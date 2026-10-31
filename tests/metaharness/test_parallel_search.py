"""Tier-1 search additions on top of the generation loop: seeds of one evaluation play at once, every
evaluation leaves a digest, the proposer's own tools (preflight / dev-eval) evaluate what the search
evaluates, the prompt reports recent latency, ctx.progress() detects plan failure, the leaderboard shows
the per-seed spread, validation finalists include ties, and a resumed search may carry a new task text."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from metaharness.baselines._scripted import scripted_proposer
from metaharness.cli import main
from metaharness.core.archive import Archive
from metaharness.core.contract import Budget
from metaharness.core.loop import LoopConfig, MetaHarnessLoop
from metaharness.core.proposer import build_prompt, recent_latency_text, tools_text
from metaharness.core.runner import evaluate_harness
from metaharness.models.mock import ScriptedSystem2
from metaharness.testbeds.rulepuzzle import RulePuzzleGame
from metaharness.tiers import tier1

BASE = Path(__file__).resolve().parents[2] / "metaharness" / "baselines"
SLOW = '''
import time
class H:
    def __init__(self, ctx): self.ctx = ctx
    def run_episode(self, game):
        time.sleep(0.6)
        self.ctx.act("ACTION1")
def build(ctx): return H(ctx)
'''


def _loop(root, *, proposer, iters=1, **cfg):
    archive = Archive(root)
    if not archive.candidates():
        archive.add_seed(BASE / "tier1_astra_naive.py", label="naive")
    config = LoopConfig(n_iterations=iters, seeds_train=[0, 1], seeds_heldout=[100], clock_mode="virtual",
                        budget=Budget(max_system2_calls=3, max_wall_s=20), harness_timeout_s=20, reflect=False,
                        ablate=False, **cfg)
    return MetaHarnessLoop(archive=archive, game_factory=lambda **k: RulePuzzleGame(**k), game_kwargs={},
                           task_md="task", contract_doc="contract", proposer=proposer,
                           system2=ScriptedSystem2(replies=["ACTION1"], latency_s=0.2), system1=None,
                           config=config, game_name="rulepuzzle")


def test_parallel_seeds_play_at_once_and_report_in_seed_order(tmp_path):
    h = tmp_path / "h.py"
    h.write_text(SLOW)
    t0 = time.monotonic()
    res = evaluate_harness(h, lambda **k: RulePuzzleGame(**k), [3, 1, 2], system2=ScriptedSystem2(), system1=None,
                           out_dir=tmp_path / "out", clock_mode="virtual", harness_timeout_s=10, parallel=3)
    assert time.monotonic() - t0 < 1.5                      # 3 × 0.6 s sequential would be 1.8 s
    assert [e.seed for e in res.episodes] == [3, 1, 2] and all(e.steps == 1 for e in res.episodes)
    assert res.metadata["parallel"] == 3 and (tmp_path / "out" / "digest.md").exists()
    shared = evaluate_harness(h, lambda **k: RulePuzzleGame(**k), [3, 1], system2=ScriptedSystem2(), system1=None,
                              clock_mode="virtual", harness_timeout_s=10, parallel=3, memory_mode="shared")
    assert shared.metadata["parallel"] == 1               # ordered memory: never parallel


def test_digest_indexes_steps_and_model_calls(tmp_path):
    res = evaluate_harness(BASE / "tier1_astra_naive.py", lambda **k: RulePuzzleGame(**k), [0],
                           system2=ScriptedSystem2(replies=["ACTION1"]), system1=None, out_dir=tmp_path,
                           clock_mode="virtual", budget_per_episode=Budget(max_system2_calls=3), harness_timeout_s=10)
    d = (tmp_path / "digest.md").read_text()
    assert "## seed 0" in d and d.count("system2 call") == 3 and d.count("| step ") == 3
    assert f"ended by {res.episodes[0].ended_by}" in d


def test_ctx_progress_counts_stalls_and_rejected_actions():
    from metaharness.core.contract import HarnessContext, VirtualClock
    from metaharness.models.base import TraceSink
    g = RulePuzzleGame()
    g.reset(1)
    ctx = HarnessContext(system2=None, system1=None, trace=TraceSink(), clock=VirtualClock(), budget=Budget(),
                         memory={}, game_kind="turn", act_fn=g.step, observe_fn=g.observe, done_fn=lambda: g.done,
                         score_fn=lambda: g.score)
    with pytest.raises(ValueError):
        ctx.act("BOGUS")
    assert ctx.progress()["illegal_actions"] == 1 and ctx.progress()["acts"] == 0
    for _ in range(3):
        ctx.act("ACTION5")                        # the observation changes (actions_left), the score does not
    p = ctx.progress()
    assert p["acts"] == 3 and p["since_score_change"] == 2 and p["score"] == 0.0 and p["since_obs_change"] == 0


def _fake_eval(scores: list[float], game: str = "rulepuzzle") -> dict:
    return {"game": game, "mean_score": sum(scores) / len(scores), "n_episodes": len(scores), "system2_calls": 0,
            "metadata": {"expected_episodes": len(scores), "seeds": list(range(len(scores)))},
            "episodes": [{"seed": i, "score": s} for i, s in enumerate(scores)]}


def test_leaderboard_marks_identical_seed_scores(tmp_path):
    archive = Archive(tmp_path / "eq")
    for scores in ([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]):
        c = archive.new_candidate()
        c.harness_path.write_text("def build(ctx): return None\n")
        (c.dir / "eval.json").write_text(json.dumps(_fake_eval(scores)))
    lb = archive.write_leaderboard()
    assert "seeds median / min" in lb and "3.000 / 3.000 (= all seeds)" in lb and "2.000 / 1.000 |" in lb


def test_tools_text_names_the_commands_and_the_dev_budget():
    t = tools_text("/py", "/repo", [50, 51], 2)
    assert "PYTHONPATH=/repo /py -m metaharness preflight" in t and "dev-eval" in t and "[50, 51]" in t
    assert "at most 2 dev evaluation" in t
    assert "dev-eval" not in tools_text("/py", "/repo", [], 2)


def test_prompt_reports_recent_system2_latency(tmp_path):
    archive = Archive(tmp_path / "lat")
    assert recent_latency_text(archive) == ""
    for p50, mx, failed in ((10.0, 20.0, 0), (50.0, 80.0, 3)):
        c = archive.new_candidate()
        c.harness_path.write_text("def build(ctx): return None\n")
        e = _fake_eval([1.0])
        e.update(system2_calls=10, failed_s2=failed)
        e["episodes"][0]["latency_s2"] = {"n": 5, "p50": p50, "max": mx}
        (c.dir / "eval.json").write_text(json.dumps(e))
    text = recent_latency_text(archive)
    assert "median 50 s" in text and "worst 80 s" in text and "3 of 23 calls failed" in text
    assert text in build_prompt(archive, archive.new_candidate(), "C", "T")
    assert "digest.md" in build_prompt(archive, archive.new_candidate(), "C", "T")


def test_preflight_and_dev_eval_commands_read_the_archive_manifest(tmp_runs, capsys):
    rc = main(["loop", "--tier", "1", "--game", "rulepuzzle", "--system1", "none", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--seeds", "0", "--heldout", "",
               "--dev-seeds", "9", "--s2-budget", "4", "--no-reflect", "--name", "t", "--runs-dir", str(tmp_runs)])
    assert rc == 0
    capsys.readouterr()                                      # drain the loop's own output
    root = tmp_runs / "t"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["run"]["game"] == "rulepuzzle" and manifest["seeds_dev"] == [9]
    harness = str(root / "candidates" / "0001" / "harness.py")
    assert main(["preflight", harness, "--archive", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main(["dev-eval", harness, "--archive", str(root)]) == 0
    out = capsys.readouterr().out
    assert "seed 9:" in out and "mean score:" in out
    dev = list((root / "candidates" / "0001" / "dev").iterdir())
    assert len(dev) == 1 and (dev[0] / "digest.md").exists() and (dev[0] / "trace_seed9.jsonl").exists()
    with pytest.raises(SystemExit, match="not dev seeds"):
        main(["dev-eval", harness, "--archive", str(root), "--seeds", "0"])
    prompt = (root / "candidates" / "0001" / "prompt.md").read_text()
    assert "metaharness preflight" in prompt and "dev-eval" in prompt


def test_dev_seeds_must_not_overlap_the_scored_seeds(tmp_path):
    with pytest.raises(ValueError, match="dev seeds overlap"):
        _loop(tmp_path / "o", proposer=scripted_proposer(tier1), seeds_dev=[1])


def test_resume_accepts_a_changed_task_text_but_not_changed_budgets(tmp_path):
    root = tmp_path / "res"
    _loop(root, proposer=scripted_proposer(tier1), iters=1).run()
    again = MetaHarnessLoop(archive=Archive(root), game_factory=lambda **k: RulePuzzleGame(**k), game_kwargs={},
                            task_md="a different task text", contract_doc="contract",
                            proposer=scripted_proposer(tier1), system2=ScriptedSystem2(replies=["ACTION1"]),
                            system1=None, game_name="rulepuzzle",
                            config=LoopConfig(n_iterations=1, seeds_train=[0, 1], seeds_heldout=[100],
                                              clock_mode="virtual", budget=Budget(max_system2_calls=3, max_wall_s=20),
                                              harness_timeout_s=20, reflect=False, ablate=False))
    assert again.task_changed
    again.run()
    assert "task text differs" in (root / "loop.log").read_text()
    with pytest.raises(ValueError, match="Search configuration changed \\(budget\\)"):
        MetaHarnessLoop(archive=Archive(root), game_factory=lambda **k: RulePuzzleGame(**k), game_kwargs={},
                        task_md="task", contract_doc="contract", proposer=scripted_proposer(tier1),
                        system2=ScriptedSystem2(replies=["ACTION1"]), system1=None, game_name="rulepuzzle",
                        config=LoopConfig(n_iterations=1, seeds_train=[0, 1], seeds_heldout=[100], clock_mode="virtual",
                                          budget=Budget(max_system2_calls=9, max_wall_s=20), harness_timeout_s=20,
                                          reflect=False, ablate=False))


def test_validation_finalists_include_ties_on_the_training_mean(tmp_path, monkeypatch):
    root = tmp_path / "ties"
    loop = _loop(root, proposer=scripted_proposer(tier1), iters=0, seeds_val=[7], finalists=2)
    archive = loop.archive
    for score in (3.0, 3.0, 3.0, 3.0, 1.0):              # after the loop exists: a scored archive needs a manifest
        c = archive.new_candidate()
        c.harness_path.write_text(SLOW)
        (c.dir / "eval.json").write_text(json.dumps(_fake_eval([score])))

    def fake_many(requests, **kw):
        out = []
        for cand, seeds, tag, games in requests:
            e = _fake_eval([2.0])
            (cand.dir / tag).mkdir(exist_ok=True)
            (cand.dir / tag / "eval.json").write_text(json.dumps(e))
            out.append(e)
        return out
    monkeypatch.setattr(loop, "evaluate_many", fake_many)
    assert loop.select_champion() is not None
    validated = sorted(c.dir.name for c in archive.candidates() if c.validation())
    assert validated == ["0001", "0002", "0003", "0004"]            # all four tied at 3.0; 0005 is out (0000 = baseline, unscored)
