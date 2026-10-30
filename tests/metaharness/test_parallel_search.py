"""Parallel search: seeds of one evaluation play at once, several proposals per iteration, the top-k
re-evaluation on validation seeds decides the final selection, proposals name their parent, every
evaluation leaves a digest, and the proposer's own tools (preflight / dev-eval) evaluate what the
search evaluates."""
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
from metaharness.core.proposer import tools_text
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


def _loop(root, *, proposer, iters=1, **cfg):
    archive = Archive(root)
    if not archive.candidates():
        archive.add_seed(BASE / "tier1_astra_naive.py", label="naive")
    config = LoopConfig(n_iterations=iters, seeds_train=[0, 1], seeds_heldout=[100], clock_mode="virtual",
                        budget=Budget(max_system2_calls=3, max_wall_s=20), harness_timeout_s=20, **cfg)
    return MetaHarnessLoop(archive=archive, game_factory=lambda **k: RulePuzzleGame(**k), game_kwargs={},
                           task_md="task", contract_doc="contract", proposer=proposer,
                           system2=ScriptedSystem2(replies=["ACTION1"], latency_s=0.2), system1=None,
                           config=config, game_name="rulepuzzle")


class Slotted(ScriptedSystem2):
    """A proposer that records which slot it was told and writes a harness naming its parent."""

    def __init__(self):
        super().__init__(latency_s=0.0)
        self.slots: list[str] = []

    def run(self, prompt, *, cwd=None, session_id=None, system=None, agentic=False, timeout_s=600.0, trace=None,
            images=None, full_access=False, web_search=False):
        import re
        self.calls.append(prompt)
        m = re.search(r"yours is slot (\d+) of (\d+)", prompt)
        self.slots.append(m.group(0) if m else "single")
        cand = re.search(r"candidate `(\d{4})`", prompt).group(1)
        d = Path(cwd) / "candidates" / cand
        d.joinpath("harness.py").write_text((BASE / "tier1_reference.py").read_text())
        d.joinpath("proposal.md").write_text(f"parent: 0000\n\nslot test ({self.slots[-1]})\n")
        from metaharness.models.base import System2Result
        return System2Result(text="done", session_id="s")


def test_parallel_proposals_fill_an_iteration_and_name_their_parent(tmp_path):
    prop = Slotted()
    loop = _loop(tmp_path / "a", proposer=prop, iters=2, proposals_per_iter=3, parallel_seeds=2)
    loop.run()
    archive = Archive(tmp_path / "a")
    assert len(archive.candidates()) == 1 + 6 and loop.iterations_done() == 2
    assert sorted(prop.slots) == sorted(["yours is slot 1 of 3", "yours is slot 2 of 3", "yours is slot 3 of 3"] * 2)
    assert all(c.parent() == "0000" for c in archive.candidates()[1:])
    lb = (tmp_path / "a" / "leaderboard.md").read_text()
    assert "| parent |" in lb and "| 0003 | 0003 | 0000 |" in lb
    # a sibling without eval.json is announced as being proposed / evaluated in parallel
    from metaharness.core.proposer import build_prompt
    sib = archive.new_candidate()
    prompt = build_prompt(archive, archive.new_candidate(), "C", "T", slot=(1, 2))
    assert f"candidates/{sib.dir.name}/" in prompt and "being proposed / evaluated in parallel" in prompt
    assert "yours is slot 2 of 2" in prompt and "Do not refine the leader" in prompt
    for d in (sib.dir, archive.candidates()[-1].dir):
        d.rmdir()
    # resume: the archive already holds 2 iterations of 3
    again = _loop(tmp_path / "a", proposer=prop, iters=2, proposals_per_iter=3)
    again.run()
    assert len(Archive(tmp_path / "a").candidates()) == 7


def test_validation_reevaluates_the_top_k_and_selects_by_combined_mean(tmp_path, monkeypatch):
    """Two candidates: the training leader scores 2 on train, 0 on validation (combined 1.0); the
    runner-up scores 1.5 everywhere. The combined mean picks the runner-up; held-out runs on it."""
    root = tmp_path / "v"
    archive = Archive(root)
    for name in ("lucky", "steady"):
        c = archive.new_candidate()
        c.harness_path.write_text(SLOW)
        (c.dir / "proposal.md").write_text("parent: none\n")
    loop = _loop(root, proposer=scripted_proposer(tier1), iters=0, validate_top=2, seeds_val=[7, 8])
    real = loop.evaluate

    def fake_eval(cand, seeds, tag="", **kw):
        res = real(cand, seeds, tag, **kw)
        score = {"0000": {"": 2.0, "validation": 0.0}, "0001": {"": 1.5, "validation": 1.5}}[cand.dir.name].get(tag, 1.0)
        p = (cand.dir / tag if tag else cand.dir) / "eval.json"
        e = json.loads(p.read_text())
        e["mean_score"] = score
        for ep in e["episodes"]:
            ep["score"] = score
        p.write_text(json.dumps(e))
        loop.archive.write_leaderboard()
        return res
    monkeypatch.setattr(loop, "evaluate", fake_eval)
    best = loop.run()
    assert best.dir.name == "0001" and (best.dir / "heldout" / "eval.json").exists()
    assert archive.candidates()[0].validation()["mean_score"] == 0.0
    lb = (root / "leaderboard.md").read_text()
    assert "train+validation 1.500" in lb and "Best so far: `0001`" in lb


def test_seed_groups_must_not_overlap(tmp_path):
    with pytest.raises(ValueError, match="train and dev seeds overlap"):
        _loop(tmp_path / "o", proposer=scripted_proposer(tier1), seeds_dev=[1])
    with pytest.raises(ValueError, match="validation seeds"):
        _loop(tmp_path / "o2", proposer=scripted_proposer(tier1), validate_top=2)


def test_tools_text_names_the_commands_and_the_dev_budget():
    t = tools_text("/py", "/repo", [50, 51], 2)
    assert "PYTHONPATH=/repo /py -m metaharness preflight" in t and "dev-eval" in t and "[50, 51]" in t
    assert "at most 2 dev evaluation" in t
    assert "dev-eval" not in tools_text("/py", "/repo", [], 2)


def test_preflight_and_dev_eval_commands_read_the_archive_manifest(tmp_runs, capsys):
    rc = main(["loop", "--tier", "1", "--game", "rulepuzzle", "--system1", "none", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--seeds", "0", "--heldout", "",
               "--dev-seeds", "9", "--s2-budget", "4", "--name", "t", "--runs-dir", str(tmp_runs)])
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
    # the proposer was told about the tools (the archive has dev seeds)
    prompt = (root / "candidates" / "0001" / "prompt.md").read_text()
    assert "metaharness preflight" in prompt and "dev-eval" in prompt


def test_seed_check_warns_when_training_seeds_do_not_vary(tmp_path):
    class Same(RulePuzzleGame):
        def reset(self, seed):
            return super().reset(0)                      # every seed replays the same episode
    loop = _loop(tmp_path / "same", proposer=scripted_proposer(tier1), iters=0)
    loop.game_factory = lambda **k: Same(**k)
    loop.qualify_seeds()
    seeds = json.loads((tmp_path / "same" / "seeds.json").read_text())
    assert any("same observation" in w for w in seeds["warnings"])
    assert any("held-out seeds start exactly like" in w for w in seeds["warnings"])
