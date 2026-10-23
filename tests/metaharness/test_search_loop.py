"""The search loop and what the proposer is told: interrupted searches stay resumable, seeds are
checked before anything is evaluated, held-out results are written once, the task text states the
evaluation that actually runs, and model latency is calibrated once per archive."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from metaharness.baselines._scripted import scripted_proposer
from metaharness.core.archive import Archive
from metaharness.core.calibrate import calibrate
from metaharness.core.contract import Budget
from metaharness.core.loop import LoopConfig, MetaHarnessLoop, SeedRefused, _archive_lock
from metaharness.models.base import ModelUnavailable, System2Result
from metaharness.models.mock import HeuristicSystem1, ScriptedSystem2
from metaharness.testbeds.snake import SnakeGame
from metaharness.tiers import tier1, tier2

BASE = Path(__file__).resolve().parents[2] / "metaharness" / "baselines"


class Quota(ScriptedSystem2):
    """The same profile as the scripted stand-in, out of quota (resuming keeps the manifest)."""

    def run(self, *args, **kwargs):
        raise ModelUnavailable("usage limit reached")


def _loop(root, *, proposer, system2=None, iters=1, factory=None, heldout=(100,), **cfg):
    archive = Archive(root)
    if not archive.candidates():
        archive.add_seed(BASE / "tier2_jev_only.py", label="tier2_jev_only")
    config = LoopConfig(n_iterations=iters, seeds_train=[0, 1], seeds_heldout=list(heldout), clock_mode="virtual",
                        budget=Budget(max_system2_calls=4, max_system1_calls=200, max_wall_s=30),
                        harness_timeout_s=30, **cfg)
    return MetaHarnessLoop(archive=archive, game_factory=factory or (lambda **k: SnakeGame(max_ticks=30, **k)),
                           game_kwargs={}, task_md="task", contract_doc="contract", proposer=proposer,
                           system2=system2 or ScriptedSystem2(latency_s=0.5), system1=HeuristicSystem1(latency_s=0.1),
                           config=config, game_name="snake")


def test_model_outage_skips_heldout_and_the_search_resumes(tmp_path):
    loop = _loop(tmp_path / "a", proposer=Quota())
    loop.run()
    assert loop.interrupted and "model unavailable" in loop.interrupted
    assert not list((tmp_path / "a" / "candidates").glob("*/heldout*"))       # archive stays resumable
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2))
    best = loop.run()
    assert loop.interrupted is None and best is not None and (best.dir / "heldout" / "eval.json").exists()


def test_evaluation_cut_by_an_outage_is_redone_on_resume(tmp_path):
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), system2=Quota(), iters=0, heldout=())
    loop.archive.add_seed(BASE / "tier2_astra_only.py", label="tier2_astra_only")
    loop.run()
    astra = loop.archive.candidates()[1]
    assert loop.interrupted and astra.eval()["fatal"].startswith("model unavailable")
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=())
    loop.run()
    assert astra.eval()["fatal"] is None and astra.eval()["n_episodes"] == 2
    assert list(astra.dir.glob("eval.interrupted-*.json"))


def test_refused_seed_stops_the_search_before_any_evaluation(tmp_path):
    class Picky(SnakeGame):
        def reset(self, seed):
            if seed == 100:
                raise RuntimeError("seed 100 is not admissible")
            return super().reset(seed)

    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), factory=lambda **k: Picky(max_ticks=30, **k))
    with pytest.raises(SeedRefused, match="heldout seed 100"):
        loop.run()
    assert all(c.eval() is None for c in loop.archive.candidates())
    assert json.loads((tmp_path / "a" / "seeds.json").read_text())["refused"]


def test_archive_lock_admits_one_process(tmp_path):
    with _archive_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another Meta-Harness process"):
            with _archive_lock(tmp_path):
                pass
    with _archive_lock(tmp_path):            # released again
        pass


def test_heldout_results_are_written_once(tmp_path):
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0)
    best = loop.run()
    p = best.dir / "heldout" / "eval.json"
    first = p.read_text()
    _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0).run()
    assert p.read_text() == first


def test_task_states_the_evaluation_that_runs():
    b = Budget(max_system2_calls=7, max_system1_calls=70, max_wall_s=300)
    no_heldout = tier2.task_md("snake", b, "wall", {}, study="RULES", generality=False)
    assert "Evaluation scope" in no_heldout and "Generality requirement" not in no_heldout
    assert "implement exact" in no_heldout and "300 s" in no_heldout and "70 calls" in no_heldout
    with_heldout = tier2.task_md("snake", b, "wall", {}, study="RULES", generality=True)
    assert "Generality requirement" in with_heldout and "implement exact" not in with_heldout
    t1 = tier1.task_md("rulepuzzle", b, "virtual", {}, generality=False)
    assert "mean levels completed over seeds" not in t1 and "re-drawn" not in t1 and "Jev" not in t1
    assert "mean score over the training seeds" in t1 and "fewer actions" in t1


def test_calibration_is_measured_once_and_rendered(tmp_path):
    class Fake2(ScriptedSystem2):
        name = "fake-astra"

        def run(self, prompt, **kw):
            self.calls.append(prompt)
            return System2Result(text="ok", latency_s=12.0 if "checklist" in prompt else 3.0,
                                 usage={"output_tokens": 800 if "checklist" in prompt else 2})

    class Fake1(HeuristicSystem1):
        name = "fake-jev"

    s2 = Fake2()
    cal = calibrate(tmp_path, system2=s2, system1=Fake1(), game_factory=lambda **k: SnakeGame(**k),
                    game_kwargs={}, seed=0, log=lambda m: None)
    assert cal["system2_long"]["output_tokens"] == 800 and "latency_s" in cal["system1"]
    assert calibrate(tmp_path, system2=s2, system1=Fake1(), game_factory=SnakeGame, game_kwargs={}, seed=0,
                     log=lambda m: None) == cal and len(s2.calls) == 2           # cached, not re-measured
    text = tier2.task_md("snake", Budget(), "wall", {}, calibration=cal)
    assert "Measured model latency" in text and "~800 output tokens: 12 s" in text
    assert calibrate(tmp_path / "x", system2=ScriptedSystem2(), system1=HeuristicSystem1(), game_factory=SnakeGame,
                     game_kwargs={}, seed=0) is None                              # offline stand-ins


def test_proposer_timeout_keeps_a_harness_without_proposal(tmp_path):
    from metaharness.core.proposer import propose

    class TimedOut(ScriptedSystem2):
        def run(self, prompt, *, cwd, **kwargs):
            d = sorted((cwd / "candidates").iterdir())[-1]
            (d / "harness.py").write_text("build = lambda ctx: None\n")
            err = TimeoutError("CLI deadline")
            err.session_id = "thread-1"
            raise err

    a = Archive(tmp_path / "archive")
    c = propose(TimedOut(), a, contract_doc="contract", task_md="task", iteration=(2, 3), timeout_s=600)
    assert c.harness_path.exists() and "timed out" in (c.dir / "proposal.md").read_text()
    assert json.loads((c.dir / "session.json").read_text())["session_id"] == "thread-1"
    prompt = (c.dir / "prompt.md").read_text()
    assert "iteration 2 of 3" in prompt and "10 minutes" in prompt


def test_leaderboard_has_no_pseudo_games_and_explains_ineligibility(tmp_path):
    a = Archive(tmp_path / "archive")
    ok, broken = a.new_candidate(), a.new_candidate()
    ok.eval_path.write_text(json.dumps({"game": "snake", "mean_score": 2.0, "n_episodes": 1, "metadata": {"seeds": [0]},
                                        "episodes": [{"seed": 0, "score": 2.0, "ended_by": "game_done", "wall_s": 3}]}))
    broken.eval_path.write_text(json.dumps({"game": "unknown", "fatal": "load error: SyntaxError: bad", "n_episodes": 0,
                                            "mean_score": None}))
    text = a.write_leaderboard()
    assert a.games() == ["snake"] and "selection (mean / worst" not in text
    assert "✗ fatal: load error: SyntaxError: bad" in text and "seed 0: 2.00 game_done 3s" in text


def test_an_evaluation_cut_by_a_dead_game_is_redone_on_resume(tmp_path):
    down = {"on": True}

    class Flaky(SnakeGame):
        def reset(self, seed):
            if down["on"]:
                raise ConnectionError("game server is down")
            return super().reset(seed)

    factory = lambda **k: Flaky(max_ticks=30, **k)          # noqa: E731
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=(), factory=factory,
                 qualify_seeds=False)
    loop.run()
    assert loop.interrupted and "game failed in every episode" in loop.interrupted
    down["on"] = False
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=(), factory=factory,
                 qualify_seeds=False)
    loop.run()
    e = loop.archive.candidates()[0].eval()
    assert loop.interrupted is None and e["errors"] == [] and e["n_episodes"] == 2


def test_heldout_outage_is_completed_on_the_same_candidate(tmp_path):
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, system2=Quota(), heldout=(100, 101))
    loop.archive.add_seed(BASE / "tier2_jev_only.py", label="tier2_jev_only_b")
    best = loop.run()
    assert loop.interrupted is None and best is not None
    # the held-out stage of the selected candidate ran with a working player; now break it mid-way
    held = best.dir / "heldout" / "eval.json"
    e = json.loads(held.read_text())
    e["fatal"], e["n_episodes"], e["episodes"] = "model unavailable: usage limit", 1, e["episodes"][:1]
    held.write_text(json.dumps(e))
    with pytest.raises(ValueError, match="held-out results"):           # exposed: no new proposals
        _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=1, heldout=(100, 101))
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=(100, 101))
    assert loop.heldout_best().dir == best.dir
    loop.run()
    assert json.loads(held.read_text())["n_episodes"] == 2                # completed, same candidate


def test_interrupted_heldout_resumes_with_the_same_iters(tmp_path):
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=1, heldout=(100, 101))
    best = loop.run()
    held = best.dir / "heldout" / "eval.json"
    e = json.loads(held.read_text())
    e["fatal"], e["n_episodes"], e["episodes"] = "model unavailable: usage limit", 1, e["episodes"][:1]
    held.write_text(json.dumps(e))
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=1, heldout=(100, 101))   # same command
    loop.run()
    assert json.loads(held.read_text())["n_episodes"] == 2
    assert loop.proposals_done() == 1


def test_an_empty_proposal_directory_does_not_count(tmp_path):
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=())
    loop.archive.new_candidate()            # a proposer killed before writing anything
    assert loop.proposals_done() == 0


def test_a_candidate_that_breaks_the_game_is_retried_once_then_kept(tmp_path):
    class Fragile(SnakeGame):
        def reset(self, seed):
            if seed in (0, 1) and Fragile.broken:
                raise ConnectionError("server crashed")
            return super().reset(seed)
    Fragile.broken = True
    factory = lambda **k: Fragile(max_ticks=30, **k)          # noqa: E731
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=(), factory=factory,
                 qualify_seeds=False)
    loop.run()
    assert loop.interrupted                                   # first time: the game looks down
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=(), factory=factory,
                 qualify_seeds=False)
    loop.run()
    assert loop.interrupted is None                           # redone once, failed the same way: kept
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=0, heldout=(), factory=factory,
                 qualify_seeds=False)
    loop.run()
    assert loop.interrupted is None and len(list(loop.archive.candidates()[0].dir.glob("eval.interrupted-*"))) == 1


def test_iters_counts_the_archive_proposals(tmp_path):
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=1, heldout=())
    loop.run()
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=1, heldout=())
    loop.run()
    assert sum("-" not in c.dir.name for c in loop.archive.candidates()) == 1      # resumed: nothing left
    loop = _loop(tmp_path / "a", proposer=scripted_proposer(tier2), iters=2, heldout=())
    loop.run()
    assert sum("-" not in c.dir.name for c in loop.archive.candidates()) == 2


def test_refused_seed_does_not_pin_the_archive(tmp_path):
    class Picky(SnakeGame):
        def reset(self, seed):
            if seed == 1:
                raise RuntimeError("seed 1 is not admissible")
            return super().reset(seed)

    def make(seeds):
        archive = Archive(tmp_path / "a")
        if not archive.candidates():
            archive.add_seed(BASE / "tier2_jev_only.py", label="tier2_jev_only")
        cfg = LoopConfig(n_iterations=0, seeds_train=seeds, seeds_heldout=[], clock_mode="virtual",
                         budget=Budget(max_system2_calls=2, max_system1_calls=50, max_wall_s=30), harness_timeout_s=30)
        return MetaHarnessLoop(archive=archive, game_factory=lambda **k: Picky(max_ticks=20, **k), game_kwargs={},
                               task_md="task", contract_doc="c", proposer=scripted_proposer(tier2),
                               system2=ScriptedSystem2(latency_s=0.5), system1=HeuristicSystem1(), config=cfg)

    with pytest.raises(SeedRefused):
        make([0, 1]).run()
    make([0, 2]).run()                       # other seeds: accepted, the first manifest never pinned
    assert json.loads((tmp_path / "a" / "manifest.json").read_text())["seeds_train"] == [0, 2]


def test_tuple_game_kwargs_do_not_break_resume(tmp_path):
    def make():
        archive = Archive(tmp_path / "a")
        if not archive.candidates():
            archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")
        cfg = LoopConfig(n_iterations=0, seeds_train=[0], seeds_heldout=[], clock_mode="virtual",
                         budget=Budget(max_wall_s=30), harness_timeout_s=30)
        def factory(**k):
            assert k == {"size": (12, 12)}       # the caller's kwargs, not their JSON round trip
            return SnakeGame(max_ticks=10)
        return MetaHarnessLoop(archive=archive, game_factory=factory,
                               game_kwargs={"size": (12, 12)}, task_md="task", contract_doc="c",
                               proposer=scripted_proposer(tier2), system2=None, system1=HeuristicSystem1(),
                               config=cfg)

    make().run()
    make().run()                              # the stored manifest has a list: still equal


def test_heldout_adaptor_directory_brings_its_kwargs_and_seeds(tmp_path):
    import argparse
    from metaharness import cli
    d = tmp_path / "held"
    d.mkdir()
    (d / "ADAPTOR.md").write_text("held-out adaptor")
    (d / "game.py").write_text('''
from metaharness.testbeds.lanerunner import LaneRunnerGame
SEEDS = {"train": [1, 2], "heldout": [7, 8]}
EVAL_KWARGS = {"max_ticks": 33}
class Held(LaneRunnerGame):
    name = "held"
def make_game(**kw):
    return Held(**kw)
''')
    a = argparse.Namespace(heldout_game=str(d), heldout_game_kwargs='{"density": 0.2}')
    key, kwargs, seeds = cli._resolve_heldout_game(a)
    assert kwargs == {"max_ticks": 33, "density": 0.2} and seeds == [7, 8]
