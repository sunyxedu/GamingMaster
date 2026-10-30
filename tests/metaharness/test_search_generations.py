"""Generations: directives with parent sampling, duplicate rejection, parallel child-process
evaluation, the validation pick, multi-game training and the reflection record."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from metaharness.cli import main
from metaharness.core import generation
from metaharness.core.archive import Archive
from metaharness.core.contract import Budget
from metaharness.core.loop import GameSpec, LoopConfig, MetaHarnessLoop
from metaharness.models.mock import HeuristicSystem1, ScriptedSystem2
from metaharness.testbeds.lanerunner import LaneRunnerGame
from metaharness.testbeds.snake import SnakeGame
from metaharness.tiers import tier2

BASE = Path(__file__).resolve().parents[2] / "metaharness" / "baselines"
REF = (BASE / "tier2_reference.py").read_text()
IDLE = (BASE / "tier2_idle.py").read_text()


def _eval(score, errors=(), n=2, game="snake"):
    return {"game": game, "mean_score": score, "n_episodes": n, "errors": list(errors),
            "metadata": {"seeds": list(range(n))},
            "episodes": [{"seed": i, "score": score, "ended_by": "game_done"} for i in range(n)]}


def _archive(tmp_path, scores):
    a = Archive(tmp_path / "a")
    for i, (score, errors) in enumerate(scores):
        c = a.new_candidate()
        c.harness_path.write_text(IDLE + f"\n# v{i}\n")
        c.eval_path.write_text(json.dumps(_eval(score, errors)))
    return a


def test_directives_cover_the_modes_and_sample_parents(tmp_path):
    a = _archive(tmp_path, [(1.0, ()), (5.0, ()), (3.0, ()), (9.0, ["seed 1: harness_error"])])
    plan = generation.plan(a, 4, generation=0, rng=random.Random(1))
    modes = [d["mode"] for d in plan]
    assert modes[:3] == ["refine", "rewrite", "crossover"] and modes[3] == "repair"   # 0003 scores best but erred
    assert plan[0]["parents"] == [1] and plan[0]["parent_dirs"] == ["0001"]            # refine the best
    assert plan[2]["parents"][0] == 1 and plan[2]["parents"][1] != 1                   # crossover: best + another
    assert plan[3]["parents"] == [3]
    assert all(d["generation"] == 0 for d in plan)
    # one repair slot per errored candidate: once a repair child exists, sampling is among the eligible ones
    child = a.new_candidate()
    (child.dir / "directive.json").write_text(json.dumps({"mode": "repair", "parents": [3], "parent_dirs": ["0003"]}))
    assert generation.plan(a, 1, 0)[0]["mode"] == "refine"
    # weights: the best and unexplored candidates are drawn most; nobody has probability zero
    draws = [generation.plan(a, 1, g, rng=random.Random(g))[0]["parents"][0] for g in range(1, 200, 4)]
    assert set(draws) <= {0, 1, 2} and draws.count(1) > draws.count(0)
    # an archive without eligible candidates: free exploration
    assert generation.plan(Archive(tmp_path / "empty"), 2, 0) == [
        {"mode": "explore", "parents": [], "parent_dirs": [], "generation": 0}] * 2
    assert "Mode: refine" in generation.describe(plan[0]) and "candidates/0001/" in generation.describe(plan[0])


def test_children_lower_a_parents_weight(tmp_path):
    a = _archive(tmp_path, [(5.0, ()), (5.0, ())])
    for _ in range(6):                                     # six children of 0000, none of 0001
        c = a.new_candidate()
        (c.dir / "directive.json").write_text(json.dumps({"mode": "refine", "parents": [0], "parent_dirs": ["0000"]}))
    assert generation.children_counts(a) == {0: 6}
    draws = [generation.plan(a, 1, g, rng=random.Random(g))[0]["parents"][0] for g in range(1, 400, 4)]
    assert draws.count(1) > 3 * draws.count(0)


def test_near_duplicates_are_detected_but_edits_are_not(tmp_path):
    a = _archive(tmp_path, [(1.0, ())])
    same = a.new_candidate()
    same.harness_path.write_text(IDLE + "\n# a comment changes nothing\n")
    other, sim = generation.duplicate_of(a, same)
    assert other.idx == 0 and sim == 1.0
    changed = a.new_candidate()
    changed.harness_path.write_text(IDLE.replace("while not self.ctx.done:", "for _ in range(3):\n            if self.ctx.done: break"))
    assert generation.duplicate_of(a, changed) is None
    one_line = a.new_candidate()                            # a one-line change is a proposal, however long the file
    one_line.harness_path.write_text(IDLE + "\n" + "\n".join(f"X{i} = {i}" for i in range(500)) + "\n")
    longer = a.new_candidate()
    longer.harness_path.write_text(one_line.harness_path.read_text() + "Y = 1\n")
    assert generation.duplicate_of(a, longer) is None
    assert generation.similarity(IDLE, REF) < 0.5


def test_parallel_generations_validate_and_pick_a_champion(tmp_runs):
    # two proposals per generation with a scripted proposer: the second is a duplicate of the first and is not
    # evaluated; evaluations run in child processes; the finalists are re-evaluated on the validation seeds
    rc = main(["loop", "--tier", "2", "--game", "snake", "--system1", "heuristic", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--proposals", "2", "--parallel", "3",
               "--seeds", "0,1", "--val-seeds", "7", "--heldout", "100", "--finalists", "2",
               "--name", "gen", "--runs-dir", str(tmp_runs)])
    assert rc == 0
    a = Archive(tmp_runs / "gen")
    names = [c.dir.name for c in a.candidates()]
    assert names == ["0000-tier2_jev_only", "0001-tier2_astra_only", "0002-tier2_idle", "0003", "0004"]
    c3, c4 = a.candidates()[3], a.candidates()[4]
    assert c3.directive()["mode"] == "refine" and c3.directive()["parents"] == [c3.parent()]
    assert c3.eval()["n_episodes"] == 2 and [ep["seed"] for ep in c3.eval()["episodes"]] == [0, 1]
    for s in (0, 1):                                      # one child process per seed, traces merged back
        assert (c3.dir / "seeds" / f"seed-{s}" / "evalworker.json").exists() and (c3.dir / f"trace_seed{s}.jsonl").exists()
    assert c3.eval()["metadata"]["expected_episodes"] == 2 and c3.eval()["metadata"]["seeds"] == [0, 1]
    assert c4.eval()["fatal"].startswith("duplicate of candidates/0003")
    assert "Sibling proposers are writing candidates/0004/" in (c3.dir / "prompt.md").read_text()
    champion = json.loads((a.root / "champion.json").read_text())
    assert champion["seeds_val"] == {"snake": [7]} and len(champion["finalists"]) == 2
    assert c3.validation()["n_episodes"] == 1 and all(r["status"] == "ok" for r in champion["finalists"])
    expected = max(champion["finalists"], key=lambda r: (round(r["combined"], 9), -r["rank"]))["candidate"]
    assert champion["candidate"] == expected                                          # training + validation together
    winner = next(c for c in a.candidates() if c.dir.name == expected)
    assert (winner.dir / "heldout" / "eval.json").exists()                            # only the champion
    assert not any((c.dir / "heldout").exists() for c in a.candidates() if c.dir != winner.dir)
    lb = (a.root / "leaderboard.md").read_text()
    assert "Validation seeds" in lb and f"Final pick: `{expected}`" in lb and "(refine of 0000-tier2_jev_only)" in lb
    assert (a.root / "lessons.md").exists() and "## Generation 1" in (a.root / "lessons.md").read_text()
    ablation = json.loads((winner.dir / "ablation-no-models" / "eval.json").read_text())   # the pick, models off
    assert ablation["n_episodes"] == 2 and ablation["system1_calls"] == 0 and ablation["system2_calls"] == 0
    assert "Ablation: `" + expected + "` without runtime System-1 / System-2" in (a.root / "leaderboard.md").read_text()
    assert json.loads((a.root / "manifest.json").read_text())["run"]["system1"] == "heuristic"
    # the archive is closed: validation / held-out results exist, so no further proposals
    with pytest.raises(ValueError, match="held-out results"):
        main(["loop", "--tier", "2", "--game", "snake", "--system1", "heuristic", "--system2", "scripted",
              "--proposer", "scripted", "--clock", "virtual", "--iters", "2", "--proposals", "2", "--parallel", "3",
              "--seeds", "0,1", "--val-seeds", "7", "--heldout", "100", "--finalists", "2",
              "--name", "gen", "--runs-dir", str(tmp_runs)])


def test_multi_game_training_combines_per_game_scores(tmp_runs):
    rc = main(["loop", "--tier", "2", "--game", "snake,lanerunner", "--system1", "heuristic", "--system2", "scripted",
               "--proposer", "scripted", "--clock", "virtual", "--iters", "1", "--parallel", "2", "--seeds", "0",
               "--heldout", "", "--name", "mg", "--runs-dir", str(tmp_runs)])
    assert rc == 0
    a = Archive(tmp_runs / "mg")
    assert a.games() == ["snake", "lanerunner"]
    best = a.best()
    e = best.eval()
    assert set(e["per_game"]) == {"snake", "lanerunner"} and e["n_episodes"] == 2
    assert (best.dir / "game-snake" / "eval.json").exists() and (best.dir / "game-lanerunner" / "trace_seed0.jsonl").exists()
    task = (a.root / "task.md").read_text()
    assert "### Training game `snake`" in task and "### Training game `lanerunner`" in task
    assert "Several training games" in task and "Generality requirement" in task
    assert "selection (mean / worst norm.)" in (a.root / "leaderboard.md").read_text()


def test_default_seed_blocks_carve_validation_from_declared_train_seeds(tmp_path):
    import argparse
    from metaharness import cli
    d = tmp_path / "ad"
    d.mkdir()
    (d / "ADAPTOR.md").write_text("x")
    (d / "game.py").write_text('from metaharness.testbeds.snake import SnakeGame\nSEEDS = {"train": [10, 11, 12, 13, 14, 15, 16, 17], "heldout": [90, 91]}\ndef make_game(**kw):\n    return SnakeGame(**kw)\n')
    a = argparse.Namespace(game_kwargs="", seeds=None, heldout=None, val_seeds="")
    declared, = cli._game_specs(a, [(cli.register_adaptor(d), str(d))])
    assert declared.seeds_train == [10, 11, 12, 13, 14, 15] and declared.seeds_val == [16, 17]
    assert declared.seeds_heldout == [90, 91] and declared.adaptor_dir == str(d)
    plain, = cli._game_specs(a, [("snake", None)])
    assert (plain.seeds_train, plain.seeds_val, plain.seeds_heldout) == ([0, 1, 2], [3, 4], [100, 101, 102])
    explicit, = cli._game_specs(argparse.Namespace(game_kwargs="", seeds="5,6", heldout="", val_seeds=""),
                                [("snake", None)])
    assert (explicit.seeds_train, explicit.seeds_val, explicit.seeds_heldout) == ([5, 6], [], [])


def test_task_text_for_several_games_lists_each_tick():
    text = tier2.task_md(["snake", "lanerunner"], Budget(), "wall", [{}, {}])
    assert "Tick: every 0.25s." in text and "Tick: every 0.5s." in text and "at its own pace" in text
    single = tier2.task_md("snake", Budget(), "wall", {})
    assert "ticks every 0.25s" in single and "Training game" not in single


def test_reflection_text_is_kept_when_the_model_has_no_tools(tmp_path):
    archive = Archive(tmp_path / "a")
    archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")
    notes = ScriptedSystem2(replies=["- idle scored 0 on both seeds (game_done at tick 30): nothing acts\n"], latency_s=0.1)
    cfg = LoopConfig(n_iterations=0, seeds_train=[0, 1], seeds_heldout=[], clock_mode="virtual",
                     budget=Budget(max_wall_s=30), harness_timeout_s=30)
    loop = MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=30, **k), game_kwargs={},
                           task_md="task", contract_doc="c", proposer=notes, system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    loop.run()
    loop.reflect(3, archive.candidates())
    text = (archive.root / "lessons.md").read_text()
    assert text.startswith("## Generation 3") and "idle scored 0" in text
    assert (archive.root / "reflections" / "gen_003_prompt.md").exists()
    assert "candidates/0000-tier2_idle — seed" in (archive.root / "reflections" / "gen_003_prompt.md").read_text()


def test_in_process_multi_game_loop_with_game_specs(tmp_path):
    archive = Archive(tmp_path / "a")
    archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")
    extra = GameSpec("lanerunner", lambda **k: LaneRunnerGame(max_ticks=20, **k), {}, seeds_train=[3])
    cfg = LoopConfig(n_iterations=0, seeds_train=[0], seeds_heldout=[9], clock_mode="virtual",
                     budget=Budget(max_wall_s=30), harness_timeout_s=30, more_games=[extra])
    loop = MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=20, **k), game_kwargs={},
                           task_md="task", contract_doc="c", proposer=ScriptedSystem2(), system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    best = loop.run()
    e = best.eval()
    assert e["per_game"]["lanerunner"]["episodes"][0]["seed"] == 3 and e["per_game"]["snake"]["episodes"][0]["seed"] == 0
    held = json.loads((best.dir / "heldout" / "eval.json").read_text())
    assert held["per_game"]["snake"]["episodes"][0]["seed"] == 9 and held["per_game"]["lanerunner"]["episodes"][0]["seed"] == 9
    seeds = json.loads((archive.root / "seeds.json").read_text())["results"]
    assert "lanerunner:train:3" in seeds and "train:0" in seeds and "heldout:9" in seeds


def test_per_seed_evaluations_merge_into_the_joint_result(tmp_path):
    from metaharness.core.evalworker import merge_seed_evals
    from metaharness.core.runner import evaluate_harness

    def run(seeds, out):
        return evaluate_harness(BASE / "tier2_reference.py", lambda **k: SnakeGame(max_ticks=40, **k), seeds,
                                system2=ScriptedSystem2(latency_s=0.5), system1=HeuristicSystem1(latency_s=0.1),
                                out_dir=out, clock_mode="virtual", budget_per_episode=Budget(max_wall_s=60),
                                harness_timeout_s=60, game_name="snake").to_json()

    joint = run([0, 1, 2], tmp_path / "joint")
    merged = merge_seed_evals([run([s], tmp_path / f"s{s}") for s in (2, 0, 1)], [0, 1, 2])
    for key in ("mean_score", "n_episodes", "system1_calls", "system2_calls", "stale_fraction", "mean_steps",
                "failed_s1", "failed_s2", "ended_by", "errors", "s1_informed_fraction", "s2_informed_fraction",
                "score_std", "game"):
        assert merged[key] == joint[key], key
    assert [ep["seed"] for ep in merged["episodes"]] == [0, 1, 2]
    assert merged["metadata"]["seeds"] == [0, 1, 2] and merged["metadata"]["expected_episodes"] == 3
    out = merge_seed_evals([{"fatal": "load error: x", "episodes": []},
                            {"fatal": "model unavailable: quota", "episodes": []}], [0, 1])
    assert out["fatal"].startswith("model unavailable") and out["n_episodes"] == 0 and out["mean_score"] is None


def test_a_missing_reflection_is_caught_up_and_a_timed_out_one_finishes(tmp_path):
    from metaharness.models.base import ModelTimeout, System2Result

    class Notes(ScriptedSystem2):
        def __init__(self, timeout_first=False):
            super().__init__(latency_s=0.1)
            self.timeout_first = timeout_first

        def run(self, prompt, *, session_id=None, **kw):
            self.calls.append((prompt, session_id))
            if self.timeout_first and len(self.calls) == 1:
                raise ModelTimeout("deadline", session_id="r-1")
            return System2Result(text=f"- note {len(self.calls)}", session_id="r-1")

    archive = Archive(tmp_path / "a")
    archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")
    done = archive.new_candidate()                              # a generation-1 proposal, evaluated, no lessons
    done.harness_path.write_text(IDLE + "\n# v1\n")
    (done.dir / "directive.json").write_text(json.dumps({"mode": "explore", "parents": [], "generation": 0}))
    cfg = LoopConfig(n_iterations=1, seeds_train=[0], seeds_heldout=[], clock_mode="virtual",
                     budget=Budget(max_wall_s=30), harness_timeout_s=30)
    notes = Notes(timeout_first=True)
    loop = MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=20, **k), game_kwargs={},
                           task_md="task", contract_doc="c", proposer=notes, system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    loop.run()
    text = (archive.root / "lessons.md").read_text()
    assert text.startswith("## Generation 1") and "note 2" in text           # caught up, after a finishing call
    assert notes.calls[1][1] == "r-1" and "Your time is up" in notes.calls[1][0]
    loop.run()                                                                # already there: no second reflection
    assert (archive.root / "lessons.md").read_text() == text


def test_an_archive_that_pinned_validation_seeds_still_resumes(tmp_path):
    archive = Archive(tmp_path / "a")
    archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")

    def make(val):
        cfg = LoopConfig(n_iterations=0, seeds_train=[0], seeds_heldout=[], seeds_val=val, clock_mode="virtual",
                         budget=Budget(max_wall_s=30), harness_timeout_s=30)
        return MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=10, **k), game_kwargs={},
                               task_md="task", contract_doc="c", proposer=ScriptedSystem2(), system2=None,
                               system1=HeuristicSystem1(), config=cfg, game_name="snake")
    make([]).run()
    m = json.loads((archive.root / "manifest.json").read_text())
    m["seeds_val"] = [5]                                    # what an earlier version wrote
    (archive.root / "manifest.json").write_text(json.dumps(m))
    make([7, 8])                                            # other validation seeds: still the same protocol


def test_patience_stops_a_search_whose_best_is_stale(tmp_path):
    a = Archive(tmp_path / "a")
    cfg = LoopConfig(n_iterations=8, proposals_per_iteration=1, patience=2, seeds_train=[0, 1], seeds_heldout=[],
                     clock_mode="virtual", budget=Budget(max_wall_s=30), harness_timeout_s=30, preflight=False)
    loop = MetaHarnessLoop(archive=a, game_factory=lambda **k: SnakeGame(max_ticks=10, **k), game_kwargs={},
                           task_md="t", contract_doc="c", proposer=ScriptedSystem2(), system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    seed = a.new_candidate("baseline")
    seed.eval_path.write_text(json.dumps(_eval(1.0)))
    for gen, score in ((0, 5.0), (1, 4.0), (2, 3.0)):          # generation 1 found the best, 2 and 3 did not
        c = a.new_candidate()
        c.eval_path.write_text(json.dumps(_eval(score)))
        (c.dir / "directive.json").write_text(json.dumps({"mode": "refine", "parents": [0], "generation": gen}))
    assert loop._plateaued(3) and not loop._plateaued(2)
    cfg.patience = 3
    assert not loop._plateaued(3) and loop._plateaued(4)
    cfg.patience = 0
    assert not loop._plateaued(7)


def test_a_reflection_that_writes_nothing_leaves_a_numeric_summary(tmp_path):
    class Silent(ScriptedSystem2):
        def run(self, prompt, **kw):
            raise RuntimeError("upstream error")

    archive = Archive(tmp_path / "a")
    archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")
    cfg = LoopConfig(n_iterations=0, seeds_train=[0], seeds_heldout=[], clock_mode="virtual",
                     budget=Budget(max_wall_s=30), harness_timeout_s=30)
    loop = MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=10, **k), game_kwargs={},
                           task_md="t", contract_doc="c", proposer=Silent(), system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    loop.run()
    loop.reflect(2, archive.candidates())
    text = (archive.root / "lessons.md").read_text()
    assert "## Generation 2 (automatic summary" in text and "candidates/0000-tier2_idle" in text and "mean score" in text


def test_a_reflection_that_rewrites_lessons_keeps_the_earlier_sections(tmp_path):
    class Rewriter(ScriptedSystem2):
        def run(self, prompt, *, cwd, **kw):
            (cwd / "lessons.md").write_text("## Generation 4\n- only the new section\n")
            return System2Result(text="done", session_id="x")

    from metaharness.models.base import System2Result
    archive = Archive(tmp_path / "a")
    archive.add_seed(BASE / "tier2_idle.py", label="tier2_idle")
    (archive.root / "lessons.md").write_text("## Generation 3\n- an earlier lesson\n")
    cfg = LoopConfig(n_iterations=0, seeds_train=[0], seeds_heldout=[], clock_mode="virtual",
                     budget=Budget(max_wall_s=30), harness_timeout_s=30)
    loop = MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=10, **k), game_kwargs={},
                           task_md="t", contract_doc="c", proposer=Rewriter(), system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    loop.run()
    loop.reflect(4, archive.candidates())
    text = (archive.root / "lessons.md").read_text()
    assert "an earlier lesson" in text and "only the new section" in text
    assert text.index("## Generation 3") < text.index("## Generation 4")


def test_older_missing_reflections_get_summaries_and_only_the_latest_a_model_call(tmp_path):
    from metaharness.models.base import System2Result

    class Notes(ScriptedSystem2):
        def run(self, prompt, **kw):
            self.calls.append(prompt)
            return System2Result(text="- the latest generation", session_id="x")

    archive = Archive(tmp_path / "a")
    for gen in (0, 1, 2):
        c = archive.new_candidate()
        c.harness_path.write_text(IDLE + f"\n# g{gen}\n")
        (c.dir / "directive.json").write_text(json.dumps({"mode": "explore", "parents": [], "generation": gen}))
    cfg = LoopConfig(n_iterations=3, seeds_train=[0], seeds_heldout=[], clock_mode="virtual",
                     budget=Budget(max_wall_s=30), harness_timeout_s=30, preflight=False)
    notes = Notes()
    loop = MetaHarnessLoop(archive=archive, game_factory=lambda **k: SnakeGame(max_ticks=10, **k), game_kwargs={},
                           task_md="t", contract_doc="c", proposer=notes, system2=None,
                           system1=HeuristicSystem1(), config=cfg, game_name="snake")
    loop.run()
    text = (archive.root / "lessons.md").read_text()
    assert len(notes.calls) == 1 and "## Generation 3" in text
    assert text.count("automatic summary: no reflection was recorded") == 2
