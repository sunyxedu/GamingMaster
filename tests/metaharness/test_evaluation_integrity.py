"""Regressions for comparisons that previously produced misleading scores."""
from __future__ import annotations

import json
import threading
import time

import pytest

from metaharness.core.archive import Archive
from metaharness.core.contract import Budget, BudgetExceeded
from metaharness.core.runner import evaluate_harness
from metaharness.games.base import ActionSpec
from metaharness.models.base import ModelUnavailable
from metaharness.models.mock import ScriptedSystem2


class CounterGame:
    name = "counter"
    action_spec = ActionSpec(names=["add"], descriptions={"add": "increase score"})

    def reset(self, seed):
        self.score = 0
        self.done = False
        return self.observe()

    def observe(self):
        return {"score": self.score}

    def step(self, action):
        self.score += int(action)
        return self.observe()

    def info(self):
        return {"actions_used": 0}

    def describe(self):
        return "counter"


def evaluate(tmp_path, code, **kwargs):
    p = tmp_path / "harness.py"
    p.write_text(code)
    return evaluate_harness(p, CounterGame, [0, 1], system1=None,
                            out_dir=tmp_path / "result", **kwargs)


def test_cold_starts_reset_memory_and_module_globals(tmp_path):
    code = '''
from types import SimpleNamespace
n = 0
def build(ctx):
    def run(game):
        global n
        n += 1
        ctx.memory['n'] = ctx.memory.get('n', 0) + 1
        ctx.act(n + ctx.memory['n'])
    return SimpleNamespace(run_episode=run)
'''
    cold = evaluate(tmp_path, code, system2=None)
    assert [e.score for e in cold.episodes] == [2, 2]
    shared = evaluate(tmp_path, code, system2=None, memory_mode="shared")
    assert [e.score for e in shared.episodes] == [2, 3]
    assert shared.metadata["memory_mode"] == "shared"
    assert len(cold.metadata["harness_sha256"]) == 64


def test_swallowed_unavailable_is_still_fatal(tmp_path):
    class Unavailable(ScriptedSystem2):
        def run(self, *args, **kwargs):
            raise ModelUnavailable("quota exhausted")

    r = evaluate(tmp_path, '''
from types import SimpleNamespace
def build(ctx):
    def run(game):
        try: ctx.think('test')
        except RuntimeError: pass
    return SimpleNamespace(run_episode=run)
''', system2=Unavailable())
    assert r.fatal == "model unavailable: quota exhausted"
    assert not r.episodes


def test_wall_budget_bounds_build_and_late_actions(tmp_path):
    r = evaluate(tmp_path, '''
import time
from types import SimpleNamespace
def build(ctx):
    time.sleep(.1)
    ctx.act(99)
    return SimpleNamespace(run_episode=lambda game: None)
''', system2=None, budget_per_episode=Budget(max_wall_s=.025), harness_timeout_s=2)
    # the deadline closes the episode (score kept, no error); the late action is refused
    assert all(e.error is None and e.ended_by == "deadline" for e in r.episodes)
    assert all(e.wall_s < .5 for e in r.episodes)
    p = tmp_path / "result" / "trace_seed0.jsonl"
    before = p.read_text()
    time.sleep(.15)
    assert p.read_text() == before
    assert all(e.score == 0 for e in r.episodes)


def test_request_is_archived_even_when_response_misses_episode_deadline(tmp_path):
    class Slow(ScriptedSystem2):
        def run(self, *args, **kwargs):
            time.sleep(.08)
            return super().run(*args, **kwargs)

    r = evaluate(tmp_path, '''
from types import SimpleNamespace
def build(ctx):
    return SimpleNamespace(run_episode=lambda game: ctx.think('slow request'))
''', system2=Slow(), budget_per_episode=Budget(max_wall_s=.025))
    assert all(e.error is None and e.ended_by == 'deadline' for e in r.episodes)
    rows = [json.loads(line) for line in
            (tmp_path / 'result' / 'trace_seed0.jsonl').read_text().splitlines()]
    request = next(row for row in rows if row['type'] == 'model_request')
    assert request['prompt'] == 'slow request' and request['kind'] == 'system2'
    assert rows[-1]['type'] == 'episode_end'


def test_call_reservation_is_atomic_and_does_not_count_rejected_calls():
    b = Budget(max_system2_calls=3)
    accepted = []
    def reserve():
        try:
            b.reserve("system2")
            accepted.append(1)
        except BudgetExceeded:
            pass
    workers = [threading.Thread(target=reserve) for _ in range(20)]
    for t in workers: t.start()
    for t in workers: t.join()
    assert len(accepted) == b.system2_calls == 3


def test_partial_or_failed_evaluations_cannot_win(tmp_path):
    a = Archive(tmp_path / "archive")
    good = a.new_candidate()
    def put(c, **extra):
        c.eval_path.write_text(json.dumps({"mean_score": 1, "n_episodes": 2,
                                         "metadata": {"seeds": [0, 1]}, **extra}))
    put(good)
    put(a.new_candidate(), mean_score=100, fatal="model unavailable")
    put(a.new_candidate(), mean_score=100, n_episodes=1)
    put(a.new_candidate(), mean_score=100, errors=["timeout"])
    put(a.new_candidate(), mean_score=float("nan"))
    assert a.best().idx == good.idx


def test_action_efficiency_does_not_reward_early_realtime_death(tmp_path):
    a = Archive(tmp_path / "archive")
    short, long = a.new_candidate(), a.new_candidate()
    for c, steps in ((short, 3), (long, 120)):
        c.eval_path.write_text(json.dumps({"mean_score": 1, "n_episodes": 1,
                                          "mean_steps": steps, "stale_fraction": 0.0}))
    assert a.best().idx == long.idx
    for c in (short, long):
        value = c.eval()
        value["stale_fraction"] = None
        value["metadata"] = {"game_kind": "turn"}
        c.eval_path.write_text(json.dumps(value))
    assert a.best().idx == short.idx


def test_manifest_prevents_reusing_incomparable_scores(tmp_path):
    from metaharness.cli import main
    args = ["loop", "--tier", "1", "--game", "rulepuzzle", "--system1", "none",
            "--system2", "scripted", "--proposer", "scripted", "--clock", "virtual",
            "--iters", "0", "--seeds", "0", "--heldout", "", "--s2-budget", "2",
            "--name", "test", "--runs-dir", str(tmp_path)]
    main(args)
    with pytest.raises(ValueError, match="configuration changed"):
        main(args + ["--memory-mode", "shared"])
    with pytest.raises(ValueError, match="overlap"):
        main(args + ["--heldout", "0"])
    a = Archive(tmp_path / "test")
    held = a.candidates()[0].dir / "heldout"
    held.mkdir()
    (held / "eval.json").write_text('{}')
    with pytest.raises(ValueError, match="held-out results"):
        main(args + ["--iters", "1"])


def test_hidden_rule_diagnostics_do_not_reach_harness_or_archive(tmp_path):
    from metaharness.core.runner import _GameView
    from metaharness.testbeds.rulepuzzle import RulePuzzleGame
    game = RulePuzzleGame()
    assert 'move_map' in game.info()
    view = _GameView(game, threading.Event())
    assert 'move_map' not in view.info() and 'family' not in view.info()
    code = '''
from types import SimpleNamespace
def build(ctx):
    return SimpleNamespace(run_episode=lambda game: ctx.act(1))
'''
    r = evaluate(tmp_path, code, system2=None)
    assert 'actions_used' not in r.episodes[0].info
    p = tmp_path / 'result' / 'trace_seed0.jsonl'
    assert sum(json.loads(line)['type'] == 'episode_start' for line in p.read_text().splitlines()) == 1
    # Re-evaluation replaces each episode trace instead of appending a second trajectory.
    evaluate(tmp_path, code, system2=None)
    assert sum(json.loads(line)['type'] == 'episode_start' for line in p.read_text().splitlines()) == 1


def test_game_reset_failure_is_an_episode_error(tmp_path):
    from metaharness.core.runner import evaluate_harness
    from metaharness.testbeds.snake import SnakeGame

    class Picky(SnakeGame):
        def reset(self, seed):
            if seed == 1:
                raise RuntimeError("scenario 1 is not admissible")
            return super().reset(seed)

    h = tmp_path / "h.py"
    h.write_text("class H:\n    def __init__(self, ctx): self.ctx = ctx\n"
                 "    def run_episode(self, game):\n        pass\n"
                 "def build(ctx): return H(ctx)\n")
    r = evaluate_harness(h, lambda **k: Picky(max_ticks=5, **k), [0, 1, 2], system2=None, system1=None,
                         clock_mode="virtual")
    assert r.fatal is None and len(r.episodes) == 3
    assert r.episodes[1].error and "not admissible" in r.episodes[1].error
    assert r.episodes[0].error is None and r.episodes[2].error is None
