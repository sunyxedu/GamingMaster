"""Seed baselines must be game-agnostic: they run (without KeyError) on every registered game."""
from __future__ import annotations

from pathlib import Path

import pytest

from metaharness.baselines._generic import obs_text, parse_action
from metaharness.core.runner import evaluate_harness
from metaharness.games.registry import make_game
from metaharness.models.mock import HeuristicSystem1, ScriptedSystem2

BASE = Path(__file__).resolve().parents[2] / "metaharness" / "baselines"


def test_parse_action_prefers_last_mention_and_keeps_args():
    names = ["ACTION1", "ACTION2", "ACTION6", "up", "keep"]
    assert parse_action("ACTION1 moved left. Now try ACTION2.", names) == "ACTION2"
    assert parse_action("click: ACTION6 10 12", names) == "ACTION6 10 12"
    assert parse_action("nothing useful", names, default="keep") == "keep"
    assert parse_action("I would go Up now", names) == "up"


def test_obs_text_generic():
    assert obs_text({"a": 1, "grid": "..\n.."}).endswith("..\n..")
    assert obs_text({"x": [1, 2]}) == '{"x":[1,2]}'


@pytest.mark.parametrize("game,kw", [("rulepuzzle", {"size": 6, "n_levels": 1, "budget_per_level": 6})])
def test_tier1_baseline_runs_on_any_turn_game(game, kw, tmp_path):
    r = evaluate_harness(BASE / "tier1_astra_naive.py", lambda **k: make_game(game, **k), [0],
                         system2=ScriptedSystem2(replies=["ACTION3"], latency_s=0.01), system1=None,
                         clock_mode="virtual", game_kwargs=kw)
    assert r.fatal is None and r.episodes[0].error is None and r.episodes[0].system2_calls >= 1


@pytest.mark.parametrize("game,kw", [("snake", {"max_ticks": 20}), ("lanerunner", {"max_ticks": 20})])
def test_tier2_baselines_run_on_any_realtime_game(game, kw, tmp_path):
    for name in ("tier2_jev_only.py", "tier2_astra_only.py", "tier2_idle.py"):
        r = evaluate_harness(BASE / name, lambda **k: make_game(game, **k), [0],
                             system2=ScriptedSystem2(replies=["stay"], latency_s=0.3),
                             system1=HeuristicSystem1(latency_s=0.2), clock_mode="virtual", game_kwargs=kw)
        assert r.fatal is None and r.episodes[0].error is None, (name, game, r.episodes[0].error)


def test_parse_action_handles_names_with_spaces_and_prefixes():
    from metaharness.baselines._generic import parse_action
    names = ["move 1", "move 2", "switch 3", "mega", "megaevolve"]
    assert parse_action("I will use move 2 now", names) == "move 2"
    assert parse_action("switch 3 then move 1", names) == "move 1"           # the last one mentioned
    assert parse_action("MOVE 2", names) == "move 2"                           # canonical spelling
    assert parse_action("megaevolve", names) == "megaevolve"                   # not the shorter prefix
    assert parse_action("ACTION6 10 12", ["ACTION6", "ACTION1"]) == "ACTION6 10 12"
    assert parse_action("nothing useful", names, default="move 1") == "move 1"
