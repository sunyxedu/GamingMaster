"""training/arena.py (game-agnostic match runner + statistics) and its othello
instance (training/arena_othello.py: the model's pure policy vs Egaroucid).

- statistics: Wilson intervals, the Elo transform, the Bradley–Terry ladder fit
  recovers known ratings from synthetic games (anchor held fixed, the prior draws
  keep a sweep finite), refuses a player no game connects to an anchor, and its
  bootstrap CI covers the truth;
- openings: distinct non-terminal positions, reproducible from the seed;
- pairing: an opening pair with a failed game is dropped whole;
- othello: a tiny random network plays only legal moves through the runner
  (against the random opponent: no engine needed), the rules' forced passes are
  never asked of a player, and every game ends in a scored terminal position;
- Egaroucid round trip (only where the engine is installed: scripts/engines/
  fetch_engines.sh egaroucid): the CLI end to end on one opening at level 1.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from training import arena

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_wilson_and_elo_transform():
    lo, hi = arena.wilson_interval(0.5, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.19, abs=0.01)
    assert arena.wilson_interval(0.0, 8)[0] == 0.0 and arena.wilson_interval(0.3, 0) is None
    assert arena.elo_from_score(0.5) == 0.0
    assert arena.elo_from_score(10 / 11) == pytest.approx(400.0, abs=1e-6)
    assert math.isinf(arena.elo_from_score(1.0))
    assert math.isfinite(arena.elo_from_score(1.0, eps=0.01))


def _synthetic_games(true, pairs, n_per, seed=0):
    rng = random.Random(seed)
    games, k = [], 0
    for a, b in pairs:
        p = 1.0 / (1.0 + 10 ** ((true[b] - true[a]) / 400.0))
        for _ in range(n_per):
            games.append((a, b, float(rng.random() < p), k))
            k += 1
    return games


def test_bradley_terry_recovers_known_ratings():
    true = {"L1": 0.0, "L2": 200.0, "L3": 450.0, "model": 300.0}
    pairs = [("L1", "L2"), ("L2", "L3"), ("model", "L1"), ("model", "L2"), ("model", "L3")]
    games = _synthetic_games(true, pairs, 2000)
    fit = arena.fit_ratings(games, {"L1": 0.0}, prior_draws=1.0)
    for p, v in true.items():
        assert fit["ratings"][p] == pytest.approx(v, abs=25.0), p
    assert fit["ratings"]["L1"] == 0.0 and "L1" not in fit["se"]
    assert all(0 < fit["se"][p] < 25 for p in ("L2", "L3", "model"))
    # a clean sweep stays finite with the prior draws
    sweep = [("model", "L1", 1.0, i) for i in range(20)]
    assert math.isfinite(arena.fit_ratings(sweep, {"L1": 0.0})["ratings"]["model"])
    with pytest.raises(ValueError, match="connect"):
        arena.fit_ratings(games + [("x", "y", 1.0, 0)], {"L1": 0.0})


def test_ladder_confidence_intervals_cover_the_truth():
    """95% intervals over 12 synthetic ladders: each kind covers the true rating in
    most of them (a single draw misses 5% of the time by design)."""
    true = {"L1": 0.0, "L2": 300.0, "model": 150.0}
    fisher = boot = 0
    for seed in range(12):
        games = _synthetic_games(true, [("L1", "L2"), ("model", "L1"), ("model", "L2")],
                                 300, seed=seed)
        m = arena.ladder_fit(games, {"L1": 0.0}, n_boot=100, seed=seed)["ratings"]["model"]
        fisher += m["ci95_fisher"][0] < true["model"] < m["ci95_fisher"][1]
        boot += m["ci95_bootstrap"][0] < true["model"] < m["ci95_bootstrap"][1]
    assert fisher >= 9 and boot >= 9, (fisher, boot)


# ---------------------------------------------------------------------------
# openings, pairing
# ---------------------------------------------------------------------------

def _othello():
    from training import arena_othello

    return arena_othello


def test_random_openings_are_distinct_and_reproducible():
    rules = _othello().OthelloRules()
    a = arena.random_openings(rules, 12, 4, seed=3)
    assert a == arena.random_openings(rules, 12, 4, seed=3)
    keys = {rules.position_key(arena.replay(rules, o)) for o in a}
    assert len(keys) == 12
    assert all(not rules.is_terminal(arena.replay(rules, o)) for o in a)
    with pytest.raises(ValueError, match="distinct"):
        arena.random_openings(rules, 50, 1, seed=0)          # only 4 one-ply openings


def test_incomplete_pairs_are_dropped_whole():
    def rec(opening, seating, ok=True):
        return arena.GameRecord(match="a vs b", opening=opening, seating=seating,
                                seats=("a", "b"), moves=[],
                                scores=[1.0, 0.0] if ok else None,
                                error=None if ok else "engine died")
    recs = [rec(0, 0), rec(0, 1), rec(1, 0), rec(1, 1, ok=False), rec(2, 0)]
    good, failed, incomplete = arena.complete_pairs(recs, 2)
    assert list(good) == [0] and failed == 1 and incomplete == 2


# ---------------------------------------------------------------------------
# othello: the model side through the runner
# ---------------------------------------------------------------------------

def test_othello_model_plays_legal_moves_against_random():
    ao = _othello()
    from model.games import build_game_network

    torch.manual_seed(0)
    net = build_game_network("othello", d_model=32, n_heads=4, d_ff=64, n_st_layers=1,
                             dropout=0.0, core_type="transformer", core_layers=1,
                             temporal_mixer="recurrent").eval()
    rules = ao.OthelloRules()
    player = arena.NetworkPlayer(net, ao.OthelloCodec(), device=torch.device("cpu"))
    rand = arena.RandomOpponent(rules, seed=1)
    openings = arena.random_openings(rules, 3, 4, seed=0)
    specs = arena.match_specs("model", "random", len(openings))
    records = arena.run_games(rules, specs, openings, {"model": player, "random": rand})
    assert len(records) == 6 and all(r.ok for r in records)
    for r in records:
        # the recorded game is legal throughout and ends in a scored terminal position
        moves = [ao.PASS if m == "pass" else ao.parse_square(m) for m in r.moves]
        st = arena.replay(rules, moves)
        assert rules.is_terminal(st) and r.scores == rules.seat_scores(st)
        assert sum(r.scores) == pytest.approx(1.0)
    good, failed, incomplete = arena.complete_pairs(records, 2)
    assert len(good) == 3 and failed == 0 and incomplete == 0


# ---------------------------------------------------------------------------
# Egaroucid round trip
# ---------------------------------------------------------------------------

def _egaroucid():
    for p in (os.environ.get("GM_EGAROUCID"),
              os.path.join(os.environ.get("GM_ENGINES", ""), "bin", "egaroucid"),
              os.path.expanduser("~/gm-engines/bin/egaroucid")):
        if p and os.path.exists(p):
            return p
    return None


@pytest.mark.skipif(_egaroucid() is None, reason="Egaroucid not installed "
                    "(scripts/engines/fetch_engines.sh egaroucid)")
def test_egaroucid_end_to_end(tmp_path):
    from data_processing.games.dataset import save_shard
    from tests.test_core_moe import _board_samples
    from training import train_multi_game

    (tmp_path / "oth").mkdir()
    save_shard(str(tmp_path / "oth" / "a.pt"), "othello", _board_samples("othello", 6, 6))
    train_multi_game.main([
        "--data", f"othello={tmp_path / 'oth'}", "--steps", "2", "--batch_size", "2",
        "--eval_every", "0", "--device", "cpu", "--save_dir", str(tmp_path / "ck"),
        "--override", "d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "core_type=transformer", "core_layers=1", "temporal_mixer=recurrent"])
    out = tmp_path / "res.json"
    arena.main(["othello", "--ckpt", str(tmp_path / "ck" / "multi_final.pt"),
                "--levels", "1", "2", "--openings", "2", "--workers", "1", "--device", "cpu",
                "--out", str(out), "--engine", _egaroucid(),
                "--ladder_cache", str(tmp_path / "ladder.jsonl")])
    import json
    res = json.loads(out.read_text())
    games = [json.loads(line) for line in (tmp_path / "res.games.jsonl").read_text().splitlines()]
    assert len(games) == 2 * 2 * 2 and all(g["error"] is None for g in games)   # 2 levels
    assert res and (tmp_path / "ladder.jsonl").exists()                         # calibration
