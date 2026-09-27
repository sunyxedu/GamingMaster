"""The DouZero self-play generator (scripts/generate_doudizhu.py): its bidding
heuristic, its reading of ambiguous card sets, and — the part that decides whether
the DouZero agents play at full strength — the features it feeds them: at every
decision of real generated games, the ``InfoSet`` the generator rebuilds from the
gmrules state gives exactly the features (``get_obs``: ``x_no_action`` and the move
history ``z``) of DouZero's own ``GameEnv`` stepped through the same game.

The DouZero sources are fetched (pinned commit, SHA-256 checked) into
``$GM_REFERENCE_CACHE`` by the generator / scripts/reference/doudizhu_douzero_dump.py;
the feature test runs when they are cached (or with ``GM_REFERENCE=1``, which fetches
them), the end-to-end generation test when the checkpoints are cached too.
"""

from __future__ import annotations

import collections
import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import generate_doudizhu as gen

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "doudizhu" / "douzero_selfplay.jsonl.gz"


def _douzero_pkg():
    pkg = gen.cache_root() / f"douzero-{gen.DOUZERO_COMMIT[:12]}"
    have = all((pkg / rel).exists() for rel in gen.DOUZERO_FILES)
    if not have and os.environ.get("GM_REFERENCE") != "1":
        pytest.skip("DouZero sources not cached (run the generator once, or GM_REFERENCE=1)")
    return gen.fetch_douzero()


def _checkpoints_cached(agents=("adp", "wp")) -> bool:
    root = gen.cache_root() / f"douzero-baselines-{gen.WEIGHTS_REVISION[:12]}"
    return all((root / f"douzero_{a.upper()}" / f"{p}.ckpt").exists()
               for a in agents for p in gen.POSITIONS)


def test_bidding_heuristic():
    # card ids: rank c % 13 (0 = 3 ... 12 = 2), 52 / 53 jokers
    rocket = [52, 53] + list(range(0, 12)) + [13, 14, 15]              # no 2, no bomb
    assert gen.hand_strength(rocket) == 8 and gen.max_bid(8) == 3
    twos = [12, 25, 38] + list(range(0, 11)) + [13, 14, 15]          # three 2s: 6
    assert gen.hand_strength(twos) == 6 and gen.max_bid(6) == 2
    bomb2 = [12, 25, 38, 51] + list(range(0, 13))[:-1] + [13]         # a bomb of 2s: 6, not 14
    assert gen.hand_strength(bomb2) == 6
    small = [52, 12] + list(range(0, 11)) + [13, 14, 15, 16]          # small joker + one 2
    assert gen.hand_strength(small) == 5
    assert [gen.max_bid(s) for s in (0, 2, 3, 4, 5, 6, 7)] == [0, 0, 1, 1, 2, 2, 3]


def test_play_cards_and_readings():
    assert gen.play_cards("333444-5(BWJ)") == (3, 3, 3, 4, 4, 4, 5, 20)
    assert gen.play_cards("(BWJ)(CJ)") == (20, 30) and gen.play_cards("Pass") == ()
    assert gen.play_cards("TJQKA") == (10, 11, 12, 13, 14) and gen.play_cards("22") == (17, 17)

    class MD:        # DouZero's detector verdicts for the ambiguous 333444555666
        @staticmethod
        def get_move_type(move):
            return {"type": 10, "rank": 3, "len": 4}

    names = ["333444555666", "333444555-666", "444555666-333"]
    assert gen.reading(names, gen.play_cards(names[0]), MD) == "333444555666"
    MD.get_move_type = staticmethod(lambda m: {"type": 11, "rank": 4, "len": 3})
    assert gen.reading(names, gen.play_cards(names[0]), MD) == "444555666-333"


def _fixture():
    with gzip.open(FIXTURE, "rt") as fh:
        return [json.loads(line) for line in fh]


def test_generator_features_equal_douzero_gameenv():
    """For every play decision of the fixture games: the generator's InfoSet gives
    DouZero's exact ``get_obs`` features (and the recorded move is one of DouZero's
    legal moves but for the documented kicker differences)."""
    pkg = str(_douzero_pkg())
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from douzero.env.env import get_obs
    from douzero.env.game import GameEnv

    class Scripted:
        def __init__(self):
            self.moves = []

        def act(self, infoset):
            return self.moves.pop(0)

    checked = collections.Counter()
    for rec in _fixture():
        if not rec["plays"]:
            continue
        landlord = [s for s, b in rec["auction"] if b > 0][-1]
        hands = [h + rec["kitty"] if s == landlord else h for s, h in enumerate(rec["hands"])]
        tr = gen.Tracker(hands, landlord)
        vals = {s: sorted(gen.card_value(c) for c in hands[s]) for s in range(3)}
        pos = tr.pos
        player = Scripted()
        env = GameEnv({p: player for p in gen.POSITIONS})
        kitty = sorted(gen.card_value(c) for c in rec["kitty"])
        env.card_play_init({pos[s]: list(vals[s]) for s in range(3)}
                           | {"three_landlord_cards": kitty})
        for seat, name in rec["plays"]:
            assert env.acting_player_position == pos[seat]
            theirs = env.game_infoset
            cards = list(gen.play_cards(name))
            ours = tr.infoset(seat, [cards])
            a = get_obs(_with_legal(theirs, [cards]))
            b = get_obs(ours)
            for k in ("x_no_action", "z", "x_batch", "z_batch"):
                assert np.array_equal(a[k], b[k]), (rec["id"], seat, name, k)
            checked["decisions"] += 1
            if cards not in theirs.legal_actions:
                checked["gmrules_only"] += 1           # DouZero's engine would refuse it
                break
            player.moves.append(cards)
            env.step()
            tr.play(seat, tuple(cards))
        else:
            assert env.game_over
            winner = env.get_winner()
            assert (rec["returns"][landlord] > 0) == (winner == "landlord"), rec["id"]
            assert env.get_bomb_num() == tr.bombs
            checked["games"] += 1
    assert checked["decisions"] > 1000, checked
    assert checked["gmrules_only"] <= checked["decisions"] // 1000, checked


def _with_legal(infoset, legal):
    s = copy.copy(infoset)
    s.legal_actions = legal
    return s


@pytest.mark.skipif(not _checkpoints_cached() and os.environ.get("GM_REFERENCE") != "1",
                    reason="DouZero checkpoints not cached (run the generator once, or "
                           "GM_REFERENCE=1)")
def test_generate_and_convert(tmp_path):
    """A few games end to end with the real agents: deterministic per (seed, index)
    whatever the part size, and every record verified by the converter."""
    gmdata = ROOT / "data_processing" / "target" / "release" / "gmdata"
    if not gmdata.exists():
        pytest.skip("gmdata not built (cargo build --release -p gmdata)")
    run = [sys.executable, str(ROOT / "scripts" / "generate_doudizhu.py"), "--games", "6",
           "--seed", "5", "--q_top", "3"]
    subprocess.run(run + ["--out", str(tmp_path / "a"), "--part_size", "6"], check=True,
                   capture_output=True, text=True, timeout=600)
    subprocess.run(run + ["--out", str(tmp_path / "b"), "--part_size", "4"], check=True,
                   capture_output=True, text=True, timeout=600)

    def load(d):
        return [json.loads(line) for p in sorted(d.glob("*.jsonl.gz"))
                for line in gzip.open(p, "rt")]

    a, b = load(tmp_path / "a"), load(tmp_path / "b")
    assert a == b and len(a) == 6
    assert all(r["meta"]["bidding"] == "heuristic_v1" for r in a)
    out = subprocess.run([str(gmdata), "doudizhu", "convert", "--input", str(tmp_path / "a"),
                          "--out", str(tmp_path / "shards"), "--hq"], check=True,
                         capture_output=True, text=True)
    st = json.loads(out.stdout.strip().splitlines()[-1])
    assert st["converted"] == 6 and st["illegal"] == st["returns_mismatch"] == 0
