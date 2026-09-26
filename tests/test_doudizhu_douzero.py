"""Differential tests of the gmrules Dou Dizhu PLAY PHASE against DouZero's game engine
(kwai/DouZero ``douzero/env``, the rules the DouZero agents that generate our IL data
were trained with).  DouZero has no auction and no spring; a DouZero game is mapped to
a gmrules trajectory whose deal gives the landlord seat 0 (it bids 3 at once).

At every decision the legal moves are compared as card multisets (gmrules' full plays,
kicker decisions walked, projected to cards; DouZero's generated moves) and the move
DouZero played is applied (its interpretation, for card sets that are several
airplanes, from DouZero's move type); winners and bomb counts are compared at the end.

Documented rule differences (excluded, counted, never silently ignored):
  * DouZero lets the rocket be kickers (four+2 solos, airplane solos) and a bomb be
    an airplane's solo wings; the standard rules (OpenSpiel's kicker rules) do not;
  * airplanes whose solo wings form a trio: DouZero ranks airplanes+solos by their
    highest trio and its detector rejects some such plays; the standard rules rank by
    the chain (OpenSpiel's interpretation).

Fixture tests/fixtures/doudizhu/douzero_reference.json.gz: 40 DouZero random playouts
(scripts/reference/doudizhu_douzero_dump.py --games 40 --seed 0); live
(``GM_REFERENCE=1``, network the first time): fresh playouts, and gmrules' random
playouts replayed through DouZero.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gmsim
import pytest

from tests.test_doudizhu_reference import full_plays, reference_events

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "doudizhu" / "douzero_reference.json.gz"
DUMP = ROOT / "scripts" / "reference" / "doudizhu_douzero_dump.py"
GAME = "doudizhu"
STANDARD: Dict = {}
POS = {"landlord": 0, "landlord_down": 1, "landlord_up": 2}
#: DouZero card value of each rank (3..A, 2, black / coloured joker)
VALUE = list(range(3, 15)) + [17, 20, 30]
CHAR = dict(zip(VALUE, list("3456789TJQKA2") + ["(BWJ)", "(CJ)"]))
TYPE_SERIAL_TRIPLE, TYPE_SERIAL_3_1 = 10, 11


def dz_cards(play: str) -> Tuple[int, ...]:
    """A gmrules play string as DouZero cards (sorted rank values)."""
    s = play.replace("-", "").replace("(BWJ)", "B").replace("(CJ)", "R")
    char_value = {c: v for v, c in CHAR.items() if len(c) == 1}
    char_value.update({"B": 20, "R": 30})
    return tuple(sorted(char_value[c] for c in s)) if play != "Pass" else ()


def card_value(card: int) -> int:
    """gmrules / OpenSpiel card id → DouZero value."""
    return VALUE[card - 52 + 13 if card >= 52 else card % 13]


def deal_events(deal: Dict) -> List[Tuple[int, int]]:
    """A gmrules deal giving DouZero's landlord seat 0 (+ its bid of 3): face-up position
    0, seat 1 = landlord_down, seat 2 = landlord_up, the kitty = three_landlord_cards."""
    taken = collections.Counter()

    def card_of(v: int) -> int:
        if v >= 20:
            return 52 if v == 20 else 53
        r = VALUE.index(v)
        c = taken[v] * 13 + r
        taken[v] += 1
        return c

    kitty = collections.Counter(deal["three_landlord_cards"])
    landlord = list((collections.Counter(deal["landlord"]) - kitty).elements())
    hands = [landlord, deal["landlord_down"], deal["landlord_up"]]
    cards = [[card_of(v) for v in sorted(h)] for h in hands]
    assert all(len(h) == 17 for h in cards)
    ev = [(-1, 0)] + [(-1, 51 + cards[k % 3][k // 3]) for k in range(51)]
    return ev + [(0, 3)]


def rocket_kicker(m) -> bool:
    return 20 in m and 30 in m and len(m) > 2


def bomb_wing(m) -> bool:
    return len(m) >= 16 and 4 in collections.Counter(m).values()


def trio_wings(m) -> bool:
    """An airplane with solo wings among which a trio (more trios than the chain)."""
    n3 = sum(1 for v in collections.Counter(m).values() if v == 3)
    return len(m) >= 12 and len(m) % 4 == 0 and 4 * n3 > len(m)


def airplane_solo_shape(m) -> bool:
    n3 = sum(1 for v in collections.Counter(m).values() if v >= 3)
    return len(m) >= 8 and len(m) % 4 == 0 and 4 * n3 >= len(m)


def pick(names: List[str], dz_type) -> str:
    """gmrules' interpretation of a DouZero move (card sets that are several airplanes):
    a serial triple is the kicker-free airplane, a serial 3+1 the chain DouZero names."""
    if len(names) == 1:
        return names[0]
    t, rank, length = dz_type
    if t == TYPE_SERIAL_TRIPLE:
        return next(n for n in names if "-" not in n)
    if t == TYPE_SERIAL_3_1:
        chain = "".join(CHAR[v] * 3 for v in range(rank, rank + length))
        return next(n for n in names if n.split("-")[0] == chain)
    return sorted(names)[0]           # a move DouZero's detector rejects (trio wings)


def compare_game(game, config=STANDARD) -> Dict[str, int]:
    st = gmsim.State(GAME, config)
    for e in deal_events(game["deal"]):
        st.apply(e)
    stats = collections.Counter()
    rival, passes = None, 0
    landlord_plays, farmer_plays, bombs = 0, 0, 0
    for i, step in enumerate(game["steps"]):
        seat = POS[step["pos"]]
        kind, seats = st.turn()
        assert kind == "act" and seats == [seat], (i, kind, seats, step["pos"])
        plays = full_plays(st, seat)
        ours: Dict[Tuple[int, ...], List[str]] = collections.defaultdict(list)
        for name in plays:
            ours[dz_cards(name)].append(name)
        theirs = {tuple(m) for m in step["legal"]}

        def excluded(m) -> str:
            if rocket_kicker(m) or bomb_wing(m):
                return "douzero_kickers"
            if rival is not None and (trio_wings(rival) and airplane_solo_shape(m) and len(m) == len(rival)
                                      or airplane_solo_shape(rival) and trio_wings(m)):
                return "trio_wings"
            return ""

        for m in theirs - set(ours):
            why = excluded(m)
            assert why, (i, "DouZero only", m, step["hand"])
            stats[why] += 1
        for m in set(ours) - theirs:
            assert excluded(m) == "trio_wings", (i, "gmrules only", m, ours[m], step["hand"])
            stats["trio_wings"] += 1
        if "rejected" in step:
            assert excluded(tuple(step["rejected"])) == "trio_wings", step["rejected"]
            stats["rejected"] += 1
            return stats
        move = tuple(step["move"])
        if move not in ours:
            stats["diverged"] += 1                   # DouZero played a DouZero-only move
            return stats
        for a in plays[pick(ours[move], step["type"])]:
            st.apply_action(seat, a)
        stats["steps"] += 1
        if move:
            rival, passes = move, 0
            bombs += (len(set(move)) == 1 and len(move) == 4) or move == (20, 30)
            if seat == 0:
                landlord_plays += 1
            else:
                farmer_plays += 1
        else:
            passes += 1
            if passes == 2:
                rival, passes = None, 0
    if game["over"]:
        kind, _ = st.turn()
        assert kind == "terminal"
        r = st.returns()
        assert (r[0] > 0) == (game["winner"] == "landlord")
        assert bombs == game["bomb_num"]
        spring = landlord_plays == 1 or farmer_plays == 0
        assert abs(r[0]) == 2 * 3 * 2 ** (bombs + spring)
    return stats


with gzip.open(FIXTURE, "rt") as _fh:
    _FIXTURE = json.load(_fh)


def test_fixture_play_phases_match_douzero():
    total = collections.Counter()
    for game in _FIXTURE["games"]:
        total.update(compare_game(game))
    assert total["steps"] > 1500 and total["diverged"] == 0, total


def test_fixture_covers_the_move_types():
    types = collections.Counter(s["type"][0] for g in _FIXTURE["games"] for s in g["steps"])
    # pass, single, pair, trio, bomb, rocket, 3+1, 3+2, serial single, serial 3+1, 4+2
    for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13):
        assert types[t] > 0, t
    assert {g["winner"] for g in _FIXTURE["games"]} == {"landlord", "farmer"}


def test_rule_difference_predicates():
    assert rocket_kicker((3, 3, 3, 3, 20, 30)) and not rocket_kicker((20, 30))
    assert bomb_wing((3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 9, 9, 9, 9))
    assert trio_wings((3, 3, 3, 4, 4, 4, 5, 5, 5, 9, 9, 9)) and not trio_wings((3, 3, 3, 4, 4, 4, 5, 6))
    assert dz_cards("333444-5(BWJ)") == (3, 3, 3, 4, 4, 4, 5, 20) and dz_cards("Pass") == ()
    ev = deal_events(_FIXTURE["games"][0]["deal"])
    assert len(ev) == 53 and ev[-1] == (0, 3)


live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1",
                          reason="live reference tests: set GM_REFERENCE=1 (network the first time)")


def _run_dump(tmp_path, *args):
    out = tmp_path / "dz.json.gz"
    subprocess.run([sys.executable, str(DUMP), "--out", str(out), *args], check=True,
                   capture_output=True, text=True, timeout=1800)
    with gzip.open(out, "rt") as fh:
        return json.load(fh)


def douzero_game(events):
    """A gmrules trajectory's play phase in DouZero's format, and the landlord seat
    (None: nobody bid)."""
    ref = reference_events(STANDARD, events)
    cards = [int(i) - 51 for s, i in ref[1:52]]
    kitty = sorted(set(range(54)) - set(cards))
    landlord, moves = None, []
    for seat, x in ref[52:]:
        if isinstance(x, int):          # auction: the last bidder is the landlord
            if x > 0:
                landlord = seat
            continue
        moves.append((seat, [] if x == "Pass" else list(dz_cards(x))))
    if landlord is None:
        return None, None
    hands = [[card_value(c) for i, c in enumerate(cards) if i % 3 == s] for s in range(3)]
    pos = {landlord: "landlord", (landlord + 1) % 3: "landlord_down", (landlord + 2) % 3: "landlord_up"}
    deal = {"landlord": sorted(hands[landlord] + [card_value(c) for c in kitty]),
            "landlord_down": sorted(hands[(landlord + 1) % 3]),
            "landlord_up": sorted(hands[(landlord + 2) % 3]),
            "three_landlord_cards": sorted(card_value(c) for c in kitty)}
    return {"deal": deal, "moves": [[pos[s], m] for s, m in moves]}, landlord


@live
def test_live_douzero_playouts(tmp_path):
    data = _run_dump(tmp_path, "--games", "300", "--seed", "5")
    total = collections.Counter()
    for game in data["games"]:
        total.update(compare_game(game))
    assert total["steps"] > 10000


@live
def test_live_gmrules_play_phases_replayed_in_douzero(tmp_path):
    """Our random playouts through DouZero: every move accepted (but the documented
    trio-wing cases), same winners and bomb counts."""
    games, ours = [], []
    for seed in range(300):
        ev = gmsim.random_trajectory(GAME, STANDARD, seed)
        g, landlord = douzero_game(ev)
        if g is not None:
            games.append(g)
            ours.append((ev, landlord))
    (tmp_path / "ours.json").write_text(json.dumps({"games": games}))
    data = _run_dump(tmp_path, "--replay", str(tmp_path / "ours.json"))
    rejected = 0
    for game, (ev, landlord) in zip(data["games"], ours):
        if any("rejected" in s for s in game["steps"]):
            rej = next(s for s in game["steps"] if "rejected" in s)
            assert trio_wings(tuple(rej["rejected"])) or any(trio_wings(tuple(s["move"])) for s in game["steps"]
                                                             if "move" in s), rej
            rejected += 1
            continue
        assert game["over"]
        r = gmsim.replay(GAME, STANDARD, ev)
        assert (r["returns"][landlord] > 0) == (game["winner"] == "landlord")
    assert rejected <= len(games) // 20
