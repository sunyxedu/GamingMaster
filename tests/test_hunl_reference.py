"""Randomized differential tests of the gmrules ``hunl`` engine (ACPC rules) against
PokerKit, an independent implementation (scripts/reference/pokerkit_hunl.py; MIT,
never a project dependency), in both directions and four configs — the ACPC game
(50 / 100 / 20000), the 2009 game (1 / 2 / 400), short stacks (1 / 2 / 13: all-ins,
all-in-only raises) and odd blinds (3 / 7 / 150: odd raise arithmetic):

- PokerKit's own random hands stepped through ``gmsim.State``: at every step the
  same turn kind, acting seat, the SAME legal action set (fold / check-call / every
  raise-to amount from PokerKit's minimum to its maximum, as total commitments),
  chance = every undealt card uniformly, and the same payoffs (showdowns: PokerKit's
  hand evaluator vs ours);
- gmrules' random hands (exact amounts and our abstract candidates' amounts)
  replayed through PokerKit: PokerKit accepts every event and pays the same.

Committed fixture: tests/fixtures/hunl/pokerkit_reference.json (write it with
``uv run python -c "from tests.test_hunl_reference import write_fixture;
write_fixture()"``); live (``GM_REFERENCE=1``, uv + network the first time): fresh
hands in both directions (``GM_REFERENCE_HANDS`` per variant, default 300).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hunl" / "pokerkit_reference.json"
SCRIPT = ROOT / "scripts" / "reference" / "pokerkit_hunl.py"
POKERKIT = "pokerkit==0.7.6"
VARIANTS = {
    "acpc": {"small_blind": 50, "big_blind": 100, "stack": 20000},
    "acpc2009": {"small_blind": 1, "big_blind": 2, "stack": 400},
    "short": {"small_blind": 1, "big_blind": 2, "stack": 13},
    "odd": {"small_blind": 3, "big_blind": 7, "stack": 150},
}


def compare(config: Dict, traj: Dict) -> int:
    """Step gmsim.State through a PokerKit trajectory, comparing every step (see the
    module docstring); returns the number of steps."""
    assert "error" not in traj, traj.get("error")
    st = gmsim.State("hunl", config)
    dealt = set()
    for i, step in enumerate(traj["steps"]):
        kind, seats = st.turn()
        if step[0] == -1:
            assert kind == "chance", (i, kind)
            outs = st.chance_outcomes()
            assert [o for o, _ in outs] == [c for c in range(52) if c not in dealt], i
            assert all(abs(p - 1.0 / len(outs)) < 1e-12 for _, p in outs)
            st.apply_chance(step[1])
            dealt.add(step[1])
        else:
            seat, action, fold, call, lo, hi = step
            assert (kind, seats) == ("act", [seat]), (i, kind, seats, step)
            want = [0] * fold + [1] * call + (list(range(lo, hi + 1)) if lo >= 0 else [])
            got = st.legal_actions(seat)
            assert got == want, (i, got[:3], got[-1], len(got), step, str(st))
            st.apply_action(seat, action)
    kind, _ = st.turn()
    assert kind == traj["final"] == "terminal", (kind, traj["final"])
    assert st.returns() == pytest.approx(traj["returns"], abs=1e-9)
    return len(traj["steps"])


def our_hand(config: Dict, rng: random.Random) -> List[List[int]]:
    """A gmrules random hand: chance uniform; per decision fold (15%), raise (35%: half
    a random ABSTRACT candidate's amount — gmsim.State.decode of a legal candidate —
    half an exact amount: min, max or uniform), else check / call."""
    st = gmsim.State("hunl", config)
    while True:
        kind, seats = st.turn()
        if kind == "terminal":
            return st.history().tolist()
        if kind == "chance":
            st.apply_chance(rng.choice(st.chance_outcomes())[0])
            continue
        s = seats[0]
        legal = st.legal_actions(s)
        a = 1
        if legal[0] == 0 and rng.random() < 0.15:
            a = 0
        elif legal[-1] >= 2 and rng.random() < 0.35:
            raises = [x for x in legal if x >= 2]
            if rng.random() < 0.5:
                mask = st.frame(s)["obs"]["action_mask"]
                k = rng.choice([k for k in range(2, len(mask)) if mask[k]])
                a = st.decode(s, [k])
            else:
                a = rng.choice([raises[0], raises[-1], rng.choice(raises)])
        st.apply_action(s, a)


def run_reference(out: Path, *args: str) -> Dict:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not installed")
    cmd = [uv, "run", "--no-project", "--python", "3.12", "--with", POKERKIT, "python",
           str(SCRIPT), "--out", str(out), *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip(f"PokerKit reference unavailable: {res.stderr[-500:]}")
    with open(out) as fh:
        return json.load(fh)


def reference_data(tmp: Path, hands: int, seed: int) -> Dict:
    """Both directions: PokerKit's hands, and ours replayed through PokerKit."""
    pk = run_reference(tmp / "pk.json", "--hands", str(hands), "--seed", str(seed))
    rng = random.Random(seed)
    ours = {"variants": [{"key": k, "config": c, "trajectories": [our_hand(c, rng) for _ in range(hands)]}
                         for k, c in VARIANTS.items()]}
    (tmp / "ours.json").write_text(json.dumps(ours))
    back = run_reference(tmp / "back.json", "--replay", str(tmp / "ours.json"))
    return {"source": pk["source"], "pokerkit_hands": pk["variants"], "gmrules_hands_in_pokerkit": back["variants"]}


def write_fixture(path: Path = FIXTURE, hands: int = 30, seed: int = 0) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        data = reference_data(Path(d), hands, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def _check(data: Dict) -> Dict[str, int]:
    n = {"steps": 0, "hands": 0, "showdowns": 0, "raises": 0}
    for direction in ("pokerkit_hands", "gmrules_hands_in_pokerkit"):
        for v in data[direction]:
            for t in v["trajectories"]:
                n["steps"] += compare(v["config"], t)
                n["hands"] += 1
                n["raises"] += sum(1 for s in t["steps"] if s[0] >= 0 and s[1] >= 2)
                n["showdowns"] += sum(1 for s in t["steps"] if s[0] == -1) == 9
    return n


@pytest.fixture(scope="module")
def fixture_data() -> Dict:
    with open(FIXTURE) as fh:
        return json.load(fh)


@pytest.mark.parametrize("direction", ["pokerkit_hands", "gmrules_hands_in_pokerkit"])
def test_fixture_matches_pokerkit(fixture_data, direction):
    keys = set()
    for v in fixture_data[direction]:
        assert v["config"] == VARIANTS[v["key"]]
        for t in v["trajectories"]:
            compare(v["config"], t)
        keys.add(v["key"])
    assert keys == set(VARIANTS)
    n = _check(fixture_data)
    assert n["hands"] >= 200 and n["showdowns"] >= 40 and n["raises"] >= 200, n


@pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1", reason="live reference: GM_REFERENCE=1")
def test_live_randomized_differential(tmp_path):
    hands = int(os.environ.get("GM_REFERENCE_HANDS", "300"))
    data = reference_data(tmp_path, hands, seed=int(os.environ.get("GM_REFERENCE_SEED", "1")))
    n = _check(data)
    assert n["hands"] == 2 * len(VARIANTS) * hands, n
    print(f"hunl vs {data['source']}: {n}")


def test_our_random_hands_cover_the_rules():
    """The gmrules side of the fixture exercises folds, all-ins and every street."""
    rng = random.Random(5)
    kinds = {"fold": 0, "showdown": 0, "all_in": 0}
    for _ in range(60):
        ev = our_hand(VARIANTS["acpc"], rng)
        r = gmsim.replay("hunl", VARIANTS["acpc"], ev)
        assert r["terminal"]
        acts = [i for s, i in ev if s >= 0]
        kinds["fold"] += 0 in acts
        kinds["all_in"] += 20000 in acts
        kinds["showdown"] += 0 not in acts
    assert all(v > 3 for v in kinds.values()), kinds
