"""Reference data from OpenSpiel's bridge for the differential tests of the gmrules
bridge (tests/test_bridge_reference.py).  Standalone: runs in its own throw-away uv
environment, never in the project one:

    # OpenSpiel's own random playouts (boards = duplicate board numbers: dealer and
    # vulnerability by the standard rotation)
    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/openspiel_bridge_dump.py --episodes 40 --boards 1 6 7 12 --out ref.json
    # gmrules trajectories (JSON {"variants": [{"board", "trajectories": [[[seat, id], ...]]}]})
    # replayed through OpenSpiel
    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/openspiel_bridge_dump.py --replay ours.json --out ref.json
    # real records (the WBridge5 dataset: one OpenSpiel history per line, board 1)
    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/openspiel_bridge_dump.py --records test.txt --max-records 500 --out ref.json

Every step records what OpenSpiel says — the turn ("chance": the outcome ids as a
bit mask ``mask`` and their common probability ``p``; "act": the acting seat and its
legal actions; "terminal") — then the event applied (``[seat, id]``, seat -1 =
chance), and at the end the returns (and ``ok``: every event was legal).  Ids are
identical in gmrules and OpenSpiel (cards 0..51, calls 52..89); gmrules configs
``{"board": b}`` have no board chance event, so a trajectory is an OpenSpiel history
verbatim.  ``--summary`` drops the steps (returns only: large record sets).

OpenSpiel (Apache-2.0, https://github.com/google-deepmind/open_spiel) is not a
dependency of the project.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional

import pyspiel

#: duplicate board → (dealer, dealer_vul, non_dealer_vul), OpenSpiel's
#: NewDuplicateBridgeInitialState table
DEALER_VUL = [False, False, False, True, True, True, True, False,
              False, True, False, False, True, False, True, True]
NON_DEALER_VUL = [False, True, True, True, False, False, True, False,
                  True, True, False, True, True, False, False, False]


def load(board: int):
    b = (board - 1) % 16
    return pyspiel.load_game(
        f"bridge(use_double_dummy_result=false,dealer={b % 4},"
        f"dealer_vul={str(DEALER_VUL[b]).lower()},non_dealer_vul={str(NON_DEALER_VUL[b]).lower()})")


def turn_of(state) -> Dict:
    if state.is_terminal():
        return {"kind": "terminal"}
    if state.is_chance_node():
        outs = state.chance_outcomes()
        ps = {round(p, 15) for _, p in outs}
        assert len(ps) == 1, "bridge chance outcomes are uniform"
        mask = 0
        for a, _ in outs:
            mask |= 1 << int(a)
        return {"kind": "chance", "mask": str(mask), "p": float(outs[0][1])}
    p = state.current_player()
    return {"kind": "act", "seat": p, "legal": [int(a) for a in state.legal_actions(p)]}


def play(game, rng: Optional[random.Random] = None, events: Optional[List[List[int]]] = None,
         summary: bool = False) -> Dict:
    """One trajectory: random (``rng``) or the given ``[seat, id]`` events (seats are
    checked against OpenSpiel's current player)."""
    state = game.new_initial_state()
    steps, ok, i = [], True, 0
    while not state.is_terminal():
        turn = turn_of(state)
        if events is not None:
            if i >= len(events):
                break
            seat, a = events[i]
            i += 1
            legal = (state.legal_actions() if turn["kind"] == "act"
                     else [o for o, _ in state.chance_outcomes()])
            want = -1 if turn["kind"] == "chance" else turn["seat"]
            if seat != want or a not in legal:
                ok = False
                break
        elif turn["kind"] == "chance":
            outs = state.chance_outcomes()
            seat, a = -1, int(rng.choice(outs)[0])
        else:
            seat, a = turn["seat"], rng.choice(turn["legal"])
        if not summary:
            steps.append({"turn": turn, "event": [seat, int(a)]})
        state.apply_action(int(a))
    out = {"final": turn_of(state)["kind"], "ok": ok,
           "returns": [float(r) for r in state.returns()]}
    if not summary:
        out["steps"] = steps
    return out


def record_events(line: str) -> List[List[int]]:
    """A WBridge5 dataset line (an OpenSpiel history of board 1) as [seat, id] events:
    52 chance deals, then the calls and cards with OpenSpiel's acting seats."""
    game = load(1)
    state = game.new_initial_state()
    events = []
    for tok in line.split():
        a = int(tok)
        seat = -1 if state.is_chance_node() else state.current_player()
        events.append([seat, a])
        state.apply_action(a)
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boards", type=int, nargs="*", default=[1, 6, 7, 12])
    ap.add_argument("--replay", default=None, help="JSON of gmrules trajectories to replay")
    ap.add_argument("--records", default=None, help="WBridge5 dataset file (OpenSpiel histories)")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--summary", action="store_true", help="returns only (no steps)")
    a = ap.parse_args()
    out = {"source": f"open_spiel {getattr(pyspiel, '__version__', '?')}", "variants": []}
    if a.records:
        with open(a.records) as fh:
            lines = [ln for ln in fh if ln.strip()][:a.max_records]
        game = load(1)
        runs = []
        for ln in lines:
            ev = record_events(ln)
            run = play(game, events=ev, summary=a.summary)
            run["events"] = ev
            runs.append(run)
        out["variants"].append({"board": 1, "records": a.records, "trajectories": runs})
    elif a.replay:
        with open(a.replay) as fh:
            for v in json.load(fh)["variants"]:
                game = load(v["board"])
                out["variants"].append({"board": v["board"], "trajectories": [
                    play(game, events=t, summary=a.summary) for t in v["trajectories"]]})
    else:
        for board in a.boards:
            game = load(board)
            rng = random.Random(f"{a.seed}:bridge:{board}")
            out["variants"].append({"board": board, "trajectories": [
                play(game, rng=rng, summary=a.summary) for _ in range(a.episodes)]})
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
