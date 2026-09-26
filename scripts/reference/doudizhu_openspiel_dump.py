"""Reference Dou Dizhu trajectories from OpenSpiel's ``dou_dizhu`` for the differential
tests of the gmrules game (tests/test_doudizhu_reference.py).  Standalone: runs in
its own throw-away uv environment, never in the project one:

    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/doudizhu_openspiel_dump.py --games 40 --seed 2 --out ref.json.gz
    # replay trajectories produced by gmrules (JSON {"trajectories": [[[seat, id or
    # play string], ...], ...]}; seat -1 = chance) through OpenSpiel
    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/doudizhu_openspiel_dump.py --replay ours.json --out ref.json.gz

Every step records the state as OpenSpiel sees it, then the event applied:

    chance   {"kind": "chance", "outcomes": hex bitmask of the outcome ids, "p": [min, max]
              probability, "event": [-1, id]}
    auction  {"kind": "act", "phase": "auction", "seat", "legal": [ids], "event": [seat, id]}
    play     {"kind": "act", "phase": "play", "seat", "hand": the acting hand (OpenSpiel's
              rank string), "legal": [OpenSpiel action strings], "event": [seat, string],
              "id": OpenSpiel action id}

plus the final "returns".  Play actions are compared by their OpenSpiel strings
(``ActionToString``: every card by rank, airplanes with kickers as ``chain-kickers``),
which name OpenSpiel's play ids one-to-one — the gmrules game factors plays with
kickers into several decisions, so its ids differ for those.

OpenSpiel (Apache-2.0, https://github.com/google-deepmind/open_spiel) is not a
dependency of the project.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from typing import Dict, List

import pyspiel


def _hand_of(state, seat: int) -> str:
    # ObservationString: "My hand <cards>\nPlayed cards ..."
    first = state.observation_string(seat).split("\n", 1)[0]
    assert first.startswith("My hand "), first
    return first[len("My hand "):]


def turn_of(state) -> Dict:
    if state.is_chance_node():
        outs = state.chance_outcomes()
        mask = 0
        for a, _ in outs:
            mask |= 1 << int(a)
        ps = [float(p) for _, p in outs]
        return {"kind": "chance", "outcomes": hex(mask), "p": [min(ps), max(ps)]}
    seat = state.current_player()
    legal = [int(a) for a in state.legal_actions()]
    # the auction always offers a bid (ids 1..3, never legal while playing)
    phase = "auction" if any(1 <= a <= 3 for a in legal) else "play"
    step = {"kind": "act", "phase": phase, "seat": seat}
    if phase == "auction":
        step["legal"] = legal
    else:
        strs = [state.action_to_string(seat, a) for a in legal]
        assert len(set(strs)) == len(strs), "OpenSpiel play strings are one-to-one"
        step["legal"] = strs
        step["hand"] = _hand_of(state, seat)
    return step


def play_game(game, rng: random.Random, events=None) -> Dict:
    """One trajectory: uniformly random events, or ``events`` (gmrules' format:
    [seat, id] for chance / bids, [seat, play string] for plays)."""
    state = game.new_initial_state()
    steps: List[Dict] = []
    i = 0
    while not state.is_terminal():
        if events is not None and i >= len(events):
            break
        step = turn_of(state)
        if step["kind"] == "chance":
            if events is None:
                outs = state.chance_outcomes()
                a = rng.choices([o for o, _ in outs], [p for _, p in outs])[0]
            else:
                seat, a = events[i]
                assert seat == -1, (i, events[i])
            step["event"] = [-1, int(a)]
            state.apply_action(int(a))
        else:
            seat = step["seat"]
            legal = state.legal_actions()
            if events is None:
                a = rng.choice(legal)
            else:
                s, ev = events[i]
                assert s == seat, (i, s, seat)
                if step["phase"] == "auction":
                    a = int(ev)
                else:
                    by_str = {state.action_to_string(seat, x): x for x in legal}
                    if ev not in by_str:
                        step["rejected"] = ev
                        steps.append(step)
                        break
                    a = by_str[ev]
            if step["phase"] == "auction":
                step["event"] = [seat, int(a)]
            else:
                step["event"] = [seat, state.action_to_string(seat, a)]
                step["id"] = int(a)
            state.apply_action(a)
        steps.append(step)
        i += 1
    out = {"steps": steps, "terminal": bool(state.is_terminal())}
    if state.is_terminal():
        out["returns"] = [float(r) for r in state.returns()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", help="JSON file of gmrules trajectories to replay")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    game = pyspiel.load_game("dou_dizhu")
    rng = random.Random(args.seed)
    if args.replay:
        with open(args.replay) as fh:
            todo = json.load(fh)["trajectories"]
        trajs = [play_game(game, rng, ev) for ev in todo]
    else:
        trajs = [play_game(game, rng) for _ in range(args.games)]
    data = {"game": "dou_dizhu", "num_distinct_actions": game.num_distinct_actions(),
            "max_chance_outcomes": game.max_chance_outcomes(), "seed": args.seed,
            "trajectories": trajs}
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt") as fh:
        json.dump(data, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
