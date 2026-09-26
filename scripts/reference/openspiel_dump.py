"""Reference trajectories from OpenSpiel for the differential tests of gmrules games
(tests/test_candidate_reference.py).  Standalone: runs in its own throw-away uv
environment, never in the project one:

    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/openspiel_dump.py --episodes 40 --seed 0 --out ref.json
    # replay trajectories produced by gmrules (JSON {"variants": [{"key", "trajectories":
    # [[[seat, id], ...], ...]}]}) through OpenSpiel and record what it says
    uv run --no-project --python 3.12 --with open_spiel \\
        python scripts/reference/openspiel_dump.py --replay ours.json --out ref.json

Every step records the state as OpenSpiel sees it — the turn ("chance" with its
outcomes and probabilities, "act" with every acting seat's legal actions,
"terminal"), then the events applied (``[seat, id]``, seat -1 = chance; the actions of
a simultaneous turn in seat order) — and the final returns.  Pattern for other games:
map the reference's action / outcome ids to the gmrules canonical ids (identity here),
dump from the reference, compare step by step against ``gmsim.State``.

OpenSpiel (Apache-2.0, https://github.com/google-deepmind/open_spiel) is not a
dependency of the project.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List

import pyspiel

#: gmrules variant key → (OpenSpiel game string, gmrules game, gmrules config)
VARIANTS = {
    "kuhn2": ("kuhn_poker(players=2)", "kuhn_poker", {"players": 2}),
    "kuhn3": ("kuhn_poker(players=3)", "kuhn_poker", {"players": 3}),
    "kuhn4": ("kuhn_poker(players=4)", "kuhn_poker", {"players": 4}),
    "goof4x2": ("goofspiel(num_cards=4,players=2,points_order=random,imp_info=False,"
                "returns_type=win_loss)", "goofspiel",
                {"num_cards": 4, "players": 2, "points_order": "random"}),
    "goof5x3": ("goofspiel(num_cards=5,players=3,points_order=random,imp_info=False,"
                "returns_type=win_loss)", "goofspiel",
                {"num_cards": 5, "players": 3, "points_order": "random"}),
    "goof4x4d": ("goofspiel(num_cards=4,players=4,points_order=descending,imp_info=False,"
                 "returns_type=win_loss)", "goofspiel",
                 {"num_cards": 4, "players": 4, "points_order": "descending"}),
    "pig2": ("pig(players=2,winscore=10,diceoutcomes=6,horizon=40)", "pig",
             {"players": 2, "winscore": 10, "diceoutcomes": 6, "horizon": 40}),
    "pig3": ("pig(players=3,winscore=8,diceoutcomes=3,horizon=12)", "pig",
             {"players": 3, "winscore": 8, "diceoutcomes": 3, "horizon": 12}),
}


def turn_of(state, n: int) -> Dict:
    if state.is_terminal():
        return {"kind": "terminal"}
    if state.is_chance_node():
        return {"kind": "chance",
                "outcomes": [[int(a), float(p)] for a, p in state.chance_outcomes()]}
    if state.is_simultaneous_node():
        seats = [p for p in range(n) if state.legal_actions(p)]
    else:
        seats = [state.current_player()]
    return {"kind": "act", "seats": seats,
            "legal": {str(p): [int(a) for a in state.legal_actions(p)] for p in seats}}


def apply(state, turn: Dict, events: List[List[int]], n: int) -> None:
    if turn["kind"] == "chance":
        (seat, oid), = events
        assert seat == -1
        state.apply_action(oid)
    elif state.is_simultaneous_node():
        acts = [0] * n
        for seat, a in events:
            acts[seat] = a
        state.apply_actions(acts)
    else:
        (seat, a), = events
        assert seat == state.current_player()
        state.apply_action(a)


def play(game, n: int, rng: random.Random = None, events: List[List[int]] = None) -> Dict:
    """One trajectory: random (``rng``) or the given flat ``events``."""
    state = game.new_initial_state()
    steps, i = [], 0
    while not state.is_terminal():
        turn = turn_of(state, n)
        if events is not None:
            k = 1 if turn["kind"] == "chance" else len(turn["seats"])
            if i + k > len(events):
                break
            ev = [list(e) for e in events[i:i + k]]
            i += k
        elif turn["kind"] == "chance":
            ids, probs = zip(*turn["outcomes"])
            ev = [[-1, rng.choices(ids, weights=probs)[0]]]
        else:
            ev = [[p, rng.choice(turn["legal"][str(p)])] for p in turn["seats"]]
        steps.append({"turn": turn, "events": ev})
        apply(state, turn, ev, n)
    return {"steps": steps, "final": turn_of(state, n),
            "returns": [float(r) for r in state.returns()]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=sorted(VARIANTS))
    ap.add_argument("--replay", default=None, help="JSON of gmrules trajectories to replay")
    a = ap.parse_args()
    out = {"source": f"open_spiel {getattr(pyspiel, '__version__', '?')}", "variants": []}
    if a.replay:
        with open(a.replay) as fh:
            todo = [(v["key"], v["trajectories"]) for v in json.load(fh)["variants"]]
    else:
        todo = [(k, None) for k in a.variants]
    for key, trajs in todo:
        spiel, gm_game, gm_config = VARIANTS[key]
        game = pyspiel.load_game(spiel)
        n = game.num_players()
        rng = random.Random(f"{a.seed}:{key}")
        runs = ([play(game, n, events=t) for t in trajs] if trajs is not None
                else [play(game, n, rng=rng) for _ in range(a.episodes)])
        out["variants"].append({"key": key, "openspiel": spiel, "gm_game": gm_game,
                                "gm_config": gm_config, "trajectories": runs})
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
