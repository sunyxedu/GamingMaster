"""PerfectDou (Netease Games AI Lab, NeurIPS 2022, Apache-2.0) as a move server for
scripts/generate_doudizhu.py.  Its feature encoder ships as a CPython 3.7 extension
(perfectdou/env/encode.cpython-37m-x86_64-linux-gnu.so), so it runs in its own Python 3.7
process — this file, kept 3.7-compatible — next to the generator.

One JSON object per stdin line: DouZero's ``InfoSet`` fields of the seat to move
(player_position, player_hand_cards, legal_actions, other_hand_cards, last_move,
last_two_moves, last_move_dict, num_cards_left_dict, played_cards, bomb_num,
card_play_action_seq; card values 3..14, 17 = 2, 20 / 30 = jokers) → one line
``{"action": [values]}`` (PerfectDou's greedy move; [] = pass), or ``{"error": ...}``.
``{"op": "ping"}`` answers ``{"pong": true}``.

    <python3.7> scripts/perfectdou_worker.py --perfectdou <checkout of the PerfectDou repo>
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfectdou", required=True, help="the PerfectDou repo checkout")
    a = ap.parse_args()
    sys.path.insert(0, a.perfectdou)
    from perfectdou.env.game import InfoSet
    from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent

    agents = {p: PerfectDouAgent(p) for p in ("landlord", "landlord_up", "landlord_down")}
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("op") == "ping":
                out.write(json.dumps({"pong": True}) + "\n")
                out.flush()
                continue
            info = InfoSet(d["player_position"])
            for k in ("player_hand_cards", "legal_actions", "other_hand_cards", "last_move", "last_two_moves",
                      "last_move_dict", "num_cards_left_dict", "played_cards", "bomb_num", "card_play_action_seq",
                      "three_landlord_cards", "last_pid", "all_handcards"):
                if k in d:
                    setattr(info, k, d[k])
            act = agents[d["player_position"]].act(info)
            out.write(json.dumps({"action": sorted(int(x) for x in act)}) + "\n")
        except Exception as e:  # noqa: BLE001 — the generator decides
            out.write(json.dumps({"error": "%s: %s" % (type(e).__name__, e)}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
