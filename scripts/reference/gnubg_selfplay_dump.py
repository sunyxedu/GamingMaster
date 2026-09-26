"""gnubg's OWN match loop as bgmatch.v1 transcripts — the independent reference for the
match-level rules of gmrules backgammon (dice handling, opening roll, doubling and
cube ownership, drops, gammons / backgammons, the Jacoby rule, scores, the Crawford
and post-Crawford games).  gnubg rolls its own dice (Mersenne Twister, ``--seed``) and
plays both sides; ``gmdata backgammon verify`` then replays every game with the rules
engine and checks every action's legality, every played-out game's result and the
score / Crawford progression between games.

Runs in a THROW-AWAY environment with GNU Backgammon's Python bindings (GPL, an
external program; see gnubg_bridge.py)::

    uv run --no-project --python 3.12 --with gnubg==1.1.0a16 \\
        python scripts/reference/gnubg_selfplay_dump.py --out m.jsonl.gz --matches 20 \\
        --match-lengths 1,3,5,7 --money-games 20 --plies 0 --seed 1
    gmdata backgammon verify --input m.jsonl.gz

Matches are one transcript each (config ``{"match_length": N}``); money games one
transcript per game (``{"match_length": 0, "jacoby": ...}``: one money game is one
episode of the rules).  gnubg resigns hopeless games: a resigned game's play is still
verified, its result (resignation value x cube) cannot be — the rules have no
resignation — so it only enters the score progression.  Seats: gnubg's X = seat 0,
O = seat 1.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnubg_bridge as bridge  # noqa: E402  (redirects gnubg's stdout; we only write files)

gnubg = bridge.gnubg


def _seat(player: str) -> int:
    return 0 if player == "X" else 1


def _actions(raw) -> list:
    """gnubg's move records of one game → bgmatch.v1 actions (a rejected resignation
    changes nothing and is dropped)."""
    out = []
    k = 0
    while k < len(raw):
        a = raw[k]
        act, p = a["action"], _seat(a.get("player", "X"))
        if act == "move":
            flat = [int(x) for step in a["move"] for x in step]
            out.append({"p": p, "a": "move", "dice": [int(d) for d in a["dice"]], "move": flat})
        elif act in ("double", "take", "drop"):
            out.append({"p": p, "a": act})
        elif act == "resign":
            if k + 1 < len(raw) and raw[k + 1].get("action") == "reject":
                k += 2
                continue
            out.append({"p": p, "a": "resign", "value": int(a.get("points", 1))})
        elif act not in ("accept", "reject"):
            raise ValueError(f"unexpected gnubg action {a!r}")
        k += 1
    return out


def _game(g) -> dict:
    info = g["info"]
    return {"score": [int(info.get("score-X", 0)), int(info.get("score-O", 0))],
            "crawford": bool(info.get("crawford", False)), "actions": _actions(g["game"]),
            "winner": _seat(info["winner"]), "points": int(info["points-won"]),
            "resigned": bool(info.get("resigned", False))}


def play_match(n: int, seed: int) -> dict:
    for c in (f"set seed {seed}", f"new match {n}"):
        bridge.command(c)
    return gnubg.match()


def play_money(jacoby: bool, seed: int) -> dict:
    for c in (f"set jacoby {'on' if jacoby else 'off'}", f"set seed {seed}", "new session", "new game"):
        bridge.command(c)
    return gnubg.match()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help=".jsonl or .jsonl.gz")
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--match-lengths", default="5", help="comma-separated, used in turn")
    ap.add_argument("--money-games", type=int, default=10)
    ap.add_argument("--plies", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    bridge.setup(args.plies)
    for c in ("set player 0 gnubg", "set player 1 gnubg", "set automatic roll on",
              "set automatic game on", "set rng mersenne", "set crawford on"):
        bridge.command(c)
    engine = f"gnubg {gnubg.version} {args.plies}-ply (own match loop)"
    lines = []
    lengths = [int(x) for x in args.match_lengths.split(",")]
    for i in range(args.matches):
        seed = args.seed * 100003 + i
        n = lengths[i % len(lengths)]
        m = play_match(n, seed)
        games = [_game(g) for g in m["games"]]
        last = games[-1]
        final = list(last["score"])
        final[last["winner"]] += last["points"]
        lines.append({"format": "bgmatch.v1", "id": f"gnubg-selfplay-m{n}-s{seed}",
                      "config": {"match_length": n},
                      "players": [engine, engine], "meta": {"seed": seed, "source": "gnubg match loop"},
                      "games": games, "winner": 0 if final[0] >= n else 1,
                      "score": final})
    bridge.command("set automatic game off")
    for i in range(args.money_games):
        seed = args.seed * 100003 + 50000 + i
        jacoby = i % 2 == 0
        m = play_money(jacoby, seed)
        g = _game(m["games"][0])
        g["score"] = [0, 0]
        lines.append({"format": "bgmatch.v1", "id": f"gnubg-selfplay-money-j{int(jacoby)}-s{seed}",
                      "config": {"match_length": 0, "jacoby": jacoby},
                      "players": [engine, engine], "meta": {"seed": seed, "source": "gnubg money game"},
                      "games": [g], "winner": g["winner"]})
    data = "".join(json.dumps(x) + "\n" for x in lines).encode()
    if args.out.endswith(".gz"):
        with open(args.out, "wb") as raw, gzip.GzipFile("", "wb", 9, raw, mtime=0) as f:
            f.write(data)
    else:
        with open(args.out, "wb") as f:
            f.write(data)
    sys.stderr.write(f"{len(lines)} transcripts -> {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
