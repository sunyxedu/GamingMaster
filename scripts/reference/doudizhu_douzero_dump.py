"""Reference Dou Dizhu play phases from DouZero's game engine (kwai/DouZero,
``douzero/env``: the rules DouZero was trained and evaluated with; Apache-2.0) for the
differential tests of the gmrules game (tests/test_doudizhu_douzero.py).

The engine is five pure-Python files, fetched at a pinned commit (SHA-256 checked)
into ``$GM_REFERENCE_CACHE`` (default ``~/.cache/gm-reference``); no install, no torch:

    python scripts/reference/doudizhu_douzero_dump.py --games 40 --seed 0 --out ref.json.gz
    # replay play phases produced by gmrules (JSON {"games": [{"deal": card_play_data,
    # "moves": [[position, [cards]], ...]}, ...]}) through DouZero
    python scripts/reference/doudizhu_douzero_dump.py --replay ours.json --out ref.json.gz

DouZero has no auction: a game starts from ``card_play_data`` (the landlord's 20 cards
incl. the 3 kitty cards, 17 each for ``landlord_down`` — who plays after the landlord
— and ``landlord_up``); cards are rank values 3..14 (3..A), 17 (2), 20 / 30 (jokers).
Every decision records the acting position, its hand, DouZero's legal moves (sorted,
de-duplicated), the move played and DouZero's type of it (``get_move_type``: type id,
rank, length); every game its winner ("landlord" / "farmer") and bomb count (bombs and
rockets).  Random playouts pick uniformly among DouZero's legal moves.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

COMMIT = "718a5c920bf3361e34178a38f3b80458e176b351"
FILES = {
    "game.py": "09e76db9d90eb8dffbb747325fdd5ac86d7fc00873e6c862e3dd3dbadd4d0e96",
    "move_generator.py": "157cb79cccc89110b6b6c49700e4860bf09b287b5356811ea7a1f81552649685",
    "move_detector.py": "d1a264e4084138b870c5dbb31ce723347164047653b872737bd511eb10cc4cfe",
    "move_selector.py": "40aa53c9284617a606d89168cc70fc9581ce66f6ced131dcfba3f4cf1df16c38",
    "utils.py": "c4dab1343fa9e189c0f66a24363c5fa80f5bc3f1517dc0e45ef419367be5d48c",
}
DECK = [v for v in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]


def fetch_engine() -> Path:
    """The pinned DouZero env package (downloaded once, hashes checked)."""
    root = Path(os.environ.get("GM_REFERENCE_CACHE", Path.home() / ".cache" / "gm-reference"))
    pkg = root / f"douzero-{COMMIT[:12]}"
    env = pkg / "douzero" / "env"
    env.mkdir(parents=True, exist_ok=True)
    for init in (pkg / "douzero" / "__init__.py", env / "__init__.py"):
        init.touch()
    for name, digest in FILES.items():
        path = env / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            url = f"https://raw.githubusercontent.com/kwai/DouZero/{COMMIT}/douzero/env/{name}"
            data = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise SystemExit(f"{url}: unexpected SHA-256")
            path.write_bytes(data)
    return pkg


class Player:
    """Plays scripted moves (replay) or uniformly random legal ones."""

    def __init__(self, rng: random.Random, script: Optional[List] = None):
        self.rng, self.script, self.log = rng, script, []

    def act(self, infoset):
        legal = infoset.legal_actions
        if self.script is None:
            move = self.rng.choice(legal)
        else:
            move = sorted(self.script.pop(0))
        self.log.append((infoset, move))
        return move


def play(game_env, md, deal: Dict, rng: random.Random, moves=None) -> Dict:
    script = None if moves is None else [m for _, m in moves]
    positions = None if moves is None else [p for p, _ in moves]
    player = Player(rng, script)
    env = game_env({p: player for p in ("landlord", "landlord_up", "landlord_down")})
    env.card_play_init(copy.deepcopy(deal))
    steps = []
    rejected = None
    while not env.game_over:
        info = env.game_infoset
        pos = env.acting_player_position
        legal = sorted({tuple(sorted(m)) for m in info.legal_actions})
        if moves is not None:
            if not player.script:
                break
            if positions[len(steps)] != pos:
                raise SystemExit(f"replay: move {len(steps)} by {positions[len(steps)]}, DouZero expects {pos}")
            if tuple(sorted(player.script[0])) not in legal:
                rejected = sorted(player.script[0])
                steps.append({"pos": pos, "hand": sorted(info.player_hand_cards),
                              "legal": [list(m) for m in legal], "rejected": rejected})
                break
        env.step()
        _, move = player.log[-1]
        t = md.get_move_type(list(move))
        steps.append({"pos": pos, "hand": sorted(info.player_hand_cards), "legal": [list(m) for m in legal],
                      "move": list(move), "type": [t["type"], t.get("rank", 0), t.get("len", 1)]})
    out = {"deal": deal, "steps": steps, "over": bool(env.game_over)}
    if env.game_over:
        out["winner"] = env.get_winner()
        out["bomb_num"] = env.get_bomb_num()
    return out


def random_deal(rng: random.Random) -> Dict:
    d = DECK.copy()
    rng.shuffle(d)
    deal = {"landlord": sorted(d[:20]), "landlord_up": sorted(d[20:37]),
            "landlord_down": sorted(d[37:54]), "three_landlord_cards": sorted(d[17:20])}
    return deal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", help="JSON file of gmrules play phases to replay")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(fetch_engine()))
    from douzero.env import move_detector as md
    from douzero.env.game import GameEnv

    rng = random.Random(args.seed)
    if args.replay:
        with open(args.replay) as fh:
            todo = json.load(fh)["games"]
        games = [play(GameEnv, md, g["deal"], rng, g["moves"]) for g in todo]
    else:
        games = [play(GameEnv, md, random_deal(rng), rng) for _ in range(args.games)]
    data = {"engine": f"kwai/DouZero@{COMMIT}", "seed": args.seed, "games": games}
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt") as fh:
        json.dump(data, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
