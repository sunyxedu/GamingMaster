"""Dou Dizhu self-play records from the public DouZero agents — the imitation corpus
of the ``doudizhu`` gmrules game (there is no public human data).

    uv run python scripts/generate_doudizhu.py --out data/doudizhu/raw --games 1000000 --workers 32
    gmdata doudizhu convert --input data/doudizhu/raw --out data/doudizhu/train --hq

(``scripts/fetch_doudizhu.sh`` runs both, plus a validation split.)

Rules: the gmrules game itself (``gmsim.State("doudizhu")``, the standard rules =
OpenSpiel's ``dou_dizhu`` + four-with-two) decides every legal move, the auction and
the score; the converter (``gmdata doudizhu convert``) replays and verifies every
record again.

Players.  PLAY PHASE: DouZero (kwai/DouZero, Zha et al., ICML 2021, Apache-2.0) —
its pretrained per-position networks (landlord / landlord_down = plays after the
landlord / landlord_up) score every legal move with DouZero's own feature encoder
(``douzero/env/env.py get_obs``, fetched at a pinned commit, SHA-256 checked) and play
the best one, exactly as DouZero's ``DeepAgent`` does (greedy; a single legal move is
played without the network).  The candidates are the gmrules legal plays (projected
to DouZero's card lists; a card set with several readings — e.g. ``333444555666`` —
is played as DouZero's move detector reads it), so every recorded move is legal
under the standard rules (DouZero alone would also play the rocket or a bomb as
kickers).  Checkpoints (``--agents``, drawn per game and role): ``adp`` (objective:
average difference points) and ``wp`` (winning percentage) — the two strongest
public DouZero agents — and ``sl`` (DouZero's supervised baseline, weaker; not in the
default mix).  They come from the Hugging Face mirror of DouZero's ``baselines/``
(palemoky/douzero-baselines, Apache-2.0, pinned revision + SHA-256; the official
copies are on Baidu Netdisk only, the Google Drive folder is gone; the WP files are
byte-identical to an independent 2021 copy in tianqiraf/DouZero_For_HappyDouDiZhu)
and are loaded with ``torch.load(weights_only=True)``.

``perfectdou`` (``--agents perfectdou``): PerfectDou (Netease Games AI Lab, Yang et al.,
NeurIPS 2022, Apache-2.0 — stronger than DouZero in its paper), its pretrained ONNX policy
per position, greedy, through ``scripts/perfectdou_worker.py``: its feature encoder is a
CPython 3.7 extension, so the worker runs in a separate Python 3.7 (``--perfectdou_python``,
with onnxruntime 1.7 + numpy) on a checkout of its repo (``--perfectdou_repo``, commit
594404922ee3).  It sees the same InfoSet fields (its encoder reads no hidden card — the
opponents' split changes no feature — but the landlord's unplayed kitty cards, kept here as
its GameEnv does); a move outside the gmrules candidates (never seen so far) falls back to the
first candidate, counted in the record's meta.

AUCTION (DouZero has none) — ``--bidding heuristic_v1``, marked in every record's
meta: the classic hand-strength count of the 17 dealt cards — rocket 8 (else big
joker 4, small joker 3), each bomb 6, each 2 outside a bomb 2 — gives a maximum bid
(>= 7: 3, >= 5: 2, >= 3: 1, else none); in turn (OpenSpiel's auction: from the holder
of the face-up card until two passes follow a bid or someone bids 3) a player bids
its maximum when it beats the standing bid, else passes.  All three passing ends
the game with no score (OpenSpiel's rule; such records are kept).  The RL phase
learns bidding by itself.

Record (JSONL, one game per line; the ``gmdata doudizhu convert`` format — see
data_processing/rs/gmdata/src/doudizhu.rs)::

    {"id": "dz0-17", "hands": [[17 card ids] x 3], "kitty": [3 card ids],
     "face_up": card id (dealt face up: its holder bids first),
     "auction": [[seat, bid], ...] (0 = pass), "plays": [[seat, "33344"|"Pass"], ...],
     "returns": [r0, r1, r2], "agents": [per seat: "douzero_wp", ...],
     "q": [[i, [[play, Q], ...]], ...]  (plays[i]'s top --q_top moves by DouZero Q),
     "meta": {"bidding": "heuristic_v1", "generator": ..., "seed": ..., "index": ...}}

Card ids are OpenSpiel's: ``c < 52`` rank ``c % 13`` (3 4 ... A 2), suit ``c // 13``;
52 / 53 = black / coloured joker.  A game's deal and agents are a pure function of
``(--seed, index)`` and the play is greedy, so on one machine the output does not
depend on ``--workers`` / ``--part_size`` (across CPU types the networks' float
rounding may break a near-tie differently); output parts are written atomically and
existing parts are skipped (resumable).
Throughput (CPU, one torch thread per worker): ~22 games/s per worker on an Apple
M-series core (~115 games/s with 6 workers); a game averages ~36 decision frames
(2800 games: 100,127 frames, 78 MB of shards).

Measured through this pipeline (seat 0 landlord, 300 random deals each): WP landlord
vs random peasants wins 99.7%, random landlord vs WP peasants 0.3%; landlord win rate
WP/WP 40%, ADP/ADP 47% — DouZero's own ballpark; with heuristic_v1 choosing the
landlord the generated corpus has ~65% landlord wins.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

GENERATOR = "generate_doudizhu.py/2"
PERFECTDOU_COMMIT = "594404922ee3810e2d84b80bb2c2846cb20e5390"
DOUZERO_COMMIT = "718a5c920bf3361e34178a38f3b80458e176b351"
#: DouZero sources needed for inference (feature encoder, move detector, models)
DOUZERO_FILES = {
    "douzero/env/env.py": "43012a3b7672e054b81212dc5b2456b7648ad1d46ebb1217363d897f74233330",
    "douzero/env/game.py": "09e76db9d90eb8dffbb747325fdd5ac86d7fc00873e6c862e3dd3dbadd4d0e96",
    "douzero/env/move_generator.py":
        "157cb79cccc89110b6b6c49700e4860bf09b287b5356811ea7a1f81552649685",
    "douzero/env/move_detector.py":
        "d1a264e4084138b870c5dbb31ce723347164047653b872737bd511eb10cc4cfe",
    "douzero/env/move_selector.py":
        "40aa53c9284617a606d89168cc70fc9581ce66f6ced131dcfba3f4cf1df16c38",
    "douzero/env/utils.py": "c4dab1343fa9e189c0f66a24363c5fa80f5bc3f1517dc0e45ef419367be5d48c",
    "douzero/dmc/models.py": "3ad5483bf3d82b3a4174ee3a8893891dafcd635590e9cd7372fabe5988e8a190",
}
WEIGHTS_REPO = "palemoky/douzero-baselines"
WEIGHTS_REVISION = "57b3914046c2a0877016b8b8830fd07cf5b0ba08"
POSITIONS = ("landlord", "landlord_down", "landlord_up")
#: checkpoint SHA-256 per agent and position (the mirror's LFS object ids)
CHECKPOINTS = {
    "adp": {
        "landlord": "6f2971813495e9c509cbd4a101213c49587472942fa0f018d04c139a6a647c2d",
        "landlord_down": "c173dbe30258f19b6392c30ab1e330b6b3090bbd9ccdc89a9b18123a6ab6e2d9",
        "landlord_up": "7fb7c495ff095db2ddadd8f9f8fd53cad840211578dabc83bbcc8d75085f843e",
    },
    "wp": {
        "landlord": "132e26479fcb69f457ddbf1ad32a7bdc3aa73cde80a28cc91f29a62c6075955a",
        "landlord_down": "964720faf583f4905662c94aaf73b1a97352b76ae3686fcc949beabbacdd95d1",
        "landlord_up": "b849ea518e66f088d635d3a29956da880f7b5ca89fabc5f70fc356c106ca6a15",
    },
    "sl": {
        "landlord": "6cea57a61b8310a0b270e12e72dfb721d15dfcfe77a00caad0e872ad09996a2b",
        "landlord_down": "ba544e205e18c8b9d1b7d9d41d8fb38e712d9a77231d57f5eed19af653ca69e7",
        "landlord_up": "b78b2eee9c79ba55b33911108b678988462fefdcb658b04fc2bd1c6f6d648373",
    },
}
BIDDING = ("heuristic_v1",)
GAME = "doudizhu"

#: DouZero card value of each gmrules rank (3..A, 2, black / coloured joker)
VALUE = list(range(3, 15)) + [17, 20, 30]
RANKS = "3456789TJQKA2"
CHAR_VALUE = {c: v for c, v in zip(RANKS, VALUE)}
CHAR_VALUE.update({"B": 20, "R": 30})
VALUE_CHAR = {v: c for c, v in zip(RANKS, VALUE)}


# ---------------------------------------------------------------------------
# fetching (pinned, hashed)
# ---------------------------------------------------------------------------

def cache_root(cache: Optional[str] = None) -> Path:
    return Path(cache or os.environ.get("GM_REFERENCE_CACHE",
                                        Path.home() / ".cache" / "gm-reference"))


def _download(url: str, path: Path, digest: str) -> None:
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(url, timeout=300).read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit(f"{url}: unexpected SHA-256 (expected {digest})")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch_douzero(cache: Optional[str] = None) -> Path:
    """The pinned DouZero inference sources (downloaded once, hashes checked); returns
    the directory to put on ``sys.path`` (shared with
    scripts/reference/doudizhu_douzero_dump.py's cache)."""
    pkg = cache_root(cache) / f"douzero-{DOUZERO_COMMIT[:12]}"
    for rel, digest in DOUZERO_FILES.items():
        url = f"https://raw.githubusercontent.com/kwai/DouZero/{DOUZERO_COMMIT}/{rel}"
        _download(url, pkg / rel, digest)
    for init in ("douzero/__init__.py", "douzero/env/__init__.py", "douzero/dmc/__init__.py"):
        (pkg / init).touch()
    return pkg


def fetch_checkpoints(agents: Sequence[str],
                      cache: Optional[str] = None) -> Dict[str, Dict[str, Path]]:
    """``{agent: {position: checkpoint path}}`` (downloaded once, hashes checked)."""
    root = cache_root(cache) / f"douzero-baselines-{WEIGHTS_REVISION[:12]}"
    out: Dict[str, Dict[str, Path]] = {}
    for agent in agents:
        out[agent] = {}
        for pos in POSITIONS:
            path = root / f"douzero_{agent.upper()}" / f"{pos}.ckpt" if agent != "sl" else \
                root / "sl" / f"{pos}.ckpt"
            sub = path.parent.name
            url = (f"https://huggingface.co/{WEIGHTS_REPO}/resolve/{WEIGHTS_REVISION}/"
                   f"checkpoints/{sub}/{pos}.ckpt")
            _download(url, path, CHECKPOINTS[agent][pos])
            out[agent][pos] = path
    return out


# ---------------------------------------------------------------------------
# DouZero agents
# ---------------------------------------------------------------------------

class DouZeroAgent:
    """One pretrained DouZero agent: a network per position; ``q_values`` = DouZero's
    ``DeepAgent`` forward (``get_obs`` features, one value per legal move)."""

    def __init__(self, name: str, paths: Dict[str, Path], device: str = "cpu"):
        import torch
        from douzero.dmc.models import model_dict
        from douzero.env.env import get_obs

        self.name = f"douzero_{name}"
        self._torch, self._get_obs, self.device = torch, get_obs, torch.device(device)
        self.models = {}
        for pos in POSITIONS:
            net = model_dict[pos]()
            net.load_state_dict(torch.load(paths[pos], map_location="cpu", weights_only=True),
                                strict=True)
            self.models[pos] = net.to(self.device).eval()

    def q_values(self, infoset) -> List[float]:
        torch = self._torch
        obs = self._get_obs(infoset)
        z = torch.from_numpy(obs["z_batch"]).float().to(self.device)
        x = torch.from_numpy(obs["x_batch"]).float().to(self.device)
        with torch.no_grad():
            y = self.models[infoset.player_position].forward(z, x, return_value=True)["values"]
        return y.reshape(-1).cpu().tolist()


class PerfectDouAgent:
    """PerfectDou's greedy policy in its Python 3.7 move server (module docs)."""

    FIELDS = ("player_position", "player_hand_cards", "legal_actions", "other_hand_cards", "last_move",
              "last_two_moves", "last_move_dict", "num_cards_left_dict", "played_cards", "bomb_num",
              "card_play_action_seq", "three_landlord_cards")

    def __init__(self, python: str, repo: str):
        import subprocess

        self.name = "perfectdou"
        worker = str(Path(__file__).resolve().parent / "perfectdou_worker.py")
        self.proc = subprocess.Popen([python, worker, "--perfectdou", repo], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1, cwd=repo)
        self.fallbacks = 0

    def act(self, infoset) -> List[int]:
        msg = {k: getattr(infoset, k) for k in self.FIELDS}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("perfectdou worker died")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(f"perfectdou worker: {r['error']}")
        return r["action"]


# ---------------------------------------------------------------------------
# cards and plays
# ---------------------------------------------------------------------------

def card_rank(card: int) -> int:
    return card - 52 + 13 if card >= 52 else card % 13


def card_value(card: int) -> int:
    return VALUE[card_rank(card)]


def play_cards(play: str) -> Tuple[int, ...]:
    """A gmrules play string (``333444-5(BWJ)``, ``Pass``) as sorted DouZero values."""
    if play == "Pass":
        return ()
    s = play.replace("-", "").replace("(BWJ)", "B").replace("(CJ)", "R")
    return tuple(sorted(CHAR_VALUE[c] for c in s))


def hand_strength(cards: Sequence[int]) -> int:
    """heuristic_v1's count: rocket 8 (else big joker 4, small joker 3), each bomb 6,
    each 2 outside a bomb 2."""
    n = collections.Counter(card_rank(c) for c in cards)
    s = 8 if n[13] and n[14] else 3 * n[13] + 4 * n[14]
    s += sum(6 for r in range(13) if n[r] == 4)
    return s + (2 * n[12] if n[12] < 4 else 0)


def max_bid(strength: int) -> int:
    return 3 if strength >= 7 else 2 if strength >= 5 else 1 if strength >= 3 else 0


def full_plays(st, seat: int) -> Dict[str, List[int]]:
    """Every legal full play of ``seat`` → its decision path (a combination with
    kickers is a main decision followed by one kicker decision per unit)."""
    out: Dict[str, List[int]] = {}

    def walk(s, path: List[int]) -> None:
        for a in s.legal_actions(seat):
            name = s.action_string(seat, a)
            if "*" in name:
                nxt = s.clone()
                nxt.apply_action(seat, a)
                walk(nxt, path + [a])
            else:
                out[name] = path + [a]

    walk(st, [])
    return out


def reading(names: List[str], cards: Tuple[int, ...], md) -> str:
    """The gmrules play among ``names`` (several readings of one card set) that
    DouZero's move detector sees: a serial triple is the kicker-free airplane, a
    serial 3+1 the airplane on the chain it names; otherwise the first."""
    if len(names) == 1:
        return names[0]
    t = md.get_move_type(list(cards))
    if t["type"] == 10:
        plain = [n for n in names if "-" not in n]
        if plain:
            return plain[0]
    if t["type"] in (11, 12):
        chain = "".join(VALUE_CHAR[v] * 3 for v in range(t["rank"], t["rank"] + t["len"]))
        hit = [n for n in names if n.split("-")[0] == chain]
        if hit:
            return hit[0]
    return sorted(names)[0]


class Tracker:
    """DouZero's view of the play phase (its ``InfoSet`` fields), kept from the
    gmrules plays: hands, played cards, last moves, the action sequence, bombs."""

    def __init__(self, hands: List[List[int]], landlord: int, kitty: Sequence[int] = ()):
        self.pos = {landlord: "landlord", (landlord + 1) % 3: "landlord_down",
                    (landlord + 2) % 3: "landlord_up"}
        self.hands = {self.pos[s]: sorted(card_value(c) for c in hands[s]) for s in range(3)}
        # the kitty cards the landlord has not played yet (PerfectDou's GameEnv keeps these)
        self.kitty = sorted(card_value(c) for c in kitty)
        self.played = {p: [] for p in POSITIONS}
        self.last_move_dict = {p: [] for p in POSITIONS}
        self.seq: List[List[int]] = []
        self.bombs = 0

    def infoset(self, seat: int, legal: List[List[int]]):
        """DouZero's ``InfoSet`` of ``seat`` to move (GameEnv.get_infoset's fields)."""
        me = self.pos[seat]
        last: List[int] = []                 # GameEnv.get_last_move
        if self.seq:
            last = self.seq[-2] if not self.seq[-1] and len(self.seq) >= 2 else self.seq[-1]
        last_two: List[List[int]] = [[], []]  # GameEnv.get_last_two_moves
        for m in self.seq[-2:]:
            last_two = ([list(m)] + last_two)[:2]
        # GameEnv.get_infoset's order (the DouZero features count cards; PerfectDou's
        # encoder reads the order)
        others = [c for p in ("landlord", "landlord_up", "landlord_down") if p != me for c in self.hands[p]]
        return SimpleNamespace(three_landlord_cards=list(self.kitty),
            player_position=me, player_hand_cards=list(self.hands[me]), legal_actions=legal,
            other_hand_cards=others, last_move=list(last), last_two_moves=last_two,
            last_move_dict={p: list(m) for p, m in self.last_move_dict.items()},
            num_cards_left_dict={p: len(h) for p, h in self.hands.items()},
            played_cards={p: list(m) for p, m in self.played.items()}, bomb_num=self.bombs,
            card_play_action_seq=[list(m) for m in self.seq])

    def play(self, seat: int, cards: Tuple[int, ...]) -> None:
        p = self.pos[seat]
        move = list(cards)
        for v in move:
            self.hands[p].remove(v)
            if p == "landlord" and v in self.kitty:
                self.kitty.remove(v)
        self.played[p] += move
        self.last_move_dict[p] = move
        self.seq.append(move)
        if (len(move) == 4 and len(set(move)) == 1) or move == [20, 30]:
            self.bombs += 1


# ---------------------------------------------------------------------------
# one game
# ---------------------------------------------------------------------------

def in_auction(st, seat: int) -> bool:
    return any(1 <= a <= 3 for a in st.legal_actions(seat))


def play_game(seed: int, index: int, agents: Dict[str, DouZeroAgent], mix: Sequence[str],
              q_top: int, md, gmsim) -> Dict:
    """Game ``index`` of run ``seed``: deal, heuristic auction, DouZero play."""
    rng = random.Random(f"doudizhu:{seed}:{index}")
    deck = list(range(54))
    rng.shuffle(deck)
    pos = rng.randrange(51)               # the face-up card's deal position
    st = gmsim.State(GAME, {})
    st.apply_chance(pos)
    for c in deck[:51]:
        st.apply_chance(51 + c)
    hands = [sorted(deck[k] for k in range(51) if k % 3 == s) for s in range(3)]
    strength = [hand_strength(h) for h in hands]
    rec: Dict = {"id": f"dz{seed}-{index}", "hands": hands, "kitty": sorted(deck[51:]),
                 "face_up": deck[pos], "auction": [], "plays": []}
    landlord = None
    while True:
        kind, seats = st.turn()
        if kind != "act" or not in_auction(st, seats[0]):
            break
        seat = seats[0]
        legal = st.legal_actions(seat)
        want = max_bid(strength[seat])
        bid = want if want in legal else 0
        st.apply_action(seat, bid)
        rec["auction"].append([seat, bid])
        if bid:
            landlord = seat
    roles = {}
    if st.turn()[0] == "act":
        kitty = rec["kitty"]
        full = [h + kitty if s == landlord else h for s, h in enumerate(hands)]
        tr = Tracker(full, landlord, kitty)
        roles = {s: agents[rng.choice(list(mix))] for s in range(3)}
        q_rows = []
        fallbacks = 0
        while True:
            kind, seats = st.turn()
            if kind != "act":
                break
            seat = seats[0]
            plays = full_plays(st, seat)
            by_cards: Dict[Tuple[int, ...], List[str]] = collections.defaultdict(list)
            for name in plays:
                by_cards[play_cards(name)].append(name)
            options = list(by_cards)
            if len(options) == 1:
                best = 0
            elif not hasattr(roles[seat], "q_values"):
                # a policy (PerfectDou): its move among the candidates
                act = tuple(roles[seat].act(tr.infoset(seat, [list(o) for o in options])))
                if act in by_cards:
                    best = options.index(act)
                else:
                    best, fallbacks = 0, fallbacks + 1
            else:
                q = roles[seat].q_values(tr.infoset(seat, [list(o) for o in options]))
                best = max(range(len(q)), key=lambda i: (q[i], -i))
                if q_top > 0:
                    top = sorted(range(len(q)), key=lambda i: (-q[i], i))[:q_top]
                    q_rows.append([len(rec["plays"]),
                                   [[reading(by_cards[options[i]], options[i], md),
                                     round(q[i], 5)] for i in top]])
            cards = options[best]
            name = reading(by_cards[cards], cards, md)
            for a in plays[name]:
                st.apply_action(seat, a)
            tr.play(seat, cards)
            rec["plays"].append([seat, name])
        if q_top > 0:
            rec["q"] = q_rows
    rec["returns"] = st.returns()
    rec["agents"] = [roles[s].name if s in roles else None for s in range(3)]
    rec["meta"] = {"bidding": "heuristic_v1", "strength": strength, "generator": GENERATOR,
                   "seed": seed, "index": index, "douzero": DOUZERO_COMMIT[:12],
                   "weights": f"{WEIGHTS_REPO}@{WEIGHTS_REVISION[:12]}"}
    if any(a is not None and a.startswith("perfectdou") for a in rec["agents"]):
        rec["meta"]["perfectdou"] = PERFECTDOU_COMMIT[:12]
        rec["meta"]["perfectdou_fallbacks"] = fallbacks if roles else 0
    return rec


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

_W: Dict = {}


def _init_worker(pkg: str, ckpts: Dict[str, Dict[str, str]], device: str, threads: int,
                 perfectdou: Optional[Tuple[str, str]] = None) -> None:
    import torch

    torch.set_num_threads(threads)
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    import gmsim
    from douzero.env import move_detector as md

    _W["agents"] = {a: DouZeroAgent(a, {p: Path(x) for p, x in paths.items()}, device)
                    for a, paths in ckpts.items()}
    if perfectdou is not None:
        _W["agents"]["perfectdou"] = PerfectDouAgent(*perfectdou)
    _W["md"], _W["gmsim"] = md, gmsim


def _run_part(job: Tuple[str, int, int, int, Sequence[str], int]) -> Tuple[str, int, float]:
    out, part, lo, hi, mix, q_top = job
    seed = _W["seed"]
    path = Path(out) / f"part-{part:05d}.jsonl.gz"
    t0 = time.time()
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt") as fh:
        for i in range(lo, hi):
            rec = play_game(seed, i, _W["agents"], mix, q_top, _W["md"], _W["gmsim"])
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    tmp.replace(path)
    return str(path), hi - lo, time.time() - t0


def _init_and_seed(pkg, ckpts, device, threads, seed, perfectdou=None):
    _init_worker(pkg, ckpts, device, threads, perfectdou)
    _W["seed"] = seed


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="output directory of part-*.jsonl.gz files")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first", type=int, default=0, help="index of the first game")
    ap.add_argument("--part_size", type=int, default=1000, help="games per output part")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--agents", default="adp,wp",
                    help="agents drawn per game and role: adp, wp, sl (DouZero checkpoints), perfectdou "
                         "(comma list)")
    ap.add_argument("--perfectdou_python", default=os.environ.get("GM_PERFECTDOU_PYTHON"),
                    help="a Python 3.7 with onnxruntime 1.7 + numpy (for --agents perfectdou)")
    ap.add_argument("--perfectdou_repo", default=os.environ.get("GM_PERFECTDOU_REPO"),
                    help=f"a checkout of Netease-Games-AI-Lab-Guangzhou/PerfectDou at {PERFECTDOU_COMMIT[:12]}")
    ap.add_argument("--bidding", default="heuristic_v1", choices=BIDDING)
    ap.add_argument("--q_top", type=int, default=8,
                    help="record the top-K moves by DouZero Q per decision (0: none)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default=None, help="download cache ($GM_REFERENCE_CACHE)")
    ap.add_argument("--fetch_only", action="store_true",
                    help="only download the DouZero sources and checkpoints (a login node "
                         "with internet, before generating on compute nodes)")
    args = ap.parse_args(argv)
    mix = [a.strip() for a in args.agents.split(",") if a.strip()]
    bad = sorted(set(mix) - set(CHECKPOINTS) - {"perfectdou"})
    if bad or not mix:
        raise SystemExit(f"--agents {args.agents}: choose from {sorted(CHECKPOINTS) + ['perfectdou']}")
    perfectdou = None
    if "perfectdou" in mix:
        if not (args.perfectdou_python and args.perfectdou_repo):
            raise SystemExit("--agents perfectdou needs --perfectdou_python and --perfectdou_repo")
        perfectdou = (args.perfectdou_python, args.perfectdou_repo)
    pkg = str(fetch_douzero(args.cache))
    ckpts = {a: {p: str(x) for p, x in paths.items()}
             for a, paths in fetch_checkpoints(sorted(set(mix) - {"perfectdou"}), args.cache).items()}
    if args.fetch_only:
        print(json.dumps({"douzero": pkg, "checkpoints": ckpts}), flush=True)
        return
    if not args.out:
        raise SystemExit("--out is required (unless --fetch_only)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, lo in enumerate(range(args.first, args.first + args.games, args.part_size)):
        hi = min(lo + args.part_size, args.first + args.games)
        part = lo // args.part_size
        if (out / f"part-{part:05d}.jsonl.gz").exists():
            continue                         # resumable: parts are written atomically
        jobs.append((str(out), part, lo, hi, mix, args.q_top))
    t0, done = time.time(), 0
    init = (pkg, ckpts, args.device, args.threads, args.seed, perfectdou)
    if args.workers <= 1:
        _init_and_seed(*init)
        results = map(_run_part, jobs)
    else:
        import multiprocessing as mp

        ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_and_seed, initargs=init)
        results = ex.map(_run_part, jobs)
    for path, n, dt in results:
        done += n
        rate = done / max(time.time() - t0, 1e-9)
        print(json.dumps({"part": path, "games": n, "seconds": round(dt, 1), "total": done,
                          "games_per_s": round(rate, 2)}), flush=True)
    print(json.dumps({"out": str(out), "games": done, "skipped_parts":
                      len(range(args.first, args.first + args.games, args.part_size)) - len(jobs),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
