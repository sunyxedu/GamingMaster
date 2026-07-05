"""Tenhou mjai-format logs (.mjson) → riichi mahjong training shards.

Converts mjai event-stream logs (one JSON event per line — the format
the community tenhou-to-mjai mirrors distribute, see ``fetch.md``) into
the shared shard format consumed by ``data_processing.games.dataset``
and the mahjong adapter (:class:`model.games.mahjong.MahjongEncoder` /
``MahjongDecoder``).

One sample per (game, seat): 4 samples per game, each holding that
seat's decision-point frames across every kyoku of the game.

Decision points (exactly these frames are emitted)
--------------------------------------------------
(a) **Own turn** — after the seat's own ``tsumo`` (the drawn tile is in
    hand, focus flag on its kind).  The label is read by lookahead:

        dahai            → ACTION_DAHAI  (tile = discarded kind)
        reach + dahai    → ACTION_RIICHI (tile = discarded kind)
        ankan / kakan    → ACTION_KAN    (tile = kan kind)
        hora (tsumo win) → ACTION_HORA
        ryukyoku         → ACTION_RYUKYOKU (kyushu kyuhai)

    Post-call discards are own-turn decisions too: after the seat's own
    ``chi``/``pon`` the forced discard gets a frame (no focus tile — no
    tile was drawn), label ACTION_DAHAI.

(b) **Call opportunity** — after another player's ``dahai``, for seat s
    (state already includes the discard; focus flag on its kind, focus
    actor on the discarder).  v1 emits a frame only when:

        - s actually called on that discard (``chi``/``pon``/
          ``daiminkan``/``hora``-ron, including multi-ron) — label is
          the call (chi tile = base/lowest kind of the run; ron carries
          no tile), or
        - s is cheaply detectable as call-eligible and did NOT call —
          label ACTION_PASS.  Cheap eligibility:
            pon/kan: hand count of the kind ≥ 2,
            chi:     s is the discarder's shimocha (s == discarder+1)
                     and holds one of the (k−2,k−1) / (k−1,k+1) /
                     (k+1,k+2) partner pairs within the suit.
          Ron eligibility is NOT detected (needs a win checker), so
          missed-ron pass frames are not emitted; seats that already
          declared riichi are skipped (they cannot call chi/pon/kan);
          no pass frames at wall 0 (no calls possible off the last
          discard except ron, which we don't detect anyway).

State tracked per kyoku: concealed hands (tehais + draws + calls −
discards), red-five possession, melds, discard ponds (a claimed tile
moves from the pond into the caller's meld), riichi flags, dora
markers, scores (resynced from every ``start_kyoku``, updated by
hora/ryukyoku deltas and riichi sticks), and wall count (70 − draws;
exact, since kan replacement draws are ``tsumo`` events too).
Ignored simplifications (documented): kuikae restrictions are not
checked for chi eligibility, and furiten is not modelled.

Value target
------------
``value`` [T] is constant per sample: the seat's **final game
placement**, ranked by end-of-game score with ties broken by seat
index (Tenhou convention — the seat closer to the initial East wins
the tie), mapped ``1st → +1, 2nd → +1/3, 3rd → −1/3, 4th → −1``.

CLI
---
    uv run python -m data_processing.games.mahjong.convert \
        --input mjai_2009.zip --output_dir shards/mahjong \
        --shard_size 128 --max_games 500

``--input`` may be a ``.zip`` of ``.mjson`` files (read via zipfile
streaming, per-member gzip transparently handled — the community
archives store gzipped members), a directory of ``.mjson``/
``.mjson.gz`` files, or a single file.  Games that fail to parse are
counted and skipped.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import sys
import zipfile
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from data_processing.games.dataset import save_shard
from model.games.mahjong import (
    ACTION_CHI,
    ACTION_DAHAI,
    ACTION_HORA,
    ACTION_KAN,
    ACTION_NAMES,
    ACTION_PASS,
    ACTION_PON,
    ACTION_RIICHI,
    ACTION_RYUKYOKU,
    N_MJ_PLAYERS,
    N_TILE_KINDS,
    PLAYER_FEATURE_DIM,
    TABLE_FEATURE_DIM,
    TILE_FEATURE_DIM,
)

GAME_NAME = "mahjong"
WALL_TILES = 70            # 136 − 14 dead wall − 4×13 dealt
DISCARD_SCALE = 24.0
SCORE_SCALE = 100000.0
PLACEMENT_VALUES = (1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)

_SUITS = "mps"
_HONORS = ["E", "S", "W", "N", "P", "F", "C"]   # winds, then white/green/red dragons

#: mjai tile string → 34-kind index.  Red fives ("5mr"/"0m" style both
#: accepted) map to their ordinary 5 kind; redness is a separate flag.
TILE_KIND: Dict[str, int] = {}
for _si, _s in enumerate(_SUITS):
    for _n in range(1, 10):
        TILE_KIND[f"{_n}{_s}"] = _si * 9 + _n - 1
    TILE_KIND[f"5{_s}r"] = _si * 9 + 4
    TILE_KIND[f"0{_s}"] = _si * 9 + 4
for _i, _h in enumerate(_HONORS):
    TILE_KIND[_h] = 27 + _i


class SkipGame(Exception):
    """Raised when a game cannot be converted (inconsistent/hidden state)."""


def tile_kind(pai: str) -> int:
    try:
        return TILE_KIND[pai]
    except KeyError:
        raise SkipGame(f"unknown tile {pai!r}") from None


def is_red(pai: str) -> bool:
    return pai.endswith("r") or pai.startswith("0")


def dora_from_marker(k: int) -> int:
    """Dora kind indicated by a marker kind (successor with wraparound)."""
    if k < 27:                      # suits: 9 → 1 of the same suit
        base = (k // 9) * 9
        return base + (k - base + 1) % 9
    if k < 31:                      # winds E→S→W→N→E
        return 27 + (k - 27 + 1) % 4
    return 31 + (k - 31 + 1) % 3    # dragons P→F→C→P


def is_terminal_or_honor(k: int) -> bool:
    return k >= 27 or k % 9 in (0, 8)


# ---------------------------------------------------------------------------
# Kyoku state tracking
# ---------------------------------------------------------------------------

class KyokuState:
    """Full public + own-hand state, updated event by event."""

    def __init__(self, ev: Dict):
        self.oya: int = ev["oya"]
        self.bakaze: str = ev["bakaze"]
        self.kyoku: int = ev.get("kyoku", 1)
        self.honba: int = ev.get("honba", 0)
        self.kyotaku: int = ev.get("kyotaku", 0)
        self.scores: List[int] = list(ev["scores"])
        self.dora_markers: List[int] = [tile_kind(ev["dora_marker"])]
        self.hands = [[0] * N_TILE_KINDS for _ in range(4)]
        self.reds = [set() for _ in range(4)]     # kinds with a red 5 in hand
        self.melds = [[0] * N_TILE_KINDS for _ in range(4)]  # tile counts
        self.meld_n = [0] * 4
        self.kan_n = [0] * 4
        self.discards = [[0] * N_TILE_KINDS for _ in range(4)]
        self.disc_n = [0] * 4
        self.riichi = [False] * 4
        self.riichi_pending = [False] * 4
        self.wall = WALL_TILES
        for s, tehai in enumerate(ev["tehais"]):
            if len(tehai) != 13:
                raise SkipGame(f"seat {s} dealt {len(tehai)} tiles")
            for p in tehai:
                if p == "?":
                    raise SkipGame("hidden tehai (not a full game record)")
                self._add(s, p)

    # -- hand bookkeeping ------------------------------------------------

    def _add(self, s: int, pai: str) -> None:
        k = tile_kind(pai)
        self.hands[s][k] += 1
        if is_red(pai):
            self.reds[s].add(k)

    def _remove(self, s: int, pai: str) -> int:
        k = tile_kind(pai)
        if self.hands[s][k] <= 0:
            raise SkipGame(f"seat {s} removes {pai} not in hand")
        self.hands[s][k] -= 1
        if is_red(pai) or self.hands[s][k] == 0:
            self.reds[s].discard(k)
        return k

    def draw(self, s: int, pai: str) -> int:
        self.wall -= 1
        self._add(s, pai)
        return tile_kind(pai)

    def discard(self, s: int, pai: str) -> int:
        k = self._remove(s, pai)
        self.discards[s][k] += 1
        self.disc_n[s] += 1
        if self.riichi_pending[s]:
            self.riichi_pending[s] = False
            self.riichi[s] = True
        return k

    def call(self, ev: Dict) -> None:
        """chi / pon / daiminkan: consumed leave the hand, the claimed
        tile leaves the discarder's pond, everything enters the meld."""
        c, t = ev["actor"], ev["target"]
        k = tile_kind(ev["pai"])
        if self.discards[t][k] <= 0:
            raise SkipGame(f"claimed {ev['pai']} missing from pond of {t}")
        self.discards[t][k] -= 1
        self.disc_n[t] -= 1
        self.melds[c][k] += 1
        for pai in ev["consumed"]:
            self.melds[c][self._remove(c, pai)] += 1
        self.meld_n[c] += 1
        if ev["type"] == "daiminkan":
            self.kan_n[c] += 1

    def ankan(self, ev: Dict) -> None:
        c = ev["actor"]
        for pai in ev["consumed"]:
            self.melds[c][self._remove(c, pai)] += 1
        self.meld_n[c] += 1
        self.kan_n[c] += 1

    def kakan(self, ev: Dict) -> None:
        c = ev["actor"]
        self.melds[c][self._remove(c, ev["pai"])] += 1
        self.kan_n[c] += 1          # upgrades an existing pon: meld_n unchanged

    # -- call eligibility (cheap detection, see module docstring) ---------

    def can_pon(self, s: int, k: int) -> bool:
        return self.hands[s][k] >= 2

    def can_chi(self, s: int, discarder: int, k: int) -> bool:
        if s != (discarder + 1) % 4 or k >= 27:
            return False
        r, base, h = k % 9, k - k % 9, self.hands[s]

        def has(x: int) -> bool:
            return h[base + x] > 0

        return (
            (r >= 2 and has(r - 2) and has(r - 1))
            or (0 < r < 8 and has(r - 1) and has(r + 1))
            or (r <= 6 and has(r + 1) and has(r + 2))
        )


# ---------------------------------------------------------------------------
# Frame encoding (layouts documented in model/games/mahjong.py)
# ---------------------------------------------------------------------------

def encode_frame(
    st: KyokuState,
    seat: int,
    focus_kind: Optional[int],
    own_turn: bool,
    focus_actor: Optional[int] = None,
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    """One decision-point frame from ``seat``'s perspective →
    (tiles [34, 16], players [4, 12], table [12]) as nested lists."""
    shimocha, toimen, kamicha = (seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4
    dora_kinds = {dora_from_marker(m) for m in st.dora_markers}
    marker_counts = [0] * N_TILE_KINDS
    for m in st.dora_markers:
        marker_counts[m] += 1

    tiles = []
    for k in range(N_TILE_KINDS):
        row = [0.0] * TILE_FEATURE_DIM
        row[min(st.hands[seat][k], 4)] = 1.0
        row[5] = st.melds[seat][k] / 4.0
        row[6] = (st.melds[shimocha][k] + st.melds[toimen][k]
                  + st.melds[kamicha][k]) / 4.0
        row[7] = st.discards[seat][k] / 4.0
        row[8] = st.discards[shimocha][k] / 4.0
        row[9] = st.discards[toimen][k] / 4.0
        row[10] = st.discards[kamicha][k] / 4.0
        row[11] = 1.0 if k in dora_kinds else 0.0
        visible = (st.hands[seat][k] + marker_counts[k]
                   + sum(st.melds[p][k] + st.discards[p][k] for p in range(4)))
        row[12] = min(visible, 4) / 4.0
        row[13] = 1.0 if is_terminal_or_honor(k) else 0.0
        row[14] = 1.0 if k in st.reds[seat] else 0.0
        row[15] = 1.0 if k == focus_kind else 0.0
        tiles.append(row)

    players = []
    for rel in range(N_MJ_PLAYERS):
        p = (seat + rel) % 4
        row = [0.0] * PLAYER_FEATURE_DIM
        row[0] = st.scores[p] / SCORE_SCALE
        row[1] = 1.0 if st.riichi[p] else 0.0
        row[2 + (p - st.oya) % 4] = 1.0          # seat wind (oya = East)
        row[6] = 1.0 if p == st.oya else 0.0
        row[7] = st.meld_n[p] / 4.0
        row[8] = st.disc_n[p] / DISCARD_SCALE
        row[9] = st.kan_n[p] / 4.0
        row[10] = 1.0 if p == focus_actor else 0.0
        row[11] = 1.0
        players.append(row)

    table = [0.0] * TABLE_FEATURE_DIM
    table["ESWN".index(st.bakaze)] = 1.0
    table[4] = (st.kyoku - 1) / 4.0
    table[5] = st.honba / 8.0
    table[6] = st.kyotaku / 4.0
    table[7] = max(st.wall, 0) / float(WALL_TILES)
    table[8] = len(st.dora_markers) / 5.0
    table[9] = 1.0 if own_turn else 0.0
    table[10] = sum(st.riichi) / 4.0
    table[11] = 1.0
    return tiles, players, table


# ---------------------------------------------------------------------------
# Game conversion
# ---------------------------------------------------------------------------

def _own_turn_label(events: List[Dict], i: int, actor: int) -> Optional[Tuple[int, int]]:
    """Label for the decision after ``events[i]`` (the actor's tsumo)."""
    if i + 1 >= len(events):
        return None
    nxt = events[i + 1]
    t = nxt["type"]
    if t == "dahai" and nxt["actor"] == actor:
        return ACTION_DAHAI, tile_kind(nxt["pai"])
    if t == "reach" and nxt["actor"] == actor:
        if i + 2 < len(events) and events[i + 2]["type"] == "dahai" \
                and events[i + 2]["actor"] == actor:
            return ACTION_RIICHI, tile_kind(events[i + 2]["pai"])
        return None
    if t == "ankan" and nxt["actor"] == actor:
        return ACTION_KAN, tile_kind(nxt["consumed"][0])
    if t == "kakan" and nxt["actor"] == actor:
        return ACTION_KAN, tile_kind(nxt["pai"])
    if t == "hora" and nxt["actor"] == actor and nxt.get("target") == actor:
        return ACTION_HORA, 0
    if t == "ryukyoku":
        return ACTION_RYUKYOKU, 0   # kyushu kyuhai (abortive draw on own turn)
    return None


def _discard_calls(events: List[Dict], i: int, discarder: int) -> Dict[int, Tuple[int, int]]:
    """Actual calls made on the discard at ``events[i]`` → {seat: (type, tile)}."""
    calls: Dict[int, Tuple[int, int]] = {}
    j = i + 1
    while j < len(events) and events[j]["type"] == "hora" \
            and events[j].get("target") == discarder \
            and events[j]["actor"] != discarder:
        calls[events[j]["actor"]] = (ACTION_HORA, 0)
        j += 1
    if not calls and j < len(events):
        nx = events[j]
        if nx["type"] in ("chi", "pon", "daiminkan") and nx.get("target") == discarder:
            k = tile_kind(nx["pai"])
            if nx["type"] == "chi":
                base = min([k] + [tile_kind(p) for p in nx["consumed"]])
                calls[nx["actor"]] = (ACTION_CHI, base)
            elif nx["type"] == "pon":
                calls[nx["actor"]] = (ACTION_PON, k)
            else:
                calls[nx["actor"]] = (ACTION_KAN, k)
    return calls


def convert_game(events: List[Dict], name: str = "",
                 min_frames: int = 1) -> List[Dict]:
    """One mjai event stream → up to 4 samples (one per seat).

    Raises :class:`SkipGame` on inconsistent/hidden state.
    """
    frames = {
        s: {"tiles": [], "players": [], "table": [], "type": [], "tile": []}
        for s in range(4)
    }

    def emit(seat, label, focus_kind, own_turn, focus_actor=None):
        ti, pl, ta = encode_frame(st, seat, focus_kind, own_turn, focus_actor)
        f = frames[seat]
        f["tiles"].append(ti)
        f["players"].append(pl)
        f["table"].append(ta)
        f["type"].append(label[0])
        f["tile"].append(label[1])

    st: Optional[KyokuState] = None
    n_kyoku = 0
    for i, ev in enumerate(events):
        t = ev["type"]
        if t == "start_kyoku":
            st = KyokuState(ev)
            n_kyoku += 1
            continue
        if st is None:
            continue

        if t == "tsumo":
            a = ev["actor"]
            drawn = st.draw(a, ev["pai"])
            label = _own_turn_label(events, i, a)
            if label is not None:
                emit(a, label, drawn, own_turn=True)
        elif t == "dahai":
            a = ev["actor"]
            k = st.discard(a, ev["pai"])
            calls = _discard_calls(events, i, a)
            for rel in (1, 2, 3):
                s = (a + rel) % 4
                if s in calls:
                    emit(s, calls[s], k, own_turn=False, focus_actor=a)
                elif (not st.riichi[s] and st.wall > 0
                      and (st.can_pon(s, k) or st.can_chi(s, a, k))):
                    emit(s, (ACTION_PASS, 0), k, own_turn=False, focus_actor=a)
        elif t in ("chi", "pon"):
            st.call(ev)
            # Forced post-call discard is an own-turn decision (no draw,
            # so no focus tile).
            if i + 1 < len(events) and events[i + 1]["type"] == "dahai" \
                    and events[i + 1]["actor"] == ev["actor"]:
                emit(ev["actor"],
                     (ACTION_DAHAI, tile_kind(events[i + 1]["pai"])),
                     None, own_turn=True)
        elif t == "daiminkan":
            st.call(ev)              # replacement draw arrives as a tsumo event
        elif t == "ankan":
            st.ankan(ev)
        elif t == "kakan":
            st.kakan(ev)
        elif t == "reach":
            st.riichi_pending[ev["actor"]] = True
        elif t == "reach_accepted":
            st.scores[ev["actor"]] -= 1000
            st.kyotaku += 1
        elif t == "hora":
            st.scores = [s0 + d for s0, d in zip(st.scores, ev.get("deltas", [0] * 4))]
            st.kyotaku = 0
        elif t == "ryukyoku":
            st.scores = [s0 + d for s0, d in zip(st.scores, ev.get("deltas", [0] * 4))]
        elif t == "dora":
            st.dora_markers.append(tile_kind(ev["dora_marker"]))
        # start_game / end_kyoku / end_game: nothing to do

    if st is None or n_kyoku == 0:
        raise SkipGame("no kyoku")

    # Final placement (score desc, ties → lower seat index wins: Tenhou rule)
    order = sorted(range(4), key=lambda s: (-st.scores[s], s))
    placement = {s: r for r, s in enumerate(order)}

    samples = []
    for s in range(4):
        f = frames[s]
        T = len(f["type"])
        if T < min_frames:
            continue
        value = PLACEMENT_VALUES[placement[s]]
        samples.append({
            "obs": {
                "tiles_seq": torch.tensor(f["tiles"], dtype=torch.float32),
                "players_seq": torch.tensor(f["players"], dtype=torch.float32),
                "table_seq": torch.tensor(f["table"], dtype=torch.float32),
            },
            "tgt": {
                "action_type": torch.tensor(f["type"], dtype=torch.long),
                "tile_idx": torch.tensor(f["tile"], dtype=torch.long),
                "value": torch.full((T,), value, dtype=torch.float32),
            },
            "meta": {
                "name": name,
                "seat": s,
                "kyokus": n_kyoku,
                "placement": placement[s] + 1,
                "final_score": st.scores[s],
            },
        })
    return samples


# ---------------------------------------------------------------------------
# Input iteration (zip / dir / single file, transparent gzip)
# ---------------------------------------------------------------------------

def _decode(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_events(text: str) -> List[Dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def iter_games(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (name, mjson text) from a zip, a directory, or one file."""
    if os.path.isfile(path) and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.endswith((".mjson", ".mjson.gz")):
                    continue
                yield info.filename, _decode(z.read(info))
    elif os.path.isdir(path):
        pats = ("**/*.mjson", "**/*.mjson.gz")
        files = sorted(
            f for pat in pats
            for f in glob.glob(os.path.join(path, pat), recursive=True)
        )
        for f in files:
            with open(f, "rb") as fh:
                yield os.path.basename(f), _decode(fh.read())
    else:
        with open(path, "rb") as fh:
            yield os.path.basename(path), _decode(fh.read())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True,
                   help=".zip of .mjson files, a directory, or one .mjson")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=128)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--min_frames", type=int, default=8,
                   help="skip per-seat samples with fewer frames")
    args = p.parse_args(argv)

    stats = {
        "games": 0, "skipped": 0, "samples": 0, "frames": 0, "shards": 0,
        "action_hist": {n: 0 for n in ACTION_NAMES},
    }
    pending: List[Dict] = []
    shard_idx = 0

    def flush(force: bool = False):
        nonlocal shard_idx, pending
        while len(pending) >= args.shard_size or (force and pending):
            chunk, pending = pending[:args.shard_size], pending[args.shard_size:]
            save_shard(
                os.path.join(args.output_dir, f"mahjong_{shard_idx:05d}.pt"),
                GAME_NAME, chunk,
            )
            shard_idx += 1
            stats["shards"] += 1

    for name, text in iter_games(args.input):
        if args.max_games is not None and stats["games"] >= args.max_games:
            break
        try:
            samples = convert_game(parse_events(text), name=name,
                                   min_frames=args.min_frames)
        except (SkipGame, KeyError, ValueError, json.JSONDecodeError) as e:
            stats["skipped"] += 1
            if stats["skipped"] <= 10:
                print(f"skip {name}: {e}", file=sys.stderr)
            continue
        stats["games"] += 1
        for s in samples:
            stats["samples"] += 1
            stats["frames"] += int(s["tgt"]["action_type"].shape[0])
            for a in s["tgt"]["action_type"].tolist():
                stats["action_hist"][ACTION_NAMES[a]] += 1
        pending.extend(samples)
        flush()

    flush(force=True)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
