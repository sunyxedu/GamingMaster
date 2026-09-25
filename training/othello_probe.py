"""Othello machinery for the DIAL step-0 headroom probe (scripts/dial_probe.py):
bitboard rules, the training-frame semantics of `gmdata othello convert`, an
exact endgame solver, a batched streaming evaluator around ``GameNetwork.step``
and a depth-limited lookahead search that uses the network's own value head.

Frame semantics — pinned bit-for-bit to the converter by tests/test_dial_probe.py
(against tests/fixtures/golden/othello.json):

  * one training sample = one whole game; frame t is the position BEFORE the
    t-th half-move, seen from the side to move (planes: mover discs /
    opponent discs / empty, a pure channel swap).  Both colours' frames sit
    in the SAME sequence, alternating, so the temporal state of the network
    carries the whole game history whoever is to move.
  * a side with no legal move while the opponent has one gets a PASS frame
    (tgt.place_idx 0, tgt.passed True); the sequence then continues with the
    opponent's frame of the same board.
  * the terminal position (neither side can move) is never a frame.
  * tgt.value[t] = sign(final own discs − final opponent discs) for the mover
    of frame t, in {−1, 0, +1} (W/D/L; empties never change the sign).

Perspective / sign conventions of the search:

  * V(frame), the network value of a frame, estimates that value target: the
    expected W/D/L outcome for THAT FRAME'S MOVER.
  * every search value is expressed for the ROOT mover m:
        leaf decision frame with mover d:  +V if d == m else −V
        terminal position:                 sign(final m discs − opponent discs)
  * after a move the stream continues with the next frame the training data
    would contain.  If the side to move must pass, that is its pass frame
    followed by the other side's decision frame; the pass frame is fed (it is
    part of the training history) but not evaluated, and the value is read at
    the decision frame — so when m's move forces the opponent to pass, the
    leaf is m's own frame and its value is taken with sign +1.
  * interior nodes take the max over children when their decision player is
    m and the min otherwise (depth-limited minimax, i.e. negamax written in
    the root mover's frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data_processing.games.othello.rules import BLACK, WHITE

PASS = -1
FULL = (1 << 64) - 1
_FILE_A = 0x0101010101010101            # col 0 (square = row * 8 + col)
_NOT_A = FULL & ~_FILE_A
_NOT_H = FULL & ~(_FILE_A << 7)
# (shift, wrap mask) per direction; the mask drops bits that wrapped a row
_LEFT = ((1, _NOT_A), (8, FULL), (9, _NOT_A), (7, _NOT_H))    # E, S, SE, SW
_RIGHT = ((1, _NOT_H), (8, FULL), (9, _NOT_H), (7, _NOT_A))   # W, N, NW, NE

INITIAL_BLACK = (1 << (3 * 8 + 4)) | (1 << (4 * 8 + 3))
INITIAL_WHITE = (1 << (3 * 8 + 3)) | (1 << (4 * 8 + 4))


# ---------------------------------------------------------------------------
# Bitboard rules (same rules as data_processing/games/othello/rules.py)
# ---------------------------------------------------------------------------

def other(player: int) -> int:
    return BLACK + WHITE - player


def legal_moves(own: int, opp: int) -> int:
    """Bitboard of ``own``'s legal moves (Kogge-Stone style fills; a run of at
    most 6 opponent discs fits between two squares of a line)."""
    empty = FULL & ~(own | opp)
    moves = 0
    for s, m in _LEFT:
        mo = opp & m
        x = (own << s) & mo
        x |= (x << s) & mo
        x |= (x << s) & mo
        x |= (x << s) & mo
        x |= (x << s) & mo
        x |= (x << s) & mo
        moves |= (x << s) & m & empty
    for s, m in _RIGHT:
        mo = opp & m
        x = (own >> s) & mo
        x |= (x >> s) & mo
        x |= (x >> s) & mo
        x |= (x >> s) & mo
        x |= (x >> s) & mo
        x |= (x >> s) & mo
        moves |= (x >> s) & m & empty
    return moves


def flips(own: int, opp: int, sq: int) -> int:
    """Discs flipped if ``own`` plays ``sq`` (0 ⇔ illegal; ``sq`` assumed empty)."""
    bit = 1 << sq
    f = 0
    for s, m in _LEFT:
        mo = opp & m
        x = (bit << s) & mo
        if x:
            x |= (x << s) & mo
            x |= (x << s) & mo
            x |= (x << s) & mo
            x |= (x << s) & mo
            x |= (x << s) & mo
            if (x << s) & m & own:
                f |= x
    for s, m in _RIGHT:
        mo = opp & m
        x = (bit >> s) & mo
        if x:
            x |= (x >> s) & mo
            x |= (x >> s) & mo
            x |= (x >> s) & mo
            x |= (x >> s) & mo
            x |= (x >> s) & mo
            if (x >> s) & m & own:
                f |= x
    return f


def play(own: int, opp: int, sq: int) -> Tuple[int, int]:
    """(own', opp') after ``own`` plays ``sq``; ValueError if illegal."""
    if (own | opp) >> sq & 1:
        raise ValueError(f"square {sq} is occupied")
    f = flips(own, opp, sq)
    if not f:
        raise ValueError(f"square {sq} flips nothing")
    return own | f | (1 << sq), opp & ~f


def iter_bits(b: int) -> Iterator[int]:
    while b:
        low = b & -b
        yield low.bit_length() - 1
        b ^= low


def is_game_over(own: int, opp: int) -> bool:
    return not legal_moves(own, opp) and not legal_moves(opp, own)


def sides(black: int, white: int, player: int) -> Tuple[int, int]:
    """(player's discs, opponent's discs)."""
    return (black, white) if player == BLACK else (white, black)


def colours(own: int, opp: int, player: int) -> Tuple[int, int]:
    """Inverse of :func:`sides`: (black, white)."""
    return (own, opp) if player == BLACK else (opp, own)


def n_empties(black: int, white: int) -> int:
    return 64 - (black | white).bit_count()


def outcome_sign(own: int, opp: int) -> float:
    """sign(own discs − opponent discs) — the value-target scale."""
    a, b = own.bit_count(), opp.bit_count()
    return float((a > b) - (a < b))


def final_diff(own: int, opp: int) -> int:
    """Final disc differential for ``own`` with the empties going to the
    winner (WTHOR / tournament scoring; same sign as the raw count)."""
    p, o = own.bit_count(), opp.bit_count()
    if p > o:
        return 64 - 2 * o
    if p < o:
        return 2 * p - 64
    return 0


def board_from_bytes(board: Sequence[int]) -> Tuple[int, int]:
    """rules.py bytearray (64 squares, EMPTY/BLACK/WHITE) → (black, white)."""
    black = white = 0
    for i, v in enumerate(board):
        if v == BLACK:
            black |= 1 << i
        elif v == WHITE:
            white |= 1 << i
    return black, white


_SHIFTS = np.arange(64, dtype=np.uint64)


def planes_batch(own_opp: Sequence[Tuple[int, int]]) -> np.ndarray:
    """[N, 3, 8, 8] float32 own / opponent / empty planes — the layout of
    ``rules.board_planes`` and of the converter's obs.board_seq."""
    if not own_opp:
        return np.zeros((0, 3, 8, 8), dtype=np.float32)
    own = np.array([o for o, _ in own_opp], dtype=np.uint64)[:, None]
    opp = np.array([p for _, p in own_opp], dtype=np.uint64)[:, None]
    a = ((own >> _SHIFTS) & np.uint64(1)).astype(np.float32)
    b = ((opp >> _SHIFTS) & np.uint64(1)).astype(np.float32)
    return np.stack([a, b, 1.0 - a - b], axis=1).reshape(-1, 3, 8, 8)


# ---------------------------------------------------------------------------
# Training-frame semantics (gmdata othello convert)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """One training frame: the board BEFORE ``mover`` acts, and the action
    taken (a square, or PASS on a forced-pass frame)."""
    black: int
    white: int
    mover: int
    action: int

    def own_opp(self) -> Tuple[int, int]:
        return sides(self.black, self.white, self.mover)


def replay_wthor(moves: Sequence[int]) -> Tuple[List[Frame], int, int]:
    """WTHOR move list (squares, passes implicit) → training frames with the
    forced pass frames inserted, exactly like the converter's ``replay``.
    Returns (frames, final black, final white); ValueError on an illegal or
    truncated record."""
    black, white, player = INITIAL_BLACK, INITIAL_WHITE, BLACK
    frames: List[Frame] = []
    for n, sq in enumerate(moves):
        while True:
            own, opp = sides(black, white, player)
            if legal_moves(own, opp):
                break
            if not legal_moves(opp, own):
                raise ValueError(f"move {n}: game already over but "
                                 f"{len(moves) - n} recorded moves remain")
            frames.append(Frame(black, white, player, PASS))
            player = other(player)
        own, opp = sides(black, white, player)
        if not (0 <= sq < 64) or not (legal_moves(own, opp) >> sq & 1):
            raise ValueError(f"move {n}: {sq} is illegal for player {player}")
        frames.append(Frame(black, white, player, sq))
        black, white = colours(*play(own, opp, sq), player)
        player = other(player)
    if not is_game_over(black, white):
        raise ValueError("record truncated: game not over after last recorded move")
    return frames, black, white


def frames_to_tensors(frames: Sequence[Frame], black: int, white: int
                      ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """(obs, tgt) tensors of one game in the converter's dtypes/shapes."""
    sign = float((black.bit_count() > white.bit_count()) - (black.bit_count() < white.bit_count()))
    obs = {"board_seq": torch.from_numpy(planes_batch([f.own_opp() for f in frames]))}
    tgt = {
        "place_idx": torch.tensor([max(f.action, 0) for f in frames], dtype=torch.int64),
        "passed": torch.tensor([f.action == PASS for f in frames], dtype=torch.bool),
        # -sign for white, as the converter writes it (a draw is -0.0 there)
        "value": torch.tensor([sign if f.mover == BLACK else -sign for f in frames],
                              dtype=torch.float32),
    }
    return obs, tgt


def frames_from_sample(sample: Dict) -> List[Frame]:
    """Recover a game's frames from a shard sample (tgt.place_idx / passed),
    checking every action's legality and every frame's planes against
    obs.board_seq, so a converter semantic drift fails loudly."""
    idx = sample["tgt"]["place_idx"].tolist()
    passed = sample["tgt"]["passed"].tolist()
    black, white, player = INITIAL_BLACK, INITIAL_WHITE, BLACK
    frames: List[Frame] = []
    for t, (sq, ps) in enumerate(zip(idx, passed)):
        own, opp = sides(black, white, player)
        legal = legal_moves(own, opp)
        if ps:
            if legal or not legal_moves(opp, own):
                raise ValueError(f"frame {t}: pass frame but the mover "
                                 f"{'has legal moves' if legal else 'faces a finished game'}")
            frames.append(Frame(black, white, player, PASS))
        else:
            if not (legal >> int(sq) & 1):
                raise ValueError(f"frame {t}: move {sq} is illegal")
            frames.append(Frame(black, white, player, int(sq)))
            black, white = colours(*play(own, opp, int(sq)), player)
        player = other(player)
    got = torch.from_numpy(planes_batch([f.own_opp() for f in frames]))
    if not torch.equal(got, sample["obs"]["board_seq"].float()):
        raise ValueError("obs.board_seq disagrees with the replayed positions")
    return frames


# ---------------------------------------------------------------------------
# Exact endgame solver
# ---------------------------------------------------------------------------

def solve(own: int, opp: int, alpha: int = -64, beta: int = 64) -> int:
    """Exact final disc differential (empties to the winner) for the side to
    move, fail-soft negamax alpha-beta.  A side without a move passes (no
    depth is consumed); the game ends when neither side can move."""
    moves = legal_moves(own, opp)
    if not moves:
        if not legal_moves(opp, own):
            return final_diff(own, opp)
        return -solve(opp, own, -beta, -alpha)
    kids = []
    for sq in iter_bits(moves):
        f = flips(own, opp, sq)
        kids.append((opp & ~f, own | f | (1 << sq)))         # opponent to move
    if len(kids) > 1 and 64 - (own | opp).bit_count() > 5:
        kids.sort(key=lambda k: legal_moves(k[0], k[1]).bit_count())  # fastest-first
    best = -65
    for o2, p2 in kids:
        v = -solve(o2, p2, -beta, -alpha)
        if v > best:
            best = v
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    break
    return best


def exact_move_values(own: int, opp: int) -> Dict[int, int]:
    """Exact final disc differential (mover's view) of every legal move."""
    out = {}
    for sq in iter_bits(legal_moves(own, opp)):
        f = flips(own, opp, sq)
        out[sq] = -solve(opp & ~f, own | f | (1 << sq))
    return out


def _exact_job(pos: Tuple[int, int]) -> Dict[int, int]:
    return exact_move_values(*pos)


def solve_positions(positions: Sequence[Tuple[int, int]], workers: int = 0
                    ) -> List[Dict[int, int]]:
    """exact_move_values for many (own, opp) positions, in a process pool
    when ``workers > 1`` (spawned, not forked: the caller has usually
    initialised CUDA already)."""
    if workers > 1 and len(positions) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(_exact_job, positions, chunksize=4))
    return [exact_move_values(*p) for p in positions]


# ---------------------------------------------------------------------------
# Batched streaming evaluator around GameNetwork.step
# ---------------------------------------------------------------------------

class CacheBlock:
    """Stage-1 temporal caches of ``n`` histories that all hold ``t`` frames
    (``cache`` None ⇔ empty history).  A history is referenced as
    ``(block, row)``."""

    __slots__ = ("cache", "t", "n")

    def __init__(self, cache: Optional[List[torch.Tensor]], t: int, n: int):
        self.cache, self.t, self.n = cache, t, n


Ref = Tuple[CacheBlock, int]


class NetEvaluator:
    """Appends one frame to many game histories at once with
    ``GameNetwork.step`` and returns policy logits, values and the new
    histories.

    The Stage-1 cache is the whole history state (every core runs per frame
    AFTER Stage 1 and carries nothing across frames), so a history is
    independent of the depth r used to read it, and cloning a history for
    each candidate move is an index-select of its cache rows (cache layout:
    ``[batch * S, ...]`` with S = game tokens + 2 decision tokens).  Rows
    are evaluated independently (eval-mode BatchNorm, per-token MoE
    routing), so batching changes results only up to float summation order.
    """

    def __init__(self, net, device: torch.device, max_batch: int = 1024,
                 bf16: bool = False):
        self.net = net.eval()
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.bf16 = bf16 and device.type == "cuda"
        self.S = net.obs_encoder.n_tokens + 2
        self.calls = 0
        self.rows = 0
        self._empty = CacheBlock(None, 0, 0)

    def empty_ref(self) -> Ref:
        return (self._empty, 0)

    # -- cache plumbing -----------------------------------------------------

    def _rows(self, c: torch.Tensor, n: int, idx: torch.Tensor) -> torch.Tensor:
        rest = c.shape[1:]
        return c.reshape(n, self.S, *rest).index_select(0, idx).reshape(-1, *rest)

    def _gather(self, refs: Sequence[Ref]) -> Optional[List[torch.Tensor]]:
        if refs[0][0].cache is None:
            return None
        blocks: Dict[int, Tuple[CacheBlock, List[int], List[int]]] = {}
        for pos, (blk, row) in enumerate(refs):
            blocks.setdefault(id(blk), (blk, [], []))
            blocks[id(blk)][1].append(pos)
            blocks[id(blk)][2].append(row)
        parts, order = [], []
        for blk, pos, rows in blocks.values():
            if rows == list(range(blk.n)):
                parts.append(blk.cache)
            else:
                idx = torch.tensor(rows, device=self.device)
                parts.append([self._rows(c, blk.n, idx) for c in blk.cache])
            order.extend(pos)
        if len(parts) == 1:
            return parts[0]
        inv = torch.as_tensor(np.argsort(np.asarray(order), kind="stable"), device=self.device)
        return [self._rows(torch.cat([p[i] for p in parts]), len(order), inv)
                for i in range(len(parts[0]))]

    @torch.no_grad()
    def _step(self, planes: np.ndarray, cache, t: int, r: int, keep: bool):
        B = planes.shape[0]
        logits, values, caches = [], [], []
        for a in range(0, B, self.max_batch):
            b = min(B, a + self.max_batch)
            sub = (None if cache is None
                   else [c[a * self.S:b * self.S] for c in cache])
            obs = {"board_seq": torch.from_numpy(planes[a:b]).to(self.device)}
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                out = self.net.step(obs, legal=None, st_cache=sub, step_idx=t, act_steps=r)
            logits.append(out["place_logits"].float().cpu())
            values.append(out["value"].float().cpu())
            if keep:
                caches.append(out["st_cache"])
            self.calls += 1
            self.rows += b - a
        new = None
        if keep:
            new = caches[0] if len(caches) == 1 else [
                torch.cat([c[i] for c in caches]) for i in range(len(caches[0]))]
        return torch.cat(logits).numpy(), torch.cat(values).numpy(), new

    # -- public -------------------------------------------------------------

    def feed(self, refs: Sequence[Ref], planes: np.ndarray, r: int,
             keep_cache: bool = True):
        """Append ``planes[i]`` to history ``refs[i]`` for every i, read the
        frame at depth ``r``.  Returns (place logits [N, 64], values [N],
        new refs or None)."""
        N = len(refs)
        logits = np.zeros((N, planes.shape[-1] * planes.shape[-2]), dtype=np.float32)
        values = np.zeros(N, dtype=np.float32)
        new_refs: Optional[List[Ref]] = [None] * N if keep_cache else None  # type: ignore
        by_t: Dict[int, List[int]] = {}
        for i, (blk, _) in enumerate(refs):
            by_t.setdefault(blk.t, []).append(i)
        for t, idxs in by_t.items():
            cache = self._gather([refs[i] for i in idxs])
            lg, v, new = self._step(planes[idxs], cache, t, r, keep_cache)
            logits[idxs], values[idxs] = lg, v
            if keep_cache:
                blk = CacheBlock(new, t + 1, len(idxs))
                for j, i in enumerate(idxs):
                    new_refs[i] = (blk, j)
        return logits, values, new_refs


# ---------------------------------------------------------------------------
# Players and lookahead search (evaluator-agnostic: anything with
# empty_ref() and feed(refs, planes, r, keep_cache))
# ---------------------------------------------------------------------------

def masked_argmax(logits: np.ndarray, legal: int) -> int:
    """RAW policy: the legal square with the largest place logit."""
    mask = np.zeros(logits.shape[-1], dtype=bool)
    mask[list(iter_bits(legal))] = True
    return int(np.where(mask, logits, -np.inf).argmax())


@dataclass
class Root:
    """A decision frame to search from: the board, the side to move (which
    has a legal move), the history INCLUDING this frame, and the frame's
    policy logits (the tie-break)."""
    black: int
    white: int
    mover: int
    ref: Ref
    logits: np.ndarray


class _Node:
    __slots__ = ("black", "white", "mover", "ref", "parent", "move", "value", "root")

    def __init__(self, black, white, mover, ref, parent, move, root):
        self.black, self.white, self.mover, self.ref = black, white, mover, ref
        self.parent, self.move, self.root, self.value = parent, move, root, None


def search(ev, roots: Sequence[Root], depth: int, r: int
           ) -> List[Tuple[int, Dict[int, float]]]:
    """Depth-limited minimax over ``depth`` plies of real decisions (forced
    passes are free) with the network value, read at depth ``r``, at the
    leaves and exact W/D/L at terminal positions — see the module docstring
    for the sign conventions.  All roots are searched together, one batched
    ``feed`` per tree level (per history length).

    Returns per root (chosen move, {move: value for the root mover}); the
    move maximises the value, ties broken by the root policy logit."""
    if depth < 1:
        raise ValueError("search depth must be >= 1")
    levels: List[List[_Node]] = [[_Node(rt.black, rt.white, rt.mover, rt.ref, -1, PASS, i)
                                  for i, rt in enumerate(roots)]]
    for i, nd in enumerate(levels[0]):
        if not legal_moves(*sides(nd.black, nd.white, nd.mover)):
            raise ValueError(f"root {i} has no legal move (pass frames need no search)")
    for lvl in range(1, depth + 1):
        leaf = lvl == depth
        kids: List[_Node] = []
        for pi, nd in enumerate(levels[-1]):
            if nd.mover is None:
                continue                                   # terminal
            own, opp = sides(nd.black, nd.white, nd.mover)
            for sq in iter_bits(legal_moves(own, opp)):
                b, w = colours(*play(own, opp, sq), nd.mover)
                kids.append(_Node(b, w, other(nd.mover), None, pi, sq, nd.root))
        # advance every child to its next decision frame (through a forced pass)
        first: List[Tuple[int, int]] = []                  # (kid, frame mover)
        second: List[int] = []                             # kids behind a pass frame
        for i, k in enumerate(kids):
            own, opp = sides(k.black, k.white, k.mover)
            m = roots[k.root].mover
            if legal_moves(own, opp):
                first.append((i, k.mover))
            elif legal_moves(opp, own):
                first.append((i, k.mover))                 # pass frame of k.mover
                second.append(i)
                k.mover = other(k.mover)                   # decision player after it
            else:                                          # game over: exact W/D/L
                k.value = outcome_sign(*sides(k.black, k.white, m))
                k.mover = None
        passed = set(second)
        pending = {}                                       # kid -> value of its last frame fed
        for keep in (True, False):
            sel = [(i, fm) for i, fm in first if (not leaf or i in passed) == keep]
            if not sel:
                continue
            planes = planes_batch([sides(kids[i].black, kids[i].white, fm) for i, fm in sel])
            refs = [levels[-1][kids[i].parent].ref for i, _ in sel]
            _, vals, new = ev.feed(refs, planes, r, keep_cache=keep)
            for j, (i, _) in enumerate(sel):
                pending[i] = float(vals[j])
                if keep:
                    kids[i].ref = new[j]
        if second:
            planes = planes_batch([sides(kids[i].black, kids[i].white, kids[i].mover)
                                   for i in second])
            _, vals, new = ev.feed([kids[i].ref for i in second], planes, r,
                                   keep_cache=not leaf)
            for j, i in enumerate(second):
                pending[i] = float(vals[j])                # value at the decision frame
                kids[i].ref = new[j] if not leaf else None
        if leaf:
            for i, v in pending.items():
                k = kids[i]
                k.value = v if k.mover == roots[k.root].mover else -v
        levels.append(kids)
    # back up: max where the decision player is the root mover, else min
    for lvl in range(depth, 0, -1):
        agg: Dict[int, List[float]] = {}
        for k in levels[lvl]:
            agg.setdefault(k.parent, []).append(k.value)
        for pi, vals in agg.items():
            p = levels[lvl - 1][pi]
            p.value = max(vals) if p.mover == roots[p.root].mover else min(vals)
    out: List[Tuple[int, Dict[int, float]]] = [(PASS, {}) for _ in roots]
    for k in levels[1]:
        out[k.root][1][k.move] = k.value
    for i, (_, vals) in enumerate(out):
        lg = roots[i].logits
        best = max(vals, key=lambda sq: (vals[sq], float(lg[sq])))
        out[i] = (best, vals)
    return out


# ---------------------------------------------------------------------------
# Paired matches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerSpec:
    """depth 0 = RAW (greedy legal-masked policy), d >= 1 = d-ply lookahead
    with network values at the leaves; r = recurrent-depth iterations."""
    depth: int
    r: int

    @property
    def name(self) -> str:
        return f"{'raw' if self.depth == 0 else f'la{self.depth}'}@{self.r}"


def parse_player(text: str, default_r: int) -> PlayerSpec:
    kind, _, r = text.partition("@")
    kind = kind.strip().lower()
    if kind == "raw":
        depth = 0
    elif kind.startswith("la") and kind[2:].isdigit() and int(kind[2:]) >= 1:
        depth = int(kind[2:])
    else:
        raise ValueError(f"player {text!r}: expected raw | la<d> with optional @r")
    return PlayerSpec(depth, int(r) if r else default_r)


class _Game:
    __slots__ = ("black", "white", "to_move", "ref", "ply", "opening", "specs",
                 "logits", "action", "is_pass", "tag", "actions")

    def __init__(self, opening, specs, ref, tag):
        self.black, self.white, self.to_move = INITIAL_BLACK, INITIAL_WHITE, BLACK
        self.ref, self.ply, self.opening, self.specs, self.tag = ref, 0, opening, specs, tag
        self.logits = None
        self.action = PASS
        self.is_pass = False
        self.actions: List[int] = []


def play_games(ev, games: Sequence[Tuple[Sequence[int], PlayerSpec, PlayerSpec]],
               parallel: int = 64, progress: Optional[Callable[[int], None]] = None
               ) -> List[Dict]:
    """Play every (opening actions, black player, white player) game to the
    end, ``parallel`` games in lockstep; each game's frames stream through
    one shared history exactly as a training sample would (opening frames
    included).  Returns per game {black, white, frames, actions}: final disc
    counts, number of frames, and the action of every frame (PASS = -1)."""
    results: List[Optional[Dict]] = [None] * len(games)
    for c0 in range(0, len(games), max(1, parallel)):
        active = [_Game(list(op), {BLACK: b, WHITE: w}, ev.empty_ref(), c0 + j)
                  for j, (op, b, w) in enumerate(games[c0:c0 + parallel])]
        while active:
            live = []
            for g in active:
                if is_game_over(g.black, g.white):
                    results[g.tag] = {"black": g.black.bit_count(),
                                      "white": g.white.bit_count(), "frames": g.ply,
                                      "actions": list(g.actions)}
                    if progress is not None:
                        progress(g.tag)
                else:
                    live.append(g)
            active = live
            if not active:
                break
            by_r: Dict[int, List[_Game]] = {}
            for g in active:
                own, opp = sides(g.black, g.white, g.to_move)
                g.is_pass = not legal_moves(own, opp)
                book = g.ply < len(g.opening)
                # the history is r-independent: frames nobody decides on are read at r=1
                r = 1 if (book or g.is_pass) else g.specs[g.to_move].r
                by_r.setdefault(r, []).append(g)
            for r, gs in by_r.items():
                planes = planes_batch([sides(g.black, g.white, g.to_move) for g in gs])
                logits, _, new = ev.feed([g.ref for g in gs], planes, r)
                for j, g in enumerate(gs):
                    g.ref, g.logits = new[j], logits[j]
            searches: Dict[PlayerSpec, List[_Game]] = {}
            for g in active:
                own, opp = sides(g.black, g.white, g.to_move)
                spec = g.specs[g.to_move]
                if g.ply < len(g.opening):
                    g.action = g.opening[g.ply]
                    if (g.action == PASS) != g.is_pass:
                        raise ValueError(f"opening action {g.action} at ply {g.ply} "
                                         f"contradicts the rules")
                elif g.is_pass:
                    g.action = PASS
                elif spec.depth == 0:
                    g.action = masked_argmax(g.logits, legal_moves(own, opp))
                else:
                    searches.setdefault(spec, []).append(g)
            for spec, gs in searches.items():
                res = search(ev, [Root(g.black, g.white, g.to_move, g.ref, g.logits)
                                  for g in gs], spec.depth, spec.r)
                for g, (mv, _) in zip(gs, res):
                    g.action = mv
            for g in active:
                if g.action != PASS:
                    own, opp = sides(g.black, g.white, g.to_move)
                    g.black, g.white = colours(*play(own, opp, g.action), g.to_move)
                g.actions.append(g.action)
                g.to_move = other(g.to_move)
                g.ply += 1
    return results  # type: ignore[return-value]


def elo_from_score(s: float) -> float:
    return 400.0 * float(np.log10(s / (1.0 - s)))


def match_stats(pair_scores: Sequence[Tuple[float, float]], n_boot: int = 2000,
                seed: int = 0) -> Dict:
    """W/D/L, score and Elo (with 95% CIs) of player A from per-opening
    (score as black, score as white) pairs.  CIs: normal approximation on
    the mean paired score (pairs are the independent units), and a
    percentile bootstrap over opening pairs.  Scores are clamped to
    [ε, 1−ε] with ε = 0.5 / games before the Elo transform so a clean sweep
    stays finite."""
    flat = np.array([s for p in pair_scores for s in p], dtype=np.float64)
    n_games = len(flat)
    if n_games == 0:
        return {"games": 0}
    eps = 0.5 / n_games

    def elo(s):
        return elo_from_score(min(max(float(s), eps), 1.0 - eps))

    pairs = np.array([(a + b) / 2 for a, b in pair_scores], dtype=np.float64)
    score = float(flat.mean())
    n = len(pairs)
    se = float(pairs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    out = {
        "games": n_games, "pairs": n,
        "W": int((flat == 1.0).sum()), "D": int((flat == 0.5).sum()), "L": int((flat == 0.0).sum()),
        "score": score, "elo": elo(score), "score_se": se,
        "elo_ci95_normal": ([elo(score - 1.96 * se), elo(score + 1.96 * se)]
                            if n > 1 else None),
    }
    if n > 1:
        rng = np.random.RandomState(seed)
        boot = pairs[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
        out["elo_ci95_bootstrap"] = [elo(np.percentile(boot, 2.5)), elo(np.percentile(boot, 97.5))]
    else:
        out["elo_ci95_bootstrap"] = None
    return out


def game_score(result: Dict, a_is_black: bool) -> float:
    d = result["black"] - result["white"]
    s = 1.0 if d > 0 else (0.0 if d < 0 else 0.5)
    return s if a_is_black else 1.0 - s


# ---------------------------------------------------------------------------
# Validation positions: value vs exact solver, move agreement
# ---------------------------------------------------------------------------

@dataclass
class ValGame:
    index: int
    frames: List[Frame]
    values: List[float]          # tgt.value (W/D/L for each frame's mover)


def iter_val_games(dataset, order: Iterable[int], skipped: Optional[List[str]] = None
                   ) -> Iterator[ValGame]:
    """Validated games of a GameShardDataset in ``order``; games whose
    recorded actions / planes contradict the rules are skipped (noted in
    ``skipped``)."""
    for i in order:
        sample = dataset[i]
        try:
            frames = frames_from_sample(sample)
        except ValueError as e:
            if skipped is not None:
                skipped.append(f"game {i}: {e}")
            continue
        yield ValGame(i, frames, sample["tgt"]["value"].float().tolist())


def analyse_positions(ev, games: Sequence[Tuple[ValGame, set, set]], depths: Sequence[int],
                      exact: Dict[Tuple[int, int], Dict[int, int]], parallel: int = 64
                      ) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Stream every selected game through the network (lockstep, ``parallel``
    games at a time) and, on the selected decision frames, compare the value
    and the RAW / LOOKAHEAD-1 moves at every depth r against the human move
    (``human`` frames) and the exact solver (``endgame`` frames).

    ``games``: (game, human frame indices, endgame frame indices);
    ``exact[(game_pos, t)]``: exact move values for the endgame frames."""
    acc = {"human": {r: {} for r in depths}, "endgame": {r: {} for r in depths}}

    def add(kind, r, key, v):
        d = acc[kind][r]
        d[key] = d.get(key, 0.0) + float(v)

    for c0 in range(0, len(games), max(1, parallel)):
        chunk = games[c0:c0 + parallel]
        refs: List[Ref] = [ev.empty_ref()] * len(chunk)
        T = max(len(g.frames) for g, _, _ in chunk)
        for t in range(T):
            act = [i for i, (g, _, _) in enumerate(chunk) if t < len(g.frames)]
            an = [i for i in act if t in chunk[i][1] or t in chunk[i][2]]
            an_set = set(an)
            rest = [i for i in act if i not in an_set]
            if rest:
                planes = planes_batch([chunk[i][0].frames[t].own_opp() for i in rest])
                _, _, new = ev.feed([refs[i] for i in rest], planes, 1)
                for j, i in enumerate(rest):
                    refs[i] = new[j]
            if not an:
                continue
            fr = [chunk[i][0].frames[t] for i in an]
            planes = planes_batch([f.own_opp() for f in fr])
            old = [refs[i] for i in an]
            per_r = {}
            for k, r in enumerate(depths):
                logits, values, new = ev.feed(old, planes, r, keep_cache=(k == 0))
                if k == 0:
                    for j, i in enumerate(an):
                        refs[i] = new[j]
                per_r[r] = (logits, values)
            for r in depths:
                logits, values = per_r[r]
                la = search(ev, [Root(f.black, f.white, f.mover, refs[i], logits[j])
                                 for j, (i, f) in enumerate(zip(an, fr))], 1, r)
                for j, (i, f) in enumerate(zip(an, fr)):
                    g, hum, end = chunk[i]
                    raw_mv = masked_argmax(logits[j], legal_moves(*f.own_opp()))
                    la_mv = la[j][0]
                    v = float(values[j])
                    if t in hum:
                        add("human", r, "n", 1)
                        add("human", r, "raw_top1", raw_mv == f.action)
                        add("human", r, "la1_top1", la_mv == f.action)
                        add("human", r, "value_sq_err_vs_result", (v - g.values[t]) ** 2)
                    if t in end:
                        ex = exact[(c0 + i, t)]
                        best = max(ex.values())
                        z = float((best > 0) - (best < 0))       # exact W/D/L target
                        add("endgame", r, "n", 1)
                        add("endgame", r, "value_sq_err", (v - z) ** 2)
                        if z != 0:
                            add("endgame", r, "n_decisive", 1)
                            add("endgame", r, "value_sign_ok", np.sign(v) == z)
                        for name, mv in (("raw", raw_mv), ("la1", la_mv)):
                            add("endgame", r, f"{name}_best", ex[mv] == best)
                            add("endgame", r, f"{name}_regret", best - ex[mv])
                            zc = float((ex[mv] > 0) - (ex[mv] < 0))
                            add("endgame", r, f"{name}_wdl_blunder", zc < z)
    out: Dict[str, Dict[int, Dict[str, float]]] = {"human": {}, "endgame": {}}
    for r in depths:
        h = acc["human"][r]
        if h.get("n"):
            n = h["n"]
            out["human"][r] = {"n": int(n), "raw_top1": h["raw_top1"] / n,
                               "la1_top1": h["la1_top1"] / n,
                               "value_mse_vs_result": h["value_sq_err_vs_result"] / n}
        e = acc["endgame"][r]
        if e.get("n"):
            n = e["n"]
            nd = e.get("n_decisive", 0.0)
            out["endgame"][r] = {
                "n": int(n), "n_decisive": int(nd),
                "value_sign_acc": (e["value_sign_ok"] / nd) if nd else float("nan"),
                "value_mse_wdl": e["value_sq_err"] / n,
                **{f"{p}_{k}": e[f"{p}_{k}"] / n
                   for p in ("raw", "la1") for k in ("best", "regret", "wdl_blunder")},
            }
    return out
