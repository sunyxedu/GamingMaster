"""Othello as a :class:`~env.vec_game.VecGameEnv`: N games on numpy uint64 bitboards.

Frame semantics are exactly ``gmdata othello convert``'s (data_processing/rs/
gmdata/src/othello.rs; the IL data), so a rollout stream IS an IL sample:

  * one row per game = ONE interleaved stream; frame t is the position BEFORE
    the t-th half-move, seen from the side to move (planes: mover discs /
    opponent discs / empty — a pure channel swap, no geometric flip);
  * a side with no legal move while the opponent has one gets a forced PASS
    frame (``forced``; applied action place_idx 0, passed True — the IL
    target); the stream continues with the opponent's frame of the same board;
  * the terminal position (neither side can move) is never a frame;
  * the game-ending step pays each seat sign(own − opponent discs), the IL
    value target of every frame of that seat (white's is written ``-sign``,
    so a draw reads −0.0 for white exactly as in the shards).

Seats: 0 = black (moves first; rules player 1), 1 = white (rules player 2).
Rules are data_processing/games/othello/rules.py's (and the Rust converter's),
evaluated with the Kogge-Stone fills of training/othello_probe.py on whole
batches at once; pinned to rules.py by tests/test_vec_game.py.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

N_SQUARES = 64
_U = np.uint64
_ZERO = _U(0)
_ONE = _U(1)
_FULL = _U(0xFFFFFFFFFFFFFFFF)
_NOT_A = _U(0xFEFEFEFEFEFEFEFE)      # drops bits that wrapped into column 0
_NOT_H = _U(0x7F7F7F7F7F7F7F7F)      # drops bits that wrapped into column 7
# (shift, wrap mask) per direction, square = row * 8 + col
_LEFT = ((_U(1), _NOT_A), (_U(8), _FULL), (_U(9), _NOT_A), (_U(7), _NOT_H))
_RIGHT = ((_U(1), _NOT_H), (_U(8), _FULL), (_U(9), _NOT_H), (_U(7), _NOT_A))
_SHIFTS = np.arange(N_SQUARES, dtype=np.uint64)

INITIAL_BLACK = _U((1 << (3 * 8 + 4)) | (1 << (4 * 8 + 3)))
INITIAL_WHITE = _U((1 << (3 * 8 + 3)) | (1 << (4 * 8 + 4)))


def legal_moves(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """[N] uint64 bitboards of ``own``'s legal moves (a run of at most 6
    opponent discs fits between two squares of a line)."""
    empty = ~(own | opp)
    moves = np.zeros_like(own)
    for s, m in _LEFT:
        mo = opp & m
        x = (own << s) & mo
        for _ in range(5):
            x |= (x << s) & mo
        moves |= (x << s) & m & empty
    for s, m in _RIGHT:
        mo = opp & m
        x = (own >> s) & mo
        for _ in range(5):
            x |= (x >> s) & mo
        moves |= (x >> s) & m & empty
    return moves


def flips(own: np.ndarray, opp: np.ndarray, bit: np.ndarray) -> np.ndarray:
    """[N] uint64: discs flipped when ``own`` plays the single-bit squares
    ``bit`` (0 ⇔ illegal; the squares are assumed empty)."""
    f = np.zeros_like(own)
    for s, m in _LEFT:
        mo = opp & m
        x = (bit << s) & mo
        for _ in range(5):
            x |= (x << s) & mo
        f |= np.where(((x << s) & m & own) != _ZERO, x, _ZERO)
    for s, m in _RIGHT:
        mo = opp & m
        x = (bit >> s) & mo
        for _ in range(5):
            x |= (x >> s) & mo
        f |= np.where(((x >> s) & m & own) != _ZERO, x, _ZERO)
    return f


def unpack(b: np.ndarray) -> np.ndarray:
    """[N] uint64 → [N, 64] uint8 bits (square order)."""
    return ((b[:, None] >> _SHIFTS) & _ONE).astype(np.uint8)


def popcount(b: np.ndarray) -> np.ndarray:
    return unpack(b).sum(axis=1, dtype=np.int64)


def planes(own: np.ndarray, opp: np.ndarray) -> torch.Tensor:
    """[N, 3, 8, 8] float32 own / opponent / empty — ``rules.board_planes``."""
    a = unpack(own).astype(np.float32)
    b = unpack(opp).astype(np.float32)
    out = np.stack([a, b, 1.0 - a - b], axis=1).reshape(-1, 3, 8, 8)
    return torch.from_numpy(np.ascontiguousarray(out))


class OthelloVecEnv(VecEnvBase):
    """N othello games, one interleaved stream each (see the module docstring).

    Actions: ``place_idx`` / ``action_place`` [N] long (row * 8 + col); a
    ``passed`` / ``action_pass`` entry is accepted and ignored — passes are
    rules-forced (``forced`` frames take place_idx 0 / passed True whatever is
    sent; an illegal placement on a decision frame raises).
    """

    game = "othello"
    n_seats = 2
    rows_per_env = 1
    interleaved = True
    ACTION_KEYS = {"action_place": "place_idx", "action_pass": "passed"}

    def __init__(self, n_envs: int = 1, *, autoreset: bool = False,
                 seed: Optional[int] = 0):
        super().__init__(n_envs, autoreset, seed)
        self._alloc(self._n)
        self.reset()

    # ------------------------------------------------------------------

    def _alloc(self, n: int) -> None:
        self._n = n
        self.black = np.zeros(n, dtype=np.uint64)
        self.white = np.zeros(n, dtype=np.uint64)
        self.mover = np.zeros(n, dtype=np.int64)       # seat to move
        self.ply = np.zeros(n, dtype=np.int64)         # frames so far (= step_idx)
        self.n_pass = np.zeros(n, dtype=np.int64)
        self.done = np.ones(n, dtype=bool)
        self.new_episode = np.zeros(n, dtype=bool)
        self._legal = np.zeros(n, dtype=np.uint64)     # mover's legal moves

    def _sides(self, idx=slice(None)) -> Tuple[np.ndarray, np.ndarray]:
        blk = self.mover[idx] == 0
        own = np.where(blk, self.black[idx], self.white[idx])
        opp = np.where(blk, self.white[idx], self.black[idx])
        return own, opp

    def _start(self, ids: np.ndarray) -> None:
        self.black[ids] = INITIAL_BLACK
        self.white[ids] = INITIAL_WHITE
        self.mover[ids] = 0
        self.ply[ids] = 0
        self.n_pass[ids] = 0
        self.done[ids] = False
        self.new_episode[ids] = True
        own, opp = self._sides(ids)
        self._legal[ids] = legal_moves(own, opp)

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self.done.copy())

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        """Othello starts from one position, so ``seed`` only reseeds the env
        RNGs (the torch one drives ``random_actions`` without a generator)."""
        if seed is not None:
            self._seed_rngs(seed)
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("resize (n) and env_ids are exclusive")
            self._alloc(int(n))
        self.new_episode[:] = False
        self._start(self._env_ids(env_ids))
        return self.observe()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {"board_seq": ((3, 8, 8), torch.float32)}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        n = self._n
        own, opp = self._sides()
        active = ~self.done
        legal_bits = np.where(active, self._legal, _ZERO)
        forced = active & (self._legal == _ZERO)
        return VecObs(
            obs={"board_seq": planes(own, opp)},
            legal={"place": torch.from_numpy(unpack(legal_bits).astype(bool)),
                   "pass": torch.from_numpy(forced.copy())},
            to_play=torch.from_numpy(self.mover.copy()),
            env_id=torch.arange(n, dtype=torch.long),
            stream_id=torch.arange(n, dtype=torch.long),
            step_idx=torch.from_numpy(self.ply.copy()),
            new_episode=torch.from_numpy(self.new_episode & active),
            active=torch.from_numpy(active.copy()),
            forced=torch.from_numpy(forced),
            stream_seats=torch.ones(n, self.n_seats, dtype=torch.bool),
        )

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        n = self._n
        acts = self.replay_actions(actions)
        self._require(acts, ("place_idx",))
        place = acts["place_idx"].cpu().numpy().astype(np.int64)
        active = ~self.done
        forced = active & (self._legal == _ZERO)
        move = active & ~forced

        idx = np.flatnonzero(move)
        if idx.size:
            sq = place[idx]
            bad = (sq < 0) | (sq >= N_SQUARES)
            ok = ~bad
            ok[ok] = ((self._legal[idx[ok]] >> sq[ok].astype(np.uint64)) & _ONE) == _ONE
            if not ok.all():
                i = int(idx[~ok][0])
                raise ValueError(
                    f"othello env {i}: place_idx {int(place[i])} is illegal for seat "
                    f"{int(self.mover[i])} at frame {int(self.ply[i])} (use legal['place'])")
            bit = _ONE << sq.astype(np.uint64)
            own, opp = self._sides(idx)
            f = flips(own, opp, bit)
            own = own | f | bit
            opp = opp & ~f
            blk = self.mover[idx] == 0
            self.black[idx] = np.where(blk, own, opp)
            self.white[idx] = np.where(blk, opp, own)

        applied = {
            "place_idx": torch.from_numpy(np.where(move, place, 0).astype(np.int64)),
            "passed": torch.from_numpy(forced.copy()),
        }
        episode_len = np.zeros(n, dtype=np.int64)
        self.new_episode[:] = False
        self.ply[active] += 1
        self.n_pass[forced] += 1
        self.mover[active] ^= 1

        rewards = np.zeros((n, self.n_seats), dtype=np.float32)
        winner = np.full(n, -1, dtype=np.int64)
        ended = np.zeros(n, dtype=bool)
        score = np.zeros((n, self.n_seats), dtype=np.int64)
        act = np.flatnonzero(active)
        if act.size:
            own, opp = self._sides(act)
            lm = legal_moves(own, opp)
            self._legal[act] = lm
            stuck = lm == _ZERO
            if stuck.any():
                s_idx = act[stuck]
                opp_can = legal_moves(opp[stuck], own[stuck]) != _ZERO
                ended[s_idx[~opp_can]] = True
        e_idx = np.flatnonzero(ended)
        if e_idx.size:
            b = popcount(self.black[e_idx])
            w = popcount(self.white[e_idx])
            sign = np.sign(b - w).astype(np.float32)
            rewards[e_idx, 0] = sign
            rewards[e_idx, 1] = -sign
            winner[e_idx] = np.where(b > w, 0, np.where(w > b, 1, -1))
            score[e_idx, 0], score[e_idx, 1] = b, w
            episode_len[e_idx] = self.ply[e_idx]
            self.done[e_idx] = True
        info: Dict[str, Any] = {
            "terminated": torch.from_numpy(ended.copy()),
            "truncated": torch.zeros(n, dtype=torch.bool),
            "winner": torch.from_numpy(winner),
            "episode_len": torch.from_numpy(episode_len),
            "acted": torch.from_numpy(active.copy()),
            "applied": applied,
            "illegal": torch.zeros(n, dtype=torch.bool),
            "score": torch.from_numpy(score),
            "n_pass": torch.from_numpy(np.where(ended, self.n_pass, 0)),
        }
        if self.autoreset and e_idx.size:
            self._start(e_idx)
        return self.observe(), torch.from_numpy(rewards), torch.from_numpy(ended), info

    # ------------------------------------------------------------------

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        legal = obs.legal["place"]
        w = legal.float()
        w[~legal.any(-1)] = 1.0                          # forced / inactive rows
        place = torch.multinomial(w, 1, generator=self._generator(generator)).squeeze(-1)
        return {"place_idx": place, "passed": obs.forced.clone()}
