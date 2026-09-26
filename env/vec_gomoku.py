"""Gomoku as a :class:`~env.vec_game.VecGameEnv`: N games on numpy boards.

Frame semantics are exactly ``gmdata gomoku convert``'s (data_processing/rs/
gmdata/src/gomoku.rs; the IL data), so a rollout stream IS an IL sample:

  * one row per game = ONE interleaved stream; frame t is the position BEFORE
    the t-th move, seen from the side to move (planes: mover stones /
    opponent stones / empty — a pure channel swap, no geometric flip);
  * black (seat 0) moves first, the movers strictly alternate: there is no
    pass and no forced frame (a board with an empty point always has a legal
    move, a full board ends the game);
  * the move that completes five or more in a row (``rule="freestyle"``, the
    Gomocup freestyle leagues; ``"standard"``: exactly five, an overline does
    not win) ends the game and pays +1 to the winner and −1 to the loser —
    the IL value target (±1 for the mover) of every frame of that seat; the
    terminal position is never a frame;
  * a full board without a five is a draw (0, 0).  The converter drops
    undecided games, so draws never occur in the IL data;
  * ``max_moves``: optional truncation after that many moves (a soft terminal
    the env scores as a draw, 0 / 0, ``info["truncated"]``).

Legality: ``legal["place"]`` [R, H·W] = the empty points (row · W + col);
``legal["pass"]`` [R] is always False.  The gomoku decoder has no pass head
(model/games/registry.py builds it with ``allow_pass=False``); a network built
with one may send a pass flag, which is ignored — the env always applies the
placement (``info["applied"]["passed"]`` is always False).

Board size: ``board_size`` (int N, or (H, W)); the joint trainer takes it from
the network config (``GomokuSpec.env_kwargs``: ``board_h`` / ``board_w``).
The default 20 is the Gomocup freestyle board (scripts/fetch_gomocup.sh).

Win detection is incremental: only the lines through the last move, via a
precomputed ray table (per point, per direction and sign, the next K points
and whether they are still on the board — no row wrap-around), K = 4
(freestyle) or 5 (standard: a sixth stone on either side makes an overline).
Pinned to the converter frame by frame on real Gomocup games by
tests/test_vec_game_golden.py and to a brute-force line scan on random games
by tests/test_vec_game.py.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

EMPTY = 0             # board values: 0 empty, 1 black (seat 0), 2 white (seat 1)
RULES = ("freestyle", "standard")
# the 4 line directions (row, col step); the rays take them with both signs
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def ray_table(h: int, w: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """For every point, direction d (0..3, then the same 4 reversed) and step
    1..k: ``(index, on_board)``, both [h·w, 8, k] — the flat index of the
    point that far along the ray (0 once the ray has left the board) and
    whether it is still on the board (a straight line never re-enters)."""
    r = np.arange(h * w) // w
    c = np.arange(h * w) % w
    steps = np.arange(1, k + 1)
    index = np.zeros((h * w, 8, k), dtype=np.int64)
    on_board = np.zeros((h * w, 8, k), dtype=bool)
    for s, sign in enumerate((1, -1)):
        for d, (dr, dc) in enumerate(DIRECTIONS):
            rr = r[:, None] + sign * dr * steps
            cc = c[:, None] + sign * dc * steps
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            index[:, 4 * s + d] = np.where(inside, rr * w + cc, 0)
            on_board[:, 4 * s + d] = inside
    return index, on_board


def line_runs(board: np.ndarray, rays: Tuple[np.ndarray, np.ndarray], rows: np.ndarray,
              sq: np.ndarray, color: np.ndarray) -> np.ndarray:
    """[M, 4] length of the ``color`` line through the stone at ``sq`` of
    board row ``rows`` ([N, h·w] board), per direction, counting at most k
    stones each way (``rays`` = :func:`ray_table`)."""
    index, on_board = rays
    vals = board[rows[:, None, None], index[sq]]                   # [M, 8, k]
    same = (vals == color[:, None, None]) & on_board[sq]
    cnt = np.logical_and.accumulate(same, axis=-1).sum(-1)          # [M, 8]
    return 1 + cnt[:, :4] + cnt[:, 4:]


def _board_dims(board_size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(board_size, (int, np.integer)):
        h = w = int(board_size)
    else:
        h, w = (int(x) for x in board_size)
    if h < 1 or w < 1:
        raise ValueError(f"gomoku board must be at least 1x1, got {h}x{w}")
    return h, w


class GomokuVecEnv(VecEnvBase):
    """N gomoku games, one interleaved stream each (see the module docstring).

    Actions: ``place_idx`` / ``action_place`` [N] long (row · W + col); a
    ``passed`` / ``action_pass`` entry is accepted and ignored (there is no
    pass).  A placement on an occupied or off-board point of an active row
    raises.
    """

    game = "gomoku"
    n_seats = 2
    rows_per_env = 1
    interleaved = True
    ACTION_KEYS = {"action_place": "place_idx", "action_pass": "passed"}

    def __init__(self, n_envs: int = 1, *, board_size: Union[int, Sequence[int]] = 20,
                 rule: str = "freestyle", max_moves: Optional[int] = None,
                 autoreset: bool = False, seed: Optional[int] = 0):
        super().__init__(n_envs, autoreset, seed)
        if rule not in RULES:
            raise ValueError(f"gomoku rule must be one of {RULES}, got {rule!r}")
        self.board_h, self.board_w = _board_dims(board_size)
        self.n_points = self.board_h * self.board_w
        if max_moves is not None and int(max_moves) < 1:
            raise ValueError(f"max_moves must be >= 1 (or None), got {max_moves}")
        self.rule = rule
        self.exact_five = rule == "standard"
        self.max_moves = None if max_moves is None else int(max_moves)
        self._rays = ray_table(self.board_h, self.board_w, 5 if self.exact_five else 4)
        self._alloc(self._n)
        self.reset()

    # ------------------------------------------------------------------

    def _alloc(self, n: int) -> None:
        self._n = n
        self._board = np.zeros((n, self.n_points), dtype=np.int8)
        self.ply = np.zeros(n, dtype=np.int64)          # moves so far (= step_idx)
        self.done = np.ones(n, dtype=bool)
        self.new_episode = np.zeros(n, dtype=bool)

    @property
    def board(self) -> np.ndarray:
        """[N, H·W] int8: 0 empty, 1 black (seat 0), 2 white (seat 1)."""
        return self._board

    @property
    def mover(self) -> np.ndarray:
        """[N] seat to move (black moves on even plies)."""
        return self.ply & 1

    def _start(self, ids: np.ndarray) -> None:
        self._board[ids] = EMPTY
        self.ply[ids] = 0
        self.done[ids] = False
        self.new_episode[ids] = True

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self.done.copy())

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        """Gomoku starts from the empty board, so ``seed`` only reseeds the env
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
        return {"board_seq": ((3, self.board_h, self.board_w), torch.float32)}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        n = self._n
        board = torch.from_numpy(self._board)
        mover = self.mover
        own = torch.from_numpy((mover + 1).astype(np.int8))[:, None]
        active = ~self.done
        empty = board == EMPTY
        planes = torch.stack([board == own, board == 3 - own, empty], 1).to(torch.float32)
        act = torch.from_numpy(active)
        zeros = torch.zeros(n, dtype=torch.bool)
        return VecObs(
            obs={"board_seq": planes.view(n, 3, self.board_h, self.board_w)},
            legal={"place": empty & act[:, None], "pass": zeros},
            to_play=torch.from_numpy(mover),
            env_id=torch.arange(n, dtype=torch.long),
            stream_id=torch.arange(n, dtype=torch.long),
            step_idx=torch.from_numpy(self.ply.copy()),
            new_episode=torch.from_numpy(self.new_episode & active),
            active=act,
            forced=zeros.clone(),
            stream_seats=torch.ones(n, self.n_seats, dtype=torch.bool),
        )

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        n, P = self._n, self.n_points
        acts = self.replay_actions(actions)
        self._require(acts, ("place_idx",))
        place_t = acts["place_idx"]
        if place_t.dim() == 2 and place_t.shape[1] == 1:
            place_t = place_t[:, 0]
        if place_t.dim() != 1:
            # one point per row: an [N, k] tensor (a batched decode's [B, T]
            # output, per-unit actions, ...) is a caller bug — never silently
            # apply one of its columns
            raise ValueError(f"gomoku: place_idx must be [N] (or [N, 1]), got shape "
                             f"{tuple(place_t.shape)}")
        place = place_t.cpu().numpy().astype(np.int64)
        active = ~self.done
        idx = np.flatnonzero(active)
        won = np.zeros(n, dtype=bool)
        if idx.size:
            sq = place[idx]
            ok = (sq >= 0) & (sq < P)
            ok[ok] = self._board[idx[ok], sq[ok]] == EMPTY
            if not ok.all():
                i = int(idx[~ok][0])
                raise ValueError(
                    f"gomoku env {i}: place_idx {int(place[i])} is illegal for seat "
                    f"{int(self.mover[i])} at frame {int(self.ply[i])} (use legal['place'])")
            color = (self.mover[idx] + 1).astype(np.int8)
            self._board[idx, sq] = color
            run = line_runs(self._board, self._rays, idx, sq, color)
            five = (run == 5) if self.exact_five else (run >= 5)
            won[idx] = five.any(-1)

        applied = {
            "place_idx": torch.from_numpy(np.where(active, place, 0).astype(np.int64)),
            "passed": torch.zeros(n, dtype=torch.bool),
        }
        winner = np.where(won, self.mover, -1).astype(np.int64)    # seat of the last move
        self.new_episode[:] = False
        self.ply[active] += 1
        full = active & ~won & (self.ply >= P)
        terminated = won | full
        truncated = np.zeros(n, dtype=bool)
        if self.max_moves is not None:
            truncated = active & ~terminated & (self.ply >= self.max_moves)
        ended = terminated | truncated

        rewards = np.zeros((n, self.n_seats), dtype=np.float32)
        w_idx = np.flatnonzero(won)
        rewards[w_idx, winner[w_idx]] = 1.0
        rewards[w_idx, 1 - winner[w_idx]] = -1.0
        episode_len = np.where(ended, self.ply, 0)
        self.done |= ended
        info: Dict[str, Any] = {
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "winner": torch.from_numpy(winner),
            "episode_len": torch.from_numpy(episode_len),
            "acted": torch.from_numpy(active.copy()),
            "applied": applied,
            "illegal": torch.zeros(n, dtype=torch.bool),
        }
        e_idx = np.flatnonzero(ended)
        if self.autoreset and e_idx.size:
            self._start(e_idx)
        return self.observe(), torch.from_numpy(rewards), torch.from_numpy(ended.copy()), info

    # ------------------------------------------------------------------

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        # inverse CDF over the legal points (~10x faster than torch.multinomial
        # on 400 points): the k-th legal point, k uniform in [0, n_legal); rows
        # without a legal point (inactive) get the last point, which step ignores
        legal = obs.legal["place"]
        cum = legal.to(torch.int32).cumsum(-1)                    # [R, P]
        n_legal = cum[:, -1:]
        u = torch.rand(n_legal.shape, generator=self._generator(generator))
        k = (u * n_legal).long().clamp(max=(n_legal - 1).clamp(min=0))
        place = (cum <= k).sum(-1).clamp(max=legal.shape[-1] - 1)
        return {"place_idx": place, "passed": torch.zeros_like(obs.forced)}
