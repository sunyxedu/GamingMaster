"""Two-player self-play environment for the placement board games.

Wraps the exact rules engines the IL converters use (othello, gomoku) and
produces observations through the same ``board_planes`` helpers, so RL
rollouts encode positions identically to the IL shards by construction.

Alternating-move, mover-POV: :meth:`obs` is the position BEFORE the side
to move acts, from that side's perspective (channel swap) — exactly one
IL sample frame.  Othello passes are rules-forced (the game has no
voluntary pass): when :meth:`legal_mask` is all-False the only step is
``step(None)``, and the trainer excludes that frame from the policy loss.

Stream convention (shared by IL and RL, see training/train_rl_game.py):
a game is ONE interleaved sequence — every ply (forced passes included) is
a frame from :attr:`to_play`'s perspective, both movers' frames in play
order, which is how every board IL converter lays out its samples.  The
RL trainer streams all of them through one model cache (step index = ply)
and puts the loss on the learner seats' frames only.

Winner encoding: 0 = draw / in progress, else the rules engine's player
id (1 = black/first mover, 2 = white).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from data_processing.games.gomoku.rules import (
    board_planes as gomoku_planes,
    apply_move as gomoku_apply,
    board_full,
    initial_board as gomoku_initial,
    wins,
)
from data_processing.games.othello.rules import (
    board_planes as othello_planes,
    apply_move as othello_apply,
    disc_counts,
    has_legal_move,
    initial_board as othello_initial,
    is_game_over,
    legal_moves,
    opponent,
)

OTHELLO_SIZE = 8


class BoardSelfPlayEnv:
    """One game of othello or gomoku, played to completion by callers
    alternating :meth:`step`."""

    def __init__(self, game: str, board_size: Optional[int] = None):
        if game == "othello":
            if board_size not in (None, OTHELLO_SIZE):
                raise ValueError("othello is 8×8 only")
            self.size = OTHELLO_SIZE
        elif game == "gomoku":
            self.size = board_size or 15
        else:
            raise ValueError(f"unsupported game {game!r} (othello / gomoku)")
        self.game = game
        self.reset()

    def reset(self) -> None:
        self.board = (othello_initial() if self.game == "othello"
                      else gomoku_initial(self.size))
        self.mover: int = 1
        self.done: bool = False
        self.winner: int = 0
        self.ply: int = 0

    # ------------------------------------------------------------------

    @property
    def to_play(self) -> int:
        """Seat to move (1 = black / first mover, 2 = white) — the owner of
        the current frame; read-only view of :attr:`mover`."""
        return self.mover

    def obs(self) -> Dict[str, torch.Tensor]:
        """Mover-POV observation, same keys/planes as the IL shards."""
        planes = (othello_planes(self.board, self.mover)
                  if self.game == "othello"
                  else gomoku_planes(self.board, self.size, self.mover))
        return {"board_seq": planes}

    def legal_mask(self) -> torch.Tensor:
        """[size²] bool placement legality for the mover; all-False means
        the mover must pass (othello only)."""
        n = self.size * self.size
        mask = torch.zeros(n, dtype=torch.bool)
        if self.done:
            return mask
        if self.game == "othello":
            for r, c in legal_moves(self.board, self.mover):
                mask[r * self.size + c] = True
        else:
            board = torch.tensor(list(self.board), dtype=torch.uint8)
            mask = board == 0
        return mask

    def legal(self) -> Optional[Dict[str, torch.Tensor]]:
        """The decoder legality dict ``{"place": [size²] bool}`` for the
        mover, or ``None`` when there is no placement to choose: a forced
        pass (othello; the only step is ``step(None)``) or a finished game."""
        mask = self.legal_mask()
        return {"place": mask} if bool(mask.any()) else None

    # ------------------------------------------------------------------

    def step(self, place_idx: Optional[int]) -> bool:
        """Apply the mover's placement (``None`` = forced pass) and swap
        the mover.  Returns ``done``."""
        if self.done:
            raise RuntimeError("game is over; call reset()")
        r, c = divmod(place_idx, self.size) if place_idx is not None else (-1, -1)

        if self.game == "othello":
            if place_idx is None:
                if has_legal_move(self.board, self.mover):
                    raise ValueError("pass with legal moves available")
            else:
                othello_apply(self.board, self.mover, r, c)
            if is_game_over(self.board):
                self.done = True
                black, white = disc_counts(self.board)
                self.winner = 1 if black > white else 2 if white > black else 0
        else:
            if place_idx is None:
                raise ValueError("gomoku has no pass")
            gomoku_apply(self.board, self.size, self.mover, r, c)
            if wins(self.board, self.size, self.mover, r, c):
                self.done = True
                self.winner = self.mover
            elif board_full(self.board):
                self.done = True

        self.ply += 1
        self.mover = opponent(self.mover)
        return self.done
