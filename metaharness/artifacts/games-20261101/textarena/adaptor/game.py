"""Offline TextArena adaptor for a small, verified Tier 1 game mix.

The wrapper keeps TextArena's public text observation and string action protocol while
handling baseline turns internally.  It intentionally never places TextArena's private
state in the player observation.
"""
from __future__ import annotations

import hashlib
import itertools
import threading
from contextlib import contextmanager
from importlib.metadata import version
import random
import re
from typing import Any

import textarena as ta
from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True

# Alternating entries keep every split balanced between the two verified games.
SEEDS = {
    "train": [10, 11, 12, 13, 14, 15, 16, 17],
    "heldout": [110, 111, 112, 113, 114, 115, 116, 117],
    "dev": [210, 211],
    "validation": [310, 311, 312, 313, 314, 315, 316, 317],
}
EVAL_KWARGS: dict[str, Any] = {}

_RNG_LOCK = threading.RLock()

_TTT_ACTIONS = [f"[{i}]" for i in range(9)]
_MM_ACTIONS = [f"[{a} {b} {c} {d}]" for a, b, c, d in itertools.permutations(range(1, 7), 4)]
_TTT_WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
             (2, 5, 8), (0, 4, 8), (2, 4, 6))


def _stable_seed(seed: int, label: str) -> int:
    raw = hashlib.blake2b(f"{seed}:{label}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "big")


class TextArenaGame:
    name = "TextArena (TicTacToe / Mastermind)"

    def __init__(self) -> None:
        if version("textarena") != "0.7.4":
            raise RuntimeError("This adaptor requires exactly textarena==0.7.4")
        self.action_spec = self._ttt_spec()
        self._rng_state = random.Random(0).getstate()
        self._texts = {0: "", 1: ""}
        self._guesses: set[str] = set()
        self._attempts = 0
        self._invalid_count = 0
        self._termination = "in_progress"
        self._raw_rewards: dict[str, float] = {}
        self._env: Any = None
        self._game_id = ""
        self._seed = 0
        self._learner = 0
        self._current = 0
        self._text = ""
        self._board = [None] * 9
        self._history: list[str] = []
        self._last_error: str | None = None
        self._done = False
        self._score = 0.0
        self._raw_reward: float | None = None
        self._terminal_reason = ""
        self._move_count = 0
        self._valid = True
        self._closed = False
        self._baseline_rng = random.Random(0)

    @property
    def done(self) -> bool:
        return bool(self._done)

    @property
    def score(self) -> float:
        return float(self._score)

    def reset(self, seed: int) -> dict[str, Any]:
        self.close()
        self._seed = int(seed)
        self._game_id = "TicTacToe-v0" if seed % 2 == 0 else "Mastermind-v0"
        self._learner = ((seed // 2) & 1) if self._game_id == "TicTacToe-v0" else 0
        self._baseline_rng = random.Random(_stable_seed(seed, "baseline"))
        self._rng_state = random.Random(_stable_seed(seed, "constructor")).getstate()
        self._closed = False
        with self._native_rng():
            kwargs = {"max_turns": 2, "code_length": 4, "num_numbers": 6, "duplicate_numbers": False} if self._game_id == "Mastermind-v0" else {}
            self._env = ta.make(self._game_id + "-raw", **kwargs)
            self._env.reset(num_players=2 if self._game_id == "TicTacToe-v0" else 1,
                            seed=_stable_seed(seed, "environment"))
        self._texts = {0: "", 1: ""}
        self._guesses = set()
        self._attempts = 0
        self._invalid_count = 0
        self._termination = "in_progress"
        self._raw_rewards = {}
        self._current = 0
        self._text = ""
        self._board = [None] * 9
        self._history = []
        self._last_error = None
        self._done = False
        self._score = 0.0
        self._raw_reward = None
        self._terminal_reason = ""
        self._move_count = 0
        self._valid = True
        self._closed = False
        self.action_spec = self._ttt_spec() if self._game_id == "TicTacToe-v0" else self._mm_spec()
        self._read_observation()
        self._advance_baseline()
        return self.observe()

    def _ttt_spec(self) -> ActionSpec:
        return ActionSpec(
            names=["noop"] + list(_TTT_ACTIONS),
            noop="noop",
            descriptions={"noop": "Keep the position; consumes one wrapper attempt.", **{a: f"Place your mark in square {a[1:-1]} (row-major 0-8)." for a in _TTT_ACTIONS}},
            notes="Submit [n], where n is an empty square. The board is shown row-major from 0 through 8.",
        )

    def _mm_spec(self) -> ActionSpec:
        return ActionSpec(
            names=["noop"] + list(_MM_ACTIONS),
            noop="noop",
            descriptions={"noop": "Keep the position; consumes one wrapper attempt.", **{a: "Guess the four distinct digits inside the brackets." for a in _MM_ACTIONS}},
            notes="Submit [a b c d], four distinct digits from 1 to 6. Feedback reports exact and misplaced pegs.",
        )

    @contextmanager
    def _native_rng(self):
        # The audited environments use Python random, including State.__init__.
        # Save/restore also prevents interleaved adaptor instances from coupling.
        with _RNG_LOCK:
            outer = random.getstate()
            random.setstate(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = random.getstate()
                random.setstate(outer)

    def _read_observation(self) -> None:
        player, messages = self._env.get_observation()
        self._current = int(player)
        self._append_messages(self._current, messages)
        if self._done and self._current != self._learner:
            # Terminal turns do not rotate. Read only the learner's addressed queue,
            # never the opponent's prompt or private state.
            messages = self._env.state.observations[self._learner]
            self._append_messages(self._learner, messages)
            self._env.state.observations[self._learner] = []
        self._text = self._texts[self._learner]

    def _append_messages(self, player: int, messages: list) -> None:
        for sender, text, _kind in messages:
            role = "GAME" if sender == -1 else f"Player {sender}"
            self._texts[player] += f"[{role}] {text}\n"

    def _advance_baseline(self) -> None:
        if self._game_id != "TicTacToe-v0" or self._done:
            return
        # A bounded loop prevents a broken native environment from hanging reset/step.
        for _ in range(9):
            if self._done or self._current == self._learner:
                return
            move = self._choose_ttt_move()
            self._apply_native(f"[{move}]", actor=self._current, baseline=True)
        if not self._done and self._current != self._learner:
            self._valid = False
            self._last_error = "baseline turn loop exceeded bound"
            self._done = True
            self._termination = "infrastructure_failure"

    def _choose_ttt_move(self) -> int:
        legal = [i for i, v in enumerate(self._board) if v is None]
        if not legal:
            return 0
        me = self._current
        opp = 1 - me
        # Randomized tie ordering changes only equally strong moves; minimax remains perfect.
        order = list(legal)
        self._baseline_rng.shuffle(order)

        def winner(board: list[int | None]) -> int | None:
            for a, b, c in _TTT_WINS:
                if board[a] is not None and board[a] == board[b] == board[c]:
                    return int(board[a])
            return None

        def minimax(board: list[int | None], turn: int, depth: int) -> tuple[int, int]:
            w = winner(board)
            if w is not None:
                return (10 - depth if w == me else depth - 10, -1)
            moves = [i for i, v in enumerate(board) if v is None]
            if not moves:
                return (0, -1)
            vals = []
            for i in moves:
                board[i] = turn
                val, _ = minimax(board, 1 - turn, depth + 1)
                board[i] = None
                vals.append((val, i))
            best = max(vals) if turn == me else min(vals)
            return best

        # Preserve randomized order on ties while retaining minimax value/depth preference.
        best_val = -10**9
        best_moves: list[int] = []
        for i in order:
            self._board[i] = me
            val, _ = minimax(self._board, opp, 1)
            self._board[i] = None
            if val > best_val:
                best_val, best_moves = val, [i]
            elif val == best_val:
                best_moves.append(i)
        return best_moves[0] if best_moves else order[0]

    @staticmethod
    def _parse_ttt(action: Any) -> int | None:
        if not isinstance(action, str) or len(action) > 128:
            return None
        m = re.fullmatch(r"\s*\[\s*([0-8])\s*\]\s*", action)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_mm(action: Any) -> tuple[int, ...] | None:
        if not isinstance(action, str) or len(action) > 128:
            return None
        m = re.fullmatch(r"\s*\[\s*([1-6])\s+([1-6])\s+([1-6])\s+([1-6])\s*\]\s*", action)
        if not m:
            return None
        vals = tuple(int(x) for x in m.groups())
        return vals if len(set(vals)) == 4 else None

    def _apply_native(self, native: str, *, actor: int, baseline: bool = False) -> None:
        try:
            with self._native_rng():
                done, _info = self._env.step(native)
            self._move_count += 1
            self._history.append(("baseline" if baseline else "learner") + ": " + native)
            if self._game_id == "TicTacToe-v0":
                move = int(re.search(r"\d", native).group())
                self._board[move] = actor
            self._done = bool(done)
            self._read_observation()
            if self._done:
                self._finalize()
        except Exception as exc:
            self._valid = False
            self._last_error = f"native TextArena failure: {type(exc).__name__}: {exc}"
            self._done = True
            self._termination = "infrastructure_failure"

    def _finalize(self) -> None:
        try:
            state = self._env.state
            rewards = state.rewards
            if not isinstance(rewards, dict) or self._learner not in rewards:
                raise RuntimeError("Native terminal state has no learner reward")
            self._raw_rewards = {str(k): float(v) for k, v in rewards.items()}
            self._raw_reward = float(rewards.get(self._learner, 0.0))
            if not -1.0 <= self._raw_reward <= 1.0:
                raise RuntimeError("Native terminal reward is outside the audited bounds")
            self._score = (self._raw_reward + 1.0) / 2.0 if self._game_id == "TicTacToe-v0" else float(self._raw_reward)
            native_info = (state.game_info or {}).get(self._learner, {})
            self._terminal_reason = str(native_info.get("reason", "terminal"))
            self._termination = "native_invalid_forfeit" if native_info.get("invalid_move") else "native_terminal"
            if native_info.get("invalid_move"):
                self._score = 0.0
        except Exception as exc:
            self._valid = False
            self._last_error = f"terminal scoring failure: {type(exc).__name__}: {exc}"
            self._termination = "infrastructure_failure"
            self._score = 0.0

    def observe(self) -> dict[str, Any]:
        legal: list[str]
        if self._game_id == "TicTacToe-v0":
            legal = [f"[{i}]" for i, v in enumerate(self._board) if v is None] if not self._done else []
            public: dict[str, Any] = {"board": list(self._board), "symbols": {"0": "O", "1": "X"}}
        else:
            legal = [a for a in _MM_ACTIONS if a not in self._guesses] if not self._done else []
            public = {"feedback_history_in_text": True}
        out: dict[str, Any] = {
            "game_id": self._game_id,
            "player_id": self._learner,
            "current_player": self._current,
            "observation_text": self._text,
            "public_state": public,
            "legal_actions": legal,
            "move_count": self._move_count,
            "remaining_attempts": max(0, 32 - self._attempts),
            "done": self._done,
            "score": self._score,
            "terminal_reason": self._terminal_reason,
            "history": self._history[-8:],
        }
        if self._last_error:
            out["last_error"] = self._last_error
        return out

    def step(self, action: Any) -> dict[str, Any]:
        if self._done:
            return self.observe()
        self._attempts += 1
        if isinstance(action, str) and action == "noop":
            self._last_error = "No move applied; wrapper attempt consumed."
            self._check_budget()
            return self.observe()
        if self._game_id == "TicTacToe-v0":
            move = self._parse_ttt(action)
            if self._current != self._learner:
                self._advance_baseline()
                return self.observe()
            if move is None or self._board[move] is not None:
                self._reject("invalid action; expected an empty square as [0] through [8]")
                return self.observe()
            self._last_error = None
            self._apply_native(f"[{move}]", actor=self._learner)
            self._advance_baseline()
        else:
            guess = self._parse_mm(action)
            native = "[" + " ".join(map(str, guess)) + "]" if guess else ""
            if guess is None or native in self._guesses:
                self._reject("invalid or repeated guess; expected a new [a b c d] with four distinct digits 1-6")
                return self.observe()
            self._last_error = None
            self._guesses.add(native)
            self._apply_native(native, actor=0)
        self._check_budget()
        return self.observe()

    def _reject(self, message: str) -> None:
        self._last_error = message
        self._invalid_count += 1
        if self._invalid_count >= 3:
            self._done = True
            self._score = 0.0
            self._termination = "wrapper_invalid_forfeit"
            self._terminal_reason = "Three invalid learner actions in one episode."
        self._check_budget()

    def _check_budget(self) -> None:
        if not self._done and self._attempts >= 32:
            self._done = True
            self._score = 0.0
            self._termination = "wrapper_truncation"
            self._terminal_reason = "32 learner attempts exhausted (including noop and invalid actions)."

    def info(self) -> dict[str, Any]:
        return {
            "valid_trial": bool(self._valid),
            "invalid_reason": self._last_error if not self._valid else None,
            "game_id": self._game_id,
            "seed": self._seed,
            "learner_seat": self._learner,
            "opponent": "Perfect minimax v1 (full game-tree search, seeded optimal tie breaks)" if self._game_id == "TicTacToe-v0" else None,
            "raw_reward": self._raw_reward,
            "raw_rewards": dict(self._raw_rewards),
            "termination": self._termination,
            "wrapper_truncated": self._termination == "wrapper_truncation",
            "learner_attempts": self._attempts,
            "invalid_actions": self._invalid_count,
            "game_weight": 0.5,
            "normalized_score": self._score,
            "terminal_reason": self._terminal_reason,
            "move_count": self._move_count,
        }

    def score_breakdown(self) -> dict[str, float]:
        return {"normalized_score": float(self._score), "raw_reward": float(self._raw_reward or 0.0),
                "literal_win": float(self._done and self._score == 1.0),
                "wrapper_truncation": float(self._termination == "wrapper_truncation"),
                f"game/{self._game_id}": float(self._score)}

    def describe(self) -> str:
        return (
            "TextArena is a deterministic offline turn-based suite. reset(seed) selects one frozen scenario: "
            "even seeds are TicTacToe-v0 against a perfect minimax baseline, and odd seeds are single-player "
            "Mastermind-v0. In TicTacToe, place O/X in row-major squares 0-8 using [n]; three in a row wins, "
            "and a full board is a draw. The learner seat is balanced by seed, and baseline turns are applied "
            "before the next learner observation. In Mastermind, guess four distinct digits 1-6 using [a b c d]; "
            "the native prompt says 2 turns but this release actually allows 3 distinct guesses; feedback gives "
            "black (right place) and white (right digit, wrong place) pegs. "
            "Observation text is native TextArena feedback; public_state contains only public board data. "
            "Score is win/draw/loss = 1/0.5/0 for TicTacToe. Mastermind uses the native fixed numeric reward: "
            "1 when solved, otherwise (black + 0.5*white)/4 on the last guess at the turn limit. "
            "In-progress scores are zero. Noop does not move; every call consumes one of 32 learner attempts. "
            "Three invalid actions in total or exhausting 32 attempts forfeits with score zero. "
            "Repeated Mastermind guesses are invalid. legal_actions lists playable native moves; noop is always "
            "available before termination. Fields include game_id, learner player_id, native current_player "
            "(possibly the last mover after termination), observation_text, public_state (0/1 occupants; null empty), "
            "legal_actions, move_count (native plies), remaining_attempts, done, score, terminal_reason, recent history "
            "and optional last_error. Local rules models and search using visible information are allowed."
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                self._valid = False
                if not self._last_error:
                    self._last_error = "TextArena close failed"


def make_game(**kwargs: Any) -> TextArenaGame:
    """Construct one isolated local TextArena episode wrapper."""
    if kwargs:
        raise TypeError(f"Unknown make_game options: {sorted(kwargs)}")
    return TextArenaGame()
