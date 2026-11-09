"""Headless Atari 2600 Breakout adaptor backed by official ALE.

The emulator is deliberately owned by this object and is advanced one native
frame by each call to :meth:`BreakoutGame.tick`.  The metaharness supplies the
real-time clock, so there is no second background stepping thread here: a
background thread would race the evaluator's ticker and advance the ROM twice.
"""
from __future__ import annotations

import copy
import math
import numbers
import threading
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np

from metaharness.games.base import ActionSpec

try:  # The pinned study interface uses this read-only RAM decoder.
    from ocatari.ram import breakout as _ocatari_breakout
    from ocatari.ram.extract_ram_info import detect_objects_ram as _detect_objects_ram
except Exception:  # pragma: no cover - requirements.txt supplies OCAtari
    _ocatari_breakout = None
    _detect_objects_ram = None


TIER = 2
DETERMINISTIC = True
# These are ordinary mode-0/difficulty-0 seeds.  The wall is fixed by the ROM;
# seeds mainly affect ALE's seeded serve and sticky-action random draws.
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [100, 101, 102, 103, 104, 105, 106, 107],
}

ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT")
ACTION_INDICES = {name: i for i, name in enumerate(ACTION_NAMES)}
ROW_VALUES = (7, 7, 4, 4, 1, 1)
NATIVE_FPS = 60.0


def _cartridge_score(ram: np.ndarray) -> int:
    """Decode Breakout's three displayed BCD score digits from RAM."""
    return (
        100 * (int(ram[76]) & 15)
        + 10 * (int(ram[77]) >> 4)
        + (int(ram[77]) & 15)
    )


def _fallback_block_bitmap(ram: np.ndarray) -> np.ndarray:
    """Small copy of OCAtari's audited Breakout RAM bit ordering.

    Normal operation calls OCAtari 2.2.1's private helper.  Keeping this
    fallback makes the adaptor fail gracefully if a deployment has ALE but
    omitted OCAtari; it is intentionally the same 6x18 bit permutation rather
    than an inferred brick count.
    """
    array = np.asarray(ram[:36], dtype=np.uint8).reshape(-1, 6)
    blocks_str = ""
    for row in np.asarray(array).T:
        row_str = ""
        for j, bitnumber in enumerate(row):
            if j == 0:
                row_str = f"{int(bitnumber):06b}"[::-2] + row_str
            elif j == 5:
                row_str = f"{int(bitnumber):08b}"[1::-2] + row_str
            else:
                row_str = f"{int(bitnumber):08b}"[::-2] + row_str
        blocks_str = row_str + "\n" + blocks_str
    bits = np.asarray([list(el) for el in blocks_str.split("\n") if el], dtype=np.uint8)
    return bits.T[[0, 4, 3, 2, 1, 5, 6, 7, 8, 11, 12, 16, 15, 14, 13, 17, 18, 19]].T


def _json_float(value: Any) -> float:
    """Return a finite builtin float for a reward or metric."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value {value!r}")
    return result


class BreakoutGame:
    """RealtimeGame facade for ``ALE/Breakout-v5`` in symbolic RAM mode."""

    name = "Atari 2600 Breakout"
    tick_s = 1.0 / NATIVE_FPS
    action_spec = ActionSpec(
        names=list(ACTION_NAMES),
        descriptions={
            "NOOP": "Remove directional input for one native Atari frame; it does not guarantee an instant paddle stop.",
            "FIRE": "Press the cartridge fire button to serve the initial ball or the next ball after a miss.",
            "RIGHT": "Move the paddle right for one native Atari frame.",
            "LEFT": "Move the paddle left for one native Atari frame.",
        },
        noop="NOOP",
        notes=(
            "Gym indices are NOOP=0, FIRE=1, RIGHT=2, LEFT=3. The v5 baseline keeps "
            "frameskip=1 and 0.25 sticky-action probability; a tick is one native frame."
        ),
    )

    def __init__(
        self,
        *,
        mode: int = 0,
        difficulty: int = 0,
        frameskip: int = 1,
        repeat_action_probability: float = 0.25,
        max_num_frames_per_episode: int = 108_000,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(map(str, kwargs)))
            raise TypeError(f"unknown Breakout adaptor option(s): {unknown}")
        if int(frameskip) != 1:
            raise ValueError("the tier-2 Breakout adaptor requires frameskip=1 (one native frame per tick)")
        if not 0.0 <= float(repeat_action_probability) <= 1.0:
            raise ValueError("repeat_action_probability must be between 0 and 1")

        # Registration is explicit in Gymnasium 1.x.  It is safe to call more
        # than once and is needed before gym.make resolves the ALE id.
        gym.register_envs(ale_py)
        spec = gym.spec("ALE/Breakout-v5")
        if spec is None:  # pragma: no cover - gym.spec raises on failure
            raise RuntimeError("ALE/Breakout-v5 is not registered")

        self.mode = int(mode)
        self.difficulty = int(difficulty)
        self.frameskip = int(frameskip)
        self.repeat_action_probability = float(repeat_action_probability)
        self.max_num_frames_per_episode = int(max_num_frames_per_episode)
        if self.max_num_frames_per_episode <= 0:
            raise ValueError("max_num_frames_per_episode must be positive")

        self._lock = threading.RLock()
        self._closed = False
        self._env = gym.make(
            "ALE/Breakout-v5",
            obs_type="ram",
            render_mode=None,
            mode=self.mode,
            difficulty=self.difficulty,
            full_action_space=False,
            frameskip=self.frameskip,
            repeat_action_probability=self.repeat_action_probability,
            max_num_frames_per_episode=self.max_num_frames_per_episode,
        )
        meanings = list(self._env.unwrapped.get_action_meanings())
        if meanings != list(ACTION_NAMES):
            self._env.close()
            raise RuntimeError(
                "unexpected ALE minimal action order: "
                f"{meanings!r}; expected {list(ACTION_NAMES)!r}"
            )

        self._objects: Any = None
        self._last_ram = np.zeros(128, dtype=np.uint8)
        self._observation: dict[str, Any] = self._empty_observation()
        self._seed: int | None = None
        self._env_info: dict[str, Any] = {}
        self._cumulative_reward = 0.0
        self._last_reward = 0.0
        self._last_action = "NOOP"
        self._last_requested_action: Any = None
        self._action_error: str | None = None
        self._terminated = False
        self._truncated = False
        self._done = False
        self._ticks = 0
        self._valid_trial = True
        self._invalid_reason: str | None = None
        self._previous_ball_xy: tuple[int, int] | None = None
        self._ball_velocity_valid = False
        self._wall_index = 0
        self._walls_cleared = 0
        self._score_match = True
        self._last_decoder_error: str | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        """Start a new seeded ALE episode without creating a second emulator."""
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"reset seed must be an integer, got {seed!r}")
        with self._lock:
            self._ensure_open()
            observation, info = self._env.reset(seed=int(seed))
            ram = np.asarray(observation, dtype=np.uint8)
            if ram.shape != (128,):
                raise RuntimeError(f"ALE RAM observation has shape {ram.shape}, expected (128,)")
            self._seed = int(seed)
            self._env_info = self._plain_info(info)
            self._last_ram = ram.copy()
            self._objects = (
                _ocatari_breakout._init_objects_ram(hud=False)
                if _ocatari_breakout is not None
                else None
            )
            self._cumulative_reward = 0.0
            self._last_reward = 0.0
            self._last_action = "NOOP"
            self._last_requested_action = None
            self._action_error = None
            self._terminated = False
            self._truncated = False
            self._done = False
            self._ticks = 0
            self._valid_trial = True
            self._invalid_reason = None
            self._previous_ball_xy = None
            self._ball_velocity_valid = False
            self._wall_index = 0
            self._walls_cleared = 0
            self._score_match = True
            self._last_decoder_error = None
            self._update_observation()
            return self.observe()

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._observation)

    def tick(self, action: Any) -> None:
        """Apply one current action and advance exactly one native ALE frame.

        The metaharness runner calls this method from its own 60-Hz ticker.  A
        malformed action is converted to NOOP and reported in the next JSON
        observation instead of raising and spoiling the episode.
        """
        with self._lock:
            if self._closed or self._done:
                return
            normalized, error = self._normalize_action(action)
            self._last_requested_action = self._json_value(action)
            if error:
                self._action_error = error
            else:
                self._action_error = None
            self._last_action = normalized
            try:
                observation, reward, terminated, truncated, info = self._env.step(
                    ACTION_INDICES[normalized]
                )
                ram = np.asarray(observation, dtype=np.uint8)
                if ram.shape != (128,):
                    raise RuntimeError(
                        f"ALE RAM observation has shape {ram.shape}, expected (128,)"
                    )
                self._last_ram = ram.copy()
                self._last_reward = _json_float(reward)
                self._cumulative_reward += self._last_reward
                self._terminated = bool(terminated)
                self._truncated = bool(truncated)
                self._done = self._terminated or self._truncated
                self._env_info = self._plain_info(info)
                self._ticks += 1
                self._update_observation()
            except Exception as exc:  # infrastructure failure: let evaluator retry
                self._mark_invalid(f"ALE step failed: {type(exc).__name__}: {exc}")
                self._done = True
                self._terminated = False
                self._truncated = False
                self._observation = self._error_observation()

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._cumulative_reward)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def info(self) -> dict[str, Any]:
        """Return evaluator-only diagnostics, including trial validity."""
        with self._lock:
            ram_score = _cartridge_score(self._last_ram)
            bricks = self._brick_count(self._last_ram)
            return {
                "valid_trial": bool(self._valid_trial),
                "invalid_reason": self._invalid_reason,
                "seed": self._seed,
                "mode": self.mode,
                "difficulty": self.difficulty,
                "frameskip": self.frameskip,
                "repeat_action_probability": self.repeat_action_probability,
                "max_num_frames_per_episode": self.max_num_frames_per_episode,
                "ale_package": getattr(ale_py, "__version__", "0.12.1"),
                "gymnasium_package": getattr(gym, "__version__", "unknown"),
                "ticks": self._ticks,
                "episode_frame_number": int(self._env_info.get("episode_frame_number", self._ticks)),
                "frame_number": int(self._env_info.get("frame_number", self._ticks)),
                "terminated": bool(self._terminated),
                "truncated": bool(self._truncated),
                "score": float(self._cumulative_reward),
                "cartridge_score": ram_score,
                "score_match": bool(self._score_match),
                "lives": self._lives(self._env_info, self._last_ram),
                "bricks_remaining": bricks,
                "wall_index": self._wall_index,
                "walls_cleared": self._walls_cleared,
                "both_walls_cleared": bool(self._walls_cleared >= 2 or ram_score >= 864),
                "decoder": "ocatari.ram.breakout 2.2.1" if _ocatari_breakout is not None else "builtin fallback",
                "decoder_error": self._last_decoder_error,
            }

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            ram_score = _cartridge_score(self._last_ram)
            bricks = self._brick_count(self._last_ram)
            return {
                "ale_reward": float(self._cumulative_reward),
                "cartridge_score": float(ram_score),
                "lives_remaining": float(self._lives(self._env_info, self._last_ram)),
                "bricks_remaining": float(bricks),
                "wall_index": float(self._wall_index),
                "walls_cleared": float(self._walls_cleared),
                "elapsed_emulated_frames": float(self._ticks),
                "both_walls_cleared": 1.0 if self._walls_cleared >= 2 or ram_score >= 864 else 0.0,
            }

    def describe(self) -> str:
        return (
            "Atari 2600 Breakout, ordinary single-player cartridge mode 0 at difficulty 0. "
            "The objective is to keep the ball in play with the horizontal paddle and clear "
            "two six-row walls for points: rows from top to bottom are worth 7, 7, 4, 4, 1, "
            "and 1, with 18 columns per row and a manual maximum of 864 points. A game starts "
            "with five lives/balls. FIRE serves the initial ball and a replacement after a miss; "
            "a missing ball in the observation means the game is waiting for FIRE, not that the "
            "episode was reset. ALE ends naturally when the lives are exhausted and truncates at "
            "the configured frame cap.\n\n"
            "This is a real-time interface: one tick is one native Atari frame at 60 Hz "
            "(tick_s=1/60), frameskip=1, with the v5 sticky-action probability 0.25. The runner "
            "calls tick(action) on its own clock; a late or missing decision repeats the runner's "
            "current NOOP/last action. NOOP removes directional input but does not promise an "
            "instantaneous paddle brake. Gym action names and indices are NOOP=0, FIRE=1, "
            "RIGHT=2, LEFT=3; use names, not raw ALE enum values. Malformed actions are applied "
            "as NOOP and reported in action_error.\n\n"
            "Each observation is compact JSON RAM state: frame and global frame counters, "
            "cumulative unmodified ALE reward (score), a separate decoded cartridge_score "
            "cross-check, lives, paddle and ball rectangles, ball dx/dy when a previous ball "
            "position exists, ball_in_play/serve_required, the exact six-by-18 brick_bitmap "
            "and occupied bricks with row values, bricks_remaining, wall_index, walls_cleared, "
            "both_walls_cleared, last_reward, last_action, and termination flags. A ball is null "
            "when absent; no screenshot or display is used."
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._env.close()
            except Exception as exc:  # close is idempotent; retain diagnostics, do not leak
                self._mark_invalid(f"ALE close failed: {type(exc).__name__}: {exc}")

    # ---- decoding and bookkeeping ---------------------------------------------------------

    def _update_observation(self) -> None:
        ram = self._last_ram
        try:
            if self._objects is not None and _detect_objects_ram is not None:
                _detect_objects_ram(self._objects, ram, "Breakout", False)
            if _ocatari_breakout is not None:
                bitmap = np.asarray(_ocatari_breakout._make_block_bitmap(ram), dtype=np.uint8)
            else:
                bitmap = _fallback_block_bitmap(ram)
            if bitmap.shape != (6, 18):
                raise RuntimeError(f"Breakout bitmap has shape {bitmap.shape}, expected (6, 18)")
            self._last_decoder_error = None
        except Exception as exc:
            # A decoder failure is an infrastructure failure, not a bad action.
            self._last_decoder_error = f"{type(exc).__name__}: {exc}"
            self._mark_invalid(f"RAM decoder failed: {self._last_decoder_error}")
            bitmap = _fallback_block_bitmap(ram)

        bitmap_list = [[int(v) for v in row] for row in bitmap.tolist()]
        brick_count = int(bitmap.sum())
        ram_score = _cartridge_score(ram)
        self._score_match = math.isclose(
            float(self._cumulative_reward), float(ram_score), rel_tol=0.0, abs_tol=1e-6
        )
        # The ordinary cartridge gives 432 points per wall.  The bitmap is used
        # for current occupancy; score thresholds make wall transitions robust
        # to a one-frame all-zero transition in RAM.
        self._walls_cleared = max(
            self._walls_cleared, min(2, max(0, ram_score // 432))
        )
        self._wall_index = min(1, self._walls_cleared)

        paddle_x = int(ram[72]) - 47
        ball: dict[str, Any] | None = None
        ball_xy: tuple[int, int] | None = None
        ball_present = int(ram[101]) != 0 and int(ram[101]) + 9 <= 196
        if ball_present:
            ball_xy = (int(ram[99]) - 49, int(ram[101]) + 9)
            velocity_valid = self._previous_ball_xy is not None
            dx = ball_xy[0] - self._previous_ball_xy[0] if velocity_valid else 0
            dy = ball_xy[1] - self._previous_ball_xy[1] if velocity_valid else 0
            ball = {
                "x": ball_xy[0],
                "y": ball_xy[1],
                "w": 2,
                "h": 4,
                "dx": int(dx),
                "dy": int(dy),
            }
            self._ball_velocity_valid = velocity_valid
        else:
            self._ball_velocity_valid = False
        self._previous_ball_xy = ball_xy

        bricks = [
            {
                "row": row,
                "col": col,
                "x": 8 + col * 8,
                "y": 57 + row * 6,
                "w": 8,
                "h": 6,
                "value": ROW_VALUES[row],
            }
            for row in range(6)
            for col in range(18)
            if bitmap_list[row][col]
        ]
        frame = int(self._env_info.get("episode_frame_number", self._ticks))
        global_frame = int(self._env_info.get("frame_number", self._ticks))
        lives = self._lives(self._env_info, ram)
        self._observation = {
            "frame": frame,
            "global_frame": global_frame,
            "score": float(self._cumulative_reward),
            "cartridge_score": int(ram_score),
            "score_match": bool(self._score_match),
            "lives": lives,
            "lives_from_ram": int(ram[57]),
            "paddle": {"x": paddle_x, "y": 189, "w": 16, "h": 4},
            "ball": ball,
            "ball_velocity_valid": bool(self._ball_velocity_valid),
            "ball_in_play": ball is not None,
            "serve_required": bool(ball is None and lives > 0 and not self._done),
            "brick_bitmap": bitmap_list,
            "bricks": bricks,
            "bricks_remaining": brick_count,
            "wall_index": int(self._wall_index),
            "walls_cleared": int(self._walls_cleared),
            "both_walls_cleared": bool(self._walls_cleared >= 2 or ram_score >= 864),
            "last_reward": float(self._last_reward),
            "cumulative_reward": float(self._cumulative_reward),
            "last_action": self._last_action,
            "requested_action": self._last_requested_action,
            "action_error": self._action_error,
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "done": bool(self._done),
            "ticks": int(self._ticks),
        }

    def _error_observation(self) -> dict[str, Any]:
        result = copy.deepcopy(self._observation)
        result["done"] = True
        result["terminated"] = bool(self._terminated)
        result["truncated"] = bool(self._truncated)
        result["action_error"] = self._invalid_reason
        return result

    def _empty_observation(self) -> dict[str, Any]:
        return {
            "frame": 0,
            "global_frame": 0,
            "score": 0.0,
            "cartridge_score": 0,
            "score_match": True,
            "lives": 0,
            "lives_from_ram": 0,
            "paddle": {"x": 99, "y": 189, "w": 16, "h": 4},
            "ball": None,
            "ball_velocity_valid": False,
            "ball_in_play": False,
            "serve_required": False,
            "brick_bitmap": [[0] * 18 for _ in range(6)],
            "bricks": [],
            "bricks_remaining": 0,
            "wall_index": 0,
            "walls_cleared": 0,
            "both_walls_cleared": False,
            "last_reward": 0.0,
            "cumulative_reward": 0.0,
            "last_action": "NOOP",
            "requested_action": None,
            "action_error": None,
            "terminated": False,
            "truncated": False,
            "done": False,
            "ticks": 0,
        }

    @staticmethod
    def _brick_count(ram: np.ndarray) -> int:
        if _ocatari_breakout is not None:
            bitmap = _ocatari_breakout._make_block_bitmap(ram)
        else:
            bitmap = _fallback_block_bitmap(ram)
        return int(np.asarray(bitmap).sum())

    @staticmethod
    def _lives(info: dict[str, Any], ram: np.ndarray) -> int:
        value = info.get("lives", int(ram[57]))
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(ram[57])

    @staticmethod
    def _plain_info(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, numbers.Integral) and not isinstance(value, bool):
                result[str(key)] = int(value)
            elif isinstance(value, numbers.Real) and not isinstance(value, bool):
                result[str(key)] = float(value)
            elif isinstance(value, (str, bool)) or value is None:
                result[str(key)] = value
            elif isinstance(value, (tuple, list)):
                result[str(key)] = [int(v) if isinstance(v, numbers.Integral) else v for v in value]
        return result

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return repr(value)
            return value
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            value = float(value)
            return value if math.isfinite(value) else repr(value)
        return repr(value)[:200]

    @staticmethod
    def _normalize_action(action: Any) -> tuple[str, str | None]:
        if isinstance(action, str):
            candidate = action.strip().upper()
            if candidate in ACTION_INDICES:
                return candidate, None
            return "NOOP", f"illegal action {action!r}; applied NOOP"
        if isinstance(action, numbers.Integral) and not isinstance(action, bool):
            index = int(action)
            if 0 <= index < len(ACTION_NAMES):
                return ACTION_NAMES[index], None
            return "NOOP", f"illegal action index {action!r}; applied NOOP"
        return "NOOP", f"malformed action {type(action).__name__}; applied NOOP"

    def _mark_invalid(self, reason: str) -> None:
        self._valid_trial = False
        self._invalid_reason = str(reason)[:500]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("BreakoutGame is closed")


def make_game(**kwargs: Any) -> BreakoutGame:
    """Construct one fresh emulator instance for an evaluator episode."""
    return BreakoutGame(**kwargs)
