"""Real Atari 2600 Robot Pong; headless ALE, one frame per runner tick."""
from __future__ import annotations

import copy
import numbers
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
from metaharness.games.base import ActionSpec

TIER = 2
DETERMINISTIC = True
# Fixed benchmark scenarios: hold out the second cartridge variation, not RNG aliases.
SEEDS = {"train": [0, 1, 2, 3], "heldout": [4, 5, 6, 7]}
EVAL_KWARGS = {"max_frames": 60_000}

_ACTIONS = {"NOOP": 0, "FIRE": 1, "RIGHT": 2, "LEFT": 3,
            "RIGHTFIRE": 4, "LEFTFIRE": 5}
_ROM_MD5 = "60e0ea3cbe0913d39803477945e9e5ec"
_TOP, _BOTTOM = 34, 194  # Visible court rows; bottom is exclusive.


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class AtariPong:
    name = "Atari 2600 Pong (ALE Robot Pong)"
    tick_s = 1.0 / 60.0
    action_spec = ActionSpec(
        names=list(_ACTIONS),
        descriptions={
            "NOOP": "Keep the paddle input unchanged; filtered movement may settle briefly.",
            "FIRE": "Press the red button for the variation's contact feature; serves are automatic.",
            "RIGHT": "Move the controlled right paddle UP (screen y decreases), without the button.",
            "LEFT": "Move the controlled right paddle DOWN (screen y increases), without the button.",
            "RIGHTFIRE": "Move the right paddle UP while pressing the red button.",
            "LEFTFIRE": "Move the right paddle DOWN while pressing the red button.",
        },
        noop="NOOP",
        notes="One emulator frame per tick, no hidden controller or macros. Integers 0..5 are also accepted in the listed order. Malformed actions become NOOP.",
    )

    def __init__(self, *, max_frames: int = 60_000, mode: int | None = None,
                 difficulty: int | None = None) -> None:
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        for label, value, choices in (("mode", mode, (0, 1)),
                                      ("difficulty", difficulty, (0, 1, 2, 3))):
            if value is not None and (type(value) is not int or value not in choices):
                raise ValueError(f"{label} must be None or one of {choices}")
        self.max_frames = max_frames
        self._mode_override, self._difficulty_override = mode, difficulty
        self._env: Any | None = None
        self._seed: int | None = None
        self._mode = self._difficulty = self._scenario_index = 0
        self._clear_episode()

    def _clear_episode(self) -> None:
        self._ticks = self._frame_number = 0
        self._own_points = self._opponent_points = self._last_reward = 0
        self._done = self._terminated = self._truncated = False
        self._valid_trial = True
        self._invalid_reason: str | None = None
        self._last_action_error: str | None = None
        self._last_ram = np.zeros(128, dtype=np.uint8)
        self._screen = np.zeros((210, 160, 3), dtype=np.uint8)
        self._ale_info: dict[str, Any] = {}
        self._last_ball: tuple[float, float] | None = None
        self._observation: dict[str, Any] = {}

    def _new_env(self) -> Any:
        gym.register_envs(ale_py)
        return gym.make(
            "ALE/Pong-v5", obs_type="ram", frameskip=1,
            repeat_action_probability=0.0, full_action_space=False,
            mode=self._mode, difficulty=self._difficulty, render_mode=None,
            # Count applied ticks ourselves: mode selection consumes a native setup frame.
            max_num_frames_per_episode=0,
        )

    def reset(self, seed: int) -> dict[str, Any]:
        if not isinstance(seed, numbers.Integral) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.close()
        self._clear_episode()
        self._seed = int(seed)
        self._scenario_index = self._seed % 8
        self._mode = (self._scenario_index // 4 if self._mode_override is None
                      else self._mode_override)
        self._difficulty = (self._scenario_index % 4 if self._difficulty_override is None
                            else self._difficulty_override)
        try:
            self._env = self._new_env()
            meanings = self._env.unwrapped.get_action_meanings()
            if list(meanings) != list(_ACTIONS):
                raise RuntimeError(f"Unexpected ALE minimal action set: {meanings!r}")
            ram, info = self._env.reset(seed=self._seed % (2**32))
            self._capture(ram, info)
            if int(self._last_ram[22]) != self._mode:
                raise RuntimeError("ALE did not select the requested cartridge variation")
            self._check_scores()
            self._observation = self._parse_observation()
        except Exception:
            self.close()
            raise
        return self.observe()

    def _capture(self, ram: np.ndarray, info: dict[str, Any]) -> None:
        self._last_ram = np.asarray(ram, dtype=np.uint8).copy()
        self._ale_info = dict(info)
        self._frame_number = int(info["frame_number"])
        self._screen = self._env.unwrapped.ale.getScreenRGB().copy()
        if self._screen.shape != (210, 160, 3) or self._last_ram.shape != (128,):
            raise RuntimeError("Unexpected Pong ROM screen/RAM geometry")

    def _check_scores(self) -> None:
        # Independently check reward event accounting using ALE's documented score bytes.
        if (int(self._last_ram[14]), int(self._last_ram[13])) != (
            self._own_points, self._opponent_points
        ):
            raise RuntimeError("Reward counters disagree with cartridge scores")

    @staticmethod
    def _action_index(action: Any) -> tuple[int, str | None]:
        if isinstance(action, str) and len(action) <= 32:
            name = action.strip().upper()
            if name in _ACTIONS:
                return _ACTIONS[name], None
        elif isinstance(action, numbers.Integral) and not isinstance(action, bool):
            index = int(action)
            if 0 <= index < len(_ACTIONS):
                return index, None
        # Avoid arbitrary repr(): it can raise, expose input secrets, or be extremely large.
        return 0, "Ignored malformed action; applied NOOP instead."

    def tick(self, action: Any) -> None:
        if self._env is None or self._done:
            return
        index, self._last_action_error = self._action_index(action)
        try:
            ram, reward, terminated, truncated, info = self._env.step(index)
            self._ticks += 1
            if reward not in (-1.0, 0.0, 1.0):
                raise RuntimeError(f"Unexpected one-frame Pong reward: {reward!r}")
            self._last_reward = int(reward)
            self._own_points += int(reward > 0)
            self._opponent_points += int(reward < 0)
            self._terminated = bool(terminated)
            self._truncated = bool(truncated or (
                self._ticks >= self.max_frames and not self._terminated
            ))
            self._done = self._terminated or self._truncated
            self._capture(ram, info)
            self._check_scores()
            self._observation = self._parse_observation()
        except Exception as exc:
            self._valid_trial = False
            self._invalid_reason = f"ALE tick failed: {type(exc).__name__}: {exc}"[:500]
            self._done = True
            self._observation.update({
                "done": True, "phase": "invalid_trial", "match_result": "invalid",
                "ticks": self._ticks, "frame": self._frame_number,
                "own_points": self._own_points, "opponent_points": self._opponent_points,
                "return": self._own_points - self._opponent_points,
                "action_error": self._last_action_error,
                "serve_ready": False, "serve_waiting": False,
            })

    def _paddle(self, x: int, background: np.ndarray, ball_color: np.ndarray
                ) -> tuple[float | None, int | None]:
        strip = self._screen[_TOP:_BOTTOM, x:x + 4]
        # Ball pixels must not become a phantom paddle after a miss or at contact.
        mask = np.any(strip != background, axis=2) & np.any(strip != ball_color, axis=2)
        b = _bounds(mask)
        if b is None:
            return None, None
        return (b[1] + b[3]) / 2.0 + _TOP, b[3] - b[1] + 1

    def _ball(self, ball_color: np.ndarray) -> tuple[float, float] | None:
        # Full width preserves the ball at paddle contacts and after a miss.
        b = _bounds(np.all(self._screen[_TOP:_BOTTOM] == ball_color, axis=2))
        if b is None:
            return None
        x0, y0, x1, y1 = b
        # TIA register updates can shear a 2x4 sprite across five scanlines.
        # Report uncertain perception as absent, never invalidate an otherwise real match.
        if x1 - x0 > 2 or y1 - y0 > 5:
            return None
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0 + _TOP

    def _result(self) -> str:
        if not self._valid_trial:
            return "invalid"
        if self._terminated:
            if self._own_points >= 21:
                return "win"
            if self._opponent_points >= 21:
                return "loss"
            return "terminated"
        return "truncated" if self._truncated else "in_progress"

    def _parse_observation(self) -> dict[str, Any]:
        # Sparse color sample avoids sorting 18,000 RGB triplets every frame.
        sample = self._screen[_TOP:_BOTTOM:31, 24:136:29].reshape(-1, 3)
        colors, counts = np.unique(sample, axis=0, return_counts=True)
        background = colors[int(np.argmax(counts))]
        ball_color = self._screen[25, 0]  # Ball shares the upper/lower border color.
        left_y, left_height = self._paddle(16, background, ball_color)
        right_y, right_height = self._paddle(140, background, ball_color)
        ball = self._ball(ball_color)
        # Independently verified against actual NOOP serve traces, not inferred from pixels.
        serve_frames = int(self._last_ram[11])
        rally_active = not self._done and serve_frames == 0
        dx = dy = None
        if rally_active and ball is not None and self._last_ball is not None:
            raw_dx, raw_dy = ball[0] - self._last_ball[0], ball[1] - self._last_ball[1]
            if abs(raw_dx) <= 16 and abs(raw_dy) <= 16:
                dx, dy = raw_dx, raw_dy
        self._last_ball = ball if rally_active else None
        sx = None if dx is None else (1 if dx > 0 else -1 if dx < 0 else 0)
        sy = None if dy is None else (1 if dy > 0 else -1 if dy < 0 else 0)
        return {
            "game": self.name, "seed": self._seed, "scenario_index": self._scenario_index,
            "mode": self._mode, "difficulty": self._difficulty,
            "cartridge_game": int(self._last_ram[22]) + 1,
            "variation": f"Robot Pong {self._mode + 1}",
            "left_difficulty_switch": "A" if self._difficulty & 1 else "B",
            "right_difficulty_switch": "A" if self._difficulty & 2 else "B",
            "left_paddle_size": "half" if self._difficulty & 1 else "full",
            "right_paddle_size": "half" if self._difficulty & 2 else "full",
            "left_paddle_nominal_height_px": 8 if self._difficulty & 1 else 16,
            "right_paddle_nominal_height_px": 8 if self._difficulty & 2 else 16,
            "frame": self._frame_number, "ticks": self._ticks,
            "episode_frame": int(self._ale_info.get("episode_frame_number", 0)),
            "frames_remaining": max(0, self.max_frames - self._ticks),
            "ram_frame_counter_mod_256": int(self._last_ram[8]),
            "left_paddle_y": left_y, "left_paddle_height_px": left_height,
            "right_paddle_y": right_y, "right_paddle_height_px": right_height,
            "ball_visible": ball is not None,
            "ball_x": None if ball is None else ball[0],
            "ball_y": None if ball is None else ball[1],
            "ball_vx": dx, "ball_vy": dy, "ball_vx_sign": sx, "ball_vy_sign": sy,
            "ball_approaching_right": None if sx is None else sx > 0,
            "serve_ready": False, "serve_automatic": True,
            "serve_waiting": bool(serve_frames > 0 and not self._done),
            "serve_countdown_frames": serve_frames if not self._done else 0,
            "phase": ("finished" if self._done else "serve_delay" if serve_frames else
                      "rally" if ball is not None else "rally_ball_occluded"),
            "last_reward": self._last_reward, "own_points": self._own_points,
            "opponent_points": self._opponent_points,
            "return": self._own_points - self._opponent_points,
            "done": self._done, "terminated": self._terminated, "truncated": self._truncated,
            "match_result": self._result(), "action_error": self._last_action_error,
        }

    def observe(self) -> dict[str, Any]:
        return copy.deepcopy(self._observation)

    @property
    def done(self) -> bool:
        return bool(self._done)

    @property
    def score(self) -> float:
        return float(self._own_points - self._opponent_points)

    @property
    def ticks(self) -> int:
        return int(self._ticks)

    def score_breakdown(self) -> dict[str, float]:
        result = self._result()
        return {
            "own_points": float(self._own_points),
            "opponent_points": float(self._opponent_points), "return": self.score,
            "win": float(result == "win"), "loss": float(result == "loss"),
        }

    def info(self) -> dict[str, Any]:
        return {
            "valid_trial": self._valid_trial, "invalid_reason": self._invalid_reason,
            "seed": self._seed, "scenario_index": self._scenario_index,
            "mode": self._mode, "difficulty": self._difficulty,
            "cartridge_game": self._mode + 1, "ticks": self._ticks,
            "frame_number": self._frame_number, "max_frames": self.max_frames,
            "own_points": self._own_points, "opponent_points": self._opponent_points,
            "return": self.score, "final_score": f"{self._own_points}:{self._opponent_points}",
            "result": self._result(), "terminated": self._terminated,
            "truncated": self._truncated, "rom_md5": _ROM_MD5,
            "observation_source": "RGB sprite geometry + verified RAM serve timer",
            "ale_version": ale_py.__version__,
            "ale_info": {key: int(value) for key, value in self._ale_info.items()
                         if key in ("episode_frame_number", "frame_number", "lives")
                         and isinstance(value, numbers.Integral)},
        }

    def describe(self) -> str:
        return (
            "Play real Atari 2600 Robot Pong through headless ALE/Pong-v5. You control the "
            "right paddle; the cartridge controls the left. Return the ball past the opponent "
            "to earn +1; missing costs -1. Score is own_points minus opponent_points. First to "
            "21 wins. A frame-budget timeout is truncated, never a match win or loss. "
            "Both Robot Pong variations serve AUTOMATICALLY, initially after 59 frames and "
            "after points after about 63 frames; FIRE is not needed to start. "
            "Actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE (or integer indices 0..5). "
            "RIGHT moves the right paddle UP (smaller y); LEFT moves it DOWN. FIRE presses "
            "the red button for the selected variation's contact feature; FIRE variants combine "
            "it with movement. NOOP stops changing the input, but hardware filtering can settle "
            "for a few frames. There are no macros or built-in paddle policies. Invalid actions "
            "become NOOP and action_error reports this for one tick. The runner repeats the last "
            "posted action when decisions are late. One tick advances one emulator frame at "
            "60 Hz (tick_s=1/60); observe never advances time. No adaptor background clock runs. "
            "Observations are JSON: coordinates are centers of VISIBLE sprite pixels on a "
            "160x210 screen, x rightward, y downward. Court rows 34..193; left paddle columns "
            "16..19, right paddle 140..143. Paddle y/height_px describe visible extents, which "
            "clip at the walls; nominal_height_px and size report 16/full or 8/half. Null means "
            "absent/occluded, not coordinate zero. Ball vx/vy are consecutive visible-position "
            "differences in pixels/frame; signs are -1/0/1 or null. Velocity is unknown on "
            "first sight, during serves or after occlusion; clipping perturbs wall samples. "
            "ball_approaching_right is inferred from vx. phase distinguishes serve_delay, rally, "
            "rally_ball_occluded and finished. serve_countdown_frames is a verified RAM timer; "
            "serve_waiting means that timer is active, not just missing pixels. serve_automatic "
            "is true; serve_ready is false because no manual serve is needed. A stale ball may "
            "remain visible on a point frame; phase takes precedence. own_points/opponent_points "
            "count signed reward events; last_reward and return are the latest/total differential. "
            "done, terminated, truncated and match_result report outcome. ticks counts applied "
            "ticks since reset; frame/episode_frame are ALE counters (mode selection may consume "
            "a setup frame); frames_remaining is the adaptor limit. ram_frame_counter_mod_256 "
            "is the cartridge's wrapping timer. seed/scenario_index identify the start. Default "
            "seed modulo 8 selects mode=floor(index/4), difficulty=index%4. Modes 0/1 select "
            "cartridge games 1/2 (Robot Pong), identified by variation/cartridge_game. Difficulty "
            "bit 0 selects the left A/B switch and bit 1 the right: A halves its paddle, B is full. "
            "Train seeds 0..3 and held-out seeds 4..7 are disjoint configurations. Same settings "
            "and actions repeat deterministically; a seed alone does not randomize this cartridge."
        )

    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            env.close()
        # Dropping this reference releases native ALE/Stella. No child process, connection,
        # window, thread or open output file is owned by this adaptor. Idempotent.


def make_game(**kwargs: Any) -> AtariPong:
    return AtariPong(**kwargs)
