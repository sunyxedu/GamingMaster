"""Tier-2 adaptor for the headless ``flappy-bird-gymnasium`` environment.

The package is the game engine selected by the study phase.  This module only
adds the Meta-Harness realtime mailbox, a JSON feature observation, and score
bookkeeping; it does not run a wall-clock thread.  The evaluator's ticker owns
the clock and calls :meth:`FlappyBirdGame.tick` once per simulation frame.
"""
from __future__ import annotations

import math
import numbers
import os
import threading
from typing import Any

# Pygame is a dependency of the selected package.  These defaults make its
# import and no-render mode safe on the stated display-less Linux host, while
# still allowing an explicit user setting to win.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import flappy_bird_gymnasium  # noqa: F401  - registers FlappyBird-v0 with Gymnasium
import gymnasium

from metaharness.games.base import ActionSpec


TIER = 2
DETERMINISTIC = True

# Eight or more seeds in each group, with the same package scenario family in
# both groups.  A seed changes Gymnasium's pipe-gap RNG stream; it is not a map
# or a hand-authored level identifier.
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [101, 102, 103, 104, 105, 106, 107, 108],
}

# The study-selected default: the package's fixed 288x512 world and 30 FPS
# simulation cadence.  EVAL_KWARGS is JSON-serialisable and is passed to
# make_game by the evaluator.
EVAL_KWARGS = {"tick_s": 1.0 / 30.0}

_FEATURE_NAMES = (
    "last_pipe_x",
    "last_pipe_top_y",
    "last_pipe_bottom_y",
    "next_pipe_x",
    "next_pipe_top_y",
    "next_pipe_bottom_y",
    "next_next_pipe_x",
    "next_next_pipe_top_y",
    "next_next_pipe_bottom_y",
    "bird_y",
    "bird_vertical_velocity",
    "bird_rotation",
)
_PIPE_LABELS = ("last", "next", "next_next")
_ACTION_DESCRIPTIONS = {
    "IDLE": "Apply no flap impulse for this frame; gravity and pipe motion still advance.",
    "FLAP": "Apply one upward flap impulse for this frame.",
}

# These are the fixed constants of flappy-bird-gymnasium 0.4.0's default
# world.  Keeping them here avoids importing private package constants into the
# public observation, while the engine remains the source of truth for play.
_SCREEN_SIZE = (288, 512)
_BIRD_X = int(_SCREEN_SIZE[0] * 0.2)
_BIRD_WIDTH = 34
_BIRD_HEIGHT = 24
_PIPE_WIDTH = 52
_PIPE_SPEED = 4
_GRAVITY = 1
_FLAP_VELOCITY = -9
_GROUND_Y = _SCREEN_SIZE[1] * 0.79


class FlappyBirdGame:
    """Headless, seeded, frame-clocked Flappy Bird game.

    ``tick`` is deliberately fast and never sleeps.  A valid action becomes
    the held action for this and subsequent frames.  ``None`` means no fresh
    action and repeats that held action; an invalid action is ignored, repeats
    the held action, and is reported in the next observation.
    """

    name = "Flappy Bird (flappy-bird-gymnasium)"
    tick_s = 1.0 / 30.0
    action_spec = ActionSpec(
        names=["IDLE", "FLAP"],
        descriptions=dict(_ACTION_DESCRIPTIONS),
        noop="IDLE",
        notes=(
            "Actions are one-frame commands. The runner's real-time ticker applies "
            "the held action again when no fresh action is posted. String names are "
            "case-insensitive; integer aliases 0=IDLE and 1=FLAP are accepted. "
            "Malformed actions are ignored and recorded rather than raising."
        ),
    )

    def __init__(
        self,
        *,
        tick_s: float = 1.0 / 30.0,
        max_ticks: int | None = None,
        screen_size: tuple[int, int] | list[int] = _SCREEN_SIZE,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        **kwargs: Any,
    ) -> None:
        """Construct the adaptor without starting a process or display.

        ``screen_size``, ``pipe_gap``, and visual options are explicit robustness
        knobs for experiments.  The benchmark defaults above are never changed
        by a seed.  ``max_ticks`` is disabled by default so the selected engine
        remains the source of episode termination; callers may set it for an
        explicitly time-limited experiment.
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected constructor option(s): {unknown}")
        if isinstance(tick_s, bool) or not isinstance(tick_s, numbers.Real):
            raise TypeError("tick_s must be a finite positive number")
        self.tick_s = float(tick_s)
        if not math.isfinite(self.tick_s) or self.tick_s <= 0.0 or self.tick_s >= 10.0:
            raise ValueError("tick_s must be finite, greater than 0, and below 10 seconds")

        if max_ticks is not None:
            if isinstance(max_ticks, bool) or not isinstance(max_ticks, numbers.Integral):
                raise TypeError("max_ticks must be a positive integer or None")
            if int(max_ticks) <= 0:
                raise ValueError("max_ticks must be positive when supplied")
            self.max_ticks: int | None = int(max_ticks)
        else:
            self.max_ticks = None

        if not isinstance(screen_size, (tuple, list)) or len(screen_size) != 2:
            raise TypeError("screen_size must be a pair of positive integers")
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or int(v) <= 0 for v in screen_size):
            raise ValueError("screen_size must be a pair of positive integers")
        self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        self._bird_x = int(self.screen_size[0] * 0.2)
        self._bird_width = _BIRD_WIDTH
        self._bird_height = _BIRD_HEIGHT
        self._pipe_width = _PIPE_WIDTH
        self._ground_y = self.screen_size[1] * 0.79
        if isinstance(pipe_gap, bool) or not isinstance(pipe_gap, numbers.Integral) or int(pipe_gap) <= 0:
            raise ValueError("pipe_gap must be a positive integer")
        self.pipe_gap = int(pipe_gap)
        if not isinstance(bird_color, str) or not bird_color:
            raise ValueError("bird_color must be a non-empty string")
        if not isinstance(pipe_color, str) or not pipe_color:
            raise ValueError("pipe_color must be a non-empty string")
        if background is not None and (not isinstance(background, str) or not background):
            raise ValueError("background must be a non-empty string or None")
        self.bird_color = bird_color
        self.pipe_color = pipe_color
        self.background = background

        self._lock = threading.RLock()
        self._env: Any | None = None
        self._closed = False
        self._reset_state()

    def _reset_state(self) -> None:
        """Clear wrapper state; called while holding ``_lock``."""
        self._seed = 0
        self._features = [0.0] * 12
        self._score = 0.0
        self._max_score_seen = 0.0
        self._ticks = 0
        self._last_reward = 0.0
        self._shaped_reward = 0.0
        self._survival_shaping = 0.0
        self._pipe_reward = 0.0
        self._death_penalty = 0.0
        self._flap_count = 0
        self._current_action = "IDLE"
        self._last_applied_action = "IDLE"
        self._last_requested_action = "IDLE"
        self._last_action_error: str | None = None
        self._invalid_action_count = 0
        self._terminated = False
        self._truncated = False
        self._done = False
        self._termination_cause: str | None = None
        self._invalid_reason: str | None = None
        self._last_engine_info: dict[str, Any] = {"score": 0}

    def _new_env(self) -> Any:
        """Build one fresh no-render Gymnasium environment."""
        # Importing flappy_bird_gymnasium above registers this id.  These
        # options intentionally keep the benchmark symbolic and headless.
        return gymnasium.make(
            "FlappyBird-v0",
            render_mode=None,
            use_lidar=False,
            normalize_obs=False,
            audio_on=False,
            screen_size=self.screen_size,
            pipe_gap=self.pipe_gap,
            bird_color=self.bird_color,
            pipe_color=self.pipe_color,
            background=self.background,
            score_limit=None,
            debug=False,
            disable_env_checker=True,
        )

    @staticmethod
    def _finite(value: Any, *, name: str) -> float:
        """Convert a scalar from NumPy/Python into a strict JSON number."""
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} is not numeric: {type(value).__name__}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} is not finite")
        return result

    def _set_features(self, raw: Any) -> None:
        values = list(raw.tolist() if hasattr(raw, "tolist") else raw)
        if len(values) != 12:
            raise ValueError(f"expected 12 feature values, got {len(values)}")
        self._features = [self._finite(v, name=f"feature[{i}]") for i, v in enumerate(values)]

    @staticmethod
    def _safe_action_text(action: Any) -> str:
        if action is None:
            return "<none>"
        if isinstance(action, str):
            return action[:80]
        if isinstance(action, (numbers.Integral, numbers.Real)) and not isinstance(action, bool):
            return repr(action)[:80]
        if isinstance(action, dict):
            return "<dict>"
        return f"<{type(action).__name__}>"

    def _normalise_action(self, action: Any) -> tuple[str | None, str | None]:
        """Return (canonical action, error); ``None`` means repeat held action."""
        if action is None:
            return None, None
        if isinstance(action, dict):
            if "action" not in action:
                return None, "invalid_action_dict"
            action = action.get("action")
        if isinstance(action, str):
            name = action.strip().upper()
            aliases = {"0": "IDLE", "1": "FLAP", "NOOP": "IDLE", "WAIT": "IDLE"}
            name = aliases.get(name, name)
            if name in self.action_spec.names:
                return name, None
            return None, "invalid_action"
        if isinstance(action, numbers.Integral) and not isinstance(action, bool):
            if int(action) == 0:
                return "IDLE", None
            if int(action) == 1:
                return "FLAP", None
            return None, "invalid_action"
        return None, "invalid_action"

    def reset(self, seed: int) -> dict[str, Any]:
        """Start a fresh seeded episode and return its first observation."""
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        with self._lock:
            old_env = self._env
            self._env = None
            if old_env is not None:
                try:
                    old_env.close()
                except Exception:
                    # A fresh local environment is still usable; no external
                    # process or connection is left behind by this package.
                    pass
            self._reset_state()
            self._seed = int(seed)
            self._closed = False
            try:
                self._env = self._new_env()
                raw, engine_info = self._env.reset(seed=self._seed)
                self._set_features(raw)
                self._last_engine_info = self._clean_engine_info(engine_info)
                self._score = self._engine_score(self._last_engine_info)
                self._max_score_seen = self._score
                return self._observation()
            except Exception:
                if self._env is not None:
                    try:
                        self._env.close()
                    except Exception:
                        pass
                self._env = None
                self._closed = True
                raise

    @staticmethod
    def _clean_engine_info(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        cleaned: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                v = float(value)
                if math.isfinite(v):
                    cleaned[str(key)] = v
            elif isinstance(value, (str, bool)) or value is None:
                cleaned[str(key)] = value
        return cleaned

    @staticmethod
    def _engine_score(info: dict[str, Any]) -> float:
        value = info.get("score", 0.0)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return 0.0
        result = float(value)
        return result if math.isfinite(result) else 0.0

    def _mark_engine_failure(self, error: BaseException) -> None:
        self._invalid_reason = f"local_environment_error: {type(error).__name__}: {error}"[:500]
        self._done = True
        self._terminated = False
        self._truncated = True
        self._termination_cause = "infrastructure_failure"

    def _infer_termination_cause(self, terminated: bool, truncated: bool) -> str | None:
        if truncated:
            return "time_limit" if self.max_ticks is not None else "environment_truncated"
        if not terminated:
            return None
        env = self._env
        if env is None:
            return "collision"
        try:
            raw = env.unwrapped
            player_y = float(raw._player_y)
            ground_y = float(raw._ground["y"])
            if player_y + self._bird_height >= ground_y - 1:
                return "ground_collision"
            return "pipe_collision"
        except Exception:
            return "collision"

    def tick(self, action: Any) -> None:
        """Apply one held action and advance exactly one 30 FPS game frame."""
        with self._lock:
            if self._done or self._closed:
                return
            if self._env is None:
                self._mark_engine_failure(RuntimeError("reset(seed) was not called"))
                return

            self._last_requested_action = self._safe_action_text(action)
            requested, error = self._normalise_action(action)
            if error is None and requested is not None:
                self._current_action = requested
            elif error is not None:
                self._invalid_action_count += 1
            self._last_action_error = error
            applied = self._current_action
            self._last_applied_action = applied
            self._ticks += 1
            if applied == "FLAP":
                self._flap_count += 1

            try:
                raw, reward, terminated, truncated, engine_info = self._env.step(
                    1 if applied == "FLAP" else 0
                )
                self._set_features(raw)
                reward_value = self._finite(reward, name="reward")
                self._last_reward = reward_value
                self._shaped_reward += reward_value
                self._last_engine_info = self._clean_engine_info(engine_info)
                old_score = self._score
                self._score = self._engine_score(self._last_engine_info)
                self._max_score_seen = max(self._max_score_seen, self._score)
                score_delta = max(0.0, self._score - old_score)
                self._pipe_reward += score_delta
                if reward_value == 0.1:
                    self._survival_shaping += reward_value
                elif reward_value < -0.5:
                    self._death_penalty += -reward_value

                self._terminated = bool(terminated)
                self._truncated = bool(truncated)
                if self.max_ticks is not None and self._ticks >= self.max_ticks and not (self._terminated or self._truncated):
                    self._truncated = True
                self._done = self._terminated or self._truncated
                self._termination_cause = self._infer_termination_cause(
                    self._terminated, self._truncated
                )
            except Exception as error_obj:  # a local engine failure invalidates, not crashes, a trial
                self._mark_engine_failure(error_obj)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """One-frame compatibility helper for replay and local smoke tests."""
        with self._lock:
            before = self._score
            self.tick(action)
            return (
                self._observation(),
                float(self._score - before),
                bool(self._terminated),
                bool(self._truncated),
                self.info(),
            )

    def _pipe_records(self) -> list[dict[str, float | str]]:
        records: list[dict[str, float | str]] = []
        for i, label in enumerate(_PIPE_LABELS):
            offset = i * 3
            x, top, bottom = self._features[offset : offset + 3]
            records.append(
                {
                    "name": label,
                    "x": x,
                    "gap_top_y": top,
                    "gap_bottom_y": bottom,
                    "gap_center_y": (top + bottom) / 2.0,
                    "gap_height": bottom - top,
                }
            )
        return records

    def _observation(self) -> dict[str, Any]:
        pipes = self._pipe_records()
        target = pipes[0]
        bird_y = self._features[9]
        top = float(target["gap_top_y"])
        bottom = float(target["gap_bottom_y"])
        top_clearance = bird_y - top
        bottom_clearance = bottom - (bird_y + self._bird_height)
        target_x = float(target["x"])
        if target_x + self._pipe_width < self._bird_x:
            phase = "behind"
        elif target_x <= self._bird_x + self._bird_width:
            phase = "crossing"
        else:
            phase = "approach"

        return {
            "game": "flappy_bird",
            "seed": int(self._seed),
            "frame": int(self._ticks),
            "tick_s": float(self.tick_s),
            "score": float(self._score),
            "pipe_pairs_passed": float(self._score),
            "last_reward": float(self._last_reward),
            "shaped_reward": float(self._shaped_reward),
            "done": bool(self._done),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "termination_cause": self._termination_cause,
            "valid_trial": self._invalid_reason is None,
            "invalid_reason": self._invalid_reason,
            "previous_action": self._last_applied_action,
            "held_action": self._current_action,
            "last_requested_action": self._last_requested_action,
            "last_action_error": self._last_action_error,
            "features": list(self._features),
            "feature_names": list(_FEATURE_NAMES),
            "pipes": pipes,
            "target_pipe": dict(target),
            "bird": {
                "x": float(self._bird_x),
                "y": float(bird_y),
                "vertical_velocity": float(self._features[10]),
                "rotation": float(self._features[11]),
                "width": float(self._bird_width),
                "height": float(self._bird_height),
            },
            "geometry": {
                "screen_width": float(self.screen_size[0]),
                "screen_height": float(self.screen_size[1]),
                "pipe_width": float(self._pipe_width),
                "ground_y": float(self._ground_y),
                "pipe_speed_px_per_frame": float(_PIPE_SPEED),
                "gravity_px_per_frame2": float(_GRAVITY),
                "flap_velocity_px_per_frame": float(_FLAP_VELOCITY),
            },
            "target_geometry": {
                "phase": phase,
                "top_clearance": float(top_clearance),
                "bottom_clearance": float(bottom_clearance),
            },
            "diagnostics": {
                "flap_count": float(self._flap_count),
                "invalid_action_count": float(self._invalid_action_count),
                "max_score_seen": float(self._max_score_seen),
            },
        }

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return self._observation()

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            # This is intentionally the environment's direct pipe-passage
            # score, not its +0.1/frame shaped reward.
            return float(self._score)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            return {
                "primary_score_pipe_pairs": float(self._score),
                "pipe_pairs_passed": float(self._score),
                "max_score_seen": float(self._max_score_seen),
                "alive_frames": float(self._ticks),
                "shaped_reward": float(self._shaped_reward),
                "survival_shaping_reward": float(self._survival_shaping),
                "pipe_reward": float(self._pipe_reward),
                "death_penalty": float(self._death_penalty),
                "flap_count": float(self._flap_count),
            }

    def info(self) -> dict[str, Any]:
        with self._lock:
            result = {
                "name": self.name,
                "tier": 2,
                "environment": "flappy-bird-gymnasium==0.4.0",
                "seed": int(self._seed),
                "score": float(self._score),
                "primary_score": float(self._score),
                "pipe_pairs_passed": float(self._score),
                "max_score": float(self._max_score_seen),
                "alive_frames": int(self._ticks),
                "ticks": int(self._ticks),
                "shaped_reward": float(self._shaped_reward),
                "flap_count": int(self._flap_count),
                "termination_cause": self._termination_cause,
                "terminated": bool(self._terminated),
                "truncated": bool(self._truncated),
                "done": bool(self._done),
                "valid_trial": self._invalid_reason is None,
                "invalid_reason": self._invalid_reason,
                "invalid_action_count": int(self._invalid_action_count),
                "last_action_error": self._last_action_error,
                "tick_s": float(self.tick_s),
                "score_breakdown": self.score_breakdown(),
                "config": {
                    "screen_size": [int(self.screen_size[0]), int(self.screen_size[1])],
                    "pipe_gap": int(self.pipe_gap),
                    "bird_color": self.bird_color,
                    "pipe_color": self.pipe_color,
                    "background": self.background,
                    "max_ticks": self.max_ticks,
                },
            }
            return result

    def describe(self) -> str:
        return (
            "Play a headless Flappy Bird clone at one simulation frame per tick "
            f"({self.tick_s:.6f} seconds, nominally 30 FPS). The world advances "
            "whether or not a new decision is posted. Gravity moves the bird "
            "downward by 1 px/frame² up to 10 px/frame; FLAP sets its vertical "
            "velocity to -9 px/frame, and IDLE applies no impulse. Pipes move left "
            "4 px/frame. The 288x512 default world has a fixed 100 px pipe gap, "
            "34x24 bird, 52 px pipe width, and ground at y=404.48. A collision "
            "with a pipe or the ground ends the episode; touching y<0 incurs a "
            "top-edge shaping penalty but is not itself terminal.\n\n"
            "Objective: maximize the direct score, which is the number of pipe "
            "pairs passed. The environment also returns shaped reward (+0.1 for "
            "an ordinary surviving frame, +1 on a pipe passage, -1 on death, with "
            "a top-edge penalty), but shaped survival reward is diagnostic and is "
            "not the primary score. Seeds are passed to Gymnasium reset(seed); "
            "they select the seeded vertical offsets of the initial and recycled "
            "pipe pairs without changing default physics or spacing.\n\n"
            "Observation is strict JSON. features is the package's exact 12-value "
            "unnormalized vector in feature_names order: three pipe records (last, "
            "next, next-next), each with x and the upper/lower gap boundaries, "
            "followed by bird y, vertical velocity, and rotation. For an upcoming "
            "pipe still beyond the right edge, the package reports x=screen width "
            "and gap boundaries 0 and screen height. pipes, target_pipe, bird, "
            "target_geometry, score, frame, action, and termination fields are "
            "named convenience views of that same state.\n\n"
            "Action grammar: send the bare string IDLE or FLAP (case-insensitive); "
            "integer aliases 0 and 1 are accepted. A dictionary {'action': 'FLAP'} "
            "is also accepted for adapters that carry metadata. A valid action is "
            "held on later ticks until replaced. A missing action repeats the held "
            "action, initially IDLE. Malformed actions are ignored, recorded in "
            "last_action_error, and do not raise. The evaluator's tick counter is "
            "the number of simulation frames applied."
        )

    def close(self) -> None:
        """Close the local Gymnasium environment; safe to call repeatedly."""
        with self._lock:
            env, self._env = self._env, None
            self._closed = True
            if not self._done and env is not None:
                self._done = True
                self._termination_cause = self._termination_cause or "closed"
            if env is not None:
                try:
                    env.close()
                except Exception:
                    # close_game should never be held up by a renderer cleanup;
                    # this environment has no process or network resource.
                    pass


def make_game(**kwargs: Any) -> FlappyBirdGame:
    """Factory used by the evaluator."""
    return FlappyBirdGame(**kwargs)
