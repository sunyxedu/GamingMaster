"""Turn based MiniGrid/BabyAI adaptor.

The adaptor talks directly to the installed Gymnasium environment.  It keeps the
observation public (the egocentric symbolic image and mission) and never reads
the hidden grid to drive play or scoring.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid and BabyAI IDs
import numpy as np

from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True

# Keep every group alike: each is a balanced sample from this same ordered mix.
# The IDs were reset-audited with minigrid 3.0.0 on Python 3.12.
SCENARIO_WEIGHTS = (
    ("MiniGrid-Empty-Random-6x6-v0", 1),
    ("MiniGrid-DoorKey-6x6-v0", 1),
    ("MiniGrid-MultiRoom-N2-S4-v0", 1),
    ("BabyAI-GoToRedBallGrey-v0", 1),
    ("BabyAI-GoToObj-v0", 1),
    ("BabyAI-PickupLoc-v0", 1),
)

SEEDS = {
    # Each large group has the same 8-seed difficulty mix: two Empty/door
    # starts and one each of MultiRoom, GoToRedBallGrey, GoToObj, PickupLoc.
    "train": [5, 1, 3, 6, 14, 2, 8, 7],
    "heldout": [9, 12, 10, 19, 15, 4, 13, 25],
    "dev": [33, 46],
    "validation": [16, 26, 27, 22, 17, 21, 20, 45],
}

_OBJECT_NAMES = {0: "unknown", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
_COLOR_NAMES = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
_STATE_NAMES = {0: "open", 1: "closed", 2: "locked"}
_DIRECTIONS = ("east", "south", "west", "north")


def _digest_int(domain: str, external_seed: int) -> int:
    """Stable SHA-256 domain-separated integer, independent of Python hash/RNG."""
    raw = hashlib.sha256(f"minigrid-adaptor-v1:{domain}:{int(external_seed)}".encode("ascii")).digest()
    return int.from_bytes(raw[:8], "big")


def _scenario_for_seed(seed: int) -> str:
    point = _digest_int("scenario", seed) % sum(w for _, w in SCENARIO_WEIGHTS)
    for env_id, weight in SCENARIO_WEIGHTS:
        if point < weight:
            return env_id
        point -= weight
    return SCENARIO_WEIGHTS[-1][0]


def _json_value(value: Any) -> Any:
    """Convert Gymnasium/NumPy values into strict JSON primitives."""
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items() if isinstance(k, (str, int, float))}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MiniGridGame:
    name = "MiniGrid / BabyAI"

    def __init__(self, env_id: str | None = None, environment_id: str | None = None,
                 id: str | None = None, env_kwargs: dict[str, Any] | None = None,
                 **kwargs: Any) -> None:
        self._requested_id = env_id or environment_id or id or os.environ.get("MINIGRID_ENV_ID")
        self._env_kwargs = dict(env_kwargs or {})
        # Accept a JSON-friendly `kwargs` mapping for evaluator integrations.
        if isinstance(kwargs.get("kwargs"), dict):
            self._env_kwargs.update(kwargs["kwargs"])
        self._env: Any | None = None
        self._env_id: str | None = None
        self._external_seed: int | None = None
        self._reset_seed: int | None = None
        self._obs: dict[str, Any] | None = None
        self._mission = ""
        self._steps = 0
        self._max_steps = 1
        self._last_reward = 0.0
        self._raw_return = 0.0
        self._terminated = False
        self._truncated = False
        self._success = False
        self._last_action: str | None = None
        self._history: list[dict[str, Any]] = []
        self._last_error: str | None = None
        self._closed = False
        self._runtime = Path(__file__).resolve().parent / "runtime"
        self._runtime.mkdir(parents=True, exist_ok=True)
        self.action_spec = ActionSpec(
            names=["left", "right", "forward", "pickup", "drop", "toggle", "done"],
            descriptions={
                "left": "turn 90 degrees counter-clockwise",
                "right": "turn 90 degrees clockwise",
                "forward": "move one cell forward if it is traversable",
                "pickup": "pick up the object directly in front of you",
                "drop": "drop the carried object in the cell directly in front of you",
                "toggle": "interact with the door or other toggleable object directly in front of you",
                "done": "declare that the mission is complete (the environment may ignore this)",
            },
            noop=None,
            notes="Actions are turn based: one step is consumed per call. Invalid or blocked actions are still environment steps. The view is a 7x7 (or task-configured) egocentric symbolic grid; x increases right and y increases toward the rear of the view.",
        )

    def _make_env(self, env_id: str) -> None:
        if env_id not in gym.registry:
            raise ValueError(f"unknown MiniGrid environment id: {env_id}")
        self._env = gym.make(env_id, **self._env_kwargs)
        self._env_id = env_id
        spec_steps = getattr(getattr(self._env, "spec", None), "max_episode_steps", None)
        core_steps = getattr(getattr(self._env, "unwrapped", self._env), "max_steps", None)
        self._max_steps = int(spec_steps or core_steps or 1)
        self._closed = False

    def reset(self, seed: int) -> dict[str, Any]:
        self.close()
        self._external_seed = int(seed)
        self._reset_seed = _digest_int("reset", self._external_seed) % (2**31 - 1)
        if self._reset_seed == 0:
            self._reset_seed = 1
        env_id = self._requested_id or _scenario_for_seed(self._external_seed)
        self._make_env(env_id)
        raw_obs, _info = self._env.reset(seed=self._reset_seed)
        # BabyAI computes its mission-dependent budget during reset (the
        # pre-reset value is zero), so resolve the effective budget again now.
        spec_steps = getattr(getattr(self._env, "spec", None), "max_episode_steps", None)
        core_steps = getattr(getattr(self._env, "unwrapped", self._env), "max_steps", None)
        self._max_steps = int(spec_steps or core_steps or 1)
        self._obs = raw_obs
        self._steps = 0
        self._last_reward = 0.0
        self._raw_return = 0.0
        self._terminated = False
        self._truncated = False
        self._success = False
        self._last_action = None
        self._history = []
        self._last_error = None
        self._mission = str(raw_obs.get("mission", "")) if isinstance(raw_obs, dict) else ""
        self._write_log({"event": "reset", "external_seed": self._external_seed, "reset_seed": self._reset_seed, "environment_id": self._env_id, "kwargs": self._env_kwargs, "minigrid_version": getattr(minigrid, "__version__", "unknown"), "gymnasium_version": getattr(gym, "__version__", "unknown")})
        return self.observe()

    def _write_log(self, record: dict[str, Any]) -> None:
        try:
            with (self._runtime / "episodes.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_json_value(record), allow_nan=False) + "\n")
        except OSError:
            pass

    def _decode_observation(self) -> dict[str, Any]:
        raw = self._obs or {}
        image = np.asarray(raw.get("image", []))
        grid: list[list[dict[str, Any]]] = []
        inventory: dict[str, str] | None = None
        if image.ndim == 3 and image.shape[-1] >= 3:
            width, height = int(image.shape[0]), int(image.shape[1])
            cx, cy = width // 2, height - 1
            carry_code = int(image[cx, cy, 0]) if 0 <= cx < width and 0 <= cy < height else 1
            if carry_code not in (0, 1, 3, 10):
                inventory = {"object": _OBJECT_NAMES.get(carry_code, f"object_{carry_code}"), "color": _COLOR_NAMES.get(int(image[cx, cy, 1]), f"color_{int(image[cx, cy, 1])}")}
            for y in range(height):
                row: list[dict[str, Any]] = []
                for x in range(width):
                    obj, color, state = (int(image[x, y, i]) for i in range(3))
                    # Positive relative y is forward, matching MiniGrid's egocentric
                    # image convention (the cell immediately ahead is (0, 1)).
                    cell: dict[str, Any] = {"x": x - cx, "y": (height - 1) - y, "object": _OBJECT_NAMES.get(obj, f"object_{obj}")}
                    if obj not in (0, 1, 2, 3, 10):
                        cell["color"] = _COLOR_NAMES.get(color, f"color_{color}")
                    if obj == 4:
                        cell["state"] = _STATE_NAMES.get(state, f"state_{state}")
                    row.append(cell)
                grid.append(row)
        direction = int(raw.get("direction", 0)) if isinstance(raw, dict) else 0
        result: dict[str, Any] = {
            "environment_id": self._env_id,
            "mission": self._mission,
            "direction": direction,
            "direction_name": _DIRECTIONS[direction % len(_DIRECTIONS)],
            "view_convention": "grid cells use x right, y forward; the agent is at (0,0), front is (0,1)",
            "grid": grid,
            "inventory": inventory,
            "steps_used": self._steps,
            "max_steps": self._max_steps,
            "remaining_steps": max(0, self._max_steps - self._steps),
            "last_action": self._last_action,
            "last_reward": float(self._last_reward),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
        }
        if self._history:
            result["recent_actions"] = list(self._history[-8:])
        if self._last_error:
            result["action_error"] = self._last_error
        return result

    def observe(self) -> dict[str, Any]:
        return self._decode_observation()

    def _action_value(self, action: Any) -> tuple[int | None, str | None]:
        names = list(self.action_spec.names)
        if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
            value = int(action)
            if 0 <= value < len(names):
                return value, names[value]
        if isinstance(action, str):
            token = action.strip().lower()
            if token in names:
                return names.index(token), token
            if token.isdigit() and 0 <= int(token) < len(names):
                value = int(token)
                return value, names[value]
        return None, None

    def step(self, action: Any) -> dict[str, Any]:
        if self._env is None:
            self._last_error = "step called before reset"
            return self.observe()
        if self.done:
            self._last_error = "episode already finished"
            return self.observe()
        value, name = self._action_value(action)
        if value is None or name is None:
            self._last_error = f"ignored malformed action {action!r}"
            return self.observe()
        self._last_error = None
        raw_obs, reward, terminated, truncated, info = self._env.step(value)
        self._obs = raw_obs
        self._steps += 1
        self._last_reward = float(reward)
        self._raw_return += float(reward)
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._last_action = name
        self._history.append({"action": name, "reward": float(reward), "step": self._steps})
        if isinstance(info, dict) and info.get("success") is True:
            self._success = True
        # MiniGrid's audited sparse tasks return positive reward only on success.
        # The official BabyAI verifier can emit its positive terminal reward in
        # the same transition that sets either termination flag.  Positive
        # reward is audited as success for the sparse scenario mix.
        if float(reward) > 0:
            self._success = True
        if self.done:
            self._write_log({"event": "finish", "external_seed": self._external_seed, "reset_seed": self._reset_seed, "environment_id": self._env_id, "steps_used": self._steps, "success": self._success, "raw_return": self._raw_return, "terminated": self._terminated, "truncated": self._truncated})
        return self.observe()

    @property
    def done(self) -> bool:
        return bool(self._terminated or self._truncated)

    @property
    def score(self) -> float:
        if not self._success:
            return 0.0
        remaining_fraction = max(0.0, min(1.0, (self._max_steps - self._steps) / self._max_steps))
        return float(0.1 + 0.9 * remaining_fraction)

    def score_breakdown(self) -> dict[str, float]:
        remaining = max(0, self._max_steps - self._steps)
        fraction = remaining / self._max_steps if self._max_steps else 0.0
        return {"success": float(self._success), "steps_used": float(self._steps), "max_steps": float(self._max_steps), "remaining_steps": float(remaining), "remaining_fraction": float(fraction), "score": float(self.score), "raw_return": float(self._raw_return), "terminated": float(self._terminated), "truncated": float(self._truncated)}

    def info(self) -> dict[str, Any]:
        return {"valid_trial": True, "external_seed": self._external_seed, "reset_seed": self._reset_seed, "environment_id": self._env_id, "kwargs": dict(self._env_kwargs), "success": bool(self._success), "steps_used": self._steps, "max_steps": self._max_steps, "remaining_steps": max(0, self._max_steps - self._steps), "remaining_fraction": max(0.0, min(1.0, (self._max_steps - self._steps) / self._max_steps)), "raw_reward": self._last_reward, "raw_return": self._raw_return, "terminated": self._terminated, "truncated": self._truncated, "minigrid_version": getattr(minigrid, "__version__", "unknown"), "gymnasium_version": getattr(gym, "__version__", "unknown")}

    def describe(self) -> str:
        return ("MiniGrid/BabyAI is a synchronous, turn-based gridworld. The mission is the natural-language objective. "
                "Each action consumes one step; the episode ends on task success, a hazard/failure, or the step limit. "
                "The observation gives the mission, facing direction, carried object, and the currently visible egocentric symbolic grid; unknown cells are occluded. "
                "Grid coordinates in the observation use x increasing right and y increasing forward, with the agent at (0,0) and the cell in front at (0,1). "
                "Turn left/right changes facing, forward moves one cell, pickup/drop manipulate the facing cell, toggle opens or interacts with it, and done declares completion. "
                "Blocked and otherwise invalid in-game actions still consume a step. Success is scored as 0.1 + 0.9 times the fraction of the step budget remaining; failed episodes score zero. "
                "The evaluator may provide an environment_id; otherwise a stable SHA-256 mapping selects an audited MiniGrid/BabyAI environment from the declared scenario mix.")

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
        self._closed = True


def make_game(**kwargs: Any) -> MiniGridGame:
    return MiniGridGame(**kwargs)


EVAL_KWARGS = {}
