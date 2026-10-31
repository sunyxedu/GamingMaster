"""Meta-Harness adaptor for the local Crafter benchmark.

The upstream release used by this adaptor (1.8.3) is a synchronous, turn-based
environment.  A fresh ``crafter.Env`` is created for every reset so that the
seed controls the generated world and a fresh instance is deterministic.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Crafter imports pygame in some releases.  This keeps the adaptor usable on a
# server without an X display while still allowing a user to override it.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import crafter
from crafter import constants

from metaharness.games.base import ActionSpec


TIER = 1
DETERMINISTIC = True

# These are procedural world seeds, grouped for search and evaluation.  Every
# group uses the same simple integer-seed distribution and is disjoint.
SEEDS = {
    "train": [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108],
    "heldout": [2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208],
    "dev": [3301, 3302],
    "validation": [4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408],
}


_FALLBACK_ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down", "do",
    "sleep", "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]

_ACTION_DESCRIPTIONS = {
    "noop": "Wait for one simulation tick.",
    "move_left": "Move one tile west and face west.",
    "move_right": "Move one tile east and face east.",
    "move_up": "Move one tile north and face north.",
    "move_down": "Move one tile south and face south.",
    "do": "Interact with the tile or creature directly in front (gather, drink, eat, or attack).",
    "sleep": "Start or continue sleeping to restore energy; sleeping ends when rested or hurt.",
    "place_stone": "Place one stone block in the facing tile when legal.",
    "place_table": "Place a crafting table in the facing tile when legal.",
    "place_furnace": "Place a furnace in the facing tile when legal.",
    "place_plant": "Plant a sapling in the facing grass tile when legal.",
    "make_wood_pickaxe": "Craft a wood pickaxe beside a table.",
    "make_stone_pickaxe": "Craft a stone pickaxe beside a table.",
    "make_iron_pickaxe": "Craft an iron pickaxe beside a table and furnace.",
    "make_wood_sword": "Craft a wood sword beside a table.",
    "make_stone_sword": "Craft a stone sword beside a table.",
    "make_iron_sword": "Craft an iron sword beside a table and furnace.",
}


def _plain(value: Any) -> Any:
    """Convert NumPy/scalar containers from Crafter into strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class CrafterGame:
    """TurnGame wrapper around the official Crafter Python environment."""

    name = "Crafter"

    def __init__(self, **kwargs: Any):
        self._env_kwargs = dict(kwargs)
        self._env: Any | None = None
        self._seed: int | None = None
        self._steps = 0
        self._done = False
        self._last_reward = 0.0
        self._last_action = "noop"
        self._last_info: dict[str, Any] = {}
        self._invalid_action: str | None = None
        self._obs_cache: dict[str, Any] | None = None

        # constants.actions is the package's authoritative action order.  A
        # fallback keeps import-time failures clear for compatible old releases.
        names = list(getattr(constants, "actions", _FALLBACK_ACTIONS))
        self.action_spec = ActionSpec(
            names=names,
            descriptions={name: _ACTION_DESCRIPTIONS.get(name, name) for name in names},
            noop="noop" if "noop" in names else names[0],
            notes=(
                "Actions are bare names from this list. Each step advances exactly one game tick. "
                "The local_map grid is centered on the player; coordinates are [x, y]. "
                "Collect and craft achievements are cumulative counters, while score is the fraction "
                "of the standard achievement set unlocked in this episode."
            ),
        )

    def reset(self, seed: int) -> dict[str, Any]:
        """Start a fresh procedurally generated episode for ``seed``."""
        self.close()
        self._seed = int(seed)
        # Crafter 1.8.x seeds in its constructor and hashes (seed, episode) in
        # reset().  A new Env with episode zero therefore gives deterministic
        # first observations for each integer seed.
        self._env = crafter.Env(seed=self._seed, **self._env_kwargs)
        self._env.reset()
        self._steps = 0
        self._done = False
        self._last_reward = 0.0
        self._last_action = "noop"
        self._last_info = {}
        self._invalid_action = None
        self._obs_cache = self._serialize()
        return self._obs_cache

    def _action_id(self, action: Any) -> tuple[int, str, str | None]:
        names = list(self.action_spec.names)
        if isinstance(action, str):
            if action in names:
                return names.index(action), action, None
            return names.index(self.action_spec.noop or names[0]), self.action_spec.noop or names[0], (
                f"unknown action name {action!r}"
            )
        if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
            number = int(action)
            if 0 <= number < len(names):
                return number, names[number], None
            return names.index(self.action_spec.noop or names[0]), self.action_spec.noop or names[0], (
                f"action id {number} is outside 0..{len(names) - 1}"
            )
        return names.index(self.action_spec.noop or names[0]), self.action_spec.noop or names[0], (
            f"action must be a Crafter name or integer id, got {type(action).__name__}"
        )

    def step(self, action: Any) -> dict[str, Any]:
        """Apply one action synchronously and return the next semantic observation."""
        if self._env is None:
            self.reset(0)
        if self._done:
            # A post-terminal action is harmless and keeps malformed calls from
            # turning infrastructure validation into a game error.
            self._invalid_action = "step called after the episode ended"
            self._obs_cache = self._serialize()
            return self._obs_cache

        action_id, action_name, invalid = self._action_id(action)
        self._invalid_action = invalid
        result = self._env.step(action_id)
        if len(result) == 4:
            _, reward, done, info = result
        elif len(result) == 5:
            _, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"Crafter step returned {len(result)} values, expected 4 or 5")
        self._steps += 1
        self._done = bool(done)
        self._last_reward = float(reward)
        self._last_action = action_name
        self._last_info = info if isinstance(info, dict) else {}
        self._obs_cache = self._serialize()
        return self._obs_cache

    def _object_name(self, obj: Any) -> str | None:
        if obj is None:
            return None
        name = type(obj).__name__.lower()
        if hasattr(obj, "ripe"):
            name += "_ripe" if bool(obj.ripe) else "_growing"
        return name

    def _serialize(self) -> dict[str, Any]:
        env = self._env
        if env is None or getattr(env, "_player", None) is None:
            return {
                "game": self.name,
                "seed": self._seed,
                "status": "not_reset",
                "step": self._steps,
                "done": self._done,
                "score": 0.0,
            }
        player = env._player
        world = env._world
        pos = np.asarray(player.pos, dtype=int)
        radius = 4
        rows: list[list[dict[str, Any]]] = []
        for y in range(int(pos[1]) - radius, int(pos[1]) + radius + 1):
            row: list[dict[str, Any]] = []
            for x in range(int(pos[0]) - radius, int(pos[0]) + radius + 1):
                material, obj = world[(x, y)]
                cell: dict[str, Any] = {"x": x, "y": y, "material": material or "outside"}
                object_name = self._object_name(obj)
                if object_name:
                    cell["object"] = object_name
                    if hasattr(obj, "health"):
                        cell["health"] = int(obj.health)
                row.append(cell)
            rows.append(row)
        inventory = _plain(getattr(player, "inventory", {}))
        achievements = _plain(getattr(player, "achievements", {}))
        unlocked = sorted(name for name, count in achievements.items() if count > 0)
        achievement_count = max(1, len(achievements))
        score = len(unlocked) / achievement_count
        return {
            "game": self.name,
            "seed": self._seed,
            "step": self._steps,
            "done": self._done,
            "score": score,
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "player": {
                "position": [int(pos[0]), int(pos[1])],
                "facing": _plain(getattr(player, "facing", (0, 1))),
                "health": int(player.health),
                "sleeping": bool(getattr(player, "sleeping", False)),
                "daylight": float(world.daylight),
            },
            "inventory": inventory,
            "achievements": achievements,
            "unlocked": unlocked,
            "local_map": {"radius": radius, "rows": rows},
            **({"invalid_action": self._invalid_action} if self._invalid_action else {}),
        }

    def observe(self) -> dict[str, Any]:
        return self._obs_cache if self._obs_cache is not None else self._serialize()

    @property
    def done(self) -> bool:
        return bool(self._done)

    @property
    def score(self) -> float:
        obs = self.observe()
        value = obs.get("score", 0.0) if isinstance(obs, dict) else 0.0
        return float(value)

    def score_breakdown(self) -> dict[str, float]:
        obs = self.observe()
        achievements = obs.get("achievements", {}) if isinstance(obs, dict) else {}
        unlocked = sum(1 for value in achievements.values() if value > 0)
        total = max(1, len(achievements))
        breakdown = {
            "achievements_unlocked": float(unlocked),
            "achievement_fraction": float(unlocked / total),
            "achievement_total": float(total),
        }
        # Keep each named binary success visible to episode aggregators.  The
        # benchmark's geometric mean is computed across held-out episodes from
        # these per-achievement components rather than from shaped reward.
        breakdown.update({str(name): float(value > 0) for name, value in achievements.items()})
        return breakdown

    def info(self) -> dict[str, Any]:
        return {
            "valid_trial": True,
            "seed": self._seed,
            "steps": self._steps,
            "reward_last_step": self._last_reward,
            "crafter_info": _plain(self._last_info),
        }

    def describe(self) -> str:
        return (
            "Crafter is a synchronous open-world survival episode. The world advances one tick per "
            "step, and ends when the player dies or the configured episode length is reached. Gather "
            "wood, stone, coal, iron and diamond; drink water, manage food/energy, place blocks and "
            "plants, craft tools and swords near the required table/furnace, and defeat zombies and "
            "skeletons. The observation is JSON with player position/facing/vitals, inventory, "
            "achievement counters, and a 9x9 semantic local_map centered on the player. Coordinates "
            "are [x,y], and do interacts with the tile or creature one step in front. Score is the "
            "fraction of the 22 standard achievement names unlocked; held-out episode aggregation "
            "can compute the benchmark per-achievement rates and their geometric mean."
        )

    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            closer = getattr(env, "close", None)
            if callable(closer):
                closer()


def make_game(**kwargs: Any) -> CrafterGame:
    """Construct a fresh adaptor instance."""
    return CrafterGame(**kwargs)


EVAL_KWARGS: dict[str, Any] = {}
