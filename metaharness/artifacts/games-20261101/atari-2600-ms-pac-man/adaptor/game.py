"""Realtime Atari 2600 Ms. Pac-Man adaptor.

The emulator is deliberately driven by the evaluator's clock rather than by a
background thread.  One ``tick`` advances one fixed ALE action quantum, which
keeps the interface deterministic and makes the real-time boundary explicit to
the harness.
"""
from __future__ import annotations

import math
import numbers
import threading
from typing import Any

from metaharness.games.base import ActionSpec


# OCAtari/ALE's minimal action set is stable for this ROM.  These are action
# indices in the Gymnasium environment, not the larger ALE full action enum.
_ACTIONS: dict[str, int] = {
    "noop": 0,
    "up": 1,
    "right": 2,
    "left": 3,
    "down": 4,
    "up_right": 5,
    "up_left": 6,
    "down_right": 7,
    "down_left": 8,
}
_DIRECTIONS = ("up", "right", "down", "left")
_GHOST_COLOURS = ("orange", "cyan", "pink", "red")
_FRUIT_NAMES = ("cherry", "strawberry", "orange", "pretzel", "apple", "pear", "banana")


TIER = 2
DETERMINISTIC = True
# These are reproducible ALE seeds for the standard four-ghost stratum.  The
# emulator's opening position is intentionally fixed; seed affects ALE's RNG
# streams after reset rather than selecting a different maze.
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [100, 101, 102, 103],
}
# The main scored configuration.  Other ghost-count strata are available with
# make_game(ghost_mode=1), 2, or 3 and should be reported separately.
EVAL_KWARGS = {
    "ghost_mode": 0,
    "frameskip": 4,
    # ALE's documented v5 sticky-action stream is part of the seeded scenario;
    # keeping it fixed and nonzero makes train/held-out seeds meaningfully
    # different while reset(seed) remains reproducible.
    "repeat_action_probability": 0.25,
}


_ACTION_SPEC = ActionSpec(
    names=list(_ACTIONS),
    descriptions={
        "noop": "Keep the joystick neutral for this action quantum.",
        "up": "Hold the joystick up.",
        "right": "Hold the joystick right.",
        "left": "Hold the joystick left.",
        "down": "Hold the joystick down.",
        "up_right": "Hold up and right (ALE diagonal; cardinal movement is recommended).",
        "up_left": "Hold up and left (ALE diagonal; cardinal movement is recommended).",
        "down_right": "Hold down and right (ALE diagonal; cardinal movement is recommended).",
        "down_left": "Hold down and left (ALE diagonal; cardinal movement is recommended).",
    },
    noop="noop",
    notes=(
        "Each tick advances the fixed ALE frame skip (normally four Atari frames, "
        "about 67 ms at 60 Hz). The game continues between decisions; if no new "
        "action is posted, the previous action repeats. Ms. Pac-Man can reverse, "
        "but turns must be taken at legal maze corridors."
    ),
)


def _as_int(value: Any, default: int = 0) -> int:
    """Convert NumPy/Python scalar values without leaking them into JSON."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _pair(value: Any, default: tuple[int, int] = (0, 0)) -> list[int]:
    try:
        values = list(value)
        if len(values) >= 2:
            return [_as_int(values[0]), _as_int(values[1])]
    except (TypeError, ValueError):
        pass
    return [default[0], default[1]]


class MsPacmanGame:
    """A headless, symbolic, fixed-cadence Atari 2600 Ms. Pac-Man game."""

    name = "Atari 2600 Ms. Pac-Man"
    action_spec = _ACTION_SPEC

    def __init__(
        self,
        *,
        ghost_mode: int = 0,
        mode: int | None = None,
        frameskip: int = 4,
        repeat_action_probability: float = 0.25,
        **_ignored: Any,
    ) -> None:
        # ``mode`` is accepted as a convenient curriculum alias.  OCAtari's
        # own ``mode`` parameter means RAM/vision, so it is never passed there.
        if mode is not None:
            ghost_mode = mode
        if isinstance(ghost_mode, bool) or not isinstance(ghost_mode, numbers.Integral):
            raise ValueError("ghost_mode must be an integer from 0 through 3")
        self.ghost_mode = int(ghost_mode)
        if self.ghost_mode not in (0, 1, 2, 3):
            raise ValueError("ghost_mode must be 0 (four), 1 (one), 2 (two), or 3 (three ghosts)")
        if isinstance(frameskip, bool) or not isinstance(frameskip, numbers.Integral) or int(frameskip) <= 0:
            raise ValueError("frameskip must be a positive integer")
        self.frameskip = int(frameskip)
        probability = _as_float(repeat_action_probability, -1.0)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("repeat_action_probability must be between 0 and 1")
        self.repeat_action_probability = probability
        self.tick_s = self.frameskip / 60.0

        # Importing here keeps game.py loadable for tools that inspect the
        # declarations before installing the optional emulator dependencies.
        try:
            import gymnasium
            import ale_py
            gymnasium.register_envs(ale_py)
            from ocatari.core import OCAtari
        except Exception as exc:  # pragma: no cover - installation failure path
            raise RuntimeError(
                "Atari adaptor needs gymnasium, ale-py, and ocatari; install requirements.txt"
            ) from exc

        # OCAtari gives named dynamic objects from RAM.  No render mode or RGB
        # stack is requested, so this remains headless and compact.
        self._env = OCAtari(
            "ALE/MsPacman-v5",
            mode="ram",
            hud=True,
            obs_mode="obj",
            render_mode=None,
            frameskip=self.frameskip,
            repeat_action_probability=self.repeat_action_probability,
            create_buffer_stacks=[],
        )
        # OCAtari reserves its public ``mode`` keyword for extraction mode.
        # Its wrapped AtariEnv stores the actual ALE game variation here and
        # reloads it on every seeded reset.
        self._env._env.unwrapped._game_mode = self.ghost_mode
        self._ale = self._env._ale
        self._lock = threading.RLock()
        self._closed = False
        self._seed = 0
        self._ticks = 0
        self._last_action = 0
        self._last_action_name = "noop"
        self._last_reward = 0.0
        self._total_reward = 0.0
        self._hud_score = 0
        self._last_info: dict[str, Any] = {}
        self._done = False
        self._invalid_reason = ""
        self._last_action_error = ""
        self._last_lives = 0
        self._deaths = 0
        self._maze_number = 1
        self._mazes_cleared = 0
        self._initial_collectibles = 0
        self._previous_collectibles = 0
        self._pills_eaten = 0
        self._last_observation: dict[str, Any] = {}

    def reset(self, seed: int) -> dict[str, Any]:
        """Start a fresh seeded ALE episode and return its first observation."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot reset a closed game")
            self._env._env.unwrapped._game_mode = self.ghost_mode
            try:
                _raw_obs, info = self._env.reset(seed=int(seed))
            except Exception as exc:
                self._mark_invalid(f"ALE reset failed: {type(exc).__name__}: {exc}")
                # Reset errors are infrastructure errors; re-raise here because
                # there is no valid state from which to continue an episode.
                raise
            self._seed = int(seed)
            self._ticks = 0
            self._last_action = 0
            self._last_action_name = "noop"
            self._last_reward = 0.0
            self._total_reward = 0.0
            self._done = bool(self._ale.game_over())
            self._invalid_reason = ""
            self._last_action_error = ""
            self._last_info = dict(info) if isinstance(info, dict) else {}
            self._last_lives = self._lives()
            self._deaths = 0
            self._maze_number = 1
            self._mazes_cleared = 0
            self._pills_eaten = 0
            self._hud_score = self._read_hud_score()
            self._initial_collectibles = self._collectible_count()
            self._previous_collectibles = self._initial_collectibles
            self._last_observation = self._build_observation()
            return self._last_observation

    def observe(self) -> dict[str, Any]:
        with self._lock:
            if self._closed and self._last_observation:
                return self._last_observation
            self._last_observation = self._build_observation()
            return self._last_observation

    def tick(self, action: Any = None) -> None:
        """Post an action and advance exactly one fixed ALE action quantum.

        Malformed actions are deliberately non-fatal: the previous action is
        retained and a short diagnostic is exposed in the observation. This
        matches a real-time mailbox, where a late or malformed command must
        not stop the world.
        """
        with self._lock:
            if self._closed or self._done:
                return
            parsed = None if action is None else self._parse_action(action)
            if parsed is not None:
                self._last_action, self._last_action_name = parsed
            try:
                # OCAtari updates its object list after step().  Its 2.2.1
                # wrapper swaps terminated/truncated positions, so terminal
                # truth comes from ALE directly below.
                result = self._env.step(self._last_action)
                self._last_reward = _as_float(result[1] if len(result) > 1 else 0.0)
                self._total_reward += self._last_reward
                if len(result) > 4 and isinstance(result[4], dict):
                    self._last_info = dict(result[4])
                self._ticks += 1
                self._done = bool(self._ale.game_over())
                self._update_progress()
                self._last_observation = self._build_observation()
            except Exception as exc:  # never let a bad live tick crash a trial
                self._mark_invalid(f"ALE step failed: {type(exc).__name__}: {exc}")
                self._done = True
                self._last_observation = self._build_observation()

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done or (not self._closed and self._ale.game_over()))

    @property
    def score(self) -> float:
        # ALE's reward is the score delta for this ROM.  The HUD decoded score
        # is exposed separately and cross-checks the cumulative reward.
        with self._lock:
            return _as_float(self._total_reward)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def info(self) -> dict[str, Any]:
        with self._lock:
            ram = self._ram()
            return {
                "valid_trial": not bool(self._invalid_reason),
                "invalid_reason": self._invalid_reason,
                "seed": int(self._seed),
                "ghost_mode": int(self.ghost_mode),
                "ghosts_configured": int(self._configured_ghosts()),
                "frameskip": int(self.frameskip),
                "repeat_action_probability": float(self.repeat_action_probability),
                "episode_frame": self._episode_frame(),
                "ticks": int(self._ticks),
                "lives": int(self._lives()),
                "deaths": int(self._deaths),
                "mazes_cleared": int(self._mazes_cleared),
                "hud_score": int(self._hud_score),
                "ale_score": float(self._total_reward),
                "ram_mode_field": int(ram[19]) if len(ram) > 19 else 0,
                "terminal": bool(self._done),
            }

    def score_breakdown(self) -> dict[str, float]:
        """Numeric scoring components retained for evaluator reporting."""
        with self._lock:
            visible = self._collectible_count()
            return {
                "ale_score": float(self._total_reward),
                "hud_score": float(self._hud_score),
                "mazes_cleared": float(self._mazes_cleared),
                "visible_collectibles_eaten": float(self._pills_eaten),
                "visible_collectibles_remaining": float(visible),
                "lives_remaining": float(self._lives()),
                "deaths": float(self._deaths),
                "ghost_mode": float(self.ghost_mode),
                "terminal": 1.0 if self._done else 0.0,
            }

    def describe(self) -> str:
        return (
            "Ms. Pac-Man on the Atari 2600 CX2675 ROM runs continuously in a local "
            "headless ALE emulator. Your objective is to maximize score while eating "
            "pills and power pills, collecting fruit when safe, surviving, and clearing "
            "mazes. A normal pill is worth 10 points, a power pill 50, fruit and ghost "
            "chains add score, and losing all lives ends the episode. The opening maze "
            f"uses ALE ghost mode {self.ghost_mode} ({self._configured_ghosts()} configured "
            "ghosts); mode 0 is the main four-ghost evaluation and modes 1/2/3 are "
            "one-/two-/three-ghost curriculum strata. "
            "\n\n"
            "Actions are the nine names in action_spec: noop, up, right, left, down, "
            "up_right, up_left, down_right, and down_left. Each tick applies the latest "
            f"posted action and advances {self.frameskip} Atari frames (tick_s is "
            f"{self.tick_s:.6f} seconds at 60 Hz); before the first action, noop applies, "
            "and a missed decision repeats the last action. Cardinal actions are safest. "
            "The emulator does not pause while the model reasons, and the game may need "
            "its automatic opening sequence before Ms. Pac-Man begins moving. "
            "\n\n"
            "Observations are symbolic JSON: episode frame, cumulative ALE score and "
            "last reward, lives, terminal state, mode, player position/direction/velocity, "
            "four ghost slots, fruit and decoded fruit kind, power pills, remaining pills "
            "with row/column coordinates, HUD score, maze-progress counters, a raw 128-byte "
            "RAM snapshot, and a maze-map note. RAM does not encode wall geometry, so the "
            "maze map is explicitly marked unverified; use object coordinates and movement "
            "to infer legal corridors. Ghost RGB is their normal colour and is not a "
            "reliable frightened-state flag. The primary score is cumulative ALE reward; "
            "HUD score and survival/progress components are reported separately."
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._env.close()
            except Exception as exc:  # close must remain idempotent/non-blocking
                self._mark_invalid(f"ALE close failed: {type(exc).__name__}: {exc}")

    # ---- internal state/serialization helpers ---------------------------------

    def _mark_invalid(self, reason: str) -> None:
        if not self._invalid_reason:
            self._invalid_reason = str(reason)[:500]

    def _parse_action(self, action: Any) -> tuple[int, str] | None:
        if isinstance(action, str):
            key = action.strip().lower().replace("-", "_").replace(" ", "_")
            if key in _ACTIONS:
                self._last_action_error = ""
                return _ACTIONS[key], key
            self._last_action_error = f"unknown action {action!r}"[:300]
            return None
        if isinstance(action, numbers.Integral) and not isinstance(action, bool):
            index = int(action)
            if 0 <= index < len(_ACTIONS):
                self._last_action_error = ""
                name = list(_ACTIONS)[index]
                return index, name
        # NumPy integer scalars are Integral on supported NumPy versions; any
        # other object is intentionally ignored rather than raising.
        self._last_action_error = f"malformed action {action!r}"[:300]
        return None

    def _ram(self) -> list[int]:
        try:
            return [_as_int(x) & 255 for x in list(self._ale.getRAM())[:128]]
        except Exception as exc:
            self._mark_invalid(f"ALE RAM unavailable: {type(exc).__name__}: {exc}")
            return [0] * 128

    def _episode_frame(self) -> int:
        try:
            return _as_int(self._ale.getEpisodeFrameNumber())
        except Exception:
            return _as_int(self._last_info.get("episode_frame_number", self._ticks * self.frameskip))

    def _lives(self) -> int:
        try:
            return max(0, _as_int(self._ale.lives()))
        except Exception:
            return max(0, _as_int(self._last_info.get("lives", self._last_lives)))

    def _configured_ghosts(self) -> int:
        # ALE's Ms. Pac-Man mode numbering is not a simple count: mode 0 is
        # the standard four-ghost game, while modes 1/2/3 have one/two/three.
        return {0: 4, 1: 1, 2: 2, 3: 3}.get(self.ghost_mode, 0)

    def _all_objects(self) -> list[Any]:
        try:
            return list(self._env.objects)
        except Exception:
            return []

    def _category(self, obj: Any) -> str:
        return str(getattr(obj, "category", type(obj).__name__))

    def _visible(self, obj: Any) -> bool:
        try:
            return bool(obj)
        except Exception:
            return bool(getattr(obj, "_visible", False))

    def _rgb(self, obj: Any) -> list[int]:
        value = getattr(obj, "rgb", getattr(obj, "_rgb", (0, 0, 0)))
        result = _pair(value, (0, 0))
        try:
            values = list(value)
            if len(values) >= 3:
                return [_as_int(values[0]) & 255, _as_int(values[1]) & 255, _as_int(values[2]) & 255]
        except (TypeError, ValueError):
            pass
        return [result[0], result[1], 0]

    def _entity(self, obj: Any, category: str, slot: int) -> dict[str, Any]:
        xy = _pair(getattr(obj, "xy", (0, 0)))
        wh = _pair(getattr(obj, "wh", (0, 0)))
        return {
            "slot": int(slot),
            "category": category,
            "visible": self._visible(obj),
            "xy": xy,
            "x": int(xy[0]),
            "y": int(xy[1]),
            "wh": wh,
            "dx": _as_int(getattr(obj, "dx", 0)),
            "dy": _as_int(getattr(obj, "dy", 0)),
            "rgb": self._rgb(obj),
        }

    def _read_hud_score(self) -> int:
        for obj in self._all_objects():
            if self._category(obj) == "Score":
                return max(0, _as_int(getattr(obj, "value", getattr(obj, "_value", 0))))
        ram = self._ram()
        if len(ram) >= 123:
            return self._score_from_ram(ram)
        return 0

    @staticmethod
    def _score_from_ram(ram: list[int]) -> int:
        def dec(number: int) -> int:
            high = number >> 4
            low = number & 15
            return (min(high, 9) * 10) + min(low, 9)

        return dec(ram[122]) * 10000 + dec(ram[121]) * 100 + dec(ram[120])

    def _collectible_count(self) -> int:
        objects = self._all_objects()
        return sum(
            1
            for obj in objects
            if self._category(obj) in ("Pill", "PowerPill") and self._visible(obj)
        )

    def _update_progress(self) -> None:
        lives = self._lives()
        if self._last_lives and lives < self._last_lives:
            self._deaths += self._last_lives - lives
        self._last_lives = lives
        current = self._collectible_count()
        previous = self._previous_collectibles
        if current <= previous:
            self._pills_eaten += previous - current
        elif previous > 0 and current > previous + 20 and previous < max(20, self._initial_collectibles // 2):
            # A cleared maze repopulates the pill grid.  This is intentionally
            # conservative so temporary decoder changes cannot fake a clear.
            self._mazes_cleared += 1
            self._maze_number += 1
            self._initial_collectibles = current
        self._previous_collectibles = current
        self._hud_score = self._read_hud_score()

    def _fruit_kind(self, ram: list[int]) -> str:
        value = ram[123] if len(ram) > 123 else 255
        for index, upper in enumerate((16, 32, 48, 64, 80, 96, 112)):
            if value < upper:
                return _FRUIT_NAMES[index]
        return "unknown"

    def _build_observation(self, error: str = "") -> dict[str, Any]:
        ram = self._ram()
        objects = self._all_objects()
        # The RAM decoder's stable first ten slots are player, four ghosts,
        # fruit, and four power pills.  The next 252 slots are the pill grid.
        if len(objects) >= 10:
            player_obj = objects[0]
            ghost_objs = objects[1:5]
            fruit_obj = objects[5]
            power_objs = objects[6:10]
            pill_objs = objects[10:262]
        else:
            # This only occurs after an emulator failure; keep JSON shape valid.
            from types import SimpleNamespace
            empty = SimpleNamespace(xy=(0, 0), wh=(0, 0), dx=0, dy=0, rgb=(0, 0, 0), category="NoObject")
            player_obj = empty
            ghost_objs = [empty] * 4
            fruit_obj = empty
            power_objs = [empty] * 4
            pill_objs = []

        direction = _DIRECTIONS[(ram[56] & 3) if len(ram) > 56 else 0]
        player = self._entity(player_obj, "Player", 0)
        player["direction"] = direction

        ghosts = []
        for slot, obj in enumerate(ghost_objs):
            entity = self._entity(obj, "Ghost", slot)
            entity["colour"] = _GHOST_COLOURS[slot]
            ghosts.append(entity)

        fruit = self._entity(fruit_obj, "Fruit", 0)
        fruit["kind"] = self._fruit_kind(ram)
        power_pills = [self._entity(obj, "PowerPill", slot) for slot, obj in enumerate(power_objs)]

        pills: list[dict[str, Any]] = []
        for obj in pill_objs:
            if not self._visible(obj):
                continue
            grid = getattr(obj, "grid_ij", (0, 0))
            row_col = _pair(grid)
            xy = _pair(getattr(obj, "xy", (0, 0)))
            pills.append({
                "row": int(row_col[0]),
                "col": int(row_col[1]),
                "x": int(xy[0]),
                "y": int(xy[1]),
                "visible": True,
            })

        score = float(self._total_reward)
        hud_score = int(self._hud_score)
        visible_collectibles = len(pills) + sum(1 for obj in power_objs if self._visible(obj))
        actual_mode_field = int(ram[19]) if len(ram) > 19 else self.ghost_mode - 1
        observation: dict[str, Any] = {
            "frame": int(self._episode_frame()),
            "episode_frame": int(self._episode_frame()),
            "score": score,
            "hud_score": hud_score,
            "reward_delta": float(self._last_reward),
            "lives": int(self._lives()),
            "lives_remaining": int(self._lives()),
            "deaths": int(self._deaths),
            "game_over": bool(self._done),
            "mode": int(self.ghost_mode),
            "ghost_count_configured": int(self._configured_ghosts()),
            "ghost_count_ram": int(actual_mode_field + 1),
            "ticks": int(self._ticks),
            "last_action": self._last_action_name,
            "player": player,
            "ghosts": ghosts,
            "fruit": fruit,
            "power_pills": power_pills,
            "pills": pills,
            "visible_collectibles": int(visible_collectibles),
            "pills_eaten": int(self._pills_eaten),
            "mazes_cleared": int(self._mazes_cleared),
            "maze_number": int(self._maze_number),
            "maze": {
                "map_id": "cx2675_pill_grid_1" if (ram and ram[0] == 0) else "cx2675_pill_grid_2",
                "known_walls_or_turn_graph": "not encoded in Atari RAM; infer or calibrate from RGB",
                "map_confidence": 0.0,
            },
            "raw_ram": ram,
        }
        if error or self._last_action_error:
            observation["action_error"] = str(error or self._last_action_error)[:300]
        elif self._invalid_reason:
            observation["infrastructure_error"] = self._invalid_reason
        return observation


def make_game(**kwargs: Any) -> MsPacmanGame:
    """Construct a fresh Atari 2600 Ms. Pac-Man real-time game."""
    return MsPacmanGame(**kwargs)
