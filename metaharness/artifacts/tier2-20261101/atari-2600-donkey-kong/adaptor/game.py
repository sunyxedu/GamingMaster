"""Headless Atari 2600 Donkey Kong adaptor.

The emulator is ALE/Stella (through Gymnasium) and OCAtari is used only for
its RAM object decoder.  The public game object deliberately exposes JSON
state rather than NumPy arrays or OCAtari objects.
"""
from __future__ import annotations

import hashlib
import threading
from numbers import Integral
from typing import Any

import ale_py
import gymnasium as gym
from metaharness.games.base import ActionSpec
from ocatari.core import OCAtari


TIER = 2
# A reset is deterministic at the opening state.  With the default v5 sticky
# action probability, the subsequent action trace is intentionally stochastic.
DETERMINISTIC = True

# All integer seeds are valid for this single fixed cartridge scenario.  The
# held-out group is separate so the evaluator can use fresh ALE RNG streams.
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [100, 101, 102, 103, 104, 105, 106, 107],
}

# Standard ALE v5 stochasticity is the evaluation default.  Passing
# repeat_action_probability=0.0 to make_game is useful for deterministic
# calibration and clone/restore experiments.
EVAL_KWARGS = {
    "frameskip": 4,
    "repeat_action_probability": 0.25,
    "difficulty": 0,
    "full_action_space": True,
}

_EXPECTED_ROM_MD5 = "36b20c427975760cb9cf4a47e41369e4"
_NTSC_FPS = 60.0

# These are the ALE action indices, in the exact order exposed by the
# DonkeyKong action set.  Names are intentionally short, stable strings for a
# model controller; numeric indices are accepted as a convenience too.
_ACTIONS: tuple[tuple[str, int], ...] = (
    ("noop", 0),
    ("fire", 1),
    ("up", 2),
    ("right", 3),
    ("left", 4),
    ("down", 5),
    ("upright", 6),
    ("upleft", 7),
    ("downright", 8),
    ("downleft", 9),
    ("upfire", 10),
    ("rightfire", 11),
    ("leftfire", 12),
    ("downfire", 13),
    ("uprightfire", 14),
    ("upleftfire", 15),
    ("downrightfire", 16),
    ("downleftfire", 17),
)
_ACTION_NAME_TO_INDEX = dict(_ACTIONS)
_ACTION_INDEX_TO_NAME = {index: name for name, index in _ACTIONS}

_ACTION_DESCRIPTIONS = {
    "noop": "Hold the joystick neutral; does not start the attract screen.",
    "fire": "Press FIRE; starts the cartridge and jumps in play.",
    "up": "Climb a ladder when Mario is aligned with one.",
    "right": "Walk right on the current girder.",
    "left": "Walk left on the current girder.",
    "down": "Climb down a ladder when aligned.",
    "upright": "Move up and right; climb or jump depending on alignment.",
    "upleft": "Move up and left; climb or jump depending on alignment.",
    "downright": "Move down and right; climb or walk depending on alignment.",
    "downleft": "Move down and left; climb or walk depending on alignment.",
    "upfire": "Jump while steering upward/right as accepted by the cartridge.",
    "rightfire": "Jump/steer right with FIRE.",
    "leftfire": "Jump/steer left with FIRE.",
    "downfire": "Jump/steer down with FIRE.",
    "uprightfire": "Jump while steering up and right.",
    "upleftfire": "Jump while steering up and left.",
    "downrightfire": "Jump while steering down and right.",
    "downleftfire": "Jump while steering down and left.",
}

_LADDERS = (
    (108, 176, 4, 17),
    (48, 148, 4, 17),
    (80, 148, 4, 17),
    (88, 120, 4, 17),
    (108, 120, 4, 17),
    (48, 92, 4, 17),
    (68, 92, 4, 17),
    (108, 64, 4, 17),
    (76, 40, 4, 17),
    (32, 40, 4, 17),
)


class DonkeyKongGame:
    """A RealtimeGame backed by one local ALE/Stella episode.

    No worker thread is needed: the evaluator owns the clock and each call to
    :meth:`tick` advances exactly one ALE/Gymnasium frameskip quantum.
    """

    name = "Atari 2600 Donkey Kong"

    def __init__(
        self,
        *,
        frameskip: int = 4,
        repeat_action_probability: float = 0.25,
        difficulty: int = 0,
        full_action_space: bool = True,
        max_num_frames_per_episode: int | None = 108000,
        **kwargs: Any,
    ) -> None:
        # OCAtari uses ``mode`` for its extraction mode, so Atari's mode is
        # fixed at the cartridge's only supported mode (0) and is consumed
        # here if a caller supplies it.
        atari_mode = kwargs.pop("mode", 0)
        if atari_mode not in (0, None):
            raise ValueError("Donkey Kong only supports Atari mode 0")
        obs_type = kwargs.pop("obs_type", "ram")
        if obs_type != "ram":
            raise ValueError("This adaptor requires obs_type='ram'")
        render_mode = kwargs.pop("render_mode", None)
        if render_mode is not None:
            raise ValueError("This headless adaptor requires render_mode=None")
        # Do not allow callers to replace OCAtari's own extraction settings.
        kwargs.pop("hud", None)
        kwargs.pop("obs_mode", None)
        kwargs.pop("create_buffer_stacks", None)
        continuous = kwargs.pop("continuous", False)
        if continuous:
            raise ValueError("This adaptor exposes the discrete ALE action set")

        if isinstance(frameskip, bool) or not isinstance(frameskip, Integral) or int(frameskip) <= 0:
            raise ValueError("frameskip must be a positive integer")
        frameskip = int(frameskip)
        probability = float(repeat_action_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("repeat_action_probability must be between 0 and 1")
        if isinstance(difficulty, bool) or not isinstance(difficulty, Integral) or int(difficulty) != 0:
            raise ValueError("Donkey Kong only supports difficulty 0")
        difficulty = int(difficulty)

        self._lock = threading.RLock()
        self._closed = False
        self._has_reset = False
        self._seed: int | None = None
        self._frameskip = frameskip
        self._repeat_action_probability = probability
        self._valid_trial = True
        self._invalid_reason: str | None = None

        self._ticks = 0
        self._frame = 0
        self._score = 0.0
        self._last_reward = 0.0
        self._last_action = "noop"
        self._last_action_index = 0
        self._posted_action: tuple[str, int] | None = None
        self._last_error: str | None = None
        self._terminated = False
        self._truncated = False
        self._started = False
        self._screens_completed = 0
        self._ram: tuple[int, ...] = (0,) * 128
        self._previous_ram: tuple[int, ...] | None = None
        self._last_env_info: dict[str, Any] = {}

        # Verify the exact maintained ROM before constructing Stella.  ALE's
        # lookup also validates the hash and honors ALE_ROMS_DIR when set.
        rom_path = ale_py.roms.get_rom_path("donkey_kong")
        if rom_path is None or not rom_path.is_file():
            raise RuntimeError("ALE could not find the donkey_kong ROM")
        with rom_path.open("rb") as rom_file:
            rom_md5 = hashlib.md5(rom_file.read()).hexdigest()
        if rom_md5 != _EXPECTED_ROM_MD5:
            raise RuntimeError(
                f"unexpected Donkey Kong ROM MD5 {rom_md5}; expected {_EXPECTED_ROM_MD5}"
            )
        self._rom_path = str(rom_path)
        self._rom_md5 = rom_md5

        # Register ALE's Gymnasium environments before OCAtari asks Gymnasium
        # to create ALE/DonkeyKong-v5.  No display or screenshot buffer is made.
        gym.register_envs(ale_py)
        ocatari_kwargs: dict[str, Any] = {
            "mode": "ram",
            "hud": True,
            "obs_mode": "obj",
            "render_mode": None,
            "create_buffer_stacks": [],
            "obs_type": "ram",
            "frameskip": frameskip,
            "repeat_action_probability": probability,
            "full_action_space": bool(full_action_space),
            "difficulty": difficulty,
        }
        if max_num_frames_per_episode is not None:
            if (
                isinstance(max_num_frames_per_episode, bool)
                or not isinstance(max_num_frames_per_episode, Integral)
                or int(max_num_frames_per_episode) <= 0
            ):
                raise ValueError("max_num_frames_per_episode must be positive or None")
            ocatari_kwargs["max_num_frames_per_episode"] = int(max_num_frames_per_episode)
        # There should be no remaining arguments in normal use.  Passing
        # unknown ALE kwargs is useful for compatible local ALE builds.
        ocatari_kwargs.update(kwargs)
        self._env = OCAtari("ALE/DonkeyKong-v5", **ocatari_kwargs)

        if getattr(self._env.action_space, "n", None) != len(_ACTIONS):
            self.close()
            raise RuntimeError(
                f"ALE exposed {getattr(self._env.action_space, 'n', None)} actions, "
                f"expected {len(_ACTIONS)}"
            )

        self.action_spec = ActionSpec(
            names=[name for name, _ in _ACTIONS],
            descriptions=dict(_ACTION_DESCRIPTIONS),
            noop="noop",
            notes=(
                "Each action is held for one fixed four-frame ALE quantum by default "
                "(about 66.7 ms at NTSC speed). FIRE is needed to leave the initial "
                "attract/idle state. UP/DOWN only climb when aligned with a usable "
                "ladder; direction+FIRE jumps. The default v5 sticky-action probability "
                "is 0.25, so the emulator may repeat the previous joystick action."
            ),
        )

    @property
    def tick_s(self) -> float:
        """The fixed ALE frameskip quantum at the NTSC cartridge cadence."""
        return self._frameskip / _NTSC_FPS

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._terminated or self._truncated)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._score)

    def describe(self) -> str:
        return (
            "Atari 2600 Donkey Kong is a local NTSC cartridge emulated by ALE/Stella. "
            "The objective is to guide Mario up the barrel/ramp screen, then remove "
            "all rivets on the rivet screen and reach the goal while preserving lives. "
            "The cartridge awards points for jumps, rivets, hammer smashes, and the "
            "screen bonus; score is the cumulative unclipped ALE reward, not the HUD "
            "bonus byte. A fresh reset always opens the same mode-0, difficulty-0 "
            "cartridge; the seed selects ALE's RNG stream and does not select a level. "
            "With the default standard-v5 sticky-action probability (0.25), an action "
            "can be repeated by the emulator; set repeat_action_probability=0.0 when "
            "calibrating deterministic traces.\n\n"
            "Call tick(action) once per real-time quantum. Actions are the literal names "
            "in action_spec (noop, fire, up, right, left, down, upright, upleft, "
            "downright, downleft, upfire, rightfire, leftfire, downfire, uprightfire, "
            "upleftfire, downrightfire, downleftfire); integer ALE indices 0-17 are "
            "also accepted. Passing None means no fresh decision and repeats the last "
            "posted action (or noop before the first post). A malformed action is treated "
            "as noop and reported in the next observation instead of raising.\n\n"
            "The JSON observation contains frame/tick timing, stage, Mario's bounding "
            "box and motion, visible barrel/fireball hazards with slots and motion, "
            "the ten fixed ladder boxes, hammer, four rivet-row status bytes, a derived "
            "rivets_removed count, BCD score display and bonus, cumulative score, lives, "
            "screens_completed, rewards, and terminal flags. raw_ram is the complete "
            "128-byte RIOT RAM snapshot; the game-specific fields use the verified NTSC "
            "Donkey Kong RAM map (player at 0x93/0x9B, screen at 0x90, rivets at "
            "0x9F-0xA2, lives at 0xA3, bonus at 0xA4)."
        )

    def reset(self, seed: int) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot reset a closed Donkey Kong game")
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer")
            seed = int(seed)
            # Reset the episode bookkeeping before asking the actual engine to
            # reset, so a failed reset cannot accidentally retain a prior score.
            self._seed = seed
            self._ticks = 0
            self._frame = 0
            self._score = 0.0
            self._last_reward = 0.0
            self._last_action = "noop"
            self._last_action_index = 0
            self._posted_action = None
            self._last_error = None
            self._terminated = False
            self._truncated = False
            self._started = False
            self._screens_completed = 0
            self._previous_ram = None
            self._valid_trial = True
            self._invalid_reason = None
            try:
                _, info = self._env.reset(seed=seed)
                self._last_env_info = self._safe_info(info)
                self._frame = self._info_int(info, "episode_frame_number", 0)
                self._has_reset = True
                self._refresh_ram()
            except Exception as exc:  # infrastructure failure, not bad play
                self._has_reset = True
                self._mark_invalid(f"reset failed: {type(exc).__name__}: {exc}")
            return self.observe()

    def observe(self) -> dict[str, Any]:
        with self._lock:
            ram = self._ram
            stage = self._stage(ram)
            player_obj = self._object_by_category("Player")
            player = self._spatial_object(
                player_obj,
                fallback=(int(ram[19]) - 4, int(ram[27]) + 22, 8, 17),
            )

            hazards: list[dict[str, Any]] = []
            objects = list(getattr(self._env, "objects", ()) or ())
            for slot in range(4):
                obj = objects[4 + slot] if len(objects) > 4 + slot else None
                present = self._object_present(obj)
                if not present and int(ram[slot]) != 0:
                    x = int(ram[65 + slot]) - 4
                    y = int(ram[slot]) + 40
                    hazards.append(
                        {
                            "kind": "fireball" if stage == "rivet" else "barrel",
                            "slot": slot,
                            "x": x,
                            "y": y,
                            "w": 8,
                            "h": 8,
                            "dx": 0,
                            "dy": 0,
                            "visible": True,
                        }
                    )
                elif present:
                    item = self._spatial_object(obj)
                    item.update(
                        {
                            "kind": "fireball" if stage == "rivet" else "barrel",
                            "slot": slot,
                        }
                    )
                    hazards.append(item)

            ladders: list[dict[str, Any]] = []
            for slot, fallback in enumerate(_LADDERS):
                obj = objects[8 + slot] if len(objects) > 8 + slot else None
                if self._object_present(obj):
                    ladder = self._spatial_object(obj, fallback=fallback)
                else:
                    ladder = {
                        "x": fallback[0],
                        "y": fallback[1],
                        "w": fallback[2],
                        "h": fallback[3],
                        "dx": 0,
                        "dy": 0,
                        "visible": False,
                    }
                ladder["slot"] = slot
                ladders.append(ladder)

            hammer_obj = self._object_by_category("Hammer")
            if self._object_present(hammer_obj):
                hammer = self._spatial_object(hammer_obj)
                hammer = {
                    "present": True,
                    "x": hammer["x"],
                    "y": hammer["y"],
                    "w": hammer["w"],
                    "h": hammer["h"],
                    "visible": True,
                }
            else:
                hammer = {
                    "present": False,
                    "x": 0,
                    "y": 0,
                    "w": 0,
                    "h": 0,
                    "visible": False,
                }

            rivet_rows = [int(v) for v in ram[31:35]]
            bonus_bcd = int(ram[36])
            observation: dict[str, Any] = {
                "frame": int(self._frame),
                "tick": int(self._ticks),
                "stage": stage,
                "screen_index": int(ram[16]),
                "game_state": int(ram[12]),
                "player": player,
                "hazards": hazards,
                "ladders": ladders,
                "hammer": hammer,
                "rivet_rows": rivet_rows,
                "rivets_removed": int(self._rivets_removed(rivet_rows)),
                "bonus_display_bcd": bonus_bcd,
                "current_bonus": int(self._bcd_to_int(bonus_bcd) * 100),
                "score_display_bcd": [int(ram[7]), int(ram[8])],
                "score_total": float(self._score),
                "episode_return": float(self._score),
                "last_reward": float(self._last_reward),
                "screens_completed": int(self._screens_completed),
                "lives_remaining": int(ram[35]),
                "life_lost_flag": bool(ram[13]),
                "last_action": str(self._last_action),
                "last_action_index": int(self._last_action_index),
                "last_error": self._last_error,
                "terminated": bool(self._terminated),
                "truncated": bool(self._truncated),
                "termination_reason": self._termination_reason(),
                "raw_ram": [int(v) for v in ram],
            }
            return observation

    def tick(self, action: Any) -> None:
        """Apply one action quantum and return immediately.

        The runner supplies the wall-clock cadence.  No sleeping happens here,
        which leaves the runner free to call this method on its own clock.
        """
        with self._lock:
            if self._closed:
                self._last_error = "tick called after close()"
                return
            if not self._has_reset:
                # The normal contract resets before play; this fallback keeps a
                # late/misordered controller call harmless and observable.
                self.reset(0)
            if action is None and self._posted_action is not None:
                # ``None`` represents a missed controller decision: the
                # RealtimeGame contract repeats the last posted action.
                action_name, action_index = self._posted_action
                error = None
            elif action is None:
                action_name, action_index, error = "noop", 0, None
            else:
                action_name, action_index, error = self._coerce_action(action)
                if error is None:
                    self._posted_action = (action_name, action_index)
            self._last_action = action_name
            self._last_action_index = action_index
            self._last_error = error
            if self.done:
                return
            try:
                # OCAtari 2.2.1 swaps the two terminal flags in its public
                # step() return.  Step its underlying Gymnasium wrapper for the
                # canonical (terminated, truncated) order, then explicitly run
                # OCAtari's RAM detector to refresh the object list.
                inner = getattr(self._env, "_env", None)
                if inner is not None and callable(getattr(inner, "step", None)):
                    _, reward, terminated, truncated, info = inner.step(action_index)
                    self._env.detect_objects()
                else:  # compatibility fallback for a changed OCAtari wrapper
                    _, reward, truncated, terminated, info = self._env.step(action_index)
                self._last_reward = float(reward)
                self._score += float(reward)
                self._terminated = bool(terminated)
                self._truncated = bool(truncated)
                self._last_env_info = self._safe_info(info)
                self._frame = self._info_int(info, "episode_frame_number", self._frame + self._frameskip)
                self._ticks += 1
                self._refresh_ram()
            except Exception as exc:  # an emulator/connection failure invalidates this trial
                self._mark_invalid(f"step failed: {type(exc).__name__}: {exc}")

    def info(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "valid_trial": bool(self._valid_trial),
                "seed": self._seed,
                "rom_md5": self._rom_md5,
                "rom_path": self._rom_path,
                "frameskip": int(self._frameskip),
                "tick_s": float(self.tick_s),
                "repeat_action_probability": float(self._repeat_action_probability),
                "ticks": int(self._ticks),
                "frame": int(self._frame),
                "terminated": bool(self._terminated),
                "truncated": bool(self._truncated),
                "score_total": float(self._score),
                "screens_completed": int(self._screens_completed),
                "current_bonus": int(self._current_bonus()),
                "rivets_removed": int(self._rivets_removed(self._ram[31:35])),
                "lives_remaining": int(self._ram[35]),
                "engine_info": dict(self._last_env_info),
            }
            if self._invalid_reason is not None:
                result["invalid_reason"] = str(self._invalid_reason)
            if self._last_error is not None:
                result["last_error"] = str(self._last_error)
            return result

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            return {
                "ale_reward": float(self._score),
                "screens_completed": float(self._screens_completed),
                "current_bonus": float(self._current_bonus()),
                "rivets_removed": float(self._rivets_removed(self._ram[31:35])),
                "lives_remaining": float(self._ram[35]),
                "terminated": 1.0 if self._terminated else 0.0,
                "truncated": 1.0 if self._truncated else 0.0,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._env.close()
            finally:
                # Drop references to the native emulator promptly.  No process
                # is started by this adaptor, but this releases Stella state.
                self._env = None  # type: ignore[assignment]

    # ----- state and serialization helpers ---------------------------------

    def _refresh_ram(self) -> None:
        ram_obj = self._env._ale.getRAM()
        ram = tuple(int(value) for value in ram_obj)
        if len(ram) != 128:
            raise RuntimeError(f"ALE returned {len(ram)} RAM bytes, expected 128")
        previous = self._previous_ram
        was_started = self._started
        if previous is not None:
            previous_rivets = previous[31:35]
            if (
                was_started
                and int(previous[16]) == 1
                and int(ram[16]) == 0
                and all(int(value) >= 18 for value in previous_rivets)
            ):
                self._screens_completed += 1
        if int(ram[12]) == 255:
            self._started = True
        self._previous_ram = ram
        self._ram = ram

    def _mark_invalid(self, reason: str) -> None:
        self._valid_trial = False
        self._invalid_reason = str(reason)[:500]
        self._truncated = True
        self._last_error = self._invalid_reason

    def _coerce_action(self, action: Any) -> tuple[str, int, str | None]:
        if isinstance(action, str):
            key = action.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            aliases = {
                "none": "noop",
                "nooperation": "noop",
                "nothing": "noop",
            }
            key = aliases.get(key, key)
            if key in _ACTION_NAME_TO_INDEX:
                return key, _ACTION_NAME_TO_INDEX[key], None
            return "noop", 0, f"ignored malformed action {action!r}; applied noop"
        if isinstance(action, Integral) and not isinstance(action, bool):
            index = int(action)
            if index in _ACTION_INDEX_TO_NAME:
                return _ACTION_INDEX_TO_NAME[index], index, None
            return "noop", 0, f"ignored illegal ALE action index {index}; applied noop"
        return "noop", 0, f"ignored malformed action of type {type(action).__name__}; applied noop"

    @staticmethod
    def _safe_info(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in info.items():
            if key == "seeds" and isinstance(value, (tuple, list)):
                result[str(key)] = [int(v) for v in value]
            elif isinstance(value, bool):
                result[str(key)] = bool(value)
            elif isinstance(value, Integral):
                result[str(key)] = int(value)
            elif isinstance(value, float):
                result[str(key)] = float(value)
            elif isinstance(value, (str, type(None))):
                result[str(key)] = value
        return result

    @staticmethod
    def _info_int(info: Any, key: str, default: int) -> int:
        if isinstance(info, dict):
            value = info.get(key)
            if isinstance(value, Integral) and not isinstance(value, bool):
                return int(value)
        return int(default)

    @staticmethod
    def _bcd_to_int(value: int) -> int:
        value = int(value) & 0xFF
        return ((value >> 4) & 0x0F) * 10 + (value & 0x0F)

    @staticmethod
    def _rivets_removed(rows: Any) -> int:
        # Each row is 0 (none), 6 (left), 12 (right), or 18 (both).
        total = 0
        for value in rows:
            total += min(18, max(0, int(value))) // 6
        return int(total)

    def _current_bonus(self) -> int:
        return int(self._bcd_to_int(self._ram[36]) * 100)

    @staticmethod
    def _object_present(obj: Any) -> bool:
        if obj is None:
            return False
        category = str(getattr(obj, "category", ""))
        if category == "NoObject":
            return False
        try:
            return bool(obj)
        except Exception:
            return True

    def _object_by_category(self, category: str) -> Any:
        for obj in list(getattr(self._env, "objects", ()) or ()):
            if str(getattr(obj, "category", "")) == category:
                return obj
        return None

    def _spatial_object(
        self,
        obj: Any,
        fallback: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        if obj is not None and self._object_present(obj):
            try:
                x, y, w, h = (int(value) for value in obj.xywh)
                dx = int(getattr(obj, "dx", 0))
                dy = int(getattr(obj, "dy", 0))
                return {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "dx": dx,
                    "dy": dy,
                    "visible": True,
                }
            except Exception:
                pass
        x, y, w, h = fallback or (0, 0, 0, 0)
        return {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "dx": 0,
            "dy": 0,
            "visible": False if fallback is None else True,
        }

    def _stage(self, ram: tuple[int, ...]) -> str:
        if self._terminated:
            return "game_over"
        if self._truncated:
            return "transition"
        if not self._has_reset or not self._started:
            return "idle"
        # $8D is asserted during a life-loss transition.  Other non-active
        # values in the game-state byte are transition/level-complete states.
        if int(ram[13]) != 0 or int(ram[12]) != 255:
            return "transition"
        if int(ram[16]) == 0:
            return "ramp"
        if int(ram[16]) == 1:
            return "rivet"
        return "unknown"

    def _termination_reason(self) -> str:
        if self._valid_trial is False:
            return "infrastructure_error"
        if self._terminated:
            return "game_over"
        if self._truncated:
            return "truncated"
        return ""


def make_game(**kwargs: Any) -> DonkeyKongGame:
    """Construct one fresh local emulator-backed episode controller."""
    return DonkeyKongGame(**kwargs)
