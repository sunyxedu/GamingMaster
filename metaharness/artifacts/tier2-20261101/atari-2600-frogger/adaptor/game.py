"""Headless Atari 2600 Frogger adaptor backed by ALE and Gymnasium.

The adaptor deliberately uses one ALE emulator frame per realtime tick.  ALE
exposes the Frogger cartridge's 128 bytes of RAM; this module decodes the
cartridge's named player, score, home, timer, and object tables into compact
JSON while retaining the raw RAM for auditability.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym
from metaharness.games.base import ActionSpec

TIER = 2
DETERMINISTIC = True

# The primary benchmark is ordinary one-player Frogger (ALE mode 0, difficulty
# 0).  Eight seeds per split gives the search several independent phases while
# keeping train and held-out episodes disjoint.
SEEDS = {
    "train": [101, 103, 107, 109, 113, 127, 131, 137],
    "heldout": [139, 149, 151, 157, 163, 167, 178, 179],
}

# Secondary sets are intentionally separate from SEEDS: the evaluator should
# construct a separate make_game(mode=..., difficulty=...) for them and never
# pool their scores with the primary mode.
SCENARIO_SEEDS = {
    "mode_1_harder": [211, 223, 227, 229, 233, 6, 241, 251],
    "mode_2_speedy": [257, 263, 269, 271, 277, 281, 283, 293],
    "difficulty_1_wrap": [307, 311, 313, 317, 331, 337, 347, 349],
}

EVAL_KWARGS = {
    "mode": 0,
    "difficulty": 0,
    "frameskip": 1,
    "sticky_action_probability": 0.0,
}

_ACTIONS = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
_ACTION_INDEX = {name: i for i, name in enumerate(_ACTIONS)}
_EXPECTED_ROM_MD5 = "081e2c114c9c20b61acf25fc95c71bf4"

# Cartridge RAM is mirrored by ALE as RAM[0] == $80.  These are names used in
# the community disassembly bundled under runtime/ and checked against RGB
# rendering during adaptor review.
_ADDR = {
    "object_a": 0x80,
    "object_b": 0x8B,
    "object_a_tia": 0x96,
    "object_b_tia": 0xA1,
    "frog_row": 0xAC,
    "lady_row": 0xAD,
    "fly_row": 0xAE,
    "frog_x": 0xB0,
    "lady_x": 0xB1,
    "fly_x": 0xB2,
    "frog_tia_x": 0xB4,
    "lady_tia_x": 0xB5,
    "fly_tia_x": 0xB6,
    "nusiz_table": 0xB8,
    "nusiz_table_b": 0xBA,
    "sprite_table_a": 0xC0,
    "sprite_table_b": 0xC2,
    "activity_table": 0xC4,
    "bonus_flags": 0xC7,
    "score_high": 0xCC,
    "score_low": 0xCE,
    "reserve": 0xD0,
    "level": 0xD2,
    "state_pointer": 0xD4,
    "music_pointer": 0xD6,
    "frame_counter": 0xDB,
    "variation": 0xDD,
    "game_state": 0xDE,
    "timer": 0xE0,
    "collision_state": 0xE3,
    "collision_row": 0xE5,
    "player_index": 0xE6,
    "hazard_flags": 0xE7,
    "background_pointer": 0xE8,
    "lady_flags": 0xF9,
    "fly_phase": 0xFA,
}

_ROW_NAMES = {
    0: "road_lane_1",
    1: "road_lane_2",
    2: "road_lane_3",
    3: "road_lane_4",
    4: "road_lane_5",
    5: "median",
    6: "river_row_1",
    7: "river_row_2",
    8: "river_row_3",
    9: "river_row_4",
    10: "river_row_5",
}
# Pixel centers in ALE's 160x210 RGB render.  They are descriptive only; the
# RAM row and world x are the authoritative symbolic coordinates.
_ROW_SCREEN_Y = {0: 165, 1: 150, 2: 135, 3: 125, 4: 110, 5: 95,
                 6: 85, 7: 70, 8: 60, 9: 45, 10: 35}
_HOME_X = (16, 48, 80, 112, 144)

# Atari color-register values observed in this exact ROM.  The numeric code is
# always included, so a later level with a new palette remains inspectable.
_COLOR_NAMES = {
    0: "black",
    1: "white",
    2: "red",
    3: "orange",
    4: "dark_green",
    5: "light_green",
    6: "olive",
    7: "yellow",
    8: "orange",
    9: "brown",
    10: "blue",
    11: "purple",
    12: "pink",
    13: "light_orange",
    14: "light_green",
    15: "white",
}


def _bcd_byte(value: int) -> int:
    """Decode one packed BCD byte, tolerating transient display values."""
    return ((int(value) >> 4) & 0x0F) * 10 + (int(value) & 0x0F)


def _score_from_ram(ram: list[int]) -> int:
    """Decode ALE Frogger's decimal score at 6502 addresses $CC and $CE."""
    low = _bcd_byte(ram[0xCE - 0x80])
    high = _bcd_byte(ram[0xCC - 0x80])
    return high * 100 + low


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_phase(seed: int) -> int:
    """Select a reproducible live-emulator traffic phase for a seed.

    This cartridge has a fixed reset layout and does not randomise its first
    frame from ALE's RNG.  Once the cartridge has entered play, advancing the
    real emulator with NOOP changes the live traffic/platform phase without
    editing RAM or synthesising objects.  The phase is short enough that an
    episode still has nearly the full 30-second Frogger timer.
    """
    x = int(seed) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return 1 + (x % 90)


class FroggerGame:
    """One isolated ALE Frogger episode implementing :class:`RealtimeGame`."""

    name = "Atari 2600 Frogger"
    tick_s = 1.0 / 60.0

    action_spec = ActionSpec(
        names=list(_ACTIONS),
        descriptions={
            "NOOP": "Leave the frog's joystick centered for this Atari frame.",
            "UP": "Make one hop toward the river and home bays.",
            "RIGHT": "Make one hop to the right.",
            "LEFT": "Make one hop to the left.",
            "DOWN": "Make one hop toward the starting sidewalk.",
        },
        noop="NOOP",
        notes=(
            "One tick is one 60 Hz emulator frame.  In games 1-4 a direction "
            "is edge-triggered as one hop; use NOOP between intentional hops. "
            "Mode 2 is Speedy Frogger, where a held direction can repeat."
        ),
    )

    def __init__(
        self,
        *,
        mode: int = 0,
        difficulty: int = 0,
        frameskip: int = 1,
        sticky_action_probability: float = 0.0,
        max_num_frames_per_episode: int = 108000,
    ) -> None:
        if int(mode) not in (0, 1, 2):
            raise ValueError("mode must be ALE Frogger mode 0, 1, or 2")
        if int(difficulty) not in (0, 1):
            raise ValueError("difficulty must be 0 or 1")
        if int(frameskip) != 1:
            raise ValueError("FroggerGame requires frameskip=1 for one-frame ticks")
        if not 0.0 <= float(sticky_action_probability) <= 1.0:
            raise ValueError("sticky_action_probability must be between 0 and 1")

        gym.register_envs(ale_py)
        self.mode = int(mode)
        self.difficulty = int(difficulty)
        self.frameskip = 1
        self.sticky_action_probability = float(sticky_action_probability)
        self.max_num_frames_per_episode = int(max_num_frames_per_episode)
        self._env = gym.make(
            "ALE/Frogger-v5",
            obs_type="ram",
            frameskip=1,
            repeat_action_probability=self.sticky_action_probability,
            full_action_space=False,
            mode=self.mode,
            difficulty=self.difficulty,
            max_num_frames_per_episode=self.max_num_frames_per_episode,
        )

        rom_path = ale_py.roms.get_rom_path("frogger")
        if rom_path is None:
            self._env.close()
            raise RuntimeError("ALE did not provide the Frogger ROM")
        self.rom_path = Path(rom_path)
        self.rom_hash = _md5(self.rom_path)
        if self.rom_hash != _EXPECTED_ROM_MD5:
            self._env.close()
            raise RuntimeError(
                f"unexpected ALE Frogger ROM MD5 {self.rom_hash}; "
                f"expected {_EXPECTED_ROM_MD5}"
            )
        self._rom = self.rom_path.read_bytes()

        self._lock = threading.RLock()
        self._closed = False
        self._seed: int | None = None
        self._startup_frames = 0
        self._seed_phase_frames = 0
        self._rng_seeds: list[int] = []
        self._ram = [0] * 128
        self._ale_info: dict[str, Any] = {}
        self._score = 0
        self._previous_score = 0
        self._score_delta = 0
        self._total_reward = 0.0
        self._last_reward = 0.0
        self._last_action = "NOOP"
        self._invalid_action: str | None = None
        self._ticks = 0
        self._done = False
        self._truncated = False
        self._deaths = 0
        self._frogs_home_count = 0
        self._home_mask = 0
        self._last_home_added = 0
        self._time_bonus_awarded = 0
        self._last_time_bonus_awarded = 0
        self._previous_lives = 4
        self._previous_timer = 30
        self._terminal_reason: str | None = None
        self._valid_trial = True
        self._invalid_reason: str | None = None
        self._reset_done = False
        self._previous_object_x: tuple[int, ...] | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot reset a closed FroggerGame")
            requested_seed = int(seed)
            try:
                observation, initial_info = self._env.reset(seed=requested_seed)

                # ALE resets the cartridge before it leaves its attract/start
                # phase.  The minimal Frogger action set has no FIRE, so wait
                # internally until the cartridge's own state byte reaches the
                # playable state.  These are not agent ticks and are recorded
                # explicitly in metadata.
                startup_frames = 0
                while self._ram_value(observation, _ADDR["game_state"]) == 0x08:
                    if startup_frames >= 1200:
                        raise RuntimeError("Frogger did not leave its start phase")
                    observation, _, terminated, truncated, step_info = self._env.step(0)
                    startup_frames += 1
                    if bool(terminated) or bool(truncated):
                        raise RuntimeError("Frogger ended during its start phase")
                phase_frames = _seed_phase(requested_seed)
                phase_info: dict[str, Any] = self._plain_info(step_info if startup_frames else initial_info)
                for _ in range(phase_frames):
                    observation, _, terminated, truncated, phase_step_info = self._env.step(0)
                    if bool(terminated) or bool(truncated):
                        raise RuntimeError("seed phase ended before play could start")
                    phase_info = self._plain_info(phase_step_info)
            except Exception as exc:  # infrastructure failure, reported in info()
                self._valid_trial = False
                self._invalid_reason = f"ALE reset failed: {type(exc).__name__}: {exc}"[:500]
                self._done = True
                self._seed = requested_seed
                self._startup_frames = 0
                self._seed_phase_frames = 0
                self._rng_seeds = []
                self._reset_done = True
                return self._observation()

            self._seed = requested_seed
            self._startup_frames = startup_frames
            self._seed_phase_frames = phase_frames
            self._rng_seeds = [int(x) for x in initial_info.get("seeds", ())]
            self._ale_info = self._plain_info(phase_info)
            # The first step info normally contains no seed tuple; preserve
            # Gymnasium's reset metadata too.
            if "seeds" not in self._ale_info and self._rng_seeds:
                self._ale_info["seeds"] = list(self._rng_seeds)
            self._ram = self._ram_list(observation)
            self._score = _score_from_ram(self._ram)
            self._previous_score = self._score
            self._score_delta = 0
            self._total_reward = 0.0
            self._last_reward = 0.0
            self._last_action = "NOOP"
            self._invalid_action = None
            self._ticks = 0
            self._done = False
            self._truncated = False
            self._deaths = 0
            self._home_mask = self._home_occupancy_mask()
            self._frogs_home_count = self._home_mask.bit_count()
            self._last_home_added = 0
            self._time_bonus_awarded = 0
            self._last_time_bonus_awarded = 0
            self._previous_lives = int(self._ale_info.get("lives", self._ram_at(_ADDR["reserve"])))
            self._previous_timer = self._ram_at(_ADDR["timer"])
            self._terminal_reason = None
            self._valid_trial = True
            self._invalid_reason = None
            self._reset_done = True
            self._previous_object_x = self._object_positions()
            return self._observation()

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return self._observation()

    def tick(self, action: Any) -> None:
        """Apply one action for one emulator frame; malformed actions become NOOP."""
        with self._lock:
            if self._closed or not self._reset_done or self._done:
                return
            action_name = self._normalise_action(action)
            if action_name is None:
                self._invalid_action = repr(action)[:120]
                action_name = "NOOP"
            else:
                self._invalid_action = None
            self._last_action = action_name
            old_ram = list(self._ram)
            old_timer = self._ram_at(_ADDR["timer"])
            old_lives = int(self._ale_info.get("lives", self._ram_at(_ADDR["reserve"])))
            try:
                observation, reward, terminated, truncated, info = self._env.step(
                    _ACTION_INDEX[action_name]
                )
            except Exception as exc:  # emulator outage is not bad play
                self._valid_trial = False
                self._invalid_reason = f"ALE step failed: {type(exc).__name__}: {exc}"[:500]
                self._done = True
                self._terminal_reason = "infrastructure_error"
                return

            self._ticks += 1
            self._last_reward = float(reward)
            self._total_reward += float(reward)
            self._ale_info = self._plain_info(info)
            self._ram = self._ram_list(observation)
            new_score = _score_from_ram(self._ram)
            self._previous_score = self._score
            self._score_delta = int(new_score - self._score)
            self._score = new_score

            old_home_mask = self._home_mask
            new_home_mask = self._home_occupancy_mask()
            self._last_home_added = new_home_mask & ~old_home_mask
            if self._last_home_added:
                self._frogs_home_count += self._last_home_added.bit_count()
                # The score routine adds home points, optional lady/fly points,
                # and then the remaining-time award.  Keep a conservative exact
                # lower bound and expose the current available bonus separately.
                base_home = 5 * self._last_home_added.bit_count()
                if old_home_mask != 0x1F and new_home_mask == 0x1F:
                    base_home += 100
                remainder = max(0, self._score_delta - base_home)
                self._last_time_bonus_awarded = min(remainder, max(0, old_timer) * 2)
                self._time_bonus_awarded += self._last_time_bonus_awarded
            else:
                self._last_time_bonus_awarded = 0
            self._home_mask = new_home_mask

            lives = int(self._ale_info.get("lives", self._ram_at(_ADDR["reserve"])))
            if lives < old_lives:
                self._deaths += old_lives - lives
                self._terminal_reason = "timeout" if old_timer <= 1 else "life_lost"
            self._previous_lives = lives
            self._previous_timer = old_timer
            self._truncated = bool(truncated)
            self._done = bool(terminated or truncated)
            if self._truncated:
                self._terminal_reason = "frame_limit"
            elif terminated:
                if self._ram_at(_ADDR["reserve"]) == 0xFF:
                    self._terminal_reason = "no_frogs"
                elif self._terminal_reason is None:
                    self._terminal_reason = "terminal"

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            # The decoded on-screen score is the primary machine score; ALE's
            # cumulative reward is available in observe()/score_breakdown().
            return float(self._score)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "valid_trial": bool(self._valid_trial),
                "invalid_reason": self._invalid_reason,
                "seed": self._seed,
                "startup_frames": self._startup_frames,
                "seed_phase_frames": self._seed_phase_frames,
                "rng_seeds": list(self._rng_seeds),
                "mode": self.mode,
                "difficulty": self.difficulty,
                "rom": "frogger",
                "rom_hash": self.rom_hash,
                "frameskip": self.frameskip,
                "sticky_action_probability": self.sticky_action_probability,
                "ticks": self._ticks,
                "episode_frame_number": int(self._ale_info.get("episode_frame_number", self._ticks)),
                "frame_number": int(self._ale_info.get("frame_number", self._ticks)),
            }

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            return {
                "score": float(self._score),
                "ale_reward": float(self._total_reward),
                "score_delta": float(self._score_delta),
                "frogs_home": float(self._frogs_home_count),
                "deaths": float(self._deaths),
                "reserve_frogs": float(self._reserve_frogs()),
                "time_bonus": float(self._time_bonus_awarded),
                "time_bonus_available": float(self._ram_at(_ADDR["timer"]) * 2),
                "level": float(self._level()),
            }

    def describe(self) -> str:
        return (
            "Atari 2600 Parker Brothers Frogger, emulated headlessly by the "
            "official ALE/Stella integration. The adaptor starts the cartridge's "
            "own start phase, then advances a seed-selected live traffic phase; "
            "agent ticks begin at the playable frog. Move one frog from the "
            "bottom sidewalk across five alternating road lanes, a median, and "
            "five river rows into one of five unoccupied home bays. Cars kill; "
            "water and unsafe/diving carriers kill; a timeout or exhausted "
            "reserve ends the run. Five occupied homes advance the cartridge to a "
            "harder level. Score follows the on-screen BCD score: a forward hop "
            "is 1, a home is 5, the five-home bonus is 100, and the manual also "
            "allows lady-frog, fly, and remaining-time bonuses.\n\n"
            "This is real-time at 60 Hz: traffic and carriers animate during a "
            "NOOP. Legal action strings are exactly NOOP, UP, RIGHT, LEFT, and "
            "DOWN; each tick applies one of them for one emulator frame. In the "
            "standard modes a direction is a single edge-triggered hop, so use "
            "NOOP between hops. Mode 2 is Speedy Frogger and can repeat held "
            "directions. A malformed action is ignored as NOOP and reported in "
            "the next observation.\n\n"
            "observe() is strict JSON. It includes raw RAM for $80-$FF, exact "
            "score/reward/life/timer fields, frog row and x, both decoded object "
            "tracks per road/river row with x, direction, velocity, carrier "
            "color/shape and diving hazard, five home occupancy records, bonus "
            "frog/fly candidates, collision flags, and ROM/configuration metadata. "
            "Object x values are the cartridge's 0-160 world pixels; tia_x is the "
            "render-position byte. mode=0,difficulty=0 is primary. Separate "
            "mode=1, mode=2, and difficulty=1 constructions are supported."
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._env.close()
            finally:
                self._env = None  # type: ignore[assignment]

    # ---- RAM/table decoder -------------------------------------------------

    @staticmethod
    def _plain_info(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, (int, float, str, bool)):
                result[str(key)] = value
            elif isinstance(value, (tuple, list)):
                converted: list[Any] = []
                for item in value:
                    if hasattr(item, "item"):
                        try:
                            converted.append(int(item.item()))
                        except Exception:
                            continue
                    elif isinstance(item, (int, float, str, bool)):
                        converted.append(item)
                result[str(key)] = converted
            elif hasattr(value, "item"):
                try:
                    result[str(key)] = int(value.item())
                except Exception:
                    pass
        return result

    @staticmethod
    def _ram_list(observation: Any) -> list[int]:
        values = observation.tolist() if hasattr(observation, "tolist") else list(observation)
        return [max(0, min(255, int(value))) for value in values[:128]]

    @staticmethod
    def _ram_value(observation: Any, address: int) -> int:
        values = observation.tolist() if hasattr(observation, "tolist") else list(observation)
        index = int(address) - 0x80
        return int(values[index]) if 0 <= index < len(values) else 0

    def _ram_at(self, address: int) -> int:
        index = int(address) - 0x80
        if 0 <= index < len(self._ram):
            return int(self._ram[index])
        return 0

    def _u16(self, address: int) -> int:
        return self._ram_at(address) | (self._ram_at(address + 1) << 8)

    def _rom_byte(self, address: int) -> int:
        offset = int(address) - 0xF000
        return int(self._rom[offset]) if 0 <= offset < len(self._rom) else 0

    def _table_byte(self, pointer_address: int, row: int) -> int:
        pointer = self._u16(pointer_address)
        return self._rom_byte(pointer + int(row))

    @staticmethod
    def _normalise_action(action: Any) -> str | None:
        if not isinstance(action, str):
            return None
        candidate = action.strip().upper()
        return candidate if candidate in _ACTION_INDEX else None

    @staticmethod
    def _circular_delta(current: int, previous: int | None) -> int:
        if previous is None:
            return 0
        delta = int(current) - int(previous)
        if delta > 80:
            delta -= 160
        elif delta < -80:
            delta += 160
        return delta

    def _object_positions(self) -> tuple[int, ...]:
        values: list[int] = []
        for row in range(11):
            values.append(self._ram_at(0x80 + row))
            values.append(self._ram_at(0x8B + row))
        return tuple(values)

    def _kind(self, row: int, color_code: int, sprite_ptr: int) -> tuple[str, bool]:
        if row <= 4:
            return "vehicle", True
        if color_code == 10:
            return "diving_turtle", True
        if color_code == 9:
            return "log", False
        if color_code == 6:
            return "turtle", False
        # Later modes/levels can swap the exact sprite/color table. Preserve a
        # useful carrier category and mark unfamiliar forms conservatively.
        if sprite_ptr in (0xB1, 0xB8):
            return "river_carrier", False
        return "river_hazard_or_carrier", True

    def _objects(self) -> list[dict[str, Any]]:
        previous = self._previous_object_x
        current = self._object_positions()
        rows: list[dict[str, Any]] = []
        sprite_table_address_a = self._u16(_ADDR["sprite_table_a"])
        sprite_table_address_b = self._u16(_ADDR["sprite_table_b"])
        for row in range(11):
            sprite_ptr = self._table_byte(_ADDR["sprite_table_a"], row)
            sprite_ptr_b = self._table_byte(_ADDR["sprite_table_b"], row)
            sprite_address = (sprite_table_address_a & 0xFF00) | sprite_ptr
            sprite_address_b = (sprite_table_address_b & 0xFF00) | sprite_ptr_b
            nusiz_a = self._table_byte(_ADDR["nusiz_table"], row)
            # B8 and BA usually point at the same table, but read both to keep
            # the two rendered objects explicit in harder modes.
            nusiz_b = self._table_byte(_ADDR["nusiz_table_b"], row)
            color_a = self._table_byte(0xBC, row)
            color_b = self._table_byte(0xBE, row)
            activity_mask = self._table_byte(_ADDR["activity_table"], row)
            direction_code = self._rom_byte(0xFD47 + row)
            direction = "left" if direction_code & 0x80 else "right"
            object_list: list[dict[str, Any]] = []
            for index, (x_addr, tia_addr, color, shape, nusiz) in enumerate(
                (
                    (0x80 + row, 0x96 + row, color_a, sprite_ptr, nusiz_a),
                    (0x8B + row, 0xA1 + row, color_b, sprite_ptr_b, nusiz_b),
                )
            ):
                x = self._ram_at(x_addr)
                previous_x = previous[row * 2 + index] if previous is not None else None
                kind, hazard = self._kind(row, color, shape)
                width = self._render_width(nusiz)
                end = x + width
                x_intervals = [[x, min(160, end)]]
                if end > 160:
                    x_intervals.append([0, end - 160])
                object_list.append(
                    {
                        "id": f"{_ROW_NAMES[row]}_{index}",
                        "index": index,
                        "x": x,
                        "x_interval": x_intervals[0],
                        "x_intervals": x_intervals,
                        "width_px_estimate": width,
                        "tia_x": self._ram_at(tia_addr),
                        "direction": direction,
                        "velocity_px_per_tick": self._circular_delta(x, previous_x),
                        "kind": kind,
                        "hazard": bool(hazard),
                        "color_code": color,
                        "color": _COLOR_NAMES.get(color, "unknown"),
                        "shape_ptr": shape,
                        "shape_address": sprite_address if index == 0 else sprite_address_b,
                        "nusiz": nusiz,
                        "update_scheduled_this_frame": (activity_mask & self._ram_at(_ADDR["frame_counter"])) == 0,
                    }
                )
            rows.append(
                {
                    "row": row,
                    "name": _ROW_NAMES[row],
                    "zone": "road" if row <= 4 else ("median" if row == 5 else "river"),
                    "screen_y_px": _ROW_SCREEN_Y[row],
                    "direction": direction,
                    "activity_mask": activity_mask,
                    "objects": object_list if row != 5 else [],
                }
            )
        self._previous_object_x = current
        return rows

    @staticmethod
    def _render_width(nusiz: int) -> int:
        # A conservative collision/render envelope for Atari NUSIZ codes. The
        # exact register is included alongside this estimate for controllers
        # that want to use their own margin.
        return {0: 8, 1: 16, 2: 24, 3: 24, 4: 16, 5: 24, 6: 32, 7: 24}.get(int(nusiz) & 7, 8)

    def _home_occupancy_mask(self) -> int:
        # E3 low five bits are set when the corresponding home bay is filled;
        # LFCD9 in the cartridge checks exactly this mask against $1f.
        return self._ram_at(_ADDR["collision_state"]) & 0x1F

    def _homes(self) -> list[dict[str, Any]]:
        mask = self._home_occupancy_mask()
        target = self._ram_at(_ADDR["collision_row"])
        frog_row = self._ram_at(_ADDR["frog_row"])
        target_slot = target if frog_row == 11 and 0 <= target < 5 else None
        return [
            {
                "slot": slot,
                "x_center": x,
                "occupied": bool(mask & (1 << slot)),
                "available": not bool(mask & (1 << slot)),
                "target": target_slot == slot,
                # The ROM has no separate home-crocodile byte. C7/F9 are
                # retained below as named bonus/hazard flags rather than
                # pretending that a missing head is safe.
                "crocodile": "unknown",
                "fly_bonus": "unknown",
            }
            for slot, x in enumerate(_HOME_X)
        ]

    @staticmethod
    def _row_zone(row: int) -> str:
        if row == 0xFF:
            return "start_sidewalk"
        if 0 <= row <= 4:
            return "road"
        if row == 5:
            return "median"
        if 6 <= row <= 10:
            return "river"
        if row == 0x49:
            return "home_complete_respawn"
        if row == 11:
            return "home_approach"
        return "transition_or_unknown"

    def _frog(self) -> dict[str, Any]:
        row = self._ram_at(_ADDR["frog_row"])
        return {
            "x": self._ram_at(_ADDR["frog_x"]),
            "tia_x": self._ram_at(_ADDR["frog_tia_x"]),
            "row": row,
            "row_name": _ROW_NAMES.get(row, "home" if row == 11 else "off_board"),
            "zone": self._row_zone(row),
            "moving_with_platform": bool(self._ram_at(_ADDR["hazard_flags"]) & 0x80),
            "state": self._ram_at(_ADDR["game_state"]),
        }

    def _bonus_frog(self, row_addr: int, x_addr: int, tia_addr: int, kind: str) -> dict[str, Any]:
        row = self._ram_at(row_addr)
        present = 0 <= row <= 10
        return {
            "kind": kind,
            "present": present,
            "row": row if present else None,
            "row_name": _ROW_NAMES.get(row, "absent") if present else "absent",
            "x": self._ram_at(x_addr) if present else None,
            "tia_x": self._ram_at(tia_addr) if present else None,
            "hazard": False,
        }

    def _level(self) -> int:
        # D2 is the cartridge's current game/difficulty progression value. It
        # starts at 1, is 3 for ALE mode 1, and advances as homes are completed.
        return max(1, self._ram_at(_ADDR["level"]))

    def _reserve_frogs(self) -> int:
        raw = self._ram_at(_ADDR["reserve"])
        return 0 if raw == 0xFF else raw

    def _observation(self) -> dict[str, Any]:
        d0 = self._ram_at(_ADDR["reserve"])
        timer = self._ram_at(_ADDR["timer"])
        e3 = self._ram_at(_ADDR["collision_state"])
        e5 = self._ram_at(_ADDR["collision_row"])
        c7 = self._ram_at(_ADDR["bonus_flags"])
        f9 = self._ram_at(_ADDR["lady_flags"])
        frame = int(self._ale_info.get("episode_frame_number", self._ticks))
        object_rows = self._objects()
        return {
            "game": self.name,
            "seed": self._seed,
            "startup_frames": self._startup_frames,
            "seed_phase_frames": self._seed_phase_frames,
            "rng_seeds": list(self._rng_seeds),
            "mode": self.mode,
            "difficulty": self.difficulty,
            "rom": "frogger",
            "rom_hash": self.rom_hash,
            "frameskip": self.frameskip,
            "sticky_action_probability": self.sticky_action_probability,
            "tick_s": self.tick_s,
            "tick": self._ticks,
            "frame": frame,
            "frame_number": int(self._ale_info.get("frame_number", frame)),
            "raw_ram": list(self._ram),
            "decoder_confidence": "calibrated_ram_and_render_alignment",
            "ram_fields": {
                "score_cc": self._ram_at(_ADDR["score_high"]),
                "score_ce": self._ram_at(_ADDR["score_low"]),
                "reserve_d0": d0,
                "frog_x_b0": self._ram_at(_ADDR["frog_x"]),
                "frog_tia_x_b4": self._ram_at(_ADDR["frog_tia_x"]),
                "frog_row_ac": self._ram_at(_ADDR["frog_row"]),
                "lady_x_b1": self._ram_at(_ADDR["lady_x"]),
                "lady_row_ad": self._ram_at(_ADDR["lady_row"]),
                "fly_x_b2": self._ram_at(_ADDR["fly_x"]),
                "fly_row_ae": self._ram_at(_ADDR["fly_row"]),
                "timer_e0": timer,
                "state_de": self._ram_at(_ADDR["game_state"]),
                "collision_e3": e3,
                "collision_row_e5": e5,
                "hazard_e7": self._ram_at(_ADDR["hazard_flags"]),
                "bonus_flags_c7": c7,
                "lady_flags_f9": f9,
                "home_occupancy_mask": e3 & 0x1F,
                "level_counter_d2": self._ram_at(_ADDR["level"]),
                "variation_dd": self._ram_at(_ADDR["variation"]),
            },
            "frog": self._frog(),
            "road": [row for row in object_rows if row["zone"] == "road"],
            "river": [row for row in object_rows if row["zone"] == "river"],
            "median": {
                "row": 5,
                "safe_pause": True,
                "screen_y_px": _ROW_SCREEN_Y[5],
            },
            "homes": self._homes(),
            "lady_frog": self._bonus_frog(_ADDR["lady_row"], _ADDR["lady_x"], _ADDR["lady_tia_x"], "lady_frog"),
            "fly": self._bonus_frog(_ADDR["fly_row"], _ADDR["fly_x"], _ADDR["fly_tia_x"], "fly"),
            "bonus_flags": {
                "raw_c7": c7,
                "fly_or_bonus_active": bool(c7 & 0x80),
                "fly_collected_or_in_play": bool(c7 & 0x40),
                "lady_raw_f9": f9,
                "home_crocodile": "not separately exposed by this ROM",
            },
            "score": int(self._score),
            "score_delta": int(self._score_delta),
            "reward": float(self._last_reward),
            "total_reward": float(self._total_reward),
            "frogs_home": int(self._frogs_home_count),
            "homes_occupied_current_level": int((e3 & 0x1F).bit_count()),
            "last_home_added_mask": int(self._last_home_added),
            "deaths": int(self._deaths),
            "reserve_frogs": int(self._reserve_frogs()),
            "frogs_remaining_including_active": (
                0 if d0 == 0xFF else int(self._reserve_frogs() + 1)
            ),
            "reserve_frogs_raw": int(d0),
            "lives_raw": int(self._ale_info.get("lives", d0)),
            "timer_s": int(timer),
            "time_bonus": int(self._time_bonus_awarded),
            "time_bonus_awarded_last": int(self._last_time_bonus_awarded),
            "time_bonus_available": int(timer * 2),
            "level": int(self._level()),
            "last_action": self._last_action,
            "invalid_action": self._invalid_action,
            "terminal": bool(self._done),
            "truncated": bool(self._truncated),
            "terminal_memory": bool(d0 == 0xFF),
            "terminal_reason": self._terminal_reason,
        }


def make_game(**kwargs: Any) -> FroggerGame:
    """Construct a fresh isolated Frogger episode instance."""
    return FroggerGame(**kwargs)
