"""Headless Atari 2600 Q*bert adaptor backed by ALE/Stella.

The evaluator owns the clock.  Each :meth:`QbertGame.tick` call sends one
current joystick action to Gymnasium/ALE and advances exactly one configured
ALE frameskip quantum.  The public state is ordinary JSON: the 128-byte Atari
RAM snapshot is retained for auditability, while the ROM-specific OCAtari RAM
decoder supplies named cubes, Q*bert, balls, Coily, Sam, and discs.
"""
from __future__ import annotations

import copy
import hashlib
import math
import numbers
import threading
from collections import deque
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
from metaharness.games.base import ActionSpec

try:
    # OCAtari's Q*bert decoder is a read-only decoder over the ALE RAM.  It
    # does not own an emulator or a display.
    from ocatari.ram import qbert as _qbert_decoder
    from ocatari.ram.extract_ram_info import detect_objects_ram as _detect_objects_ram
except Exception:  # pragma: no cover - requirements.txt supplies this path
    _qbert_decoder = None
    _detect_objects_ram = None


TIER = 2
# The opening position is fixed, but a fresh seeded ALE instance reproduces its
# first state.  With v5 sticky actions, later action traces are intentionally
# stochastic and held-out seeds exercise fresh sticky-action streams.
DETERMINISTIC = True
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "heldout": [100, 101, 102, 103, 104, 105, 106, 107],
}
EVAL_KWARGS = {
    "frameskip": 4,
    "repeat_action_probability": 0.25,
    "difficulty": 0,
    "mode": 0,
    "warm_start": True,
}

_EXPECTED_ROM_MD5 = "484b0076816a104875e00467d431c2d2"
_NTSC_FPS = 60.0

_ACTIONS: tuple[tuple[str, int], ...] = (
    ("NOOP", 0),
    ("FIRE", 1),
    ("UP", 2),
    ("RIGHT", 3),
    ("LEFT", 4),
    ("DOWN", 5),
)
_ACTION_TO_INDEX = dict(_ACTIONS)
_INDEX_TO_ACTION = {index: name for name, index in _ACTIONS}

# The 21-node Atari 2600 pyramid, in row-major order (1+2+3+4+5+6).
_CUBE_POSITIONS: tuple[tuple[int, int], ...] = (
    (68, 34),
    (56, 62), (84, 62),
    (44, 91), (68, 91), (96, 91),
    (32, 120), (56, 120), (84, 120), (108, 120),
    (20, 149), (44, 149), (68, 149), (96, 149), (120, 149),
    (8, 178), (32, 178), (56, 178), (84, 178), (108, 178), (132, 178),
)
_CUBE_RAM_INDICES: tuple[int, ...] = (
    21, 52, 54, 83, 85, 87, 98, 100, 102, 104,
    1, 3, 5, 7, 9, 32, 34, 36, 38, 40, 42,
)

# Each pair is (neighbour node, ALE action).  UP from a left child is UP and
# LEFT from a right child is the inverse; the names are the official ALE names,
# not screen-cardinal guesses.
def _make_board() -> dict[int, list[tuple[int, str]]]:
    rows: list[list[int]] = []
    cursor = 0
    for row_size in range(1, 7):
        rows.append(list(range(cursor, cursor + row_size)))
        cursor += row_size
    board: dict[int, list[tuple[int, str]]] = {i: [] for i in range(21)}
    for row in range(5):
        for col, node in enumerate(rows[row]):
            left_child = rows[row + 1][col]
            right_child = rows[row + 1][col + 1]
            board[node].extend(((left_child, "DOWN"), (right_child, "RIGHT")))
            board[left_child].append((node, "UP"))
            board[right_child].append((node, "LEFT"))
    return board


_BOARD = _make_board()
_CYCLE_LABELS = {
    1: ("S", "D"),
    2: ("S", "I", "D"),
    3: ("S", "D"),
    4: ("S", "I", "D", "I"),
    5: ("S", "I", "D", "S"),
}

# Values are the palette byte stored by the cartridge for the opening and the
# Level 1 destination.  Other values remain exposed as raw_color and are
# assigned a cycle phase by the transition tracker, rather than guessed from
# an arcade board.
_START_COLOR = 148
_LEVEL1_DESTINATION = 26
_COLOR_NAMES = {
    148: "start_blue",
    26: "level1_destination_orange",
    210: "yellow",
    98: "green",
    146: "purple",
    181: "qbert_orange",
}

# The Stella RGB buffer is used only to correct sprite visibility/position
# when a RAM sprite slot still contains its previous animation position.  The
# palette families are Atari colours, not a screenshot observation.
_QBERT_RGB = ((181, 83, 40),)
_GREEN_RGB = (
    (0, 68, 0), (26, 102, 26), (50, 132, 50), (72, 160, 72),
    (92, 186, 92), (111, 210, 111), (128, 232, 128), (144, 252, 144),
)
_PURPLE_RGB = (
    (132, 0, 100), (151, 25, 122), (168, 48, 143), (184, 70, 162),
    (198, 89, 179), (212, 108, 195), (224, 124, 210), (236, 140, 224),
    (80, 0, 132), (104, 25, 154), (125, 48, 173), (146, 70, 192),
    (164, 89, 208), (181, 108, 224), (197, 124, 238), (212, 140, 252),
)
_RED_RGB = ((195, 144, 61), (210, 164, 74), (223, 183, 85), (236, 200, 96))
_DISK_RGB = (
    (48, 56, 0), (80, 89, 22), (109, 118, 43), (136, 146, 62),
    (160, 171, 79), (183, 194, 95), (204, 216, 110), (224, 236, 124),
)

_ACTION_DESCRIPTIONS = {
    "NOOP": "Hold the joystick neutral for one ALE quantum; use this only for a deliberate wait.",
    "FIRE": "Press FIRE. It is not a movement direction and may restart/reset the cartridge in this ROM.",
    "UP": "Hop toward the upper-right edge of the diamond; from a left child this returns toward its parent.",
    "RIGHT": "Hop down/right to the right child of the current cube when that edge exists.",
    "LEFT": "Hop up/left to the parent of a right child when that edge exists.",
    "DOWN": "Hop down/left to the left child of the current cube when that edge exists.",
}


class QbertGame:
    """A RealtimeGame facade for the Atari 2600 Q*bert cartridge."""

    name = "Atari 2600 Q*bert"
    action_spec = ActionSpec(
        names=[name for name, _ in _ACTIONS],
        descriptions=dict(_ACTION_DESCRIPTIONS),
        noop="NOOP",
        notes=(
            "Official minimal ALE indices are NOOP=0, FIRE=1, UP=2, RIGHT=3, "
            "LEFT=4, DOWN=5.  One tick holds the chosen action for the configured "
            "frameskip (default four NTSC frames, about 66.7 ms); v5 sticky actions "
            "can repeat the previous joystick action with probability 0.25.  Do not "
            "use the arcade 28-cube map: this Atari port has 21 cubes."
        ),
    )

    def __init__(
        self,
        *,
        mode: int = 0,
        difficulty: int = 0,
        frameskip: int = 4,
        repeat_action_probability: float = 0.25,
        obs_type: str = "ram",
        render_mode: Any = None,
        full_action_space: bool = False,
        max_num_frames_per_episode: int | None = 108_000,
        warm_start: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(str(k) for k in kwargs))
            raise TypeError(f"unknown Q*bert adaptor option(s): {unknown}")
        if obs_type != "ram":
            raise ValueError("this adaptor requires obs_type='ram'")
        if render_mode is not None:
            raise ValueError("this headless adaptor requires render_mode=None")
        if isinstance(mode, bool) or not isinstance(mode, numbers.Integral) or int(mode) != 0:
            raise ValueError("Atari 2600 Q*bert only supports mode=0")
        if isinstance(difficulty, bool) or not isinstance(difficulty, numbers.Integral) or int(difficulty) not in (0, 1):
            raise ValueError("difficulty must be 0 (novice) or 1 (expert)")
        if isinstance(frameskip, bool) or not isinstance(frameskip, numbers.Integral) or int(frameskip) <= 0:
            raise ValueError("frameskip must be a positive integer")
        probability = float(repeat_action_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("repeat_action_probability must be between 0 and 1")
        if max_num_frames_per_episode is not None:
            if (
                isinstance(max_num_frames_per_episode, bool)
                or not isinstance(max_num_frames_per_episode, numbers.Integral)
                or int(max_num_frames_per_episode) <= 0
            ):
                raise ValueError("max_num_frames_per_episode must be a positive integer or None")
            max_num_frames_per_episode = int(max_num_frames_per_episode)

        # This adaptor intentionally uses the six-action minimal Q*bert set.
        if bool(full_action_space):
            raise ValueError("full_action_space=True is not supported; use the six-action Q*bert set")

        rom_path = ale_py.roms.get_rom_path("qbert")
        if rom_path is None or not rom_path.is_file():
            raise RuntimeError(
                "ALE could not find qbert.bin; install ale-py==0.12.1 or set ALE_ROMS_DIR "
                "to a directory containing the legal checksum-matching Q*bert ROM"
            )
        with rom_path.open("rb") as rom_file:
            rom_md5 = hashlib.md5(rom_file.read()).hexdigest()
        if rom_md5 != _EXPECTED_ROM_MD5:
            raise RuntimeError(
                f"unexpected Q*bert ROM MD5 {rom_md5}; expected {_EXPECTED_ROM_MD5}"
            )

        gym.register_envs(ale_py)
        env_kwargs: dict[str, Any] = {
            "obs_type": "ram",
            "render_mode": None,
            "mode": int(mode),
            "difficulty": int(difficulty),
            "full_action_space": False,
            "frameskip": int(frameskip),
            "repeat_action_probability": probability,
        }
        if max_num_frames_per_episode is not None:
            env_kwargs["max_num_frames_per_episode"] = max_num_frames_per_episode
        self._env = gym.make("ALE/Qbert-v5", **env_kwargs)
        meanings = list(self._env.unwrapped.get_action_meanings())
        if meanings != [name for name, _ in _ACTIONS]:
            self._env.close()
            raise RuntimeError(
                f"unexpected ALE Q*bert action order {meanings!r}; expected "
                f"{[name for name, _ in _ACTIONS]!r}"
            )

        self.mode = int(mode)
        self.difficulty = int(difficulty)
        self.frameskip = int(frameskip)
        self.repeat_action_probability = probability
        self.max_num_frames_per_episode = max_num_frames_per_episode
        self.warm_start = bool(warm_start)
        self._rom_path = str(rom_path)
        self._rom_md5 = rom_md5
        self._lock = threading.RLock()
        self._closed = False
        self._has_reset = False

        self._seed: int | None = None
        self._env_info: dict[str, Any] = {}
        self._last_ram = np.zeros(128, dtype=np.uint8)
        self._objects: Any = None
        self._decoder_error: str | None = None
        self._decoder_name = "ocatari.ram.qbert 2.2.1" if _qbert_decoder is not None else "builtin cube fallback"

        self._ticks = 0
        self._warmup_steps = 0
        self._frame = 0
        self._score = 0.0
        self._decoded_score = 0
        self._score_match = True
        self._last_reward = 0.0
        self._last_action = "NOOP"
        self._posted_action: tuple[str, int] | None = None
        self._last_requested_action: Any = None
        self._action_error: str | None = None
        self._terminated = False
        self._truncated = False
        self._done = False
        self._valid_trial = True
        self._invalid_reason: str | None = None

        self._lives = 0
        self._previous_lives = 0
        self._life_lost_count = 0
        self._completed_rounds = 0
        self._extra_life_count = 0
        self._last_event = "none"
        self._event_counts = {"cube": 0, "sam": 0, "green_ball": 0, "coily": 0, "round": 0}
        self._event_points = {"cube": 0, "sam": 0, "green_ball": 0, "coily": 0, "round": 0}

        self._tracking_level = 1
        self._cube_previous: list[int | None] = [None] * 21
        self._cube_phase_index: list[int] = [0] * 21
        self._cube_start_color: list[int | None] = [None] * 21
        self._previous_cube_states: list[str] = ["unknown"] * 21
        self._red_ball_previous: tuple[int, int] | None = None
        self._red_ball_visual: dict[str, Any] | None = None
        self._visual_components: dict[str, list[tuple[int, int, int, int, int]]] = {}
        self._visual_previous: dict[str, tuple[int, int]] = {}
        self._observation: dict[str, Any] = self._empty_observation()

    @property
    def tick_s(self) -> float:
        return float(self.frameskip) / _NTSC_FPS

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._score)

    def describe(self) -> str:
        return (
            "Atari 2600 Q*bert is a local NTSC Parker Brothers cartridge emulated by "
            "ALE/Stella; no display, account, server, GPU, or network service is used. "
            "The player hops on a six-row, 21-cube pyramid.  A cube landing changes its "
            "colour according to the current level's finite cycle; finish the required "
            "destination phase on all cubes while preserving Q*bert lives.  Level 1 uses "
            "S->D, Level 2 S->I->D, Level 3 S->D->S, Level 4 S->I->D->I, and Level 5 "
            "S->I->D->S; Level 5 repeats after the first five levels.  The port has four "
            "rounds per level.  Avoid Red Balls (expert difficulty), Purple Balls and "
            "Coily; Sam reverses cube progress and is worth 300, Green Ball is worth 100, "
            "and luring Coily off a disc is worth 500.  Cube destination changes are 25 "
            "points and a completed round is 3,100 points.  Survival and completing a "
            "round are more important than optional bonuses.\n\n"
            "This is a real-time interface: call tick(action) once per tick_s.  The default "
            "is ALE/Qbert-v5 with frameskip=4 (four emulated NTSC frames, about 66.7 ms) "
            "and repeat_action_probability=0.25.  A late tick with None repeats the last "
            "posted action, or applies NOOP before the first action; malformed names or "
            "indices are ignored, a NOOP is applied, and action_error is reported in the "
            "next observation.  Legal names are exactly NOOP, FIRE, UP, RIGHT, LEFT, DOWN; "
            "integer indices 0..5 are accepted as a convenience.  FIRE is not a normal "
            "movement action.  UP/RIGHT/LEFT/DOWN are the four joystick edges, and an edge "
            "can be lethal if it is absent from the current node's safe_moves.  Do not use "
            "the 28-node arcade board.\n\n"
            "The JSON observation includes frame/tick counters, lives and other resources, cumulative ALE score, "
            "decoded BCD cartridge score and score_match, level/round progress, qbert_node, rendered "
            "Q*bert position/motion, hop/fall phase, and named cartridge timers/scheduler states.  It "
            "contains cube_state and needed_landings for all 21 fixed nodes plus destination/intermediate/"
            "incomplete counts; red_ball/purple_ball/coily/sam/green_ball objects when visible with kind, "
            "position and motion; hazard distances/risks; available disc_exits; graph legal_moves and "
            "conservative hazard-blocked/safe_moves; last reward/event/action, raw 128-byte RAM, and "
            "terminal/truncation flags.  The decoder cross-checks RAM sprite slots against the current "
            "ALE RGB palette internally so stale animation placeholders are not shown as live enemies; "
            "the RGB frame itself is never exposed.  reset performs a bounded NOOP bootstrap by "
            "default so the first observation is the playable Level 1 Round 1 opening rather "
            "than the cartridge's attract-screen blank frames; those bootstrap frames are not "
            "counted as agent ticks.  Seeds select ALE's seeded sticky-action RNG stream, not "
            "a new board layout.  Set frameskip=1 and repeat_action_probability=0 only for a "
            "separate deterministic decoder calibration run."
        )

    def reset(self, seed: int) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"reset seed must be an integer, got {seed!r}")
        with self._lock:
            self._ensure_open()
            try:
                observation, info = self._env.reset(seed=int(seed))
                self._check_ram(observation)
                self._seed = int(seed)
                self._env_info = self._plain_info(info)
                self._last_ram = np.asarray(observation, dtype=np.uint8).copy()
                self._objects = _qbert_decoder._init_objects_ram(False) if _qbert_decoder is not None else None
                self._decoder_error = None
                self._ticks = 0
                self._warmup_steps = 0
                self._frame = self._info_int(self._env_info, "episode_frame_number", 0)
                self._score = 0.0
                self._decoded_score = self._decode_score(self._last_ram)
                self._score_match = self._score == float(self._decoded_score)
                self._last_reward = 0.0
                self._last_action = "NOOP"
                self._posted_action = None
                self._last_requested_action = None
                self._action_error = None
                self._terminated = False
                self._truncated = False
                self._done = False
                self._valid_trial = True
                self._invalid_reason = None
                self._lives = self._info_lives(self._env_info, self._last_ram, default=4)
                self._previous_lives = self._lives
                self._life_lost_count = 0
                self._extra_life_count = 0
                self._completed_rounds = 0
                self._last_event = "none"
                self._event_counts = {"cube": 0, "sam": 0, "green_ball": 0, "coily": 0, "round": 0}
                self._event_points = {"cube": 0, "sam": 0, "green_ball": 0, "coily": 0, "round": 0}
                self._tracking_level = 1
                self._cube_previous = [None] * 21
                self._cube_phase_index = [0] * 21
                self._cube_start_color = [None] * 21
                self._previous_cube_states = ["unknown"] * 21
                self._red_ball_previous = None
                self._red_ball_visual = None
                self._visual_components = {}
                self._visual_previous = {}
                self._has_reset = True

                if self.warm_start:
                    self._bootstrap_to_play()
                self._refresh_observation()
                return self.observe()
            except Exception as exc:
                self._has_reset = True
                self._mark_invalid(f"ALE reset failed: {type(exc).__name__}: {exc}")
                self._refresh_observation()
                return self.observe()

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._observation)

    def tick(self, action: Any) -> None:
        """Apply one action quantum; the evaluator, not this method, owns timing."""
        with self._lock:
            if self._closed or self._done:
                return
            if not self._has_reset:
                self.reset(0)
                if self._done:
                    return

            normalized, error = self._normalize_action(action)
            self._last_requested_action = self._json_value(action)
            self._action_error = error
            self._last_action = normalized
            if error is None and action is not None:
                self._posted_action = (normalized, _ACTION_TO_INDEX[normalized])
            try:
                observation, reward, terminated, truncated, info = self._env.step(
                    _ACTION_TO_INDEX[normalized]
                )
                self._check_ram(observation)
                self._last_ram = np.asarray(observation, dtype=np.uint8).copy()
                self._last_reward = self._finite_float(reward)
                self._score += self._last_reward
                self._env_info = self._plain_info(info)
                self._frame = self._info_int(
                    self._env_info, "episode_frame_number", self._frame + self.frameskip
                )
                self._terminated = bool(terminated)
                self._truncated = bool(truncated)
                self._done = bool(self._terminated or self._truncated)
                self._ticks += 1
                self._record_reward(self._last_reward)
                current_lives = self._info_lives(self._env_info, self._last_ram, default=self._lives)
                if current_lives < self._previous_lives:
                    self._life_lost_count += self._previous_lives - current_lives
                elif current_lives > self._previous_lives:
                    self._extra_life_count += current_lives - self._previous_lives
                self._previous_lives = current_lives
                self._lives = current_lives
                self._refresh_observation()
            except Exception as exc:
                self._mark_invalid(f"ALE step failed: {type(exc).__name__}: {exc}")
                self._refresh_observation()

    def info(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "valid_trial": bool(self._valid_trial),
                "seed": self._seed,
                "rom_md5": self._rom_md5,
                "rom_path": self._rom_path,
                "ale_package": str(getattr(ale_py, "__version__", "0.12.1")),
                "gymnasium_package": str(getattr(gym, "__version__", "unknown")),
                "decoder": self._decoder_name,
                "decoder_error": self._decoder_error,
                "mode": self.mode,
                "difficulty": self.difficulty,
                "frameskip": self.frameskip,
                "tick_s": float(self.tick_s),
                "repeat_action_probability": self.repeat_action_probability,
                "warm_start": self.warm_start,
                "warmup_steps": self._warmup_steps,
                "ticks": self._ticks,
                "episode_frame_number": self._frame,
                "frame_number": self._info_int(self._env_info, "frame_number", self._frame),
                "terminated": bool(self._terminated),
                "truncated": bool(self._truncated),
                "done": bool(self._done),
                "score": float(self._score),
                "decoded_score": int(self._decoded_score),
                "score_match": bool(self._score_match),
                "lives": int(self._lives),
                "completed_rounds": int(self._completed_rounds),
                "completed_levels": int(self._completed_rounds // 4),
                "life_lost_count": int(self._life_lost_count),
                "extra_life_count": int(self._extra_life_count),
                "terminal_step_reward": float(self._last_reward) if self._done else None,
                "terminal_step_reward_zero": bool(self._done and self._last_reward == 0.0),
                "engine_info": dict(self._env_info),
            }
            if self._invalid_reason is not None:
                result["invalid_reason"] = str(self._invalid_reason)
            return result

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            return {
                "ale_reward": float(self._score),
                "decoded_cartridge_score": float(self._decoded_score),
                "cube_change_points": float(self._event_points["cube"]),
                "sam_points": float(self._event_points["sam"]),
                "green_ball_points": float(self._event_points["green_ball"]),
                "coily_points": float(self._event_points["coily"]),
                "round_bonus_points": float(self._event_points["round"]),
                "completed_rounds": float(self._completed_rounds),
                "lives_remaining": float(self._lives),
                "life_losses": float(self._life_lost_count),
                "extra_lives_awarded": float(self._extra_life_count),
                "terminated": 1.0 if self._terminated else 0.0,
                "truncated": 1.0 if self._truncated else 0.0,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            env = self._env
            self._env = None  # type: ignore[assignment]
            if env is not None:
                try:
                    env.close()
                except Exception:
                    # close() is best effort and idempotent; the evaluator gets
                    # the actual infrastructure error through info if needed.
                    pass

    # ---- emulator / observation bookkeeping ---------------------------------

    def _bootstrap_to_play(self) -> None:
        """Skip the cartridge attract-screen blank frames without agent ticks."""
        # At frameskip=1 Q*bert enters the playable opening around frame 150;
        # use a little margin for compatible ALE builds and stop as soon as the
        # player RAM position is live.
        max_steps = max(1, int(math.ceil(240 / self.frameskip)))
        for _ in range(max_steps):
            if self._is_playable(self._last_ram):
                break
            observation, reward, terminated, truncated, info = self._env.step(0)
            self._check_ram(observation)
            self._last_ram = np.asarray(observation, dtype=np.uint8).copy()
            self._env_info = self._plain_info(info)
            self._frame = self._info_int(self._env_info, "episode_frame_number", self._frame + self.frameskip)
            warm_reward = self._finite_float(reward)
            self._score += warm_reward
            if warm_reward:
                self._record_reward(warm_reward)
            self._warmup_steps += 1
            if bool(terminated) or bool(truncated):
                self._terminated = bool(terminated)
                self._truncated = bool(truncated)
                self._done = True
                break
        self._decoded_score = self._decode_score(self._last_ram)
        self._score_match = math.isclose(self._score, float(self._decoded_score), rel_tol=0.0, abs_tol=1e-6)
        self._lives = self._info_lives(self._env_info, self._last_ram, default=self._lives)
        self._previous_lives = self._lives

    @staticmethod
    def _is_playable(ram: np.ndarray) -> bool:
        # $8C is the game-state byte and RAM[67] is Q*bert's active vertical
        # position.  This avoids treating the attract screen's stale enemy
        # bytes as named game objects.
        return int(ram[12]) == 176 and 16 < int(ram[67]) < 210

    def _refresh_observation(self) -> None:
        self._decoded_score = self._decode_score(self._last_ram)
        self._score_match = math.isclose(
            self._score, float(self._decoded_score), rel_tol=0.0, abs_tol=1e-6
        )
        try:
            self._decode_objects()
            self._decoder_error = None
        except Exception as exc:
            self._decoder_error = f"{type(exc).__name__}: {exc}"
            # A decoder problem is reported but does not make ordinary game
            # play an infrastructure failure; raw RAM and fixed cube geometry
            # remain available.
        self._update_cube_tracking()
        self._observation = self._build_observation()

    def _decode_objects(self) -> None:
        playable_state = int(self._last_ram[12]) == 176
        if _qbert_decoder is None or _detect_objects_ram is None:
            self._objects = None
        else:
            if self._objects is None:
                self._objects = _qbert_decoder._init_objects_ram(False)
            if self._is_playable(self._last_ram):
                _detect_objects_ram(self._objects, self._last_ram, "Qbert", False)
            else:
                # The decoder's Player() default is a placeholder at (78,103),
                # not evidence that a sprite is currently visible.
                self._objects = _qbert_decoder._init_objects_ram(False)

        self._visual_components = self._capture_visual_components() if playable_state else {}
        self._red_ball_visual = self._visual_object(
            None, "red_ball", "red", unique=True
        ) if self.difficulty == 1 else None

    def _capture_visual_components(self) -> dict[str, list[tuple[int, int, int, int, int]]]:
        """Decode small Atari sprite components from the native RGB frame.

        RAM remains the primary decoder.  The RGB pass is a calibration/cursor
        correction: it prevents the OCAtari RAM helper's default Player object
        or a stale Sam/PurpleBall slot from being presented as a live object
        during an edge fall or sprite transition.  No pixels leave this method.
        """
        if self._env is None:
            return {}
        try:
            screen = np.asarray(self._env.unwrapped.ale.getScreenRGB(), dtype=np.uint8)
            if screen.ndim != 3 or screen.shape[-1] != 3:
                return {}
            return {
                "qbert": self._component_bboxes(screen, _QBERT_RGB, 8, 12, 24),
                "green": self._component_bboxes(screen, _GREEN_RGB, 8, 14, 24),
                "purple": self._component_bboxes(screen, _PURPLE_RGB, 8, 14, 24),
                "red": self._component_bboxes(screen, _RED_RGB, 8, 14, 16, 4),
                "disc": self._component_bboxes(screen, _DISK_RGB, 4, 12, 4),
            }
        except Exception:
            return {}

    @staticmethod
    def _component_bboxes(
        screen: np.ndarray,
        colors: tuple[tuple[int, int, int], ...],
        minimum_pixels: int,
        maximum_width: int,
        maximum_height: int,
        minimum_height: int = 1,
    ) -> list[tuple[int, int, int, int, int]]:
        mask = np.zeros(screen.shape[:2], dtype=bool)
        for color in colors:
            mask |= np.all(screen == np.asarray(color, dtype=np.uint8), axis=2)
        ys, xs = np.nonzero(mask)
        pixels = set(zip(xs.tolist(), ys.tolist()))
        components: list[tuple[int, int, int, int, int]] = []
        while pixels:
            start = pixels.pop()
            stack = [start]
            component = [start]
            while stack:
                x, y = stack.pop()
                for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if neighbour in pixels:
                        pixels.remove(neighbour)
                        stack.append(neighbour)
                        component.append(neighbour)
            if len(component) < minimum_pixels:
                continue
            xx = [point[0] for point in component]
            yy = [point[1] for point in component]
            x0, x1, y0, y1 = min(xx), max(xx), min(yy), max(yy)
            width, height = x1 - x0 + 1, y1 - y0 + 1
            if width <= maximum_width and height <= maximum_height and height >= minimum_height:
                components.append((len(component), x0, y0, width, height))
        return components

    def _visual_object(
        self,
        obj: Any,
        kind: str,
        family: str,
        *,
        unique: bool = False,
    ) -> dict[str, Any] | None:
        raw = self._object_json(obj, kind)
        components = self._visual_components.get(family, [])
        best: tuple[int, int, int, int, int] | None = None
        if components:
            available = [
                (index, component)
                for index, component in enumerate(components)
                if (family, index) not in self._visual_claimed
            ]
            if available and unique:
                if unique and kind == "qbert":
                    # Q*bert's head and body can be separated by black sprite
                    # pixels.  Combine nearby same-palette components into one
                    # rendered bounding box before reporting position/motion.
                    selected = [component for _, component in available]
                    x0 = min(component[1] for component in selected)
                    y0 = min(component[2] for component in selected)
                    x1 = max(component[1] + component[3] - 1 for component in selected)
                    y1 = max(component[2] + component[4] - 1 for component in selected)
                    if x1 - x0 + 1 <= 12 and y1 - y0 + 1 <= 24:
                        best = (
                            sum(component[0] for component in selected),
                            x0, y0, x1 - x0 + 1, y1 - y0 + 1,
                        )
                        for index, _ in available:
                            self._visual_claimed.add((family, index))
                    else:
                        index, best = max(available, key=lambda pair: pair[1])
                        self._visual_claimed.add((family, index))
                else:
                    index, best = max(available, key=lambda pair: pair[1])
                    self._visual_claimed.add((family, index))
            elif available and raw is not None:
                index, best = min(
                    available,
                    key=lambda pair: abs(pair[1][1] - raw["x"])
                    + abs(pair[1][2] - raw["y"]),
                )
                distance = abs(best[1] - raw["x"]) + abs(best[2] - raw["y"])
                # A same-colour Sam and Green Ball must be matched to the RAM
                # slot; never silently relabel a distant sprite.
                if distance > 12:
                    best = None
                else:
                    self._visual_claimed.add((family, index))
        if best is None:
            if raw is not None and kind == "qbert":
                raw["visible"] = False
                raw["source"] = "ram_decoder_unrendered"
                return raw
            return None
        _, x, y, width, height = best
        previous = self._visual_previous.get(kind)
        dx = x - previous[0] if previous is not None else (raw["dx"] if raw else 0)
        dy = y - previous[1] if previous is not None else (raw["dy"] if raw else 0)
        self._visual_previous[kind] = (x, y)
        return {
            "kind": kind,
            "x": int(x), "y": int(y), "w": int(width), "h": int(height),
            "dx": int(dx), "dy": int(dy), "visible": True,
            "source": "ALE_RGB_palette_component",
        }

    def _decode_objects_for_observation(self) -> tuple[Any, Any, Any, Any, Any, list[Any]]:
        player = self._object_at(0)
        coily = self._object_at(24)
        purple = self._object_at(25)
        green = self._object_at(26)
        sam = self._object_at(27)
        disks = [self._object_at(22), self._object_at(23)]
        return player, purple, coily, sam, green, disks

    def _object_at(self, index: int) -> Any:
        if not self._is_playable(self._last_ram):
            return None
        if self._objects is None or index >= len(self._objects):
            return None
        obj = self._objects[index]
        if obj is None:
            return None
        category = str(getattr(obj, "category", ""))
        if not category or category == "NoObject":
            return None
        return obj

    @staticmethod
    def _object_json(obj: Any, kind: str | None = None) -> dict[str, Any] | None:
        if obj is None:
            return None
        try:
            x, y, w, h = (int(value) for value in obj.xywh)
            return {
                "kind": kind or str(getattr(obj, "category", "object")).lower(),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "dx": int(getattr(obj, "dx", 0)),
                "dy": int(getattr(obj, "dy", 0)),
                "visible": True,
            }
        except Exception:
            return None

    def _update_cube_tracking(self) -> None:
        colors = [int(self._last_ram[index]) for index in _CUBE_RAM_INDICES]
        level = self._current_level
        if level != self._tracking_level:
            # A new level can use a different palette/cycle.  Resynchronise
            # against the first visible colour instead of carrying a Level 1
            # phase index into the reversible Level 3-5 rules.
            self._tracking_level = level
            self._cube_previous = [None] * 21
            self._cube_phase_index = [0] * 21
            self._cube_start_color = [None] * 21
        cycle_length = len(_CYCLE_LABELS[level])
        for i, color in enumerate(colors):
            previous = self._cube_previous[i]
            if previous is None:
                self._cube_start_color[i] = color if color not in (0, 255) else _START_COLOR
                self._cube_phase_index[i] = 0
            elif color != previous:
                if color == self._cube_start_color[i] or color == _START_COLOR:
                    self._cube_phase_index[i] = 0
                else:
                    self._cube_phase_index[i] = (self._cube_phase_index[i] + 1) % cycle_length
            self._cube_previous[i] = color
        self._previous_cube_states = [self._phase_label(i, colors[i], level) for i in range(21)]

    def _phase_label(self, index: int, color: int, level: int) -> str:
        if color in (0, 255):
            return "unknown"
        start = self._cube_start_color[index]
        if start is not None and color == start:
            return "S"
        if level == 1 and color == _LEVEL1_DESTINATION:
            return "D"
        labels = _CYCLE_LABELS[level]
        if self._cube_phase_index[index] < len(labels):
            return labels[self._cube_phase_index[index]]
        return "unknown"

    @property
    def _current_level(self) -> int:
        return ((self._completed_rounds // 4) % 5) + 1

    @property
    def _current_round(self) -> int:
        return self._completed_rounds + 1

    def _needed_landings(self, index: int, phase: str, level: int) -> int:
        if phase == "unknown":
            return -1
        labels = _CYCLE_LABELS[level]
        if level == 1 and phase == "D":
            return 0
        current = self._cube_phase_index[index] % len(labels)
        for distance in range(0, len(labels) + 1):
            if labels[(current + distance) % len(labels)] == "D":
                return int(distance)
        return -1

    def _build_observation(self) -> dict[str, Any]:
        self._visual_claimed: set[tuple[str, int]] = set()
        player, purple_obj, coily_obj, sam_obj, green_obj, disks = self._decode_objects_for_observation()
        qbert = self._visual_object(player, "qbert", "qbert", unique=True)
        purple = self._visual_object(purple_obj, "purple_ball", "purple")
        coily = self._visual_object(coily_obj, "coily", "purple")
        sam = self._visual_object(sam_obj, "sam", "green")
        green_ball = self._visual_object(green_obj, "green_ball", "green")
        qbert_node = self._qbert_node(qbert)
        qbert_in_hop = bool(qbert is not None and (abs(int(qbert["dx"])) + abs(int(qbert["dy"])) > 0))
        level = self._current_level
        phases = list(self._previous_cube_states)
        colors = [int(self._last_ram[index]) for index in _CUBE_RAM_INDICES]
        cube_details = []
        needed = []
        for i, ((x, y), color, phase) in enumerate(zip(_CUBE_POSITIONS, colors, phases)):
            required = self._needed_landings(i, phase, level)
            needed.append(required)
            cube_details.append({
                "node": i,
                "row": self._node_row(i),
                "col": self._node_col(i),
                "x": int(x),
                "y": int(y),
                "raw_color": int(color),
                "color": _COLOR_NAMES.get(int(color), f"palette_{int(color)}"),
                "phase": phase,
                "needed_landings": int(required),
            })

        disc_exits = []
        for side, disk in zip(("left", "right"), disks):
            item = self._visual_object(disk, f"disc_{side}", "disc")
            if item is not None:
                item["kind"] = "disc"
            if item is None:
                item = {
                    "kind": "disc", "x": 12 if side == "left" else 140, "y": 138,
                    "w": 8, "h": 2, "dx": 0, "dy": 0, "visible": False,
                    "source": "RAM_slot_not_currently_rendered",
                }
            item.update({
                "side": side,
                "available": bool(disk is not None),
                "validated_exit": None,
                "safe_now": False,
            })
            disc_exits.append(item)

        destination_cubes = int(phases.count("D"))
        intermediate_cubes = int(phases.count("I"))
        unknown_cubes = int(phases.count("unknown"))
        incomplete_cubes = int(21 - destination_cubes)

        hazard_objects = [
            ("red_ball", self._red_ball_visual, True),
            ("purple_ball", purple, True),
            ("coily", coily, True),
            ("sam", sam, False),
            ("green_ball", green_ball, False),
        ]
        hazards, legal_moves, blocked_moves, safe_moves = self._hazard_snapshot(
            qbert, qbert_node, hazard_objects
        )
        qbert_motion = self._qbert_motion_snapshot(qbert, qbert_in_hop)
        next_extra_life_round = (
            5 if self._completed_rounds < 5
            else 5 + 4 * ((self._completed_rounds - 5) // 4 + 1)
        )
        resources = {
            "lives": int(self._lives),
            "life_losses": int(self._life_lost_count),
            "extra_lives_awarded": int(self._extra_life_count),
            "next_extra_life_round": int(next_extra_life_round),
            "completed_rounds": int(self._completed_rounds),
        }
        terminal_reason = self._termination_reason()
        return {
            "frame": int(self._frame),
            "global_frame": int(self._info_int(self._env_info, "frame_number", self._frame)),
            "tick": int(self._ticks),
            "ticks": int(self._ticks),
            "level": int(level),
            "round": int(self._current_round),
            "round_in_level": int(((self._completed_rounds) % 4) + 1),
            "completed_rounds": int(self._completed_rounds),
            "completed_levels": int(self._completed_rounds // 4),
            "difficulty": "expert" if self.difficulty == 1 else "novice",
            "lives": int(self._lives),
            "score": float(self._score),
            "cumulative_ale_reward": float(self._score),
            "decoded_score": int(self._decoded_score),
            "score_match": bool(self._score_match),
            "qbert_node": qbert_node,
            "qbert": qbert,
            "qbert_in_hop": qbert_in_hop,
            "qbert_motion": qbert_motion,
            "timers": self._timer_snapshot(),
            "resources": resources,
            "cube_state": phases,
            "cube_phase": phases,
            "cube_colors": colors,
            "cube_details": cube_details,
            "destination_cubes": destination_cubes,
            "intermediate_cubes": intermediate_cubes,
            "unknown_cubes": unknown_cubes,
            "incomplete_cubes": incomplete_cubes,
            "needed_landings": needed,
            "red_ball": self._red_ball_visual,
            "purple_ball": purple,
            "coily": coily,
            "sam": sam,
            "green_ball": green_ball,
            "hazards": hazards,
            "disc_exits": disc_exits,
            "legal_moves": legal_moves,
            "blocked_moves": blocked_moves,
            "safe_moves": safe_moves,
            "last_reward": float(self._last_reward),
            "last_event": self._last_event,
            "event_counts": {str(k): int(v) for k, v in self._event_counts.items()},
            "life_lost_count": int(self._life_lost_count),
            "last_action": self._last_action,
            "requested_action": self._last_requested_action,
            "action_error": self._action_error,
            "warmup_steps": int(self._warmup_steps),
            "terminal": bool(self._terminated),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "done": bool(self._done),
            "termination_reason": terminal_reason,
            "terminal_step_reward_zero": bool(self._done and self._last_reward == 0.0),
            "raw_ram": [int(value) for value in self._last_ram.tolist()],
        }

    def _timer_snapshot(self) -> dict[str, Any]:
        ram = self._last_ram
        return {
            "hop_animation_timer": int(ram[56]),
            "object_scheduler": int(ram[74]),
            "green_ball_scheduler": int(ram[103]),
            "sam_scheduler": int(ram[105]),
            "purple_ball_state": int(ram[119]),
            "falling_flag": bool(int(ram[59]) == 255),
            "sprite_state": int(ram[126]),
            "sprite_palette_state": int(ram[127]),
            "units": "cartridge frames/ROM state bytes; hop_animation_timer counts down during a hop",
        }

    def _qbert_motion_snapshot(self, qbert: dict[str, Any] | None, in_hop: bool) -> dict[str, Any]:
        falling = bool(int(self._last_ram[59]) == 255)
        if falling:
            phase = "falling"
        elif qbert is None or not qbert.get("visible", False):
            phase = "not_visible"
        elif in_hop:
            phase = "hopping"
        else:
            phase = "standing"
        return {
            "phase": phase,
            "falling": falling,
            "rendered": bool(qbert is not None and qbert.get("visible", False)),
            "ram_x": int(self._last_ram[43]),
            "ram_y": int(self._last_ram[67]),
            "animation_timer": int(self._last_ram[56]),
            "sprite_state": int(self._last_ram[126]),
        }

    @staticmethod
    def _object_node(obj: dict[str, Any] | None) -> int | None:
        if obj is None:
            return None
        x, y = int(obj["x"]), int(obj["y"])
        distances = [
            (x + int(obj.get("w", 0)) // 2 - (cx + 10)) ** 2
            + (y + int(obj.get("h", 0)) // 2 - (cy + 2)) ** 2
            for cx, cy in _CUBE_POSITIONS
        ]
        return int(min(range(21), key=lambda i: distances[i]))

    def _hazard_snapshot(
        self,
        qbert: dict[str, Any] | None,
        qbert_node: int | None,
        objects: list[tuple[str, dict[str, Any] | None, bool]],
    ) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
        hazards: list[dict[str, Any]] = []
        lethal_nodes: set[int] = set()
        blocked_moves: set[str] = set()
        for kind, obj, lethal in objects:
            if obj is None:
                continue
            item = dict(obj)
            item["lethal"] = bool(lethal)
            item["estimated_node"] = self._object_node(obj)
            if qbert is None:
                item["distance_to_qbert"] = None
                item["risk"] = "unknown"
            else:
                distance = int(round(math.hypot(
                    (int(obj["x"]) + int(obj.get("w", 0)) // 2)
                    - (int(qbert["x"]) + int(qbert.get("w", 0)) // 2),
                    (int(obj["y"]) + int(obj.get("h", 0)) // 2)
                    - (int(qbert["y"]) + int(qbert.get("h", 0)) // 2),
                )))
                item["distance_to_qbert"] = distance
                item["risk"] = "immediate" if distance <= 20 else "near" if distance <= 40 else "distant"
            hazards.append(item)
            if lethal:
                node = item["estimated_node"]
                if node is not None:
                    lethal_nodes.add(int(node))
        legal_moves: list[str] = []
        if qbert_node is not None and not self._done:
            legal_moves = [action for _, action in _BOARD.get(qbert_node, ())]
            for target, action in _BOARD.get(qbert_node, ()):
                if target in lethal_nodes:
                    blocked_moves.add(action)
            if qbert_node in lethal_nodes:
                blocked_moves.update(legal_moves)
        safe_moves = [action for action in legal_moves if action not in blocked_moves]
        return hazards, legal_moves, sorted(blocked_moves), safe_moves

    def _qbert_node(self, qbert: dict[str, Any] | None) -> int | None:
        if qbert is None:
            return None
        qx, qy = int(qbert["x"]), int(qbert["y"])
        # The decoder's player box is six pixels right and 17 pixels above
        # the cube anchor when standing.  Nearest-anchor remains useful during
        # the hop animation and makes the estimate explicit rather than hiding
        # a pixel array from the harness.
        distances = [
            (qx - (x + 6)) ** 2 + (qy - (y - 17)) ** 2
            for x, y in _CUBE_POSITIONS
        ]
        return int(min(range(21), key=lambda i: distances[i]))

    @staticmethod
    def _node_row(node: int) -> int:
        total = 0
        for row in range(6):
            if node < total + row + 1:
                return row
            total += row + 1
        return 5

    @classmethod
    def _node_col(cls, node: int) -> int:
        row = cls._node_row(node)
        start = sum(range(1, row + 1))
        return int(node - start)

    def _record_reward(self, reward: float) -> None:
        self._last_event = "none"
        if reward <= 0:
            return
        if reward >= 3100:
            event, points = "round", 3100
            self._completed_rounds += 1
        elif math.isclose(reward, 500.0, abs_tol=1e-6):
            event, points = "coily", 500
        elif math.isclose(reward, 300.0, abs_tol=1e-6):
            event, points = "sam", 300
        elif math.isclose(reward, 100.0, abs_tol=1e-6):
            event, points = "green_ball", 100
        elif math.isclose(reward, 25.0, abs_tol=1e-6):
            event, points = "cube", 25
        else:
            event, points = "other", int(round(reward))
        self._last_event = event
        if event in self._event_counts:
            self._event_counts[event] += 1
            self._event_points[event] += points

    def _termination_reason(self) -> str:
        if not self._valid_trial:
            return "infrastructure_error"
        if self._terminated:
            return "game_over"
        if self._truncated:
            return "frame_limit"
        return ""

    def _mark_invalid(self, reason: str) -> None:
        self._valid_trial = False
        self._invalid_reason = str(reason)[:500]
        self._done = True
        self._action_error = self._invalid_reason

    def _check_ram(self, observation: Any) -> None:
        array = np.asarray(observation, dtype=np.uint8)
        if array.shape != (128,):
            raise RuntimeError(f"ALE Q*bert RAM observation has shape {array.shape}, expected (128,)")

    @staticmethod
    def _decode_score(ram: np.ndarray) -> int:
        low = int(ram[0x5B])
        middle = int(ram[0x5A])
        high = int(ram[0x59])
        return (
            10 * (low >> 4) + (low & 15)
            + 1000 * (middle >> 4) + 100 * (middle & 15)
            + 100000 * (high >> 4) + 10000 * (high & 15)
        )

    @staticmethod
    def _finite_float(value: Any) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"ALE returned non-finite reward {value!r}")
        return result

    @staticmethod
    def _plain_info(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, bool):
                result[str(key)] = bool(value)
            elif isinstance(value, numbers.Integral):
                result[str(key)] = int(value)
            elif isinstance(value, numbers.Real):
                value = float(value)
                if math.isfinite(value):
                    result[str(key)] = value
            elif isinstance(value, (str, type(None))):
                result[str(key)] = value
            elif isinstance(value, (tuple, list)):
                result[str(key)] = [int(v) if isinstance(v, numbers.Integral) else v for v in value]
        return result

    @staticmethod
    def _info_int(info: dict[str, Any], key: str, default: int) -> int:
        value = info.get(key)
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
        return int(default)

    @staticmethod
    def _info_lives(info: dict[str, Any], ram: np.ndarray, default: int) -> int:
        value = info.get("lives")
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return max(0, int(value))
        return max(0, int(default))

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            value = float(value)
            return value if math.isfinite(value) else repr(value)
        return repr(value)[:200]

    def _normalize_action(self, action: Any) -> tuple[str, str | None]:
        if action is None:
            if self._posted_action is not None:
                return self._posted_action[0], None
            return "NOOP", None
        if isinstance(action, str):
            candidate = action.strip().upper().replace("-", "_").replace(" ", "_")
            aliases = {"NONE": "NOOP", "NO_OPERATION": "NOOP", "NOTHING": "NOOP"}
            candidate = aliases.get(candidate, candidate)
            if candidate in _ACTION_TO_INDEX:
                return candidate, None
            return "NOOP", f"ignored malformed action {action!r}; applied NOOP"
        if isinstance(action, numbers.Integral) and not isinstance(action, bool):
            index = int(action)
            if index in _INDEX_TO_ACTION:
                return _INDEX_TO_ACTION[index], None
            return "NOOP", f"ignored illegal ALE action index {index}; applied NOOP"
        return "NOOP", f"ignored malformed action of type {type(action).__name__}; applied NOOP"

    def _empty_observation(self) -> dict[str, Any]:
        return {
            "frame": 0,
            "global_frame": 0,
            "tick": 0,
            "ticks": 0,
            "level": 1,
            "round": 1,
            "round_in_level": 1,
            "completed_rounds": 0,
            "completed_levels": 0,
            "difficulty": "expert" if self.difficulty == 1 else "novice",
            "lives": 0,
            "score": 0.0,
            "cumulative_ale_reward": 0.0,
            "decoded_score": 0,
            "score_match": True,
            "qbert_node": None,
            "qbert": None,
            "qbert_in_hop": False,
            "qbert_motion": {
                "phase": "not_visible", "falling": False, "rendered": False,
                "ram_x": 0, "ram_y": 0, "animation_timer": 0, "sprite_state": 0,
            },
            "timers": {},
            "resources": {
                "lives": 0, "life_losses": 0, "extra_lives_awarded": 0,
                "next_extra_life_round": 5, "completed_rounds": 0,
            },
            "cube_state": ["unknown"] * 21,
            "cube_phase": ["unknown"] * 21,
            "cube_colors": [0] * 21,
            "cube_details": [],
            "destination_cubes": 0,
            "intermediate_cubes": 0,
            "unknown_cubes": 21,
            "incomplete_cubes": 21,
            "needed_landings": [-1] * 21,
            "red_ball": None,
            "purple_ball": None,
            "coily": None,
            "sam": None,
            "green_ball": None,
            "hazards": [],
            "disc_exits": [],
            "legal_moves": [],
            "blocked_moves": [],
            "safe_moves": [],
            "last_reward": 0.0,
            "last_event": "none",
            "event_counts": {"cube": 0, "sam": 0, "green_ball": 0, "coily": 0, "round": 0},
            "life_lost_count": 0,
            "last_action": "NOOP",
            "requested_action": None,
            "action_error": None,
            "warmup_steps": 0,
            "terminal": False,
            "terminated": False,
            "truncated": False,
            "done": False,
            "termination_reason": "",
            "terminal_step_reward_zero": False,
            "raw_ram": [0] * 128,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("QbertGame is closed")


def make_game(**kwargs: Any) -> QbertGame:
    """Construct a fresh local ALE/Stella Q*bert episode facade."""
    return QbertGame(**kwargs)
