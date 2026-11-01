"""Real Atari 2600 Asteroids via the installed ALE/Stella emulator."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym
from metaharness.games.base import ActionSpec

from asteroid_objects import ObjectDecoder, displayed_score

TIER = 2
DETERMINISTIC = True
SEEDS = {"train": list(range(0, 16, 2)), "heldout": list(range(1, 16, 2))}
EVAL_KWARGS = {"scenario_suite": "bounded-launch-v1"}
FPS = 60
FRAME_BUDGET = 18000
HERE = Path(__file__).resolve().parent
SCENARIO_VERSION = "bounded-launch-v1"
ROM_SHA256 = "5ba6f91851b2331a37a3fe6a950c9b464fd764cb035a3d25d4b4388e9b26768f"
# One parameterized physical prelude, not a seed-dependent gameplay policy.
# Even and odd durations interleave the same family in train and heldout.
SCENARIOS = {s: {"family": "clockwise-thrust", "action": "UPRIGHT",
                 "frames": 60 + 4 * s} for s in range(16)}

ACTION_DESCRIPTIONS = {
    "NOOP": "Release all controls; inertia and the game continue.",
    "FIRE": "Press the fire button without thrust or rotation.",
    "UP": "Thrust along the current nose direction.",
    "RIGHT": "Rotate clockwise without thrust.",
    "LEFT": "Rotate counterclockwise without thrust.",
    "DOWN": "Invoke the cartridge defense (hyperspace in mode 0); not braking.",
    "UPRIGHT": "Thrust and rotate clockwise.",
    "UPLEFT": "Thrust and rotate counterclockwise.",
    "UPFIRE": "Thrust and press fire.",
    "RIGHTFIRE": "Rotate clockwise and press fire.",
    "LEFTFIRE": "Rotate counterclockwise and press fire.",
    "DOWNFIRE": "Invoke defense and press fire.",
    "UPRIGHTFIRE": "Thrust, rotate clockwise, and press fire.",
    "UPLEFTFIRE": "Thrust, rotate counterclockwise, and press fire.",
}


class AsteroidsGame:
    name = "Atari 2600 Asteroids"
    tick_s = 1.0 / FPS

    def __init__(self, *, scenario_suite=SCENARIO_VERSION, include_ram=True):
        if scenario_suite not in (SCENARIO_VERSION, "canonical"):
            raise ValueError("scenario_suite must be bounded-launch-v1 or canonical")
        if not isinstance(include_ram, bool):
            raise TypeError("include_ram must be bool")
        self.scenario_suite = scenario_suite
        self.include_ram = include_ram
        self._env = None
        self._closed = False
        self._started = False
        self._score = 0.0
        self._ticks = self._frame = self._lives = self._lost = self._gained = 0
        self._terminated = self._truncated = False
        self._valid_trial = True
        self._invalid_reason = None
        self._reason = None
        self._snapshot = {}
        self._last_action = "NOOP"
        self._action_status = "not_started"
        self._prelude = {}
        self._seed = None
        self._decoder = ObjectDecoder()
        gym.register_envs(ale_py)
        try:
            self._env = gym.make(
                "ALE/Asteroids-v5", obs_type="ram", render_mode=None,
                frameskip=1, repeat_action_probability=0.0, full_action_space=False,
                mode=0, difficulty=0, max_num_frames_per_episode=FRAME_BUDGET)
            meanings = tuple(self._env.unwrapped.get_action_meanings())
            if set(meanings) != set(ACTION_DESCRIPTIONS):
                raise RuntimeError(f"Unexpected ALE action set: {meanings}")
            self._actions = {name: i for i, name in enumerate(meanings)}
            self.action_spec = ActionSpec(
                names=meanings, descriptions=dict(ACTION_DESCRIPTIONS), noop="NOOP",
                notes=("Each tick is one 1/60-second emulator frame. Names are case-sensitive. "
                       "Optional ACTION@FRAME applies only while FRAME <= current frame < FRAME+6; "
                       "FRAME is an observed absolute episode frame. Expired/invalid input releases controls. "
                       "Plain names apply on every tick the runner repeats them; no hidden action latch or sleep."))
            self._versions = {p: importlib.metadata.version(p) for p in ("ale-py", "gymnasium", "numpy")}
            rom = Path(ale_py.roms.get_rom_path("asteroids"))
            self._rom_sha256 = hashlib.sha256(rom.read_bytes()).hexdigest()
            if self._rom_sha256 != ROM_SHA256:
                raise RuntimeError(f"Asteroids ROM hash mismatch: {self._rom_sha256}")
            self._available_modes = [int(v) for v in self._env.unwrapped.ale.getAvailableModes()]
            self._available_difficulties = [int(v) for v in self._env.unwrapped.ale.getAvailableDifficulties()]
        except Exception:
            self.close()
            raise

    def _fail(self, exc):
        self._valid_trial = False
        self._invalid_reason = f"{type(exc).__name__}: {str(exc)[:400]}"
        self._reason = "infrastructure_error"

    def reset(self, seed: int):
        if self._closed:
            raise RuntimeError("Cannot reset a closed game; construct a fresh instance")
        if type(seed) is not int or seed not in SCENARIOS:
            raise ValueError("Valid seeds are the declared integers 0..15")
        self._seed = seed
        self._score = 0.0
        self._ticks = self._frame = self._lives = self._lost = self._gained = 0
        self._ram, self._objects, self._native_seeds = [], {}, []
        self._terminated = self._truncated = False
        self._valid_trial, self._invalid_reason, self._reason = True, None, None
        self._last_action, self._action_status = "NOOP", "reset"
        self._decoder = ObjectDecoder()
        self._started = True
        config = SCENARIOS[seed] if self.scenario_suite == SCENARIO_VERSION else {
            "family": "canonical-fixed-start", "action": "NOOP", "frames": 0}
        self._prelude = {"family": config["family"], "action": config["action"],
                         "planned_frames": config["frames"], "frames": 0,
                         "reward": 0.0, "lives_lost": 0, "lives_gained": 0}
        try:
            ram, native_info = self._env.reset(seed=seed)
            self._native_seeds = [int(v) for v in native_info.get("seeds", ())]
            self._lives = int(native_info["lives"])
            self._ram = [int(v) for v in ram]
            self._objects = self._decoder.decode(self._ram, 0)
            for _ in range(config["frames"]):
                if self.done:
                    break
                self._advance(config["action"])
                self._prelude["frames"] = self._frame
            self._prelude.update(reward=self._score, lives_lost=self._lost, lives_gained=self._gained)
            self._action_status = "prelude_complete" if config["frames"] else "reset"
        except Exception as exc:
            self._fail(exc)
        self._publish()
        return self.observe()

    def _advance(self, action):
        ram, reward, terminated, truncated, info = self._env.step(self._actions[action])
        self._score += float(reward)
        frame = int(info["episode_frame_number"])
        if frame != self._frame + 1:
            raise RuntimeError(f"Expected one ALE frame, got {self._frame} -> {frame}")
        self._frame = frame
        lives = int(info["lives"])
        self._lost += max(0, self._lives - lives)
        self._gained += max(0, lives - self._lives)
        self._lives = lives
        self._terminated = bool(terminated)
        self._truncated = bool(truncated or self._frame >= FRAME_BUDGET)
        if self._terminated:
            self._reason = "game_over"
        elif self._truncated:
            self._reason = "frame_budget" if self._frame >= FRAME_BUDGET else "emulator_truncated"
        self._last_action = action
        self._ram = [int(v) for v in ram]
        self._objects = self._decoder.decode(self._ram, frame)

    def _parse_action(self, action):
        if not isinstance(action, str) or len(action) > 64:
            return "NOOP", "invalid_action_noop"
        if action in self._actions:
            return action, "applied"
        match = re.fullmatch(r"([A-Z]+)@([0-9]{1,6})", action)
        if match and match[1] in self._actions:
            observed = int(match[2])
            if observed <= self._frame < observed + 6:
                return match[1], "applied_fresh"
            return "NOOP", "stale_or_future_action_noop"
        return "NOOP", "invalid_action_noop"

    def tick(self, action: Any) -> None:
        if self._closed or not self._started or self.done:
            return
        chosen, self._action_status = self._parse_action(action)
        try:
            self._advance(chosen)
            self._ticks += 1
        except Exception as exc:
            self._fail(exc)
        self._publish()

    def _publish(self):
        if not hasattr(self, "_ram"):
            self._ram = []
            self._objects = {}
        self._snapshot = {
            "game": self.name, "frame": self._frame, "ticks": self._ticks,
            "elapsed_s": round(self._frame / FPS, 6), "frame_budget": FRAME_BUDGET,
            "frames_remaining": max(0, FRAME_BUDGET - self._frame),
            "observation_frame": self._frame, "observation_age_frames": 0,
            "score": self._score, "displayed_score": displayed_score(self._ram) if self._ram else None,
            "lives": self._lives, "lives_lost": self._lost, "lives_gained": self._gained,
            "done": self.done, "terminated": self._terminated, "truncated": self._truncated,
            "end_reason": self._reason, "last_action": self._last_action, "action_status": self._action_status,
            "settings": {"mode": 0, "difficulty": 0, "sticky_action_probability": 0.0,
                         "frames_per_tick": 1, "frames_per_second": FPS,
                         "scenario_suite": self.scenario_suite},
            "prelude": copy.deepcopy(self._prelude),
            "screen": {"width": 160, "height": 210, "coordinate_units": "pixels; x right, y down"},
            **copy.deepcopy(self._objects),
            "coverage": {"geometry": "approximate RAM-derived reference boxes, not collision masks",
                         "asteroid_source_slots": 15, "player_missile_slots": 2,
                         "enemy_craft": "not decoded; null does not mean absent",
                         "enemy_projectiles": "not decoded; null does not mean absent",
                         "motion": "finite difference up to 4 frames; reset at gaps, size changes and large jumps/wraps; silent slot reuse may be missed",
                         "wrap_period": "unverified; no invented wrap velocities"},
        }
        if self.include_ram:
            self._snapshot["ram"] = list(self._ram)

    def observe(self):
        return copy.deepcopy(self._snapshot)

    @property
    def done(self) -> bool:
        return bool(self._terminated or self._truncated or not self._valid_trial)

    @property
    def score(self) -> float:
        return float(self._score)

    @property
    def ticks(self) -> int:
        return int(self._ticks)

    def score_breakdown(self):
        return {"ale_reward": self.score}

    def info(self):
        return {
            "valid_trial": self._valid_trial, "invalid_reason": self._invalid_reason,
            "seed": self._seed, "scenario_suite": self.scenario_suite, "prelude": copy.deepcopy(self._prelude),
            "surviving_frames": self._frame, "surviving_seconds": self._frame / FPS,
            "lives_lost": self._lost, "lives_gained": self._gained, "lives": self._lives,
            "displayed_score": self._snapshot.get("displayed_score"),
            "terminated": self._terminated, "truncated": self._truncated, "end_reason": self._reason,
            "settings": copy.deepcopy(self._snapshot.get("settings", {})),
            "versions": dict(self._versions), "rom_sha256": self._rom_sha256,
            "native_seeds": list(getattr(self, "_native_seeds", [])),
            "available_modes": list(self._available_modes),
            "available_difficulties": list(self._available_difficulties),
        }

    def describe(self):
        return (
            "Real Atari 2600 Asteroids, one-player ALE mode 0/difficulty 0. Destroy rocks with the nose-mounted "
            "gun, avoid collisions, and preserve ships. Rocks fragment, cleared fields are replaced, inertia "
            "continues without thrust, and bodies wrap at boundaries. NOOP is not a brake. LEFT increments and RIGHT decrements the orientation code, typically every four frames. Four starting ships; "
            "extra ships follow cartridge rules. The objective is the sum of original unclipped ALE rewards "
            "before true game over or 18000 emulated frames (300 seconds at 60 Hz). Score rollover is handled by ALE; "
            "survival and lives are separate diagnostics, never score bonuses. FIRE is a button level, not an automatic "
            "rapid-fire macro; release then repress FIRE to shoot repeatedly; holding it does not continuously shoot. DOWN invokes mode-0 hyperspace, not reverse thrust.\n"
            "The runner calls tick every 1/60 second, applying one current action for exactly one native frame. "
            "There is no background loop or sleeping: the runner must keep ticking while models think. Repeated "
            "plain action names remain held. Optional ACTION@FRAME (e.g. UP@120) expires after six frames relative "
            "to that observed episode frame; future, expired, or malformed actions apply NOOP while time advances. "
            "Use short bursts and a local controller; timestamps are simulated frames, not wall-clock time.\n"
            "Observation is JSON: frame includes the prelude; ticks counts only external tick calls; score is cumulative "
            "reward; displayed_score wraps at 100000; lives are authoritative native lives. elapsed_s and frames_remaining "
            "include all prelude frames. settings and prelude disclose the scenario and its actual reward/life costs. "
            "The default bounded-launch-v1 catalogue holds UPRIGHT for 60+4*seed frames, seeds 0..15, then gives "
            "control to you. Even seeds train, odd seeds held out: both sample the same launch family. This compensates "
            "for the ROM's identical seeded canonical starts; it is a disclosed robustness task, not random new worlds. "
            "canonical suite disables that prelude for separately reported fixed-start runs.\n"
            "player has presence, pixel x/y, approximate width/height and orientation_code: 0 up, 4 left, 8 down, "
            "12 right; 16 possible codes. Intermediate heading angles are not calibrated. asteroids and "
            "player_missiles contain source_slot, geometry, raw_xy, size/direction codes and estimated "
            "velocity_px_per_frame. Source slots can be reused; they are not permanent physical identities. "
            "Motion uses up to four prior frames; null means insufficient history or a discontinuity, not zero. "
            "Large jumps/wraps, respawns and size changes invalidate velocity rather than inventing a wrap period. "
            "Geometry is approximate and unclipped, not collision masks; flicker can hide active objects in one RGB frame. Enemy craft/projectiles are not decoded: "
            "null means unknown. Raw 128-byte RAM is included by default for transparency. observation_frame stamps "
            "the immutable snapshot, age is zero at emission; compare its frame to the latest frame for decision age. "
            "Only mode 0/difficulty 0 is supported. snapshot_png() can explicitly save a real RGB image for debugging.\n"
            + self.action_spec.describe())

    def snapshot_png(self) -> str:
        """Explicit diagnostic, never called by observe; no Pillow dependency."""
        if self._closed or not self._started:
            raise RuntimeError("Snapshot requires an open, reset game")
        image = self._env.unwrapped.ale.getScreenRGB()
        h, w, _ = image.shape
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        data = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
               chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b""))
        directory = HERE / "runtime" / "screenshots"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.scenario_suite}-seed{self._seed}-frame{self._frame}-{hashlib.sha256(png).hexdigest()[:12]}.png"
        path.write_bytes(png)
        return str(path)

    def close(self):
        if self._closed:
            return
        self._closed = True
        env, self._env = self._env, None
        if env is not None:
            env.close()


def make_game(**kwargs):
    return AsteroidsGame(**kwargs)
