"""Realtime adaptor for the Atari 2600 Space Invaders ROM.

The emulator's observation is RAM (the cheap, canonical ALE channel).  ALE does
not publish Space Invaders object symbols, so the adaptor reads the emulator's
current RGB buffer only inside this module and turns the ROM-specific palette
into compact JSON objects for the harness.
"""
from __future__ import annotations

import threading
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np

from metaharness.games.base import ActionSpec


TIER = 2
# Sticky actions are deliberately enabled at the standard v5 probability.  The
# first frame is a fixed cartridge opening, but the seeded sticky-action RNG
# makes otherwise identical action sequences diverge between scenarios.
DETERMINISTIC = False
SEEDS = {
    "train": [0, 1, 2, 3],
    "heldout": [100, 101],
}
EVAL_KWARGS = {
    "mode": 0,
    "difficulty": 0,
    "frameskip": 4,
    "repeat_action_probability": 0.25,
}


# Gym's compressed six-action set for this ROM.  Do not substitute the numeric
# values of ale_py.Action: Gym maps its Discrete values to the minimal ALE set.
_ACTIONS = {
    "noop": 0,
    "fire": 1,
    "right": 2,
    "left": 3,
    "right_fire": 4,
    "left_fire": 5,
}

_ACTION_SPEC = ActionSpec(
    names=list(_ACTIONS),
    descriptions={
        "noop": "Release the joystick and fire button; the formation and shots continue moving.",
        "fire": "Press FIRE without moving; only one player laser can be in flight.",
        "right": "Hold the joystick RIGHT for this emulator tick.",
        "left": "Hold the joystick LEFT for this emulator tick.",
        "right_fire": "Move RIGHT and press FIRE on the same tick.",
        "left_fire": "Move LEFT and press FIRE on the same tick.",
    },
    noop="noop",
    notes=(
        "Each tick advances four NTSC Atari frames by default (~66.7 ms). "
        "A tick with no fresh valid action repeats the last valid action; before "
        "the first action it uses noop. Reversing is allowed, but bombs and the "
        "descending formation continue during every tick."
    ),
)


class _ObjectDecoder:
    """Decode the canonical NTSC ROM's RGB palette into object-level state.

    This is intentionally a ROM-specific decoder rather than a claim that ALE's
    128 RAM bytes have a public object schema.  Formation movement is estimated
    by matching the known six-by-six sprite lattice from the previous frame.
    """

    INVADER = (134, 134, 29)
    SHIELD = (181, 83, 40)
    PLAYER = (50, 132, 50)
    READY_PLAYER = (162, 134, 56)
    PROJECTILE = (142, 142, 142)
    UFO = (151, 25, 122)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ox = 0
        self._oy = 0
        self._initialized = False
        self._last_alive = 0
        self._previous_projectiles: list[tuple[float, float]] = []

    @staticmethod
    def _mask(image: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
        return np.all(image == np.asarray(colour, dtype=np.uint8), axis=2)

    @staticmethod
    def _count(mask: np.ndarray, x: int, y: int, width: int = 12, height: int = 14) -> int:
        h, w = mask.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + width), min(h, y + height)
        if x0 >= x1 or y0 >= y1:
            return 0
        return int(mask[y0:y1, x0:x1].sum())

    def _match_formation(self, invader: np.ndarray, pixel_count: int) -> tuple[int, int, list[list[int]]]:
        # A new wave starts near the canonical origin.  At ordinary speeds the
        # formation moves only a few pixels between four-frame observations;
        # broad searches are reserved for the first/new full formation.
        broad = (not self._initialized) or (self._last_alive <= 2 and pixel_count > 500)
        if broad:
            x_candidates = range(-6, 7)
            y_candidates = range(-4, 13)
        else:
            x_candidates = range(self._ox - 6, self._ox + 7)
            y_candidates = range(self._oy - 4, self._oy + 5)

        best: tuple[float, int, int, list[list[int]]] | None = None
        for ox in x_candidates:
            for oy in y_candidates:
                counts = [
                    [self._count(invader, 20 + ox + 16 * col, 29 + oy + 18 * row)
                     for col in range(6)]
                    for row in range(6)
                ]
                total = sum(sum(row) for row in counts)
                # Prefer the prior lattice on a near-tie.  The actual sprite
                # contains 30--46 pixels, while an incorrectly shifted window
                # loses enough pixels to make this a useful alignment score.
                distance = abs(ox - self._ox) + abs(oy - self._oy)
                score = float(total) - 0.001 * distance
                if best is None or score > best[0]:
                    best = (score, ox, oy, counts)

        if best is None:
            return self._ox, self._oy, [[0] * 6 for _ in range(6)]
        _, self._ox, self._oy, counts = best
        self._initialized = True
        alive = sum(1 for row in counts for value in row if value >= 8)
        self._last_alive = alive
        return self._ox, self._oy, counts

    @staticmethod
    def _x_components(mask: np.ndarray, y0: int, y1: int) -> list[tuple[int, int, int]]:
        """Return (xmin, xmax, pixel_count) for colored lower-screen objects."""
        region = mask[max(0, y0):min(mask.shape[0], y1), :]
        xs = np.flatnonzero(np.any(region, axis=0))
        if not len(xs):
            return []
        starts = xs[np.r_[True, np.diff(xs) > 2]]
        ends = xs[np.r_[np.diff(xs) > 2, True]]
        return [(int(start), int(end), int(region[:, start:end + 1].sum()))
                for start, end in zip(starts, ends)]

    def _player(self, image: np.ndarray) -> dict[str, int]:
        # During the cartridge's ready/respawn frames the player sprite uses
        # the yellow palette; during play it is green.  Restricting to the
        # bottom playfield excludes the score text and reserve-ship glyphs.
        green = self._mask(image, self.PLAYER)
        yellow = self._mask(image, self.READY_PLAYER)
        # The ROM shows a yellow ready/player glyph while the green spare-ship
        # glyph may flicker in the same bottom band.  Prefer yellow whenever it
        # is present; once play begins it disappears and green is the live ship.
        candidates = self._x_components(yellow, 180, 195)
        if not candidates:
            candidates = self._x_components(green, 180, 195)
        if not candidates:
            return {"x": 80, "y": 190}
        # A player sprite is the widest lower component.  Ties are resolved by
        # the component closest to the center of the 160-pixel playfield.
        xmin, xmax, _ = max(candidates, key=lambda c: (c[1] - c[0], -abs((c[0] + c[1]) / 2 - 80)))
        return {"x": int(round((xmin + xmax) / 2)), "y": 190}

    @staticmethod
    def _components(mask: np.ndarray) -> list[dict[str, int]]:
        """Small connected-component finder for the gray projectile palette."""
        ys, xs = np.where(mask)
        remaining = {(int(y), int(x)) for y, x in zip(ys, xs)}
        components: list[dict[str, int]] = []
        while remaining:
            start = remaining.pop()
            stack = [start]
            points = [start]
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not dx and not dy:
                            continue
                        p = (y + dy, x + dx)
                        if p in remaining:
                            remaining.remove(p)
                            stack.append(p)
                            points.append(p)
            py = [p[0] for p in points]
            px = [p[1] for p in points]
            components.append({
                "x": int(round(sum(px) / len(px))),
                "y": int(round(sum(py) / len(py))),
                "xmin": min(px),
                "xmax": max(px),
                "ymin": min(py),
                "ymax": max(py),
                "pixels": len(points),
            })
        return components

    def _projectiles(self, image: np.ndarray, player_x: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        gray = self._mask(image, self.PROJECTILE)
        # Score text is above y20.  y180 is the playfield boundary; the
        # bottom border and reserve ships are not projectiles.
        roi = np.zeros_like(gray)
        roi[20:180, :] = gray[20:180, :]
        components = self._components(roi)
        shot_candidates: list[dict[str, Any]] = []
        bombs: list[dict[str, Any]] = []
        for component in components:
            x, y = component["x"], component["y"]
            prior = min(
                self._previous_projectiles,
                key=lambda p: abs(p[0] - x) + abs(p[1] - y),
                default=None,
            )
            delta_y = y - prior[1] if prior is not None and abs(prior[0] - x) <= 8 and abs(prior[1] - y) <= 28 else 0
            # A laser rises from the cannon; bombs descend from the formation.
            # A fresh component close to the cannon is consequently a useful
            # fallback when a sprite was flickered in the preceding frame.
            is_shot = delta_y < -1 or (delta_y == 0 and y >= 154 and abs(x - player_x) <= 12)
            if is_shot:
                shot_candidates.append(component)
            elif y >= 28:
                bombs.append({"x": int(x), "y": int(y), "velocity": "down"})

        self._previous_projectiles = [(float(c["x"]), float(c["y"])) for c in components]
        if shot_candidates:
            shot = min(shot_candidates, key=lambda c: (abs(c["x"] - player_x), -c["y"]))
            player_shot = {"active": True, "x": int(shot["x"]), "y": int(shot["y"])}
        else:
            player_shot = {"active": False, "x": None, "y": None}
        # A few overlapping gray scanline fragments can describe the same
        # bomb.  The component finder normally already merges them; cap the
        # public list so an explosion cannot make observations large.
        return player_shot, bombs[:12]

    def decode(self, image: np.ndarray, *, frame: int, score: float, lives: int,
               game_over: bool, wave: int, last_action_error: str | None) -> dict[str, Any]:
        invader = self._mask(image, self.INVADER)
        pixel_count = int(invader.sum())
        ox, oy, counts = self._match_formation(invader, pixel_count)
        alive = [[count >= 8 for count in row] for row in counts]
        player = self._player(image)
        player_shot, bombs = self._projectiles(image, player["x"])

        invaders: list[dict[str, Any]] = []
        for row in range(6):
            for col in range(6):
                invaders.append({
                    "row": row + 1,
                    "col": col,
                    "x": int(26 + ox + 16 * col),
                    "y": int(35 + oy + 18 * row),
                    "alive": bool(alive[row][col]),
                    "value": int(5 * (row + 1)),
                })

        red = self._mask(image, self.SHIELD)
        shields: list[dict[str, Any]] = []
        for index in range(3):
            x0, y0 = 42 + 32 * index, 157
            rows: list[int] = []
            for row in range(9):
                bits = 0
                for col in range(8):
                    if bool(red[y0 + 2 * row:y0 + 2 * row + 2, x0 + col].any()):
                        bits |= 1 << (7 - col)
                rows.append(int(bits))
            shields.append({
                "x": int(x0 + 4),
                "y": 166,
                "health_mask": rows,
                "pixels": int(red[y0:y0 + 18, x0:x0 + 8].sum()),
            })

        purple = self._mask(image, self.UFO)
        # The normal UFO is rendered in the purple palette in the score-line
        # band (y12--19); this is distinct from the green/yellow score glyphs.
        purple[0:10] = False
        purple[20:] = False
        ufo_y, ufo_x = np.where(purple)
        ufo = {
            "active": bool(len(ufo_x)),
            "x": int(round(float(ufo_x.mean()))) if len(ufo_x) else None,
            "y": int(round(float(ufo_y.mean()))) if len(ufo_y) else None,
        }

        return {
            "observation_version": 1,
            "frame": int(frame),
            "score": float(score),
            "lives": int(lives),
            "game_over": bool(game_over),
            "wave": int(wave),
            "player": player,
            "player_shot": player_shot,
            "invaders": invaders,
            "bombs": bombs,
            "shields": shields,
            "ufo": ufo,
            "last_action_error": last_action_error,
        }


class SpaceInvadersGame:
    """A synchronous RealtimeGame backed by one headless ALE instance."""

    name = "Atari 2600 Space Invaders"
    action_spec = _ACTION_SPEC

    def __init__(self, *, mode: int = 0, difficulty: int = 0, frameskip: int = 4,
                 repeat_action_probability: float = 0.25, max_ticks: int | None = None,
                 **_: Any) -> None:
        self.mode = int(mode)
        self.difficulty = int(difficulty)
        self.frameskip = max(1, int(frameskip))
        self.repeat_action_probability = float(repeat_action_probability)
        self.tick_s = self.frameskip / 60.0
        self.max_ticks = None if max_ticks is None else max(1, int(max_ticks))
        self._lock = threading.RLock()
        self._env: Any = None
        self._closed = False
        self._seed: int | None = None
        self._ticks = 0
        self._frame = 0
        self._score = 0.0
        self._lives = 3
        self._done = False
        self._truncated = False
        self._terminal_cause: str | None = None
        self._invalid_reason: str | None = None
        self._last_action = "noop"
        self._last_action_error: str | None = None
        self._last_reward = 0.0
        self._previous_alive: int | None = None
        self._wave = 1
        self._waves_cleared = 0
        self._invaders_killed = 0
        self._ufo_hits = 0
        self._decoder = _ObjectDecoder()
        self._observation: dict[str, Any] = self._empty_observation()
        self._build_env()

    def _build_env(self) -> None:
        gym.register_envs(ale_py)
        self._env = gym.make(
            "ALE/SpaceInvaders-v5",
            obs_type="ram",
            frameskip=self.frameskip,
            repeat_action_probability=self.repeat_action_probability,
            mode=self.mode,
            difficulty=self.difficulty,
        )
        self._closed = False

    def _empty_observation(self) -> dict[str, Any]:
        return {
            "observation_version": 1,
            "mode": int(self.mode),
            "difficulty": int(self.difficulty),
            "frame": 0,
            "score": 0.0,
            "lives": 3,
            "game_over": False,
            "terminal_cause": None,
            "wave": 1,
            "waves_cleared": 0,
            "invaders_killed": 0,
            "player": {"x": 80, "y": 190},
            "player_shot": {"active": False, "x": None, "y": None},
            "invaders": [],
            "bombs": [],
            "shields": [],
            "ufo": {"active": False, "x": None, "y": None},
            "last_action_error": None,
        }

    def reset(self, seed: int) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._env is None:
                self._build_env()
            self._seed = int(seed)
            self._ticks = 0
            self._frame = 0
            self._score = 0.0
            self._lives = 3
            self._done = False
            self._truncated = False
            self._terminal_cause = None
            self._invalid_reason = None
            self._last_action = "noop"
            self._last_action_error = None
            self._last_reward = 0.0
            self._previous_alive = None
            self._wave = 1
            self._waves_cleared = 0
            self._invaders_killed = 0
            self._ufo_hits = 0
            self._decoder.reset()
            _, info = self._env.reset(seed=self._seed)
            self._lives = int(info.get("lives", 3))
            self._refresh_observation()
            return self._observation

    def _refresh_observation(self) -> None:
        if self._env is None:
            return
        image = np.asarray(self._env.unwrapped.ale.getScreenRGB())
        decoded = self._decoder.decode(
            image,
            frame=self._frame,
            score=self._score,
            lives=self._lives,
            game_over=self._done,
            wave=self._wave,
            last_action_error=self._last_action_error,
        )
        alive = sum(1 for item in decoded["invaders"] if item["alive"])
        if self._previous_alive is not None:
            if alive < self._previous_alive:
                self._invaders_killed += self._previous_alive - alive
            if not self._done and self._previous_alive > 0 and alive == 0:
                self._waves_cleared += 1
                self._wave += 1
            elif not self._done and self._previous_alive <= 2 and alive >= 30:
                # A short transition can be skipped by frameskip.
                self._waves_cleared += 1
                self._wave += 1
        self._previous_alive = alive
        decoded["mode"] = int(self.mode)
        decoded["difficulty"] = int(self.difficulty)
        decoded["wave"] = int(self._wave)
        decoded["waves_cleared"] = int(self._waves_cleared)
        decoded["invaders_killed"] = int(self._invaders_killed)
        decoded["game_over"] = bool(self._done)
        decoded["terminal_cause"] = self._terminal_cause
        decoded["score"] = float(self._score)
        decoded["lives"] = int(self._lives)
        decoded["last_action_error"] = self._last_action_error
        self._observation = decoded

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return self._observation

    def _set_terminal(self, *, terminated: bool, truncated: bool, info: dict[str, Any]) -> None:
        self._truncated = bool(truncated)
        self._done = bool(terminated or truncated)
        if truncated:
            self._terminal_cause = "time_limit"
        elif terminated:
            self._terminal_cause = "all_lives_lost" if self._lives <= 0 else "game_over"

    def tick(self, action: Any) -> None:
        """Post an action and advance one real-time ALE tick.

        Invalid actions are ignored rather than escaping as an evaluator game
        error.  The error is visible in the next JSON observation.
        """
        with self._lock:
            if self._done or self._env is None:
                return
            if isinstance(action, str) and action in _ACTIONS:
                self._last_action = action
                self._last_action_error = None
            elif isinstance(action, (int, np.integer)) and int(action) in range(6):
                self._last_action = next(name for name, value in _ACTIONS.items() if value == int(action))
                self._last_action_error = None
            elif action is not None:
                self._last_action_error = f"ignored malformed action {str(action)[:120]!r}"

            try:
                _, reward, terminated, truncated, info = self._env.step(_ACTIONS[self._last_action])
                self._last_reward = float(reward)
                self._score += float(reward)
                self._ticks += 1
                self._frame = int(info.get("episode_frame_number", info.get("frame_number", self._ticks * self.frameskip)))
                self._lives = int(info.get("lives", self._lives))
                if float(reward) == 200.0:
                    self._ufo_hits += 1
                self._set_terminal(terminated=bool(terminated), truncated=bool(truncated), info=info)
                if self.max_ticks is not None and self._ticks >= self.max_ticks and not self._done:
                    self._done = True
                    self._truncated = True
                    self._terminal_cause = "time_limit"
                self._refresh_observation()
            except Exception as exc:  # infrastructure failure: keep the protocol alive
                self._done = True
                self._invalid_reason = f"ALE step failed: {type(exc).__name__}: {exc}"[:500]
                self._terminal_cause = "infrastructure_error"
                self._last_action_error = self._invalid_reason
                try:
                    self._refresh_observation()
                except Exception:
                    # Preserve the last valid symbolic state if the emulator
                    # failed before its screen buffer could be read.
                    fallback = dict(self._observation)
                    fallback["game_over"] = True
                    fallback["terminal_cause"] = self._terminal_cause
                    fallback["last_action_error"] = self._invalid_reason
                    fallback["score"] = float(self._score)
                    fallback["lives"] = int(self._lives)
                    self._observation = fallback

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._score)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "valid_trial": self._invalid_reason is None,
                "invalid_reason": self._invalid_reason,
                "seed": self._seed,
                "mode": self.mode,
                "difficulty": self.difficulty,
                "frameskip": self.frameskip,
                "repeat_action_probability": self.repeat_action_probability,
                "frame": self._frame,
                "ticks": self._ticks,
                "last_action": self._last_action,
                "last_reward": self._last_reward,
                "terminal_cause": self._terminal_cause,
                "lives_remaining": self._lives,
                "waves_cleared": self._waves_cleared,
                "wave": self._wave,
                "invaders_killed": self._invaders_killed,
                "ufo_hits": self._ufo_hits,
                "truncated": self._truncated,
            }

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            return {
                "atari_score": float(self._score),
                "lives_remaining": float(self._lives),
                "waves_cleared": float(self._waves_cleared),
                "invaders_killed": float(self._invaders_killed),
                "ufo_hits": float(self._ufo_hits),
                "survival_ticks": float(self._ticks),
            }

    def describe(self) -> str:
        return (
            "Atari 2600 Space Invaders running locally in ALE/Stella, canonical "
            f"mode={self.mode}, difficulty={self.difficulty}. The world is real-time: "
            f"each tick advances {self.frameskip} NTSC frames ({self.tick_s:.4f} s), "
            "even if the controller repeats or misses a decision. Clear the six-by-six "
            "invader formation, survive with three lives, and maximize the ALE score. "
            "Invader values are 5, 10, 15, 20, 25, and 30 points from top row to "
            "bottom row; a normal UFO is worth 200 and a complete 36-invader wave is "
            "630 points. The game ends when all lives are lost.\n\n"
            "Actions are strings from action_spec: noop, fire, right, left, "
            "right_fire, and left_fire. Combined actions move and fire on the same "
            "tick. The player has one laser in flight at a time; firing while it is "
            "active does not create a second beam. The formation moves sideways, "
            "descends at an edge, and drops gray bombs; shields absorb some hits but "
            "are destructible. A late action cannot rewind the emulator.\n\n"
            "Observation coordinates are the 160x210 NTSC screen coordinate system "
            "(x=0..159, y=0..209), not a screenshot. Fields are versioned JSON: mode, "
            "difficulty, frame, score, lives, game_over, terminal_cause, wave, waves_cleared, "
            "invaders_killed, player, player_shot, 36 invaders with row/col/x/y/alive/value, "
            "bombs, three shields, ufo, and any malformed "
            "action warning. Shield health_mask has nine integer rows; each row is an "
            "8-bit left-to-right occupancy mask. Bomb and laser positions are decoded "
            "from the ROM's gray projectile pixels. The primary Gym observation is RAM; "
            "RGB is read internally only for this symbolic decoder.\n\n"
            "reset(seed) selects ALE's seeded emulator/sticky-action RNG. The benchmark "
            "uses explicit frameskip=4 and repeat_action_probability=0.25 with mode and "
            "difficulty fixed at zero; changing cartridge modes or difficulty can alter "
            "palette, shields, bomb rules, or multiplayer behavior and is not covered "
            "by the canonical decoder."
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            env, self._env = self._env, None
            if env is not None:
                try:
                    env.close()
                except Exception:
                    # close() is deliberately idempotent and never blocks a
                    # validator on a best-effort ALE cleanup.
                    pass


def make_game(**kwargs: Any) -> SpaceInvadersGame:
    """Construct a fresh headless Atari 2600 Space Invaders episode client."""
    return SpaceInvadersGame(**kwargs)
