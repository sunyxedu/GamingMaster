"""Tier-2 ViZDoom adaptor for classic Doom and compatible Freedoom IWADs.

The evaluator talks to :class:`DoomRealtimeGame`; it never receives the native
ViZDoom object.  A single controller thread owns calls which advance the native
engine.  ``tick`` is a mailbox update and therefore remains cheap while the
engine continues to consume the last action at 35 Doom tics per second.
"""
from __future__ import annotations

import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from metaharness.games.base import ActionSpec

TIER = 2
# Native async timing and the controller thread make scheduler-level first-state
# timing nondeterministic even though ViZDoom's seed is applied to every episode.
DETERMINISTIC = False

# The two groups deliberately use different RNG seeds.  Their map families
# overlap, while held-out also adds unseen maps; this tests new instances rather
# than making the whole held-out group a different game.
SEEDS = {
    "train": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
               1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018],
    "heldout": [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
                2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
}

TICRATE = 35.0
DEFAULT_TIMEOUT_TICS = 126_000
_INIT_CWD_LOCK = threading.Lock()

_FAMILY_MAPS: dict[str, tuple[str, ...]] = {
    "doom1": tuple(f"E{episode}M{mission}" for episode in range(1, 5) for mission in range(1, 10)),
    "freedoom1": tuple(f"E{episode}M{mission}" for episode in range(1, 5) for mission in range(1, 10)),
    "doom2": tuple(f"MAP{i:02d}" for i in range(1, 33)),
    "freedoom2": tuple(f"MAP{i:02d}" for i in range(1, 33)),
}

# Early warm-up maps are intentionally small.  Held-out retains that family
# mix and adds unseen maps, rather than turning held-out into a transfer-only
# test.  VIZDOOM_SKILL=5 is the documented separate stress evaluation.
_TRAIN_MAPS: dict[str, tuple[str, ...]] = {
    "doom1": ("E1M1", "E1M2", "E1M3", "E1M4"),
    "freedoom1": ("E1M1", "E1M2", "E1M3", "E1M4"),
    "doom2": ("MAP01", "MAP02", "MAP03", "MAP04"),
    "freedoom2": ("MAP01", "MAP02", "MAP03", "MAP04"),
}
_HELDOUT_MAPS: dict[str, tuple[str, ...]] = {
    "doom1": tuple(f"E1M{i}" for i in range(1, 9)),
    "freedoom1": tuple(f"E1M{i}" for i in range(1, 9)),
    "doom2": tuple(f"MAP{i:02d}" for i in range(1, 9)),
    "freedoom2": tuple(f"MAP{i:02d}" for i in range(1, 9)),
}

# Values are deliberately bounded secondary progress references, not map
# completion claims.  The ViZDoom API exposes current counts but not per-map
# totals; exact percentages are therefore null unless a caller supplies totals.
_PROGRESS_REFERENCE = {"kills": 30.0, "items": 30.0, "secrets": 5.0}

_ACTION_NAMES = (
    "noop",
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "strafe_left",
    "strafe_right",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
    "fire",
    "fire_forward",
    "fire_strafe_left",
    "fire_strafe_right",
    "use",
    "next_weapon",
    "prev_weapon",
    "select_weapon_1",
    "select_weapon_2",
    "select_weapon_3",
    "select_weapon_4",
    "select_weapon_5",
    "select_weapon_6",
    "select_weapon_7",
)

_ACTION_DESCRIPTIONS = {
    "noop": "Hold no buttons; safe initial and missed-decision action.",
    "forward": "Run forward while keeping the current view.",
    "backward": "Run backward; useful for retreating from a threat.",
    "turn_left": "Turn left in place.",
    "turn_right": "Turn right in place.",
    "strafe_left": "Run sideways to the left without turning.",
    "strafe_right": "Run sideways to the right without turning.",
    "forward_left": "Run forward and strafe left (a moving turn/circle).",
    "forward_right": "Run forward and strafe right (a moving turn/circle).",
    "backward_left": "Run backward and strafe left.",
    "backward_right": "Run backward and strafe right.",
    "fire": "Fire the selected weapon without moving.",
    "fire_forward": "Fire while running forward.",
    "fire_strafe_left": "Fire while strafing left.",
    "fire_strafe_right": "Fire while strafing right.",
    "use": "Pulse USE once for a door, switch, or exit.",
    "next_weapon": "Pulse selection of the next available weapon.",
    "prev_weapon": "Pulse selection of the previous available weapon.",
    "select_weapon_1": "Pulse selection of weapon slot 1 (if owned).",
    "select_weapon_2": "Pulse selection of weapon slot 2 (if owned).",
    "select_weapon_3": "Pulse selection of weapon slot 3 (if owned).",
    "select_weapon_4": "Pulse selection of weapon slot 4 (if owned).",
    "select_weapon_5": "Pulse selection of weapon slot 5 (if owned).",
    "select_weapon_6": "Pulse selection of weapon slot 6 (if owned).",
    "select_weapon_7": "Pulse selection of weapon slot 7 (if owned).",
}

_ACTION_SPEC = ActionSpec(
    names=_ACTION_NAMES,
    descriptions=_ACTION_DESCRIPTIONS,
    noop="noop",
    notes=(
        "Doom runs at 35 tics/s. Movement and fire actions are held until a new action is "
        "posted; USE and weapon-selection actions are one-tic pulses. The optional compact "
        "grammar is 'move <forward> <turn>', where each integer is -1, 0, or 1 and positive "
        "turn means right, or 'select <1-7>'. Unknown or malformed actions are ignored and "
        "reported in the next observation."
    ),
)


class _Command:
    __slots__ = ("label", "vector", "pulse")

    def __init__(self, label: str, vector: list[int], pulse: bool = False) -> None:
        self.label = label
        self.vector = vector
        self.pulse = pulse


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _int_value(value: Any, default: int = 0) -> int:
    return int(round(_finite(value, float(default))))


def _bool_value(value: Any) -> bool:
    return bool(_int_value(value))


def _family_from_path(path: Path, explicit: str | None = None) -> str:
    stem = path.stem.lower()
    inferred: str | None = None
    if "freedoom1" in stem:
        inferred = "freedoom1"
    elif "freedoom2" in stem:
        inferred = "freedoom2"
    elif "doom2" in stem:
        inferred = "doom2"
    elif stem == "doom" or stem.startswith("doom_") or stem.endswith("_doom"):
        inferred = "doom1"
    if explicit:
        family = explicit.strip().lower()
        if family not in _FAMILY_MAPS:
            raise ValueError(
                "VIZDOOM_IWAD_FAMILY must be one of doom1, doom2, freedoom1, freedoom2"
            )
        if inferred is not None and family != inferred:
            raise ValueError(
                f"IWAD {path.name!r} identifies as {inferred}, but "
                f"VIZDOOM_IWAD_FAMILY={family}; refusing to mix Doom and Freedoom families"
            )
        return family
    if inferred is not None:
        return inferred
    raise ValueError(
        f"cannot identify the IWAD family from {path.name!r}; set VIZDOOM_IWAD_FAMILY "
        "to doom1, doom2, freedoom1, or freedoom2 rather than mixing map families"
    )


def _resolve_wad() -> tuple[Path, str, bool]:
    """Resolve a user IWAD, or copy the package's explicit Freedoom fallback into runtime."""
    env_present = "VIZDOOM_WAD_PATH" in os.environ
    raw = os.environ.get("VIZDOOM_WAD_PATH", "").strip()
    if env_present and not raw:
        raise RuntimeError(
            "VIZDOOM_WAD_PATH is set but empty; provide an absolute path to doom.wad, "
            "doom2.wad, or a compatible Freedoom IWAD"
        )
    if raw:
        wad = Path(raw).expanduser().resolve()
        if not wad.is_file():
            raise FileNotFoundError(
                f"VIZDOOM_WAD_PATH does not point to a file: {wad}. "
                "Use a legally obtained Doom IWAD or Freedoom IWAD."
            )
        family = _family_from_path(wad, os.environ.get("VIZDOOM_IWAD_FAMILY"))
        return wad, family, False

    # ViZDoom 1.3.2 ships Freedoom assets.  Copying, rather than referencing,
    # the site-package path keeps all runtime assets below this adaptor and
    # makes the no-account fallback explicit in info()/ADAPTOR.md.
    try:
        import vizdoom
    except ImportError as exc:  # pragma: no cover - requirements normally installs it
        raise RuntimeError(
            "ViZDoom is not installed. Install requirements.txt in the evaluator venv, "
            "or set VIZDOOM_WAD_PATH to a legal IWAD."
        ) from exc
    source = Path(vizdoom.__file__).resolve().parent / "freedoom2.wad"
    if not source.is_file():
        raise FileNotFoundError(
            "VIZDOOM_WAD_PATH is unset and the ViZDoom package has no bundled freedoom2.wad; "
            "set VIZDOOM_WAD_PATH to a legal IWAD"
        )
    runtime = Path(__file__).resolve().parent / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    target = runtime / "freedoom2.wad"
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    configured_family = os.environ.get("VIZDOOM_IWAD_FAMILY", "").strip().lower()
    if configured_family and configured_family != "freedoom2":
        raise ValueError(
            "VIZDOOM_WAD_PATH is unset, so the bundled asset is freedoom2.wad; "
            "VIZDOOM_IWAD_FAMILY must be freedoom2 or omitted"
        )
    return target, "freedoom2", True


def _seed_group(seed: int) -> str:
    if seed in SEEDS["train"]:
        return "train"
    if seed in SEEDS["heldout"]:
        return "heldout"
    return "custom"


class DoomRealtimeGame:
    """One fresh, headless ViZDoom episode controlled at the Doom tic cadence."""

    name = "Doom (ViZDoom ASYNC_PLAYER)"
    action_spec = _ACTION_SPEC
    tick_s = 1.0 / TICRATE

    def __init__(
        self,
        *,
        wad_path: str | None = None,
        map_name: str | None = None,
        skill: int | None = None,
        episode_timeout: int | None = None,
        mode: str | None = None,
    ) -> None:
        # Resolve at construction so load_adaptor gives an actionable error,
        # while actual engine creation remains inside reset(seed).
        if wad_path is not None:
            old = os.environ.get("VIZDOOM_WAD_PATH")
            try:
                os.environ["VIZDOOM_WAD_PATH"] = str(wad_path)
                self._wad_path, self._family, self._bundled_fallback = _resolve_wad()
            finally:
                if old is None:
                    os.environ.pop("VIZDOOM_WAD_PATH", None)
                else:
                    os.environ["VIZDOOM_WAD_PATH"] = old
        else:
            self._wad_path, self._family, self._bundled_fallback = _resolve_wad()
        self._explicit_map = (map_name or os.environ.get("VIZDOOM_MAP", "")).strip() or None
        raw_skill = skill if skill is not None else os.environ.get("VIZDOOM_SKILL", "")
        self._explicit_skill = int(raw_skill) if str(raw_skill).strip() else None
        if self._explicit_skill is not None and not 1 <= self._explicit_skill <= 5:
            raise ValueError("VIZDOOM_SKILL must be an integer from 1 through 5")
        raw_timeout = episode_timeout if episode_timeout is not None else os.environ.get(
            "VIZDOOM_EPISODE_TIMEOUT_TICS", str(DEFAULT_TIMEOUT_TICS)
        )
        self._episode_timeout = int(raw_timeout)
        if self._episode_timeout <= 0:
            raise ValueError("VIZDOOM_EPISODE_TIMEOUT_TICS must be positive")
        requested_mode = (mode or os.environ.get("VIZDOOM_MODE", "ASYNC_PLAYER")).upper()
        if requested_mode != "ASYNC_PLAYER":
            raise ValueError(
                "This tier-2 adaptor requires VIZDOOM_MODE=ASYNC_PLAYER; PLAYER is only a "
                "synchronous fallback and would violate the real-time contract"
            )
        self._mode_name = requested_mode

        import threading as _threading  # keeps module import simple for adaptor loaders

        self._state_lock = _threading.RLock()
        self._native_lock = _threading.RLock()
        self._action_lock = _threading.Lock()
        self._lifecycle_lock = _threading.RLock()
        self._stop_event = _threading.Event()
        self._engine_thread: threading.Thread | None = None
        self._game: Any = None
        self._closed = False
        self._done = False
        self._valid_trial = True
        self._invalid_reason: str | None = None
        self._terminal_reason: str | None = None
        self._last_error: str | None = None
        self._last_state: Any = None
        self._last_reward = 0.0
        self._total_reward = 0.0
        self._engine_tics = 0
        self._ticks = 0
        self._start_time = 0.0
        self._seed: int | None = None
        self._map = ""
        self._skill = 3
        self._variable_names: list[str] = []
        self._button_names: list[str] = []
        self._button_index: dict[str, int] = {}
        self._action_epoch = 0
        self._pulse_epoch: int | None = None
        self._current_command = _Command("noop", [])

    def _scenario(self, seed: int) -> tuple[str, int]:
        if self._explicit_map:
            map_name = self._explicit_map.upper()
        else:
            group = _seed_group(seed)
            maps = _TRAIN_MAPS[self._family] if group == "train" else _HELDOUT_MAPS[self._family]
            if group == "custom":
                maps = _HELDOUT_MAPS[self._family]
            map_name = maps[abs(int(seed)) % len(maps)]
        if map_name not in _FAMILY_MAPS[self._family]:
            raise ValueError(
                f"map {map_name!r} is not valid for {self._family}; Doom and Freedoom map "
                "families are kept separate"
            )
        if self._explicit_skill is not None:
            skill = self._explicit_skill
        elif _seed_group(seed) == "train":
            skill = 2 if abs(seed) % 2 else 3
        elif _seed_group(seed) == "heldout":
            skill = 4
        else:
            skill = 3
        return map_name, skill

    @staticmethod
    def _variable_list(vzd: Any) -> list[tuple[str, Any]]:
        names = [
            "HEALTH", "ARMOR", "DEAD", "ON_GROUND", "ATTACK_READY",
            "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO",
            *(f"AMMO{i}" for i in range(10)),
            *(f"WEAPON{i}" for i in range(10)),
            "KILLCOUNT", "ITEMCOUNT", "SECRETCOUNT", "DAMAGE_TAKEN",
            "POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE",
            "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z",
        ]
        return [(name, getattr(vzd.GameVariable, name)) for name in names]

    def _create_game(self, seed: int, map_name: str, skill: int) -> Any:
        import vizdoom as vzd

        scenarios = Path(vzd.__file__).resolve().parent / "scenarios"
        cfg_name = f"{self._family}.cfg"
        cfg = scenarios / cfg_name
        if not cfg.is_file():
            # The official doom configs are equivalent for a compatible custom IWAD.
            cfg = scenarios / ("doom.cfg" if self._family.endswith("1") else "doom2.cfg")
        # ViZDoom creates/uses a relative _vizdoom.ini during construction and
        # init. Keep both operations in the runtime directory. The lock protects
        # the process-wide cwd transition when episodes are made in parallel.
        runtime = Path(__file__).resolve().parent / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        game: Any = None
        try:
            with _INIT_CWD_LOCK:
                old_cwd = os.getcwd()
                try:
                    os.chdir(runtime)
                    game = vzd.DoomGame()
                    if not game.load_config(str(cfg)):
                        raise RuntimeError(f"ViZDoom rejected scenario configuration {cfg}")
                    game.set_doom_game_path(str(self._wad_path))
                    game.set_doom_map(map_name)
                    game.set_doom_skill(skill)
                    game.set_seed(int(seed) & 0xFFFFFFFF)
                    game.set_episode_timeout(self._episode_timeout)
                    game.set_mode(vzd.Mode.ASYNC_PLAYER)
                    game.set_window_visible(False)
                    game.set_sound_enabled(False)
                    game.set_audio_buffer_enabled(False)
                    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
                    game.set_screen_format(vzd.ScreenFormat.GRAY8)
                    game.set_render_hud(False)
                    game.set_render_messages(False)
                    game.set_automap_buffer_enabled(False)
                    game.set_labels_buffer_enabled(True)
                    game.set_objects_info_enabled(False)
                    game.set_sectors_info_enabled(False)
                    game.set_map_exit_reward(1.0)
                    game.set_kill_reward(0.0)
                    game.set_item_reward(0.0)
                    game.set_secret_reward(0.0)
                    game.set_living_reward(0.0)
                    game.clear_available_game_variables()
                    variable_list = self._variable_list(vzd)
                    for _, variable in variable_list:
                        game.add_available_game_variable(variable)
                    game.init()
                finally:
                    os.chdir(old_cwd)
        except Exception:
            if game is not None:
                try:
                    game.close()
                except Exception:
                    pass
            raise
        self._variable_names = [name for name, _ in variable_list]
        self._button_names = [getattr(button, "name", str(button)) for button in game.get_available_buttons()]
        self._button_index = {name.upper(): i for i, name in enumerate(self._button_names)}
        return game

    def _make_vector(self, buttons: Iterable[str]) -> list[int]:
        vector = [0] * len(self._button_names)
        for button in buttons:
            index = self._button_index.get(button.upper())
            if index is not None:
                vector[index] = 1
        return vector

    def _command(self, label: str, buttons: Iterable[str] = (), pulse: bool = False) -> _Command:
        return _Command(label, self._make_vector(buttons), pulse)

    def _parse_action(self, action: Any) -> _Command | None:
        if not isinstance(action, str):
            return None
        raw_text = action.strip().lower()
        text = raw_text.replace("-", "_")
        if text in _ACTION_NAMES:
            sustained: dict[str, tuple[str, ...]] = {
                "noop": (),
                "forward": ("SPEED", "MOVE_FORWARD"),
                "backward": ("SPEED", "MOVE_BACKWARD"),
                "turn_left": ("TURN_LEFT",),
                "turn_right": ("TURN_RIGHT",),
                "strafe_left": ("SPEED", "STRAFE", "MOVE_LEFT"),
                "strafe_right": ("SPEED", "STRAFE", "MOVE_RIGHT"),
                "forward_left": ("SPEED", "STRAFE", "MOVE_FORWARD", "MOVE_LEFT"),
                "forward_right": ("SPEED", "STRAFE", "MOVE_FORWARD", "MOVE_RIGHT"),
                "backward_left": ("SPEED", "STRAFE", "MOVE_BACKWARD", "MOVE_LEFT"),
                "backward_right": ("SPEED", "STRAFE", "MOVE_BACKWARD", "MOVE_RIGHT"),
                "fire": ("ATTACK",),
                "fire_forward": ("ATTACK", "SPEED", "MOVE_FORWARD"),
                "fire_strafe_left": ("ATTACK", "SPEED", "STRAFE", "MOVE_LEFT"),
                "fire_strafe_right": ("ATTACK", "SPEED", "STRAFE", "MOVE_RIGHT"),
            }
            if text in sustained:
                return self._command(text, sustained[text])
            if text == "use":
                return self._command(text, ("USE",), pulse=True)
            if text == "next_weapon":
                return self._command(text, ("SELECT_NEXT_WEAPON",), pulse=True)
            if text == "prev_weapon":
                return self._command(text, ("SELECT_PREV_WEAPON",), pulse=True)
            if text.startswith("select_weapon_"):
                slot = text.rsplit("_", 1)[-1]
                return self._command(text, (f"SELECT_WEAPON{slot}",), pulse=True)
        parts = raw_text.split()
        if len(parts) == 3 and parts[0] == "move":
            try:
                forward, turn = int(parts[1]), int(parts[2])
            except ValueError:
                return None
            if forward not in (-1, 0, 1) or turn not in (-1, 0, 1):
                return None
            buttons: list[str] = ["SPEED"] if forward or turn else []
            if forward > 0:
                buttons.append("MOVE_FORWARD")
            elif forward < 0:
                buttons.append("MOVE_BACKWARD")
            if turn > 0:
                buttons.append("TURN_RIGHT")
            elif turn < 0:
                buttons.append("TURN_LEFT")
            return self._command(f"move {forward} {turn}", buttons)
        if len(parts) == 2 and parts[0] == "select" and parts[1].isdigit():
            slot = int(parts[1])
            if 1 <= slot <= 7:
                return self._command(f"select {slot}", (f"SELECT_WEAPON{slot}",), pulse=True)
        return None

    def _stop_engine(self) -> None:
        self._stop_event.set()
        thread = self._engine_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.tick_s * 8.0))
        self._engine_thread = None

    def reset(self, seed: int) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("reset(seed) requires an integer seed")
        map_name, skill = self._scenario(seed)
        with self._lifecycle_lock:
            self._stop_engine()
            with self._state_lock:
                old_game = self._game
                self._game = None
            if old_game is not None:
                try:
                    old_game.close()
                except Exception:
                    pass
            game = self._create_game(seed, map_name, skill)
            with self._state_lock:
                self._game = game
                self._closed = False
                self._done = False
                self._valid_trial = True
                self._invalid_reason = None
                self._terminal_reason = None
                self._last_error = None
                self._last_state = None
                self._last_reward = 0.0
                self._total_reward = 0.0
                self._engine_tics = 0
                self._ticks = 0
                self._start_time = time.monotonic()
                self._seed = int(seed)
                self._map = map_name
                self._skill = skill
                self._action_epoch = 0
                self._pulse_epoch = None
                self._current_command = _Command("noop", [0] * len(self._button_names))
                self._stop_event = threading.Event()
                try:
                    self._last_state = game.get_state()
                except Exception:
                    self._last_state = None
                self._engine_thread = threading.Thread(
                    target=self._engine_loop,
                    name="vizdoom-ticker",
                    daemon=True,
                )
                self._engine_thread.start()
                return self._observation_locked()

    def _mark_invalid_locked(self, reason: str) -> None:
        self._valid_trial = False
        self._invalid_reason = reason[:500]
        self._done = True
        self._terminal_reason = "infrastructure_error"

    def _update_terminal_locked(self, game: Any) -> None:
        try:
            finished = bool(game.is_episode_finished())
            timeout = bool(game.is_episode_timeout_reached())
            dead = bool(game.is_player_dead())
        except Exception as exc:
            self._mark_invalid_locked(f"ViZDoom terminal query failed: {type(exc).__name__}: {exc}")
            return
        if not finished:
            return
        self._done = True
        if self._terminal_reason == "infrastructure_error":
            return
        if timeout:
            self._terminal_reason = "timeout"
        elif dead:
            self._terminal_reason = "death"
        else:
            # In single-player Doom a non-death, non-timeout terminal episode
            # is the map exit.  This also handles compatible IWAD exit specials.
            self._terminal_reason = "exit"

    def _engine_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._state_lock:
                game = self._game
                if game is None or self._closed or self._done:
                    return
            with self._action_lock:
                command = self._current_command
                epoch = self._action_epoch
            try:
                # The native binding is owned by this thread while an episode
                # is live.  Do not hold state_lock over advance_action: tick()
                # is a mailbox operation and must not wait behind a native tic.
                with self._native_lock:
                    game.set_action(command.vector)
                    game.advance_action(1)
                    engine_tics = max(0, _int_value(game.get_episode_time()))
                    last_reward = _finite(game.get_last_reward())
                    total_reward = _finite(game.get_total_reward())
                    state = game.get_state()
                    finished = bool(game.is_episode_finished())
                    timeout = bool(game.is_episode_timeout_reached())
                    dead = bool(game.is_player_dead())
                with self._state_lock:
                    if self._game is not game:
                        return
                    self._engine_tics = max(self._engine_tics, engine_tics)
                    self._last_reward = last_reward
                    self._total_reward = total_reward
                    if state is not None:
                        self._last_state = state
                    if finished:
                        self._done = True
                        if timeout:
                            self._terminal_reason = "timeout"
                        elif dead:
                            self._terminal_reason = "death"
                        elif self._terminal_reason != "infrastructure_error":
                            self._terminal_reason = "exit"
            except Exception as exc:  # native crashes become invalid trials, not harness errors
                if not self._stop_event.is_set():
                    with self._state_lock:
                        self._mark_invalid_locked(
                            f"ViZDoom controller failed: {type(exc).__name__}: {exc}"
                        )
                return
            if command.pulse:
                with self._action_lock:
                    if self._action_epoch == epoch:
                        self._current_command = _Command(
                            "noop", [0] * len(self._button_names)
                        )
                        self._pulse_epoch = None

    def tick(self, action: Any) -> None:
        """Post the current action; the controller thread advances the live world."""
        # These fields are only Python mailbox state; avoid both locks around
        # native advance_action so a late controller decision remains cheap.
        if self._closed or self._game is None or self._done:
            return
        parsed = self._parse_action(action)
        if parsed is None:
            self._last_error = f"ignored malformed or illegal action: {action!r}"[:300]
            return
        with self._action_lock:
            self._current_command = parsed
            self._action_epoch += 1
            self._pulse_epoch = self._action_epoch if parsed.pulse else None
            self._ticks += 1
        self._last_error = None

    @property
    def ticks(self) -> int:
        with self._action_lock:
            return int(self._ticks)

    def _values_locked(self) -> dict[str, Any]:
        state = self._last_state
        values: dict[str, Any] = {}
        if state is None:
            return values
        raw = getattr(state, "game_variables", None)
        if raw is None:
            return values
        for name, value in zip(self._variable_names, raw):
            if name in {"DEAD", "ON_GROUND", "ATTACK_READY", "WEAPON0", "WEAPON1", "WEAPON2",
                        "WEAPON3", "WEAPON4", "WEAPON5", "WEAPON6", "WEAPON7", "WEAPON8", "WEAPON9"}:
                values[name] = _bool_value(value)
            elif name in {"HEALTH", "ARMOR", "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO",
                          "KILLCOUNT", "ITEMCOUNT", "SECRETCOUNT", "DAMAGE_TAKEN",
                          *(f"AMMO{i}" for i in range(10))}:
                values[name] = _int_value(value)
            else:
                values[name] = _finite(value)
        return values

    @staticmethod
    def _labels(state: Any) -> list[dict[str, Any]]:
        labels = getattr(state, "labels", None) or []
        result: list[dict[str, Any]] = []
        for label in labels[:128]:
            try:
                result.append({
                    "id": _int_value(label.object_id),
                    "name": str(label.object_name),
                    "category": str(label.object_category),
                    "value": _int_value(label.value),
                    "bbox": [_int_value(label.x), _int_value(label.y),
                             _int_value(label.width), _int_value(label.height)],
                    "world": [_finite(label.object_position_x), _finite(label.object_position_y),
                              _finite(label.object_position_z)],
                    "velocity": [_finite(label.object_velocity_x), _finite(label.object_velocity_y),
                                 _finite(label.object_velocity_z)],
                })
            except Exception:
                continue
        return result

    def _observation_locked(self) -> dict[str, Any]:
        state = self._last_state
        values = self._values_locked()
        tic = _int_value(getattr(state, "tic", self._engine_tics)) if state is not None else self._engine_tics
        self._engine_tics = max(self._engine_tics, tic)
        ammo = {str(i): int(values.get(f"AMMO{i}", 0)) for i in range(10)}
        weapons = {str(i): bool(values.get(f"WEAPON{i}", False)) for i in range(10)}
        player = None
        if all(key in values for key in ("POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE")):
            player = {
                "x": values["POSITION_X"],
                "y": values["POSITION_Y"],
                "z": values["POSITION_Z"],
                "angle": values["ANGLE"],
            }
        total_reward = _finite(self._total_reward)
        elapsed_tics = max(0, tic)
        observation: dict[str, Any] = {
            "game": "Doom",
            "interface": "ViZDoom ASYNC_PLAYER",
            "iwad": self._family,
            "iwad_name": self._wad_path.name,
            "iwad_source": "bundled_freedoom2" if self._bundled_fallback else "VIZDOOM_WAD_PATH",
            "map": self._map,
            "skill": int(self._skill),
            "seed": int(self._seed) if self._seed is not None else None,
            "tic": elapsed_tics,
            "elapsed_tics": elapsed_tics,
            "elapsed_seconds": round(elapsed_tics / TICRATE, 3),
            "engine_tics": int(self._engine_tics),
            "ticks": int(self._ticks),
            "health": int(values.get("HEALTH", 0)),
            "armor": int(values.get("ARMOR", 0)),
            "dead": bool(values.get("DEAD", False)),
            "on_ground": bool(values.get("ON_GROUND", False)),
            "attack_ready": bool(values.get("ATTACK_READY", False)),
            "weapon": int(values.get("SELECTED_WEAPON", 0)),
            "weapon_ammo": int(values.get("SELECTED_WEAPON_AMMO", 0)),
            "ammo": ammo,
            "weapons": weapons,
            "kills": int(values.get("KILLCOUNT", 0)),
            "items": int(values.get("ITEMCOUNT", 0)),
            "secrets": int(values.get("SECRETCOUNT", 0)),
            "damage_taken": int(values.get("DAMAGE_TAKEN", 0)),
            "player": player,
            "velocity": {
                "x": values.get("VELOCITY_X", 0.0),
                "y": values.get("VELOCITY_Y", 0.0),
                "z": values.get("VELOCITY_Z", 0.0),
            },
            "visible": self._labels(state) if state is not None else [],
            "reward": _finite(self._last_reward),
            "vizdoom_total_reward": total_reward,
            "terminal": bool(self._done),
            "terminal_reason": self._terminal_reason,
            "last_action": self._current_command.label,
        }
        if self._last_error:
            observation["action_error"] = self._last_error
        return observation

    def observe(self) -> dict[str, Any]:
        # The controller snapshots state after every native tic. Returning the
        # cached snapshot keeps observation latency independent of native IPC.
        with self._state_lock:
            return self._observation_locked()

    @property
    def done(self) -> bool:
        with self._state_lock:
            return bool(self._done)

    def _progress(self, values: dict[str, Any]) -> tuple[float, float, float, float]:
        kill_p = min(max(_finite(values.get("KILLCOUNT", 0)) / _PROGRESS_REFERENCE["kills"], 0.0), 1.0)
        item_p = min(max(_finite(values.get("ITEMCOUNT", 0)) / _PROGRESS_REFERENCE["items"], 0.0), 1.0)
        secret_p = min(max(_finite(values.get("SECRETCOUNT", 0)) / _PROGRESS_REFERENCE["secrets"], 0.0), 1.0)
        return kill_p, item_p, secret_p, 0.6 * kill_p + 0.25 * item_p + 0.15 * secret_p

    @property
    def score(self) -> float:
        with self._state_lock:
            values = self._values_locked()
            _, _, _, normalized = self._progress(values)
            exit_component = 1000.0 if self._terminal_reason == "exit" and self._valid_trial else 0.0
            progress_component = 25.0 * normalized
            health_component = 3.0 * min(max(_finite(values.get("HEALTH", 0)) / 100.0, 0.0), 1.0)
            armor_component = 1.5 * min(max(_finite(values.get("ARMOR", 0)) / 100.0, 0.0), 1.0)
            ammo_total = sum(max(0, _int_value(values.get(f"AMMO{i}", 0))) for i in range(10))
            ammo_component = min(ammo_total / 200.0, 1.0)
            timeout_s = max(self._episode_timeout / TICRATE, 1.0)
            elapsed = max(0.0, (self._engine_tics or 0) / TICRATE)
            time_component = max(0.0, 1.0 - elapsed / timeout_s)
            reward_component = max(-5.0, min(5.0, _finite(self._total_reward)))
            return float(
                exit_component + progress_component + health_component + armor_component
                + ammo_component + time_component + reward_component
            )

    def score_breakdown(self) -> dict[str, float]:
        with self._state_lock:
            values = self._values_locked()
            kill_p, item_p, secret_p, normalized = self._progress(values)
            exit_component = 1000.0 if self._terminal_reason == "exit" and self._valid_trial else 0.0
            progress_component = 25.0 * normalized
            health_component = 3.0 * min(max(_finite(values.get("HEALTH", 0)) / 100.0, 0.0), 1.0)
            armor_component = 1.5 * min(max(_finite(values.get("ARMOR", 0)) / 100.0, 0.0), 1.0)
            ammo_total = sum(max(0, _int_value(values.get(f"AMMO{i}", 0))) for i in range(10))
            ammo_component = min(ammo_total / 200.0, 1.0)
            timeout_s = max(self._episode_timeout / TICRATE, 1.0)
            time_component = max(0.0, 1.0 - max(0, self._engine_tics) / TICRATE / timeout_s)
            reward_component = max(-5.0, min(5.0, _finite(self._total_reward)))
            return {
                "exit_success": exit_component,
                "normalized_progress": progress_component,
                "kill_progress": kill_p,
                "item_progress": item_p,
                "secret_progress": secret_p,
                "health": health_component,
                "armor": armor_component,
                "remaining_ammo": ammo_component,
                "time_tiebreak": time_component,
                "vizdoom_reward": reward_component,
                "total": float(
                    exit_component + progress_component + health_component + armor_component
                    + ammo_component + time_component + reward_component
                ),
            }

    def info(self) -> dict[str, Any]:
        with self._state_lock:
            values = self._values_locked()
            elapsed_tics = max(0, self._engine_tics)
            totals = {"kills": None, "items": None, "secrets": None}
            return {
                "valid_trial": bool(self._valid_trial),
                "invalid_reason": self._invalid_reason,
                "iwad": self._family,
                "iwad_name": self._wad_path.name,
                "iwad_source": "bundled_freedoom2" if self._bundled_fallback else "VIZDOOM_WAD_PATH",
                "map": self._map,
                "skill": int(self._skill),
                "seed": int(self._seed) if self._seed is not None else None,
                "scenario_group": _seed_group(self._seed) if self._seed is not None else None,
                "completed_by_exit": self._terminal_reason == "exit" and self._valid_trial,
                "terminal_reason": self._terminal_reason,
                "death_or_timeout": self._terminal_reason in {"death", "timeout"},
                "elapsed_tics": elapsed_tics,
                "elapsed_seconds": elapsed_tics / TICRATE,
                "vizdoom_total_reward": _finite(self._total_reward),
                "kills": int(values.get("KILLCOUNT", 0)),
                "items": int(values.get("ITEMCOUNT", 0)),
                "secrets": int(values.get("SECRETCOUNT", 0)),
                "map_totals": totals,
                "kill_percent": None,
                "item_percent": None,
                "secret_percent": None,
                "health": int(values.get("HEALTH", 0)),
                "armor": int(values.get("ARMOR", 0)),
                "remaining_ammo": sum(max(0, _int_value(values.get(f"AMMO{i}", 0))) for i in range(10)),
                "ticks": int(self._ticks),
                "engine_tics": int(self._engine_tics),
            }

    def describe(self) -> str:
        return (
            "Classic Doom single-player on one IWAD map per episode. The engine runs headlessly "
            "at 35 tics per second in ViZDoom ASYNC_PLAYER mode; monsters, projectiles, doors, "
            "weapon cooldowns, and damage continue while the controller is thinking. Reach and "
            "activate the map exit to succeed. Death and the episode timeout end the episode. "
            "The seed selects a deterministic map/skill scenario and seeds ViZDoom's own RNG: "
            "declared train seeds use early maps at skills 2-3, held-out seeds use a mixed set "
            "including unseen maps at skill 4, and VIZDOOM_SKILL=5 is a separate stress run. "
            "Doom 1 uses E1M1-E4M9 and Doom II uses MAP01-MAP32; Doom and Freedoom are never "
            "mixed.\n\n"
            + self.action_spec.describe()
            + "\n\nObservations are compact JSON with IWAD family, map, skill, seed, Doom tic and "
            "elapsed time, health/armor/death, weapon/ammo ownership, kill/item/secret counts, "
            "player position/angle/velocity, visible labels with names/categories and pixel "
            "boxes, the latest and total ViZDoom reward, and terminal status. Labels are only "
            "currently rendered visible objects; whole-map objects, sectors, and omniscient "
            "automap data are disabled. Map totals are not exposed by the native API, so exact "
            "kill/item/secret percentages are null; bounded count progress is secondary to exit "
            "success."
        )

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed and self._game is None:
                return
            self._closed = True
            self._stop_engine()
            with self._state_lock:
                game = self._game
                self._game = None
            if game is not None:
                try:
                    game.close()
                except Exception:
                    # close() is intentionally idempotent; validation can report
                    # native teardown errors without leaving a process running.
                    pass


def make_game(**kwargs: Any) -> DoomRealtimeGame:
    """Construct an adaptor instance; the evaluator calls ``reset(seed)`` next."""
    return DoomRealtimeGame(**kwargs)
