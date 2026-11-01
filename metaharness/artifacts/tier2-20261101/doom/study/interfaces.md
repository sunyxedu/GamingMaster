# Doom programmatic interfaces

## Recommendation

Use the **official ViZDoom Python API directly**, with `DoomGame` in `ASYNC_PLAYER` mode, hidden-window software rendering, the labels buffer, and named game variables. This is the only option here that gives the adaptor low-latency native actions plus structured Doom state without requiring a display, GPU, account, or server. Keep a synchronous `PLAYER`/`make_action(action, tics)` path as a deterministic test fallback, but do not use it as the production tier-2 loop because it makes the world wait for the adaptor.

The API is mature, maintained by the Farama Foundation, and maps closely to the engine. Its documented lifecycle is `DoomGame` -> configure -> `init()` -> `get_state()`/`set_action()` or `make_action()` -> terminal checks -> `new_episode()`/`close()`. See the [DoomGame API](https://vizdoom.farama.org/api/python/doom_game/), [Python quick start](https://vizdoom.farama.org/introduction/python_quickstart/), [GameState API](https://vizdoom.farama.org/api/python/game_state/), and the official [basic Python example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/basic.py).

## Comparison of practical interfaces

| Interface | Maturity and latency | What it exposes/control surface | Legal / ToS risk | Cost and verdict |
|---|---|---|---|---|
| **ViZDoom Python API (chosen)** | High; native C++ engine behind Python bindings. A direct `set_action` in async mode avoids screenshot/OCR overhead. | `GameState` tic/time, reward, configured game variables, screen/depth/automap/audio/labels buffers, visible `Label` metadata, and optionally objects/sectors. Sends binary or delta button vectors; supports map, skill, seed, timeout, and reward configuration. | ViZDoom is open source. It does not redistribute commercial IWADs; using `doom.wad`/`doom2.wad` requires a legally obtained copy. Freedoom avoids that asset issue. No multiplayer anti-cheat or online ToS is involved. | Free software; only asset acquisition may cost money. **Best choice.** |
| **ViZDoom Gymnasium wrapper** | High and maintained in the same project; `step()` calls `make_action` with configurable `frame_skip`, so it is low overhead but synchronous. `reset(seed=...)` maps a Gym seed to ViZDoom's seed. | A Gymnasium observation dictionary containing screen, optional depth/labels/automap/audio/notifications, and `gamevariables`; binary/continuous/combined action spaces. `VizdoomScenarioEnv` covers bundled scenario configs. Native `env.state.labels` remains available, but custom full-Doom async behavior is less direct. | Same IWAD/licensing considerations; Gymnasium itself adds no online risk. | Free, but requires Gymnasium and its wrapper dependencies. **Good test harness; not the production async control path.** See the [wrapper source](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/base_gymnasium_env.py) and [wrapper API](https://vizdoom.farama.org/api/python/gymnasium_wrapper/). |
| **GZDoom/ZDoom mod (ACS/ZScript) plus a local process bridge** | Mature engine/mod ecosystem; action/state latency can be low if the bridge is native or IPC-based, but writing and maintaining the mod/bridge is substantial. | Can expose custom HUD/console/ACS variables, event hooks, map geometry and scripted rewards; input can be injected through engine bindings or a local protocol. More flexible for custom scenarios than stock ViZDoom, but not a ready symbolic agent API. | Open-source engine is generally permissive, but commercial IWADs remain the user's responsibility. A mod that reads process memory or joins public servers may violate server rules; avoid public multiplayer. | Free software plus engineering time. **Fallback if a custom WAD needs ACS instrumentation.** Start with ViZDoom instead. Sources: [ViZDoom custom environments](https://vizdoom.farama.org/environments/creating_custom/), [ViZDoom SLADE/config guidance](https://vizdoom.farama.org/environments/slade_config/). |
| **Screen capture + keyboard/input injection** | Easy to prototype but high and variable latency: capture, image parsing/OCR, and X input add scheduling jitter. Requires a window or virtual X server and gives no reliable game variables. | Screenshot/video and perhaps visible HUD; controls via X11/`xdotool`, `python-xlib`, `evdev/uinput`, or an equivalent virtual keyboard. | Usually acceptable for a local copy, but it is brittle and can look like automation on a multiplayer service. No display is available here, so it needs Xvfb and extra failure modes. | Free tools; CPU overhead and engineering cost. **Last-resort fallback only.** |
| **DOSBox/Chocolate Doom/emulator hooks** | Low maturity for this use; timing and memory addresses vary by executable/version. Emulator debugger or process-memory reads can be fast but are fragile. | Raw video/input or emulator memory/RAM snapshots; potentially exact internal variables if reverse engineered. No portable semantic label API. | Use only a lawfully owned executable/WAD. Memory inspection or automation of an online port can violate its rules; keep it offline. | Free/open-source emulator, high maintenance. **Not suitable when ViZDoom is available.** |
| **Multiplayer bot/server libraries (Zandronum/ZDaemon-style)** | Mature for their own multiplayer ecosystems, but server tick/network latency and protocol details are unnecessary here. | Server-side actor/event state and player command packets; often excellent for bots in a server-controlled match, not for single-player episode completion. | Public servers may impose anti-cheat, bot, and automation rules; would need a server/account and contradict the offline constraint. | Software may be free, but setup/server cost and ToS risk are avoidable. **Reject.** |
| **Bundled ViZDoom benchmark environments via Gym** | High for their named benchmark tasks, but they are not full classic Doom campaigns. | Standard observations/actions for Basic, Deadly Corridor, My Way Home, Deathmatch, etc. Useful smoke tests, not a substitute for `doom.cfg`/`doom2.cfg`. | Same open-source/WAD considerations; no online service. | Free. **Use only for adaptor smoke tests.** The [official scenario list](https://vizdoom.farama.org/environments/default/) says these are benchmark scenarios. |

## Chosen interface: exact adaptor contract

### Software and assets

Use the evaluator interpreter, not the system interpreter:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/doom/.venv/bin/python -m pip install "vizdoom==1.3.2"
```

`vizdoom==1.3.2` is the current stable release in the [PyPI release list](https://pypi.org/project/vizdoom/) and the project publishes Linux manylinux wheels for supported CPython versions; Python 3.12 support was added in the project's release history ([release notes](https://vizdoom.farama.org/release_notes/index.html), [build metadata](https://github.com/Farama-Foundation/ViZDoom/blob/main/pyproject.toml)). Do not install from an untrusted archive or run downloaded code from its directory.

The adaptor must receive one of:

- a legally obtained `doom.wad` for Doom/Ultimate Doom maps, or
- a legally obtained `doom2.wad` for Doom II maps, or
- a compatible Freedoom IWAD for the no-account/offline path (`freedoom1.wad`/`freedoom2.wad` as appropriate).

ViZDoom's original-level guide explains the required WAD placement and explicitly says original levels cannot be redistributed by the project: [original Doom levels](https://vizdoom.farama.org/environments/original_doom_levels/). Never put a credential or a copied commercial asset into the study files.

### Configuration sequence

The adaptor phase should perform these steps:

1. Verify that the evaluator Python imports `vizdoom`, report its package version, and resolve the absolute IWAD path from an environment variable. Fail with an actionable message if the WAD is missing; do not silently substitute a different game.
2. Construct `vzd.DoomGame()` and either load the official `doom.cfg`/`doom2.cfg` or set the equivalent fields. The official configs show the action order, `PLAYER` mode, 320x240 output, automap/audio/message options, 126000-tic timeout, and `map_exit_reward = 1`: [doom.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom.cfg), [doom2.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom2.cfg).
3. Set the WAD (`set_doom_game_path` or the config's `doom_game_path`), map (`set_doom_map`), skill (`set_doom_skill`, 1--5), seed (`set_seed`), and episode timeout before `init()`. Use a curated map allowlist matching the selected IWAD.
4. Set `Mode.ASYNC_PLAYER` for production. Set `window_visible(False)`, `sound_enabled(False)`, a small `GRAY8` or `DOOM_256_COLORS8` screen format, and a modest resolution. No X display or GPU should be required; the screen buffer is read in memory and never displayed.
5. Enable `labels_buffer` and request named variables. Recommended baseline variables are `HEALTH`, `ARMOR`, `DEAD`, `ON_GROUND`, `ATTACK_READY`, `SELECTED_WEAPON`, `SELECTED_WEAPON_AMMO`, `AMMO0`--`AMMO9`, `WEAPON0`--`WEAPON9`, `KILLCOUNT`, `ITEMCOUNT`, `SECRETCOUNT`, `DAMAGE_TAKEN`, `POSITION_X`, `POSITION_Y`, `POSITION_Z`, `ANGLE`, and `VELOCITY_X/Y`. Position/angle/velocity are unavailable when the `+viz_nocheat` flag is enabled; make this an explicit scenario choice.
6. Optionally enable `automap_buffer` in normal/explored-only mode. Leave `objects_info` and `sectors_info` disabled for the fair default because they expose all actors or geometry; enable them only for debugging or an explicitly omniscient evaluation. The official [objects/sectors example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/objects_and_sectors.py) documents the distinction.
7. After `init()`, poll `get_state()` and project every state into JSON. Preserve the variable-to-name mapping in the adaptor because `state.game_variables` is ordered by the configured list. Include `tic`, elapsed seconds, terminal flags, reward, and a stable label list.
8. Let Jev choose from a small typed action vocabulary (`MOVE_AND_TURN`, `ATTACK_AND_STRAFE`, `USE`, `RETREAT`, `SELECT_WEAPON`, `SEARCH/ROUTE`) and let code translate that choice to the exact button vector. In async mode update the held vector with `set_action` every 4--7 tics; do not call the Python model while holding the engine loop.
9. For a deterministic smoke test, use `Mode.PLAYER` and `make_action(action, tics)` with a small repeat (1--4 tics). This is a test fallback only; it pauses the world between calls and therefore is not the production tier-2 timing model.
10. Detect `is_episode_finished()`, `is_episode_timeout_reached()`, and `is_player_dead()`; record the terminal state, total reward, `KILLCOUNT`/items/secrets, elapsed tics, and reason. For each reset, set the requested seed/map/skill before `init()` or the next `new_episode()` as appropriate, then call `new_episode()` and call `close()` at process shutdown.

The exact API names above are listed in the [DoomGame reference](https://vizdoom.farama.org/api/python/doom_game/). The official [basic example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/basic.py) demonstrates configuration, `get_state`, `make_action`, action vectors, rewards, and terminal handling.

### Observation JSON projection

The model-facing state should be compact and symbolic, for example:

```text
{
  "tic": 842,
  "map": "E1M3",
  "health": 67,
  "armor": 50,
  "weapon": 3,
  "weapon_ammo": 18,
  "ammo": {"0": 42, "1": 18, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0},
  "kills": 5,
  "items": 8,
  "secrets": 1,
  "player": {"x": 1024.0, "y": -320.0, "z": 0.0, "angle": 90.0},
  "visible": [
    {"id": 17, "name": "ShotgunGuy", "category": "Monster", "bbox": [120, 62, 35, 64], "world": [1100.0, -280.0, 0.0], "label_value": 23}
  ],
  "terminal": false
}
```

The numeric values above are illustrative; the implementation must serialize the actual configured game variables. `state.labels` is the semantic list; `labels_buffer` is an 8-bit image whose values identify visible objects. The official [labels example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/labels_buffer.py) documents `object_id`, `object_name`, `value`, world position, and bounding-box fields. A raw screen array can remain in the adaptor for a fallback vision prompt, but the primary Jev input should be the JSON projection requested by the user.

Do not include all actors or full map sectors in the production JSON unless the benchmark explicitly allows omniscience. `objects` and `sectors` are valuable for debugging but make navigation substantially easier than a human's view.

### Action encoding and timing

Read `get_available_buttons()` after config and build a name-to-index map. Do not assume that a Gym wrapper's action ordering equals a hand-built direct vector. For the official Doom config, the binary action vector has the buttons listed in `doom.cfg`; a held movement/turn/fire action is repeated by the engine. Weapon selection and `USE` should normally be one-tic pulses. Code should clamp impossible combinations, avoid selecting an unowned weapon, and preserve a safe movement/turn action if Jev is late.

Use a single adaptor owner for the native `DoomGame` object. If Astra and Jev run in separate workers, exchange JSON plans and typed decisions through a queue; do not concurrently mutate the C++ binding from two threads without a tested synchronization layer. Emergency code rules (death, zero ammo, imminent collision, close hitscanner, exit/use prompt) must be able to override a stale model decision.

## Credentials, environment, and user-provided values

There is no server, account, API key, or online credential for the chosen offline interface. The only user-provided secret-like input is a local path to a legally obtained IWAD; keep its value out of all committed files.

Suggested environment variables:

| Variable | Required | Meaning |
|---|---:|---|
| `VIZDOOM_WAD_PATH` | yes | Absolute path to `doom.wad`, `doom2.wad`, or a compatible Freedoom IWAD. |
| `VIZDOOM_MAP` | no | Map ID such as `E1M1` or `MAP01`; the scenario generator may override this per reset. |
| `VIZDOOM_SKILL` | no | Integer 1--5; default to 3 when omitted. |
| `VIZDOOM_SEED` | no | Unsigned seed for deterministic episode randomness. |
| `VIZDOOM_MODE` | no | `ASYNC_PLAYER` for production; `PLAYER` for deterministic smoke tests. |
| `VIZDOOM_EPISODE_TIMEOUT_TICS` | no | Positive timeout; use the config's 126000-tic ceiling unless the evaluator defines a shorter budget. |
| `VIZDOOM_SCENARIO_CFG` | no | Absolute path to a copied/configured scenario file if the adaptor does not build settings in Python. |

Do not create values for these variables in repository files; the launch environment must supply them.

## Known limitations and risks

- **IWAD availability:** The package alone is not a legal copy of commercial Doom. Exact original maps cannot start without the user's WAD. Freedoom is the practical no-account substitute, but its maps/assets are not identical.
- **Headless rendering:** `window_visible=False` avoids a GUI, but the adaptor should still disable sound and avoid code paths that import a display-only viewer. Keep buffers in memory.
- **Async synchronization:** `ASYNC_PLAYER` is the correct tier-2 mode, but state polling and action updates are asynchronous. Timestamp JSON states and tolerate one stale state; never infer that a model response is still current.
- **Fairness/cheat state:** `objects_info`, `sectors_info`, map-wide automap modes, and unrestricted position data can reveal information not visible to the marine. Choose and document a fair observation policy.
- **Variable availability:** Variables are only present if configured, and some world-space values disappear under `+viz_nocheat`. The adaptor must validate the configured list before play.
- **Termination:** `get_state()` may be `None` after termination. Read terminal flags before dereferencing state and always close the game.
- **No universal map route:** Doom maps are deterministic but map-specific. Astra needs a route memory keyed by IWAD/map/skill, while code handles local movement, target tracking, and resource bookkeeping.

## Sources

- [ViZDoom documentation](https://vizdoom.farama.org/)
- [DoomGame Python API](https://vizdoom.farama.org/api/python/doom_game/)
- [Enums: modes, buttons, variables](https://vizdoom.farama.org/api/python/enums/)
- [GameState and labels](https://vizdoom.farama.org/api/python/game_state/)
- [Original Doom level setup](https://vizdoom.farama.org/environments/original_doom_levels/)
- [Official Doom and Doom II configs](https://github.com/Farama-Foundation/ViZDoom/tree/main/scenarios)
- [Official Python basic example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/basic.py)
- [Official labels-buffer example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/labels_buffer.py)
- [Official objects/sectors example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/objects_and_sectors.py)
- [Official Gymnasium wrapper source](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/base_gymnasium_env.py)
