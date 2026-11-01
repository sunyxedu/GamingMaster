# Programmatic interfaces for Atari 2600 Donkey Kong

## Recommendation

Use **ALE/Stella through `ale-py` + Gymnasium, with OCAtari's RAM decoder as the object layer**. In other words, let the official maintained Atari environment own stepping, terminal handling, seeding, and rewards; keep the raw 128-byte RAM on every observation; and use OCAtari 2.2.1 to turn the RAM into named objects for Jev and Astra. This is the only option found that is simultaneously headless, local, low-latency, Python 3.12-compatible, supplied with a matching Donkey Kong ROM, and already able to expose `Player`, `DonkeyKong`, `Hammer`, obstacles, ladders, life, and HUD values.

The evaluator's Python was tested directly:

```text
Python 3.12.3
ale-py 0.12.1
Gymnasium 1.4.0
OCAtari 2.2.1
```

The test created `ALE/DonkeyKong-v5` with RAM observations and verified `Discrete(18)`, a `(128,)` `uint8` observation, and a working `OCAtari(..., mode="ram")` object list. The `ale-py` wheel supplied the exact 4 KiB NTSC image whose MD5 is `36b20c427975760cb9cf4a47e41369e4`; the same image is audited frame-by-frame in the [Plicerin Atari 2600 port](https://github.com/Plicerin/donkeykong-rom-port).

### Why not use only OCAtari?

OCAtari is the best symbolic convenience layer, but its current 2.2.1 `step` implementation returns the Gymnasium terminal flags in the order `(obs, reward, truncated, terminated, info)`, while Gymnasium specifies `(obs, reward, terminated, truncated, info)`. The flags are equal during ordinary play, so this can remain hidden until a death or game-over transition. The adaptor should either (a) step the underlying Gymnasium object and call the OCAtari RAM detector explicitly, or (b) use OCAtari but swap/guard the two flags at its boundary after checking the installed version. Always retain direct ALE state and raw RAM; OCAtari does not label every game-specific fact.

## Chosen stack: exact setup contract

### Software

Install into the evaluator venv, not the system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-donkey-kong/.venv/bin/pip install \
  ale-py==0.12.1 gymnasium==1.4.0 ocatari==2.2.1
```

`ocatari` brings OpenCV, NumPy, and pygame dependencies. No display server, GPU, browser, account, API key, or remote server is required for `mode="ram"` and `render_mode=None`. The ALE and Gymnasium setup patterns are documented in the [ALE getting-started guide](https://ale.farama.org/getting-started/), [ALE Python interface](https://ale.farama.org/python-interface/), and [OCAtari README](https://github.com/k4ntz/OC_Atari).

### Environment variables and credentials

There are **no required credentials or service variables**.

- `ALE_ROMS_DIR` — optional. Set it only if the adaptor must load a user-supplied compatible ROM directory; otherwise use `ale_py.roms.get_rom_path("donkey_kong")`, which found the bundled ROM in the evaluator test. Do not put a ROM path or ROM bytes in a credential file or commit a copyrighted dump.
- `SDL_VIDEODRIVER=dummy` — optional fallback if a future OCAtari/pygame code path initializes SDL. It is not needed for the tested RAM-only, no-render path.

If the evaluator intentionally removes bundled ROMs, the user must supply a legally obtained Atari 2600 Donkey Kong image and set `ALE_ROMS_DIR`; no network account or key can substitute for that file.

### Configuration

Use the canonical Gymnasium ID `ALE/DonkeyKong-v5`. During adapter development, use:

- `obs_type="ram"` — 128 raw bytes, no screenshot dependency.
- `frameskip=4` — matches the documented v5 environment and gives a 66.7 ms control quantum at NTSC speed. Use `frameskip=1` only for fine-grained calibration or a very fast local controller.
- `repeat_action_probability=0.0` for deterministic adapter debugging. Evaluate a robust policy again at the documented v5 default `0.25` if the harness wants the standard stochastic protocol.
- `render_mode=None` — no window and no image allocation. If an RGB audit is needed, use `render_mode="rgb_array"` without a display; it is diagnostic, not the primary state channel.
- `full_action_space=True` is harmless; Donkey Kong's minimal and legal sets are already all 18 actions in the tested ALE release.
- `mode=0`, `difficulty=0` — Donkey Kong exposes only these values according to the [environment page](https://ale.farama.org/environments/donkey_kong/).

Register ALE's Gymnasium environments before `gymnasium.make`, as shown by the [official Gymnasium interface](https://ale.farama.org/gymnasium-interface/). The first active action must be `FIRE` (index 1); otherwise the game remains in its idle/attract state.

### Observation contract to hand to the models

Do not send a NumPy array or Python object repr directly to Jev. Convert every step to JSON primitives and retain both layers:

```text
{
  "frame": integer,
  "stage": "ramp" | "rivet" | "idle" | "transition" | "unknown",
  "player": {"x": integer, "y": integer, "w": integer, "h": integer,
              "dx": integer, "dy": integer, "visible": boolean},
  "hazards": [{"kind": "barrel_or_fireball", "x": integer, "y": integer,
               "dx": integer, "dy": integer, "slot": integer}],
  "ladders": [{"x": integer, "y": integer, "w": integer, "h": integer,
               "visible": boolean}],
  "hammer": {"present": boolean, "x": integer, "y": integer},
  "rivet_rows": [integer, integer, integer, integer],
  "bonus_display_bcd": integer,
  "score_total": integer,
  "lives_remaining": integer,
  "episode_return": number,
  "terminated": boolean,
  "truncated": boolean,
  "raw_ram": [128 integers]
}
```

`raw_ram` can be omitted from Jev's prompt when token pressure matters, but keep it in the adaptor log and give it to Astra on a diagnostic request. A `score_total` field should be the cumulative sum of ALE rewards, not merely the one-byte HUD bonus.

### OCAtari object layer

Create the object wrapper with RAM extraction, HUD enabled, no human renderer, and object observations. For a lean headless path, avoid the default RGB stack by passing an empty `create_buffer_stacks` list; the object list itself remains available. The important public attributes are `env.objects`, `env.action_space`, `env.reset(seed=...)`, `env.step(action)`, and the underlying ALE handle (`env._ale`/`env._env.unwrapped.ale`, names vary slightly by wrapper version).

For each object, serialize:

- `category` — class/category name;
- `xywh` — screen-space bounding box;
- `visible` — do not treat `NoObject` placeholders as hazards;
- `dx`, `dy`, or previous position if available;
- `value` for `Score`/other `ValueObject`s;
- `hud` — HUD values are not physical collision objects.

The current RAM decoder has these fixed slots:

| Category | Maximum slots | Notes |
|---|---:|---|
| `Player` | 1 | Mario; position is decoded from RAM |
| `DonkeyKong` | 1 | static enemy sprite |
| `Girlfriend` | 1 | static goal sprite |
| `Hammer` | 1 | hidden after pickup/timeout |
| `Barrel` | 4 | generic obstacle slots; on the rivet screen the decoder still calls the obstacle class `Barrel`, so normalize it to `hazard`/`fireball` from the stage |
| `Ladder` | 10 | fixed geometry slots; not a promise that every slot is usable in every state |
| `Life` | 1 | HUD value/shape |
| `Score` | 1 | HUD display value, not the complete accumulated score |

OCAtari's **vision** mode also works in the tested installation and is a fallback when a ROM variant changes RAM semantics, but it requires RGB frames and OpenCV and is less exact for headless symbolic play. The [OCAtari README](https://github.com/k4ntz/OC_Atari) documents RAM Extraction Mode (REM), Vision Extraction Mode (VEM), `env.objects`, and the coverage table; the implementation is visible in the [Donkey Kong RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/donkeykong.py) and [vision decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/vision/donkeykong.py).

### Raw RAM decoder and audit fields

ALE exposes the 128 bytes of RIOT RAM through `getRAM()`. The verified cartridge disassembly names the game variables using absolute 6502 addresses `$80`–`$FF`; the ALE array index is `address - $80`. These fields are enough to augment OCAtari without reverse-engineering a screenshot:

| ALE index | Absolute address | Meaning from the verified port | Use |
|---:|---:|---|---|
| 0–3 | `$80–$83` | obstacle vertical positions | nonzero obstacle slots |
| 5 | `$85` | jump hang-time state | jump/death timing audit |
| 6 | `$86` | hammer time | whether the temporary hammer is active |
| 7–8 | `$87–$88` | BCD player score bytes | score audit only; prefer reward sum |
| 12 | `$8C` | game state | idle, active, level-complete transition, or game-over bits |
| 13 | `$8D` | losing-life flag | death transition |
| 16 | `$90` | `gameScreen`: `0` ramps/barrels, `1` rivets/fireballs | stage classifier |
| 19 | `$93` | Mario horizontal position | player x |
| 27 | `$9B` | Mario vertical position | player y/walkway estimate |
| 31–34 | `$9F–$A2` | four rivet-row status bytes | left/right/complete rivet bookkeeping |
| 35 | `$A3` | number of remaining Marios | life bookkeeping |
| 36 | `$A4` | BCD bonus timer/display | urgency and completion bonus |
| 65–68 | `$C1–$C4` | obstacle horizontal positions | hazard x for four slots |
| 69–72 | `$C5–$C8` | obstacle directions | relative-motion estimate |
| 73–76 | `$C9–$CC` | obstacle ladder numbers | barrel path audit |

The authoritative labels and constants are in the [verified `donkey_kong_2600.asm`](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm). That source uses `HORIZ_LEFT_RIVIT=$36`, `HORIZ_RIGHT_RIVIT=$6A`, values `6/12/18`, and `STARTING_BONUS=$50` (BCD 5000). It also documents that the current port has four obstacle slots and alternating `gameScreen` values. Keep the decoder table versioned: RAM locations are cartridge-specific and should not be reused for arcade Donkey Kong.

### Actions and timing

Use the ALE action indices exactly as documented in the [environment specification](https://ale.farama.org/env-spec/):

```text
0 NOOP, 1 FIRE, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN,
6 UPRIGHT, 7 UPLEFT, 8 DOWNRIGHT, 9 DOWNLEFT,
10 UPFIRE, 11 RIGHTFIRE, 12 LEFTFIRE, 13 DOWNFIRE,
14 UPRIGHTFIRE, 15 UPLEFTFIRE, 16 DOWNRIGHTFIRE, 17 DOWNLEFTFIRE
```

The action is held for `frameskip` emulated frames by the Gymnasium wrapper. Do not let Astra's multi-second call stop the actor. Jev should choose or validate an immediate action every one or two four-frame steps, while a safe previously chosen direction/jump is held by the adaptor. A background loop must also detect `terminated`/`truncated` and stop sending actions after the terminal state until `reset`.

### Reset, seeding, state cloning, and scoring

- `reset(seed=s)` resets the normal cartridge start. It does not choose an arcade-style level or opponent; Donkey Kong has only mode 0 and difficulty 0.
- `reset(seed=s)` seeds Gymnasium/ALE randomness and returns seed metadata, but it does not select a level. With fixed frameskip and `repeat_action_probability=0.0`, the same ROM/action trace may be identical across seeds because the cartridge's own obstacle RNG is deterministic from its emulated state/input; use the same seed/options/action trace for reproducibility, and do not promise seed variation that the ROM does not provide.
- `repeat_action_probability=0.0` is best for adapter calibration; `0.25` is the documented v5 default and is the meaningful held-out-seed robustness scenario because sticky-action RNG can alter which action is applied.
- `ale.lives()` and `ale.game_over()` are available on the direct ALE handle. Gymnasium's `info` includes lives.
- `ALEInterface.cloneState()` and `restoreState()` are available for deterministic planning experiments. `cloneState(include_rng=True)` captures the sticky-action RNG as well; the [Python interface documentation](https://ale.farama.org/python-interface/) explains the distinction. Do not use save-state cloning to bypass the life objective in evaluation; it is for local search/calibration.
- Add each Gymnasium step reward to `score_total`/`episode_return`. The v5 wrapper returns the sum of rewards from its four internal frames. Keep the raw reward, without replacing it by a binary win signal.

## Alternative interfaces

### 1. Direct `ale-py` / Gymnasium RAM or RGB

- **Maturity:** High; this is the maintained Farama ALE interface over Stella, with an official Gymnasium wrapper.
- **Latency:** Very low local native emulation latency; no network round trip. RAM observations avoid image processing.
- **Observations:** 128-byte RAM, RGB `(210,160,3)`, or grayscale `(210,160)`; rewards; lives; terminal/truncation; seed; clone/restore state. It does not supply named objects or a Donkey Kong rivet decoder.
- **Control:** The 18 Atari joystick/button actions above.
- **Cost:** Free/open source; `ale-py` wheels work on the evaluator's CPython 3.12 Linux.
- **Legal/ToS:** No service or anti-cheat risk. The emulator is open source, but the cartridge image remains copyrighted; use the image supplied by the package or a legally obtained user dump and do not redistribute it.
- **Assessment:** The most stable stepping API and the required fallback if OCAtari changes. It needs a small, versioned RAM-to-JSON decoder for symbolic state.
- **Sources:** [ALE Python interface](https://ale.farama.org/python-interface/), [DonkeyKong specification](https://ale.farama.org/environments/donkey_kong/).

### 2. OCAtari REM/VEM

- **Maturity:** Medium; research-oriented MIT-licensed wrapper, current PyPI release tested successfully.
- **Latency:** Low for RAM mode; VEM adds RGB acquisition and OpenCV segmentation.
- **Observations:** Named object slots, coordinates, sizes, visibility/history, HUD values; RAM mode is object-centric without needing a screenshot. It does not expose explicit rivet objects and labels rivet-screen obstacles as `Barrel` in its current decoder.
- **Control:** Delegates to ALE/Gymnasium's 18 actions.
- **Cost:** Free, but installation is heavier (`opencv-python`, `pygame`, scientific packages). No GPU is needed in RAM mode.
- **Legal/ToS:** MIT wrapper and local ROM/emulator caveats only.
- **Assessment:** Best symbolic state shortcut, but keep raw RAM and guard its swapped terminal return flags. Use VEM only as a diagnostic cross-check.
- **Sources:** [OCAtari README/API](https://github.com/k4ntz/OC_Atari), [PyPI release](https://pypi.org/project/ocatari/2.2.1/).

### 3. Stable-Retro with its Stella Atari 2600 core

- **Maturity:** High as a generic Gym/libretro project; Atari 2600 Stella core is supported on Linux, but the current integration list does **not** include `DonkeyKong-Atari2600-v0` (it includes other Atari 2600 titles and Donkey Kong on other systems).
- **Latency:** Low native libretro stepping once built.
- **Observations:** Standard image/action interface plus optional memory-map/reward/episode integration fields for games that have an integration. No ready Donkey Kong object schema.
- **Control:** Libretro joystick/button callbacks.
- **Cost/setup:** Free; Python 3.11–3.14 supported, but a Donkey Kong integration must be added/imported, including a legally obtained ROM, RAM addresses, reward function, and end condition. This is substantially more adaptor work than ALE.
- **Legal/ToS:** MIT framework/core licenses vary by core; ROM is not included and remains copyrighted.
- **Assessment:** Useful if the harness already standardizes on Stable-Retro or if a custom integration is wanted, not the first choice here. The [README's Atari table and ROM-import instructions](https://github.com/Farama-Foundation/stable-retro) and [supported game list](https://github.com/Farama-Foundation/stable-retro/blob/main/docs/supported_games.md) confirm the distinction.

### 4. Standalone Stella and its debugger

- **Maturity:** Very high as an Atari 2600 emulator/debugger.
- **Latency:** Native, but interactive debugger use is not a clean low-latency agent API.
- **Observations:** The integrated debugger can display/change CPU, TIA, RIOT RAM, watches, traps, disassembly, and frame/scanline state. It supports `-debug`, prompt commands, batch scripts, and frame advance ([debugger manual](https://stella-emu.github.io/docs/debugger.html)).
- **Control:** Keyboard/joystick event mapping and debugger-side memory/register manipulation. The normal desktop program is SDL/UI-oriented; the documentation does not promise a stable Python/headless frame-control protocol.
- **Cost/setup:** Free/open source standalone program; Linux packages/build tools and a user ROM are needed.
- **Legal/ToS:** Stella is GPLv2; cartridge rights are separate. Local offline use has no online terms/anti-cheat issue.
- **Assessment:** Excellent for validating RAM labels, breakpoints, and collision timing while developing the decoder; poor as the production Astra/Jev interface without writing a custom frontend.

### 5. Stella libretro core + RetroArch or `libretro.py`

- **Maturity:** Stella/libretro core is mature; Python frontend options are less turnkey. `libretro.py` describes itself as a Python binding intended for core testing and is currently alpha, requiring Python 3.12+ ([repository](https://github.com/JesseTG/libretro.py)).
- **Latency:** Native and potentially headless; callback overhead and frontend setup are higher than `ale-py`.
- **Observations:** Video/audio callbacks, input descriptors, save states, and (where the frontend exposes the core's memory descriptors) the 128-byte Atari RAM domain. It is raw memory, not Donkey Kong objects.
- **Control:** Libretro joypad callbacks; a custom frontend can submit actions per frame.
- **Cost/setup:** Free, but requires a compatible Stella core binary, Python/libretro binding or RetroArch frontend, and a legal ROM. `libretro.py` may require compilation/toolchain details and has less battle-tested Gym integration.
- **Legal/ToS:** Stella core is GPLv2; frontend/binding licenses differ; ROM rights remain separate. No account or server.
- **Assessment:** A viable fallback if direct core memory access or save-state behavior is required, but it duplicates the already-tested ALE/Stella stack and has more integration risk. See [Stella libretro documentation](https://docs.libretro.com/library/stella/) and [libretro memory monitoring](https://docs.libretro.com/guides/memorymonitoring/).

### 6. Browser/JavaScript verified port

- **Maturity:** The [Plicerin port](https://github.com/Plicerin/donkeykong-rom-port) is unusually valuable for audit: it translates the original cartridge instruction-by-instruction and reports frame-by-frame verification, but it is an unofficial preservation project rather than a maintained Python environment.
- **Latency:** Fast in a local browser/Node runtime; a browser automation layer adds overhead.
- **Observations:** Internal JavaScript machine/TIA/RAM state and a canvas; no documented JSON or Python agent API. A custom Node adaptor would need to expose `dkMachine`/input state.
- **Control:** Keyboard/gamepad/touch controls in the browser port; not a stable symbolic action contract.
- **Cost/setup:** Free source; headless Chromium/Playwright would be needed on this server if used as an agent environment. The repository deliberately does not include the cartridge image.
- **Legal/ToS:** Unofficial code and Nintendo/Coleco cartridge rights; no network account if served locally.
- **Assessment:** Best independent oracle for validating Atari-specific mechanics and MD5, not the production interface.

### 7. Screen capture plus input injection

- **Maturity:** Mature building blocks, fragile game adaptor.
- **Possible stack:** Stella or RetroArch, `Xvfb` for a virtual display, `ffmpeg`/`import`/Pillow/OpenCV for capture, and `xdotool`, `pyautogui`, `evdev/uinput`, or a RetroArch input pipe for injection.
- **Latency:** Usually tens of milliseconds to multiple frame periods for capture, OCR/vision, and input; worse and more variable than RAM. A 60 Hz game can punish frame loss.
- **Observations:** Screenshots only unless the adaptor adds CV/OCR. It must infer Mario, ladders, barrels/fireballs, rivets, score, and lives; flicker and palette/scale make this brittle.
- **Control:** Synthetic keyboard/joystick events; requires a virtual X/SDL setup despite the headless server.
- **Cost/legal:** Software is free; same ROM copyright caveat, no remote ToS. More packages and more failure modes.
- **Assessment:** Keep as a last-resort validation path when a ROM cannot be read or a decoder must be learned, not as the chosen symbolic interface.

### 8. Legacy `atari-py`, generic MAME, and small hobby emulators

- **`atari-py`:** The old OpenAI package is deprecated in favor of `ale-py`; do not make it a Python 3.12 dependency. It has no current object-level Donkey Kong contract.
- **MAME/arcade Stella variants:** They emulate a different arcade/arcade-like Donkey Kong ruleset. They are not interchangeable with the Atari 2600 two-screen ROM and would produce the wrong ladder, rivet, hazard, and scoring behavior.
- **Small Python emulators such as `pytari2600`:** Useful educational code, but no established exact-ROM, object-level, Python 3.12 adaptor was found. Their maturity and timing fidelity are insufficient compared with ALE/Stella.

## Adapter acceptance tests

Before giving the interface to the game-playing harness, verify:

1. `ale_py.roms.get_rom_path("donkey_kong")` exists and its MD5 is `36b20c427975760cb9cf4a47e41369e4`; otherwise stop rather than silently evaluating a different port.
2. RAM mode is `(128,)`, `uint8`; no SDL window is opened; no screenshot is required.
3. `reset(seed=0)` reports two spare lives and an idle/ramp initial state; an index-1 `FIRE` action starts play.
4. Action indices match the [ALE table](https://ale.farama.org/env-spec/); especially, `RIGHT=3`, `LEFT=4`, `UP=2`, `DOWN=5`, and `FIRE=1`.
5. A JSON observation contains the player, four hazard slots, ladder slots, stage, bonus, rivet bytes, lives, cumulative reward, and terminal flags.
6. The adaptor notices a life-loss transition without confusing `truncated` and `terminated`, and stops after true terminal game-over.
7. A fixed seed/action trace is repeatable with sticky actions disabled; a separate run at `repeat_action_probability=0.25` exercises robustness.
8. The cumulative reward, not just the displayed bonus, is reported as score.
