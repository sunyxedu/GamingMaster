# Programmatic interfaces for Atari 2600 Pong

## Recommendation

Use **ALE/`ale-py` through Gymnasium**, with the bundled `pong.bin` ROM, `obs_type="ram"`, `frameskip=1`, `repeat_action_probability=0.0`, and `render_mode=None`. This is the only option found that combines a maintained native Atari 2600 emulator, a supported CPython 3.12 wheel, a direct Python step API, RAM access, deterministic reset controls, and no display server. It is also the interface used by the well-documented `ALE/Pong-v5` environment.

ALE is an emulator/environment, not an Atari-operated online service. No account, server, API key, or network connection is required after installation. The package is GPL-2.0; the bundled Atari ROM's distribution rights should be checked for the deployment context. Do not add a ROM dump to this repository or redistribute it merely because an emulator can load it.

Primary references: [ALE installation](https://ale.farama.org/getting-started/), [ALE Pong](https://ale.farama.org/environments/pong/), [ALE Gymnasium interface](https://ale.farama.org/gymnasium-interface/), [ALE Python interface](https://ale.farama.org/python-interface/), [ALE source and license](https://github.com/Farama-Foundation/Arcade-Learning-Environment), and [ALE on PyPI](https://pypi.org/project/ale-py/).

## Chosen interface: ALE/ale-py + Gymnasium

### Maturity, cost, latency, and risk

- **Maturity:** High. ALE is a long-running research emulator built on Stella, has an official Gymnasium integration, and publishes native wheels. The Pong page specifies the ROM environment, actions, observations, modes, difficulties, and v5 timing options.
- **Latency:** Native emulation is fast enough to run frame-by-frame in-process. `frameskip=1` gives one Atari frame per `step`; the Python call itself is not a real-time wall-clock guarantee, so the adaptor should run a tight loop and never wait for Astra/Jev before issuing the next low-level action. The default v5 setting uses four frames per step and sticky actions with probability 0.25; disable both for a reproducible control experiment, or deliberately retain them for a benchmark matching the documented default.
- **Observation:** 128 bytes of RAM (`uint8`) without a display; optionally 210×160×3 RGB or 210×160 grayscale for calibration and a fallback object detector. A reward and frame metadata arrive with every step. Native ALE also provides screen, RAM, legal-action, game-over, lives, and emulator-state methods.
- **Control:** Six minimal actions: `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`. Full 18-action Atari input is available but unnecessary. The labels are controller directions, not guaranteed screen-Y directions; calibrate which of `RIGHT` and `LEFT` moves the right paddle up/down in the selected ROM/variation.
- **Cost:** Free software; pip install only. No GPU, X server, SDL, account, or credentials are needed for `render_mode=None`.
- **Legal/ToS:** No online anti-cheat or service ToS applies. ALE is GPL-2.0. Atari ROMs and manuals remain copyrighted; use the package's permitted bundled ROM or a ROM the operator is licensed to use, and do not commit ROM binaries.

### Install on the evaluator's Python

Run these commands with the evaluator interpreter, not a system `python`:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-pong/.venv/bin/python -m pip install -U pip
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-pong/.venv/bin/python -m pip install ale-py gymnasium
```

`ale-py` currently packages the Atari ROM files and publishes CPython wheels for modern Linux/Python versions. `numpy` is installed as a dependency. The installation documentation also describes the native C++ build: C++17, CMake 3.14+, zlib, and optional SDL; the C++ route is only needed if Python/native-wheel performance is inadequate.

Do **not** use the obsolete `gym` Atari environments, `atari-py`, or an automatic ROM downloader for this adaptor. Their APIs and ROM licensing assumptions differ from the current ALE package.

### Minimal environment setup

The adaptor phase should implement the following setup sequence (this is an API recipe, not a game-playing policy):

1. Import `gymnasium` and `ale_py`, then call `gym.register_envs(ale_py)`.
2. Construct `gym.make("ALE/Pong-v5", obs_type="ram", frameskip=1, repeat_action_probability=0.0, render_mode=None)`. Keep `full_action_space=False`.
3. Call `obs, info = env.reset(seed=seed)` before each episode. Record `info["episode_frame_number"]` and `info["frame_number"]` when present.
4. Call `env.unwrapped.get_action_meanings()` once and assert the six action indices. If a ROM/variation is changed, run an input calibration episode: apply action 2 and action 3 for a few frames, observe the right paddle Y movement, and store the mapping.
5. Feed `obs` to a ROM-specific state parser. During development, make a second environment with `obs_type="rgb"` or inspect `env.unwrapped.ale.getScreenRGB()` to validate the parser. Do not send RGB pixels to Jev if the parser can provide the requested JSON state.
6. On each frame, call `obs, reward, terminated, truncated, info = env.step(action)`. Reset only when `terminated or truncated` is true. Count positive and negative rewards rather than inferring points from elapsed frames.
7. Call `env.close()` at the end of the run. Use `clone_state`/`restore_state` only for local planning or test replay, not as a substitute for the evaluator's reset contract.

The lower-level API can avoid Gymnasium if needed:

```text
from ale_py import ALEInterface, roms
ale = ALEInterface()
ale.loadROM(roms.get_rom_path("pong"))
legal_actions = ale.getLegalActionSet()
reward = ale.act(action)
ram = ale.getRAM()
```

The exact lower-level method names exposed by the installed wheel should be checked with `dir(ale)`; the documented Python interface includes `loadROM`, `getLegalActionSet`, `act`, `game_over`, and the current binding additionally exposes RAM/screen/state methods. Gymnasium is preferred because it standardizes reset, step, termination, and observation spaces.

### ROM and configuration details

- The bundled lookup is `ale_py.roms.get_rom_path("pong")`. It validates the ROM hash and returns the package path. `ALE_ROMS_DIR` can optionally override the ROM directory, but it is not needed and should remain unset for the bundled copy.
- `ALE/Pong-v5` documents modes `[0, 1]` and difficulties `[0, 1, 2, 3]`, both defaulting to zero. The adaptor should expose these as explicit constructor settings for scenario generation.
- Default v5 settings are `frameskip=4` and `repeat_action_probability=0.25`; they are useful for benchmark compatibility but make precise paddle control and replay less predictable.
- A raw `seed` is not a substitute for scenario selection. The reverse-engineering notes for Video Olympics report no internal random-number generator; if the same mode/difficulty is used, held-out seeds may reproduce the same opening. Vary mode/difficulty or a documented initial state when meaningful generalization is required.

### State representation for Jev/Astra

ALE does not provide a ready-made Pong JSON object model. The adaptor should expose a compact state such as:

```text
{
  "frame": 1234,
  "right_paddle_y": 104,
  "left_paddle_y": 96,
  "ball_x": 132,
  "ball_y": 87,
  "ball_vx_sign": 1,
  "ball_vy_sign": -1,
  "ball_approaching_right": true,
  "own_points": 7,
  "opponent_points": 5,
  "serve_ready": false,
  "difficulty": 0
}
```

That is the adaptor's derived schema, not an ALE guarantee. Begin with RGB object detection (paddles, ball, scoreboard) and use it to label RAM traces. The FujiNet Video Olympics porting notes provide useful ROM-specific candidates: variation `$96`, raw paddle values `$84-$85`, filtered paddle positions `$BB-$BE`, and player-position cells around `$B2-$B3`; they also warn that the stock ROM reads TIA paddle mirrors at `$38-$3B`. Treat those addresses as candidates to validate against the exact dump, not as universal Atari RAM conventions. Source: [Video Olympics PORTING.md](https://raw.githubusercontent.com/FujiNetWIFI/fujinet-2600-video-olympics/main/PORTING.md).

The parser should output positions and signs/phase, not ask Jev to perform arithmetic. Code should project a ball path to the right paddle and give Jev only typed recognition/choice questions such as “Is the ball approaching the right paddle: yes/no?” or “Which calibrated direction moves the paddle toward the projected target: LEFT/RIGHT/HOLD?”

## Other practical interfaces

### 1. ALE C++ library directly

- **Maturity:** Same mature emulator as `ale-py`; the official installation page documents a C++17/CMake build and `ale::ale-lib`.
- **Latency:** Lowest overhead and straightforward frame stepping; suitable if a Python parser or model bridge becomes the bottleneck.
- **Exposes:** ALE actions, RAM, screen, rewards, modes/difficulties, and emulator state. A small local JSON/pipe bridge would still have to be written.
- **Cost:** Free/open source; SDL is optional and can remain disabled for headless use.
- **Legal/ToS:** GPL-2.0 and ROM copyright considerations; no online service.
- **Why not chosen:** It adds a compiler/CMake/native-bridge maintenance burden when the CPython 3.12 wheel already exposes the needed controls and RAM.
- **Sources:** [ALE installation/C++ build](https://ale.farama.org/getting-started/), [ALE repository](https://github.com/Farama-Foundation/Arcade-Learning-Environment).

### 2. Stella standalone emulator and debugger

- **Maturity:** Very high accuracy and mature Atari 2600 emulator; official project and documentation.
- **Latency:** Emulator execution is fast, but automation through its debugger console/script files is interactive and not a stable low-latency Python IPC interface.
- **Exposes:** CPU/TIA state, zero-page RAM, ROM, scanline/frame stepping, breakpoints/traps, save/load states, disassembly labels, and debugger scripts. Commands documented in the debugger include `ram`, `dump`, `step`, `scanLine`, `frame`, `saveState`, `loadState`, and `exec`.
- **Control:** Configure a paddle controller and issue debugger/controller input. The public documentation does not promise a remote real-time action protocol or a pure headless object API.
- **Cost:** Free, GPL-licensed.
- **Legal/ToS:** Local emulator only; same need to have rights to the ROM.
- **Best use:** Validate the ALE ROM, discover RAM labels, inspect collision timing, and make save-state regression tests. It is a poor primary adaptor because there is no documented external frame/action socket and no guaranteed display-less runtime.
- **Sources:** [Stella guide](https://stella-emu.github.io/docs/index.html), [Stella debugger](https://stella-emu.github.io/docs/debugger.html), [Stella source](https://github.com/stella-emu/stella).

### 3. Libretro Stella core / RetroArch

- **Maturity:** Mature emulator core and broad frontend ecosystem. The Stella core supports Atari paddle-style devices and the libretro input model.
- **Latency:** A native core can step frames quickly, but a RetroArch frontend or an extra Python binding introduces IPC/ctypes and configuration latency.
- **Exposes:** Pixels/audio, input descriptors, savestates depending on frontend, and libretro memory facilities. The core documentation does not define a stable Pong-specific JSON/RAM map. Memory monitoring is documented, but direct arbitrary RAM access depends on the frontend/binding.
- **Control:** RetroPad/directional input plus the paddle device mapping; specialized paddle input may be necessary because a normal joystick mapping is not equivalent to the original analog knob.
- **Cost:** Free; RetroArch/core installation varies by distribution.
- **Legal/ToS:** No online account required; ROM rights still apply. RetroArch netplay is unnecessary and should be disabled.
- **Best use:** A fallback if the evaluator already has RetroArch/libretro infrastructure, not a first choice for a new headless Python adaptor. `libretro.py` is a low-level Python binding (documented as requiring Python 3.12+) but still requires obtaining/configuring the Stella core and writing the state bridge.
- **Sources:** [Stella libretro core](https://docs.libretro.com/library/stella/), [libretro.py](https://github.com/warmenhoven/libretro.py), [Libretro API](https://www.libretro.com/index.php/api/).

### 4. MAME Atari 2600 driver/debugger

- **Maturity:** MAME is mature, but its Atari 2600 driver/debugger is a general emulation target rather than a Gym-style environment.
- **Latency:** Native frame execution is adequate; extracting RAM and injecting paddle input through the debugger or frontend is cumbersome for a per-frame Python agent.
- **Exposes:** Video, emulated input, debugger CPU/RAM access, and state tools. A FujiNet Video Olympics porting project documents configuring MAME's Atari paddle ports as `-joyport1 pad -joyport2 pad`, which is a useful reference for the unusual controller type.
- **Cost:** Free/open source.
- **Legal/ToS:** Local only; use a legally obtained ROM.
- **Best use:** Cross-check emulation/controller behavior or use an existing MAME test harness. It lacks ALE's ready-made reward/termination/action-space contract, so it is not chosen.
- **Sources:** [MAME](https://www.mamedev.org/), [MAME documentation](https://docs.mamedev.org/), [FujiNet Video Olympics port notes](https://raw.githubusercontent.com/FujiNetWIFI/fujinet-2600-video-olympics/main/PORTING.md).

### 5. Screen capture plus input injection

- **Maturity:** Technically universal but fragile. A practical stack would be Stella or RetroArch under Xvfb, capture via a screenshot API/OpenCV, and inject keyboard/controller events with an evdev/uinput or X11 tool.
- **Latency:** Worst option: rendering, capture, image processing, and event injection add scheduling jitter. A model call must never be in the capture/control loop.
- **Exposes:** Only pixels and whatever scoreboard OCR can recover; no RAM, collision latches, reward, or reliable termination signal.
- **Control:** Keyboard events mapped to left/right/fire, or a virtual paddle device. Atari's analog paddle semantics are difficult to reproduce with ordinary keys.
- **Cost:** Free packages, but Xvfb and GUI dependencies complicate the offline server; no GPU is strictly required for software rendering.
- **Legal/ToS:** No online risk, but local input injection needs host permissions. Screenshot pipelines may accidentally write game images/artifacts.
- **Best use:** RGB validation and emergency fallback only. ALE's `rgb_array` mode is simpler and more deterministic than a desktop capture loop.

### 6. Small Python Atari 2600 emulators

- **Candidates:** [pytari2600](https://github.com/ajgrah2000/pytari2600) and [pytari](https://github.com/asterick/pytari).
- **Maturity:** Hobby/early-stage projects with limited maintenance and no ALE-compatible Pong contract. `pytari2600` documents pygame/pyglet-oriented operation and reports roughly one-third real-time performance in one configuration; GUI/display coupling is a problem on this server.
- **Latency/exposure:** Potentially inspectable Python CPU/RAM state, but no stable, tested symbolic Pong observation or frame-step API was found. Pygame/display dependencies and old versions make CPython 3.12 compatibility uncertain.
- **Cost/legal:** Free projects (check each repository license); ROM rights still apply.
- **Best use:** Educational reference, not an evaluator adaptor. They lose on accuracy, maintenance, headless readiness, and performance.

## Credentials and environment variables

No credentials, account, server address, or API key are required. Leave the optional `ALE_ROMS_DIR` variable unset to use the wheel's hash-checked bundled ROM. If an operator supplies a licensed external ROM directory, set `ALE_ROMS_DIR` to that directory and ensure it contains the expected `pong.bin`; never write the value into source or study files.

## Operational failure modes

- Installing into system Python instead of `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-pong/.venv/bin/python`.
- Enabling default sticky actions/frameskip while assuming deterministic one-frame control.
- Treating the 128 RAM bytes as a standardized object schema; the parser must be ROM/version validated.
- Using joystick up/down or screen left/right assumptions for a paddle ROM; calibrate action 2/3.
- Requiring SDL, `render_mode="human"`, X11, or a GPU on the headless evaluator.
- Assuming `seed` changes a deterministic Video Olympics opening; vary mode/difficulty or explicit initial state for scenario diversity.
- Distributing an Atari ROM or a modified ROM without checking the applicable rights.
