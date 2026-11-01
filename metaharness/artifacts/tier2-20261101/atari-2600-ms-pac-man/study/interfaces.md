# Programmatic interfaces for Atari 2600 Ms. Pac-Man

## Recommendation

Use **ALE/ale-py running the bundled Atari 2600 ROM, wrapped by OCAtari's RAM Extraction Mode (REM)**. The base emulator is local, deterministic when configured so, headless, fast, and exposes the Gymnasium action/reward/reset contract plus the 128-byte Atari RAM. OCAtari adds a maintained MsPacman decoder for Ms. Pac-Man, ghosts, fruit, power pills, ordinary pills, score and lives, which is much closer to the requested object-level JSON state than a screenshot pipeline.

The recommendation is not to trust OCAtari as the sole source of truth. Keep the raw 128-byte RAM and (optionally) the 210x160 RGB frame alongside the serialized objects. OCAtari cannot recover maze walls from RAM; its own paper calls this out for Pac-Man/Ms. Pac-Man. Calibrate a wall/turn graph from RGB once, or retain a verified CX2675 map, then use REM/RAM for the dynamic state. This still avoids screenshots during normal action selection.

## Chosen stack: ALE + OCAtari REM

### Software and compatibility

The evaluator is Python 3.12.3. The following versions were installed and exercised in an isolated target using that interpreter on a headless Linux host:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-ms-pac-man/.venv/bin/python
  gymnasium==1.4.0
  ale-py==0.12.1
  ocatari==2.2.1
```

Install explicitly into the evaluator's venv, rather than relying on the system Python:

```text
$PY -m pip install "gymnasium==1.4.0" "ale-py==0.12.1" "ocatari==2.2.1"
```

where `$PY` is the evaluator path above. OCAtari's package declares Python versions above 3.9 and depends on Gymnasium, NumPy below 2, OpenCV, scikit-image, termcolor, seaborn, pandas, scikit-learn, keyboard, tqdm, pygame and pyfiglet. Its PyPI metadata does not declare all of those dependencies reliably, so let pip resolve them and explicitly install `ale-py` as well. `ale-py 0.12.1` publishes a CPython 3.12 manylinux x86-64 wheel and requires Python >=3.10.

The current ale-py package contains `ale_py/roms/ms_pacman.bin`; its size is 8,192 bytes and its MD5 matches ALE's supported MsPacman checksum. This avoids asking the user for an external ROM. If a future package stops bundling ROMs, follow the official ALE/Gymnasium ROM-install instructions and obtain a legally usable CX2675-compatible ROM; do not put a ROM in this study directory or commit one.

*Sources:* [OCAtari README/install instructions](https://github.com/k4ntz/OC_Atari), [OCAtari PyPI](https://pypi.org/project/ocatari/), [ale-py PyPI](https://pypi.org/project/ale-py/), [ALE getting started](https://ale.farama.org/getting-started/), [ALE ROM/checksum declaration](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.hpp).

### Construction and reset

The adaptor phase should:

1. Use the evaluator's Python 3.12.3 and install the pinned packages above.
2. Import `ale_py` and register its Gymnasium environments (`gymnasium.register_envs(ale_py)` for current releases).
3. Construct `OCAtari("ALE/MsPacman-v5", mode="ram", hud=True, obs_mode="obj", render_mode=None, ...)`. `mode="ram"` here means OCAtari extraction mode; it is not the Atari game variation.
4. Pass `frameskip=1, repeat_action_probability=0.0` while calibrating and debugging. For a benchmark matching the documented v5 defaults, use `frameskip=4, repeat_action_probability=0.25`; record which setting was used.
5. Call `reset(seed=seed)`, retain `info`, and serialize both `env.objects` and `env.ale.getRAM()` immediately. Call `close()` after an episode.
6. Use `env.ale.setMode(0)`, `setMode(1)`, `setMode(2)` or `setMode(3)` before a reset when a separate one-, two-, three- or four-ghost scenario is desired. Because OCAtari reserves the keyword `mode` for extraction, do not try to pass the Gym game-mode integer in that same keyword. Verify the selected variation by checking the visible ghost count or RAM byte 19.
7. Treat `env.ale.game_over()` and the underlying ALE/Gym termination state as authoritative. OCAtari 2.2.1's `step` implementation returns the Gymnasium termination flags in swapped tuple positions (`obs, reward, truncated, terminated, info` despite the usual order); normalize this in the adaptor or use the underlying ALE terminal query. This is a known wrapper hazard, not a game rule.

The default `ALE/MsPacman-v5` identifier is the supported current identifier. Do not assume that `ALE/MsPacman-ram-v5` exists: with ale-py 0.12.1/Gymnasium 1.4.0, `obs_type="ram"` on `ALE/MsPacman-v5` works, while the separate `-ram-v5` identifier does not.

*Sources:* [OCAtari core API](https://oc-atari.readthedocs.io/en/latest/ocatari/core.html), [OCAtari source](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/core.py), [OCAtari swapped-return issue](https://github.com/k4ntz/OC_Atari/issues/53), [ALE MsPacman environment](https://ale.farama.org/environments/ms_pacman/).

### Action API and timing

`env.action_space` is `Discrete(9)`:

```text
0 NOOP       1 UP       2 RIGHT       3 LEFT       4 DOWN
5 UPRIGHT    6 UPLEFT  7 DOWNRIGHT    8 DOWNLEFT
```

The adaptor should normally emit only 0–4. Each ALE step advances the emulator by the configured frame skip. Atari runs at 60 Hz, so `frameskip=1` is approximately 16.7 ms and `frameskip=4` is approximately 66.7 ms. v5's default sticky-action probability is 0.25; set it to zero for repeatable calibration/evaluation or retain it deliberately as a robustness scenario. A direction is not a turn-based move: the world, Ms. Pac-Man and ghosts continue moving while Astra or Jev is thinking.

The base Gymnasium observation can be selected as:

* `obs_type="ram"`: `uint8[128]`, the raw Atari RAM.
* default/RGB: `uint8[210, 160, 3]` (height, width, RGB), useful only for map calibration or a fallback visual recognizer.
* grayscale: `uint8[210, 160]`.

`reward` is the score delta for the frames advanced. `info` includes at least `lives`, `episode_frame_number` and `frame_number` in current ALE. The direct ALE object additionally exposes `act`, `getRAM`, `getScreenRGB`, `getLegalActionSet`, `getMinimalActionSet`, `getAvailableModes`, `setMode`, `lives`, `game_over`, `getEpisodeFrameNumber` and `getFrameNumber`.

*Sources:* [ALE MsPacman action/observation specification](https://ale.farama.org/environments/ms_pacman/), [ALE Python interface](https://ale.farama.org/python-interface/), [ALE environment specification](https://ale.farama.org/env-spec/).

### OCAtari object state to serialize

With `hud=True`, OCAtari's released MsPacman RAM decoder allocates these object slots:

* one `Player` (Ms. Pac-Man);
* four `Ghost` slots, represented by `NoObject` when absent;
* one `Fruit`;
* four `PowerPill` slots;
* 252 `Pill` slots, with `Pill.grid_ij` identifying the decoder's row/column location;
* HUD `Score` and `Life` value objects.

Do not send the Python object repr directly to Jev. Serialize each slot as stable JSON with at least `slot`, `category`, `visible`, `xy`, `wh`, `dx`, `dy`, and, where present, `value`, `rgb`, `orientation` and `grid_ij`. In OCAtari, `bool(obj)` is the visibility test; an absent slot is a `NoObject`. The released 2.2.1 decoder exposes Player/Ghost/Fruit/PowerPill/Pill positions and Score/Life values. A newer source revision adds a player orientation from RAM byte 56, but do not assume that field exists in 2.2.1; include it only when present or derive it from raw RAM.

The released decoder's relevant RAM interpretation is:

| RAM byte | Meaning used by the MsPacman decoder |
|---:|---|
| 6–9 | ghost X positions (orange, cyan, pink, red slots) |
| 10 | player X |
| 11 | fruit X |
| 12–15 | ghost Y positions |
| 16 | player Y |
| 17 | fruit Y |
| 19 | number of additional ghosts; 3 means four total, 0/1/2 mean one/two/three total |
| 47 | nonzero when the ghost group is active/visible in the decoder |
| 56 | direction/orientation in the upstream decoder: low two bits 0/1/2/3 = up/right/down/left |
| 59 onward | pill-presence bitfields, interpreted with one of two hard-coded grids |
| 117 | bits 2–5 indicate the four remaining power pills |
| 120–122 | BCD-like score bytes, decoded by OCAtari into `Score.value` |
| 123 | fruit/level and spare-life display byte; OCAtari uses `byte % 4` for its `Life.value` |

The exact object coordinates are screen-pixel coordinates after OCAtari's offsets (for example player X is RAM byte 10 minus 13 and player Y is byte 16 plus 1). Keep raw bytes too, because these are reverse-engineered fields rather than a stable official schema. The byte 19 mode convention is confirmed by probing ALE's four modes and by OCAtari's decoder source.

Important omissions and quirks:

* REM provides dynamic objects, not the maze's wall geometry. OCAtari's paper explicitly says that Pac-Man/Ms. Pac-Man maze information is not in RAM. Its decoder uses two hard-coded pill grids; do not interpret that as a complete four-maze wall map.
* The `Ghost.rgb` values in the released RAM decoder are normal ghost colours; they do not reliably say that a ghost is currently frightened/blue. Use an RGB calibration stream or a temporal state detector if Jev needs frightened status.
* `Life.value` is the spare-life icon count (initially two in the decoder), while ALE `info["lives"]` reports the current life count (initially three). Report the latter as total lives.
* OCAtari's `obs_mode="obj"` is a numeric stacked observation, not JSON. In the tested 2.2.1 configuration, `hud=True` produces 265 object slots and a four-frame object stack; use `env.objects`/raw RAM for a named JSON state instead of relying on the flattened vector ordering.

*Sources:* [OCAtari RAM documentation](https://oc-atari.readthedocs.io/en/latest/ocatari/ram.html), [released MsPacman decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py), [OCAtari object model](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/game_objects.py), [OCAtari paper and RAM limitations](https://arxiv.org/html/2306.08649v2).

### Headless operation

Use `render_mode=None` for normal operation. ALE does not need an X display or GPU for RAM observations and action stepping. `render_mode="rgb_array"` is still headless and is useful for one-time wall-map calibration, but it costs memory/bandwidth and should not be sent to Jev every question. Do not use OCAtari's human rendering or overlay on the evaluator; those paths initialize Pygame. If some dependency initializes a display unexpectedly, `SDL_VIDEODRIVER=dummy` is an optional environment setting, not a credential and not required for the no-render path.

## Other practical interfaces

### 1. Raw `ale-py` / Gymnasium (high maturity, preferred fallback)

* **What it exposes:** local Stella/ALE emulation; reset with a seed; 9 actions; 128-byte RAM or RGB/grayscale frames; reward score delta; lives and frame counters; direct ALE RAM/screen/mode/terminal methods.
* **Latency:** excellent. Native C++ emulation runs much faster than real time; with `frameskip=1` it can provide every 60-Hz frame, and with `frameskip=4` it provides the standard 67-ms action quantum.
* **Maturity:** high and widely used in Atari RL. It is the least fragile fallback if OCAtari's decoder changes. It requires the adaptor to maintain the MsPacman RAM map and a maze map itself.
* **Cost:** free/open source package; no account or server. The ROM remains copyrighted and must be used under the applicable ALE/ROM terms.
* **ToS/legal risk:** no online service or anti-cheat risk. Do not redistribute the Atari ROM outside the permitted package/use. ALE and Stella are open-source projects, but the game asset is not thereby public domain.
* **How to use:** `gymnasium.make("ALE/MsPacman-v5", obs_type="ram", frameskip=..., repeat_action_probability=...)`; use the direct ALE handle for `getRAM`, `getScreenRGB`, `setMode` and `game_over`.

*Sources:* [ALE getting started](https://ale.farama.org/getting-started/), [ALE Python interface](https://ale.farama.org/python-interface/), [ALE MsPacman](https://ale.farama.org/environments/ms_pacman/), [ALE GitHub](https://github.com/Farama-Foundation/Arcade-Learning-Environment).

### 2. Direct ALE/`ALEInterface` with a ROM path (high maturity, less convenient)

The lower-level Python interface loads a ROM explicitly, then exposes `setInt`, `loadROM`, `getLegalActionSet`, `act`, `getRAM`, `getScreen`, `game_over`, and related methods. It is useful for experiments that need precise ALE state control, savestate integration, or no Gymnasium wrapper. It is less convenient for the adaptor because it returns raw rewards and RAM rather than the standard five-value Gymnasium step result and requires a legal ROM path. Prefer Gymnasium unless direct ALE methods are needed.

*Latency:* excellent. *Cost:* free package plus legally obtained ROM. *Risk:* same ROM copyright; lower-level API changes and Python bindings need pinning. *Maturity:* high for research, but less ergonomic than Gymnasium.

*Source:* [official ALE Python interface](https://ale.farama.org/python-interface/).

### 3. OCAtari Vision Extraction Mode (medium maturity, fallback for missing RAM fields)

`OCAtari(..., mode="vision", render_mode="rgb_array")` applies handwritten colour/shape rules to the rendered frame and returns objects with category, position, dimensions and RGB values. It can detect visual state unavailable in REM, such as a frightened/blue ghost if the detector recognises it, and can validate the maze map. It is slower and less reliable than REM around blinking/overlap; it also needs the RGB frame. Use it for calibration or a hybrid REM+vision check, not as the primary Jev feed.

*Latency:* still local and usually adequate, but slower than direct RAM; varies with OpenCV/Python. *Cost:* same pip install. *Risk:* version-specific handwritten detectors; no online/anti-cheat risk. *Maturity:* useful research software, not an official Atari API.

*Sources:* [OCAtari README](https://github.com/k4ntz/OC_Atari), [OCAtari paper (VEM vs REM)](https://arxiv.org/html/2306.08649v2).

### 4. Stella native emulator/debugger (mature emulator, poor live-agent interface)

Stella is the underlying Atari 2600 emulator family. Its command line can load a cartridge, choose speed/software rendering, start the debugger, inspect/edit RAM, step frames or scanlines, and manipulate joystick registers. Debugger scripts can automate a ROM-specific session. This is valuable for validating the CX2675 ROM, discovering RAM addresses, and investigating a terminal or collision quirk.

* **Latency:** native-fast when driven locally, but debugger command/file interaction is not a clean per-step Python API.
* **Observation/control:** raw CPU/RAM/debugger state, screenshots and joystick commands; no ready-made JSON MsPacman objects.
* **Headless maturity:** the emulator is mature, but the documented debugger is GUI-oriented and a true no-display live-control mode is not promised. Xvfb/software rendering may be needed.
* **Cost/legal:** free/open source; use a legally obtained ROM. No account, server or anti-cheat risk.

Use Stella as a diagnostic fallback, not the main harness interface.

*Sources:* [Stella documentation](https://stella-emu.github.io/docs/index.html), [Stella debugger](https://stella-emu.github.io/docs/debugger.html).

### 5. RetroArch/libretro Atari 2600 core (mature generic emulator, low symbolic maturity)

A libretro Atari 2600 core under RetroArch can run a ROM without a physical display, accept joypad input, and expose video frames. A libretro frontend or plugin can drive it at a chosen rate; FFmpeg/Xvfb or a memory video callback can provide frames. This is useful if the deployment already standardises on RetroArch, but it has no MsPacman-specific RAM/object decoder and requires a custom bridge for state, timing and input.

* **Latency:** potentially excellent, but bridge and video-copy latency are the adaptor's responsibility.
* **Observation/control:** RGB/video frames and generic buttons; core-specific memory/debug APIs are not as portable as ALE.
* **Maturity/cost:** mature emulator ecosystem, free software; setup is considerably more involved than pip ALE.
* **Risk:** ROM copyright and configuration complexity; no online ToS risk.

*Sources:* [libretro documentation](https://docs.libretro.com/), [RetroArch documentation](https://www.retroarch.com/).

### 6. MAME or another debugger-capable console emulator (possible, not recommended)

MAME can emulate many consoles and has debugger/console and scripting facilities. It can be made to run an Atari 2600 cartridge, but it is a general emulator: the adaptor must identify the correct machine, inject joystick events, and locate/interpret RAM. It offers no ready-made MsPacman object state and is redundant with ALE/Stella for this target.

*Latency:* high. *Observation/control:* debugger memory and generic video/input. *Maturity:* high emulator, low target-adaptor maturity. *Cost/legal:* free/open source with the same ROM obligations. *Risk:* version/configuration and headless-launch complexity.

*Source:* [MAME documentation](https://docs.mamedev.org/).

### 7. Screen capture plus input injection (last-resort fallback)

Run Stella, RetroArch, or a browser emulator inside Xvfb; capture the 160x210/scale-up frame with an X11 capture library or `mss`; inject joystick directions with XTest/`xdotool`/`pynput`/a libretro input callback. A CV layer can locate Ms. Pac-Man, ghosts, fruit and maze pixels.

* **Maturity:** low-to-medium for this game; generic tools are mature but the detector and calibration are new work.
* **Latency:** poor relative to RAM. Screenshot copies, CV and IPC can exceed the approximately 67-ms default action quantum; buffering is essential.
* **Observation/control:** only pixels and synthetic key/joystick events; no score/RAM unless OCR/CV is added.
* **Cost:** free packages, but Xvfb and a software renderer must be installed; no GPU required.
* **Legal/ToS:** local emulator has no service ToS, but browser-hosted versions may prohibit automation and may rate-limit or change their frontend. Do not use a public browser game for evaluation.

This is useful only if ALE/OCAtari cannot be installed or if a visual frightened-state/map check is needed.

*Sources:* [Stella documentation](https://stella-emu.github.io/docs/index.html), [AtariOnline browser version](https://atarionline.org/atari-2600/ms-pac-man), [RetroGames browser version](https://www.retrogames.cz/play_076-Atari2600.php?language=EN).

### 8. Game-specific bot libraries and online/browser APIs (not viable here)

There is no official Atari/Namco Ms. Pac-Man gameplay API, account service, or mature bot library for the CX2675 Atari 2600 ROM. Arcade Ms. Pac-Man bots and MAME libraries target a different arcade ROM, maze, timing and memory layout; they cannot be substituted without changing the game. Browser sites expose a canvas/emulator for human play, not a stable local state/action API, and the user has no display or account. These options have high maintenance/ToS risk and should not be selected.

## Chosen interface contract for the next adaptor phase

The adaptor should expose a JSON observation containing:

```text
frame: native ALE frame number
score: ALE HUD/decoded score
reward_delta: reward from the last ALE step
lives: ALE info lives (not OCAtari spare-icon value)
game_over: ALE terminal truth
mode: 0..3 ghost-count variation
player: {x, y, direction_if_available, dx, dy}
ghosts: [{slot, x, y, dx, dy, colour, visible}]
fruit: {visible, x, y, kind_if_decoded}
power_pills: [{x, y, visible}]
pills: [{row, col, x, y, visible}]
maze: {map_id, known_walls_or_turn_graph, map_confidence}
raw_ram: [128 unsigned bytes]
```

The adaptor controls one of the nine ALE action integers and should retain the last direction/action, because the world continues between model calls. It should provide a short action queue so Jev/Astra latency does not pause emulation, stop or replan the queue at a collision/turn hazard, and keep Astra's slower strategic state separate from Jev's per-state typed direction choice. No credentials, server address, account, or API key are required for this local design.
