# Programmatic interfaces for Atari 2600 Space Invaders

## Recommendation

Use the **local ALE emulator through `ale-py` and Gymnasium, with `obs_type="ram"` as the primary observation and RGB only as a calibration/fallback channel**. The evaluator's Python 3.12.3 was tested with `gymnasium[atari]`: current packages installed were `ale-py 0.12.1` and `gymnasium 1.4.0`; `space_invaders.bin` was present in the package and the environment reset successfully with no display. ALE is native Stella, fast, deterministic when configured, supports seeds, raw RAM, score/lives/termination, and clone/restore snapshots. It has no server, account, display, GPU, anti-cheat, or game-website ToS dependency.

Important limitation: ALE does **not** publish a game-specific object API for this ROM. `obs_type="ram"` is the 128 raw console bytes, not a JSON list of aliens. The Computer Archaeology RAM page currently labels locations generically (`m80` through `mFF`) and does not provide a complete supported semantic map ([RAM page](https://www.computerarcheology.com/Atari2600/SpaceInvaders/RAMUse.html)). The adaptor must therefore normalize RAM into JSON itself, using a decoder validated against RGB frames and score/life transitions. That still avoids screenshots in the Astra/Jev interface: the models receive a compact object state, while the emulator remains local and headless.

## Install and smoke test

Run all commands with the evaluator interpreter, not the system Python:

```bash
PY=/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-space-invaders/.venv/bin/python
"$PY" -m pip install --upgrade 'gymnasium[atari]'
```

The current ALE README says that `ale-py` packages the Atari ROMs and that `gymnasium[atari]` installs the required modules and ROMs ([official README](https://github.com/Farama-Foundation/Arcade-Learning-Environment#quick-start)). There is no need for AutoROM or an online ROM download with the current package. If an older locked environment is used, follow that release's ROM/license instructions rather than silently downloading a ROM.

Minimal **interface setup** (not a policy) is:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make(
    "ALE/SpaceInvaders-v5",
    obs_type="ram",
    frameskip=1,                 # use 4 for the standard v5 step size after calibration
    repeat_action_probability=0.0, # set 0 for repeatable calibration
    mode=0,
    difficulty=0,
)
obs, info = env.reset(seed=seed)
# adaptor calls env.step(one of 0..5), then closes with env.close()
```

For the direct interface, which is useful when Astra wants snapshot-based planning or when Gym wrappers are undesirable:

```python
from ale_py import ALEInterface, roms
ale = ALEInterface()
ale.setInt("random_seed", seed)
ale.setFloat("repeat_action_probability", 0.0)
ale.setInt("frame_skip", 1)
ale.loadROM(roms.get_rom_path("space_invaders"))
ale.reset_game()
ram = ale.getRAM()
frame = ale.getScreenRGB()
state = ale.cloneState(include_rng=True)
```

Settings that affect emulation must be set before `loadROM`. The direct API and its state methods are documented in the [ALE Python interface](https://ale.farama.org/python-interface/) and [environment specification](https://ale.farama.org/env-spec/).

## Chosen interface details

### ALE/Gymnasium RAM environment — chosen

- **Maturity:** Production/stable ALE package, widely used Atari RL interface; native Stella emulator. Current official package metadata supports CPython 3.11–3.15, so the supplied 3.12.3 interpreter is in range.
- **Latency:** Native emulator step is far faster than real time on a CPU. With `frameskip=1`, one action advances one Atari frame (~16.7 ms NTSC); v5's standard `frameskip=4` advances ~66.8 ms. Python observation allocation and an optional NumPy decoder are the main overhead. `repeat_action_probability=0` gives deterministic calibration; v5's documented default is 0.25 for robust evaluation.
- **Observation:** `obs_type="ram"` is `Box(0,255,(128,),uint8)`. Gym's `info` includes at least `lives`, `episode_frame_number`, and `frame_number` for this environment. A parallel `obs_type="rgb"` environment returns `(210,160,3)` `uint8`; grayscale returns `(210,160)`. `reward` is the score increment, and `terminated` covers game over; `truncated` is an adaptor/time-limit condition.
- **Control:** Gym `Discrete(6)` is exactly `NOOP=0`, `FIRE=1`, `RIGHT=2`, `LEFT=3`, `RIGHTFIRE=4`, `LEFTFIRE=5`. Use combinations so movement and firing can happen on the same step. `full_action_space=True` exposes the 18-action Atari set, but the extra up/down actions are ineffective here and should not be sent.
- **Snapshots:** Direct `ALEInterface.cloneState()`/`restoreState()` support planning and regression tests. The default clone omits the sticky-action RNG; use `include_rng=True` when exact future replay is required. Serialized states are not guaranteed compatible across ALE versions.
- **Cost/legal risk:** No account or service cost. `ale-py` is GPL-2.0-only; the packaged Atari ROM is copyrighted game data, so do not redistribute the ROM or assume the package license grants independent commercial ROM rights. Use the packaged ROM under the package/evaluator's permitted terms and keep it local. There is no online anti-cheat or game-server ToS issue.

### How to produce the adaptor's object JSON

The model-facing observation should be a compact, versioned JSON object, for example:

```text
{
  "frame": integer,
  "score": integer,
  "lives": integer,
  "game_over": boolean,
  "wave": integer_or_null,
  "player": {"x": integer, "y": integer},
  "player_shot": {"active": boolean, "x": integer, "y": integer_or_null},
  "invaders": [{"row": 1..6, "col": 0..5, "x": integer, "y": integer, "alive": boolean}],
  "bombs": [{"x": integer, "y": integer}],
  "shields": [{"x": integer, "y": integer, "health_mask": ...}],
  "ufo": {"active": boolean, "x": integer_or_null}
}
```

Build it in two stages:

1. **Calibration/development:** run a second `obs_type="rgb"` stream with `frameskip=1` and no sticky actions. The initial canonical frame has six evenly spaced columns and six rows; in the current ALE RGB coordinates invader boxes begin around x=22,38,54,70,86,102 and y=31,49,67,85,103,121, three shield regions are around x=42,74,106 and y=157–174, and the player is near the bottom around y=185–194. These are calibration observations, not a promise that every PAL/variation frame has identical coordinates. Detect the known palette/object geometry and account for sprite flicker.
2. **Make the RGB fallback symbolic, not a screenshot prompt:** for the canonical NTSC ROM, classify the background `[0,0,0]`, invader `[134,134,29]`, shield `[181,83,40]`, player `[162,134,56]`, and lower/UI `[50,132,50]`/`[80,89,22]` colors only inside fixed ROIs. Cluster invader pixels into the 6x6 grid (about 16 pixels of column spacing and 18 pixels of row spacing), mark a cell alive when its expected sprite mask has enough pixels, locate the player as the contiguous lower sprite, and retain a per-shield occupancy mask. Infer moving projectiles/UFOs from non-background components and frame-to-frame differences, then reject candidates in the score/border ROIs. Calibrate thresholds and palette at reset and use reward/life transitions as ground truth for hit/death tests; do not assume these colors survive PAL or special invisible modes.
3. **Runtime:** use RAM as the cheap primary signal, maintain prior state, and verify the decoder against occasional RGB observations during tests. If a RAM semantic cannot be proven across a wave/variation, retain the RGB-derived object for that field rather than inventing a RAM address. Never present the raw 128 bytes directly to Jev and call them object state; include decoded positions, alive bits, projectile hazards, score, and life/terminal flags.

The Computer Archaeology [disassembly](https://www.computerarcheology.com/Atari2600/SpaceInvaders/Code.html) and [RAM page](https://www.computerarcheology.com/Atari2600/SpaceInvaders/RAMUse.html) are useful for validating a decoder, but they are reverse-engineering notes, not a stable ALE API. If exact object symbols are essential, the safer fallback is a checked-in ROM-specific decoder plus golden RAM/frame fixtures, not guessed addresses.

## Other practical interfaces

| Interface | Maturity / latency | What it exposes | Legal/ToS and cost | Verdict |
|---|---|---|---|---|
| **`ale-py` direct `ALEInterface`** | Production native API; lowest Python overhead and frame-level control | ROM loading, minimal/legal action set, reward, game-over, 128-byte RAM, RGB/grayscale/raw screen, settings, clone/restore state | Local only; same GPL/ROM provenance issue; free | Nearly as good as the chosen Gym wrapper. Prefer it for snapshot search, custom frame stepping, or a direct adaptor. |
| **ALE Gymnasium RGB / grayscale** | Production; same emulator latency, but pixel arrays and Python wrapper overhead | `(210,160,3)` RGB or `(210,160)` grayscale, actions, reward, lives, reset/step API | Local; no account/cost; ROM rights still apply | Reliable fallback/calibration. More CPU and harder for Jev than decoded JSON. |
| **ALE Gymnasium `NoFrameskip`/older v0/v4 IDs** | Mature compatibility variants | RGB by default; v0/v4 have old stochastic-frame-skip/sticky-action semantics | Same local ROM issue; old APIs may disappear | Use only to reproduce an existing benchmark. For new work use `ALE/SpaceInvaders-v5` and explicitly set frameskip/stickiness. The variant table is in the [official environment page](https://ale.farama.org/environments/space_invaders/). |
| **ALE C++ library** | Native and fast; requires a C++17/CMake build | Direct Stella access, RAM/screen/state, settings and stepping | Local; build dependencies and GPL/ROM considerations; free | Good for a very high-rate adaptor, but unnecessary for Astra/Jev and harder to install in the supplied Python venv. See [C++ setup](https://ale.farama.org/cpp-interface/). |
| **Stella standalone plus debugger** | Very mature emulator; interactive debugger is useful for reverse engineering, not a service API | ROM, screen, joystick; debugger can inspect/modify RAM, disassemble, break on memory access, single-step | Local and free; ROM copyright; debugger is interactive and normally GUI/terminal driven | Development aid for confirming RAM fields, not the runtime adaptor. See [Stella debugger](https://stella-emu.github.io/docs/debugger.html) and [Stella docs](https://stella-emu.github.io/docs/index.html). |
| **RetroArch/libretro Stella core** | Mature frontend/core ecosystem; headless operation is possible but frontend configuration adds latency/fragility | Video frames, configured controller input, saves/states; memory access depends on core/frontend bindings and is not a stable object API | Local/free frontend; ROM rights remain; no account | A viable generic emulator fallback when ALE cannot load a supplied ROM, but worse than ALE for RAM and Python state access. See [Stella libretro](https://github.com/libretro/stella-libretro). |
| **Javatari / browser emulator** | Good visual/manual play and preservation demos; browser/JS event latency | Canvas pixels, keyboard/joystick events, emulator UI; no standard Python object/RAM API | Local if assets are vendored; browser/DOM dependency; ROM copyright; free | Useful for human validation only. The Computer Archaeology page links a playable Javatari instance, but it is not headless. |
| **Screen capture + input injection** (`Xvfb`, SDL/RetroArch/Stella, `ffmpeg`/OpenCV, `xdotool`/`evdev`) | Lowest maturity and highest latency; image parsing and virtual display can drop frames | Screenshots/video and synthetic left/right/fire events; no score/RAM unless OCR/pixel inferred | No online ToS, but many moving pieces; free; requires a virtual display and more CPU | Last-resort fallback. It violates the preference for symbolic state and makes a 0.1–0.25 s reaction budget unreliable. |

## Environment variables and credentials

There is no server address, account, API key, or other credential. The normal chosen setup needs no environment variables and no `DISPLAY`; do not request or write credentials.

- `ALE_ROMS_DIR` is an **optional** override if an evaluator deliberately keeps ROMs outside the installed package. Put a directory path there, never a ROM or credential value. With current `ale-py`, omit it and use `roms.get_rom_path("space_invaders")`.
- A Stella/RetroArch GUI fallback may need `DISPLAY` or `SDL_VIDEODRIVER=dummy`/Xvfb configuration, but the chosen ALE path does not.
- Use exactly `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-space-invaders/.venv/bin/python`; do not install into system Python.

## Reset, seeds, and reproducibility

`env.reset(seed=n)` is the Gym entry point. For direct ALE, set `random_seed` before `loadROM`; ALE uses this for emulator randomness such as sticky actions. For deterministic calibration set `frameskip=1`, `repeat_action_probability=0.0`, and a fixed seed. For evaluation, test both the normal v5 sticky setting (0.25) and a documented deterministic setting if the benchmark requires repeatability. Keep `mode` and `difficulty` fixed within a comparison. After `terminated`, ALE does not continue emulating until reset.

Recommended adaptor setup sequence:

1. Verify the evaluator Python and import `ale_py`, `gymnasium`, and NumPy.
2. Register ALE environments and make `ALE/SpaceInvaders-v5` with `obs_type="ram"`.
3. Set `mode=0`, `difficulty=0`, and explicit frameskip/sticky settings; do not rely on changing package defaults.
4. On reset, record seed, mode, difficulty, frame number, lives, and initial RAM.
5. Decode each RAM observation into the model JSON; during decoder development compare against an RGB environment/fixture.
6. Map only the six supported Gym actions; step until `terminated or truncated`, summing rewards and recording terminal reason/lives.
7. Close the environment and discard snapshots between episodes.
