# Programmatic interfaces for Atari 2600 Breakout

Study date: 2026-10-30. The target is the Atari 2600 Breakout cartridge represented by `ALE/Breakout-v5`, not a web game and not Super Breakout. The evaluator has no display, GPU, account, or server. The recommendation below is deliberately local, synchronous at the emulator boundary, and symbolic at the adaptor boundary.

## Decision

**Choose: `emulator` — official ALE/Gymnasium with a RAM observation, plus the OCAtari Breakout RAM decoder as a pinned, read-only symbolic decoder.**

Use the official `ale-py`/Gymnasium environment as the owner of stepping, reset, reward, termination, seeding, and timing. Use OCAtari only to decode Breakout RAM into named objects, or port the small Breakout decoder logic into the future adaptor. This is better than using OCAtari as the outer environment because the official Gymnasium step flags are correct and the official API exposes frame counters, lives, exact RAM, emulator state cloning, and configurable frameskip/sticky actions. The decoder supplies the object-level view Astra/Jev need without screen capture.

The tested evaluator-compatible pin set is:

```text
ale-py==0.12.1
gymnasium==1.4.0
ocatari==2.2.1
numpy==1.26.4
opencv-python==4.11.0.86
```

`ale-py==0.12.1` alone is sufficient for the official RAM environment. `ocatari==2.2.1` brings a research wrapper and decoder plus substantial optional dependencies; install it when using its object classes rather than reimplementing the small decoder. The supplied Python 3.12.3 venv successfully imported and smoke-tested all five pins headlessly. No game-playing policy is included here.

## Installation and credentials

Run pip through the evaluator interpreter, not system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-breakout/.venv/bin/python -m pip install \
  ale-py==0.12.1 gymnasium==1.4.0 ocatari==2.2.1 \
  numpy==1.26.4 opencv-python==4.11.0.86
```

The modern `ale-py` wheel packages Atari ROMs, including `breakout.bin`; there is no account, API key, server address, or network service. The package's ROMs and the Atari copyright are a **rights concern**, not an authentication mechanism. Use the packaged ROM only in an environment whose distribution and evaluation terms permit it. Do not copy a ROM into this repository or commit it. If the adaptor must use an evaluator-provided ROM instead, set:

- `ALE_ROMS_DIR` (optional, `required: false`): directory containing the legally supplied ALE ROM files. This is the exact override name read by `ale_py.roms.get_rom_path`; the selected ROM must match ALE's expected hash. Do not put a ROM value or path into source control.

No `DISPLAY` is required. Use `render_mode=None`; do not start Pygame or an X server. OCAtari's wheel installs Pygame as a dependency, but RAM mode does not need a window. `opencv-python` is used by the wrapper's optional image modes; it is not used for the chosen RAM path.

Relevant installation and licensing references: [ALE installation](https://ale.farama.org/getting-started/), [ALE v0.9 ROM packaging release note](https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.9.0), [ALE GitHub license](https://github.com/Farama-Foundation/Arcade-Learning-Environment), and [OCAtari PyPI](https://pypi.org/project/ocatari/).

## Chosen interface: exact setup contract

The adaptor phase should perform these steps:

1. Import `gymnasium` and `ale_py`, then call `gym.register_envs(ale_py)` before `gym.make`; Gymnasium 1.x no longer relies on the old plugin side effect. Check `gym.spec("ALE/Breakout-v5")` and fail loudly if registration or the ROM is absent.
2. Create the official environment with `obs_type="ram"`, `render_mode=None`, `mode=0`, `difficulty=0`, `full_action_space=False`, `frameskip=1`, and `repeat_action_probability=0.25`. `frameskip=1` makes the real-time controller able to refresh every native frame; the v5 default is four. Preserve 0.25 sticky actions for the baseline. A deterministic diagnostic may set it to 0.0, but must label that run nonbaseline.
3. Set `max_num_frames_per_episode=108000` unless the evaluator specifies another cap. This is an ALE frame cap, approximately 30 minutes at 60 Hz, and is a truncation rather than a natural game-over.
4. Call `obs, info = env.reset(seed=seed)`. Record `info["lives"]`, `info["episode_frame_number"]`, `info["frame_number"]`, and, after a seeded reset, `info["seeds"]`. Pass a Python integer seed, never model-generated text.
5. Resolve the actual action order from `env.unwrapped.get_action_meanings()` and assert it is `['NOOP', 'FIRE', 'RIGHT', 'LEFT']`. Use the Gym action **indices** 0, 1, 2, 3. Do not substitute raw `ale_py.Action` enum values: raw RIGHT is 3 and raw LEFT is 4, while Gym's discrete index 3 means LEFT.
6. Treat NOOP as removal of directional input, not a guaranteed instantaneous paddle brake. The actual ROM can carry/decay lateral motion for several frames after a direction; observe paddle x and counter-steer early when needed. Keep this in the controller calibration and do not implement “NOOP means paddle freezes” as an assumed physics rule.
7. At each emulator step, call `obs, reward, terminated, truncated, info = env.step(action)`. Copy `obs` before sending it to another thread/process. Keep the emulator itself single-owner and single-threaded.
8. Convert the 128-byte RAM array to a JSON-safe symbolic record. Include frame number, score, lives, paddle, ball (or `null` when absent), brick occupancy/row values, the last reward, and both flags. Never send a NumPy array or Python object repr to Jev.
9. Initialize OCAtari's Breakout objects once per reset and decode the official environment's RAM into them, using the exact decoder calls below. **Do not create a second OCAtari environment alongside the official one**: it would have independent emulator state. The outer-OCAtari option is an alternative, not a companion. Do not use that wrapper's `step` flags without correction.
10. Close with `env.close()` even on episode failure. Do not use screen rendering as a hidden fallback.

The minimal official setup has the following API shape (this is an interface example, not a playing policy):

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make(
    "ALE/Breakout-v5",
    obs_type="ram",
    render_mode=None,
    mode=0,
    difficulty=0,
    full_action_space=False,
    frameskip=1,
    repeat_action_probability=0.25,
    max_num_frames_per_episode=108_000,
)
obs, info = env.reset(seed=seed)
# action is one of 0:NOOP, 1:FIRE, 2:RIGHT, 3:LEFT
# obs, reward, terminated, truncated, info = env.step(action)
```

### Exact decoder hookup (verified with the pinned versions)

These are observation-only calls, not a controller or playing policy:

```python
from ocatari.ram import breakout
from ocatari.ram.extract_ram_info import detect_objects_ram

# Once after every official env.reset:
objects = breakout._init_objects_ram(hud=False)
# Immediately after reset and after each official env.step, with its RAM obs:
detect_objects_ram(objects, obs, "Breakout", False)
bitmap = breakout._make_block_bitmap(obs)  # int array, shape (6, 18)
# objects[0] = Player, objects[1] = Ball or false-valued NoObject.
# Other populated objects are contiguous brick runs; bitmap is authoritative.
```

The generic `detect_objects_ram` saves previous object positions before updating them. Recreate `objects` after every reset; invalidate velocity on absent-to-present ball transitions, because a newly created Ball has no meaningful motion history. The helper functions beginning with `_` are private APIs, so pin OCAtari and test these calls after upgrades. Use `hud=False`: its Breakout HUD classes are bounding boxes, **not numeric score/life value objects**. Read numbers from official info/RAM. Convert all NumPy coordinates, bits, counts, and booleans to ordinary JSON scalars. Do not use generic `detect_objects_raw` for this game: in 2.2.1 the specialized raw extractor has a differently named entry point.

A full installed dependency snapshot is provided in `requirements-tested.txt`; `python -m pip check` passed. Importing `ocatari.ram` still imports the package's core dependencies, so decoder-only usage is not a magically dependency-free install. A future standalone decoder can avoid those dependencies, but must preserve any copied OCAtari MIT copyright/license notices.

The official environment's observation is `np.uint8` with shape `(128,)`. The official `info` keys are `lives`, `episode_frame_number`, and `frame_number`, with `seeds` on a seeded reset. `ALEInterface` additionally exposes `getRAM`, `getScreenRGB`, `getScreenGrayscale`, `getRAMSize`, `lives`, `getFrameNumber`, `getEpisodeFrameNumber`, `cloneState`, `cloneSystemState`, `restoreState`, `restoreSystemState`, `setRAM`, and `game_over`/`game_truncated`. State cloning is useful for future exact search and diagnostics, but do not serialize a native state handle to a model or assume it is stable across ALE versions.

API references: [ALE Gymnasium interface](https://ale.farama.org/gymnasium-interface/), [ALE Python interface](https://ale.farama.org/python-interface/), [ALE environment specification](https://ale.farama.org/env-spec/), [installed/current `AtariEnv` source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/python/env.py), and [registration defaults](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/python/registration.py).

## Symbolic observation contract

The future adaptor should expose one compact JSON object per control tick, rather than the entire 128-byte RAM or an image. A recommended shape is:

```json
{
  "frame": 1234,
  "score": 76,
  "lives": 4,
  "wall_index": 0,
  "ball": {"x": 91, "y": 153, "w": 2, "h": 4, "dx": -1, "dy": 1},
  "paddle": {"x": 86, "y": 189, "w": 16, "h": 4},
  "bricks": [{"row": 0, "col": 0, "x": 8, "y": 57, "w": 8, "h": 6, "value": 7}],
  "ball_in_play": true,
  "last_reward": 0,
  "terminated": false,
  "truncated": false
}
```

The exact JSON field names are an adaptor choice, but these semantics are not:

- `ram[0:36]` encodes the brick bitmap. Decode it into six rows and 18 columns, preserving occupancy. The OCAtari source's `_make_block_bitmap` contains the bit reversal and column order for this ROM. Do not infer occupancy from the number of variable-width `Block` objects.
- Paddle left x is `ram[72] - 47`; the standard decoder reports y=189, width=16, height=4. Difficulty can change actual geometry, and the OCAtari decoder does not model every difficulty-dependent size, so verify geometry when supporting difficulty 1.
- The ball is present when the standard decoder's test `ram[101] != 0 and ram[101] + 9 <= 196` passes. Its standard coordinates are x=`ram[99]-49`, y=`ram[101]+9`, width=2, height=4. Represent absence as `null`, not `(0,0)`.
- Lives are `ram[57]`. The official info `lives` is preferred for the public record, with RAM as a cross-check.
- Match ALE's exact score formula: `100*(int(ram[76]) & 15) + 10*(int(ram[77]) >> 4) + (int(ram[77]) & 15)`. These are three BCD digits, not ordinary binary hundreds/ones. Compare with cumulative unmodified rewards; never silently clamp a mismatch. The optional OCAtari raw helper uses a more general BCD converter, but numeric HUD objects do not contain the score.
- Row colors in the decoder are row 0 red, row 1 orange, row 2 yellow, row 3 green, row 4 aqua, row 5 blue. The points are 7, 7, 4, 4, 1, 1. Store the numeric row value directly so colors are not a model dependency.
- OCAtari's `env.objects` has one paddle, one ball slot, 108 block slots, and—when `hud=True`—score, lives, and player-number HUD slots. Objects have `category`, `xy`, `wh`, `rgb`, `dx`, and `dy`; `NoObject` is false. Its current `ns_state` contains only x/y-like values; `obs_mode="obj"` returns a four-**observation** stack by default, not necessarily four native frames. Neither contains the complete geometry/numeric HUD needed for JSON.
- OCAtari groups contiguous bricks into a single variable-width `Block`. That is useful for a visual summary but not for counting or selecting an individual brick. Use the 6x18 bitmap for bookkeeping.

Sources: [OCAtari README](https://github.com/k4ntz/OC_Atari), [OCAtari documentation](https://oc-atari.readthedocs.io/en/latest/), [OCAtari Breakout decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py), [OCAtari game-object source](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/game_objects.py), and the [installed-source flag issue](https://github.com/k4ntz/OC_Atari/issues/53).

## Timing and concurrency contract

Atari emulates at about 60 native frames/second. The official v5 registration repeats an action four times and sets `repeat_action_probability=0.25`; four frames are about 66.7 ms of game time. The chosen `frameskip=1` configuration exposes a 16.7-ms emulator tick. ALE does not advance while a synchronous caller waits for an LLM, so a real-time tier-2 adaptor needs:

- a stepping thread/process that owns the emulator and ticks against a monotonic clock;
- a latest-state queue and an atomic latest-action/short action-hold buffer;
- a fallback action that remains safe when Jev/Astra is late;
- model calls operating on immutable copied JSON, never on the live emulator object;
- a small action horizon (normally 1–3 frames near a descending ball, longer only when the ball is safely high/ascending);
- explicit telemetry for wall-clock delay, emulated-frame delay, action age, sticky-action setting, and dropped states.

On this evaluator host, a diagnostic run of 4,000 `frameskip=1` ticks measured a median 0.123 ms ALE step, 0.271 ms for OCAtari RAM/object plus bitmap decoding, and 0.402 ms combined (95th percentile 0.189, 0.387, and 0.557 ms respectively). This is a local CPU measurement, not a guaranteed deadline; model/network latency and thread scheduling dominate. It does show that symbolic decoding—not emulation—is cheap enough to run every native frame.

A single FIRE tick can also be swallowed by the 0.25 sticky-action draw when the previous action was NOOP. The serve state machine should issue a short bounded FIRE retry/hold and wait for the ball-present predicate, rather than assuming that one discrete FIRE call served the ball. Likewise, the probe showed paddle x can continue changing for several NOOP frames after a directional command, so the controller must observe and counter-steer rather than treating NOOP as an instantaneous brake.

A synchronous adaptor is still useful for search and tests, but call it “paused emulation.” A model call must not accidentally change the claimed game pace. The published DQN paper also used 60-Hz Atari emulation and selected actions every four frames, which is a useful comparison point, not a license to hide long Astra calls behind a paused emulator. [DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

## Alternatives

### 1. Raw official ALE/Gymnasium RAM — maturity: high; latency: best

- **What it exposes:** 128-byte RAM, optional 210x160 RGB/grayscale frames, four minimal Breakout actions, 18-action full ALE set, rewards, lives, frame counters, natural termination/truncation, seeded reset, and emulator state cloning. C++ emulation is fast and display-free.
- **Install/cost:** `ale-py` and `gymnasium`; free software/CPU. The current `ale-py` 0.12.1 wheel includes a verified `breakout.bin`; no GPU or account.
- **Strengths:** smallest dependency surface, correct Gymnasium flag order, direct frameskip and sticky-action controls, exact reproducibility controls, and a maintained official API.
- **Weaknesses:** RAM is bytes, not names or JSON; the adaptor must decode Breakout offsets/bitmap. RAM mappings are game-ROM-specific. Included Atari ROM rights still need checking.
- **Legal risk:** ALE is GPL-2.0; ROM use/distribution is separate and may be restricted. [ALE repository](https://github.com/Farama-Foundation/Arcade-Learning-Environment).

This is the chosen base even though the final observation is decoded symbolically.

### 2. OCAtari RAM mode — maturity: medium/high research package; latency: low, but more Python overhead

Install `ocatari`, then use `OCAtari("ALE/Breakout-v5", mode="ram", hud=True, obs_mode="obj", render_mode=None)`. It uses RAM-based handwritten detectors (REM), returns object slots, and can also expose image/vision modes. It is substantially easier than writing object classes, and its Breakout source is transparent enough to audit.

**Critical caveat:** the tested 2.2.1 `OCAtari.step` unpacks the official `(obs, reward, terminated, truncated, info)` and returns `(obs, reward, truncated, terminated, info)`. GitHub issue [#53](https://github.com/k4ntz/OC_Atari/issues/53) documents the same runtime swap. Use official ALE for flags or correct the wrapper at one narrow boundary and add a regression test. The issue/PR status may change; inspect the installed version rather than trusting this study forever. OCAtari also has a default four-frame object buffer and the decoder groups contiguous bricks. These make it a useful decoder, not the sole source of truth.

- **Cost:** free, but it pulls NumPy, OpenCV, scikit-image, SciPy, Matplotlib, Pygame, and other packages; no GPU is required for RAM mode.
- **Risk:** research API/version drift, heavy install, internal decoder functions, wrapper bug, and uncertain difficulty-width handling.
- **Use:** chosen as an optional symbolic decoder, not as the episode/termination owner.

### 3. HackAtari — maturity: experimental research framework; latency: low-to-medium

[HackAtari](https://github.com/k4ntz/HackAtari) builds on OCAtari and ALE and supports object-centric observations, RAM/vision modes, and custom interventions/rewards. It can be useful for controlled experiments, RAM hacks, or alternative reward shaping, but modifying Breakout changes the game being evaluated. It inherits the OCAtari/ALE dependency and version risks; do not add it to the baseline just to obtain a decoder. No account or display is needed for RAM use. [HackAtari docs](https://hackatari.readthedocs.io/en/latest/hackatari/core.html)

### 4. Stable-Retro / Stella Gymnasium environment — maturity: medium/high; latency: low-to-medium

[Stable-Retro lists `Breakout-Atari2600-v0` as a stable Stella-backed integration](https://stable-retro.farama.org/supported_games/). Its Gymnasium API supports `stable_retro.make`, discrete or multi-discrete controls, `render_mode=None`, RGB or RAM observations, savestates, and variables in `info`; it does **not** provide OCAtari-style named objects, so the same Breakout RAM decoder would still be needed. The current PyPI package advertises Python 3.10–3.14 and is compatible with the evaluator's 3.12. It is a real alternative if ALE's wrapper is unavailable, but game integrations normally require importing a legally obtained ROM with `python -m stable_retro.import`; the package does not solve Atari ROM rights. Headless/display dependencies and exact state/frameskip semantics need a separate smoke test. It is less attractive than direct ALE because it adds a second emulator integration and does not expose the official ALE reward/lives/termination contract as directly. [Stable-Retro Python API](https://stable-retro.farama.org/python/), [installation](https://stable-retro.farama.org/getting_started/), [PyPI](https://pypi.org/project/stable-retro/)

### 5. OpenEnv Atari server — maturity: emerging service wrapper; latency: medium/high

[OpenEnv's Atari environment](https://huggingface.co/docs/openenv/environments/atari) wraps ALE behind a local FastAPI/Docker server. It supports Breakout, RGB/grayscale/**raw 128-byte RAM**, minimal or full actions, mode/difficulty, frameskip, and sticky-action configuration; the client talks over HTTP/WebSocket and can be async. This is practical when the harness requires a service boundary and works offline with a local server, but it needs Docker or a separately launched server, adds serialization/network jitter to a game with approximately 16.7-ms ticks, and returns raw RAM rather than symbolic objects. It is therefore a fallback, not the primary interface. No account or API key is documented; the possible `base_url` is local (for example `http://localhost:8000`). [OpenEnv Atari source](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/atari_env)

### 6. Stella desktop emulator / Stella-Lua fork — maturity: high emulator, low adaptor maturity; latency: variable

[Stella](https://stella-emu.github.io/docs/index.html) is a mature Atari 2600 emulator with command-line launch and a debugger. It can run without a normal display in suitable command-line configurations, but its documented interface is a desktop/debugger workflow rather than a stable Python RL API. [Stella-Lua](https://github.com/tcr/stella-lua) adds debugger Lua access such as CPU registers and memory peeks, which is attractive for reverse engineering. Neither is as turnkey as ALE for per-frame Gym steps, structured rewards, seeded reset, and action batches. It needs a legally supplied cartridge image. Use it to validate ROM behavior or inspect memory, not for the production adaptor unless ALE fails.

### 7. Libretro Stella core with `libretro.py` or `pdretro` — maturity: core high, Python bridge medium/low; latency: potentially excellent

The Libretro [Stella/TIA core](https://docs.libretro.com/library/tia/) exposes frame callbacks, controller input, memory descriptors, and serialization through the generic C API. [libretro.py](https://libretropy.readthedocs.io/) and [pdretro](https://pypi.org/project/pdretro/) offer Python-facing routes, with pdretro advertising headless frame/input/memory access. This can provide precise frame stepping and RAM, but it requires separately locating/building the core, mapping Stella memory to Breakout variables, handling callback buffers, and supplying a legal ROM. The bridge is less standard for this task than ALE, so setup and held-out reproducibility are riskier. No account is needed; software is generally free, but core/ROM licensing remains relevant.

### 8. Screen capture plus injected input — maturity: generic tools high, game adaptor low; latency: poor

Run Stella or another emulator under Xvfb, capture RGB frames using `mss`, `PIL`, or OpenCV, and inject keys/buttons with `xdotool`, `uinput`, `evdev`, or a browser automation layer. This works even when no RAM API is available, but it adds a virtual display, calibration, image segmentation, input-queue latency, focus failures, and no direct score/lives/brick state. It is the fallback only. It also requires a legal ROM and may violate the target's terms if used against an online service; this local task has no such service.

## Operational risks and tests for the adaptor phase

Before training or playing, assert:

- Python executable is the evaluator venv and package versions match the tested pins.
- `gym.register_envs(ale_py)` succeeds; `ALE/Breakout-v5` resolves; `obs.shape == (128,)` and `dtype == uint8` for RAM.
- `get_action_meanings()` is exactly the four-action order above.
- `reset(seed)` gives five lives, score 0, absent ball; FIRE causes a ball to appear without requiring a display.
- `episode_frame_number` advances by 1 with frameskip 1 and by 4 with frameskip 4; this catches an accidental timing change.
- A lost ball decrements lives but does not cause the adaptor to reset the whole game; FIRE is required to serve the next one.
- Natural game-over is reported as `terminated=True`, frame cap as `truncated=True` by official ALE. Add a test specifically guarding the OCAtari swap.
- RAM-derived brick count, score delta, lives, and object coordinates agree with a diagnostic RGB frame when RGB is deliberately enabled in a separate test. Production must remain `render_mode=None`.
- Report package, ROM hash/path policy, mode, difficulty, frameskip, sticky-action probability, seed, and score metrics with every evaluation.
