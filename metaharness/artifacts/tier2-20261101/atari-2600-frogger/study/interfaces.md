# Programmatic interfaces for Atari 2600 Frogger

## Recommendation

Choose **ALE (`ale-py`) through its Gymnasium API, with `obs_type="ram"` and a small Frogger RAM decoder**. It is the only option found that combines a maintained, native, headless Atari emulator, a built-in Parker Brothers Frogger ROM/integration, deterministic seeding, direct 128-byte RAM access, score/life extraction, snapshots, and a five-action minimal action set. It does not provide object names directly, so the adaptor must turn RAM into JSON; that is still preferable to a screenshot pipeline on this server.

Use `ale-py==0.12.1` and `gymnasium==1.4.0` in the evaluator’s Python 3.12.3 venv (or pin the current compatible versions at adaptor build time). The current ALE package supports CPython 3.11–3.15 and bundles supported Atari ROMs; its ROM table includes `frogger.bin` ([ALE README](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [ALE ROM manifest](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/roms/md5.json), [PyPI ale-py](https://pypi.org/project/ale-py/)). No account, server, API key, `DISPLAY`, or GPU is needed.

## Comparison of practical options

| Interface | Maturity | Latency | What it exposes | Legal/ToS risk | Cost and practical verdict |
|---|---|---|---|---|---|
| **ALE Python/Gymnasium (chosen)** | High; official Farama project, native Stella core | Very low; in-process C++/Python, no rendering | `ram` (128 bytes), optional RGB/grayscale, score reward, lives, frame counters, five minimal actions, modes/difficulties, clone/restore state | ALE is GPL-2.0; bundled/game ROM copyright and redistribution terms still need checking for the deployment jurisdiction. No online anti-cheat/ToS issue | Free/open source package. Best fit; only missing layer is a RAM decoder |
| Stable-Retro custom Stella integration | High emulator/framework maturity; higher integration effort | Low once native core is running | Libretro/Stella state, custom `data.json` named memory variables, scenario reward/done, savestate | Same ROM copyright issue; custom integration should use a legally obtained ROM if the bundled ALE copy is not acceptable | Free/open source. Good fallback if a named-memory JSON integration is worth maintaining, but its supported-game table lists `Frogger-Genesis-v0`, not Parker Brothers Atari 2600 Frogger, so a custom integration is required ([supported games](https://stable-retro.farama.org/supported_games/), [integration guide](https://stable-retro.farama.org/integration/)) |
| Stella desktop emulator/debugger | Mature and accurate Atari 2600 emulator with debugger | Low for a human/local process, but poor for an external agent | Debugger can inspect/write RAM, breakpoints/traps, CPU/TIA state, scripts and joystick commands; normal use is GUI-centric | ROM copyright; no game-server rules | Free/open source. Useful for reverse engineering/calibration, not a production adaptor: no documented remote socket/IPC API and no documented no-display agent loop ([Stella guide](https://stella-emu.github.io/docs/index.html), [debugger](https://stella-emu.github.io/docs/debugger.html)) |
| Libretro Stella core / RetroArch | Mature emulator ecosystem and low per-frame latency | Low in-process; headless operation depends on frontend | Libretro callbacks can provide controller input and core memory; Stella core has no Frogger-specific semantic decoder | ROM copyright; frontend/core licenses vary | Free/open source, but requires writing/maintaining a C/C++ or ctypes bridge and locating the correct core/BIOS-free setup. More work than ALE ([libretro API](https://docs.libretro.com/development/libretro/), [Stella2014 core](https://github.com/libretro/stella2014)) |
| Screen capture plus input injection | Generic and widely available, but fragile | Highest: frame capture, image processing, virtual display and input IPC | Pixels only unless the adaptor adds computer vision; keyboard/joystick events | Local offline use has no online anti-cheat, but copying/redistributing the ROM still has copyright risk; input tools can be brittle | Free packages, but needs `stella`, `Xvfb`/SDL setup, capture library and `xdotool`/uinput. Last resort only; violates the symbolic-state preference |

There is no useful remote game API or account-based service for this offline cartridge. Browser/online emulators are excluded because they add latency, cannot guarantee deterministic reset, and conflict with the no-display/no-account constraint.

## Chosen ALE setup

### Software

Install into the exact evaluator venv, not the system Python:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/.venv/bin/python -m pip install ale-py==0.12.1 gymnasium==1.4.0
```

`numpy` is installed as an ALE dependency. Do not install SDL or a desktop window for the RAM configuration. The direct package also works without Gymnasium if the adaptor wants `ALEInterface` instead of `gym.make` ([installation](https://ale.farama.org/getting-started/), [Python interface](https://ale.farama.org/python-interface/)).

### Environment and reset configuration

The adaptor phase should:

1. Import `gymnasium` and `ale_py`, then register ALE environments with `gym.register_envs(ale_py)`.
2. Create `ALE/Frogger-v5` with `obs_type="ram"`, `frameskip=1`, `repeat_action_probability=0.0`, `full_action_space=False`, `mode=0`, `difficulty=0`, and no render mode. `frameskip=1` exposes every emulator frame; sticky actions are disabled so a typed direction does not unexpectedly repeat.
3. On `reset(seed)`, call Gymnasium reset and retain both the returned `info["seeds"]` and the 128-byte initial RAM. Use a fresh reset for each scenario rather than trying to press the physical Reset switch.
4. Assert the loaded ROM is the ALE `frogger` ROM (`ale_py.roms.get_rom_path("frogger")`) and let ALE’s hash check reject an accidental different cartridge. Do not silently substitute Starpath Official Frogger.
5. Cache the action meaning table. The minimal action indices are `0 NOOP`, `1 UP`, `2 RIGHT`, `3 LEFT`, `4 DOWN` ([Frogger environment specification](https://ale.farama.org/environments/frogger/)).
6. After every one-frame `step(action)`, build a JSON observation containing the raw RAM, ALE reward, cumulative decoded score, lives, frame counters, terminal/truncated flags, and named Frogger fields.
7. During decoder development, compare RAM-derived positions and platform tracks with `getScreenRGB()` on a few local runs, but do not make RGB or a display a production dependency.
8. Keep a clone before any Astra search branch and use `clone_state(include_rng=True)`/`restore_state` when exact future replay matters. Do not let planning rollouts mutate Jev’s live state.

### RAM-to-JSON decoder contract

ALE’s `getRAM()` returns the console’s 128 bytes for addresses `$80`–`$FF`; read address `a` as `ram[a - 0x80]`. The following are high-confidence and should be implemented first:

- **score:** ALE’s Frogger integration decodes the BCD/decimal score from addresses `0xCC` and `0xCE`; expose `score`, `score_delta`, and `total_reward` separately;
- **reserve/life/terminal:** address `0xD0`; ALE’s terminal test is `0xD0 == 0xFF`, and its reset value is `4`;
- **frame:** use Gymnasium `info["episode_frame_number"]` and `info["frame_number"]`, not a guessed RAM timer;
- **selected variation:** ALE uses `0xDD` while selecting the single-player variation.

For object-level state, add the public disassembly’s candidate fields only after calibration against this exact ROM: `0xB0`/`0xB4` for frog position candidates, `0xE0` for the 30-second countdown, and `0xE3`/`0xE5`/`0xE7`/`0xAC` for state, collision, and row/object candidates. Mark them `decoder_confidence="calibrated"` only after controlled probes; otherwise retain `raw_ram` and `decoder_confidence="hypothesis"`. The disassembly is community reverse engineering, not an ALE API guarantee ([ALE Frogger.cpp](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp), [community disassembly](https://gist.github.com/erik-nilcoast/cc55644c916c729e6a82ae07b681ac2b)).

A useful adaptor JSON shape is:

```text
{
  "frame": 123,
  "ram": [0, ... 127 bytes ...],
  "frog": {"x": ..., "row": ..., "phase": ...},
  "road": [{"lane": ..., "objects": [{"x": ..., "velocity": ..., "collision": ...}]}],
  "river": [{"row": ..., "platforms": [{"x": ..., "velocity": ..., "kind": ..., "diving": ...}]}],
  "homes": [{"slot": ..., "occupied": ..., "crocodile": ..., "fly": ...}],
  "timer_s": ..., "frogs_reserve": ..., "score": ..., "score_delta": ...,
  "terminal": false,
  "truncated": false
}
```

The exact object lists are a decoder output, not a promise that ALE itself names objects. Preserve `raw_ram` even once the fields are stable so Astra can inspect an unknown state and the adaptor can be repaired without changing the interface.

### Direct ALE alternative

If Gymnasium’s wrapper overhead is undesirable, use `from ale_py import ALEInterface, roms`, set `random_seed`, load `roms.get_rom_path("frogger")`, call `reset_game()`, read `getRAM()`, and send one member of `getLegalActionSet()` to `act()`. The direct API also exposes `getMinimalActionSet()`, `game_over()`, `cloneState()`, and `restoreState()` ([ALE Python interface source](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/main/docs/python-interface.md)). The wrapper is preferable because `reset(seed)` and `info` already provide standard episode bookkeeping.

## Other interface setup details

### Stable-Retro

Stable-Retro supports the Stella Atari 2600 emulator, but its supported-game list does not provide the Parker Brothers Atari 2600 Frogger integration. To use it, install Stable-Retro, obtain a legally usable `.a26` ROM, create a custom `Frogger-Atari2600` integration with a first-playable savestate, `data.json` memory fields, `scenario.json` reward/done rules, metadata and ROM hash, then register the custom integration with `retro.data.Integrations.add_custom_path(...)`. It is attractive because `data.json` can turn RAM addresses into named fields, but the ALE Frogger integration already supplies score, lives and actions and therefore has lower setup risk ([supported emulators](https://stable-retro.farama.org/supported_emulators/), [custom integration guide](https://stable-retro.farama.org/integration/)). No account or secret is needed.

### Stella debugger

Install the Stella binary/package and launch the ROM in its debugger (`stella -debug frogger.bin`) for manual reverse engineering. The debugger documentation describes `ram`, `dump`, `print`, `step`, `frame`, `run`, traps, `joy0Up/Down/Left/Right`, `exec` scripts and autoexec scripts. It does not document a remote socket, GDB protocol, or reliable headless runtime API, so use it to validate RAM labels or make a `.sym`/`.lst`, not as the Astra/Jev transport ([Stella debugger manual](https://stella-emu.github.io/docs/debugger.html)).

### Screen/input fallback

If the RAM decoder cannot be completed, run Stella under a virtual X server (`Xvfb`) or a supported SDL dummy backend, capture frames with a local library such as `mss`/Pillow/OpenCV, and inject joystick keys with `xdotool`, `uinput`, or `evdev`. This is technically possible but has the worst latency and gives Jev pixels rather than typed object state. Use it only for decoder validation or emergency fallback; it is not the chosen adaptor.

## Environment variables and user-supplied material

- **Credentials:** none.
- **Server/account/API key:** none.
- **Required environment variables:** none when using the ROM bundled in `ale-py`.
- **Optional `ALE_ROMS_DIR`:** a directory of ALE-hash-compatible ROM files if deployment policy requires an external ROM location; the user must supply a legally usable Frogger ROM and must not put its contents or a secret in study files ([ALE ROM loading](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/docs/python-interface.md#roms)).
- **Optional display variables:** none; do not require `DISPLAY`, `SDL_VIDEODRIVER`, or a GPU for `obs_type="ram"`.

## Risks and mitigations

1. **ROM provenance:** the pip distribution says supported ROMs are bundled, but cartridge copyright remains distinct from emulator licensing. Keep the bundled copy or an externally supplied copy only under the user’s applicable license.
2. **Decoder drift:** pin `ale-py` and the ROM hash; keep raw RAM in every observation and run calibration tests after upgrades.
3. **Life semantics:** ALE’s `lives=4` at reset is the reserve field, not the manual’s total of five frogs. Preserve both raw and derived names.
4. **Timing:** default v5 sticky actions and frameskip are useful for benchmark comparability but bad for precise control. Use one-frame deterministic stepping for the harness; if a benchmark requires v5 defaults, record that choice in the episode metadata.
5. **No perfect symbolic API:** ALE exposes bytes, not “car 2” or “diving turtle.” Complete and test the decoder; use screenshots only as an offline validation aid.
