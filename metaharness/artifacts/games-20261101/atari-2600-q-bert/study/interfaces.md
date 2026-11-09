# Programmatic interfaces for Atari 2600 Q*bert

## Recommendation

Use the **local Arcade Learning Environment (ALE) through `ale-py`/Gymnasium**, with `obs_type="ram"`, no render mode, and the bundled Q*bert ROM. It is the only option found that combines a maintained Atari 2600 emulator, a standard reset/step API, a 128-byte console-RAM observation, extracted Q*bert reward/lives, seeding, and emulator state cloning. It is local and headless: no display, GPU, server, account, API key, or network connection is needed at run time.

The recommended public environment is `ALE/Qbert-v5`. For the final adaptor, use Gymnasium's wrapper for reset/seed/step and reach through `env.unwrapped.ale` only for clone/restore or the native RAM/screen methods. The native `ALEInterface` is a good lower-overhead alternative if the wrapper is unavailable.

The ALE package is GPL-2.0-only and its current Python wheel packages Atari ROMs, including `qbert.bin`; the supported ROM is identified by MD5 `484b0076816a104875e00467d431c2d2` ([ALE package](https://pypi.org/project/ale-py/), [ALE README](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [Q*bert settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). The emulator/library license does not automatically resolve every jurisdiction's rights to redistribute Atari/Parker Brothers ROM data. Keep the bundled evaluator copy private and do not download or redistribute an unverified ROM.

## Installation and setup

Use the evaluator interpreter, not the system interpreter:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-q-bert/.venv/bin/python -m pip install "ale-py==0.12.1" "gymnasium>=1.1.0,<2"
```

`numpy>1.20` is an ALE dependency. If the evaluator pins a newer compatible ALE, verify the wheel has Python 3.12 support and that `ale_py.roms.get_rom_path("qbert")` resolves to the expected ROM. The official setup is [`pip install ale-py`](https://ale.farama.org/getting-started/); the package's Gymnasium interface requires registering ALE environments (the current package may auto-register them, but explicit registration is harmless):

```text
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make(
    "ALE/Qbert-v5",
    obs_type="ram",
    frameskip=4,
    repeat_action_probability=0.25,
    full_action_space=False,
    mode=0,
    difficulty=0,
    render_mode=None,
)
```

Do not request `render_mode="human"`; it enables SDL/display and is unnecessary on the headless server. `render_mode="rgb_array"` is an optional diagnostic fallback, not the primary observation.

### Environment variables and credentials

There is no server address, account, token, or API key. Leave `DISPLAY` unset and do not require X11. `ALE_ROMS_DIR` is optional: if set, it must be a directory containing a legal, checksum-matching `qbert.bin`; if unset, `ale_py` uses its packaged ROM directory. Never put a credential or a ROM URL/value in the study files.

### Adaptor setup checklist

1. Install the two packages with the evaluator's Python and record their versions.
2. Import/register `ale_py`, create `ALE/Qbert-v5` with `obs_type="ram"`, and assert the observation shape is `(128,)` and dtype `uint8`.
3. Confirm that the ROM resolves as `qbert`, and record/check its MD5 rather than accepting an arbitrary Q*bert ROM.
4. Call `reset(seed=seed)` and retain `info["lives"]`, `info["episode_frame_number"]`, and `info["frame_number"]`.
5. Start with the six-action minimal set. Calibrate which of actions 2–5 is each board-diagonal direction once, because the manual rotates the joystick into a diamond.
6. Decode score/lives and maintain a logical board/enemy state (details below). Add RGB capture only as a validation/fallback channel.
7. Use `terminated` for game over and `truncated` for the evaluator time/frame limit; always reset before stepping after either flag.
8. For action search, use `clone_state(include_rng=True)`/`restore_state`; for ordinary live play, do not mutate RAM or use save-state cheats.

## Chosen interface in detail: Gymnasium/ALE RAM

### Observation

Each observation is a `uint8[128]` copy of Atari 2600 RIOT RAM. It is symbolic at the console-memory level, not a ready-made object list: ALE does **not** expose named cube, Q*bert, Coily, Sam, or ball objects. The adaptor must decode the game-specific bytes or maintain a logical board from actions/rewards and use RGB only during calibration. The wrapper also supplies:

* `info["lives"]`: ALE's remaining-life count;
* `info["episode_frame_number"]` and `info["frame_number"]`;
* step reward: the score delta accumulated across the skipped frames;
* `terminated`/`truncated` flags.

The official general interface documents the three observations—RGB `(210,160,3)`, grayscale `(210,160)`, and RAM `(128,)`—and the standard reset/step return ([ALE environments](https://ale.farama.org/environments/), [Python interface](https://ale.farama.org/python-interface/)).

ALE's Q*bert wrapper gives two particularly useful verified RAM facts. It calls `readRam` with CPU-style addresses `$88` for lives and `$DB,$DA,$D9` for the three BCD score bytes ([source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). Because `getRAM()` is the 128-byte console-RAM view and ALE masks addresses with `& 0x7f`, the corresponding array indices are:

* life byte: `ram[0x08]` (source address `$88`);
* score low/middle/high BCD bytes: `ram[0x5b]`, `ram[0x5a]`, `ram[0x59]` (source addresses `$DB`, `$DA`, `$D9`).

The score formula used by ALE is:

```text
score = 10*(ram[0x5b] >> 4) + (ram[0x5b] & 15)
      + 1000*(ram[0x5a] >> 4) + 100*(ram[0x5a] & 15)
      + 100000*(ram[0x59] >> 4) + 10000*(ram[0x59] & 15)
```

Use the cumulative returned rewards as the authoritative score for the adaptor, and use this decoder as a consistency check. ALE intentionally reports zero on Q*bert's terminal step, so do not expect the last step to contain a negative life penalty. The raw life byte starts at `0x02` in ALE's wrapper bookkeeping and can become `0xfe` at terminal; prefer `info["lives"]` for the public state.

For the object-level JSON given to Astra/Jev, expose at least:

```text
frame, lives, score, level, round,
qbert_node, cube_state[21], needed_landings[21],
red_ball, purple_ball, coily, sam, green_ball,
disc_exits, safe_moves, terminal, truncated
```

The 21-node board and enemy positions need calibration against this specific ROM. Do not infer them from arcade Q*bert's 28-node board. A robust first adaptor can use the RGB frame for calibration and then replace visual fields with RAM/action-trace decoding; keeping raw RAM in every JSON state makes later decoder fixes possible.

### Actions and timing

With `full_action_space=False`, the Q*bert action space is `Discrete(6)`:

```text
0 NOOP, 1 FIRE, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN
```

This is the official mapping ([Q*bert action table](https://ale.farama.org/environments/qbert/)). `FIRE` is not a useful movement action and can reset/restart the cartridge in this ROM; use it only for an intentional reset. The 18-action full Atari set adds combinations, but ALE marks only the six actions above as minimal for Q*bert ([ALE action specification](https://ale.farama.org/environments/#action-space)).

`frameskip=4` repeats the selected action over four emulated frames (about 66.7 ms at NTSC timing), while `repeat_action_probability=0.25` may substitute the previous action. The standard v5 defaults are useful for robustness. For exact transition calibration and clone-state search, temporarily use `frameskip=1` and `repeat_action_probability=0.0`; do not confuse that debug configuration with the default evaluation configuration. Keep a short queued action macro so a 0.5-second Jev call does not leave the game without input.

### Reset, seeds, and state cloning

Use `obs, info = env.reset(seed=seed)`. The wrapper derives ALE and NumPy seeds and reports them in `info["seeds"]` when a seed is supplied. Q*bert's ordinary start is fixed; a seed chiefly changes sticky-action randomness, not the pyramid layout. The Gymnasium wrapper exposes `clone_state(include_rng=False)` and `restore_state`; use `include_rng=True` when a branch must reproduce the exact future under sticky actions ([current wrapper source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)). Serialized states are version-dependent.

The lower-level native interface is available through `ale_py.ALEInterface`:

```text
ALEInterface()
setInt("random_seed", n)
setFloat("repeat_action_probability", p)
setInt("frame_skip", n)
loadROM(roms.get_rom_path("qbert"))
reset_game(); act(action)
getRAM(); getScreenRGB(); lives(); game_over()
cloneState(include_rng=True); restoreState(state)
```

The native API returns the reward from `act` and permits direct RAM access with less Gymnasium overhead. It is the fallback if the wrapper's registration changes, but the wrapper is preferable for the evaluator's conventional `reset(seed)`/`step` contract.

## Alternatives and fallbacks

| Interface | Maturity and latency | What it exposes | Legal/ToS risk and cost | Decision |
|---|---|---|---|---|
| **ALE/Gymnasium `ALE/Qbert-v5`** | Production/stable; native C++ emulation is local and much faster than real time, with only Python/model latency in the loop. | Six actions, RAM/RGB/grayscale, score reward, lives, frame counters, seeds, clone/restore. | ALE is GPL-2.0-only; ROM rights still matter. No account or online ToS; free CPU software. | **Chosen.** Best standard headless adaptor and easiest reset/step contract. |
| **Native `ale-py.ALEInterface`** | Same mature emulator and slightly less wrapper overhead; frame-level control. | Direct `getRAM`, screen, `act`, game-over, lives, settings, RAM writes, clone/restore. No ready Gymnasium spaces. | Same GPL/ROM issue; free. RAM writes can create an unfair/invalid benchmark if misused. | Keep as a fallback and for calibration/search. |
| **Legacy ALE/Gymnasium IDs (`QbertNoFrameskip-v4`, `Qbert-v4`, `Qbert-v0`)** | Mature compatibility wrappers, but legacy semantics: v4 removes sticky actions; v0/v4 use old IDs and stochastic frame-skip conventions. | Same underlying observations/actions; v4 no-sticky and NoFrameskip variants are useful for debugging. | Same local GPL/ROM issue; free. Version drift makes results less comparable. | Use only when an evaluator explicitly requires a legacy ID; prefer v5. |
| **Standalone Stella** | Very mature Atari 2600 emulator. Debugger can step instruction/scanline/frame, inspect/edit RAM, set watches, save states, and run scripts; it is interactive and has no documented low-latency external RPC. | CPU/RAM/TIA debugger data, screenshots, input/debugger commands. `stella -debug ROM` and `autoexec.script` are documented. | GPL emulator; cartridge rights remain. GUI/debugger automation on a headless server needs a display workaround; free. | Excellent for reverse-engineering the RAM decoder, poor as the live Astra/Jev API. [Debugger docs](https://stella-emu.github.io/docs/debugger.html). |
| **Libretro Stella/Stella 2023 via RetroArch** | Mature core, but an extra frontend/FFI layer; latency depends on frontend and Python binding. | Video/audio, joypad input, save states; libretro memory descriptors can expose main RAM, but no named Q*bert objects or ALE score wrapper. | GPLv2 core and ROM rights; free; more packages/configuration. | Viable if ALE cannot install, but less direct. [Stella 2023 docs](https://docs.libretro.com/library/stella2023/), [core info](https://github.com/libretro/libretro-core-info/blob/master/stella2023_libretro.info). |
| **Python `pdretro`/custom libretro FFI** | `pdretro` is an early convenience wrapper; its public documentation does not establish a supported Stella build, so runtime compatibility is uncertain. | Intended API includes frames, buttons, memory regions, and video; still raw RAM/pixels, not game objects. | MIT wrapper, GPL core, ROM rights; free but adds an unverified dependency. | Experimental fallback only; do not make it a prerequisite. [PyPI](https://pypi.org/project/pdretro/). |
| **Screen capture plus injected keyboard/joystick** | Mature individual tools but high/brittle latency: X11, virtual display, scaling, focus, and screenshot conversion sit in the loop. | Pixels only (e.g. Stella RGB frame) and synthetic key events; no score/lives unless OCR or visual inference is added. | No online ToS for a local emulator; free. Requires Xvfb/X11 packages and is fragile on a headless server. | Last-resort compatibility path, not preferred. `Xvfb` explicitly supplies an X server without display hardware ([man page](https://linux.die.net/man/1/xvfb)). |
| **MAME arcade Q*bert** | Mature debugger/Lua ecosystem. | Arcade Q*bert memory/input, not the Atari 2600 cartridge. | Arcade ROM rights and a different game/version. | **Not equivalent**; reject for this task. |
| **ALE WebAssembly in Node** | Official ALE build but an unnecessary JS/Node boundary for this Python evaluator. | Similar emulator actions/RAM/screen if wired through the WASM API. | GPL/ROM considerations; free, but extra runtime and no advantage here. | Not chosen; useful only for a browser/Node harness. |

## Failure handling

If RAM decoding is not ready, retain the raw 128 bytes and use `rgb_array` only for a short calibration phase; do not silently switch to screenshot-only play. If the packaged ROM is missing, fail with an actionable “install/verify `ale-py` or set `ALE_ROMS_DIR`” error rather than accepting a different Q*bert release. If a model call is late, continue the last validated short action macro and mark the observation stale; never block the emulator waiting for Astra.

## Sources

* [ALE getting started](https://ale.farama.org/getting-started/)
* [ALE Q*bert environment](https://ale.farama.org/environments/qbert/)
* [ALE Python interface](https://ale.farama.org/python-interface/)
* [ALE environment specification](https://ale.farama.org/environments/)
* [ALE Q*bert reward/lives implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)
* [ALE Python Gymnasium wrapper](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)
* [ALE PyPI package](https://pypi.org/project/ale-py/)
* [Stella debugger](https://stella-emu.github.io/docs/debugger.html)
* [Libretro Stella 2023](https://docs.libretro.com/library/stella2023/)
* [Xvfb manual](https://linux.die.net/man/1/xvfb)
