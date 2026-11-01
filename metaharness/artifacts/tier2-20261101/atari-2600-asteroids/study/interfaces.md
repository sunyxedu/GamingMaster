# Programmatic interfaces — Atari 2600 Asteroids

Research date: 2026-10-30. This is an implementation specification, **not game-playing code**. The game and engine are real; no substitute Asteroids clone should be used.

## Recommendation

Use **native Gymnasium/ALE (`ale-py`) with a small, read-only Asteroids RAM-to-objects decoder**, using OCAtari's Asteroids RAM extraction as the documented reference. The environment owns score, lives, reset, termination and action timing; the decoder only produces symbolic observations. Preserve optional RGB snapshots for debugging. This fits offline CPU-only Linux and avoids screen capture/input injection and cloud services. Full OCAtari is an optional comparison tool, not the recommended runtime dependency. [A1–A4, O1–O4]

This choice is deliberate: OCAtari 2.2.1's source swaps terminated/truncated in `step`, has wrapper-chain-dependent helper methods, leaves Asteroids HUD lives unupdated, lacks saucer/enemy-shot classes, and recreates the player and missile objects on every detection. A full wrapper is not automatically a reliable object oracle. See extraction caveats below. [O2, O3; downloaded 2.2.1 source]

## Practical alternatives

| Interface | Maturity / exposed state / control | Latency | Legal/ToS risk; cost and constraints | Recommendation |
|---|---|---|---|---|
| Native ALE + Gymnasium + read-only decoder | Mature research emulator; 128-byte RAM, RGB, score rewards, lives, frame numbers, game over; joystick actions, seeds, modes | In-process CPU calls; no model/network latency; benchmark locally | ALE is GPL-2.0-only; no service ToS or account; ROM provenance/license remains relevant | **Chosen** |
| Direct `ale_py.ALEInterface` + same decoder | Same emulator; exact `act`, `getRAM`, `getScreenRGB`, `reset_game`, `getMinimalActionSet`, `lives`, `game_over` | Similar or less wrapper overhead | Same GPL-2.0-only emulator and ROM considerations; more manual seeding, truncation, and action mapping work | Good fallback |
| Legacy `atari-py` / classic Gym Atari | Historical ALE Python wrapper; RGB/RAM observations and discrete controls | Local native calls, but old bindings add setup overhead | Archived-style dependency/ROM-import path; likely unsuitable for Python 3.12 and current Gymnasium; ROM terms still apply | Avoid; native ale-py supersedes it |
| Full OCAtari RAM mode | Existing named objects with positions/sizes/orientation; wraps ALE | Low CPU overhead but Python matching/dependencies; no verified latency guarantee | MIT wrapper; ALE/ROM licenses separate; broad install dependencies and extractor gaps | Useful for cross-check, fix documented bugs if used |
| OCAtari vision / custom RGB parsing | CPU color/shape object extraction, no GPU needed | More work than RAM; frame flicker and tracking ambiguity | Open-source local wrapper; same ALE/ROM constraints; no account | Debugging or missing-object fallback, not primary |
| AtariARI / annotated RAM | Older research RAM labels, usually not rich normalized object tracks | Low overhead | MIT research code; local ROM rights still apply; legacy Gym dependencies and partial labels need adaptation | Reference, not turnkey current runtime |
| Stella debugger / custom emulator extension | Mature emulator/debugger exposes RAM, registers, frames and input; no standard current Python object API | Fast emulation; integration/scripting overhead | Free/open-source local tool; ROM rights still apply; display/debugger may require SDL/virtual display | ALE already solves this problem more directly |
| libretro Stella core via a native frontend | `retro_run`, video callbacks, input callbacks, memory access and serialization | Frame-level native calls | Free/open-source interfaces; ROM provenance remains the user's responsibility; need C ABI/frontend integration and ROM; object decoder still needed | Viable engineering fallback |
| Gym Retro / Stable-Retro integration | Emulator action arrays and RAM with game data/integration definitions | Local per-step; native extension compatibility must be checked | Open-source local frameworks; no service ToS, but Asteroids integration/ROM and Python 3.12 wheels not verified | Do not make this the critical path |
| Screen+input adapter (`stella` + `Xvfb` + `mss`/PyAutoGUI or `xdotool`) | Screenshot pixels, OCR/vision score and keyboard/joystick events; no RAM unless debugger is separately bridged | Highest jitter: virtual display, capture, OCR and focus/input queues | Local open-source/tools but ROM rights still apply; X11 virtual display and extra packages required; no server ToS | Last resort |
| Browser emulator / ALE WebAssembly | Pixels and potentially JS hooks/step interface | Browser/headless overhead | Free/open-source option may still inherit ROM/site terms; browser dependency and possible network | No advantage over local Python |
| Official commercial port/cloud account | No identified official game-server automation API for this cartridge | N/A | Accounts/platform purchase potentially required | Not needed |

No network opponent, anti-cheat service, server address, paid API, or credentials are required for the chosen local route. Treat software licensing and ROM rights separately; a freely available emulator is not a universal grant to redistribute ROMs. Use the installed ALE-distributed asset according to its terms, or an authorized ROM if required. Do not fetch arbitrary ROM packs. [A4, A5]

## Software and setup

The **actual evaluator interpreter** is:
`/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python` (3.12.3).

Successfully installed in that interpreter during the study:
- `ale-py==0.12.1`
- `gymnasium==1.3.0`
- `numpy==1.26.4`
- resolved dependencies `cloudpickle==3.1.2`, `typing_extensions==4.16.0`, `farama-notifications==0.0.6`

Installation command (setup only):
```
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python -I -m pip install ale-py==0.12.1 gymnasium==1.3.0 numpy==1.26.4
```

ALE 0.12.1 ships ROM assets; its release notes flag 0.12.0 as accidentally lacking ROMs. Do not follow older AutoROM/`accept-rom-license` recipes blindly. A current native environment created and reset successfully without supplying a separate ROM. NumPy 1.26.4 is a conservative compatible pin, not an ALE requirement. [A4, A5; local smoke test]

**User-provided environment variables: none.** Optional `ASTEROIDS_ROM_PATH` can mean an authorized replacement cartridge path for a direct-ALE fallback; it is an adaptor convention, not a built-in ALE environment variable. SDL dummy drivers and `MPLBACKEND=Agg` are only optional workarounds for full OCAtari/rendering imports; plain ALE worked without them. Never enable human rendering on this host.

### Native environment contract

1. Import `gymnasium` and `ale_py`; call `gym.register_envs(ale_py)` explicitly.
2. Create `ALE/Asteroids-v5` with `obs_type='ram'`, `render_mode=None`, `frameskip=1`, `repeat_action_probability=0.0`, `mode=0`, `difficulty=0`, and reduced actions. Use a disclosed sticky-action variant (0.25) only as a separate benchmark setting.
3. Read and save `env.unwrapped.get_action_meanings()`, installed versions, ROM identity, settings, and supported mode/difficulty lists.
4. Call `env.reset(seed=seed)` and take `(obs, info)`. Reset score accumulator, object tracking, action latch and simulated-time counters. Initial smoke test: `(128,) uint8`, lives 4, frame number 0.
5. Call `env.step(action_index)` for one frame and unpack **obs, reward, terminated, truncated, info**. Accumulate raw rewards before any wrapper. Life loss is not a reset.
6. Read `env.unwrapped.ale.getRAM()` (same 128-byte RAM); get `getScreenRGB()` only for debugging. RGB is `(210,160,3)` uint8 on the tested ROM. `info` carries `lives`, `episode_frame_number`, `frame_number`; reset also returned `seeds`.
7. Stop at true termination or adaptor frame budget; close the environment. Native ALE's tested default `max_num_frames_per_episode` was 108,000 frames, but the proposed benchmark deliberately uses 18,000 and should either enforce that in the adaptor or pass `max_num_frames_per_episode=18000`. Never let “episodic life” or reward clipping wrappers silently change score/episodes.
8. For variable action duration, keep underlying frameskip=1 and repeat no more than the promised number of frames; stop immediately on termination. Avoid hidden double frame-skipping.

**Gym action mapping:** 0 NOOP; 1 FIRE; 2 UP; 3 RIGHT; 4 LEFT; 5 DOWN; 6 UPRIGHT; 7 UPLEFT; 8 UPFIRE; 9 RIGHTFIRE; 10 LEFTFIRE; 11 DOWNFIRE; 12 UPRIGHTFIRE; 13 UPLEFTFIRE. Raw ALE enumeration numbers differ: obtain them from `getMinimalActionSet`, never pass Gym indices directly to `act`. In the tested ale-py 0.12.1 build, the direct minimal mapping is `NOOP=0, FIRE=1, UP=2, RIGHT=3, LEFT=4, DOWN=5, UPRIGHT=6, UPLEFT=7, UPFIRE=10, RIGHTFIRE=11, LEFTFIRE=12, DOWNFIRE=13, UPRIGHTFIRE=14, UPLEFTFIRE=15`; full action space additionally includes `DOWNRIGHT=8`, `DOWNLEFT=9`, `DOWNRIGHTFIRE=16`, and `DOWNLEFTFIRE=17`. Always query the runtime enum rather than hard-code it. [A1, A2; local validation]

For the direct `ale_py.ALEInterface` fallback, the concrete calls are `setInt('random_seed', value)`, `loadROM(path)`, optional `setMode(mode)`/`setDifficulty(difficulty)`, `getAvailableModes()`, `getAvailableDifficulties()`, `reset_game()`, `act(ale_action)`, `getRAM()`, `getScreenRGB()`, `getMinimalActionSet()`, `getLegalActionSet()`, `lives()`, `game_over()`, `game_truncated()`, `getEpisodeFrameNumber()`, and `getFrameNumber()`. Use `cloneSystemState`/`restoreSystemState` only for diagnostics/search, not as an action shortcut; direct ALE has no Gymnasium `terminated`/`truncated` tuple, so the adaptor must maintain its own frame budget and reward accumulator. [A3, A4; local API inspection]

## Symbolic state specification

Expose JSON with frame/elapsed time, score total, displayed score, lives, terminal/truncated status, mode/difficulty, legal action names, player, asteroid list, player missile list, extraction coverage and confidence, and optional raw RAM. Use plain Python numbers, not NumPy scalar values. Include previous action and the age of the observation delivered to a model. Player presence should be explicit rather than fabricating a ship at `(0,0)`.

### Known RAM map (128-byte array indices, not CPU addresses)

Cast RAM bytes to Python `int` before arithmetic; uint8 subtraction can wrap.

- Lives: `RAM[60] >> 4`; player orientation: `RAM[60] & 15` (same packed byte). ALE CPU address `$BC` means array offset 60.
- Displayed score: `1000 * BCD(RAM[61]) + 10 * BCD(RAM[62])`, with BCD(byte) = 10*(byte>>4)+(byte&15). Prefer accumulated ALE rewards as total score across rollover. [A2, O3]
- Player: encoded X `RAM[73]`; Y `100 + 2*(RAM[74]-41)`. `RAM[74]==224` means absent in the extractor. Orientation 0 up, 4 left, 8 down, 12 right per source comments; there are 16 possible low-nibble values, not only eight headings. Calibrate intermediate values before assigning angles. X is adjusted -1 at orientation 4, +1 at 12. [O3]
- Asteroid source slots: Y indices `3,4,5,6,7,8,9,12,13,14,15,16,17,18,19`; X at index+18; size/state at index+36. Active if X byte is nonzero and Y byte's high bit is clear. Y pixels = `184-2*(80-Ybyte)`. Size code `RAM[index+36]&127`: <32 gives 16x28, <48 gives 8x15, otherwise 4x8. Preserve source-slot identity and raw size code. [O3]
- Two missile source slots: X at 83/84, Y at 86/87, direction at 89/90. Presence test is nonzero X and clear Y high bit; X pixel = decoded X +1; Y pixel = `175-2*(80-Ybyte)+2`. [O3]
- X decoding from OCAtari: let low nibble L=value&15, A=8*((value>>7)&1), S=(value>>4)&7. Values 0 and 1 map specially to 64 and 4. Other even L: `97+15*(L/2-1)+A-S`; odd L: `10+15*((L-1)/2-1)+A-S`. Returned integer-valued coordinates need validation near wrap boundaries. [O3]

This is **approximate extracted geometry**, not exact collision masks. OCAtari clips rock boxes at X=160 and Y=194; do not infer the physical wrap period from clipping alone. The source has stale caveats about X interpretation and a suspicious chained condition for player width. Preserve raw observations and mark uncertain geometry; do not silently declare unknown hazards absent. [O3]

### Coverage and wrapper traps

- No saucer/satellite/enemy-shot classes in either inspected Asteroids RAM or vision extractor. Hard-difficulty scenarios are not fully observable through this extractor; add and validate coverage before advertising them.
- The extractor scans 15 source slots but allocates 30 asteroid placeholders. Do not claim 30 physical asteroids from the allocation alone.
- `_init_objects_ram` creates 34 base slots versus 33 declared by max-object counts; `obs_mode='obj'` dimensional assumptions deserve checking.
- New Player/PlayerMissile instances each frame invalidate reliance on their inherited `dx/dy`. Track previous coordinates independently. Assignment of asteroid bounding boxes can swap identities at crossing/splitting/wrap.
- HUD PlayerScore and Lives are geometric objects, not authoritative numeric values; read native ALE lives/reward or packed RAM.
- `_augment_info_asteroids_ram` is old/inconsistent helper code. Do not use it as the primary decoder.
- `mode='ram'` in OCAtari is the extraction mode, **not** ALE's cartridge mode number. A native wrapper avoids this keyword collision.

## Seeds, fairness and latency

A deterministic emulator can have identical starts under different seeds. Seeding the action space is not enough. In local validation, seeds `0, 1, 2, 3, 42, 123` produced identical initial RAM and identical RAM after 120 NOOP frames under mode 0/difficulty 0; a longer fixed mixed-input diagnostic was also identical across those seeds. Treat this as evidence that `reset(seed)` alone is not a meaningful start randomizer for this ROM, not as a promise for every mode or action sequence. Test reproducibility and diversity with fixed diagnostic inputs and state hashes.

For a comparable main score, keep mode 0/difficulty 0 fixed and report repeated deterministic runs; do not pretend seed IDs are independent worlds. For a genuine held-out robustness suite, define a versioned scenario catalogue before evaluation: map a seed to explicit mode/difficulty/defense/speed variants, or to a measured, bounded control prelude whose frames and rewards remain inside the episode budget. Report each variant separately because speed, extra-life threshold and defenses change the task. Do not randomize RAM or discard a dangerous prelude. Never hash a seed into a memorized action schedule. Retain sticky actions as an explicit robustness suite. The native step median was about 0.09 ms and p95 about 0.11 ms on this host for 600 one-frame NOOP steps, excluding model time; do not treat that as a real-time guarantee on another host. [local `validation.json`]

ALE does not advance during API/model inactivity. A tier-2 harness must make the clock contract explicit and tick with a short-lived last action or deterministic safe fallback while Astra/Jev are busy. One emulator thread/process should own all step calls; publish immutable timestamped observations. Reject stale decisions. Do not hold thrust or defense indefinitely on timeout.

## Sources

[A1] https://ale.farama.org/environments/asteroids/
[A2] https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/Asteroids.cpp
[A3] https://ale.farama.org/python-interface/ and https://ale.farama.org/gymnasium-interface/
[A4] https://ale.farama.org/getting-started/ and https://pypi.org/project/ale-py/0.12.1/
[A5] https://ale.farama.org/release_notes/index.html
[A6] Atari 2600 NTSC timing reference: https://problemkaputt.de/2k6specs.htm
[O1] https://github.com/k4ntz/OC_Atari and https://oc-atari.readthedocs.io/en/latest/
[O2] https://github.com/k4ntz/OC_Atari/blob/master/ocatari/core.py
[O3] https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/asteroids.py
[O4] https://github.com/k4ntz/OC_Atari/blob/master/ocatari/vision/asteroids.py
Other interface references: https://stella-emu.github.io/docs/debugger.html ; https://github.com/tcr/stella-lua ; https://docs.libretro.com/development/cores/developing-cores/ ; https://github.com/mila-iqia/atari-representation-learning ; https://github.com/openai/atari-py ; https://github.com/Farama-Foundation/stable-retro ; https://python-mss.readthedocs.io/stable/usage.html ; https://pyautogui.readthedocs.io/

Downloaded reference sources are under `sources/`; they are reference material, not executed agent code. Full OCAtari 2.2.1 source was inspected; only the minimal native stack was installed and smoke-tested.
