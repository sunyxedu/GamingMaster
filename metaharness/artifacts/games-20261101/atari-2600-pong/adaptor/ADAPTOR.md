# Atari 2600 Pong adaptor

This tier-2 adaptor runs the real bundled 2 KiB USA Video Olympics `pong.bin` ROM through ALE/ale-py and Gymnasium, headlessly and in-process. The verified ROM MD5 is `60e0ea3cbe0913d39803477945e9e5ec`; the ROM is supplied by the installed ALE package and is not copied into this adaptor. No display, GPU, account, server or credentials are needed. One `tick` advances one emulator frame; the evaluator owns the 60 Hz clock. Frameskip is 1 and sticky actions are disabled.

## Setup and lifecycle

Install `requirements.txt` into the evaluator Python environment. `metaharness` must also be importable there. Construct with `make_game()`, call `reset(seed)`, then drive `tick(action)`, `observe()`, `score`, and `done`. `close()` releases the in-process ALE/Stella object and is idempotent. The adaptor starts no child process, server, connection, display, or background thread.

## Scenarios and deterministic resets

Seeds 0–3 are training/calibration scenarios: ALE mode 0, cartridge Robot Pong game/variation 1, difficulties 0–3. Seeds 4–7 are held-out scenarios: ALE mode 1, Robot Pong game/variation 2, difficulties 0–3. Bit 0 of the ALE difficulty selects the left Difficulty A/B switch; bit 1 selects the controlled right switch. A is half-size and B full-size, so right difficulties 2/3 are half-paddle and 0/1 full-paddle cases. `reset(seed)` maps by `seed % 8`, reloads the ROM, and is deterministic. A seed alone is not assumed to randomize this cartridge; mode/variation and switches provide scenario diversity. Optional `mode` and `difficulty` constructor overrides are available for local calibration, while declared evaluator seeds use the mapping above.

## Observation and actions

Observations are compact strict JSON. The adaptor detects paddle and ball sprite geometry from the ALE RGB buffer internally and exposes symbolic fields: coordinates/heights, velocity differences/signs, approach, automatic serve countdown, phase, mode/difficulty/variation, and score. No screenshot path is exposed or written. Coordinates are screen pixels, x rightward and y downward; visible paddle extents can clip at the walls. `null` means absent/occluded/unknown. The serve timer and score bytes are cross-checked against actual RAM traces; signed ALE rewards remain authoritative for score counters. The six legal actions are `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE`, and `LEFTFIRE`. Calibration confirms `RIGHT` moves the controlled right paddle up and `LEFT` down. Robot Pong serves automatically, so `FIRE` is a variation-specific button feature rather than a required serve action. Malformed actions are ignored as `NOOP` and reported in `action_error`.

## Score and limits

`score` is positive reward count minus negative reward count. `own_points`, `opponent_points`, the latest reward, final `own:opponent` score, termination, truncation, and result are reported separately. A normal match ends when one side reaches 21; the default frame budget is 60,000 and a frame-budget end is reported as truncation rather than loss/win. `info()["valid_trial"]` becomes false only for an emulator/integrity failure, with `invalid_reason` for evaluator retry. Pixel parsing is specific to this ROM and deliberately reports uncertainty rather than raising for a transient sheared ball sprite. No training, network calls, or runtime files are required while playing.
