# Atari 2600 Asteroids adaptor

This is the real cartridge running in ALE 0.12.1 / Stella through Gymnasium 1.3.0,
not a clone or simulated substitute. CPU-only, no display, GPU, service, account,
credential, or external process. Installed ALE supplies its cartridge asset.

## Setup

Install requirements.txt into the evaluator's Python environment, or run setup.sh with its Python path. For this run:

```sh
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python -I -m pip install -r requirements.txt
```

`make_game()` returns a tier-2 object implementing the contract. `reset(seed)`
starts a fresh native episode. `close()` releases the environment and is idempotent.
No required environment variables; see env.example.

## Clock, controls and score

One tick is one native frame at 60 Hz, `tick_s=1/60`, with underlying frameskip=1.
The evaluator is the sole clock owner; no duplicate background thread, sleeps, or
autonomous stepping. The evaluator keeps ticking while model calls run. `ticks`
counts external applied ticks only, and resets to zero. `frame` includes reset's
explicit scenario prelude. Plain names are the current joystick/button levels,
not pulse macros. Query the runtime's fourteen reduced Gym action meanings.
Optional `ACTION@FRAME` is valid only while FRAME <= current frame < FRAME+6 (six frames). Malformed,
stale or future input releases controls for one frame instead of raising.
The runner cannot signal whether an identical plain action is fresh or repeated;
use timestamped inputs/local controller for safe expiry rather than claiming that
an undocumented timeout overrides current controls.

Score is precisely accumulated original, unclipped ALE rewards; includes prelude
rewards. Stop on true game over or 18000 emulated frames, counting the prelude.
No episodic-life resets, shaping, life/survival bonuses, reward clipping, hidden
frame skipping, RAM writes or save-state manipulation. ALE handles display rollover.
`info()` separately reports elapsed frames/seconds, displayed score, lives gained
and lost, settings, termination/truncation reason, package versions and ROM SHA256.
Emulator failures set `valid_trial=False` and an `invalid_reason`.

## Versioned scenarios and fairness

Canonical mode-0/difficulty-0 starts and fixed-input trajectories did not vary
with native reset seeds in study tests. We do not pretend seed labels create
random worlds. Default evaluation is the disclosed **bounded-launch-v1** robustness
suite: seed s in 0..15 means UPRIGHT held for 60+4*s frames immediately after native
reset. This is one parameterized control prelude, not a memorized action schedule
or a seed-aware gameplay controller. It uses only real controls and keeps ROM,
mode 0, difficulty 0, zero sticky actions, and 18000-frame budget fixed.
Train seeds are 0,2,4,6,8,10,12,14; held-out seeds 1,3,5,7,9,11,13,15. Both groups
interleave the same launch family. Prelude frames, actual rewards, lives lost and
gained are retained and disclosed. No outcome rejection or life reset occurs.

`make_game(scenario_suite="canonical")` disables the prelude for separately
reported original fixed-start runs. Different canonical seeds are **not** independent
instances. Do not pool these scores with bounded-launch-v1, or claim transfer to
unseen modes. Fast/defense/hard-difficulty variants are intentionally not offered
until mode mapping and missing enemy observation coverage are validated.

## Observation and limits

Strict JSON, object-level RAM state, raw RAM by default, no automatic screenshots.
`include_ram=False` omits RAM without changing dynamics. player presence and packed
orientation (0 up, 4 left, 8 down, 12 right), fifteen asteroid source slots, two
player missiles, approximate positions/sizes, raw encoded coordinates, score and
native lives are exposed. Source slots are not permanent object identities; same-size immediate reuse can be missed.
Finite-difference velocity uses up to four frames and is invalidated at gaps, size
changes, large jumps/wraps and respawns. Exact wrap periods and intermediate angles
are deliberately not invented. Boxes are unclipped approximate reference geometry,
not exact collision masks. Display flicker can hide active objects in one RGB frame. `enemy_craft` and `enemy_projectiles` are null/unknown,
never claims of absence. See THIRD_PARTY.md.

Snapshots are immutable copies stamped with the emulated frame. They contain no
wall-time/id/path noise; fresh same-seed resets are reproducible. A snapshot's
`observation_age_frames=0` applies at emission; clients calculate later age from
its stamp. Optional `snapshot_png()` saves actual ALE RGB under runtime/screenshots
and returns the path. Runtime never modifies source files.

## Validation

Run the generic `metaharness.games.adaptor.validate_adaptor` with expected_tier=2.
Additional real-emulator checks and their measured outcomes are recorded under
runtime/; no fixture or mock emulator is used for integration tests.
