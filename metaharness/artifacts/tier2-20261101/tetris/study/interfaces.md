# Programmatic interfaces

## Recommendation

Use **`tetris-gymnasium` as a local, headless Gymnasium environment**, wrapped by a thin JSONL adaptor that adds an explicit real-time clock, the full symbolic state, deterministic reset/seed, and the chosen scoring/level policy.  It is the best fit for this run because it is pure local software, MIT-licensed, configurable, has no account or server, supports `render_mode=None`/`ansi`, exposes a dictionary board observation, and already implements the logical actions (movement, rotation, hard drop, hold, no-op).  Its public source is [GitHub](https://github.com/Max-We/Tetris-Gymnasium), its [installation docs](https://max-we.github.io/Tetris-Gymnasium/introduction/installation/) say `pip install tetris-gymnasium`, and PyPI metadata at [pypi.org/pypi/tetris-gymnasium/json](https://pypi.org/pypi/tetris-gymnasium/json) reports Python `>=3.10,<4.0`, MIT license, and current dependencies including Gymnasium, NumPy, JAX/Chex, and OpenCV.

The adaptor should prefer the base environment’s board dictionary over an RGB wrapper.  The documented environment API has `board`, `active_tetromino_mask`, `holder`, and `queue` arrays, `Discrete` actions for `move_left`, `move_right`, `move_down`, clockwise/counter-clockwise rotation, `swap`, `hard_drop`, and `no_op`, optional gravity, optional `ansi`/RGB rendering, `reset(seed=...)`, and an info field such as `lines_cleared` ([environment source](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/envs/tetris.py), [action mapping](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/mappings/actions.py)).  `GroupedActionsObservations` is useful for a fast placement adaptor: it enumerates `x * 4 + rotation` destinations and returns an `action_mask`, but it should be used only after validating its semantics against the pinned package version ([wrapper source](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/wrappers/grouped.py)).

Do not ask Astra or Jev to control a GUI.  The engine owns legality, SRS, collision, gravity, lock delay, scoring, and shortest-input generation; Jev chooses among already legal candidates and Astra supplies longer-horizon strategy.

## Option A — local `tetris-gymnasium` (chosen)

**Maturity:** actively documented, modular Gymnasium environment with tests and a published package; newer than many small Tetris Gym packages, so pin and smoke-test the exact version.  It is a general RL environment rather than a finished agent protocol; the JSONL adaptor remains necessary.

**Latency:** in-process `step()` latency is normally milliseconds.  It has no network round trip.  Real-time pacing is the adaptor’s responsibility: run a monotonic-clock loop and call/update gravity independently of model calls.  If the selected version advances one cell per `step`, the adaptor should subdivide elapsed time into deterministic ticks or implement the clock around the engine rather than pretending one model call equals one turn.

**Observation:** symbolic NumPy arrays: a padded board, active-piece mask, hold, and queue.  Convert them to JSON lists plus explicit `piece`, `orientation`, `x`, `y`, `ghost_y`, `can_hold`, timers, score, lines, level, combo, B2B, and seed.  If exact piece identities are not recoverable from the installed package’s arrays, maintain them in the adaptor’s piece source and verify them in a reset/step smoke test.  Do not use `RgbObservation` unless debugging; it intentionally converts state to pixels ([observation wrapper source](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/wrappers/observation.py)).

**Actions:** expose named JSON actions and map them to the package’s integer mapping: left, right, down/soft drop, rotate clockwise, rotate counter-clockwise, hard drop, swap/hold, and no-op.  Reject unknown actions before calling the environment.  If using grouped placements, map the selected `(column, rotation)` through `GroupedActionsObservations`, check `info["action_mask"]`, then emit the shortest legal low-level sequence.  The adaptor must label whether an action is rejected, buffered, or committed.

**Cost/legal risk:** free/open-source local installation; no account, API key, server, or online ToS.  The package’s own license is MIT; comply with its license and dependencies.  It is not the trademarked Tetris.com service, and the local ruleset should be described as Guideline-compatible rather than falsely calling the package official.

### Installation and setup

Run all installation commands with the evaluator interpreter, not system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/tetris/.venv/bin/python -m pip install --upgrade tetris-gymnasium
```

The adaptor phase should:

1. Pin the tested `tetris-gymnasium` version in its requirements/lock file and record `python --version`.
2. Import the package and create `Tetris(render_mode=None, width=10, height=20, gravity=...)` (or the package’s registered Gymnasium ID after verifying the installed release).
3. Configure standard I/J/L/O/S/T/Z tetrominoes, seven-bag randomizer, standard action mapping, and the chosen reward/scoring mapping; do not rely on undocumented defaults without a smoke test.
4. Set `reset(seed=seed)` and record the seed, first queue, and initial active piece.  Verify that two resets with the same seed produce identical sequences and different seeds produce meaningful starts.
5. Verify the action indices from the installed `actions.py`; do not hard-code indices without checking the pinned package.
6. Set `render_mode=None`; do not open an OpenCV window.  Use `ansi` only for diagnostics.
7. Add a monotonic clock/worker loop that advances gravity, lock delay, and line-clear/entry timing while model requests are pending.  If the package is configured as a discrete drop environment, implement the real-time wrapper explicitly and document the tick duration.
8. Serialize observations and events as JSONL; include an engine version/ruleset identifier so a replay is reproducible.
9. Test wall kicks, hold-once-per-piece, hard drop, line clear, top-out, scoring, seed repeatability, and model-call stalls before evaluation.

**Credentials/environment:** none required.  The `env` list in `study.json` is empty.  No display, GPU, account, server address, or API key is needed.

## Option B — `gym-simpletetris`

[GitHub](https://github.com/tristanrussell/gym-simpletetris) and [PyPI](https://pypi.org/project/gym-simpletetris/) provide a small pure-Python/OpenAI-Gym-style environment.  It offers a `ram` observation (grid array), grayscale, or RGB, and configurable `lock_delay`/`step_reset`; its README documents `pip install gym-simpletetris` and line-clear reward variants ([raw README](https://raw.githubusercontent.com/tristanrussell/gym-simpletetris/master/README.md)).

* **Maturity:** simple and understandable, but its public documentation does not clearly specify current Python/Gym compatibility, action meanings, seed behavior, or Guideline fidelity; treat it as a fallback, not the primary engine.
* **Latency:** local/in-process; no network latency.  Its default lock delay is expressed in steps, so a real-time adaptor still needs a clock.
* **Exposure:** RAM/grid is closer to symbolic state than pixels, but active/settled-piece semantics and queue/hold support must be audited in the pinned source.
* **Legal/cost:** free local package; no credentials.  Confirm its license and dependency versions before redistribution because the README/PyPI page was not sufficiently explicit in the available documentation.

## Option C — ALE/Gymnasium Atari Tetris

Farama’s [ALE Tetris documentation](https://ale.farama.org/environments/tetris/) exposes `gymnasium.make("ALE/Tetris-v5")`.  The default action space is `Discrete(5)` (NOOP, FIRE, RIGHT, LEFT, DOWN), `full_action_space=True` provides 18 actions, and observations are RGB `(210,160,3)` or RAM `(128,)`; the environment uses frame skip 4 and action-repeat probability 0.25.  Install `ale-py` per [ALE getting started](https://ale.farama.org/getting-started/), then install/register the Gymnasium integration as appropriate for the pinned release.

* **Maturity:** mature Atari emulator and standard Gymnasium interface.
* **Latency:** local emulator, but actions are frame-based and the state is NES/Atari-specific.  The default stochastic frame behavior is a poor match for a transparent Guideline symbolic engine.
* **Exposure:** pixels or opaque 128-byte RAM, not a clean modern 10x20/hold/queue JSON state.  RAM reverse engineering would be required.
* **Legal/cost:** software is open source, but ROM distribution/licensing must be checked for the exact ALE release and ROM package.  It also implements an Atari/NES-like variant, not the selected Guideline rules.  No credentials are needed, but ROM/legal setup can block a reproducible offline evaluator.

## Option D — `gym-tetris` / NES emulator (`nes-py`)

[Kautenja/gym-tetris](https://github.com/Kautenja/gym-tetris) wraps NES Tetris with `nes-py` and Gymnasium.  It offers full 256-button actions or `MOVEMENT` (12) / `SIMPLE_MOVEMENT` (6), and its info can include current/next piece, score, lines, board height, and statistics.  It is useful only if the evaluator explicitly wants NES Tetris.

* **Maturity:** recognizable emulator-based project with a conventional Gym interface; release/Python support and ROM setup must be checked against the exact pinned version.  The public repository currently advertises modern Python CI, but this may not match Python 3.12.3.
* **Latency:** local emulator/frame-level, but extra emulation and screen/RAM translation add complexity.
* **Exposure:** emulator frames plus metadata; not the clean Guideline board/hold/SRS state wanted here.
* **Legal/cost:** free software, but NES ROM acquisition/distribution has legal/licensing risk.  No game account is needed.  Its rules differ from the chosen engine (history randomizer, NRS, no hold/hard drop/wall kicks), so strategy prompts would need a separate NES prior.

## Option E — emulator hooks

**PyBoy** ([documentation](https://docs.pyboy.dk/), [repository](https://github.com/Baekalfen/PyBoy)) embeds a Game Boy emulator in Python and exposes memory reads/writes, button presses, frame ticking, and screenshots.  A Game Boy Tetris ROM could be driven headlessly, but the adaptor would need a legally obtained ROM and reverse-engineered RAM addresses for board/piece/score.

**Nintaco** has an external API and Tetris bot examples, e.g. [nintaco-go-api-tetris-bot](https://github.com/meatfighter/nintaco-go-api-tetris-bot).  It can control an emulated NES at fine granularity, but requires the emulator, ROM, API process, and a ruleset-specific memory map.  These are valuable research hooks, not good choices for this offline run.

* **Maturity/latency:** mature enough for emulator experiments; local frame latency is low, but API/process synchronization and memory decoding increase failure modes.
* **Exposure:** raw memory and/or pixels; no canonical JSON unless the adaptor creates it.
* **Legal/cost:** emulator packages may be free/open source; ROMs are the user’s responsibility.  No cloud credentials, but legal provenance and version drift are risks.

## Option F — screen capture + input injection

A desktop/browser Tetris client can be observed with screenshots and controlled with X11/uinput/xdotool/PyAutoGUI.  This is the least suitable interface on the stated server: it needs a display or virtual framebuffer, pixel recognition, focus management, and a timing-sensitive input path.  It exposes screenshots rather than exact board state, and latency/jitter is highest.  It also risks violating a service’s rules if used online.

TETR.IO’s official [rules](https://tetr.io/about/rules/) explicitly prohibit third-party utilities, macro programs, bots, and assistance tools.  Do not use an online account or screen bot for this study.  An official Tetris.com developer/control API was not found; public bots and the [Tetris Bot Protocol MVP](https://gist.github.com/MinusKelvin/de519f7e922a1c0821c3ac55de10384c) are community interfaces, not an endorsed offline game engine.  The protocol uses JSON messages over stdin/stdout or web-worker messaging, observes board/queue/hold/combo/B2B, and requests placements, but it leaves timing/opponent details unspecified and supports Guideline-conforming frontends rather than supplying one.

## Interface contract for the next phase

The adaptor should expose a local process such as:

```text
reset(seed) -> {"state": {...}, "ruleset": "guideline-marathon-v1"}
step({"action": "...", "dt_s": ...}) -> {"state": {...}, "events": [...], "done": false}
```

A state should contain JSON-native values:

```json
{
  "board": [[0,0]], "active": {"type":"T","orientation":0,"x":4,"y":19,"cells":[[4,19]]},
  "ghost_cells": [], "hold": null, "can_hold": true, "queue": ["J","L","I","O","S"],
  "score": 0, "lines": 0, "level": 1, "combo": -1, "back_to_back": false,
  "gravity_s_per_cell": 1.0, "lock_remaining_s": null, "game_over": false,
  "seed": 0, "ruleset": "guideline-marathon-v1"
}
```

The example is schematic; the adaptor must report all 10x20 cells, exact active coordinates, and actual timers.  `dt_s` should be measured by the adaptor, not invented by Jev.  Every response should include line-clear/lock/top-out events and an `illegal_action` count.  This contract makes it possible to replay a seed without GUI, network, credentials, or hidden state.
