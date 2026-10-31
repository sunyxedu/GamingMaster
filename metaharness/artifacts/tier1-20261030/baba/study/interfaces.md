# Programmatic interfaces and adaptor specification

## Recommendation and present blocker

Choose the **Baba Is AI Python environment in process**, using its native transition engine and lossless symbolic board state. It fits Linux without a GUI, offline execution, exact turn boundaries, and the requested benchmark. Prefer its public environment API; use documented/source-inspected internals only where its public observation omits essential information.

**This choice is conditional on obtaining the actual package source.** No installation or runtime validation was possible during this study. The target venv exists and runs Python 3.12.3, but contains only pip. The session cannot resolve PyPI or GitHub; no local Baba distribution was located. It also permits writes only in the study directory and `/tmp`, so the requested external venv is not writable through this session's allowed filesystem scope. No installation was attempted there and no request to override these restrictions is needed for writing this study.

The exact upstream owner, release, license, import name, registration IDs, API signatures, action enum, observation encoding, success flag, and native level-selection method remain unverified. This document supplies a concrete integration contract and inspection procedure, but cannot honestly supply a tested invocation without those artifacts. **The next phase must resolve this source-acquisition blocker before claiming a working adaptor.** Do not substitute a similarly named clone or make up a Gym ID.

## Interface comparison

Latencies below describe expected architecture, not measurements.

| Interface | Maturity / availability | Observation and control | Latency | Cost and legal considerations | Decision |
|---|---|---|---|---|---|
| `baba-is-ai` Python package, public environment API | User-designated target; exact release and maintenance unverified | Expected native reset/step plus board observation; actual protocol must be inspected | Synchronous local calls; no network or input/render loop required | Intended free route; confirm package, assets, and level licenses | **Chosen**, subject to source verification |
| Same package's direct engine/grid access | Depends on source and release; internal interfaces may change | Full object layers, words, rules, directions, win state; possible snapshot support | Local memory access | Same licensing; pin source commit to prevent drift | Supplement public API if needed; never maintain a second rules engine unnecessarily |
| Another free Baba-like implementation or research bot library | Different implementations exist; none verified here as a drop-in replacement | Potential symbolic simulator, solver API, or browser state | Local simulation can be quick; browser bridge adds overhead | Audit repository and asset licenses; mechanical and level mismatch is the main risk | Only with explicit benchmark equivalence; not an automatic fallback |
| Commercial Baba Is You with mod/plugin or internal scripting | Mature game, but no supported benchmark automation API verified | Potential internal level/object access through a mod; requires exact version integration | Local, but input timing and frame loop matter | Paid proprietary game and asset/mod redistribution questions; fails free-software constraint | Reject for this task |
| Commercial game with screen capture and input injection (Xvfb, capture, keyboard tools) | Generic automation is mature; game recognition and headless setup are custom | Pixels and key presses; hidden overlap/rules hard to recover | Rendering/capture/input synchronization overhead; no need for Jev just because it has frames | Paid game; platform/automation terms require review; no online competitive anti-cheat expected in local puzzles | Reject: unnecessary GUI and fragile perception |
| Browser clone controlled through JavaScript/Playwright | Possible for a compatible free clone, but none validated here | Canvas screenshots, DOM/JS state if exposed, discrete key input | Browser startup and bridge overhead | Browser itself free; clone code/assets/levels need separate license review | Different game unless exact benchmark assets and mechanics match |
| Emulator hooks | No relevant native emulator interface identified for the requested Python benchmark | Would expose machine memory/input, not automatically meaningful puzzle state | Extra integration and interpretation cost | Distribution rights depend on the game image | No practical reason to use |

Do not confuse “a solver for Baba Is You” with “the environment used by this evaluation.” Solvers can suggest search techniques, but their level syntax and physics may differ.

## Acquisition, software, and offline installation

Use **only** this interpreter for environment inspection, installation, and final checks:

`/home/hemlix/GamingMaster-mh-tier1/runs/venvs/baba/bin/python`

Required distribution: `baba-is-ai`, or the upstream source package identified by its package metadata. Required transitive dependencies must come from that release's `Requires-Dist` / `pyproject.toml`; **Gym versus Gymnasium, NumPy, pygame, Pillow, and their versions are not established**. Do not install a large guessed stack. Git is needed only for a Git checkout, not for an already acquired wheel/archive. No browser, display server, emulator, paid game, GPU, or account is part of the chosen interface.

Source/discovery URLs (unretrieved in this session):

- Package page: https://pypi.org/project/baba-is-ai/
- Machine-readable metadata: https://pypi.org/pypi/baba-is-ai/json
- Repository discovery: https://github.com/search?q=baba-is-ai&type=repositories
- General Gymnasium API, **only if the package uses it**: https://gymnasium.farama.org/api/env/
- Legacy Gym compatibility context: https://gymnasium.farama.org/introduction/migration_guide/
- pip download and offline-install documentation: https://pip.pypa.io/en/stable/cli/pip_download/ and https://pip.pypa.io/en/stable/cli/pip_install/

Do not guess a GitHub owner. Follow `project_urls` from trusted package metadata and verify the README actually describes this benchmark. Record distribution version, repository commit if applicable, license, and artifact SHA-256 hashes in a local manifest. Include levels and any needed assets; a package that tries to fetch them on first reset is not yet offline-ready.

Provisioning sequence for the adaptor phase:

1. Obtain the package source/wheel and metadata from a connected provisioning machine or an existing offline artifact store. If the package name does not resolve, use an identified upstream checkout and record that change explicitly.
2. Inspect `Requires-Python`, build configuration, dependency bounds, licenses, level files, and imports before choosing versions. Confirm Python **3.12** support, not merely “Python 3.” Old dependencies that require removed stdlib modules or lack cp312 wheels are a real blocker. Pure Python wheels may still contain incompatible code.
3. On a Linux x86_64 machine with the same Python 3.12 ABI, build/download the complete dependency closure into `wheelhouse/`. `python3.12 -m pip download --dest wheelhouse baba-is-ai` is an acquisition template, not a command successfully executed here. Pin the selected version once known. Source-only dependencies may need building there; bring final wheels and hashes to the offline server.
4. With write access to the designated evaluator venv, install locally:

   ```sh
   /home/hemlix/GamingMaster-mh-tier1/runs/venvs/baba/bin/python -m pip install --no-index --find-links ./wheelhouse baba-is-ai
   /home/hemlix/GamingMaster-mh-tier1/runs/venvs/baba/bin/python -m pip check
   ```

   Run from the directory containing the acquired wheelhouse. Select the pinned version or exact wheel in the finalized command. For an upstream source archive, preferably build a wheel first; offline isolated builds also require their build dependencies.
5. Save the resolved package list and artifact identities. Verify the environment can import, construct, reset, observe, step, and close with no network and no display. Import success alone does not establish usability. Record actual timings after this check rather than assuming a millisecond budget.
6. Run the acceptance checks below before connecting a model.

No game-playing code is included in this study. These are dependency provisioning and source-inspection instructions.

## Credentials and environment variables

**No credentials, API key, server address, login, or network service are required by the intended local interface.** `study.json` therefore has an empty `interface.env` array. Model service configuration is outside this game's interface and must not be recorded as credential values here.

If pygame/SDL is actually imported and initializes display/audio, optional process variables are `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`, set before importing it. They are not requirements for a pure symbolic backend and do not repair a package that unconditionally needs a graphical window. Prefer rendering disabled. Do not set `DISPLAY` or install Xvfb without evidence it is needed.

## Establish the actual native API

Inspect installed distribution metadata (`pip show -f baba-is-ai`), README/examples, registration code, environment constructor, reset, step, rule evaluator, and bundled levels. Record a version-specific table with these answers before implementing the bridge:

| Required fact | Where to establish it | Why it matters |
|---|---|---|
| Import module and environment constructor / IDs | Metadata, package initialization, registration code, examples | Distribution names need not equal import names |
| Backend: Gymnasium, legacy Gym, or custom | Imports and method signatures | Determines reset/step tuple handling and wrappers |
| Action enum and exact integers | Enum definition and step dispatch | Never infer compass mapping from integer order |
| Available no-op, undo, reset controls | Enum and engine methods | Keyboard controls in the commercial game are not API guarantees |
| Observation type, shape, axes, encoding, and visibility | Observation space and encoder | RGB, compact integer grid, and dicts are not interchangeable |
| Cell stacking and object orientation | Grid storage and movement logic | A flat visible tile can omit future-relevant state |
| Win, loss, reward, step limit | Transition and wrapper code | Termination and reward need not mean success |
| Level selection and seed behavior | Reset implementation, constructors, level loader | Reseeding alone may never change the puzzle |
| Supported grammar and interaction order | Rule parser and transition pipeline | Full Baba Is You rules must not be assumed |
| State export/restore or safe copying | Public methods and mutable engine members | Simulation branches must not mutate the live game |

If Gymnasium is used, the usual protocol is reset returning `(observation, info)` and step returning `(observation, reward, terminated, truncated, info)`. That is **framework documentation, not verification that this package uses it**. Old Gym can return four values from step and a bare observation from reset. Select a compatibility path from the inspected version, and preserve timeout versus true terminal distinctions. Avoid broad exception handlers that turn a real engine error into a fake unsuccessful puzzle.

## Recommended adaptor contract (ours, not claimed upstream fields)

Expose a JSON state with:

- Scenario: stable level identifier, package/version identity, seed, board dimensions, step count, and action budget if present.
- Board: explicit coordinate convention, normally `(x,y)` with origin top-left and y increasing downward; each cell contains a **list** of objects. Each object distinguishes a physical entity from a word tile, records its literal noun/property/operator, and includes direction and other transition-relevant attributes when the backend has them. Preserve overlap and multiplicity. Do not confuse NumPy row/column with x/y.
- Rules: active rules and their text coordinates, preferably exported by the engine, plus current controlled-object identities and current goals. If a derived parser is unavoidable, label its output as derived and compare it with engine behavior. Do not present unsupported syntax as parsed with certainty.
- Outcome: separate `success`, `terminated`, `truncated`, and reason; optionally expose raw reward for diagnostics. Identify the exact source condition mapped to `success`.
- Actions: a name-to-native-action table established at startup; directional names should be unambiguous. Mark wait/undo/restart available only when source and evaluation policy allow them.

Read the engine's actual observation fidelity first. If the public observation is partial, adding internal state changes the task and requires agreement with the evaluation contract. For a fully observable puzzle board, lossless engine extraction is preferable to images. Do not leak reference solutions, future random draws, or hidden evaluation labels through observation.

Each action request advances **exactly one** native turn and returns the resulting state. Observation and rendering are read-only. When Astra proposes a macro, issue its primitives sequentially and compare the predicted state with each actual result; do not skip outcomes. Reject invalid action names without advancing the game, according to a documented adaptor policy. Preserve native behavior for valid but blocked moves.

Do not advertise an undo capability unless the benchmark authorizes it. A search snapshot is a private planning operation only if local branching is allowed; it is not an uncounted live undo. If exact copying is unavailable but branching is allowed, reconstruct a simulation from the pinned initial state and replay its action history. Include RNG state and step count where necessary. Cache only states whose full future dynamics match.

## Seed-to-puzzle mapping and score

A seed selects a **scenario**, not necessarily a native environment seed. Build a checked-in manifest of actual supplied levels and their identities. Use the package's documented native selection mechanism if it already fulfills this contract. Otherwise define and version a stable deterministic mapping from integer seed to level ID plus any legitimate level randomization. Do not rely on process-randomized Python `hash()` or unsorted filesystem order.

Repeated seeds must reproduce initial state and action outcomes. Distinct seeds can collide on a finite corpus; report distinct level count and never claim that infinitely many seeds mean infinitely many puzzles. Partition held-out levels by identity or template **before** assigning seed ranges. Prevent the same static layout or cosmetic variant appearing on both sides and being called generalization. Do not randomly scramble words or walls without a solvability guarantee. No invented procedural generator is needed if the benchmark supplies a finite corpus.

Score one point per scheduled puzzle solved, at most once per episode. Total solved is the primary user-requested score; solved fraction and attempts accompany it. Read explicit success or the inspected engine win predicate; never guess from reward positivity. Step-limit exhaustion, no-control deadlocks, and failure terminations score zero. Log underlying reward separately if useful. Restart attempts within one scheduled puzzle share its total budget unless evaluation specifies otherwise. Make comparisons on the same scenario manifest and action budgets.

## Acceptance checks for the next phase

These are required verification tasks, **not results already observed**:

1. Import and headless construction work in the exact Python 3.12.3 venv; `pip check` passes; assets resolve without network or a working directory dependency.
2. Enumerate real environments/levels and supported rules. Confirm a reset produces a valid board and that selected seeds cover multiple intended puzzles.
3. Observe twice with a delay and verify unchanged logical state. Verify each directional enum on a known small board. Confirm blocked commands and wait semantics.
4. Compare serialized coordinates, every layer in occupied cells, directions, and active rules against engine state. A screenshot may be used as an optional diagnostic, never the sole ground truth.
5. On a supplied trivial fixture/reference trace, confirm the terminal win transition scores exactly one. Separately verify timeout, loss, and repeated post-terminal calls cannot add points.
6. Replay identical seed/action traces in fresh instances and compare states. If snapshots are used, prove branch mutations do not affect live state and restore includes all future-relevant state.
7. Check rule changes immediately affect control/blocking as the source specifies. Test only supported interactions, with special attention to transformation, stacked objects, and simultaneous movement.
8. Persist package hashes, level manifest, seed mapping version, action mapping, resolved dependencies, and actual API facts beside the adaptor.

Completion of these checks is the gate for calling the adaptor operational. The study's recommended contract must not be mistaken for a verified upstream API.
