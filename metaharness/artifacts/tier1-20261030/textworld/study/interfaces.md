# TextWorld interfaces and adaptor handoff

## Recommendation and evidence status

Use **TextWorld's native Python API**, generating a distinct game for each scenario seed and running the compiled **Glulx `.ulx`** story through `textworld.start`. Keep the generated `.json` sidecar beside the story. The native environment avoids Gym version adapters, supplies typed score/terminal fields, and lets us declare exactly which textual assistance is allowed. No GUI, account, paid service, or persistent server is needed. [T1–T4]

The interface contract below reflects the established TextWorld API and is a concrete implementation target. **It has not been executed against this machine:** the exact evaluator venv is Python 3.12.3 with TextWorld absent, network DNS failed, and the venv is outside the permitted write roots. Upstream URLs are verification targets; only the Jericho fixture references marked L1 were read locally. See `research/sources.md`. Do not mark compatibility verified until the acceptance checks below pass.

## Practical interface comparison

All latency descriptions are qualitative, not measurements from this machine. There is no need for subsecond decisions. Startup/compilation can be much slower than individual steps.

| Interface | Maturity and exposure | Latency / operational cost | Licensing, ToS, and suitability |
|---|---|---|---|
| **Native `textworld.start` + `EnvInfos`** | First-party research API. Text feedback plus selectable description, inventory, objective, score/max score, moves, win/loss, admissible actions; optionally privileged internals. Compiled-story interpreter behind API. | Local call plus interpreter IPC; expected far below an Astra call, but benchmark it. CPU only; no per-call fee. One process/story per active episode. | Intended for automated research. No online-game ToS or anti-cheat. TextWorld project is MIT; inspect bundled third-party licenses separately. **Chosen.** |
| `textworld.gym` registration and batch environments | First-party convenience layer over TextWorld: `register_game`/`register_games`, `gym.make`, text observations, info dictionaries, vectorized games and episode limits. | Similar game stepping; batching helps throughput but adds lifecycle and return-shape handling. Free. | Same underlying licenses. Established **legacy Gym** integration; do not assume Gymnasium compatibility or modern five-element `step`. Avoid this dependency surface for one turn-by-turn agent. [T6] |
| Native symbolic/JSON environment (`TextWorldEnv` backend) | First-party logical state transition backend for generated JSON games; useful for validation and planning research. Text rendering and available infos can differ from interpreter play. | No compiled story interpreter; potentially lower overhead. Free. | Direct state access can reveal hidden facts. Not a drop-in natural-language benchmark without parity checks. Do not silently substitute it if compilation fails. [T7] |
| `tw-play` terminal CLI / stdin–stdout automation | First-party human-play/demo route. Terminal feedback and typed commands; CLI agents can also be selected in supported versions. | Process/PTY orchestration and prompt parsing; inexpensive but less robust than API. | No account or fee. Loses typed score/info benefits; appropriate for manual diagnosis. [T1] |
| Jericho / Frotz for generated `.z8` Z-machine stories | Mature separate IF research interface, `FrotzEnv`, text, score, done, snapshots and other utilities. Local upstream tests explicitly exercise TextWorld `.z8` games. **Not a `.ulx` interpreter and not a generator.** | Local native interpreter calls. Free, headless. | TextWorld integration and enhanced-info support vary by story/version. Jericho delta rewards differ from native TextWorld cumulative score. Additional dependency/license surface. Good fallback only if a separately specified Z-machine suite is validated. [L1, T8] |
| Standalone Glulx/Git or Z-machine/Frotz interpreter driven via pipes | Mature IF VMs can play compiled compatible stories without TextWorld. Exposes narrative and status-line text; rich infos require game-specific instrumentation. | Small local process overhead; terminal parsing can be brittle. Free implementations available. | Must match VM to file format and preserve binary/third-party licenses. No advantage over TextWorld's built-in interpreter bridge. |
| Browser/graphical IF frontends, screenshots + keyboard injection | Possible presentation layer for compiled games; TextWorld visualization/debug facilities are also presentation aids. Pixels/DOM provide less reliable state than the native API. | Highest setup/parsing overhead; virtual display/browser potentially required. | No benefit on this headless server; additional licensing/assets to provision. Not required. |
| Community agents/frameworks, e.g. GATA, KG-A2C, DRRN-related implementations | Policy/research code built around environments, not an independent official control interface. Can provide memory/action-selection designs; may request privileged infos or use training data. | Training/checkpoints/ML frameworks can be costly; inference is optional. | Review repository and checkpoint licenses. No external model/service is required to run TextWorld itself. Reuse ideas, not heavyweight agent packages, for this task. [C1–C3] |

No mod, emulator screen hook, paid game purchase, remote host, or multiplayer account is necessary. The main risk is old research-package dependency compatibility, not anti-cheat.

## Software and offline provisioning

**Interpreter to use for every package/build/validation command:**

`/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textworld/bin/python`

Do not substitute system Python, install into another game's venv, or assume a package built for Python 3.10/3.11 works with 3.12. The study verified Python 3.12.3, `gcc`, `g++`, `make`, Python 3.12 headers, and a libffi header. It did not validate TextWorld or its native extensions.

Required installation inputs:

1. A **pinned, validated TextWorld release or source commit**, supplied initially through `pip textworld` as requested. Resolve the exact release using PyPI metadata and upstream source; this study cannot honestly name a verified current version.
2. All transitive runtime and build dependencies from that pin, locked with versions/hashes. TextWorld generations have depended on NumPy, NetworkX, TatSu, CFFI, hashids, tqdm, and related packages; treat this as a compatibility checklist, **not a complete or current requirements file**. Some releases also pull Gym/Jericho even when not choosing those interfaces. Inspect actual package metadata.
3. Linux native build tools where no compatible wheels exist: C/C++ compilers, `make`, Python 3.12 development headers, libffi development files, and packaging build requirements. Git/curl may be provisioning tools but runtime must not need the network.
4. The exact Inform compiler tooling and Glulx interpreter assets required by the pinned TextWorld build. Verify that these are packaged or staged locally and executable. An sdist or source checkout may fetch/build assets during installation; a wheel bundle alone is not sufficient unless that behavior has been audited. Preserve license notices for TextWorld and every bundled component. Legacy Inform distributions have distinct terms; do not equate zero price with an OSI-approved license. If “free software” means strictly libre, confirm every bundled compiler component satisfies that constraint before freezing the build. [T1, T9]

Provisioning procedure, for a phase with network and venv write access:

- Fetch release metadata/source and inspect Python version classifiers, `Requires-Python`, dependencies, build scripts, and licenses. Use a matching CPython 3.12/Linux x86_64 machine to resolve/build a wheelhouse, including build dependencies and any non-wheel compiler assets. Preserve source versions and patches.
- Install into the **specified** venv and validate there. On a network-enabled provisioning machine, the initial intent is `<exact-python> -m pip install textworld`; the final reproducible install must use a lock, not the floating name.
- For offline installation after staging a hash-locked requirements file and complete wheelhouse in the study directory, the command is:

  ```sh
  /home/hemlix/GamingMaster-mh-tier1/runs/venvs/textworld/bin/python -m pip install --no-index --find-links=/home/hemlix/GamingMaster-mh-tier1/runs/textworld/study/wheelhouse --require-hashes -r /home/hemlix/GamingMaster-mh-tier1/runs/textworld/study/requirements.lock
  ```

  `wheelhouse/` and `requirements.lock` are **future provisioning outputs**, not files delivered by this study. If building from source, stage all build requirements first; pip build isolation can otherwise try to download them.
- Run `pip check`, import/generation/compile/reset/step checks, and an entire episode with networking disabled. Lock the successful TextWorld/dependency/compiler versions and story checksums.
- Python 3.12 removed `distutils` and exposes incompatibilities in old NumPy/Gym/native build stacks. A dependency conflict is not permission to change the evaluator interpreter. Prefer a compatible upstream release or a documented minimal build patch. Do not silently substitute a different game/backend.

**Credentials/environment:** no user credentials, account, API key, server address, display server, or required game-specific environment variable. Therefore `interface.env` is an empty array. Model credentials belong to the surrounding harness, not TextWorld. Supply artifact/cache paths as adaptor configuration. Use a writable per-game build directory and ensure the packaged interpreter is executable there. Any release-specific compiler override should be derived from that release's source; do not invent environment-variable names.

## Generation, difficulty mix, and `reset(seed)`

These are **proposed benchmark settings**, not upstream defaults. Freeze them before comparing agents. Use generic generated quests for the baseline; this avoids silently conflating different cooking/treasure challenge settings.

| Profile index | Weight | `nb_rooms` | `nb_objects` | `quest_length` | Submitted-command cap |
|---|---:|---:|---:|---:|---:|
| 0: easy | 1/3 | 3 | 5 | 2 | 100 |
| 1: medium | 1/3 | 5 | 10 | 5 | 200 |
| 2: hard | 1/3 | 10 | 20 | 10 | 400 |

Use nonnegative integer episode seeds below 2^32. Set `profile_index = seed % 3`; use the seed itself as the root generation seed. Construct a fresh `textworld.GameOptions`, set `options.seeds`, `options.nb_rooms`, `options.nb_objects`, `options.quest_length`, and a unique `options.path` ending in `.ulx`, then use `textworld.make(options)` to obtain `(game_file, game)`. Confirm this convenience function and the `.ulx` output choice against the pinned version. Do not set only the environment's RNG: resetting a fixed story generally restarts that story and does **not** generate a new map. [T4]

Use a cache key containing suite version, profile, root seed, retry index, TextWorld version, and generation configuration. Keep the `.ulx` and `.json` together. Store a private manifest with full settings, actual generation seeds, compiler/version hashes, story hash, max score, and cap. Never expose `game`, its quests, walkthrough, or hidden facts to the player.

Generation may reject constraints or time out. Define a bounded deterministic policy before evaluation: up to five attempts with `generation_seed = (root_seed + 1000003 * retry_index) mod 2^32`, retry indices 0–4; preserve the original profile. Log every rejected attempt and reason. If all fail, report an infrastructure/scenario error, not an agent loss, and do not switch to an easier profile. Use an explicit compilation/generation timeout chosen during provisioning and freeze it. Validate positive max score and generated solvability using privileged checks **outside** the agent run.

Held-out seeds must be disjoint from development seeds; use balanced triples of seeds for exact equal weighting. Map topology, item placement, names/descriptions, quest dependencies, and distractors should vary under the generator's seeded RNGs. Inspect manifests to detect duplicated outputs; procedural seeds alone do not prove diversity. If generation is inconsistent across versions, preserve the compiled artifacts rather than promising cross-version seed identity.

Cooking is an optional **separately versioned** suite: inspect `tw-make tw-cooking --help` and challenge source for recipe count, retrieval count, map-size settings, `open`, `cook`, `cut`, and inventory/drop flags. Map-size parameters in cooking are not interchangeable with generic `nb_rooms`. Do not add cooking or tune profiles after observing held-out scores. [T5]

## Native environment lifecycle and observation contract

The adaptor should perform this sequence (API notation is a contract, not a player implementation):

1. Close any previous environment. Resolve/generate the selected seed's validated story.
2. Construct `textworld.EnvInfos` requesting `description`, `inventory`, `objective`, `admissible_commands`, `score`, `max_score`, `won`, `lost`, and `moves` where supported by the pinned release.
3. Call `textworld.start(game_file, request_infos=infos)` and then `env.reset()`. Native reset returns a `GameState`, **not** Gymnasium's `(observation, info)` tuple. Read its text fields and requested infos. Inspect and test field availability, including reset-time terminal flags.
4. Submit one command string with `env.step(command)`. Native return is `(game_state, score, done)`, where **score is cumulative**. Serialize the new observation, update diagnostics, and stop on `done`. Also stop on the benchmark command cap, marking truncation separately.
5. Call `env.close()` on success, failure, exception, timeout, and reset. Do not leak interpreter subprocesses or share one mutable instance across games. [T3, T7]

Recommended agent-visible JSON fields:

| Field | Source / semantics |
|---|---|
| `feedback` | Latest native feedback string, preserved verbatim |
| `description`, `inventory`, `objective` | Requested native strings; distinguish missing from empty |
| `admissible_commands` | Current native string list; this is an explicitly allowed action-selection handicap |
| `score`, `max_score` | Native numeric values; validate positive max |
| `won`, `lost`, `terminated` | Native flags / `done`; `done` alone does not mean victory |
| `truncated` | Adaptor's cap or external evaluator cutoff, distinct from native loss |
| `commands_used`, `command_limit`, `commands_remaining` | Adaptor counters; every submitted command counts, even invalid/no-op |
| `moves` | Engine count if supported; never use it in place of command budget |
| `last_command`, `last_score_delta` | Adaptor diagnostics; delta = new cumulative score minus previous cumulative score |

No screenshot or object-tree serializer is needed. Do not serialize all of `GameState` blindly: it may hold internal state. Use an allowlist. Disable `facts`, `game`, `policy_commands`, `intermediate_reward`, and `extras` such as `walkthrough`; also avoid revealing `entities`/location lists that may enumerate unseen content. Treat generated JSON and manifests as evaluator-private. `admissible_commands` may itself reveal executable interactions not obvious in prose; disclose it as assistance, and do not compare directly to a text-only benchmark. [T3]

If text-only evaluation is desired later, create a distinct information-policy version: remove admissible actions and obtain needed observations under a clearly defined command-cost policy. Do not switch midway through a suite.

## Actions, scoring, and recovery

Accept one string per step. Reject malformed adaptor payloads without advancing the game, with an explicit protocol error. Do not run model text through a shell. For engine commands, preserve canonical names; pass supported single commands to TextWorld and let it return game feedback. Do not concatenate a planned route or multiple actions into one string to evade the command cap.

When admissible commands are provided, select an exact member by stable index/string. Adaptor bookkeeping should not claim a move or pickup succeeded until feedback/new observations support it. Invalid actions are not fatal infrastructure errors. Avoid automatically issuing `look`/`inventory` after each action if equivalent requested infos are already available; extra steps can consume budget and change the game.

Score = **final native cumulative score / native max score**, per episode, then arithmetic mean across the balanced suite. No policy/intermediate reward shaping, shortest-path bonus, reward summation, or replacing partial credit with a win indicator. Keep the last authoritative cumulative score on cap truncation. Preserve partial credit on genuine loss unless a separately declared benchmark requires otherwise. If the returned score disagrees with a requested score field, treat it as a contract bug to investigate, not an opportunity to choose the higher value.

A process crash, missing sidecar, failed compile, or absent score is an infrastructure error with logs and deterministic replay information. Do not quietly count it as score zero or regenerate a different game during play. Do not expose privileged snapshots or use trial `step` calls and undo to enumerate outcomes unless a future benchmark expressly allows that search interface.

## Acceptance checks before writing the full player

- **Exact environment:** install/import with the named Python 3.12.3; `pip check` passes; record package and native asset versions.
- **Offline operation:** from a cold process with networking disabled, generate and compile one game from each profile, reset, obtain every allowlisted info, step, and close. Repeat without cached generated outputs to catch hidden downloads.
- **Scoring:** in a private diagnostic run, use a generated solution to confirm final score equals max and `won`; compare successive native scores to show cumulative vs delta semantics. Keep this diagnostic oracle out of the player path.
- **Partial credit and failure:** inspect a suitable generated quest/fixture with intermediate score and a destructive action; verify score retention, `lost`, `done`, and zero division handling. Not every generic game has partial rewards.
- **Determinism/diversity:** same seed+config+pin yields the same initial observation and replay; different seeds vary generated content; reset does not continue the prior episode.
- **Budget and terminal handling:** invalid/informational commands count against the adaptor cap; cap yields truncation without fake native loss; no steps after termination; cleanup works on exceptions.
- **Observation policy:** player JSON and logs supplied to the model contain no facts, quest graph, walkthrough, game object, future policy commands, or private manifest. Admissible actions remain allowed consistently.
- **Latency/lifecycle:** measure generation, reset, and step separately; verify wall-clock waiting does not change a game; run repeated reset/close cycles to detect resource leaks. No subsecond policy deadline is required.

## Sources

- [T1] https://github.com/microsoft/TextWorld
- [T2] https://arxiv.org/abs/1806.11532
- [T3] https://textworld.readthedocs.io/en/latest/textworld.html ; https://github.com/microsoft/TextWorld/blob/main/textworld/core.py
- [T4] https://github.com/microsoft/TextWorld/tree/main/textworld/generator ; https://github.com/microsoft/TextWorld/blob/main/textworld/helpers.py
- [T5] https://github.com/microsoft/TextWorld/blob/main/textworld/challenges/tw_cooking/cooking.py
- [T6] https://github.com/microsoft/TextWorld/tree/main/textworld/gym
- [T7] https://github.com/microsoft/TextWorld/tree/main/textworld/envs
- [T8] https://github.com/microsoft/jericho ; https://jericho-py.readthedocs.io/en/latest/
- [T9] https://github.com/microsoft/TextWorld/blob/main/LICENSE.txt ; https://github.com/microsoft/TextWorld/blob/main/setup.py ; https://github.com/ganelson/inform
- [L1] https://github.com/microsoft/jericho/blob/752ba6720d522909a4a2a495b5a03dd535f9adf1/tests/test_textworld_games.py (locally read; saved under `research/`)
- [C1] GATA: https://github.com/xingdi-eric-yuan/GATA-public
- [C2] KG-A2C: https://github.com/rajammanabrolu/KG-A2C
- [C3] DRRN: https://arxiv.org/abs/1606.02654 ; https://github.com/microsoft/tdqn
