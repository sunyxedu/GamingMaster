# Programmatic interfaces and adaptor specification

**Recommendation: use the local `jericho.FrotzEnv` Python API with a pinned, licensed, checksum-identified corpus of instrumented Z-machine stories.** This gives direct text input/output, game-native scoring, deterministic seeding, and optional state inspection without a GUI or network service. The evaluator Python is `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/jericho/bin/python`, version **3.12.3**, verified locally. It currently contains pip but no Jericho. [J1–J3]

Evidence limit: fresh network requests to GitHub raw source, Read the Docs, PyPI, IF Archive, Brass Lantern, and Eristic failed DNS resolution on 2026-10-30. No upstream page, story license, or package metadata was successfully retrieved. References in this file are verification targets grounded in known projects, not claims of live retrieval. No runtime compatibility or game test has been completed. This matters: the adaptor is specified below, but it is not yet provisioned. Local checks and outstanding gates are recorded in `research-log.md`.

## Practical options

Latency estimates here are engineering expectations, not measurements on this host. Story complexity and command generation can dominate interpreter overhead.

| Interface | Maturity and latency | Observation / control | Cost and legal constraints | Assessment |
|---|---|---|---|---|
| **Jericho Python / modified Frotz** | Established IF research environment; native execution normally far faster than model calls. Ordinary steps often millisecond-scale; valid-action enumeration can take much longer. Python 3.12 needs a build/import test. | Parser text in/out; native score/reward, terminal flags, saves in memory; supported games also expose objects, world changes, candidate/valid commands. | Free software; inspect upstream and bundled dependency licenses. Story licenses are separate. No account, anti-cheat, or server ToS for local execution. | **Chosen.** Closest fit to requested environment and score metric. [J1–J5] |
| Frotz / `dfrotz` subprocess, pipes or PTY | Long-established CLI Z-machine interpreter. Local command round trips normally milliseconds. Prompt buffering/pagination and save dialogs complicate automation. | Human transcript and status line; enter text through stdin; save/restore through interpreter commands. No standard native Python score/object API. | Free software; check Frotz's license and each story's license. No service fee. | Useful fallback for running a legal ROM or diagnosing interpreter behavior, but score/done extraction becomes story-specific. Not a silent replacement for Jericho benchmarks. [F1] |
| TextWorld wrappers for existing Z-machine games | Mature research project; additional dependencies and version coupling. Python 3.12 support must be checked separately. | Can wrap supported games and present a different environment contract. Generated TextWorld games have richer game-specific metadata. | Free framework; stories remain separately licensed. | Use only if another evaluation layer requires it. Generated TextWorld quests change this task; do not substitute them for Jericho scenarios. [T1] |
| Direct Frotz/Jericho native C library via `ctypes`/FFI | Same underlying VM, but private ABI and lifecycle issues; low execution overhead, much higher maintenance burden. | Memory, objects, parser buffers, save/restore, native instrumentation depending on build. | Free native code subject to its license; no account. | Python wrapper already solves the useful integration. Only justified for a confirmed wrapper defect or missing hook. [J3, F1] |
| RemGlk + compatible interpreter, structured subprocess protocol | Established headless IF tooling. Interpreter/game support varies; low local latency. | JSON events/windows and input requests, text buffers; not a universal semantic game-state API. A Glk-capable interpreter is still required. | Free implementations; verify component/story licenses. | Viable when supporting multiple VM families, but Glulx stories and Glulxe are outside Jericho's Z-machine instrumentation. More work to recover score. [G1] |
| Browser interpreters such as Parchment, automated with Playwright | Mature player-facing IF tools; headless browser possible. More overhead and fragile UI selectors. | Rendered transcript/DOM, text entry, sometimes interpreter-specific JS APIs. | Free interpreter tooling; offline bundle required; story rights remain separate. | Technically possible, no benefit over native API for this server. [P1] |
| GUI interpreter + screenshots/OCR + key injection | Mature desktop interpreters; integration brittle and slow relative to direct text. Requires a display server or virtual display. | Pixels and keyboard events; semantic score/state must be parsed. | Software can be free, but adds dependencies; story rights unchanged. | Poor fit to a GUI-free Linux host. No reason to install for this task. |
| Existing research agents (DRRN/KG-based agents) on Jericho | Research implementations often depend on old ML stacks and external models. | Usually call the same Jericho API and construct candidate commands/knowledge graphs. | Code/checkpoint licenses vary; GPU or model downloads may be needed. | Read as strategy references; do not add a second agent runtime or paid API to the game interface. [R1, R2] |

## What the chosen API supplies

The following is the **expected established API contract to verify against the pinned installed release**, not a locally observed trace. Inspect the installed source and docs before binding it. Do not assume Gymnasium return shapes.

| Operation | Expected contract | Implementation consequence |
|---|---|---|
| Construct | `jericho.FrotzEnv(story_file, seed=vm_seed)` | Absolute local story path; explicit deterministic seed. If the selected release uses positional seed, adapt to its signature. |
| Reset | `env.reset()` → `(observation, info)` | Capture initial banner/prompt, native initial score and move count. `reset(seed=...)` is the evaluator wrapper's contract, not necessarily FrotzEnv's. Recreate the environment for a new game/seed. |
| Step | `env.step(command)` → `(observation, reward, done, info)` | Four returns, not Gymnasium's five. Reward is normally native score change, not cumulative score. Verify it against native score before using it. |
| Score/moves | `get_score()`, `get_max_score()`, `get_moves()` | Advanced game bindings must work for the exact ROM. `info` commonly also includes score and moves; verify keys, do not assume all versions match. |
| Terminal | `victory()`, `game_over()`; `done` from step | Keep win and game-over distinctions; validate on known traces. Some unsupported games lack reliable detection. |
| State persistence | `get_state()`, `set_state(state)` | Opaque snapshot/restore for internal inspection or explicitly allowed search. Do not serialize as JSON or assume cross-version portability. Verify RNG and parser context restoration. |
| Objects | `get_player_location()`, `get_inventory()`, `get_world_objects()`; interactive-object identification helpers such as `identify_interactive_objects()` | Verify optional helper names and signatures in the installed source. Some methods expose VM objects/IDs/attributes; exact return types vary by method/version. Build a JSON representation, never send Python objects directly. Whole-world access can reveal hidden information. |
| Candidate actions | `get_valid_actions()` and template action generation (`TemplateActionGenerator`) | Uses templates plus objects, often with execution-based change filtering. This is an instrumentation aid, not an exhaustive English grammar or proof of safety. |
| Change/hash | `world_changed()` / `get_world_state_hash()` where supported | Useful for detecting object changes and caching; not proof two complete states have identical future behavior. Timers, score, RNG, pending questions can differ. |
| Dictionary/walkthrough | `get_dictionary()`, `get_walkthrough()` where supported | Dictionary is vocabulary, not an action list. Walkthrough access is privileged: keep out of scored policy observations. It may be useful solely to validate a fixture if permitted. |
| Lifecycle | `close()` | Close prior VM on reset and release resources. Use process isolation for concurrent episodes rather than assuming native VM state is thread-safe. |

The core step/reset interface can work for stories whose scoring, object cleaning, or terminal detection is unsupported. Therefore “it prints an opening scene” is an insufficient admission test. Jericho's support is often tied to exact story hashes/builds and game-specific bindings. Check support indications available in the installed version, its binding lookup/source, and runtime warnings. Never reinterpret a default zero maximum or always-false terminal flag as valid instrumentation. [J2–J4]

### Baseline observation boundary

Recommended baseline is **human-visible text plus evaluator score and step metadata**. Expose a JSON object with:

- Manifest game ID and episode ID; initial text or last command response, verbatim.
- Last submitted action, native current score, native maximum, native score delta, native move count when reliable, evaluator step count.
- `terminated`, `truncated`, and a reason; separate verified win/loss flags when supported.
- Last *observed* room description and inventory, with the step on which each was observed; both can become stale.
- Agent-owned memory: explored graph, object facts with provenance, failed commands, unresolved puzzles, active plan, hazard observations.

Do not silently call `look` and `inventory` after each real action: those are parser commands and can advance time or change state. Prefer transcript-derived observations. If fresh auxiliary views are necessary, take an internal snapshot, issue the view commands on a clone/restored branch, then restore exactly; validate this is side-effect-free, and label the assisted observation mode. At a pending clarification prompt, `look` may answer the question incorrectly, so skip auxiliary commands entirely. Pure native getters such as score are preferable to entering `score` repeatedly.

Two optional modes need explicit, fixed benchmark policy:

1. **Action-assisted mode:** expose `get_valid_actions()` output. Helpful for reducing an enormous vocabulary, but it leaks affordances discovered by simulator trials. Record whether this is enabled consistently across games/seeds. Ensure it does not mutate the live state.
2. **State/search-assisted mode:** expose selected object facts and permit bounded snapshot search. Whole-world objects may reveal unseen rooms/objects; walkthroughs and arbitrary VM memory are a much stronger oracle. Do not call this text-only play. Keep official counters and consumed resources consistent with the declared search budget.

Use text-only sequential decisions as the portable minimum. For harness search, action assistance is a useful upgrade when supported: freeze and report the observation mode across a comparison, and retain an unrestricted ordinary-command channel because the generated action set is incomplete. This is a benchmark design choice, not a requirement to ask for user permission. Jericho remains useful for reliable native scoring even in text-only mode. [J2, J5]

### Actions, end conditions, and persistence

Accept one bounded-length text string per evaluator step; preserve quotes, punctuation, and case where the game needs them. Reject NULs/newlines and scripted multi-command submissions; do not strip punctuation blindly because spells, quoted speech, and parser answers can require it. The adapter should not provide an arbitrary shell or interpreter file path input.

Allow ordinary movement/manipulation, `look`, `inventory`, examination, and answers to parser clarification or yes/no prompts. Score queries are available from native metadata. Keep lifecycle commands (`restart`, `restore`, `save`, `quit`, `undo`, transcript-file commands) under evaluator control. Reject these without inventing an in-game score change; count repeated invalid submissions against the evaluator action budget to prevent unbounded loops. Game-specific vocabulary that overlaps a meta command needs an explicit allowlist decision.

Use `done` plus validated terminal hooks to terminate; death, victory, and “the game asks whether to restart” must not automatically reset. Freeze the final native score before cleanup. A configurable command cap (a proposed starting value is 1,000 submitted player inputs per episode) causes truncation, not defeat. Log native moves separately: parser errors and clarification replies need not increment native turns but must count toward evaluator interactions. A wall-time cap is infrastructure policy, not the game's decision deadline.

Save all commands, observations, native score changes, scenario IDs, binary hashes, VM seed, package/build versions, action-assistance mode, and termination reason. A replay using the same assets and seed should reproduce the transcript and score. Avoid untrusted pickle deserialization of snapshots. [J2, V1]

### Minimum adaptor contract for implementation

These are proposed adaptor operations, not additional Jericho methods:

| Adaptor operation | Required behavior |
|---|---|
| `reset(seed)` | Validate the seed and frozen manifest; choose game/VM seed using the mapping below; close the old environment; construct the new one and perform the release's documented reset sequence; capture the reset return without issuing a gameplay action. Return the initial observation and metadata. Confirm that constructor plus reset does not accidentally discard the requested seed. |
| `step(command)` | If the episode is already over, reject the call. Validate one bounded input, read the prior native score, submit exactly one parser line, then capture text, score, moves and terminal status. Increment the evaluator submission counter, including rejected submissions, and apply the cap. Return observation, score delta, terminated, truncated, and metadata according to the evaluator's wrapper format. |
| `close()` | Release native resources and flush transcript metadata. Do not submit `quit` and lose the final state while trying to close. |

A suggested serializable observation has fields `game_id`, `episode_seed`, `vm_seed`, `text`, `last_action`, `score`, `max_score`, `score_delta`, `native_moves`, `step_count`, `terminated`, `truncated`, `won`, `lost`, `termination_reason`, and `observation_mode`. The first observation has no last action and a zero transition delta. Use JSON null for unavailable facts; never coerce unknown moves or unknown win/loss into verified zero/false. `valid_actions`, if exposed, is an explicitly assisted field. Agent map/memory belongs to the harness and must not be presented as authoritative VM data.

At a normal step, the returned reward should equal current native score minus the previous native score. Treat a discrepancy as an instrumentation problem to diagnose. At a terminal transition capture the score before close/reset. If the final permitted action wins, report a terminal win rather than solely a step-limit truncation. A response such as “You can't go that way” is an ordinary game response, not a Python exception and not necessarily turn-free.

For native crashes, unsupported scoring, or parser output overflow, return a distinguishable environment error and retain the last successfully captured state in logs. Do not silently convert an infrastructure failure into a player loss or award fabricated final points. A bounded worker-process timeout protects against interpreter hangs; it is separate from Astra's decision time and should not count time while waiting for the next command. Use one process per live VM if concurrency is needed.

Known API/version pitfalls to inspect before implementation:

- Jericho's four-element step return differs from Gymnasium's five-element return. Termination and evaluator truncation must be separated by the wrapper.
- Game-specific score/terminal/object support can depend on the story hash. File extension, title, and a working opening screen do not prove support. Raw Z-code is the conservative input format; do not assume Blorb containers, Glulx, TADS, or other IF formats are accepted.
- Optional NLP helpers may need language resources even when core stepping does not. Test the actual imported package path; some releases may import optional-looking dependencies eagerly.
- Template generation and valid-action filtering can be much slower than an ordinary VM step and may use multiprocessing. Check serial execution options in the pinned implementation before adding process pools inside evaluator workers.
- NumPy/native dependency availability on Python 3.12, `pkg_resources` availability, removed `distutils`, and source-build requirements are compatibility checks, not established defects of an uninspected release. Avoid a speculative patch or an unrequested downgrade to another Python.
- A snapshot must restore RNG, timers, flags, and pending parser context, not just the object tree. Treat its representation as opaque and process/version-specific.

## Software and offline provisioning

Verified available on 2026-10-30: Linux server, evaluator Python 3.12.3, `/usr/bin/gcc`, `/usr/bin/make`, and `/usr/include/python3.12/Python.h`. Only pip (26.2.2) was listed in the evaluator venv. No story files or Jericho distributions were located in the study workspace; `vendor/` is empty. PyPI could not resolve; this does **not** establish that Jericho is unavailable for Python 3.12. The earlier draft also reported a failed pip download, but that command was not rerun during this revision.

Required provisioned assets:

- `jericho`, pinned to a tested release or commit; upstream project and PyPI metadata are the sources of truth. Do not assume current master, a historical wheel, or an old advertised Python minimum proves Python 3.12 compatibility.
- Its **actual locked transitive dependencies** from that selected release. NumPy and NLP/template-action dependencies such as NLTK occur in this ecosystem; inspect package metadata. If the chosen optional functions require spaCy or language data, bundle compatible packages/model resources explicitly. Do not unconditionally install a large NLP stack based on guesswork.
- Build tooling: pip, setuptools, wheel, a C compiler, make, and Python 3.12 headers as needed by the selected package. Jericho includes native interpreter code. Some releases may import `pkg_resources` or use older build APIs: pin a compatible setuptools build/runtime version if the source requires it; Python 3.12 removed stdlib `distutils`.
- A locally staged wheelhouse containing Python 3.12 / Linux x86_64-compatible wheels (including any native Jericho wheel), or complete source distributions and build dependencies. Never reuse a CPython 3.10 wheel as if it were CPython 3.12-compatible.
- A manifest of admissible game files, matching legal notices, and recorded hashes. Source-build tools such as Inform are optional only if legally building a compatible story from source; a recompiled binary may lose Jericho's exact-build support.

No GUI, GPU, paid service, account, game server, or API credential is required by the game interface. Foundation-model orchestration is outside this offline game-runtime setup.

### Provisioning procedure for the next phase

1. On a network-enabled staging machine with compatible Python 3.12/Linux x86_64, retrieve upstream source and inspect its license, setup metadata, supported-game bindings, and relevant API signatures. Select a concrete version/commit. Build wheels if PyPI has only source or incompatible wheels. Record versions and hashes, including build dependencies.
2. Resolve and lock all runtime dependencies. If action-generation helpers require external tokenizers, corpora, or model data, exercise those helpers on the staging machine and package their exact resources. Disable runtime downloads. A minimal text-step/score profile should avoid optional NLP helpers where possible.
3. Obtain only games whose **specific binary** may legally be redistributed, together with the permission text and any attribution/source obligations. Verify that those exact bytes have working Jericho bindings and positive native maximum score. See the candidate-source section below; no ROM is approved by this study.
4. Stage `vendor/wheels/`, `vendor/requirements.lock`, `games/manifest.json`, story binaries, and licenses on this machine. These are planned paths, not currently supplied artifacts. Keep game IDs and sorted manifest order stable across evaluation releases.
5. Install into the evaluator venv, not system Python. The intended offline command is:

   ```sh
   /home/hemlix/GamingMaster-mh-tier1/runs/venvs/jericho/bin/python -m pip install --no-index --find-links=/home/hemlix/GamingMaster-mh-tier1/runs/jericho/study/vendor/wheels --require-hashes -r /home/hemlix/GamingMaster-mh-tier1/runs/jericho/study/vendor/requirements.lock
   ```

   The lock must enumerate all transitive packages with hashes and all wheels must already exist. Prefer prebuilt wheels; source builds in an isolated offline environment also need their build requirements staged. The current study sandbox only permits writes in the study directory and `/tmp`, so it cannot modify this sibling venv. The adaptor/provisioning phase needs filesystem permission to install there; no installation was attempted outside the permitted roots.
6. Run `python -m pip check` with the full evaluator interpreter path, import Jericho, instantiate each approved story, and verify the API contract above. Build one short legal fixture trace per story and confirm native score changes and reproducibility. Do not infer compatibility from installation success alone.
7. Run the acceptance checklist below with network unavailable. Freeze the accepted manifest and dependency lock. If a story fails rights, scoring, or instrumentation checks, exclude it with a recorded reason; if no story passes, report provisioning blocked rather than substituting commercial Zork files or an unrelated game.

### Configuration (no secrets)

These are **proposed adapter variables**, not Jericho's built-in environment-variable API:

| Variable | Required? | Value to supply |
|---|---|---|
| `JERICHO_GAME_MANIFEST` | Yes, unless a manifest path is passed as an equivalent argument | Absolute path to the approved local JSON manifest; manifest entries contain relative story/license paths. |
| `JERICHO_MAX_STEPS` | Optional | Positive command cap, suggested initial default 1000. Freeze it for comparisons. |
| `JERICHO_OBSERVATION_MODE` | Optional | `text` by default; explicitly approved `actions` or `state_search` modes only. |
| `NLTK_DATA` | Only if the pinned optional helpers use NLTK data | Path to prepackaged local resources. Validate the exact corpora required rather than downloading all data. |

No credential values are needed or should be written into these files.

## Legal game corpus: concrete acquisition plan, unresolved approval

A public download, GitHub mirror, or academic ROM archive is **not** sufficient permission to redistribute. Jericho's software license does not license Zork, Infocom stories, or community games. “Abandonware” is not a license. Current upstream changes could alter a title's status, so inspect today's actual license rather than assuming old proprietary or new open-source status. [J1, Z3]

Start rights investigation with these concrete sources; each is a **candidate, not a verified approved asset**:

| Candidate | Where to look | Why investigate / required gate |
|---|---|---|
| Adventure / Colossal Cave, Z-machine port | https://www.ifarchive.org/indexes/if-archive/games/zcode/ and https://www.ifarchive.org/indexes/if-archive/games/source/ | Classic scoring adventure, commonly represented in IF benchmarks. Identify exact author/port/release, preserve permission notices for both original and port, then match Jericho support. Do not confuse the original game's status with every port's license. |
| Balances, by Graham Nelson | https://www.ifarchive.org/indexes/if-archive/games/zcode/ and https://www.ifarchive.org/indexes/if-archive/games/source/ | Small puzzle/score-oriented Inform game and plausible benchmark candidate. Verify explicit binary/source distribution terms and exact supported build. Source availability is not enough. |
| Other independently authored supported games | Jericho support data: https://github.com/microsoft/jericho/tree/master/jericho ; discovery: https://ifdb.org/ | Intersect Jericho's supported hashes with games having documented redistribution permission and usable scores. Do not admit a zero-score vignette merely to increase pool size. |
| Zork I/II/III | https://github.com/historicalsource/zork1 and documentation https://infodoc.plover.net/ | Useful strategy references; **do not ship by default**. Verify an explicit applicable license or authorized distribution grant for the chosen build. Historical promotional free downloads do not automatically grant permission to redistribute. |

If a release now has an appropriate open license, retain that license and verify its scope covers the story binary or the ability to build/distribute one. Rebuilding from licensed sources is an option, but can change hashes and score/terminal offsets; add and validate bindings transparently rather than pretending it is an existing supported ROM.

Manifest entry requirements: stable game ID/title, story-relative path, SHA-256, any native support hash (commonly MD5 for matching, not security), story release/version/checksum, source URL, license URL, local license text, redistribution rationale, pinned Jericho version/commit, maximum score, initial score, native scoring/terminal support status, timed-input exclusion, fixture validation result, and train/held-out split. This is an adaptor manifest specification, not a claim that such a manifest currently exists.

Operationally, build the corpus by intersecting (a) exact builds in the selected Jericho support data, (b) binaries whose distribution terms permit this offline bundle, and (c) games with validated positive score maxima and untimed input. Start with Adventure and Balances as investigations, not preapproved names. The number of approved games is currently **zero**; no precise ROM URL, checksum, or license grant was verified. This is a real provisioning blocker. Do not create a manifest full of placeholders and label it runnable. A historical academic game collection can help identify filenames/builds but cannot substitute for permission to redistribute each game.

## Reset and score semantics

Define a stable nonnegative integer evaluator seed `s`, a frozen ordered list of `N` approved games, game index `s mod N`, and positive VM seed `1 + (floor(s / N) mod 32767)`. This is a simple proposed mapping; verify the selected Jericho seed range and determinism before freezing it. Avoid VM seed zero if the interpreter treats it as nondeterministic. Different seeds select games and change random events within a game; they do not procedurally generate a new puzzle map. Log both seeds and the selected binary hash. Recreate the VM with that story and seed, then capture the ordinary initial state. Never apply unlogged random warm-up moves.

Held-out integers alone do not create held-out games. For meaningful generalization, partition the approved **game IDs** into training and held-out pools before mapping seeds; additionally hold out RNG seeds within games to test robust execution. Record the pool version. Tiny pools cannot support strong claims of novel-puzzle generalization. A single approved game is a working smoke test, not a varied benchmark.

For each admitted game with native maximum `M > 0`, define episode score `S_final / M`. Use the final native score at termination or truncation, not peak score and not a sum that loses negative rewards. Initial points remain part of the requested native-score ratio; record `S_initial` separately. If a game's legitimate native score can be negative, preserve the signed ratio unless the evaluator explicitly requires a documented clamped display metric. If score exceeds the approved maximum, flag a manifest/instrumentation error rather than silently changing the denominator. Zero/unknown maxima and nonscoring games are ineligible under the requested metric. Alternate progress metrics need a separate task agreement.

Use the arithmetic mean of episode normalized scores with balanced per-game sampling; also report per-game results. Do not sum raw points across games with different maxima. Win rate and steps-to-score are diagnostics, not replacements for native score. Reward deltas should telescope to `S_final - S_initial`; verify this invariant. [J2, J5]

## Acceptance checklist before evaluation

- With the exact evaluator Python 3.12.3, install/import works and native library loads; dependencies are pinned and `pip check` passes.
- Every shipped story has explicit redistribution evidence, preserved notices, a matching checksum, known positive maximum, and verified native score/move/terminal behavior.
- Same game/seed/actions reproduce the same responses and scores; distinct game-selection seeds exercise all manifest entries. Ignore no RNG initialization warnings.
- State snapshot/restore, if used, reproduces text, score, RNG-sensitive outcomes, and clarification prompts. Action enumeration and auxiliary inspection do not advance the live state.
- Invalid text and ambiguity prompts are handled without deadlock. Long outputs and pagination do not block waiting for an unseen keypress.
- Victory, death, and step-cap truncation preserve final score and never auto-restart; meta commands cannot rewind a scored episode.
- A long pause while awaiting the next command leaves the admitted game's state unchanged. Exclude wall-clock timed-input stories.
- Multiple simultaneous episodes, if needed, are isolated in processes and do not share native state. Clean close/reset does not leak resources.
- All tests work with network disabled; no corpus/model auto-download, accounts, GUI, or remote calls occur in the game runtime.

## References to verify

- [J1] https://github.com/microsoft/jericho
- [J2] https://jericho-py.readthedocs.io/en/latest/api.html
- [J3] https://github.com/microsoft/jericho/blob/master/jericho/jericho.py
- [J4] https://github.com/microsoft/jericho/tree/master/jericho
- [J5] https://arxiv.org/abs/1909.05398
- Package metadata: https://pypi.org/project/jericho/
- [F1] https://github.com/DavidGriffith/frotz
- [T1] https://github.com/microsoft/TextWorld
- [G1] https://github.com/erkyrath/remglk
- [P1] https://github.com/curiousdannii/parchment
- [R1] He et al., *Deep Reinforcement Learning with a Natural Language Action Space*: https://arxiv.org/abs/1511.04636
- [R2] Ammanabrolu and Hausknecht, *Graph Constrained Reinforcement Learning for Natural Language Action Spaces*: https://arxiv.org/abs/2001.08837
- [V1] https://inform-fiction.org/zmachine/standards/z1point1/sect15.html
- [Z3] https://github.com/historicalsource/zork1
