# TextArena interface and offline adaptor specification

## Recommendation and evidence

Choose the **official local Python environment API**, with a thin per-game compatibility layer and a player-specific text/history adaptor. It directly exposes turns and outcomes, runs headless, and avoids network play, screenshot parsing, and input timing. Preserve the native game rules and select games through a frozen manifest.

Target interpreter: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena/bin/python`, verified as **3.12.3, linux-x86_64**. TextArena is **not installed**; only pip is installed. Neither a package version nor a game/opponent manifest was supplied. Network requests failed with `Could not resolve host`. No cached TextArena wheel/source was found in the searched workspace and pip/uv caches. The reference API below is prior knowledge of the public TextArena interface, **not a successful runtime verification of a release**. Treat its version-sensitive points as setup gates.

Official references: [repository and examples](https://github.com/LeonGuertler/TextArena), [PyPI metadata](https://pypi.org/project/textarena/), [environment sources](https://github.com/LeonGuertler/TextArena/tree/main/textarena/envs), [wrappers](https://github.com/LeonGuertler/TextArena/tree/main/textarena/wrappers), [agents](https://github.com/LeonGuertler/TextArena/tree/main/textarena/agents). These URLs were not successfully retrieved in this session. In particular, “latest”, Python compatibility, and a particular dependency set are not verified claims.

## Practical interface choices

Latency descriptions are architectural expectations, not benchmarks measured here.

| Interface | Maturity / exposure | Latency | Cost and legal/ToS considerations | Suitability |
| --- | --- | --- | --- | --- |
| Official local `textarena` environment | Primary intended programmatic path: player observations, string actions, terminal status, native rewards; per-game implementation details vary | In-process call overhead is low; game generation/search may dominate | Local execution has no per-turn service fee. Project is commonly distributed as MIT; verify the acquired release's LICENSE and asset notices | **Chosen**, subject to a pinned release and offline audit |
| Official observation/action wrappers | Useful convenience layer over the same API; may aggregate messages or transform output | Local formatting overhead | Same package; no credentials inherently required | Use only wrappers whose history and transformation behavior has been verified |
| Direct game classes / internal state | Source offers rich boards, counters, state and helpers, but internal layouts are unstable and may include secrets | Low local overhead | Same source license; exposing hidden state would invalidate fair evaluation | Useful for evaluator scoring and validating public moves; not a player-facing omniscient interface |
| Supplied `ta.agents` classes | Agents consume observations; these are policies, not a separate environment interface. Availability of a generic legal random bot is **unverified** | Human input blocks; local inference varies; hosted LLMs add network/model latency | Hosted classes can require API keys and paid services; optional local weights have separate licenses/costs | Inventory source before calling any class a free offline baseline |
| Custom local process / JSON transport around official env | Not a separately verified upstream service; adaptor may isolate one env per worker | Adds small IPC/serialization cost | Free local software | Good if isolation is needed; retain the official turn semantics |
| Official online arena / `make_online` family | Remote matchmaking/model submission exists in the public project; exact current protocol, account fields and API stability unverified | Network, queue, server deadline and opponent latency | Requires Internet and potentially registration/model credentials; service terms and rate limits apply | Excluded by offline requirement |
| Human terminal interface / rendering | Can display textual boards and accept manual strings | Input and terminal parsing overhead | Local and usually free | Debugging only; unnecessary for an automated policy |
| Browser/screen capture plus injected keys | No need for this when the Python API exists; no specific supported desktop hook verified | Rendering, OCR and input latency | Remote site would add service/account constraints | No GUI on server; inferior to the native interface |
| Gymnasium/PettingZoo adapter, mods, emulator hooks | No compatible upstream adapter or emulator integration was verified | Depends on added translation | Third-party package terms/dependencies vary | Do not assume standard Gym reset/step signatures or buy/install a GUI stack |

## Software, credentials, and offline installation

Required runtime software is a **version-pinned TextArena distribution plus its declared dependency closure**, installed into the evaluator venv. The Python interpreter already exists. A source checkout is useful for audit, but the installed release must match it. No account, game server, API key, display server, browser, or environment variable is required for the chosen local path. `study.json.interface.env` is therefore empty. Never configure a remote model wrapper merely to obtain a baseline opponent.

Do not assume every catalogue entry is offline just because the package imports. Audit imports, reset, stepping and scoring for HTTP calls, downloads, hosted judges, API-based opponents, and datasets loaded on demand. Word games may need dictionaries/corpora; chess-like games may need optional libraries; semantic games may invoke a judge. These are conditional dependencies, not a verified package requirements list. If a selected game requires network resources, stage licensed assets before offline evaluation or exclude it **before the manifest is frozen**. Required manifest entries must not disappear silently at runtime.

Installation handoff, once package access is available:

1. Obtain the evaluator's version and manifest if predefined. Otherwise select a release from PyPI with compatible `Requires-Python`, inspect its metadata and license, and freeze its version and wheel/source SHA-256. Do not invent a version pin from this report.
2. On a connected Linux x86_64 Python 3.12 staging machine, resolve that release and its full dependencies into a wheelhouse. Prefer wheels; build and retain wheels for source-only dependencies on a compatible system. Include any audited game assets and their license notices. This is preparation, not a runtime Internet requirement.
3. Produce a fully pinned, hashed `requirements.lock`, package hashes, and an asset manifest. Verify the dependency closure is complete, including optional imports used by the selected games. Metadata compatibility alone does not prove Python 3.12 runtime compatibility.
4. Install offline with the **evaluator Python**, not system pip:

   `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena/bin/python -m pip install --no-index --find-links ./wheelhouse --require-hashes -r requirements.lock`

   Then run that interpreter's `-m pip check`, verify the imported module path and distribution version, and record them. The wheelhouse and lock file are future setup artifacts; neither exists as a deliverable of this study.
5. Inventory the installed registry and source. Record which IDs are constructible, their player counts, constructor parameters, local assets, and optional dependencies. Match each game to an observation parser, action serializer, scoring extractor, baseline policy, and termination rule.
6. Run the acceptance checks below with network unavailable and no display. Only then freeze the manifest and package lock for held-out evaluation.

No install was attempted into an arbitrary alternative interpreter. Installation could not be completed here: the package is absent, downloads are unavailable, and this session's writable roots are the study directory and `/tmp`, not the evaluator venv. The next phase needs both package artifacts and write access to the specified venv; switching to system Python would not meet the requirement.

## Reference API contract

The established public usage pattern is:

| Operation | Reference call / expected meaning | Verification required |
| --- | --- | --- |
| Import | `import textarena as ta` | Check installed version and import side effects |
| Construct | `ta.make(env_id="TicTacToe-v0")` | Example ID, not a verified installed ID; inspect available registry and kwargs |
| Optional text wrapper | `ta.wrappers.LLMObservationWrapper(env=env)` | Confirm class/export and whether it accumulates history |
| Start | `env.reset(num_players=N, seed=seed)` | Confirm signature, accepted seed type, player counts, and whether reset returns a value |
| Observe | `player_id, observation = env.get_observation()` | Confirm observation type and consumption/history behavior |
| Act | `done, info = env.step(action=action_string)` | Expected TextArena two-value pattern; do not assume a Gym five-tuple |
| Finish | `rewards = env.close()` in familiar examples | Confirm return type and where terminal outcome/game score actually lives; cache final data before cleanup |

The core may also expose `env.state`, pending observations, current player, game state, reward dictionaries, and logs. These are implementation details, not a stable public JSON schema. Some versions may change state subclasses or terminal-return shapes. Bind one explicit compatibility adapter to the installed release, asserting expected types at startup. Do not guess at tuple meanings in the middle of an evaluation episode.

Raw observations have historically been message sequences with sender IDs and text; an LLM wrapper presents a text history. Confirm this against the installed release. **Observation reads may drain pending messages or update wrapper history**: read once per turn, cache, and reuse that cached value for logging/model calls. Repeated reads should not be used as a polling loop. Reset all per-player histories between episodes. An initial rules message is part of the observation and must survive history compression.

`step` does not select an arbitrary player by an action-dictionary key. Submit the action for the current player. Never drive the two-player game by a fixed `0,1,0,1` loop: invalid retries, phase changes and extra turns can violate that pattern. Native rewards may only become available at termination; no dense reward should be invented from message sentiment.

## Adaptor behavior, sufficient for implementation once artifacts are supplied

### Episode manifest

For every selected entry store: exact game ID and release, constructor kwargs/variant, number of players, evaluation weight, learner seat policy, baseline name/version/config, observation mode, grammar/range conventions, maximum actions, score formula, asset hashes, and whether a local rules model is allowed. Distinguish **native environment rules** from any wrapper action budget or reward mapping.

Candidate minimal mix, only if the evaluator has not prescribed one: tic-tac-toe and Connect Four for two-player public boards; Sudoku and Mastermind for single-player constraints/inference. Common reference IDs are `TicTacToe-v0`, `ConnectFour-v0`, `Sudoku-v0`, `Mastermind-v0`; existence, player count and configuration need registry confirmation. Expansion to Othello, Nim, Battleship, Wordle or a sliding puzzle should follow the same audit. This is a proposal, not a claim about the hidden evaluator mix.

### Reset and reproducibility

Use the external episode seed to choose a manifest entry, variant/puzzle seed, learner seat, and opponent variant through a stable documented mapping. Use fixed arithmetic or a stable digest, not Python's randomized `hash()`. Keep independent deterministic random streams for environment, opponent and scenario selection. Some environments may seed Python's global RNG, seed too late, use NumPy separately, or create randomness in their constructor; audit and test instead of assuming `reset(seed)` is sufficient.

A simple deterministic game schedule can cycle through manifest indices by `seed modulo number_of_games`, with other seed bits or a stable digest supplying the puzzle and seat seeds. Evaluation should use stratified schedules or paired seats rather than trust a tiny random sample to balance games. Empty-board deterministic games offer little start diversity: vary opponent policy/seed and seat; use legal seeded opening prefixes only if the evaluator permits them and records them as scenario settings. Do not vary rules invisibly.

Use separate worker processes if global RNG or mutable module state would couple games. Do not expose seed-derived puzzle answers, secret-generation streams, or private initial state to the policy. An opaque episode identifier is sufficient for player-side memory.

### Turn controller

1. Construct/reset the selected environment and reset baseline and learner memory.
2. Fetch and cache exactly one observation for the active player. Retain roles/sender IDs and separate all player histories.
3. If it is a baseline's turn, pass **only that player's permitted observation** to its local policy and step its action. Repeat until the learner is active or the game terminates. Cap this internal loop so a buggy baseline cannot hang the adaptor.
4. Present the learner with the cached raw text plus any reliably parsed public state and legal actions. Never present a raw dump of `env.state.game_state`: it can contain the target solution, opponent hand, ship placement, or deck order.
5. Accept one learner action, validate its type, size and known grammar, serialize it exactly, and call native `step` once. Re-fetch active player/status; do not infer that a turn advanced.
6. If the game ended, capture authoritative native results and metric fields before cleanup, normalize through the manifest's scoring function, and return a terminal record. Otherwise advance baseline turns and return the next learner observation.

Validation must have defined semantics. Prefer deterministic legality checks or provided legal-move lists. Do not “test” candidate actions by submitting them to the live environment. Local search should use a separate public-state rules model if permitted, not mutate and undo an unsnapshotable game or inspect hidden state. If invalid actions are rejected before `step`, count them and enforce a fixed retry budget; otherwise the wrapper gives unlimited free turns. A parser failure should fall back to the raw prompt, not manufacture a default board or secret.

### Player-facing data contract (proposed, not native TextArena fields)

Expose a JSON object with `game_id`, `variant`, `player_id`, `rules_text`, `observation_text`, `public_state`, `self_private_state`, `legal_actions`, `remaining_moves`, and `history_summary`. Set unavailable fields to null/omit them; never imply a complete action list when only a shortlist is known. Keep enough recent verbatim feedback to audit parsing and preserve original coordinate labels. The policy need not see the evaluator seed, scoring-only internal data, opponent RNG, or future draws.

In most games, return a single native action string. For games accepting free text, do not globally strip prose or all brackets: exact per-game syntax governs. For structured moves, keep explanation separate from the emitted command; avoid multiple bracketed alternatives that the parser could interpret as extra moves. Formatting helpers must not change the chosen move silently.

### Offline opponents

“Baseline agents” in the user notes does not identify an upstream baseline class. Inspect the supplied agent module; many convenience agents in LLM game suites are remote model clients, not offline bots. Prefer evaluator-provided baseline policies. If none are supplied, define and freeze transparent local baselines during the adaptor phase: legal uniform random and/or a modest fixed heuristic policy. Do not pretend such custom baselines are supplied by TextArena.

Each baseline needs legal move generation for its particular game and the same information restrictions as the learner. A uniform random string generator is not a legal random policy. Do not default to an optimal solver as the baseline without declaring the difficulty: solved two-player games can make wins impossible from some seats. Pin baseline randomness and prevent it from consuming the environment's RNG stream. No opponent implementation is included in this study.

### Scores and external budgets

For win/loss games extract outcome directly, then apply the frozen win/draw/loss mapping. For single-player games distinguish binary completion from numeric score; use normalized numeric score only when a defined measure exists. Retain raw rewards and the chosen formula. Do not assume `close()` always returns a dictionary with integer keys or that reward 0 means failure.

Track native terminal, invalid-action forfeit, wrapper action cap, external timeout and infrastructure exception separately. Decide their score treatment before evaluation. Record per-game and overall weighted scores, learner seat, opponent and episode counts. Local turn waiting has no known deadline; provisionally permit minute-scale Astra calls, but measure total episode cost against the evaluator's unknown outer timeout. `decision_latency_s=-1.0` in the JSON denotes an unbounded native wait, not an evaluator guarantee.

## Acceptance checks for the adaptor phase

* Import and construct every frozen entry under the exact Python 3.12.3 venv with no network or display; verify optional assets are already local.
* Complete at least one natural terminal episode per game and verify scalar/dictionary/tuple return shapes, player IDs, rewards, and score normalization against known outcomes. Check wins, losses, draws where possible, and every single-player metric's extrema.
* Verify first observation contains rules, observations do not leak secrets, cached reads preserve messages, and wrappers do not duplicate histories. Verify learner seat 1 and opponent-first games.
* Use documented examples to validate coordinate bases, pass syntax, guesses, promotions and other special actions. Check invalid syntax, invalid legality, retry/forfeit behavior and action-cap handling in disposable episodes.
* Replay identical seeds with identical action traces: observations, outcomes and baseline actions should match. Verify different seeds actually vary scenarios and do not perturb unrelated episodes.
* Audit timer code and compare state before/after an idle interval **without stepping**; this supports tier 1 for the selected games. Separately measure reset and step costs; no latency measurements were obtained in this study.
* Freeze a release/manifest/asset checksum record and keep terminal logs free of credentials and hidden information in the player channel.

The remaining blockers are concrete: acquire a compatible pinned package and dependencies, install in the specified venv, identify or freeze the game/opponent manifest, and verify the reference contract. There is no evidence of a need for a paid service, GUI, emulator, or account for the chosen local design.
