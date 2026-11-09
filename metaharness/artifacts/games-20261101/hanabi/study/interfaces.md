# Programmatic interfaces and offline adaptor specification

## Recommendation and evidence boundary

**Choose a local, headless, standard-library Python simulator of standard Hanabi, exposing the JSON contract below.** This is an equivalent simulator explicitly allowed by the task. It is the operational choice because this machine is offline, the evaluator venv contains no simulator packages, and no cached HLE distribution was located. It needs no account, network, GPU, GUI, browser, or extra Python package. Build it in the adaptor phase, not this study phase. Use HLE as an optional conformance oracle if a source bundle becomes available; do not make runtime depend on fetching it.

HLE would be the first choice if a compatible installation were already available: it avoids implementing rules and is a widely used research reference. Do not claim that the offline fallback is already implemented, tested, or a certified clone of a particular HLE release. Its complete proposed profile is specified here so the adaptor can be built without web access.

**Verification status:** all external URLs in this document are references from prior knowledge; DNS failures prevented live retrieval. No simulator import, build, or latency benchmark succeeded or was claimed. The evaluator at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/hanabi/bin/python` was directly verified as Python 3.12.3, linux-x86_64. It has pip but lacks `hanabi_learning_environment`, `pyspiel`, `pettingzoo`, `numpy`, `cffi`, `setuptools`, and `wheel`. Python headers, CMake 3.28.3, and g++ 13.3.0 exist. Current sandbox policy also permits writes only to the study directory and `/tmp`, not the evaluator venv; no installation was attempted. The chosen fallback requires no venv modification.

## Practical interface families

Latency descriptions are architectural expectations, **not measurements**. Every local option is turn-based and can wait indefinitely for the model.

| Interface | Maturity and observation/control | Latency and deployment | Cost, license/ToS considerations | Suitability |
|---|---|---|---|---|
| **Local Python equivalent simulator (chosen)** | New implementation requiring rules tests. Full control of hidden-state engine; expose only current-seat JSON and legal actions. | Expected sub-millisecond to millisecond simple steps; no network, native build, or GUI. | Zero service cost; use an original implementation and retain any licenses on reused code. No online-service ToS. | Guaranteed dependency availability on this Python; rules fidelity and masking are the main engineering work. |
| **DeepMind HLE Python API / C++ library** [I1] | Established research reference. Python `rl_env` supplies structured observations, vector encoding, legal moves and rewards; `pyhanabi` provides lower-level game/state access. C++ interface is also usable directly. | Local in-process native simulation, normally much faster than a model call. Older packaging/native dependencies need a Python 3.12 build check. | Free, repository historically Apache-2.0; verify bundled notices at chosen revision. No account/ToS for local use. | Best existing specialized backend if offline source/wheels are supplied. |
| **PettingZoo `hanabi_v5`** [I2] | Established multi-agent AEC wrapper around HLE; current agent, vector observation, action mask, integer actions, rewards/terminations. | Adds small Python overhead; depends on a working HLE install plus Gymnasium/NumPy and package-specific dependencies. | Free open source; check pinned wrapper and dependency licenses. No account locally. | Helpful for standard RL tooling; does not solve the HLE build issue and vector-only prompts are inconvenient. |
| **OpenSpiel `hanabi`** [I3] | Mature general game framework. `pyspiel` state, current player, legal action IDs, observation strings/tensors, chance nodes, terminal returns. Hanabi availability can depend on build configuration. | Fast local native engine; potentially substantial build/toolchain footprint. Must verify a Python 3.12 wheel includes the game. | Free, core Apache-2.0; dependencies carry their own notices. | Good alternate installed backend, but absent here; do not assume every wheel has Hanabi. |
| **Community bot clients, especially Hanab Live and H-Group tooling** [I4, I5] | Rich event logs, card-order IDs, clues, convention trackers; bots send protocol actions. Community conventions can be studied without running the client. | Remote WebSocket/network latency; self-hosting needs the server and its full dependency stack. | Usually free access/open source components; inspect each repo license. Public-service bot rules, accounts and automation approval must be checked, not assumed. | Online service violates this task's offline constraint; self-hosting is excessive compared with an engine. |
| **Other research bot repositories/frameworks** [I6] | Rule agents, convention agents, search and learned-policy examples. Often custom Java/C++/Python engines or HLE wrappers rather than a supported public API. | Local when dependencies/checkpoints are bundled; variable maintenance, Python compatibility and compute needs. | Usually no service fee; verify code/checkpoint licenses individually. | Useful strategy references or test policies, not a reason to import a large training stack for this task. |
| **Browser automation / DOM or WebSocket hooks on a web implementation** [I4] | Can observe rendered cards and inject clicks or speak an undocumented protocol. DOM/internal client state can reveal forbidden information. | Browser startup plus event synchronization; needs browser binaries; remote sites need network. | Site automation/ToS restrictions; account may be necessary. | No benefit over local simulation; offline copy would require vendoring all assets and services. |
| **Screen capture + input injection; desktop/tabletop game mods** | Generic fallback through screenshots, OCR/vision, mouse events, or mod/plugin scripts. Must find a specific implementation first. | Highest overhead; display server or virtual framebuffer, fragile card coordinates. | A commercial tabletop host can cost money; platform account and automation policies apply. | Incompatible with free/no-GUI goals when a clean simulator exists. |
| **Emulator memory hooks** | Only relevant for a specific console/desktop port; no canonical Hanabi emulator API identified. | Platform-specific reverse engineering and event input. | ROM/game license issues and possible acquisition cost. | No reason to choose this route for a card game with open simulators. |

## Software, credentials, and deployment

### Chosen dependency-free route

1. Use exactly `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/hanabi/bin/python` for validation and execution. Do not assess compatibility using system Python.
2. In the adaptor phase, implement the rules and contract in this document with Python 3.12 standard-library facilities (`random`, `json`, and ordinary collections/dataclasses are sufficient). No pip installation is required. Keep simulator internals separate from the policy-facing observation serializer.
3. Freeze the profile below and a simulator version string; record them in results. Run the conformance checklist before policy evaluation.
4. Use a local RNG per episode seeded on reset. Persist no private engine state into model prompts or shared policy memory. Execute no remote downloads or telemetry during games.
5. If HLE later becomes available, run differential rules tests and either retain the tested fallback or replace its engine while preserving the public contract. Document any discovered rule differences.

**Required credentials: none. Required environment variables: none.** The `env` list in `study.json` is empty. No server address, account, API key, renderer, display variable, or pretrained model is required for the game. Astra/Jev transport is supplied by the harness; it is not a Hanabi dependency or a reason for the simulator to access the network.

### Optional HLE installation when artifacts are provided

This is a contingency, not a successfully tested installation recipe. Obtain a pinned HLE source archive, license files, and its complete dependency wheelhouse outside the offline evaluation environment, then place them locally. Record the source commit/hash. A downloaded PyPI package alone may not include everything needed for a native build; inspect `setup.py`, `pyproject.toml` if present, and CMake files in that exact release.

HLE's historical Python binding uses **CFFI with a C/C++ shared library**. Expect `cffi` and its dependency `pycparser`; provision setuptools/wheel as required by the chosen build, plus NumPy if that release or wrapper imports it. Use CFFI and any other native dependency builds supporting CPython 3.12, not old wheels for 3.8/3.10. Native building needs CMake, a C++ compiler, make/Ninja as configured, Python headers, and possibly libffi development files. Do not assume that pip will resolve old packaging on 3.12: `distutils` was removed from the standard library.

When venv writes are available to the adaptor phase, use that venv's `python -m pip` with `--no-index --find-links <local-wheelhouse>` for dependencies, then install the pinned source with `--no-build-isolation` only after supplying its declared build requirements. A local prebuilt wheel matching CPython 3.12 and linux x86_64 is preferable. Run `pip check`, import `rl_env` and `pyhanabi`, construct/reset a game, inspect observation fields, and exercise all action categories. Do not repair an unavailable package by silently switching evaluator Python versions. If required artifacts are absent, use the chosen standard-library route.

## Fixed evaluation profile

Default to **2 seats** as an explicit study assumption, since the user did not specify a count. This produces meaningful cooperative hidden-information self-play with minimal seat-management overhead. Support 2–5 for separately labelled evaluation suites; never silently change player count between comparisons.

| Parameter | Exact value/behavior |
|---|---|
| Colors | `R`, `Y`, `G`, `W`, `B`; no rainbow/multicolor/special cards |
| Ranks in public adaptor | Integers **1–5**; engine mappings must be explicit |
| Copies per color | Rank 1: 3; ranks 2/3/4: 2 each; rank 5: 1 |
| Initial hands | 5 each at 2/3 seats; 4 each at 4/5 seats |
| Initial/capped clues | 8 |
| Initial lives | 3 |
| Initial player | Absolute seat 0; turns in increasing seat order modulo N |
| Discard at full clues | Illegal |
| Clue | One color or rank, another player, at least one match, all matches indicated; requires ≥1 token |
| Repeated clue | Legal if still truthful and nonempty; still spends a token |
| Final round | N further player actions after last draw, beginning with next seat and ending with the drawer |
| Early terminal | Lives reach zero or five stacks reach 5 |
| Failure score | Zero on exhausted lives; otherwise sum of heights |

### State and reset semantics

Keep private engine state: shuffled deck, all actual hands, per-seat clue constraints on every card, public discarded cards, five stack heights, clues, lives, current seat, monotonic public card IDs, turn count, and final-round counter. Deck order and RNG state are never observations. Use one standard 50-card multiset, shuffle with a local seeded RNG, and deal round-robin from a documented end of the shuffled list. Fixed draw/deal convention and fixed engine version make reset reproducible; **equal seeds need not produce equal deals across different engines**.

Within a hand, index 0 is oldest and the last index is newest. Removing a card shifts subsequent indices left; append the replacement at the end. Give each dealt/drawn card a unique opaque ID sequentially at deal/draw time, **not an ID encoding its identity or future deck position**. Clue knowledge follows the ID, never a stale index. New cards start with all 25 color/rank pairs possible by literal clue history. Public information later narrows beliefs independently.

`reset(seed)` discards all engine state, seat memories, convention commitments, previous model sessions, and episode-local search caches. It returns the current seat's observation. Keep seed only in evaluation metadata, **not the policy input**: knowing the shuffle seed or retrieving engine RNG state defeats hidden information. The harness should also avoid giving policy code a way to inspect private simulator fields.

### Recommended observation JSON

This is the **adaptor contract to implement**, not a claim that these names are native HLE fields. All values must be plain JSON; no live game/state objects.

| Field | Meaning |
|---|---|
| `rules_version`, `num_players`, `observer`, `current_player`, `turn` | Fixed profile ID, seat count, observer/current absolute seat, elapsed valid player actions. `observer == current_player` for nonterminal decisions. |
| `fireworks` | Map from five color letters to heights 0–5. |
| `information_tokens`, `life_tokens`, `deck_size` | Public counts, excluding cards already held. |
| `final_turns_remaining` | `null` before final draw, then N at the observation immediately following it, decreasing after each subsequent action. |
| `hands` | Array by absolute seat; each card carries `card_id`, current `index`, positive clue flags and literal candidate masks. Own cards have `color: null`, `rank: null`; other hands have actual color/rank. Never fill own truth merely because the policy might infer it. |
| `discard_pile` | Public color/rank cards, including failed plays; optionally also counts by identity. Successful plays are represented by stacks. |
| `history` | Ordered public action events: actor, action kind, target, clue attribute and touched IDs, or revealed played/discarded identity and outcome; public draw ID/drawer only. Never include the drawn identity in shared history. Current visible hands supply legal private visibility. |
| `legal_actions` | Complete, deterministic list of valid JSON action objects for this decision; empty after terminal. May include explicit clue `touches` as derived observation metadata, not an extra communication action. |
| `fireworks_sum`, `score`, `done`, `end_reason` | Progress total, benchmark score, terminal flag, and `null`/`perfect`/`lives_exhausted`/`deck_exhausted`. |

Literal clue candidates must capture both positive and negative information. On a color clue, touched target cards retain only that color; untouched target cards exclude it. Rank clues are analogous. This applies to cards in that hand **at clue time**, never replacement cards drawn later. These masks report literal clue consistency, not guaranteed remaining physical availability and not an unlabelled convention inference.

Seat-specific persistent memory may also retain that seat's own previous observations. **Do not concatenate observation logs from different seats.** A card in A's hidden hand is visible when the same agent acts for B; feeding B's private view back to A would make the benchmark effectively open-hand. Use separate Astra sessions per seat and episode, separate private scratchpads, and a shared store containing only public events and the preagreed convention definition. No shared chain-of-thought containing visible card identities. Apply the same rule to Jev inputs and search caches. If isolation cannot be enforced, explicitly label the result centralized/omniscient and do not compare it with standard Hanabi scores.

### Actions, stepping, and rejection

Exactly four action object forms:

* `{"type":"PLAY","card_index":0}`
* `{"type":"DISCARD","card_index":0}`
* `{"type":"REVEAL_COLOR","target":1,"color":"R"}`
* `{"type":"REVEAL_RANK","target":1,"rank":1}`

`target` is an **absolute seat ID**. Indices refer to the actor's current hand; do not accept a stale slot from a previous turn. Index validity determines play legality; the simulator must not leak hidden identity by omitting misplays from `legal_actions`. Discards similarly remain legal for any own card when tokens are below eight. Clue actions may enumerate only attributes present in the target's hand, which is already visible to the actor.

Apply a valid action atomically. For plays, test rank against the height, update stack or life/discard, and recover a clue on a successful 5. For discards, recover one clue. For clues, deduct one clue and update target constraints. If the game has not immediately ended, replace a played/discarded card when the deck permits; if that draw empties the deck, set the final counter to N and do **not** decrement it for the triggering turn. On each later valid action decrement once, including clues. Advance current seat after a nonterminal turn. At counter zero, end. Check exhausted lives before assigning terminal score. Ending immediately on zero lives or perfect stacks can omit a useless final replacement draw; make this convention explicit in tests.

`step(action)` returns the next current-seat observation, `reward`, `done`, and diagnostic `info` if the harness wants a Gym-like tuple. Define reward as change in **benchmark score**, so a bomb-out can produce a negative final reward. Evaluator score is the terminal `score`, not a count of positive rewards or a sum of repeated cumulative scores. Empty legal lists at terminal are normal. Reject an illegal action without state mutation or turn loss; return a structured error for correction and impose a harness retry limit. Never silently auto-clue, pass, or replace an invalid move with a random legal one unless a separately documented fallback policy is intentionally being tested.

## Using HLE instead: mapping guide

These are historically established HLE interfaces, **not locally verified against an installed release**. Inspect the pinned package before writing version-specific calls. [I1]

* Import path: `hanabi_learning_environment.rl_env`, with lower-level `hanabi_learning_environment.pyhanabi`. The familiar convenience constructor is `rl_env.make('Hanabi-Full', num_players=N)`; for deterministic seeds and explicit profile use `rl_env.HanabiEnv(config)` instead. Expected config keys include `players`, `colors`, `ranks`, `hand_size`, `max_information_tokens`, `max_life_tokens`, `seed`, `random_start_player`, and `observation_type`. Historically observation type **1** is ordinary card knowledge and **2** is seer; verify constants and select ordinary knowledge explicitly.
* Historical API: `reset()` returns an observation dictionary; `step(action)` returns `(observations, reward, done, info)`, not Gymnasium's five-tuple. Typically `observations['current_player']` is the acting absolute seat and `observations['player_observations']` contains one entry per player. Select **only** the current player's entry before serialization. Do not let the policy access the containing all-player object.
* Common per-seat fields: `current_player`, `current_player_offset`, `observed_hands`, `card_knowledge`, `discard_pile`, `fireworks`, `information_tokens`, `life_tokens`, `deck_size`, `legal_moves`, `legal_moves_as_int`, `vectorized`. Exact availability is version-dependent. Hands and knowledge are usually ordered **relative to the observer**, with index 0 self; convert offset `d` to absolute `(observer+d) % N`. Own card sentinels can be `None`/negative ranks: sanitize into explicit nulls.
* Structured native moves use `action_type`: `PLAY`/`DISCARD` with `card_index`; `REVEAL_COLOR` with `target_offset` and color; `REVEAL_RANK` with `target_offset` and rank. Target offsets are relative, not absolute. Convert `target_offset = (target - actor) mod N`, which must be nonzero. Historical HLE card ranks are **0–4**, while stack heights are **0–5**; convert rank fields by ±1 at the boundary. Use engine color utilities rather than guessing integer color order.
* Obtain native legal move dictionaries/IDs from the current observation; do not assume an action-ID layout. If a wrapper exposes only an action mask, keep an explicit verified bidirectional map. Flat `card_knowledge` fields may show only directly known attributes; do not assume they encode all negative clues. Reconstruct constraints from the public transcript or inspect the lower-level knowledge API.
* The RL wrapper normally resolves chance deals/draws internally. Lower-level `pyhanabi` access may require dealing chance events until the next player. Do not ask Astra to choose chance outcomes. Reconstruct the environment with a seeded configuration for each reset if the pinned API lacks `reset(seed=...)`; do not assume reseeding Python's global RNG controls the C++ generator.
* Derive `final_turns_remaining` with a public wrapper counter when absent: detect the first transition from positive `deck_size` to zero, set N then decrement on later decisions. Track own stable IDs externally if the engine exposes only slots.
* Verify score on life exhaustion against `hanabi_state.cc`; do not substitute stack sum for benchmark failure score. Never pass the internal `state`, full deck, seer observations or seed to the policy.

### Other wrapper details

PettingZoo normally exposes `hanabi_v5.env(...)`, AEC `agent_iter()`, `last()`, and `step(action)`, with a dict containing `observation` and `action_mask`. Do not use the simultaneous Parallel API for a sequential game. Dead/terminated agents may require `step(None)` under the AEC contract; that is wrapper cleanup, not a Hanabi pass. Arguments, reward placement and version suffix must be checked in the installed version. [I2]

For OpenSpiel, first verify `'hanabi'` is among registered games. Generic entry points include `pyspiel.load_game`, `new_initial_state`, `current_player`, `legal_actions`, `apply_action`, `is_chance_node`, `chance_outcomes`, `is_terminal`, and `returns`. Sample chance outcomes with the seeded private engine RNG; do not maximize over them. Use a player-specific observation/information-state interface supported by that game, not `str(state)` as an assumed-safe observation: state debug strings may expose hidden cards. Check whether terminal returns match the requested score and normalize only through an explicitly documented score mapping. [I3]

## Required conformance checks in the adaptor phase

These are tests to implement later, not game-playing code included in this study.

1. **Deck and dealing:** 50 cards with exact multiplicities; initial deck sizes 40/35/34/30 for 2/3/4/5 seats; eight clues, three lives, zero stacks; no card duplication/loss.
2. **Hidden-information boundary:** own truth/seed/deck absent from serialized state; other hands visible; no unsafe simulator objects; separate per-seat memories. Two legal underlying states differing only in the actor's hidden cards/deck must have identical observations when their literal knowledge and public/visible facts agree. Legal PLAY/DISCARD lists must not distinguish those states.
3. **Clues:** all and only matches touched; untouched negative exclusions; no nonexistent attribute, self clue or zero-token clue; repeat clue legal; clue does not draw. Drawn replacements do not inherit earlier clue masks.
4. **Play:** next rank succeeds; skipped/duplicate rank loses a life and enters discards; correct 5 recovers at most one clue; third failed play ends at score zero; a misplay is legal before its outcome is known.
5. **Discard/indexing:** no discard at eight tokens; otherwise increments once; shift slots and append replacement; stable IDs and knowledge follow survivors; no replacement after empty deck.
6. **Final round:** after last draw exactly N further valid actions absent early termination, including the original drawer; clues also decrement; trigger turn does not decrement; shrinking hands have correct legal indices.
7. **Terminal and score:** perfect game ends at 25, deck exhaustion reports stack sum, fuse-out reports zero despite nonzero stacks. Terminal has no actionable turn; repeated observe is read-only; illegal actions and terminal steps cannot mutate state.
8. **Determinism:** same engine version/profile/seed/action sequence reproduces observations and score; different seeds vary deals; reset clears memories; inspect final engine transcript only in test/debug context.
9. **Backend comparison when available:** use fixed explicitly constructed fixtures for stack/clue/final-round transitions rather than expecting different RNG implementations to agree on seed-to-deal mapping. Assert the selected profile before using HLE/PettingZoo/OpenSpiel results interchangeably.

## Scenario generation

Primary evaluation: fresh uniformly shuffled standard decks, fixed two-seat profile, fixed starting seat, shared policy with isolated seat information, and disjoint training/search versus held-out seed lists. All seats cooperate; there are no opponents to randomize. Do not choose favorable seeds or tune from held-out transcripts. Evaluate enough deals to reveal score variance and bomb-outs. Report mean raw score and per-seed scores; perfect rate and bomb-out rate are secondary diagnostics. A separate extension suite may vary 3/4/5 players, labelled and stratified by player count. Do not mix rainbow, random rule changes, or easier open-hand states into the standard 0–25 result. Scripted near-endgame states are unit tests, not substitute benchmark episodes.

## Source references

* **I1:** https://github.com/google-deepmind/hanabi-learning-environment ; legacy https://github.com/deepmind/hanabi-learning-environment ; package metadata reference https://pypi.org/project/hanabi-learning-environment/ . Inspect `README.md`, `setup.py`, `hanabi_learning_environment/rl_env.py`, `hanabi_learning_environment/pyhanabi.py`, `hanabi_lib/hanabi_state.cc`, and license in a pinned archive.
* **I2:** https://pettingzoo.farama.org/environments/classic/hanabi/ and https://github.com/Farama-Foundation/PettingZoo . API/version and dependency documentation.
* **I3:** https://github.com/google-deepmind/open_spiel ; https://openspiel.readthedocs.io/ ; https://pypi.org/project/open-spiel/ . Generic engine API and build availability.
* **I4:** https://hanab.live/ and https://github.com/Hanabi-Live/hanabi-live . Browser/community server implementation; no online use proposed.
* **I5:** https://hanabi.github.io/ . H-Group community conventions and links to tooling; conventions are not game rules.
* **I6:** https://arxiv.org/abs/1902.00506 . Hanabi benchmark, representative agents, self-play and ad-hoc evaluation.
* Rules cross-reference: https://www.ultraboardgames.com/hanabi/game-rules.php . Live retrieval of all references remains outstanding due to network restrictions.
