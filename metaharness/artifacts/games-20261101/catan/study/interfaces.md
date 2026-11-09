# CATAN interfaces and adaptor plan

This document assumes the standard base game and a local, offline evaluator. The references are linked in [`sources.md`](sources.md). In particular, [Catanatron](https://github.com/bcollazo/catanatron) is the preferred candidate, but the session could not resolve DNS, so its current release, exact method names, and Python 3.12 wheel availability still have to be checked in the adaptor build. Pin the inspected commit/release; do not silently track `main`.

## Choice

Use the **Catanatron Python engine directly** (a bot-library/core-engine interface), with a small adaptor that emits a filtered JSON observation and maps a JSON action ID back to an engine-owned action. It is the best fit for a Python evaluator; JSettlers2 and Pioneers are also plausible free headless engines but require separate processes and protocol integration. It avoids browser timing and image recognition and lets the adaptor ask for legal actions instead of guessing them. A Gym/Gymnasium wrapper can be added after the direct engine works, but it may alter decision granularity (discarding, robber victim, and initial placement) and its package/version contract is unverified.

The exact Catanatron symbols are release-sensitive. Historical APIs expose a `Game`/state object, player colors, typed actions, a legal-action list, and a way to apply one action. The lookup guide below is prior knowledge, **not a verified API contract for an installed version**. Before writing the adaptor, obtain one pinned version and inspect its source, then run a smoke probe that (1) constructs a four-player game with a fixed seed, (2) serializes the initial state, (3) enumerates legal actions, (4) applies one legal action, and (5) reaches a terminal state against built-in bots. If a symbol differs, use the pinned source's API rather than adding a compatibility layer based on guesses.

## Candidate interfaces

| Interface | Observation/control | Maturity and latency | Legal/cost risk | Decision |
|---|---|---|---|---|
| **Catanatron core Python engine** ([C1](https://github.com/bcollazo/catanatron), [C2](https://catanatron.readthedocs.io/)) | In-process board/state objects and typed legal actions; bots and seeded games | Purpose-built bot engine; local calls should be fast but neither action nor bot latency was measured | Free/open-source candidate; verify the pinned license and base-game fidelity | **Choose** |
| Catanatron Gym/Gymnasium wrapper ([C1](https://github.com/bcollazo/catanatron)) | `reset`/`step` observations and action integers or dictionaries, depending on release | Convenient RL loop, but wrapper may flatten or hide multi-step decisions; local latency | Free candidate; Gym/Gymnasium and NumPy version compatibility must be checked | Secondary, only after core probe |
| JSettlers2 ([A1](https://github.com/jdmonin/JSettlers2)) | Java server/client protocol, robot players, text/network messages | Mature open-source implementation; process/protocol latency is low but integration is substantial | Free; check GPL obligations and whether robot protocol exposes private/public state correctly | Fallback if Catanatron cannot install |
| Pioneers ([A2](https://pioneers.sourceforge.net/)) | C client/server, computer opponents, GUI-oriented protocol | Older project; headless setup and current build tooling are uncertain | Free; license/build and rule-variant compatibility need checking | Fallback only |
| OpenSpiel ([A3](https://github.com/google-deepmind/open_spiel)) or PettingZoo ([A4](https://pettingzoo.farama.org/environments/)) | Excellent generic multi-agent APIs | No standard CATAN environment is established by this study; adding one would be new game code | Free, but implementing and validating CATAN is out of scope | Do not select |
| Browser/online CATAN (Colonist, CATAN Universe) ([A5](https://colonist.io/terms-of-service), [A6](https://catanuniverse.com/)) | DOM/WebSocket or screen/input automation | Network and browser latency; accounts and UI changes are external state | Terms, anti-cheat, account, and network risks; not offline | Do not use |
| Tabletop sandbox scripting / emulator hooks | Lua or memory/input hooks could expose objects or automate a commercial digital edition | Host-specific integration; often lacks rule enforcement and built-in CATAN opponents; no suitable target verified | Commercial host/game assets may cost money or require licenses; unsupported binary hooks are fragile | Does not satisfy the free headless simulator requirement |
| Screen capture plus mouse/keyboard injection | Screenshot, OCR/CV, `xdotool`/uinput-like controls | Highest latency and brittle; GUI is unavailable on the target server | Online ToS and display-server dependencies; free tools do not solve fidelity | Last-resort debugging only |

No official public CATAN research API, supported bot plugin, or emulator environment has been verified. That is an evidence limit, not a claim that none exists. A commercial game's private network protocol is not an official supported API. General frameworks only become useful when a rules-complete CATAN implementation is available. Local free-software engines have no online account/anti-cheat service in the selected setup; preserve their licenses when redistributing source or binaries. All latency descriptions above are qualitative, not measurements.

## Software and installation

The target interpreter is `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/catan/bin/python` (Python 3.12.3, Linux x86_64). Install and test only with that interpreter. With network access during provisioning, the adaptor phase should inspect the package metadata, select a release supporting Python 3.12, record the exact version and commit, and run the smoke probe. For an already prepared offline machine, put the pinned wheel and all transitive wheels in a local wheelhouse and install with pip's offline mode:

```text
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/catan/bin/python -m pip install --no-index --find-links "$CATAN_WHEELHOUSE" -r requirements.lock
```

This is a **future provisioning command**: neither `requirements.lock` nor a wheelhouse exists in this study. The provisioner must create the lock with an exact distribution name/version and exact transitive dependencies. Inspect both [catanatron metadata](https://pypi.org/pypi/catanatron/json) and the historical [catanatron-core distribution](https://pypi.org/project/catanatron-core/); older source trees split `catanatron_core`, experimental players, and Gym into separate projects. An import named `catanatron` does not by itself identify the distribution. Do not install competing core distributions into the same venv.

`CATAN_WHEELHOUSE` is a path supplied by the provisioning system, not a secret. Follow [pip's offline installation guidance](https://pip.pypa.io/en/stable/cli/pip_install/) and [wheel download guidance](https://pip.pypa.io/en/stable/cli/pip_download/). Acquire/build artifacts on a connected Linux x86_64 Python 3.12 environment, including build requirements if no compatible wheel exists. Preserve hashes and the license with the artifacts. The package may require NumPy, NetworkX, Gym/Gymnasium, or other dependencies; use inspected metadata, then record `pip freeze` and run `pip check` with the target interpreter. Avoid GUI, web, training, or GPU extras unless an included opponent requires them. A README claiming Python 3 support is insufficient evidence for 3.12: older NumPy/Gym pins may fail on this interpreter. Do not change the evaluator to another Python to make an old dependency work. No server address, account, or API key is required for a local run. No simulator is installed now; the venv currently contains only pip. This session's writable roots do not include the venv, so provisioning requires a different execution context with that write access.

Required setup gates:

1. Verify the venv path, Python version, platform, package version, and license.
2. Install the pinned Catanatron distribution and declared dependencies from the wheelhouse.
3. Locate the public game/state/action classes in the pinned source and verify construction, seeding, legal-action enumeration, action application, and terminal/winner reporting.
4. Confirm the engine's rule switches: base game versus expansions, number-token/map randomization, initial snake placement, trade availability, development-card timing, seven-discard handling, robber victim selection, awards, and hidden information.
5. Build a player-view serializer. It must expose the agent's own resources and development-card identities, but only opponent resource counts and development-card counts; never pass the full engine object or hidden deck order to Astra/Jev.
6. Add deterministic `reset(seed, opponents, options)` and a hard episode cap. Save the seed, engine version, map, opponent policies, winner, true VP, public VP, turn count, and termination reason.
7. Run fixed-seed games against each included bot and a random policy. Check invariants after every action: resource conservation, piece counts, distance rule, connectivity, one-roll/one-development-card timing, and award ownership.

## Engine API lookup guide and control loop

Use the upstream [repository](https://github.com/bcollazo/catanatron) and locally search its files with `rg --files` and `rg` for the following symbols. Paths may start at `catanatron/` or under a historical `catanatron_core/catanatron/` directory. These are **candidate historical names to inspect**, not tested imports:

| Need | Candidate names / locations | Verify before using |
|---|---|---|
| Game construction | `Game` in `catanatron.game`, sometimes re-exported at package root; `Game(players, seed=...)` | Argument signature, standard map selection, target VP, whether player order is shuffled |
| Player callback | `Player`, `Color`, `RandomPlayer` in `catanatron.models.player` or package root; `decide(self, game, playable_actions)` | Callback is synchronous; it must return one offered engine action object |
| Decision step | `Game.play_tick`, `Game.execute`, `Game.play` | Whether a tick calls a player, whether execute validates legality, and which operation advances only one decision; `play()` can run a whole game |
| Legal action list | `game.state.playable_actions`; action-generation code | It may be a list attribute, not a `playable_actions()` method. Determine who owns the pending decision |
| Action representation | `Action`, `ActionType` in `catanatron.models.enums`; fields such as `color`, `action_type`, `value` | Resource/edge/robber tuple layout; random outcomes may appear in executed action values |
| State / active player | `game.state`, `current_color()`, `current_player_index`, `current_prompt` | Pending discarder may differ from turn owner; do not assume the same index controls all phases |
| Board / map | `state.board`, `state.board.map`, `land_tiles`, node and edge structures | Stable topology IDs, node adjacency to tiles/edges/ports, desert and coast treatment |
| Player state / scores | `state.player_state`, `color_to_index`; historical keys such as `P0_VICTORY_POINTS`, `P0_ACTUAL_VICTORY_POINTS` | Which key includes hidden VP/awards; key prefixes follow actual shuffled seat order |
| Winner | `game.winning_color()` or equivalent | Turn-sensitive victory timing and internal turn-limit behavior; winner can be absent when truncated |
| Built-in opponents | `RandomPlayer`; candidates `WeightedRandomPlayer` in `catanatron.players.weighted_random`, `ValueFunctionPlayer` in `catanatron.players.value` | Actual availability, dependencies, policy parameters, compute cost, and whether bots read hidden hands |

Minimal integration is a player callback plus a single-decision driver. Create one controlled player and three installed built-in players. At each step, obtain the **pending actor**, not merely the normal turn owner. For a bot decision, call that bot with the current legal list and apply its chosen action. For the controlled player's decision, return a filtered observation and pause; when the evaluator supplies an action ID, resolve it to the retained legal object and apply it. Then advance until the next controlled-player decision or terminal state. A synchronous callback can instead invoke Astra directly if the evaluator supports blocking calls. Do not call `play()` in a reset/step API if it would consume the entire episode before the evaluator can answer.

Always retain an observation revision and action-ID-to-object mapping. Reject stale IDs. Rebuild the list after every mutation, including trades. Returning an already enumerated action is safer than constructing arbitrary enum tuples. Setup may use the same `BUILD_SETTLEMENT` and `BUILD_ROAD` enums as ordinary construction. A robber action may combine hex and victim; a Road Building card may trigger two subsequent road prompts; a `DISCARD` action may have no parameter because the engine chooses randomly. Expose precisely the supported choices and do not fabricate choices or change simulator rules to fit the proposed JSON names.

Catanatron releases may omit domestic negotiation, implement discards randomly, and combine/alter standard phases. Those differences directly affect strategy. Add an episode `capabilities` record with verified booleans/values for domestic trading, chosen discards, pre-roll development cards, robber/victim decision format, and full versus reduced board. Keep unsupported actions out of prompts. This study cannot certify those capabilities without source access.

## JSON adaptor contract

The adaptor should normalize the engine into JSON before invoking a model. A single observation should contain at least:

```text
phase, current_player, turn_number, pending_decision,
board {hexes: [{id, terrain, number, robber, vertices}], vertices: [{id, adjacent_hexes, neighbors, building, owner, port}], edges: [{id, endpoints, owner}]},
self {resources, development_cards, playable_old_cards, played_knights, pieces, ports, public_vp, true_vp},
opponents [{id, resource_count, development_count, played_knights, pieces, ports, public_vp}],
awards {longest_road_owner, longest_road_length, largest_army_owner, largest_army_size},
legal_actions [{id, type, parameters}], dice_result, bank_counts, terminal
```

`legal_actions` is authoritative. Parameters must use stable IDs for vertices, edges, hexes, players, resources, and cards. Include a public event log (last roll, production, trade/build, robber move, and award changes) so Astra can reason without replaying opaque objects. When the phase is a seven-discard, robber move, victim choice, or initial placement, expose only the legal choices for that pending decision. If the engine can resolve random theft internally, report the victim choice but not the stolen card before it is revealed.

After theft, only the thief and victim learn the stolen resource identity; a third-party observer sees a transfer of one card. Development purchases reveal only a count to others; played development cards are public. Initial resources and ordinary production are public events, so code may maintain **beliefs** about opponent resources from history, but must not use engine-private hands to make those beliefs exact after unknown steals/discards. Do not serialize raw engine action logs without privacy review. Our own true VP is safe to expose; opponents' true VP is scoring-only at episode end unless the rules reveal it. Bank counts should be exposed only to the extent the chosen information convention treats them as public. Never pass hidden state to model-guided search; construct sampled plausible states if search is later implemented.

The adaptor sends one action at a time and then re-serializes state. Typical action types are `place_settlement`, `place_road`, `roll`, `discard`, `move_robber`, `choose_victim`, `trade_maritime`, `trade_domestic` (only if the engine supports it), `build_road`, `build_settlement`, `build_city`, `buy_development`, `play_knight`, `play_road_building`, `play_year_of_plenty`, `play_monopoly`, and `end_turn`. Translate these to the pinned engine's typed action constructors and reject anything not in `legal_actions`; never let a model invent an edge or bypass a phase. Opponent bots and dice should advance only when the engine says so.

## Other interface setup notes

* **Gym wrapper:** After the core smoke test, inspect the installed package for a `gym`/`gymnasium` environment. Confirm its observation space does not leak hidden state, its action mapping covers all pending decisions, and `reset(seed=...)` actually seeds board and dice. If it flattens actions, retain a reversible ID map in the adaptor. Gymnasium itself is not a CATAN implementation.
* **JSettlers2:** A fallback adaptor would build the Java project, launch a local server with three built-in robot opponents, and connect the controlled fourth player using a robot client over its documented protocol. It must translate protocol messages into the same JSON contract and handle asynchronous prompts. No external account or internet is needed, but Java build dependencies and protocol details must be verified from [A1](https://github.com/jdmonin/JSettlers2). Disable any configured robot/turn timeout for Astra calls and use loopback-only networking.
* **Pioneers:** Build the local server and robot/client components from [A2](https://pioneers.sourceforge.net/), then verify a headless mode and protocol. Treat it as a rules-compatible fallback, not a drop-in Catanatron replacement.
* **Browser/screen automation:** These require a display server, OCR/CV, focus management, and online service access. They conflict with the Linux headless/offline constraint and may violate service terms; no credentials should be requested or stored for them.

## Failure and reproducibility checks

At startup, fail with an actionable message if the venv, pinned package, or wheelhouse is missing. Include package/version and seed in every episode record. If a game reaches the cap, report `truncated` and current true/public VP; do not count truncation as a win. If the engine and rulebook disagree, record the exact engine behavior in the run metadata and update the strategy prompt to match the evaluator rather than silently emulating a different edition.

Use a starting harness cap of 1,000 completed player turns and 20,000 individual actions, then calibrate on built-in-bot games. This is a proposed evaluation setting, not a CATAN rule. Preserve a lower engine-imposed cap as a documented limitation. Define a turn as one player's completed normal turn, excluding setup and interrupt decisions; four turns form a four-player round. Report both action count and completed turns.

Seeding must cover board/ports/tokens, shuffled seats, dice, shuffled development deck, theft, randomized discards, and randomized bot choices. Inspect whether the engine uses global Python/NumPy RNGs; isolate episodes and planning rollouts so hypothetical search cannot consume the live game's randomness. Replaying a fixed seed **and action sequence** should reproduce events. Different actions may consume random draws differently; identical seeds alone do not guarantee identical dice sequences across policies. For fair comparisons, rotate the controlled color through all four seats, store the realized seat order, and use a fixed train/held-out seed split with the same bot mix and caps. Keep map randomness inside the engine's valid generator; do not create illegal token placements manually. Vary opponent seeds and policies independently of seat rotation. Compare win rate with uncertainty (for example Wilson intervals), report mean true VP and truncation rate, and include legal-action rejection/timeout counts. Audit whether built-in bots use hidden information and disclose it if unavoidable.

There is no game clock in a direct engine call. Runtime latency is unmeasured; benchmark setup, typical action application, and opponent thinking separately on this server. Allow roughly 60 seconds per substantive Astra decision as an operational starting point, automatically handle forced singleton actions, and reuse the current plan for routine choices. This is not an in-game deadline. `study.json` uses `-1.0` as a documented numeric sentinel for **no finite game-imposed decision timeout**.
