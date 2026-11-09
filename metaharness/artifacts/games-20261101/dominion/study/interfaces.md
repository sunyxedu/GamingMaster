# Programmatic interfaces and adaptor specification

## Recommendation and readiness

Use **a local Python bot-library adaptor around pyminion**, initially two players and a tested whitelist of Base cards. Its intended niche—simulating Dominion with programmable players—fits this task better than controlling a client. Keep Astra outside the rules engine: the engine owns card resolution, legality, shuffles, score, and termination; the adaptor supplies choices and filtered observations.

**This selection is provisional, not an installation or compatibility claim.** The target Python is confirmed as **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python`. It currently contains pip only. Neither pyminion source nor dependencies are available locally. DNS/network access failed and this study sandbox cannot write the target venv. No upstream API, source commit, dependency versions, license, supported cards, or successful match has been verified. The source-inspection gate below is mandatory. Do not build against guessed method signatures.

Sources: https://github.com/evanofslack/pyminion ; https://pypi.org/project/pyminion/ . All external URLs here are reference targets, not successfully retrieved pages; see `sources.md`.

## Practical interface survey

Latency figures below are **engineering expectations, not measurements**. Engine computation should be small relative to Astra's seconds-to-minute inference; measure actual callbacks after installation.

| Route | Observation and control | Maturity / likely latency | Cost, licensing/ToS, and suitability |
|---|---|---|---|
| **Pyminion local Python simulator / bot extension** | In-process game/card/player objects; adapt player decision functions into JSON prompts. Potential access to all state requires deliberate hidden-information filtering. Exact hooks and card coverage require source inspection. | Small community project; packaging, tests, maintenance and 3.12 compatibility unverified. Local callback/queue operations expected sub-second. | Intended free/open-source candidate; inspect LICENSE and bundled assets before pinning. No remote account, anti-cheat, subscription, or runtime network. Best integration fit. |
| **DominionSim, Java simulator** | Source-level access to game state, scripted bot definitions and simulation loop. To make an interactive agent, add a blocking decision bridge with stdin/stdout JSON or a local socket; do not assume an existing step API. | Established community simulation project, but present maintenance/card coverage unverified. Java process startup once; individual decisions expected sub-second. | Candidate free/open-source fallback; inspect exact license and build dependencies. JVM plus Python subprocess bridge increases work. https://github.com/Geronimoo/DominionSim |
| **Androminion / Java VDOM engine** | Android application with Java game engine and AI lineage. Investigate isolating engine/player logic and using a headless Java bridge rather than emulating Android UI. Confirm actual source layout and protocol. | Longstanding community project; current JVM-only build and coverage unverified. Core callbacks should be local; Android UI adds overhead. | Candidate free/open-source alternative, subject to its exact license/assets. Android SDK/emulator is unnecessary if engine can be separated, otherwise poor server fit. https://github.com/mehtank/androminion |
| **Official Dominion Online** | Browser UI; possible DOM/log observation and browser input automation. Private network protocols are not a supported public bot API. | Mature hosted human-play service; network/animation latency and server clocks. Current bot/API policy not verified. | Requires internet/account, some content may require paid access. Automation may be disallowed; authorization/terms would need checking. Fails offline requirement; do not use. https://dominion.games/ |
| **Official Temple Gates digital client** | UI input/screenshot OCR; official built-in AI. No supported programmable-agent API identified in this study. | Mature commercial client, UI-dependent, platform-specific installation. OCR/animation delay and focus problems. | Base access may be free, expansions may cost money; proprietary software fails the free-software constraint regardless of price. Offline capabilities/initial activation require checking. https://www.templegatesgames.com/dominion/ |
| **Screen capture + injected input** | Xvfb/virtual desktop, screenshot recognition, mouse/keyboard or browser automation for a selected client. | Feasible generic technique, but no semantic legality/state guarantee. Hundreds of milliseconds to seconds, plus recognition errors. | Inherits selected client's license, costs and terms. Headless display adds dependencies; no reason to use it when a source engine is available. |
| **New custom rules engine / Gymnasium-style wrapper** | Full custom `reset`/`step`, JSON observations and exact rule implementation. Wrapping a simulator is reasonable; writing the engine from scratch is a separate substantial task. | No existing standard Dominion Gym/RLCard/OpenSpiel environment was verified. Never assume `gym.make('Dominion-v0')` works. | No service cost, but rules bugs and incomplete cards are a major benchmark risk. Not the recommended fallback while a usable open-source engine exists. |

These cover the practical approaches under the stated server constraints. No official free offline Dominion agent API was identified. Project-specific APIs and service terms remain unverified, rather than proven absent.

## Installation and provisioning handoff

### Software and credentials

Required: existing CPython **3.12.3** venv, a pinned pyminion source/wheel, all of its **resolved runtime dependencies for Python 3.12/Linux x86_64**, and standard-library JSON/threading/queue or subprocess facilities. Exact dependency names must come from the chosen distribution's `METADATA`/`pyproject.toml`/`setup.py`; this study could not inspect them. Do not invent a version pin or install an unbounded latest release. Optional: pytest and the upstream test dependencies for verification. No GUI, browser, display, CUDA, or model SDK is needed by the game engine.

**No account, server address, API key, credential, or required environment variable.** Configure seed/kingdom/opponent through the harness configuration or reset parameters. The `env` array in `study.json` is intentionally empty. Model-serving infrastructure is separate from the offline game simulator; do not add a network model client to the game environment or write any secrets to files.

All commands below are for the **adaptor/provisioning phase**, not commands successfully performed by this study:

1. On a network-enabled preparation machine, obtain the pyminion source from the repository/package-index URLs above. Save the version or commit, license, distribution metadata, source archive, and SHA-256 hashes. Inspect the source before choosing a release. Prefer a pure Python wheel when available.
2. Build/download a complete wheelhouse for **CPython 3.12, Linux x86_64**. If the project needs a patch to run on 3.12, retain the patch and build a local wheel from the pinned source. Python minimum-version metadata alone does not establish 3.12 support. Old packaging may use removed `distutils`, obsolete `collections` imports, or incompatible dependencies; diagnose actual failures rather than predicting success.
3. Transfer the wheelhouse and a fully pinned requirements lock (including hashes of all allowed wheels) under this study directory, e.g. `vendor/wheels/` and `vendor/requirements.lock`. Include source and license text in `vendor/`. The lock is a future artifact, not present now.
4. With filesystem permission to install into the target venv, install **offline** using the target executable, never system Python:

   ```sh
   /home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -m pip install --no-index --find-links ./vendor/wheels --require-hashes -r ./vendor/requirements.lock
   /home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -m pip check
   /home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -m pip show pyminion
   ```

5. Record actual import path, installed version and `Requires-Python`; inspect source and run the upstream tests under this same interpreter. Run a legal two-bot match **only in the implementation phase**, followed by the conformance checks below. A successful import alone is not sufficient.
6. If pyminion cannot run or cannot expose all needed decisions, prefer a small documented compatibility/decision-hook patch to a fork. Otherwise inspect DominionSim and evaluate a JVM JSON bridge. Do not silently fall back to online play or a partial homemade game while calling it Dominion.

Provisioning is currently blocked by absent network/artifacts and venv write scope. This does not mean the user needs to supply credentials. An external phase with the stated venv permission and a staged wheelhouse can resolve it. No installation is possible purely from the study files as they stand.

## Mandatory source inspection: map real hooks before coding

Search the acquired source (using `rg`) for `class .*Player`, `Bot`, `BigMoney`, `choose`, `select`, `decision`, `play`, `gain`, `discard`, `trash`, `reaction`, `shuffle`, `random`, `winner`, and `score`. These are **search hints, not asserted exported names**. Read README examples, player abstractions, concrete bots, game runner, card implementations and tests. Do not presume that a library's human-player CLI callback is its only choice surface.

Record this mapping in an implementation manifest:

| Required capability | What to locate and verify |
|---|---|
| Game construction | Actual constructor, number of players, starting player, explicit 10-card kingdom, edition/card registry and pile sizing |
| Rules runner | Blocking run/play loop versus one-action stepping; find every path that can request agent input |
| Player strategy boundary | Actual player class/interface; every abstract/overridable callback and return type; distinguish returning a card from directly mutating state |
| Nested effects | Every Base card that asks the player a question, including victim choices during the other player's turn |
| Legal choices | Engine-generated candidate sets if available; otherwise derive masks from exact callback constraints and engine helpers |
| Randomness | All shuffle/choice sites, global versus injected RNG, constructor-time RNG use, stochastic bots, default random kingdoms |
| State zones and public events | Card identity, owner, hand/deck/discard/play/trash, Supply, public reveals, counters, turn counts, current controller |
| Score and finish | Victory scoring, Gardens rounding, Curse scoring, end-of-turn finish check, VP/turn-count tie resolution |
| Built-in opponents | Actual policy behavior, configurability, determinism and whether callbacks accidentally use hidden information |

There is no verified upstream `Game.step()`, `Game(seed=...)`, `legal_actions`, `BigMoney` import path, or observation serializer in this study. Those names must not be treated as existing API promises. The following protocol is **our proposed adaptor contract**, independent of upstream names.

## Suspend at decisions, not just whole-turn boundaries

The externally visible `step` should answer one decision, then advance deterministic effects and scripted opponent behavior until the next decision controlled by the agent or the terminal state. An agent response can be needed on the opponent's turn (Militia/Moat); do not run an entire opponent turn without checking for these.

Preferred implementation if the engine is synchronous:

1. Keep a live game in a worker thread or isolated process. A custom player implementation replaces **all** agent decision callbacks. Each callback constructs a legal prompt, publishes it, blocks on a queue/condition, validates a response, and returns the expected engine object/value.
2. The outer adaptor reads prompts and submits responses. The engine remains suspended with its real call stack; it does not rerun card effects when a choice arrives. Use condition/queue blocking, not busy waiting.
3. Use a process per game if the upstream engine uses module-global RNG or mutable card singletons. Start fresh workers at reset; close old workers explicitly. Avoid simultaneously advancing two games sharing global state.
4. Put JSON communication on a dedicated pipe/queue or reserve stdout for JSON and send logs to stderr. Never parse prose game logs as the authoritative choice protocol when structured callbacks exist.
5. If a callback internally loops over multiple decisions, bridge each decision or submit a constrained ordered sequence according to the effect. Never advance unresolved private draws merely to collect a batch of future questions.

For planning, provide clones only if source inspection proves they copy all game/RNG state correctly. A blocked Python stack is not an ordinary serializable game snapshot. Replay from reset plus recorded decisions is a simpler restoration approach, but the policy still cannot inspect hidden future randomness.

## Proposed observation contract

A compact JSON object at every pending decision:

| Field | Required meaning |
|---|---|
| `episode_id`, `decision_id` | Stable episode identity and monotonically increasing decision identity; reject stale replies |
| `viewer`, `active_player`, `decision_player` | Seat identities; the turn owner may differ from the person answering a reaction/discard |
| `turns_taken`, `phase`, `effect_stack` | Per-seat completed/started-turn accounting documented consistently; phase plus source card/current effect explanation |
| `resources` | Current active player's Actions, Buys, coins; any temporarily modified costs relevant to this prompt |
| `kingdom` and `supply` | Exact card IDs, edition, types, printed/effective costs, text or normalized effects, remaining count per pile |
| `self` | Current hand with selectable instance IDs; own discard and in-play cards; deck count; owned-card multiset inferred from tracked events; legally known top cards, if any |
| `opponent` | Hand/deck/discard counts, visible cards in play, public gains/trashes/reveals, and ownership/VP inferred from public events where exact. Never current hidden hand or draw order |
| `public_log_delta` | Ordered public events since previous prompt, including reveals and attack outcomes; private events filtered by recipient |
| `prompt` | Kind, source effect, textual instruction, legal option IDs, min/max selections, duplicate/ordering rules, and whether declining is legal |
| `derived` | Exact known VP, estimated/unknown values explicitly tagged, empty piles, Provinces left, available spend, safe bookkeeping features |
| `terminal` | On genuine completion: final full VP, turn counts, winner/tie, end reason; no pending prompt |

**Information boundary:** simulator object access is not permission to reveal private state. Hide both players' unrevealed deck order, opponent current hand, seed/RNG state, and engine-private future choices. It is fair to track all publicly gained/trash/discarded/revealed cards as a perfect-memory player could. A public ownership multiset is not the same as knowing current hand or future shuffle. Preserve privacy on opponent discards/reveals according to the specific card effect. Treat hidden-zone instance IDs as opaque and do not expose ordering through numeric IDs or candidate enumeration. If arbitrary starting decks make ownership unknown, mark it unknown instead of reading it from the engine.

Do not send the reset seed to the playing model: the simulator needs it for reproducibility, but deriving future deck orders from it is information leakage. A policy performing rollouts must sample hidden states consistent with public information, not clone the actual hidden state exposed by the evaluator.

## Proposed action contract and validation

Response fields: `decision_id` and `option_ids` (an ordered array if order matters). An optional `mode_id` is another legal enumerated option. For binary decisions use two explicit option IDs. Use card **instances** when duplicates could have different contexts; otherwise a multiset count choice is acceptable if the engine's callback supports it.

Prompt kinds to support for the Base whitelist:

- `play_action`: one legal Action in hand or explicit end-Action option.
- `play_treasure`: if needed by the implementation; optionally collapse ordinary Base Treasures into an engine-validated play-all transition.
- `buy`: one affordable nonempty pile or stop-buying; issue another buy prompt if Buys remain.
- `gain`: legal gains constrained by effect, including cost, card type and destination; no coin/Buy charge unless the effect says so.
- `trash` / `discard`: constrained selection with min/max, including zero only when allowed; Militia requires exactly enough to reach 3 cards.
- `topdeck` / `order`: choose cards and their order; distinguish first-to-draw from first-to-place-on-stack.
- `reveal_reaction`: e.g. optionally reveal Moat for the current attack, without removing it from hand.
- `choose_mode` / `resolve_effect`: effect-specific decisions such as Vassal's optional Action play, Library's optional Action set-aside, Sentry's trash/discard/keep and remaining-card order, or Throne Room's target.

Never execute model-generated Python or accept a raw card name as authority. Validate the decision ID, legal IDs, selection count, ordering and uniqueness against the **still pending prompt**. Invalid responses must not mutate the game; return the validation error with the same prompt for repair. For unrecoverable inference failure, use a documented legal fallback or mark the episode errored; record fallback count. Never silently skip required effects. A Base-only automatic Treasure convenience must stop being assumed safe if expansions introduce Treasure ordering/buy interactions.

Limit redundant reasoning: auto-resolve a prompt with exactly one legal response and deterministic forced effects; retain a trace. Do not auto-choose optional trashing or a buy just because its value seems obvious.

## Scoring, seeds, opponents and acceptance tests

Terminal report: `agent_vp`, `opponent_vp`, `vp_margin`, `outcome`, `win_score`, `turns_taken`, `end_reason`, `truncated`, `illegal_action_count`, `fallback_count`, `kingdom_id`, `opponent_id`, engine version and seed metadata. Winner calculation is VP first, then fewer turns among VP-tied players. Avoid a made-up weighted sum of margin and win unless the evaluator specifies one. Temporary VP is a feature, not a dense training reward that overrides final objectives.

Reset configuration should include a scenario ID, master seed and agent seat. Derive separate deterministic RNG streams for kingdom selection, shuffles, seat assignment, and any stochastic opponent policy, where supported; otherwise isolate/reset all global RNGs in one worker. Record the precise implementation so repeated episodes are reproducible. Standard starting decks are always 7 Copper + 3 Estate. Vary shuffle order (including 4/3 and 5/2 openings), seat, tested kingdom, and disclosed opponent family. Both players' draws vary. A reset should not change card rules, deck composition or opponent policy secretly.

Use curated suites from `strategies.md`, plus deterministic 10-card sampling **only from implemented/tested cards**. Pair seat-swapped games for each kingdom/seed, with a documented player/seed mapping. Report results by opponent and kingdom as well as overall; use disjoint training and held-out seeds and held-out kingdom combinations. One fixed kingdom with different shuffles measures draw adaptation, not general Dominion skill. Do not promise exact mirrored draw sequences after player policies diverge.

Start with the explicitly defined money policy in `strategies.md`; then add a small validated action-aware baseline. A library's bot named "BigMoney" may not implement the same policy—inspect it or label the difference. Opponents get their own legal observations and no agent-hidden information.

Minimum implementation checks, under the evaluator Python:

1. **Installation:** import plus dependency check, upstream tests appropriate to the chosen revision, offline run with no external runtime services.
2. **Setup:** 10-card starting decks, initial 5-card hands, exactly 10 kingdom piles, correct two-player Supply counts and no accidental randomized replacement of requested cards.
3. **Resource/rules:** Action spending, +Actions, multi-buy accounting, gain destination, shuffle only when needed, no buying empty/too-expensive piles; pass/stop behavior.
4. **Choice plumbing:** Chapel 0–4 trash; Cellar multiple discards; Remodel trash then constrained gain; Mine optional trash and gain to hand; Militia/Moat during opponent turn; Sentry's choices/order; Throne Room repeats with separate choices. Add Bandit, Vassal, Library, Artisan and other card-specific cases before allowing them in random kingdoms.
5. **End/scoring:** Province ending, three-pile ending, finishing all current-turn effects, Curse penalties, Gardens floor-per-copy, VP tie/fewer-turn win and true draw. End through gains as well as purchases.
6. **Privacy:** observations with identical public information but different opponent hands/deck permutations contain no distinguishing private fields. No seed/order leak in IDs or logs. Own draw events reach only their owner until publicly revealed.
7. **Determinism:** same config/seed and response sequence reproduce the same prompts/results. An invalid action leaves state unchanged; reset leaves no stale prompt or worker. Pausing a prompt does not change state.
8. **Baseline games:** a modest batch of completed legal games, both seats and all curated kingdoms; inspect trace and scores. No benchmark win-rate claims until measured. A high safety limit on turns/decisions is an infrastructure truncation, reported separately, not Dominion's end condition.

All implementation work above is for the next phase. This study contains specifications and findings only; it does not contain a working adaptor or gameplay code.
