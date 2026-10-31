Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `jericho-interactive-fiction` — You are playing an offline Jericho Z-machine interactive fiction story. The objective is to solve the story and maximize its native score. Each step accepts exactly one parser command as a string; use ordinary commands such as 'east', 'look', 'take carrot', or 'put carrot in chest'. The observation includes the latest parser text, score, move count, and terminal flags. Failed parser commands are still submitted turns. The episode ends on victory/game over or at the evaluator step limit.

Actions:
- command: One ordinary parser command, such as 'east' or 'put carrot in chest'.
(real-time games: ticks apply 'look' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
The action value is a single string command (UTF-8, at most 200 characters, no newline or NUL). Commands are interpreted by the story's parser; one command is submitted per turn.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40 calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of 1800 s, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals.

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Use Astra for clue interpretation, puzzle dependencies, route planning, and adapting after surprising responses, while code maintains exact scores, a directed map, inventory provenance, and parser context. Seek reachable prerequisites such as light and access before risky exploration, and bank treasure when the selected game's scoring rules reward it. Execute one command at a time, predict its effect, and revise plans after failures, theft, hazards, or changed exits. Generate commands from observed nouns and relevant templates while retaining unusual free-text verbs and clarification answers; optional Jericho valid actions are helpful but incomplete and not safety guarantees. Track scarce turns and unique items, avoid repeated grammar failures and irreversible losses, and distinguish preparation from immediate reward. Optional bounded snapshot search must be a declared assisted mode with exact restoration and trial limits. Jev is unnecessary for timing but can batch typed hazard, parser-response, and candidate-relevance classifications; do not assign planning or arithmetic to it.
- scoring: For each admitted exact story build with verified maximum M > 0, report final native score S divided by M at termination or truncation, plus S, M, initial score, win/loss, and steps as diagnostics. Use final score, not peak score, summed positive rewards, or exploration bonuses. Retain genuine negative native scores unless the evaluator specifies a different convention; scores above M indicate an instrumentation/manifest error. Exclude unknown/zero maxima. Aggregate normalized episode scores with balanced per-game sampling.
- interface.observation: JSON wrapping verbatim initial/latest parser text, last action, manifest game ID and seeds, native current/max score and delta, native moves when reliable, evaluator submission count, termination/truncation and verified win/loss metadata. Room/inventory memory is transcript-derived and timestamped. Optional valid actions or VM-object/search observations must be identified as assisted; never inject walkthroughs or assume extra look/inventory commands are free.
- interface.actions: Submit one bounded parser text line through FrotzEnv.step, such as north, examine lamp, open window, take sword, put treasure in case, or an answer to a pending clarification. Ordinary free-text commands remain available even with generated candidates. Reject multiline/compound bypasses and evaluator lifecycle commands; track rejected submissions against the step cap. Evaluator reset/close are separate from scored actions.

### study/strategies.md
# Playing Jericho text adventures well

This is a strategy prior for a turn-based, partially observed parser game, not an executable policy. The score objective is final native score divided by that story's verified maximum. Native moves, evaluator submissions, and wall-clock thinking time are different quantities. Astra can spend a minute reasoning without draining a turn-based lamp; entering ten speculative commands can drain it.

Evidence status, 2026-10-30: external source requests failed DNS resolution. The URLs below identify established documentation, community hints, and research to consult; their contents were not retrieved in this session. General heuristics and familiar Zork examples below are prior knowledge, not a tested walkthrough for a provisioned ROM. Proposed memory formats, question schemas, and action budgets are design recommendations, not claims from the cited papers. No game has been installed or played here.

## Division of labor

| Responsibility | Put it in | Concrete behavior |
|---|---|---|
| Transcript, counters, score normalization | **Code** | Preserve exact input/output, calculate deltas and ratios, distinguish native turns from submissions, never ask a model to do score arithmetic. |
| Room graph and object ledger | **Code + Astra prompts** | Code stores directed edges and provenance. Astra interprets descriptions and proposes facts; uncertain identities stay hypotheses. |
| Puzzle dependencies and route priorities | **Astra prompts** | Explain a reachable subgoal, prerequisites, evidence, likely point gain, and a safe exit. Plan several steps but execute one at a time. |
| Candidate generation | **Code + Astra prompts** | Combine observed nouns and a small verb/template set; Astra supplies unusual verbs, conversation topics, spells, and puzzle-specific commands. |
| Hazard recognition | **Astra prompts; optional Jev questions** | Recognize darkness, combat, failing light, rising water, pending clarification, and parser rejection from the latest response. |
| Candidate safety/affordance tags | **Optional Jev questions** | Independently tag candidates as supported/unsupported/unknown or hazardous/unknown. Jev does not sequence a plan or estimate distances. |
| Search, caching, repetition control | **Code**, if search is part of the benchmark | Bound branches and actions, restore complete snapshots, track failed hypotheses and cache by an adequate state key. |
| Novel puzzle interpretation | **Astra prompts** | Use wording, object descriptions, world rules, and observed effects to propose and revise explanations. |

Jev is **not required** for Tier 1. Start with Astra and exact bookkeeping. Add batched recognition only if it reduces cost without losing useful context; there is no real-time control loop to maintain. [S1, S2]

## Opening procedure for an unfamiliar story

1. **Read the banner and immediate prompt.** Identify title/release if available, stated mission, starting equipment, exits, urgency, and whether the game is asking a menu or clarification question. Do not issue `take lamp` into an introductory yes/no prompt. Code records the game ID/hash; Astra interprets the visible premise. [S3, S4]
2. **Establish what is already known.** Use the opening description before spending another command on `look`. Obtain inventory if it is unknown and safe to query. Examine conspicuous writing, containers, switches, and unusual objects. Treat turns as potentially costly until that game's behavior is known. Astra chooses observations; code labels their freshness. [S3, S4]
3. **Find the first reachable prerequisite.** Typical targets are illumination, entry through a visible opening, a container holding a useful object, or a clue naming an operation. Prefer clue-backed actions to enumerating every noun/verb combination. Astra plans; code records the supporting transcript. [S3, S4]
4. **Map while progressing.** Record actual transitions and blocked exits, collect useful portable tools, and identify a safe storage/scoring location. Do not assume a room is empty because its initial description omits scenery details. Code maintains the graph; Astra chooses which unexplored exit is worthwhile. [S2–S4]
5. **Replan at evidence changes.** New score, object transformations, an NPC arriving, a changed exit, theft, death warnings, or a failed prerequisite invalidate a cached plan. Code detects mechanical changes; Astra revises the plan. [S1, S2]

There is no universal build order. A treasure hunt, a detective story, and a spell puzzle can share parser syntax while rewarding different behavior.

## The decision loop and its memory

Give Astra a compact current packet: last exact response; score delta/current/max; known location and inventory with observation times; current hazard and parser mode; immediate objective; relevant map edges; unresolved prerequisites; and recently failed actions. Preserve the full transcript on disk so the model can recover exact wording. Do not feed a growing unstructured transcript as the only memory.

Ask Astra to choose **one command** and state a short expected effect, a success signal, and what would invalidate the plan. The harness submits that command, records the result, and evaluates the prediction. Never execute a ten-command route blindly: an NPC, closing door, or clarification prompt can change the meaning of its second command. This loop is an engineering recommendation motivated by the partial observability and combinatorial action space discussed in [S1, S2].

Suggested records, maintained by **code** with **Astra** interpretations:

- **Rooms:** stable internal ID, observed names/descriptions, exits, visits, hazard facts, and confidence. Names alone are not IDs: mazes reuse descriptions, and one room may change its description.
- **Edges:** source, exact command, destination, evidence step, condition, whether reverse traversal was tested. North followed by south need not return to the starting room.
- **Objects:** observed aliases, last known owner/container/location, open/locked/lit/worn state when known, portability, and evidence step. Track nested containment; being in the room does not mean being carried.
- **Puzzles:** obstacle, goal, prerequisite hypotheses, supporting clues, attempted interventions and results, next cheapest discriminating test.
- **Resources:** native moves, observed lamp warnings, consumable quantities, known carrying constraints, and remaining evaluator steps. Keep unknown limits unknown; never invent a universal lamp lifetime or carrying limit.
- **Events:** points gained/lost, doors changed, objects stolen/destroyed, NPC state changes, and possible irreversible actions.

Promote a hypothesis to a fact only after sufficient observation. For example, “the key may fit the iron door” is not “the door is unlocked.” If a thief stole the key, all plans depending on carrying it need reconsideration. [S2, S4, S5]

## High-value heuristics and their failure cases

| Situation | Strong response | Common mistake / countermeasure | Owner; sources |
|---|---|---|---|
| Many possible commands | Start from salient nouns, described mechanisms, and a few relevant templates: `examine X`, `open X`, `take X`, `put X in Y`, `unlock X with Y`, `give X to Y`. | Trying every item on every item wastes turns. Expand only after a specific hypothesis or clue. | **Code** generates/deduplicates; **Astra** prioritizes and adds unusual actions. [S1–S4] |
| Parser says it does not understand | Separate unknown verb, unknown noun, absent object, wrong preposition, and unmet world precondition. Reuse nouns from the prose and simplify the command. | Repeating an identical string without a state change cannot fix grammar. Never conclude a puzzle is impossible from one wording failure. | **Astra** diagnoses; **code** stores failed wording; **Jev** may classify response. [S3, S4] |
| “Which … do you mean?” | Enter the distinguishing noun/adjective required by the question; keep parser mode pending. | Issuing a route command into a clarification context can answer the wrong question or waste a turn. | **Code** maintains mode; **Astra/Jev** recognize and select referent. [S3, S4] |
| No immediate reward after a useful action | Track prerequisites and newly reachable states, while preserving native score as the evaluation metric. | Sparse rewards make necessary preparation look useless; exploration bonuses must not replace reported score. | **Astra** evaluates dependencies; **code** keeps native score exact. [S1, S2] |
| Dark location or failing lamp | Obtain/activate light before further movement, preserve an escape route, and reduce low-value exploration. In safe lit areas, conserve portable light if the game's mechanics support it. | Walking randomly in darkness can kill the player. Turning the only light off in a dark room is not conservation. | **Astra** plans; **code** tracks warnings/moves; **Jev** can flag danger. [S4, S5] |
| Carrying capacity reached | Store unneeded objects at a known retrievable location; keep immediate puzzle tools and light. Prefer banking eligible treasure when feasible. | `take all` can overload the player, take harmful objects, or hide which item mattered. An item on the ground may be stolen or become inaccessible. | **Astra** chooses kit/storage; **code** tracks ownership. [S4, S5] |
| Treasure acquired | Determine where/how it scores and whether it remains a puzzle tool. Deposit it when safe and useful. | Carrying treasure is not always full credit; sacrificing a unique treasure may lose score or make a puzzle unsolvable. | **Astra** handles dependency tradeoff; **code** records actual score events. [S5] |
| Hostile NPC | Use observed combat rules and a suitable weapon, retreat if feasible, or seek the intended noncombat solution. Verify inventory after theft/disarming. | Repeated attacks while unarmed or injured can consume turns without improving survival. Never assume every creature is meant to be killed. | **Astra** chooses response; **Jev** flags attack/disarm/death text; **code** records changes. [S4, S5] |
| Maze with repeated room text | Use arrival direction, tested exits, and persistent distinctive observations to separate nodes; cautiously use expendable markers if the game permits. | Merging all “twisty passages” rooms into one produces false routes. Markers can be moved or stolen. | **Code** graph; **Astra** disambiguation. [S3–S5] |
| Turn deadline or moving NPC | Prepare prerequisites before initiating the sequence; execute the shortest known safe sequence with response checks. | Confusing a turn deadline with a wall-clock deadline; extra `look`, failed commands, or `wait` can matter. | **Code** counts turns; **Astra** plans sequence. [S4, S5] |
| Irreversible operation | Check the dependency graph before burning, consuming, throwing away, giving away, breaking, or closing access. | Many classic games permit unwinnable live states. Apparent progress can destroy later prerequisites. | **Astra** assesses; **Jev** may tag candidates as irreversible; **code** retains evidence. [S4, S5] |
| Repeated failure with no new facts | Change the hypothesis, seek a missing clue/tool, or pursue a different reachable puzzle. | Random synonym churn and cycling between the same rooms consumes the episode. | **Code** detects recurrence; **Astra** changes plan. [S1–S4] |

Use recurrence thresholds as tunable harness heuristics, not game rules. For example, after two identical failures in an unchanged context, require a different command or an explicit reason before retrying. Apparent unchanged text does not prove unchanged timers or RNG; combat retries can be justified while grammar retries usually are not.

## Zork I: useful opening and puzzle patterns

**Scope:** these are familiar Zork I patterns, conditional on the game identity and observed rooms. They are not shared rules for Zork II/III or other Jericho stories. Verify release-specific details against the selected legal binary and the manual/hints [S4, S5]. Zork files have not been admitted to a redistribution manifest by this study.

…(truncated; full text in prior/strategies.md)

Search needs more than positive reward: death penalties, resource consumption, irreversible loss, information gain, and future reachability matter. A zero-point door opening can be essential; a point-gaining action can still make a later treasure impossible. **Astra** supplies the strategic evaluation and the code applies only exact comparisons or explicitly declared heuristic weights. Never silently equate the object-tree hash with complete game state: globals, timers, RNG, parser context, and previous events can differ. Do not use a transposition table to discard states solely because room text matches.

Walkthrough APIs or bundled solutions are useful for isolated instrumentation tests, but revealing the next walkthrough command to the policy changes the task to solution replay. Keep fixtures out of runtime observations. If title-specific hints are intentionally part of the prior, declare that separately from simulator search. [S8]

## Optional Jev typed questions

Supply a JSON state containing the latest verbatim response, pending prompt context, a small set of **observed** facts with provenance, and candidate commands with stable IDs. Batch independent questions; do not make one answer depend on another answer from the same batch. Include `unknown`/`unclear` choices so Jev need not invent a fact.

| Typed question | Answer domain | Use and limit |
|---|---|---|
| What kind of response is this? | `normal`, `clarification`, `yes_no`, `parser_error`, `explicit_death`, `explicit_victory`, `unclear` | Route prompt handling. Native tested terminal hooks remain authoritative for evaluator termination. |
| Does the text explicitly warn of darkness, failing light, attack, flooding, or another immediate threat? | One independent `yes/no/unclear` per hazard | Escalate to Astra; do not derive turns remaining from prose using Jev. |
| Does this candidate target an object named in the supplied current evidence? | `yes/no/unclear` per candidate | A relevance tag, not proof the item is reachable or the command legal. |
| Does the supplied text state that this item was taken, lost, dropped, or destroyed? | `taken`, `lost`, `dropped`, `destroyed`, `not_stated`, `unclear` per item | Suggest ledger updates with a matching evidence span/ID if the question format permits; Astra resolves ambiguity. |
| Which supplied referent best matches the clarification question? | Candidate object ID or `unclear` | Choose among explicit referents only. Do not invent an absent object. |
| Could this candidate consume or irreversibly discard the named unique item? | `yes/no/unclear` per candidate | A conservative flag for Astra, not a blanket action veto. |

Do **not** ask Jev to calculate normalized score, count a long route, remember unprovided inventory, devise a multi-step solution, compare dependent future plans, or infer an entire hidden world. A fast classification error should not corrupt persistent memory without evidence. Because the world waits, uncertainty can simply go to Astra.

## Priorities for harness experiments

1. **Implement in code first:** exact transcript/score logging, deterministic game/seed mapping, one-command stepping, terminal handling, directed map and inventory provenance, parser-mode preservation, and bounded history retrieval.
2. **Prompt Astra next:** dependency-oriented planning, expected outcomes, hazard/resource awareness, graceful grammar repair, and explicit revision after surprising results.
3. **Then compare assistance modes:** text-only candidates versus validated Jericho action assistance; add a bounded-search arm only if the benchmark definition includes it. Report action/trial budgets consistently.
4. **Add Jev last:** measure whether batched classification reduces total cost or improves decisions; it is not needed to meet a game deadline.

Judge variants by balanced per-game final normalized score, then inspect deaths, parser failures, repeat loops, and unproductive turns as diagnostics. A change that explores more rooms but ends with less native score is not an improvement under the requested metric. Freeze the game manifest, native seed mapping, observation mode, and command budget for comparisons. Hold out entire titles as well as RNG seeds when enough legally usable titles exist. [S1, S2]

## Sources and verification targets

- **[S1]** Hausknecht et al., *Interactive Fiction Games: A Colossal Adventure* (Jericho; partial observability, action generation, environment facilities): https://arxiv.org/abs/1909.05398
- **[S2]** Ammanabrolu and Hausknecht, *Graph Constrained Reinforcement Learning for Natural Language Action Spaces* (knowledge graphs and action constraints): https://arxiv.org/abs/2001.08837
- **[S3]** Brass Lantern, *A Beginner's Guide to Playing Interactive Fiction* (parser literacy and general playing advice): https://brasslantern.org/beginners/playif.html
- **[S4]** Infocom documentation archive (select the manual/reference card for the exact title): https://infodoc.plover.net/
- **[S5]** Eristic, Zork I community hints/walkthrough (spoilers; verify against the selected release): https://www.eristic.net/games/infocom/zork1.html
- **[S6]** IF Archive Z-code collection and descriptions: https://www.ifarchive.org/indexes/if-archive/games/zcode/ ; source collection: https://www.ifarchive.org/indexes/if-archive/games/source/
- **[S7]** Interactive Fiction Database, title/release and community-resource discovery: https://ifdb.org/
- **[S8]** Jericho project, API, and implementation: https://github.com/microsoft/jericho ; https://jericho-py.readthedocs.io/en/latest/ ; https://github.com/microsoft/jericho/blob/master/jericho/jericho.py

The sources above support investigation of the relevant mechanics and methods; this session cannot certify their current contents, exact downloadable assets, or licenses. The division between code, Astra, and Jev is this study's proposed design.

### study/brief.md
# Jericho interactive fiction: study brief

Research status, 2026-10-30: the evaluator interpreter was checked locally and is Python 3.12.3. Jericho is absent. Fresh attempts to fetch GitHub raw source, Read the Docs, PyPI, IF Archive, Brass Lantern, and Eristic all failed DNS resolution in this sandbox. Consequently, the links below are source references for verification, **not pages successfully read in this session**. The mechanics and API descriptions are based on established Jericho/Z-machine knowledge; release details, licenses, exact supported ROMs, and Python 3.12 behavior remain unverified. No game-playing code has been written. See `interfaces.md` for a concrete implementation and acceptance plan and `research-log.md` for local evidence.

## What this game is

Jericho is a Python research environment around a modified Frotz Z-machine interpreter, rather than one adventure with universal rules. It runs supported compiled interactive-fiction stories. Zork I, II, and III are separate games; other stories differ in parser vocabulary, maximum score, hazards, victory conditions, and even whether scoring exists. Use a fixed manifest of individually approved story binaries. Do not mistake support for many Z-machine files for full instrumentation support for every release of every story. [J1–J4]

The player reads prose, types a command, and receives the narrator's response. Rooms form a graph with directional exits; movable objects can be carried, manipulated, combined, or given to characters. Puzzles impose dependencies: obtain a key before unlocking a door, find illumination before entering darkness, learn a word before casting a spell. The world is only partially observed, and text can omit relevant state. Parser errors, ambiguity questions, scripted events, and changing room descriptions all matter. [J5, I1, Z1]

## Objective and score

The evaluation objective is **normalized in-game score**, as requested: final native score divided by the verified positive maximum for that exact story build. Report raw final score, maximum, normalized score, win/loss, and truncation separately. Do not replace this with exploration bonuses, the sum of only positive rewards, a language-model estimate, or the highest score ever seen. Death can cost points; final score must reflect that. A successful ending and maximum score are not necessarily equivalent. `interfaces.md` specifies edge cases and aggregation. [J2, J5]

Zork I illustrates the genre: find treasures, solve obstacles, and put treasures in the living-room trophy case. Discovering/acquiring a treasure and depositing it can contribute different points; merely carrying everything does not finish the task. Its familiar maximum is 350 points, but the manifest must verify the exact loaded release rather than assuming all games share this scale. Light lifetime, carrying capacity, hostile creatures, and irreversible mistakes constrain routes. The 350 figure is a known Zork I reference value, not a locally tested ROM measurement. [Z1–Z3]

## Commands and parser behavior

| Purpose | Typical inputs | Caveat |
|---|---|---|
| Movement | `north`, `n`, `south`, `east`, `west`, `up`, `down`, `in`, `out` | Exits may be one-way, conditional, or non-geometric. Never infer the reverse exit is valid. |
| Observe | `look`, `inventory`, `examine lamp`, `read leaflet` | Abbreviations such as `l`, `i`, and `x` depend on the parser; even observation commands must not be assumed turn-free across stories. |
| Manipulate | `take lamp`, `drop rope`, `open window`, `unlock door with key`, `put jewel in case` | Use names and verbs from the game's prose. Containers and supporters differ. |
| Operate/fight | `turn on lamp`, `tie rope to railing`, `attack troll with sword` | Preconditions and wording are game-specific. These can consume scarce turns or provoke death. |
| Communicate | `ask person about topic`, `give item to person`, a quoted word | Conversation grammars vary. An answer to a clarification prompt may be just a noun. |
| Wait | `wait` or often `z` | Explicitly advances game time; waiting for the model does not. |
| Meta | `score`, `save`, `restore`, `restart`, `quit`, sometimes `undo` | Evaluator should control persistence and lifecycle; do not expose reset/restore/undo as ordinary scored player actions. |

Submit one parser input per step. A compound line such as `n.then open door` can execute several turns or behave differently between parsers. Preserve clarification prompts: after “Which key do you mean?” the next input should answer that question rather than launch a fresh plan. [I1, Z1]

## Pace and tier

**Tier 1 is correct for the selected untimed stories.** Simulation normally advances when input is processed, not while the player thinks. Combat, lamp drain, flooding, NPC movement, and timed puzzles are usually measured in game turns. Astra may deliberate for seconds or a minute without taking an in-game turn. Repeated inspection, failed actions, or `wait` may still advance game time depending on the story. [J2, I1, Z1]

The Z-machine specification does allow timed input, so “all interactive fiction waits forever” is too broad. Exclude stories using wall-clock timed input from this Tier 1 pool, or prove the selected interpreter handles them in a deterministic, turn-based way. Evaluation service wall-clock limits are separate from game mechanics and were not supplied. [V1]

There is **no finite game-imposed decision deadline** for an admitted untimed game. In `study.json`, `decision_latency_s = -1.0` is an explicitly documented sentinel meaning unbounded, not a negative actual timeout. If a downstream schema requires a nonnegative number, the adaptor owner must define an unbounded convention; do not silently invent a 60-second game deadline.

## Sources to verify

- [J1] Jericho project and installation overview: https://github.com/microsoft/jericho
- [J2] Jericho API documentation: https://jericho-py.readthedocs.io/en/latest/api.html
- [J3] Python implementation, exact signatures and bindings: https://github.com/microsoft/jericho/blob/master/jericho/jericho.py
- [J4] Game-specific data/support: https://github.com/microsoft/jericho/tree/master/jericho
- [J5] Hausknecht et al., *Interactive Fiction Games: A Colossal Adventure* (Jericho paper): https://arxiv.org/abs/1909.05398
- [I1] Brass Lantern, *A Beginner's Guide to Playing Interactive Fiction*: https://brasslantern.org/beginners/playif.html
- [Z1] Infocom documentation archive; Zork manuals and reference cards: https://infodoc.plover.net/
- [Z2] Zork I community hints (spoilers; verify wording against selected release): https://www.eristic.net/games/infocom/zork1.html
- [Z3] Zork I source reference (source availability alone does not establish redistribution permission): https://github.com/historicalsource/zork1
- [V1] Z-machine standard, input/output and timed input: https://inform-fiction.org/zmachine/standards/z1point1/sect15.html

### adaptor/ADAPTOR.md
# Jericho interactive fiction adaptor

This Tier 1 adaptor runs a local Z-machine story with `jericho.FrotzEnv` (Jericho 3.3.1). It has no server, GUI, account, or network requirement. `reset(seed)` selects the ordered manifest entry with `seed % N` and passes `1 + ((seed // N) % 32767)` as Jericho's VM seed. The VM is recreated for every reset, so the opening transcript is deterministic for a fixed story and seed.

The bundled manifest contains Jericho's supported `tw-game.z8` generated fixture. Its exact SHA-256 and native maximum score (3) are checked before the VM starts. A deployment with additional legally approved, supported stories can point `JERICHO_GAME_MANIFEST` at a manifest with the same fields; every entry must have a positive `max_score` and a matching SHA-256. The single bundled title is a smoke-test corpus, so held-out seeds test RNG/replay robustness rather than novel story content.

The action is one parser command string per turn (maximum 200 characters, no newline or NUL). `describe()` documents the grammar and `action_spec` advertises the command channel plus `look` as the noop. Lifecycle commands (`restart`, `restore`, `save`, `quit`, `undo`) are reserved for the evaluator and are reported as an action error without changing the VM. Parser output is returned verbatim in a compact JSON observation with score, native moves, transition delta, seeds, game hash, and terminal flags. No automatic `look` or `inventory` commands are issued.

`score` is the final native score divided by the manifest maximum; negative native scores remain negative. `score_breakdown()` and `info()` retain native score, maximum, initial score, win/loss, seed mapping, and termination reason. A submitted-command cap (default 1000) produces truncation and preserves the final score. `close()` releases the Frotz environment and is idempotent.

The bundled story fixture and Jericho source are covered by the upstream repository's distribution notices. Before adding third-party stories, verify their exact redistribution terms and preserve their license text in the manifest or runtime bundle.

### study/interfaces.md
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

…(truncated; full text in prior/interfaces.md)

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


## Measured model latency (at archive creation, 2026-10-30 05:32)
- System-2 (Astra), one-word answer: 44 s
- System-2, an answer of ~969 output tokens: 81 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).