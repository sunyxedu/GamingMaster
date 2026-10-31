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

From the usual **West of House** start, a conventional route is `north` to North of House, then `east` to Behind House, `open window`, and `enter` to the kitchen. From the kitchen, `west` reaches the living room. The living room normally supplies the brass lantern and sword; `move rug` exposes the trap door. Take appropriate equipment, open the trap door, ensure the lamp is on, and only then descend. Check each response and use the nouns actually printed by the parser. The mailbox/leaflet are useful introductory context but are not prerequisites for entering the house. Put route selection and recovery in **Astra prompts**, graph/inventory checks in **code**; do not hardwire the sequence as a universal opening. [S4, S5]

The **trophy case in the living room** is the central scoring destination. Acquiring treasure and depositing it can award separate points. Record actual deltas so an object already deposited is not treated as a fresh scoring opportunity. The underground return route needs planning; do not assume the trap-door transition is freely reversible. **Astra** chooses collection/banking trips; **code** tracks deposited objects and verified edges. [S5]

The **troll** and **thief** are different problems. The sword is the conventional weapon for the troll encounter, but outcomes depend on game state and randomness. The thief can move objects and disrupt inventory assumptions. Do not turn a known successful transcript into a fixed RNG-independent combat macro. Re-read combat responses, react to disarming or wounds, and preserve valuable tools. **Astra** handles tactical decisions; **code** detects inventory/score changes; **Jev** may flag combat outcome text. [S4, S5]

Representative dependency chains illustrate how to reason beyond the opening:

- A rope and a described attachment point suggest securing a descent before climbing. In Zork I, the dome/railing puzzle makes this concrete. **Astra** matches clue to tool; **code** tracks whether the rope is now installed rather than carried. [S5]
- A narrow passage or an “only certain items fit” constraint calls for a minimal inventory plan with safe storage on the accessible side. Dropping everything without planning can strand the light or needed tool. **Astra** plans the crossing; **code** preserves object locations. [S5]
- Containers, boats, and fragile mechanisms have operational prerequisites. For the inflatable boat, investigate inflation equipment and sharp carried objects before boarding; do not repeatedly try `enter boat` without addressing the failure. **Astra** interprets warnings; **code** tracks prerequisite completion. [S5]
- A magically responsive object may require an agent/tool with a specific capability rather than direct force. The jeweled egg is a classic case where trying to break it open is a poor default. Treat this as a clue to seek a capability, not a universal command to give every locked container to a thief. **Astra** reasons about capabilities. [S5]

These spoilers may help a Zork scenario, but the transferable strategy is to infer dependencies and test predictions. For a held-out-game experiment, keep title-specific hint modules distinct from the generic prompt and report whether they are enabled.

## Other classic adventures: transfer cautiously

In **Adventure/Colossal Cave**, treasure return and useful equipment from the starting building are familiar motifs, but object names, command grammar, scoring, dwarf/pirate behavior, and maze details vary by port. Inspect the chosen port's instructions instead of importing a 350-point solution into every version. Maintain a route to the deposit location, note clues and magic words, and avoid consuming unique objects without a reason. **Code:** map, deposit ledger, moves/resources. **Astra:** route, riddle/word interpretation, hazard response. Source/discovery: [S6, S7].

In **Balances** and spell-oriented Inform adventures, learn the actual casting grammar from the text and record each spell's observed effect and limitations. A spell is an action with prerequisites, not a generic synonym for `use`. Distinguish known spell names from spells currently available to cast; copying or consuming a resource may be significant. Do not assume a spell list or a maximum score without loading the exact story. **Code:** spell/resource ledger. **Astra:** effects, target choice, dependency planning. **Jev:** optional recognition of explicit success/failure. Source/discovery: [S6, S7].

For conversation-heavy games, record NPC topics and exact named clues. `ask guard about key`, `tell guard about ...`, `give ...`, and quoted commands are different grammars, and some games use menus. **Astra** infers the supported conversation style; **code** preserves pending choices and topic history. A generic two-object action generator is insufficient. [S3, S4]

## Action generation and search

**Text-only baseline:** code combines observed entities with a bounded set of likely templates; Astra proposes other commands when the context calls for them. Separate information actions, movement, prerequisite operations, scoring actions, and risky/irreversible actions. This prevents `examine` from being discarded merely because it gives zero reward. Keep a few plausible alternatives rather than an enormous list. [S1, S2]

**Jericho action assistance:** when the benchmark exposes `get_valid_actions()`, use it as a candidate source, not a complete legal-action oracle. A command can parse and be useful without producing the object changes used by filtering. Examination, conversation, unusual puzzle verbs, or context-dependent answers may be absent. “Valid” also does not mean “safe” or “high value.” Preserve an ordinary free-text command path. Cache only when the underlying relevant state is unchanged; prompts, inventory, timers, and location can all affect usefulness. **Code** handles enumeration/caching; **Astra** selects and adds missing candidates. [S1, S8]

**Snapshot search, only in an explicitly defined assisted benchmark mode:** code may branch from an opaque complete state, try a small candidate set, compare native reward/terminal status and new facts, then restore the root exactly. A practical starting budget is 8–16 one-step candidates; deeper search should focus on a concrete puzzle hypothesis, not every English string. Log trial actions separately and impose a trial budget. Do not claim these trials were ordinary text-only play or allow an unbounded free oracle. [S1, S8]

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
