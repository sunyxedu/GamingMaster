Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `TextWorld generic Glulx quests (admissible commands)` — Play a native TextWorld generated text adventure, one command per turn. Achieve the objective shown in the observation. Explore rooms, move objects and operate containers/doors with matching keys. Eating is irreversible and can make a goal impossible. There are no opponents and waiting in real time changes nothing. Send one exact string from the current admissible_commands list; look and inventory are also always accepted. Names in action_spec are these concrete strings and change with state. Grammar: look; inventory; go <north|south|east|west>; examine <object>; take <object> [from <container/supporter>]; drop <object>; open <target>; close <target>; unlock <target> with <key>; lock <target> with <key>; put <object> on <supporter>; insert <object> into <container>; eat <food>. Preserve exact names and spelling. No multi-command strings, undo, restart, debug or save/load. Every string attempt costs one command, including look/inventory, invalid strings and noop (look). Malformed non-string payloads are rejected without advancing or using the budget. Easy/medium/hard quests have 3/5/10 rooms, 5/10/20 objects, generation lengths 2/5/10 and command caps 100/200/400. Seeds vary the world and text. Observation fields: feedback is the latest native response; description, inventory and objective are current native text. admissible_commands lists currently executable interactions, an explicitly allowed assistance level that does not imply safety. score and max_score are native cumulative values; won/lost/terminated are native outcomes, truncated means the command cap was reached. moves is the engine count; commands_used/command_limit/commands_remaining track the separate budget. last_command and last_score_delta describe the previous submission; protocol_error explains rejected input; infrastructure_error means the trial must be retried. The evaluator score is final cumulative native score / positive max_score, preserving any partial credit on loss or truncation. No score for speed or exploration. No hidden facts, walkthroughs, future policy commands or intermediate rewards are exposed. After termination further steps have no effect.

Actions:
- inventory: List carried objects, costing one submitted command.
- look: Describe the current room; the noop, costing one submitted command.
(real-time games: ticks apply 'look' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Names are the current concrete command strings; this list changes after each step. Send one exact string from observation.admissible_commands, or look/inventory. Admissibility is assistance, not a guarantee of safety. No command batching.

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

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds.

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
- strategy_summary: Astra should translate the literal objective into required object relations and states, then plan backward through prerequisites while choosing one current command at a time. Code should maintain an observed room graph, item locations, inventory, explicit uncertainty, score and command budget, and use graph search for routes over known edges. Prefer actions completing a goal or prerequisite, then targeted information gathering and systematic exploration; verify outcomes before advancing the plan. Treat admissibility as executability rather than safety, and review eating or other irreversible transformations against the actual objective. Prevent state-conditioned failure loops and finish all required end conditions after partial score increases. For a separately enabled cooking suite, read the recipe and maintain an exact ingredient/preparation checklist; Jev can optionally classify feedback or match commands but should not plan or do arithmetic.
- scoring: For each accepted scenario, report final native cumulative game score divided by its strictly positive native max_score; preserve partial score on loss or command-cap truncation. Validate the suite's score range instead of hiding out-of-range values. Do not sum cumulative step scores, substitute oracle intermediate rewards, or replace partial credit with win rate. Average per-game normalized scores over a balanced fixed difficulty mix; log win/loss, truncation and command count separately. Infrastructure failures must be reported separately from agent losses.
- interface.observation: An allowlisted JSON serialization of native GameState text fields: feedback, description, inventory and objective; current admissible command strings; cumulative score, positive max_score, won/lost/done, optional engine moves; and adaptor-maintained last action, score delta, submitted-command count, remaining budget and truncation flag. Current admissible commands are an explicitly allowed assistance level, not a guarantee of safety. Hidden logical state, private manifests, generated solution policies and walkthroughs are excluded.
- interface.actions: One textual game command per turn via env.step(command), preferably an exact string from the current admissible_commands list. Examples include go north, take item from container, open container, unlock door with key, put item on supporter and insert item into container. Irreversible operations such as eating require task-specific justification. Native reset returns GameState and native step returns (GameState, cumulative_score, done), subject to validation against the pinned release.

### study/strategies.md
# TextWorld strategy prior for Astra and the harness

## Scope and evidence

The recommended suite uses generic generated quests with easy/medium/hard profiles; see `interfaces.md`. Cooking advice below is conditional on using a separately declared cooking suite. There is no universal opening move, enemy counter, or fixed solution across seeds. The useful equivalent is a reliable cycle: understand the objective, gather missing information, plan prerequisites, take one verified action, update memory, and replan when evidence changes.

References [T1–T5] are upstream technical sources; [C1–C4] are community/research resources for action ranking and persistent world representations. Web access failed in this session, so these are **verification targets**, not claims of freshly read papers, reproduced scores, or winning leaderboard methods. The strategies are reasoned recommendations informed by the established TextWorld model. [L1] is a directly inspected Microsoft fixture test demonstrating multi-step score and irreversible failure. See `research/sources.md`.

## Ownership and priorities

| Strategy | Put in code | Put in Astra prompts | Jev questions, if useful |
|---|---|---|---|
| Exact command execution | Schema validation, current command-index map, one step per action, command budget | Choose an exact current command; ask for information before ambiguous actions | Identify which offered command matches a stated immediate action |
| Objective decomposition | Persist explicit checklist and completed observations | Turn goal prose into required relations/states and prerequisite subgoals | Extract object/container names or classify a stated requirement |
| Map and object memory | Store observed graph, object locations, inventory, door states, uncertainty, recent failures | Decide where to explore and which remembered clue matters | Classify a feedback message or match an entity mention |
| Navigation | BFS over **observed traversable** edges; detect stale edge state | Select destination and key/door subgoal | No planning or route arithmetic |
| Irreversible-action guard | Flag eat/cook/cut/discard and missing-target actions for review; no blanket ban | Confirm task relevance and required state before destructive action | Is this candidate consumptive/destructive? Is this item a named goal object? |
| Loop prevention | State-conditioned action history, no-progress counters | Reconsider unmet precondition or information gap | Did feedback say success, rejection, ambiguity, or no change? |
| Score and budget | Cumulative score/deltas, normalization, remaining commands | Prioritize attainable progress and finish exact end conditions | No score arithmetic or route-cost calculation |
| Cooking checklist | Ingredient/preparation table and inventory accounting | Recipe interpretation, retrieval order, safe assembly plan | Match required preparation to a candidate/tool |

Astra is the planner. Jev is not needed for timely control in tier 1. Use it only when typed extraction/classification reduces prompt size or cost, and preserve an `unknown` option. Deterministic accounting belongs in code; a language model should not maintain the sole authoritative score, command counter, or map data structure. [T2; C1–C3; design recommendation]

## 1. Opening: interpret the actual objective

**Astra:** read objective, room description, inventory, current score/max score, and offered commands. Rewrite the goal as explicit conditions, e.g. “carrot inside chest” AND “chest closed,” not simply “find carrot.” Identify conditions that can be undone and items that must survive. Do not infer a puzzle from flavor text alone. The room prose can be whimsical while the logical task is literal. [T2, T4, L1]

**Code:** expose current allowed information without issuing hidden extra turns. Initialize a durable memory with known rooms/exits, visible objects and their locations, current inventory, known locks/keys, target conditions, unresolved questions, and the last few action outcomes. Mark unknowns explicitly rather than treating an unseen item as nonexistent.

**Opening policy:** if the goal and immediate safe prerequisite are clear, act on them. If the objective points to a recipe, read it before preparing anything. Otherwise inspect the relevant visible item/container or begin bounded exploration. Do not reflexively take every object, eat food, or type `help` every episode. Already supplied description/inventory fields make repeated `look` and `inventory` unnecessary.

**Counter to irrelevant narrative:** copy canonical object names from observations and admissible commands, and keep quest requirements separate from descriptive prose. If wording is ambiguous, inspect before committing. No hidden walkthrough or `policy_commands` should be included in the prompt.

## 2. Plan backward from the final condition

**Astra:** form a short prerequisite chain, then execute its first currently supported action. To put an object in a locked container: locate the target and container; obtain the matching key if needed; unlock; open; acquire target; insert; close only if required. Order may change with room layout and initial states. Do not issue this as a hard-coded sequence without checking observations. [T2–T4]

Separate a **goal** (“deliver carrot”) from its next **action** (“take carrot from table”). Retain a small goal stack: final condition → blocking prerequisite → next action. This makes it easier to resume after a detour to locate a key.

**Code:** maintain the explicit prerequisite checklist and search only the known room graph for routes. Exact graph search is useful bookkeeping; planning over the generator's full hidden logical state would change the evaluation. Do not use evaluator-side `game.quests`, oracle policies, simulator forks, or walkthrough replay for the live player.

**Counter to premature completion:** after a score increase, check terminal state and the remaining goal conditions. Partial score is evidence of progress, not proof of victory. In the local fixture, depositing the carrot yields 2/3; the chest must then be closed. [L1]

## 3. Explore systematically and remember what is known

**Code:** represent each observed room with exits and destinations, objects, supporters/containers, and door states. Record room identity carefully; similar narrative wording need not imply the same room. An untested exit is not a confirmed edge. A failed movement through a closed door is a blocked edge, not a new room. Update inventory/object locations only on supported outcomes. [T2; C1, C2]

**Astra:** prefer unexplored exits, unexamined relevant containers, and rooms plausibly needed by the current objective. Search openable containers when a required item is missing; search already explored rooms only when new information or a changed state makes that worthwhile. In generic quests, no map location is universally “the best opening.”

**Code:** use shortest known routes when returning to a discovered item or destination. Reverse directions are a useful hypothesis, not an excuse to add unseen reciprocal edges unconditionally. Account for whether a known door must be opened or unlocked; a BFS over rooms alone does not solve key dependencies.

**Counter to wandering:** remember which room/container searches were completed in which state. Avoid repeating an unchanged search because it fell out of the model's context. Keep discovered but not carried tools/keys in memory so the agent can return for them.

## 4. Select actions by task value, not just admissibility

An admissible list solves much of command syntax discovery. It does not identify the optimal action and does not guarantee safety. `eat carrot` is executable but loses the inspected carrot-delivery fixture. [T3, L1]

**Code:** present a stable, indexed list of the **current** commands; map a selected index back to the exact string. Never reuse an index from a previous turn. Preserve the full list or a transparent filtered view with a way to recover omitted commands; aggressive top-k pruning can hide the only useful action.

**Astra ranking heuristic:** first consider a safe action completing a required condition, then a safe prerequisite, then targeted information gathering, then useful exploration. Penalize repeating an unsuccessful command in an unchanged relevant state. Destructive actions require affirmative task justification. The priority ordering is a heuristic, not a universal proof of optimality.

**Community connection:** DRRN-style methods rank variable textual action candidates by state relevance; graph-based systems such as GATA and KG-A2C motivate persistent relational memory. Use those architectural ideas without assuming their trained models generalize to this generated suite or share its information restrictions. [C1–C3]

**Jev candidate questions:** “Which offered command takes the stated object?”, “Does this command consume or irreversibly transform an item?”, “Which status category matches this feedback?” Batch independent classifications. Do not ask Jev to construct a plan, calculate shortest paths, or infer a hidden prerequisite chain.

## 5. Guard against irreversible mistakes

**Astra:** before eating, cutting, cooking, or otherwise destroying/changing a quest object, establish all three: the objective calls for this operation, the object is the correct one, and its current state permits the intended result. When evidence is insufficient, inspect or defer. There is normally no time-pressure benefit to guessing. [T4, T5, L1]

**Code:** flag these action classes and preserve target-object identities and required states. A guard should trigger contextual review, not universally forbid the actions: eating the final meal may be necessary to win. Do not silently replace the chosen action with another action.

**Counter to plausible common sense:** real-world assumptions are not rules. A key may not match a thematically similar lock; cutting food differently may permanently invalidate a recipe; “food is for eating” fails in a delivery quest. Trust observed mechanics and explicit recipe/objective text.

**Recovery:** after a suspected mistake, inspect the resulting observation and `lost`/`done`. If terminal, stop and report native score. If nonterminal, reassess feasibility and remaining scoring goals without resetting the same evaluation episode to erase the error. Partial score can remain valuable, but do not assume an impossible main objective can be repaired.

## 6. Learn from parser failures without getting stuck

**Code:** distinguish adaptor schema error, engine command rejection, missing precondition, successful state change, score change, and terminal event. Cache failed `(relevant observed state, command)` pairs. The same action may become valid after acquiring a tool or opening a container, so do not blacklist it globally. Count every submitted command toward the benchmark cap.

**Astra:** on rejection, change the specific cause: object spelling, `in` versus `on`, missing carried key, closed container, wrong location, or ambiguous referent. Use exact admissible strings when available. Do not repeatedly paraphrase a command when feedback says the object is elsewhere. If the parser asks for disambiguation, prefer the fully qualified canonical command exposed by the environment rather than inventing conversational replies.

**Counter to stale plans:** verify each action's effect before taking the next step. `score_delta = 0` does not mean failure: walking, unlocking, and taking tools often make essential progress without score. Conversely, a friendly narrative response does not establish that the target relation changed.

…(truncated; full text in prior/strategies.md)

**Counter to reward fixation:** use narrative/objective progress and observed relations to judge intermediate utility. Do not use `intermediate_reward` or policy distance from TextWorld in the live agent; those are oracle assistance. Do not farm repeated state changes expecting repeated rewards, and do not stop on the first point unless the game is terminal.

## 8. Cooking-specific playbook — optional suite only

Cooking is a useful structured challenge but introduces mechanics beyond the generic baseline. Verify the chosen challenge version's exact command templates, appliances, preparation rules, and inventory constraints. The local cooking fixture confirms a cookbook, kitchen appliances, food, and knife, but does not establish every rule below; use the upstream challenge source for implementation. [T5, L1]

1. **Astra: read the cookbook first.** Extract each ingredient's full identity, required cut, and required cooking method. The objective can say “cook a meal” while the recipe supplies the real conditions.
2. **Code: keep a table** with ingredient, known location, carried?, required cut, observed cut, required cook, observed cook, and completion. Keep observations distinct from intentions. Do not mark an ingredient prepared because the plan mentioned it.
3. **Astra: acquire required ingredients/tools.** Plan exploration around missing ingredients and return routes to the kitchen. Handle closed containers and doors as prerequisites. Avoid hauling distractors when an inventory cap/drop mechanic is enabled.
4. **Astra: transform each ingredient exactly once as required.** In common cooking games, stove/oven/BBQ correspond to fried/roasted/grilled, and knife actions distinguish chopped/sliced/diced. Confirm the selected game's rules and commands rather than using those mappings blindly. An ingredient already in the required state should not be cooked again.
5. **Code + Astra: manage inventory safely.** If dropping is necessary, place recoverable nonessential items at a known accessible location. Preserve ingredients and the tool needed for the current operation. Do not assume generic games and cooking have identical capacity limits.
6. **Astra: prepare only after checking every condition.** Use the supported `prepare meal` action at the required location, then eat the completed meal if the objective requires it. Prematurely eating ingredients can make completion impossible.

Common failure counters: exact names distinguish similar foods; a per-ingredient checklist prevents wrong preparation; current state checks prevent burning by recooking; reading the recipe prevents applying a remembered recipe from another seed. **Jev** can match recipe requirements to candidates or classify whether observed preparation agrees; it should not determine an entire retrieval/preparation order.

## Prompt and memory design

An Astra turn should include the objective, current text, inventory, score/max score, remaining commands, current admissible commands, compact persistent map/object memory, current subgoal, and recent action outcomes. Ask for **one command** plus a concise reason/expected effect and memory correction. Keep long-term memory as explicit data, not a chain of unverified summaries. Source any memory update to an observation or mark it as a hypothesis.

Suggested standing instructions: “Solve the stated goal and preserve required objects. Select one exact current command. Before irreversible actions, verify the required target state. After feedback, update the observed map and item locations; do not assume success. If blocked, identify the missing prerequisite and seek evidence. Stop on termination.” This is a prompt recommendation, not player code.

Typed Jev outputs can use choices such as `success / blocked / ambiguous / terminal / unknown`, `relevant / irrelevant / uncertain`, or an offered command index plus `none`. Keep arithmetic, budget decisions, route search, and state reconciliation in code/Astra. A cheap recognition error should not authorize consuming a quest item without the planner's contextual check.

## Development without held-out leakage

Use disjoint development seeds to test command formatting, memory persistence, prerequisite plans, and safety checks. Keep the suite's information policy and difficulty profiles fixed. A generated solution can validate evaluator setup in a private test, but never become the player's prompt or a hidden action oracle. Compare improvements by average normalized score and per-profile results, with failure categories such as wrong object, irreversible loss, loop, exploration shortfall, parser error, and infrastructure failure. Separate infrastructure failures from agent performance.

## References

- [T1] Project: https://github.com/microsoft/TextWorld
- [T2] Environment and generated quest design: https://arxiv.org/abs/1806.11532
- [T3] API and assistance fields: https://textworld.readthedocs.io/en/latest/textworld.html ; https://github.com/microsoft/TextWorld/blob/main/textworld/core.py
- [T4] Generator and logical rules: https://github.com/microsoft/TextWorld/tree/main/textworld/generator
- [T5] Cooking rules: https://github.com/microsoft/TextWorld/blob/main/textworld/challenges/tw_cooking/cooking.py
- [C1] GATA public implementation: https://github.com/xingdi-eric-yuan/GATA-public
- [C2] Knowledge Graph A2C: https://github.com/rajammanabrolu/KG-A2C
- [C3] DRRN, *Deep Reinforcement Learning with a Natural Language Action Space*: https://arxiv.org/abs/1606.02654 ; related baselines https://github.com/microsoft/tdqn
- [C4] General parser-game player guidance: https://www.ifwiki.org/How_to_play_interactive_fiction (unretrieved community reference; generated TextWorld rules take precedence)
- [L1] Local upstream TextWorld tests: https://github.com/microsoft/jericho/blob/752ba6720d522909a4a2a495b5a03dd535f9adf1/tests/test_textworld_games.py

### study/brief.md
# TextWorld study brief

## Status and scope

TextWorld is Microsoft's framework for generating and running text adventure games, not one fixed adventure. This study recommends its native Python environment with generated games, a fixed three-profile difficulty mix, and normalized final game score. It is **tier 1**: the world advances when a command is submitted. Astra can deliberate for seconds or minutes without in-game time passing. Jev is optional; no real-time controller is needed. [T1, T2]

**Evidence limitation, 2026-10-30:** live requests to GitHub, Read the Docs, PyPI, and arXiv failed at DNS resolution. TextWorld is absent from the specified Python 3.12.3 venv. Upstream citations below are reference/verification targets, not pages successfully retrieved in this session. Locally inspected Microsoft Jericho TextWorld tests provide narrower direct evidence, marked [L1]. See `research/sources.md`. No adaptor or game-playing implementation has been written.

## Rules and objective

A generated world contains named rooms, directional connections, doors, containers, supporters, portable objects, food, and sometimes keys. Commands change logical relations and properties: the player moves to another room; an object moves from a container to inventory; a door changes from locked to unlocked to open. Exact object names and action preconditions matter. Objects on a supporter use `take … from …` / `put … on …`; objects in a container use `take … from …` / `insert … into …`. A locked door generally needs its matching key before it can be opened. [T1–T4]

Each game supplies an objective. Read it literally: merely obtaining the target may be insufficient if it must be deposited in a particular place and the container then closed. Generated quests may have intermediate rewards and failure conditions. Solvability at generation time does not imply every subsequent state remains solvable. In the inspected upstream fixture, putting a carrot into a chest awards 2 points, closing it awards the final point and wins, while eating the carrot at the start loses immediately. This is one fixture, not a universal scoring formula. [L1]

Some TextWorld challenges use different rule sets. Cooking games ask the player to find a recipe, acquire its ingredients, perform exact cutting/cooking operations, prepare a meal, and eat it. Required ingredients may be lost through premature eating, wrong preparation, or burning. Cooking-specific advice is in `strategies.md`; cooking is an optional separate suite, **not part of the recommended generic baseline**. Treasure-hunter and other challenges also require their own generation/settings contracts. [T4, T5]

There is no opponent, combat rotation, or resource build order common to all generated games. The principal difficulties are partial observability, remembering a map and object locations, discovering action preconditions, maintaining a multi-step plan, and avoiding irreversible mistakes. [T1, T2]

## Observation and actions

The normal interaction is textual: room description, command feedback, inventory, objective, score, and terminal status. The native API can request additional fields through `EnvInfos`. Some are substantial assistance, especially `admissible_commands`; others expose hidden world facts or a solution. The recommended benchmark explicitly allows current description/inventory and admissible commands, but withholds hidden facts, game objects, walkthroughs, and policy commands. Use the same information policy on all seeds. [T3]

Typical commands, subject to the generated grammar:

| Purpose | Examples |
|---|---|
| Inspect | `look`, `inventory`, `examine cookbook` |
| Navigate | `go north`, `go south`, `go east`, `go west` |
| Acquire/release | `take brass key`, `take carrot from chest`, `drop brass key` |
| Door/container | `open chest`, `close chest`, `unlock chest with brass key`, `lock chest with brass key` |
| Place | `put carrot on table`, `insert carrot into chest` |
| Consume | `eat carrot` — potentially irreversible and losing |
| Cooking variants | `slice onion with knife`, `cook onion with stove`, `prepare meal`, `eat meal` |

Prefer exact strings supplied in the current admissible-command list when that list is enabled. “Admissible” means currently executable according to the environment, **not beneficial or safe**. The legal `eat carrot` can destroy a quest. A parser rejection, a successful action with no score, and a terminal loss are distinct outcomes. [T3, L1]

## Score and progress

Report **terminal raw game score / that game's positive maximum score**, bounded to [0, 1] only after validating the supported suite's score assumptions. The native `step` contract returns a cumulative game score rather than a per-action reward; do not sum returned scores. Track score deltas separately for diagnostics. Preserve the game's partial score on loss or step-limit truncation; winning is a separate diagnostic. Require `max_score > 0` when accepting a generated scenario. Do not obtain it by guessing from prose. [T3]

For a benchmark collection, take the arithmetic mean of per-game normalized scores with equal numbers from each difficulty profile. Summing raw scores across games would overweight quests with larger maxima. Record `won`, `lost`, submitted-command count, engine move count if available, profile, generation seed, and whether the evaluator truncated the episode. None is an extra score component. A loop cannot earn extra benchmark credit by repeatedly claiming the same score. [Design recommendation]

## Pace and decision budget

The generated game is advanced by `step(command)`; wall-clock waiting between steps is not a game action. Commands may consume in-game turns, and quest rules may react to commands. Some informational or invalid commands may not count as interpreter moves, so enforce the evaluator's command budget independently. The proposed limits are 100/200/400 submitted commands for easy/medium/hard games. These are benchmark choices, not TextWorld defaults. [T3; design recommendation]

There is **no finite in-game decision deadline**. External infrastructure may still impose a run timeout or model budget. `study.json` uses **`decision_latency_s: 0.0` as an explicit sentinel for “unbounded/no wall-clock deadline,” not “must respond immediately.”** Standard JSON cannot encode infinity. Any consumer should honor this documented convention rather than divide by this field.

## References

- [T1] Project and installation overview: https://github.com/microsoft/TextWorld
- [T2] Côté et al., *TextWorld: A Learning Environment for Text-based Games*: https://arxiv.org/abs/1806.11532
- [T3] Native API, `EnvInfos`, environment contract: https://textworld.readthedocs.io/en/latest/textworld.html ; source https://github.com/microsoft/TextWorld/blob/main/textworld/core.py
- [T4] Generator: https://github.com/microsoft/TextWorld/tree/main/textworld/generator
- [T5] Cooking challenge: https://github.com/microsoft/TextWorld/blob/main/textworld/challenges/tw_cooking/cooking.py
- [L1] Locally read upstream fixture tests, commit `752ba6720d522909a4a2a495b5a03dd535f9adf1`: https://github.com/microsoft/jericho/blob/752ba6720d522909a4a2a495b5a03dd535f9adf1/tests/test_textworld_games.py . Saved reference: `research/jericho_textworld_tests.reference.txt`.

### adaptor/ADAPTOR.md
# TextWorld adaptor

This Tier 1 adaptor uses the official native TextWorld 1.6.2 Python API and its bundled Git-Glulx interpreter. Each `reset(seed)` closes the previous interpreter, deterministically generates a fresh generic quest with `textworld.make(GameOptions)`, validates its private walkthrough, and starts the compiled `.ulx` story with `textworld.start`. Generation and interpreter calls run in bounded, disposable worker processes; no network is needed.

Seeds select a balanced profile by `seed % 3`: easy (3 rooms, 5 objects, quest length 2, 100 submitted commands), medium (5 rooms, 10 objects, length 5, 200), or hard (10 rooms, 20 objects, length 10, 400). Generation retries up to five deterministic alternate generation seeds while retaining the profile. No account, server, GUI, or environment variable is required.

The observation is an allowlist of native feedback, description, inventory, objective, admissible commands, cumulative score/max score, terminal flags, move count, and command-budget diagnostics. Facts, walkthroughs, policy commands, and internal game objects are withheld. `step` accepts one exact current admissible command (plus `look` and `inventory`) and blocks until the native environment applies it. Malformed non-string actions are rejected without advancing; other rejected strings consume one command and are reported in `protocol_error`. Command batching, debug commands, undo/restart, and save/load are rejected. `look` is the noop.

The reported `score` is the native cumulative score divided by strictly positive native `max_score`; partial score remains after loss or command-cap truncation. `info()` separately reports win, loss, truncation, command count, and infrastructure validity. `close()` is idempotent and releases the worker and native interpreter. Runtime artifacts are disposable and live only under `runtime/`.

This adaptor has no opponents. The benchmark is the generated quest itself. TextWorld's native admissible-command list is intentionally exposed as a declared assistance level. TextWorld is MIT licensed; bundled Inform 7 is Artistic License 2.0, Git-Glulx is MIT, and Jericho is GPLv2 (licenses are recorded by the installed package and upstream metadata). A complete hash-locked wheelhouse is included under `runtime/wheelhouse`; `setup.sh` installs it offline into the evaluator venv.

### study/interfaces.md
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

…(truncated; full text in prior/interfaces.md)

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


## Measured model latency (at archive creation, 2026-10-30 22:50)
- System-2 (Astra), one-word answer: 62 s
- System-2, an answer of ~938 output tokens: 94 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).