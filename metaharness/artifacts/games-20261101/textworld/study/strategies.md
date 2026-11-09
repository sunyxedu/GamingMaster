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

**Loop policy, proposed:** after two rejections of the same intended action in an unchanged state, force a prerequisite review. After several steps with neither new information nor meaningful state/goal progress, ask Astra to state the missing assumption and choose a different information source or route. Thresholds are engineering heuristics; benchmark them on development seeds only.

## 7. Optimize normalized score under the command budget

**Code:** maintain native cumulative score, max score, score delta, command count, and remaining budget exactly. Sum no cumulative scores. Report final normalized score even when an episode is truncated. Native move counters and submitted-command budgets are different. [T3]

**Astra:** choose plans with achievable required outcomes, avoid unnecessary round trips, and batch retrieval decisions mentally before moving. Execute one command at a time with feedback. When the cap is near, prefer a known feasible scoring subgoal over distant speculative exploration, but do not sacrifice the sole required object for an uncertain reward. A partial-score strategy is useful only when the quest actually offers independent partial credit.

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
