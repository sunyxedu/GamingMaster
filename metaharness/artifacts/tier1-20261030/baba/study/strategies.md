# Strategy prior for Baba Is AI

Keep tier 1. Use Astra to choose rule changes and subgoals, and deterministic code to maintain exact state and execute/check moves. Jev is optional: no real-time response requirement justifies putting a second model in the control loop. The requested score is puzzles solved, so preserve solvability and use thinking time where it improves completion probability.

**Evidence status, 2026-10-30:** GitHub and PyPI retrieval failed because DNS resolution is unavailable. No upstream implementation, level corpus, research paper, or community page was retrieved in this session. The tactics below are general Baba Is You knowledge and explicitly proposed engineering deductions, not measured performance claims or a verified Baba Is AI solver. Links are reference targets for subsequent verification. Only enable a mechanic after the selected package proves it supports it. The exact transition engine overrides these descriptions.

## Source key and scope

| Key | Reference URL | Intended support and limitation |
|---|---|---|
| S1 | https://www.hempuli.com/baba/ | Creator's description of the original game's rule-changing premise; does not specify the Python benchmark API. |
| S2 | https://babaiswiki.fandom.com/wiki/Rules | Community rule grammar reference target; page path/content not checked here. |
| S3 | https://babaiswiki.fandom.com/wiki/YOU | Player control and multiple controlled objects. |
| S4 | https://babaiswiki.fandom.com/wiki/WIN | Victory property, including alternatives to reaching a conventional flag. |
| S5 | https://babaiswiki.fandom.com/wiki/PUSH | Push mechanics and interaction with other properties. |
| S6 | https://babaiswiki.fandom.com/wiki/STOP | Blocking depends on rules. |
| S7 | https://babaiswiki.fandom.com/wiki/SINK | Destructive overlap and sacrificial-object tactics. |
| S8 | https://babaiswiki.fandom.com/wiki/OPEN and https://babaiswiki.fandom.com/wiki/SHUT | Matching properties, not object appearance, govern opening. |
| S9 | https://babaiswiki.fandom.com/wiki/MOVE | Automatic movement and direction; exact timing must be inspected. |
| S10 | https://jsokoapplet.sourceforge.io/sokoban/deadlocks.html | Sokoban deadlock concepts as an analogy. Baba's changeable rules invalidate unconditional Sokoban pruning. |
| S11 | https://babaiswiki.fandom.com/wiki/HOT, https://babaiswiki.fandom.com/wiki/MELT, and https://babaiswiki.fandom.com/wiki/DEFEAT | Hazard vocabulary; supported combinations/order are engine-specific. |

No claim below that is specific to this benchmark is established by these original-game references. Source inspection remains necessary, especially for transformations, overlaps, simultaneous moves, and hazard precedence.

## Opening analysis for every puzzle

There is no universal opening move or opponent rotation. Start by understanding which rules can be changed and what each candidate win would require. Do not spend live actions merely to inspect the board.

1. **Code:** produce a lossless board, active-rule list, current controlled objects, current WIN objects, and supported action vocabulary. Include text coordinates, physical-object coordinates, and the distinction between them. If the native observation hides information, preserve that restriction rather than silently exposing privileged state. [S1–S6]
2. **Astra prompts:** identify the currently possible win conditions: reach a WIN object, make a reachable object WIN, transfer YOU to an object near WIN, or obtain YOU and WIN on the same entity if the implementation allows it. Do not assume the flag is the only target. [S3, S4]
3. **Code + Astra prompts:** inventory editable words and accessible sides of those words. Separate fixed-looking text from text that is actually unreachable under all currently feasible plans. An inaccessible rule can become editable after a blocker changes. Record rule dependencies such as “cannot reach WIN while WALL IS STOP remains active.” [S2, S5, S6]
4. **Astra prompts:** work backward from one candidate victory to a short chain of prerequisites. Example: reach flag → open corridor → break WALL IS STOP → reach the correct side of STOP. This is a candidate plan, not proof that each step is physically possible. Compare it with changing the goal or controlled noun. [S1–S6]
5. **Code:** verify the next subgoal with the authoritative simulator if local simulation is permitted. Otherwise execute a conservative primitive action and inspect the result. **Astra prompts:** revise when the predicted outcome differs. Never execute a long stale action list after a rule changes. [Engineering recommendation derived from S1–S6]

## Tactics, mistakes, and responses

| Situation / strong tactic | Exact checks and common failure | Where it belongs | Sources |
|---|---|---|---|
| Change the obstacle's rule | Breaking WALL IS STOP can remove solidity. First establish whether the relevant word can be pushed from the needed side, whether another WALL IS STOP remains, and whether another property still blocks passage. | Astra chooses the rule to change; code recomputes actual properties and reachability. | S1, S2, S6 |
| Change the victory condition | A reachable ROCK IS WIN may be easier than walking to a flag. Check that the noun, IS, and WIN can actually be assembled in a valid line, and that a controlled entity survives the final interaction. | Astra proposes candidate goal nouns; code verifies grammar, placement, and terminal success. | S2, S4 |
| Transfer control | Make ROCK IS YOU when rocks can access useful terrain. Account for all rocks becoming controlled. Establish replacement control before losing old control when possible. | Astra plans the handoff; code tracks all controlled objects and the transition where rules update. | S2, S3 |
| Keep enough control to finish | Moving YOU or IS out of a sentence can strand the position. No YOU is a strong warning, but is only a proven dead end if autonomous dynamics cannot restore control or win. | Code can prove a frozen no-control state under verified rules; Astra judges uncertain recoveries. | S3, S9 |
| Preserve access behind words | A pushable tile needs both a destination and an accessible pushing position. A word at the edge can become unrecoverable without pulling, transformation, or another supported mechanic. | Code calculates legal pushes; Astra preserves maneuvering space and chooses ordering. | S5, S10 |
| Use the whole sentence geometry | A word may participate in both a row and a column. Disabling a blocker can also disable YOU or activate a hazard. Check both axes after every text movement, including pushed chains. | Code parses all supported grammar; Astra reviews coupled rules before committing. | S2, S3, S11 |
| Exploit multiple YOU objects | Shared direction commands can let one object push a remote word while another stays safe, or let a wall hold one object while another moves. Independent path plans are invalid. | Astra coordinates roles; code simulates all objects together with native resolution order. | S3, S5, S6 |
| Use expendable objects | Under supported SINK rules, sacrificing an object can clear a hazard. Under OPEN/SHUT rules, a matching object can remove a barrier. Preserve unique words and any object needed later. | Astra assigns resources and order; code verifies destruction, remaining resources, and actual overlaps. | S7, S8 |
| Wait for turn-driven motion | MOVE objects can change position while YOU waits or issues a blocked move, if the engine advances them on those actions. Direction and phase are part of state. | Code handles timing and verifies available wait semantics; Astra chooses the interaction. | S9 |
| Transform a noun | X IS Y can change every X object, including ones needed for control or resources. Y's current properties then matter. Do not assume transformation is reversible or that simultaneous transformations follow ordinary-language intuition. | Astra considers transformations as major subgoals; code uses engine semantics. | S1, S2 |
| Use optional advanced grammar | AND, NOT, HAS, conditions, or other words radically change planning only if present and implemented. A guessed universal parser can fabricate legal rules. | Code exposes the verified vocabulary/grammar; Astra's prompt includes only supported constructs. | S2; benchmark source required |

“Counters” in this puzzle mean responses to constraints, not opponent strategies. For an apparently sealed goal, consider removing blocking, changing WIN, or moving YOU to another noun before spending moves forcing a path. For an immovable word, consider whether its property or surrounding terrain can change. For a narrow passage, consider push order and where the player can stand after each push. These are candidate transformations, never permission to assume unimplemented mechanics. [S1–S6, S10]

## What exact code should do

These are implementation requirements for the later harness, not game-playing code supplied by this study.

**Use the native transition engine.** A handwritten approximation often disagrees on text pushes, overlapping objects, simultaneous control, or the order of rule updates and destruction. If exact cloning or replay is allowed, use that engine for branch evaluation. Keep hypothetical branches separate from the live episode. If the evaluation allows only submitted turns and no simulator branching, restrict code to observation, bookkeeping, and permitted calculations; do not invent free undo. [Engineering recommendation from S2–S9]

**Represent future-relevant state.** Include all cell layers and multiplicities, object type/text literal, orientation, any per-object attributes, and dynamic state exposed by the implementation. Store level identity, step budget, RNG state if relevant, and terminal status. Rule text positions are needed even when active rules match: different word positions allow different future moves. A hash of the player position or rendered image is insufficient. Object identity/order may matter to engine resolution; canonicalize only after verifying that permutations preserve transitions. [Engineering recommendation from S2, S3, S9]

**Use exact small searches where they are economical.** Breadth-first search can solve tiny deterministic boards; A* or best-first search can handle larger subgoals when paired with careful state keys. Search the full puzzle if its state space is small; otherwise let Astra nominate rule/position subgoals and use bounded search to realize them. Maintain visited states and executable parent-action traces. Avoid revisiting a logical state with less remaining action budget when a dominating state exists, but only if time/phase and all other future dynamics match. No algorithm's success rate or runtime has been measured here. [Engineering recommendation; S10 supplies deadlock background, not evidence for a Baba implementation]

**Separate navigation from manipulation cautiously.** Static BFS on walkable cells is suitable only for a phase where walking does not push words/objects, change rules, trigger hazards or motion, alter relevant orientation, or affect another controlled object. Otherwise even a routine walk must use full transitions. Push-level search can reduce repeated walking on a Sokoban-like level, but reachable-region equivalence becomes unsound with multiple YOU, MOVE, or direction-sensitive dynamics. Macro actions must still be replayed as individual native actions and counted. [Engineering deduction from S3, S5, S9, S10]

**Use soft heuristics for changeable obstacles.** Useful estimates include distance to a currently viable WIN, pushes needed to access a target word, number of blocked prerequisites, and preservation of access to required pushing sides. Manhattan distance to the current flag is not generally an admissible lower bound: a nearer WIN can be created or control transferred. A heuristic that rewards keeping all current rules also misses deliberate rule destruction. If optimality matters, use only proved lower bounds; the user's completion score usually favors a practical bounded search over proving shortest solutions. [Engineering deduction from S1–S6]

**Prune only proven deadlocks.** Hard pruning is appropriate when the exact state cannot yield success under any allowed continuation, such as a fully frozen state with no control and no autonomous change. Treat “box in a corner,” “word against an edge,” and “goal unreachable under current rules” as warnings unless terrain, control, and relevant properties are provably fixed. Reversible-looking configurations may be dead; apparently dead configurations may become solvable through rule changes. A solver that silently applies Sokoban deadlock tables to all Baba states can discard the only solution. [S1, S2, S10; pruning policy is our recommendation]

**Verify and recover.** Store actual observations, primitive actions, active-rule changes, and terminal outcomes. If a plan fails, diagnose whether the cause was coordinate error, missed simultaneous movement, a lost rule, a blocked pushing side, or unsupported assumed physics. Only use undo/restart if exposed and authorized; charge restarts to the same scenario budget under the chosen evaluation policy. Count engine-confirmed success once, independently of reward shaping. [Engineering recommendation; S2–S9]

## What Astra should receive and decide

Give Astra a compact exact board legend, coordinates, layers, directions where relevant, active rules with word positions, action vocabulary, remaining budget, previous action/result, and the specific supported-mechanics list. Include the current hypothesis and any disproved assumption. Avoid a long transcript of unchanged boards or every search node.

Recommended planning instruction:

> Identify one feasible win condition and the rule changes needed to enable it. For each proposed word push, check the side from which it must be pushed and the space needed afterward. Track all YOU objects and every rule changed by that push. Preserve the ability to finish after each irreversible action. Choose a short next subgoal, state the expected rule/control changes, and supply the next action or a validated short sequence. Replan from the observed state whenever expectations fail. Use only the mechanics listed by the adapter; name uncertainty instead of assuming original-game features.

Prefer explicit subgoals such as “move STOP out of the WALL sentence while preserving BABA IS YOU” over “explore right.” Ask Astra to compare alternatives when a plan requires many irreversible pushes. A concept-level plan should still be rejected if the exact search cannot find its physical realization within the budget. When the current decomposition stalls, reconsider the intended controlled noun and goal noun rather than repeating movement around the same room. [S1–S6; prompt structure is our proposal]

After a failed puzzle, retain general lessons about verified mechanics and planner errors. Do not place held-out solution traces, level-specific answer scripts, or future evaluation identities into the strategy prior. [Evaluation hygiene recommendation]

## Optional Jev questions

Jev is unnecessary for timing. If used, batch independent recognition or ranking questions about already computed facts. Code supplies rule deltas, hazard facts, and candidate metadata; Jev must not become an approximate physics engine. Always allow `unknown` when the supplied state is insufficient.

| Typed question | Answer type | Proper use / limit |
|---|---|---|
| Which supplied bottleneck best describes this state: inaccessible rule word, blocked corridor, control transfer, or coordination of several YOU objects? | Choice plus `unknown` | Route the state to an Astra prompt pattern; do not treat the label as a proof. |
| Does the supplied verified rule delta remove the final current YOU rule without adding replacement control? | Yes / no / unknown | Optional redundant flag for Astra; code can and should calculate the underlying fact. No assertion that the state is permanently lost. |
| Which of these simulator-validated subgoals most directly advances the stated plan? | Candidate ID or `unknown` | Rank short candidates; code handles validity and arithmetic, Astra resolves strategic tradeoffs. |
| Is this observation consistent with the stated expected rule-change category? | Yes / no / unknown | Triage a mismatch already detected by code; never replace an exact state comparison. |

Do not ask Jev to solve a level, compute a push sequence, count shortest-path moves, infer collision order, or predict arbitrary transformations. With a symbolic board, most low-level recognition is cheaper and more reliable in code. Start with Astra plus exact state handling, and add Jev only if profiling shows a useful role. [Architecture recommendation based on the supplied model capabilities]

## Failure checklist and evaluation priorities

Before an irreversible push, check: required word not marooned; all simultaneous YOU movements accounted for; both rule-reading axes checked; needed pushing side still accessible; replacement control available when required; supported hazards resolved by engine; score tied to actual success. These checks belong in code where exact facts are available and in Astra's planning prompt for strategic consequences. [S2–S11]

Evaluate on a versioned manifest of distinct held-out puzzles, with fixed action budgets and no training/evaluation duplicates. Report solved count and solved fraction first, then steps, planning time, and failure category for diagnosis. Choose timeouts as evaluation policy rather than pretending the game has a real-time deadline. Tune search and prompts on development puzzles; never use known answer traces as a seed-to-solution lookup. [Evaluation recommendation; the user's requested objective is puzzles solved]
