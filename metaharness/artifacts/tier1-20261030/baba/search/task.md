Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Baba Is AI` — Baba Is AI is a fully observable, symbolic, turn-based puzzle. Push physical and rule tiles to change rules; an active '<object> is you' object is controlled, and touching an active '<object> is win' object solves the scheduled puzzle. The observation contains every stacked board object, active rules, coordinates, direction, and terminal status. Coordinates are (x,y), origin at the top-left. Actions are the five names in action_spec; each advances one turn. A puzzle scores exactly one point only when the native engine reports win; loss or step limit scores zero.

Actions:
- noop: Wait one game turn without moving.
- up: Move the controlled object one cell upward (north).
- right: Move the controlled object one cell right (east).
- down: Move the controlled object one cell downward (south).
- left: Move the controlled object one cell left (west).
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Submit one primitive action per turn. Invalid action values are ignored and do not advance the engine.

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
- strategy_summary: Inspect all active text rules, controlled objects, goals, and available word pushes before moving. Consider changing the goal, changing control, or removing blocking rules as alternatives to walking toward the current WIN object. Plan backward from a candidate win, preserve pushing access and a path to future control, and reassess every rule affected by a text push in both axes. Use the native engine for exact simulation or carefully bounded search where authorized; treat ordinary Sokoban deadlock heuristics as warnings until fixed-rule conditions are proved. Execute primitive turns with observation checks and replan whenever the actual transition differs from expectation. Astra owns subgoals and adaptation; code owns exact state, legal transitions, bookkeeping, and score. Jev is optional for typed candidate ranking, with no timing-critical control role.
- scoring: One point per scheduled puzzle if and only if the native engine confirms success, with at most one point per episode. Report total puzzles solved and attempted, solved fraction, seed/level identity, step count, and termination reason. Raw reward or truncation alone does not imply a solved puzzle.
- interface.observation: Prefer the environment's native symbolic observation, serialized as JSON with scenario identity, dimensions, every cell layer and word tile, active rules where exposed, directions and other future-relevant attributes, turn count, and separate success/termination/truncation fields. The exact upstream format is unverified; do not expose private engine fields if the evaluation contract intends partial observability.
- interface.actions: Submit one verified native action for each of north, east, south, and west; expose wait, undo, or restart only if the selected release and evaluation policy permit them. Reset by deterministic scenario seed. Exact enum values and signatures must be read from source; no integer mapping is assumed.

### study/strategies.md
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

### study/brief.md
# Baba Is AI — study brief

**Recommendation:** use the free Baba Is AI Python environment, with a symbolic state interface and one action per turn. Keep **tier 1**. Score completed puzzles, not movement, time spent thinking, or intermediate rules made.

## Evidence and scope

Study date: 2026-10-30. The user identifies the target as `baba-is-ai`, says seeds select puzzles, and specifies puzzles solved as the score. This is a benchmark/reimplementation inspired by **Baba Is You**, not a request to run the commercial game.

Verified locally: the evaluator `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/baba/bin/python` is **Python 3.12.3**, with only pip installed. `baba-is-ai`, Gymnasium, NumPy, and pygame are absent. No matching source or wheel was found in the searched workspace and local uv cache. HTTPS retrieval of PyPI and GitHub failed with `curl: (6) Could not resolve host`. Consequently **no upstream web page, package metadata, source implementation, or benchmark API was inspected successfully**. All web links below are reference/discovery targets, not claims of live retrieval. General game knowledge is useful planning guidance; exact benchmark mechanics must be checked against its source. `interfaces.md` records the integration blocker rather than inventing environment IDs, action integers, or dependencies.

## Objective and rules

Baba Is You is a grid puzzle in which movable words form the rules of the current board. A sentence such as `BABA IS YOU` gives Baba objects player control; `FLAG IS WIN` makes flags goals; `WALL IS STOP` makes walls block movement. A flag is not intrinsically a goal, a wall is not intrinsically solid, and Baba is not intrinsically the player. Changing the words changes these properties. Official game reference: https://www.hempuli.com/baba/ . Community reference targets: https://babaiswiki.fandom.com/wiki/Rules and https://babaiswiki.fandom.com/wiki/YOU .

Basic sentences read left to right or top to bottom in contiguous aligned cells. Text is normally pushable. Word tiles and the objects they name are distinct: moving the word `ROCK` is not moving an ordinary rock. One word may participate in both a horizontal and a vertical sentence. Text arrangements can therefore modify several rules at once. More elaborate grammar exists in the original game; do not assume the benchmark supports it. Reference: https://babaiswiki.fandom.com/wiki/Rules .

The standard win condition is an object with `YOU` contacting an object with `WIN`; a controlled object can sometimes itself gain `WIN`. Use the benchmark's actual success signal and interaction ordering as authority. Movement commands act on all currently controlled objects, not a selected individual. Push chains, blocking, destructive interactions, and rule changes can make a single command affect much of the board. Reference targets: https://babaiswiki.fandom.com/wiki/WIN and https://babaiswiki.fandom.com/wiki/PUSH .

Typical original-game properties include `STOP` (blocking), `PUSH` (pushable), `DEFEAT` (destroys contacting controlled objects), `SINK` (destructive overlap), `HOT`/`MELT`, `OPEN`/`SHUT`, and `MOVE` (automatic movement on game turns). `X IS Y` may transform objects rather than give a property. These are **a vocabulary to look for, not a verified list of Baba Is AI features**. Exact supported words, transformations, overlap rules, and resolution order belong to package inspection. Do not silently import the full commercial rule system into a small benchmark.

## Controls and pace

| Operation | Intended meaning | Benchmark status |
|---|---|---|
| North / east / south / west | Issue one grid movement command to controlled objects | Map to source-defined enum; integers unverified |
| Wait | Advance a turn without directional movement; useful with moving objects | Expose only if the installed environment supplies it |
| Undo | Restore a previous turn in the commercial game | Not assumed available or legal in this benchmark |
| Restart | Reset a puzzle | Evaluator policy must define whether it is allowed and how actions are counted |
| Reset with scenario seed | Begin an evaluation puzzle | User requirement; native seed-to-level behavior unverified |

The world waits between submitted turns. Automatic objects, if supported, advance on turns rather than while Astra thinks. Seconds or a minute of reasoning do not intrinsically harm the position. A move limit, step-shaped reward, evaluator wall-clock timeout, or model budget can still penalize excessive work; those are separate constraints. Polling or reading a state must never advance it. Failed movement and wait actions may still advance the engine and consume the action budget.

`study.json` uses **`decision_latency_s: -1.0` as an explicit sentinel for no finite in-game thinking deadline**. JSON has no standard infinity literal. It does not mean that the evaluator grants unlimited wall-clock time. The adaptor should translate this sentinel into “no real-time controller needed.”

## Score and progress

For the requested evaluation, a puzzle contributes **1 exactly once if the engine reports success, otherwise 0**. Report total solved and number attempted; also report solved/attempted when comparing runs with different numbers of puzzles. Record seed, selected level, step count, termination reason, and success independently of raw environment reward. A positive shaped reward or `terminated=True` alone is not proof of a win. Truncation is not success unless the same final transition explicitly establishes success.

Intermediate milestones—access to a word, a newly formed useful rule, or reaching a new room—help planning but do not increase the official score. Multiple completions after replaying the same evaluation puzzle must not inflate score. Seeds must select reproducible puzzles from a versioned manifest, not merely reseed a fixed level's random-number generator. Details and leakage controls are in `interfaces.md`.

### adaptor/ADAPTOR.md
# Baba Is AI adaptor

This adaptor runs the open-source `nacloos/baba-is-ai` engine in process, headlessly and offline. The adaptor vendors the inspected upstream `baba` package source (version 0.0.1) under `baba/`; its runtime dependencies are listed in requirements.txt. No GUI, server, account, or credentials are used.

`reset(seed)` maps declared seeds to a versioned, sorted manifest of native environment IDs. Train, held-out, dev, and validation groups use disjoint environment IDs (held-out and validation use the same difficulty family but different from training). Seeds outside the declaration select from the full manifest by stable modulo mapping. A seed therefore selects a native puzzle template/configuration; the engine's own RNG, seeded with that integer, instantiates the board.

The game is Tier 1. `step` accepts `noop`, `up`, `right`, `down`, `left` (or native integers 0–4), applies exactly one native turn, and returns the next structured observation. Invalid actions are ignored and reported in `last_error`. Observations preserve every object stack in `(x,y)` cells, active rules, agent position/direction, step count, and terminal state.

Score is one point only when the engine's `is_win` predicate is true. Losses and step-limit termination score zero. `score_breakdown()` reports solved count, attempted count, and solved fraction; `info()` includes seed, level identity, step count, and termination reason. The finite manifest is intentionally limited, so arbitrary seeds can collide modulo its size.

The adaptor does not start background processes. `close()` calls the engine close hook when present and is idempotent.

### study/interfaces.md
# Programmatic interfaces and adaptor specification

## Recommendation and present blocker

Choose the **Baba Is AI Python environment in process**, using its native transition engine and lossless symbolic board state. It fits Linux without a GUI, offline execution, exact turn boundaries, and the requested benchmark. Prefer its public environment API; use documented/source-inspected internals only where its public observation omits essential information.

**This choice is conditional on obtaining the actual package source.** No installation or runtime validation was possible during this study. The target venv exists and runs Python 3.12.3, but contains only pip. The session cannot resolve PyPI or GitHub; no local Baba distribution was located. It also permits writes only in the study directory and `/tmp`, so the requested external venv is not writable through this session's allowed filesystem scope. No installation was attempted there and no request to override these restrictions is needed for writing this study.

The exact upstream owner, release, license, import name, registration IDs, API signatures, action enum, observation encoding, success flag, and native level-selection method remain unverified. This document supplies a concrete integration contract and inspection procedure, but cannot honestly supply a tested invocation without those artifacts. **The next phase must resolve this source-acquisition blocker before claiming a working adaptor.** Do not substitute a similarly named clone or make up a Gym ID.

## Interface comparison

Latencies below describe expected architecture, not measurements.

| Interface | Maturity / availability | Observation and control | Latency | Cost and legal considerations | Decision |
|---|---|---|---|---|---|
| `baba-is-ai` Python package, public environment API | User-designated target; exact release and maintenance unverified | Expected native reset/step plus board observation; actual protocol must be inspected | Synchronous local calls; no network or input/render loop required | Intended free route; confirm package, assets, and level licenses | **Chosen**, subject to source verification |
| Same package's direct engine/grid access | Depends on source and release; internal interfaces may change | Full object layers, words, rules, directions, win state; possible snapshot support | Local memory access | Same licensing; pin source commit to prevent drift | Supplement public API if needed; never maintain a second rules engine unnecessarily |
| Another free Baba-like implementation or research bot library | Different implementations exist; none verified here as a drop-in replacement | Potential symbolic simulator, solver API, or browser state | Local simulation can be quick; browser bridge adds overhead | Audit repository and asset licenses; mechanical and level mismatch is the main risk | Only with explicit benchmark equivalence; not an automatic fallback |
| Commercial Baba Is You with mod/plugin or internal scripting | Mature game, but no supported benchmark automation API verified | Potential internal level/object access through a mod; requires exact version integration | Local, but input timing and frame loop matter | Paid proprietary game and asset/mod redistribution questions; fails free-software constraint | Reject for this task |
| Commercial game with screen capture and input injection (Xvfb, capture, keyboard tools) | Generic automation is mature; game recognition and headless setup are custom | Pixels and key presses; hidden overlap/rules hard to recover | Rendering/capture/input synchronization overhead; no need for Jev just because it has frames | Paid game; platform/automation terms require review; no online competitive anti-cheat expected in local puzzles | Reject: unnecessary GUI and fragile perception |
| Browser clone controlled through JavaScript/Playwright | Possible for a compatible free clone, but none validated here | Canvas screenshots, DOM/JS state if exposed, discrete key input | Browser startup and bridge overhead | Browser itself free; clone code/assets/levels need separate license review | Different game unless exact benchmark assets and mechanics match |
| Emulator hooks | No relevant native emulator interface identified for the requested Python benchmark | Would expose machine memory/input, not automatically meaningful puzzle state | Extra integration and interpretation cost | Distribution rights depend on the game image | No practical reason to use |

Do not confuse “a solver for Baba Is You” with “the environment used by this evaluation.” Solvers can suggest search techniques, but their level syntax and physics may differ.

## Acquisition, software, and offline installation

Use **only** this interpreter for environment inspection, installation, and final checks:

`/home/hemlix/GamingMaster-mh-tier1/runs/venvs/baba/bin/python`

Required distribution: `baba-is-ai`, or the upstream source package identified by its package metadata. Required transitive dependencies must come from that release's `Requires-Dist` / `pyproject.toml`; **Gym versus Gymnasium, NumPy, pygame, Pillow, and their versions are not established**. Do not install a large guessed stack. Git is needed only for a Git checkout, not for an already acquired wheel/archive. No browser, display server, emulator, paid game, GPU, or account is part of the chosen interface.

Source/discovery URLs (unretrieved in this session):

- Package page: https://pypi.org/project/baba-is-ai/
- Machine-readable metadata: https://pypi.org/pypi/baba-is-ai/json
- Repository discovery: https://github.com/search?q=baba-is-ai&type=repositories

…(truncated; full text in prior/interfaces.md)

Repeated seeds must reproduce initial state and action outcomes. Distinct seeds can collide on a finite corpus; report distinct level count and never claim that infinitely many seeds mean infinitely many puzzles. Partition held-out levels by identity or template **before** assigning seed ranges. Prevent the same static layout or cosmetic variant appearing on both sides and being called generalization. Do not randomly scramble words or walls without a solvability guarantee. No invented procedural generator is needed if the benchmark supplies a finite corpus.

Score one point per scheduled puzzle solved, at most once per episode. Total solved is the primary user-requested score; solved fraction and attempts accompany it. Read explicit success or the inspected engine win predicate; never guess from reward positivity. Step-limit exhaustion, no-control deadlocks, and failure terminations score zero. Log underlying reward separately if useful. Restart attempts within one scheduled puzzle share its total budget unless evaluation specifies otherwise. Make comparisons on the same scenario manifest and action budgets.

## Acceptance checks for the next phase

These are required verification tasks, **not results already observed**:

1. Import and headless construction work in the exact Python 3.12.3 venv; `pip check` passes; assets resolve without network or a working directory dependency.
2. Enumerate real environments/levels and supported rules. Confirm a reset produces a valid board and that selected seeds cover multiple intended puzzles.
3. Observe twice with a delay and verify unchanged logical state. Verify each directional enum on a known small board. Confirm blocked commands and wait semantics.
4. Compare serialized coordinates, every layer in occupied cells, directions, and active rules against engine state. A screenshot may be used as an optional diagnostic, never the sole ground truth.
5. On a supplied trivial fixture/reference trace, confirm the terminal win transition scores exactly one. Separately verify timeout, loss, and repeated post-terminal calls cannot add points.
6. Replay identical seed/action traces in fresh instances and compare states. If snapshots are used, prove branch mutations do not affect live state and restore includes all future-relevant state.
7. Check rule changes immediately affect control/blocking as the source specifies. Test only supported interactions, with special attention to transformation, stacked objects, and simultaneous movement.
8. Persist package hashes, level manifest, seed mapping version, action mapping, resolved dependencies, and actual API facts beside the adaptor.

Completion of these checks is the gate for calling the adaptor operational. The study's recommended contract must not be mistaken for a verified upstream API.


## Measured model latency (at archive creation, 2026-10-30 19:31)
- System-2 (Astra), one-word answer: 19 s
- System-2, an answer of ~946 output tokens: 64 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).