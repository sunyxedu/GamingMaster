Design a harness for a **real-time** game (tier 2).

Game: `Overcooked-AI` — Overcooked-AI symbolic gridworld, pinned to HumanCompatibleAI/overcooked_ai commit 739950a079cdaed5a44fcc662efc40244c205d06. This is a realtime two-chef cooperative kitchen: the world advances exactly one MDP transition every 0.166667 seconds, for at most 400 ticks. You control one chef; a deterministic scripted partner controls the other, and the observation says which index is yours. Walkable cells are spaces in terrain_mtx; O/T/D/P/S/X mean onion/tomato/dish dispenser, pot, serving window, and counter. A chef faces the direction of its last movement. INTERACT operates the feature in the facing cell: collect an ingredient/dish, place an ingredient in a pot, start cooking under modern dynamics, take ready soup with a dish, place/pick up an object on a counter, or deliver soup. Pots cook automatically after explicit start; ingredients and recipes are defined by state.all_orders, and a delivery is worth the active recipe value (bonus_orders can multiply it). The score is cumulative sparse team delivery reward only; shaped subgoal rewards are diagnostics and never added to score. The complete JSON observation includes the terrain, serialized players and objects (including soup ingredients/cook state), orders, tick, sparse reward, recent events, action history, partner task, and diagnostics. Use exactly one of the six advertised primitive action names per tick; directional actions are not multi-step plans. A missed action repeats the previous valid action, while malformed input becomes stay. Seeds 0/1/2 select cramped_room, asymmetric_advantages, and coordination_ring; held-out seed 100 validates counter_circuit_o_1order, while later held-out seeds cover explicit handoff, mixed tomato, counter-circuit, reversed-agent, and randomized-start cases.

Actions:
- north: Move one walkable tile north; facing becomes north.
- south: Move one walkable tile south; facing becomes south.
- east: Move one walkable tile east; facing becomes east.
- west: Move one walkable tile west; facing becomes west.
- stay: Do not move and keep the current facing direction.
- interact: Operate the feature immediately in front of this chef (pick up, put down, fill/start a pot, or serve).
(real-time games: ticks apply 'stay' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
One primitive is applied per 1/6-second tick. A missing decision repeats the last valid action; a malformed action is replaced by stay and is reported in the next observation. There are no action parameters and no commercial-game mechanics: this is the pinned Overcooked-AI grid MDP.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.16666666666666666s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
900 s (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals. Clock mode: wall.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Treat each accepted soup as a pipelined job: reserve a recipe and pot, fetch ingredients, fill it, explicitly start cooking under modern dynamics, stage a dish, pick up the ready soup, and deliver. Code should own exact state bookkeeping, recipe/value arithmetic, reachability, collision-safe shortest paths, interaction preconditions, timers, and stale-action rejection. Astra should allocate roles, choose among near-equal recipes, set circulation and handoff protocols, and adapt when a layout or partner violates the initial plan. Jev should answer small typed recognition/choice questions over legal code-generated task candidates, such as pot phase, delivery urgency, traffic priority, or handoff status; it should not emit raw directions or do arithmetic. Give ready-soup delivery and traffic recovery priority over speculative ingredient work, and use a custom protocol rather than stock greedy behavior on forced-coordination or mixed-recipe layouts.
- scoring: Report cumulative sparse team reward: sum of env.step's sparse_r_by_agent or env_info episode ep_sparse_r over the 400-tick episode. This counts only accepted soup deliveries and is the true benchmark objective; do not add shaped rewards. Also report deliveries, score by agent, episode length, event counts, collision/stuck ticks, pot idle time, and handoff failures as diagnostics. In this environment an ordinary accepted recipe is commonly worth 20, bonus orders use order_bonus*recipe.value, and layouts may override recipe values/times, so compute values from the active state/layout rather than hardcoding.
- interface.observation: Canonical observation is a JSON-safe snapshot containing episode_id, layout_name, terrain_mtx, horizon, state.to_dict(), cumulative sparse score, last sparse reward, sparse_r_by_agent, shaped_r_by_agent, event flags, partner's last action/task, controlled_agent_index, current tick, and wall-clock deadline. state.to_dict() contains players with (x,y) position, orientation, held object, unheld objects including soup ingredients/cooking tick/ready flags, bonus_orders, all_orders, and timestep. Include terrain_mtx because the state dictionary alone does not identify counters, dispensers, pots, or serving tiles. The environment also provides lossless_state_encoding_mdp(state) and featurize_state_mdp(state), but those tensors are optional policy inputs rather than the Astra/Jev canonical representation.
- interface.actions: The adaptor controls one chef with one primitive per tick and supplies a joint action to OvercookedEnv.step((action0, action1)). Native actions are Direction.NORTH=(0,-1), SOUTH=(0,1), EAST=(1,0), WEST=(-1,0), Action.STAY=(0,0), and Action.INTERACT='interact'; Action.ALL_ACTIONS orders them as north, south, east, west, stay, interact. INTERACT operates the feature immediately ahead. The partner implements Agent.action(state) -> (action, info), normally via GreedyHumanModel(env.mlam) on standard three-onion layouts. Use MotionPlanner/MLAM to turn high-level tasks into safe primitive actions, replan after interactions/collisions, and validate all preconditions in code. A missing model reply becomes STAY rather than blocking the clock.

### study/strategies.md
# Overcooked strategy prior

Target is the symbolic HumanCompatibleAI/Overcooked-AI environment, not commercial Overcooked's order-clock/combo game. The useful unit of strategy is a repeated soup-production pipeline under a 400-tick horizon. Strong play maximizes delivered accepted recipes per tick while keeping pots, counters, hands, and traffic moving. The document labels where each idea belongs:

- **CODE**: deterministic bookkeeping, geometry, legality, path search, state machine, and exact arithmetic.
- **ASTRA**: high-level allocation, role/route planning, unusual-layout adaptation, recovery, and partner-model updates.
- **JEV**: fast typed recognition/choice over a compact JSON snapshot; never ask it to invent plans, perform long arithmetic, or replace collision-safe path code.

Sources: official [README](https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/README.md), [MDP implementation](https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_mdp.py), [agents](https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/agent.py), [planners](https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/planning/planners.py), [layouts](https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/layouts), and the original [benchmark paper](https://arxiv.org/abs/1910.05789).

## 1. Universal production loop

For each accepted recipe, the physical loop is:

1. Choose a recipe and pot; reserve both.
2. Fetch the missing ingredients from dispensers or counters.
3. Add ingredients in a recipe-preserving order.
4. Start cooking with an empty-handed interaction under modern dynamics.
5. During cooking, fetch a dish and position the delivery chef.
6. When ready, face the pot with the dish and interact.
7. Carry soup to a serving tile and interact.
8. Immediately launch the next pot, using the partner's state rather than returning to a fixed role script.

In the default three-onion layout, a soup needs three onion pickups, three pot interactions, one explicit start-cooking interaction, 20 cooking ticks, one dish pickup, one soup pickup, a path to `S`, and delivery. Cooking advances during the same transition after interactions/movement; a soup started at tick `t` is ready after its configured cooking increments, not before. This makes it valuable to start the pot as soon as the final ingredient lands, then exploit the cooking window for dishes/ingredients. [MDP]

**CODE — production state machine.** Maintain exact per-pot state: `empty`, idle with ingredient count, `cooking` with `cook_time_remaining`, or `ready`. Maintain each player's held object and reserved task. Do not ask a model whether a soup is ready when `SoupState.is_ready` and `_cooking_tick` are available. Reject an interaction unless the facing tile and hand/pot preconditions are true; otherwise treat it as a no-op and replan.

**ASTRA — batch/role allocation.** At the beginning of an episode and after major events, choose which chef handles ingredient runs, which handles dishes/cooking/delivery, and which pot is next. Reallocate when one chef is trapped, a soup is nearly ready, a counter handoff is advantageous, or the partner's behavior differs from its assumed role. Plan a rolling pipeline, not a single soup followed by idle time.

**JEV — event recognition.** Ask typed questions such as:

```json
{"question":"choice","choices":["fetch_ingredient","fill_pot","start_cooking","fetch_dish","pickup_ready_soup","deliver","unstick","wait"],"state":{...}}
```

Only ask after code has generated legal candidate tasks. A second question can classify `pot_phase` as `empty|idle_partial|idle_full|cooking|ready` or identify `urgent_ready_pot` from the compact state. Jev should not choose a raw direction.

## 2. Openings and build orders

### A. Cramped Room — simple three-onion soup

The compact 5-by-4 map has a top pot, two onion dispensers near the top corners, the chefs near the center, dish dispenser at the bottom-left, and serving station bottom-right. The best robust opening is asymmetric:

- Chef A takes the ingredient/production side: collect an onion and place it; continue with the next onion rather than both chefs contesting the same dispenser.
- Chef B clears traffic, fetches a dish only when a pot is cooking/nearly ready, and becomes the first delivery runner.
- After the third ingredient, interact empty-handed to start cooking immediately (unless the layout explicitly sets old dynamics).
- While the pot cooks, Chef A can start the next pot or fetch an ingredient; Chef B obtains a dish, waits near the pot/serving route, then picks up and delivers.
- After delivery, swap roles if the current positions make the next assignment cheaper. Never return both chefs to their original sides merely for symmetry.

The stock `GreedyHumanModel` follows a related priority: without an object it fetches a dish if a soup is ready/cooking and no partner has a dish; otherwise it starts a suitable pot or fetches onions; with an onion it targets a pot; with a dish it targets a nearly-ready soup; with soup it delivers. It can be an effective partner baseline on this layout, but its own source asserts that the sole order is three onions. [Agents]

**CODE:** shortest-path movement to interaction poses; orientation-aware goal representation; exact `start_cooking` interaction; pot reservations; collision/stuck detector; score and event ledger.

**ASTRA prompt:** “Assign A/B roles for the next two soups. Prefer parallel ingredient collection and dish/delivery, avoid both chefs entering the same one-cell approach, and state the handoff/role-switch trigger.”

**JEV questions:** “Is a dish already held?”, “Is any pot ready?”, “Is the partner blocking the only approach?”, “Which of these legal medium tasks is urgent?”

### B. Asymmetric Advantages — exploit side specialization

This map deliberately gives chefs different access/approaches to resources. Do not force both agents to perform every subtask. At reset, compute which chef has lower motion-planner cost to each dispenser, pot, dish station, and serving station. Assign the nearest chef to repeated trips, while the other handles the complementary resource and makes counter handoffs where paths cross. A side specialist can feed a pot while a delivery specialist stages dishes and soup. If the kitchen bottleneck is a narrow crossing, schedule crossings rather than having both chefs greedily select the same nearest goal.

**CODE:** calculate a cost matrix using `MotionPlanner`; cache route lengths; mark inaccessible goals as infinity; track which resource side each chef owns; detect when a route is blocked by the partner.

**ASTRA:** decide whether specialization or alternating batch work is faster after seeing the first soup; formulate a protocol such as “ingredient specialist supplies pot, delivery specialist never takes onions unless all pots are empty.”

**JEV:** classify the current bottleneck (`ingredient|pot|dish|delivery|traffic`) and whether a handoff is beneficial from a small candidate list.

### C. Coordination Ring — travel/crossing discipline

The ring puts resources around separated corners. A naive “nearest feature” policy can create long loops and collisions. Assign a clockwise or counterclockwise circulation convention at reset. Reserve one lane/approach for the carrying chef and avoid crossing the partner at the central choke point. The chef who reaches the pot first can launch cooking; the other should use the cooking delay to fetch the dish and stage at the ready-pot approach, not make an uncoordinated second ingredient trip.

Use route reservations only as soft priorities: if a soup is ready, delivery has precedence over a low-value ingredient trip. When two planned paths conflict, have the lower-urgency chef STAY or take a legal detour; never allow two simultaneous opposite moves through a one-cell gate. The MDP itself resolves a collision by keeping both positions, which can silently waste multiple ticks. [MDP]

**CODE:** A* / `MotionPlanner` shortest paths, time-expanded conflict checks for the next few steps, priority labels (`ready_delivery > start_cooking > fill_reserved_pot > dish_stage > ingredient > idle`), and automatic replan after an actual collision.

**ASTRA:** choose circulation direction and lane policy, then update it if the first cycle creates deadlock.

**JEV:** yes/no “would these two next moves collide or swap?”, “is delivery urgent?”, and choice among safe high-level detours generated by code.

### D. Counter Circuit — pass ingredients across counters

`counter_circuit` is not merely a longer room: it has a counter barrier and mixed orders. The layout's current file specifies onion/tomato recipes, a bonus onion-tomato order, onion value 21, tomato value 13, onion cook time 15, and tomato cook time 7. The source also supplies `counter_circuit_o_1order`, a simpler one-order variant. Read those parameters at runtime. [Layouts]

A strong plan uses counters as a conveyor:

1. A supply-side chef picks up an ingredient and deposits it on a counter accessible to the other side.
2. The receiving chef picks it up, carries it to the correct pot, and deposits it.
3. Batch compatible ingredients to reduce crossings, but never fill all counters with low-priority ingredients if a dish/soup handoff needs space.

…(truncated; full text in prior/strategies.md)

- accepted and bonus recipes with value/time;
- current tick, horizon, sparse score, and last transition events;
- partner's last task/action and reliability statistics;
- candidate task list with exact motion cost, preconditions, and conflict flags.

Ask Astra for a **short-horizon allocation**, not primitive motor commands:

> “At tick T, choose role/task assignments for the next event boundary. Candidates are already legal and have exact costs. Protect any ready soup, maintain at least one usable counter for handoff, and state which event invalidates the plan. Return JSON: `{assignments, priority, handoff_protocol, replan_triggers}`. Do not output directions.”

Re-query at reset, delivery, pot start/finish, failed interaction, collision streak, partner deviation, or when no queued medium-level task remains. Do not invoke it every tick.

## 7. Typed Jev question bank

Run independent questions in one batch over the same JSON state. Keep outputs closed-world and small:

1. `pot_phase(pot_id) -> empty | idle_partial | idle_full | cooking | ready`
2. `urgent_delivery -> yes | no`
3. `partner_holding_dish -> yes | no`
4. `partner_holding_soup -> yes | no`
5. `blocked_or_stuck(agent) -> yes | no`
6. `handoff_status(counter) -> free | staged | occupied_wrong_type | acknowledged`
7. `task_choice(candidates) -> candidate_id`
8. `traffic_priority -> self | partner | yield`
9. `ingredient_matches_recipe -> yes | no`
10. `replan_needed -> yes | no`

Candidates must be generated by code and include IDs, exact precondition booleans, and no more than a few alternatives. Jev's output is advisory: the adaptor revalidates it against the current tick/state and falls back to a safe candidate.

## 8. Search/evaluation plan

### Baselines

Measure at least:

1. STAY/random controls (sanity floor).
2. `GreedyHumanModel` paired with itself on `cramped_room`.
3. Controlled agent with shortest-path finite-state production paired with stock greedy.
4. Astra/Jev hierarchical controller paired with stock greedy.
5. Same controller on held-out layouts and controlled-agent index reversal.

Report mean and standard error over many seeds, cumulative sparse score, deliveries, episode length, and event counts. Also report useful diagnostics: collisions, unchanged-position streaks, wasted interactions, pot idle time, soup pickup-to-delivery ticks, and handoff timeout count. The official `AgentEvaluator` and `env.game_stats` provide relevant event/stat patterns. [Benchmarking]

### Scenario split

Train/reason on `cramped_room`, `asymmetric_advantages`, and `coordination_ring`; validate on `counter_circuit_o_1order`; hold out `forced_coordination`, `counter_circuit`, tomato/mixed variants, and randomized valid starts for the final test. Do not tune to a single starting side. Vary controlled-agent index, initial positions where the layout permits, recipe configurations, and partner style (stock greedy, deterministic role script, mild stochastic/unstuck partner).

A meaningful seed should control: scenario/layout selection, valid start-position choice, orientations if randomized, partner tie-breaking, and any object-initialization randomization. Keep a fixed list of test seeds and do not expose it to Astra's prompt. Record the exact upstream commit and layout files.

## Sources

- Rules, rewards, transitions, collision handling, pot states, and state encodings: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_mdp.py
- Action symbols and ordering: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/actions.py
- Environment horizon, `step`, reset, reward/event info: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_env.py
- Scripted agents and greedy high-level priorities: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/agent.py
- MLAM and motion planner: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/planning/planners.py
- Layout configuration: https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/layouts
- Benchmark design and 400-timestep evaluation: https://arxiv.org/abs/1910.05789
- Official project overview/install: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/README.md
- Evaluation helpers/trajectory stats: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/benchmarking.py

### study/brief.md
# Overcooked — study of the Overcooked-AI target

Research date: 2026-10-30. Target: **HumanCompatibleAI/overcooked_ai**, not the commercial Overcooked executable. Inspected upstream revision: `739950a079cdaed5a44fcc662efc40244c205d06` (package metadata version 2.0.0). The requested environment is local, CPU-only, headless Linux, symbolic observations, with one controlled chef and one scripted teammate. See `interfaces.md` for installation, exact API, compatibility evidence, and the required real-time driver.

## Objective and score

Two chefs cooperate to produce and deliver soup. In the recommended initial benchmark, the only accepted recipe is three onions, cooking takes 20 **simulation ticks**, and each accepted delivery earns 20 shared points. Run a fixed 400-tick episode; maximize **sum of the sparse team reward returned by `OvercookedEnv.step`**. Count both chefs' deliveries, not just the controlled chef's. The original benchmark also evaluates cumulative reward over 400 timesteps. [S1–S3]

There is no win/lose threshold or native star progression in this simulator. Its MDP has no terminal states; the environment supplies the finite horizon. Different layouts can specify ingredient-dependent recipe values/times and static bonus recipes. Read actual configuration rather than assuming every soup is worth 20. `all_orders` is the allowed recipe set, **not a FIFO queue**. Serving does not remove recipes. A disallowed soup is consumed at delivery but gives zero reward. Bonus recipes receive `order_bonus * recipe.value` (default multiplier 2). [S2]

Shaping signals are diagnostic/training aids, not the score: the default configuration awards 3 for potting an ingredient, 3 for a useful dispenser dish pickup, and 5 for soup pickup. Do not add these to the reported sparse score; incomplete busywork must not outrank completed meals. [S2–S3]

## Rules that must be implemented faithfully

1. Walk on floor cells. Terrain codes: space = floor; `X` = counter/wall; `O` = onion dispenser; `T` = tomato dispenser; `D` = dish dispenser; `P` = pot; `S` = serving station. Layout digits identify initial players and become floor. Coordinates are `(x,y)`, origin top-left, x rightward, y downward; `terrain_mtx[y][x]`. [S2,S4]
2. A chef carries at most one object: onion, tomato, dish, or soup. Dispensers are unlimited. Face an adjacent dispenser and interact with empty hands to collect its object. [S2]
3. Face a pot with an ingredient and interact to deposit it into an idle, non-full soup. Ingredients cannot be removed through a player action; cooking locks the mixture. Default capacity is three. [S2]
4. **Modern default `old_dynamics=False`: depositing the third onion does NOT start cooking. Interact again with empty hands to start.** Starting a 1- or 2-ingredient soup is mechanically possible but yields no reward in the three-onion-only benchmark. With explicitly selected `old_dynamics=True`, three ingredients trigger cooking automatically; do not confuse these versions. [S2,S5]
5. Cooking advances after interactions and movements on every joint tick, including all-STAY ticks. A newly started soup advances to cooking tick 1 in that same transition. It becomes ready at its recipe's cook time. A dish interaction on the transition that finishes cooking is too early: interactions precede the final cooking increment. [S2]
6. Face ready soup while carrying a dish and interact. The dish is replaced by the soup in hand and the pot becomes empty. Deliver by facing `S` and interacting. No dish washing or reusable plate return exists. [S2]
7. Counters hold one object. Interact to put down a held object on an empty counter, or retrieve an object with empty hands. No direct hand-to-hand transfer, floor drop, or throw is available. Cooperative transfers use shared counters. [S2]
8. Ready soup does **not** burn or expire. No chopping, dash, fire, extinguishers, moving platforms, dirty dishes, order-expiry penalty, tip combo, or commercial star thresholds are present in the selected core environment. Its urgency comes from limited ticks, travel, pot occupancy, and coordination. [S2–S4]

## Actions and collision rules

| Symbolic action | Native action | Index at pinned revision | Effect |
|---|---|---:|---|
| NORTH | `(0,-1)` | 0 | Move up / face up |
| SOUTH | `(0,1)` | 1 | Move down / face down |
| EAST | `(1,0)` | 2 | Move right / face right |
| WEST | `(-1,0)` | 3 | Move left / face left |
| STAY | `(0,0)` | 4 | Wait without changing orientation |
| INTERACT | `"interact"` | 5 | Operate the feature immediately in front |

Use `Action.INDEX_TO_ACTION` rather than hardcoding another library's ordering. All six actions are formally legal everywhere; an ineffective interaction is a no-op, not an invalid action. A directional action into a blocked tile still changes orientation. A successful move also changes orientation. Moving/turning and interacting are separate actions. [S2,S4]

Both chefs act on the same pre-transition state. Interactions resolve first in player-index order (0 then 1), then simultaneous movement, then cooking/timestep advancement. If attempted movement puts chefs on one cell or swaps their positions, both retain old positions; their attempted orientations can still change. Moving into the cell a partner vacates is possible when it is not a swap. Player-order resolution matters when both interact with one pot or counter. [S2]

## Pace and tier decision

**Retain tier 2, with an explicit real-time wrapper contract.** Native `OvercookedEnv` is synchronous: it advances only when called. It does NOT autonomously tick while Astra thinks. A bare blocking `observe -> model -> step` loop would therefore be tier 1 and must not be presented as tier 2. [S2–S3]

To preserve the requested tier, the adaptor must run the simulator in an independent ticking worker. Recommended cadence: **6 ticks/second**, based on the official demo's `play_game(..., fps=6)` setting, with 400 logical ticks per round (about 66.7 seconds). That exact fixed-horizon, nonblocking worker is our adaptor design, not an existing native environment feature. The demo itself can block for an NPC action and is not a drop-in guarantee of independent ticking. [S6–S7]

…(truncated; full text in prior/brief.md)

- Astra may take seconds to a minute. Give it layout analysis, role allocation, traffic policy, and event-triggered strategic revision; never await Astra on the tick thread. A robust default task executor must already be running.
- Precompute planners and initialize partners before starting the clock. Reject stale decisions by episode ID, observation tick, target validity, and task preconditions.
- In debug/replay mode, pause or step synchronously, but explicitly label that mode non-real-time and do not use it to measure tier-2 performance.

## Initial scenarios

Start with `cramped_room`, `asymmetric_advantages`, and `coordination_ring`, using the upstream `GreedyHumanModel` as the scripted partner and modern explicit-start cooking. Add `counter_circuit_o_1order` as a harder connected layout after testing. Do not confuse it with `counter_circuit`, whose current configuration has mixed recipes and nonstandard values/times. Exclude `forced_coordination` from the default stock-partner pool: its isolated supply/cooking areas require deliberate handoffs that the stock greedy partner explicitly does not guarantee. [S8–S9]

Use seeded layout selection, controlled-player index, valid initial positions/orientations, and partner randomization. A seed is not meaningful if it resets to an identical deterministic episode every time. Full seeded-reset and evaluation rules are in `interfaces.md`.

## Primary sources

- S1: Official project/benchmark description: https://github.com/HumanCompatibleAI/overcooked_ai
- S2: Pinned rules/state/reward implementation: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_mdp.py
- S3: Pinned environment API/horizon/statistics: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_env.py
- S4: Pinned actions: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/actions.py
- S5: Core tests, including old/new cooking dynamics: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/testing/overcooked_test.py
- S6: Demo clock: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo/server/app.py
- S7: Demo action queues/NPC blocking: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo/server/game.py
- S8: Agent contracts and greedy-partner restrictions: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/agent.py
- S9: Layout definitions: https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/layouts
- S10: Carroll et al., On the Utility of Learning about Humans for Human-AI Coordination: https://arxiv.org/abs/1910.05789

### adaptor/ADAPTOR.md
# Overcooked adaptor

This adaptor connects the Meta-Harness to the local, open-source
`HumanCompatibleAI/overcooked_ai` symbolic MDP at commit
`739950a079cdaed5a44fcc662efc40244c205d06`. It does not launch the commercial
Overcooked game, a GUI, a server, or an account. The installed evaluator Python
imports the upstream `OvercookedGridworld` and `OvercookedEnv` directly, with
`SDL_VIDEODRIVER=dummy` for headless pygame imports.

## Interface and clock

`make_game()` returns a tier-2 `RealtimeGame`. The upstream environment is a
synchronous transition oracle, so the adaptor exposes one real MDP transition
on each `tick(action)` at `tick_s = 1/6` seconds. `ticks` is the authoritative
`OvercookedState.timestep`; the harness runner owns the wall-clock ticker. The
adaptor itself starts no process or thread. A fresh valid action replaces the
current mailbox action; `None` repeats it, and malformed input is converted to
`stay` and recorded in the next observation rather than raising.

The controlled chef uses one of `north`, `south`, `east`, `west`, `stay`, and
`interact`. Directions are single-tile primitives. `interact` acts on the tile
immediately in front of the chef. The partner is scripted: the standard one
three-onion-order scenarios use the upstream `GreedyHumanModel`; mixed and
forced-coordination scenarios use a deterministic feature/path planner. The
forced layouts use an explicit shared-counter handoff protocol and are not fed
to
`GreedyHumanModel`, whose upstream source documents that it is not suitable for
unilateral completion there.

## Observations

Observations are strict JSON-safe dictionaries under 200 kB. They include the
layout terrain, scenario/seed, controlled-agent index, tick and deadline,
`state.to_dict()` (players, held objects, counter/pot soups, recipes and
`timestep`), sparse team score, per-agent sparse/shaped transition rewards,
recent and cumulative event counts, actions, and the partner's current task.
No screenshot or binary perception is needed: the selected interface is
symbolic.

## Seeds and scenarios

The fixed split in `game.py` is:

- train/development: `0` cramped_room, `1` asymmetric_advantages, `2`
  coordination_ring;
- held out: `100` counter_circuit_o_1order validation, `101` forced_coordination,
  `102` forced_coordination with controlled-agent index reversed, `103`
  counter_circuit, `104` cramped_room_tomato, `105` asymmetric_advantages_tomato
  with reversed control and a randomized valid start, `106`
  forced_coordination_tomato with a randomized valid start, and `107`
  counter_circuit_o_1order with a randomized valid start.

Seeded selection controls the layout, controlled-agent index, valid starting
positions/orientations for randomized cases, and deterministic partner
style/tie-breaking. All starts are made through the upstream MDP's valid
walkable-position set. The upstream recipe/pot state is authoritative; no
commercial-game timers, burning, tips, combos, or star thresholds are added.

## Score and diagnostics

`score` is exactly the cumulative sparse team reward: the sum of
`OvercookedEnv.step`'s `sparse_r_by_agent`, or equivalently the terminal
`ep_sparse_r`. It counts accepted soup deliveries using active recipe values
and bonus orders; shaped rewards are never included. `score_breakdown()` and
`info()` additionally report sparse reward by agent, deliveries, episode
length, event counts, collision/stuck ticks, pot idle time, and handoff
attempt/success/failure diagnostics.

There are no credentials or external services. The package dependencies and
pinned upstream source are listed in `requirements.txt`; optional headless
settings are comments in `env.example`. Runtime-generated files, if later
added, belong under `runtime/` (the current implementation needs none).

### study/interfaces.md
# Programmatic interfaces for Overcooked

Research date: 2026-10-30. The user specifically requests the open-source **HumanCompatibleAI/overcooked_ai** environment, headless Linux, no account, symbolic state, and a scripted partner. That is the recommended interface. The commercial game and GUI approaches are documented as alternatives for completeness, but should not be used for this run.

## Decision

### Chosen: official Overcooked-AI Python MDP/environment, plus an adaptor-owned 6-Hz ticking worker

Use `OvercookedGridworld` for exact deterministic transitions and `OvercookedEnv` for episode bookkeeping. Construct a concrete layout with `OvercookedGridworld.from_layout_name(layout_name, **overrides)`, wrap it with `OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)`, and use `env.state` as the canonical symbolic observation. Call `env.step((action0, action1))` with native action objects. Use `env.reset()` or `mdp.get_random_start_state_fn(...)()` for reset variants. Serialize/deserialize through `OvercookedState.to_dict()` / `OvercookedState.from_dict()`; use `env.lossless_state_encoding_mdp(state)` only for a tensor policy, not as the canonical model prompt.

The next adaptor controls one index and supplies the second action from the scripted partner. The cleanest interface is to make the adaptor an `Agent` with `action(state) -> (action, info)` and run the partner as a second `Agent` in `AgentPair`, or call the pair's actions directly. A strong baseline is `GreedyHumanModel(env.mlam)`; its implementation returns a native action and `{"action_probs": ...}`. It is intentionally limited: its own docstring says it is intended for the 3-onion order and may not work on `forced_coordination`. For mixed recipes or forced handoffs, write a deterministic partner around `MediumLevelActionManager` / `MotionPlanner` or use a fixed plan, rather than pretending the stock partner is general.

This beats every other option here because it is local, open-source/MIT, account-free, deterministic, symbolic, CPU-capable, and exposes the exact transition state and reward. It avoids screenshots, OCR, GUI focus, network variance, commercial game state ambiguity, anti-cheat, and input-injection fragility. It is also the reference implementation used by the benchmark paper. The one important addition is ours: a worker thread/process that calls one joint transition every 1/6 second while Astra/Jev reason off-thread. Without that scheduler the native API is synchronous and should be treated as tier 1 in execution, despite the study's requested tier-2 model architecture.

…(truncated; full text in prior/interfaces.md)

- Package metadata and Python constraint: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/pyproject.toml
- Environment API: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_env.py
- MDP/state/rules/rewards: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_mdp.py
- Actions: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/actions.py
- Agents: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/agent.py
- Benchmark evaluator: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/benchmarking.py
- Layout files: https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/layouts
- Original benchmark paper: https://arxiv.org/abs/1910.05789
- Official demo clock/server: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo/server/app.py
- Official demo game/action queue: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo/server/game.py


## Measured model latency (at archive creation, 2026-10-30 15:00)
- System-2 (Astra), one-word answer: 630 s
- System-2, an answer of ~1225 output tokens: 78 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 156 ms (= 0.9 ticks), p90 173 ms, max 183 ms over 12 warm requests; first request on a new connection 173 ms — a timeout below the tail turns slow answers into failed calls
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).

## Jev model context and usage for Meta-Harness

Snapshot: 2026-10-29 (docs.typesafe.ai: API, models, primitives, confidence, patterns, cookbooks, request
rules and the jev-1.13 known-failure page; plus the design of deployed open-source Jev agents). This is design context for the Astra proposer and for runtime Astra
when it designs Jev questions. The Python interface below is GamingMaster's wrapper, which differs
from the provider SDK. The model profile and episode budgets in the experiment manifest are
authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It reads text and structured JSON. (The provider has an experimental image
input; this wrapper does not send images, so a visual game needs a separate perception
step — code, or Astra via `ctx.think(images=[...])` — before Jev can judge its state.)
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

Model ids: `jev-latest` and `jev-preview` both resolve to jev-1.14.0 at this snapshot (released
2026-10-14; jev-1.13.0 remains available). An alias can move between calls; every answer records
the version that produced it, and the leaderboard warns when an archive mixes versions. Limits:
64k tokens per request, 32k per question; English is the primary training language.

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Known failure modes (provider's list for jev-1.13) and what to do instead

* **Literal reading.** It answers the question written, not the one meant: state the
  exact condition in `instructions` and put boundary cases in the criteria.
* **Math, counting, comparisons of numbers and dates.** Unreliable: compute them in
  code and send the result or a named bucket ("adjacent", "two cells away"), never ask
  Jev to count, add or compare magnitudes.
* **Numeric representations.** Semantic words beat raw numbers: colour names over RGB/hex
  or palette indices, "wall to the left" over coordinates. Translate in code first.
* **Indirection.** Several hops (a property of a property, double negatives) cost
  accuracy: name the relevant state field directly, e.g. `threats.nearest`.
* **Large states with irrelevant detail.** Unrelated content distracts and costs accuracy
  ("context rot"): filter in code and send only the fields this decision needs.
* **Contradictory instructions and criteria.** Keep them aligned; a Noul whose `true`
  means "no" performs worse.
* **Choice option order.** It leans toward the option listed first: do not always list a
  default/no-op first; when a decision matters, ask the same question twice with the
  options reordered (one request, two question ids) and act only if both agree.
* **Generation.** It is not a generator: turn extraction into a Choice over candidates
  that code (or Astra) produced.

### Context that each request must carry

Jev does not inherit the proposer's prompt, Astra's conversation, `ctx.memory`,
previous requests, game description or observations automatically. This wrapper
sends only the supplied `state` and `questions` with the configured model.
Construct the necessary context explicitly on every call.

Prefer a named JSON object containing the current observation summary, the
relevant rule/field meanings, Astra's current subgoal and constraints, and any
short recent evidence needed for this decision. Include candidate descriptions
when asking Jev to compare candidates. Supply actual values, not variable names
or a reference to an inaccessible file. Keep irrelevant history out; preserve
evidence that distinguishes the options. English instructions are preferred for
this model. A request must fit 64k tokens, and the state plus any single question
32k. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Patterns that deployed Jev systems share (provider cookbooks, open-source agents)

* **Code computes facts, Jev judges, code decides.** Successful controllers send a small JSON
  state of *derived facts* (distances as named buckets, what is adjacent, time to contact,
  the last action and its measured outcome, the measured age of the observation) rather than
  raw grids or numbers, then ask atomic questions and combine the probabilities in code.
* **One request, several atomic questions.** A per-tick request typically carries 3–8
  independent questions (e.g. a Choice over the legal actions, a Score of immediate danger, a
  Noul per precondition or per candidate), including speculative ones whose answers code may
  ignore; a request with many questions costs about the same latency as one with one.
* **Structured criteria.** Option descriptions as objects (`what`, `not_for`, `examples`) and
  instructions that name the state fields (`` `threats.nearest` ``) measurably beat prose.
* **Freshness and fallbacks.** Every answer is tagged with the observation / plan version it
  judged; a late answer is dropped, and an executable fallback (the last safe action, a
  conservative rule) runs while waiting. Deadlines per question are short and explicit.
* **Confidence gates per consequence.** Irreversible or lethal actions need a higher top
  probability than cheap ones; a top probability under ~0.6 is treated as "undecided" (route to
  the fallback or to System-2), not as a weak yes. Ask high-stakes Choices twice with the options
  reordered and act only when both agree.
* **Large option sets** (more than 255, or many near-identical options) are handled in two
  stages: Score or Noul each candidate independently, then one Choice over the shortlist.
* **System-2 authors, System-1 runs.** The slow model designs the state schema, the questions,
  the thresholds and the fallback policy — and re-authors them when the evidence says the
  questions are wrong — rather than being asked per tick. Harnesses that let System-2 revise
  the System-1 interface on observed failures improved most across iterations.

### Exact local call and answer contract

Import `Choice`, `Score`, `Noul` from `metaharness.models.jev`. Call
`ctx.decide(state, questions, timeout_s=5.0)`; it returns
`dict[str, System1Answer]` directly, keyed by your question IDs.

| Question constructor | Meaning of `answer.value` | Other local fields |
|---|---|---|
| `Choice(instructions, criteria={option: description, ...})` | Selected option key, a string | `.probabilities`, `.confidence` |
| `Score(instructions, criteria=[low_description, ..., high_description])` | Fractional level index, from 0 to number of levels minus 1 | `.probabilities` with string index keys, `.confidence` |
| `Noul(instructions, criteria={"true": description, "false": description})` | Probability of yes, between 0 and 1 | `.probabilities` contains `"true"`/`"false"`; `.confidence` is normally `None` |

All answers also have `.type` and `.top(k)`. There is no `.answers` container,
`.choice`, `.score`, `.noul`, generated explanation, or Score `.legend` on the
local return objects. Keep the Score level descriptions in your own code.
`Noul` criteria are optional. JSON-serializable question dictionaries with
`type`, `instructions` and appropriate `criteria` also work. Credentials and
the HTTP client are managed outside the harness; do not instantiate `JevClient`
or bypass `ctx.decide`. See the [HTTP API](https://docs.typesafe.ai/api) for the
provider's different wire format.

For Choice, define distinguishable options and include an explicit unknown or
no-match option when needed; there can be at most 255 options. Structured
descriptions can spell out inclusion, exclusion and examples when labels are
ambiguous. See [Choice](https://docs.typesafe.ai/primitives/choice).
For Score, use 2–10 concrete ordered levels for one dimension (the API rejects more
than 10; the constructor raises); the returned number is an expected level, not
necessarily an integer or a probability — use it for thresholds, not to interpolate
magnitudes. See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

What `ctx.decide` raises: `TimeoutError` when the answer (including retries of
rate-limited or transient server errors) does not arrive within `timeout_s`;
`RuntimeError` for any other failed call (e.g. a malformed question rejected by
the API); `BudgetExceeded` / `EpisodeOver` when the System-1 budget is used up or
the episode is over; `ModelUnavailable` when the key is rejected or the account is
persistently throttled (the run stops; catching it does not hide it).

### Small usage example, not a game adapter

This helper illustrates one batched request about an Astra plan and an
observation summary. The proposer still designs the summaries, questions,
thresholds, action mapping and scheduling for its candidate.

```python
from metaharness.models.jev import Choice, Noul, Score


def assess_plan(ctx, plan, observation_summary):
    state = {"plan": plan, "observation": observation_summary}
    answers = ctx.decide(state, {
        "condition": Choice(
            "Do the facts in `observation` support `plan.preconditions`?",
            {
                "supported": "All required preconditions are supported by the observations.",
                "contradicted": "An observed fact contradicts a required precondition.",
                "unknown": "The observations cannot establish all required preconditions.",
            },
        ),
        "progress": Score(
            "How much of `plan.goal` is visibly achieved in `observation`?",
            ["No visible progress.", "Some progress, but the goal is incomplete.",
             "The stated goal is visibly complete."],
        ),
        "finished": Noul(
            "Does `observation` explicitly establish completion of `plan.goal`?"
        ),
    }, timeout_s=5.0)
    condition = answers["condition"]
    return {
        "condition": condition.value,
        "condition_probability": condition.probabilities.get(condition.value, 0.0),
        "condition_confidence": condition.confidence,
        "progress_level": answers["progress"].value,
        "completion_probability": answers["finished"].value,
    }
```

### Timing and the generated bridge

One admitted `ctx.decide` request consumes one System-1 call, even with several
questions. Requests are synchronous unless submitted via `ctx.spawn`. Batching
reduces round trips but does not promise constant latency. Measure actual
latency from traces and the task's calibration: Jev can take longer than a game
tick, and the first request on a new connection (each thread keeps its own) pays
the connection setup. Keep an executable fallback while waiting, limit outstanding
requests, and associate each response with the observation and plan version that
produced it. Recheck relevant preconditions and legality on the latest observation
before acting. A cached semantic interpretation may outlive an individual action
decision; define its invalidation rule rather than treating every answer as
permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, let `ModelUnavailable` propagate, and stop
background work when `ctx.done`. The standard inference calls already log request,
response, latency and token usage. Add harness events explaining when a Jev
answer changes a chosen action, invalidates a plan, or triggers Astra; call
counts alone do not establish useful cooperation.

In `proposal.md`, specify the Jev state schema, each question's role, how its
answer affects control, freshness/confidence handling, and the hypothesis about
when Astra should revise the interface. These choices belong to the generated
harness; this guide supplies no environment-specific adapter.

When runtime Astra is asked to create or revise Jev questions, include this
guide (or its relevant interface sections) in that `ctx.think` prompt too.
The proposer context is not automatically forwarded to runtime Astra. A harness
can obtain the exact same bundled text with
`from metaharness.models.jev import usage_guide` and append `usage_guide()` to
the interface-design prompt, together with public observations and action
specifications. Do not send the full guide to Jev on every tick; send the
decision-specific state and questions it describes.
