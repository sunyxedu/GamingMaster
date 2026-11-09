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
4. Keep a particular counter reserved for dish or soup transfer when serving routes require it.
5. Do not assume ingredient order is interchangeable when orders have different values and times; choose a target recipe before crossing.

**CODE:** inventory each counter; track reservations and object ownership; use exact recipe matching and reward arithmetic; prohibit drops on occupied counters; select access poses for both sides.

**ASTRA:** choose whether to prioritize the 2-ingredient bonus order or a 3-ingredient order based on value, cook time, travel, and whether it unlocks a pot. Do not optimize raw value while ignoring throughput.

**JEV:** classify a counter as `free|ingredient_reserved|dish_reserved|soup_handoff`, identify whether a held ingredient matches the selected recipe, and choose `deposit|retrieve|continue` among legal candidates.

### E. Forced Coordination — explicit handoff protocol

This layout intentionally forces collaboration; the stock greedy partner's docstring warns it may not work there. One chef may be unable to complete the whole soup alone. Use a pre-agreed protocol: supply-side chef fetches and passes ingredients through the accessible counter/gate; pot-side chef owns pot filling/start; delivery-side chef owns dish pickup and serving. If a chef can physically pass a held object only by putting it on a counter, reserve a counter and define the receiver's pickup trigger.

Do not let “nearest pot” override the protocol. A chef holding the wrong ingredient should walk to the transfer counter, not wander to an inaccessible pot. If a partner fails to retrieve a staged item, Astra should switch to a recovery plan: retrieve it with the other chef if possible, use another counter, or abandon the reservation and clear the station.

**CODE:** reachability test for every goal/chef pair; explicit finite-state handoff (`carrying -> staged -> acknowledged -> retrieved`); timeout and recovery; no impossible goal passed to Jev.

**ASTRA:** plan role ownership and handoff timing, infer partner reliability from recent acknowledgements, and select recovery after a missed handoff.

**JEV:** yes/no “did partner retrieve the staged item?”, “is handoff overdue?”, “is the target counter occupied?”

## 3. Recipe and priority strategy

### Exact recipe choice

For each candidate idle pot, enumerate legal recipes that extend its current ingredient multiset. A recipe is accepted only if it is in `state.all_orders`; value is zero otherwise. Bonus recipes in `state.bonus_orders` receive `order_bonus * base_value`. The implementation's default recipe capacity is three, but layouts may override `max_num_ingredients`. [MDP]

Do not use commercial-game assumptions such as a FIFO order queue, tip decay, combo multiplier, or expiry. In this environment, “next order” is a planning choice among allowed recipes; the delivery reward is determined by recipe membership. For a mixed layout, compute a priority such as:

```text
priority(recipe) = (reward * expected_delivery_probability)
                   / (ingredient_travel_ticks + interaction_ticks
                      + cook_time + pickup/delivery_ticks)
```

Use exact code arithmetic for this score and include pot occupancy/partner access. A high-value recipe that blocks both pots or requires an impossible handoff can be worse than a short recipe.

**CODE:** parse `Recipe.ingredients`, `Recipe.value`, `Recipe.time`, `state.all_orders`, and `state.bonus_orders`; enumerate missing ingredients; maintain a reservation so two chefs do not target incompatible mixtures. Never ask Astra/Jev to calculate the reward.

**ASTRA:** choose the objective weights/throughput policy when multiple recipes are close, and revise after observing actual travel/collision costs.

**JEV:** answer categorical recipe compatibility (`matches_target`, `needs_onion`, `needs_tomato`, `should_abandon`) from code-provided candidates.

### Pot scheduling

With multiple pots, fill one pot and start it, then use the other for a staggered batch if resources permit. Avoid putting a low-value partial mixture into a pot that is the only route to a bonus recipe. Start cooking as soon as the intended recipe is full/valid; in modern dynamics this is an explicit empty-hand interact. During cook time, shift labor to the next pot or dish staging. A ready soup has priority over speculative prep because leaving it in the pot blocks that pot and delays score.

The environment treats `ready`, `cooking`, and idle `x_items` as distinct pot states. A pot with ingredients is not cooking just because it is full under modern dynamics. [MDP]

**CODE:** maintain a pot ledger and enforce “at most one intended recipe per pot”; invoke `get_pot_states`; identify exact interaction pose; account for the one tick needed to start and the final cook increment.

**ASTRA:** choose pipeline order and decide when to sacrifice a partial/zero-value pot to free capacity if no valid completion is possible.

**JEV:** classify pot urgency and select among code-generated `start`, `fill`, `dish_pickup`, `soup_pickup`, `ignore` choices.

## 4. Traffic and collision heuristics

The transition resolves interactions before movement and cancels both players' movement when they collide or swap. This makes the following rules high value:

- Do not route both chefs through a single approach tile at once.
- A chef carrying soup has high priority; the other yields with STAY or a detour.
- Do not stand directly in front of the only pot/counter interaction tile while the partner is trying to use it.
- After a failed movement, compare positions to the previous state. If unchanged for several ticks, replan rather than repeating the same action.
- STAY preserves orientation; use it intentionally when waiting in front of a pot or yielding a lane.
- An attempted blocked directional move still changes orientation, so a subsequent interact may unexpectedly target a different feature. Keep orientation in the state machine.
- Replan after every successful interaction because the held-object class changes the valid useful goals.

**CODE:** implement collision prediction from `compute_new_positions_and_orientations` semantics; detect position stagnation; maintain interaction preconditions. The official greedy partner has an `auto_unstuck` mechanism that samples an action changing positions if two consecutive states are equal; borrow the principle, but make recovery deterministic for reproducibility. [Agents, MDP]

**ASTRA:** decide traffic priority for a layout and whether to pause a low-urgency task.

**JEV:** fast yes/no collision and stuck recognition, not path planning.

## 5. Common mistakes and counters

| Mistake | Why it loses | Countermeasure / owner |
|---|---|---|
| Treating `step` as paused while Astra reasons | In tier-2 adaptor the clock continues; soup cooks and work opportunities pass | Nonblocking planner, safe fallback action, stale-result rejection (**CODE**) |
| Forgetting explicit start-cooking | Modern default leaves a full idle pot, so no cooking progress | Detect `idle_full` and schedule empty-hand `INTERACT` (**CODE/JEV recognition**) |
| Starting a soup before its intended recipe is complete | Once cooking starts, ingredients cannot be added; delivery may be zero | Recipe reservation and precondition check (**CODE**) |
| Asking both chefs to fetch the same ingredient/pot | They collide, duplicate work, or block each other | Role/target reservations (**ASTRA**, enforced in **CODE**) |
| Using a dish too early | A dish only picks up a ready soup; it may then waste a hand/counter | Ask Jev whether a target is nearly ready; code checks exact `is_ready` (**JEV/CODE**) |
| Carrying soup to a blocked/invalid serving route | Soup occupies one hand and can deadlock a narrow kitchen | Reachability and delivery priority (**CODE**) |
| Dropping on an occupied counter | Interaction is a no-op; the chef stays loaded and misses ticks | Validate counter occupancy before interact (**CODE**) |
| Ignoring partner handoffs | Forced layouts cannot be solved by unilateral nearest-goal greed | Acknowledge/stage/retrieve protocol (**ASTRA/CODE**) |
| Treating all recipes as 20 points/20 ticks | Mixed layouts customize values/times; bonus order multiplier changes reward | Parse layout/config every reset (**CODE**) |
| Reporting shaping reward as score | Busywork can look good without a delivered soup | Report cumulative sparse reward only (**CODE**) |
| Using raw integer actions in the native env | Native `OvercookedEnv.step` expects action objects, not Gym indices | Translate at the adaptor boundary (**CODE**) |
| Sharing a stateful partner object across indices | Agent index/history gets overwritten | Separate agents or use documented `AgentPair` duplicate mode (**CODE**) |
| Assuming the stock greedy partner is universal | It asserts a sole three-onion order and warns about forced coordination | Scenario-specific partner or custom planner (**ASTRA/CODE**) |
| Overusing `STAY` when no plan is ready | It is safe but loses one tick and can leave a pot blocked | Jev fast fallback among legal precomputed tasks (**JEV**) |

## 6. Prompt design for Astra

Give Astra a compact symbolic snapshot, stable coordinate map, recent event window, current assignments, and planner-generated candidate tasks. Include:

- layout name and terrain grid;
- both positions/orientations/held-object summaries;
- every object/counter/pot and soup ingredients/cook remaining/ready;
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
