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

- Primitive-action deadline: one tick, approximately **0.166667 seconds**. A missing valid primitive/plan continuation becomes STAY; the partner and cooking continue. A delay loses productive opportunities, not an automatic score deduction.
- Jev's approximately 0.5-second latency spans about three ticks, so it should choose among **prevalidated medium-level tasks**, asynchronously. Deterministic code handles movement and interactions every tick.
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
