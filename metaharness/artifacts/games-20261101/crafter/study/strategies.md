# Crafter strategy prior

The progression and mechanics below are reconstructed from the upstream environment and the benchmark paper: [Crafter repository](https://github.com/danijar/crafter), [environment source](https://github.com/danijar/crafter/blob/main/crafter/env.py), [world/player mechanics](https://github.com/danijar/crafter/tree/main/crafter), and [Crafter paper](https://arxiv.org/abs/2109.06780). Verify names and recipes against the installed package at startup.

## Reliable opening and dependency order

1. **Orient and secure basics.** Inspect the nearby tile grid, face a tree, and use `do` to collect wood. Keep a mental return direction and avoid spending steps wandering. This exact adjacency/facing bookkeeping belongs in **code**; selecting which resource to pursue belongs in **Astra**.
2. **Workbench and wooden tools.** Collect enough wood to place a table, then make a wooden pickaxe and sword through the game's `do` interaction/crafting rules. The adaptor should expose recipe prerequisites and inventory deltas to Astra; recipe accounting is **code**.
3. **Stone, furnace and coal.** Mine exposed stone with the appropriate pickaxe, place a furnace, gather coal, and upgrade to a stone pickaxe/sword. Keep a safe route and return to the table/furnace. Tool selection and exact craft/placement checks are **code**; route choice is **Astra**.
4. **Water and food.** Drink at water and eat plants or cows before thirst/food reaches zero. Plant saplings when a safe tile and resources are available. Threshold alarms (“need drink now”, “food below X”) are **Jev yes/no questions** backed by exact vitals in **code**.
5. **Iron and diamond.** Search stone/caves for iron, smelt/collect it, craft iron pickaxe and sword, then descend/search for diamond. Use the strongest available pickaxe; do not attempt diamond blocks with a weak tool. Search planning and retreat decisions are **Astra**; inventory/tool validity is **code**.
6. **Combat and night survival.** Fight zombies/skeletons with a sword, one attack at a time while kiting around obstacles; preserve health and retreat to open ground or a lit/safe area. Enemy-present and “attack versus retreat” classification is **Jev choice**; multi-step combat and escape planning are **Astra**.

## Heuristics and failure modes

- Prefer exposed resources and short loops over deep expeditions before a food/water reserve exists (**Astra** route planning).
- Face the target before `do`; a movement/action off-by-one can waste ticks or hit the wrong object (**code** assertion from player facing and adjacent tile).
- Batch a short plan, then re-observe after every meaningful action. Plants, mobs, hunger, thirst and day/night change on each `step`; do not open-loop hundreds of actions (**Astra** replanning, **Jev** recognition of changed local state).
- Avoid no-op spam: it advances the world and can cause starvation or attacks (**code** action budget).
- Do not carry unneeded material into combat; return to a known safe tile after mining and maintain a retreat vector (**Astra**).
- Never assume a fixed map or resource route across seeds. Use the semantic radius grid and coordinates; held-out seeds are the benchmark's point (**code** map bookkeeping plus **Astra** adaptation).
- If health, food or drink is critical, prioritize the nearest valid recovery action over a new achievement. Death discards the run's remaining opportunities (**Jev urgency classification, Astra choice).
- Common implementation mistake: treating reward as the score. Reward is shaped/sparse feedback; achievement flags and episode-level success are the reportable progress (**code** metric calculation).

## Where each model should help

**Code** should maintain coordinates/facing, tile and entity grid, inventory quantities, vitals, recipes, tool validity, achievement set, action legality, no-op/step budgets, and the geometric-mean evaluation metric. It should reject impossible craft/place actions and serialize a compact JSON state.

**Astra prompts** should ask for a dependency-aware objective (which achievement next), a safe route, resource reserves, and a contingency after each observation. Give it the recent trajectory and explicit costs; it can trade exploration against survival and change plans when a seed lacks an exposed resource.

**Jev questions** should be small typed queries over JSON: “Is water/food urgent?”, “Which adjacent tile is the target?”, “Is the facing correct?”, “Do we attack or retreat?”, “Which of these legal actions advances the current subgoal?”, and “Did this step unlock an achievement?”. Jev should classify and rank candidates, never invent recipes or perform arithmetic that code can do exactly.

