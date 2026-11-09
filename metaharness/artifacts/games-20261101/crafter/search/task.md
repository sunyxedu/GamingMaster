Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Crafter` — Crafter is a synchronous open-world survival episode. The world advances one tick per step, and ends when the player dies or the configured episode length is reached. Gather wood, stone, coal, iron and diamond; drink water, manage food/energy, place blocks and plants, craft tools and swords near the required table/furnace, and defeat zombies and skeletons. The observation is JSON with player position/facing/vitals, inventory, achievement counters, and a 9x9 semantic local_map centered on the player. Coordinates are [x,y], and do interacts with the tile or creature one step in front. Score is the fraction of the 22 standard achievement names unlocked; held-out episode aggregation can compute the benchmark per-achievement rates and their geometric mean.

Actions:
- noop: Wait for one simulation tick.
- move_left: Move one tile west and face west.
- move_right: Move one tile east and face east.
- move_up: Move one tile north and face north.
- move_down: Move one tile south and face south.
- do: Interact with the tile or creature directly in front (gather, drink, eat, or attack).
- sleep: Start or continue sleeping to restore energy; sleeping ends when rested or hurt.
- place_stone: Place one stone block in the facing tile when legal.
- place_table: Place a crafting table in the facing tile when legal.
- place_furnace: Place a furnace in the facing tile when legal.
- place_plant: Plant a sapling in the facing grass tile when legal.
- make_wood_pickaxe: Craft a wood pickaxe beside a table.
- make_stone_pickaxe: Craft a stone pickaxe beside a table.
- make_iron_pickaxe: Craft an iron pickaxe beside a table and furnace.
- make_wood_sword: Craft a wood sword beside a table.
- make_stone_sword: Craft a stone sword beside a table.
- make_iron_sword: Craft an iron sword beside a table and furnace.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Actions are bare names from this list. Each step advances exactly one game tick. The local_map grid is centered on the player; coordinates are [x, y]. Collect and craft achievements are cumulative counters, while score is the fraction of the standard achievement set unlocked in this episode.

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
- strategy_summary: Strong play follows the dependency chain wood -> table/wood tools -> stone -> furnace/coal -> iron -> diamond, while drinking, eating and planting before those vitals become critical. Maintain a semantic local map, face targets before do, batch only short safe action sequences, and replan after every meaningful step because mobs, day/night and resources change. Crafting legality, inventory, recipes, coordinates and achievement accounting belong in code; Astra chooses objectives and routes; Jev answers fast urgency, target, combat and legal-action questions. Kite enemies and retreat when health or supplies are low rather than pursuing a marginal achievement.
- scoring: Report achievements unlocked per episode and the benchmark Crafter score: aggregate achievement success rates over held-out seeds (the paper defines the aggregate as a geometric mean). Do not use shaped reward as the primary score; it does not represent the named objectives.
- interface.observation: Public observations provide RGB image plus inventory and achievements mappings (and reward/discount, with exact keys release-dependent). The adaptor should add JSON text containing a radius tile/entity grid, player coordinates/facing/vitals, inventory, achievements and recent transition; use env private world/player objects when present and retain image optionally.
- interface.actions: Submit one discrete integer per step: noop, four directional moves, do/interact, sleep, and place stone/table/furnace/plant. Resolve IDs from env.action_space/package names at startup and advance only through Env.step(action).

### study/strategies.md
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

### study/brief.md
# Crafter study brief

Crafter is Danijar Hafner's procedurally generated, Minecraft-like 2-D survival benchmark. The canonical upstream project and paper are [the GitHub repository](https://github.com/danijar/crafter) and [Crafter: A Procedural Benchmark for Open-Ended Language and Vision](https://arxiv.org/abs/2109.06780). It is a local Python environment, so there is no server, account, matchmaking, or online opponent.

## Objective and scoring

An episode starts with a player in a randomly generated world. The player gathers wood, stone, coal, iron and diamond; crafts tools and weapons; places blocks and plants; eats and drinks; and fights zombies and skeletons. Progress is represented by the named achievements exposed in the observation. The benchmark reports one success rate per achievement over many episodes and aggregates them as the Crafter score (the paper describes the aggregate as the geometric mean of achievement success rates). For this harness, report the current number/list of achievements unlocked and the normalized aggregate when evaluating a batch; never substitute raw reward for achievement progress. See [the benchmark README](https://github.com/danijar/crafter/blob/main/README.md) and [the paper](https://arxiv.org/abs/2109.06780).

The achievement set in released versions includes resource collection (`collect_wood`, `collect_stone`, `collect_coal`, `collect_iron`, `collect_diamond`, `collect_sapling`, `collect_drink`), food (`eat_plant`, `eat_cow`), crafting (`make_wood_pickaxe`, `make_stone_pickaxe`, `make_iron_pickaxe`, `make_wood_sword`, `make_stone_sword`, `make_iron_sword`), placement (`place_table`, `place_furnace`, `place_stone`, `place_plant`), defeating enemies (`defeat_zombie`, `defeat_skeleton`), and waking from sleep (`wake_up`)—22 in the standard benchmark. Check `env._achievements`/the returned `achievements` mapping at runtime because releases have changed the exact list.

## Controls and state transitions

The official environment is Gym-compatible. `env.reset(seed=seed)` returns an observation; `env.step(action)` returns `(observation, reward, terminated, truncated, info)` (older releases return the four-item Gym tuple). The discrete actions in the released Crafter environment are:

| id | action |
|---:|---|
| 0 | noop |
| 1–4 | move left, right, up, down |
| 5 | do/interact (gather, attack, drink, eat depending on facing/target) |
| 6 | sleep |
| 7–10 | place stone, table, furnace, plant |

Use `env.action_space` and the package's action-name table as the authority rather than hard-coding an order across versions; the names are documented in [crafter/env.py](https://github.com/danijar/crafter/blob/main/crafter/env.py). A step advances the simulation by one game tick. Hunger, thirst, energy, day/night, plants and hostile mobs advance on steps, not while Python is waiting between calls. Therefore this is tier 1 for the harness: Astra can spend seconds planning and the next state is not sampled until it submits an action. A practical adaptor deadline is about 60 seconds per action (wall-clock budget), while the game itself imposes no decision-time penalty.

Observations normally include a 64x64 RGB `image`, scalar `reward`, `discount`, an `inventory` mapping, and an `achievements` mapping; exact keys depend on release. Use the semantic mappings when available and retain the image for debugging. The world is finite and generated from a seed; an episode ends at the configured length or player death. `crafter.Recorder` can write videos/JSON logs (see [recording documentation](https://crafter.readthedocs.io/en/latest/)).

## Pace and reset

Each `step` is synchronous. A no-op still consumes a tick, so long sequences of no-ops waste hunger/energy and permit enemies to move. Reset with a fresh integer seed for evaluation; fixed seeds are useful for debugging only. The paper's benchmark evaluates many held-out procedural worlds, which is why a policy must adapt its local map instead of memorizing a route.

### adaptor/ADAPTOR.md
# Crafter adaptor

This adaptor runs the official `crafter==1.8.3` Python simulator directly in the evaluator's
Crafter virtual environment. It starts no server, opens no account, and makes no network requests.
`SDL_VIDEODRIVER` defaults to `dummy` for headless Linux. There are no required credentials; the
optional setting is documented in `env.example`.

Crafter is tier 1 because its simulation advances only when `step(action)` is called. `reset(seed)`
creates a fresh `crafter.Env(seed=seed)` and calls its reset, preserving the standard 64x64 world and
10,000-tick episode length unless evaluator kwargs override them. The declared train, dev, validation,
and held-out seeds all select independent procedural worlds; no route is memorized.

The action vocabulary is read from `crafter.constants.actions` at construction. Actions may be submitted
as their names (the normal harness form) or integer IDs. Invalid actions are converted to `noop` and
reported in the next observation. Each observation is strict JSON: player coordinate/facing/vitals,
inventory, cumulative achievement counters and unlocked names, the last action/reward, and a 9x9 local
semantic map. Map cells contain coordinates, material, and nearby object names (including health and
plant growth status where available). The optional RGB observation from Crafter is intentionally omitted
to keep the interface textual and compact; the official environment remains available for debugging.

`score` is the fraction of the standard achievement set unlocked in the current episode, and
`score_breakdown()` reports the unlocked count and denominator. For a batch of held-out episodes, the
benchmark Crafter score is obtained by calculating each achievement's success rate and taking the
geometric mean, as described in the study. Raw environment reward is exposed only as `last_reward` and
is not used as the primary score.

`close()` is idempotent and releases the current environment. No runtime files or processes are created.

### study/interfaces.md
# Crafter interfaces

## Recommended: official Python environment (API / official env)

Install the upstream package in the evaluator interpreter: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/crafter/bin/python -m pip install crafter` (pin the version that is available to the offline wheel cache; the current PyPI line is commonly `crafter==1.8.x`). The source of truth is [github.com/danijar/crafter](https://github.com/danijar/crafter), with API details in [`crafter/env.py`](https://github.com/danijar/crafter/blob/main/crafter/env.py) and [Read the Docs](https://crafter.readthedocs.io/en/latest/).

Typical adaptor shape:

```python
import crafter
env = crafter.Env()                 # optionally configure length/area/size
obs = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action_id)
```

Use `env.observation_space` and `env.action_space`, not assumptions from a different Gym version. Exposed semantic fields are the inventory and achievements mappings (plus reward/discount); image is RGB. For the requested textual map, add a serializer that reads the local world/player state from the installed release (`env._world`, `env._player`, and nearby object/entity arrays are the usual private attributes) and emits a radius grid with coordinates, block/entity names, facing, health, food, drink, energy, inventory and achievements. Treat those attributes as a version-checked adapter and fail loudly if absent. The public fallback is to pass the RGB image through a local tile classifier and expose the resulting grid as text; no GUI is needed. Keep a raw image option for Astra debugging.

Control is one integer action per call: movement, `do`, `sleep`, and four placement actions. Calling `step` is synchronous and has negligible API latency. There are no credentials, server addresses, API keys, or ToS restrictions for local benchmark use. Cost is free software; package dependencies (NumPy, Pillow, Pygame/Gym or Gymnasium according to release) must be present in the offline wheel cache. On headless Linux set `SDL_VIDEODRIVER=dummy` if Pygame attempts to initialise a display.

Setup checklist:

1. Install `crafter` and its pinned dependencies into the evaluator venv, with wheels available offline.
2. Run a smoke test that prints `env.action_space`, observation keys/shapes, and one seeded reset.
3. Discover the installed action-name mapping and semantic/private world fields; record the package version in adaptor logs.
4. Implement `reset(seed)` and a JSON serializer containing local map, player vitals, inventory and achievements.
5. Implement `step(action)` with tuple-shape compatibility for old/new Gym APIs and close the environment in `finally`.
6. Run headless (`SDL_VIDEODRIVER=dummy`) and retain optional `crafter.Recorder` output for diagnosis.

This beats every alternative because it is the benchmark's own deterministic simulator, exposes exact actions and achievement accounting, runs offline without a GUI or account, and advances only when the adaptor calls `step`.

## Other practical routes

| Route | Maturity / latency / exposure | Cost and risk |
|---|---|---|
| Direct private-engine inspection (`env._world`, player/entity objects) | Mature in the upstream implementation but version-sensitive; zero-copy and low latency; gives exact tiles, entities, vitals and coordinates. | Free and local. Private API can break on upgrade; pin and smoke-test. No legal/ToS issue for this open-source benchmark. |
| RGB observation plus local semantic decoder | Public API and release-stable; one-step latency; gives only pixels, so inventory/achievement fields should still be taken from the observation. | Free, headless. Decoder engineering error and ambiguity around hidden tiles; no external service. |
| `crafter.Recorder` / JSON episode logs | Mature wrapper for recording frames and metrics, not a controller; writes after each step and adds I/O latency. | Free; disk usage only. Useful for replay/evaluation, not as the primary action interface. |
| Screen capture plus keyboard/mouse injection (SDL/Pygame, Xvfb) | Possible only by launching a human renderer; high latency and brittle focus/display dependencies; pixels but no semantic state. | Free but poor fit for a server without GUI; no benefit over the API. |
| Generic Gym/Gymnasium wrappers or RL bot libraries | Wrappers can normalize reset/step and vectorize environments, but do not expose extra game semantics; latency is low. | Free; dependency/version mismatch risk. They are optional around the official env, not a separate game interface. |

No network endpoint, mod/plugin protocol, emulator hook, or account credential is required. Do not put credential values in configuration; for this local route there are no required environment variables. `SDL_VIDEODRIVER` is an optional runtime variable, not a secret.


## Measured model latency (at archive creation, 2026-10-30 07:54)
- System-2 (Astra), one-word answer: 55 s
- System-2, an answer of ~1004 output tokens: 81 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).