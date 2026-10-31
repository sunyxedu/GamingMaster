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
