# MiniGrid / BabyAI study

Evidence status (2026-10-30): the evaluator executable was checked directly and is CPython 3.12.3. `pip show minigrid gymnasium numpy pygame` found none installed. Requests to the official documentation, PyPI, and GitHub source failed DNS resolution in this sandbox. Therefore the game details below are a knowledge-based study with explicit source URLs for verification, not a claim that those pages were retrieved or that a particular release was tested. Installation and source-level checks are specified in interfaces.md. No game-playing code was written.

## Game and objective

MiniGrid is a family of small, procedural, grid-world environments maintained by the Farama Foundation. BabyAI is an instruction-following task suite incorporated into modern MiniGrid. These are single-agent benchmark environments, not an online service. A mission can require reaching a goal, approaching an object, picking something up, opening/unlocking doors, placing one object next to another, remembering a cue, or performing several instructions in order. The mission and environment family determine success; there is no universal “walk onto the green square” rule. [B1, B2, B5]

The agent occupies one cell and faces one of four directions. Walls, closed doors, and most objects block forward movement. The agent can carry only one object at a time. A matching-color key unlocks a locked door; ordinary unlocking does not consume the key. Doors, balls, keys, and boxes have colors, and some tasks distinguish otherwise identical objects by location. Toggling a box can reveal its contents. Lava can be entered but is fatal. Not all apparently traversable cells are safe. [B3, B4]

Most environments give a small, egocentric, partially observable symbolic view, normally 7×7 cells, a direction, and a textual mission. The “image” in the default observation is an array of object/color/state codes, not RGB pixels. Occlusion and unseen cells must be preserved in text; unseen is not empty. Some tasks deliberately change visibility, size, action semantics, or reward. [B1, B3, B6]

## Actions and their costs

The standard action enum is below. Read the installed enum rather than hard-coding assumptions for every subclass. Each submitted action normally increments the environment step counter, including a blocked move, unsuccessful interaction, or otherwise ineffective action. [B3, B4]

| ID | Name | Effect |
|---|---|---|
| 0 | left | Turn 90° counterclockwise in place. |
| 1 | right | Turn 90° clockwise in place. |
| 2 | forward | Move one cell in the facing direction if allowed. |
| 3 | pickup | Pick up the immediately front object if pickupable and the hand is empty. |
| 4 | drop | Drop the carried object into an empty front cell. |
| 5 | toggle | Interact with the front object: open/close a door, unlock with a matching key, open a box. |
| 6 | done | Base MiniGrid usually does nothing beyond spending a step; some task/verifier modes use an explicit completion action. Never assume this means success. |

There is no standard backward, strafe, diagonal, click-object, or teleport action. Interactions require facing the adjacent cell. Some environment families expose fewer actions or override them; DynamicObstacles and specialized tasks need explicit audits. See interfaces.md for family restrictions. [B3, B5]

BabyAI distinctions matter: “go to” generally means becoming adjacent to and facing a qualifying object, “pick up” means actually carrying it, “open” means the required door opening event, and “put next” means a qualifying placement beside the destination, usually cardinal adjacency rather than diagonals. Exact event checks, strict ordering, and explicit-done behavior belong to the installed instruction verifier. Arbitrarily dropping beside an object or standing nearby is not a substitute for checking that verifier. [B7]

## Score, progress, and termination

For many standard sparse-reward tasks, success reward is

`1 - 0.9 * (step_count / max_steps)`.

Failure and unsuccessful timeout normally yield zero. Thus successful shorter episodes score better, and successful episodes near the limit still yield about 0.1. This is not a safe assumption for every registered environment: distance shaping, intermediate rewards, special failure rewards, and task-specific termination exist. Restrict the benchmark to audited sparse-success families or define a separate per-family success extractor. [B3, B5]

Report `success`, `steps_used`, `max_steps`, `remaining_steps = max(0, max_steps - steps_used)`, `remaining_fraction`, raw reward/return, `terminated`, and `truncated`. For a mixed-size benchmark use the standard success reward as the primary scalar for the chosen sparse families: `score = success * (0.1 + 0.9 * remaining_fraction)`. Also publish success rate and mean remaining fraction among successes; do not give a failed episode positive credit merely because it ended early. Compare raw remaining steps only within the same task/budget. Mission subgoals and explored cells are diagnostic progress, not score. [B3; benchmark design recommendation]

The Gymnasium API separates terminal events (`terminated`, such as success or lava death) from time limits (`truncated`). Neither flag alone means success. Both may be true on a successful last allowed step, so retain terminal success in that case. Stop stepping once either is true and reset for the next episode. Maximum budgets depend on environment and dimensions; read the actual instance and account for an outer TimeLimit wrapper. [B3, B8]

## Pace and model roles

**Tier 1 is correct.** The environment advances through synchronous `step(action)` calls. It does not move while the model is thinking. Even moving-obstacle tasks update in steps, not according to elapsed wall time. Astra can deliberate for seconds or a minute without spending in-game time. A surrounding evaluator can impose an independent wall-clock timeout; none was supplied. Render FPS is not a decision deadline. [B3, B5, B8]

There is no finite intrinsic decision latency limit. In study.json, `decision_latency_s: -1.0` is an explicitly documented sentinel for “unbounded by game mechanics”; it is not a literal negative timeout. Do not implement it as an immediate timeout. Jev is optional for this tier and should not control an asynchronous action loop.

## Source URLs

These are authoritative verification targets, not pages successfully retrieved in this session. Links to `master` are moving references; the adaptor phase should record the installed version and use its corresponding source tag/commit.

- [B1] Official documentation: https://minigrid.farama.org/ and https://minigrid.farama.org/content/basic_usage/
- [B2] Maintained project and BabyAI integration: https://github.com/Farama-Foundation/Minigrid
- [B3] Core reset, observation, action, reward, and step implementation: https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/minigrid_env.py
- [B4] Object behavior and enums: https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/world_object.py ; https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/actions.py ; https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/constants.py
- [B5] Environment catalog and implementations: https://minigrid.farama.org/environments/minigrid/ ; https://minigrid.farama.org/environments/babyai/ ; https://github.com/Farama-Foundation/Minigrid/tree/master/minigrid/envs
- [B6] Wrappers: https://minigrid.farama.org/api/wrappers/ ; https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/wrappers.py
- [B7] BabyAI instruction verification: https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/babyai/core/verifier.py ; https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/babyai/core/roomgrid_level.py
- [B8] Gymnasium environment API: https://gymnasium.farama.org/api/env/
