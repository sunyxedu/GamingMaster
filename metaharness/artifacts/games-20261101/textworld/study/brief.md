# TextWorld study brief

## Status and scope

TextWorld is Microsoft's framework for generating and running text adventure games, not one fixed adventure. This study recommends its native Python environment with generated games, a fixed three-profile difficulty mix, and normalized final game score. It is **tier 1**: the world advances when a command is submitted. Astra can deliberate for seconds or minutes without in-game time passing. Jev is optional; no real-time controller is needed. [T1, T2]

**Evidence limitation, 2026-10-30:** live requests to GitHub, Read the Docs, PyPI, and arXiv failed at DNS resolution. TextWorld is absent from the specified Python 3.12.3 venv. Upstream citations below are reference/verification targets, not pages successfully retrieved in this session. Locally inspected Microsoft Jericho TextWorld tests provide narrower direct evidence, marked [L1]. See `research/sources.md`. No adaptor or game-playing implementation has been written.

## Rules and objective

A generated world contains named rooms, directional connections, doors, containers, supporters, portable objects, food, and sometimes keys. Commands change logical relations and properties: the player moves to another room; an object moves from a container to inventory; a door changes from locked to unlocked to open. Exact object names and action preconditions matter. Objects on a supporter use `take … from …` / `put … on …`; objects in a container use `take … from …` / `insert … into …`. A locked door generally needs its matching key before it can be opened. [T1–T4]

Each game supplies an objective. Read it literally: merely obtaining the target may be insufficient if it must be deposited in a particular place and the container then closed. Generated quests may have intermediate rewards and failure conditions. Solvability at generation time does not imply every subsequent state remains solvable. In the inspected upstream fixture, putting a carrot into a chest awards 2 points, closing it awards the final point and wins, while eating the carrot at the start loses immediately. This is one fixture, not a universal scoring formula. [L1]

Some TextWorld challenges use different rule sets. Cooking games ask the player to find a recipe, acquire its ingredients, perform exact cutting/cooking operations, prepare a meal, and eat it. Required ingredients may be lost through premature eating, wrong preparation, or burning. Cooking-specific advice is in `strategies.md`; cooking is an optional separate suite, **not part of the recommended generic baseline**. Treasure-hunter and other challenges also require their own generation/settings contracts. [T4, T5]

There is no opponent, combat rotation, or resource build order common to all generated games. The principal difficulties are partial observability, remembering a map and object locations, discovering action preconditions, maintaining a multi-step plan, and avoiding irreversible mistakes. [T1, T2]

## Observation and actions

The normal interaction is textual: room description, command feedback, inventory, objective, score, and terminal status. The native API can request additional fields through `EnvInfos`. Some are substantial assistance, especially `admissible_commands`; others expose hidden world facts or a solution. The recommended benchmark explicitly allows current description/inventory and admissible commands, but withholds hidden facts, game objects, walkthroughs, and policy commands. Use the same information policy on all seeds. [T3]

Typical commands, subject to the generated grammar:

| Purpose | Examples |
|---|---|
| Inspect | `look`, `inventory`, `examine cookbook` |
| Navigate | `go north`, `go south`, `go east`, `go west` |
| Acquire/release | `take brass key`, `take carrot from chest`, `drop brass key` |
| Door/container | `open chest`, `close chest`, `unlock chest with brass key`, `lock chest with brass key` |
| Place | `put carrot on table`, `insert carrot into chest` |
| Consume | `eat carrot` — potentially irreversible and losing |
| Cooking variants | `slice onion with knife`, `cook onion with stove`, `prepare meal`, `eat meal` |

Prefer exact strings supplied in the current admissible-command list when that list is enabled. “Admissible” means currently executable according to the environment, **not beneficial or safe**. The legal `eat carrot` can destroy a quest. A parser rejection, a successful action with no score, and a terminal loss are distinct outcomes. [T3, L1]

## Score and progress

Report **terminal raw game score / that game's positive maximum score**, bounded to [0, 1] only after validating the supported suite's score assumptions. The native `step` contract returns a cumulative game score rather than a per-action reward; do not sum returned scores. Track score deltas separately for diagnostics. Preserve the game's partial score on loss or step-limit truncation; winning is a separate diagnostic. Require `max_score > 0` when accepting a generated scenario. Do not obtain it by guessing from prose. [T3]

For a benchmark collection, take the arithmetic mean of per-game normalized scores with equal numbers from each difficulty profile. Summing raw scores across games would overweight quests with larger maxima. Record `won`, `lost`, submitted-command count, engine move count if available, profile, generation seed, and whether the evaluator truncated the episode. None is an extra score component. A loop cannot earn extra benchmark credit by repeatedly claiming the same score. [Design recommendation]

## Pace and decision budget

The generated game is advanced by `step(command)`; wall-clock waiting between steps is not a game action. Commands may consume in-game turns, and quest rules may react to commands. Some informational or invalid commands may not count as interpreter moves, so enforce the evaluator's command budget independently. The proposed limits are 100/200/400 submitted commands for easy/medium/hard games. These are benchmark choices, not TextWorld defaults. [T3; design recommendation]

There is **no finite in-game decision deadline**. External infrastructure may still impose a run timeout or model budget. `study.json` uses **`decision_latency_s: 0.0` as an explicit sentinel for “unbounded/no wall-clock deadline,” not “must respond immediately.”** Standard JSON cannot encode infinity. Any consumer should honor this documented convention rather than divide by this field.

## References

- [T1] Project and installation overview: https://github.com/microsoft/TextWorld
- [T2] Côté et al., *TextWorld: A Learning Environment for Text-based Games*: https://arxiv.org/abs/1806.11532
- [T3] Native API, `EnvInfos`, environment contract: https://textworld.readthedocs.io/en/latest/textworld.html ; source https://github.com/microsoft/TextWorld/blob/main/textworld/core.py
- [T4] Generator: https://github.com/microsoft/TextWorld/tree/main/textworld/generator
- [T5] Cooking challenge: https://github.com/microsoft/TextWorld/blob/main/textworld/challenges/tw_cooking/cooking.py
- [L1] Locally read upstream fixture tests, commit `752ba6720d522909a4a2a495b5a03dd535f9adf1`: https://github.com/microsoft/jericho/blob/752ba6720d522909a4a2a495b5a03dd535f9adf1/tests/test_textworld_games.py . Saved reference: `research/jericho_textworld_tests.reference.txt`.
