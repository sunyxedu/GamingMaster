# Sokoban study brief

## What the game is

Sokoban is a deterministic, single-player grid puzzle. A level contains walls, floor, goals, a player, and one box for each goal. The player walks on free floor and can push an adjacent box one square when the square behind the box is free; boxes cannot be pulled. A level is solved when every goal is occupied by a box. The compact rules and the conventional symbols are documented by the Sokoban overview and XSB level-format references ([Wikipedia](https://en.wikipedia.org/wiki/Sokoban), [Sokoban Wiki: Level format](http://sokobano.de/wiki/index.php?title=Level_format)).

Use these ASCII symbols when presenting state to Astra or Jev:

| Symbol | Meaning |
|---|---|
| `#` | wall (impassable) |
| space or `-` | floor (the parser should normalize one chosen empty-floor spelling) |
| `.` | goal |
| `@` | player on floor |
| `+` | player on goal |
| `$` | box on floor |
| `*` | box on goal |

Rows may have different lengths; cells outside a row are treated as walls. Preserve the original rectangular dimensions in observations so coordinates remain stable. A level is invalid if it has no player, or if the number of boxes and goals differs.

## Objective and progress

The primary objective is to solve as many seeded levels as possible. Within a level, report `solved`, number of boxes on goals, pushes, walking steps, total actions, and whether the action cap was reached. A push is the meaningful scarce action; walking is still counted against the evaluator's action budget. Classic Sokoban traditionally rewards fewer pushes and then fewer moves, while many engines expose a shaped reward for newly filled goals ([Wikipedia](https://en.wikipedia.org/wiki/Sokoban); [gym-sokoban environment](https://github.com/mpSchrader/gym-sokoban/blob/master/gym_sokoban/envs/sokoban_env.py)). The harness should therefore rank first by solved levels, then by boxes on goals, then by pushes and total actions; do not treat an intermediate reward as a proof of solvability.

The benchmark-specific score is a local contract, not an upstream Sokoban rule: for each `reset(seed)`, select one level, stop at the first solved state or `MAX_ACTIONS`, and return the number of solved levels and boxes placed (plus diagnostic counts). Keep the raw counters so a judge can apply its own weighting. A solved level must have every goal occupied; a box merely moved closer does not count.

## Actions and transition semantics

Expose exactly four primitive actions, encoded as `up`, `down`, `left`, and `right` (or `U,D,L,R` at the wire boundary). On an action:

1. If the destination is a wall or outside the board, leave the state unchanged and count one attempted action.
2. If the destination is empty floor or goal, move the player there and count a walking step.
3. If the destination contains a box and the next cell in the same direction is empty floor or goal, move the box and player together and count one push.
4. If the box cannot be pushed, leave the state unchanged and count one attempted action.

Goals are properties of cells, so walking off a goal or pushing a box off a goal must retain that goal underneath. A solved state is terminal for scoring; the adaptor may reject further actions or record them as no-ops. Gym-sokoban's action/state conventions are a useful compatibility reference, but its exact reward and API are version-dependent and must be inspected before use ([repository](https://github.com/mpSchrader/gym-sokoban), [implementation](https://github.com/mpSchrader/gym-sokoban/blob/master/gym_sokoban/envs/sokoban_env.py)).

For search, derive player reachability by flood fill with boxes treated as obstacles. A push is legal only if the player can reach the square behind the box and the destination is free. Include the complete symbolic board, player coordinate, box coordinates, goals, last action, and counters in each observation; these fields let Astra reason without a screenshot.

## Pace and tier

This is Tier 1. The board changes only after an agent action and there is no real-time clock, opponent, animation, or network server. Astra may spend seconds to a minute planning between actions without an in-game penalty; the only practical limit is the harness wall-clock and action cap. Set the adaptor's documented decision budget to 60 seconds per model turn (and make it configurable), while treating the game's intrinsic decision latency as unbounded. If a run-level timeout is needed, timeout handling belongs to the harness rather than the game rules.

## Sources and scope

Rules and notation: [Wikipedia](https://en.wikipedia.org/wiki/Sokoban), [Sokoban Wiki level format](http://sokobano.de/wiki/index.php?title=Level_format). Deadlock and solver terminology: [Deadlocks](http://sokobano.de/wiki/index.php?title=Deadlocks), [Solver](http://sokobano.de/wiki/index.php?title=Solver), and Junghanns & Schaeffer, *Sokoban: Enhancing General Single-Agent Search Methods Using Domain Knowledge* ([DOI](https://doi.org/10.1016/S0004-3702(01)00109-6)). Boxoban is an appropriate offline corpus; obtain and pin its revision and license from [google-deepmind/boxoban-levels](https://github.com/google-deepmind/boxoban-levels) and [LICENSE](https://github.com/google-deepmind/boxoban-levels/blob/master/LICENSE). The local source-status record notes that these URLs were not reachable from this sandbox, so release details must be verified when staging the corpus.
