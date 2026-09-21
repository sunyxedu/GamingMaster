# Snake — study brief

Research date: 2026-09-21. Scope: single-player, classic grid Snake. No account, multiplayer, power-ups, or specific commercial release was requested. This study recommends a local implementation in the adaptor phase; it does not implement the game or a player.

## Rules and objective

Steer an automatically moving snake toward food. Eating increases its length; colliding with a wall or its own body ends the run. The practical objective is to collect as much food as possible and ultimately fill the board. Arrow keys commonly select direction; direct reversal is disallowed. These shared mechanics are documented by the [Original Snake recreation](https://theoriginalsnake.com/how-to-play). Implementations disagree on board dimensions, wrapping, speed changes, and point multipliers: for example, [AppsFactory](https://www.appsfactory.dev/games/snake) distinguishes wrapping from solid-wall modes, while [1700 Games](https://1700games.com/arcade/classic-snake) uses ten points per food and accelerating levels. These are examples of variants, not authoritative specifications for every Snake release.

The following **study-selected ruleset**, `classic-walled-v1`, resolves those ambiguities:

| Property | Required behavior |
|---|---|
| Board | Empty rectangular grid; default 10 × 10; solid exterior walls. |
| Snake | Initial length 3; ordered head-to-tail body; one cell of motion per tick. |
| Food | Exactly one food in a uniformly sampled empty cell. Replenish immediately after eating unless the board is full. |
| Growth | One extra segment per food; the tail stays in place on that move. No delayed growth. |
| Direction | `UP`, `RIGHT`, `DOWN`, `LEFT`; opposite-direction requests are ignored and motion continues in the current direction. |
| Collision | Leaving the grid or entering retained body cells is fatal. Entering the old tail cell is permitted on a non-growing move because that tail leaves simultaneously. |
| Score | One point per food, starting at zero. Length = 3 + score. |
| Victory | Length equals board area; stop immediately, with no attempt to spawn another food. |
| Pace | Fixed 0.200 seconds per cell, no automatic acceleration in the primary profile. |
| Other rules | No starvation, shrinking, obstacles, rivals, wraparound, or bonus foods. |

The simultaneous tail movement and food-growth convention agrees with the formal movement model in [Graafsma, Manthey, and Skopalik, sections 2.2–2.3](https://arxiv.org/html/2506.21281v1). Their paper adds other theoretical conventions; this study does not adopt its repeated-position loss rule. The exact board size and tick duration above are benchmark decisions, not measured Nokia timings.

## Timing and tier

**Tier 2.** A running episode advances every 0.200 seconds regardless of observations or pending model calls. A snapshot taken just after a tick provides at most 0.200 seconds to affect the next move; later snapshots provide less. Missing a deadline continues the last heading, which can be fatal on that next tick. Several seconds of straight corridor may occasionally provide slack, but it is state-dependent and cannot be assumed.

Astra's seconds-to-minute calls cannot steer directly. Jev's approximately 0.5-second calls also span multiple ticks. Generated local control code must execute the current policy every tick, validate model suggestions against the current state, and remain functional when neither model responds. Jev selects among computed strategy options asynchronously; Astra designs or revises policies. This is a timing requirement, not evidence that Jev can reliably choose every turn.

A synchronous `step(action)` environment would instead be tier 1 **for that interface**, with no gameplay deadline. Provide such a mode only for debugging/replay, label it separately, and do not report its results as real-time performance. Pausing while models think changes the task.

## Score and progress reporting

Report `food_eaten` as the raw game score, `length`, `board_area`, `won`, `ticks`, and `end_reason`. For comparisons across board sizes, the primary scalar is `food_eaten / (board_area - initial_length)`, in [0, 1]. Also report win rate over held-out seeds. Among equal completion scores, fewer ticks is better; report efficiency separately rather than paying positive reward for surviving in circles. A collision does not erase food already earned.

The evaluator may stop a run after `2 × board_area²` ticks. This is an external truncation, not a classic Snake death or starvation rule. Mark it `truncated=true`, preserve the earned score, and distinguish it from collision or victory. At a 100-cell board the cap is 20,000 ticks, or about 66.7 minutes at the selected pace; ordinary successful runs can still last minutes. Faster offline evaluation is useful but belongs to a separately labeled profile.

## Confidence and limits

Rules are well supported; exact timing is selected explicitly because no platform was specified. External repositories were inspected through documentation/source, not installed or executed. Network retrieval through the web tool worked, but a shell `curl` request failed in this workspace. No software installation or account setup has occurred. See `interfaces.md` for the complete adaptor contract and `strategies.md` for the control design.
