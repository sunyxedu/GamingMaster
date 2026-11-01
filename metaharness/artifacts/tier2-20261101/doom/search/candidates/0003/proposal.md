# Candidate 0003: Jev-only control with an anti-stall execution guard

## Parent and targeted hypothesis

Parent: `candidates/0000-tier2_jev_only/`. I kept its architecture: one fresh Jev `Choice` request per control decision, the complete observation as state, no Astra calls, and the same action bridge. The one targeted change is a deterministic stuck/recovery watchdog between Jev and `ctx.act`.

**Hypothesis:** the parent's largest loss is not merely Jev latency or lack of a long route plan. On several maps Jev settles on `forward` after the initial corridor, and ASYNC_PLAYER faithfully holds that action against a wall or blocked route. A cheap position-based recovery can break this absorbing loop and let subsequent Jev observations contain a new view. It should improve survival/progress without spending any additional model budget.

## Trace evidence

The parent trace repeatedly shows the same failure, including no combat pressure:

- `trace_seed1002.jsonl`: the episode ended by `call_budget` at score `4.3196`, after `4000` System-1 calls and `22717` game ticks (`18718` stale ticks). The action distribution was `forward: 4000`; around call 11 and still at call 4000 the state was `MAP03`, `player=(224, 912, 0)`, angle `270`, health `100`, kills `0`.
- `trace_seed1004.jsonl`: score `4.3183`, `call_budget`, `4000` calls and `18878` stale ticks. The action distribution was `forward: 3999` (one malformed/empty answer); the player stopped at `MAP01`, `(688, -192, -64)`, angle `0`, health `100`, with no kills.
- The same pattern appears in seeds 1006, 1008, 1012, and 1014: each consumed all 4000 Jev calls, stayed at a fixed position after the first straight run, and never reached a terminal game state. In contrast, the parent traces that ended in `game_done` had varied actions and did not spend thousands of calls at one coordinate.

This is also consistent with the parent's overall `83%` stale-tick rate: a held action is useful while it advances through a corridor, but disastrous when it has become an unobserved collision loop.

## What changed

At each Jev decision I retain the previous and current `player.x/y/z` from the public observation. If the previously posted action is translational and the 3-D displacement is less than 8 map units for three consecutive decision observations, I classify that as an execution stall. The threshold is deliberately far below ordinary movement over the roughly 5--7 game tics between parent decisions, and including `z` prevents a moving lift from looking stationary.

On a stall, the bridge temporarily overrides the Jev choice with a legal three-step maneuver, alternating sides between recoveries:

1. retreat diagonally (`backward_right` or `backward_left`),
2. turn on that side, and
3. move diagonally forward on that side.

If a reduced test action set omits one of these, the code chooses the longest legal fallback (backward/turn/strafe). The queue is completed even if the first step moves, so the view is actually changed rather than immediately returning Jev to the same heading. The normal Jev question and cadence are otherwise unchanged; no Astra calls or extra System-1 calls are introduced. Every posted action is checked against `action_spec.names`.

## System-1 interface and bridge

### State schema

The state passed to Jev is the adaptor's complete compact JSON observation, unchanged from the parent:

- episode identity and timing: `game`, `interface`, IWAD/source, `map`, `skill`, `seed`, `tic`, elapsed/engine/controller ticks;
- survivability and weapon facts: `health`, `armor`, `dead`, `on_ground`, `attack_ready`, selected `weapon`, selected ammo, `ammo`, and `weapons`;
- progress and feedback: `kills`, `items`, `secrets`, `damage_taken`, latest/total reward and terminal fields;
- kinematics: `player.x/y/z/angle` and velocity;
- currently rendered `visible` labels, including id/name/category, pixel box, world position and velocity.

The watchdog reads only `player.x/y/z` and the prior posted action locally. It does not add omniscient geometry or hidden map data to Jev's state.

### Question

Each request batches one typed question:

- `act`: `Choice` over exactly the legal `game.action_spec.names`, with each option described by `action_spec.descriptions`. Its instruction is to choose the best immediate action that preserves a retreat path. There are no numeric questions and no generated action strings.

The typed `.value` is accepted only when it is a legal action name. A malformed or arbitrary stand-in answer becomes the advertised noop. A `TimeoutError` or ordinary failed Jev call posts no replacement and waits for a tick, so ASYNC_PLAYER keeps holding the last valid action. Budget/episode/model-unavailable exceptions retain the contract's normal termination behavior.

### Freshness, thresholds, and fallback

- A stall sample is considered only after a fresh observation and only when the last action was one of the movement or movement-plus-fire actions. `use`, weapon pulses, stationary fire, and turning do not accumulate stall evidence.
- `displacement < 8.0` counts as no progress; three such samples trigger recovery. Recovery actions are posted after the current Jev call and are associated with that current observation; while the queue runs, the game continues ticking and the queue is not interrupted by another Jev answer.
- The fallback is intentionally local and reversible, not a claim about the hidden map. It alternates left/right to avoid repeatedly choosing the same side of a blocked corridor. If it fails, a later three-sample stall can trigger the opposite maneuver again.
- The initial/missed-decision behavior remains the adaptor's noop; an inference failure leaves the previous held action in place. Terminal status is still governed by `ctx.done` and the game.

## Falsifiable outcome

The hypothesis is supported if evaluation traces show `stuck_recovery` events followed by changing position/angle, fewer long fixed-coordinate runs and call-budget endings, and higher kills/progress or exit success on the six parent failure seeds without a large regression on the parent `game_done` seeds. It is refuted if the maneuvers fail to change the fixed states, or if they interrupt ordinary lifts/combat often enough that score falls despite eliminating the forward loops. The experiment therefore isolates execution-level recovery rather than conflating it with a new planner or a larger prompt.
