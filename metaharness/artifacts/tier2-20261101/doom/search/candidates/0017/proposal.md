# Candidate 0017 — heading-frontier exploration with a position-stall escape lease

## Parents and hypothesis

This is a crossover of `candidates/0011/` and `candidates/0003/`.

- **0011's mechanism:** a code-owned, continuously refreshed sweep/escape motor, with local `heading_visits` used to choose an under-observed turn. Jev only supplies bounded macro hints and Astra only revises a bounded policy, so slow inference cannot stop the ASYNC_PLAYER motor.
- **0003's mechanism:** a cheap execution watchdog that detects when the action selected by Jev is not changing the player's position and forces a short retreat/turn/translation sequence.

**Hypothesis:** 0011's heading frontier reduces translational loops but has a distinct remaining trap: the controller can spend long periods turning without changing position. In the worst 0011 episode, the final state was at `(752,-304,-64)`, health 6, and the last 40 actions were almost entirely alternating `turn_left`/`turn_right`; the trace logged 17 collision escapes overall but ended in a fixed-position turn loop. In the parent action order, immediate-target combat was checked before draining a committed escape queue, so visible threats could also interrupt an intended escape. The older 0003 guard would not catch the turn-only case because it accumulated stall evidence only for translational actions. Combining the frontier motor with a position-based recovery lease should break both wall-push and turn-only traps, expose a new view, and preserve the leader's low-staleness behavior.

This is intentionally an execution hypothesis, not a claim that an exit label is always rendered. If it works, the extra movement should turn 0011's already high secondary exploration into more route/exit opportunities without adding map knowledge or seed-specific routes.

## Evidence and targeted change

The 0011 seed-1012 trace had 2,737 ticks and ended at score 3.213 with health 6. It used 1,179 `turn_left`, 498 `turn_right`, and only 85 `backward` actions. The final observations stayed at `(752.0,-304.0,-64.0)` while a Doom monster was intermittently visible. The harness logged 17 `macro_escape` events, but its final combat alignment path continued to issue turns at that fixed location. In the long 0011 seed-1008 trace, the heading frontier reached 281 measured cells and 290 escapes yet still made 10,075 left and 4,921 right turns; the episode reached the deadline without an exit. This is evidence that heading counting is useful but does not by itself prevent a turn lock.

The implementation retains 0011's local heading frontier, visible-label taxonomy, deterministic weapon/USE checks, asynchronous Jev/Astra pipeline, and per-tick legal fallback. It adds three code-owned pieces:

1. `update()` accumulates **absolute position displacement**, including Z, even when the prior action was a turn, fire, USE, or weapon pulse. A sample with less than 0.75 map units of displacement adds its elapsed tics to `stationary_tics`; any real displacement resets it. This catches the failure class that 0003 did not observe.
2. When stationary for 32 tics and not in a recent recovery (90-tic cooldown), `action()` starts a bounded `position_stall` escape before ordinary target alignment. Close projectiles still outrank it. The escape uses the frontier-selected heading, five tics of diagonal backward movement, the measured turn, and fourteen tics of diagonal forward movement. The diagonal departure is the 0003 retreat idea combined with 0011's least-observed heading, rather than another fixed left/right maneuver.
3. A committed escape is consumed before ordinary goals and combat on subsequent ticks (except the projectile emergency). Starting it clears the stalled counter and logs `position_stall_recovery`, so it cannot be repeatedly restarted by the same frozen observation.

No omniscient map, object list, training-seed route, or additional model call is introduced. The watchdog is geometry-agnostic and only uses public player kinematics.

## System-1 interface and bridge

Jev receives the compact derived JSON state already used by 0011:

```text
{
  task,
  rules,
  health: critical|wounded|healthy,
  armor: none|present,
  damage_recent: boolean,
  motion: {phase, blocked, novelty, visited_cells, same_cell_visits},
  threats: [id, name, category, range, direction, level,
            hitscan, major, body_blocker],
  local_goals: [id, name, category, range, direction, level, ...],
  goal: {kind, range}|null,
  macro: {hand, sweep, turn_bias, previous_action}
}
```

Code computes ranges, bearings, threat classes, motion cells, heading counts, damage recency, weapon legality, and the position-stall counter. Jev does not perform arithmetic or invent actions. Each request batches:

- `route: Choice` over `advance`, `turn_left`, `turn_right`, `use`, `retreat`, `unknown`; an accepted turn only biases the next frontier macro, `use` requests one bounded pulse, and `retreat` can start a semantic escape only when no immediate threat owns control.
- `dodge: Choice` over `left`, `right`, `back`, `none`; this only biases combat strafing.
- `open: Noul`; a value of at least `0.72` requests a bounded USE pulse.

A Choice is accepted only for a legal option with selected-option probability at least 0.57 for `route` or 0.54 for `dodge`; an explicit confidence below 0.10, malformed answer, timeout, ordinary runtime error, or age above 55 Doom tics is ignored. `Noul` needs at least 0.72. The deterministic motor continues while the request is pending, and all late answers are associated with their request tic rather than applied to a new episode. A Jev failure holds no new semantic hint; the code motor remains executable. `ModelUnavailable` and budget/episode exceptions follow the harness contract.

Astra is launched at the same bounded initial/review cadence as 0011, at most twice, with a short state/history payload. It may return only `hand`, `sweep`, and `combat` JSON fields. Prose, unknown fields, stale/cancelled answers, and malformed output leave the prior policy intact. Astra never chooses a button, target, heading, or hidden route.

## Fallback and tier-2 timing

Every processed observation posts a legal deterministic action and waits for the next engine tick. Jev runs in a background future with a 1.35-second timeout and the motor uses the prior/cached hint while it is pending. Astra runs in a background future with a 180-second timeout; its answer can only affect a future macro boundary. Terminal/deadline paths cancel outstanding futures without joining a new control loop. The initial action remains the advertised noop until the first post.

Priority in the action bridge is:

1. close-projectile lateral dodge;
2. already committed escape lease;
3. new 32-tic position-stall escape;
4. immediate threat combat and legal weapon selection;
5. visible interaction/pickup steering and bounded USE;
6. collision/novelty frontier escape and normal sweep.

The 32-tic threshold is long enough not to interrupt ordinary one-tic turning or a lift transition, while much shorter than the multi-minute deadline. The 90-tic recovery cooldown prevents an immobile door face from causing a busy recovery loop; the existing collision/novelty mechanisms still handle a genuinely blocked route. The final legality check maps every result to `action_spec.names`.

## Falsification

The hypothesis is supported if traces show `position_stall_recovery` events followed by changed position/heading, fewer long fixed-coordinate turn runs, fewer low-health deaths in the 0011 failure seeds, and—most importantly—any increase in `completed_by_exit` or time spent reaching new cells. It is refuted if the recovery lease cannot move the player, repeatedly ejects the controller from useful combat/door alignment, increases damage or reduces survival without increasing exit completion, or merely raises kills/items while all episodes still fail at the same geometric loops. Secondary progress alone is not confirmation: the primary test is exit success.
