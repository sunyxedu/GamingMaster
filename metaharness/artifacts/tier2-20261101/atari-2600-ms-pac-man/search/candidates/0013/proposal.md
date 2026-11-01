# Candidate 0013 — defer reserve pills until a life transition

## Parent and concrete diagnosis

Parent: `candidates/0010/`. I retain its observation-derived corridor graph, continuous
cardinal executor, ghost guard, asynchronous Astra policy, and asynchronous Jev branch
arbitration. The parent is the current best candidate at **1861.7** mean score with
zero stale ticks, but it still ended all six training episodes with three deaths, zero
maze clears, and zero lives remaining.

The parent did not actually preserve its reserve before the first collision. In its
worst episode, `candidates/0010/trace_seed3.jsonl`, the trace records
`reserve_escape_started` for `[148,14]`, then `power_escape_consumed` with
`power_eaten: 3`, followed by `reserve_escape_started` for `[148,146]` and
`power_escape_consumed` with `power_eaten: 4`—all before the first death. The first
loss is then observed at frame **1896**, score **1000**, with Ms. Pac-Man at
`[81,99]` moving right and three ghosts clustered at approximately `[81,81]`;
the final loss in that episode leaves score **1500** and 30 visible pills. The same
pattern appears in seed 0: both reserve targets are consumed before the first death,
which is observed near `[61,109]` at frame **2580**. Thus the parent’s
`level-2 hazard -> start reserve escape` rule converts its emergency detector into a
routine four-pill route, without saving the recurrent centre/lower-corridor collision.

This is not evidence that power pills are useless. It is evidence that the parent’s
hazard trigger is too eager to spend a supposedly protected resource. A reserve that
has already disappeared cannot help on the next life.

## Isolated hypothesis

**Only trigger a reserve-pill route after a life transition (or when already on the
last life), not merely because the code labels a normal-life hazard “immediate.”**
The first two pills remain ordinary route checkpoints. The other pills remain excluded
from ordinary dot BFS and stay visible through the first life. After a death, the
existing committed reserve selection is still used; the route remains hard until the
pill disappears. On a last life, an already measured hazard may also use a reachable
reserve. Before those points, the existing graph/ghost branch guard must choose a
cardinal escape without spending the reserve.

This isolates resource timing rather than adding more model calls, changing the map,
or changing the Jev interface. It tests whether the parent’s three-death result is
primarily caused by premature reserve consumption.

## What changed

Starting from 0010, the only gameplay-policy change is the reserve gate. In
`_escape_action`, starting a new reserve route is gated by `force_escape_after_death`
or `lives <= 1`. The parent’s post-death path still sets
`force_escape_after_death`, selects a reachable target, rotates away from a failed
target, and commits to it. The Jev state builder uses the same gate: before a life
transition it does not expose a reserve direction as `escape_option`, so the Noul
cannot reintroduce the parent’s early diversion through the asynchronous bridge.
The ordinary checkpoint limit, reserve-node exclusion, route search, candidate ghost
risk, Jev questions/thresholds, Astra policy, and tick scheduling are unchanged.

When an immediate hazard occurs on life one or two without a committed escape, the
existing deterministic `_escape_action` scores legal graph neighbors by projected
ghost distance, corridor danger, and continuity. It never waits for Jev and never
posts `noop`; Jev can still arbitrate only a fresh, legal local branch. This is the
fallback that makes the experiment meaningful if saving the pill is the wrong choice.

## Tier-2 Jev bridge

The state sent to Jev remains a strict JSON summary, not raw RAM:

```text
observation: {
  frame, tick, score, lives, deaths, player_xy, player_direction,
  visible_pills, visible_power_pills, pill_active, maze_clears
}
plan: {
  risk_mode, goal_kind, goal_xy, life_phase, fallback_action,
  legal_actions, at_graph_node, graph_degree, policy_version
}
threats: {summary, nearest_distance_bucket, closing_same_lane_count, immediate}
options: {
  action: {continues_goal_route, corridor_risk, reaches_checkpoint, branch_degree}
}
escape_option: action-or-none
```

One asynchronous `ctx.decide` request (the parent’s `timeout_s=1.2`) batches two
identical `Choice` questions over the legal cardinal options with reversed option
order, plus a `Noul` asking whether the already-computed power-pill escape is safer.
The bridge accepts a direction only when both choices agree, their selected-option
probabilities pass the parent’s ordinary (`0.40`) or hazard (`0.55`) gate, the Noul
passes its `0.60` true gate when used, and frame/death/node/policy-version freshness
and current legality still hold. Answers older than 18 native frames are dropped.
A normal Jev answer cannot abandon a checkpoint or committed escape; malformed,
uncertain, stale, contradictory, or late answers are ignored.

The deterministic fallback is the current route, then the ghost guard, then a legal
cardinal hold/reverse. Astra remains background-only and is parsed only for an
explicit `MODE=survival|balanced|score` marker. Garbage stand-in model output is
therefore harmless during preflight.

## Confirmation and refutation

Confirmation would be a trace with the reserve pills still visible through the first
life, a post-death `reserve_escape_selected`/`reserve_escape_started` event followed
by actual pill consumption, and either fewer than three deaths, a later first/second
demise, or a maze clear at comparable early score. The strongest confirmation is
surviving the recurrent `[81,99]` or `[61,109]` approach because a reserve remains
available for the next life.

Refutation would be the same three deaths at comparable frames while reserve pills
remain untouched, or a large first-life score collapse caused by the no-diversion
policy. If reserve timing helps first-life coverage but post-death consumption still
fails at the same coordinates, the next iteration should target the graph’s ghost
projection/turn geometry rather than further alter pill allocation.
