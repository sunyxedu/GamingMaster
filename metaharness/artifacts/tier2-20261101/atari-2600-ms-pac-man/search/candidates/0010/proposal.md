# Candidate 0010 — reserve-aware power-pill escapes

## Parent and diagnosis

Parent: `candidates/0008/`, with the continuous observation-derived graph,
zero-stale-tick loop, and local Jev arbitration retained. Its mean was 1608.3,
well above 0007's 1028.3, but every episode still ended with three deaths and
zero maze clears.

The new diagnosis comes directly from the 0008 traces. The checkpoint mechanism
consumed all four power pills early in every run: for example, seed 2 reports
`power_checkpoint` for `[8,147]`, `[8,15]`, `[148,14]`, and `[148,146]` before
its first death at frame 2008. The first death observations then repeatedly
show Ms. Pac-Man at the spawn/centre crossing (`[81,99]`) with several ghosts
clustered immediately above (`[79-80,81]`), while the controller continues
right. Later losses recur at the lower branch (`[61,109]` in seeds 0, 2 and 4)
and the final life has no pill left to convert a hazard into an escape. Thus
0008 improved coverage but spent its only recovery resource before it needed it;
adding more local Jev choices would not restore a consumed pill.

## Hypothesis

A small **reserve of power pills**, combined with a committed escape route, will
improve survival without giving up 0008's useful early coverage. The controller
should visit at most two pills as ordinary checkpoints, then route to dots while
leaving the remaining pills untouched. When a measured immediate hazard occurs,
or after a life loss, it should choose one remaining reachable pill and commit
to that target until the object is actually consumed. If an escape approach
still ends in a death, that particular target is avoided on the next life and a
different reserve is tried.

This tests a structural resource-allocation hypothesis: the issue is not merely
that the ghost threshold is too low or Jev is too slow, but that the route policy
has no reserve when the repeated centre/lower-corridor trap is encountered.

## What changed

Starting from 0008, the harness changes only the resource and recovery layer:

* `EARLY_PILL_LIMIT = 2` makes the first two reachable pills normal checkpoint
  goals. Once they are consumed, ordinary dot BFS cannot select another pill.
* Remaining visible pills are a reserve. A level-2 hazard can start a reserve
  escape immediately; a level-1 hazard can do so only under the existing
  survival policy. A death explicitly requests a reserve escape on the next
  life, even if the early-pill budget is already spent. Normal dot BFS excludes
  reserve nodes both as goals and as intermediate nodes, so the reserve cannot
  be consumed accidentally while sweeping dots.
* A reserve target is represented as `goal_mode="escape"` and is a hard route
  goal. The route is recomputed only as the player reaches graph nodes or when
  the target/maze/life changes; Jev cannot abandon it at an ordinary junction.
  This avoids the parent's one-tick power diversion followed by a return to the
  same dot route.
* Pill disappearance is still the only confirmation of consumption. It clears
  the escape goal and starts the short `pill_active` bookkeeping interval; no
  ghost RGB is interpreted as frightened state.
* If a life ends while a committed reserve target is still visible, that target
  is stored in `failed_escape_points` and excluded from the next reserve choice.
  This is soft target memory, not a fabricated wall, and it resets on a maze
  change.
* Continuous cardinal actions, graph legality, ghost projection, the two
  reversed-order Jev Choices, and the asynchronous Astra risk policy remain
  unchanged. Astra still cannot block a tick or post a joystick action.

## Jev state, questions, and bridge

The state remains a strict JSON object derived by code:

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

One `ctx.decide` request batches two `Choice` questions over the same legal
cardinal actions, with the option order reversed in the second question, plus
one independent `Noul` about the already-computed pill escape. Jev is asked to
recognise a local branch, not calculate paths, compare numeric distances, read
RAM, or infer frightened colour. The returned action is used only when both
Choices agree, the selected probabilities clear the existing ordinary/hazard
thresholds, the request is fresh (at most 18 native frames), the node/death/
policy version is compatible, and the action remains legal. A normal answer
cannot override either `checkpoint` or committed `escape` routing; an immediate
hazard may use the still-legal Jev choice. Missing, malformed, low-confidence,
disagreeing, or late answers are ignored.

The deterministic fallback is always executable: graph route first, then the
velocity/proximity ghost guard, then a cardinal hold/reverse if no graph edge
is available. A late Jev answer never causes a wait or `noop`. Astra's text is
accepted only for an explicit `MODE=survival|balanced|score` marker; arbitrary
stand-in text is harmless.

## Scheduling and error handling

The harness posts one valid cardinal action and waits for exactly one tick on
every loop iteration, so model latency produces no stale ticks. Jev calls run
through `ctx.spawn` with a short timeout and are tagged by frame/death/node/
policy version. Astra runs at most twice in the background with a long timeout
and is never on the action path. `BudgetExceeded`, `EpisodeOver`, timeouts and
ordinary model failures are treated as normal late-inference fallbacks; a
persistent `ModelUnavailable` is allowed to propagate as required by the
contract. No custom background loop survives `ctx.done`.

## Evidence that would confirm or refute the hypothesis

Confirmation would show fewer than three deaths, a maze clear, or a later death
than 0008 at comparable early score, together with trace events showing two
early checkpoints followed by a reserve target selected at a hazard or life
transition. The strongest confirmation is surviving the recurring `[81,99]`
and `[61,109]` states with a reserve pill still visible or consumed for escape.

Refutation would be unchanged three-life termination after the reserve remains
available, repeated `reserve_escape_selected` targets that do not alter the
route, or a large score collapse before the first death caused by reserving too
many pills. If reserve consumption occurs but deaths remain at the same states,
the next iteration should inspect the geometric collision projection/turn
hysteresis rather than add more model calls.
