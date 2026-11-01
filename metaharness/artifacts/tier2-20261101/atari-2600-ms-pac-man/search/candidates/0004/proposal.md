# Candidate 0004 — graph execution with local hazard arbitration

## Parent and diagnosis

Parent: `candidates/0002-tier2_idle/` (the idle reference).  The strongest
available comparator is `candidates/0000-tier2_jev_only/`.

The parent proves that merely waiting is worth 60 points, but it also proves
that `noop` is not a safe policy: in its seed-2 trace, with no posted actions,
the opening carries Ms. Pac-Man from `[75, 99]` to `[29, 99]` and eats six
dots by ticks 68–85, then the episode ends at tick 483 with three deaths.
The Jev-only trace improves this to 110 on the same seed, but still ends at
tick 393 with lives 0 and no maze cleared.  At tick 247 it is moving down at
`[29,107]` while the orange ghost is at `[46,99]` and closing along the
starting lane; it continues down to `[29,155]` while the red ghost reaches the
bottom corridor.  The controller has no representation of a legal turn or an
escape route with which to react.

More generally, `0000` sends a full raw observation (about 7,750 input tokens
in the trace) to Jev once per synchronous request.  The requests take roughly
0.16–0.20 s, while a tick is 0.0667 s; its leaderboard row consequently has
63.4% stale ticks and 1,231 S1 calls.  Its first several answers in the bad
seed are low-confidence `noop`/direction guesses while the player is still in
the automatic opening.  It manages only 27.5 visible collectibles per episode
on average, clears no maze, and loses all lives.  There is no `lessons.md` in
the supplied archive to contradict this diagnosis.

## Hypothesis

The flat/low result is primarily a control-structure failure, not a shortage
of model calls.  Ms. Pac-Man's visible pill coordinates are a sufficiently
stable description of the CX2675 corridor lattice.  If code builds and retains
a graph from those coordinates, routes through currently remaining dots, and
posts a cardinal direction on every tick, it can keep moving and pre-turn
without waiting for Jev.  Jev should only arbitrate a small set of already
legal branches when a junction or a measured closing ghost makes the route
ambiguous.  A short deterministic ghost guard should override dot greed only
inside a collision horizon.  Astra is too slow for movement, so it should
choose a risk policy in the background rather than emit joystick commands.

This tests a structural prediction: the same model used as a local recognizer
should be more useful when given three candidate corridors and named danger
facts than when asked to select among nine actions from a raw RAM dump.

## What changed

`0004/harness.py` changes all three layers relative to the idle parent and the
Jev-only comparator:

* **Code/map layer.**  On the first observation it treats each visible pill
  coordinate as a corridor node, links adjacent same-row/same-column dots
  (12-pixel spacing), and retains the graph after dots disappear.  This is not
  an imported arcade map: it is rebuilt from the current ROM's observation.
  It detects a side-tunnel wrap only after an observed coordinate wrap.  A
  BFS route targets the nearest still-visible dot, with a direction-preserving
  clockwise tie break, and buffers the next turn by posting that cardinal
  action on every tick.  Maze counters reset the graph for a new maze.
* **Code safety layer.**  Ghost coordinates and recent velocities are reduced
  to a short-horizon closing/corridor test.  A close approaching ghost causes
  the fallback to select the safest computed branch, or a path toward a
  currently decoded power pill.  It never uses normal ghost RGB as a
  frightened flag.  The route, map and model answers are invalidated after a
  death or maze change.
* **System-1 layer.**  Jev is launched with `ctx.spawn`, so the current
  direction continues while a roughly 0.18 s answer is in flight.  It is
  queried only near a degree-3-or-more graph node, on a planned turn, or at a
  hazard; a compact derived state replaces the raw 128-byte RAM and 150-pill
  list.  Two identical `Choice` questions use opposite option order to reduce
  first-option bias.  A third, independent `Noul` judges whether the
  already-computed power escape is safer.  Code accepts a move only if both
  Choices agree, the action is still legal and fresh, and their top
  probabilities clear the consequence-dependent threshold.
* **System-2 layer.**  One Astra call is launched asynchronously at opening
  calibration.  It sees the observed graph size, pill/power locations, ghost
  positions/velocities, lives and score and returns an explicit
  `MODE=survival|balanced|score` policy.  The parser accepts only a nearby
  explicit mode marker; arbitrary or malformed text leaves the deterministic
  `balanced` policy unchanged.  Astra never blocks the opening or a turn.
  The last life forces `survival` regardless of Astra's suggestion.

The loop calls `ctx.act` before every `ctx.wait_for_tick`, including while
model futures are pending.  A late/failed model answer therefore falls back
to the graph route or ghost guard instead of raising an uncaught timeout.
Only contract-normal budget/episode endings are returned; model-unavailable
errors are allowed to stop the run as required by the harness contract.

## Jev interface and bridge

Every Jev request carries strict JSON with this schema (all fields are
concrete values, not references to inaccessible code):

```text
observation:
  frame, tick, score, lives, deaths, player_xy, player_direction,
  visible_pills, maze_clears
plan:
  risk_mode, fallback_action, legal_actions, at_graph_node,
  graph_degree, policy_version
threats:
  summary, nearest_distance_bucket, closing_same_lane_count, immediate
options:
  <cardinal action>:
    continues_code_route, corridor_risk, is_power_escape, branch_degree
escape_option: <cardinal action or "none">
```

`corridor_risk` is computed by code as `open`, `caution`, `danger`, or
`unknown`; Jev is not asked to count pixels, perform BFS, infer the wall map,
or add score values.  The state contains no raw RAM and no unbounded pill
array, so the question is a local recognition task.

The two `Choice` questions are independent duplicates:

1. choose exactly one value from `plan.legal_actions`, preferring the code
   fallback unless the immediate-threat/option facts justify changing it;
2. the same choice with criteria in reverse order.

`escape` is an independent `Noul`: whether `escape_option` is safer now than
the fallback.  Its answer is used only for explanatory arbitration and only
when code already reports an approaching hazard and has a legal power route.

A returned choice is usable only when:

* its request is no more than 24 native frames old (six default ticks), has
  the same death count, and is still at the requested node unless the current
  hazard is immediate;
* both Choices return the same currently legal cardinal name;
* both probabilities for that name are at least 0.34 at a normal junction or
  0.50 when a hazard is present.  Missing probabilities are uncertainty, not
  permission to act.

An accepted answer is a temporary override for at most 16 native frames and
is discarded when its node/legality changes.  A disagreement, low confidence,
malformed stand-in answer, timeout, or stale answer is ignored without parsing
or execution errors.

## Fallback and scheduling policy

The graph route is the primary policy.  At a normal corridor it is posted
continuously, so a slow Jev/Astra call adds no stale ticks.  Immediate measured
collision risk invokes the deterministic escape guard; an approaching risk is
also treated as an escape in `survival` mode.  If no graph is available, the
player's observed cardinal direction (then `left`) is used, never a malformed
model string.  `noop` is only a contract fallback, not an intentional opening
strategy.

Astra is deliberately not re-entered for every turn: its 29 s one-word
latency is incompatible with the 67 ms game clock.  Its background policy can
change the route's risk tolerance when it arrives; the code guard and the
last-life rule remain authoritative.  Jev is the fast local exception handler,
not the route planner.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a materially higher mean score than 275 with lower
stale-tick fraction, more than the comparator's 27.5 visible collectibles,
and ideally a maze clear or lives remaining.  Trace evidence should show
long cardinal runs following graph edges, fresh actions posted while Jev is
pending, and Jev changes concentrated at junctions/hazards rather than every
tick.

Refutation would be a score near the 60 idle baseline despite a connected
observed graph, repeated deaths on graph edges whose code risk labels were
`open`, or a large number of stale/invalid Jev overrides.  If the graph links
non-corridors, the trace should show the first offending coordinate edge;
that would support tightening the observation-derived edge rule rather than
returning to raw-state per-tick deliberation.  If deaths occur with a ghost
inside the projected horizon, the relevant follow-up is to revise the guard's
projection/escape features, not to increase Astra frequency.
