# Candidate 0006 — strategic subgoals, code-owned movement

## Parent and diagnosis

Parent: `candidates/0001-tier2_astra_only/`.  The strongest comparator is
`candidates/0004/`, whose graph executor is retained as the deterministic base.

The parent establishes the failure of putting a real-time joystick behind
Astra: its 29-second-scale calls left all 483 ticks stale and every seed ended
at 60 points.  The useful part of that experiment is the opening observation
and the fact that the ROM continues safely for a short time before a first
command; it is not evidence that Astra can be the per-tick controller.

Candidate 0004 proves the opposite structural point: posting a code action on
every tick raised the mean from 275 to 778.3, with zero stale ticks and 62–107
dots eaten.  However, every episode still lost all three lives and cleared no
maze.  Its trace exposes a specific remaining failure.  On seeds 0 and 3 the
first life reaches the bottom-right sweep at about `[135,159]` and 240 points;
then accepted Jev direction overrides alternate around the threatened branch:
for seed 0 they include `right` at frame 604, `left` at 616, 628, 652, 664,
then `right` at 712, `left` at 724, `right` at 740, and `left` at 752.  The
same pattern appears in seed 3.  The controller is still moving, but Jev is
being asked to choose a joystick direction while all supplied branches are
marked dangerous.  The controller never consumes a power pill in the recorded
runs; all four remain visible in the final observations.  The second life has
a similarly repeatable lower-lane death around `[61,109]`, after a long `down`
commitment.  Thus 0004's graph is useful for dot coverage, but the model bridge
can fight the graph at exactly the time a stable escape commitment is needed.

## Hypothesis

The next bottleneck is not stale inference or insufficient local direction
judgments.  It is **authority at a junction**: a slow/uncertain recognizer is
allowed to replace a legal, precomputed movement direction many times while a
ghost is close, causing reversals and preventing the route from reaching an
escape pill.  If Jev chooses only an abstract local subgoal (`continue dots`,
`take a power pill`, or `safe branch`) and code owns the entire path to that
subgoal, the controller should retain 0004's dot throughput while escaping
rather than oscillating.  The immediate collision guard remains code-owned and
can force a power-pill route even if Jev is late or malformed.

Astra is changed from a policy that can indirectly affect individual branch
choices to a slow author of the maze-level objective: preserve lives and dots,
choose whether power pills are emergency-only or acceptable as a score
opportunity, and identify when a route should be abandoned.  It never emits an
action and never blocks the tick loop.  Jev is a fast recognizer of *which
already-computed subgoal is appropriate now*, not a planner or joystick driver.

## What changed

* The observation-derived pill graph, persistent map nodes, BFS dot sweep,
  coordinate normalization, per-tick action posting, and deterministic fallback
  from 0004 remain.  A malformed model answer cannot reach `ctx.act`.
* Directional Jev overrides are removed.  A Jev answer can set a short-lived
  subgoal, but code then computes and commits to a graph path.  The chosen
  escape direction is held until a graph node/target transition, not replaced
  by a new answer every few ticks.
* Immediate hazards first invoke a code safety policy.  If a visible power pill
  has a reachable path whose first branch is not already a measured collision,
  the controller routes to it.  If no such path exists it scores legal branch
  candidates by projected ghost separation and preserves the safest committed
  branch.  On the last life, an approaching hazard also enters this policy.
* Power pills are real route goals, rather than a boolean attached to one Jev
  direction.  Reaching a power-pill coordinate invalidates the escape route and
  returns to dot coverage.  This specifically tests why 0004 left every power
  pill visible.
* Jev is queried only on a new junction/escape situation, with two batched
  duplicate `Choice` questions over the named subgoals and one independent
  `Noul` about whether the computed power route is preferable.  Answers are
  accepted only when fresh, same-life, same-map, and agreeing; disagreement or
  low probability leaves the code policy in charge.
* Astra remains at most two asynchronous strategic calls (opening and after a
  life/maze transition if budget permits).  Its result is parsed only from an
  explicit `MODE=`/`PILL=` marker; arbitrary text, empty stand-in output, and
  failed calls leave a conservative deterministic policy.

## Jev interface and bridge

Each request is strict JSON and contains only derived facts:

```text
observation: frame, tick, score, lives, deaths, maze clears, player cell,
             current direction, remaining-dot count, visible-power-pill count
plan:        objective, route_kind, fallback_subgoal, route_epoch,
             legal_subgoals, current graph node, graph degree
threats:     none/approaching/immediate, nearest-distance bucket,
             closing-lane count, power_route_available,
             power_route_first_branch, safe_branch_count
candidates:  each subgoal's already-computed meaning and consequence
```

`continue_dots` means follow the existing legal dot route and do not divert
only for a speculative ghost chain. `take_power_pill` means follow the
already-computed shortest viable power route. `safe_branch` means abandon the
dot target and follow the code-selected safest branch; it does not mean invent
a direction. The `Noul` asks whether the supplied power route is safer than
continuing the supplied fallback, not whether Jev can calculate distances.

The two Choices use the same options in opposite orders to reduce the known
first-option bias.  The bridge requires the same string value in both answers,
a probability of at least .50 for an immediate hazard or .35 otherwise, and a
currently valid subgoal.  A returned power choice is never allowed to override
the code's immediate-collision check; a returned continue choice is never
allowed to suppress an immediate code escape.  The answer is tagged with the
request frame, death count, map key, and route epoch.  It expires after 32
native frames or once the player leaves the requested decision node.

## Astra interface

Astra receives a compact map summary, not raw RAM, and is asked for at most:
`MODE=survival|balanced|score`, `PILL=emergency|opportunity`, and one short
abandon condition.  `survival` is forced with one life; `emergency` still loses
to the code immediate-hazard rule but prevents unnecessary pill diversions.
Astra is launched through `ctx.spawn`, so its roughly 17–35 second latency
cannot create stale ticks.  Runtime policy is advisory and versioned; the map,
legality, ghost projection, scoring arithmetic, and route execution stay in
code.

## Fallback and late decisions

The harness posts a cardinal action before every `ctx.wait_for_tick`.  While
Jev or Astra is pending, it continues the last legal route action.  The
fallback order is: committed emergency power route; code-selected safe branch
for an immediate/last-life threat; existing dot route; current observed
cardinal direction; `left`.  `noop` is only a final contract fallback.  All
future-result parsing is defensive and catches timeout/runtime/garbage answers;
no answer is required for episode progress.

## Evidence that would confirm or refute the hypothesis

Confirmation would preserve 0004's zero stale ticks and roughly 60+ dots while
showing fewer deaths, at least one consumed power pill, a surviving last life,
or a maze clear.  Traces should show long uninterrupted cardinal segments and
subgoal changes at hazards rather than alternating direction overrides.

Refutation would be deaths at the same cells despite no model direction
override, or a score near idle because the graph route cannot reach power
nodes.  If power routes are unreachable, the next diagnosis should inspect the
observation-derived node-to-power association; if the player survives but
stalls, compare route epochs, blocked-motion recovery, and the exact first
non-moving cell.  A high score with all pills still visible would refute the
power-goal part specifically even if subgoal arbitration helps survival.
