# Candidate 0016 — route-goal council, not delayed joystick arbitration

## Parent and evidence
Parent: **0009**, rewritten using its observation-derived dot graph and uninterrupted
cardinal executor. Best comparator: **0012** (the executable policy equals 0010).
Read the supplied lessons, both modules and selected JSONL traces. This is a new
control architecture, not a reserve-count ablation or claimed docstring-only change.

In **0009 seed 2**, the last six Jev request states (lines 1312–1343) keep the player
at `[97,135]`, score 1520, 18 dots. The fallback is `left` while Jev's accepted events
repeatedly select `right` (frames 3372–3424); both options are labelled merely
`danger`. Terminal frame 3433 has a ghost at `[99,135]`. In **0012 seed 4**,
requests at frames 3428–3480 show `[53,60]`, a code fallback `up`, and repeated
accepted `down` advice; terminal frame 3489 has the ghost at `[53,58]`, score 1410,
29 dots. These are evidence of conflict and an impoverished local interface, NOT
proof that the stationary periods caused death: death animation can freeze position.

**0012 seed 2** demonstrates another independent necessary correction: observation
line 2013, frame 5416, repopulates from zero to 153 collectibles with strawberry,
but `mazes_cleared=0` and `map_id=cx2675_pill_grid_1` remain unchanged. Its consumed
power records therefore survive into the next level. It ultimately scores 3990 and
dies with one dot left. No seed-specific route, timing or action sequence is encoded.

## Hypothesis
The continuous executor already solves latency. Giving a delayed classifier the
last word over a single joystick direction is the wrong division of labour when
every nearby candidate is described as danger. Instead, **code compares executable
short routes for interception, Jev selects a bounded semantic route goal, and Astra
audits which route-selection rubric to use after actual evidence**. A goal can
survive several moving observations; a dangerous action cannot. This should reduce
contradictory overrides and improve repeatable completion without waiting for models.

## Structural changes
- Retain dot-coordinate graph construction, power-grid snapping, cardinal pre-turn
  execution and the first two coverage checkpoints. Edges remain inferred, never
  advertised as verified walls. Ignore initialization ghost displacements.
- Compute weighted graph distances and short-horizon ghost interception profiles
  for continued route, weighted dot route, reachable power route and branch-to-next-
  junction alternatives. Physically misaligned perpendicular actions are excluded
  from immediate arbitration. Code remains the final safety authority every tick.
- Replace `move_a/move_b/escape` direction overrides with route-goal leases, rebuilt
  from the current position and revalidated against current hazards before adoption.
  Risk dominates reward; within comparable risks, continuity and actual dot progress
  dominate empty retreat. Bounded reversal holds reduce command oscillation.
- Detect sustained dot repopulation independently of adaptor maze counters; renew
  graph/resource/goal state and invalidate previous-epoch answers. Incidental power
  consumption is logged too. Recent consumption is uncertain protection, never an
  asserted frightened timer or a ghost-chasing license.
- Delay Astra until observed gameplay exists, then ask for a one-line bounded rubric
  for the Jev interface. Revisit after life loss, maze renewal or supported route-trap
  evidence, with cooldown and at most three calls. No slow immediate-action commands.

## Jev schema and bridge
`rules` gives objective, uncertain pill protection, route-choice semantics and risk
labels. `plan` gives rubric, resource epoch, cleanup/sweep stage, last-life status
and goal kind. `options` contains only non-dominated, currently executable routes,
with purpose, risk label, exit-branch boolean, semantic dot yield, reversal flag,
resource, continuity and inferred-geometry warning. `recent` gives score progress,
power event and recent guard intervention. Raw RAM and arithmetic are excluded.

One asynchronous request batches two reordered `Choice`s (`route_a`, `route_b`)
and a `Noul` (`renew`) about stalled progress with repeated interventions. Both
Choices must agree with selected-option probabilities at least **0.70**. Missing
probabilities are uncertainty. Noul >= **0.80**, plus code-confirmed lack of score
progress, can request an Astra interface audit. Every answer is tagged by frame,
maze epoch, life and rubric version. At most **48 native frames** old, and the goal
must still exist and have a freshly rebuilt risk no worse than the best current
route class. A lease lasts at most **120 native frames**, continuously revalidated;
there is no cached joystick override. Unknown, disagreement, malformed values,
transient failures or stale answers leave the executable code policy in control.

Jev requests are event/interval gated (>=24 native frames apart), one in flight,
with **2.5s timeout** versus archived ~0.2s normal latency and >1.2s failures. Astra
uses **100s timeout**, one in flight, a short answer, and never blocks actions.
Late model results are not assumed fresh simply because they succeeded. No private
model clients or background Python loops; all inference uses ctx and all control
waits on ctx.wait_for_tick. Budget and episode termination remain normal exits.

## Confirmation and refutation
Confirm with error-free/near-zero-stale runs, fewer contradictory turn episodes,
logged `jev_route_applied` followed by route execution, higher mean score than 0009
and more independently detected dot refills. On progression, require `maze_epoch`
then new `power_window` events; do not rely on the broken reported maze counter.
Astra benefit requires `astra_interface` followed by meaningful different route
choices; completed calls alone do not establish benefit.

Refute if conservative interception keeps circling empty corridors, coverage falls
while survival does not improve, or Jev route choices never survive revalidation.
The graph and constant-speed horizon are approximations and may overpredict threat.
The experiment intentionally does not claim learned ghost personalities, guaranteed
frightened duration, corrected adaptor implementation or seed-specific solutions.

## Validation
Implementation written before refinement. Syntax, arbitrary-model-output contract
checks and archived-observation replay will be checked without running long games
or contacting model backends. Results are added after those checks.
