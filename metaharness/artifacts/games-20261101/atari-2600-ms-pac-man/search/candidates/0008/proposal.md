# Candidate 0008 — power-pill checkpoints and life-specific escape routes

## Parent and diagnosis

Parent: `candidates/0004/`, whose graph executor raised the score from the idle/Jev
baselines to 778.3 mean with zero stale ticks. I retain its useful continuous
corridor execution, but test a different failure hypothesis. The strongest
counter-evidence is not model latency: every 0004 episode ended with three deaths,
zero mazes cleared, and **zero power pills eaten**. In seed 0 the controller had
already scored 240 by tick 146, then died at tick 236; after the respawn it repeated
the same dot sweep and died again at ticks 406 and 469. The final observation still
showed all four power-pill objects at `[148,146]`, `[148,14]`, `[8,147]`, and `[8,15]`.
The same all-pill-preserved pattern appears in the other five final observations.

The parent labels some branches `is_power_escape`, but power pills are only a
conditional emergency target. Its ordinary nearest-dot BFS therefore repeatedly
walks through exposed lanes, and a last-second branch can be too late or oscillate
between equally dangerous options. The resulting score is a good first-maze dot
sweep, not survival control.

## Hypothesis

The missing structural primitive is a **consumed escape resource**, not another
local direction judgment. If the controller deliberately visits a power pill before
risk accumulates, then treats its disappearance as a verified frightened interval,
it can convert the first maze's exposed dot sweep into a route with a resettable
safety buffer. If a death occurs, changing the route phase and forcing a different
power-pill target should prevent repeating the parent's same lethal loop.

This is intentionally stronger than merely lowering a hazard threshold: it tests
whether explicit power-pill checkpoints plus life-specific route rotation improve
lives/maze progress, even if the first checkpoint sacrifices a few nearby dots.

## What changed

The harness keeps 0004's observation-derived corridor graph, continuous action
posting, and asynchronous Jev arbitration, but changes the route state:

* power-pill coordinates are retained as graph goals and connected to the nearest
  corridor nodes using the same observed coordinate lattice; they are no longer
  only a fallback considered after a collision warning;
* during the opening/death-reset calibration, each life has a checkpoint phase;
  the first life heads for a safe reachable power pill once the opening is moving,
  the second chooses the opposite-side checkpoint, and the last life chooses the
  nearest remaining checkpoint. A checkpoint is abandoned only when its object
  disappears, the maze/death changes, or the route is measurably unreachable;
* disappearance of a power-pill object near the player is detected in code. The
  controller then enters a conservative, time-bounded `pill_active` phase and
  returns to dot clearing rather than trying to infer frightened colour from RGB.
  It never asks Jev to identify frightened state;
* an immediate ghost threat now interrupts the current route and selects an
  escape branch by projected graph distance to a power-pill goal. A normal route
  is not resumed until the target is reached or invalidated. On a death, the
  route target and tie-break phase rotate, so the next life does not replay the
  same first route;
* the graph is built from both ordinary pills and normalized power-pill goals,
  while persistent nodes are retained after collection. The harness uses the
  adaptor's supplied ghost velocity fields and ignores the opening `(0,0,0,0)`
  sentinel rather than treating it as an active nearby ghost;
* Jev remains a local tie-breaker only at a checkpoint approach, junction, or
  immediate hazard. It cannot veto an explicitly required power-pill checkpoint
  when a legal route exists. All actions are posted every tick while calls run.
  Astra still writes at most one slow maze-level risk mode in the background and
  cannot block execution.

## Jev interface and bridge

Each request is strict JSON and contains only derived facts:

```text
observation: {frame, tick, score, lives, deaths, player_xy, direction,
              visible_pills, visible_power_pills, pill_active, maze_clears}
plan: {mode, goal_kind, goal_xy, fallback_action, legal_actions,
       graph_degree, route_version, life_phase}
threats: {level, nearest_bucket, closing_lane_count, immediate}
options: {cardinal: {continues_goal_route, corridor_risk,
                     reaches_checkpoint, branch_degree}}
```

`Choice` questions ask which of the already computed legal cardinal branches to
use. Two independent copies use reversed option order to reduce Jev's documented
first-option bias. A `Noul` asks whether the supplied checkpoint escape is safer
than the normal fallback; its answer can select only the code-computed checkpoint
branch. No question asks Jev to count, calculate a path, read RAM, or infer ghost
colour.

A result is accepted only if both Choices agree, the answer has a probability of
at least 0.40 in ordinary route selection or 0.55 during an immediate hazard, the
request is at most 18 native frames old, the death/maze/goal version is unchanged,
and the selected action is still legal. An escape Noul needs probability at least
0.60 and is advisory; code's immediate threat guard remains authoritative.
Malformed stand-in answers, missing probabilities, disagreement, timeout, and
stale futures are ignored. The deterministic route/guard action is always used.

## Astra, fallback, and scheduling

Astra receives the compact graph/checkpoint summary once at opening and can choose
`MODE=survival`, `MODE=balanced`, or `MODE=score`; explicit last-life survival in
code overrides it. The mode affects the existing ghost-guard tolerance, while the
checkpoint rule remains code-owned. Astra never posts a joystick action. Its result
is parsed only for an explicit marker, so arbitrary pre-flight text is harmless.

At every loop iteration the harness computes a legal cardinal fallback, posts it,
and waits for the next tick. Jev and Astra run in the background. If the graph is
not yet usable, it holds the observed direction (then left); it never intentionally
waits on `noop`. Model exceptions other than the required backend-unavailable
termination are swallowed and execution continues. No background loop survives
`ctx.done`.

## Evidence that would confirm or refute the hypothesis

Confirmation is more than a higher raw score: traces should show a power-pill
object disappearing while the player reaches its checkpoint, later deaths reduced
below three or a maze clear, and life resets selecting different checkpoint goals.
A score above 778 with comparable zero stale ticks would be strong confirmation.

Refutation is all six episodes preserving every power pill and dying at the same
locations as 0004, or reaching a checkpoint but still dying before any additional
maze progress. If explicit checkpoint routing lowers score but leaves lives and
maze clears unchanged, the hypothesis that pills are a sufficient safety resource
is weakened; the next revision should inspect legal-goal normalization or the
collision projection rather than add more Jev calls.
