# Candidate 0014 — event contracts and an interval motor

## Parent and diagnosis

Parent: `candidates/0007/`, with `0005` as the best-scoring reference.  I kept
0005/0007's important property: code posts a legal action before waiting for a
model, and the model is never on the 60-Hz critical path.

The traces identify a lower-level failure rather than a stale-tick failure.
0007 seed 3 had only 2 stale ticks in 7155 frames and still lost at frames
3318, 4845, 5888, 6635, and 7155.  The action windows immediately before those
losses contain fresh three-frame reversals such as `RIGHT,RIGHT,RIGHT` then
`LEFT,LEFT,LEFT`, or repeated `LEFT/RIGHT` cycles.  Its contact calibration
also produced offsets of `+9`, `-9`, `+7`, `-10`, and other changing values;
there is no stable enough table for those offsets to be a safe actuator model.
The best 0005 trace shows the same symptom more plainly: its seed-4 loss at
frame 918 is preceded by a long NOOP run and then `RIGHT,LEFT,RIGHT`, while
later losses have repeated three-frame direction pulses.  Thus extra Jev
modes, and the failed slow Astra calls (0007 timed out on all six), are not a
solution by themselves.

## Hypothesis

The parent treats the projected paddle crossing as a point and recomputes a
velocity-aware direction on every frame.  Near a collision, a one-pixel
projection change or one sticky-action draw can therefore turn a safe overlap
into a fresh opposite command.  Conversely, a long NOOP run does not prove
that the paddle has stopped.  Breakout only requires *interval overlap*, so a
controller should plan a catch interval once after the ball is below the brick
band, lease a short motor pulse toward that interval, and use at most one
measured counter-pulse when motion would leave it.  Replan only on a real
velocity/flight event, not on every target fluctuation.

This is a structural rewrite of the division of labour as well as a low-level
test.  Code owns exact reflection, overlap geometry, event bookkeeping, and
motor timing.  Jev is no longer asked for a per-frame tactical mode; it judges
a long-lived rally contract only at serves, life/wall transitions, and paddle
bounces.  Astra proposes the same kind of contract once, asynchronously.  A
model contract can shift only the broad early/home anchor and is rejected for
low lives, fast/uncertain flight, or an open edge.  Neither model can replace
the catch interval or emit an action.

## Harness changes

* `_make_return_contract` computes the reflected ball landing and the complete
  paddle-left overlap interval, with a small interior margin.  It waits until
  the descending ball is below the lower brick band, so brick collisions do
  not masquerade as a paddle return.
* The interval is latched through a clean descent.  A new contract is allowed
  after a serve, life loss, wall transition, paddle bounce, brick/side event,
  or a measured velocity change.  The motor uses two-frame early/one-frame
  near-impact leases and a single brake pulse plus neutral cooldown.  It
  observes paddle displacement and the adaptor's effective action fields, but
  does not import the unstable contact-offset table from 0007.
* While the ball is ascending or in the collision zone, code parks in a wide
  home interval.  The optional strategic contract can move that home interval
  only with plentiful lives; the physical catch interval always wins.
* FIRE is emitted whenever `serve_required` is true and lives remain.  A null
  ball is not an episode reset, and the loop stops only on game/deadline or the
  normal contract exceptions.

## Jev state, questions, thresholds, and bridge

Each boundary request sends strict JSON with the following derived state:

```text
{
  frame, event, lives, score, serve_required,
  phase: serve|ascending|collision_zone|descending|uncertain_descent,
  urgency: serve|early|near|imminent,
  speed: normal|fast|unknown, landing_lane, catch_interval: wide|narrow,
  paddle_motion, eta, fallback_action,
  rally_contract, astra_plan, left_edge, right_edge,
  left_count, right_count, bricks_remaining, wall_index,
  contract_epoch, rally_version
}
```

One request batches `contract_a` and `contract_b`, the same `Choice` with
rotated and reversed option order.  Options are `PRESERVE`, `LEFT`, `RIGHT`,
and `NO_CHANGE`; they describe only an early/home preference.  The bridge
requires both answers to agree and the weaker selected probability to be at
least `0.55`.  It then rechecks the current lives and edge bitmap: low lives
(2 or fewer), `NO_CHANGE`, an open requested edge, or a changed rally version
becomes `PRESERVE`.  A late, malformed, timed-out, or unavailable answer is
ignored.  The fallback is always the interval motor; Jev never chooses FIRE
or a raw directional action.

Requests are event-driven with a 36-frame cooldown, not periodic reflex calls.
This tests whether meaningful boundary judgments are more useful than the
parent's mostly `TRACK`/`SAFE` labels that rarely changed control.  A fresh
answer can affect only the next safe/home phase; it cannot affect an
imminent/near return.

## Astra handoff and revision trigger

Astra receives a compact state containing the current derived facts, side-edge
buckets, ball and paddle rectangles, and the same usage restriction.  Its
short-token result is parsed defensively as `PRESERVE`, `LEFT`, or `RIGHT` and
is stored as an untrusted prior.  Jev must validate it at a boundary, and the
code gates it by lives, speed, edge openness, and phase.  If Astra times out,
returns prose without a token, or is cancelled, no state changes.

A future Astra revision is warranted if traces show that interval catches are
surviving but the first wall remains largely intact: then the strategic
contract should be expanded into a verified tunnel subgoal.  If losses still
cluster while the paddle is inside the logged catch interval, the hypothesis
is refuted and the next experiment should inspect collision timing/ROM
geometry rather than add model authority.  If the trace instead shows frequent
brake pulses and rapid lease changes, the motor response estimate or sticky
handling needs revision.

## Evidence that would confirm or refute it

Confirmation is fewer lives lost at comparable frames and a mean above 288
without increasing stale ticks or episode errors.  Supporting events would be
stable `return_contract` intervals, few lease reversals, and brake pulses only
when measured motion exits an interval.  Refutation is earlier loss than
0005/0007, many contracts whose intervals are physically catchable but still
missed, or survival with no score improvement; those outcomes would indicate
that the public RAM velocity/reflection model is insufficient and require a
small verified transition search instead.
