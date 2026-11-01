# Candidate 0010 — phase-gated committed landing intervals

## Parent and hypothesis

Parent: `candidates/0008/`; comparison baseline: the best candidate,
`candidates/0005/`.

The best controller is already fast and code-owned.  The remaining failure is
not stale ticks or a missing strategic idea.  In the worst `0008` trace
(seed 1), the command stream around the first loss repeatedly reverses at the
impact window (`LEFT`/`RIGHT` at frames 1053--1062, then alternating again at
1065--1070).  The same pattern is visible around later losses and is called out
in the generation-2 lesson for `0007`.  `0008`'s extra measured-velocity servo
and contact mode did not help: it scored 234 versus `0005`'s 288, and all
six Astra calls timed out.  The `0005` trace shows that ordinary reflected
interception is useful, but it still changes a point target every frame.

**Hypothesis:** a point-target servo is the wrong abstraction for this ROM.
During the last part of a return, the ball's horizontal landing is already
catchable over a range of paddle positions, while one-frame RAM velocity and
sticky actions make the exact point move slightly.  Recomputing a point and
reversing immediately can turn a safe overlap into an edge miss.  A controller
that (1) waits until the descending ball has cleared the brick band, (2) makes
one reflected prediction, (3) aims for a paddle-overlap interval, and (4) latches
a direction for a short commitment with an explicit neutral/brake phase should
lose fewer lives.  This is a structural change rather than another margin
constant.

## What changed

- The wall is a phase boundary.  While the ball is ascending or still inside
  the bottom-brick collision band, code parks the paddle in a wide center
  interval.  It does not trust a trajectory that can still be changed by a
  brick.
- Once a descending ball is below the brick band, code builds an ensemble of
  reflected landing estimates.  The action target is the legal paddle-left
  interval that overlaps the landing ball with an interior margin, not the
  landing point itself.
- The landing interval is latched through the return.  It is invalidated by a
  serve, life loss, paddle hit, wall transition, brick hit, or a large observed
  inconsistency.  Small frame-to-frame prediction changes do not trigger a
  new direction.
- The paddle actuator has a two-frame directional latch, a one-frame neutral
  phase, and at most one counter-steer when measured momentum would carry the
  paddle away from the catch interval.  Position deltas, and not an assumed
  NOOP brake, decide whether the brake worked.
- `last_action`/`requested_action` mismatches are counted as sticky-action
  evidence and widen the code-owned landing ensemble.  Malformed or missing
  model output cannot affect legality or the catch.
- There is no Astra call in this candidate.  The archived Astra calls were
  late (all six `0008` calls timed out) and supplied no control evidence; a
  delayed tunnel plan is not useful while the phase-gated code controller is
  already executing.  Removing that stale dependency is deliberate and also
  leaves the System-2 tie-breaker favorable.  Exact serving, geometry,
  bookkeeping, and fallback remain in code as required by the study.

## Jev interface and bridge

Jev receives a small strict-JSON state on an asynchronous mailbox request.  It
contains:

- `observation.frame`, `event`, `lives`, `lives_low`, `serve_required`;
- named `flight_phase` (`ascending`, `collision_zone`, `descending`, or
  `unknown`) and bucketed landing urgency;
- whether the code plan is `latched`, whether the current paddle is in the
  computed overlap interval, `paddle_motion`, and `speed_regime`; and
- wall/brick progress as side context, never as arithmetic Jev must perform.

One request batches two independent `Choice` questions, `mode_a` and `mode_b`.
They use the same three options in different orders:

- `TRACK_RETURN`: the ball is a stable descending return and the code plan is
  latched;
- `PARK_HOME`: the ball is ascending or still in the brick collision band;
- `NO_DECISION`: ambiguity, imminent contact, low lives, or contradictory facts.

The instructions explicitly tell Jev not to calculate coordinates or invent an
action.  An answer is accepted only if both choices agree, both probabilities
support that choice at **0.55 or higher**, the response is no more than 75
frames old, and the landing-plan version is unchanged.  Jev's accepted value
is logged as an advisory phase judgment; the deterministic controller still
owns every frame and ignores it near an impact.  `PARK_HOME` can only intersect
a computed catch interval when one life remains and the return is not near;
otherwise Jev cannot change a latched target, reverse a paddle, serve, or
bypass the overlap interval.  The request is launched about every 90 frames
(or sooner after a
major event) with a 1.25-second timeout, so a late/failed answer simply remains
in the mailbox and the code fallback continues.

Arbitrary pre-flight answers, missing probability fields, timeouts, and other
model exceptions are caught.  The answer parser checks the value against the
closed option set before using it.

## Astra handoff and revision trigger

This iteration deliberately has no Astra handoff: the verified exact rules and
all strategic output that arrived in the archive are slower than the remaining
reflex problem, and the previous Astra requests timed out without changing an
action.  A later iteration should reintroduce one short Astra call only if
traces show stable survival with a clear first-wall bottleneck and a measurable
need for a tunnel policy.  It should then author a mode proposal for Jev, never
post a frame action, and be invalidated at a life or wall transition.  Until
that evidence exists, going back to Astra would test latency plumbing rather
than the stated collision hypothesis.

## Fallback and evaluation test

The immediate fallback is always legal: FIRE while `serve_required` is true,
NOOP on termination or an absent ball, and otherwise the interval controller's
latched RIGHT/LEFT/NOOP action.  `ctx.wait_for_tick()` is used on every real-time
iteration, and no inference call blocks the emulator loop.

Evidence supporting the hypothesis would be fewer life-loss transitions at
similar frame counts, fewer consecutive one-frame direction reversals near a
loss, and a higher score/brick value on the same seeds without stale ticks.
It would be refuted if losses occur while the paddle is stably inside the
computed overlap interval, or if waiting below the brick band causes a clear
survival regression before the first return.  The trace events
`landing_plan_created`, `landing_plan_invalidated`, and the action stream make
those cases distinguishable.
