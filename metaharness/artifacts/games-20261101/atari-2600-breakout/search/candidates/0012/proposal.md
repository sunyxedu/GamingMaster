# Candidate 0012 — receding-horizon capture-set control

## Parent and hypothesis

Parent: `candidates/0006/`.  I also used the best evaluated implementation,
`candidates/0005`, as the performance comparison.  The parent and its
successors already established that the real-time loop must be code-owned;
this candidate does not return to synchronous model steering.

**Hypothesis:** the remaining losses are caused by the actuator/controller
interface, not by a missing Jev label.  The prior controllers mostly reduce a
predicted landing to one target point and choose a fresh directional pulse
from that point.  With sticky actions and paddle drift, the same target can
be reached by several short pulse sequences, and a point servo repeatedly
reverses while the paddle is still moving.  A controller that (1) computes a
paddle *capture set* for an uncertain landing and (2) searches short
LEFT/NOOP/RIGHT sequences through an online measured paddle-response model
should reduce misses without waiting for inference.

This is a structural test: action selection changes from a per-frame
point-target servo to a receding-horizon, sequence-level reachability search.
It is not another threshold or Jev-prompt tweak.

## Evidence from the archive

All of the useful candidates have essentially fresh actions, so latency is no
longer a sufficient explanation for the misses:

- `0005/trace_seed0.jsonl` loses lives near frames 2492, 3809, 7580, and
  9262.  The action stream is fresh but repeatedly changes direction in
  short blocks, for example RIGHT at 2477--2480, NOOP at 2481--2482, LEFT at
  2483--2485, then RIGHT at 2488--2491 before the first loss.
- `0007/trace_seed3.jsonl` has the same collision-window pattern at its first
  loss: RIGHT at 3312--3314, then LEFT at 3315--3319.  Its mode answer was
  `TRACK`, so this is not an ambiguous Jev decision.
- `0011/trace_seed4.jsonl` still shows RIGHT 956--959, NOOP 960--961, LEFT
  962--964 immediately before the loss at 965, and LEFT/RIGHT alternation at
  2845--2847.  Its stale fraction was only 0.000315.
- `0010` tried a latched landing interval, but its phase gate produced long
  NOOP runs (for example frames 323--339 and 713--729 in the recorded
  traces) and scored only 5.5.  Thus simply freezing a plan or widening an
  interval is not enough; the controller must actively plan the paddle's
  future motion and continue steering toward a reachable corridor.

The parent’s smoothed velocity idea is useful plumbing, but it still chooses
one immediate action from a point target.  The new test keeps the measured
velocity and event invalidation while replacing that bridge with short action
sequence search.

## What changed

`harness.py` implements:

1. **Capture-set geometry.**  It projects an ensemble of reflected ball
   trajectories using the public RAM ball position and `dx/dy`, with small
   position, velocity, and contact-time perturbations.  The result is an
   interval of legal paddle-left coordinates that gives interior horizontal
   overlap, rather than a single landing coordinate.  The uncertain brick
   band uses a broad central corridor, but the paddle is actively driven into
   that corridor instead of being left on NOOP.
2. **Collision/event invalidation.**  Life loss, serve, wall changes, brick
   hits, horizontal bounces, and an observed descending-to-ascending change
   near the paddle invalidate the old capture plan.  A missing ball is
   treated as serve/game state, not an episode reset.
3. **Measured actuator model.**  Consecutive paddle rectangles update
   direction-specific movement gains, current velocity, and NOOP persistence.
   The local model explicitly allows velocity to persist through NOOP and
   models an opposite pulse as braking before reversal.
4. **Receding-horizon search.**  For each current state, code enumerates at
   most `3**5` short sequences over `LEFT`, `NOOP`, and `RIGHT`.  It simulates
   both a nominal path and a sticky-action path, scores worst-case distance
   from the capture interval, penalizes residual velocity and direction
   reversals, and posts only the first action.  The search runs again after
   the next native frame, so a planned pulse is never a stale mailbox.  A
   sequence can reverse when the capture set is clearly unreachable, but a
   near tie prefers the previous command rather than chattering.
5. **Model authority is reduced further.**  Astra returns only
   `SURVIVE`, `SHAPE_LEFT`, or `SHAPE_RIGHT`; Jev validates a short-lived
   version of that preference.  A shaping preference selects an edge *inside*
   the code-computed safe capture set only when the return is early,
   normal-speed, settled, and there are at least three lives.  Neither model
   can choose a coordinate, pulse, FIRE timing, or frame-level action.
6. **Defensive real-time plumbing.**  The loop posts an action before
   launching new inference, waits on `ctx.wait_for_tick()`, retries FIRE while
   `serve_required` is true, and treats malformed/late model output as
   advisory failure.  All model work is asynchronous and each background
   task is one bounded context call.

## Jev interface

### State schema

Every request carries a strict JSON object containing:

- `game`: fixed ordinary mode-0/difficulty-0 and native-60-Hz description;
- `observation`: frame, event (`serve`, `brick_hit`, `paddle_hit`, etc.), lives
  and `low_lives`, score, flight phase, landing urgency, landing lane,
  return-uncertainty bucket, speed regime, measured paddle motion,
  `shape_is_code_eligible`, left/right outer-edge status, bricks remaining,
  wall index, and walls cleared;
- `strategy_proposal`: the latest Astra preference; and
- `rules`: the three legal preferences and the fact that code owns geometry,
  arithmetic, and action execution.

The state does not ask Jev to count bricks, predict pixels, compare numeric
coordinates, or invent actions.  All values sent to it are named buckets
computed by code.

### Questions and bridge

One request batches two independent `Choice` questions over
`SURVIVE`, `SHAPE_LEFT`, and `SHAPE_RIGHT`.  The option order is rotated for
the first question and reversed for the second, reducing the documented
first-option bias.  The instructions explicitly require `SURVIVE` for low
lives, near/imminent contact, fast or unknown speed, wide/unknown return
uncertainty, an open requested edge, or `shape_is_code_eligible == false`.

A response is usable only when both answers agree, the selected option is in
the allow-list, and the weaker selected-option probability is at least
`0.52`.  It must also be no more than 90 native frames old and match the
capture epoch and strategic plan version.  The newest observation then
rechecks all safety preconditions; a failed recheck converts shaping to
`SURVIVE`.  The accepted preference expires on the next event invalidation or
90 frames.  Jev is requested only outside serve state and no closer than the
current scheduling interval; the current action never waits for it.

Astra's one-token strategic answer is parsed defensively and fed back through
this Jev gate.  A missing, timed-out, malformed, or stale answer leaves the
default `SURVIVE` strategy and the deterministic capture controller intact.

## Fallback and thresholds

- `serve_required` always posts `FIRE`; an absent ball without serve permission
  posts `NOOP`.
- Ascending, unknown, or still-in-brick-band flight uses the central
  paddle-left corridor `[58, 94]`.  The search drives toward it and accounts
  for measured drift; it does not assume that NOOP brakes instantly.
- A clean descent uses the reflected capture interval.  The action search has
  a horizon of one to five native frames (exact contact when close, progressive
  movement when far), includes a nominal and sticky branch, and gives
  catchability higher weight than shaping or pulse count.
- At near/imminent contact, code ignores a strategic edge preference and uses
  the full safe interval.  If an inference call is late, raises, disagrees,
  or returns garbage, this same search remains the fallback.
- Invalid action strings are normalized to `NOOP` before `ctx.act`; all model
  answer parsing is inside exception-safe advisory paths.

## Confirmation or refutation

The hypothesis is supported if, relative to `0005` (288 mean) and `0006`
(200.8 mean), the traces show fewer short direction reversals near observed
paddle contacts, longer survival, and higher score without increasing stale
ticks or episode errors.  Useful controller evidence is convergence of the
measured gains and a sequence whose simulated sticky endpoint remains inside
the capture set.

It is refuted if the sequence model systematically understeers or oversteers
compared with observed paddle rectangles, if misses persist with the same
collision-window action pattern, or if the central uncertainty corridor
preserves lives but prevents brick progress.  In the latter case a later
iteration should retain the sequence-level actuator model and investigate
actual paddle-section bounce outcomes/tunnel shaping, rather than adding more
model prose.
