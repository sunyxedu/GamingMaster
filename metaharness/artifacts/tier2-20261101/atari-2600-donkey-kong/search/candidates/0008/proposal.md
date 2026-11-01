# Candidate 0008: closed-loop route phases instead of nearest-ladder reaction

## Parent and hypothesis

Parent: `candidates/0004/`, with `0003` used as a negative control for the
fixed-route idea.

The best parent did not fail because its asynchronous actor was late.  Its
first active trace (`trace_seed0`) posts `fire` at tick 1, then `up` at ticks
2--238.  The first active observation has Mario at x=45 and the bottom ladder
at x=108, while the parent selects the above-and-near ladder at approximately
x=50/y=148.  Thus it spends the whole opening holding `UP` away from the only
bottom ladder.  When a barrel finally approaches on the same bottom walkway
(the parent sees it at x=117, 109, 101, ... y=189), the hard guard starts
`leftfire` only around tick 238, when it is already near x=77.  It then ends
in `transition` and never posts the restart FIRE; the final state at the
900-second deadline still has score 0, screens 0, and one spare life.

`0003` tried a fixed ladder table, but it mixed that table with synchronous
Jev calls, repeated FIRE during startup, long jump holds, and a raw-RAM jump
index that was not the documented `$85` jump byte.  Its traces contain long
`leftfire` runs and still no progress.  This leaves a distinct testable
hypothesis:

> Donkey Kong needs a feedback controller with explicit route phases and
> short, committed action primitives.  A ladder must be approached from the
> current walkway and accepted only after UP produces measured vertical
> progress; a jump must be started early and held for one calibrated jump, not
> chosen afresh from a stale observation.  Correcting route phase and restart
> state should matter more than adding model calls.

## What changed

* The ramp controller uses the verified six height levels and the nominal
  bottom-right/alternating route (`108`, then `48/80`, `108/88`, `48/68`,
  `108`, `32/76`).  It walks to the current level's target before pressing
  UP.  It increments a phase only after a climb attempt has moved Mario at
  least 12 pixels upward.  If UP makes no progress for a bounded number of
  quanta, it nudges diagonally and then tries the other ladder at that same
  level; it never selects an arbitrary ladder merely because it is above.
* Hazard handling is a separate hard interrupt.  It classifies only visible
  same-row obstacles, uses named closing-distance buckets, starts a jump at
  `soon` as well as `immediate`, and commits to one direction for nine
  four-frame quanta (approximately the documented 31-frame jump).  It prefers
  jumping away from the obstacle except at an edge.  The same primitive is
  used to cross a rivet gap.
* The rivet screen is a state-derived serpentine: the lowest incomplete row is
  selected from the raw row bytes, the first side alternates by row, and a
  row-byte change arms a gap jump.  It approaches an unremoved rivet from
  outside the row, sweeps across it, then uses the observed byte change rather
  than guessed coordinates to decide that the rivet was removed.  It climbs
  from the finishing side of each completed row.
* Transition handling is explicit.  The actor releases input during the
  transition and posts a sparse FIRE attempt after 18 quanta, retrying at a
  long interval while a spare Mario remains.  Idle receives one controlled
  start edge, and stage/life changes reset all route and jump state.
* Astra is asynchronous and low-authority: at stage/life/screen changes it
  may select only validated `left|right` rivet-first and conservative/balanced
  fields.  Invalid or arbitrary text is ignored.  Jev receives a compact
  derived state and can recognize `hold_route`, `jump_left`, `jump_right`, or
  `wait`; a result is consumed once and is allowed to suggest a jump only when
  it agrees with the code's current hazard guard.  Neither model can suppress
  a code-level imminent jump, walk through a known gap, or turn a late answer
  into a persistent hold.

## Evidence that would confirm or refute it

Confirmation would be a trace in which Mario first moves from x≈45 toward the
bottom-right ladder, advances at least one ramp level, or earns any rivet/
jump reward, and in which a life-loss transition is followed by a new active
stage rather than 13,000 noop ticks.  The strongest confirmation is a positive
`ale_reward` or `screens_completed` with low stale ticks, even if the run does
not yet finish a screen.

Refutation is a run with the correct observed bottom-ladder approach but no
vertical progress across both nominal and bounded alternate targets, or a
run that reaches the rivet screen yet cannot change any rivet byte despite
safe row traversal.  Repeated deaths while the actor is visibly aligned and
using early nine-quantum jumps would refute the jump-duration/threshold part;
reaching transition but never restarting would refute the transition timing.

## Jev interface and control contract

The state sent on every request is strict JSON and contains only derived facts:

* `stage`; player horizontal/vertical bands, motion, and whether a jump is
  committed;
* route target kind (`ladder` or `rivet_or_gap`), target side/band, ramp level,
  rivet row/masks, `climb_started`, bounded no-progress, and `gap_pending`;
* one same-row hazard with named side, motion, `distance_bucket`, `closing`,
  and urgency (`clear`, `watch`, `soon`, `immediate`);
* progress buckets for lives, screens, rivets, and bonus, plus the last posted
  action.

Each batched request has two independent questions:

1. `tactic`: a `Choice` over `hold_route`, `jump_left`, `jump_right`, and
   `wait`.  It is a recognition/choice question, not a coordinate or math
   question.  The code-computed route remains the fallback.
2. `closing`: a `Noul` asking whether the normalized hazard is a same-row
   closing obstacle needing a jump soon.  It is logged for diagnostics and
   does not override the hard arithmetic guard.

Requests are asynchronous, at least eight quanta apart unless a stage or
hazard edge triggers one, with a 0.9-second timeout.  A response is usable only
once, in its original stage, immediately after collection.  A jump suggestion
must match the code-selected direction; otherwise the deterministic route is
used.  Low confidence is not converted to an action probability, and malformed
answers, timeout, or runtime failures simply leave the fallback active.

The fallback is executable every tick: FIRE in idle, sparse FIRE in a live
transition, phase route in ramp/rivet, and NOOP for terminal/dead states.  The
controller calls `ctx.wait_for_tick()` before each new observation/action, so a
slow Astra request or Jev tail cannot stop the game clock.  Astra calls are
also spawned and their parsed fields are whitelisted; arbitrary stand-in model
text cannot crash pre-flight.
