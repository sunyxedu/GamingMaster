# Candidate 0005 — released-edge control with an asynchronous supervisor

## Parents and diagnosis

This is a crossover of `0000-tier2_jev_only` and `0002-tier2_idle`.  The Jev
parent earns the only meaningful score (mean 6.0), while the idle parent is the
robust zero-call reference: six clean episodes and no model or harness errors.
The failure is visible in the seed-103 trace, not just in the aggregate: Jev
answers `UP` repeatedly (for example, the first answers have `UP` around 0.51
and `NOOP` around 0.37), but the posted action then remains `UP` for roughly
nine more emulator ticks while the next 0.15–0.20 s request is in flight.
The action log shows `UP` on ticks 13–21, 22–32, and so on.  Seed 103 receives
points 1 through 5, then ends at score 5 with four deaths and zero homes.  All
six Jev episodes have four deaths and zero homes; the best score is therefore
mostly forward-hop credit, not successful Frogger play.  The idle parent instead
shows that simply waiting is a safe, error-free fallback, although it scores 0.
There is no `lessons.md` in the archive to contradict this diagnosis.

## Hypothesis

The flat Jev score is primarily an action-bridge error: a real-time directional
judgment is being treated as a continuously held command.  Releasing every
accepted direction after the next tick should turn Jev's useful preference for
forward progress into discrete hops while preserving the idle parent's safe
behavior whenever a model is late, malformed, unavailable, or uncertain.  The
second hypothesis is that a short Astra policy can improve *which* discrete hop
is chosen, but must never block the real-time loop.

## What changed

* Each Jev decision is converted into an edge: `ctx.act(direction)`, wait for
  exactly one game tick, then immediately post `NOOP`.  During the next Jev
  request, the game therefore waits rather than repeatedly applying the old
  direction.  A Jev timeout/error falls back to waiting (or one `UP` on an
  explicitly detected fresh/dead frog), so the harness remains clean like the
  idle parent.
* A rotating, typed Jev action question chooses among literal actions.  Its
  state is a compact, strict JSON summary: frog row/x/state candidates, phase
  hint, score/home/death/reserve/timer/level bookkeeping, selected RAM fields,
  recent outcomes, event since the last summary, the current supervisor advice,
  and the explicit one-frame control contract.  Jev also answers an independent
  `wait_is_safe` Noul and a decoder-phase Choice.  A strong wait judgment vetoes
  only a low-probability directional answer; unknown/terminal phase blocks
  lateral invention but does not make ordinary forward progress impossible.
* Astra is a background supervisor, never a blocking action source.  It is asked
  for a bounded JSON policy (`goal`, `preferred_action`, `wait_rule`, `avoid`) at
  opening and, when possible, after the first death/reset and the first level
  change.  Its arbitrary or malformed text is reduced to a bounded advisory
  dictionary; it enters the Jev state rather than being executed directly.
  The Jev question explicitly checks the advice against current preconditions.
* The supervisor is polled only between decisions, and all futures/model
  failures are treated as advisory failures.  The loop always has a legal NOOP
  fallback and exits normally on budget/deadline/game completion.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a materially higher score or any nonzero `frogs_home` with
fewer deaths than 0000, especially if the trace shows directional actions
followed by one consumed tick and an immediate NOOP rather than long same-action
runs.  A Jev→act rate can remain high even though stale ticks fall only modestly;
the primary signal is progress beyond the repeated single-hop score.

The hypothesis is refuted if the released-edge harness still dies after roughly
one forward point per frog while the trace confirms one-frame directions, or if
it only reaches the idle score because Jev's state has no actionable information.
That result would point to missing object perception/RAM calibration rather than
held-input repetition; the next iteration should then use Astra to calibrate
RAM/object candidates or add a verified route search, not merely alter timing.

## Jev contract and fallback

The state sent to Jev is JSON with these fields:

* `frog`: conservative `row_candidate`, `x_candidate`, `screen_x_candidate`,
  `state_candidate`, and a non-authoritative zone hint;
* `score`, `frogs_home`, `deaths`, `reserve_frogs`, `timer_s`, `level`,
  `terminal`, `decoder_confidence`, `ram_focus`, and tick/frame counters;
* `event`, score/death deltas, up to four recent summaries, and
  `last_posted_action`/`last_observed_action`;
* `supervisor` plus `supervisor_version`, and a literal statement that the next
  direction is released after one frame.

The three batched questions are:

1. `action`: a Choice over `NOOP`, `UP`, `RIGHT`, `LEFT`, and `DOWN`, asking for
   the best *single next frame* and explicitly forbidding long-horizon planning;
2. `wait_is_safe`: a Noul asking whether one-frame waiting is safer than any
   directional hop, with `true` defined as a one-frame judgment rather
   than indefinite idling;
3. `current_phase`: a Choice over `START`, `ADVANCED`, `TERMINAL`, and `UNKNOWN`
   based only on named fields.

The action answer is accepted only if its value is one of the legal action names.
A strong wait probability (at least .78) vetoes a direction only when the
selected action probability is below .56.  An explicit terminal/unknown phase
also vetoes lateral actions.  These gates are conservative around lethal
sideways moves but do not reject a normal forward answer merely because Jev's
confidence is imperfect.  If the answer is missing, malformed, times out, or
System-1 is unavailable, code uses `UP` only for a clearly fresh/reset frog and
otherwise uses the validated Astra recommendation or `NOOP`; terminal state
always maps to `NOOP`.  Every posted direction is debounced by the edge bridge.
The Jev timeout is 0.65 s: it is comfortably above the archived 0.4 s maximum,
while the game continues on NOOP during the request.  Astra is requested with a
120 s timeout but in a background future and is limited to 100 output tokens, so
its long latency cannot hold a joystick or block a tick.

Astra is revisited only on meaningful events and is capped at three background
calls.  Its parsed fields are context for Jev, not a bypass of the typed-action
contract.  If the training traces show lower stale action runs but no home
progress, the next experiment should test the study's platform/traffic decoder
rather than increase Astra frequency.
