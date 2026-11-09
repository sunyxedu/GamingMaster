# Candidate 0003 — stale-action safety interlock

## Parent and targeted hypothesis

Parent: `candidates/0000-tier2_jev_only/` (the current best candidate). I keep its
architecture: no Astra, one synchronous Jev `Choice` request at a time, and the
complete adaptor observation as Jev state. The single change is a code-side
safety interlock around that request.

**Hypothesis:** the parent is losing lives primarily because its previous joystick
edge is held while the next Jev answer is in flight, rather than because Jev
cannot recognize Q*bert's local state. A decision is about 0.16–0.20 seconds
later (two or three 66.7 ms ticks), so an edge that was safe when the request
started can be unsafe by the time its answer is posted. If the harness brakes an
already-unsafe previous edge before waiting and validates the returned choice
against a fresh observation, it should preserve more lives without changing the
model's route preference.

The strongest concrete trace evidence is in the parent’s low-score episodes:

* In `trace_seed4.jsonl`, the Jev state at tick 40 reports
  `qbert_node=15`, `safe_moves=['UP']`, while `last_action='DOWN'`. The answer
  is `UP`, but the old `DOWN` is held during the request; ticks 41 and 42 still
  post `DOWN`. The subsequent ticks continue `UP` from the now-falling edge,
  and at tick 48 the observation reports `lives=3`, `qbert_motion.phase='falling'`
  and `falling=true`. The episode eventually reaches only 175 points.
* The same trace shows the first model call after arriving at the bottom-left
  edge choosing `UP` with high confidence, so simply asking Jev for more
  confidence or changing option order would not fix the delay. The dangerous
  action was the *previous* `DOWN` during the call.
* In `trace_seed5.jsonl`, tick 38/40 states likewise show the bottom-left
  `safe_moves=['UP']` while `last_action='DOWN'`; the episode later loses a life
  at tick 122 while still in the bottom-edge recovery pattern and finishes at
  150 points. Across the parent, all six episodes lose four lives, complete no
  round, and the lowest two scores are 175 and 150.

This is deliberately narrower than adding a route planner, cube bookkeeping, or
Astra. It tests whether stale joystick state is the first bottleneck.

## What changed

For each loop:

1. Read the current JSON observation and its adaptor-computed `safe_moves`.
2. If `last_action` is not one of those code-certified directions, immediately
   post the first certified safe direction (or `NOOP` if none exists) **before**
   making the synchronous Jev call. This prevents the old action from being
   repeated for the request's latency.
3. Ask the same kind of Jev question as the parent, but expose only the current
   safe directional candidates. `FIRE` and unlisted edges are never model
   options.
4. After Jev returns, read a fresh observation. Accept its value only if it is
   still in the fresh `safe_moves`; otherwise keep a currently safe previous
   action, choose the adaptor's first safe edge, or use `NOOP` if no safe edge is
   certified.

The harness never parses prose, trusts an arbitrary answer, presses `FIRE`, or
invents an action when the decoder reports no safe edge. Timeout/runtime failures
keep the guard action for a tick and retry. The Jev timeout is 2.0 s: above the
parent's observed 0.345 s maximum and the live trial's 0.485 s maximum, while
still bounding a pathological wait. This is a reliability setting, not a second
gameplay change; the guard action remains posted during the call. Budget and
backend exceptions retain the normal contract behavior.

## System-1 interface

### State schema

The state is the complete strict-JSON object returned by `game.observe()`, as in
the parent. Thus Jev receives the frame/tick, lives and score, level/round,
21-element cube phases and needed landings, Q*bert node and motion, hazards,
`legal_moves`, `safe_moves`, last action/event, and terminal fields. No raw
pixels or new inferred numeric calculation is added. Code extracts only
`safe_moves` and `last_action` for the interlock; Jev sees the full state so the
architecture and recognition task remain comparable to the parent.

### Question

There is one batched question, `act`:

* Type: `Choice`.
* Candidate keys: the current observation's `safe_moves` filtered to the four
  directional names. If that set is empty, the only key is `NOOP`.
* Instructions: the observation's safe set is code-verified; choose exactly one
  listed useful safe hop, never `FIRE` or an unlisted edge.
* Criteria: the normal action descriptions from `game.action_spec`, without
  adding unsupported moves.

There is no confidence interpolation and no score arithmetic in Jev. A Jev
answer is accepted only by exact string membership in the *new* safe set. The
hard safety threshold is therefore `answer.value in fresh.safe_moves`; a
low-confidence answer cannot bypass it, while a high-confidence answer is also
rejected if it became stale. I intentionally do not impose a second confidence
cutoff: the parent’s fatal bottom-edge answer was already high-confidence and
confidence is not a substitute for freshness.

## Fallback and freshness policy

* Before a request, if the current `last_action` is still safe, leave it alone;
  this avoids needless reversals in the middle of a hop. If it is unsafe, post
  a safe edge immediately. If no edge is certified, post `NOOP`.
* A Jev answer is fresh only until the next `game.observe()` after that call.
  The answer is posted only if it remains in the new safe set.
* If an answer is missing, malformed, timed out, or no longer safe, retain a
  currently safe `last_action`, otherwise use the adaptor's first safe edge,
  otherwise `NOOP`. `FIRE` is never a fallback.
* If the game ends or the context closes while a request is running, return or
  propagate only the contract’s normal budget/backend termination exceptions;
  no post-game action is attempted.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a material reduction in life losses and edge/falling
observations, especially in seeds 4 and 5, with similar cube points per safe
landing and a lower stale-tick fraction or comparable score at the same Jev
budget. The useful signature is a guard replacement at states where
`last_action` is absent from `safe_moves`, followed by survival past the parent’s
first bottom-left failure.

Refutation would be unchanged bottom-edge life-loss timing despite the guard,
or a large score decrease while lives/falling events are unchanged. That would
suggest the dominant problem is route selection/coverage, not stale execution,
and the next iteration should alter Jev state/questions or add a planner rather
than another timing guard. A decrease caused merely by excessive neutral waits
would also falsify the proposed minimal timing fix.
