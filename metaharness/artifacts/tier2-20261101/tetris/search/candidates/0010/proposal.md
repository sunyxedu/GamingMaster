# Candidate 0010 — latency-separated, exact placement controller

## Parent and hypothesis

Parent: `candidates/0007/` (the successful one-preview controller).  I also used
`0008` as a negative control for search changes, but did not copy its executor.

The concrete remaining failure in 0007 is timing rather than survival.  Its worst
training episode (`trace_seed4`) still survived all 200 pieces and cleared 78 lines,
but 409 of 1,522 ticks were stale.  The trace shows the safe `noop` at tick 9 followed
by stale noops at ticks 10–11, and similar gaps at ticks 19–20 and 25–26 while the
next placement was being computed.  Replaying 0007's planner on archived observations
also makes the bottleneck visible: `_plans` takes roughly 90–230 ms for one state,
while a game tick is 50 ms.  It repeatedly enumerates the next piece for every
current candidate.  Thus even a correct target spends several clock ticks repeating
an old noop before it can post the first steering action.

The hypothesis is that we can lower stale ticks, without sacrificing 0007's strong
200-piece/78-line behavior, by separating *planning latency* from the clock.  A
background code planner can compute the same exact SRS-shaped, one-preview shortlist
while the foreground loop posts a fresh safe `noop` for each tick.  Once a plan is
ready, the existing verified executor takes over.  This tests a timing/architecture
hypothesis, not a new Tetris heuristic: 0008's depth-3 beam reached only 20.7 pieces
because its coordinate/hold execution was faulty, so this candidate deliberately
keeps 0007's executor, freshness checks, SRS tables, and conservative evaluator.

## What changed

* The 0007 placement generator and scoring remain the baseline.  It still owns board
  conversion, holes, line removal, SRS-shaped placement enumeration, hold handling,
  candidate validation, and the hard-drop executor; Jev cannot invent an action.
* Planning for a newly observed falling piece is submitted to `ctx.spawn` as finite
  ordinary Python work.  The main loop never calls `future.result()` until the future
  is done.  While it is pending, it posts `noop` and waits for exactly one tick, so a
  slow search does not let the prior piece's `hard_drop` be reapplied and does not
  intentionally block on model inference.
* The background result is tagged with a compact piece/board signature and is accepted
  only if the same falling piece and settled board are still present.  If gravity has
  moved the piece, the target's orientation and x are still revalidated by the
  existing executor; if a lock or hold changes the signature, the result is discarded.
  A completed plan is never used after a terminal observation.
* Jev remains an asynchronous advisory tie-breaker between the exact shortlist, with
  two independently ordered Choice questions and the 0007 agreement/probability
  gate.  Astra remains a slow policy reviewer, but neither can delay the foreground
  clock.  Garbage, malformed model answers, timeouts, and an unavailable result all
  fall back to the deterministic code plan.

## System-1 state, questions, and thresholds

The state sent to Jev is the compact JSON object made by `_jev_state`:

```text
{
  ruleset, policy, piece_instance,
  board: {height bucket, holes bucket, surface bucket, lines, level,
          combo active/inactive, back_to_back},
  active_piece: {type, orientation, anchor_x}, hold, can_hold,
  next_queue, fallback, candidates: {C0: description, ...}
}
```

Each candidate description names only a code-generated legal destination: current
versus hold, SRS orientation and anchor column, clear kind, projected height/hole/
surface buckets, and well depth.  `pick_a` and `pick_b` are independent `Choice`
questions over those same labels, with the second criteria dictionary reversed to
reduce first-option bias.  They are accepted only when both `.value` fields agree,
the label is in the request's candidate set, and each selected-label probability is
at least `0.30`; otherwise the deterministic plan remains active.  A Jev result is
fresh only for its original piece instance and candidate snapshot, and is rechecked
for hold/type compatibility before it can replace the fallback.  Jev never computes
coordinates, scores, arithmetic, or input sequences.

## Foreground fallback and timing policy

At every loop the code observes the game.  On a new piece it immediately posts a
valid `noop`, starts one planner future, and then repeats this pattern while that
future is pending: `ctx.act(noop)` followed by `ctx.wait_for_tick()`.  This is the
safe fallback because it neither changes the board nor consumes hold, and it avoids
repeating the previous piece's hard drop.  If planning fails or returns no legal destination,
the fallback is a validated `hard_drop`; if a movement/rotation does not change the
pose, the target is abandoned and a hard drop is used rather than repeating a blocked
input.  A hold is sent at most once and all actions are checked against
`action_spec.names`.

Only one planner future and one Jev future are outstanding.  Planner exceptions are
caught and converted to an empty result; model exceptions are caught as advisory
failures, while `ModelUnavailable` is allowed to propagate as required by the
contract.  The planner is finite (no unbounded loop), checks `ctx.done` between
candidate branches, and is discarded on terminal/deadline exit.  The harness never
waits for a model or planner future during cleanup.

## Confirmation or refutation

Confirmation would be: stale fraction materially below 0007's 29.5% (especially the
multi-tick gaps in the trace), no increase in illegal actions or top-outs, and scores,
lines, holes, and 200-piece completion close to 0007.  A score improvement would be
extra evidence that earlier steering rather than a changed heuristic matters.

The hypothesis is refuted if the planner future does not reduce stale ticks, if
background results are systematically invalidated by gravity/hold, or if moving the
same search off the foreground reduces line clears/survival.  A preflight garbage
model response, a planner timeout, or an empty plan must still produce a valid
fallback and no escaping exception.
