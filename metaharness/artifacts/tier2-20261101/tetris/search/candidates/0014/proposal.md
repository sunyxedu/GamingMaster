# Candidate 0014 proposal

## Parents and hypothesis

Parent: `candidates/0009/`, with `0007` as the strongest comparison and
`0010`/`0011` as timing controls.

The archive separates two effects that the recent candidates conflated.  The
verified placement executor is the source of the large gain: `0007` reached 200
pieces and 78.33 lines on average.  But its best-seed-quality controller waits
for synchronous search at each boundary.  In `0007` seed 4, a fresh `noop` at
tick 9 is followed by stale `noop` ticks 10--11, and the same pattern repeats
at ticks 25--29 and 34--38.  Moving that exact search to a worker removed stale
ticks, but `0011` seed 3 shows the other failure: after piece 3, fresh noops
(ticks 19--21) precede steering only at tick 22; later a piece waits through
five fresh noops (ticks 73--77) before the fallback drops it.  The resulting
run survives 200 pieces but scores 43,600 and 75 lines, versus 48,974 and 76
lines for `0007` on that seed.  The worker is timely enough to make a plan,
but the foreground policy while it is pending is strategically blind.

The hypothesis is that the controller needs **two code-owned plans, not one**:
a cheap, immediately executable guardian placement and a deeper refinement
placement.  The guardian should steer the live piece while the exact preview
search runs, rather than repeating `noop` until the worker finishes.  When the
refinement is ready, it may replace the guardian only after freshness and hold
checks.  This should retain `0007`'s placement quality without paying its
stale-noop spans, and avoid `0011`'s wait-until-lock fallback.

The division of labour also changes.  Code computes all geometry, legal
placements, role portfolios, queue lookahead, and input routes.  Jev no longer
selects a coordinate from nine similar descriptions.  It chooses a *role*
(`safe`, `clear`, `future`, or `recovery`) from four code-built portfolios;
code then selects the exact placement inside that role.  This makes the fast
model judge a strategic tradeoff it can recognize without doing arithmetic or
inventing SRS input.  Astra is an asynchronous policy author, not an action
source: it reviews a meaningful board after the opening and, later, a danger
transition, returning one risk regime that changes future code weights and the
role descriptions.  A malformed, late, or cancelled review leaves the current
regime in force.

## What changes in `harness.py`

* Keep the proven SRS-shaped board conversion, line-removal simulation,
  one-preview exact search, hold bookkeeping, and verified one-action-per-tick
  executor.  Search runs in one finite `ctx.spawn` worker and is tagged with
  `(pieces, active.type)`.
* Add a synchronous **guardian search**: immediate current/hold placements only,
  with no preview recursion.  It is installed before the worker's one-preview
  search and is executed while that worker is pending.  Near lock it hard-drops
  rather than waiting.  The guardian is a real placement, not a no-op.
* Replace rotate-first steering with a small SRS-aware breadth-first route over
  `(orientation, x, y)` poses.  It chooses the next legal horizontal or rotation
  input from the current observed pose and replans after every tick.  If the
  target is unreachable or a movement is blocked, the current verified plan is
  abandoned safely; no stale target can issue an unbounded sequence.
* Convert Jev's interface from coordinate selection to portfolio selection.
  The code constructs safe/clear/future/recovery representatives from the legal
  refinement shortlist.  Jev receives only derived buckets, queue/hold facts,
  and one description per role, then answers two reversed-order `Choice`
  questions.  A role is accepted only if both values agree, are valid, and each
  selected-label probability is at least `0.45`; otherwise the code's safe
  role remains active.
* Start Astra after a few real placements, not on the empty spawn board, and
  optionally review once more when a high-stack/hole danger transition occurs.
  Its leading `BALANCED`, `RECOVER`, or `WELL` token changes the evaluator for
  future searches only.  It never blocks the clock or supplies a keypress.

## System-1 state, questions, and bridge

Each Jev request is strict JSON of the following form:

```text
{
  ruleset,
  policy: BALANCED | RECOVER | WELL,
  piece_instance,
  board: {
    height: low | moderate | high | critical,
    holes: hole-free | few shallow holes | several holes | many or deep holes,
    surface: flat | uneven | rough,
    lines, level, combo: active | inactive, back_to_back
  },
  active_piece: {type, orientation, anchor_x},
  hold, can_hold, next_queue,
  fallback_role: safe,
  roles: {
    safe: description,
    clear: description,
    future: description,
    recovery: description
  }
}
```

`safe` is the code-ranked low-hole/low-height representative; `clear` is the
best non-dangerous line-clear representative when one exists; `future` favors
the strongest preview-supported surface; and `recovery` minimizes existing
holes and hole depth.  The labels are semantic, while every underlying plan
already passed code simulation.  Jev is explicitly told not to count cells,
compare raw numbers, calculate score, or emit primitive actions.  `pick_a` and
`pick_b` ask the same role judgment with reversed criteria order.  A response
is fresh only for its original piece and role snapshot.  It is discarded after
a lock, hold replacement, timeout, disagreement, low probability, malformed
stand-in output, or failed compatibility validation.

The foreground policy is always executable: post one action and wait for one
tick.  On a new piece, post a safe `noop` only to cross the boundary, compute
the guardian immediately, start the deep worker, and execute the guardian on
subsequent ticks.  When the worker completes, adopt its compatible top plan and
launch the role question after an action has already been posted.  A pending
Jev answer never stalls the game.  If the guardian/deep plan is absent, the
active orientation and x form the emergency plan; if near lock, `hard_drop` is
the final valid fallback.  All actions are checked against `action_spec`.

## Evidence that would confirm or refute it

Confirmation is: zero illegal actions, stale ticks materially below `0007`,
200-piece completion, and line/score quality close to or above `0007` rather
than `0011`; traces should show steering or a hard drop while `search_ready` is
pending, followed by occasional role changes that remain same-piece valid.
An improvement concentrated on seeds where the worker was late would support
the guardian hypothesis.  A higher accepted-role rate with no increase in
holes would support the new Jev bridge.

It is refuted if the guardian's immediate placements create more holes than
`0011`, if BFS routes regularly fail despite legal targets, if refinement plans
are invalidated by gravity/hold, or if score/lines fall while stale ticks stay
low.  If score matches `0007` but Jev role changes never alter a plan, the code
architecture is confirmed and the role model is only diagnostic.  If Astra
never completes before a danger transition, its calls should be removed in a
later iteration rather than allowed to affect execution.
