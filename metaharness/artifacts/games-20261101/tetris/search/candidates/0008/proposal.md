# Candidate 0008 — queue-aware beam search with an advisory Jev gate

## Parent and hypothesis

Parent: `candidates/0004/` (with `0003` as the placement-search comparison).

The concrete failure in `0004` is visible in its traces, not just in its score. In
seed 1, after the first placement, ticks 11–15 repeatedly apply the previous
`hard_drop` while the new piece is already falling. The next Jev request is not
even made until tick 15. The resulting locks are mostly at spawn x=3 and
orientation 0: pieces 2–6 build a central staircase, holes rise from 1 to 9,
and no line is cleared. Seed 0 has the same shape: after the first hold/left
sequence, pieces 2–5 lock at x=3 and holes rise from 0 to 6. Jev's later answers
are often fresh only after the fallback has already committed the piece. The
best 0004 episode reaches 474 only because one later I placement happens to
clear a line; its final board still has 29 holes and top-outs at 16 pieces.

The hypothesis is that the weak result is not primarily Jev recognition. It is
the one-ply, model-gated control structure: a fallback target can become stale
between piece transition and answer harvesting, and the score function has no
multi-piece commitment to preserve a usable surface. A fast deterministic beam
over exact legal placements for the visible queue should make every tick carry a
fresh executable action while using future pieces to avoid holes and value line
clears. Jev can still recognize a strategic choice, but it must be unable to
replace an invalid or stale geometry.

This is a structural change from `0004`: the beam is the controller, not a
waiting fallback. Hold choices and four future pieces are searched in code;
Jev is an asynchronous, two-view tie-breaker among the top legal first
placements. Astra is deliberately not called: the archived Astra calls were
cancelled at game-over and never preceded an action, so spending the slow model
on a real-time policy review is not a useful test of this hypothesis.

## What changed

* Code converts the public top-to-bottom board to bottom-origin cells, derives
  SRS shapes from the observed active coordinate convention, enumerates all
  hard-drop orientations and columns, removes rows, and computes holes,
  hole-depth, aggregate/max height, bumpiness, wells, row transitions and line
  clears.
* A beam (width 6, depth 3 at standard gravity; width 5, depth 2 when
  gravity is fast) branches over current, existing-hold, and empty-hold
  choices. It consumes the observed queue with the correct hold transition and
  scores future boards, rather than only looking one piece ahead. The
  deterministic top plan is immediately executable even if Jev is late or
  absent; the smaller fast-gravity beam keeps search near the tick budget.
* Execution posts exactly one valid action and waits for the next tick every
  loop. It re-observes after each rotation/move, verifies active type and target
  origin, issues hold at most once, and replans if a replacement or stale answer
  changes the piece. There is no previous-action stream caused by a synchronous
  model call.
* Jev receives only a compact state plus up to nine named plans. Two
  independent Choice questions reverse option order. A response is accepted
  only for the same falling piece (piece count, active type, hold value and
  hold permission must still match), when both choices agree and each selected
  probability is at least 0.42; steering x/orientation may change because
  those are revalidated by the executor. Otherwise the code beam remains in
  control. All model output is treated as optional and malformed output falls
  back safely.

## System-1 interface

The JSON state sent to Jev contains:

```text
{
  ruleset, policy, piece_instance,
  board: {height bucket, holes bucket, surface bucket, lines, level},
  active: {type, orientation, x, y},
  hold, can_hold, queue,
  plans: {C0: description, ...},
  fallback: "C0"
}
```

Each plan description states whether hold is used, the target orientation and
origin, clear kind, resulting height bucket, hole bucket and surface bucket.
These are facts computed by code; Jev is not asked to count cells, perform
arithmetic, generate SRS inputs, or invent a plan.

`pick_a` asks for the plan that best balances a clear with a low, flat,
hole-free result. `pick_b` asks the same question with reversed option order to
expose first-option bias. Agreement and the 0.42 per-choice probability gate
are sufficient because every option is already legal and the deterministic
beam is the fallback. A late answer is ignored if the piece count, active type,
hold value or hold permission no longer matches the request; movement and
rotation changes alone do not invalidate it because the settled board is
unchanged.

## Fallback and timing policy

The beam target is selected before any model call. A background Jev request never
blocks `ctx.act` or `ctx.wait_for_tick`; while it runs, the harness rotates,
translates and hard-drops the deterministic target. If it arrives before the
same piece is committed, it can replace the target only with a validated plan
from the original shortlist. If it times out, disagrees, has low probability,
contains arbitrary values, or arrives after a hold/board change, it is ignored.
A missing target uses a valid hard drop. `BudgetExceeded` and
`EpisodeOver` end normally; ordinary inference failures are swallowed, while
`ModelUnavailable` is allowed to propagate as required by the contract.

## Confirmation or refutation

The hypothesis is confirmed by a large reduction in stale ticks (the loop
should post an action on essentially every tick), survival beyond the 13–16
pieces reached by `0004`, lower final holes, and materially more lines/score.
A strong result with few accepted Jev choices would specifically show that the
queue search, rather than model prose, fixed the failure. It is refuted if the
beam still builds the same spawn-column staircase, if the public coordinate/SRS
assumptions produce illegal or mispositioned locks, or if depth-4 search is not
better than the deterministic one-ply fallback on comparable seeds. If every
Jev answer is rejected but the score improves, later work should remove the
advisory call; if accepted plans consistently hurt the beam, raise the gate or
use Jev only for risk classification.
