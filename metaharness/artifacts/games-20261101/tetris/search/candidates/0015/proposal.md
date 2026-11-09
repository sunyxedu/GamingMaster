# Candidate 0015 proposal

## Parent and hypothesis

Parent: `candidates/0014/`.  I kept its guardian/refinement controller,
verified SRS executor, one-preview search, hold handling, asynchronous Jev
role tie-breaker, and non-blocking Astra reviewer unchanged.

The parent solved the dangerous failure mode: it completed all 200 pieces in
all six episodes with zero illegal actions and almost no stale ticks.  Its
remaining weakness is scoring quality, not survival.  The worst parent run,
`candidates/0014/trace_seed5.jsonl`, ends with the public final observation
showing `lines: 78`, `score: 56942`, `holes: 0`,
`score_breakdown.line_clear: 45800`, and
`score_breakdown.back_to_back: 0`.  Its lock observations contain many safe
single clears and mini T-spins; for example, the trace's piece-7 lock is a
one-line `t_spin_mini`, followed by an ordinary one-line lock at piece 9,
which breaks the difficult-clear chain.  The other low-scoring run,
`trace_seed2.jsonl`, ends at `lines: 75`, `aggregate_height: 51`,
`score: 58220`, and also `back_to_back: 0`.  Thus the controller is stacking
safely but is not consistently creating the multi-line/difficult clears that
produce the remaining score headroom.  This is consistent with the aggregate
leaderboard breakdown: 0014 averages only 200 back-to-back points despite
49,400 line-clear points and 77.5 lines.

**Hypothesis:** the parent’s code ranking is too willing to take a safe single
when a still-safe double/triple/Tetris is available.  Its line term is
`128*lines + 26*lines^2`; that makes a four-line clear worth only 928 ranking
points before the structural penalties, while four independent singles are
worth 616.  Increasing only the nonlinear multi-line term to
`128*lines + 70*lines^2` makes the corresponding values 198/536/1014/1632 for
one/two/three/four lines.  The unchanged hole, hole-depth, height, bumpiness,
landing, and delta penalties remain the safety envelope.  This is deliberately
not a T-spin detector or an opener rewrite: it tests whether a stronger
multi-line preference alone recovers difficult-clear/B2B score without
sacrificing the proven survival controller.

## What changed

In `_placement_value` in `harness.py`, I changed exactly the quadratic
multi-line coefficient from `26.0` to `70.0` and documented why.  No model
prompt, action route, freshness rule, search depth, or fallback changed.  The
term is applied to guardian, refinement, and role portfolios because all of
them share the same code-owned evaluator.  A clear is still rejected or
out-ranked when it makes holes or an unsafe stack; the model cannot turn an
illegal placement into an action.

This keeps the experiment identifiable: compared with 0014, only the
relative preference for larger line clears is different.  Because the
simulator already computes the exact post-placement line count, the harness
never asks Jev to count cells or estimate score.

## Evidence that would confirm or refute it

Confirmation would be a mean score increase, especially in
`line_clear_points`, combo points, and `back_to_back_bonus`, with line count
at least as high as 0014 and no material increase in holes, top-out height,
or illegal/stale ticks.  The most diagnostic trace evidence would be more
placements with `lock.lines` 2--4 and consecutive difficult clears rather
than a long run of `lock.lines: 1` followed by `back_to_back: false`.
Improvement on seed 5 or seed 2 would directly support the stated failure
mode; improvement only through extra drop points would not.

It is refuted if the coefficient causes tall/well-dependent placements,
additional holes or top-outs, or lower lines/score while the trace still has
few multi-line clears.  If score and line distribution are unchanged, then
0014's main limitation is likely the shallow one-preview geometry or the
executor's placement choices, not the clear-vs-safety weighting.  If score
falls despite more Tetrises, the coefficient is too aggressive and the
safety envelope needs a later, separate refinement; this candidate does not
silently add that second change.

## Tier-2 System-1 interface retained from the parent

Jev remains advisory and never emits a primitive action.  Each request is
strict JSON with this schema:

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

The role descriptions are code-generated from already simulated legal
placements.  `safe` is the normal code-ranked placement, `clear` is the best
non-dangerous line-clear alternative, `future` preserves the preview surface,
and `recovery` reduces existing holes/top-out risk.  The new coefficient is
already reflected in all these descriptions' underlying ranks; Jev is not
told to perform the arithmetic.

One Jev request batches two independent `Choice` questions, `pick_a` and
`pick_b`, with the same four role criteria in opposite order.  The
instructions explicitly say to prefer recovery for holes/danger, clear only
when its described result is safe, and otherwise choose safe/future; they
forbid counting cells, raw-number comparisons, score calculation, and action
emission.  A role answer is accepted only when both choices are the same
valid role and each selected-label probability is at least `0.45`.  A
malformed answer, disagreement, low probability, timeout, or answer for a
stale piece is ignored.  The code then revalidates the selected plan against
the current active type and hold permission before using it.

The parent’s timing policy is retained: launch Jev only after an executable
action has been posted; while a request or deep search is pending, the
guardian plan continues.  If the selected plan is unavailable or becomes
incompatible, use the first compatible refinement plan, then the guardian,
then the current-pose emergency plan.  If steering is blocked or lock time is
urgent, post a validated `hard_drop`; no model call can stall the real-time
clock.  Astra remains an asynchronous, token-gated policy reviewer and can
change only the BALANCED/RECOVER/WELL weighting regime, never a keypress.
