# Candidate 0021 — route council plus a hard escape governor

## Parents and diagnosis

Parents: `candidates/0016/` and `candidates/0019/`.

I retain 0016's route-goal council (continuous code-owned movement, inferred pill graph,
weighted ghost projection, and Jev choosing a semantic goal) and 0019's one-reserve-per-life
power-pill ledger.  The new test is a resource-aware safety governor, not a new
pill quota.

The concrete failure is visible in 0019's worst trace, `trace_seed1.jsonl`.  On the last life,
the controller repeatedly changed direction near the same exposure:

* frames 8640--8676 alternate `right`, `left`, `right`, `left`, with code-guard margins
  `24.31`, `4.21`, `3.31`, `-5.23`;
* at frame 8716 Jev accepted a route whose recomputed profile was `risk=3, margin=-15.54`;
* the episode ended at frame 8769 with the player at `[11,99]` and a ghost at `[13,99]`,
  score 2910, after the second maze epoch had left 59 dots and both power pills visible.

The same issue is present in the best parent: 0016 seed 3 installs a Jev dot route at
frame 6828 with `risk=3, margin=-18.0` and loses its next life at frame 6892; earlier it
alternates code-guard retreat packets with negative margins at frames 4200/4208 before the
frame-4340 loss.  Thus the parents do not merely lack reserve resources.  Their comparison is
allowed to choose among equally bad paths: `_poll_jev` accepts a choice when its risk is no
worse than the best available candidate, even if that best candidate is risk 3.  The parent
executor then replans often enough to turn a deteriorating local situation into left/right
oscillation.  The reserve lease in 0019 was exercised (`reserve_lease_started` and
`reserve_lease_consumed`), but it does not prevent this later-life local collision.  Also,
its protected-pill rule filters power goals but does not stop a dot route from crossing a
protected power node: at frame 5204 the second life logged `reserve_lease_unavailable` with
`visible=0`, so the intended future resource set had already been spent.

## Hypothesis

The route council and resource lease are useful in calm corridors, but two code invariants
are missing.  First, a reserve is not protected if an ordinary dot path can cross its node.
Second, the action bridge is unsafe when all route packets look bad.  A resource-aware
**escape corridor governor** should improve survival: ordinary routes must avoid every
currently protected power node, while a selected packet in the collision class or with margin
below four native pixels is vetoed and replaced by the best executable branch.  The governor
then holds that cardinal direction for a short pre-turn interval, revalidating it each tick
rather than replacing it merely because a new Jev answer names another route.  If every branch
is bad, it chooses the least-bad projected margin rather than an arbitrary risk-3 Jev route.

This tests both trace-supported failures without importing arcade geometry.  It does not
assume ghost colour means frightened, does not claim the inferred dot graph is a verified wall
map, and leaves calm routes unchanged except where they would spend a leased resource
incidentally.  The hard invariants are intentionally conservative: a lower-score surviving
life with an escape pill is preferable to a high-score collision.

## What changed

1. Keep 0016's graph/council machinery and 0019's epoch-reset and one-reserve-per-life
   lease.  Dot repopulation renews the lease ledger, so old consumed-pill state cannot leak
   into a new level.
2. Add protected-resource routing.  The graph search removes visible power nodes that are
   not currently leased/allowed from ordinary route traversal, and stale dot leases crossing
   such a node are discarded.  Reserve selection temporarily bypasses this filter only while
   finding its one reachable lease target; after that target is leased, other pills remain
   protected.
3. Add `goal_safety_veto`: immediately before the parent executor follows a goal, recompute
   its current risk profile.  A collision-class route (`risk >= 2`), a negative margin, or a
   route crossing a protected node is removed, including a route installed by Jev.
4. Add `escape_governor_lock`.  The governor derives executable alternatives from the
   parent's already computed packets, adds a small direct same-lane/approach check for ghosts
   near the first node, ranks safety class then margin then branch/continuity, and holds the
   chosen cardinal action for up to 20 native frames (8 for a risk-2 least-bad branch,
   4 for risk 3).  The lock is revalidated each tick and is released on a life/epoch change,
   invalid direction, or a newly unusable branch.
5. The governor is not a blind reversal rule.  It may select a retreat, tunnel/power route,
   or a continuing branch; it uses the least-bad margin if all choices are dangerous.  Model
   answers can still choose goals only in calm conditions, and cannot post a joystick action.
6. Add the governor state to Jev's existing compact state so Jev knows that code owns the
   safety veto and short lock.  Astra remains an asynchronous rubric auditor and never blocks
   the tick loop.

## Confirmation and refutation

Confirmation would be traces with protected pills still visible after ordinary dot routes,
`escape_governor_lock` before the previously observed left/right deterioration, no action
following a Jev goal with negative margin, fewer three-life terminal runs, or survival beyond
the comparable terminal frame without a large loss of first-life dot coverage.  A particularly
strong result is a later maze epoch with a life remaining.

Refutation is unchanged or worse survival while the lock fires repeatedly, a large reduction
in dots without avoiding the same terminal separations, protected pills being consumed by
ordinary routes anyway, or locks that trap the player in a corridor.  If locks are safe but
score falls, the next iteration should shorten the lock or raise the danger threshold rather
than weaken the hard veto.  If no lock events occur in bad traces, the direct pressure detector
or graph association needs repair.

## Tier-2 Jev bridge

Jev receives the inherited strict-JSON route-council state:

```text
rules: {objective, power_semantics, choice_semantics, risk_order}
plan: {rubric, epoch, stage, last_life, goal_kind}
options: {route_name: {purpose, risk, exit_branch, dot_yield,
                       reverses_motion, resource, continues_plan, geometry}}
recent: {score_progress, recent_power_event, guard_intervened}
resources: {visible_power_count, protected_power_count, lease_active,
            lease_target, ordinary_routes_cross_protected}
safety_governor: {hard_veto, lock, lock_frames_left,
                  purpose, recent_escape_events}
```

The two reordered `Choice` questions (`route_a`, `route_b`) select one of the already
computed route goals, and must agree with selected-option probability at least `0.70`; an
`unknown` option is rejected.  The batched `Noul` (`renew`) only flags code-confirmed stalled
progress for an eventual Astra audit.  Code performs graph search, distance/risk arithmetic,
legal-action filtering, resource accounting, freshness checks, and the final safety veto.
A Jev result is tagged by frame, resource epoch, life/death count, rubric version, and goal
existence; stale, contradictory, malformed, or newly dangerous results are ignored.  A late
or failed Jev call leaves the code-owned route/escape policy running.

Requests remain asynchronous with one outstanding request, an interval/event gate, and the
parent's 2.5-second timeout against the measured roughly 0.2-second Jev latency.  Astra is
also asynchronous, short-output and bounded; its rubric can affect calm route ranking but not
the escape lock.  Every loop posts a valid cardinal action and waits on `ctx.wait_for_tick`,
so no model latency creates stale ticks or an unhandled preflight exception.
