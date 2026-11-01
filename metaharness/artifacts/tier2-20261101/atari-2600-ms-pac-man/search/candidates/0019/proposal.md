# Candidate 0019 — route-goal executor with one-life reserve leases

## Parents and diagnosis

Parents: `candidates/0016/` and `candidates/0013/`.

The useful mechanism in 0016 is structural: it posts a deterministic cardinal action every
67 ms, turns the observed pill coordinates into an executable graph, scores short route
packets with a ghost-interception profile, and lets Jev choose among route goals rather than
replace the joystick one tick at a time. That architecture reached **4018.3 mean** and
actually observed dot repopulation, while keeping stale ticks essentially zero. Its trace
also exposes the remaining failure: in `trace_seed3.jsonl` it consumed power pills at
frames approximately **436, 1220, 1612, and 2004**, then lost lives at **4340** and **6892**.
Thus its route council can spend all four escape resources long before the first later-life
collision; `power_window` is correctly marked as uncertain, but there is no resource lease.

0013 supplied the careful counter-mechanism, even though its simpler parent controller scored
only **1653.3**: two pills are routine checkpoints, reserve pills are not exposed to the
ordinary hazard diversion before a life transition, and a reserve route is committed after a
demise (or on the last life). Its trace/proposal specifically reports the parent consuming
both reserve pills before its first death. I combine that resource timing with 0016's route
council and dynamic epoch/ghost machinery, rather than copying 0013's weaker graph executor.

## Hypothesis

The high score of 0016 comes from uninterrupted route execution and weighted interception,
not from consuming every power pill early. **A one-reserve-per-life lease will preserve the
0016 first-life dot sweep while making a remaining pill available after a death.** The first
two observed power pills can remain ordinary checkpoints. After those are consumed, all
other pills are protected from normal `power_escape` packets. A death, or entering the last
life, arms exactly one reachable reserve target; the route-goal executor then owns that target
until it is eaten or becomes invalid. After eating it, other reserve pills remain protected
until the next life transition. A last-life emergency gets the same one-target lease.

This is deliberately not a frightened-ghost assumption: recent pill consumption still only
changes a short tie-break window. If no pill route is offered, the code-owned risk packets and
branch exits still choose a legal cardinal escape.

## What changed

Starting from the 0016 route-goal council, this candidate adds an explicit per-resource epoch
ledger:

* checkpoint quota is two rather than three;
* the first two pills that the graph controller actually reaches are routine resources;
* remaining pills are not candidates for ordinary weighted routes or Jev route choices;
* a death/last-life transition arms one reachable reserve point, records it as a lease, and
  commits a route-goal lease to it; consuming that target closes the lease;
* a failed reserve route is remembered for the next life, so the same target is not blindly
  replayed;
* dot repopulation resets the ledger along with the 0016 resource epoch, renewing the
  policy for the new maze rather than carrying consumed-pills state forward.

The 0016 code-owned route risk, graph inference, alignment filter, ghost velocity projection,
progress recovery, epoch detection, and short route-goal Jev council are retained. Code still
posts every tick and revalidates every model result; Astra remains an asynchronous rubric
author and never controls an action. This is a structural resource-scheduling change, not a
seed-specific route or a latency tweak.

## Evidence that would confirm or refute it

Confirmation is a trace in which the first-life route reaches similar dot counts while at
least one reserve remains visible, followed by `reserve_lease_started` after a life loss and
`reserve_lease_consumed` on that later life. Strong confirmation is fewer than three deaths,
a maze/resource epoch after the refill, or survival past the corresponding 0016 death frame
at comparable score.

Refutation is a large first-life coverage loss with no survival gain, or repeated deaths at
comparable frames while the reserve lease is both reachable and unused. If the reserve is
consumed after a death but the player still dies immediately, the next iteration should alter
the graph's interception/turn model, not merely hoard more pills. If the resource ledger
prevents all maze completion, the two-checkpoint allowance was too conservative.

## Tier-2 Jev bridge

Jev receives a compact strict-JSON state, not raw RAM or a map dump:

```text
rules: {objective, power_semantics, choice_semantics, risk_order}
plan: {rubric, epoch, goal_kind, goal_target, life_phase, reserve_lease,
       reserve_target, stage, fallback_route}
options: {
  route_name: {purpose, risk_class, exit_branch, dot_yield,
               resource, continues_plan, reverses_motion, inferred_geometry}
}
recent: {score_progress, power_window_uncertain, guard_intervened}
```

Options are already computed executable route goals: a continuing lease, a dot-cleanup path,
a one-target reserve escape, or a short branch retreat. Code removes dominated risk classes,
checks physical alignment, and excludes protected reserve pills before constructing the state.
The only questions are two reordered `Choice`s over those named route goals plus a `Noul`
about whether the supplied progress/intervention evidence warrants replacement. Jev is not
asked to calculate paths, compare numbers, infer walls, or identify frightened colours.

The two Choices must agree and each selected-option probability must be at least `0.70`;
missing probabilities, disagreement, `unknown`, stale answers, changed epoch/life/rubric,
changed goal existence, or a newly worse risk profile are rejected. Requests are at least 24
native frames apart, one is in flight, and timeout is `2.5 s`; this leaves the deterministic
route executor in control through the normal ~0.2 s Jev latency. Accepted results install a
route-goal lease for at most 120 native frames, never a raw joystick override. The Noul is
only used to flag a measured stall for a future Astra interface audit, with a `0.80` threshold.

Astra receives observed progress, resource-epoch, life-loss, and Jev-rejection evidence. It
may return only `RUBRIC=balanced`, `RUBRIC=exits`, or `RUBRIC=cleanup`; malformed or late text
is ignored. It is launched asynchronously with a 100 s timeout and cannot block or post a
move. All model exceptions except `ModelUnavailable` are treated as ordinary failed advice;
stand-in garbage output therefore falls back safely during preflight.

## Execution fallback

Every loop iteration observes, updates the graph/ledger, polls completed inference, computes a
weighted route and hazard profile, and posts a valid cardinal action before waiting for the
next tick. If Jev or Astra is late, uncertain, malformed, or unavailable, code keeps the
current goal, rejects immediate danger, uses a measured branch/escape, or holds a legal
cardinal direction. It never posts `noop` as a thinking fallback and never waits for Astra.
