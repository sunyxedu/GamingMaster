# Candidate 0012 — just-in-time dish timing and soft lane ownership

## Parent and diagnosis

Parent: `candidates/0006/` (with `0011/` as the closest refinement). I kept
only the verified ideas—symbolic recipe/pot bookkeeping, interaction poses,
BFS, explicit modern cooking start, and an asynchronous real-time motor—and
changed the production schedule and traffic contract.

The best candidates are not failing because Jev cannot name a phase. Their
low-scoring traces show a physical scheduling problem. In `0006/trace_seed3`,
the controller repeatedly commits to `stage_dish` at `[2,0]` with pose
`[2,1]` (for example ticks 138, 178, 225, 267, 307, and 348), then reaches
only 9 deliveries by tick 400. The episode ends with both chefs holding dishes
and the controlled chef at the pot approach. The diagnostics for this episode
are 108 collision ticks, 45 stuck ticks, and 105 pot-idle ticks. `0011`'s
`stage_gate_release` is a narrow escape from the same reservation and did not
change seed 3: it still delivered 9 soups/180 points. In contrast, the clean
`0006` seed 1 trace delivers 16 soups/320 points with no collision-driven
stall, so the recipe arithmetic and primitive motor are sufficient when the
shared approach is not occupied.

## Hypothesis

The long-lived `stage_dish` reservation is the avoidable bottleneck. A chef
fetches a dish shortly after cooking starts and then occupies the only useful
pot approach for most of the cooking interval. That is a poor trade: the dish
has no value until the soup is ready, while the same ticks could prepare the
next pot or let the scripted partner traverse the kitchen. The release rule in
0011 still starts from an already-contested gate and only releases in a narrow
case; it does not test the stronger alternative of never reserving the gate
while the pot is cooking.

Candidate 0012 therefore uses a **timed dish transaction**:

1. During cooking, compute a conservative shortest route from the current
   position to the dish dispenser and then to a pot interaction pose.
2. Fetch a dish only when the observed cook time remaining is close enough for
   that route (or when the pot is already ready).
3. If the dish arrives early, wait in a code-selected floor parking cell that
   is not any pot interaction pose and has a return path to the pot. The
   interaction square is entered only after the observation says the pot is
   ready.
4. While the dish is not yet due, use a legal next ingredient or counter
   buffer job. This creates useful overlap on multi-pot layouts without
   guessing at timers in the model.

A second, coupled structural change is a **soft lane assignment**. At reset,
the controlled chef owns the left or right ingredient side according to its
relative spawn position; the assignment is only a tie-breaker among legal
sources, never a hard assumption that a side exists. This prevents the
controlled chef from repeatedly crossing to the partner's dispenser after
successful deliveries. A temporary alternate lane can be accepted only from
an Astra advisory or a long-lived blockage, and every target/precondition is
revalidated from the current observation. Pot selection remains route-based,
so this does not assume that a layout has a particular number or arrangement
of pots.

This is deliberately not another gate lease: the default behavior is to free
and not claim the pot gate during cooking, rather than to hold it and invent a
release condition. The ready-soup, held-object, and explicit-start invariants
remain code-owned.

## What changed

* `TimedKitchen` has a persistent macro job, but its cooking-phase job is
  `await_dish` with a parking pose, not `stage_dish` at the pot. It converts to
  `pickup_ready` only after `is_ready` is observed.
* `_dish_due` uses only code-derived route lengths and the observed cooking
  remainder. It has a conservative buffer and falls back to fetching when the
  timer is unavailable; it never asks a model to count ticks.
* `_parking_pose` rejects every pot interaction pose, requires both an outbound
  and return route, and prefers the assigned lane. If no safe parking cell
  exists, the controller uses the least-bad legal approach rather than
  dropping the dish.
* The empty-hand scheduler prioritizes ready soup, full idle pots, and missing
  ingredients. While a pot cooks it does not fetch a dish early unless its
  route is due; it can fetch a compatible ingredient or stage one on a free
  counter. If it has just started cooking and no useful preparation is
  available, an explicit `park_cook` job moves it away from the pot before
  waiting; a bare stay at the interaction pose is not a fallback.
* The controlled side is derived from the actual initial observation and is
  only a soft tie-breaker. Directional traffic still uses current partner
  position, last action, predicted next cell, swap checks, and a non-reversing
  detour. No action is emitted from model text.
* Astra is launched once in the background as a layout-level reviewer of
  `just_in_time` versus `continuous` timing and lane stability. Its parsed
  output can only supply a bounded advisory; malformed or late output is
  ignored.

## Evidence that would confirm or refute the hypothesis

Confirmation would show a lower cooking-to-pickup gate occupancy and fewer
collision/stuck ticks on cramped-room seeds 0/3/6, while preserving the
16-delivery asymmetric traces. Harness logs should show `await_dish` parking,
then `dish_ready`/`pickup_ready`, rather than a `stage_dish` job beginning
immediately after every start. On the two-pot layouts, the confirmation is
also a lower pot-idle total because ingredient/counter work occurs during the
cook interval.

The hypothesis is refuted if just-in-time arrivals routinely miss a ready pot,
if parking blocks the only useful route more than the old gate reservation, or
if the lane tie-breaker causes repeated long detours. The decisive trace signs
would be a ready pot with a dish still far away, a repeated `await_dish`/
`pickup_ready` replan loop, or a score drop on the clean asymmetric seeds.
That would suggest retaining the parking idea but widening the due buffer or
removing lane preference, rather than adding another indefinite lease.

## Tier-2 System-1 interface and fallback

Every Jev request is a strict JSON object with no raw grid or arithmetic task:

* `tick` and `layout`;
* `self.position`, `self.held`, and `self.facing`;
* `partner.position`, `partner.held`, and `partner.last_action`;
* each pot's `position`, `phase`, `remaining`, and missing ingredient names;
* the current macro job (`kind`, target, whether it is waiting for readiness,
  and blocked-tick count);
* the derived left/right lane and the list of code-generated candidate kinds;
  and
* a short timing object containing only the active hint and candidate labels.

The one batched request asks two independent judgments:

1. `timing: Choice` over `prepare`, `fetch_dish`, and `wait`. `prepare` means
   continue a legal ingredient/counter job; `fetch_dish` means favor a ready or
   nearly-ready pot; `wait` means preserve an object and position. The
   controller accepts a closed-world value only with returned probability at
   least `0.60` (a stand-in with no usable distribution is accepted only as a
   label and still passes code legality). The hint expires after 16 ticks and
   cannot override ready-soup or held-object invariants.
2. `traffic: Noul` asks whether the partner visibly carries soup/dish and is
   approaching the shared route. Its probability-of-true must be at least
   `0.72`, and is used only when the partner actually holds `dish` or `soup`;
   it yields an empty-handed chef for at most two ticks. It cannot interrupt a
   carrying chef or create a movement direction.

Requests are asynchronous, have at most one outstanding future, and are sent
only at a job boundary or after a bounded route blockage (minimum 18 ticks
between requests). The response is tagged with the macro-job epoch; a late
answer is discarded. Timeout, malformed answer, missing probability, cancelled
future, or exhausted budget leaves the deterministic scheduler in charge.
Every `ctx.act` receives an advertised primitive name, and an exception gets a
final safe `stay` attempt. Thus model latency cannot create stale ticks or turn
an early dish judgment into an illegal interaction.

Astra's output is similarly closed-world and weak: `continuous` is recorded
only as a timing advisory and `alternate` gives a finite lane hint. It cannot
change the exact due calculation, route, interaction preconditions, or
priority of a ready delivery. All model work is started with `ctx.spawn`, so
the 6-Hz game clock continues while it reasons.
