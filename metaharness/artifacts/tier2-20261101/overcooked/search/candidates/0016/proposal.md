# Candidate 0016 — role boundaries with a bounded delivery takeover

## Parent and diagnosis

Parent: `candidates/0009/`.  I retained its useful committed physical jobs,
pose-aware BFS, partner-aware collision avoidance, and asynchronous model bridge,
but replaced its partner-claim/lease layer.  I also used the best current
candidate, `0012`, as the failure reference.

The traces show that the remaining loss is not a Jev latency problem.  In
`0012/trace_seed0`, after the first production, the controller commits
`deposit_counter` at tick 23, `park_cook` at tick 24, and then `wait` at tick
43.  Similar passive intervals occur at ticks 66--85, 235--239, 269--292, and
326--330.  The episode ends at tick 400 with only 180 points and a three-onion
pot at cooking tick 4.  The final trace also shows the controller at `[1,2]`
while the partner is at `[1,1]`; the controller has spent the last cycle
rebuilding a pot rather than starting it early enough to deliver it.
`0012/trace_seed3` has the same structure: `park_cook` at 64, `wait` at 83,
then another long build/park interval; it ends at 180 with a pot cooking at
tick 5.  By contrast, the clean `0012/trace_seed1` trace repeatedly commits
`fetch_ingredient -> fill_pot -> start_cooking` without buffering an ingredient
in front of a full idle pot and reaches 360.

The old controllers also made the partner's held dish into an implicit global
exclusion.  In `0012`, the ready branch is skipped whenever the partner holds a
dish, but there is no progress obligation for that claim.  `0009` made the
opposite mistake: its `yield_partner` transaction could become the whole policy.
Its ring seed 2 commits `yield_partner` at tick 62, then `wait` at tick 90, and
ends with 20 points, two ready soups, a partner-held dish, and 659 pot-idle
ticks.  Both indefinite yielding and indefinite parking are therefore
refuted; a partner claim must have an observable progress/expiry condition.

## Hypothesis

The controlled chef should have a **role**, not a fresh task choice on every
state change.  On a one-pot or partner-efficient layout the safest role is a
supplier: fill the next missing recipe ingredient and start a full idle pot,
while allowing the scripted partner to own a visibly progressing dish
transaction.  The supplier must not fetch an extra ingredient merely because a
pot is cooking when there is no second usable pot; that creates the counter and
parking detours seen in the 180-point traces.

The role must have one bounded exception.  If a ready pot is present and the
partner's held dish/soup has not changed position or object state for eight
observed ticks, the controlled chef becomes a delivery backup and fetches a
dish.  This is a takeover of a stalled physical transaction, not a lease: it
expires as soon as the partner progresses, and it can never override a chef
already carrying soup, an immediate start, or a legal held-object transaction.
The same event-driven boundary should preserve the 16--18 delivery behavior
on the clean asymmetric layouts while removing long no-progress intervals on
cramped layouts.

## What changed

* `RoleKitchen` has a persistent mission and a role boundary.  Code computes
  recipes, pot phases, exact interaction poses, shortest paths, collision-safe
  next moves, and all legality.  It does not repeatedly ask a model to choose
  a raw action.
* Astra is a one-shot **protocol author**.  It sees the initial terrain and
  pot/partner facts and returns only `role`, `circulation`, and `handoff` JSON.
  The parsed values are advisory and cannot alter a hard priority or primitive.
* Jev is no longer a task planner or a timer oracle.  At a mission boundary or
  bounded traffic blockage it judges (in one batched request) whether the
  current safe role is `supply`, `take_delivery`, or `yield`, and separately
  whether an already code-detected ready-pot stall warrants takeover.
* The normal scheduler gives full idle pots an immediate `start_cooking` branch
  before it stages another ingredient.  During cooking it either prepares a
  genuinely unfinished pot, fetches a dish only near the observed cook
  boundary, or parks away from every pot approach.
* The partner signal is a compact transition detector: unchanged partner
  position and held object increment `idle_ticks`; a movement or held-object
  change resets it.  It is a soft handoff trigger, never a permanent claim.
* Passive jobs now have an explicit exit condition: `park`, `wait`, and
  `wait_ready` remain valid only while the same passive task is still present in
  the newly derived task set.  A ready/partial/full-idle transition therefore
  reopens the production graph instead of preserving a stale wait forever.
* Model calls are submitted with `ctx.spawn`; timeout, garbage stand-in output,
  stale epoch, absent probabilities, or exhausted budgets leave the deterministic
  motor running.  Every posted action is validated against the advertised
  action names and the loop catches normal realtime/budget closure exceptions.

## Confirmation and refutation

Confirmation would show fewer long `park/wait` runs, lower pot-idle and stuck
counts on training seeds 0/3/6, and a final partial/cooking pot only when the
remaining horizon genuinely makes another delivery impossible.  In event
logs, a stalled partner-held dish should produce one bounded `take_delivery`
transition, not repeated yield missions.  The asymmetric seeds should retain
roughly the old 16--18-delivery production cadence because supplier work is
still selected by shortest legal route and all full pots start immediately.

The hypothesis is refuted if the eight-tick takeover threshold causes two
chefs to contest a ready-pot approach, if supplier mode starves a second pot,
or if passive-job exit causes route oscillation.  Decisive signs would be two
held dishes with no score increase, repeated `fetch_dish`/`wait_ready` cycles,
or a full idle pot surviving multiple observations while an ingredient is
being staged.  That evidence would call for a layout-level role switch or a
more explicit counter handoff, not a longer global lease.

## Tier-2 System-1 interface

Each request carries one strict JSON state (no raw full observation and no
arithmetic request):

```json
{
  "tick": 123,
  "layout": "...",
  "self": {"position": [x,y], "facing": [dx,dy], "held": ""},
  "partner": {"position": [x,y], "held": "dish", "last_action": "east",
              "idle_ticks": 4},
  "pots": [{"position\...", "phase": "ready|cooking|full_idle|partial|empty",
            "missing": ["onion"], "remaining": 5}],
  "protocol": {"role": "adaptive", "handoff": "takeover"},
  "job": {"kind": "fill_pot", "target": [x,y], "blocked": 0},
  "candidates": [{"id": "...", "kind": "...", "description": "..."}]
}
```

The batched questions are:

1. `role: Choice` with fixed closed-world choices `supply`,
   `take_delivery`, and `yield`.  `supply` means missing ingredients/full idle
   pot; `take_delivery` means a ready pot after a partner stall; `yield` means
   preserving a visibly progressing partner transaction.  The answer is used
   only for the current highest-priority band, only while its epoch is current,
   and only when its selected probability is at least `0.55` (a stand-in with
   no probability is accepted as a label and still passes code legality).
2. `takeover: Noul` asks the direct proposition “is a ready soup stranded and
   the partner not progressing?”  A true value is accepted at `0.70` or above,
   but code still requires a ready pot, an empty controlled hand, and the
   observed partner condition.  It cannot cancel delivery of a held soup.

There is at most one outstanding Jev future and at least 18 ticks between
requests.  Answers are tagged with `job_epoch`; late answers are dropped.  The
fallback is entirely code-owned: deliver held soup, pick up ready soup, start
full idle pot, fill a compatible pot, fetch a missing ingredient, perform a
bounded delivery takeover, fetch/stage a dish near readiness, park, then wait.
A Jev answer can select among safe equal-priority role branches but cannot
invent a target, do arithmetic, issue a direction, or create an unbounded
lease.  Astra's protocol has the same advisory-only status and is never awaited
by the tick thread.
