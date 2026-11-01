# 0010 — Hand-staged interaction transactions

## Parent and hypothesis

Parent: `candidates/0008`, the best evaluated candidate (mean score 2.0; seed scores
3/0/3). I used its traces and the public action contract, but this is a new
controller rather than a patch or copy of its frontier implementation.

The immediate failure is narrower than the general strategy problem. On seed 0,
0008 had already acquired a crafting table by about 9.5 seconds, then issued

```
place crafting_table b:-22,67,31 up
```

at 9.5, 36, 62.5, ... and 875.5 seconds. Every attempt was invalidated by
`adapter_feedback`; the final observation still had `crafting_table: 1` and
`held_item: null`. Seed 2 shows the same pattern after the table was crafted:
`place crafting_table b:-113,66,-64 up` was posted roughly every 26 seconds from
139.5 through 897.5 seconds. Thus the controller was not merely failing to
choose a good support: it never progressed past the same placement transaction.
All earlier candidates also generally issued a placement directly after
crafting, without an explicit `equip` stage.

**Hypothesis:** the semantic `place ITEM REFERENCE FACE` operation is sensitive
to the actual held hand and to the current interaction target. A table/furnace
placement should therefore be a transaction: (1) equip the named item in the
hand, (2) place it against a freshly observed, reachable support, and (3) accept
it only after an inventory/block postcondition. If a support rejects the action,
retire that support permanently for this episode and move to a different
interaction position. This should turn the repeated table loop into a verified
workstation and allow the already implemented recipe chain to reach tools,
stone, and exposed iron. A secondary part of the same action-bridge hypothesis
is to navigate directly to an observed block candidate when a resource is
occluded, instead of assuming that a nearby stand is the only legal approach;
this targets seed 1's hidden oak candidates.

This is intentionally a careful repair, not a long-term Minecraft route claim.
The prior candidates have not established that later Nether/stronghold actions
are executable, so 0010 first tests whether a correct hand/target transaction
unlocks the opening dependency chain.

## What changed

* `HandTransaction` persists a `table` or `furnace` intent across ticks. It
  emits `equip crafting_table hand` (or `equip furnace hand`) until the latest
  observation reports that item held, and only then emits the parameterized
  `place` command.
* A placement is confirmed only by an inventory decrease, a matching block
  change, or the observed workstation appearing. A feedback rejection or a
  no-postcondition timeout retires the exact `(item, reference-block)` pair;
  it cannot be retried later merely because the phase compiler still wants a
  table. The controller searches another visible/in-reach support, navigates to
  a new frontier, or moves on to locating the already-consumed table.
* The same transaction monitor is used for equip, craft, gather, navigation,
  smelting, eating, and movement. It records inventory, held-item, position,
  dimension, and target-block postconditions, rather than treating a nonempty
  feedback string or movement alone as universal progress.
* Resource acquisition is code-owned and dependency-driven: logs/planks/table,
  wooden pick, exposed stone/stone pick, optional weapon/food/bed, exposed
  iron/coal, furnace, real smelting, and shield. Exact counts, recipes,
  candidate filtering, target coordinates, and cooldowns stay in Python.
* A hidden but observed resource block can itself be a `navigate` target. The
  controller first tries `navigate b:x,y,z`, then gathers after the block becomes
  reachable. This is bounded to the observed candidate and never reveals a
  chunk or seed. If safe stands form a local basin, a short renewed manual
  frontier segment is used only before a usable workstation exists; it is not
  the normal movement policy.
* The code handles stale observations, lava/fire/drowning, low-health eating,
  visible hostiles, ordinary death and respawn. A rejected job is cancelled
  before replacement, and all one-shot repeats are separated by `noop` through
  the latch rule.
* No Astra call is made. The prior Astra priority calls did not break the
  placement/frontier stalls (0006 and 0007 both used them and still ended at
  opening progress), so this candidate isolates the concrete interaction
  hypothesis and preserves the wall budget for game actions.

## Jev interface and scheduling

Jev is used only when an immediate threat is present; it is not asked to count
items, find structures, select coordinates, or invent commands. At most one
background request is outstanding. While it is pending, a deterministic
survival fallback (shield, equipped attack, retreat, or noop) is posted, so a
0.2-second-ish classifier tail cannot freeze the real-time game. Answers are
accepted only if their threat signature still matches and the selected value is
one of the regenerated legal commands with probability at least 0.55. Low or
malformed answers use the code fallback. Budget exhaustion and model
unavailability retain the contract's normal behavior; ordinary timeout,
runtime, and malformed-answer failures are logged and ignored.

The strict JSON state supplied to Jev is:

```json
{
  "purpose": "Immediate survival response; code owns legality and movement.",
  "self": {
    "health_band": "critical|low|stable",
    "food_band": "low|fed",
    "dimension": "overworld|nether|end",
    "lava_or_fire": false
  },
  "threats": [
    {"kind": "creeper", "range": "touching|near"}
  ],
  "legal_actions": ["shield on", "attack 17", "navigate s:...", "noop"],
  "rule": "Choose only one supplied complete action..."
}
```

The one batched question is `response`, a `Choice` over the exact strings in
`legal_actions`. Its criteria describe each supplied response. The code's
hazard detector is authoritative: it cancels active jobs for a close threat,
uses movement controls for lava/fire/drowning, and never lets Jev fabricate a
parameter. Threat signatures are `(kind, observed id, touching/near range)`;
answers expire after 2.5 seconds or any signature change. A fresh answer is
also rechecked against the current menu immediately before posting.

## Confirmation and refutation

The hypothesis is confirmed if traces show an `equip ... hand` immediately
before a placement, a `placement_confirmed` postcondition, a visible crafting
workstation, and no long exact-support retry streak. The strongest progression
evidence would be the seed-0/2 table loop ending in a wooden pick and then an
advancement/stone inventory, plus seed 1 acquiring a log after direct block
navigation. Low stale ticks and zero harness errors are required.

It is refuted if the equipped-item placement still rejects on multiple fresh
supports, if inventory decreases without a workstation ever becoming usable,
or if direct block navigation cannot expose the seed-1 resource and the run
still remains in `find_wood`. If the controller reaches the same opening score
but spends its time in furnace/food/bed loops, that would show that this repair
fixed the interaction bridge but not the next strategic bottleneck; it would
not validate a claim of dragon completion.
