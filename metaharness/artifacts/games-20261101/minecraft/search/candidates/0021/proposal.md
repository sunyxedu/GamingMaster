# Candidate 0021 — carry the workstation to the mine

## Parents and diagnosis

Parents: `candidates/0015/` and `candidates/0019/`.

The useful mechanism in 0015 is the input-derived recipe compiler.  It turned the
parent's rejected fixed-size crafts into real opening progress on all three seeds:
0015 reached the opening score 3/3/3, and its seed-0 trace had a table at about
10 s and a wooden pick at about 27 s.  The useful mechanism in 0019 is the
postcondition-checked moving mine.  Its seed-0 trace verified a sequence of
`stair_step_confirmed` events and three cobblestone acquisitions rather than
counting surface movement as mining.

The concrete failure is at the handoff between these mechanisms.  In the 0019
seed-0 trace, the mine really did reach cobblestone:

* steps were confirmed at 163, 170, 176, 184, 192, 199, 205, 219 and later;
* `resource_confirmed` gathered blocks at 233, 237 and 242 s;
* the next action was `goto -22.00 68.00 34.00` at 242.51 s, followed by
  rejected returns to that coordinate and to `navigate b:-23,68,33` at 243.5 s
  and again around 314–315 s.

Thus the stone milestone was obtained, but the controller required a trip back to
the surface crafting table before it could make a stone pick.  The table-return
route was not a verified staircase postcondition and became the new 900-second
stall.  This is different from 0015's earlier impossible recipe count, and from
0019's already-fixed stationary excavation: the remaining boundary is a
workstation locality problem.

## Hypothesis

A portable-workstation handoff will cross the first real progression boundary:
after the opening, retain enough wood for a second crafting table, craft that
table in the 2x2 inventory grid, and place it at the current exposed mining face.
Then a stone pick, furnace, and shield can be compiled from the current audited
inventory without returning through a newly cut or unreliable staircase.

The change is intentionally structural rather than another retry/cooldown.  The
mine still only digs listed adjacent blocks and only counts a step after the
player enters the verified opened cell.  The new invariant is: **once the first
stone is collected, the next recipe must be satisfiable at a local workstation;
no old surface coordinate is a prerequisite for the stone pick.**

## What changed

* Retained and independently implemented 0015's feasible whole-recipe counts
  for planks, sticks, tables, picks, furnaces, beds and shields.  A craft is
  emitted only when the audited personal inventory can pay for at least one
  execution; cursor and container previews are not counted.
* The opening reserves four logs' worth of material (or equivalent planks) and
  explicitly makes a second crafting table after the wooden pick.  If a table
  has been consumed/placed but the candidate view lags, a short table latch and
  a coordinate-adjacent `goto` wait for the actual block candidate instead of
  inventing a table ID.
* `local_workstation()` first accepts a nearby observed table, otherwise crafts
  a carried table without a table ID, then hand-stages `equip` → `place` on a
  listed adjacent support.  The stone-pick branch invokes this at the mining
  face, not the original table.  The same handoff is used for the furnace and
  shield where possible.
* Retained 0019's moving transaction: a listed exposed target is approached,
  equipped, looked at, directly dug, checked against block disappearance/block
  changes, and followed by a `goto` into that target cell.  A vanished block,
  position change, or navigation completion alone is never credited as a
  resource.  Hidden solid interiors are never synthesized.
* After a death the controller clears local route/workstation hints and issues
  ordinary `respawn`; it does not restore dropped inventory.  Food, passive
  hunting, visible bed material, hazards, and a short frontier escape are
  opportunistic and bounded so they cannot silently replace the stone route.
* Jev remains an interrupt-level safety judge, not a command generator.  Python
  supplies a compact exact-action menu and hard vetoes lava/fire, drowning,
  falls, stale observations, critical health, and illegal/stale IDs.

## Jev interface and real-time policy

There is at most one asynchronous Jev request, only while a visible hazard
signature is present.  Routine recipes, coordinates, target IDs, inventory
arithmetic, table existence, and postconditions are code-owned.  The request
state is strict JSON:

```json
{
  "purpose": "immediate survival only",
  "self": {
    "health_band": "stable|low|critical",
    "food_band": "fed|low",
    "dimension": "overworld|the_nether|the_end",
    "lava_fire": false
  },
  "hazards": [{"kind": "zombie|lava_or_fire|...", "range": "near|touching"}],
  "legal_actions": ["equip observed_item hand", "attack observed_id", "shield on", "cancel", "controls back,jump", "noop"],
  "rule": "choose one supplied action; never invent an ID"
}
```

One batched request contains:

* `response: Choice`, whose criteria are the exact strings in
  `legal_actions`; it is accepted only if the value is still in the current
  menu, the hazard signature is unchanged, and its returned probability is at
  least 0.52;
* `safe_to_continue: Noul`, recorded as an advisory second judgment.  It does
  not override deterministic lethal checks.

The timeout is 1.5 s, below the slow game interval but above the measured warm
Jev tail.  While it is pending, the deterministic fallback attacks with an
observed weapon, eats, raises a carried shield, or retreats.  A bad/late/garbage
stand-in answer, stale value, timeout, or ordinary type/response failure is
caught and falls back; `BudgetExceeded` and `ModelUnavailable` retain the
platform's normal termination behavior.  Every posted action is revalidated by
`allowed`, and a `noop` separates repeated one-shot commands.

## Confirmation and refutation

The hypothesis is confirmed by a valid trace with the following causal order:

1. audited logs/table/wooden pick plus a retained second table (or its feasible
   planks);
2. the 0019-style `mine_transaction_begin` and multiple verified staircase
   steps;
3. three audited cobblestone items;
4. a local `equip`/`place` or observed portable table near the mining face,
   followed by `craft stone_pickaxe 1 <local-table-id>`; and
5. an iron/shield component, or at minimum a stone-pick advancement, without a
   rejected return to the original surface table.

At leaderboard level, any valid score above the universal 3-point opening,
especially `iron_shield > 0`, confirms the immediate hypothesis. A lower score
with a portable table and stone-pick postcondition but no iron would isolate the
next bottleneck as ore/furnace discovery. Zero stale ticks and no harness errors
remain required.

It is refuted if the controller still has three verified cobblestone items but
repeatedly routes to the original table, if a carried table cannot be placed
from a listed support despite the local handoff, if the table latch loops without
an observed block, or if many staircase steps are confirmed without changing
position. If all seeds remain at the opening despite a valid portable-table
transaction, the workstation hypothesis is wrong or the resource search (not
recipe locality) remains the limiting boundary.
