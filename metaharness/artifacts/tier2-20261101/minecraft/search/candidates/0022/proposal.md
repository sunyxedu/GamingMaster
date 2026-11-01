# Candidate 0022 proposal — carry the workstation across the mine boundary

## Parent and diagnosis

Parent: `candidates/0019/` (with `0015` as the best opening comparator).  I read
`0019`'s three evaluation traces and the `0015` trace that had the fastest opening.
The opening is no longer the main problem: `0015` made a table at about 10 s and a
wooden pick at about 27 s on seed 0, and `0019` made a table/pick at roughly 17/34 s
on seed 0.  Both then stop before the first defense milestone.

The concrete failure is a workstation-route boundary, not merely a recipe count.
In `0019` seed 0, the moving mine did produce real local progress:

* `stair_step_confirmed` ran 25 times from about 41 s through 228 s;
* four `resource_confirmed` events followed, and the observation contained three
  cobblestone plus `minecraft:story/mine_stone`;
* immediately afterward the controller posted `goto -22.00 68.00 34.00` at about
  242.5 s, followed by the same table approach and `navigate b:-23,68,33` retries.
  The gotos were rejected or timed out at about 243.5, 314.5, 385.5, 473, 560.5,
  648.5, 735.5 and 823 s.  It ended with a wooden pick and the three cobblestone,
  but no stone pick or iron.

The same trace shape is visible in `0015`: after a verified opening, the controller
returns to table positions repeatedly (including 11 invalidated table approaches)
instead of making the next tool.  `0019` improved mining geometry enough to reach
Stone Age, but it still assumed that a table at the spawn site remained a usable
workstation after an expedition.  `0016`'s stationary excavation and `0018`'s long
expeditions did not test a local workstation; adding more table navigation would
repeat the failure identified in `lessons.md`.

## Hypothesis

A **mobile workstation** will cross the first progression boundary.  Before leaving
the opening, the harness makes two crafting tables whenever the observed inventory
permits it.  It places one at a local surface support and keeps the second in the
personal inventory.  After the moving mine verifies its first stone/cobblestone,
it places the carried table beside the current feet cell and crafts the stone pick
there.  The furnace is handled with the same local placement rule.  Thus the mine
never needs a long path back to a table whose route is blocked by newly changed
terrain, water, or a staircase.

This is seed-agnostic: the second table is made only from audited logs/planks, its
support is an observed in-reach block, and the mine targets only observed candidates.
No structure coordinate, seed fact, or hidden underground block is assumed.

## What changed

* The opening compiler has a distinct portable-table stock phase.  It requests
  `crafting_table 2` only when the current audited planks can actually pay for two
  outputs; otherwise it makes the feasible count and gathers more observed logs.
  The wood, planks, sticks, wooden-pick and bed recipes are all bounded by current
  inventory counts.
* `ensure_local_table` prefers a nearby observed table, otherwise equips and places
  the carried spare on a new observed support.  A successful place is latched only
  after the table inventory count decreases or a matching block candidate appears.
  The latch is labelled as a derived coordinate from the observed support and is
  used only while it remains within local reach.
* The resource executor uses direct bounded `gather` for observed logs, food, coal
  and exposed iron.  For stone that is not exposed, it uses a new moving transaction:
  observed adjacent block -> look/equip -> direct `dig` -> observed disappearance ->
  `goto` the newly opened cell.  Movement alone is never a mining success.
* After three audited cobblestone/cobbled-deepslate items, the controller places
  the spare table at the mine, crafts a stone pick with that table, and continues
  the same local search.  Once eight stone is audited it crafts/places a local
  furnace, smelts real raw iron and then crafts a shield.  There is no route-back
  branch in these phases unless a local table cannot be made from real inventory.
* Failed actions are keyed and cooled, and every one-shot action has an explicit
  noop re-arm.  Jobs are cleared on death; respawn is ordinary and dropped
  inventory is not restored.  Safety code vetoes lava/fire, drowning, falls, stale
  observations and low-health hunger actions before the progression compiler.

## System-1 interface and real-time policy

Astra is intentionally not called in this candidate.  The archive traces show that
long planning calls did not repair an adapter/workstation boundary, while the
milestone arithmetic and postconditions are deterministic and observable.  The
hypothesis is therefore tested without spending the 40-call System-2 budget or
pausing a real-time mine for a slow plan.

Jev is only an interrupt classifier.  Its state is rebuilt from the current
observation and contains:

```json
{
  "purpose": "Immediate survival interrupt",
  "self": {
    "health_band": "critical|low|stable",
    "food_band": "low|fed",
    "dimension": "...",
    "in_lava_or_fire": false
  },
  "hazards": [
    {"kind": "zombie|creeper|lava_or_fire|drowning|fall|...",
     "id": "observed id or self", "range": "touching|near"}
  ],
  "legal_actions": ["exact commands compiled from observed IDs"]
}
```

One request batches two independent questions:

1. `response: Choice` selects one exact member of `legal_actions`.  Criteria describe
   the complete executable string; Jev never supplies a coordinate, count, or ID.
2. `safe_to_continue: Noul` is advisory context for the trace and is not permitted
   to override the hard code vetoes.

The request is launched in the background with a 1.3 s timeout.  While it is late,
the deterministic fallback immediately retreats, jumps out of water, equips/eats,
raises a carried shield, attacks only a visible touching entity with an observed
weapon, or stops.  A returned Choice is accepted only if its value is still in the
fresh menu, its probability is at least 0.50, and the hazard signature has not
changed.  Invalid, stale, timed-out, or arbitrary stand-in answers fall back without
escaping `run_episode`; budget exhaustion and persistent model unavailability retain
the platform's normal termination behavior.  A bounded job is cancelled before a
hazard response replaces its hand ownership.

The controller itself posts one action per real-time tick and uses `noop` while a
bounded job or transaction is active.  Candidate IDs, recipe arithmetic,
coordinate transforms, inventory deltas, workstation latches, action deadlines and
milestone transitions are code-owned.  Jev does not decide whether a table is
reachable or whether a craft count is legal.

## Confirmation and refutation

The hypothesis is confirmed by a valid trace that contains, in order:

1. an audited opening table plus a second carried table (or an equivalent verified
   local-table placement),
2. the `0019`-style moving mine and an audited cobblestone/Stone Age transition,
3. `portable_table_placed` at the mine rather than a return to the original table,
4. a successful `craft stone_pickaxe ... <local table id>`, and preferably raw iron,
   a local furnace/smelt and the shield score component,
5. fewer repeated table gotos and no harness/model errors.

At leaderboard level, any valid score above the 3-point opening plateau, especially
`iron_shield > 0`, confirms the immediate hypothesis.  A run that reaches
cobblestone but cannot place the spare table would refute the placement/execution
part and identify support selection or candidate freshness as the next repair.  A
run that crafts the stone pick locally but still cannot find iron would confirm the
workstation hypothesis and move the bottleneck to ore search, fuel or smelting.  If
all seeds remain at the opening score while the traces show two tables and no table
return, the hypothesis is refuted rather than hidden behind repeated retries.
