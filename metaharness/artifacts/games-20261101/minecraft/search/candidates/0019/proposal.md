# Candidate 0019 — make the mine move

## Parents and diagnosis

Parents: `candidates/0015/` and `candidates/0016/`.

I am crossing the mechanism that earned each parent its robustness rather than
copying either controller wholesale:

* 0015's inventory-derived recipe compiler fixed the opening transaction.  It
  reached score 3 on all three training seeds and replaced impossible requests
  such as `craft acacia_planks 8` with feasible whole-recipe outputs.  The
  seed-0 trace made a table at about 10 s and a wooden pick at about 27 s.
  Its failure after that was structural: it posted `goto -22.00 68.00 31.00`
  29 times while in the stone phase, including 11 invalidated table approaches,
  without ever acquiring a stone block.
* 0016's transaction engine was a better local executor.  It verified real
  inventory/block postconditions and its seed-0/seed-2 traces recorded 51 and
  28 `excavation_confirmed` events.  But those events were mostly removals in
  the original surface layer.  It did not enter the space it opened: after a
  wooden pick, the trace repeatedly did `look_at ...` / `dig ...` on blocks at
  the same elevation, then selected another surface block.  It accumulated
  49 `no_postcondition` plus 49 job-timeout suppressions in seed 0 and still
  had no cobblestone, iron, or shield.  Seed 1 remained in a safe-stand loop
  around the swamp tree and ended with no wood.

The common missing invariant is therefore not another recipe or a longer
cooldown.  A vanished surface block is not a mine, and movement to a table is
not stone progress.  A resource search must (1) choose only an observed block,
(2) verify its disappearance, and then (3) verify that the player moved into
the newly opened feet cell.  This is especially important because unknown
solid interiors are not exposed by the adaptor; the harness must never invent
an underground coordinate.

## Hypothesis

A short, controlled staircase will cross the first terrain layer more
reliably than a stationary surface excavation or a safe-stand survey.  After
the wooden pick is available, the controller prefers an observed exposed
stone block.  If there is none, it chooses an observed adjacent grass/dirt/
stone block at the player's current floor or one level below, never the block
supporting the player's feet.  It aims, digs, waits for the block-change
postcondition, and then navigates/gotos the removed cell as the next feet
position.  The next observation consequently has a chance to expose the next
layer.  The same moving transaction is reused with a stone pick for coal and
iron.

This tests a bold geometric change while retaining the two parent mechanisms:
feasible recipe counts and hand-staged workstations from 0015, plus the
postcondition/safety/ordinary-respawn discipline from 0016.  It does not assume
that every seed has exposed ore, a village, a cave, or a particular coordinate.

## What changed

### 1. A moving step transaction replaces stationary excavation

`begin_direct_mine` creates a `step` job with the observed block id and
position.  Its stages are:

1. find a listed safe stand or loaded feet position if the block is not in
   reach;
2. equip the currently appropriate pickaxe;
3. `look_at` the recorded block and issue one direct `dig`;
4. accept success only when the candidate disappears, a matching block change
   is observed, or a real inventory delta is seen;
5. issue a `goto` to the removed cell first (the cell is now part of a
   verified corridor), with a listed safe stand as fallback, and accept the
   step only when the current position is close to that expected feet
   position.

Every confirmed step is appended to a tiny in-memory route.  Before crafting
at the table, the controller backtracks through those already verified feet
cells in reverse order; it does not ask pathfinder to jump from a deep new
cell directly to a workstation.  This specifically addresses the live trial
failure in which `goto -21.00 68.00 34.00` timed out after the staircase had
successfully reached the first stone layer.

The supporting block directly under the current feet is vetoed.  Fluids, lava,
fire, falling hazards, sand/gravel, and visible hostiles are not used as normal
mining targets.  A failed target and its transaction are suppressed for a
bounded interval; movement alone cannot complete a mining postcondition.

Exposed valuable blocks are handled separately with `gather`, so their real
drops are collected rather than discarded by a bare `dig`.  Once three
cobblestone/cobbled-deepslate items are audited, the compiler returns to the
observed table and crafts a stone pick.  It then seeks coal/iron, eight stone
for a furnace, and smelts real raw iron before crafting a shield.

A moved step is also not allowed to become an infinite surface tunnel.  After
eight verified steps without a stone/resource postcondition, code makes one
safe frontier move to refresh the loaded candidate window, then resumes the
same step transaction.  The refresh is a bounded observation change, not
credit for mining progress.

### 2. The opening compiler is retained but independently reimplemented

Logs, planks, sticks, the crafting table, wooden pick, stone pick, furnace, and
shield all use input-derived feasible counts.  A carried table or furnace is
equipped before placement, and a consumed table/furnace is latched while the
candidate stream catches up; the controller never crafts a duplicate merely
because a block observation is lagging.  Table routing is a workstation step,
not the precondition for starting a resource search.

Wood exploration uses a persistent safe-stand frontier and prefers unvisited,
farther listed stands instead of cycling the nearest stands.  If pathfinding
has no stand in a swamp/tree basin, a three-second look-then-forward lease is
used to leave the basin and is stopped explicitly.  Food blocks are gathered
only when actually observed.  Death clears transactions and uses ordinary
`respawn`; it never restores dropped inventory.

### 3. System-1 is an interrupt, not a command generator

The harness does not ask Jev to count resources, calculate a recipe, discover
an ore coordinate, or invent a command.  Code owns the milestone compiler,
coordinate arithmetic, candidate freshness, postconditions, and legal action
strings.  This keeps the useful low latency of 0015 while avoiding a Jev call
on every routine craft/navigation choice.

## Jev interface and real-time policy

There is at most one background Jev request, created only when a current hazard
signature is nonempty or changes.  While it is pending, hard code continues an
immediate fallback; the real-time world is never paused for inference.  The
request uses this strict JSON state (rebuilt for every signature):

```json
{
  "purpose": "Immediate survival only; choose one exact supplied action.",
  "self": {
    "health_band": "stable|low|critical",
    "food_band": "fed|low",
    "dimension": "overworld|the_nether|the_end",
    "lava_fire": false
  },
  "hazards": [
    {"kind": "creeper|ranged|lava_or_fire|drowning|fall|...",
     "range": "touching|near"}
  ],
  "legal_actions": [
    "equip <observed item> hand", "attack <observed entity id>",
    "shield on", "cancel", "controls back,jump", "noop"
  ],
  "rule": "Never invent an ID; leave environmental hazards before work."
}
```

The batched questions are:

* `response: Choice` — selects exactly one member of `legal_actions`; each
  criterion describes the complete executable string.
* `safe_to_continue: Noul` — asks whether the current work can continue without
  defense.  The harness records the answer only as advisory; it does not let a
  positive answer override a lava, drowning, fall, stale-observation, death,
  or low-health veto.

A Choice is accepted only when its value is in the freshly built menu, its
probability for that value is at least `0.52`, and the hazard signature is
unchanged.  Otherwise the deterministic fallback is used.  The fallback
leaves lava/fire with `controls forward,jump`, surfaces from drowning with
`controls jump`, eats at health <=6 or food <=5 when food is carried, attacks a
visible touching hostile only with an observed weapon, raises a carried shield,
or retreats with `controls back,jump`.  A bounded job is cancelled before an
interrupt replaces it.  Jev timeouts, malformed stand-in answers, stale values,
and ordinary runtime/type failures are caught and cannot escape the episode;
`BudgetExceeded` and `ModelUnavailable` retain the platform's normal
termination behavior.  The request timeout is 1.5 s, above the measured warm
Jev tail but short relative to a dangerous real-time interval.

## Confirmation and refutation

The hypothesis is confirmed by a zero-error trace on a former 0015/0016
plateau seed containing, in order:

1. audited logs, a table, and a wooden pick;
2. `mine_transaction_begin` for an observed floor/wall block;
3. a matching `dig` block change and `stair_step_confirmed` at a new feet cell;
4. a newly exposed stone candidate and an audited cobblestone delta; and
5. a stone-pick craft, ideally followed by raw iron/iron ingots and the shield
   score component.

At leaderboard level, any valid increase above the 3-point opening plateau,
especially `iron_shield > 0`, confirms the immediate hypothesis.  A seed-1
run that leaves the swamp sooner without repeated exact-tree gathers is useful
secondary evidence.  Zero stale ticks and no harness errors are required;
unsafe death is a gameplay result, not evidence to hide.

It is refuted if the controller records many `stair_step_confirmed` events but
player Y/position never changes, if the removed cell cannot be entered despite
valid observed stands, if repeated steps expose no new candidates while the
controller is genuinely descending, or if it still returns to a table before
starting an observed stone search.  If it reaches stone/cobblestone but cannot
find iron, that confirms the moving-mine repair and identifies ore discovery
or smelting as the next bottleneck.  If all runs remain at zero because the
opening frontier cannot leave a swamp, the next proposal should change safe
terrain traversal rather than add more mining retries.
