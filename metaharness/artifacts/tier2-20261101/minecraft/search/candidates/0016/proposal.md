# Candidate 0016 — expose the mine instead of orbiting the table

## Parent, evidence, and hypothesis

Parent: `candidates/0006/` as required by the generation directive.  The
closest implementation comparator is `0012`, because it attempted explicit
`navigate -> equip -> look_at -> dig -> pickup` transactions.  `0008` and
`0013` are useful controls for the opening workstation.

The archive shows that local command validation is no longer the main missing
piece.  Several runs verified wood and a table, but the controller then put the
player back into a loop whose precondition was not a resource precondition:

* In the `0012` seed-0 trace, direct logs were confirmed at about 32, 51, and
  103 seconds.  A wooden pick was crafted at about 122 seconds.  From about
  142 seconds onward the trace repeatedly posted `navigate b:-21,68,31`, the
  table block, while the mode was `stone`; it never acquired stone.  The
  implementation returned to a table before its stone search, so it could not
  mine/excavate away from the workstation to discover stone.
* The same trace's manual frontier used `controls forward,jump`, which is not a
  safe exploration primitive in this world.  In `0012` seed 1 the controller
  repeatedly posted `controls jump` around a water spawn and died at about 151
  seconds.  Seed 2 spent the episode beginning another manual frontier every
  roughly 18 seconds without obtaining wood.  A navigator-only frontier has
  also already been tested and did not expose a useful resource on seed 1.
* Even after `0013` repaired the hand-staged table placement, its downstream
  trace still made long craft/navigation cycles rather than starting a stone
  acquisition transaction.  In contrast, the initial observations visibly
  contain ordinary `grass_block`/`dirt` surface blocks and hidden ores.  The
  interface exposes only exposed blocks, so an agent that waits for a visible
  stone candidate can be permanently blind to the first mine layer.
* `0008` demonstrates the cost of retrying an interaction without a changed
  postcondition: the same table placement was invalidated 34 times on seed 0
  and 30 times on seed 2.  `0012` likewise retried a bad table support in seed
  1 for the remainder of the episode.  A failed support must be replaced or
  abandoned, not turned into the next phase's gate.

**Hypothesis:** the missing structural operation is a short, safe *surface
excavation* transaction.  Once a wooden pick exists, code should remove
observed, adjacent low-value ground blocks (never the block under the player's
feet), verify that the block actually disappeared, and rescan the newly exposed
slice for stone, coal, or iron.  Resource search must be independent of table
proximity; the table is needed only after the resource postcondition is
verified.  Safe `s:` navigation may refresh the candidate window, but blind
manual movement must not be the primary exploration method.  This should turn
the verified wooden opening into the first verified stone/iron transition
without relying on a seed-specific coordinate.

This is deliberately a rewrite of the division of labour, not a longer retry
cooldown:

* **Code** owns milestone facts, recipe counts, safe-foot support checks,
  candidate/coordinate binding, the excavation geometry veto, action latches,
  block/inventory postconditions, table/furnace approach stands, and all
  cooldowns.  It never counts a navigation move as mining progress.
* **Jev** is not asked to choose routine craft or navigation.  It sees only a
  compact immediate-hazard menu and chooses among exact, current legal
  responses.  Code's lava, drowning, fall, low-health, and stale-observation
  rules override it.
* **Astra** is a slow coarse route adviser, launched only after about 55 seconds
  without an audited inventory/advancement change.  It chooses a mode token
  (`excavate`, `scout`, `food`, `bed`, `iron`, `shelter`, or `progress`) from a
  derived terrain summary.  The token can select whether to refresh terrain or
  prioritize the already implemented transaction, but Astra cannot emit a
  command, coordinate, recipe, or target.  A pending or unusable answer leaves
  the deterministic excavation policy in force.

## What changed in the harness

### 1. Verified block transactions

`ExcavationController` binds a current `b:x,y,z` record to one transaction:

1. choose an observed visible/in-reach target;
2. navigate to a listed nearby safe `s:x,y,z` stand (with a one-time block-goal
   fallback), rather than assuming a block itself is a safe feet position;
3. equip a tool if the transaction needs one;
4. `look_at` the recorded position and issue one direct `dig`;
5. for a valuable block, find an observed item entity or a safe stand and verify
   an authoritative inventory delta; for terrain, verify the target vanished;
6. suppress only the failed target/action after a timeout or explicit feedback.

The old target is never considered complete merely because the navigator moved.
A block directly below the player's current feet is vetoed.  Ground targets are
limited to the current surface layer, so the controller does not intentionally
dig straight down.  The same transaction machinery handles logs, stone, coal,
iron, exposed food blocks, and a visible bed.

### 2. Excavation is a new progression phase

After the wooden pick is present and before a stone pick is crafted, the
compiler first searches for exposed `stone`/`cobblestone` and then for safe
visible `grass_block`, dirt, gravel, sand, clay, or similar ground.  Removing a
side/floor block creates a new observation opportunity for the next layer.  It
does not use `controls forward,jump` to explore.  Once three cobblestone are
verified, the controller goes back to an observed table stand and crafts the
stone pick.  The same operation is reused later to expose iron and enough stone
for a furnace.  This is the critical order reversal: **mine/excavate first,
return to the table second**.

Safe stand exploration is selected only when no useful exposed target is
available.  It prefers fresh, distant listed stands and remembers stand
failures, but it never treats movement as a resource postcondition.  The
controller can therefore leave the initial candidate basin while keeping the
hazardous manual-control path as an emergency escape only.

### 3. Workstations are independent, hand-staged transactions

The table compiler accepts a table only when it is currently observed and near.
If a table is farther away it chooses a safe stand near that table, not a stale
`navigate b:` goal as its first choice.  A carried table is equipped before
placing, and a support directly under the player's feet is rejected.  Inventory
consumption plus a later `crafting_table` candidate latches a successful place;
a failed support is suppressed and another observed support or safe stand is
selected.  Furnace placement and approach use the same rule.  A lagging block
stream cannot cause a second table craft.

### 4. Recovery and survival

Every one-shot has a before-inventory/before-position latch and a short
postcondition deadline.  A changed feedback field, unchanged target, or missing
block change ends that transaction and records a bounded suppression.  A current
bounded job is continued with `noop`; a replacement is preceded by `cancel`.
Death calls ordinary `respawn` and drops the in-memory job rather than assuming
inventory restoration.  Low food/health, lava/fire, drowning, visible hostiles,
and a stale observation bypass the macro compiler.  The code avoids blind
manual movement and uses only observed entities for animal combat.

## System-1 interface, thresholds, and bridge

The one Jev request is made only when the current hazard signature is nonempty
or changes.  Its strict JSON state is:

```json
{
  "purpose": "immediate survival only",
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
    "shield on", "attack <observed entity id>",
    "equip <observed item> hand", "cancel", "noop"
  ],
  "rule": "choose one exact supplied action; never invent an id"
}
```

The batched typed questions are:

* `response: Choice` — choose one value from `legal_actions`.  The criteria
  describe each exact command as a survival response; no coordinate or recipe
  is generated.
* `safe_to_continue: Noul` — independently judge whether the current bounded
  work can continue without defense.  The controller records it but hard code
  still handles lava, drowning, and stale state.

A response is accepted only if it is a current menu member, matches the hazard
signature, and has probability at least `0.52`; an attack/retreat below that
threshold falls back to the deterministic shield/escape response.  While a
request is pending, the fallback is immediate: leave lava/fire, surface from
water, eat at health/food bands 7/5, attack a touching visible hostile only
with an observed weapon, raise a carried shield, or use `controls back,jump`.
A bounded task is cancelled before a replacement response.  Jev timeout,
malformed stand-in answers, stale values, and low probabilities are caught and
cannot escape `run_episode`; budget exhaustion and model-unavailable errors
retain the platform's normal termination semantics.  The request timeout is
1.5 seconds, comfortably above the measured warm Jev tail while preventing a
hazard from waiting for inference.

## Astra route interface and scheduling

On a no-progress interval (55 seconds since the last audited inventory or
advancement change, at least 250 wall seconds remaining, and at most one
outstanding call), Astra receives only derived facts: current code phase,
filtered inventory, semantic near/far exposed-block categories, and the number
of safe exploration rounds.  It must put one of these tokens on its first line:
`excavate`, `scout`, `food`, `bed`, `iron`, `shelter`, or `progress`.
The parser searches only that first line and discards all prose/coordinates.
There are no fixed seed locations in the prompt.  The default remains
transactional excavation when it is possible, so a 35–95 second Astra call
cannot freeze the real-time game.  A result is invalidated by a new phase or
left as a harmless preference; it is never treated as a game fact.

## Confirmation and refutation

The hypothesis is confirmed by a valid trace with these events in order on a
seed that previously plateaued: a verified wood/table opening, one or more
`transaction_begin` ground targets, `excavation_confirmed`, a newly exposed
stone target and `resource_confirmed`, then a table approach and a stone-pick
craft.  The important negative signals are absence of long repeated
`navigate b:<table>` while in the stone phase, absence of blind controls in
ordinary exploration, and no exact failed placement streak.  Leaderboard
confirmation is any increase above the 2.0 mean opening plateau, especially an
`iron_shield` component, without harness errors or stale ticks; seed 1 should
survive water instead of repeating `controls jump`.

It is refuted if safe excavation repeatedly removes blocks without exposing any
new candidate, if direct visible/in-reach digs fail despite the postcondition
latch, or if the controller still returns to the table before stone search.  A
run that reaches stone but not iron would confirm the immediate hypothesis and
identify the next bottleneck as the underground/Nether route, not as evidence
that the surface transaction failed.  A zero score with no unsafe manual
movement would instead indicate that safe-stand exploration cannot cross the
spawn terrain and that the next proposal should improve navigation/site choice,
not add more retries.
