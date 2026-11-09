# Candidate 0020 — make the first mine a bounded staircase

## Parent and hypothesis

Parent: `candidates/0016/`, with `0015` as the best-score comparator.  This is a
new structural test, not another recipe-count or table-placement repair.

The hypothesis is that the remaining plateau is caused by treating a surface
block list as a mine.  Candidate 0016 did verify its local opening, but its
post-opening transaction was still a sequence of isolated surface removals.
For example, its seed-0 trace has a wooden opening followed by
`transaction_begin ... excavate grass_block b:-23,67,32` at about 25.5 s,
then `dig -23 67 32`; the action was suppressed at about 37.5 s for
`no_postcondition`.  It then tried nearby grass blocks repeatedly.  The trace
contains several `excavation_confirmed` events (for example around 54, 118,
and 125 s), but no `resource_confirmed` stone event, and the final inventory
still contains only a wooden pick.  The generation summary reports **51+28
verified ground removals and zero stone/iron confirmations** for 0016.

The best candidate, 0015, fails at the same boundary in a different shape: its
seed-0 final observation has sticks and a few planks but no cobblestone, iron,
or shield, after the opening.  Its downstream policy can return to a table or
safe-stand survey, but it has no explicit operation that enters a newly
opened lower cell.  Thus the controller can prove that it changed terrain
without making the changed terrain useful.

I will test the stronger claim that the first progression transition requires
an explicit, small geometry protocol: remove one observed adjacent floor block,
verify it vanished, path into that new one-block-lower feet cell, and repeat a
limited number of times.  Once inside, clear only an observed two-block-high
forward branch.  This supplies the wooden pick with a real route to exposed
stone while retaining the adaptor's safety constraints.  It does not assume
that an underground block or ore is present: if no suitable candidate is
exposed, the bot goes back to a safe-stand observation refresh.

## What changed

The 0016 surface excavation loop is replaced by a finite transaction machine:

* `choose_surface_step` accepts only an exposed/in-reach ground or stone
  floor block adjacent to the current feet cell.  It rejects hazards and the
  support block under the player, and checks the supplied local-collision
  label when one is available.  Stone is included only as a descent surface
  once the appropriate pick can make the transition.
* A `step` transaction binds the exact candidate and records the inventory,
  position, and block postcondition.  It equips the best observed pick, aims,
  issues one direct `dig`, verifies the target disappeared, then sends `goto`
  to the new feet cell.  It records `stair_step_confirmed` only after the
  player actually reaches that cell.  A failed dig or path is suppressed by
  target and never counted as mining progress.
* After a step, the player can be in a lower layer.  `choose_tunnel_block`
  selects only an observed block in the next forward foot/head pair.  A
  `tunnel` transaction clears one block, enters it, and ends; the next
  observation selects the next wall block.  No guessed underground
  coordinates and no blind `controls forward,jump` exploration are used.
* The stone phase has an explicit priority: exposed stone/cobblestone first,
  then a verified staircase step, then an observed forward branch, then a
  bounded safe-stand refresh.  It does not return to the table until three
  cobblestone are in the authoritative personal inventory.
* The same transaction/postcondition and recovery machinery is reused for
  wood, table placement, stone, furnace, iron, smelting, and shield.  Recipe
  actions are emitted only for the phase's actual observed inputs; a changed
  inventory, vanished target, or reached feet position is the only success
  signal.  Stale feedback, unchanged targets, action timeouts, death, and
  unknown model values fall back without escaping the episode.
* A small opportunistic food reserve is compiled only after a pick exists and
  only from a currently exposed food block.  If no such candidate exists, the
  mine is not diverted into guessed animal or structure logic.
* The opening is retained as a deterministic compiler: exposed wood is
  gathered, planks/table/sticks/tools are crafted, and the table is explicitly
  equipped before placement.  This preserves 0015's confirmed score-3 opening
  while making the post-wooden-pick route materially different.

There is no seed-specific coordinate, structure oracle, command, or privileged
state.  `goto` is used only with current numeric positions from safe-stand or
newly exposed observations; `dig` is used only with a current candidate bound
to the transaction.

## System-1 interface and real-time bridge

Jev is reserved for immediate threat recognition, rather than routine mining
or recipe selection.  The state sent on each new hazard signature is a strict,
small JSON object:

```json
{
  "purpose": "immediate survival only",
  "self": {
    "health_band": "stable|low|critical",
    "dimension": "overworld|the_nether|the_end",
    "in_lava": false
  },
  "hazards": [
    {"kind": "zombie|skeleton|creeper|...|lava_or_fire|drowning",
     "range": "touching|near"}
  ],
  "legal_actions": [
    "equip <observed item> hand", "attack <observed entity id>",
    "shield on", "controls back,jump", "cancel", "noop"
  ],
  "rule": "choose exactly one supplied action; never invent an id"
}
```

The request batches two atomic questions:

* `response` is a `Choice` over the exact `legal_actions`.  The code checks
  membership in the current menu, and accepts it only when the returned
  probability for that value is at least **0.48**.  This is a cheap tactical
  choice, not a request to plan a mine or calculate a route.
* `continue` is a `Noul` asking whether the current work can continue without
  defense.  It is deliberately advisory and is batched for calibration; hard
  code still interrupts for lava/fire, drowning, low health with available
  food, stale observations, and a bounded task in the presence of a visible
  threat.

The hard fallback is: escape lava/fire or drowning immediately; eat when
health is at most 7 or food at most 5 if food exists; cancel an owned job when
an immediate visible hostile is present; otherwise equip an observed weapon,
attack a touching observed hostile, raise an observed shield, or retreat with
`controls back,jump`.  A synchronous Jev call has a **1.2 s** timeout, above
normal warm latency but short enough that the real-time fallback remains
active.  Timeout, malformed stand-in answers, stale/unrecognized values, and
low probability select the first legal fallback.  `BudgetExceeded` and
`ModelUnavailable` retain the platform's normal episode termination semantics;
ordinary inference errors are caught.  The harness never lets Jev create an
argument, coordinate, item count, or action outside the current menu.

Every action is revalidated against the latest observation.  A bounded job is
continued only with `noop`; replacing it first uses `cancel`.  One-shot
repeats are separated by a `noop` barrier.  The loop waits for each real-time
tick, and all model state is discarded at the end of the episode.

## Confirmation or refutation

The hypothesis is confirmed by a valid, stale-free trace on a seed that
previously reached a wooden pick but no stone containing, in order:

1. `stair_step_begin` followed by `stair_step_confirmed` for at least one
   actual lower feet position;
2. `branch_begin`/`branch_entered` or a direct exposed-stone transaction;
3. an authoritative `gather_confirmed` for cobblestone and a stone-pick craft;
4. fewer repeated `no_postcondition` surface removals and no long table-or-
   survey loop before the stone milestone.

A leaderboard improvement above the 0015 mean of 3.0, especially any
`iron_shield` points, is the strongest confirmation.  A lower but clean trace
that reaches cobblestone would confirm this immediate bottleneck and identify
iron/furnace acquisition as the next experiment.

It is refuted if the controller reaches lower feet cells and still never
obtains cobblestone despite observed reachable stone, if `goto` cannot safely
enter its own verified stair opening, or if all runs remain at opening 3 with
no stair/branch confirmations.  If no training seed exposes a valid adjacent
step, that refutes the chosen geometry as an effective interface but does not
show that surface excavation was sufficient; the next proposal should then
focus on safe navigation/biome-dependent resource acquisition rather than
adding retries.
