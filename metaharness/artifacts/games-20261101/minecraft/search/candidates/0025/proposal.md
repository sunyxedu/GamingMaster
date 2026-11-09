# Candidate 0025 — make the wooden-pick stone phase a verified staircase transaction

## Parent and diagnosis

Parent: `candidates/0015/`.  I keep its architecture: code owns the milestone
compiler, inventory arithmetic, candidate legality, postconditions, failure
cooldowns, and safety overrides; a single asynchronous Jev request chooses from
a freshly compiled exact-action menu.  Astra remains unused because the failure
in the parent is a local transaction boundary, not a missing long-range plan.

The parent reaches the opening but never crosses from a wooden pick to stone.
The seed-0 trace is especially concrete:

* `table_place_observed` occurs at `10.003 s`, and
  `craft wooden_pickaxe 1 b:-23,68,30` succeeds at `27.004 s`;
* instead of starting a stone transaction, the controller posts
  `goto -22.00 68.00 31.00` in `locate_table` at `33.0 s` and repeatedly
  returns there (including posts near `98`, `106`, `171`, `179`, `250`,
  `330`, `422`, `502`, `584`, `650`, and `716 s`); the same coordinate is
  invalidated 11 times with `adapter_feedback`;
* the trace contains no stone gather, direct dig, cobblestone inventory delta,
  or stone-pick craft before the deadline.

Seed 2 shows the same failure later: the table is observed at about `132 s`,
the wooden pick is crafted at `146.5 s`, and the controller posts the stale
`goto -121.00 63.00 -44.00` table return at `152`, `218`, `284`, …, `815 s`,
with 12 adapter-feedback invalidations and no stone attempt.  Seed 1 is the
opening-frontier worst case: it spends hundreds of seconds surveying stands,
only acquires its first wood around `827 s`, and crafts its wooden pick around
`864 s`, leaving no time for the next phase.

The parent’s early workstation and feasible-recipe transactions are therefore
working; the immediate missing operation is a resource-acquisition transaction
that can open the observed terrain.  This diagnosis is also supported by the
archive’s later comparator: candidate 0019 recorded real
`stair_step_confirmed` events and three cobblestone acquisitions on seed 0,
whereas 0015 never posted a dig at all.  I do **not** add its later table or
portable-workstation rewrite here; this candidate isolates only the missing
wooden-pick-to-stone operation.

## Hypothesis and one targeted change

The stone phase is deadlocked because it only offers exposed `gather` targets
and, after a table placement, the cached workstation latch is checked before
that phase.  A code-owned **observed staircase transaction** should cross the
boundary without assuming an underground block or a return route:

1. when a wooden pick exists, fewer than three cobblestone are audited, and no
   stone pick exists, choose a listed adjacent, non-hazard ground block one
   block below the current feet level (never an underfoot support);
2. approach it, equip the wooden pick, look at it, and issue one direct `dig`;
3. require the listed block to disappear or an audited local block-change before
   proceeding, then use a bounded `goto` into the newly opened cell;
4. count a step only after the player actually enters that observed cell, and
   repeat with a newly compiled candidate until cobblestone reaches three; if
   no adjacent target remains, move to one listed safe stand and refresh the
   candidate window while staying in the stone phase;
5. only then allow the parent’s existing table-return and feasible
   `craft stone_pickaxe` path to run.

The resource branch is evaluated before the parent’s cached-table return branch
while this exact stone precondition is active.  Thus a stale table hint can no
longer prevent the first observed dig, but the table behavior after the real
cobblestone threshold is unchanged.  No portable table, new route heuristic,
seed fact, or hidden-coordinate search is added.  The change is one local
replacement of the wooden-pick stone phase, and it deliberately leaves the
known workstation handoff as the next separately measurable boundary.

## Implementation details and safety

`staircase_target` accepts only an adaptor-listed `b:` block adjacent to the
current feet cell, at the current level or one level below.  It rejects the
supporting block under the player, sand/gravel, fluids and known hazards, and
requires a harvestable, visible or in-reach candidate.  A `mine_job` state
machine has `approach`, `equip`, `look`, `dig`, `wait_dig`, and `move` stages.
It uses only listed `s:` stands or the listed block position for navigation;
unknown underground coordinates are never synthesized.  This candidate only
accepts floor blocks one level below the feet cell; it will not count
same-height wall cutting as staircase progress.  If no lower target is exposed,
it uses a listed safe stand to refresh the candidate window and waits for new
observations.  A timeout, missing postcondition, adapter rejection, death, or stale observation clears and cools
the exact target, allowing another observed candidate.  Inventory changes are
not treated as proof that a direct dig succeeded unless the block/step
postcondition is also satisfied.

After a successful step the normal parent compiler still owns recipe counts,
workstation IDs, and the return-to-table action.  Death still calls ordinary
`respawn` and clears the mine job without restoring dropped inventory.

## Jev state, questions, thresholds, and fallback

The state keeps the parent’s compact schema and adds the current transaction
stage:

```json
{
  "objective": "safe verified progression toward Free the End",
  "phase": "mine_stone|find_wood|make_table|place_table|wood_pick|...",
  "transaction": {"stage": "approach|equip|look|dig|move|none",
                   "target_kind": "listed ground block or none"},
  "dimension": "overworld|the_nether|the_end",
  "health": "critical|low|stable",
  "hunger": "hungry|fed",
  "inventory": {"wooden_pickaxe": 1, "cobblestone": 2},
  "threats": [{"kind": "zombie", "range": "melee|near", "id": "observed id"}],
  "options": [{"command": "exact compiled action", "purpose": "code description"}],
  "recent_failures": ["normalized exact action keys"]
}
```

As in 0015, one Jev request batches:

* `action: Choice` over the exact current option strings.  The value is
  accepted only if it is in the current menu, from the same episode, no more
  than 10 observations old, and its returned probability for that value is at
  least `0.55`.
* `safe: Noul`, asking whether no visible close hostile makes the compiled
  work unsafe.  A false probability (`safe.value < 0.35`) rejects the Jev
  action and uses the first code-legal option.

The request is launched with `ctx.spawn` and a `2.0 s` timeout.  While it is
pending, the controller posts `noop`; after `0.8 s` it commits the first legal
compiled option so real-time physics keeps advancing.  Late, stale, malformed,
unknown, low-probability, or arbitrary pre-flight answers are discarded.  The
first legal option (or `noop`) is the fallback.  Code always overrides Jev for
death, lava/fire, drowning, critical hunger/health, stale observations, active
jobs, and illegal IDs.  `BudgetExceeded` and `ModelUnavailable` retain the
platform’s normal episode termination behavior; ordinary model/parse failures
are contained.

## Confirmation or refutation

Confirmation is a valid zero-error trace in which a wooden pick is followed by
`mine_transaction_begin`, a real `dig`, a disappearance/block-change
postcondition, several `stair_step_confirmed` events, and an audited
cobblestone/`minecraft:story/mine_stone` transition **before** repeated table
returns.  A subsequent feasible stone-pick craft or any iron/shield progress
would show that the next handoff also works, but is not required to confirm
this isolated resource hypothesis.

The hypothesis is refuted if the controller still spends the wooden-pick phase
in stale table `goto`s without a dig, or if observed adjacent targets are
repeatedly rejected despite the staged dig/move postconditions.  If it produces
verified cobblestone but then stalls at the unchanged table return, that is a
positive result for this candidate’s single change and identifies the next
boundary rather than refuting the staircase hypothesis.  If seed 1 remains at
zero because no wood is exposed, that is the parent’s separate opening-frontier
limitation, not evidence against the stone transaction.

The latest five-minute live smoke test on seed 0 ended without a harness error
at score 3, but did **not** exercise the new mine branch: after the opening
table, its additional wood gather did not change inventory, and the parent’s
cached-table return then repeated.  This is inconclusive for the hypothesis
because the wooden-pick precondition was never reached.  The earlier smoke run with the initial horizontal-target selector did exercise
the transaction (27 verified entered cells) but mostly cut a same-height
surface path, reaching only one block below its first mining level; that is why
the final candidate rejects same-height cuts and only accepts a one-block
descent.
