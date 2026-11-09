# 0018 — Separate observation-window expeditions from verified transactions

## Parent and hypothesis

Parent: `candidates/0015` (with `0016` and `0017` as diagnostic comparators).
This is a structural rewrite, not a recipe-count tweak.  The parent proved that
feasible recipe compilation is useful locally: seed 0 of 0015 reached
`craft wooden_pickaxe 1 b:-23,68,30` at about 26 s and all three episodes earned
the opening score.  It did not prove that a block-candidate list is a usable
stone route.

The concrete hypothesis is:

> The missing transition is an ownership problem.  A candidate ID is an
> interaction opportunity, not a navigation route.  When no useful block is in
> the current window, repeatedly selecting safe stands (or returning to a
> workstation coordinate) consumes the episode without changing the window in a
> controlled way.  The controller needs a separate, time-bounded directional
> expedition which renews movement controls, and must return to a transaction
> compiler as soon as an observation-grounded resource appears.

The parent trace makes this failure unusually clear.  In 0015 seed 0, after the
wooden pick at 27.0 s, the controller entered `phase: stone_pick` and its Jev
menu contained only `navigate s:...` options.  It posted 83 navigations and 29
`goto -22.00 68.00 31.00` returns by the deadline, with no stone or iron
postcondition.  The same trace records navigation feedback invalidations at
33.0, 106.0, 184.0, 264.5, 352.5, and later times.  The action compiler had
correct arithmetic but no action that could expose the next layer.

The comparator traces distinguish two related mistakes.  0016 did verify many
surface removals (`excavation_confirmed` at 54.0, 69.5, 118.5, 125.0 s and
later), but its seed-0 run spent 109 `dig`/`look_at` transactions without a
cobblestone postcondition; it treated removal itself as the route rather than
making a bounded resource-or-exploration decision.  On its seed-1 run it made
119 navigation posts and retried `gather b:-12,64,-21` without inventory change.
0017 renewed controls, but seed 1 still logged 1,617 `survey_begin` events
without a stone transition.  Thus neither “more stands” nor “remove arbitrary
ground forever” is sufficient.

0018 tests whether a two-level controller fixes that boundary: code executes a
small, auditable interaction transaction when the current observation supports
one; a separate expedition changes the observation window when it does not.

## What changed

### Code owns exact local work

`compile_options` is a dependency compiler with real postconditions.  It owns
inventory parsing, feasible recipe counts, phase transitions, support-block
selection, exposure safety, action latching, feedback cooldowns, and ordinary
respawn.  It never asks a model to count, calculate a coordinate, or invent an
ID.

The local phases are `wood`, `table`, `wood_tools`, `stone`, `iron`, `smelt`,
`shield`, and `expedition`.  In particular:

* a log is gathered only from a current exposed candidate;
* planks, sticks, tables, picks, furnaces, and shields are requested only when
  the authoritative personal inventory makes the recipe legal;
* workstation placement is staged as equip → place and is latched until the
  table appears or the placed item is observed consumed;
* an exposed stone block is gathered directly, while a safe adjacent dirt/sand
  face is a finite `look_at` → `dig` exposure transaction; the block under the
  player's feet is vetoed;
* a gather/dig is complete only on an inventory delta or an observed block
  change.  Navigation progress alone never completes a resource job;
* the stone exposure fallback is capped at six safe face transactions before
  the controller must launch a directional expedition.  This explicit cap
  prevents 0016's verified-but-unproductive 109-dig loop from returning under a
  new name; after an expedition the cap resets and the newly loaded window is
  checked again.
* every failed target gets a cooldown and an observation re-read, rather than
  an immediate identical retry.

### Expeditions own movement, not transactions

When the compiler has no current legal resource/workstation action, 0018 starts
one `expedition` with a compass bearing.  It aims once and renews
`controls forward,jump` (and sprint only with a food reserve) for at most 13
seconds.  It then stops and re-enters compilation.  A newly visible target
cancels the expedition with `controls none` before another movement leg.  The
fallback rotates east/south/west/north; it does not use a fixed seed or a hidden
map.  This is deliberately different from selecting a succession of `s:`
stands: movement is continuous enough to change the loaded terrain window, but
is still bounded and interruptible.

### Division of labour

* **Python:** exact facts, recipe arithmetic, phase/dependency order,
  candidate validation, safe excavation, cooldowns, postconditions, hazard
  vetoes, movement leases, and action formatting.
* **Jev:** a fast choice among at most seven exact, freshly compiled local
  transaction strings.  It can prefer food versus an already legal resource
  action, but cannot author one.  For high-stakes emergency states it answers
  two identical-choice questions with reversed option order; code accepts a
  response only when both values agree and the first option probability is at
  least 0.45.  A late/uncertain answer uses the highest-priority compiled
  option, after Python revalidates it.
* **Astra:** a sparse macro adviser invoked only after a phase change or at
  least 45 s without verified progress, with a 120 s call spacing and a
  105 s background timeout.  It chooses a short-lived coarse expedition mode
  (`cave`, `surface`, `food`, `shelter`, `portal`, or `progress`) and compass
  bearing.  It cannot emit commands, coordinates, IDs, or recipe counts.  The
  bearing changes the next bounded expedition; local code still decides
  whether any observed target is legal.  Invalid, late, or empty Astra text
  leaves the deterministic rotation policy intact.

## Jev state schema and questions

Every local-menu request receives a small JSON state, rebuilt from the latest
observation:

```json
{
  "objective": "Free the End through the next verified survival milestone",
  "phase": "stone",
  "dimension": "overworld",
  "health_band": "stable|low|critical",
  "food_band": "fed|low",
  "inventory": {"wooden_pickaxe": 1, "stick": 2, "cobblestone": 0},
  "visible_threats": [{"kind": "skeleton", "range": "near"}],
  "options": [{"command": "gather b:x,y,z 1", "purpose": "..."}],
  "rule": "Return only an exact command from options"
}
```

The normal request batches three independent typed judgments:

1. `action`: a `Choice` over the exact compiled commands.
2. `action_check`: the same `Choice` with criteria reversed, reducing the
   provider's first-option bias.
3. `continue`: a `Noul` about whether the highest-priority compiled
   transaction is safe from the supplied local facts.

The harness accepts `action` only if it is in the current menu, the episode and
sequence are fresh (no more than eight snapshots old), it agrees with
`action_check`, and its returned probability is at least 0.45.  The Noul is
informational because Python has hard hazard and legality vetoes; it never makes
an illegal action executable.  While Jev is in flight, the first compiled
option is the fallback after 0.45 s, so a slow request cannot stall the world.
Malformed stand-in answers, missing probabilities, unknown strings, timeouts,
and stale answers fall back to that option without escaping `run_episode`.

Emergency requests use the same compact hazard schema but only legal responses
such as `cancel`, `shield on`, `attack <visible-id>`, `controls back,jump`, or
`noop`.  Environmental hazards and critical hunger/health are handled before
Jev.  A task or transaction is cancelled before a replacement combat action;
otherwise the deterministic escape/food policy runs immediately.  Jev is
consulted only to choose between supplied emergency responses, and must agree
across the reversed pair at the same 0.45 threshold.

## Safety and late-answer policy

The controller always waits on the real-time tick and posts `noop` while a
bounded job owns the hands.  It cancels before replacing navigation, digging,
or item use.  Observation age over three seconds cancels a task or stays still.
Lava/fire, drowning, death, hunger, visible close hostiles, and missing held
items are code-level checks.  Normal death clears the transaction and uses the
ordinary `respawn` action; inventory is not restored.

A background Astra answer is never allowed to replace a current job.  A
background Jev answer is tagged with the menu's episode/sequence and hazard
signature.  It is discarded when the world has advanced, a target disappeared,
or the hazard set changed.  The fallback is always a legal current action or
`noop`; no stale model output is posted.

## Confirmation or refutation

The hypothesis is supported by a zero-error run in which, after the wooden
opening, at least one of the following occurs without a long workstation-goto
loop:

1. a bounded expedition changes position and exposes a stone/cobblestone row;
2. the code completes a gather or safe exposure transaction and observes an
   inventory/block postcondition;
3. the verified cobblestone count crosses three and a stone pick is crafted;
4. subsequent iron, furnace, or shield progress is reached.

The strongest confirmation is a trace with a finite number of `expedition_begin`
/`expedition_end` pairs, a `resource_confirmed` or `excavation_confirmed` event,
and fewer than a handful of repeated navigation invalidations.  It is refuted
if the bot still cycles on one workstation coordinate, if continuous expeditions
never change the candidate window on seeds where surface movement should reveal
resources, or if safe exposure repeatedly removes blocks without ever checking
for the newly exposed block.  If it reaches stone but not Nether progression,
that confirms this candidate's targeted boundary while leaving later route
planning as the next experiment.
