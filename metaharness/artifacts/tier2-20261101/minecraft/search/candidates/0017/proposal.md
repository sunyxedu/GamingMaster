# 0017 — Renew the expedition lease, not the failed stand

## Parents and hypothesis

Parents: `candidates/0008/` and `candidates/0004/`.

The useful mechanism in 0004 is its **compiled progression chain**: code knows
that planks precede a table, a table and sticks precede picks, and a real furnace
must be loaded and waited on.  It also supplied a small, validated Astra macro
enum rather than asking a model to emit Minecraft syntax.  The useful mechanism
in 0008 is **failure-driven bookkeeping**: it normalizes action feedback, keeps
bounded jobs from being overwritten, and treats model choices as choices among
complete commands.  I retain both, but do not retain their stand-only resource
search.

The concrete failure is visible in both parents' traces.  In 0008 seed 0,
`place crafting_table b:-22,67,31 up` was posted at 9.5, 36.0, 62.5, ... and
invalidated with `adapter_feedback` every 25 seconds through 876 seconds.  The
placement target was not put in a permanent support failure ledger.  In seed 1,
`gather b:-12,64,-21 1` was retried at 24.5, 291.0, 563.0 and 828.5 after the
controller had already learned that the swamp tree could not be gathered.  In
seed 2, the same controller spent the first 84 seconds walking a sequence of
safe stands before finally acquiring a log, then returned to placement/crafting
loops.  0004 had the same underlying problem at higher inference cost: seed 0
selected `craft acacia_planks 8` for dozens of consecutive seconds, while seed 1
spent the episode on the inaccessible tree.  Its 12 Astra calls did not change
that outcome.

The lesson is not merely “quarantine more IDs.”  A listed safe stand is not a
resource action, and a real-time `controls` lease expires after 0.8 seconds.
Previous frontier/manual experiments used `controls forward,jump` and then
`noop` for several seconds; that silently stopped movement.  My hypothesis is
that the missing transition is an **active, lease-aware survey primitive**:
renew movement controls on every 0.5-second tick for a short cardinal segment,
stop as soon as an exposed resource appears, and bind an explicit
approach/equip/look/dig/pickup transaction to that fresh block.  If the survey
finds a stone or iron exposure, the verified dependency chain can continue
instead of counting navigation as progress.  A support block that rejects a
placement is quarantined by its reference ID, so a workstation failure cannot
monopolize the run.

This is a structural crossover rather than a parameter change: stand navigation
is no longer the exploration primitive, and no movement or feedback marker is
accepted as a mining postcondition.

## What changed

* A code-owned milestone executor covers opening wood/table, wooden and stone
  tools, a weapon, food/wool/bed opportunities, exposed stone/coal/iron,
  furnace timing, shield crafting, and an optional already-active Nether portal.
  Counts, recipe outputs, table IDs, tool requirements, support geometry, and
  furnace-ready time are computed in Python.
* Resource acquisition is a transaction: bind a currently visible block,
  navigate until it is in reach, equip the required tool, look at it, `dig` one
  block, then walk to an observed drop.  Only an audited inventory delta in the
  expected drop family completes it.  A moved player, a nonempty feedback
  string, or a successful `navigate` does not.
* The survey uses `look_at` once and then posts the same `controls forward,jump`
  command freshly on every tick for at most ten seconds.  It rotates through
  eight seed-independent horizontal directions, cancels the lease when a
  relevant visible candidate appears, and has explicit lava, drowning, fall,
  hostile, health and death overrides.  This is deliberately bounded rather
  than an uninterruptible movement autopilot.
* Placement is staged (`equip` then `place`) and its expected block is checked
  in the current candidates.  A failed place records the reference support as
  bad as well as recording the full command; a later support must be a different
  observed block.  Bounded jobs get action-specific postcondition deadlines.
* The 0004 macro idea remains, but Astra is advisory and bounded.  After 35
  seconds without an audited inventory/advancement change, at most one short
  background request chooses `food`, `bed`, `stone`, `iron`, `explore`, or
  `balanced`.  A choice can cause only one 60-second survey detour; it cannot
  invent a command, bypass prerequisites, or veto safety code.  This tests the
  parent mechanism without allowing a 40-second answer to freeze the player.
* Jev is reserved for immediate threats, where it can recognize the tactical
  response but cannot invent an entity ID or coordinate.  Deterministic safety
  code remains authoritative and runs while Jev is pending.

## System-1 interface and control policy

There is at most one background Jev request.  Its JSON state is deliberately
small and derived:

```json
{
  "purpose": "immediate Minecraft survival interrupt",
  "self": {
    "health_band": "critical|stable",
    "dimension": "overworld|the_nether|the_end",
    "lava_fire": false
  },
  "threats": [
    {"kind": "creeper|ranged|blaze|lava_or_fire|drowning|...",
     "range": "touching|near"}
  ],
  "legal_actions": [
    "controls back,jump", "shield on", "attack <observed id>", "cancel", "noop"
  ],
  "rule": "choose one exact supplied action; never invent an id"
}
```

The threat request batches two independent typed questions:

1. `response: Choice` selects exactly one of the compiled `legal_actions`.
2. `continue_safe: Noul` judges whether current work can continue.  It is logged
   as evidence but never overrides code's lava/drowning/death veto.

The answer is accepted only if its value is still in the current legal menu, its
winning probability is at least `0.55`, the threat signature (kind, observed ID,
and touching/near band) is unchanged, and it is less than two seconds old.
Uncertainty, an invalid value, timeout, stale signature, or a failed call uses
the deterministic fallback: leave lava/fire, surface from drowning, cancel a
job, attack a close non-ranged hostile with an observed weapon, raise an observed
shield, or back-jump.  A threat response is revalidated every loop.  There is no
request for arithmetic, target discovery, or long-term planning.

The state sent to Astra is separate and includes the current code phase, audited
inventory, a short list of visible block names/distance/reachability, recent
failed target keys and recent commands.  Astra must return only the six-value
focus enum.  Invalid or arbitrary prose is ignored; the previous focus remains.
The focus expires after 60 seconds and is used only when no immediate local food,
animal, stone, or ore transaction is available.

## Freshness, fallback, and accounting

The loop first processes audited progress and feedback, then checks death,
hazards, and observation age.  A live bounded action receives `noop` until its
postcondition or its deadline is known.  A survey instead receives a fresh
controls command every tick; it never receives noop as a substitute for a
controls renewal.  Deliberate repeated one-shot commands get a noop barrier.
Every answer is attached to the current threat signature; late answers are
harvested but cannot change a newer state.  Model timeouts and malformed text are
ordinary fallback events.  `BudgetExceeded` and `ModelUnavailable` retain their
contractual behavior, while other inference parsing errors are caught.

The harness logs `mine_begin`, `mine_confirmed`, `target_quarantined`,
`survey_begin`, `jev_threat_choice`, and `astra_focus` so action selection can be
compared with the code default.  It uses no seed, hidden terrain, commands,
console, inventory fabrication, or unobserved coordinates.

## Confirmation and refutation

The hypothesis is confirmed if traces show renewed controls during surveys,
new block candidates after a survey, `mine_confirmed` inventory deltas for stone
or iron, and no long exact-target placement/gather streaks.  A positive result
would be a reproducible improvement from the current opening plateau to food,
bed, stone, iron or shield milestones without increasing stale ticks or trial
errors.

It is refuted if controls renewal still leaves the player stationary or causes
unrecoverable deaths, if visible/in-reach direct digs do not produce audited
pickup deltas, or if the controller reaches the same wood/table plateau while
spending its time in survey cycles.  If stone/iron succeeds but the run still
stops before Nether progress, that would confirm the execution hypothesis only
and identify the next missing structure as portal/fortress planning rather than
another frontier tweak.
