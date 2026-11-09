# 0008 — Failure-driven frontier controller

## Parent and isolated hypothesis

Parent: `candidates/0005`, the best evaluated candidate (mean 2.0). I also used
`0004` as a diagnostic comparison, not as an implementation parent.

The best candidate is not primarily losing because Jev lacks a long-term plan.
It is losing because the controller does not close the loop around bounded jobs:
its target remains eligible after a gather/navigation failure, and its exploration
memory is keyed to transient candidate IDs rather than a spatial frontier.

Concrete trace evidence:

* On seed 1, the parent posted `gather b:-3,64,-25 3` at 0.5 seconds even
  though the initial oak candidates were `visible: false` and `in_reach: false`.
  It then retried that same hidden tree repeatedly (including at 7.0, 8.5,
  13.5, 20.0, 21.5, and throughout the episode), ending with an empty
  inventory and score 0.0 at the deadline.
* On seed 2, after finding wood and a table, it spent roughly 323--708 seconds
  navigating among nearby stands around `-200,-70`, with no stone, shield, food,
  or bed milestone. It then took stray damage and died several times. The
  final inventory was empty and the score was only 3.0.
* On seed 0, it earned the table/log opening points but finished with only two
  logs, planks, sticks, and a wooden pick. It repeatedly crafted planks and
  placed additional tables rather than observing a failed target and advancing
  to stone/iron.

`0005` does not read the adaptor's canonical `action_feedback` field at all.
`0004` has a failure counter, but its feedback probe is `action_result`/`feedback`
and its local visit/cooldown bookkeeping does not normalize the current
`action_feedback` plus spatially equivalent gather targets. The traces therefore
show a missing observation-to-invalidation path, not simply an insufficient
number of Jev calls. The parent also spent 11 Astra calls without escaping this
local loop.

**Hypothesis:** normalize all available action-result fields, invalidate a
specific observed target after an explicit failure or a no-progress deadline,
and select a genuinely novel safe stand as a frontier when the local resource
menu is empty. This will convert hidden/stale target churn into exploration and
let the existing recipe chain reach stone, iron, and defense. A broader threat
override should then prevent the long scout from ending in a stray death.

## What changed

This is a new closed-loop controller rather than a copy of the parent policy.

1. Python owns a small progression state machine: exposed wood/table, wooden
   pick, stone pick/weapon, exposed iron/fuel, furnace/shield, food/bed, then
   local resource exploration. Recipes and counts are computed in code.
2. `action_feedback`, `action_result`, `last_result`, and `feedback` are
   normalized into one change-detected marker. Failure words invalidate the
   active command. If a bounded gather/navigation/craft job has no inventory,
   movement, or dimension change by its safe deadline, it is also invalidated.
   Gather failures are keyed by `gather <block-id>` rather than the requested
   count, so changing 1 to 2 cannot retry the same bad block.
3. Resource gathering only accepts an exposed visible or explicitly in-reach
   candidate. A failed block is suppressed for a long cooldown. This directly
   tests the seed-1 hidden-tree failure instead of repeatedly asking Jev about
   it.
4. Navigation records quantized spatial cells and candidate cooldowns. It
   prefers the farthest novel safe stand, then permits a cooled fallback when
   all nearby stands have been seen. This is a frontier policy, not a dictionary
   of ephemeral IDs.
5. The active bounded job owns the hands until progress, failure, or deadline.
   A fresh `noop` is posted while a task is busy, and identical one-shots get an
   explicit noop re-arm. Thus a lagging `task` field cannot overwrite a newly
   issued command.
6. Code handles low health, lava/water, visible ranged threats and close
   hostiles before the normal menu. It uses a real weapon when possible, a
   shield when available, or retreats to a stand. Normal death uses ordinary
   respawn; no inventory restoration or privileged information is used.

I intentionally use **no Astra calls** in this candidate. The test is whether
correct observation/action bookkeeping is the missing structural component;
`0005` already showed that sparse Astra priority calls did not repair the
repeated-target loop. Removing that confound also preserves the 40-call budget
for a later candidate if this controller reaches a meaningful strategic
milestone.

## System-1 interface

Jev is a tactical selector, not a planner or parser. Each request receives a
strict JSON state with:

* `phase`: one code-owned phase such as `find_wood`, `stone_pick`, or
  `resource_frontier`;
* health and hunger bands, dimension, and a filtered audited inventory;
* visible hostile entities with only kind, melee/near range bucket, and ID;
* `options`: at most seven complete, currently legal command strings paired with
  their code-derived purpose;
* the last few normalized failed-command keys; and
* the rule that no coordinates, IDs, verbs, or counts may be invented.

Two independent typed questions are batched:

* `action` is a `Choice` over the exact command strings. A returned value is
  accepted only if it is still a member of the newly compiled menu, the
  episode/sequence is fresh (at most ten snapshots), and its probability is at
  least 0.55.
* `safe` is a `Noul` about whether the supplied state has a visible close
  hostile threat. It is advisory for Jev-selected work; code's emergency
  override is authoritative. A false/uncertain answer never licenses an
  uncompiled action.

The state is intentionally small: Jev does not see the full collision dump,
raw coordinates, recipe arithmetic, or stale hidden candidates. Code computes
those facts and produces semantic descriptions. The answer is discarded on
unknown values, stale sequence/episode, malformed return objects, timeout, or
low probability. `BudgetExceeded` and `ModelUnavailable` propagate normally;
ordinary timeout/runtime/type failures fall back safely.

## Scheduling and fallbacks

There is at most one background Jev request. It has a two-second timeout and
runs through `ctx.spawn`; while it is pending the harness posts `noop`, and
uses the first code-compiled option after 0.8 seconds rather than freezing the
world. A completed late answer is harvested but cannot replace an already
posted action. If System-1 is nearly exhausted, the code policy runs alone.
Every normal tick posts either a continuing noop, an emergency action, or a
freshly compiled complete command, so real-time stale ticks remain bounded.

Hard safety gates precede Jev: dead means `respawn`; stale observations cancel
jobs; active tasks are continued; low health eats carried food; lava/water use
movement controls; and visible threats can cancel work, shield, attack, or
retreat. A command is revalidated against the current option menu immediately
before posting. No controls movement is used for ordinary exploration.

## Confirmation and refutation

The hypothesis is confirmed if traces show `action_invalidated` events for
failed/unchanged targets, no long exact-target retry streaks, increasing
frontier cells, and accepted inventory deltas after exploration. Performance
confirmation is positive opening progress on the swamp/snow training cases,
then stone/iron/shield or bed points without an invalid trial; stale ticks and
System-2 calls should be no worse than `0005`.

It is refuted if visible/in-reach commands still fail without useful feedback,
if spatial frontier navigation merely oscillates despite target invalidation, or
if the controller reaches the same opening score while spending its time in
craft/navigation loops. A positive opening but no Nether progress would show
that this repair fixed local execution but not the later strategy; it would not
be evidence that the dragon route is solved.
