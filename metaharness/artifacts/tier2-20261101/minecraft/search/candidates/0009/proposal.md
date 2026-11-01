# Candidate 0009 — Spatial transactions with a macro/tactical crossover

## Parents and diagnosis

This is a crossover of `candidates/0007/` and `candidates/0008/`.

The useful mechanism from 0007 is its **transactional compiled-action menu**: code owns recipes and prerequisites, Jev selects only complete commands, and a bounded action is allowed to finish before another hand/movement command is posted. That mechanism earned opening progress on seed 0 (`score 3.0`) and seed 2 (`score 3.0`) even though the candidate averaged only 1.0. The useful mechanism from 0008 is its **normalized feedback, spatial failure memory, and frontier idea**: failed work is noticed without trusting a single feedback field, and the controller does not need to ask Jev to invent coordinates.

The strongest concrete evidence is that 0008 fixed one failure mode but exposed a deeper one:

* On seed 0 it gathered a log and crafted the table by about 10 seconds, then posted exactly `place crafting_table b:-22,67,31 up` at roughly 9.5, 36, 62.5, ... seconds. The adaptor reported `adapter_feedback` invalidation each time, and the same support/target was retried 34 times through the deadline. The controller's 25-second cooldown was on the command, not a new placement geometry; when the only generated placement option was rebuilt, it selected the same bad support again.
* Seed 2 shows the same class of error at a different location: `place crafting_table b:-113,66,-64 up` was invalidated and then retried about 30 times. Thus exact-string cooldown is not enough when the planner regenerates the same spatial goal.
* On seed 1, 0008 did avoid blindly gathering the initially hidden oak, but it spent the run cycling `navigate s:...` stands and retrying `gather b:-12,64,-21 1`; it never converted exploration into a reliable wood acquisition. The 0007 trace is the complementary failure: it treated movement as progress and then returned to `gather b:-3,64,-25 3` 631 times, even though the target remained out of reach/occluded.
* The trace shows why merely adding more Jev or Astra is unlikely to fix this. 0007's seed 2 rejected ordinary work behind a Jev safety gate for 418 calls, while 0008 used no Astra and still spent its best seeds in one invalid placement loop. Both candidates ended with zero iron, Nether, eyes, End, and dragon components; the immediate bottleneck is still reliable local execution after the opening.

## Hypothesis

The next missing structure is a **two-stage spatial transaction**. A block operation should not be represented as “place on this observed support” or “gather this candidate” alone. It should be:

1. select a currently observed resource or support;
2. select an observed safe feet position that approaches that goal without occupying the destination cell;
3. move there with a coordinate `goto`, avoiding an ephemeral candidate-ID navigation failure;
4. re-observe and compile the final block-reference command from the new snapshot;
5. accept it only if its inventory/block/advancement postcondition is visible; otherwise quarantine the spatial goal and choose another support, stand, or resource cluster.

I predict this will turn 0008's opening score into actual tool progression. In particular, a table will no longer be repeatedly placed into a cell occupied by the player or a replaceable block, and a hidden tree will receive a bounded number of goal-oriented approaches before the controller moves to a genuinely different wood/frontier cell. The longer route is still conservative: it does not invent underground terrain, structure coordinates, or a seed.

## What changed

This is a new controller, not a copy of either parent loop.

* **Spatial placement transaction.** `placement_plan` filters observed solid supports, checks the cell above for a known solid obstruction, checks local collision labels, and chooses a safe stand at a nonzero distance from the destination. It emits `goto` first when the current player cell is too close; only a fresh observation can emit `place ITEM SUPPORT up`. A failed placement is keyed by item, support coordinate, destination face, and a long spatial cooldown, so rebuilding the same command cannot revive it.
* **Coordinate staging instead of stale candidate IDs.** Resource and workstation approaches use coordinates copied from current `s:x,y,z` positions and validated to be within 48 blocks. The controller revalidates the block candidate after arriving. This directly tests whether `goto` is more reliable than the parents' repeated `navigate` calls whose feedback often said `unknown_or_stale_candidate`.
* **Goal-level failure memory.** Gather failures are keyed by block coordinates rather than count-bearing strings, and repeated failures quarantine a 5-block resource cell. Goto failures are keyed by rounded coordinates. A failed support does not suppress all nearby supports; it forces support/frontier selection to change.
* **Progression executor.** Code owns the verified opening dependency chain logs → planks → table → wooden pick → exposed stone → stone pick → fuel/iron → furnace → smelt → shield. It also has observed food, passive-mob, bed, bucket, and safety branches. Furnace timing and exact counts remain code-owned; a table/furnace action must show its postcondition before the next dependency is selected.
* **Normalized feedback and postconditions.** All public feedback fields are combined, but old feedback is not treated as a failure for a newly posted action: each transaction records its feedback baseline. Completion checks inventory, position, held item, health/food, bed, dimension, score/advancements, target block, and new block changes. A nonempty feedback marker alone is never progress.
* **Safe fallback rather than Jev deadlock.** Fire/lava, drowning, death, low health/hunger, and visible close hostiles are code overrides. A Jev safety answer is logged but never gates every ordinary action; the code gate handles lethal cases, avoiding 0007's seed-2 all-rejections. Death uses ordinary `respawn`; no inventory is restored.
* **Both model roles are retained, but off the critical path.** Astra is a sparse background macro adviser after about 35 seconds of verified stagnation or a local stall. It may choose only `progress`, `food`, `scout`, or `recover`; malformed or late text leaves the prior bias unchanged. Jev is a fast choice among at most seven exact commands already compiled from the current observation. The harness remains executable if either model is late or returns garbage.

## System-1 state and questions

Every Jev request is strict JSON and contains only derived/current information:

```json
{
  "objective": "safe verified progress toward Free the End",
  "phase": "find_wood | make_table | stage_table | approach_table | wood_pick | stone_pick | resource_progress",
  "macro_bias": "progress | food | scout | recover",
  "dimension": "overworld | nether | end",
  "health_band": "critical | low | stable",
  "hunger_band": "hungry | fed",
  "inventory": {"item": "audited integer count"},
  "visible_threats": [{"kind": "name", "range": "melee|near|far", "id": "observed id"}],
  "options": [{"command": "complete current command", "purpose": "code description"}],
  "rules": "select only one supplied command; never invent IDs, coordinates, or counts"
}
```

The batched questions are:

* `action`: a `Choice` whose criteria keys are the exact commands in `options`. Jev recognizes the best current option; it does not generate action arguments. The answer is accepted only if the selected string is still in the current menu, has the same episode and macro version, is no more than ten snapshots old, and its selected-option probability is at least **0.43**. This lower gate is intentional because code has already removed illegal and lethal choices.
* `safe`: a `Noul` about whether no listed visible close hostile/critical health state requires interruption. It is advisory/diagnostic. A low value causes no global deadlock because the deterministic emergency gate is authoritative; the answer is not allowed to authorize an uncompiled command.

Independent questions share one state and one request. The state omits raw collision grids, hidden blocks, arithmetic, and irrelevant inventory/window data. Code computes distances, reachability labels, counts, recipe prerequisites, cooldowns, and the candidate menu.

## Timing, freshness, and fallback

There is at most one background Jev future. A fresh answer is harvested before posting if it arrives within the normal short lease. While it is pending, the controller posts `noop` and waits for the next real-time tick; after **0.70 seconds** it posts the highest-priority code option rather than freezing the world. A late answer is marked committed and cannot replace that action. Ordinary Jev timeout/runtime/parse failures are logged and use the same compiled fallback; budget exhaustion and model unavailability follow the harness contract.

An active bounded job owns hands and movement. Each tick while it is active posts `noop`, and a new action is not selected until the job is complete or its postcondition deadline invalidates it. An emergency cancels first and waits for `task=null`; lava/fire, drowning, death, and stale observations take priority over Jev. Identical one-shot commands are separated by a noop barrier.

Astra is never on the action path. It starts only after a 35-second no-progress/stall signal, at most once per 185 seconds, with a short JSON-only prompt and a 120-second timeout sized for the measured slow model. Its result changes only the priority bias and increments the plan version, invalidating old Jev answers.

## Confirmation and refutation

The hypothesis is confirmed if traces show:

* no long streak of the same `place ... support up` after a placement failure;
* table placement is preceded by a safe stand or uses a destination cell that is not the current player cell;
* a failed hidden log is quarantined by spatial cell and later approaches use different current stand coordinates;
* `goto` reaches observed resource/workstation areas with fewer `unknown_or_stale_candidate` failures than the parents;
* inventory advances through wooden/stone tools and, on favorable observations, food/iron/shield rather than ending at opening; and
* no stale ticks or harness errors are introduced by background model calls.

It is refuted if the fresh-placement geometry still chooses the same occupied destination, if coordinate `goto` is rejected for ordinary observed stands, or if the controller reaches the table but stalls on a new craft/postcondition issue. A positive opening with no stone would specifically indicate that local staging is repaired but exposed-resource discovery still needs a stronger cave/frontier policy; it would not be evidence that the dragon route is complete.
