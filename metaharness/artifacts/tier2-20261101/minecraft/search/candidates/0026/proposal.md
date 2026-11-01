# Candidate 0026 — Verified checkpoint handoffs

## Parent and trace diagnosis

Parent: `candidates/0009/`. The redesign keeps its useful principle—compile executable work from public state and confirm postconditions—but replaces its large, frequently rebuilt action menu with a deterministic milestone compiler and a single-owner transaction state machine. The best comparator is `candidates/0015/`; the later verified mining traces in `0019` and portable-station trace in `0021` are used as evidence, not copied as a controller.

The archive shows two specific, adjacent failures:

* In parent `0009`, seed 1 spent all 68 work posts in `find_wood`; it retried `gather b:-3,64,-25 1` at 36, 222, and 644 seconds, and never acquired inventory. The seed-0/2 opening did not transition beyond the wooden tool: seed 0 posted 130 `stone_pick` and 122 `approach_table` actions, while seed 2 accumulated 31 goal invalidations. A new coordinate or a recompiled command was not verified resource progress.
* Best candidate `0015` fixed feasible recipe counts and reached its opening score on all three seeds, but no episode earned iron/shield. Its seed-0 action trace made a table and wooden pick quickly, then cycled through observed stands and `goto -22.00 68.00 31.00`; after 11 failed table returns, it still had no stone. Candidate `0023` removed that table-return gate and still failed to complete a stone-pick handoff, so a different retry interval is not the answer.
* A staircase can produce real cobble, but it is slow and error-prone. Candidate `0019` seed 0 confirmed three cobblestone pickups after a verified moving mine, then spent the rest of the episode failing to return to its original workstation. Candidate `0021` did craft a stone pick in seed 2 at about 558 seconds after a local table handoff, but only one of its nine Generation-7 comparisons made the stone pick; its other traces show many rejected dig/look/navigation operations. The missing test is a reliable, locally compiled handoff from *audited cobble* to *nearby crafting table* to *stone pick*, while avoiding repeat loops on failed resource targets.

## Hypothesis

A bounded transaction executor with **checkpointed local station creation** will turn occasional stone acquisition into more verified tool progression. The controller should not treat a remembered surface-table coordinate as a prerequisite: after three audited cobble and two real sticks, it can craft a table from four carried planks using the inventory grid, place it on a currently observed adjacent support, wait for the actual table candidate, and then craft the stone pick. Resource work is separately checked against personal-inventory deltas; movement alone never counts as mining progress.

This is a careful test of the nearest known bottleneck, not a claim that the harness can already complete the End. It retains ordinary survival, bounded actions, and fail-closed behavior. It also opportunistically seeks observed food and a listed bed after the wooden pick, because the score awards those opening milestones directly and the earlier best candidates left those points unearned.

## What changed

* **New division of labor:** code owns a compact milestone compiler, all recipe bounds, observed target selection, stage transitions, cooldowns, and action legality. There is no Jev action choice over routine crafting/navigation transactions. A single active transaction owns hands/movement until feedback or a postcondition settles it.
* **Two-stage resource confirmation:** an exposed candidate is gathered only when listed and harvestable. If a resource candidate is in reach, the adaptor's `gather` primitive handles the tool/pickup; a missing block without an inventory delta enters a short real-drop pickup stage. When no exposed stone is available, one staircase step targets an observed adjacent safe block only; direct `dig` is followed by an observed-cell movement check. Falling blocks, liquids, unknown chunks, and the player's support block are excluded.
* **Local station checkpoint:** opening logs/planks are budgeted for a modest reserve, with a 95-second cap on waiting for an ideal reserve. At the stone checkpoint the compiler uses the stationless 2x2 table recipe from carried planks, explicitly equips and places the table on a current support, waits for a nearby candidate, then crafts the stone pick. A remembered original table can trigger at most one bounded recovery attempt; it cannot trap the mine in an unconditional return loop.
* **Failure is goal-level:** failed or timed-out gathers/mines cool their observed target; failed navigation cools the target/work transaction. The executor will not count changed position or stale feedback as resource acquisition. Death causes ordinary respawn and no inventory restoration.
* **Astra is advisory and asynchronous:** at an early quiet checkpoint or after a long no-progress interval, Astra sees a compact milestone/inventory summary and selects only a supplied macro focus (`balanced`, `food`, `bed`, `stone`, `iron`, or `recover`). Its response can change optional priority, not emit a command, skip a prerequisite, or override hard safety. It is polled in the main loop; late, malformed, or timed-out text is ignored. At most three calls are started, with a long cooldown.
* **Jev is tactical and asynchronous:** it is called only for a live visible hazard. The state is a compact JSON summary of health/food bands, dimension, current phase/focus, named hazards with distance buckets, and exact defensive response strings (cancel, shield, attack/equip where legal, navigate to a listed stand, or back+jump). It answers two batched `Choice`s over the same options in opposite orders plus one diagnostic `Noul`. The code accepts an answer only if both Choices agree on a current action, each selected probability is at least 0.62, and episode/sequence/hazard-menu freshness still matches. It cannot authorize a command outside the generated menu.
* **Real-time fallback:** lava/fire, drowning, stale observations, critical hunger/health, and death are handled in code immediately. While Jev is pending or uncertain, code cancels a hazardous owned job, otherwise retreats/uses a shield; it never waits for inference while movement persists. While a bounded job owns the adaptor, the harness posts `noop` and waits for the next tick.

## Jev state and question policy

The request schema is:

```json
{
  "purpose": "immediate hazard response only",
  "self": {"health_band": "critical|low|stable", "food_band": "low|fed",
           "dimension": "observed dimension", "on_ground": true},
  "current_phase": "code-owned milestone phase",
  "astra_focus": "bounded macro preference, advisory only",
  "snapshot_freshness": "fresh|aging",
  "hazards": [{"kind": "observed label", "range": "touching|near", "observed_id": "public entity/block ID"}],
  "legal_responses": ["exact full action strings"],
  "rule": "choose only a supplied response; never invent IDs or coordinates"
}
```

* `response_a` and `response_b` are independent `Choice`s over exactly the listed response strings. `response_b` reverses the option ordering. Their result is used only when both select the same currently legal command with selected-option probability ≥ **0.62** and the source episode/sequence is no more than **8** snapshots old.
* `continue_work` is a `Noul` diagnostic about whether a named hazard warrants interruption. It is logged but never overrides code safety.
* A late response is discarded. The fallback is deterministic and executable from the newest state: hard-hazard escape; cancel an owned job near a hostile; otherwise shield or the safest generated retreat. No model output is parsed as an action or coordinate. Resolved or unusable calls impose a **1.0–1.5 s** hazard-signature cooldown to avoid spending the 4,000-call budget repeating the same judgment every half-second.

## Evidence that would confirm or refute

A one-minute live smoke test on seed 0 reached score **3**: it collected three real logs, crafted and placed a crafting table, made a wooden pick, and entered three observed cells with **0/119 stale ticks**. This verifies the opening/transaction plumbing, not the stone handoff. A two-minute seed-1 smoke run remained at **0** and hit repeated failed safe-stand navigation near an oak target without error; the difficult-spawn/wood discovery bottleneck is therefore still open and is an explicit refutation condition, not claimed solved.

**Confirm** if the harness has no episode errors/stale ticks; every mined resource milestone follows an audited inventory delta; a seed that obtains three cobble and has reserve planks places a local table and crafts a stone pick without returning to the original table; and the number of repeated rejected target/table actions is materially below the parent/best traces. Additional confirmation would be more food/bed opening points from observed sources without extra deaths.

**Refute** if gathering/tree acquisition still retries the same block after a failure, the staircase never turns exposed adjacent blocks into inventory, local station placement repeatedly consumes the table without an observed nearby block, or episodes with audited three-cobble inputs still fail to issue/complete the stone-pick recipe. If no seed reveals a reachable stone source, the local-station hypothesis remains untested; the trace should distinguish that from a failed craft handoff. Any increased death rate or time spent waiting for a reserve would also refute the risk-adjusted version.
