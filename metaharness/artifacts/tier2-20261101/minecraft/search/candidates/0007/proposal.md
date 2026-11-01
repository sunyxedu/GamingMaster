# Candidate 0007 — Transactional progression with Jev action choice

## Parent mechanism and observed failure

Crossover parents: `candidates/0004/` and `candidates/0005/`.

Both parents found the important structural improvement over the original Jev-only baseline: they compiled complete, observed commands rather than posting bare verbs. That is why they earned the opening milestone on favorable seeds. `0004` reached score 3 on seeds 0 and 2, and `0005` reached score 3 on seeds 0 and 2 while using fewer Jev calls. The useful mechanisms are therefore:

- `0004`: code-owned recipes, prerequisite ordering, safe candidate grounding, and background Astra macro advice;
- `0005`: a lighter Jev layer, opportunistic food/bed branches, and a bounded idle loop that does not block on Astra.

The traces also show the remaining failure mode. On seed 1, `0004` repeatedly posted `gather b:-3,64,-25 1` from the opening through the deadline even though the initial observation marked that oak log as not in reach. `0005` repeated the same target with count 3, interleaving occasional `goto`/`navigate` calls, but returned to the stale log target and finished at score 0. On seed 0, `0004` ended with `craft acacia_planks 8` repeated through the final ticks; on seed 2 it similarly repeated `craft spruce_planks 8` after the opening milestone. `0005` was less wasteful, but its successful seeds still stopped at opening score 3 with no iron, shield, food, bed, Nether, or later milestone.

## Hypothesis

The bottleneck is now action lifecycle, not merely action syntax. A compiled action is still unsafe to retry forever when its target is unreachable, its prerequisite is stale, or its previous attempt produced no observable progress. The next improvement should be a small transactional controller:

1. stage to a safe observed stand before gathering a distant block;
2. post one concrete action and preserve it while its bounded job runs;
3. verify progress from inventory, position, dimension, score, advancements, bed state, block changes, or feedback;
4. back off a failed/no-progress action and choose another currently executable option;
5. keep Jev as a fast selector over this fresh menu, with deterministic code fallback;
6. let Astra change only a macro priority such as food, bed, or iron.

This is a structural crossover rather than a parameter tweak: the controller now owns action transactions, postcondition checks, per-command backoff, and staged target approach.

## What changed

`harness.py` is a new stateful controller, not a copy of either parent loop.

- It reads only the authoritative personal inventory for prerequisites. Client previews, cursor contents, containers, and recipe previews are not counted.
- It derives a dependency-driven opening: logs → planks → crafting table → wooden pickaxe → exposed stone → stone pickaxe → exposed iron/fuel → furnace → smelted iron → shield.
- It keeps food and bed work as parallel optional branches and lets the Astra priority reorder those alternatives without overriding prerequisites or emergencies.
- For a block outside reach, it chooses a listed safe stand near that block and uses `navigate` before offering `gather`. It does not invent coordinates or inspect hidden terrain.
- It never lets a stale concrete command retry indefinitely. Every non-noop action gets a baseline signature and bounded limit. Progress includes inventory, position, dimension, score, advancements, bed state, block changes, or positive feedback. Failure, rejection, timeout, or expiry adds a temporary action-specific backoff.
- One-shot repeats are separated by `noop`; active bounded jobs receive `noop` rather than being restarted. Stale observations cancel a current job before new work is selected. Death uses ordinary `respawn`.
- The controller uses an asynchronous Jev request only over the current compiled options. If Jev is late, malformed, stale, low-confidence, or unsafe, code falls back to the highest-priority live option. No model output can invent an argument.
- Astra runs asynchronously and returns only `{"priority":"balanced|food|bed|iron|explore"}`. Invalid or slow Astra output leaves the prior macro priority unchanged.

## Jev interface

Each Jev request carries a compact JSON state with:

- `stage`: code-derived phase such as `logs_table`, `place_table`, `wooden_pickaxe`, `stone_pickaxe`, `resources`, or `emergency`;
- `macro_priority`: the current Astra-selected priority;
- authoritative `inventory` counts;
- health and hunger bands computed in code;
- visible hostile entities represented by kind and named range buckets (`melee`, `near`, `far`), not raw geometry;
- recent action feedback;
- `options`: at most eight complete action strings, each with purpose and code priority;
- explicit rules that Jev must select only a supplied action and must not invent parameters.

Two typed questions are batched:

1. `action`: `Choice` over the supplied complete commands plus `unknown`.
2. `safe`: `Noul` asking whether ordinary work is safe given the explicitly listed visible threats and health band.

The selected action must still exist in the latest menu, match the episode and macro-plan version, and come from an answer no more than 2.5 seconds old and 10 snapshots old. The action Choice must assign at least 0.58 probability to the selected option, and the safety Noul must assign at least 0.68 to `true`. Those are consequence-aware gates, not a substitute for code validation.

## Fallback and timing policy

The real-time loop always posts an action or `noop` before waiting for the next 0.5-second tick. While Jev or Astra is pending, the game continues with `noop`; the current bounded job is preserved. Jev is given 0.85 seconds to return before the controller uses the deterministic menu default, which is safe given the measured Jev p50/p90 latency. A late answer is discarded rather than applied to a changed world.

Astra is never on the critical action path. It starts only after local progress has been stagnant for roughly 25 seconds or multiple actions have failed, and no more often than about every 190 seconds. Its timeout is sized for the measured slow System-2 latency but leaves ample wall time for gameplay. Budget exhaustion and model-unavailability exceptions follow the harness contract; ordinary model timeout or malformed output falls back without ending the episode.

## Confirmation and refutation

The hypothesis is confirmed if traces show:

- distant resource blocks are approached through observed stands instead of endlessly retried from out of reach;
- failed craft/gather/navigation actions are backoff-limited;
- inventory and score progress continues beyond the opening on seeds where exposed stone, iron, food, or beds become available;
- Jev selections are fresh and grounded, while deterministic fallback remains executable; and
- the candidate remains error-free with low stale-tick fraction.

It is refuted if the controller still loops on one concrete action after its postcondition fails, if the staging candidates do not bring observed blocks into reach, or if inventory auditing remains too delayed for the transaction limits and causes productive actions to be blacklisted. In that case, the next iteration should narrow the postcondition windows or improve target-specific navigation rather than add more free-form model planning.

The harness is intentionally focused on converting the parent opening gains into reliable, auditable progress. It does not claim a complete Nether, stronghold, or dragon-fight policy; those should be added only after this lifecycle hypothesis is validated on live traces.
