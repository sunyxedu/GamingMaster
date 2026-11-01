# Candidate 0004 — Contact cartography and asynchronous intent control

## Parent and falsifiable hypothesis

Parent: `0001-tier2_astra_only` (structural rewrite). Comparator: `0000-tier2_jev_only`, the best eligible archived candidate. I also checked the idle evaluation and the study documents. There is no `lessons.md` in the supplied archive.

**Hypothesis:** The principal bottleneck is not model intelligence or a slightly slower request. It is the missing feedback controller between intent and motion. Neither baseline records that a commanded forward action failed, aims using visible world positions, or converts exploration into finite, testable motion. Jev consequently repeats a plausible-looking action indefinitely; Astra issues the same dangerous action with minute-scale hold times. A proprioceptive motion/aim servo plus contact-derived exploration should increase useful exploration and survival, and make semantic model advice actionable.

### Concrete evidence

- Best candidate, `trace_seed1002.jsonl`: 22,710 engine ticks use `forward`, seven use `noop`. At tics 5,605, 11,521, 17,129 and 22,727, the position is exactly `(224.0, 912.0488128662109, 0.0)` with angle `270.0000000628643`. It consumes all **4,000 Jev calls** in 649 seconds without a kill or item. The final forward vote has probability **0.62**: increasing confidence does not solve the absent displacement test.
- Best candidate, `trace_seed1005.jsonl`: 260 ticks use `forward`, 61 use `fire_forward`. At tic 92 it reaches y=`79.951171875`; at tics 174, 256 and 338 that y is unchanged, while health falls **79 → 28 → 7**. The angle remains 90 degrees, kills remain zero, and the final score is **1.6773** after 9.43 seconds. There are visible monster world positions, but the controller never turns to aim.
- Parent, `trace_seed1012.jsonl`: the first Astra answer is `fire_forward` after **118.041 seconds**. The trace contains 4,132 noop ticks followed by 1,098 fire-forward ticks. It dies at 149.52 seconds, with weapon 1, zero ammunition and one kill. The second call is cancelled by game end.
- Parent, `trace_seed1001.jsonl`: `fire_forward` arrives after **129.537 seconds**; death follows about 15.65 seconds later. It does get two kills. Thus sustained fire at a valid angle can work, but uncontrolled advance and unbounded holds are not safe.
- Parent mean 4.3100 and best mean 4.4390 are close to idle 4.2500, with **zero exits**. This motivates a structural rewrite, not another action-choice wording tweak.

## Architecture

### Code: control and experimentally observed geometry

The foreground loop observes every engine tic and posts a newly checked action. It computes world-relative bearings, steers until aligned, verifies available weapons/ammo, fires while strafing, creates space from contact enemies, and updates damage and movement history. Right-turn sign and turn step are checked against actual angle changes. Rocket/BFG fire is conservatively disabled because walls are not observable; ordinary weapons remain usable. Monster category is required before applying the declared hitscanner/major/melee name taxonomy.

Exploration uses 80-unit horizontal cells with separate elevation bands. Its nodes and directed connections are created **only from actual player traversal**. There is no WAD access, map-specific route, seed branching, automap or inferred whole-map geometry. Untried directions are hypotheses, not asserted free space. A finite probe either makes measured progress or becomes a contact test: pulse USE, allow the door time to move, measure again, then back off and suppress the failed direction. Exhausted nodes can backtrack along observed edges to an untried frontier. Useful pickup positions and explicitly mapped interactables (exit, switch, door, teleporter categories or names) are remembered only after rendering; exits get highest local priority and receive an immediate USE pulse at contact, while terminal success is still accepted only from the engine. Large vertical offsets are not treated as directly reachable. Passed-through absent objects are suppressed as unavailable, not asserted to be acquired keys.

The exact same controller is the fallback when either model is delayed, uncertain, malformed or unavailable temporarily. It never holds a stale model turn for seconds. Actor-visibility checks, finite progress clocks, contact cooldowns and ammunition filters outrank model intent.

### Jev: one persistent asynchronous worker, four independent judgments

The persistent `ctx.spawn` worker keeps the same request thread alive, avoiding repeated cold connections. Only one request can be outstanding. The nominal request cadence is 0.38 seconds in combat, 1.2 seconds otherwise, comfortably below 4,000 calls over 900 seconds. The timeout is **2.5 seconds**, above the archived two-second maximum; lateness never blocks the control loop.

Every state explicitly includes:

- Task and visibility limitations; meanings of remembered versus visible actors.
- Astra's current policy, version and short conditional `local_rule`.
- Code-derived health/armor/ammo buckets, recent-damage/projectile flags and movement/progress evidence.
- Up to five visible threat candidates and five useful observed pickup/interactable candidates, with observed exits/switches prioritized by code.
- Each candidate's identity, category, semantic range/direction/elevation/visibility, hitscanner/major/body-blocker facts.
- Previous action and controller reason. No numeric comparison or hidden-layout inference is delegated to Jev.

Questions and bridges:

1. **Posture Choice:** `engage / retreat / bypass / unknown`. Winning probability at least **0.62**. Controls whether optional combat interrupts exploration, or creates distance while fighting. Immediate threat and legal-action guards can override it.
2. **Target Choice:** up to five explicit visible candidates plus `none`. Threshold **0.56**. Supplies an actor ID to the arithmetic aim servo; the actor must still be visible, and a contact enemy can override it.
3. **Goal Choice:** observed useful pickups plus `explore / unknown`. Threshold **0.62** for a pickup. Supplies an objective ID, never coordinates invented by a model. Code rechecks reachability, utility and suppression memory. Candidate order is reversed on alternate requests to reduce fixed-order bias.
4. **Plan-invalid Noul:** asks whether observed resources/threats/stagnation clearly contradict the durable plan, not whether one collision occurred. Probability at least **0.85** on three accumulated strong judgments requests Astra revision.

Choice confidence is treated separately: present confidence below 0.12 rejects the answer; absent confidence does not create certainty. Tactical answers older than **0.85 seconds / 30 tics** are discarded; accepted tactical semantics expire after another 0.7 seconds and visibility is rechecked every tic. Goal responses may arrive within 2.5 seconds and persist for seven seconds only while the object is still useful. Any policy-version mismatch invalidates the result. `jev_bridge` events record adopted fields, fallback and revision requests; control transitions show their execution.

### Astra: delayed policy and Jev-interface revision, not buttons

Astra runs asynchronously, initially and then at a 210-second review interval, or after sustained mismatch/teleport with a minimum 110-second launch separation. It has a 240-second timeout, sized above the archive's 213-second maximum. It receives public observations, recent outcomes, observed landmarks, exploration counts and the exact relevant local Jev contract. Its bounded JSON can revise resource objective, combat policy, exploration handedness, an already observed focus ID, and the short conditional rule passed to Jev. No generated code is evaluated.

Astra's tactical locations would be hopelessly stale, so only durable policy is accepted. Unknown enums, nonexistent landmark IDs, overlong rules and non-JSON output are ignored. The previous executable policy survives arbitrary stand-in text. Plan changes invalidate pending Jev interpretations and navigation targets.

**Interface-revision hypothesis:** Astra should revise the local decision rule when repeated Jev contradiction or slow progress shows a mismatch between resource/visibility facts and the objective. It should not be consulted for individual turns, target bearings, a single blocked wall, or every new rendered label. The bounded interface deliberately prevents it from hallucinating unseen routes.

## Verification and evaluation criteria

Planned local checks are syntax/import-contract checks using stand-ins, malformed-response tests, and short synthetic controller scenarios. No game implementation, held-out results, external model backend, or long game experiment is inspected or run.

Evidence supporting the hypothesis would be: materially fewer repeated zero-displacement forward ticks; visible acquisition of useful weapons/pickups; more distinct actually visited cells; a healthier kill/resource trajectory; eventual exit success; and useful Jev/Astra bridge events rather than raw call counts. Evaluation must report score and completion, not celebrate freshness alone.

Evidence against it would be: reliable unsticking without any meaningful exploration/score gain; repeated geometric loops despite frontier memory; firing at corpse labels; excessive turning that prevents shots; target switching that loses the retreat lane; or no measurable effect of accepted semantic advice. If movement improves but exits remain absent, the next bottleneck is symbolic observation of walls/switches or exploration policy—not a reason to secretly inspect map data.

## Scope and limits

This is one candidate testing the missing-feedback hypothesis. The topology is approximate, actor labels do not expose walls or door state, and an unsuccessful USE cannot distinguish a wall from a locked door. Key inventory is not invented. The controller cannot guarantee an exit in arbitrary unseen Doom maps under those observation constraints. Actual model latency, score improvement and zero-error eligibility remain for the evaluator to establish.
