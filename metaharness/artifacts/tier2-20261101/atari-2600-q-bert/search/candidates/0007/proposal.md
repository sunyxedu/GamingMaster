# Candidate 0007 — palette-epoch coverage with guarded execution

**Mode:** crossover. **Parents:** `0005` and `0004`.

## Hypothesis: the shared ceiling is a progress-interpretation failure

The parents' confirmed mechanisms are deterministic 21-node traversal and an action loop independent of model latency. `0005` supplies fast useful-cube routing; `0004` supplies neutralized single-hop execution and freshness checks. Their asynchronous advisers are useful architectural components, but the archive does not establish that Astra caused their score gains.

The shared ~4,000-point ceiling is **not adequately explained by never completing the first pyramid**. Public observations contradict that interpretation:

* **0005, seed 0:** at tick **220**, score **850**, all 21 phases are `D` and the incomplete count is zero, with three lives. Ticks **222–259** then alternate whole-board colours while the score rises to **3950** in **31 increments of 100**. Q*bert stays at node 20; no Green Ball is visible. This matches the manual's **3,100-point completion tally**, although the decoder calls the increments `green_ball` and leaves `completed_rounds=0`.
* At tick **267**, node 1 earns 25 points but changes from decoded **D to S**, with raw colour **26→148**. Node 0 does the same at tick **275**, and node 2 at tick **417**. The controller subsequently oscillates over those already-scoring upper cubes until termination at tick 900.
* **0004, lowest-score seed 1:** tick **457** has all 21 raw colours equal to 26, all needs zero, score **825**, and two lives. After the same flashing tally, the terminal observation at tick **589** still labels the uniform next board complete; score is **3925**, with no further useful coverage.
* **0004, seed 4** supplies another view: all-D at tick **451**, flashing through tick **490**, then rewarded **26→148** landings at nodes 1 and 0 at ticks **503** and **516**. Later Jev requests repeatedly target these nodes, sometimes proposing NOOP because the target is already under Q*bert.

**Hypothesis:** the cartridge changes palette semantics across rounds, while the adaptor retains the first round's phase interpretation. Routing directly from those labels traps both parents after a real clear. A harness-owned, observation-calibrated palette model should preserve their initial traversal and continue onto subsequent boards.

This corrects an interpretation in `lessons.md`, not the evaluator itself. The harness never modifies score, observations, RAM, the adaptor, or evaluator counters. All local clear counts are explicitly labeled **inferred**.

## The crossover and new control structure

### 1. Evidence-based progress epochs

`PaletteProgress` uses public `cube_colors` and cumulative score deltas:

* A change to at least 16 cube colours in one observation starts a whole-board flashing interval. During that interval the controller posts NOOP instead of treating each palette flash as useful coverage or a Sam reversal.
* After the palette stabilizes and Q*bert returns to the top, a new coverage epoch begins. The score accumulated during the interval is logged separately. A large tally supports an inferred clear; observed local trials produced exactly **3100** each time.
* The current destination colour is learned from a local colour change with a 25-point landing component, including a possible simultaneous 300-point capture. It is **not** a hard-coded colour or a function of seed.
* Each subsequent observation recomputes unfinished cubes from the learned destination. Sam reversions and reversible visits therefore invalidate completion immediately rather than being flattened into permanent visits.
* If a uniformly visited board does not begin a completion animation after 22 ticks, the controller permits an additional pass and relearns the next useful colour. This is a bounded recovery for multi-stage progress, not an assumption that the reported level number is correct.

The code logs `palette_goal_learned`, `palette_flash_begin`, `palette_epoch`, and `decoder_progress_disagreement`. Reported round counters remain untouched.

### 2. Fast routing plus a guarded motor

The documented Atari 21-node graph is built and all-pairs graph distances are computed in code. Candidate edges are restricted to the current observation's `safe_moves` and the graph; an empty safe set is not silently replaced with all legal edges. A current-cube target cannot become a zero-length route deadlock: the next hop is ranked by its destination and subsequent distance to unfinished cubes.

A directional post is latched as one hop. It can be retried briefly before movement starts, but all moving ticks receive NOOP. Observed landing, minimum hop age, and a neutral release separate successive hops. A bounded retry recovery handles a pulse ignored during a cartridge animation. Falling, invisibility, and palette flashing are neutral states. FIRE and guessed disc exits are never used.

The main loop posts every tick and uses `ctx.wait_for_tick()`. It never waits for inference, and it runs until the game or context ends—not to an internal step limit.

### 3. Local temporal safety

Geometry-only legality is insufficient: the first live trial lost lives while hopping from a bottom cube into a descending ball's possible next branch. Code now derives short-horizon danger from public enemy positions, cube coordinates, graph adjacency, and bounded four-tick visibility memory. This also corrects cases where the decoder's nearest-node estimate lags a rendered enemy moving between rows.

Candidates distinguish an occupied landing, a possible interception, and a clear route. For Coily, a seemingly clear corner receives a substantial penalty when its only return cube is threatened: covering a corner now is not useful if Q*bert cannot leave it.

When descending balls threaten every candidate but the current cube is appreciably safer, NOOP becomes a **deliberate, bounded tactical candidate**. The wait ends when the branch resolves, the current cube becomes threatened, a nearby Coily appears, or 18 ticks elapse. An earlier ten-tick bound sometimes expired before the ball committed; the refined bound is checked against current danger every tick, not blindly held.

## System-1: schema, questions, and answer-to-action bridge

One asynchronous request is issued at most every five ticks, with at most one outstanding. Requests can run during an incoming hop so the motor need not stop for Jev. Inputs contain no raw RAM or opaque palette values:

```json
{
  "policy": {"style": "nearest", "review": "collision_first", "priority": "survival, coverage, completion"},
  "progress": {"interpretation": "destination learned from scoring palette changes", "board_epoch": 1},
  "position": {"node": 17, "motion": "hopping"},
  "candidates": [{
    "action": "LEFT", "landing": "transit", "danger": "clear",
    "escape_space": "multiple", "route": "one_hop_away", "code_preferred": true
  }],
  "proposed_action": "LEFT",
  "threats": [{"kind": "purple_ball", "proximity": "farther"}]
}
```

Code computes all distances, landing usefulness, escape counts, and rankings. Jev receives semantic buckets rather than arithmetic tasks.

Each request batches three independent questions:

1. **`a`: Choice** over supplied edges, a deliberate NOOP when code offers it, and `unknown`. Select the safest useful alternative using danger, escape space, and the code preference.
2. **`b`: Choice** with the **same instructions and reversed option order**. It does not depend on `a`; agreement is checked in code.
3. **`danger`: Noul** asks whether the proposed action has an immediate, supported lethal threat. Mere distant visibility is not sufficient.

**Acceptance:** both Choices must agree and each selected-option probability must be at least **0.60**. Missing/invalid probabilities become zero; absent `.confidence` is never treated as certainty. Current code still requires the chosen alternative to be within **2.0** cost units of the best candidate and no more than **0.2** worse in computed danger. A Noul probability of at least **0.82** can veto the proposed edge only in favor of a currently available alternative whose danger is no more than **0.1** worse. There is no model-authorized off-board action.

**Freshness:** advice carries originating node, tick, life epoch, palette epoch, policy version, and the relevant hazard-node signature. All must still match, and age must be **0–8 ticks**. Current safety and cost are recomputed before execution. Stale answers, disagreement, `unknown`, missing fields, malformed values, and transient call failures all retain the executable code route. `jev_changes_action` and `jev_veto_applied` identify actual interventions rather than merely counting calls.

**Timing:** Jev timeout is **1.0 s**. Final-controller trials on seeds 3 and 4 measured median latencies about **0.173 s**, with maxima **0.673/0.699 s**. Thus synchronous per-tick use would miss several game ticks; asynchronous calls and a fresh deterministic action are necessary. Only one Jev request and one Astra request can be outstanding.

## System-2: durable policy and interface revision

Astra receives current score/lives, inferred versus reported progress, recent rewarded colour-change evidence, and the active policy. Its prompt explicitly explains the compact Jev state, typed Choice/Noul roles, confidence gates, and freshness checks; it does not assume inheritance of the proposer's context.

It returns only two bounded fields:

* `style`: `nearest`, `top_first`, or `frontier`, selecting small coverage tie biases;
* `review`: `collision_first` or `escape_space`, selecting Jev's safety wording and a corresponding escape-space tie bias.

No generated code, unvalidated joystick route, or arbitrary model-authored instruction is executed. Garbage text or unsupported values keep the current policy. Policy changes invalidate old local advice and are logged as `astra_policy_changed`.

Calls are asynchronous, capped at three, with a 100-second timeout and a 300-tick revision cooldown. Initial planning, progress-epoch changes, life loss, and a 150-tick scoring stall can trigger a review while calls remain. The hypothesis is that repeated failure evidence should change the **recognition criterion** rather than ask Astra to drive individual frames. In practice, early epoch reviews can consume the small cap before later deaths; this is a remaining limitation.

Astra latency in the final seed-3/4 trials reached **60.84/53.87 s** despite compact requested output, validating the long timeout and off-clock architecture. The observed score gains must not be attributed to Astra merely because its requests completed.

## Local experiments and interpretation

All trials used the archive's default sticky-action configuration and **training seeds only**. All listed episodes ended `game_done` except the final seed-0 repeat, which reached the external **five-minute trial deadline while still playing**. None ended with a harness error. These are development trials across successive refinements, **not** the official six-seed evaluation or a pooled estimate of the final candidate.

| Revision | Seed | Astra | Score | Inferred 3100-point clear tallies | Stale ticks |
|---|---:|---|---:|---:|---:|
| Palette + guarded motor, before tactical waits | 0 | off | 11,925 | 3 | 1 / 1823 |
| Same initial structure | 1 | on | 11,925 | 3 | 1 / 1807 |
| Ten-tick tactical wait | 0 | off | 23,425 | 6 | 3 / 4092 |
| Ten-tick tactical wait | 2 | on | 12,050 | 3 | 0 / 1792 |
| Final return-path guard + branch-resolving wait | 3 | on | 22,875 | 6 | 13 / 3682 |
| Final return-path guard + branch-resolving wait | 4 | on | 22,850 | 6 | 14 / 3295 |
| Final controller, repeat check | 5 | on | 22,850 | 6 | 5 / 3295 |
| Final controller, five-minute deadline | 0 | on | 26,675 | 7 | 8 / 4499 |

Trace directories under `tries/` retain all requests, answers, timing, and evidence. Final-controller seeds 3 and 4 had 672/599 Jev calls and three Astra calls each, with zero failed model calls; both preserved all four reported lives through their first four inferred clears. The final seed-5 repeat used 607 Jev calls, with no failed model calls. The capped final seed-0 repeat used 837 Jev calls and had one **handled one-second Jev timeout**; deterministic execution continued and the episode ended normally at its trial deadline. All four final-controller trials used three Astra calls. Seed 0's 26,675 is a capped score, not a game-over score or a full 900-second evaluation.

The main hypothesis is supported directly: the harness learned **148** as the next destination after the initial **26**, did not get trapped on the upper two/three cubes, and produced additional independent 3100-point tallies. The official adaptor still reports zero completed rounds, so that counter alone is not a valid progression test here.

The initial Astra-on/off trials both scored 11,925; an intermediate Astra policy change was logged but did not establish improvement. In all four final-controller trials Astra retained the existing policy. **There is no demonstrated independent Astra performance benefit.** Different seeds and refinements also prevent treating the 23,425 versus 12,050 pair as a causal ablation.

Syntax compilation, graph reciprocity/bounds checks, palette replay on the parent archive, a synthetic full-loop replay with arbitrary/malformed model answers, and **89 motor-safety cases** covering each node and motion state plus invalid nodes passed. These checks supplement—not replace—the evaluator's official pre-flight. The final tested harness SHA-256 is `7af12e8e4888ec4739e89e42f9d77edf665b65ccb4e6c07584cd7bcb2a6b7fb8`.

## Confirmation, refutation, and next research step

Confirm on the full training evaluation if the controller consistently crosses the first palette inversion, maintains low stale fractions, and earns further verified scoring transitions without errors. Refute the palette hypothesis if inferred new goals repeatedly fail to predict useful landings, if flashing does not correspond to large score tallies and a new usable board, or if upper-cube oscillation returns despite correct calibration.

Remaining headroom is later-board survival and route quality. The current colour model resynchronizes from observations but is not an optimal reversible-cycle planner. Enemy prediction is conservative and approximate, disc exits remain unvalidated, and every game-over development trial eventually exhausted its supply; one final repeat was still running at the five-minute cap. A later iteration should test a time-expanded escape planner and a controlled same-seed Astra ablation, rather than crediting adviser call counts for code-owned gains.
