# Candidate 0013 — late-board rolling-hazard occlusion guard

**Mode:** refine. **Parent:** `candidates/0007/` (the current best).

## Hypothesis and scope

The parent's palette epochs, graph routing, neutralized single-hop motor, and asynchronous advisers are retained. The one targeted change is its **interpretation of a missing rolling-hazard sprite on later boards**:

> Missing rendered hazard data is sometimes uncertainty rather than proof of an empty branch. Preserve a short directional shadow of a recently seen rolling ball, and distinguish the observed purple-family state 36 from ordinary visible-Coily state 30. This may prevent a subset of later-board collisions without perturbing the reliable opening.

This is an experimental safety heuristic, **not a calibrated decode of state 36**, and not a claim that every empty-hazard loss has this cause. No game implementation, RAM writes, hidden state, or held-out seed information is used. The guard becomes eligible after four **scored, inferred** clears, matching the demonstrated failure regime rather than an episode tick or seed.

## Trace evidence and negative controls

The parent's worst official episode is `0007/trace_seed0.jsonl`: **15,700**, ending after 2,682 ticks. It retained four reported lives through its fourth scored palette rebase at tick **2017**, then lost lives at **2222, 2322, and 2422**.

The second loss is the relevant evidence:

* The observation at **2310** has Q*bert near node **4**, `hazards=[]`, all four moves in `safe_moves`, and `timers.purple_ball_state=36`. Observations at **2270, 2280, 2290, and 2300** also show empty hazards and state 36.
* At **2312**, `coverage_status` records `node=4`, `action="UP"`, `threats=[]`, `lives=3`, and 12 unfinished cubes. The action is applied at **2313**. A life loss is recorded at **2322**; the adjacent-purple warning appears in a Jev request only around that loss.
* At **2402**, state 36 and empty hazards recur. Q*bert visits the apex at **2413**, then DOWN is applied at **2415–2416**, and another loss occurs at **2422**. This last case is a limitation: a climb preference alone does not protect a downward apex departure.

There is also a shorter visibility-gap pattern in the parent's seed-1 and seed-4 traces. At **1964**, the controller moves DOWN from node **13** with `threats=[["purple_ball",8]]`. It lands at **18** at **1973**, waits, then posts UP at **1981**. A Jev request around **1983** labels both parents clear and has `threats=[]`; the life loss follows at **1987**. This supports testing whether the four-tick retention horizon is shorter than a lethal visibility gap. Sparse score-change observations do not establish the exact ball trajectory or the exact timer value at the collision.

Two archive lessons constrain this experiment:

* `0011` changed a clear corner commitment to a more immediately threatened non-corner at tick 200 through its unconditional +40 cost. It scored **15,079.17**, versus the parent's **20,708.33**, with zero corner-deferral events. No new corner penalty is added here.
* `0009` combined broad radar shadows, route changes, and a different motor and regressed to **15,454.17**. That does not establish a benefit from shadows. This candidate keeps the parent controller and changes only its late-board occlusion facts. It does not add persistent targets, edge-failure learning, a generic Coily radius, or a new landing debounce.

## Implementation

### One late-board occlusion layer

Before four inferred clears, hazard record expiry and deterministic scoring match the parent. Afterward:

1. **Recent rolling ball:** purple/red records expire after **10 ticks**, instead of four; all other hazards still expire after four. During ages **5–10**, the last rendered node and its two DOWN/RIGHT descendant layers form a short possible branch. Existing danger at the last observed position/child persists; a stale descendant outside the immediate children adds **1.25** danger. The shadow is a possibility, not a new observed enemy.
2. **Never-localized uncertainty:** `purple_ball_state == 36`, with neither a rendered purple ball nor Coily, sets `hidden_ball_active`. State **30 does not qualify**: the archive and local trial show it alongside ordinary visible Coily. No enemy node is invented from the timer.
3. **Bounded response:** on entry through a changed state value, a **three-tick** neutral guard may run, with a **12-tick rearm cooldown**. An already-launched hop is not interrupted. While state 36 remains unlocalized, UP/LEFT candidates add **0.20** danger; DOWN/RIGHT add none. At the parent's danger multiplier of 12, this is 2.4 cost units—less than the four-unit difference between a graph step toward a target and one away from it. It biases close choices without making a persistent target unreachable solely because climbing costs more.

The existing `safe_moves` intersection is still the hard legality filter. No corner ban, FIRE, guessed disc exit, fixed seed route, or score modification is introduced. Palette flashing, inferred clears, life recovery, motor timing, coverage styles, and stopping on true terminal/context signals remain the parent's behavior. A life counter reaching zero is not a stop condition.

`hidden_ball_guard` records phase-transition guards. `occlusion_decision` records executed choices and all candidate costs/dangers when timer uncertainty or a stale branch is present. Existing `hop_landed`, `control_epoch`, `palette_epoch`, and model-intervention events remain available for attribution.

## System-1 state and questions

Jev retains the parent's compact request: game/rule summary, current policy, inferred palette epoch, node/motion, candidate facts, proposed action, and lethal-threat summaries. Code owns legality, graph distances, progress arithmetic, branch construction, expiry, and numerical comparisons.

Each candidate adds:

```json
{
  "action": "UP",
  "landing": "unfinished",
  "danger": "clear",
  "escape_space": "multiple",
  "rolling_branch": "timer_active",
  "route": "on_target",
  "code_preferred": false
}
```

`rolling_branch` is `none`, `stale_descendant`, or `timer_active`. Threats add a visibility category; timer-only uncertainty is explicitly `{"kind":"purple_ball_timer","proximity":"uncertain","visibility":"occluded"}`. Raw timer numbers and RAM are not sent to Jev for arithmetic or interpretation. Here `rendered` is the parent's fresh-record bucket, including its original four-tick memory; it does not certify a sprite appeared on this exact tick.

One asynchronous request batches three independent questions:

* **`a`: Choice** over supplied legal edges, a bounded NOOP when offered by code, and `unknown`. Prefer clear useful routes and the code preference. A stale descendant is a recent possible rolling branch; a timer-only flag is uncertainty, not proven contact.
* **`b`: Choice** with identical instructions and reversed option order, to limit first-option bias.
* **`danger`: Noul** asks whether proposed-action facts support immediate lethal danger. A recent branch may support caution, but timer-only uncertainty or a distant threat alone does not establish contact.

Both Choices must agree on a supplied option and each selected-option probability must be **at least 0.60**. The existing execution gate requires cost no more than **2.0** above the fallback and danger no more than **0.2** above it. A Noul at **0.82** can veto only toward an alternative no more than **0.1** worse in computed danger. Missing probabilities become zero; absent confidence is not certainty.

Advice must match the node, life epoch, palette epoch, policy version, hazard-node/age-class signature, and timer-uncertainty signature. It expires after **8 ticks**. Latest candidate membership and legality are rechecked before execution. Stale replies, disagreement, unknown, malformed model output, or exceptions retain deterministic control. Only an actual intervention logs `jev_changes_action` or `jev_veto_applied`; call count/freshness is not evidence of model benefit.

## Astra, timing, and fallbacks

Astra remains the parent's off-clock adviser: at most **three** asynchronous calls, **100-second** timeout, **300-tick** revision cooldown. Its unchanged prompt supplies policy/evidence and the local Jev Choice/Noul bridge. It returns only `style: nearest|top_first|frontier` and `review: collision_first|escape_space`. Valid changes alter the existing coverage tie-bias/review wording and invalidate Jev advice. Unsupported or malformed output preserves the current policy. Astra cannot change the guard, authorize an edge, or emit executable code.

The retained review hypothesis is that life loss, palette changes, or a 150-tick scoring stall may justify a durable coverage/review change rather than per-frame reasoning. This candidate does not claim an independent Astra benefit; the archive has not established one, and the small call cap can be consumed before late losses.

Jev launches at most once per **five ticks**, with one outstanding request and a **one-second** timeout. Archived Jev latency around 0.2 s already exceeds several ticks; Astra takes roughly a minute and has reached 94 s. Neither call is awaited by the motor. Each tick receives a deliberate post and then `ctx.wait_for_tick()`.

Fallbacks remain: neutral during flashing/falling/unanchored motion; finish the existing hop pulse and release; apply any eligible brief uncertainty guard; otherwise rank freshly legal edges deterministically, including the parent's bounded descending-ball tactical wait. Models can only select within those recomputed options. The episode runs to game/context termination, not a harness step limit.

## Development evidence and limitations

An **initial, superseded** one-minute seed-0 smoke test (`tries/seed0-234437`) reached **600 points**, 899 ticks, zero stale ticks, 173 S1 calls, no S2, and an external deadline ending. It did not validate the hypothesis. Inspection found two implementation problems: an edit had removed record expiry, retaining the old purple record indefinitely, and the broad `>=20` timer trigger also penalized ordinary visible-Coily state 30. It also changed the opening route at tick 29. These are not presented as final-controller results.

The corrected version restores expiry, excludes state 30, restricts the new deterministic behavior to after four scored clears, reduces the uncertainty preference to avoid defeating target approach, and adds explicit diagnostic events. Syntax and synthetic tests check timer distinction, record expiry, and graph-safe outputs. Tests supplement rather than replace the evaluator's garbage-model virtual-clock pre-flight. Final live results, if any, are recorded separately below; a short opening trial cannot test a post-fourth-clear mechanism.

## Confirmation and refutation

Support requires retaining the parent's early scoring/low stale rate **and** fewer losses following actual stale-branch or state-36 interventions, followed by more fifth-and-later-board scoring. A changed action alone is not confirmation; examine the ensuing landed path, score, and loss window. Only scored flash/rebase sequences count as inferred progression, not the decoder's unreliable round counters or dividing all apparent Green Ball points by 3,100.

Refute if record expiry and signature invalidation work but the same hidden-ball loss pattern persists, if uncertainty creates prolonged oscillation, or if score/survival regresses without reducing the targeted losses. State 36 could be a broad cooldown rather than an imminent spawn signal; this is a central uncertainty of the experiment. Visible Coily/corner deaths, unobserved spawn location, downward apex collisions, and optimal reversible-cycle routing remain unsolved and outside this refinement.

## Corrected-version smoke test

The final controller's **45-second** training-seed-0 trial (`tries/seed0-235753`) reached
**4,325 points**, **674 ticks**, **0 stale ticks**, **118 S1 calls**, and **0 S2 calls**, ending
at the external deadline. It passed the initial palette handoff. This is an opening smoke
test, not a measurement of the late-board guard or the official training mean.

Additionally, **315 synthetic opening-state combinations** (21 nodes, five timer values,
three hazard configurations) matched the parent's candidate action/cost/danger ranking
and produced only graph-legal actions or NOOP. Separate checks passed late-state gating,
record expiry after ten ticks, syntax compilation, and malformed Jev/Astra output handling.
No final-version full-episode or post-fourth-clear live benefit has been demonstrated.

The corrected smoke test logged two unsuccessful S1 calls; the fallback continued.
Observed Jev median latency was **0.175 s**, with a **1.028 s** maximum. The first
scored palette rebase occurred at tick **482**, tally **3,100**, with all four reported
lives. Final harness SHA-256: `43f3a848b999ae77b53da816b0da00377fc3a447ed892492c173142d144f927e`.
