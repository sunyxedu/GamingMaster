# Candidate 0008 — reusable feedback certificates and exception-only inference

## Parent and hypothesis

Parent: `candidates/0005/`. This is a structural exploration, not a claim that the parent fails to survive. It already reaches the 900-second deadline with score 716 on every archived training episode. Horizontal motion and pipe spacing are not controllable, so there is no honest policy change that promises a higher passage rate while changing neither physics nor the deadline. The remaining selection opportunities are reliability, stale ticks, and fewer System-2 calls.

**Hypothesis:** the successful controller can be converted from a continuously re-solved, continuously model-advised policy into an event-driven **feedback certificate**. The same public obstacle geometry admits a whole table of safe feedback actions, not just one action for one observed state. Reusing that table until its assumptions change, and reserving models for actual certificate violations rather than routine preference judgments, should retain deadline-limited score with less scheduling work and no routine Astra calls. This also tests the archive's explicit open question: whether the parent's rare semantic interventions are necessary at all.

This combines a control representation change and a model-authority ablation; it is not a clean estimate of the individual causal contribution of either. Runtime counters and short equivalence tests distinguish the mechanisms, but a matched follow-up would be needed if performance changes.

## Concrete evidence from the permitted archive

All parent episodes tie on score. I inspected seeds 0 and 5, including their final observations and model events; seed 5 includes a supervisor timeout and seed 0 includes one stale tick.

- Parent `eval.json`: mean score **716**, **26,999 frames** per episode, all six end by **deadline**, zero deaths, **11,074 Jev calls**, **12 Astra calls**, **2 Astra timeouts**, and stale fraction **0.000024692272553304443**. This is a wall-clock ceiling, not a repeated survival failure.
- Parent seed-0 trace, line **34081**, ends at score 716 with `stale_ticks=1`, `system1_calls=1845`, `system2_calls=2`, and final bird `y=193, v=-4`. Final clearances are **23/53 pixels**. Its final controller summary, line **34071**, still has an allowed IDLE continuation at margin 10.
- Seed 0 contains **587 accepted `jev_profile` events**, **1,258 discarded judgments**, but only **14 `jev_changes_action` events**. The first intervention is not until frame **12005** (line **14533**): LIFT changes a safe IDLE to a safe FLAP. Most successful play already has no semantic intervention.
- I separately checked every successful Jev response in the six parent traces: all **11,074** profile Choices agreed with their code-computed direction (`HIGHER→LIFT`, `LOWER→DROP`, `LEVEL/UNKNOWN→CRUISE`), including the reverse-order question. This verifies a seed-independent semantic bridge that can be compiled into the normal fast path rather than repeatedly asking Jev to recognize a deterministic label.
- Seed 0 requests Astra at frames **385** and **2770** after **15** and **12** rejected judgments. The returned `astra_plan` events (lines **2963** and **4656**) both leave the configuration at **CURRENT_GAP / margin 10**, merely incrementing its version. Both reviews occur with no physics-mismatch events in the trace. Rejected judgments alone are a poor proxy for a broken control model.
- Parent seed 5 records **1,843 Jev calls**, **603 accepted profiles**, only **8 action changes**, and zero stale ticks. Its first Astra request at frame **397** times out at frame **3999**, immediately followed by another review request. The accepted review again returns **CURRENT_GAP / margin 10**. It nevertheless reaches score 716. This separates supervisor availability from basic control success.
- The parent's initial seed-0 public observations show `(y,v)=(244,-9)` at frame 0 and `(236,-8)` after one IDLE frame. Its `physics_verified` event at frame 12 reports zero mismatches and `new_velocity`. The new harness checks this observation-to-observation relationship again at runtime.
- I read `lessons.md` and the 0003/0004 proposals and 0004 controller. 0003 tested pulse release without a controller; 0004 tested a per-frame beam search with asynchronous tie-breaking; 0005 tested per-frame integer reachability with continuous profile judgments. None tested retained time-indexed feedback certificates plus exception-only model authority. Unfinished sibling directories were not inspected.

## Structural changes

### 1. Compile a feedback tube, then index it each tick

The parent's integer-lattice reachability idea remains the safety foundation. A certificate stores a bitset of feasible vertical positions for every `(future frame, vertical velocity)` through all currently visible pipe corridors. The backward transition is exact under the verified public physics:

- IDLE: `v_next=min(v+1,10)`, `y_next=y+v_next` (or the alternate integration order if observed evidence establishes it).
- FLAP: `v_next=-9`, `y_next=y-9`.
- Safety bands use the observed bird/pipe rectangles, a two-pixel horizontal cushion, and the parent's **10,6,2,0** vertical margin sequence.

The bitset recurrence is `K[t,v] = safe[t] AND (preimage_IDLE(K[t+1]) OR preimage_FLAP(K[t+1]))`. A current action is accepted only if its next observed-lattice state belongs to the next layer. A certificate is therefore a **feedback policy over many states**, not a script of memorized flaps. State disturbances need not invalidate it when the actual new y/v is still represented.

The terminal layer is strengthened to the maximal controlled invariant in the public empty-sky band, computed by monotone fixed-point iteration. This prevents accepting a terminal position whose momentum makes eventual ceiling/ground avoidance impossible. It makes no claim about undisclosed future pipes. Newly revealed real gaps trigger rebuilding before their constraints are used.

### 2. Rebuild on changed assumptions, not every frame

Certificate identity contains the public pipe intercepts `x + speed*frame` and gap edges, bird/pipe dimensions, ground, pipe speed, observed integration order, and y-lattice fraction. It does **not** contain seed or a private pipe list. Rebuild when that identity changes, a one-step physics check fails, or the tube approaches its terminal horizon. Margin-specific tubes are built lazily. Previously unseen offscreen placeholders are ignored, as in the parent; no hidden gap is invented.

A normal tick requires two bit-membership checks rather than reconstructing every backward layer. The same conservative margin relaxation order and anchor equations are retained. The verified semantic mapping is compiled into the fast path: `HIGHER` selects the bounded LIFT anchor, `LOWER` selects DROP, and LEVEL/UNKNOWN selects CRUISE. The chosen preference is still filtered by the certificate, so this is not an unsafe action override.

### 3. Retain the proven pulse executor

The controller posts a fresh action each observed tick and calls `ctx.wait_for_tick()`. An observed previous FLAP is revoked with IDLE before search or model handling. Before posting a newly computed impulse, a second observation checks that the source frame is still current. Any straddled tick discards the old result. Optional model launches and periodic trace serialization happen after posting the action.

A certificate is not a guarantee against an arbitrarily long scheduler stall: the runner still holds the last action if the process is not scheduled. Reducing hot-path work is a hypothesis about reducing that exposure, not a change to the real-time contract.

## Model division of labour: deliberately dormant on normal play

The routine parent model path is ablated. **Expected normal operation: zero Jev calls, zero Astra calls, and no model intervention.** The normal profile bridge is now verified code: direction labels are derived from the current public state and map to the same bounded LIFT/DROP/CRUISE offsets that Jev selected in every archived parent response. This is an explicit test of whether Jev was adding information or only latency, not an attempt to present an unverified model answer as code. Models remain available only when an observed certificate assumption is contradicted.

An exception starts when either:

1. no action has a certified continuation even at zero extra vertical margin; or
2. at least three consecutive-observation physics mismatches occur within the last 90 validated comparisons.

Relaxing from margin 10 to margin 6 or 2 is normal control, not a model exception. Late/uncertain judgments do not themselves trigger more inference or a slow review.

### Jev state schema

Every exception request is self-contained:

- `rules`: passage objective and the code-owned single-frame execution/safety boundary;
- `plan`: version, CURRENT_GAP/FOLLOWING_GAP focus, and stable-arc recovery goal;
- `route`: observed target identity, APPROACH/INSIDE phase, and the code-computed HIGHER/LOWER/LEVEL/UNKNOWN priority direction;
- `motion`: RISING or FALLING_OR_LEVEL;
- `evidence`: repeated-transition-mismatch and no-certified-action-at-zero-margin booleans;
- `safety`: current certified action names and the non-overridable certificate rule.

No seed, raw arrays, hidden geometry, arithmetic task, or inaccessible history is sent.

### Questions and action bridge

One request batches:

1. A Choice over **CRUISE/LIFT/DROP/UNKNOWN**, selecting a bounded recovery preference from the named direction.
2. The same Choice with reversed options to detect order sensitivity.
3. A Noul asking whether either named exception flag actually supports a slow review.

Both Choices must agree. Each winning probability must be at least **0.80**, with supplied confidence at least **0.40**. If confidence is absent, the winning-probability threshold rises to **0.90**. Malformed/unknown answers are discarded, including non-string Choice values.

Accepted LIFT/DROP preferences use the parent's bounded anchor offsets (-6/+5 pixels) and **only** choose among the latest certificate-approved actions. In normal play these offsets come from the verified direction mapping; during an active exception a fresh Jev consensus may supply the bounded profile. If no certified action exists, the model has no authority to declare one safe: code uses the explicit last-resort CRUISE/ground/ceiling recovery rule and logs the failed certification. `profile_changes_action` records whether a change came from compiled direction or actual Jev intervention.

### Freshness, latency, and budgets

- At most one Jev call is outstanding; exception requests are at least **30 frames** apart, with a hard **24-call episode cap**.
- The asynchronous **3-second timeout** exceeds the parent's measured seed-0 max **1.338 seconds** and seed-5 max **1.283 seconds**. It is not an action deadline; control never waits for it.
- Answers must be no older than **24 frames** and match incident, target, phase, named direction, motion class, and plan version. Their lease lasts no later than request frame +30. Resolving the incident invalidates the lease immediately.
- Every ordinary, late-answer, parse-failure, and pending-answer tick runs the deterministic certified controller, including the verified direction-to-profile mapping. IDLE is only the explicit pulse release/malformed-observation action, not a replacement for sustained control.

### Astra revision hypothesis

Astra is useful only if a **persistent, corroborated certificate exception** suggests the recovery question has the wrong focus. It is not useful merely because Jev answers expired.

A review requires a fresh Noul probability at least **0.95**, an active exception lasting **30 frames**, and more than **150 seconds** of remaining budget. At most **one** asynchronous Astra review is permitted, with a **120-second timeout**. The prompt includes the relevant exact local Jev contract and current evidence. Astra may return only a JSON focus of CURRENT_GAP or FOLLOWING_GAP and a short reason. This changes the direction label and bounded recovery question on future Jev calls; it cannot alter physics, margins, observations, certificates, or the actuator. Garbage text is ignored. No model-generated code is executed.

Background tasks consist only of a single managed inference call, not custom loops. Budget and episode exceptions propagate as normal termination; ordinary model failures leave control intact. No work is joined past the episode deadline.

## Confirmation and refutation

**Primary confirmation:** all training episodes remain error-free and deadline-limited at the parent's observed score ceiling, with negligible stale ticks and **zero routine System-2 calls**. `certificate_summary` should show many more cache hits than tube builds, no unmodeled transition errors, and zero or very few exception requests.

**Model-ablation/compilation evidence:** survival to deadline with no routine semantic calls, while the code profile mapping produces the same bounded preferences as all archived Jev responses, supports the claim that normal Jev inference was latency rather than information. It does not prove that models are never useful or that unseen geometry cannot need adaptation.

**Refutation:** lower scores from premature certificate expiry, failures after newly revealed gaps, empty viable sets not seen with fresh recomputation, or larger stale fractions. Physics mismatch would invalidate the exact-model premise. If inference is repeatedly requested despite otherwise certified play, the exception trigger itself is flawed. If evaluation shows fewer calls but lower score, economy loses to the primary objective.

A larger pipe score is **not** promised; changing the ticker, deadline, seeds, collision rules, or scoring would invalidate this experiment.

## Validation status

The complete harness and proposal are written before refinement. No model backend calls, real game evaluation, held-out results, game implementation inspection, or long experiments have been used. Short syntax checks, certificate recurrence/equivalence tests, and garbage-answer stand-in contract tests will be run locally; actual benchmark performance remains for the evaluator.
