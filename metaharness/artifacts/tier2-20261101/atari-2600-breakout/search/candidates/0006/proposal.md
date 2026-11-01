# Candidate 0006 — stateful interception and persistent control contracts

## Parent and hypothesis

Rewrite parent: `0001-tier2_astra_only`. Comparative evidence: evaluated candidate `0005`, especially its worst episode, seed 4. No sibling candidates or game implementation source were inspected.

The parent establishes that a slow model must not hold frame-level control: its six evaluated episodes all score zero, with 100% stale ticks and Astra latencies up to 198 seconds. Preserve its use of legal named actions and budgeted context inference, but replace the synchronous action loop completely.

Candidate 0005 establishes that autonomous interception works: mean 288 points with effectively zero stale ticks. Its remaining failure is therefore not adequately explained by model latency alone. **Hypothesis: treating a quantized one-frame RAM displacement as a continuous velocity creates a moving, noisy interception target; transient model biases further complicate control. A collision-aware temporal velocity estimate plus persistent, safety-checked model contracts should reduce target chatter and improve survival.** This is a testable hypothesis, not a verified identification of cartridge physics.

## Concrete archive evidence

- `0005/trace_seed4.jsonl`, tick 697: score 4, lives 5, paddle x=72, ball `(126,95)`, `dx=-1, dy=2`. Ticks 701–715 then alternate LEFT/RIGHT every tick, all marked fresh. There is no latency excuse for this chatter.
- The same trace has ball `dy=3` at scoring ticks 3435 and 3499, then `dy=4` at 3563 and `dy=3` at 3629. Single samples do not establish a constant flight slope; displacement quantization or speed changes must be distinguished with recent history.
- At tick 3629 it still has four lives and score 46; by scoring tick 3800 only three lives remain. The episode ends at tick 6821 with score 143 and zero lives despite zero stale ticks. Score-change observations are sparse: they do **not** prove the exact contact geometry of these misses.
- Its source computes `(paddle_y-ball_h-ball_y)/dy` directly from the current sample and treats every nonpositive `dy` as ascending. There is no temporal velocity estimate. Its advisory CENTER can replace the projected landing with the middle of the screen while urgency is early, and biases disappear near contact.
- `0005/eval.json` reports five failed Astra calls, four timed out, out of six calls. A 60-second strategic timeout is inconsistent with the parent’s observed latency tail. Longer asynchronous calls need not stall control.
- `lessons.md` also records two proposer deadline failures with no proposal. This candidate is written immediately rather than depending on a long experiment.

## Structural redesign

### Code: continuous perception, geometry, execution

Code owns serving and every native-frame action. It maintains up to eight recent displacement observations per axis, averages displacement over elapsed frames, preserves the last nonzero direction through zero-displacement samples, and resets the appropriate history on direction reversal, score change, observation gaps, life loss or a new ball. A three-sample estimate is a prediction-disagreement diagnostic rather than a second competing target.

The averaged flight is projected with side-wall reflection. The controller computes a safe paddle interval and travel slack, measures actual paddle displacement, and uses a modest observed-drift allowance when choosing a one-frame pulse. NOOP is not assumed to brake instantly. The established candidate's provisional reflection limits (8–150 for ball left coordinate) and paddle limits (8–144) remain explicit assumptions; this iteration does not claim to have measured exact collision boundaries or the bounce-angle table. All geometry is recomputed from public observations. No cloning, hidden state or seed-dependent rules are used.

### Astra: persistent preference rather than delayed action

Astra chooses **STABILITY** or **ECONOMY**, a general tracking preference that remains meaningful after a long inference. It never supplies paddle coordinates, a side, or executable code. The prompt explains exactly how the preference changes Jev’s contract and how code gates execution. The first call starts after 180 frames of evidence, with a 240-second timeout; at most one review follows a later life-loss/wall boundary, after at least 3600 frames. The review tests whether accumulated losses justify changing the interface preference.

Only an exact valid token is accepted. Garbage preserves the default STABILITY policy. A policy revision increments its version and invalidates any old Jev contract. Answers older than 15000 frames are ignored. This is deliberately a fixed, auditable question interface with a model-selected preference, not runtime code generation.

### Jev: choose a reusable tracking contract, not a direction

Each request includes the complete relevant context:

- `game`: mode/objective description;
- `plan`: current policy and its explicit meaning;
- `flight`: phase, prediction certainty (`settled/unsettled`), catch slack (`comfortable/tight/urgent`), speed (`ordinary/fast`), and landing lane;
- `progress`: `scoring/stalled`;
- `life_reserve`: `last_ball/multiple_balls`;
- `rules`: precise meanings and safety preconditions of the contracts.

There is no raw bitmap, numerical comparison task, or hidden reference to Astra history. Code supplies all comparisons.

One request batches three questions:

1. `mode_a`: Choice of STABLE, QUIET, UNKNOWN.
2. `mode_b`: the identical judgment with reversed option order.
3. `veto`: Noul asking whether supplied uncertainty, urgency, last-ball status or STABILITY preference makes reduced movement inappropriate.

**STABLE** centers the paddle at the current smoothed landing. **QUIET** can suppress unnecessary movement only while the paddle is already inside the freshly recomputed safe corridor and travel slack remains comfortable. UNKNOWN falls back. Neither contract can change geometry, serve, or select a speculative tunnel angle.

Acceptance requires both Choices to agree, both selected probabilities at least 0.70, both explicit confidence values at least 0.15, and matching flight epoch and policy version. Missing confidence is not certainty. QUIET additionally requires veto probability at most 0.15. Replies older than 90 frames are discarded; accepted contracts expire after 120 more frames or a collision/score event. Every tick independently overrides QUIET on uncertainty, tightness or the last ball.

Calls are asynchronous, with at most one outstanding per model. Jev starts only when the ball is descending and estimated time to contact exceeds measured model latency plus eight frames. Minimum spacing is 36 frames, increasing with observed latency. Timeout is three seconds; reserves prevent exhaustion of the call budget. Astra and Jev failures never require the controller to wait for a new action.

## Fallback and lifecycle

FIRE is posted while `serve_required` is true. An absent ball is not a reset. STABLE interception runs while inference is pending, late, contradictory, malformed or unavailable through transient errors. The main loop posts before scheduling new work and uses `ctx.wait_for_tick()` rather than spinning. Background tasks perform one context inference each, not uncontrolled worker loops. Budget/episode termination is normal; persistent backend unavailability remains visible to the evaluator.

Events record model acceptance/rejection, actual Jev-caused action changes, controller state, paddle-return detection and preceding facts on life loss. This separates semantic cooperation from merely making model calls.

## Confirmation, refutation and limitations

Confirmation: fewer rapid action reversals during settled descents, smaller prediction spread near contact, longer survival and higher mean reward than 288, without materially increased stale ticks or episode errors. Model influence is assessed through `jev_changes_action`, not just call counts.

Refutation: smoothed predictions lag discrete speed changes enough to increase misses, or fresh stable interception still loses the same number of lives. That would motivate measured collision/paddle dynamics rather than stronger semantic steering. If survival improves but reward stagnates, a later candidate should learn contact-section outcomes and then add targeted brick/tunnel planning; this iteration intentionally does not invent an angle table.

The proposer was instructed to stop and write immediately. No new emulator experiment, model call, syntax test or preflight was run for this candidate. The code and proposal are complete submissions, but performance and runtime validation remain for the evaluator. The causal diagnosis above is based on archive evidence and source inspection, not an ablation result.
