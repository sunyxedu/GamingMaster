# Candidate 0015 — act on shield erosion before cover is almost gone

**Mode:** refine. **Parent:** `candidates/0012/` (mean score 346.250).

## One hypothesis

The parent's weak shield-damage sensor often produces evidence that never reaches
its deterministic controller: `_derived()` permits an escape only when the
changed mask column has at most two occupied cells remaining (`event.exposed`).
This is a second, unvalidated condition on an already heavily filtered signal.
A recent unexplained top-entry chip might justify one short detour *before* that
column is nearly empty. Waiting for near-total erosion may discard useful early
warning when decoded bombs are missing.

**The single behavioral change is to remove that exposed-only eligibility gate.**
All recent, unused, locally relevant erosion events can now request the parent's
existing bounded escape. This is a hypothesis about weak evidence, not a claim
that a shield chip proves an incoming bomb or that intact cover is unsafe.

## Archive evidence

The two lowest-scoring parent episodes are seed 7 (190 points, 529 ticks) and
seed 3 (215 points, 577 ticks). Both terminate with `all_lives_lost`, no wave
clear, and no effective model intervention.

* `0012/trace_seed7.jsonl` records seven `shield_lane_inferred` events but only
  three `shield_escape_applied` events (frames 212, 292, 532). Later evidence
  includes `{"event":"shield_lane_inferred","frame":784,"x":107.7,
  "lost_pixels":3,"confidence":"weak"}` and another at frame 1336, x=79.0.
  Neither is followed by another escape. The life signals are
  `{"event":"life_loss_signal","frame":1128,"previous_x":86.0,
  "recent_bomb":false}` and the same event at frame 1744, previous_x=88.0,
  `recent_bomb=false`.
* `0012/trace_seed3.jsonl` likewise has seven inferred events and only four
  applied escapes. It infers x=47.7 at frame 1332 and x=46.5 at frame 1372, but
  applies no subsequent shield escape before the second life signal:
  `{"event":"life_loss_signal","frame":2008,"previous_x":94.0,
  "recent_bomb":false}`. Its earlier life signal is at frame 752, x=52.0,
  also without a recent decoded bomb.
* Every sampled observation in these two traces has an empty bomb list. The
  final controller events report `jev_changed_actions=0` and
  `astra_effective=0`. Parent code explains why non-exposed cues need model
  intervention while exposed cues get an immediate code fallback.

These traces establish unused evidence and blind losses, **not** a causal
connection between a particular chip and a subsequent death. The elapsed gaps
are too long to identify the same projectile. The experiment asks whether acting
on more eligible cues improves survival and progress overall.

Generation 4's lessons explicitly leave the erosion sensor unvalidated. This
refinement tests its action gate rather than claiming the parent's score gain
proved the sensor worked. It does not retry the refuted blocked-route extension,
infer boundaries from stationary cannon observations, or store the respawn
position as a death lane.

## What stays fixed

The module remains a complete copy of the parent's architecture, with this one
gate change and corresponding event labels/documentation:

* Tick-owned broad left/right sweeps with continuous combined FIRE; no aiming,
  fire-rate, route-boundary, or movement-speed changes.
* Shield-mask differencing, top-entry filtering, recent-player-shot attribution,
  bulk-clear rejection, and reset invalidation.
* The nearest unused cue within 28 screen pixels; one consumed erosion event
  per detour, at least 48 frames between detours, and the existing escape
  geometry. Cues expire after 180 frames by default.
* Visible-bomb trajectory checks override weak cues. Boundary guards and
  current-state safety checks remain unchanged.
* One asynchronous Jev mailbox, the same duplicated/reordered questions and
  confidence thresholds, and at most one off-path Astra duration choice.

`exposed` remains available as descriptive evidence (`remaining_cover`); it no
longer prevents the deterministic controller from considering a cue. A cue
consumed by code is not actionable again when late model advice arrives.

## System-1 interface and answer-to-action bridge

Every request uses the parent's strict-JSON schema
`space_invaders_shield_erosion_v1`:

* Episode identity/freshness: `frame`, `lives`, `wave`, plus tagged signal frame
  and x coordinate outside the model state.
* Categorical `player_lane`, committed `sweep`, and `signals`: shield-change
  attribution, source, remaining-cover category, lane side and age, and visible
  bomb-risk category. Empty bomb observations explicitly do not certify safety.
* Code-derived `safe.left`, `safe.right`, `safe.fallback_escape`, cover category,
  and a short plan carrying Astra's current signal-retention duration.

One request batches `intent_a` and `intent_b`: the same Choice with reverse
option orders over `DODGE_LEFT`, `DODGE_RIGHT`, `CONTINUE`, and `UNKNOWN`.
The questions ask only for a near-term control judgment on supplied geometry;
Jev does not compute trajectories or invent actions.

Acceptance is unchanged: both answers must agree, each selected-option
probability must be at least **0.60**, and supplied confidence must be at least
**0.25**. Missing confidence is not substituted with certainty. Reception rejects
answers older than **36 frames** or from a different life count/wave. Execution
requires an age of at most **24 frames**, the same live unused cue, matching
signal frame/x, current distance at most 28 pixels, a code-safe side, and
agreement with code's away-from-lane direction. `CONTINUE`, `UNKNOWN`, malformed
answers, disagreement, and unsafe/expired advice do not change the fallback.

Queries are asynchronous and event-conditioned, at the parent's 20-frame
cadence with urgent refresh after erosion/life signals. Only one request may be
outstanding. The **0.85 s** timeout is retained: parent p50s are about
0.17–0.20 s, and its maximum is about 0.363 s. No action waits for a response.

**Fallback order:** directly predicted bomb avoidance; then one bounded escape
from a recent unexplained local erosion cue (now regardless of `exposed`);
otherwise the committed broad sweep. FIRE stays held throughout, with the ROM
providing its single-beam lock.

## Astra and late decisions

Astra is unchanged: after the first unexplained erosion or life signal, one
background call may choose `brief`, `normal`, or `extended` retention
(96/180/300 frames). It cannot change firing, geometry, model thresholds, or the
new eligibility rule. Its output is defensively parsed, with garbage leaving
the current duration unchanged. Timeout is 55 seconds, and the call is never on
the actuator path. No runtime interface rewrite is introduced in this refine;
Astra revises only the evidence-memory parameter carried into Jev's plan.

Parent Astra calls did not affect actions before game over. This candidate does
not assume that will change. Late Jev responses are revalidated, and while
inference is pending the same tick-local fallback continues. Background work
contains no independent polling loop and is canceled/closed at episode end.

## Confirmation and refutation

Confirmation requires actual escapes on **non-exposed** cues, retained firing
throughput and near-zero stale ticks, and improved training mean together with
better survival/kill progress in the lower tail. Inspect escape events and
subsequent score/life changes, rather than treating call counts as intervention
evidence. Seed identities are diagnostic references only; runtime logic contains
no seed-specific branch, frame schedule, or replay.

Refutation would be increased direction reversals without progress, reduced
scores/survival, or unchanged behavior because non-exposed cues still fail the
existing freshness/location gates. Acting on absorbed bombs or self-inflicted
shield damage could worsen the result; one-shot consumption, attribution, and
expiry limit that failure but do not eliminate it. A score increase alone cannot
establish that any particular death was prevented.

No new game evaluation is claimed here; the evaluator will measure the single
candidate under the existing training protocol.
