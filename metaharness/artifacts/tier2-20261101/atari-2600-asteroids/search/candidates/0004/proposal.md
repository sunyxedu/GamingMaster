# Candidate 0004 — frame-local safety controller with asynchronous model advice

**Parent:** `candidates/0002-tier2_idle/` (with `0000-tier2_jev_only` as the best reference).

## Hypothesis

The best candidate is not failing because Jev cannot recognize Asteroids. It is failing
because it gives Jev ownership of a real-time, one-frame action while Jev is much slower
than a native frame and the action remains held until the next answer. In the worst
training runs this turns one mistaken or stale judgment into a long thrust/rotation
macro: seed 10 posts `UPRIGHT` for 1,740 of 1,796 ticks and loses all four ships by
frame 1,896 for score 80; seed 4 similarly posts `UPRIGHT` for 1,741 of 1,816 ticks
and ends at score 110. Across the six traces, about 91% of ticks are stale. Seed 8 even
posts `DOWN` late, after three lives are already gone, without evidence that the delayed
hyperspace judgment was still safe. The idle parent establishes that simply waiting is
valid but earns only the passive 135-point reference.

I therefore test the structural hypothesis that the game needs a deterministic, frame-rate
local controller for safety, aiming, firing cadence, and clock ownership. Models should
choose only a bounded semantic bias over code-computed candidates. Jev's answer is no
longer a direct 14-action command; it is double-checked and expires. Astra is no longer
asked to act: it reviews a durable sweep/target/risk policy in the background and may
revise it after a confirmed life loss. When either model is late or malformed, the local
controller continues.

## What changed

* Every tick waits for the native frame, derives wrap-aware relative geometry, approximate
  closest approach, centrality, and a cardinal target heading using only the documented
  orientation labels, and posts a fresh legal action.
  This removes the parent/best-candidate held-action failure and makes model latency an
  observation-age issue rather than a control pause.
* Code owns the exact mechanics that should not be delegated: shortest orientation turn,
  target lead, bounded thrust escape, central drift correction, one-frame FIRE presses
  separated by release gaps, life/respawn reset, and a once-per-life hyperspace guard.
  Enemy coverage remains explicitly unknown.
* A background Astra call receives only compact derived facts and returns a tiny durable
  policy (`sweep` versus `target`, sweep direction, risk, emergency-only defense). One
  launch review and at most one post-life-loss review are allowed; no Astra text is
  interpreted as an action or timestamp.
* A background Jev request is made about every 24 frames when the previous request is
  finished. It batches three independent judgments: risk class and two reversed-order
  tactical-bias Choices. Jev selects among `follow_target`, short left/right sweeps,
  bounded evasions, `defend`, and `hold`; code still calculates and executes the actual
  action. Agreement and probability gates are required for nontrivial biases, and the
  bias expires after 42 frames. A failed/late/disagreeing response leaves the local
  policy intact.

## Jev contract and state schema

Each request sends a strict JSON object, not raw RAM. The schema is:

```json
{
  "frame": 123,
  "lives": 4,
  "score": 80,
  "player": {
    "present": true,
    "centrality": "central|off_center|edgeward",
    "motion": "still_or_unknown|drifting|fast_drift"
  },
  "field": {
    "rock_count": 4,
    "danger": "clear|approaching|emergency|unknown",
    "imminent_count": 0,
    "near_count": 1,
    "target": {
      "size": "large|medium|small|unknown",
      "distance_bucket": "contact_range|near|middle|far",
      "approach_bucket": "imminent|approaching|not_imminent",
      "clearance_bucket": "overlapping|tight|limited|clear",
      "heading_relation": "aligned|almost_aligned|turn_left|turn_right|half_turn"
    }
  },
  "plan": {"mode": "sweep|target", "risk": "conservative|balanced",
           "defense": "emergency_only", "sweep": "left|right"},
  "controller": {"current_bias": "...", "action_is_frame_local": true,
                  "defense_used_this_life": false}
}
```

The target fields are buckets made by code; Jev is not asked to compare distances,
count objects, infer velocity, or solve interception. `risk` asks it to classify the
already-computed danger. `bias_a` and `bias_b` ask the same tactical choice with reverse
option ordering, addressing Jev's documented first-option bias. The answer is used only
when both Choices agree and their selected-option probabilities are at least 0.18 for
ordinary aim/hold choices or 0.42 for evasions/defense. The risk label needs probability
0.55 for `emergency` (0.20 for a lower-stakes label). A disagreement, low probability,
unknown answer, timeout, or a response more than 42 current frames old is ignored. Jev
cannot lower the code's emergency threshold or repeatedly invoke DOWN.

## Fallback and confirmation criteria

The fallback is always executable without a model: preserve a central region, follow the
shortest heading to a useful rock or use a short clockwise/counterclockwise sweep, press
FIRE for one frame and release for several frames, and do not thrust in ordinary play.
An approaching or emergency closest-approach calculation locally takes precedence over
Jev and produces bounded escape steering; only an emergency gets a bounded thrust pulse.
Hyperspace is considered only for a computed
emergency plus an accepted emergency semantic answer, at most once per life. Missing
player or a respawn resets the tactical bias and firing/defense state. A malformed Astra
or Jev result is logged and cannot escape the loop.

The hypothesis is supported if this candidate survives materially longer than the best
Jev-only trace, has fewer stale ticks, and scores above the 135 idle reference while
retaining nontrivial firing. It is refuted if frame-local sweep/aiming still loses ships
at roughly the same early frames without scoring, or if its conservative target geometry
misses so many rocks that survival improves but score does not. In particular, later
iterations should compare action pulses and target-hit timing, not infer success from
survival alone; the official metric remains unclipped ALE score.
