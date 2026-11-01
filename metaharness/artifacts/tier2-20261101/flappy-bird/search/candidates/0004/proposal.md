# Candidate 0004 — asynchronous supervisory MPC

## Parent and hypothesis

Parent: `candidates/0002-tier2_idle/`; the comparison candidate is
`candidates/0000-tier2_jev_only/`.

The idle trace establishes that a no-op is safe but dies on the ground at frame 31
with score 0.  The Jev-only trace shows a different failure: a request takes about
0.13–0.21 s (roughly 4–6 game ticks), and each returned action is held during that
wait.  On seed 0 Jev's raw-state decisions become repeated `FLAP` actions; the
terminal observation is `bird.y=-71`, `bird.vertical_velocity=5`, 40 flaps, and a
pipe collision at frame 50, still with score 0.  In other words, the prior division
asked a slow, stale classifier to perform both geometry/arithmetic and actuation.
There is no `lessons.md` in the archive; these traces are the available negative
lesson.

**Hypothesis:** score will improve only when the live loop is never blocked by
inference and the exact, observable physics is used to choose a sparse action. A
short-horizon rollout can solve the vertical timing; Jev is useful only as a
high-stakes semantic tie-breaker between code-generated alternatives, and Astra is
useful as a slow episode-level risk supervisor rather than as an actuator.

## What changed

This is a structural rewrite, not a parameter-only change:

* Code extracts a finite named state from the strict observation, selects the first
  valid pipe that is not behind the bird, computes rectangle-safe margins, and
  simulates gravity, flap impulses, pipe motion, ground contact, and both candidate
  actions. A small beam MPC evaluates sparse sequences through the active pipe and
  executes only the first action. It re-anchors on the new observation every tick,
  so model latency cannot make its plan stale.
* The main loop posts one action before the first tick and one after every observed
  tick. While models run, the code MPC remains the fallback, so a late answer costs
  no game frame. A malformed observation or arbitrary stand-in model output falls
  back to `IDLE`/the code plan without raising.
* Astra is spawned once at episode start and asked to choose only `BALANCED` or
  `CAUTIOUS`, two explicitly defined risk postures. Its parsed token is committed
  only at a pipe transition or a quiet buffer; prose, timeout, or cancellation is
  ignored. The mode is carried into the compact Jev state as a plan revision.
* Jev receives a derived state, not the raw 12-vector. It sees the active gap,
  bird margins/velocity, phase, the code MPC candidate and its rival, and qualitative
  forecast labels. It answers two independent reversed-order Choices in one request.
  Only a fresh, same-target consensus with probability at least 0.60 for both
  judgments can override the code action. Thus Jev recognizes an urgent
  candidate, but it does not calculate, search, or hold the game hostage.

The parent established the valid no-op timing and the best candidate established
that synchronous model ownership produces stale ticks and over-flapping. This
candidate tests the stronger claim that model calls should be advisory and that
code should own geometry, search, freshness, and execution.

## Evidence that would confirm or refute it

Confirmation would be a material increase in pipe score over 0, with the action
trace showing regular fresh posts and sparse flaps, while Jev/Astra timeouts do not
end an episode. A useful secondary signal is that terminal states are inside or
near a gap rather than at the ground after an all-IDLE run or above the top edge
after a held-FLAP run.

The hypothesis is refuted if the code MPC remains at score 0 across seeds despite
fresh per-tick actions, or if its physics convention systematically chooses paths
that collide while a conservative no-op lasts longer. It is also refuted as a
model-division hypothesis if accepted Jev consensus overrides consistently safe
MPC actions and lowers score; the trace events `jev_answer_dropped` and
`jev_consensus` make that failure distinguishable. Astra's mode is intentionally
low authority: if it changes performance, the next iteration should compare the
posture at pipe boundaries rather than make Astra live-act.

## Jev contract: state, questions, thresholds, and fallback

Each request's strict-JSON state has this schema:

```json
{
  "game": "flappy_bird",
  "frame": 42,
  "plan": {"mode": "BALANCED", "revision": 0, "owner": "code_mpc"},
  "phase": "approach|crossing|passed",
  "active_gap": {"top": 160, "bottom": 260, "x": 120, "width": 52},
  "bird": {"y": 205, "vertical_velocity": 3,
            "top_margin": 45, "bottom_margin": 31},
  "forecast": {
    "code_action": "IDLE", "code_label": "wide_safe_band",
    "code_min_margin": 24,
    "alternative_action": "FLAP", "alternative_label": "edge_grazing",
    "alternative_min_margin": 5,
    "code_dead": false, "alternative_dead": false
  },
  "score": 0
}
```

All arithmetic, target selection, collision checking, and candidate labels are
computed in code. `forward` asks which named candidate is safer; `reverse` asks
the same judgment with the options reversed to counter option-order bias. Both
must return `code` or both `alternate`, and both winning-option probabilities must
be at least 0.60. Confidence is not substituted for the winning probability.
Answers older than 8 frames, from a changed target or plan revision, malformed
answers, disagreements, and timeouts are discarded. The current code MPC action
is the fallback and is posted every tick; if even that result is malformed, IDLE
is legal and safe to submit.

A Jev request is launched only at a close MPC contest, a predicted urgent
alternative, or crossing, with at least an 8-frame gap between requests. It is
asynchronous with a 0.42 s timeout (chosen above the measured ~0.18 s tail but
short enough not to block). The answer can affect at most the next fresh action;
there is no stale held-model action.

## Astra role and revision trigger

Astra's prompt supplies the public physics and the initial derived state and asks
for one of two tokens. It does not see a stream and cannot post an action. Its
output is parsed conservatively and becomes a `plan.mode` field in future Jev
states and an MPC margin choice. A valid change is held until a pipe transition
or a quiet 20-frame buffer, preventing a late supervisor answer from changing a
bird already crossing a pipe. There is no per-frame Astra call. A timeout,
`RuntimeError`, cancellation, or non-token output leaves the balanced code policy
unchanged. The background worker has no loop and stops when the context closes,
so it does not outlive the episode with harness work.
