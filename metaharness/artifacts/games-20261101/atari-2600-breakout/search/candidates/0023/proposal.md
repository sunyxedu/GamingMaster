# Candidate 0023 — stop the terminal controller at the collision horizon

## Parent and one hypothesis

Parent: `candidates/0019/`. I retain its asynchronous, code-owned 60-Hz loop,
filtered RAM velocity, reflected catch interval, fitted queued/sticky paddle
model, one-contact shaping trials, sparse-wall cleanup, and advisory Astra/Jev
bridge. The only control change in this candidate is the horizon used by the
motor's three-action receding-horizon comparison.

**Hypothesis:** the parent's `ceil(eta) + 2` motor horizon is optimizing paddle
velocity after the ball has already reached the paddle. With a delayed paddle
actuator and sticky actions, that extra tail turns a currently safe overlap into
a late reversal whose effect arrives during the collision window. Evaluating
only through the predicted collision for the last six frames should preserve
more catches without changing the tunnel/cleanup experiment.

## Trace evidence motivating the change

The parent was strong overall (mean 743.5), but its worst official episode was
seed 2 (428 points, one four-point brick left). Its `miss_window` at frame 5510
shows the controller still changing direction while the ball was already in
the paddle approach band:

- frame 5503: ball `(128,175)`, paddle x `121`, interval `[122.8,134.8]`,
  eta `3.9`, action `NOOP`;
- frame 5504: ball `(130,178)`, paddle x `125`, interval `[123.0,135.0]`,
  eta `3.0`, action `LEFT`;
- frame 5505: ball `(132,182)`, paddle x `127`, interval `[122.43,134.43]`,
  eta `1.7`, action `NOOP`;
- frame 5506: ball `(134,185)`, paddle x `125`, interval `[122.8,134.8]`,
  eta `0.9`, action `RIGHT`;
- frame 5507: ball `(136,188)`, paddle x `122`, interval `[123.0,135.0]`,
  eta `0.0`, action `RIGHT`;
- the ball then reached x `140` and was lost at frame 5510.

This is not a stale-tick problem: these are fresh native-frame actions. The
same trace has another late miss at frame 6834, where the paddle moves from x
24 to 23 to 19 to 14 while the ball descends through y 176–186 and the code
alternates LEFT/RIGHT/RIGHT. The fitted motor cannot make the sticky actuator
instantaneous. The parent proposal itself reports the unresolved failure as
“rare sticky/queued actuator excursions,” including a seed-0 miss where LEFT
was requested but existing motion continued. These observations motivate
removing the post-collision optimization tail, rather than adding more model
labels or changing the contact objective.

The change is deliberately narrower than a new braking rule: the candidate
still evaluates NOOP/LEFT/RIGHT with the parent's same actuator/sticky branches,
weights, interval-distance cost, preferred endpoint, and safety gates. Only the
number of simulated frames changes: while eta is at most six, it is
`max(1, ceil(eta))`; while the return is farther away, the parent's extra
braking tail remains (`min(10, ceil(eta)+2)`). A missing eta keeps the parent's
10-frame fallback.

## Runtime architecture

- Code reads the compact JSON observation, detects serve/life/wall/brick,
  paddle-bounce, and side/speed events, and posts a legal action every native
  frame. FIRE is posted whenever `serve_required` is true and lives remain.
- Code computes the reflected landing and paddle-left overlap interval. It
  filters the observed dx/dy cadence and uses the parent’s fitted first-order
  paddle model, including the 0.25 sticky alternatives. No model is on the
  critical path; a late answer leaves this controller running.
- The parent’s one-flight contact objective remains intact. A trial is counted
  once per real paddle contact, not once per geometry replan. At low lives,
  fast/uncertain speed, narrow intervals, or an unverified direction, the
  full catch interval wins over shaping.
- Astra supplies at most the parent’s bounded strategic LEFT/RIGHT/SURVIVE
  prior at wall/stall stages. It is parsed defensively and never emits an
  action. Jev validates a future contact objective, not arithmetic or a frame
  action. Garbage, disagreement, timeout, and stale answers are ignored.
- The loop catches normal model budget/time exceptions, continues with the
  deterministic fallback, and stops only on `ctx.done` or natural game end.

## Jev state schema and questions

Each Jev request contains strict JSON with these fields (all are derived by
code):

```text
observation:
  frame, event, lives, score, wall, bricks_remaining,
  phase, urgency, speed, landing_lane, catch_interval,
  paddle_motion, eta, fallback_action,
  rally_contract, astra_plan,
  left_edge, right_edge, left_count, right_count,
  contract_epoch, rally_version
rules:
  preserve-catch priority, legal actions, FIRE serve rule,
  model-answers-are-contact-objectives rule
```

The two batched questions are reordered `Choice`s with the same four options:

- `PRESERVE`: keep the full code-computed catch interval;
- `LEFT`: permit a safe early left-going contact preference;
- `RIGHT`: permit a safe early right-going contact preference;
- `NO_CHANGE`: leave the code-selected contract alone.

The instructions explicitly say not to calculate coordinates or output
NOOP/FIRE/LEFT/RIGHT frame actions. They tell Jev to choose PRESERVE or
NO_CHANGE for low lives, fast/unknown speed, narrow/uncertain returns, or
near/imminent urgency; directional choices only authorize the already gated
next-contact objective. The two choices must agree. A selected option must
have probability at least 0.60, match the current model/contact/wall epoch,
and be no more than 90 frames old. Current safety facts are rechecked before
acceptance. Jev acceptance changes only the next high-level contract; the
collision-horizon motor still owns every posted action.

Requests are boundary/eligibility triggered, with one outstanding request and
about a 42-frame cooldown. A 2-second timeout is normal. If Jev is unavailable,
late, malformed, or disagrees, the fallback is the deterministic full-catch
interval and the current motor action. If a malformed answer has no usable
probabilities, it is not treated as confident.

## Astra and answer-to-action bridge

Astra receives the same derived state plus the brick-edge buckets and returns a
short strategic token. The parser accepts only an unambiguous PRESERVE/LEFT/
RIGHT/SURVIVE token; all other prose is discarded. Its answer becomes a prior
for Jev and is invalidated on a wall revision. It cannot bypass life, speed,
open-edge, corridor-width, or collision-horizon safety gates. Astra is called
asynchronously at the parent’s limited wall/stall stages, with the existing
72-second timeout; the game continues while it thinks.

The final bridge is:

```text
Astra token -> Jev contact Choice -> code safety gates -> shaped catch interval
-> collision-time receding-horizon motor -> legal frame action
```

For every frame the motor simulates candidate NOOP/LEFT/RIGHT through the
same possible queued and sticky inputs as the parent. The targeted refinement
is that a near collision is a terminal objective: its rollout ends at
`ceil(eta)`, rather than spending two additional simulated frames trying to
settle post-contact paddle velocity. Distant returns retain the tail so the
controller can brake early. If the interval or velocity is unavailable, the
home/center catch fallback and FIRE serve logic remain authoritative.

## Confirmation or refutation

The hypothesis is supported if, relative to 0019, the controller has fewer
late `miss_window` records whose preceding `contact_tick`s contain direction
reversals while eta is below six, while maintaining comparable confirmed
contacts and increasing the number of completed second-wall bricks. In
particular, the seed-2 style sequence should stop issuing a reversal solely
because the simulated paddle would drift after the collision.

It is refuted if misses occur at the same or higher rate despite the shorter
horizon, if paddle positions arrive outside the catch interval earlier because
there is insufficient braking, or if survival improves but tunnel/cleanup
progress falls back toward the 0015/0017 range. If refuted, the next experiment
should measure actual collision ordering or actuator state; it should not grant
Jev direct frame control.
