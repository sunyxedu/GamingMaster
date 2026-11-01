# Flappy Bird strategies and controller prior

## What a strong player is solving

The bird has fixed horizontal position in the reference clone; only vertical control matters. Each pipe pair is an upcoming vertical corridor. Strong play is therefore a short-horizon feedback controller: identify the first pipe whose x-range is ahead of the bird, aim for a safe point near the gap centre, account for current vertical velocity, and issue sparse flaps early enough that the bird crosses the gap while descending gently. Do not confuse “keep the bird at the screen centre” with the real objective: the target is the next gap, and it changes after every pipe.

The basic human advice—focus on the upcoming gap, use deliberate rather than frantic taps, and avoid overcorrection—is consistent across [Flappy Bird strategy guidance](https://dopabrain.com/portal/blog/en/flappy-bird-strategy-tips-2026.html) and [How to get a high score](https://playbrain.games/blog/flappy-bird-tips-how-to-get-high-score). Those guides are practical community advice, not controlled experiments; their exact tap rates and difficulty numbers should not be copied as constants for the Gymnasium clone.

## Reference physics to exploit

The chosen environment publishes enough state to make a controller precise:

- screen 288x512 by default; bird rectangle 34x24;
- bird y increases downward;
- each frame, unless a flap was just applied, vertical velocity increases by 1 up to +10;
- flap sets vertical velocity to -9 for the next integration;
- pipes move left by 4 px/frame;
- pipe pairs have 52 px width and default gap 100 px;
- collision uses bird and pipe rectangles plus ground; the source also penalizes top contact;
- score is checked when the bird's horizontal midpoint crosses a pipe midpoint.

These are clone constants from [`constants.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/constants.py) and [`flappy_bird_env.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py), not guaranteed specifications of Dong Nguyen's proprietary binary. The environment's 12 features expose the gap edges, bird y, velocity, and rotation; use those rather than reconstructing geometry from a screenshot.

A useful mental model is discrete ballistic motion. If the bird is at vertical position `y` with velocity `v`, then a no-flap step approximately makes `v <- min(v+1, 10)` and `y <- y+v` (the exact source order applies the current velocity then increments future velocity, and a flap replaces velocity with -9). A flap is a large impulse, so repeated flaps every frame can cause a steep oscillation and ceiling collision. Code may simulate candidate short action sequences, but the model must be validated against the actual environment because collision and integration ordering matter.

## Strategy catalogue and placement

### 1. Identify the active pipe and target its corridor

**Rule:** From the ordered pipe records, choose the first pair whose right edge is not behind the bird (or more simply the nearest pipe with x ahead/overlapping the bird). Let `gap_top` and `gap_bottom` be its open boundaries; target `gap_mid = (gap_top + gap_bottom)/2`, with a safety margin from both boundaries. Once the bird has passed the current pair, advance to the next record immediately rather than continuing to chase the old gap.

**Put in code.** This is deterministic geometry/bookkeeping, not a question for Astra or Jev. Code should compute active pipe, horizontal distance, gap centre, gap half-height, and whether the bird is inside the pipe's x corridor. Send a compact derived JSON to Jev, for example:

```json
{
  "bird_y": 244,
  "bird_vy": 3,
  "target": {"dx": 92, "gap_top": 190, "gap_bottom": 290, "gap_center": 240},
  "errors": {"center_y_minus_bird_y": -4, "lower_margin": 34, "upper_margin": 30},
  "phase": "approach",
  "score": 7,
  "frames_since_last_flap": 4
}
```

Do not ask Jev to subtract or find the minimum; the model is for recognition/choice, not arithmetic.

### 2. Maintain a safe vertical band, not a single exact pixel

**Rule:** Prefer the middle of the gap, or slightly below its centre when descending, because the next flap can lift the bird and the lower boundary is usually easier to recover from than a ceiling hit. Stay at least one bird height plus a small margin from each pipe edge. The source's rectangle collision means the bird's top/bottom, not just its centre, must remain in the opening.

**Put in code + Jev.** Code computes margins and a coarse class: `TOO_HIGH`, `SAFE`, `TOO_LOW`, or `UNKNOWN`. Jev answers “flap now?” based on that class, current velocity sign, and approach phase. Astra can tune the margin after observing failures across seeds, but should not make per-frame arithmetic decisions.

### 3. Flap early; avoid panic tapping

**Rule:** If the bird is below the target band and descending, flap before the pipe is at the bird. If it is above target or rising, wait. A flap has a fixed impulse and needs several frames to play out, so last-second corrections are dangerous. Community guides explicitly warn that frantic tapping produces wild oscillations and recommend short, consistent taps; this is advice rather than a formal optimum ([source](https://dopabrain.com/portal/blog/en/flappy-bird-strategy-tips-2026.html)).

**Put in Jev questions.** Ask a typed binary question such as:

> `Given the JSON state, choose exactly one: FLAP if the bird will enter the target gap too low or is descending toward its lower edge; IDLE otherwise. Do not calculate numbers; use the supplied phase/margins/velocity labels.`

Code should enforce a minimum flap separation (for example, one or two frames) unless a hard collision forecast demands otherwise. Astra should choose and validate the separation policy during calibration; it should not override a fast decision on every frame.

### 4. Use velocity, not just position

**Rule:** The same y error can require opposite actions: a bird above the centre but falling may need no flap, while a bird below the centre but rising may also need no flap. Include vertical velocity and time-to-pipe/approach phase. A simple bang-bang policy is: flap when predicted crossing y is below the safe band, idle when predicted crossing is inside it, and reserve early flaps for large downward velocity.

**Put in code.** Code can use the published constants to roll forward a few no-flap/flap steps and mark `predicted_low`, `predicted_high`, or `predicted_safe`. This is exact simulation/bookkeeping and is more reliable than asking a language model to do arithmetic. Jev selects among precomputed actions or a small typed policy class. Astra can compare policies and adjust horizon/margins after whole-run results.

### 5. Replan at every pipe, with a short lookahead

**Rule:** The next gap, not the whole run, dominates immediate control. Keep the following one or two pipe records in context to avoid being surprised by a large vertical change, but do not commit to a long open-loop flap sequence. The [FlapPyBird-MPC project](https://github.com/philzook58/FlapPyBird-MPC#readme) demonstrates the principled version: predict ballistic motion and pipe constraints over a horizon, execute only the first action, then re-solve with new state. It also reports that fine long-horizon MILP solving is not real-time and uses coarser future timesteps.

**Put in code/Astra.** A lightweight code rollout over 5–20 frames is appropriate; full MILP/Gurobi is unnecessary for the baseline. Astra may select the rollout horizon, cost weights, and fallback policy from scored experiments. Jev should only choose the immediate typed action.

### 6. Handle the opening deliberately

At reset, the first pipes are ahead and the initial velocity is -9 in the chosen implementation. Do not immediately spam flaps: observe the first target and let the initial arc develop, then correct toward the first gap. The exact opening depends on the seeded first gap and is not a universal fixed sequence.

**Put in code:** reset bookkeeping and a short startup phase. **Put in Astra prompts:** explain that seed-specific first-gap geometry overrides a memorized opening. **Put in Jev questions:** only the same flap/idle judgment as normal, with `phase="opening"` supplied as a label.

### 7. Crossing and score bookkeeping

A point is scored when the bird's midpoint passes a pipe midpoint; in the reference source, score is incremented in a narrow crossing condition. Report `info["score"]` and also track the maximum score per seed. Do not use shaped reward as the headline metric: +0.1 per alive frame can reward hovering even when pipe progress is poor, while score counts successful obstacles. The package documents +1 for a passed pipe and -1 for death in addition to the survival shaping ([README](https://github.com/markub3327/flappy-bird-gymnasium#rewards)).

**Put in code.** Code logs score deltas, frame count, episode termination, and the active pipe index. **Put in Astra prompts:** compare policies by median/quantile pipe score over held-out seeds rather than one lucky run.

### 8. Recovery and impossible states

If a bird is already dangerously close to a pipe edge, a flap may be the only action with any chance; if the bird is above the upper edge and rising, idling is safer than another flap. If the state is outside the gap or collision has already terminated the episode, do not issue arbitrary recovery actions—end/reset and record the failure. If Jev times out, use the previous action only for a very short bounded hold and then use a code emergency rule; never pause the simulation waiting for a slow Astra response.

**Put in code:** termination check, timeout fallback, collision forecast, action validation. **Put in Astra prompts:** analyze recurring failure modes (too-high, too-low, late flap, overcorrection, stale target) at episode boundaries. **Put in Jev questions:** classify urgency (`NORMAL`, `URGENT`, `DEAD`) or select `FLAP`/`IDLE` from precomputed labels.

## Candidate controller policies

### Conservative centre-band controller (baseline)

1. Select the active upcoming pipe in code.
2. Compute a safe band inside the gap, subtracting bird height and a margin from its edges.
3. Project a short no-flap trajectory using the current y/v and pipe arrival frames.
4. If projection crosses below the band, ask Jev to flap; otherwise idle.
5. Enforce a short cooldown and update after every frame or 2-frame batch.

This should be the first implementation because it is interpretable, does not require training, and matches the published state. Astra should tune only margin, prediction horizon, and cooldown using multiple seeds.

### Target-tracking bang-bang controller

Use the sign of `target_y - bird_y` together with `bird_vy`: flap when below target and descending; idle when above target or rising. Add a deadband to prevent chatter. It is easy to implement but can fail when a pipe is close, so include horizontal phase and a lookahead forecast.

**Placement:** code for arithmetic and state machine, Jev for a typed override/recognition, Astra for tuning.

### Short-horizon search / MPC-lite

Enumerate action sequences of length 5–20 (or a small set of flap times), roll each through the known physics, reject any collision, and choose the sequence that maximizes minimum gap margin and next-pipe progress. Execute only the first action, then replan. This mirrors the idea in [FlapPyBird-MPC](https://github.com/philzook58/FlapPyBird-MPC) without requiring Gurobi. It is a code/search component; Astra can decide objective weights and horizon, while Jev is not a solver.

## Common mistakes and countermeasures

| Mistake | Symptom | Countermeasure | Owner |
|---|---|---|---|
| Frantic repeated flaps | bird oscillates into ceiling or upper pipe | cooldown/deadband; flap only on predicted lower-band violation | code + Jev |
| Chasing bird y without pipe target | survives open air then misses next gap | always include active gap and horizontal phase | code |
| Treating centre coordinate as collision | apparent safe centre but sprite clips pipe | use rectangle margins / bird height | code |
| Waiting until pipe overlaps bird | no time for -9 impulse to settle | lookahead by horizontal distance and -4 px/frame | code |
| Ignoring vertical velocity | same position produces wrong action | provide `vy` and predicted crossing class | code + Jev |
| Stale pipe after crossing | controller keeps aiming at old gap | recompute active index and log score delta | code |
| Model arithmetic in prompt | occasional nonsensical flap decisions | precompute margins, phase, and labels locally | code |
| Astra blocks live loop | many frames pass while reasoning | Astra tunes at reset/episode boundaries; Jev acts | harness |
| Optimizing shaped reward | long survival but poor pipe score | rank by pipe score first, shaped reward second | code/Astra |
| Overfitting one seed/opening | high training score, poor held-out runs | evaluate many held-out seeds and report distribution | Astra |
| Assuming original constants | clone controller transfers poorly | calibrate against the selected environment and log constants | Astra/code |

## Prompt division of labour

### Jev prompt template

Supply a compact JSON state with no raw arrays if possible:

```text
You are a fast Flappy Bird action classifier. Return exactly one token: FLAP or IDLE.
Do not do arithmetic. Use the already-computed labels and margins.
FLAP only when the bird is predicted to be too low for the active gap, or when URGENT says
an upward impulse is needed. Otherwise return IDLE. A flap is one discrete action frame.
STATE: { ... }
```

For batch calls, ask independent typed questions for each timestamp and include an ID, but do not ask Jev to plan a multi-pipe strategy. Keep the code-side safety validator authoritative.

### Astra prompt template

Give Astra episode summaries, not a live stream: seed, constants, score, frame count, termination cause, action/flap trace around the last two pipes, and aggregate results over seeds. Ask it to select one change among margin, horizon, cooldown, target bias, or fallback policy, explain the predicted failure mode, and define an evaluation test. Astra can also inspect source/API failures and write adaptor code in the next phase, but should not be the only live actuator.

## Evaluation plan

Use deterministic seeds for development and a held-out set for selection. For each seed, run until `terminated`/`truncated` or a fixed high frame cap, recording final pipe score, maximum score, alive frames, flap count, and termination cause. Report median, mean, minimum, and upper quantiles across seeds; one unusually long run is not enough. A valid improvement must not merely exploit a modified gap, reward, or collision rule. Keep `pipe_gap`, screen size, and constants fixed unless the scenario explicitly tests a variant.

## Sources and confidence

- [Gymnasium environment README](https://github.com/markub3327/flappy-bird-gymnasium#readme) — high confidence for API, actions, observations, and rewards.
- [Gymnasium environment source](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py) — high confidence for step order, pipe generation, collision, and score event.
- [Gymnasium constants](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/constants.py) — high confidence for this clone's numeric constants.
- [Flappy Bird how-to-play guide](https://flappybird.io/how-to-play) — useful for canonical controls/scoring, but the site is a modern browser copy.
- [Playbrain strategy guide](https://playbrain.games/blog/flappy-bird-tips-how-to-get-high-score) — practical timing/aiming advice; exact values and difficulty claims are clone-specific and low confidence.
- [Dopabrain strategy guide](https://dopabrain.com/portal/blog/en/flappy-bird-strategy-tips-2026.html) — practical anti-overcorrection advice; not a formal study.
- [FlapPyBird-MPC](https://github.com/philzook58/FlapPyBird-MPC#readme) — high confidence for the existence and limitations of an MPC/MILP approach; not a required dependency.
