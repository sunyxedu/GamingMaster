Design a harness for a **real-time** game (tier 2).

Game: `Flappy Bird (flappy-bird-gymnasium)` — Play a headless Flappy Bird clone at one simulation frame per tick (0.033333 seconds, nominally 30 FPS). The world advances whether or not a new decision is posted. Gravity moves the bird downward by 1 px/frame² up to 10 px/frame; FLAP sets its vertical velocity to -9 px/frame, and IDLE applies no impulse. Pipes move left 4 px/frame. The 288x512 default world has a fixed 100 px pipe gap, 34x24 bird, 52 px pipe width, and ground at y=404.48. A collision with a pipe or the ground ends the episode; touching y<0 incurs a top-edge shaping penalty but is not itself terminal.

Objective: maximize the direct score, which is the number of pipe pairs passed. The environment also returns shaped reward (+0.1 for an ordinary surviving frame, +1 on a pipe passage, -1 on death, with a top-edge penalty), but shaped survival reward is diagnostic and is not the primary score. Seeds are passed to Gymnasium reset(seed); they select the seeded vertical offsets of the initial and recycled pipe pairs without changing default physics or spacing.

Observation is strict JSON. features is the package's exact 12-value unnormalized vector in feature_names order: three pipe records (last, next, next-next), each with x and the upper/lower gap boundaries, followed by bird y, vertical velocity, and rotation. For an upcoming pipe still beyond the right edge, the package reports x=screen width and gap boundaries 0 and screen height. pipes, target_pipe, bird, target_geometry, score, frame, action, and termination fields are named convenience views of that same state.

Action grammar: send the bare string IDLE or FLAP (case-insensitive); integer aliases 0 and 1 are accepted. A dictionary {'action': 'FLAP'} is also accepted for adapters that carry metadata. A valid action is held on later ticks until replaced. A missing action repeats the held action, initially IDLE. Malformed actions are ignored, recorded in last_action_error, and do not raise. The evaluator's tick counter is the number of simulation frames applied.

Actions:
- IDLE: Apply no flap impulse for this frame; gravity and pipe motion still advance.
- FLAP: Apply one upward flap impulse for this frame.
(real-time games: ticks apply 'IDLE' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Actions are one-frame commands. The runner's real-time ticker applies the held action again when no fresh action is posted. String names are case-insensitive; integer aliases 0=IDLE and 1=FLAP are accepted. Malformed actions are ignored and recorded rather than raising.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.03333333333333333s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
900 s (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals. Clock mode: wall.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Use the symbolic feature observation to select the first pipe still ahead of or overlapping the bird, compute its gap center and safe interior band, and predict a few frames of ballistic motion using the published gravity, flap impulse, and pipe speed. Flap early and sparsely when the projected bird trajectory is too low or descending toward the lower edge; otherwise idle, using a deadband and short cooldown to avoid panic tapping and ceiling collisions. Put geometry, arithmetic, rollout/search, score bookkeeping, and timeout fallbacks in code; use Jev for the fast typed FLAP/IDLE judgment and Astra for episode-level tuning of margin, horizon, cooldown, and recovery policy. Replan after each frame or short batch and advance the target immediately after a pipe crossing.
- scoring: Report info["score"] / the number of pipe pairs passed as the primary score because each successful pipe passage is the game's direct progress measure. Also log final score, maximum score, alive frames, shaped reward, flap count, termination cause, and seed for diagnostics; do not rank policies primarily by the environment's +0.1-per-frame survival shaping.
- interface.observation: With use_lidar=False and normalize_obs=False, each observation is a 12-element NumPy float array: [last_pipe_x, last_pipe_top_y, last_pipe_bottom_y, next_pipe_x, next_pipe_top_y, next_pipe_bottom_y, next_next_pipe_x, next_next_pipe_top_y, next_next_pipe_bottom_y, bird_y, bird_vertical_velocity, bird_rotation]. Pipe records are ordered by x; top and bottom values are the open gap boundaries. step additionally returns reward, terminated, truncated, and info containing score. A wrapper should serialize these values as named JSON and compute active-gap center, horizontal phase, vertical margins, and prediction labels locally.
- interface.actions: Discrete action 0 means do nothing for one simulation frame and action 1 means flap once. Submit one validated action per step; Jev selects the immediate action, while code handles arithmetic, target-pipe bookkeeping, timeout safety, collision/termination checks, and any short action hold. Do not inject keyboard events or wait for a display.

### study/strategies.md
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

### study/brief.md
# Flappy Bird study brief

## Scope and identity

This brief treats **Flappy Bird** as the original one-button arcade game and, where an executable is needed on the offline server, its faithful open-source Gymnasium implementation. The original game was made by Dong Nguyen/.Gears; it is not an online service and has no supported public game API. The Gymnasium implementation is therefore the recommended experimental target, not an assertion that its constants are the proprietary original's exact constants.

## Objective and rules

- The bird continuously moves through a fixed-width playfield while gravity makes it descend.
- A flap/tap gives the bird an upward velocity impulse; no flap is the other action.
- Green pipe pairs scroll from right to left. The bird must pass through each pair's gap without touching either pipe or the ground. In the reference implementation, going above y=0 is penalized (`-0.5`) but is not itself a terminal collision; pipe and ground collisions end the episode.
- A pipe pair passed awards one point. Collision ends the run. The conventional medal thresholds are 10/20/30/40 for bronze/silver/gold/platinum, but the adaptor should report the integer score rather than a medal.
- There is no inventory, opponent, account, or persistent progression in the core game. The meaningful objective is maximize score / number of consecutive pipe pairs survived.

The original control and scoring description is corroborated by [How to Play Flappy Bird](https://flappybird.io/how-to-play) and the historical overview in [Wikipedia](https://en.wikipedia.org/wiki/Flappy_Bird). The implementation-level API and reward description are in the [Gymnasium README](https://github.com/markub3327/flappy-bird-gymnasium#readme).

## Controls and actions

Canonical action alphabet:

| Action | Meaning |
|---|---|
| `0` | do nothing for one simulation frame |
| `1` | flap once for one simulation frame |

Human versions commonly bind tap, mouse click, Space, or Arrow Up. The [FlapPyBird README](https://github.com/sourabhv/FlapPyBird#readme) documents Up/Space and Escape. In the chosen environment, use `env.step(0)` or `env.step(1)`; do not synthesize keyboard events.

## State and scoring

The best observation for an agent is the 12-feature numerical state (`use_lidar=False`): three ordered pipe records, each with pipe x, upper-gap edge y, and lower-gap edge y, followed by bird y, vertical velocity, and rotation. With normalization enabled, x/y are divided by screen dimensions, velocity by maximum vertical velocity, and rotation by 90. The alternative is 180 normalized LIDAR distances. The source code defines these fields in [`_get_observation_features`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py#L470-L526).

For the reference implementation, the default screen is 288x512, pipe x velocity is -4 pixels per simulation frame, gravity is +1, flap velocity is -9, maximum downward velocity is +10, bird rectangle is 34x24, pipe width is 52, and ground y is 79% of screen height. These are clone constants, not claimed original-game measurements; see [`constants.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/constants.py). The default pipe gap is 100 pixels in the current Gymnasium constructor, and the eight possible gap offsets are selected by the seeded NumPy RNG; see [`_get_random_pipe`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py#L388-L403).

The environment reward is +0.1 for surviving a frame, +1.0 for passing a pipe, -0.5 for touching the top, and -1.0 for death. For evaluation, use `info["score"]` / pipe count as the primary score: survival reward is shaped training feedback, while pipe count directly measures the task. The environment returns Gymnasium's `(observation, reward, terminated, truncated, info)` from `step`, and `(observation, info)` from `reset`; see the [Gymnasium Env API](https://gymnasium.farama.org/api/env/).

## Pace and tier

This is **tier 2**. The world advances every call to `step`, and the reference implementation is configured for 30 rendered frames per second (`metadata["render_fps"] = 30`), although a non-rendered harness can advance faster than wall clock. Pipes move 4 px/frame, so a pipe can traverse the 288 px screen in roughly 72 frames (2.4 s at 30 fps). A flap impulse of -9 followed by +1 gravity gives a roughly 19-frame arc before the bird returns to comparable vertical velocity; decisions should therefore be made at frame or short-burst resolution, not after Astra spends seconds planning. In the live game, a 0.5 s delay consumes about 15 frames and can remove most of the safe correction window; use Jev for immediate flap/no-flap decisions and Astra only for strategy calibration or recovery planning.

A practical initial control deadline is **0.10–0.20 s** (3–6 frames at 30 fps) when the bird is near a pipe, with at most ~0.33 s (10 frames) in a wide-open region. `decision_latency_s` in `study.json` is a conservative 0.15. This is an engineering estimate derived from the documented 30 FPS metadata and constants, not a claim about the original game's undocumented wall-clock scheduler.

## Reset and reproducibility

Call `obs, info = env.reset(seed=seed)`. The current implementation invokes `super().reset(seed=seed)`, uses its `np_random` to choose each pipe's gap offset, and creates three pipes at x = 288, 432, and 576 before play. Use held-out integer seeds to test generalization; record seed, final score, termination cause, and frame count. `reset()` without a seed continues the RNG stream as specified by Gymnasium.

## Sources

- [Flappy Bird Gymnasium README](https://github.com/markub3327/flappy-bird-gymnasium#readme)
- [Flappy Bird Gymnasium environment source](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py)
- [Flappy Bird Gymnasium constants](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/constants.py)
- [Gymnasium API](https://gymnasium.farama.org/api/env/)
- [FlapPyBird clone](https://github.com/sourabhv/FlapPyBird)
- [How to Play Flappy Bird](https://flappybird.io/how-to-play)
- [Historical overview](https://en.wikipedia.org/wiki/Flappy_Bird)

### adaptor/ADAPTOR.md
# Flappy Bird adaptor

This adaptor implements the study-selected **local Gymnasium interface** using
`flappy-bird-gymnasium==0.4.0`. It is a headless symbolic environment: no
browser, display, account, network connection, server, or credentials are
needed. Pygame is used only as the package dependency and the adaptor sets
Dummy SDL defaults before importing it, so it runs on the offline Linux host.

## Interface and timing

`make_game(**kwargs)` returns a tier-2 `RealtimeGame` named
`Flappy Bird (flappy-bird-gymnasium)`. The runner owns the clock and calls
`tick(action)` once per frame. `tick_s` is `1/30` seconds, matching the
package's advertised 30 FPS cadence. `tick()` never sleeps and advances exactly
one Gymnasium simulation frame; `ticks` is the number of frames applied.

The action vocabulary is:

- `IDLE`: no flap impulse for this frame.
- `FLAP`: one upward flap impulse for this frame.

The action spec's `noop` is `IDLE`. Bare case-insensitive strings are the normal
grammar; integer aliases `0=IDLE`, `1=FLAP`, and `{"action": "FLAP"}` are also
accepted. A valid action becomes the held action. A missing action repeats the
held action (initially `IDLE`), as does a malformed action; malformed input is
counted and reported in `last_action_error` rather than raising from `tick()`.

## Game and observations

The default engine configuration is the package's fixed 288x512 world:

- Bird is 34x24 pixels at a fixed x near 20% of the screen.
- Gravity adds 1 px/frame², capped at 10 px/frame downward.
- `FLAP` sets vertical velocity to -9 px/frame.
- Pipes move left 4 px/frame, are 52 px wide, and have a fixed 100 px gap.
- A pipe or ground collision ends the episode; touching y<0 incurs the package's
  top-edge shaping penalty but is not itself terminal.

`observe()` returns strict, compact JSON. `features` preserves the package's
12-value unnormalised feature vector in `feature_names` order:

```text
last_pipe_x, last_pipe_top_y, last_pipe_bottom_y,
next_pipe_x, next_pipe_top_y, next_pipe_bottom_y,
next_next_pipe_x, next_next_pipe_top_y, next_next_pipe_bottom_y,
bird_y, bird_vertical_velocity, bird_rotation
```

The `pipes`, `target_pipe`, `bird`, `target_geometry`, `score`, `frame`, action,
and termination fields are named views and bookkeeping around that vector. For
a pipe still beyond the right edge, the upstream package deliberately reports
screen width and gap boundaries 0/screen height; this is not replaced with a
private pipe peek. Observations also include `valid_trial` and an infrastructure
reason if the local engine fails.

## Seeds and scenarios

`reset(seed)` creates a fresh Gymnasium environment and calls
`env.reset(seed=int(seed))`. The package's seeded NumPy RNG selects the vertical
offset from its fixed set for the initial and recycled pipe pairs. Seed does not
select a hand-authored level and does not alter the default screen, physics, or
spacing. Training seeds are `0..7`; held-out seeds are `101..108`. Both groups
have eight seeds, use the same scenario family, and are disjoint.

The default `EVAL_KWARGS` is `{"tick_s": 0.03333333333333333}`. Constructor
options `screen_size`, `pipe_gap`, `bird_color`, `pipe_color`, `background`, and
`max_ticks` are explicit robustness/time-limit experiments only; they are not
used by the main benchmark and are not changed by seed. `max_ticks` is disabled
by default so ordinary termination comes from the selected game engine.

## Score and diagnostics

`score` is the environment's direct score: the number of pipe pairs passed.
That is the primary score and is exposed also as `pipe_pairs_passed` and
`primary_score`. The package's shaped reward is intentionally diagnostic only:
it normally gives `+0.1` per surviving frame, `+1` on a pipe passage, and a death
penalty. `score_breakdown()` and `info()` record final/max score, pipe passages,
alive frames, cumulative shaped reward, survival shaping, pipe reward, death
penalty, flap count, termination cause, seed, and validity. Policies should be
ranked by pipe pairs passed, not by survival shaping.

The adaptor owns no background thread or external resource. `close()` is
idempotent and closes the Gymnasium/Pygame environment; `reset()` also closes
any prior local environment before creating a fresh one.

## Limits and provenance

This is the maintained community Gymnasium clone selected in the study, not an
official API for the withdrawn commercial Flappy Bird application. Its code and
assets are supplied by the MIT-licensed PyPI dependency; the adaptor does not
copy or distribute proprietary game assets. The wrapper reports the clone's
physics and feature-vector semantics, which may differ from any commercial
binary. The dependency is pinned in `requirements.txt`.

### study/interfaces.md
# Programmatic interfaces for Flappy Bird

## Decision and recommendation

**Chosen interface: the maintained `flappy-bird-gymnasium` package, feature-vector mode, with no renderer.** It is a local Gymnasium environment with a synchronous `reset(seed)` / `step(action)` interface, deterministic seeded pipe generation, direct symbolic observations, and no display, account, server, or credentials. It is the only option below that directly matches the offline headless constraint and lets Jev answer from a compact JSON state instead of inferring pixels. It also avoids the latency, fragility, and legal ambiguity of driving a third-party web/mobile copy.

Package source and license: [markub3327/flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium), [PyPI metadata](https://pypi.org/pypi/flappy-bird-gymnasium/json), [MIT license](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/LICENSE). The repository describes itself as based on the older [Talendar/flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) and [FlapPyBird](https://github.com/sourabhv/FlapPyBird); it is a faithful clone/environment, not an official API for the withdrawn commercial app.

## Option A — Gymnasium symbolic environment (chosen)

### What to install

Use the evaluator interpreter, not system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/flappy-bird/.venv/bin/python -m pip install flappy-bird-gymnasium
```

PyPI 0.4.0 declares Python `>=3.9` and dependencies `gymnasium`, `numpy`, `pygame`, and `matplotlib`; Python 3.12.3 is within the declared range. The package's own setup metadata is visible in [`setup.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/setup.py). For reproducible evaluation, pin the resolved package versions in the adaptor environment after testing, rather than silently using a newer release.

### Exact setup and adaptor contract

1. Install the package into the evaluator venv.
2. Import `flappy_bird_gymnasium` before calling `gymnasium.make`, so the package registers `FlappyBird-v0`.
3. Create `env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False, normalize_obs=False, audio_on=False)`. `render_mode=None` avoids display initialization; feature mode gives symbolic values and is easier for a language-model harness than 180 rays.
4. Call `obs, info = env.reset(seed=int(seed))`. Do not use the old Gym 4-tuple API.
5. Convert the NumPy array into a JSON object before sending it to Jev. Preserve the field order below and include the previous action, frame number, cumulative score, reward, `terminated`, and `truncated` in the wrapper's JSON (those wrapper fields are not supplied by the package and must be bookkeeping).
6. Ask Jev for a typed binary choice (`FLAP` or `IDLE`) from the current JSON. Validate the answer locally and map it to 1 or 0; never allow model text to become an arbitrary action.
7. Call `obs, reward, terminated, truncated, info = env.step(action)`. End the episode on either flag, call `env.close()`, and report `info["score"]`.
8. During Astra-driven calibration, use short batches of steps and inspect score/termination; during live control, do not block the environment waiting for Astra.

The official package usage example is in the [README](https://github.com/markub3327/flappy-bird-gymnasium#usage), and the return contract is defined by [Gymnasium's Env API](https://gymnasium.farama.org/api/env/).

### Observation schema

With `use_lidar=False, normalize_obs=False`, the 12-vector is:

```text
[
  last_pipe_x, last_pipe_top_y, last_pipe_bottom_y,
  next_pipe_x, next_pipe_top_y, next_pipe_bottom_y,
  next_next_pipe_x, next_next_pipe_top_y, next_next_pipe_bottom_y,
  bird_y, bird_vertical_velocity, bird_rotation
]
```

…(truncated; full text in prior/interfaces.md)

- **Legal/ToS/cost:** requires an APK, Android SDK/emulator and substantial disk/CPU; app licensing and automation/leaderboard rules apply. No account is assumed, but do not hard-code device credentials or serials.
- **Verdict:** inappropriate for offline headless evaluation unless the user supplies an authorized APK and emulator image.

## Environment variables and credentials

For the chosen Gymnasium interface, **no credentials and no environment variables are required**. Do not invent or write values. The only setup is local pip installation and the adaptor's Python import.

If a future adaptor deliberately selects an alternate interface, it must request rather than hard-code:

| Variable | Meaning | Required for chosen interface? |
|---|---|---|
| `DISPLAY` | X11 display for GUI/MSS/PyAutoGUI/Xvfb | No |
| `MSS_DISPLAY` | wrapper-specific display override, e.g. `:99` | No |
| `ANDROID_SERIAL` | ADB device/emulator serial | No |
| `PLAYWRIGHT_BROWSERS_PATH` | optional browser installation location | No |
| `GAME_URL` | authorized browser game URL | No |
| `GAME_WINDOW` | wrapper-specific capture rectangle | No |

Never place API keys, passwords, account cookies, APK signing keys, or real device credentials in this directory.

## Operational failure checks

The adaptor phase should verify, using the evaluator Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/flappy-bird/.venv/bin/python -m pip install flappy-bird-gymnasium
/home/hemlix/GamingMaster-mh-tier2/runs/flappy-bird/.venv/bin/python -c 'import flappy_bird_gymnasium, gymnasium; e=gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False, normalize_obs=False); o,i=e.reset(seed=0); print(o.shape, e.action_space, i); print(e.step(0)[0].shape); e.close()'
```

If that fails because a transitive dependency has no Python 3.12 wheel, the adaptor should first try the package's current compatible dependency versions. Only then fall back to a local symbolic implementation based on the MIT source; do not silently switch to an online GUI or old Gym package.


## Measured model latency (at archive creation, 2026-10-30 17:46)
- System-2 (Astra), one-word answer: 27 s
- System-2, an answer of ~1055 output tokens: 106 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 159 ms (= 4.8 ticks), p90 181 ms, max 228 ms over 12 warm requests; first request on a new connection 186 ms — a timeout below the tail turns slow answers into failed calls
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).

## Jev model context and usage for Meta-Harness

Snapshot: 2026-10-29 (docs.typesafe.ai: API, models, primitives, confidence, patterns, cookbooks, request
rules and the jev-1.13 known-failure page; plus the design of deployed open-source Jev agents). This is design context for the Astra proposer and for runtime Astra
when it designs Jev questions. The Python interface below is GamingMaster's wrapper, which differs
from the provider SDK. The model profile and episode budgets in the experiment manifest are
authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It reads text and structured JSON. (The provider has an experimental image
input; this wrapper does not send images, so a visual game needs a separate perception
step — code, or Astra via `ctx.think(images=[...])` — before Jev can judge its state.)
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

Model ids: `jev-latest` and `jev-preview` both resolve to jev-1.14.0 at this snapshot (released
2026-10-14; jev-1.13.0 remains available). An alias can move between calls; every answer records
the version that produced it, and the leaderboard warns when an archive mixes versions. Limits:
64k tokens per request, 32k per question; English is the primary training language.

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Known failure modes (provider's list for jev-1.13) and what to do instead

* **Literal reading.** It answers the question written, not the one meant: state the
  exact condition in `instructions` and put boundary cases in the criteria.
* **Math, counting, comparisons of numbers and dates.** Unreliable: compute them in
  code and send the result or a named bucket ("adjacent", "two cells away"), never ask
  Jev to count, add or compare magnitudes.
* **Numeric representations.** Semantic words beat raw numbers: colour names over RGB/hex
  or palette indices, "wall to the left" over coordinates. Translate in code first.
* **Indirection.** Several hops (a property of a property, double negatives) cost
  accuracy: name the relevant state field directly, e.g. `threats.nearest`.
* **Large states with irrelevant detail.** Unrelated content distracts and costs accuracy
  ("context rot"): filter in code and send only the fields this decision needs.
* **Contradictory instructions and criteria.** Keep them aligned; a Noul whose `true`
  means "no" performs worse.
* **Choice option order.** It leans toward the option listed first: do not always list a
  default/no-op first; when a decision matters, ask the same question twice with the
  options reordered (one request, two question ids) and act only if both agree.
* **Generation.** It is not a generator: turn extraction into a Choice over candidates
  that code (or Astra) produced.

### Context that each request must carry

Jev does not inherit the proposer's prompt, Astra's conversation, `ctx.memory`,
previous requests, game description or observations automatically. This wrapper
sends only the supplied `state` and `questions` with the configured model.
Construct the necessary context explicitly on every call.

Prefer a named JSON object containing the current observation summary, the
relevant rule/field meanings, Astra's current subgoal and constraints, and any
short recent evidence needed for this decision. Include candidate descriptions
when asking Jev to compare candidates. Supply actual values, not variable names
or a reference to an inaccessible file. Keep irrelevant history out; preserve
evidence that distinguishes the options. English instructions are preferred for
this model. A request must fit 64k tokens, and the state plus any single question
32k. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Patterns that deployed Jev systems share (provider cookbooks, open-source agents)

* **Code computes facts, Jev judges, code decides.** Successful controllers send a small JSON
  state of *derived facts* (distances as named buckets, what is adjacent, time to contact,
  the last action and its measured outcome, the measured age of the observation) rather than
  raw grids or numbers, then ask atomic questions and combine the probabilities in code.
* **One request, several atomic questions.** A per-tick request typically carries 3–8
  independent questions (e.g. a Choice over the legal actions, a Score of immediate danger, a
  Noul per precondition or per candidate), including speculative ones whose answers code may
  ignore; a request with many questions costs about the same latency as one with one.
* **Structured criteria.** Option descriptions as objects (`what`, `not_for`, `examples`) and
  instructions that name the state fields (`` `threats.nearest` ``) measurably beat prose.
* **Freshness and fallbacks.** Every answer is tagged with the observation / plan version it
  judged; a late answer is dropped, and an executable fallback (the last safe action, a
  conservative rule) runs while waiting. Deadlines per question are short and explicit.
* **Confidence gates per consequence.** Irreversible or lethal actions need a higher top
  probability than cheap ones; a top probability under ~0.6 is treated as "undecided" (route to
  the fallback or to System-2), not as a weak yes. Ask high-stakes Choices twice with the options
  reordered and act only when both agree.
* **Large option sets** (more than 255, or many near-identical options) are handled in two
  stages: Score or Noul each candidate independently, then one Choice over the shortlist.
* **System-2 authors, System-1 runs.** The slow model designs the state schema, the questions,
  the thresholds and the fallback policy — and re-authors them when the evidence says the
  questions are wrong — rather than being asked per tick. Harnesses that let System-2 revise
  the System-1 interface on observed failures improved most across iterations.

### Exact local call and answer contract

Import `Choice`, `Score`, `Noul` from `metaharness.models.jev`. Call
`ctx.decide(state, questions, timeout_s=5.0)`; it returns
`dict[str, System1Answer]` directly, keyed by your question IDs.

| Question constructor | Meaning of `answer.value` | Other local fields |
|---|---|---|
| `Choice(instructions, criteria={option: description, ...})` | Selected option key, a string | `.probabilities`, `.confidence` |
| `Score(instructions, criteria=[low_description, ..., high_description])` | Fractional level index, from 0 to number of levels minus 1 | `.probabilities` with string index keys, `.confidence` |
| `Noul(instructions, criteria={"true": description, "false": description})` | Probability of yes, between 0 and 1 | `.probabilities` contains `"true"`/`"false"`; `.confidence` is normally `None` |

All answers also have `.type` and `.top(k)`. There is no `.answers` container,
`.choice`, `.score`, `.noul`, generated explanation, or Score `.legend` on the
local return objects. Keep the Score level descriptions in your own code.
`Noul` criteria are optional. JSON-serializable question dictionaries with
`type`, `instructions` and appropriate `criteria` also work. Credentials and
the HTTP client are managed outside the harness; do not instantiate `JevClient`
or bypass `ctx.decide`. See the [HTTP API](https://docs.typesafe.ai/api) for the
provider's different wire format.

For Choice, define distinguishable options and include an explicit unknown or
no-match option when needed; there can be at most 255 options. Structured
descriptions can spell out inclusion, exclusion and examples when labels are
ambiguous. See [Choice](https://docs.typesafe.ai/primitives/choice).
For Score, use 2–10 concrete ordered levels for one dimension (the API rejects more
than 10; the constructor raises); the returned number is an expected level, not
necessarily an integer or a probability — use it for thresholds, not to interpolate
magnitudes. See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

What `ctx.decide` raises: `TimeoutError` when the answer (including retries of
rate-limited or transient server errors) does not arrive within `timeout_s`;
`RuntimeError` for any other failed call (e.g. a malformed question rejected by
the API); `BudgetExceeded` / `EpisodeOver` when the System-1 budget is used up or
the episode is over; `ModelUnavailable` when the key is rejected or the account is
persistently throttled (the run stops; catching it does not hide it).

### Small usage example, not a game adapter

This helper illustrates one batched request about an Astra plan and an
observation summary. The proposer still designs the summaries, questions,
thresholds, action mapping and scheduling for its candidate.

```python
from metaharness.models.jev import Choice, Noul, Score


def assess_plan(ctx, plan, observation_summary):
    state = {"plan": plan, "observation": observation_summary}
    answers = ctx.decide(state, {
        "condition": Choice(
            "Do the facts in `observation` support `plan.preconditions`?",
            {
                "supported": "All required preconditions are supported by the observations.",
                "contradicted": "An observed fact contradicts a required precondition.",
                "unknown": "The observations cannot establish all required preconditions.",
            },
        ),
        "progress": Score(
            "How much of `plan.goal` is visibly achieved in `observation`?",
            ["No visible progress.", "Some progress, but the goal is incomplete.",
             "The stated goal is visibly complete."],
        ),
        "finished": Noul(
            "Does `observation` explicitly establish completion of `plan.goal`?"
        ),
    }, timeout_s=5.0)
    condition = answers["condition"]
    return {
        "condition": condition.value,
        "condition_probability": condition.probabilities.get(condition.value, 0.0),
        "condition_confidence": condition.confidence,
        "progress_level": answers["progress"].value,
        "completion_probability": answers["finished"].value,
    }
```

### Timing and the generated bridge

One admitted `ctx.decide` request consumes one System-1 call, even with several
questions. Requests are synchronous unless submitted via `ctx.spawn`. Batching
reduces round trips but does not promise constant latency. Measure actual
latency from traces and the task's calibration: Jev can take longer than a game
tick, and the first request on a new connection (each thread keeps its own) pays
the connection setup. Keep an executable fallback while waiting, limit outstanding
requests, and associate each response with the observation and plan version that
produced it. Recheck relevant preconditions and legality on the latest observation
before acting. A cached semantic interpretation may outlive an individual action
decision; define its invalidation rule rather than treating every answer as
permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, let `ModelUnavailable` propagate, and stop
background work when `ctx.done`. The standard inference calls already log request,
response, latency and token usage. Add harness events explaining when a Jev
answer changes a chosen action, invalidates a plan, or triggers Astra; call
counts alone do not establish useful cooperation.

In `proposal.md`, specify the Jev state schema, each question's role, how its
answer affects control, freshness/confidence handling, and the hypothesis about
when Astra should revise the interface. These choices belong to the generated
harness; this guide supplies no environment-specific adapter.

When runtime Astra is asked to create or revise Jev questions, include this
guide (or its relevant interface sections) in that `ctx.think` prompt too.
The proposer context is not automatically forwarded to runtime Astra. A harness
can obtain the exact same bundled text with
`from metaharness.models.jev import usage_guide` and append `usage_guide()` to
the interface-design prompt, together with public observations and action
specifications. Do not send the full guide to Jev on every tick; send the
decision-specific state and questions it describes.
