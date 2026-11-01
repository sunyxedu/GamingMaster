# Candidate 0005 — asynchronous interception with advisory models

## Parents

- `candidates/0000-tier2_jev_only/`: the best scored parent, mean 3.333, because Jev occasionally selected a direction that happened to contact bricks.
- `candidates/0002-tier2_idle/`: the robustness baseline, which never posted a harmful direction and therefore ran until the deadline with five lives, but scored 0.

## Failure diagnosis

The Jev-only parent is not failing because it cannot recognize Breakout at all; it is failing because a synchronous model call becomes the real-time controller. The seed-0 trace shows a Jev answer at frame 22 with ball `(69,131)`, `dx=-1`, `dy=1`, paddle `x=99`, followed by `RIGHT`; the next model answer is not until frame 33. The trace then contains stale runs of 18–25 repeated `RIGHT` ticks. Near the end of seed 0, the model sees frame 774 with the ball already at `(82,188)` and paddle `x=144`, still answers `RIGHT`, and the action remains stale through the final miss. Seed 1 has the same shape: 22–27 repeated `LEFT` or `RIGHT` ticks, and ends with paddle `x=144` while the ball is descending.

The aggregate confirms the mechanism: the parent posts only 477 fresh actions across 6 episodes, with a 92.6% stale-tick fraction. It reaches only 2–4 points and loses all lives before clearing the first wall. In contrast, the idle parent demonstrates that a no-op loop can be error-free and preserve the episode, but it never serves the ball and scores 0.

## Hypothesis

The primary bottleneck is action freshness and paddle inertia, not the absence of a high-level game plan. A deterministic controller that predicts the descending ball's paddle intercept and posts one action every native frame should convert idle-style robustness into actual survival and scoring. Jev should only choose a short-lived shaping bias among code-generated alternatives; Astra should only choose whether a reachable side tunnel is worth attempting. Neither model should be allowed to block the 60-Hz loop or directly control an action after its judged state is stale.

This is a structural crossover rather than a parameter tweak: the real-time control authority moves from synchronous Jev to code, while both models become asynchronous advisors.

## What changed

`harness.py` implements:

1. A reflected-coordinate projection from the current ball position and `dx/dy` to the paddle's y band, using the decoded paddle width and a deadband.
2. A deterministic `FIRE` state machine: serve whenever `serve_required` is true, without treating a missing ball as an episode reset.
3. A per-tick fallback that chooses `LEFT`, `RIGHT`, or `NOOP`, and posts it before waiting for the next tick. It therefore does not repeat a model action for the duration of a 0.2–0.5 second Jev call.
4. Asynchronous Jev requests every 18 or more frames. The compact state contains only derived facts: flight phase, projected landing lane, landing urgency, paddle relation, speed regime, event, side-opening status, and the current fallback action.
5. Two reordered Jev `Choice` questions. A steering bias is accepted only when both answers agree, the selected value is legal, and its probability is at least 0.45. The answer expires after 36 frames, after a life/wall/brick event, or when the landing becomes near/imminent.
6. At most two asynchronous Astra calls. Astra returns one token—`LEFT`, `RIGHT`, or `SURVIVE`—for a long-lived tunnel preference. The code caps its influence to a seven-pixel bias and never lets it override a near or imminent catch.
7. Defensive parsing and exception handling so arbitrary stand-in model text, malformed typed answers, timeouts, and transient model failures leave the deterministic fallback running.

## Jev interface and control policy

### State schema

Each Jev request receives a strict JSON object with:

- `game`: fixed mode/difficulty description;
- `observation`: `frame`, `lives`, `score`, `event`, `serve_required`, `flight_phase`, `landing_lane`, `landing_urgency`, `paddle_relation`, `speed_regime`, `landing_x`, `eta`, `fallback_action`, `strategy`, `left_open`, `right_open`, `bricks_remaining`, `wall_index`, and `walls_cleared`; and
- `rules`: legal action names, the serve rule, and the priority to preserve catchability before shaping.

All numeric comparisons and reflected geometry are computed in code. Jev sees named buckets rather than a raw 6-by-18 bitmap or an arithmetic task.

### Questions

The request batches two independent `Choice` questions:

- `steering_a`: choose `TRACK`, `CENTER`, `BIAS_LEFT`, or `BIAS_RIGHT`.
- `steering_b`: the same options in reverse/rotated order to reduce first-option bias.

`TRACK` follows the code projection. `CENTER` is conservative far from the paddle. The two bias options are small strategic shaping preferences only.

### Thresholds and freshness

- `TRACK` is the effective safety default when landing urgency is `near` or `imminent`.
- A Jev answer is used only when both questions agree and the weaker selected-option probability is at least `0.45`.
- An answer is discarded after 36 frames, on a life loss, brick score event, wall transition, strategy change, or plan-version mismatch.
- Jev is never asked while a serve is required; code posts `FIRE` directly.
- If Jev is late, times out, returns malformed data, or disagrees across the two questions, the code projection remains in control.

### Fallback

The fallback is always executable without either model. For a descending ball it reflects the horizontal trajectory to the paddle band, aims the paddle center at that intercept, clamps to the legal paddle range, and uses a deadband to choose `LEFT`, `RIGHT`, or `NOOP`. For an ascending ball it keeps the current projected target or a conservative center/strategic target. A missing ball with `serve_required` is handled by `FIRE`; a missing ball without serve permission uses `NOOP`.

## Astra revision trigger

Astra is not consulted per tick. It is launched in the background after the first live-ball observation and can be relaunched after a life loss or wall transition. Its prompt explicitly says that the fast controller owns geometry and asks for only a one-token strategic choice. A future revision should change this interface only if traces show one of two things:

- the geometric fallback survives but scores poorly because it never shapes a reachable side tunnel; or
- Jev biases are frequently accepted yet correlate with misses, indicating that the derived buckets or confidence gate are misleading.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a large drop in stale ticks, materially longer survival, and scores above 3.33 without game errors; useful intermediate evidence is a paddle that remains near the projected landing instead of saturating at `x=8` or `x=144`. The hypothesis is refuted if the fallback still loses lives quickly despite fresh per-tick posts, or if survival improves but score remains near zero across seeds, showing that the Atari paddle's actual movement/angle dynamics require a more explicit measured-action model or a stronger tunnel planner.
