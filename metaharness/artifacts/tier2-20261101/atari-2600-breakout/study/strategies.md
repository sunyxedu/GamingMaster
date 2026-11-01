# Atari 2600 Breakout — strategy prior

## What “strong” means here

The baseline is one player, ordinary point-scoring Breakout: mode 0, difficulty 0, five balls, two six-row walls, and the four-action ALE set. There is no opponent, economy, build order, rotation, or persistent upgrade. A strong agent first avoids losing a ball, then deliberately shapes the ball's horizontal path so a side opening/tunnel clears many bricks. The objective is the full **864 points**, not merely a long rally. The original manual's controls and paddle tips are the authoritative Atari-2600-specific facts; the tunnel plan is additionally supported by the published DQN Breakout result. [M1, M2]

The source manual labels the cartridge family “Breakaway” and describes ordinary Breakaway and Breakthru variants. Do not accidentally apply its Catch, Steerable Ball, Timed, Invisible, or Breakthru rules to ALE mode 0. [M1]

## Reliable game facts to encode

1. **Six rows, values 7/7/4/4/1/1.** From top to bottom the manual lists red 7, orange 7, yellow 4, green 4, aqua 1, blue 1. A complete wall is 432; after it is cleared a second wall appears; both walls are 864. There are five balls in a game. [M1]
2. **Serve state is a real state.** FIRE serves the first ball and serves another after a miss. A ball absent from RAM is not a ball at coordinate `(0,0)`; the controller must identify “waiting to serve” and issue FIRE without resetting the episode. [M1, I1]
3. **Paddle contact matters.** The manual divides the paddle into five sections. It says angles become progressively smaller after the third, seventh, and eleventh hit and return to the original angle after the twelfth; the center section makes the ball “jump.” It does not give a numerical section-to-angle table, so learn/measure the outgoing `dx` rather than inventing one. [M1]
4. **Speed changes are predictable but discrete.** The manual says the ball speeds up after the twelfth consecutive hit or when it hits a brick in any of the top three rows. Treat either event as a controller-mode change: shorten action holds and refresh predictions. [M1]
5. **A side tunnel is a proven long-horizon plan.** Mnih et al. report that DQN learned to “first dig a tunnel around the side of the wall,” after which the ball can travel behind the wall and destroy many blocks. Their extended data describes a value increase when breaking through to the upper level. This is not a requirement to tunnel on every seed; it is the default strategic prior when a safe side opening is available. [M2]
6. **ALE timing is not turn-based.** v5 uses four-frame action repeats and 0.25 sticky actions by default; the chosen high-control configuration uses one frame per step but preserves stickiness. The paper's Atari experiments used 60-Hz emulation and action selection every four frames. This game punishes delayed paddle movement, so the low-level loop cannot wait for Astra. [I1, M2]

## Opening and rally plan

### Opening: serve, observe, then choose a side

There is no universal fixed LEFT/RIGHT opening: the serve trajectory and sticky-action draws can differ by seed. The controller should:

1. Keep the paddle near the middle before serving. Issue FIRE as a short bounded retry/hold when the ball is absent, because v5 sticky actions can discard a single FIRE tick; stop retrying as soon as the ball-present predicate becomes true.
2. Use the first few frames to determine ball position and `dx,dy`; do not aim at the ball's current x when it is high in the field.
3. Choose a target side only after seeing the reachable trajectory and the current brick bitmap. Prefer the side with a reachable outer gap and enough paddle margin. If no safe tunnel is available, choose survival and clear whatever path the ball naturally offers.
4. Shape the first paddle contact toward the chosen side only if the projected interception remains safe. The manual establishes section-dependent angle control, but not the exact mapping; calibrate from observed outgoing displacement rather than hard-code “far-left always sends left.”
5. Once the ball has entered the wall, keep the paddle on the predicted return path. Do not make a large lateral sweep just because the ball is currently above the opposite side.

**Where it goes:** serve detection, initial state machine, frame bookkeeping, and safe-side selection belong in **code**. Astra can choose “attempt left tunnel / attempt right tunnel / survival” after seeing a summarized initial trajectory, but should not be asked to notice the serve one slow call at a time. Jev can answer typed `serve_required: yes/no` and `opening_target: left/right/survival` from the JSON summary.

### Low-level interception loop

The central skill is not visual reaction; it is predicting where the descending ball will cross the paddle's y band.

- When `dy < 0` or the ball is moving upward, keep enough paddle mobility to cover the next return; avoid noisy oscillation.
- When the ball is descending, use a forward projection to the paddle y, reflecting x at the left and right side walls. The projection must include the ball's width, paddle width, current speed regime, and a small uncertainty margin. It should be recomputed after a brick/paddle/wall event because `dx`, angle, or speed can change.
- Move toward the projected intercept with a deadband. Stop or counter-steer before the paddle edge; holding a direction until the ball is beside the paddle is too late. Use short pulses near an impact, longer pulses only when the ball is high and far away.
- Aim for overlap between the ball's horizontal interval and the paddle's interval, not for equality of centers. With the standard paddle, center coverage is the safe fallback. With a narrow difficulty paddle, enlarge the predicted-error penalty.
- After a contact, verify that the new ball object and `dx,dy` are present before choosing the next hold. Do not assume the commanded direction produced the desired angle under sticky actions.

**Where it goes:** all geometry, reflected-coordinate prediction, collision timing, margins, deadband, and action-hold scheduling belong in **code**. This is arithmetic and must not be delegated to Jev. Jev can classify `flight_phase` (ascending, descending, imminent, serve, lost) and choose among precomputed actions when prediction is ambiguous. Astra should only change the controller's mode or margin, not provide a pixel-by-pixel paddle trace.

A future adaptor can optionally use ALE state cloning for a tiny model-predictive search over one-to-three-frame LEFT/RIGHT/NOOP pulses. Search should score “ball still catchable plus safe margin” first and brick progress second. If sticky actions are retained, clone the RNG-containing state where supported; if not, treat the branch as a deterministic diagnostic only. This is a code-level search opportunity, not an LLM task.

### Paddle sections and controlled angles

The five paddle sections are a strategic actuator. For a safe rally, use the center/middle region and let the ball continue toward a predictable lane. For tunnel construction, choose a section that empirically produces the desired horizontal direction while retaining overlap at the next landing. Record a short calibration table keyed by speed regime and observed outgoing `dx`; do not assume that a generic Breakout clone's bounce formula matches this ROM.

The manual's angle changes mean the same-looking contact can produce a different trajectory later in a rally. Track observed paddle hits and invalidate the prediction after any collision or measured speed change. Reset the rally counter on life loss; do not reset it merely because a brick or wall was hit. The manual says the angle returns after the twelfth hit, not that the internal hit counter necessarily resets; treat any precise cycle or numerical angle table as unverified until measured. Astra can be told the current hit-count phase and decide whether to keep shaping or return to a conservative center. Jev can answer a recognition question such as `contact_phase: ordinary | reduced_angle | twelfth_reset` if the counter is included in state; it should not calculate the count.

## Tunnel strategy and adaptations

### Standard tunnel attempt

The tunnel prior is:

1. Pick the left or right outer region with a feasible route.
2. Preserve the rally while steering repeated contacts so the ball removes a vertical/near-vertical chain at that side.
3. Detect an actual opening from the brick bitmap and ball trajectory; “aiming at the side” is not enough.
4. Once the ball has passed behind/above the remaining wall, switch from tunnel construction to survival. The ball's repeated back-wall traversal can clear a large cluster without deliberate targeting.
5. Track the remaining bitmap. If the tunnel collapses or the ball returns through a dangerous lane, abandon the tunnel and catch the ball rather than forcing a second opening with one life left.

This is the strongest known long-horizon opening from the DQN literature, but the manual does not promise it and the exact Atari physics can make one side safer on a particular serve. Treat it as a prior with a fallback, not a script. [M2]

### Speed-brick adaptation

A hit in any top-three row speeds the ball, as does the twelfth consecutive paddle hit. On that event:

- switch the code controller to one-frame observations/action holds;
- reduce projection horizon and increase paddle-overlap margin;
- stop asking Jev questions that require a long response before the next tick;
- tell Astra only at a strategic boundary, such as after the current return or after a life, that the speed regime changed;
- prefer a safe broad paddle contact over an ambitious angle if the projected landing is uncertain.

Do not avoid all top-row bricks merely because they accelerate the ball: the top rows are worth 7 and the side tunnel's purpose is to reach the upper/back region. The correct tradeoff is survival margin versus fast wall clearance, not “never hit red.”

### Wall transition

When a depleted wall repopulates, increment `wall_index`, preserve cumulative score/lives, and clear the old tunnel target. A four-frame sample may miss an all-zero bitmap, so combine count jumps with score/progress history. Do not assume a wall transition requires FIRE: the manual only says a new wall appears. Continue catching an active ball; serve only if it is genuinely absent. The second wall has the same 432-point maximum. Astra may reselect the target when its advice is safe to apply; Jev can classify the code-provided transition event.

## Code, Astra, and Jev allocation

| Knowledge or behavior | Put it in code | Put in Astra prompts | Put in Jev questions |
|---|---|---|---|
| Action index mapping and FIRE-after-miss | Assert mapping at startup; finite-state serve/lost logic | None beyond knowing the state names | `serve_required` yes/no |
| Six-by-18 bitmap, row values, score, lives, wall index | Decode and maintain exact bookkeeping | Reason about whether progress is worth risk | `wall_phase`, `remaining_bricks_band` choice |
| Ball/paddle coordinates and `dx,dy` | Serialize; estimate velocity and projected intercept | None | `flight_phase`, `imminent` yes/no |
| Reflection/intercept arithmetic | Exact code; optional clone-state search | Never ask Astra for pixel arithmetic | Only choose among code-generated safe action candidates |
| Paddle five-section angle effects | Track observed outcomes and consecutive-hit phase | Choose “shape vs center/survive” mode | Classify contact/speed phase |
| Opening | Serve and safety controller in code | Select side-tunnel versus survival when enough context exists | Fast choice among left/right/survival |
| Tunnel | Detect corridor and terminate/retry attempt in code | Decide when to abandon a risky tunnel and when to reattempt after a wall/life | `tunnel_established` yes/no; `tunnel_side` choice |
| Speed changes | Detect top-three hit/twelfth hit; shorten tick/action horizon | Adapt risk policy and explain why | `speed_regime` and urgency classification |
| Model-latency fallback | Keep last safe action, center/edge safety, deadline telemetry | Never block the real-time loop | Select fallback category if asked within budget |
| Held-out evaluation | Seed, config, cumulative reward, flags, frame count | Interpret failures and revise high-level plan | Independent typed judgments for recognition/choice |

Astra's prompt should include a compact state window, recent event labels, current objective, lives, tunnel status, speed regime, and a menu of legal strategic modes. It should explicitly say: “Do not calculate x intercepts or issue a long sequence of individual paddle actions; code supplies candidates.” Astra is useful at the beginning of a life, after a wall transition, after a repeated failure mode, and when choosing whether to abandon a tunnel.

Jev's prompt should contain JSON only (plus a fixed schema definition) and request independent typed fields. Good parallel questions are:

- `flight_phase`: `serve | ascending | descending | imminent | lost | transition`;
- `interception_risk`: `low | medium | high`;
- `safe_action_choice`: one of `NOOP | LEFT | RIGHT | FIRE` from **code-provided candidates**;
- `tunnel_decision`: `continue_left | continue_right | abandon_for_survival`;
- `speed_regime`: `normal | fast`;
- `wall_transition`: boolean;
- `should_recenter`: boolean.

The state must already provide `projected_intercept_x`, `paddle_center`, `distance_to_paddle_frames`, and `action_age`; Jev should recognize and choose, not do arithmetic. Code should reject malformed or late answers and use its deterministic fallback.

## Strong-player heuristics and failure counters

### Heuristics

- **Survive first.** A seven-point top brick is not worth throwing away one of five balls. When projected interception is uncertain, center the paddle/expand overlap rather than force an angle.
- **Plan for the landing, not the present ball.** The ball may be near the opposite side while descending; tracking its current x produces systematic late moves.
- **Recenter while safe, not at impact.** When the ball is high, maintain a central or predicted-return position. When it is descending, commit early to the calculated lane.
- **Exploit a real opening.** Count bitmap occupancy and verify that the ball has crossed the wall; do not infer tunnel success from score alone.
- **Use the full state, not pixels.** The RAM decoder gives exact ball/paddle/brick geometry; the adaptor can detect events sooner and with less latency than image reasoning.
- **Replan at discontinuities.** Recompute after a wall bounce, paddle hit, brick hit, speed change, serve, life loss, or wall transition.

### Common mistakes and their countermeasures

1. **Wrong LEFT/RIGHT index.** Gym index 2 is RIGHT and 3 is LEFT; raw ALE enum numbers differ. Assert `get_action_meanings()`.
2. **Astra as reflex controller.** Seconds-to-minutes calls arrive after the ball is gone. Keep Astra strategic, Jev recognition/choice, and code in the 60-Hz loop.
3. **Long action holds.** Four-frame default repeats plus LLM delay can overshoot a narrow landing. Use `frameskip=1` and short holds near descent; log action age.
4. **Ignoring FIRE or assuming one tick is enough.** A reset starts with no active ball, every lost ball needs a serve, and 0.25 sticky actions can discard one FIRE tick. Use a bounded retry/hold and stop when the ball appears. Model “waiting to serve” separately from “dead/game over.”
5. **Following current x.** Predict the paddle crossing with wall reflections. Recalculate after brick/paddle events because the angle/speed changed.
6. **Oscillating at the paddle.** Use deadband and braking; a direction flip every tick can spend the impact window outside the overlap.
7. **Treating a life as an episode.** A miss decrements lives but the five-ball game continues. Only zero lives/natural game end or the explicit frame cap ends the episode.
8. **Misreading grouped blocks.** OCAtari's variable-width `Block` represents a contiguous run. Use the source bitmap for 108 individual cells and scoring.
9. **Trusting stale/absent objects.** Ball `NoObject`/`null` is meaningful; reset object histories after a serve and do not derive velocity from `(0,0)`.
10. **Using the OCAtari flags unexamined.** Version 2.2.1's wrapper returns `truncated` and `terminated` in the reverse positions. Use official ALE flags or correct/test the wrapper. [I2]
11. **Reward clipping or brick-count-only evaluation.** Row values differ. Sum raw ALE reward and report score; brick count is a progress diagnostic.
12. **Mixing variants.** Catch, Steerable, Invisible, Timed, and Breakthru change action/state semantics. Keep mode 0/difficulty 0 in the baseline and label any variant separately.
13. **Forcing a tunnel with the last life.** The tunnel is a prior, not a duty. Fall back to survival when the corridor is not reachable or the ball is fast.
14. **Assuming a generic clone's physics.** The manual does not provide a numeric bounce table. Calibrate paddle-section effects and speed regimes in the actual ROM; do not import an arcade or browser game's constants.

## Evaluation scenarios

- **Primary held-out seeds:** fixed mode 0/difficulty 0, five balls, same frameskip/sticky-action configuration, varied integer `reset(seed)` values. The brick wall is deterministic; seeds mainly vary ALE/serve and sticky-action randomness, so report this limitation instead of pretending that the brick layout changes.
- **Repeated-seed reproducibility:** run the same seed twice with the same package/config and compare initial RAM, event stream, score, and termination. A mismatch signals an unlogged RNG, background timing race, or ROM/config drift.
- **Latency stress:** run the same seed with Jev/Astra delayed or unavailable. Measure score loss as action age increases; verify the code fallback never blocks the emulator.
- **Optional robustness suite:** separately test difficulty 1 (narrower paddle) and selected ALE modes corresponding to cartridge variants. Do not combine these scores with baseline mode 0, and inspect/label the mode's actual game number before claiming ordinary Breakout rules.
- **No opponent scenario:** single-player Breakout has no opponents. Two-/three-/four-player manual rules are not meaningful evaluation scenarios for this adaptor.

## Sources

- **M1 — [Atari 2600 Breakout/Breakaway IV manual transcription](https://www.atariage.com/manual_html_page.php?SoftwareID=889)** and [searchable 1978 manual OCR](https://archive.org/download/Breakout_Breakaway_IV_1978_Atari/Breakout_Breakaway_IV_1978_Atari_djvu.txt). The local copy used during study is `research_manual/breakout_manual.txt`.
- **M2 — Mnih et al., “Human-level control through deep reinforcement learning,” Nature (2015), [paper PDF](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)**; see the Breakout tunnel paragraph, 60-Hz/action-repeat discussion, and Extended Data Figure 2.
- **I1 — [ALE Breakout environment documentation](https://ale.farama.org/environments/breakout/)** and [Gymnasium/ALE interface](https://ale.farama.org/gymnasium-interface/).
- **I2 — [OCAtari issue #53: swapped `terminated`/`truncated`](https://github.com/k4ntz/OC_Atari/issues/53)** and [Breakout RAM decoder source](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py).
- **I3 — [ALE Python `AtariEnv` source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/python/env.py)** and [v5 registration defaults](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/python/registration.py).
