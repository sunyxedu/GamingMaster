Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Breakout` — Atari 2600 Breakout, ordinary single-player cartridge mode 0 at difficulty 0. The objective is to keep the ball in play with the horizontal paddle and clear two six-row walls for points: rows from top to bottom are worth 7, 7, 4, 4, 1, and 1, with 18 columns per row and a manual maximum of 864 points. A game starts with five lives/balls. FIRE serves the initial ball and a replacement after a miss; a missing ball in the observation means the game is waiting for FIRE, not that the episode was reset. ALE ends naturally when the lives are exhausted and truncates at the configured frame cap.

This is a real-time interface: one tick is one native Atari frame at 60 Hz (tick_s=1/60), frameskip=1, with the v5 sticky-action probability 0.25. The runner calls tick(action) on its own clock; a late or missing decision repeats the runner's current NOOP/last action. NOOP removes directional input but does not promise an instantaneous paddle brake. Gym action names and indices are NOOP=0, FIRE=1, RIGHT=2, LEFT=3; use names, not raw ALE enum values. Malformed actions are applied as NOOP and reported in action_error.

Each observation is compact JSON RAM state: frame and global frame counters, cumulative unmodified ALE reward (score), a separate decoded cartridge_score cross-check, lives, paddle and ball rectangles, ball dx/dy when a previous ball position exists, ball_in_play/serve_required, the exact six-by-18 brick_bitmap and occupied bricks with row values, bricks_remaining, wall_index, walls_cleared, both_walls_cleared, last_reward, last_action, and termination flags. A ball is null when absent; no screenshot or display is used.

Actions:
- NOOP: Remove directional input for one native Atari frame; it does not guarantee an instant paddle stop.
- FIRE: Press the cartridge fire button to serve the initial ball or the next ball after a miss.
- RIGHT: Move the paddle right for one native Atari frame.
- LEFT: Move the paddle left for one native Atari frame.
(real-time games: ticks apply 'NOOP' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Gym indices are NOOP=0, FIRE=1, RIGHT=2, LEFT=3. The v5 baseline keeps frameskip=1 and 0.25 sticky-action probability; a tick is one native frame.

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

The game ticks every 0.016666666666666666s **on its own clock**; `ctx.act(a)` posts the current action (instant).
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
- strategy_summary: Use code for exact RAM bookkeeping, serve/life state, projected paddle interception with wall reflection, short action holds, speed-change detection, and optional clone-state search. Start with FIRE and a centered safe rally, then have Astra choose whether a reachable left or right side tunnel is worth attempting; the published DQN Breakout result supports digging a side tunnel so the ball can clear bricks behind the wall. Replan after paddle/brick/wall impacts, top-row speed increases, the twelfth consecutive hit, life loss, and wall transitions. Jev should classify flight phase, urgency, speed regime, tunnel status, and a choice among code-generated safe actions, not perform arithmetic. When tunnel construction becomes unsafe, prioritize preserving the remaining ball and use the projected landing rather than chasing the ball's current x. Keep the low-level controller running while Astra thinks and correct any late or malformed model output with a deterministic fallback.
- scoring: Report the unmodified cumulative ALE reward as the primary score because Breakout rewards brick destruction and preserves the 1-, 4-, and 7-point row values; cross-check it against the decoded cartridge score. Also report lives remaining, bricks remaining, wall index, elapsed emulated frames, and whether both walls were cleared. The Atari 2600 manual's maximum for two walls is 864 points.
- interface.observation: The official observation is a uint8 array of 128 RAM bytes. The adaptor should expose JSON rather than raw bytes: frame number; score from BCD-like RAM bytes 76 and 77; lives from RAM 57; paddle x=RAM[72]-47 with standard y=189, w=16, h=4; ball x=RAM[99]-49 and y=RAM[101]+9 with w=2, h=4 when RAM[101] != 0 and RAM[101]+9 <= 196, otherwise null; dx/dy from consecutive decoded states; and a six-row, 18-column brick bitmap decoded from RAM[0:36], with row values [7,7,4,4,1,1]. Include ball_in_play, wall_index, last_reward, terminated, truncated, lives, and episode/frame counters. OCAtari objects additionally expose category, xy, wh, rgb, dx, and dy, but its current ns_state only contains position-like values, obs_mode=obj returns a four-observation stack by default, and contiguous bricks may be grouped into variable-width blocks.
- interface.actions: Use the four Gym discrete actions: 0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT. FIRE serves the initial ball and a replacement after a lost ball; RIGHT/LEFT move the paddle and NOOP removes the directional input but does not guarantee an instantaneous stop of paddle movement. Issue actions at frameskip=1, normally in 1-3 frame holds near a descending ball, and account for v5 repeat_action_probability=0.25. The adaptor—not an LLM—projects the ball's paddle intercept, selects safe pulses, handles stickiness and late-response fallback, and stops only on natural termination or explicit truncation.

### study/strategies.md
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

…(truncated; full text in prior/strategies.md)

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

### study/brief.md
# Atari 2600 Breakout — game brief

## Scope and recommendation

Study date: 2026-10-30. This is **Breakout for the Atari 2600 (1978)**, not the arcade original, Super Breakout, or a modern clone. The baseline is the single-player ordinary points game: **`ALE/Breakout-v5`, cartridge mode 0, difficulty 0**. Retain **tier 2**: at native speed this is a reflex game. Use a headless local ALE emulator, Gymnasium for stepping/scoring, and OCAtari's **read-only RAM decoder** for symbolic observations. See `interfaces.md` for the complete adaptor specification.

## Objective, rules, and progress

- Move one horizontal paddle to keep a ball in play. The ball rebounds from the paddle and boundaries; striking a brick removes it and scores points. Missing below the paddle consumes a ball. Start with **five balls/lives**. Press FIRE to serve initially and again after a lost ball. [1, 2]
- The wall has **six rows of 18 bricks**: 108 bricks per wall. The manual specifies six rows; 18 columns are confirmed by OCAtari's decoder and the installed ROM's initial state. From top to bottom, row values are **7, 7, 4, 4, 1, 1**. The manual names these red, orange, yellow, green, aqua, and blue; rendered emulator palette colors need not match those English names. Use row indices, not inferred color names, for scoring. [1, 3]
- A complete wall is worth **432 points**. A second wall appears after the first is cleared. Clearing both gives the manual's maximum **864 points**. Track actual score, remaining bricks, wall transitions, and lives; neither survival time nor paddle hits alone is progress. [1]
- The manual describes five paddle contact sections. Bounce angles change after the third, seventh, and eleventh consecutive paddle hits, and return to their original size after the twelfth. Ball speed increases after the twelfth consecutive hit **or contact with a brick in the top three rows**. Contact near the paddle center makes the ball “jump.” These are qualitative rules, not a complete numerical physics table. [1]
- Do **not** import arcade-only facts such as an eight-row layout, three starting lives, 448-point walls, or automatic half-width paddle shrink after a ceiling hit. The cited 2600 manual does not establish that shrink rule. Its difficulty switches change paddle size; validate the actual ROM geometry before supporting nonbaseline difficulty. [1, 2, 3]
- The cartridge also offers steerable balls, catch, invisible walls, timed games, and Breakthru (ball penetrates bricks). These are **different tasks**, not seed variation for the baseline. ALE exposes modes 0, 4, 8, …, 44 and difficulties 0 and 1. Keep baseline mode/difficulty fixed. [1, 2]

## Controls

| Gymnasium minimal action index | Meaning | Intended use |
|---|---|---|
| 0 | NOOP | Stop issuing directional movement; game continues |
| 1 | FIRE | Serve a ball |
| 2 | RIGHT | Move paddle right |
| 3 | LEFT | Move paddle left |

The physical cartridge uses a rotary paddle controller and its red button. ALE maps directional inputs to paddle movement. **Minimal Gym indices are not raw ALE enum values**: raw `Action.RIGHT.value` is 3 and `Action.LEFT.value` is 4. Resolve the action mapping at startup. No mouse, display, keyboard injection, account, or network server is needed. [1, 2, 4]

## Pace, latency, and the tier distinction

Native Atari play is approximately **60 emulator frames/second**. Standard v5 repeats each selected action over four frames, approximately **66.7 ms of game time**; the proposed adaptor instead explicitly sets `frameskip=1` so a local controller can observe and act each frame. It retains the v5 sticky-action probability of **0.25**, unless a separately labeled deterministic diagnostic configuration is requested. [2, 4, 5]

Use **0.0167 seconds as the nominal one-tick decision budget** for the chosen `frameskip=1` loop. The controller may deliberately hold a safe action for 1–3 frames, and **0.05 seconds is an illustrative upper action-hold interval**, not a guarantee that an unhandled action can wait that long. Model advice is asynchronous and must be revalidated against the current state, rather than applied as a delayed reflex. Near a fast descending ball, even 0.05 seconds can be too slow: refresh reflex control every 1–2 frames and shorten action holds near impact. The actual deadline is time until the ball reaches the paddle minus the time required to move there. A 0.5-second Jev call consumes about 30 native frames; an Astra call takes many rallies' worth of time. Neither model can be the sole real-time paddle controller. Code must handle immediate interception; Jev chooses among already-computed safe tactical options; Astra changes longer-lived strategy.

**Important implementation distinction:** ALE/Gymnasium is synchronous. With `render_mode=None`, no frames advance merely because Python or a model is thinking. The adaptor must explicitly choose its timing contract. To realize the requested tier-2 experience, use a monotonic-clock background stepping loop at native speed while inference runs asynchronously. If instead it pauses between `step` calls, disclose that as a paused-emulator evaluation; there is then no wall-clock game penalty between calls. Do not describe a synchronous API as inherently autonomous. [4]

## Scoring and episode boundaries

Primary score: **sum of unmodified ALE rewards over the full game**. ALE's Breakout reward is the increase in the cartridge's displayed BCD score; read RAM score separately as a consistency check. Preserve 1-, 4-, and 7-point distinctions—do not clip rewards. Report lives left, bricks remaining, elapsed emulated frames, whether both walls were cleared, and the termination reason as secondary metrics. [6]

Do not use an “episodic life” wrapper: a lost ball is not the end of the five-ball game. Distinguish natural termination from an evaluation frame limit. The v5 registration has a **108,000-frame cap**, about 30 minutes of native game time; choose and report any shorter limit explicitly. ALE's Breakout natural termination logic is based on zero lives after play has started, not on a special 864-point victory predicate. If a task elects to finish at 864, record a separate task-completion reason rather than claiming ALE emitted that termination. [5, 6]

## Evidence status

Installed and imported in the supplied **Python 3.12.3** venv: `ale-py==0.12.1`, `gymnasium==1.4.0`, `ocatari==2.2.1`, `numpy==1.26.4`, `opencv-python==4.11.0.86`. Headless creation, seeded reset, RAM observation, object extraction, and basic action stepping succeeded. Initial observation: score 0, lives 5, paddle at `(99,189)` with decoder size `(16,4)`, absent ball before FIRE, and a full six-by-18 wall. No game-playing policy was implemented or benchmarked. Timing budgets above are engineering recommendations, not measured human reaction thresholds.

## Sources

1. [Original Atari 2600 Breakout manual, AtariAge transcription](https://www.atariage.com/manual_html_page.php?SoftwareID=889); [1978 Breakout/Breakaway IV manual OCR](https://archive.org/download/Breakout_Breakaway_IV_1978_Atari/Breakout_Breakaway_IV_1978_Atari_djvu.txt). Local OCR reference: `research_manual/breakout_manual.txt`.
2. [Farama ALE: Breakout](https://ale.farama.org/environments/breakout/).
3. [OCAtari Breakout RAM extraction source](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py), cross-checked against installed 2.2.1.
4. [ALE Gymnasium environment implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/python/env.py), cross-checked against installed `ale_py/env.py`; [Python interface](https://ale.farama.org/python-interface/).
5. [ALE environment registration source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/python/registration.py), cross-checked against installed `ale_py/registration.py`.
6. [ALE Breakout scoring/lives implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/Breakout.cpp).

### adaptor/ADAPTOR.md
# Atari 2600 Breakout adaptor

This adaptor connects the real `ALE/Breakout-v5` Atari 2600 cartridge to the
Meta-Harness tier-2 `RealtimeGame` interface. It uses the official
`ale-py`/Gymnasium environment as the sole emulator, reward, reset, seed, life,
and termination owner. The emulator runs headlessly with RAM observations; no
window, display server, GPU, account, network service, or screenshot is used.

## Configuration and connection

`game.py` calls `gym.register_envs(ale_py)` and constructs one
`ALE/Breakout-v5` environment per game instance with:

- `obs_type="ram"`, `render_mode=None`, `mode=0`, `difficulty=0`;
- `full_action_space=False`, `frameskip=1`;
- `repeat_action_probability=0.25` (the v5 sticky-action baseline); and
- `max_num_frames_per_episode=108000`.

The optional `ALE_ROMS_DIR` setting is not required with the tested
`ale-py==0.12.1` wheel, which supplies the packaged ALE ROM. If that package
cannot use its packaged ROM, set `ALE_ROMS_DIR` to a directory containing a
legally supplied, ALE-compatible ROM; do not copy a ROM into this adaptor.
The ROM's distribution and use remain subject to the applicable Atari and ALE
licenses/terms.

No credentials are read. `env.example` contains the optional ROM override.
`requirements.txt` pins the tested Python packages for the evaluator venv.

## Timing and actions

This is tier 2. `tick_s` is exactly `1/60` seconds, the native Atari cadence
for the selected `frameskip=1` configuration. `tick(action)` synchronously
applies one action and advances exactly one native ALE frame; the Meta-Harness
runner owns the wall-clock ticker and calls it at that cadence. This is
intentional: adding an adaptor-owned stepping thread would race the runner and
advance the single emulator twice. The emulator itself is protected by a
lock for evaluator/harness access.

The legal action names are `NOOP`, `FIRE`, `RIGHT`, and `LEFT`. Their Gym
indices are respectively `0`, `1`, `2`, and `3`; the adaptor asserts this
mapping at construction. `FIRE` serves an initial or replacement ball. `NOOP`
removes directional input but is not promised to brake the paddle immediately.
The runner's mailbox repeats its current/last action on stale ticks, while a
malformed direct action is converted to `NOOP` and recorded in the next
observation's `action_error` instead of raising.

## Observation and decoding

Every observation is a fresh strict-JSON-compatible dictionary under 200 kB.
It includes:

- native and global frame counters, tick count, last reward, cumulative raw
  ALE reward (`score`), and the separately decoded cartridge score;
- `score_match`, lives from official Gym info plus a RAM cross-check, paddle
  rectangle, and ball rectangle or `null` while waiting for a serve;
- ball `dx`/`dy` and a validity flag (velocity is invalid on an
  absent-to-present serve transition);
- the exact six-by-18 `brick_bitmap`, occupied brick records with row values
  `7,7,4,4,1,1`, remaining-brick count, wall index, walls cleared, and the
  two-wall completion flag; and
- action/serve/termination flags. The evaluator-only `info()` reports the
  reset seed and full configuration; the model observation does not expose the
  scenario identifier.

The normal decoder path calls the pinned, read-only OCAtari 2.2.1 Breakout
RAM helpers (`_make_block_bitmap` and `detect_objects_ram`) after every reset
and ALE step. The exact bitmap is authoritative; OCAtari's grouped variable-
width block objects are not used for counting. A small equivalent bitmap
fallback is retained for deployments where OCAtari cannot import. That fallback
preserves the Breakout bit permutation from OCAtari's MIT-licensed decoder; the
normal execution path uses the installed package rather than a second emulator.

The primary `score` is the unmodified cumulative ALE reward. The
`cartridge_score` uses the Breakout BCD RAM formula and is reported as a
cross-check rather than replacing reward. `score_breakdown()` and `info()`
also report lives, bricks, wall progress, elapsed emulated frames, and whether
both walls were cleared. A complete two-wall ordinary game is expected to
reach 864 points, but ALE's natural termination is controlled by lives; the
108000-frame cap is reported as truncation.

## Seeds and limits

`SEEDS` declares train seeds `0..7` and held-out seeds `100..107`. All are
ordinary mode-0/difficulty-0 episodes; the brick layout is fixed by the ROM,
so seed variation primarily affects the seeded serve/sticky-action random
stream rather than producing new maps. `reset(seed)` passes the integer
straight to Gymnasium/ALE and recreates the decoder state. With the pinned
configuration, a fresh instance reset with the same seed reproduces the first
RAM observation and trajectory.

Difficulty 1 and other ALE modes are intentionally not folded into these
baseline seeds: they change paddle or cartridge-variant rules and should be
run as separately labelled robustness experiments. The adaptor accepts
`mode`, `difficulty`, and other constructor settings as keyword overrides for
such diagnostics, but it requires `frameskip=1`; the published seed contract
and observation geometry describe the baseline.

There is no adaptor-owned process or server. `close()` closes the one ALE
instance, is idempotent, and is called after every evaluator episode.

### study/interfaces.md
# Programmatic interfaces for Atari 2600 Breakout

Study date: 2026-10-30. The target is the Atari 2600 Breakout cartridge represented by `ALE/Breakout-v5`, not a web game and not Super Breakout. The evaluator has no display, GPU, account, or server. The recommendation below is deliberately local, synchronous at the emulator boundary, and symbolic at the adaptor boundary.

## Decision

**Choose: `emulator` — official ALE/Gymnasium with a RAM observation, plus the OCAtari Breakout RAM decoder as a pinned, read-only symbolic decoder.**

Use the official `ale-py`/Gymnasium environment as the owner of stepping, reset, reward, termination, seeding, and timing. Use OCAtari only to decode Breakout RAM into named objects, or port the small Breakout decoder logic into the future adaptor. This is better than using OCAtari as the outer environment because the official Gymnasium step flags are correct and the official API exposes frame counters, lives, exact RAM, emulator state cloning, and configurable frameskip/sticky actions. The decoder supplies the object-level view Astra/Jev need without screen capture.

The tested evaluator-compatible pin set is:

```text
ale-py==0.12.1
gymnasium==1.4.0
ocatari==2.2.1
numpy==1.26.4
opencv-python==4.11.0.86
```

`ale-py==0.12.1` alone is sufficient for the official RAM environment. `ocatari==2.2.1` brings a research wrapper and decoder plus substantial optional dependencies; install it when using its object classes rather than reimplementing the small decoder. The supplied Python 3.12.3 venv successfully imported and smoke-tested all five pins headlessly. No game-playing policy is included here.

## Installation and credentials

Run pip through the evaluator interpreter, not system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-breakout/.venv/bin/python -m pip install \
  ale-py==0.12.1 gymnasium==1.4.0 ocatari==2.2.1 \
  numpy==1.26.4 opencv-python==4.11.0.86
```

The modern `ale-py` wheel packages Atari ROMs, including `breakout.bin`; there is no account, API key, server address, or network service. The package's ROMs and the Atari copyright are a **rights concern**, not an authentication mechanism. Use the packaged ROM only in an environment whose distribution and evaluation terms permit it. Do not copy a ROM into this repository or commit it. If the adaptor must use an evaluator-provided ROM instead, set:

- `ALE_ROMS_DIR` (optional, `required: false`): directory containing the legally supplied ALE ROM files. This is the exact override name read by `ale_py.roms.get_rom_path`; the selected ROM must match ALE's expected hash. Do not put a ROM value or path into source control.

No `DISPLAY` is required. Use `render_mode=None`; do not start Pygame or an X server. OCAtari's wheel installs Pygame as a dependency, but RAM mode does not need a window. `opencv-python` is used by the wrapper's optional image modes; it is not used for the chosen RAM path.

…(truncated; full text in prior/interfaces.md)

## Operational risks and tests for the adaptor phase

Before training or playing, assert:

- Python executable is the evaluator venv and package versions match the tested pins.
- `gym.register_envs(ale_py)` succeeds; `ALE/Breakout-v5` resolves; `obs.shape == (128,)` and `dtype == uint8` for RAM.
- `get_action_meanings()` is exactly the four-action order above.
- `reset(seed)` gives five lives, score 0, absent ball; FIRE causes a ball to appear without requiring a display.
- `episode_frame_number` advances by 1 with frameskip 1 and by 4 with frameskip 4; this catches an accidental timing change.
- A lost ball decrements lives but does not cause the adaptor to reset the whole game; FIRE is required to serve the next one.
- Natural game-over is reported as `terminated=True`, frame cap as `truncated=True` by official ALE. Add a test specifically guarding the OCAtari swap.
- RAM-derived brick count, score delta, lives, and object coordinates agree with a diagnostic RGB frame when RGB is deliberately enabled in a separate test. Production must remain `render_mode=None`.
- Report package, ROM hash/path policy, mode, difficulty, frameskip, sticky-action probability, seed, and score metrics with every evaluation.


## Measured model latency (at archive creation, 2026-10-30 14:58)
- System-2 (Astra), one-word answer: 6.7 s
- System-2, an answer of ~1201 output tokens: 90 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 209 ms (= 12.5 ticks), p90 496 ms, max 1.3 s over 12 warm requests; first request on a new connection 459 ms — a timeout below the tail turns slow answers into failed calls
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
