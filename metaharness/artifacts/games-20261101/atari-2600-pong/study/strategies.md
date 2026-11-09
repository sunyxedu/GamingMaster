# Atari 2600 Pong strategy prior

## What “strong play” means in this ROM

The Atari 2600 target is a paddle game, not a modern mouse/joystick Pong clone. The manual's reliable facts are: keep the paddle aligned with the rally, score when the opponent misses or hits out of bounds, win at 21, and use the red button only for the feature enabled by the selected variation. The tactics below combine those facts with geometric Pong reasoning and should be validated against the exact ALE ROM/variation before becoming hard-coded assumptions.

The canonical sources are the [Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587), its [Internet Archive scan/OCR](https://archive.org/stream/Video_Olympics_1978_Atari/Video_Olympics_1978_Atari_djvu.txt), the [ALE Pong specification](https://ale.farama.org/environments/pong/), and the [FujiNet Video Olympics reverse-engineering notes](https://raw.githubusercontent.com/FujiNetWIFI/fujinet-2600-video-olympics/main/PORTING.md).

## Core defensive policy: intercept, do not chase

1. Track the ball's current position and velocity sign.
2. If it is moving toward the right paddle, project its Y coordinate to the right collision X, reflecting the Y component at the top/bottom playfield boundaries. Move the paddle center toward that projected intercept.
3. If it is moving away, return toward a central/coverage position rather than following its current Y. This leaves room for the next reflected trajectory and avoids getting stranded at the top or bottom.
4. Stop or reverse before the paddle overshoots the predicted contact zone. A paddle that is already overlapping the ball's Y at contact should hold `NOOP`; unnecessary oscillation can turn a safe contact into a miss.
5. Use a short action horizon (one frame or a few frames) near contact and a longer macro only when the ball is far away. This is code, not an Astra/Jev decision.

The wall-reflection model is a geometric prior, not a claim that ALE publishes exact ball physics. Establish the playfield bounds, paddle extent, ball speed, and action-to-Y calibration from observed frames/RAM. A direct screen-based object detector is useful during calibration even if production uses RAM.

**Implementation placement:** exact intercept arithmetic, boundary reflection, hysteresis, paddle limits, and action timing belong in **code**. Astra can choose the target policy and diagnose a bad physics model. Jev can answer fast typed questions about phase/approach (“toward right paddle?”) or whether the current paddle is above/below the projected target, but should not calculate the intercept.

## Contact and return-angle tactics

The manual documents two button tactics:

- **Speed:** press the red controller button as the ball makes contact to add speed to the return ball.
- **WHAMMY:** press at contact to put sharper angles on return hits.

The manual also documents CATCH and JUMP for other variations, but they are not automatically available in Robot Pong or standard Pong. Do not press `FIRE` continuously: it can serve, activate a feature at the wrong time, or alter the selected variation's behavior. Detect a predicted collision window and gate `RIGHTFIRE`/`LEFTFIRE` to that window only when the variation matrix says the feature is enabled.

A general Pong prior is that contact near the paddle center is the safest defensive return, while contact toward the upper/lower portion creates a steeper trajectory. Treat this as a measured ROM hypothesis: the manual promises sharper WHAMMY angles but does not publish an exact contact-angle equation. Before using edge targeting, run controlled rallies and label outgoing Y velocity versus contact offset. If the opponent is weak at extreme angles, choose a steep return; if the ball is already fast or the paddle is poorly positioned, prefer a safe center contact.

**Implementation placement:** button timing, contact-window detection, and contact-offset bookkeeping belong in **code**. Astra should decide whether to trade safety for an attacking angle after observing the opponent. Jev can classify “contact now / not now,” “center/edge opportunity,” or “use feature / do not use feature” from the JSON state.

## Robot Pong (one-player) strategy

Robot Pong is Games 1–2 in the manual: the human controls the right paddle and the computer controls the left. The manual says the computer serves to the player who lost the previous point. Start-of-rally behavior is therefore a **serve state**, not a normal tracking state: wait for the legal serve interval and press `FIRE` only when the adaptor says the serve is ready.

The computer paddle is a predictable opponent compared with a human. During calibration, record how closely its paddle tracks the ball, whether it lags at steep trajectories, and whether a chosen difficulty changes only paddle size or also behavior. Prefer repeatable sharp/edge returns only after confirming that the computer fails to cover them; otherwise keep the paddle centered and win by returning every ball. Do not claim a universal Robot Pong exploit without testing the exact ROM/configuration.

**Implementation placement:** serve-state machine, score/termination bookkeeping, and opponent-trajectory logging belong in **code**. Astra can choose between safe rallying and an exploit after a few observations. Jev can recognize “computer has drifted above/below the projected intercept” or “ball is approaching the right side.”

## Standard Pong and multi-paddle variations

If the adaptor exposes the original cartridge rather than only ALE's one-player Pong target, preserve the variation number:

- Games 1–2: Robot Pong, one player.
- Games 3–4: two-player Pong.
- Games 5–8: four-player Pong/Pong 4-I.
- Games 9–12: Super Pong/Super Pong 4, with two paddles per side.

In four-player forms, the manual's natural division is upper/lower court coverage; partners should not both chase the same intercept. In Super Pong, treat each side's two paddles as a coordinated pair and assign a target zone. These are higher-level roles for Astra; the low-level controller still projects each active paddle's needed position.

## Difficulty and feature controls

The manual says Difficulty A halves paddle size relative to B. Include this in the state and strategy prompt. A small paddle increases the value of early predictive movement and makes late reactive correction unreliable; a full paddle allows more conservative central positioning.

Do not infer that ALE's `difficulty` integer maps one-for-one to the cartridge's physical A/B switch without checking the environment. Record both the ALE constructor setting and the observed paddle size. Likewise, ALE's `FIRE`, `RIGHTFIRE`, and `LEFTFIRE` names are generic Atari action names. Use the manual's variation matrix to decide whether the selected form offers Speed or WHAMMY.

## Common mistakes and counters

| Mistake | Symptom | Counter / owner |
|---|---|---|
| Treating the game as joystick up/down | Paddle moves in an unexpected axis or direction | Run action calibration; code stores the mapping |
| Chasing the current ball instead of future contact | Paddle arrives early/late after a wall bounce | Code projects trajectory; Jev only classifies approach |
| Holding a direction through contact | Paddle overshoots the intercept | Add target hysteresis and `NOOP` braking |
| Waiting for Astra to choose every frame | Missed collision during a model call | Jev/code local controller runs continuously; Astra sets macro policy |
| Pressing FIRE away from contact | Unwanted serve/feature/catch behavior | Variation-aware contact/serve state machine in code |
| Ignoring the one-second serve rule | Serve does not start or input is discarded | Track point time and legal serve window |
| Assuming all Pong variants are identical | Wrong number of paddles or wrong feature | Preserve game number/mode in reset state |
| Forgetting Difficulty A shrinks the paddle | Same controller target misses at high difficulty | Calibrate paddle height and widen prediction margin |
| Using only screenshots in production | OCR/segmentation jitter and latency | Use RAM-derived state, RGB only for validation |
| Assuming `seed` creates meaningful random openings | Held-out episodes repeat | Vary mode/difficulty or explicit initial states; test reset behavior |
| Reading conventional TIA input addresses | Paddle parser appears frozen/wrong | Validate ROM-specific map; FujiNet notes warn about `$38-$3B` mirrors |
| Optimizing only for survival | Long rallies but poor match score | Report +1/−1 point differential and reach 21 |

## Astra prompt guidance

Give Astra the compact derived JSON state, a short recent trajectory, the current variation/difficulty, and the controller's measured action mapping. Ask it to:

- choose safe versus attacking return style;
- decide whether a feature is legal/useful in this variation;
- diagnose systematic intercept error from recent misses;
- choose calibration experiments and scenario coverage;
- change the macro target or hysteresis, not a one-frame action.

Explicitly tell Astra that code handles arithmetic, score counters, wall reflection, and frame timing. Include uncertainty markers when the parser has not yet validated a RAM field.

## Jev question set

Jev is fast and typed, so ask independent recognition/choice questions over the JSON state:

- `approach`: `RIGHT`, `AWAY`, `VERTICAL/UNKNOWN`;
- `target_relation`: `ABOVE`, `ALIGNED`, `BELOW`;
- `contact_window`: boolean;
- `serve_ready`: boolean;
- `feature_legal`: boolean from variation metadata;
- `feature_now`: `USE`, `HOLD`, `NO`;
- `direction`: one of calibrated `UP`, `DOWN`, `HOLD` (or the corresponding ALE action index);
- `risk`: `SAFE`, `ATTACKING`, `EMERGENCY`.

Never ask Jev to add scores, integrate velocity, solve wall-bounce arithmetic, or plan a multi-rally strategy. Code should provide the candidate target and Jev should recognize the state/choice.

## Verification plan before relying on the prior

1. Reset with a fixed seed and `frameskip=1`, no sticky action; capture RGB and RAM for at least one rally.
2. Calibrate action 2/3 direction, paddle Y extent, ball/paddle colors, scoreboard, and serve timing.
3. Correlate candidate RAM bytes with screen objects while moving one paddle at a time; verify any `$84`, `$85`, `$BB-$BE`, `$B2`, `$B3`, or `$96` mapping rather than copying it blindly.
4. Run a safe center-tracking controller and record point outcomes.
5. Compare center, edge, and feature-at-contact returns; only then enable an attacking policy.
6. Repeat on each chosen mode/difficulty and keep held-out reset seeds/scenarios separate.

## Sources

- [AtariAge Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)
- [Manual OCR](https://archive.org/stream/Video_Olympics_1978_Atari/Video_Olympics_1978_Atari_djvu.txt)
- [Internet Archive manual scan](https://archive.org/details/Video_Olympics_1978_Atari)
- [ALE Pong](https://ale.farama.org/environments/pong/)
- [ALE environment specification](https://ale.farama.org/env-spec/)
- [FujiNet Video Olympics README](https://raw.githubusercontent.com/FujiNetWIFI/fujinet-2600-video-olympics/main/README.md)
- [FujiNet Video Olympics porting notes](https://raw.githubusercontent.com/FujiNetWIFI/fujinet-2600-video-olympics/main/PORTING.md)
- [FujiNet RAM/config map](https://raw.githubusercontent.com/FujiNetWIFI/fujinet-2600-video-olympics/main/tools/vo.cfg)
- [Pong strategy guide (generic Pong advice; not ROM-specific)](https://playmore.gg/guides/pong-guide)
- [Atari 2600 hardware timing reference](https://problemkaputt.de/2k6specs.htm)
