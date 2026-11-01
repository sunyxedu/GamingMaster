# Atari 2600 Pong (Video Olympics / Pong Sports)

## Scope and identity

“Pong” on the Atari 2600 is primarily the Pong family in Atari's **Video Olympics** cartridge (CX2621, also sold as Sears **Pong Sports**), not the later arcade/console games that happen to have the same name. Video Olympics is a 50-variation paddle game. The standard headless benchmark target is the bundled ALE ROM exposed as `ALE/Pong-v5`; ALE describes that environment as the player controlling the right paddle against a computer-controlled left paddle. The ROM metadata identifies `pong.bin` as a 2 KiB USA Video Olympics ROM (MD5 `60e0ea3cbe0913d39803477945e9e5ec`).

Sources: [ALE Pong environment](https://ale.farama.org/environments/pong/), [ALE repository/ROM packaging](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [Video Olympics ROM record](https://atariboxed.com/index.php?go=detail&modul_refnumber=3722), and the [Video Olympics manual OCR](https://archive.org/stream/Video_Olympics_1978_Atari/Video_Olympics_1978_Atari_djvu.txt).

The adaptor should make the exact selected variation explicit. The manual calls games 1–2 Robot Pong (one player against the computer), games 3–4 standard two-player Pong, games 5–8 four-player Pong, and games 9–12 Super Pong (two paddles per side). `ALE/Pong-v5` is the one-player benchmark form; verify the startup screen/score behavior once rather than silently assuming that a ROM reset selected a different cartridge variation.

## Objective and rules

- Rally the ball with the paddle. In the one-player form, the human controls the **right** paddle and the computer controls the **left** paddle.
- A player/team gets one point when the opponent misses or hits the ball out of bounds. The first player or team to reach **21 points** wins.
- In Robot Pong, the computer serves to the player who lost the preceding point. In the general Pong instructions, the last player/team to score serves; the manual says to wait at least one second after the point before pressing the controller button to serve.
- Paddles move vertically. The original controls are analog paddle knobs, not ordinary up/down joysticks. The red controller button is game-dependent: it can invoke Speed, WHAMMY (sharper angles), CATCH, or Jump. In the manual's matrix, Robot Pong Game 1 has Speed and Pong Game 2 has WHAMMY; confirm the selected ALE variation before relying on a button tactic.
- Moving the Difficulty switch from B to A reduces the paddle to half size. This is an important scenario variable and should not be confused with an opponent skill level.

The primary manual facts above come from the [AtariAge Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587), especially “ROBOT PONG,” “PONG,” “CONTROLLER ACTION,” “DIFFICULTY,” and the game matrix. The [Internet Archive scan](https://archive.org/details/Video_Olympics_1978_Atari) is a second copy of the manual.

## ALE actions and observations

`ALE/Pong-v5` has the following default minimal action set:

| Index | ALE name | Use |
|---:|---|---|
| 0 | `NOOP` | Hold the current paddle state / do nothing |
| 1 | `FIRE` | Red controller button; serve or activate the variation's button feature |
| 2 | `RIGHT` | One paddle-wheel/joystick direction |
| 3 | `LEFT` | The opposite paddle-wheel/joystick direction |
| 4 | `RIGHTFIRE` | Direction 2 plus button |
| 5 | `LEFTFIRE` | Direction 3 plus button |

The names are inherited from the Atari input abstraction, so `RIGHT` and `LEFT` should be calibrated against screen Y on the first reset; do not hard-code the assumption that the labels mean screen right/left or screen up/down. The ALE documentation gives the action table and says that `full_action_space=True` exposes all 18 Atari actions, but the six-action minimal set is sufficient here. A direct Python check is `env.unwrapped.get_action_meanings()`.

For an object-oriented adaptor, request `obs_type="ram"`: the observation is a 128-byte `uint8` vector. ALE also supports `obs_type="rgb"`, a 210×160×3 frame, or grayscale 210×160. RAM is fast and symbolic at the hardware level, but it is not a documented JSON object schema: the adaptor must maintain a ROM-specific map or derive ball/paddle objects. Keep RGB enabled during development to calibrate and validate the RAM map. ALE also exposes native `getRAM()`, screen, reward, lives, frame counters, and clone/restore state methods.

Sources: [Pong action/observation specification](https://ale.farama.org/environments/pong/), [Gymnasium interface](https://ale.farama.org/gymnasium-interface/), [ALE Python interface](https://ale.farama.org/python-interface/), and the [ALE environment specification](https://ale.farama.org/env-spec/).

## Pace and tier

This is **tier 2**. An NTSC Atari 2600 frame is approximately 1/60 s (262 scanlines); paddle sampling and game logic run in the frame loop. A missed contact can be decided on a single frame/scanline, so the conservative raw-game decision budget is about **0.0167 s**. The default `ALE/Pong-v5` wrapper advances four frames per `step` and has a 0.25 probability of repeating the previous action, so its nominal action opportunity is about 0.0667 s but is stochastic. For a controllable benchmark, use `frameskip=1` and `repeat_action_probability=0.0`, or keep a frame-rate controller running while higher-level reasoning is pending.

A 0.5-second Jev call is roughly 30 NTSC frames and is too slow for direct paddle reflexes. It is usable only for recognition or a short target/macro while deterministic code continues to issue frame-level actions. Astra should plan mode choice, calibration, and adaptation; it must not be placed directly on the critical collision loop. The [Atari 2600 hardware reference](https://problemkaputt.de/2k6specs.htm) documents the NTSC frame/scanline and paddle-capacitor timing; the [ALE Pong page](https://ale.farama.org/environments/pong/) documents the default frameskip and sticky-action settings.

## Measuring progress

Use the environment reward and maintain counters:

- `own_points = number of +1 rewards`;
- `opponent_points = number of -1 rewards`;
- `return = sum(reward) = own_points - opponent_points`;
- report win/loss/termination and the final score `own_points:opponent_points`.

This is preferable to a frame count: it directly measures rallies won, matches the manual's 21-point objective, and is available without OCR. Keep the reward sign convention explicit because the player is the right paddle in ALE. If the adaptor also parses the scoreboard from RGB/RAM, cross-check it against these counters rather than replacing the reward-based score.

## Sources

- [AtariAge Video Olympics manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)
- [Internet Archive manual item](https://archive.org/details/Video_Olympics_1978_Atari)
- [Manual OCR text](https://archive.org/stream/Video_Olympics_1978_Atari/Video_Olympics_1978_Atari_djvu.txt)
- [Arcade Learning Environment: Pong](https://ale.farama.org/environments/pong/)
- [ALE installation](https://ale.farama.org/getting-started/)
- [ALE Gymnasium interface](https://ale.farama.org/gymnasium-interface/)
- [ALE environment specification](https://ale.farama.org/env-spec/)
- [Arcade Learning Environment repository](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [Video Olympics ROM metadata](https://atariboxed.com/index.php?go=detail&modul_refnumber=3722)
- [Atari 2600 hardware timing reference](https://problemkaputt.de/2k6specs.htm)
