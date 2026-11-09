# Atari 2600 Asteroids — game brief

Study date: 2026-10-30. Scope: the original **Atari 2600** cartridge as exposed by `ALE/Asteroids-v5`, not the arcade machine, Atari 7800 release, a homebrew remake, or Asteroids Deluxe.

## Rules and objective

Destroy the asteroid field while preserving ships. Rocks may fragment before disappearing; clearing a field produces another. The manual gives four starting ships. Large/medium/small rocks award 20/50/100 points, satellites 200, and UFOs 1,000. Selected variants award another ship at intervals of 5,000, 10,000, or 20,000 points, or never. There is no finite campaign victory: accumulated score before game over (or a disclosed time limit) measures progress. [M1, A1, A2]

Left/right rotate, up thrusts, and FIRE shoots along the nose; the manual explicitly recommends pressing and releasing FIRE rapidly. Down is **not braking**: it invokes the selected defense. Hyperspace relocates unpredictably; shield must be released before two seconds; flip turns the ship 180 degrees; some variants have no defense. Bodies wrap at screen boundaries. Do not import the arcade version's exact split tree, hyperspace death probability, saucer AI, or bullet limit. [M1]

The cartridge has 66 variations, including alternating two-player and children's games. The proposed benchmark is single-player ALE mode 0, difficulty 0; it must record these settings explicitly. ALE supports modes 0–31 and 128 and difficulties **0 and 3**, not every integer 0–3. Difficulties correspond to console switches; verify their A/B mapping in the installed build rather than guessing. [A1, A2]

## Controls available to the adaptor

The tested reduced Gymnasium action space is `Discrete(14)`:

| Index | Meaning | Interpretation |
|---:|---|---|
| 0 | NOOP | Release joystick and fire |
| 1 | FIRE | Fire without thrust/turn |
| 2 | UP | Thrust |
| 3 | RIGHT | Clockwise rotation |
| 4 | LEFT | Counterclockwise rotation |
| 5 | DOWN | Selected defense |
| 6 | UPRIGHT | Thrust and clockwise rotation |
| 7 | UPLEFT | Thrust and counterclockwise rotation |
| 8 | UPFIRE | Thrust and fire |
| 9 | RIGHTFIRE | Clockwise rotation and fire |
| 10 | LEFTFIRE | Counterclockwise rotation and fire |
| 11 | DOWNFIRE | Defense and fire input; effect depends on defense state |
| 12 | UPRIGHTFIRE | Thrust, clockwise rotation, fire |
| 13 | UPLEFTFIRE | Thrust, counterclockwise rotation, fire |

Read `env.unwrapped.get_action_meanings()` at startup; direct ALE uses an action enumeration with different numeric values. Keep the reduced mapping separate from raw ALE actions. The native Gymnasium return order is `(observation, reward, terminated, truncated, info)`. [A1; local smoke test]

## Pace and tier

**Keep tier 2.** Real-time NTSC play runs at approximately 60 emulated frames/second (the Atari 2600 NTSC timing reference describes 60 Hz). One frame is about 16.7 ms; ordinary ALE v5 repeats an action for four frames (about 66.7 ms), with 0.25 sticky-action probability. The tested Gym wrapper reports `metadata['render_fps'] == 30` for display, so the adaptor should use ALE `episode_frame_number` and configured frameskip—not display metadata—as its physics clock. Use `frameskip=1` for explicit control, diagnostics, and bounded action durations. [A1, A3, T1; local smoke test]

Set the planning metadata `decision_latency_s` to **0.10 seconds**, an engineering reaction target rather than a guaranteed safe deadline. Nearby rocks can make even that too slow; a distant threat offers more time. Jev's approximately 0.5-second response corresponds to about 30 frames, so it cannot be the only emergency controller. Astra must never block survival reactions.

**Important clock distinction:** ALE is synchronous: it advances only when `step`/`act` is called, not while a model thinks. It can therefore be paused offline, but that does not make the underlying game turn-based. For a tier-2 evaluation, the generated harness must specify a simulated-time or wall-clock scheduler, keep ticking with a bounded fallback while requests are outstanding, and record observation age. Do not silently pause and report real-time latency tolerance.

## Score and episode boundaries

Report the **sum of native, unclipped ALE rewards**, with no penalties mixed into the public score. ALE's Asteroids settings compute score deltas, correct the 100,000-point display rollover, read lives, and terminate when lives reach zero. Keep survival frames/seconds, lives lost, displayed score, and truncation reason as secondary metrics. Loss of one ship is not episode termination. [A2]

A proposed evaluation budget is 18,000 emulated frames (300 s at 60 Hz), or actual game over, whichever comes first. This is a benchmark choice, not a cartridge rule. Keep settings and frame budget fixed across competitors; disclose any harder-mode suite separately.

## Sources

- [M1] Atari, *Asteroids Game Program Instructions*, sections 2–8: https://atariage.com/2600/manuals_old/asteroids.html ; alternate original-manual mirror: https://atariage.com/manual_html_page.php?SoftwareID=828
- [A1] Farama, Asteroids environment: https://ale.farama.org/environments/asteroids/
- [A2] ALE Asteroids game settings source: https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/Asteroids.cpp
- [A3] ALE environment specification: https://ale.farama.org/environments/ and visualization: https://ale.farama.org/visualization/
- [T1] Atari 2600 NTSC timing reference: https://problemkaputt.de/2k6specs.htm
- Local validation uses the evaluator's Python 3.12.3, ale-py 0.12.1, Gymnasium 1.3.0, NumPy 1.26.4. Headless RAM and RGB reset plus one NOOP step passed, with four lives and 14 actions; no display/GPU/account required. Fixed-input checks produced identical initial and 120-NOOP RAM hashes for seeds 0, 1, 2, 3, 42 and 123, so `reset(seed)` must not be assumed to create different starts; use a versioned scenario catalogue or measured prelude for a separate robustness suite.
