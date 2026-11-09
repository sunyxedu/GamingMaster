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
