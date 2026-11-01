Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Asteroids` — Real Atari 2600 Asteroids, one-player ALE mode 0/difficulty 0. Destroy rocks with the nose-mounted gun, avoid collisions, and preserve ships. Rocks fragment, cleared fields are replaced, inertia continues without thrust, and bodies wrap at boundaries. NOOP is not a brake. LEFT increments and RIGHT decrements the orientation code, typically every four frames. Four starting ships; extra ships follow cartridge rules. The objective is the sum of original unclipped ALE rewards before true game over or 18000 emulated frames (300 seconds at 60 Hz). Score rollover is handled by ALE; survival and lives are separate diagnostics, never score bonuses. FIRE is a button level, not an automatic rapid-fire macro; release then repress FIRE to shoot repeatedly; holding it does not continuously shoot. DOWN invokes mode-0 hyperspace, not reverse thrust.
The runner calls tick every 1/60 second, applying one current action for exactly one native frame. There is no background loop or sleeping: the runner must keep ticking while models think. Repeated plain action names remain held. Optional ACTION@FRAME (e.g. UP@120) expires after six frames relative to that observed episode frame; future, expired, or malformed actions apply NOOP while time advances. Use short bursts and a local controller; timestamps are simulated frames, not wall-clock time.
Observation is JSON: frame includes the prelude; ticks counts only external tick calls; score is cumulative reward; displayed_score wraps at 100000; lives are authoritative native lives. elapsed_s and frames_remaining include all prelude frames. settings and prelude disclose the scenario and its actual reward/life costs. The default bounded-launch-v1 catalogue holds UPRIGHT for 60+4*seed frames, seeds 0..15, then gives control to you. Even seeds train, odd seeds held out: both sample the same launch family. This compensates for the ROM's identical seeded canonical starts; it is a disclosed robustness task, not random new worlds. canonical suite disables that prelude for separately reported fixed-start runs.
player has presence, pixel x/y, approximate width/height and orientation_code: 0 up, 4 left, 8 down, 12 right; 16 possible codes. Intermediate heading angles are not calibrated. asteroids and player_missiles contain source_slot, geometry, raw_xy, size/direction codes and estimated velocity_px_per_frame. Source slots can be reused; they are not permanent physical identities. Motion uses up to four prior frames; null means insufficient history or a discontinuity, not zero. Large jumps/wraps, respawns and size changes invalidate velocity rather than inventing a wrap period. Geometry is approximate and unclipped, not collision masks; flicker can hide active objects in one RGB frame. Enemy craft/projectiles are not decoded: null means unknown. Raw 128-byte RAM is included by default for transparency. observation_frame stamps the immutable snapshot, age is zero at emission; compare its frame to the latest frame for decision age. Only mode 0/difficulty 0 is supported. snapshot_png() can explicitly save a real RGB image for debugging.
- NOOP: Release all controls; inertia and the game continue.
- FIRE: Press the fire button without thrust or rotation.
- UP: Thrust along the current nose direction.
- RIGHT: Rotate clockwise without thrust.
- LEFT: Rotate counterclockwise without thrust.
- DOWN: Invoke the cartridge defense (hyperspace in mode 0); not braking.
- UPRIGHT: Thrust and rotate clockwise.
- UPLEFT: Thrust and rotate counterclockwise.
- UPFIRE: Thrust and press fire.
- RIGHTFIRE: Rotate clockwise and press fire.
- LEFTFIRE: Rotate counterclockwise and press fire.
- DOWNFIRE: Invoke defense and press fire.
- UPRIGHTFIRE: Thrust, rotate clockwise, and press fire.
- UPLEFTFIRE: Thrust, rotate counterclockwise, and press fire.
(real-time games: ticks apply 'NOOP' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Each tick is one 1/60-second emulator frame. Names are case-sensitive. Optional ACTION@FRAME applies only while FRAME <= current frame < FRAME+6; FRAME is an observed absolute episode frame. Expired/invalid input releases controls. Plain names apply on every tick the runner repeats them; no hidden action latch or sleep.

Actions:
- NOOP: Release all controls; inertia and the game continue.
- FIRE: Press the fire button without thrust or rotation.
- UP: Thrust along the current nose direction.
- RIGHT: Rotate clockwise without thrust.
- LEFT: Rotate counterclockwise without thrust.
- DOWN: Invoke the cartridge defense (hyperspace in mode 0); not braking.
- UPRIGHT: Thrust and rotate clockwise.
- UPLEFT: Thrust and rotate counterclockwise.
- UPFIRE: Thrust and press fire.
- RIGHTFIRE: Rotate clockwise and press fire.
- LEFTFIRE: Rotate counterclockwise and press fire.
- DOWNFIRE: Invoke defense and press fire.
- UPRIGHTFIRE: Thrust, rotate clockwise, and press fire.
- UPLEFTFIRE: Thrust, rotate counterclockwise, and press fire.
(real-time games: ticks apply 'NOOP' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Each tick is one 1/60-second emulator frame. Names are case-sensitive. Optional ACTION@FRAME applies only while FRAME <= current frame < FRAME+6; FRAME is an observed absolute episode frame. Expired/invalid input releases controls. Plain names apply on every tick the runner repeats them; no hidden action latch or sleep.

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
- strategy_summary: Begin with controlled central firing and short aiming sweeps rather than continuous thrust. Prioritize imminent collision threats, then small fast rocks and clean firing opportunities, while preparing for fragments and fresh waves. Put numerical trajectory tracking, wrap-aware geometry, action timers, score/life bookkeeping and emergency reactions in code. Ask Jev to choose among precomputed safe candidates and classify field danger, not to perform arithmetic or plan long sequences. Let Astra select and adapt the strategic policy from aggregate score and death patterns without blocking the tick loop. Use defense sparingly and according to the selected mode; shield duration must be bounded and hyperspace is not guaranteed safe. Validate firing cadence, orientation mapping and enemy coverage instead of borrowing arcade-version mechanics.
- scoring: Report sum of original unclipped ALE rewards within a fixed 18000-emulated-frame budget or true game over. Asteroids ALE corrects 100000-point display rollover. Report surviving frames/seconds, lives lost, displayed score, settings and termination/truncation reason separately; do not replace score with survival or reward shaping.
- interface.observation: JSON derived from the 128-byte RAM: player presence, position, 16-valued orientation code, tracked velocity/confidence, asteroid source slots/positions/sizes, two player missile slots, native lives, cumulative reward score, displayed score, frame/time, settings and coverage flags. Optional 210x160 RGB diagnostic image. The inspected OCAtari reference lacks explicit saucer/satellite/enemy-shot classes, so default coverage must not claim those absent in hard settings.
- interface.actions: Fourteen reduced discrete Gym actions: NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE, UPRIGHTFIRE, UPLEFTFIRE. Up thrusts; left/right rotate; down invokes mode-specific defense. Use one-frame native steps with short explicitly bounded repeats; no raw RAM modification or score/life cheats.

### study/strategies.md
# Strategy prior — Atari 2600 Asteroids

Scope: Atari 2600 cartridge, initially ALE mode 0 / difficulty 0. No arcade-only rules should be imported. Tags: **CODE** = exact numeric bookkeeping, physics estimation, action execution and safety; **ASTRA** = slow strategic adaptation; **JEV** = typed judgments over already computed features. Recommendations below distinguish source-backed play advice from proposed engineering heuristics; none is a tested champion policy.

## 1. Primary strategic shape: controlled central shooting

The original manual favors central positioning with a left/right or circular aiming sweep, warns against lateral edges before a new field, and prioritizes small fast rocks. This is a better Atari 2600 prior than constant full-thrust roaming. Community accounts also favor the middle and report predominantly vertical asteroid movement, but that pattern should be measured, not assumed universal. [M1, C1]

- **CODE:** Track centrality, rock trajectories, shot availability, and imminent collision risk. Support stationary FIRE, short LEFTFIRE/RIGHTFIRE aiming pulses, and short thrust corrections. Do not thrust just because a target is far away.
- **JEV:** Choose `hold_aim / sweep_left / sweep_right / reposition` from safe pre-scored candidates; judge whether a clear firing corridor exists.
- **ASTRA:** Select an overall central-sweep versus deliberate-targeting policy from performance summaries. Change strategy after repeated observed failures, not every frame.

Opening proposal: observe the first live frames, preserve the central spawn unless threatened, fire into a clear arc, then rotate in controlled bursts. This is a proposed opening, not a guaranteed exploit. There is no economy, build order, or equipment progression.

**Variant prior:** The manual's one-player rows are grouped as games 1–8 (slow/fast, extra-life settings, hyperspace), 9–16 (shields), 17–24 (flip), and 25–32 (no feature); game 33 is the children's slow/hyperspace/5,000-point-extra-ship version. ALE exposes these as modes 0–31 plus `0x80`, so mode 0 is expected to be the first one-player row, but the adaptor should verify the selected mode on reset rather than rely on a human game number. Difficulty `b` is normal; `a` adds satellites/UFOs, with the manual reporting occasional UFO attacks after 7,500 and guaranteed attacks after 15,000 points. [M1, A1]

- **CODE:** Make mode, difficulty, defense and extra-life threshold explicit immutable state; do not let a policy silently transfer assumptions between variants.
- **JEV:** Answer whether the currently enabled defense is available and whether a candidate uses it; never infer it from a down action alone.
- **ASTRA:** Choose the benchmark variant before play and adapt defense posture to that variant. Keep hard difficulty as a separate suite until enemy craft/projectiles are actually observed.

## 2. Survival overrides scoring

A high-value small target is still a poor choice if reaching it causes an immediate collision. Handle imminent danger below model-call latency.

- **CODE:** Maintain a short horizon of candidate outcomes with conservative collision clearance. Use relative motion and approximate swept boxes/circles; account for heading change, ship drift, uncertainty, and wrap neighbors. Select a low-risk escape when all offensive candidates are unsafe.
- **JEV:** Rate already summarized risk as `safe / caution / emergency`, or rank a small set whose time-to-collision and escape clearance have been calculated. Do not ask Jev to calculate velocities or solve intersections.
- **ASTRA:** Adjust risk thresholds from deaths per minute, score per life, and missed opportunities. Do not send it frame-by-frame steering.

Suggested horizons such as 6–12 frames for emergency control and 30–60 for route assessment are starting engineering parameters, not verified optimal timing constants. Every selected maneuver needs an expiry and a newer-state safety check.

## 3. Geometry and aiming belong in code

Atari controls rotate/thrust; RIGHT does not move the ship right. Orientation and momentum must be separate variables. Use current and previous observations to infer speed, heading changes and missile direction. OCAtari's player/missile deltas cannot be trusted without independent tracking. [O1, O2]

- **CODE:** Calibrate the 16-value orientation nibble and turn rate against fixed inputs; compute shortest turn direction and stop pulses before overshoot. Fit bullet speed/lifetime and firing behavior from diagnostic observations rather than borrowing arcade constants. Solve interception numerically with target velocity and potential wrap paths; hand Jev a few named candidates.
- **JEV:** `Which candidate offers the cleanest safe shot: A/B/C/none?` Inputs should already include turn cost, intercept time, miss distance, slot availability, danger and confidence.
- **ASTRA:** Decide whether accurate intercept targeting beats sweep firing at the current density and score rate. Diagnose repeated missed shots from logs.

Do not equate facing direction with velocity, treat a teleport as extreme velocity, or use raw Euclidean distance across a wrapping edge. New or split rocks need fresh track identities/confidence rather than inherited high-confidence velocity.

## 4. Fire cadence and finite shots

The manual instructs repeated fire presses, and the reference RAM decoder exposes two missile slots. Neither fact proves an exact fire rate, cooldown, auto-repeat behavior or number of simultaneously harmful projectiles under every state. [M1, O1]

- **CODE:** Validate held FIRE versus press/release pulses, determine when shot slots become reusable, and only then optimize cadence. Releasing FIRE must preserve whichever turn/thrust state remains desired. Record shot creation and disappearance, not just reward.
- **JEV:** Prefer a clean valuable target when a shot is about to become available; otherwise continue safe sweep/dodge.
- **ASTRA:** Compare sustained scoring and deaths across pulse schedules and targeting policies. Treat any “double fire” or glitch claim as unverified unless explicitly allowed and reproduced.

Common error: holding a shooting rotation for Jev's entire response period may turn far past the target. Code should execute short pulses and reassess while the next model answer is pending.

## 5. Small-fast threats, splitting and wave transition

The manual's small-fast-rock priority has both survival and score justification (100 versus 20 for a large rock). Shooting a large rock can create additional hazards rather than clear the lane. [M1]

- **CODE:** Score candidates using collision urgency first, then expected useful score, turn time, hit probability and fragment uncertainty. Detect disappearance/splitting conservatively. An empty extractor frame is not proof of a new wave; confirm several frames and player state.
- **JEV:** `Is this a dangerous fragment cluster, a manageable field, or a likely transition?` Use supplied counts, confidence and nearest-threat features.
- **ASTRA:** Set target-priority weights and choose whether to speed up cleanup or spend time restoring safe position.

Keep away from side edges when a fresh field is about to appear. An apparently clear side lane just before wave replacement may be the worst place to commit a long thrust. [M1]

## 6. Defense is mode-specific, never a generic brake

Hyperspace can rescue an otherwise doomed ship but has an uncertain landing. Shield must not remain engaged for two seconds. Flip is an immediate reversal of orientation, not momentum. Some games have no defense. [M1]

- **CODE:** Include the selected defense type in every observation. Enforce shield duration strictly with a safety margin; apply defense as a bounded pulse with explicit release. Reset track velocity after hyperspace. Avoid repeating DOWN from a stale model result. Implement steering-only escape when defense is unavailable.
- **JEV:** Answer `controlled_dodge / emergency_defense / continue_fire` over candidates with supplied risk and defense availability.
- **ASTRA:** Choose a conservative versus emergency-only defense posture from the mode and observed outcomes; it must not invent a cooldown or assume invulnerability.

Do not import the arcade's often-repeated 25% hyperspace destruction probability into this cartridge. Any numerical cost of teleporting must be estimated for the actual environment.

## 7. Common mistakes and counters

These are the failure modes a strong harness should actively guard against:

- **Edge commitment:** thrusting to a side while the field is clearing can meet the next group at its entry edge. Counter: CODE raises edge risk during transition and ASTRA keeps a central fallback.
- **Thrust as panic:** continuous UP builds unwanted displacement while the ship is already aimed safely. Counter: CODE caps thrust pulses and JEV selects a bounded candidate, not an open-ended “move toward target.”
- **Facing equals motion:** rotating toward a rock does not stop existing drift. Counter: maintain separate heading and velocity estimates and score swept collision clearance.
- **Large-rock tunnel vision:** taking a 20-point large rock may create fragments while a 100-point small rock or collision is more urgent. Counter: threat-first candidate ordering, then value.
- **Stale model action:** a delayed LEFTFIRE or DOWN can be dangerous after a split, death or teleport. Counter: attach frame/expiry to every Jev/Astra instruction and reject it when state age or hazard class changes.
- **Hidden coverage failure:** treating missing saucer/enemy-shot objects as “none.” Counter: expose `coverage_unknown` and fall back to RGB validation or an explicitly safe policy.
- **Score-only optimization:** farming points while losing ships quickly is weak play. Counter: Astra compares score per life and survival as secondary diagnostics while the official objective remains score.
- **Arcade-mechanics transfer:** importing arcade bullet limits, saucer accuracy, or hyperspace odds can create a confident but wrong policy. Counter: label every unvalidated constant and calibrate this ROM.

## 8. Enemy craft and score farming: gated extension

UFOs and satellites belong to the harder console setting, and reward is much larger than rock hits. Community advice includes leaving one rock to keep harvesting craft and taking diagonal/vertical shots when horizontal shots appear unreliable. This is anecdotal, setting-dependent, and not a verified recommended policy here. [M1, C1]

- **CODE:** First implement and validate enemy craft/projectile observation. Current inspected OCAtari extractors omit these classes. Never feed a model `enemy_count=0` when coverage is unknown. Track enemy projectiles separately from friendly shots.
- **JEV:** Only after coverage exists: rank `evade_projectile / engage_craft / clear_rock` from computed safety/value features.
- **ASTRA:** Compare farming against wave clearing under the fixed evaluation horizon and explicit rules. Do not pursue farming in the default no-craft setting or at the expense of immediate survival.

The manual mentions that collision can award points when a ship is already doomed near an extra-life boundary. Treat deliberate ramming as a rare, verified last resort, not a general exploit or default tactic. Exact threshold bookkeeping belongs in **CODE**, and the policy decision in **ASTRA**. [M1]

## 9. Two-model control architecture

**Fast deterministic layer:** own the emulator clock, input latch, release timers, action-map conversion, score/life accounting, track fitting, collision features, candidate generation and emergency fallback. Candidate features should include observation frame, projected time to hazard, clearance, score opportunity, required turn, action duration and model-confidence expiry. Never ask Jev for arithmetic.

**Jev batch questions (~0.5 s):** ask several independent typed questions on one compact JSON state: (1) danger category; (2) safest candidate ID; (3) best safe target ID or none; (4) field regime `sparse / dense / transition / uncertain`; (5) yes/no, strategic instruction still suitable. Consume the response only if it remains compatible with the latest frame. Jev does not plan a long action sequence.

**Astra (seconds to a minute):** receive aggregated score rate, life changes, action distribution, uncertain tracks and short death summaries. Request a small declarative plan: strategy family, target ordering, risk tolerance and defense preference. Keep those preferences valid for multiple seconds, but let fast safety override them. Astra should analyze why a policy failed rather than issue late commands from an old screenshot.

Suggested JSON features: frame, observation_age_frames, lives, score, mode, defense, player_presence, heading, speed, nearest threat distance/TTC, safe escape sectors, shots_available, field density, candidate action summaries, and coverage flags. Keep raw RAM as a diagnostic side channel, not Jev's main reasoning task.

## 10. What to test later (no agent code in this study)

1. Native no-op survival versus stationary firing versus controlled circular fire; these are baselines to implement in the adaptor/harness phase, not proven winners.
2. Exact orientation mapping, turn cadence, thrust/drift, screen wrap and shot lifetimes on fixed diagnostic sequences.
3. Decoder/player presence around death and hyperspace; object identities at split/wrap.
4. Seeding reproducibility and actual start diversity; sticky action robustness separately.
5. Slow-model timeouts and Jev results that arrive after danger has changed.
6. Scoring rollover, extra life, death, true game over and frame-budget truncation.
7. Held-out performance using raw score, score/life, survival and variance across fixed settings. Never select settings separately per agent to inflate results.

## Sources and evidence grades

- **Primary manual:** https://atariage.com/2600/manuals_old/asteroids.html (sections 3, 5–8); https://atariage.com/manual_html_page.php?SoftwareID=828 [M1]
- **Community anecdote, not validated champion play:** https://forums.atariage.com/topic/90451-asteroids-any-tips/ [C1]. Direct fetch was blocked (403); available search extracts support the center/vertical-motion/UFO-farming advice, so treat it as weaker evidence.
- **Primary implementation:** https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/asteroids.py [O1]; https://github.com/k4ntz/OC_Atari/blob/master/ocatari/core.py [O2]. The downloaded PyPI 2.2.1 source was also inspected.
- **Native engine:** https://ale.farama.org/environments/asteroids/ and https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/Asteroids.cpp

Engineering heuristics above are study recommendations, not claims that the source authors prescribe this model architecture. In particular, reaction horizons, candidate scoring, safety overrides, benchmark duration and model-call cadence require empirical tuning.

### study/brief.md
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

### adaptor/ADAPTOR.md
# Atari 2600 Asteroids adaptor

This is the real cartridge running in ALE 0.12.1 / Stella through Gymnasium 1.3.0,
not a clone or simulated substitute. CPU-only, no display, GPU, service, account,
credential, or external process. Installed ALE supplies its cartridge asset.

## Setup

Install requirements.txt into the evaluator's Python environment, or run setup.sh with its Python path. For this run:

```sh
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python -I -m pip install -r requirements.txt
```

`make_game()` returns a tier-2 object implementing the contract. `reset(seed)`
starts a fresh native episode. `close()` releases the environment and is idempotent.
No required environment variables; see env.example.

## Clock, controls and score

One tick is one native frame at 60 Hz, `tick_s=1/60`, with underlying frameskip=1.
The evaluator is the sole clock owner; no duplicate background thread, sleeps, or
autonomous stepping. The evaluator keeps ticking while model calls run. `ticks`
counts external applied ticks only, and resets to zero. `frame` includes reset's
explicit scenario prelude. Plain names are the current joystick/button levels,
not pulse macros. Query the runtime's fourteen reduced Gym action meanings.
Optional `ACTION@FRAME` is valid only while FRAME <= current frame < FRAME+6 (six frames). Malformed,
stale or future input releases controls for one frame instead of raising.
The runner cannot signal whether an identical plain action is fresh or repeated;
use timestamped inputs/local controller for safe expiry rather than claiming that
an undocumented timeout overrides current controls.

Score is precisely accumulated original, unclipped ALE rewards; includes prelude
rewards. Stop on true game over or 18000 emulated frames, counting the prelude.
No episodic-life resets, shaping, life/survival bonuses, reward clipping, hidden
frame skipping, RAM writes or save-state manipulation. ALE handles display rollover.
`info()` separately reports elapsed frames/seconds, displayed score, lives gained
and lost, settings, termination/truncation reason, package versions and ROM SHA256.
Emulator failures set `valid_trial=False` and an `invalid_reason`.

## Versioned scenarios and fairness

Canonical mode-0/difficulty-0 starts and fixed-input trajectories did not vary
with native reset seeds in study tests. We do not pretend seed labels create
random worlds. Default evaluation is the disclosed **bounded-launch-v1** robustness
suite: seed s in 0..15 means UPRIGHT held for 60+4*s frames immediately after native
reset. This is one parameterized control prelude, not a memorized action schedule
or a seed-aware gameplay controller. It uses only real controls and keeps ROM,
mode 0, difficulty 0, zero sticky actions, and 18000-frame budget fixed.
Train seeds are 0,2,4,6,8,10,12,14; held-out seeds 1,3,5,7,9,11,13,15. Both groups
interleave the same launch family. Prelude frames, actual rewards, lives lost and
gained are retained and disclosed. No outcome rejection or life reset occurs.

`make_game(scenario_suite="canonical")` disables the prelude for separately
reported original fixed-start runs. Different canonical seeds are **not** independent
instances. Do not pool these scores with bounded-launch-v1, or claim transfer to
unseen modes. Fast/defense/hard-difficulty variants are intentionally not offered
until mode mapping and missing enemy observation coverage are validated.

## Observation and limits

Strict JSON, object-level RAM state, raw RAM by default, no automatic screenshots.
`include_ram=False` omits RAM without changing dynamics. player presence and packed
orientation (0 up, 4 left, 8 down, 12 right), fifteen asteroid source slots, two
player missiles, approximate positions/sizes, raw encoded coordinates, score and
native lives are exposed. Source slots are not permanent object identities; same-size immediate reuse can be missed.
Finite-difference velocity uses up to four frames and is invalidated at gaps, size
changes, large jumps/wraps and respawns. Exact wrap periods and intermediate angles
are deliberately not invented. Boxes are unclipped approximate reference geometry,
not exact collision masks. Display flicker can hide active objects in one RGB frame. `enemy_craft` and `enemy_projectiles` are null/unknown,
never claims of absence. See THIRD_PARTY.md.

Snapshots are immutable copies stamped with the emulated frame. They contain no
wall-time/id/path noise; fresh same-seed resets are reproducible. A snapshot's
`observation_age_frames=0` applies at emission; clients calculate later age from
its stamp. Optional `snapshot_png()` saves actual ALE RGB under runtime/screenshots
and returns the path. Runtime never modifies source files.

## Validation

Run the generic `metaharness.games.adaptor.validate_adaptor` with expected_tier=2.
Additional real-emulator checks and their measured outcomes are recorded under
runtime/; no fixture or mock emulator is used for integration tests.

### study/interfaces.md
# Programmatic interfaces — Atari 2600 Asteroids

Research date: 2026-10-30. This is an implementation specification, **not game-playing code**. The game and engine are real; no substitute Asteroids clone should be used.

## Recommendation

Use **native Gymnasium/ALE (`ale-py`) with a small, read-only Asteroids RAM-to-objects decoder**, using OCAtari's Asteroids RAM extraction as the documented reference. The environment owns score, lives, reset, termination and action timing; the decoder only produces symbolic observations. Preserve optional RGB snapshots for debugging. This fits offline CPU-only Linux and avoids screen capture/input injection and cloud services. Full OCAtari is an optional comparison tool, not the recommended runtime dependency. [A1–A4, O1–O4]

This choice is deliberate: OCAtari 2.2.1's source swaps terminated/truncated in `step`, has wrapper-chain-dependent helper methods, leaves Asteroids HUD lives unupdated, lacks saucer/enemy-shot classes, and recreates the player and missile objects on every detection. A full wrapper is not automatically a reliable object oracle. See extraction caveats below. [O2, O3; downloaded 2.2.1 source]

## Practical alternatives

| Interface | Maturity / exposed state / control | Latency | Legal/ToS risk; cost and constraints | Recommendation |
|---|---|---|---|---|
| Native ALE + Gymnasium + read-only decoder | Mature research emulator; 128-byte RAM, RGB, score rewards, lives, frame numbers, game over; joystick actions, seeds, modes | In-process CPU calls; no model/network latency; benchmark locally | ALE is GPL-2.0-only; no service ToS or account; ROM provenance/license remains relevant | **Chosen** |
| Direct `ale_py.ALEInterface` + same decoder | Same emulator; exact `act`, `getRAM`, `getScreenRGB`, `reset_game`, `getMinimalActionSet`, `lives`, `game_over` | Similar or less wrapper overhead | Same GPL-2.0-only emulator and ROM considerations; more manual seeding, truncation, and action mapping work | Good fallback |
| Legacy `atari-py` / classic Gym Atari | Historical ALE Python wrapper; RGB/RAM observations and discrete controls | Local native calls, but old bindings add setup overhead | Archived-style dependency/ROM-import path; likely unsuitable for Python 3.12 and current Gymnasium; ROM terms still apply | Avoid; native ale-py supersedes it |
| Full OCAtari RAM mode | Existing named objects with positions/sizes/orientation; wraps ALE | Low CPU overhead but Python matching/dependencies; no verified latency guarantee | MIT wrapper; ALE/ROM licenses separate; broad install dependencies and extractor gaps | Useful for cross-check, fix documented bugs if used |
| OCAtari vision / custom RGB parsing | CPU color/shape object extraction, no GPU needed | More work than RAM; frame flicker and tracking ambiguity | Open-source local wrapper; same ALE/ROM constraints; no account | Debugging or missing-object fallback, not primary |
| AtariARI / annotated RAM | Older research RAM labels, usually not rich normalized object tracks | Low overhead | MIT research code; local ROM rights still apply; legacy Gym dependencies and partial labels need adaptation | Reference, not turnkey current runtime |
| Stella debugger / custom emulator extension | Mature emulator/debugger exposes RAM, registers, frames and input; no standard current Python object API | Fast emulation; integration/scripting overhead | Free/open-source local tool; ROM rights still apply; display/debugger may require SDL/virtual display | ALE already solves this problem more directly |
| libretro Stella core via a native frontend | `retro_run`, video callbacks, input callbacks, memory access and serialization | Frame-level native calls | Free/open-source interfaces; ROM provenance remains the user's responsibility; need C ABI/frontend integration and ROM; object decoder still needed | Viable engineering fallback |
| Gym Retro / Stable-Retro integration | Emulator action arrays and RAM with game data/integration definitions | Local per-step; native extension compatibility must be checked | Open-source local frameworks; no service ToS, but Asteroids integration/ROM and Python 3.12 wheels not verified | Do not make this the critical path |
| Screen+input adapter (`stella` + `Xvfb` + `mss`/PyAutoGUI or `xdotool`) | Screenshot pixels, OCR/vision score and keyboard/joystick events; no RAM unless debugger is separately bridged | Highest jitter: virtual display, capture, OCR and focus/input queues | Local open-source/tools but ROM rights still apply; X11 virtual display and extra packages required; no server ToS | Last resort |
| Browser emulator / ALE WebAssembly | Pixels and potentially JS hooks/step interface | Browser/headless overhead | Free/open-source option may still inherit ROM/site terms; browser dependency and possible network | No advantage over local Python |
| Official commercial port/cloud account | No identified official game-server automation API for this cartridge | N/A | Accounts/platform purchase potentially required | Not needed |

…(truncated; full text in prior/interfaces.md)

repeated deterministic runs; do not pretend seed IDs are independent worlds. For a genuine held-out robustness suite, define a versioned scenario catalogue before evaluation: map a seed to explicit mode/difficulty/defense/speed variants, or to a measured, bounded control prelude whose frames and rewards remain inside the episode budget. Report each variant separately because speed, extra-life threshold and defenses change the task. Do not randomize RAM or discard a dangerous prelude. Never hash a seed into a memorized action schedule. Retain sticky actions as an explicit robustness suite. The native step median was about 0.09 ms and p95 about 0.11 ms on this host for 600 one-frame NOOP steps, excluding model time; do not treat that as a real-time guarantee on another host. [local `validation.json`]

ALE does not advance during API/model inactivity. A tier-2 harness must make the clock contract explicit and tick with a short-lived last action or deterministic safe fallback while Astra/Jev are busy. One emulator thread/process should own all step calls; publish immutable timestamped observations. Reject stale decisions. Do not hold thrust or defense indefinitely on timeout.

## Sources

[A1] https://ale.farama.org/environments/asteroids/
[A2] https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/Asteroids.cpp
[A3] https://ale.farama.org/python-interface/ and https://ale.farama.org/gymnasium-interface/
[A4] https://ale.farama.org/getting-started/ and https://pypi.org/project/ale-py/0.12.1/
[A5] https://ale.farama.org/release_notes/index.html
[A6] Atari 2600 NTSC timing reference: https://problemkaputt.de/2k6specs.htm
[O1] https://github.com/k4ntz/OC_Atari and https://oc-atari.readthedocs.io/en/latest/
[O2] https://github.com/k4ntz/OC_Atari/blob/master/ocatari/core.py
[O3] https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/asteroids.py
[O4] https://github.com/k4ntz/OC_Atari/blob/master/ocatari/vision/asteroids.py
Other interface references: https://stella-emu.github.io/docs/debugger.html ; https://github.com/tcr/stella-lua ; https://docs.libretro.com/development/cores/developing-cores/ ; https://github.com/mila-iqia/atari-representation-learning ; https://github.com/openai/atari-py ; https://github.com/Farama-Foundation/stable-retro ; https://python-mss.readthedocs.io/stable/usage.html ; https://pyautogui.readthedocs.io/

Downloaded reference sources are under `sources/`; they are reference material, not executed agent code. Full OCAtari 2.2.1 source was inspected; only the minimal native stack was installed and smoke-tested.


## Measured model latency (at archive creation, 2026-10-30 17:28)
- System-2 (Astra), one-word answer: 62 s
- System-2, an answer of ~900 output tokens: 96 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 153 ms (= 9.2 ticks), p90 162 ms, max 164 ms over 12 warm requests; first request on a new connection 188 ms — a timeout below the tail turns slow answers into failed calls
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
