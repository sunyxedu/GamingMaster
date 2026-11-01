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
