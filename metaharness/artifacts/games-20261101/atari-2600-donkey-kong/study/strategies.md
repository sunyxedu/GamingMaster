# Atari 2600 Donkey Kong — strategy prior

## Scope and confidence

This is a one-player real-time Atari 2600 game, so “build order” means a repeatable route and hazard policy rather than an economy opening. The reliable rules are the [AtariAge Atari 2600 manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=149), the [ALE DonkeyKong specification](https://ale.farama.org/environments/donkey_kong/), and the [frame-verified Atari 2600 disassembly/recompilation](https://github.com/Plicerin/donkeykong-rom-port). The [Atari 2600 GameFAQs walkthrough](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview) and the [archived browser bot](https://github.com/Plicerin/donkeykong-rom-port/blob/main/archive_old_port/donkey-kong-bot.js) provide useful human/bot heuristics, but community advice must be marked as advisory because many Donkey Kong guides describe the arcade game.

The harness should never import the arcade's four-board order, conveyor stage, arcade fireball rules, or arcade barrel patterns into this port. This cartridge alternates only the ramp/barrel and rivet/fireball screens, with increasing difficulty after each completed pair.

## Strategic objective

**Primary objective:** complete as many screens as possible without losing Marios. **Secondary objective:** complete quickly enough to bank the 5,000-point screen bonus and collect safe 100-point rivets/jumps. **Tertiary objective:** take the hammer for 800-point hazard smashes only when it does not delay a ladder or create a likely death.

The manual's scoring is 5,000 starting bonus per screen, +100 for jumping a barrel/fireball, +100 per rivet, and +800 per hammer smash. The bonus counts down and is banked when the screen is completed ([manual scoring section](https://atariage.com/manual_html_page.php?SoftwareLabelID=149)). Thus “survive and reach the next screen quickly” dominates risky point farming. A score-only agent can look good briefly while spending its three Marios; report both score and completed-screen/life statistics.

### Responsibility split

| Knowledge/action | Put in deterministic code | Put in Astra's prompt/planner | Ask Jev quickly |
|---|---|---|---|
| Cartridge identity, action mapping, terminal handling | Yes; fixed constants and assertions | No | No |
| Stage/lives/bonus/rivet bookkeeping | Yes; decode and serialize | Use as facts, do not recalculate | Read stage/life urgency if needed |
| Ladder coordinates and legal climb alignment | Yes; geometry and tolerance | Select the next route target when several are safe | “Which ladder/side is the immediate target?” |
| Relative hazard distance and collision-time estimate | Yes; arithmetic from positions/velocities | Decide how much risk is acceptable | “Jump now or continue?” and “which jump direction?” |
| Default ramp sequence and rivet serpentine | Store as prior/waypoint graph | Choose/replan route | Recognize whether current state matches a waypoint |
| Hammer detour | Check presence/time and prevent climbing while active | Decide whether score/risk warrants it | “Is hammer present?” / “take it now?” |
| Unusual state, broken ladder, lost route | Detect exact facts in code | Diagnose and replan | Classify immediate danger/action |
| Long-term risk versus bonus | Track score/lives/bonus | Astra | No arithmetic; ask only a typed risk choice |

Jev should never be asked to add scores, estimate time from raw coordinates, or invent a route. Code computes distances and BCD conversion; Jev recognizes the tactical situation and returns one of a small set of legal actions. Astra can reason about several seconds of route, but must not block the actor loop.

## Start and reset policy

1. After reset, the game can sit in an idle/attract state. Send `FIRE` (ALE action 1) once to start. The ALE page explicitly says that `NOOP` leaves Donkey Kong idle with no reward ([starting the game](https://ale.farama.org/environments/donkey_kong/)).
2. Do not repeatedly spam `FIRE`: in active play it is the jump button and can trigger an unnecessary jump. Code should know whether the game state is idle, active, transition, or game over.
3. At a new screen, preserve the current score/lives and re-read the stage and player position. Do not assume a previous jump direction or ladder target survived a life-loss/transition.
4. On death, release movement during the cartridge's death/reset animation, wait for the new active state, and then restart from the actual reported player coordinates. On game-over, stop until `reset`; do not use the Atari reset action as a substitute for the environment reset.

**Allocation:** steps 1–4 are code-level state-machine rules. Astra should receive a fresh-state explanation after a life loss and choose whether to switch from a score-seeking policy to a conservative survival policy. Jev can answer `start_needed`, `death_animation`, or `game_over` as yes/no, but code can usually determine these from RAM and terminal flags.

## Ramp/barrel screen policy

### Fixed route prior

The port's verified code uses a fixed ladder/ramp topology rather than an arcade board. OCAtari exposes ten ladder slots at approximately `(108,176)`, `(48,148)`, `(80,148)`, `(88,120)`, `(108,120)`, `(48,92)`, `(68,92)`, `(108,64)`, `(76,40)`, and `(32,40)` in the 160×210 ALE coordinate system. The port source also labels alternating ladder tables “starting from the bottom” and describes a right-to-left/left-to-right alternation ([decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/donkeykong.py), [disassembly ladder tables](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm)).

Use this as the opening route prior:

- From the bottom start, run toward the next usable ladder on the far side of the current walkway.
- After climbing, traverse the next ramp in the opposite horizontal direction; the ramps alternate their slope, so the correct side changes as Mario moves upward.
- Continue the zig-zag to the top rather than wandering for points.
- At every rung, confirm that Mario's x is within the ladder tolerance and that the ladder is currently unbroken/usable. If a hazard forces a jump, recompute the current walkway and target; do not keep blindly holding `UP`.

The archived bot's geometry suggests that a state-derived target ladder is safer than a hard-coded sequence: find the ladder that connects the current walkway to the next one, then walk to its x-coordinate and climb ([archived bot route helper](https://github.com/Plicerin/donkeykong-rom-port/blob/main/archive_old_port/donkey-kong-bot.js)). The bot is not an authoritative cartridge API, but its current-walkway/target-ladder abstraction is the right prior for the generated harness.

**Code:** store the fixed ladder graph, infer current walkway from player y/ramp geometry, use an x tolerance of a few Atari pixels, constrain Mario within the walkable span, and detect successful climb by y change. Keep `target_ladder`, `current_walkway`, and `route_side` in state. Search over the two or three legal next ladders if a hazard blocks the nominal one.

**Astra:** choose between nominal and alternate ladder routes when a barrel is approaching, reason about whether detouring is worth losing bonus, and replan after an unexpected landing. Astra should prefer a safe ladder that still makes upward progress over a long detour for a hammer.

**Jev:** answer typed questions such as:

- `choice(target_ladder)` from the currently legal ladder candidates;
- `yes_no(on_target_ladder)`;
- `choice(immediate_action)` among `LEFT`, `RIGHT`, `UP`, `NOOP`, `LEFTFIRE`, `RIGHTFIRE`;
- `yes_no(route_blocked_by_hazard)`.

Jev sees recognized objects and stage labels; it should not be asked to infer exact x distances or count frames.

### Barrel avoidance

The manual says barrels roll down the ramps, and that jumping is easiest when Mario and a barrel move toward each other. The verified cartridge has four obstacle slots, with vertical/horizontal positions and directions available in RAM; OCAtari calls these objects `Barrel` even when a generic obstacle is on the other screen.

Use this priority order:

1. If a barrel is on Mario's current walkway/ramp and the relative motion is closing, prioritize a jump over the nominal ladder target.
2. Start the jump early enough that Mario's collision box is above the barrel; do not wait until the sprites overlap. Code should use a calibrated collision margin, not Jev arithmetic.
3. Set the desired left/right facing before pressing `FIRE`; the cartridge latches the jump direction at the start. A directional fire action (`LEFTFIRE` or `RIGHTFIRE`) is preferable to `FIRE` when a safe landing side is known.
4. During the jump, avoid fighting the committed route with a last-moment direction change. After landing, re-detect walkway and target ladder.
5. If the barrel is on another walkway and not using an intersecting ladder, continue climbing; do not jump merely because any hazard exists somewhere on screen.

**Code:** compute same-walkway/relative-closing tests, hazard time-to-contact, edge constraints, jump cooldown, and whether a jump has already been credited. Give hazards a higher priority than score opportunities. Record a `jump_reason` (`barrel`, `fireball`, `rivet_gap`, `hammer`) for diagnostics.

**Astra:** decide whether a safe pause/route change beats a jump when the obstacle is far away or the target ladder is nearby. It can use the exact state history to recognize a repeated pattern, but should not override a code-level imminent-collision guard.

**Jev:** provide fast `danger_same_walkway`, `jump_now`, `jump_left/right`, and `can_continue_climb` judgments. A typed answer is appropriate because Jev recognizes the configuration; code supplies the numeric distance and legal option set.

### Hammer policy

The manual says the hammer can smash barrels/fireballs for 800 points, disappears after a few seconds, cannot be carried up a ladder, and does not save Mario if a barrel hits while the smash misses ([manual Step 3](https://atariage.com/manual_html_page.php?SoftwareLabelID=149)). The Atari 2600 walkthrough places it on the second-from-top floor ([GameFAQs overview](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview)).

A strong conservative policy is:

- Do not detour for the hammer when no hazard is near; the bonus and upward progress are more valuable.
- Take it when it is directly on the safe route and a barrel is closing, or when the run has a large safety margin and an 800-point smash is essentially free.
- While hammer-active, do not issue `UP` expecting a climb; route horizontally or let the timer expire, then resume ladder travel.
- Never treat the hammer as invulnerability. If the barrel is too close or on the wrong side, jump/avoid rather than attempting a hit.

**Code:** track `hammer_present` and hammer-active time from the object/RAM decoder, prohibit climb commands while active, and detect a confirmed smash from reward/state change. **Astra** decides whether the detour's expected 800 points justify time/life risk. **Jev** only answers presence/active/miss-imminent questions and chooses a safe immediate direction.

## Rivet/fireball screen policy

### Rivet bookkeeping and route

The manual says Mario removes a rivet by running or jumping over it and then continues to alternate screens after all rivets are removed. The cartridge disassembly stores four row bytes at ALE indices 31–34; left/right removal values are 6 and 12, and a completed row is 18. The common Atari 2600 guide describes eight total rivets, two per row, and recommends working from the bottom upward ([GameFAQs overview](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview)).

Use a serpentine route:

1. Start on the bottom rivet row.
2. Approach and clear the first remaining rivet.
3. Continue to the other side of the row, clearing the second rivet.
4. Use the far-side usable ladder to climb one row.
5. Reverse direction on the next row and repeat until all rows are complete.
6. When the last rivet is cleared, stop making route assumptions and wait for/observe the transition to the ramp screen.

A cleared rivet is not merely a bookkeeping point: it is a gap. The archived bot explicitly checks which side of a removed rivet Mario is on and jumps across the hole rather than walking through it ([gap-crossing logic](https://github.com/Plicerin/donkeykong-rom-port/blob/main/archive_old_port/donkey-kong-bot.js)). If an evasive jump lands on the wrong side, use a deliberate jump across the gap before continuing toward the next rivet/ladder.

**Code:** maintain a four-byte rivet mask and a `remaining_rivets` set; select the next target according to current row/side; prohibit a walking trajectory that crosses a cleared rivet; require a jump/landing plan for a gap. Code should declare the row complete from raw bytes, not from whether a rivet-colored pixel happens to be visible.

**Astra:** choose a safe row/side order when a fireball blocks the nominal serpentine path, decide whether to wait for a fireball to pass or cross a gap, and replan after a forced landing. It should retain the invariant “never walk through a cleared hole.”

**Jev:** answer `which_rivet_next`, `fireball_same_row`, `jump_gap_now`, `climb_far_ladder`, and `safe_to_cross`. Jev should identify a configuration, not count how many points remain.

### Fireball avoidance

The manual says fireballs follow Mario, normally stay on the same ramp, and sometimes jump across gaps. Treat a fireball on Mario's row as the immediate threat; a fireball on a different row is usually lower priority unless Mario is about to climb into it.

- Keep moving toward the currently selected rivet or ladder rather than freezing in the center of a row.
- Jump early when a fireball is closing; choose a landing side that does not put Mario into a cleared rivet gap.
- If a fireball is near a ladder, wait for a safe window or choose the alternate usable ladder; never climb through a hazard merely because the planner selected that ladder.
- Avoid repeated jump-button presses: jump height/hang-time and fire-button debounce create a finite commitment, and spam can destroy route control.

The verifier's source calls the stage `gameScreen=1` and changes obstacle movement constraints as rivets disappear; use raw stage/rivet state rather than applying ramp-stage barrel assumptions ([movement and rivet routines](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm)).

## Timing and action holding

ALE v5 defaults to four emulated frames per Gymnasium action and 0.25 sticky-action probability. At NTSC speed this is about 66.7 ms per observation; the source's jump hang time is about 31 frames and the bonus tick is about 127 frames. A practical controller should:

- run an immediate actor loop at one or two fixed-4-frame steps;
- keep a safe previous direction while Astra is thinking;
- ask Jev for a typed update at roughly 0.07–0.15 s cadence, or more often near a hazard;
- interrupt the macro when stage, life, hazard row, jump state, rivet mask, or terminal state changes;
- use `frameskip=1` only in calibration/search, not as an excuse to call a 0.5-second model synchronously;
- develop with sticky actions disabled and then test with v5's 0.25 probability.

**Code** owns cadence, action persistence, frame counters, and hard collision/death guards. **Astra** owns the macro target (next ladder, row/rivet order, hammer risk). **Jev** owns the fast interrupt (“jump left/right now?”, “climb or wait?”, “is this a transition?”).

## Common mistakes and counters

| Mistake | Why it fails | Counter |
|---|---|---|
| Treating this as arcade Donkey Kong | Wrong screen order, ladders, hazards, and progress | Assert ALE ROM ID/MD5 and use only the two-screen port model |
| Holding `NOOP` after reset | Game never starts; no reward | Send one controlled `FIRE`, then re-read active state |
| Holding `UP` away from a ladder | Mario walks/does nothing and loses bonus time | Code-align x to a legal ladder before climbing |
| Assuming every visible ladder is usable | Broken/incorrect-side ladder can reject the climb | Check current stage, row, side, and y; replan |
| Crossing a removed rivet on foot | The hole kills Mario | Jump over the gap; keep a cleared-rivet mask |
| Jumping at the last pixel | Collision boxes overlap before the jump rises | Use relative-motion guard and jump early |
| Spamming `FIRE` | Jumps are committed/debounced and can ruin a ladder approach | Use a jump cooldown and typed “jump now” interrupt |
| Taking every hammer | Hammer blocks ladder climbing and expires quickly | Only take it on-route or for an almost-free imminent smash |
| Chasing every 100/800 points | Bonus/lives are more valuable than risky points | Optimize screen completion and survival first |
| Thinking a `Barrel` object is always a barrel | OCAtari uses the same slot class for rivet-screen obstacles | Interpret obstacle kind from `stage` and raw state |
| Reading RAM byte 36 as total score | It is the displayed BCD bonus in this decoder | Sum ALE rewards; keep bonus as a separate field |
| Letting Astra block the actor | The 60 Hz world/hazard and bonus continue | Jev/default policy acts while Astra plans asynchronously |
| Trusting OCAtari terminal flags blindly | Current 2.2.1 wrapper swaps `terminated`/`truncated` on return | Step underlying Gymnasium or normalize at the boundary |
| Evaluating only one deterministic seed | It hides hazard timing and sticky-action failures | Hold out seeds and test v5 stickiness separately |

## Suggested typed Jev question set

These should be independent, small, and answerable from one normalized JSON state:

1. **Choice:** `immediate_action` ∈ `{NOOP, LEFT, RIGHT, UP, DOWN, LEFTFIRE, RIGHTFIRE, UPFIRE, DOWNFIRE}`.
2. **Yes/no:** `hazard_is_closing_on_player_row`.
3. **Choice:** `jump_direction` ∈ `{left, right, none}`.
4. **Yes/no:** `safe_to_start_climb`.
5. **Choice:** `next_target` ∈ `{left_rivet, right_rivet, upper_ladder, lower_ladder, wait, none}`.
6. **Yes/no:** `must_jump_cleared_gap`.
7. **Yes/no:** `hammer_is_present_and_worth_immediate_pickup` (only after code supplies legal route candidates).
8. **Choice:** `mode` ∈ `{idle_start, ramp_route, rivet_route, hazard_evasion, death_wait, screen_transition, game_over}`.

Code should validate that the chosen action is legal for the current hard guard (e.g., do not send climb through a known hole); Jev's answer is a recognition/choice, not a bypass of deterministic safety checks.

## Source list

- [AtariAge manual: controls, screens, hammer, bonus, scoring, three Marios](https://atariage.com/manual_html_page.php?SoftwareLabelID=149)
- [ALE DonkeyKong: action set, RAM observation, start behavior, modes](https://ale.farama.org/environments/donkey_kong/)
- [ALE environment spec: action semantics, frame skip, sticky actions](https://ale.farama.org/env-spec/)
- [OCAtari README and support/API](https://github.com/k4ntz/OC_Atari)
- [OCAtari Donkey Kong RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/donkeykong.py)
- [Verified Atari 2600 Donkey Kong disassembly](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm)
- [Verified port README and exact ROM provenance](https://github.com/Plicerin/donkeykong-rom-port)
- [Atari 2600 GameFAQs overview/walkthrough](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview)
- [Archived port bot's route and gap heuristics](https://github.com/Plicerin/donkeykong-rom-port/blob/main/archive_old_port/donkey-kong-bot.js)
