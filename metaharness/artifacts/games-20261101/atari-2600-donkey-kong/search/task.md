Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Donkey Kong` — Atari 2600 Donkey Kong is a local NTSC cartridge emulated by ALE/Stella. The objective is to guide Mario up the barrel/ramp screen, then remove all rivets on the rivet screen and reach the goal while preserving lives. The cartridge awards points for jumps, rivets, hammer smashes, and the screen bonus; score is the cumulative unclipped ALE reward, not the HUD bonus byte. A fresh reset always opens the same mode-0, difficulty-0 cartridge; the seed selects ALE's RNG stream and does not select a level. With the default standard-v5 sticky-action probability (0.25), an action can be repeated by the emulator; set repeat_action_probability=0.0 when calibrating deterministic traces.

Call tick(action) once per real-time quantum. Actions are the literal names in action_spec (noop, fire, up, right, left, down, upright, upleft, downright, downleft, upfire, rightfire, leftfire, downfire, uprightfire, upleftfire, downrightfire, downleftfire); integer ALE indices 0-17 are also accepted. Passing None means no fresh decision and repeats the last posted action (or noop before the first post). A malformed action is treated as noop and reported in the next observation instead of raising.

The JSON observation contains frame/tick timing, stage, Mario's bounding box and motion, visible barrel/fireball hazards with slots and motion, the ten fixed ladder boxes, hammer, four rivet-row status bytes, a derived rivets_removed count, BCD score display and bonus, cumulative score, lives, screens_completed, rewards, and terminal flags. raw_ram is the complete 128-byte RIOT RAM snapshot; the game-specific fields use the verified NTSC Donkey Kong RAM map (player at 0x93/0x9B, screen at 0x90, rivets at 0x9F-0xA2, lives at 0xA3, bonus at 0xA4).

Actions:
- noop: Hold the joystick neutral; does not start the attract screen.
- fire: Press FIRE; starts the cartridge and jumps in play.
- up: Climb a ladder when Mario is aligned with one.
- right: Walk right on the current girder.
- left: Walk left on the current girder.
- down: Climb down a ladder when aligned.
- upright: Move up and right; climb or jump depending on alignment.
- upleft: Move up and left; climb or jump depending on alignment.
- downright: Move down and right; climb or walk depending on alignment.
- downleft: Move down and left; climb or walk depending on alignment.
- upfire: Jump while steering upward/right as accepted by the cartridge.
- rightfire: Jump/steer right with FIRE.
- leftfire: Jump/steer left with FIRE.
- downfire: Jump/steer down with FIRE.
- uprightfire: Jump while steering up and right.
- upleftfire: Jump while steering up and left.
- downrightfire: Jump while steering down and right.
- downleftfire: Jump while steering down and left.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Each action is held for one fixed four-frame ALE quantum by default (about 66.7 ms at NTSC speed). FIRE is needed to leave the initial attract/idle state. UP/DOWN only climb when aligned with a usable ladder; direction+FIRE jumps. The default v5 sticky-action probability is 0.25, so the emulator may repeat the previous joystick action.

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

The game ticks every 0.06666666666666667s **on its own clock**; `ctx.act(a)` posts the current action (instant).
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
- strategy_summary: Start with one FIRE, then climb the fixed ramp/ladder topology while prioritizing survival and retaining the screen bonus. Track current walkway, target ladder, four obstacle slots, relative closing motion and jump cooldown in code; jump early toward a safe landing side and never walk through a cleared rivet gap. On the rivet screen, clear rows in a bottom-to-top serpentine route, reverse direction each row, and use raw rivet bytes to choose targets. Use the hammer only when it is on-route or an almost-free safe smash is available because it expires and blocks climbing. Astra chooses macro routes and risk tradeoffs, while Jev selects the immediate typed action/escape judgment as the emulator continues to tick.
- scoring: Report cumulative unclipped ALE reward as the primary score because it matches the cartridge's points and screen-completion bonus. Also report screens completed, current bonus, rivets removed, lives remaining, and episode termination; the displayed bonus byte is not the complete accumulated score. The manual awards 5,000 starting bonus per screen, 100 for jumping a barrel/fireball, 100 per rivet, and 800 for a hammer smash: https://atariage.com/manual_html_page.php?SoftwareLabelID=149
- interface.observation: A JSON state should contain frame, stage (idle/ramp/rivet/transition/game_over), player x/y/dx/dy and visibility, up to four generic hazard objects with positions and velocities, ten fixed ladder objects with visibility, hammer presence/position, four raw rivet-row bytes or a derived remaining-rivets set, BCD bonus, cumulative score, lives, episode return, terminal flags, and optionally the 128 raw RAM bytes. OCAtari objects expose category, xywh, visibility/history and HUD values; its Barrel class is a generic obstacle slot on the rivet screen, so use stage to interpret it as a fireball/hazard.
- interface.actions: Use ALE indices: 0 NOOP, 1 FIRE, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN, 6 UPRIGHT, 7 UPLEFT, 8 DOWNRIGHT, 9 DOWNLEFT, 10 UPFIRE, 11 RIGHTFIRE, 12 LEFTFIRE, 13 DOWNFIRE, 14 UPRIGHTFIRE, 15 UPLEFTFIRE, 16 DOWNRIGHTFIRE, 17 DOWNLEFTFIRE. FIRE starts the idle game and jumps during play; UP/DOWN climb only when aligned with a usable ladder; direction-plus-FIRE makes a directional jump. The actor must continue issuing actions every one or two four-frame steps while Astra thinks.

### study/strategies.md
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

…(truncated; full text in prior/strategies.md)

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

### study/brief.md
# Atari 2600 Donkey Kong — game brief

## Identity and scope

This study is for the **Atari 2600 retail port**, not the 1981 arcade game, ColecoVision, NES, or later Donkey Kong games. The modern ALE environment is `ALE/DonkeyKong-v5`; its bundled image is a 4 KiB NTSC cartridge. In the evaluator probe, ALE reported the ROM as `Donkey Kong (198x)` with MD5 `36b20c427975760cb9cf4a47e41369e4`, which is also the image verified frame-by-frame by the [Atari 2600 preservation/recompilation project](https://github.com/Plicerin/donkeykong-rom-port). The [AtariAge HTML manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=149) is listed under Atari 2600, although its embedded document heading calls the release “Donkey Kong (Coleco)”; use it for rules, not for assuming arcade boards or controls.

## Objective and progression

Mario must rescue his girlfriend from Donkey Kong. There are two alternating screens:

1. **Ramp/barrel screen (`gameScreen = 0`)** — climb the ramps and unbroken ladders to reach the top while avoiding or jumping barrels. The port's disassembly models a fixed alternating walkway/ladder topology and four active obstacle slots; it is much smaller than the arcade's full four-board game.
2. **Rivet/fireball screen (`gameScreen = 1`)** — remove every rivet by running or jumping over it while avoiding fireballs. The manual says that a removed rivet disappears; the cartridge logic treats the cleared location as a dangerous gap. The common Atari 2600 walkthrough describes two rivets on each of four rows (eight total); the cartridge stores four per-row rivet bytes, with left/right values `6` and `12` and complete value `18` ([verified disassembly constants and `CheckRivits`](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm)).

Completing either screen adds its remaining bonus to the score and switches to the other screen. After the first pair, the two screens continue to alternate and the action becomes harder; there is no separate arcade-style conveyor or pie-factory board in this port. The player receives three Marios total: the current Mario plus two additional lives. ALE's initial `lives` metadata is `2`, representing the two remaining Marios after the current one appears. The episode ends when no Mario remains.

**Progress variables worth reporting:** cumulative score, current screen (`ramp` or `rivet`), number of completed screens, current bonus, current lives, remaining rivets, and whether the current Mario is alive. A high score is useful, but a run that completes more screens with lives remaining is generally stronger than a risky run that farms 100/800-point bonuses and dies.

## Controls and action mapping

The manual states: hold the left joystick; left/right runs; up/down climbs when Mario is aligned with a ladder; and the red button starts a screen and makes Mario jump or grab the hammer. In ALE/Gymnasium, actions are integers in the following complete set ([ALE environment specification](https://ale.farama.org/env-spec/), [DonkeyKong page](https://ale.farama.org/environments/donkey_kong/)):

| ALE index | Action | Atari meaning in this game |
|---:|---|---|
| 0 | `NOOP` | Release joystick and button; no movement input |
| 1 | `FIRE` | Red button; starts the idle game and jumps when active |
| 2 | `UP` | Push joystick up; climb if aligned with a usable ladder |
| 3 | `RIGHT` | Run right |
| 4 | `LEFT` | Run left |
| 5 | `DOWN` | Push joystick down; descend if aligned |
| 6 | `UPRIGHT` | Up + right |
| 7 | `UPLEFT` | Up + left |
| 8 | `DOWNRIGHT` | Down + right |
| 9 | `DOWNLEFT` | Down + left |
| 10 | `UPFIRE` | Up + red button |
| 11 | `RIGHTFIRE` | Right + red button; jump while facing/moving right |
| 12 | `LEFTFIRE` | Left + red button; jump while facing/moving left |
| 13 | `DOWNFIRE` | Down + red button |
| 14 | `UPRIGHTFIRE` | Up + right + red button |
| 15 | `UPLEFTFIRE` | Up + left + red button |
| 16 | `DOWNRIGHTFIRE` | Down + right + red button |
| 17 | `DOWNLEFTFIRE` | Down + left + red button |

`ALE/DonkeyKong-v5` exposes all 18 actions. The game is initially idle: repeated `NOOP` produces no reward until `FIRE` is sent. A practical controller normally uses single-direction actions for walking/climbing and a direction-plus-`FIRE` action for a directional jump. The jump direction is latched when the jump starts, so set the desired left/right facing before pressing fire. Do not treat the integer `1` as “always jump”: it is also the start action while the attract/idle state is waiting.

## Scoring and time pressure

The manual's scoring table is:

| Event | Points |
|---|---:|
| Starting bonus on each screen | 5,000 |
| Jump over a barrel or fireball | 100 |
| Remove a rivet | 100 |
| Smash a barrel or fireball with the hammer | 800 |

While Mario is active, the displayed Score/Bonus value counts down; when a screen is completed, the remaining bonus is added to the score. The disassembly stores the bonus as BCD and decrements it every `0x7f` frames, approximately every 127 NTSC frames, or about 2.1 seconds at 60 Hz; each decrement is 100 displayed points ([manual, “How Time Flies?” and “SCORING”](https://atariage.com/manual_html_page.php?SoftwareLabelID=149), [bonus routine in the verified disassembly](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm)). If the bonus reaches zero, Mario is eliminated. Therefore speed is a real objective, but avoid sacrificing a life for an 800-point hammer hit.

For the adaptor, use the sum of **unclipped ALE rewards** as the primary score. It reflects the cartridge's actual point increments and completion bonuses. Keep `stage_completions`, `lives`, and `bonus` as auxiliary metrics. Do not report OCAtari's HUD `Score.value` alone as the episode score: in the current RAM decoder that field is derived from the displayed bonus byte, while the cumulative ALE reward is the reliable total.

## Screen mechanics

### Ramp/barrel screen

Mario begins at the bottom and must route upward through the fixed ladder topology. The verified cartridge has alternating ramps and a fixed set of ladder positions. OCAtari's current Donkey Kong decoder exposes the ten static ladder slots at screen coordinates (in the 160x210 ALE screen coordinate system):

```text
(108,176), (48,148), (80,148), (88,120), (108,120),
(48,92), (68,92), (108,64), (76,40), (32,40)
```

…(truncated; full text in prior/brief.md)

### Rivet/fireball screen

The goal is to remove all rivets, not merely reach the top. Running or jumping over a rivet removes it and gives 100 points. A removed rivet becomes a hole: walking into that gap without jumping kills Mario. Fireballs follow Mario, stay on their ramp in the normal case, and can sometimes jump across a gap. Prioritize a safe jump and a legal route over scoring every fireball.

The community Atari 2600 walkthrough recommends a bottom-to-top serpentine route: clear the two rivets on a row, move to the far side, climb the usable ladder, then clear the next row in the opposite direction. The exact walk direction should be selected from the current rivet mask and Mario's side; after one rivet is gone, jump across its hole rather than walking through it. Treat this route as a heuristic and verify it against the raw rivet bytes, because many online Donkey Kong guides describe the arcade version rather than this two-screen port ([Atari 2600 GameFAQs guide](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview), [archived port bot's explicit gap-crossing heuristic](https://github.com/Plicerin/donkeykong-rom-port/blob/main/archive_old_port/donkey-kong-bot.js)).

## Pace and tier

This is **tier 2**. The cartridge emulates an NTSC frame about every 16.7 ms; barrels/fireballs move, Mario's movement is updated every few frames, and the bonus continues to count down while a human is deciding. ALE's default `frameskip=4` advances four emulated frames per `step` (about 66.7 ms), and v5's default sticky-action probability is 0.25 ([ALE environment specification](https://ale.farama.org/env-spec/)). The source defines a 31-frame jump hang time (about 0.52 s) and a two-second-scale bonus tick ([verified disassembly](https://github.com/Plicerin/donkeykong-rom-port/blob/main/donkey_kong_2600.asm)).

A controller should refresh the immediate action every roughly 0.07–0.15 seconds (one or two fixed-4-frame steps) and keep a safe direction/jump policy running while Astra is thinking. A conservative `decision_latency_s` budget is **0.10 s** for a fresh hazard decision; a barrel can cover the remaining collision distance in a few such decisions. Jev's fast typed judgments should select the next immediate action, while Astra plans a longer ladder/rivet route asynchronously. Astra must not block the action loop for seconds; if an action call is delayed, the adaptor should continue a previously safe hold or jump policy.

## Sources

- [AtariAge Atari 2600 Donkey Kong manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=149)
- [ALE DonkeyKong environment](https://ale.farama.org/environments/donkey_kong/)
- [ALE environment specification](https://ale.farama.org/env-spec/)
- [ALE Python interface](https://ale.farama.org/python-interface/)
- [OCAtari object-centric Atari repository](https://github.com/k4ntz/OC_Atari)
- [Verified Atari 2600 ROM port and disassembly](https://github.com/Plicerin/donkeykong-rom-port)
- [Atari 2600 GameFAQs walkthrough](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview)

### adaptor/ADAPTOR.md
# Atari 2600 Donkey Kong adaptor

## Connection

`game.py` runs the maintained local ALE/Stella emulator through Gymnasium's
`ALE/DonkeyKong-v5` environment. It creates the environment with RAM
observations, fixed `frameskip=4`, no renderer, and no sound. OCAtari 2.2.1 is
used in RAM mode as a named-object decoder; the adaptor steps OCAtari's
underlying Gymnasium wrapper directly so Gymnasium's `(terminated, truncated)`
flags are not confused with OCAtari 2.2.1's swapped public return order.

The adaptor verifies the cartridge before construction. The bundled or
`ALE_ROMS_DIR` supplied `donkey_kong.bin` must have MD5
`36b20c427975760cb9cf4a47e41369e4`. It does not copy or redistribute the ROM.
There is no server, account, display, GPU, credential, or network dependency.

## Real-time contract

The returned object is Tier 2. One `tick(action)` applies one ALE action for
four emulated NTSC frames and returns immediately; the runner owns wall-clock
scheduling. `tick_s` is `4 / 60.0` seconds (about 66.7 ms), and `ticks` counts
successfully applied emulator quanta. The default `repeat_action_probability`
is the standard ALE v5 value, `0.25`, so Stella may repeat the previous action.
For deterministic calibration, construct with
`repeat_action_probability=0.0`.

`reset(seed)` always begins at the cartridge's same attract/idle opening. The
cartridge exposes only mode 0 and difficulty 0, so seeds do not choose boards
or levels. Seeds initialize ALE/Gymnasium randomness; with sticky actions
enabled they can produce different traces even though the first observation
is identical. The declared train and held-out seeds are separate RNG streams
of this same fixed scenario, not different level families.

## Actions

`action_spec` exposes all 18 ALE actions as lowercase strings:

- `noop`, `fire`, `up`, `right`, `left`, `down`
- `upright`, `upleft`, `downright`, `downleft`
- `upfire`, `rightfire`, `leftfire`, `downfire`
- `uprightfire`, `upleftfire`, `downrightfire`, `downleftfire`

`fire` is required to leave the initial idle state. UP/DOWN climb only when
aligned with a usable ladder; directional FIRE actions jump/steer. Integer
indices 0--17 are accepted as a convenience. Passing `None` means that no new
controller decision arrived, so the last posted action is repeated (or `noop`
before the first post). A malformed or illegal action is converted to `noop`
and reported in the next observation rather than raising.

## Observation

Every observation is a strict JSON dictionary under 200 kB. It includes:

- `stage` (`idle`, `ramp`, `rivet`, `transition`, `game_over`, or `unknown`),
  frame/tick counters, terminal flags, last reward/action/error;
- `player`, four-slot `hazards` (OCAtari's `Barrel` slots are called barrels
  on the ramp and fireballs on the rivet screen), ten fixed `ladders`, and
  `hammer`, with integer boxes and motion where available;
- `rivet_rows`, the four raw row status bytes (0, 6, 12, or 18 in the normal
  cartridge), and `rivets_removed` (`sum(row // 6)`);
- `bonus_display_bcd`, `current_bonus` (the BCD timer converted to points;
  `$50` is the 5,000-point starting bonus), HUD score bytes, cumulative
  `score_total`/`episode_return`, `screens_completed`, and
  `lives_remaining`; and
- the complete 128-byte `raw_ram` snapshot.

The verified RAM map used by the JSON decoder is: obstacle Y `$80--$83`, jump
state `$85`, hammer state `$86`, score bytes `$87--$88`, game state `$8C`, life
loss `$8D`, screen `$90`, player X `$93`, player Y `$9B`, rivets
`$9F--$A2`, lives `$A3`, bonus `$A4`, obstacle X `$C1--$C4`, directions
`$C5--$C8`, and ladder numbers `$C9--$CC`. OCAtari supplies the player,
hazard, ladder, hammer, and HUD object geometry. Its HUD `Score.value` is not
used as the primary score because it is only a display/timer value.

## Score and limits

`score` is the cumulative **unclipped ALE reward** from every four-frame
step, including point awards and screen-completion bonus. It is not derived
from the one-byte bonus display. `score_breakdown()` additionally reports ALE
reward, current bonus, rivets removed, lives, screens completed, and terminal
indicators.

The adaptor does not provide save-state cloning to the harness and does not
start on the rivet screen; normal evaluation uses full episodes from reset.
ALE's own clone/restore facilities remain available only inside local
calibration experiments, outside this public interface. No screenshots are
required or generated.

### study/interfaces.md
# Programmatic interfaces for Atari 2600 Donkey Kong

## Recommendation

Use **ALE/Stella through `ale-py` + Gymnasium, with OCAtari's RAM decoder as the object layer**. In other words, let the official maintained Atari environment own stepping, terminal handling, seeding, and rewards; keep the raw 128-byte RAM on every observation; and use OCAtari 2.2.1 to turn the RAM into named objects for Jev and Astra. This is the only option found that is simultaneously headless, local, low-latency, Python 3.12-compatible, supplied with a matching Donkey Kong ROM, and already able to expose `Player`, `DonkeyKong`, `Hammer`, obstacles, ladders, life, and HUD values.

The evaluator's Python was tested directly:

```text
Python 3.12.3
ale-py 0.12.1
Gymnasium 1.4.0
OCAtari 2.2.1
```

The test created `ALE/DonkeyKong-v5` with RAM observations and verified `Discrete(18)`, a `(128,)` `uint8` observation, and a working `OCAtari(..., mode="ram")` object list. The `ale-py` wheel supplied the exact 4 KiB NTSC image whose MD5 is `36b20c427975760cb9cf4a47e41369e4`; the same image is audited frame-by-frame in the [Plicerin Atari 2600 port](https://github.com/Plicerin/donkeykong-rom-port).

### Why not use only OCAtari?

OCAtari is the best symbolic convenience layer, but its current 2.2.1 `step` implementation returns the Gymnasium terminal flags in the order `(obs, reward, truncated, terminated, info)`, while Gymnasium specifies `(obs, reward, terminated, truncated, info)`. The flags are equal during ordinary play, so this can remain hidden until a death or game-over transition. The adaptor should either (a) step the underlying Gymnasium object and call the OCAtari RAM detector explicitly, or (b) use OCAtari but swap/guard the two flags at its boundary after checking the installed version. Always retain direct ALE state and raw RAM; OCAtari does not label every game-specific fact.

## Chosen stack: exact setup contract

### Software

Install into the evaluator venv, not the system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-donkey-kong/.venv/bin/pip install \
  ale-py==0.12.1 gymnasium==1.4.0 ocatari==2.2.1
```

`ocatari` brings OpenCV, NumPy, and pygame dependencies. No display server, GPU, browser, account, API key, or remote server is required for `mode="ram"` and `render_mode=None`. The ALE and Gymnasium setup patterns are documented in the [ALE getting-started guide](https://ale.farama.org/getting-started/), [ALE Python interface](https://ale.farama.org/python-interface/), and [OCAtari README](https://github.com/k4ntz/OC_Atari).

### Environment variables and credentials

There are **no required credentials or service variables**.

- `ALE_ROMS_DIR` — optional. Set it only if the adaptor must load a user-supplied compatible ROM directory; otherwise use `ale_py.roms.get_rom_path("donkey_kong")`, which found the bundled ROM in the evaluator test. Do not put a ROM path or ROM bytes in a credential file or commit a copyrighted dump.
- `SDL_VIDEODRIVER=dummy` — optional fallback if a future OCAtari/pygame code path initializes SDL. It is not needed for the tested RAM-only, no-render path.

…(truncated; full text in prior/interfaces.md)

- **MAME/arcade Stella variants:** They emulate a different arcade/arcade-like Donkey Kong ruleset. They are not interchangeable with the Atari 2600 two-screen ROM and would produce the wrong ladder, rivet, hazard, and scoring behavior.
- **Small Python emulators such as `pytari2600`:** Useful educational code, but no established exact-ROM, object-level, Python 3.12 adaptor was found. Their maturity and timing fidelity are insufficient compared with ALE/Stella.

## Adapter acceptance tests

Before giving the interface to the game-playing harness, verify:

1. `ale_py.roms.get_rom_path("donkey_kong")` exists and its MD5 is `36b20c427975760cb9cf4a47e41369e4`; otherwise stop rather than silently evaluating a different port.
2. RAM mode is `(128,)`, `uint8`; no SDL window is opened; no screenshot is required.
3. `reset(seed=0)` reports two spare lives and an idle/ramp initial state; an index-1 `FIRE` action starts play.
4. Action indices match the [ALE table](https://ale.farama.org/env-spec/); especially, `RIGHT=3`, `LEFT=4`, `UP=2`, `DOWN=5`, and `FIRE=1`.
5. A JSON observation contains the player, four hazard slots, ladder slots, stage, bonus, rivet bytes, lives, cumulative reward, and terminal flags.
6. The adaptor notices a life-loss transition without confusing `truncated` and `terminated`, and stops after true terminal game-over.
7. A fixed seed/action trace is repeatable with sticky actions disabled; a separate run at `repeat_action_probability=0.25` exercises robustness.
8. The cumulative reward, not just the displayed bonus, is reported as score.


## Measured model latency (at archive creation, 2026-10-31 02:55)
- System-2 (Astra), one-word answer: 9.4 s
- System-2, an answer of ~853 output tokens: 61 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 141 ms (= 2.1 ticks), p90 161 ms, max 164 ms over 12 warm requests; first request on a new connection 155 ms — a timeout below the tail turns slow answers into failed calls
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
