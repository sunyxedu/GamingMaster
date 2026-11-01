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

These are useful geometry hints, not a license to climb every slot blindly: ladder availability/side and the current walkway matter. Run toward the next usable ladder, align horizontally, then hold `UP`. The manual explicitly warns that Mario cannot climb a broken ladder. A current-walkway/target-ladder planner is safer than a single fixed sequence because barrels and jumps can leave Mario on a different side.

Donkey Kong releases rolling barrels. They move along the ramps and can use some ladders; the cartridge has at most four simultaneously active obstacle slots. Jump a barrel when it is approaching on Mario's walkway, preferably with Mario and the barrel moving toward one another as the manual recommends. A jump is safer than trying to squeeze around a collision, but jumping too late, jumping while misaligned with a ladder, or landing at the edge can waste a life. The hammer is a short-lived power-up (the community guide places it on the second-from-top floor): grabbing it allows Mario to smash an approaching barrel for 800 points, but Mario cannot climb while carrying it and a missed smash still kills him ([Atari 2600 GameFAQs guide](https://gamefaqs.gamespot.com/atari2600/584689-donkey-kong/faqs/80889/game-overview), [manual Step 3](https://atariage.com/manual_html_page.php?SoftwareLabelID=149)).

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
