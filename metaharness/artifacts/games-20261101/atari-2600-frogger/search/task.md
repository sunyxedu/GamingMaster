Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Frogger` — Atari 2600 Parker Brothers Frogger, emulated headlessly by the official ALE/Stella integration. The adaptor starts the cartridge's own start phase, then advances a seed-selected live traffic phase; agent ticks begin at the playable frog. Move one frog from the bottom sidewalk across five alternating road lanes, a median, and five river rows into one of five unoccupied home bays. Cars kill; water and unsafe/diving carriers kill; a timeout or exhausted reserve ends the run. Five occupied homes advance the cartridge to a harder level. Score follows the on-screen BCD score: a forward hop is 1, a home is 5, the five-home bonus is 100, and the manual also allows lady-frog, fly, and remaining-time bonuses.

This is real-time at 60 Hz: traffic and carriers animate during a NOOP. Legal action strings are exactly NOOP, UP, RIGHT, LEFT, and DOWN; each tick applies one of them for one emulator frame. In the standard modes a direction is a single edge-triggered hop, so use NOOP between hops. Mode 2 is Speedy Frogger and can repeat held directions. A malformed action is ignored as NOOP and reported in the next observation.

observe() is strict JSON. It includes raw RAM for $80-$FF, exact score/reward/life/timer fields, frog row and x, both decoded object tracks per road/river row with x, direction, velocity, carrier color/shape and diving hazard, five home occupancy records, bonus frog/fly candidates, collision flags, and ROM/configuration metadata. Object x values are the cartridge's 0-160 world pixels; tia_x is the render-position byte. mode=0,difficulty=0 is primary. Separate mode=1, mode=2, and difficulty=1 constructions are supported.

Actions:
- NOOP: Leave the frog's joystick centered for this Atari frame.
- UP: Make one hop toward the river and home bays.
- RIGHT: Make one hop to the right.
- LEFT: Make one hop to the left.
- DOWN: Make one hop toward the starting sidewalk.
(real-time games: ticks apply 'NOOP' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
One tick is one 60 Hz emulator frame.  In games 1-4 a direction is edge-triggered as one hop; use NOOP between intentional hops. Mode 2 is Speedy Frogger, where a held direction can repeat.

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
- strategy_summary: Strong play alternates short forward hops with deliberate NOOP waits: cross each traffic lane through a predicted gap, pause at the median, then ride broad logs or safe red turtles while compensating for platform drift. Never jump sideways off a platform end, remain on a light-blue diving turtle, enter an occupied or crocodile-threatened home bay, or let a Difficulty-A carrier reach the screen edge. Use code for RAM decoding, score/life bookkeeping, velocity and collision intervals, debounce, and clone/restore; have Astra choose routes and trade bonuses against survival, while Jev answers immediate hazard, safe-wait, platform, bay, and next-action questions. Re-estimate object speeds and become more conservative after each five-home difficulty increase.
- scoring: Report cumulative ALE reward and the decoded on-screen score as the primary score because ALE computes Frogger reward as the change in the BCD/decimal score at RAM 0xCC/0xCE. Also report score_delta, frogs_home, deaths, reserve_frogs, time_bonus, level, and terminal_reason: the manual awards 1 for a forward jump, 5 for a home, 100 for five homes, 20 for a lady frog, 20 for a fly, and 2 per second remaining. Sources: https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp and https://atariage.com/manual_html_page.php?SoftwareLabelID=194
- interface.observation: The adaptor should expose a JSON object with raw_ram (128 uint8 values), episode_frame_number, frame_number, ALE lives, score decoded from the BCD/decimal fields at 0xCC and 0xCE, terminal/life state from 0xD0, and calibrated named frog, road, river-platform, home-bay, timer, hazard, and level fields. ALE itself provides RAM, reward, lives, frame counters, and optional RGB/grayscale; it does not name cars or turtles, so object-level fields must be produced by the adaptor and retain decoder confidence plus raw_ram. Sources: https://ale.farama.org/python-interface/ and https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp
- interface.actions: Send one of the five minimal ALE actions each step: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN. Use one-frame stepping and debounce intentional hops in games 1-4; NOOP is the safe wait action. Fire is unnecessary. Mode 2/game 5 is Speedy Frogger and requires explicit handling of held/repeated directions. Source: https://ale.farama.org/environments/frogger/

### study/strategies.md
# Atari 2600 Frogger strategies

## Operating model

The board is a moving-gap problem, not a fixed sequence. Each frog must cross five alternating traffic lanes, use moving river platforms to cross five rows, and enter one of five open home bays within 30 seconds. The manual explicitly says traffic has different speeds and alternating directions, floating objects can carry the frog, diving turtles change from safe red to unsafe light blue, and later levels add snakes ([Parker Brothers manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)).

The Atari port’s minimal action set is only `NOOP`, `UP`, `RIGHT`, `LEFT`, `DOWN`; ALE’s exact indices are 0–4 ([ALE Frogger](https://ale.farama.org/environments/frogger/)). The agent should treat a direction as a request for one grid hop in games 1–4, then use `NOOP` to wait for a safe gap. Do not send a direction repeatedly merely because it was the last successful action. Modes 5/6 are Speedy Frogger, where holding a direction repeats hops, so the adaptor must distinguish a fresh press from a held action when that mode is deliberately tested.

## Strong-player playbook

### 1. Start and road crossing

1. **Use the sidewalk and median as observation points.** The sidewalk below the road and the riverbank between road and river are safe pauses. Do not spend the whole 30-second budget waiting, but use one or two frames/short batches to estimate traffic before committing.
2. **Cross in straight, deliberate hops.** Forward progress earns a point, but the important objective is a safe crossing. Align the frog with a gap, hop up one lane, reassess, and correct left/right only when a vehicle is not overlapping the frog’s landing corridor.
3. **Track each lane separately.** Traffic lanes move in alternating directions and at different speeds. Estimate per-lane object position and velocity from consecutive RAM observations; predict the landing interval rather than using a fixed “up every N frames” script. A gap that is safe in lane 1 may be unsafe one frame later in lane 3.
4. **Prefer a controlled wait to a blind hop.** `NOOP` is a real action. If the next lane is blocked, wait for the nearest vehicle group to pass; if the frog is already in a safe median, wait for a better phase. Never spend a life trying to gain one forward point when the landing cell is occupied.
5. **Keep a lateral escape option.** Avoid moving so close to a lane edge or vehicle that a necessary correction would land on another vehicle. If the decoder supplies bounding boxes, use frog width plus a collision margin, not just the frog’s centre pixel.

These are code-level predictions, not a fixed route: the manual’s alternating speeds mean a held-out seed or later level changes the gap timing.

### 2. River crossing

1. **Land on a platform row, not water.** Logs, turtles and alligators are usable carriers; the frog cannot swim. Before an upward hop, find a platform whose horizontal interval overlaps the frog’s landing interval.
2. **Ride with the platform and compensate for drift.** While on a log/turtle/alligator, its horizontal motion carries the frog. Keep the frog away from the screen edge and use small lateral corrections to stay on a broad central section. Maintain a per-row velocity estimate and recompute after every action.
3. **Do not jump sideways off an end.** The manual says sideways jumps off the end of a log or the end turtles land in water. Forward/backward transitions between rows are safer; lateral moves should be made while the frog is supported by the middle of the object.
4. **Treat diving turtles as time-limited platforms.** Red turtles are safe; light-blue turtles are about to dive or resurface. If the current platform changes to light blue, choose the next solid row immediately rather than waiting for another hop cycle. Never plan on a fixed turtle pattern.
5. **Avoid mouths and late-level snakes.** An alligator tail/back is safe, but its jaws are not. Snakes can appear on the riverbank and later on logs; a platform that is geometrically present is not necessarily safe. Use a hazard flag and do not choose a landing interval intersecting the mouth/head.
6. **Respect difficulty A versus B.** On Difficulty A, being carried off-screen kills the frog; on Difficulty B, the frog reappears on the opposite side, although it still cannot hop off-screen. The mode must be read from the scenario rather than assumed from a generic Frogger strategy.

### 3. Home entry

1. **Align before the final hop.** Enter a bay from directly in front of its centre. The side shrubbery is a collision hazard, and a bay already containing a frog is unavailable.
2. **Check the bay itself, not just its x-coordinate.** An alligator head showing in a bay is dangerous; a just-beginning-to-surface head may be safe. A fly is an optional 20-point bonus. Choose another bay or wait if the target is occupied or hazardous.
3. **Do not throw away a completed crossing for a bonus.** Lady frog and fly bonuses are worthwhile only when the route remains safe. The priority is five homes, the 100-point five-frog bonus, and the time bonus obtained by finishing alive.

### 4. Score and life management

The manual awards 1 point per successful forward jump, 5 for home, 100 for five homes, 20 for a lady frog, 20 for a fly, and 2 per second remaining. It also grants an extra frog at 1,000 points if fewer than four frogs remain ([manual scoring section](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)). Therefore:

- code should calculate and report the exact score from ALE’s BCD score fields rather than infer it from movement count;
- Astra should choose the safe route first, then use remaining time and bonus opportunities as tie-breakers;
- Jev should answer whether a bonus is safely available, not make the arithmetic itself;
- a life-preserving route with a small time loss is usually better than a fast route that risks a restart.

ALE’s Frogger integration defines reward as the change in decoded score and terminal as the life field reaching `0xFF`, so cumulative ALE reward is the primary evaluation score ([Frogger.cpp](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)). Report `score`, `frogs_home`, `deaths`, `time_bonus`, and terminal reason alongside it.

### 5. Progression and scenario adaptation

After five frogs reach home, the game advances to a harder level. Traffic becomes harder, floating objects become fewer or vary speed, and snakes become more frequent ([manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)). The agent should:

- retain learned object velocities only for the current level/seed;
- detect the level transition from score/home count and changed object statistics;
- widen collision margins and shorten Jev’s action horizon at higher levels;
- prefer broad, long logs and red turtle groups over narrow or diving groups;
- re-estimate every lane/river row after a death or level transition.

ALE exposes three single-player mode settings: its `mode=0,1,2` selection presses the cartridge Select switch to game 1, game 3, and game 5, respectively; the even two-player games are skipped. Difficulty has two settings corresponding to the manual’s off-screen behavior ([ALE Frogger implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)). Modes 0 and 1 should be the initial training/scenario defaults; mode 2 is specifically useful to test held/repeated input.

## What belongs where

| Knowledge or behavior | Put in code | Put in Astra prompts | Put in Jev questions |
|---|---|---|---|
| Action map, one-hop debounce, mode/difficulty, frame clock and reset/seed | Exact lookup and state machine | Explain a mode change if an observation is ambiguous | None; do not ask Jev to remember indices |
| RAM-to-address conversion, BCD score, `0xD0` terminal/lives, timer conversion, level/home/death bookkeeping | Yes; deterministic parsing and counters | Use the resulting named fields | Read typed fields, never calculate them |
| Vehicle/platform position, velocity, width, row and collision margins | Yes; track from consecutive frames and reject unsafe intervals | Ask Astra to choose a route/risk policy using the tracks | “Is landing interval safe?” / “Which of {UP, LEFT, RIGHT, DOWN, NOOP} is safe now?” |
| Road gap selection and when to wait | Candidate generation, interval collision test | Select which lane/gap to target and adapt after misses | Binary `blocked_now`, choice `best_action`, confidence |
| River carrier choice and drift compensation | Platform tracking and deterministic feasibility checks | Choose a sequence of rows and decide when to abandon a diving carrier | `safe_platform_kind`, `diving`, `edge_risk`, `escape_now` |
| Home-bay occupancy, crocodile/head/fly and side-shrub hazards | Exact hazard flags and available-bay list | Decide whether to take a bonus or use a safer bay | `home_bay_safe` and `best_bay` choice |
| Search over possible short action sequences / snapshots | Clone/restore, rollout bookkeeping, collision checks | Plan several hops, compare risk versus score, explain recovery | Evaluate the current frame only; no long-horizon plan or arithmetic |
| Failure recovery and higher-level adaptation | Detect death/reset, preserve score/death history | Change policy at harder levels and choose conservative versus aggressive play | Fast recognition of the immediate hazard and next action |

## Jev question set

Give Jev only the current normalized JSON plus a small, fixed answer schema. Good independent questions are:

- `current_zone`: `SIDEWALK`, `ROAD`, `MEDIAN`, `RIVER`, `HOME_APPROACH`, `DEATH`, `TERMINAL`;
- `immediate_hazard`: `NONE`, `CAR`, `WATER`, `DIVING_TURTLE`, `MOUTH`, `EDGE`, `TIMEOUT`;
- `safe_to_wait`: yes/no;
- `best_action_now`: one of the five ALE action labels;
- `best_landing_platform`: one row/object ID or `NONE`;
- `platform_is_diving`: yes/no/unknown;
- `home_bay_safe`: one bay ID or `NONE`;
- `escape_required`: yes/no;
- `confidence`: high/medium/low.

Jev should not be asked “what is the best route for the next 20 seconds?” or “compute the projected score.” Those are Astra/code tasks. If the decoder is uncertain, include `unknown` and let Jev choose `NOOP` or a conservative action rather than hallucinating an object.

## Common mistakes and counters

- **Holding UP/LEFT/RIGHT in games 1–4:** produces unintended repeated/edge-triggered hops. Debounce in code and use `NOOP` between intentional hops.
- **Following a fixed pixel route:** traffic/platform speeds vary and later levels change them. Track velocity each frame.
- **Jumping from a platform end sideways:** causes water deaths. First move onto the platform’s interior, then make a forward/backward transition.
- **Staying on a light-blue turtle:** it is about to dive. Ask Jev for `escape_required` and switch to a known solid platform.
- **Ignoring screen edges:** Difficulty A kills a frog carried off-screen. Predict drift and leave the edge early.
- **Entering a bay without checking occupancy/crocodile:** the final hop can kill a nearly successful frog. Maintain a bay availability table.
- **Rushing for two points per remaining second:** time bonus is secondary to preserving a frog. Use it only when the next landing has a positive safety margin.
- **Treating raw ALE `lives=4` as five:** it is the reserve field at reset. Code must distinguish reserve count, active frog, home count, and terminal.
- **Letting Astra think while Jev has no safe action:** a 30-second run is real time. Jev needs a short recurring cadence, and Astra should be given a snapshot/current plan while Jev maintains a verified `NOOP` or safe ride.

### study/brief.md
# Atari 2600 Frogger — study brief

## Identity and objective

This is the Parker Brothers Atari 2600/VCS release of *Frogger* (1982), not the later Starpath “Official Frogger” cartridge. The objective is to move as many frogs as possible from the bottom sidewalk, across five lanes of traffic and five river rows, into the five home bays at the top while maximising score. The official ALE description is the same objective: cross a highway and a perilous river without being crushed or eaten ([ALE Frogger](https://ale.farama.org/environments/frogger/)).

The board, from bottom to top, is:

1. sidewalk/start area;
2. five road lanes whose traffic alternates direction and has different speeds;
3. a riverbank/median;
4. five river rows containing logs, turtles, diving turtles, alligators and, at harder stages, snakes; and
5. five separate home bays.

The original manual is the authoritative rules source for these mechanics ([Parker Brothers manual, AtariAge](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)).

## Win/progress and failure

- A frog is safe only after entering an unoccupied home bay. Five bays can be filled, but a bay cannot be reused while occupied.
- The player starts with five frogs: one active frog and four reserve indicators. A life is lost on a vehicle collision, water, a diving turtle, an alligator/snake mouth, timeout, or (on Difficulty A) being carried off the screen by a floating object.
- The game ends when no frogs remain. Every five frogs brought home advances to a harder level: traffic becomes harder, floating objects become fewer or change speed, and snakes appear/proliferate.
- The ALE implementation exposes the game’s reserve/life field at RAM address `0xD0` and treats `0xD0 == 0xFF` as terminal; its reset value is 4 because the active frog is not counted in that field ([ALE Frogger implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)). Report both the raw ALE `info["lives"]` and a separately named “active + reserve” interpretation; do not silently call the raw value “five lives.”

## Score

The manual gives the following score events:

| Event | Points |
|---|---:|
| Successful forward jump | 1 |
| Frog reaches a home bay | 5 |
| Five frogs reach home | 100 bonus |
| Lady frog carried home | 20 |
| Fly eaten in a home bay | 20 |
| Each second remaining at home | 2 |
| Extra frog at 1,000 points when fewer than four frogs remain | one extra frog |

ALE decodes the score from RAM addresses `0xCE` and `0xCC` as decimal/BCD and emits reward as the change in that score, so cumulative ALE reward is the most convenient machine score ([Frogger.cpp](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)). The adaptor should additionally expose `raw_score`, `score_delta`, `frogs_home`, `deaths`, `level`, and `terminal`; the latter fields distinguish safe progress from a lucky or incomplete run.

## Controls and actions

The Atari joystick makes a single frog hop in four directions. In games 1–4, move the joystick once for one hop; games 5–6 are “Speedy Frogger” and repeat hops while a direction is held. The manual also defines the console Select, Reset, and left Difficulty switch; the adaptor should start/reset through the emulator rather than fake keyboard input.

ALE’s minimal action set is exactly ([ALE Frogger docs](https://ale.farama.org/environments/frogger/)):

| ALE action | Meaning |
|---:|---|
| 0 | `NOOP` |
| 1 | `UP` |
| 2 | `RIGHT` |
| 3 | `LEFT` |
| 4 | `DOWN` |

Fire is not needed. The full Atari 18-action set is unnecessary and makes action selection less safe.

## Pace and decision budget

This is not turn based. Traffic and river objects move on every emulator frame while the agent is thinking. A run has only 30 seconds from sidewalk to home; the manual gives a warning at five seconds remaining. The ALE v5 default is four emulator frames per agent step with 25% sticky-action probability, but the recommended control configuration for this adaptor is `frameskip=1` and `repeat_action_probability=0.0`, so the harness sees each frame and can make deterministic one-hop/no-op decisions ([ALE Frogger environment source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)).

Use an engineering decision budget of about **0.5 seconds** for a normal decision. A frog can wait on the sidewalk or median, but a moving car, a river platform edge, or a diving-turtle transition can punish a stale decision in a fraction of a second. Jev should therefore issue short typed action judgments continuously while Astra plans at a slower cadence; the adaptor can advance one frame or a very small fixed batch while Astra is unavailable. A long Astra call is acceptable only when Jev is holding `NOOP` or another already-safe action and the current platform/traffic gap has been checked.

For reproducible experiments, seed `reset(seed)` and report the ALE episode frame number. The official Gymnasium/ALE interface returns RAM observations of 128 bytes, lives, episode-frame number, and frame number; it does not require a display ([ALE Python interface](https://ale.farama.org/python-interface/), [ALE Gymnasium environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)).

## State representation to build

Use the 128-byte RAM observation rather than screenshots. ALE maps the console’s `$80–$FF` RAM into `ram[0:128]`, so a 6502 address `a` is read as `ram[a - 0x80]`. The official game integration gives the high-confidence fields:

- score: `0xCC`/`0xCE` (BCD/decimal score pair);
- reserve/life/terminal: `0xD0`;
- selected single-player game mode: `0xDD` is used by ALE while selecting modes.

A public 2025 community disassembly supplies useful, but **not official**, starting hypotheses for decoding the rest: frog position/state candidates around `0xB0`/`0xB4`, time counter `0xE0`, and collision/object flags around `0xE3`/`0xE5`/`0xE7` ([Parker Brothers Frogger disassembly gist](https://gist.github.com/erik-nilcoast/cc55644c916c729e6a82ae07b681ac2b)). Validate those hypotheses against the exact ALE ROM hash and controlled probes before relying on them. The production JSON observation should contain named fields plus the raw RAM array for auditability, not a screenshot-only representation.

## Sources

- [AtariAge Parker Brothers Frogger manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)
- [ALE Frogger environment](https://ale.farama.org/environments/frogger/)
- [ALE Python interface](https://ale.farama.org/python-interface/)
- [ALE Frogger-specific scoring, lives, modes and actions](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)
- [ALE Gymnasium environment implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)
- [Parker Brothers Frogger community disassembly](https://gist.github.com/erik-nilcoast/cc55644c916c729e6a82ae07b681ac2b)

### adaptor/ADAPTOR.md
# Atari 2600 Frogger adaptor

## Connection and lifecycle

`game.py` uses `ale-py==0.12.1` and `gymnasium==1.4.0` through the
headless ALE/Stella `ALE/Frogger-v5` environment. It loads the bundled Parker
Brothers Atari 2600 Frogger ROM and verifies MD5
`081e2c114c9c20b61acf25fc95c71bf4`. No display, GPU, server, account, network,
or credential is needed. `close()` closes the emulator and is idempotent.

Each instance is isolated. `reset(seed)` resets ALE, waits through the
cartridge's own attract/start phase (the five-action minimal set has no FIRE),
then advances the live emulator with NOOP for a deterministic 1--90 frame
seed phase. The startup and phase lengths are exposed in both `info()` and the
observation; they are not counted in agent `ticks`. This matters because a bare
ALE reset has the same initial traffic arrangement for every seed.

## Tier and actions

This is Tier 2. `tick(action)` applies one action to one Atari display frame;
`tick_s = 1/60` and `ticks` counts agent-applied frames. The runner owns the
clock. Legal action strings are:

- `NOOP` -- centered joystick / wait for this frame.
- `UP` -- one hop toward the river and homes.
- `RIGHT`, `LEFT`, `DOWN` -- one lateral or backward hop.

In standard games (modes 0 and 1), a direction is edge-triggered as one hop;
controllers should use NOOP between deliberate hops. Mode 2 is Speedy Frogger,
where holding a direction can repeat. A malformed action is safely converted
to NOOP and recorded as `invalid_action` in the next observation.

## Observation and decoder

Every observation is strict JSON and currently about 10 kB. `raw_ram` retains
all 128 bytes for `$80--$FF`, while named fields decode the state needed for
play:

- `frog`: row/zone, world x, TIA render x, platform-carry flag, and state.
  Rows 0--4 are road lanes, 5 is the median, 6--10 are river rows, `0xff` is
'the start sidewalk, and 11 is the home approach.
- `road` and `river`: five rows each, with two tracked objects per row. Each
  object has the cartridge world x, wrapped collision interval estimate, TIA
  position byte, direction, measured velocity in pixels/tick, shape pointer,
  NUSIZ size code, palette/color code, kind, and hazard flag. River color/shape
  decoding identifies brown `log`, olive `turtle`, and blue `diving_turtle`
  carriers; unfamiliar later-level forms remain explicitly
  `river_hazard_or_carrier` rather than being silently called safe.
- `homes`: five fixed bay centers (x 16, 48, 80, 112, 144), exact occupancy
  from `E3 & 0x1f`, availability, and the current target during home entry.
  `bonus_flags` and raw `C7/F9` are retained because this ROM does not expose a
  separate home-crocodile byte. `crocodile` is therefore `unknown`, never a
  false safety claim.
- `lady_frog` and `fly`: presence, row, x, and render position from the ROM's
  bonus-object fields.
- `score`, `score_delta`, `reward`, and `total_reward`: score is the on-screen
  packed-BCD value from `$CC/$CE`, and reward is ALE's per-frame score change.
- `reserve_frogs`, `frogs_remaining_including_active`, `deaths`, `timer_s`,
  `time_bonus`, `time_bonus_available`, `frogs_home`, `level`, terminal flags,
  and `terminal_reason`.
- `mode`, `difficulty`, `rom_hash`, `frameskip`, `sticky_action_probability`,
  requested seed, ALE RNG seeds, startup/phase frames, and frame counters.

Object tables are decoded from the exact ROM pointers held at `C0/C2`, `B8/BA`,
`BC/BE`, and `C4`; the alternating direction table is the cartridge's
`$FD47` table. `B0/B1/B2` are the direct 0--160 world x values and `B4/B5/B6`
are the render-position bytes. The raw bytes and table codes remain alongside
the friendly names so a later mode or level can be handled without losing
information. RAM/object positions were compared with ALE RGB rendering for
three seeds, modes 0/1, and 0/30/90/180 ticks; see `review.md` and the
runtime calibration report.

## Score and scenarios

`score` is the decoded on-screen score and is the primary evaluator score;
`total_reward` is cumulative ALE reward and should agree with score changes.
The manual events are 1 per forward hop, 5 per home, 100 for five homes, 20
for a lady frog, 20 for a fly, and 2 per second remaining. `score_breakdown()`
reports score, ALE reward, score delta, cumulative homes, deaths, reserve frogs,
awarded/available time bonus, and level. The current timer-derived bonus is
also exposed separately so a controller does not confuse possible points with
points already awarded.

The declared `SEEDS` train/held-out split is mode 0, difficulty 0. Every seed
uses a distinct seed-selected live traffic phase. `SCENARIO_SEEDS` provides
separate eight-seed groups for mode 1 (harder single-player), mode 2 (Speedy
Frogger), and difficulty 1 (floating objects wrap rather than killing the frog).
Use `make_game(mode=..., difficulty=...)` for those groups; scores are never
mixed between configurations. ALE mode and difficulty, ROM hash, frameskip,
sticky-action probability, and all seed metadata are recorded per episode.

## Limits

ALE's public RAM interface does not provide a first-party object graph or a
separate home-crocodile variable. The adaptor exposes exact RAM flags and
marks that one home hazard as unknown rather than inventing it. NUSIZ intervals
are conservative render/collision estimates; controllers can use the exact
world x, TIA byte, shape pointer, and raw RAM for their own margins. The
bundled cartridge remains subject to the user's applicable ROM distribution
rights.

### study/interfaces.md
# Programmatic interfaces for Atari 2600 Frogger

## Recommendation

Choose **ALE (`ale-py`) through its Gymnasium API, with `obs_type="ram"` and a small Frogger RAM decoder**. It is the only option found that combines a maintained, native, headless Atari emulator, a built-in Parker Brothers Frogger ROM/integration, deterministic seeding, direct 128-byte RAM access, score/life extraction, snapshots, and a five-action minimal action set. It does not provide object names directly, so the adaptor must turn RAM into JSON; that is still preferable to a screenshot pipeline on this server.

Use `ale-py==0.12.1` and `gymnasium==1.4.0` in the evaluator’s Python 3.12.3 venv (or pin the current compatible versions at adaptor build time). The current ALE package supports CPython 3.11–3.15 and bundles supported Atari ROMs; its ROM table includes `frogger.bin` ([ALE README](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [ALE ROM manifest](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/roms/md5.json), [PyPI ale-py](https://pypi.org/project/ale-py/)). No account, server, API key, `DISPLAY`, or GPU is needed.

## Comparison of practical options

| Interface | Maturity | Latency | What it exposes | Legal/ToS risk | Cost and practical verdict |
|---|---|---|---|---|---|
| **ALE Python/Gymnasium (chosen)** | High; official Farama project, native Stella core | Very low; in-process C++/Python, no rendering | `ram` (128 bytes), optional RGB/grayscale, score reward, lives, frame counters, five minimal actions, modes/difficulties, clone/restore state | ALE is GPL-2.0; bundled/game ROM copyright and redistribution terms still need checking for the deployment jurisdiction. No online anti-cheat/ToS issue | Free/open source package. Best fit; only missing layer is a RAM decoder |
| Stable-Retro custom Stella integration | High emulator/framework maturity; higher integration effort | Low once native core is running | Libretro/Stella state, custom `data.json` named memory variables, scenario reward/done, savestate | Same ROM copyright issue; custom integration should use a legally obtained ROM if the bundled ALE copy is not acceptable | Free/open source. Good fallback if a named-memory JSON integration is worth maintaining, but its supported-game table lists `Frogger-Genesis-v0`, not Parker Brothers Atari 2600 Frogger, so a custom integration is required ([supported games](https://stable-retro.farama.org/supported_games/), [integration guide](https://stable-retro.farama.org/integration/)) |
| Stella desktop emulator/debugger | Mature and accurate Atari 2600 emulator with debugger | Low for a human/local process, but poor for an external agent | Debugger can inspect/write RAM, breakpoints/traps, CPU/TIA state, scripts and joystick commands; normal use is GUI-centric | ROM copyright; no game-server rules | Free/open source. Useful for reverse engineering/calibration, not a production adaptor: no documented remote socket/IPC API and no documented no-display agent loop ([Stella guide](https://stella-emu.github.io/docs/index.html), [debugger](https://stella-emu.github.io/docs/debugger.html)) |
| Libretro Stella core / RetroArch | Mature emulator ecosystem and low per-frame latency | Low in-process; headless operation depends on frontend | Libretro callbacks can provide controller input and core memory; Stella core has no Frogger-specific semantic decoder | ROM copyright; frontend/core licenses vary | Free/open source, but requires writing/maintaining a C/C++ or ctypes bridge and locating the correct core/BIOS-free setup. More work than ALE ([libretro API](https://docs.libretro.com/development/libretro/), [Stella2014 core](https://github.com/libretro/stella2014)) |
| Screen capture plus input injection | Generic and widely available, but fragile | Highest: frame capture, image processing, virtual display and input IPC | Pixels only unless the adaptor adds computer vision; keyboard/joystick events | Local offline use has no online anti-cheat, but copying/redistributing the ROM still has copyright risk; input tools can be brittle | Free packages, but needs `stella`, `Xvfb`/SDL setup, capture library and `xdotool`/uinput. Last resort only; violates the symbolic-state preference |

There is no useful remote game API or account-based service for this offline cartridge. Browser/online emulators are excluded because they add latency, cannot guarantee deterministic reset, and conflict with the no-display/no-account constraint.

## Chosen ALE setup

### Software

Install into the exact evaluator venv, not the system Python:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/.venv/bin/python -m pip install ale-py==0.12.1 gymnasium==1.4.0
```

`numpy` is installed as an ALE dependency. Do not install SDL or a desktop window for the RAM configuration. The direct package also works without Gymnasium if the adaptor wants `ALEInterface` instead of `gym.make` ([installation](https://ale.farama.org/getting-started/), [Python interface](https://ale.farama.org/python-interface/)).

### Environment and reset configuration

The adaptor phase should:

1. Import `gymnasium` and `ale_py`, then register ALE environments with `gym.register_envs(ale_py)`.
2. Create `ALE/Frogger-v5` with `obs_type="ram"`, `frameskip=1`, `repeat_action_probability=0.0`, `full_action_space=False`, `mode=0`, `difficulty=0`, and no render mode. `frameskip=1` exposes every emulator frame; sticky actions are disabled so a typed direction does not unexpectedly repeat.
3. On `reset(seed)`, call Gymnasium reset and retain both the returned `info["seeds"]` and the 128-byte initial RAM. Use a fresh reset for each scenario rather than trying to press the physical Reset switch.
4. Assert the loaded ROM is the ALE `frogger` ROM (`ale_py.roms.get_rom_path("frogger")`) and let ALE’s hash check reject an accidental different cartridge. Do not silently substitute Starpath Official Frogger.

…(truncated; full text in prior/interfaces.md)

### Stella debugger

Install the Stella binary/package and launch the ROM in its debugger (`stella -debug frogger.bin`) for manual reverse engineering. The debugger documentation describes `ram`, `dump`, `print`, `step`, `frame`, `run`, traps, `joy0Up/Down/Left/Right`, `exec` scripts and autoexec scripts. It does not document a remote socket, GDB protocol, or reliable headless runtime API, so use it to validate RAM labels or make a `.sym`/`.lst`, not as the Astra/Jev transport ([Stella debugger manual](https://stella-emu.github.io/docs/debugger.html)).

### Screen/input fallback

If the RAM decoder cannot be completed, run Stella under a virtual X server (`Xvfb`) or a supported SDL dummy backend, capture frames with a local library such as `mss`/Pillow/OpenCV, and inject joystick keys with `xdotool`, `uinput`, or `evdev`. This is technically possible but has the worst latency and gives Jev pixels rather than typed object state. Use it only for decoder validation or emergency fallback; it is not the chosen adaptor.

## Environment variables and user-supplied material

- **Credentials:** none.
- **Server/account/API key:** none.
- **Required environment variables:** none when using the ROM bundled in `ale-py`.
- **Optional `ALE_ROMS_DIR`:** a directory of ALE-hash-compatible ROM files if deployment policy requires an external ROM location; the user must supply a legally usable Frogger ROM and must not put its contents or a secret in study files ([ALE ROM loading](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/docs/python-interface.md#roms)).
- **Optional display variables:** none; do not require `DISPLAY`, `SDL_VIDEODRIVER`, or a GPU for `obs_type="ram"`.

## Risks and mitigations

1. **ROM provenance:** the pip distribution says supported ROMs are bundled, but cartridge copyright remains distinct from emulator licensing. Keep the bundled copy or an externally supplied copy only under the user’s applicable license.
2. **Decoder drift:** pin `ale-py` and the ROM hash; keep raw RAM in every observation and run calibration tests after upgrades.
3. **Life semantics:** ALE’s `lives=4` at reset is the reserve field, not the manual’s total of five frogs. Preserve both raw and derived names.
4. **Timing:** default v5 sticky actions and frameskip are useful for benchmark comparability but bad for precise control. Use one-frame deterministic stepping for the harness; if a benchmark requires v5 defaults, record that choice in the episode metadata.
5. **No perfect symbolic API:** ALE exposes bytes, not “car 2” or “diving turtle.” Complete and test the decoder; use screenshots only as an offline validation aid.


## Measured model latency (at archive creation, 2026-10-31 09:59)
- System-2 (Astra), one-word answer: 7.2 s
- System-2, an answer of ~1071 output tokens: 41 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 171 ms (= 10.3 ticks), p90 200 ms, max 227 ms over 12 warm requests; first request on a new connection 192 ms — a timeout below the tail turns slow answers into failed calls
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
