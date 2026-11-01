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
