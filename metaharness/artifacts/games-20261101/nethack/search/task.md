Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `NetHack (NetHack Learning Environment)` — NetHack is a turn-based dungeon exploration game. Your objective is to survive, gain experience and gold, descend through the dungeon, and ultimately retrieve the Amulet and ascend; the primary score is NetHack's terminal score, with depth and ascension reported separately. Every step consumes one command and monsters may act during it. Observation fields include a 21x79 glyph and character map, terminal text (tty), cursor, latest message, inventory text, raw blstats and a named stats dictionary (HP, hunger, gold, depth, score and turn). Use action names from action_spec; movement is `move <n|e|s|w|ne|se|sw|nw>`, running is `run <dir>`, and prompts are answered by another command. `wait` is the no-op.

Actions:
- more: Dismiss a --More-- prompt or advance a text screen.
- move n: Move one square north (vi key k).
- move e: Move one square east (vi key l).
- move s: Move one square south (vi key j).
- move w: Move one square west (vi key h).
- move ne: Move one square northeast (vi key u).
- move se: Move one square southeast (vi key n).
- move sw: Move one square southwest (vi key b).
- move nw: Move one square northwest (vi key y).
- run n: Run north until an obstacle or event.
- run e: Run east until an obstacle or event.
- run s: Run south until an obstacle or event.
- run w: Run west until an obstacle or event.
- run ne: Run northeast until an obstacle or event.
- run se: Run southeast until an obstacle or event.
- run sw: Run southwest until an obstacle or event.
- run nw: Run northwest until an obstacle or event.
- up: Climb an upstairs when standing on one.
- down: Descend a downstairs when standing on one.
- wait: Wait one turn (the safe no-op when no decision is ready).
- kick: Kick in a direction; a following direction may be requested.
- eat: Eat an edible item; a following inventory letter may be requested.
- search: Search adjacent squares for hidden doors and traps.
(real-time games: ticks apply 'wait' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Each action consumes one NetHack turn. Directions use vi-key conventions. Actions such as kick/eat can leave a prompt in message/tty_chars; answer it with another action (MORE clears --More-- screens). Integer action indices 0..22 are accepted as an equivalent compact grammar.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40 calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of 1800 s, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals.

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Strong play explores cautiously, records the map and inventory, keeps food/healing and an escape route, and fights from favorable terrain. Exact bookkeeping for HP, hunger, encumbrance, BUC, item identities, resistances, charges, prompts and map coordinates belongs in code. Astra should plan role-specific routes, identification experiments, branch choices, combat retreats and ascension preparation from that belief state. Jev should answer fast typed questions about immediate threats, unresolved prompts, safe adjacent squares, item/monster recognition and whether a deterministic safety gate is met. Before Gehennom, require a resistance/escape ascension kit, retrieve the Amulet, and return by a safe route.
- scoring: Report the terminal NetHack score from the environment/info as the primary score because it combines gold, experience, dungeon progress and achievements; also report maximum dungeon depth and an ascension/success flag so survival and true completion are visible.
- interface.observation: A Python dict/NumPy observation: 21x79 glyphs and terminal character/color/special arrays, tty cursor, latest message, and blstats containing coordinates, HP/energy, level, strength, gold, dungeon depth, hunger, alignment and time. Convert to JSON-friendly lists/text at model boundaries.
- interface.actions: Submit one integer per turn via env.step(action), using nle.nethack.ACTIONS for vi movement, wait, inventory, item use, stairs and extended-command keys; inspect message/tty output and submit follow-up keys for menus or prompts. reset(seed=...) starts a seeded episode.

### study/strategies.md
# Strong-play knowledge and where to implement it

NetHack has enormous item, monster, dungeon and event variation. The [NetHack Guidebook](https://nethack.org/v366/Guidebook.html),
[NetHack Wiki](https://nethackwiki.com/wiki/Main_Page), and [NetHack strategy pages](https://nethackwiki.com/wiki/Strategy)
are the main references. Treat the following as priors, with Astra allowed to override them from
the current map and messages.

## Opening and early dungeon

* **Explore conservatively.** Map rooms and corridors, search suspicious squares, pick up useful
  food/armor/weapons, and identify safe escape routes before fighting. Keep hunger under control
  and do not spend turns wandering when HP is low. Encode movement legality, map memory, hunger
  and HP thresholds in **code**; ask Astra to choose routes and risk tradeoffs.
* **Use the starting inventory and role.** Read the role/race/alignment in the initial status and
  equip the best weapon/armor only after checking encumbrance and cursed status. Role skills and
  starting pets strongly affect the opening. **Code** can track equipment and cursed/uncursed
  knowledge; **Astra** plans role-specific priorities.
* **Level and resource discipline.** Gain a few safe experience levels, keep food and healing,
  and avoid unnecessary melee with unknown monsters. The wiki's [Beginning players' guide](https://nethackwiki.com/wiki/Beginner%27s_guide)
  and [conduct](https://nethackwiki.com/wiki/Conduct) explain common early choices. Put exact
  inventory counts and HP/energy alarms in **code**; Jev can answer “is retreat/healing required?”

## Combat and survival

* **Exploit terrain and distance.** Fight from corridors/doorways, use ranged attacks and wands,
  close doors, and avoid being surrounded. Retreat when HP, armor class, or escape resources are
  poor. Monsters have resistances and special attacks; consult the [monster database](https://nethackwiki.com/wiki/Monster)
  and [resistance guide](https://nethackwiki.com/wiki/Resistance). **Code** should maintain a
  visible-monster threat table and legal escape actions; **Jev** should rapidly classify the
  immediate threat and best one-turn response; **Astra** handles multi-turn positioning.
* **Identify before trusting.** Test or identify unknown potions, scrolls, rings and wands safely;
  never quaff/read unknown dangerous items while polymorphed, blind, confused, or in combat.
  Maintain object identity, BUC (blessed/uncursed/cursed), charges and known effects in **code**;
  have **Astra** plan identification experiments. Jev can answer typed recognition questions from
  an item description and known history. See [identification](https://nethackwiki.com/wiki/Identification)
  and [BUC](https://nethackwiki.com/wiki/BUC).
* **Escape beats damage.** Keep teleportation, digging, emergency healing, cancellation and a
  safe staircase available. A guaranteed escape is usually better than one more attack. Encode
  hard rules such as “do not enter lethal adjacent combat” and inventory availability in **code**;
  Astra selects among escapes when several are possible.

## Mid/late game and ascension

* **Prepare resistances and key tools.** Seek fire/cold/sleep/shock/poison/disintegration
  resistance, reflection, magic resistance, intrinsic speed and reliable AC. Keep a blessed
  weapon, food, healing and a way to dig or teleport. The [ascension kit](https://nethackwiki.com/wiki/Ascension_kit)
  and [resistances](https://nethackwiki.com/wiki/Resistance) pages give checklists. **Code** tracks
  achieved intrinsics, equipment slots and missing kit items; **Astra** prioritizes routes and
  shops.
* **Use shops and Sokoban intelligently.** Price-identify objects, buy critical food/scrolls and
  avoid shop theft unless the plan supports it. Sokoban can provide an early amulet of reflection
  or bag of holding but has boulder traps and alignment choices ([Sokoban](https://nethackwiki.com/wiki/Sokoban)).
  **Astra** decides whether optional branches are worth risk; **code** handles gold and carrying
  capacity.
* **Endgame discipline.** Before entering Gehennom, verify resistances, escape items, blessed
  weapon, charging/holy water and inventory slots. Locate the vibrating square, retrieve the
  Amulet, then return by a known safe route; avoid fighting unless needed. Use **Astra prompts**
  for route planning and endgame contingencies, with **code** enforcing readiness gates.

## Common failure modes and counters

* Walking into a monster while confused/blind, forgetting a prompt, or repeatedly attacking a
  resistant monster wastes turns. **Jev questions:** “Is there an unresolved prompt?”, “Which
  adjacent square is safest?”, “Does this monster resist the proposed attack?”
* Eating corpses can grant intrinsics but can also poison, induce illness, polymorph, or violate
  conduct. Track corpse age/type and conduct in **code**; let **Astra** decide when the benefit is
  worth the risk ([corpse](https://nethackwiki.com/wiki/Corpse)).
* Cursed items, unblessed water, inventory overload and hunger silently kill runs. Make BUC,
  encumbrance, hunger and HP checks deterministic **code** gates before risky actions.
* The game is stochastic and partially observed. Astra should maintain a concise belief state
  (map, monster locations, item identities, resistances, escape routes), update it after every
  message/map change, and ask Jev many independent typed questions rather than asking Jev to
  plan a route. This division follows the environment's per-turn observations and preserves
  exact bookkeeping in the harness.

## NLE-specific policy choices

Use `NetHackScore` for the primary objective and retain maximum depth as a secondary metric.
For curriculum/debugging, `NetHackStaircase` rewards reaching a deeper staircase and
`NetHackChallenge` exposes a harder fixed task; compare reward definitions in the [NLE README](https://github.com/facebookresearch/nle#environments).
Seeded resets should vary role/race/alignment and dungeon RNG while holding the same observation
contract. Do not memorize a single map: held-out seeds test exploration, item identification and
safe recovery.

### study/brief.md
# NetHack (NetHack Learning Environment)

## What the game is

NetHack is a single-player, turn-based dungeon crawler. The canonical goal is to descend to
the bottom of the dungeon, obtain the Amulet of Yendor, and return to the surface to ascend
and offer it (the “ascension”). A run ends in death, quitting, or successful ascension. The
official guide describes the objective and game commands in detail: [NetHack Guide, Playing the
Game](https://nethack.org/v366/Guidebook.html#_playing_the_game), [The Dungeons of Doom](https://nethack.org/v366/Guidebook.html#_the_dungeons_of_doom).

NLE is Facebook Research's open-source wrapper around the NetHack 3.6.6 engine. It exposes
NetHack as a Gym environment for reinforcement-learning agents, with deterministic seeding
available through Gym's reset API. Project documentation and source are at the [NLE
repository](https://github.com/facebookresearch/nle) and [NLE API reference](https://github.com/facebookresearch/nle/tree/main/nle).

## State and progress

The player's persistent objectives are survival, useful equipment and experience, exploration,
and eventually ascension. Practical progress signals are dungeon depth (the `dungeon depth`
field in `blstats`), player level/HP, inventory and identified objects, and game score. The
engine's score is a cumulative integer shown by `#score`; it rewards gold, experience, dungeon
progress and major achievements (see the [Guidebook scoring section](https://nethack.org/v366/Guidebook.html#_scoring)), and is the natural scalar to report. For learning benchmarks,
NLE also supplies task-specific environments (`NetHackScore`, `NetHackStaircase`, `NetHackGold`,
etc.) whose rewards are documented in [NLE environments](https://github.com/facebookresearch/nle#environments).

Death gives a final score and ends the episode. Reaching a staircase objective or the terminal
ascension gives a useful sparse success signal; report both final score and maximum depth so a
good surviving run is distinguishable from an early lucky score.

## Controls and actions

NetHack is keyboard-command driven. Movement uses `h j k l y u b n` (vi directions), `.` waits
one turn, `s` searches, `o` opens, `c` closes, `e` eats, `q` quaffs, `r` reads, `w` wields,
`W` wears armour, `P` puts on rings/amulets, `d` drops, `,` picks up, `i` inventories, `x`
examines, `>` descends stairs, `<` ascends stairs, `:` looks at a square, `Ctrl-P` recalls
messages, and `#` selects extended commands. Exact key semantics are listed in the
[NetHack command reference](https://nethack.org/v366/Guidebook.html#_commands).

NLE converts these keys to integer actions in `nle.nethack.ACTIONS`; the adaptor should send one
integer per environment step. Some commands open a yes/no, menu, or text prompt. The next
observation contains the prompt/message, and the adaptor must send the corresponding key(s) as
additional turns.

## Pace and tier

This is tier 1. NetHack advances only when an input command is accepted; monsters take their
turns during that same step. There is no wall-clock deadline between steps, so Astra can spend
seconds or a minute planning while the emulator is paused. A decision is strategically urgent
only in the game-time sense (for example, a monster adjacent to the player); waiting in the
harness does not consume turns. Set `decision_latency_s` to about 60 seconds as an engineering
budget, with zero real-time penalty, and keep action latency low once a command is chosen.

## Sources

* [NetHack Guidebook (3.6.6)](https://nethack.org/v366/Guidebook.html)
* [NetHack 3.6.6 source and documentation](https://github.com/NetHack/NetHack)
* [NLE README and environments](https://github.com/facebookresearch/nle)
* [NLE Python API source](https://github.com/facebookresearch/nle/tree/main/nle)

### adaptor/ADAPTOR.md
# NetHack adaptor

This adaptor runs the MIT-licensed NetHack Learning Environment (NLE) in process
through `nle.env.tasks.NetHackScore`. It starts no server, display, account, or
network connection. Install `nle==1.3.0` into the evaluator's Python environment
using `requirements.txt`.

The adaptor is Tier 1: `step(action)` submits one NLE action and waits for the
resulting turn. The game is paused between calls, so model deliberation does not
consume wall-clock turns. `action_spec` exposes NLE's 23 task actions as readable
strings (`move n`, `run se`, `up`, `eat`, `search`, and so on); `wait` is the
no-op. Integer indices 0 through 22 are also accepted. Prompt and menu text is
left in `message` and `tty` so the harness can answer it on a later turn.

Observations are strict JSON. They contain glyph, character, and colour arrays,
the textual map and terminal screen, cursor, message, inventory lines, raw
`blstats`, named player statistics, score, maximum depth, terminal flags, and the
last turn reward. The primary `score` is the terminal NetHack score from blstats.
`score_breakdown()` additionally reports maximum dungeon depth and an ascension
flag; `info()` reports engine status and marks a trial invalid after an engine
failure. No screenshots or runtime files are needed.

`reset(seed)` derives NetHack core, display, and level-generator RNG seeds from
the supplied integer and disables reseeding, making fresh instances reproducible
while still varying roles, races, alignments, dungeon layouts, monsters, items,
and branch choices across the declared train/dev/validation/held-out seeds.
The groups are disjoint and have the same broad mix. Seeds divisible by four run
the NLE staircase curriculum and succeed on reaching a downstairs; the others
are ordinary score episodes. Reaching a staircase is visible in the map and
depth fields, and `scenario_success` identifies staircase completion or
ascension. The final score remains the reported objective for every episode.

The bundled engine has no external credentials. `env.example` is included only
to document that fact and an optional caller-side default seed.

### study/interfaces.md
# Programmatic interfaces

## 1. Official NetHack Learning Environment (chosen)

**Maturity and exposure.** NLE is the maintained research wrapper used by the NetHack RL
community (MIT-licensed code; the bundled NetHack engine has its own permissive NetHack license).
It launches a headless NetHack process in-process and implements the Gym/Gymnasium `reset`,
`step`, `action_space`, and `observation_space` contract. The standard observation is a Python
dict containing a 21x79 glyph map (`glyphs`), terminal character/color arrays (`chars`, `colors`,
`special`, `tty_chars`, `tty_cursor`), the latest message, and numeric player statistics
(`blstats`, including x/y, HP, max HP, strength, experience, gold, dungeon depth, hunger,
alignment and time). This is richer and less error-prone than OCR. See the [NLE README](https://github.com/facebookresearch/nle)
and [observation definitions](https://github.com/facebookresearch/nle/blob/main/nle/nethack/nethack.py).

**Control and latency.** `env.step(action_integer)` performs exactly one NetHack command and
returns `(observation, reward, terminated, truncated, info)` (older releases return the four-item
Gym tuple). Actions are the integer key codes in `nle.nethack.ACTIONS`; use `nle.nethack.actions`
or the enum rather than hard-coding ASCII values. `env.reset(seed=seed)` starts a reproducible
episode when the engine/build supports seeded RNG. Headless local steps normally take
milliseconds, while model deliberation can happen between steps without game-time advancing.

**Install on the evaluator.** Use the evaluator interpreter, never system Python:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install --upgrade pip
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install nle
```

If no cp312 wheel exists, install build tools (`cmake`, `bison`, `flex`, a C/C++ compiler and
Python development headers) with the host package manager, then install from source:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install --no-binary nle nle
```

NLE's build instructions and supported versions are in [INSTALL.md](https://github.com/facebookresearch/nle/blob/main/INSTALL.md).
Verify with:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -c \
  'import nle; from nle.env import NetHackScore; e=NetHackScore(); o,i=e.reset(seed=0); print(e.observation_space, e.action_space, o.keys()); e.close()'
```

No account, server, API key, GUI, or network credential is required. Environment variables are
empty for the normal local install. Set `NLE_SEED` only in an adaptor if a harness-level seed
configuration is desired; pass the actual seed to `reset(seed=...)`.

**Legal/cost/risk.** Free and offline after installation. It runs the bundled open-source engine,
so online-server terms and anti-cheat do not apply. Pin the NLE version and engine seed because
minor releases can alter observation details. Python 3.12 wheels may be unavailable; a source
build is the main setup risk.

## 2. Direct NetHack engine / tty bot

Building NetHack from the [upstream source](https://github.com/NetHack/NetHack) and driving its
tty protocol gives complete textual screens and keystrokes. Mature for human-facing bots, but
requires parsing menus, prompts, cursor position and terminal control sequences. It is slower and
more brittle than NLE, and exposes less structured state (no direct glyph/blstats dict). Offline
and free; no credentials. Use only if NLE cannot compile.

## 3. Existing bot frameworks

[TAEB](https://github.com/ekg/taeb) and [Nethack4j](https://github.com/NullDev/Nethack4j)
demonstrate autonomous play and parsers, but are old/community projects rather than a stable
Python 3.12 interface. They can supply heuristics or a tty parser, not a drop-in evaluator API.
Their use against public servers may violate server rules; local bundled NetHack avoids that risk.

## 4. Screen capture and input injection

`pyte`/`curses` plus `xdotool` or `python-xlib` can capture a terminal and inject keys. This is a
last-resort Linux GUI/TTY method: it needs a display, has OCR/cursor synchronization failures,
adds latency, and violates the headless constraint. Free software but no advantage over NLE.

## 5. Remote public NetHack server

Telnet/SSH to a public server provides authentic games, but needs an account/server address,
network access and a terminal bot. Latency, disconnects, server policy and anti-bot rules make it
unsuitable for this offline evaluator. Never put credentials in study files.

## Adaptor contract for NLE

1. Construct `nle.env.NetHackScore` (or `NetHackChallenge` if challenge scoring is explicitly
   wanted) with a fixed `observation_keys` set that includes `glyphs`, `chars`, `colors`,
   `blstats`, `message`, `tty_chars`, and `tty_cursor`.
2. Call `reset(seed=seed)` and serialize observations to JSON-friendly lists/base64 only at the
   Astra/Jev boundary; retain NumPy arrays internally.
3. Render `glyphs` using NLE's glyph helpers or use `tty_chars` for a 21x79 textual map. Decode
   `message` and inspect `blstats` every step. Track `info`, `terminated`, and `truncated`.
4. Map a planned command to the matching `ACTIONS` member and call `step`. If a message reports
   a prompt/menu, expose it and continue sending menu-selection keys until the prompt clears.
5. Report final `info`/`blstats` score and maximum dungeon depth; close the environment on exit.

The [NLE example agents](https://github.com/facebookresearch/nle/tree/main/examples) show action
selection and observation handling.


## Measured model latency (at archive creation, 2026-10-30 07:26)
- System-2 (Astra), one-word answer: 65 s
- System-2, an answer of ~938 output tokens: 100 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).