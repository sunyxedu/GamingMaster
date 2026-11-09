Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Pokémon Showdown — Gen 9 Random Battle` — Play Pokémon Showdown Gen 9 Random Battle singles as p1 (Agent) against a seeded
heuristic opponent (or random if configured). Both teams have six randomly generated Pokémon,
with format-assigned levels, moves, items, abilities and Tera types. There is no team preview.
Win by fainting all six opponents. Pokémon types, stats, speed, priority, move accuracy/PP,
abilities, items, status, boosts, hazards, weather and terrain follow the real Gen 9 engine.
Both sides choose before a turn resolves; forced replacements and pivots create extra decisions.
The world waits while you think. You may Terastallize once per battle, when the request allows.

Send exactly 'move 1' .. 'move 4', 'move 1 terastallize' .. 'move 4 terastallize', or
'switch 1' .. 'switch 6'. Only strings in legal_actions are currently available.
Move indices refer to request.active[0].moves; switch indices refer to the CURRENT
request.side.pokemon array, whose order changes on switches. A forced switch requires a switch;
Revival Blessing instead selects a fainted teammate. 'noop' waits without advancing.
Malformed/unavailable choices report last_error without consuming a decision. A hidden trapping
effect can reject an apparently available switch; the engine then updates the request.

request is your private Showdown request: exact own HP/maxHP, status, stats, items, abilities,
moves/PP, Tera options, disabled moves and known trapping. battle_log contains the most recent
700 p1-visible protocol lines (also capped at 65k characters); recent_events contains events
since the last submitted choice (up to 300 lines/30k characters). Protocol is authoritative.
Public switches reveal species/level; opponent HP is rounded by Showdown. opponent_revealed,
active and public_summary cache common public events; summaries are conveniences, not a complete
mechanics interpreter (Illusion, Transform and unusual boost effects require reading the log).
move_dex supplies public move descriptions and base data for your moves; species_dex contains
public types/base stats for your team and revealed species. Hidden opponent sets and exact HP
are never exposed.
No scenario seed or referee score is included in observations. result and termination_reason
are populated at termination. The configured turn/decision limits adjudicate a tie.

Evaluator score is outcome (+1 win, 0 tie/unfinished, -1 loss) + 0.25 * HP margin.
Each team's HP fraction is the sum of its six individual remaining-HP/max-HP fractions divided
by six; HP margin is yours minus theirs. Winning is primary; preserving HP breaks ties in quality.
Exact score components, survivors, turns and termination reason are evaluator-only.


Actions:
- noop: Wait without advancing the battle.
- move 1: Choose this active move slot.
- move 2: Choose this active move slot.
- move 3: Choose this active move slot.
- move 4: Choose this active move slot.
- move 1 terastallize: Choose this move slot and spend the team's one Terastallization.
- move 2 terastallize: Choose this move slot and spend the team's one Terastallization.
- move 3 terastallize: Choose this move slot and spend the team's one Terastallization.
- move 4 terastallize: Choose this move slot and spend the team's one Terastallization.
- switch 1: Choose this current request.side.pokemon slot (switch or revive).
- switch 2: Choose this current request.side.pokemon slot (switch or revive).
- switch 3: Choose this current request.side.pokemon slot (switch or revive).
- switch 4: Choose this current request.side.pokemon slot (switch or revive).
- switch 5: Choose this current request.side.pokemon slot (switch or revive).
- switch 6: Choose this current request.side.pokemon slot (switch or revive).
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Only observation.legal_actions are currently available; noop always waits. Slots are 1-based and team order changes on switches. Illegal actions report an error without advancing.

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
- strategy_summary: Strong play begins by making safe progress while revealing the opponent's set, since random battles have no team preview. Track roles, exact HP/status/PP, hazards, speed and priority, and preserve the one Pokémon that checks a revealed threat. Use hazards, status, pivots and setup when they create a concrete endgame; switch according to type, ability, entry damage and likely coverage. Treat the assigned one-use Tera as an emergency or win-condition resource and update matchups after spending it. Astra should plan and adapt under uncertainty, Jev should answer typed threat/choice questions, and code should enforce protocol legality, mechanics, bookkeeping and bounded damage checks.
- scoring: Report win/loss/tie (outcome +1/0/-1), terminal HP fraction for each six-slot team, hp_margin = self_fraction - opponent_fraction, surviving counts, turns, and termination reason. If one scalar is required use outcome + 0.25*hp_margin so winning remains primary while remaining HP distinguishes wins; retain components for the evaluator's specified formula.
- interface.observation: Newline-and-pipe protocol text. The p1 stream contains public battle events plus sideupdate |request| JSON with active legal moves, PP/disabled flags, own six-Pokemon exact HP/status and revealed opponent information. Unrevealed opponent set details are absent. The referee may separately read omniscient state for scoring.
- interface.actions: Write one legal command to the p1 stream: move <1-based slot>, move <slot> terastallize, or switch <1-based team slot>; parse errors and the replacement request before retrying. Start/reset and player setup are sent as >start and >player commands to the parent stream.

### study/strategies.md
# Strong-play prior for Pokémon Showdown `gen9randombattle`

Random battles give each side six engine-generated Pokémon with curated, legal sets, variable levels, items, abilities and predetermined Tera types. There is no normal team preview or lead choice: both leads appear immediately. The opponent's species, moves, item, ability and Tera type become known only when revealed by protocol events. The format and generator are defined in [formats.ts](https://github.com/smogon/pokemon-showdown/blob/master/config/formats.ts) and [gen9 random teams](https://github.com/smogon/pokemon-showdown/blob/master/data/random-battles/gen9/teams.ts); the simulator's protocol and mechanics are authoritative ([SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md), [battle-actions.ts](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-actions.ts)). Smogon’s [SV Random Battle dex](https://www.smogon.com/dex/sv/formats/random-battle/) and [strategy dex](https://www.smogon.com/dex/sv/) are useful public references for species roles and standard move effects.

## Decision policy

| Situation / heuristic | Put in code | Put in Astra prompts | Put in Jev questions |
|---|---|---|---|
| Parse `|request|` JSON, maintain six-slot HP/status/PP, revealed moves, hazards, boosts, weather, terrain, screens and the current active slot | Exact bookkeeping and validation. Reject illegal moves and handle chained forced switches. | Explain only the resulting compact state. | Ask “is a forced switch required?” or “which legal slots are disabled?” when a fast answer is useful. |
| Type, ability, item, priority, speed and status legality; deterministic damage/type calculations using the pinned Showdown data | Exact rules, move-target legality, speed/priority ordering, and a bounded damage range. Cache data and avoid guessing from species names. | Resolve uncertain ranges and identify the opponent’s likely set from revealed evidence. | Typed threat recognition: “which move is safest?”, “does this KO range look favorable?”, “is this Pokémon immune/resistant?” |
| First turn with no preview | Enumerate legal attacks and switches; score immediate KO, safe damage, information and hazards. | Pick a provisional win condition and decide whether to scout or take immediate progress. | Choose among attack / switch / scout labels, with a short reason code. |
| Switching and momentum | Filter switches by type/ability immunity, hazards, trapping and expected damage; account for entry hazards and whether a pivot move creates a second request. | Predict the opponent’s likely switch from matchup and revealed tendencies; preserve a counter for the opponent’s strongest known threat. | “Do I switch now?”, “which of these two resists is safer?”, “is a pivot move preferable?” |
| Hazards, removal, screens and status | Track layers, groundedness, removal PP and residual damage; compute whether a layer changes a KO or switch threshold. | Choose when to set hazards versus attack, and preserve the sole remover or status absorber for the endgame. | Recognize “set hazard now” / “remove hazard now” / “status is safe” situations. |
| Setup and wallbreaking | Detect boosts, Unaware/phazing/Taunt/Encore and revenge-kill ranges; do not blindly repeat a setup move. | Plan a sequence that forces progress, identifies counters, and times a sweep after faster checks are removed. | Yes/no: “is setup safe this turn?”, “is the active foe a hard stop?” |
| Speed control, priority and Trick Room | Calculate effective speed, priority bracket, Choice Scarf/Unburden/weather/terrain effects where known; always include priority in KO checks. | Plan the finite turns of Tailwind/Trick Room and avoid spending them while the opponent can stall them. | “Who moves first?”, “is priority a winning line?”, “should Trick Room be refreshed?” |
| Terastallization | Validate the one-use resource, assigned Tera type and legal move; simulate defensive weaknesses/resistances and STAB. | Treat Tera as a win-condition or emergency resource: use it to secure a sweep/KO or survive a decisive hit, while considering the revealed opposing counterplay. | Typed yes/no “Tera now?”, “does Tera change the matchup?”, and choose the Tera-enabled move. |
| Endgame and information | Count remaining Pokémon, PP, hazards and guaranteed KOs; prefer lines that preserve a check and avoid unnecessary switching damage. | Re-plan after every reveal; trade a low-value Pokémon for the opponent’s win condition when that converts to a clean finish. | Fast terminal choice: finish / recover / protect / switch, and whether a revealed move is worth scouting. |

Astra should receive a compact state plus a legal-action list and be asked for one action, a short line of reasoning, the preserved counter/win condition, and uncertainty. Jev should receive typed questions with a small candidate set; it is useful for recognition and a final choice but should not be asked to invent a long plan or perform exact arithmetic.

## What strong random-battle play looks like

1. **Create information while making progress.** On turn one, compare an immediate KO or high-value status with a safe switch or pivot. Because there is no preview, reveal the opponent's species, ability and moves with low-risk attacks, Protect, U-turn/Volt Switch or a resistant switch. Do not reveal a crucial set unnecessarily when a guaranteed KO is available.
2. **Preserve role compression.** Random teams often have one practical answer to a threat: a Ground immunity, hazard remover, priority user, status absorber, revenge killer or setup answer. Keep that Pokémon healthy enough to perform its role; a high-HP but irrelevant wall can be less valuable than a low-HP priority attacker.
3. **Use hazards and residual damage deliberately.** Stealth Rock and Spikes punish the frequent switches caused by unknown sets; Toxic, burn, Leech Seed, weather and terrain can turn a neutral matchup into a timer. Set layers when the opponent cannot immediately punish them, and remove hazards when they cross a KO or switching threshold. Do not sacrifice the only remover merely to add one more layer.
4. **Respect revealed speed and priority.** Track every revealed move and speed clue. Choice Scarf, weather abilities, Tailwind, Trick Room, paralysis, priority and Unburden can reverse the apparent matchup. A slower Pokémon that survives one hit may still lose to a priority move; an apparently faster Pokémon may be choice-locked and exploitable.
5. **Treat Tera as a finite tactical resource.** The assigned Tera type is fixed by the generated set. Defensive Tera can erase a weakness and win a turn; offensive Tera can increase STAB or remove a check. Use it when it changes a decisive KO/survival or enables the planned endgame, not simply for a small damage increase. Record that it has been spent and update type matchups afterward.
6. **Convert advantage into an endgame.** Once a counter is eliminated, stop overpredicting: take the guaranteed damage or setup line, preserve PP and avoid needless switches. When behind, make the highest-upside line (double switch, status, hazard or Tera) only when the alternative is a forced loss; when ahead, choose robust lines that cover the opponent's known counterplay.
7. **Infer sets conservatively.** A species may have several random sets and levels are balanced by the generator. Infer only what a revealed move, item, ability, damage range or speed result supports. Never assume an OU spread, a hidden move, or that an unrevealed Pokémon lacks priority, recovery or coverage.

## Frequent mistakes and counters

* **Blind setup or greedy prediction:** setup into Taunt, Encore, Unaware, phazing, a super-effective revenge hit or a faster priority user loses immediately. Code should flag these checks; Astra decides whether the upside justifies risk.
* **Premature Tera:** spending Tera on chip damage can leave no answer to the opponent's revealed sweeper. Jev can flag the resource and Astra can reserve it for the actual win condition.
* **Ignoring entry damage:** repeatedly switching through Stealth Rock/Spikes, or forgetting that a grounded Pokémon can absorb Toxic Spikes, turns a nominally favorable matchup into a loss. Bookkeeping belongs in code.
* **Assuming standard cartridge/OU rules:** Random Battle has its own generated levels, clauses, Sleep Clause behavior, obtainable pool and format-specific sets. Use the pinned engine and request flags rather than hard-coded assumptions.
* **Forgetting action ordering:** ordinary switches, priority, effective Speed and random ties determine whether a line works. Use the engine's legal request and speed data; do not order actions by move power.
* **Overvaluing HP or damage dealt:** a healthy but trapped wall can be worthless, while eliminating the opponent's only hazard remover or priority user wins later. Astra should evaluate role and endgame, while the score still records HP margin.
* **Trying to force an illegal action:** disabled/zero-PP moves, trapped switches, fainted slots and a second sleep target are rejected by the simulator. Validate locally and handle a new request after `|error|`.

## Implementation priorities

1. Build a faithful protocol/state ledger and a deterministic legal-action filter before adding a strategic search. Include chained requests, exact own HP, rounded opponent HP, status, boosts, hazards, weather/terrain, screens, PP and Tera state.
2. Add a compact damage/type/speed evaluator backed by the pinned Showdown data. Use ranges and uncertainty rather than pretending the hidden set is known.
3. Prompt Astra for a short plan and robust action with the ledger and candidate lines. Ask Jev independent typed questions for immediate threat recognition and tie-breaks, then let code validate the selected command.
4. Evaluate across held-out seeds against the supplied heuristic/random opponent. Record win/loss/tie, exact terminal HP fractions and HP margin; inspect losses for missed information, bad Tera timing, hazard debt and speed/priority errors.

## Sources

* Showdown format and random generation: https://github.com/smogon/pokemon-showdown/blob/master/config/formats.ts and https://github.com/smogon/pokemon-showdown/tree/master/data/random-battles/gen9
* Simulator protocol and choices: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#possible-choices
* Simulator API and stream examples: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md and https://github.com/smogon/pokemon-showdown/blob/master/sim/examples/battle-stream-example.ts
* Mechanics and action order: https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-actions.ts and https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-queue.ts
* Random Battle format discussion and sets (community references): https://www.smogon.com/dex/sv/formats/random-battle/ and https://www.smogon.com/forums/

### study/brief.md
# Pokémon Showdown — Gen 9 Random Battle

Studied 2026-10-29 for a headless Linux, offline, turn-by-turn evaluator. The exact target is `gen9randombattle`, singles, in `/home/hemlix/gm-tools/pokemon-showdown`, version `0.11.11`, checkout commit `a5df8274e85b0889bf2a9b3422a08b39732374fc`. This is not OU, Battle Factory, doubles, Blitz, or a live ladder task.

## Evidence and scope

The checkout's official documentation, compiled exports, format rules, simulator, and community-maintained random-set generator were inspected locally. GitHub, Smogon, and poke-env documentation fetches failed DNS resolution; this study does **not** claim to have read current online discussions. Source links below identify the inspected files at the pinned commit. Strategy recommendations are synthesis from those rules and sets, with unverified external reading clearly separated in the other files. No game-playing code was written, no battles were played to completion, and no packages were installed.

## Rules and objective

Each player receives six generated Pokémon with competitively curated moves, abilities, items, levels, and predetermined Tera types. Each normally has up to four moves. Levels vary to balance strength; do not assume level 100, standard OU EVs, or cartridge defaults. Own six sets are visible immediately. There is **no team preview or lead selection**: both leads enter automatically, and the opponent's remaining Pokémon stay hidden until revealed. Generation uses structured roles and compatibility rules, not independent uniform move sampling. [B1, B2]

On a normal turn, both sides secretly choose a move or a voluntary switch. Once all required choices arrive, the simulator resolves actions. Ordinary switching precedes ordinary attacks; move priority precedes effective Speed, and equal-Speed ties involve randomness. Higher priority does not mean higher base power. Trick Room reverses Speed ordering within the relevant priority bracket. Moves consume PP; attacks use accuracy, damage rolls, type matchups, abilities, items, and other effects. Status, boosts, weather, terrain, hazards, recoil, recovery, and delayed effects matter. The engine is the authority for exceptions. [B3, B4]

A Pokémon at zero HP faints. Normally the battle is won when all six opposing Pokémon faint; simultaneous outcomes and certain unusual mechanics can produce a tie. Faint replacement and pivot moves such as U-turn can request an additional switch **within the same numbered turn**. A forced replacement does not also grant a fresh attack for that same request. Revival Blessing can instead request selection of a fainted teammate. Handle request state rather than assuming one model call per turn number. [B3, B5]

Once per team per battle, an eligible active Pokémon can Terastallize while selecting its move. Its assigned Tera type cannot be freely chosen at decision time. For ordinary non-Stellar Tera, defensive typing becomes the Tera type; original offensive STAB remains, a new Tera type gains STAB, and matching an original type normally increases that type's STAB to 2×. Tera persists through switching. Stellar, Ogerpon, and Terapagos have special behavior. Gen 9 Random Battle does not normally offer Mega Evolution, Z-Moves, or Dynamax. Use the request's capability fields. [B4, B5]

This pinned format explicitly has Species Clause, HP Percentage Mod, Cancel Mod, Sleep Clause Mod, Illusion Level Mod, Obtainable, and Pokémon of the Day support. **Sleep Clause allows sleep moves but prevents putting a second living foe to sleep while another is asleep due to your side**; self-induced Rest does not use that allowance. Do not import OU's Sleep Moves Clause or its banlist. Illusion disguises level as well as identity. Opponent HP is displayed as rounded percentages, while own HP is exact. [B1, B6]

## Controls and pacing

The selected interface is the official Node `BattleStream`, using `getPlayerStreams` to provide each player its permitted messages. A small future Node bridge will speak JSON lines to the evaluator's Python 3.12.3. There is no browser, account, network service, or animation delay. See `interfaces.md` for the complete implementation specification.

Examples of simulator choices: `move 1`, `move 2 terastallize`, and `switch 4`. Slots are **1-based in the latest request**, and the team array can reorder after switches. `wait: true` means submit nothing. `default` is an engine-selected legal fallback, not a strategic action. Cancel is only available before commitment and is unnecessary for the proposed adaptor. [B3, B5]

**Tier 1 is correct.** The standalone simulator does not advance battle mechanics with wall-clock time: it waits for required decisions. Astra can take seconds or minutes without an in-game penalty. Game state changes on engine events caused by submitted decisions; the opponent may act while the learner waits, but cannot advance past a required learner choice. There is no intrinsic finite decision deadline. `study.json` uses **`decision_latency_s: -1.0` as an explicit sentinel meaning unbounded/no game timer**, because JSON cannot represent infinity. An evaluator runtime watchdog is separate and must not silently become a game rule.

The network server has a separate optional timer. In this checkout its base bank/per-turn settings are 150 seconds for ladder battles and 300 for challenges, plus additional timer rules; these are **not guaranteed thinking time per turn**, and do not apply to the selected direct simulator. Avoid Blitz and do not start a server timer. The simulator also has a roughly 1,000-turn automatic tie limit; that is a game-turn limit, not elapsed seconds. [B5, B7]

## Score and progress

Report terminal win/loss/tie and exact remaining-HP margin. Suggested reporting fields are `win`, `outcome` (+1 win, 0 tie, -1 loss), `hp_fraction_self`, `hp_fraction_opponent`, `hp_margin`, `pokemon_remaining`, `turns`, and termination reason. For side s, define `H_s = sum(current_hp_i / current_max_hp_i) / 6`, counting fainted Pokémon as zero. Set `hp_margin = H_self - H_opponent`, in [-1, 1]. This values every team slot equally rather than rewarding naturally high-HP species. Use current engine max HP for form changes and read exact terminal HP in the referee only.

The user specifies win and HP margin but no scalar weighting. If one scalar is required, this study proposes `score = outcome + 0.25 * hp_margin`: a win always outranks a draw, which always outranks a loss. Keep the components so a supplied evaluator formula can replace it. The simulator's `end.score` is **surviving Pokémon counts, not HP margin**. Do not use Elo or damage dealt as the task score. Do not reward repeatable healing/damage cycles or confuse a process failure with a loss. A resource cap must report `truncated`, not invent a victory. [B5]

Intermediate progress is elimination, favorable HP trades, useful status/hazards, preserved counters, information gained, and a plausible endgame. HP alone can be misleading: a low-HP priority attacker may be more valuable than a healthy but irrelevant wall. Optimize winning first; use margin to distinguish otherwise similar winning lines.

## Sources

- [B1] Format: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/config/formats.ts
- [B2] Random generation and current sets: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/data/random-battles/gen9/teams.ts and https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/data/random-battles/gen9/sets.json
- [B3] Choice and event protocol: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/SIM-PROTOCOL.md
- [B4] Action resolution and Tera: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/battle-actions.ts and https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/battle-queue.ts
- [B5] Battle lifecycle/scoring and requests: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/battle.ts and https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/side.ts
- [B6] Clauses and visibility: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/data/rulesets.ts and https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/pokemon.ts
- [B7] Server-only timer: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/server/room-battle.ts

### adaptor/ADAPTOR.md
# Pokémon Showdown Gen 9 random battles

This Tier 1 adaptor runs the real, free, MIT-licensed Pokémon Showdown simulator
entirely offline on Linux. No server, GUI, account, credentials, downloads, or
network access are needed. `game.py` implements `make_game(**kwargs)` / TurnGame.

Run `bash setup.sh` from this directory. It checks the supplied built checkout at
`~/gm-tools/pokemon-showdown`, Node >=22.18, and the evaluator's existing Python
environment. There are no additional Python dependencies to install. Optional
`SHOWDOWN_ROOT` and `SHOWDOWN_NODE` overrides are documented in `env.example`.
The required Showdown git revision is
`a5df8274e85b0889bf2a9b3422a08b39732374fc`; the bridge rejects another revision.
Keep its source, compiled `dist`, and installed dependencies unchanged: the git
revision check does not detect local modifications or stale compiled output.
Showdown's license remains in its checkout; no engine code or data is copied here.

## Connection and lifecycle

Each reset closes the previous subprocess and starts one Node `bridge.cjs` process.
Python exchanges newline-delimited JSON over pipes. The bridge creates a first-party
`BattleStream`, starts `gen9randombattle`, and supplies both players without teams,
so Showdown generates both six-Pokémon teams. `getPlayerStreams` routes protocol
messages. Only the p1 stream feeds observations; the opponent uses only p2's stream.
Unused spectator/omniscient streams are drained and discarded. Engine internals
are accessed solely by the referee to calculate evaluator-only score components.
The bridge submits the opponent's choice before returning a p1 decision; it cannot
condition that choice on p1's forthcoming action.

`step` waits through engine resolution and any opponent replacements until p1 has
a new choice or the battle ends. Pivots and forced replacements may require more
than one decision in a turn. Invalid strings, nonstrings, and unavailable actions
set `last_error` and leave the game at its decision. Engine rejections due to hidden
trapping update the request normally. `noop` waits and clears the last error.
There is no wall-clock game timer. The transport timeout defaults to 30 seconds.
Infrastructure errors mark `done`, `info().valid_trial=False`, and an invalid reason;
they are never scored as an ordinary gameplay loss. Stderr goes under `runtime/`.
`close()` closes pipes, waits for exit, escalates to terminate/kill if needed, joins
the reader thread, and is idempotent. No source files are generated while playing.

## Scenarios and opponent

`SEEDS` in `game.py` declares 10 train, 10 held-out, 10 validation, and 2 dev seeds,
all disjoint. Arbitrary integer seeds also work. For each integer, SHA-256 hashes
the UTF-8 string `showdown-adaptor-v1:battle:<decimal seed>`; its first eight bytes
are interpreted as four big-endian unsigned 16-bit integers for Showdown's PRNG.
The domains `team-p1` and `team-p2` seed each player's random-team generator, and
`opponent` creates an independent PRNG for bot choices. This revision generates
nondeterministic teams if player seeds are omitted, even with a fixed battle seed.
All groups use
the identical generation procedure and difficulty. These seeds vary teams, leads,
levels, items, abilities, moves, Tera types, damage rolls, and tie resolution.
No fixed packed team is used in evaluation. Determinism holds for identical
configuration, engine build, seed, and action sequence; timestamps are excluded.
Scenario seeds are evaluator-only, to avoid revealing hidden teams via regeneration.

The default heuristic opponent estimates attack value using public typing, base
defenses, its own stats/boosts, STAB, accuracy, and priority. It can switch, recover,
set hazards/status, boost, and spend Tera when the immediate benefit justifies it.
Seeded small tie-breaking noise varies choices. This is a modest heuristic, not
an optimal damage calculator or an omniscient player; abilities and many special
mechanics are not modeled. `opponent="random"` instead samples available commands.
Default evaluation kwargs fix the heuristic opponent, 300 turns, and 900 accepted
decisions. Healthy opponents and random matchups leave substantial scoring headroom.

## Observations and actions

`describe()` and `ActionSpec` provide the agent's full initial instructions.
Actions are literal `move 1` through `move 4`, each optionally suffixed with
` terastallize`, and `switch 1` through `switch 6`. Choose from the current
`legal_actions`; these are request-derived candidates, with the engine final
authority for previously hidden restrictions. Slot numbers are one-based and
the team array order changes after switches. Revival Blessing uses a switch
command to select a fainted teammate. `noop` always waits.

The private `request` is Showdown's p1 JSON: exact own team HP/status, moves,
stats, items, abilities, PP/disabled flags, and Tera/trap availability. The public
`battle_log` is a rolling 700-line / 65k-character history; `recent_events` is
the most recent step's protocol, limited to 300 lines / 30k characters. `|t:|`
timestamps are removed. Repeated `observe()` calls are side-effect free.
`opponent_revealed`, `active`, and `public_summary` track common public identities,
rounded HP, revealed moves/items/abilities, Tera, boosts, effects, hazards, weather,
and terrain. `move_dex` and `species_dex` add public reference data only for the
player's moves/team and revealed species. The protocol is authoritative: summary
tracking is not a complete client mechanics interpreter, especially for Illusion,
Transform, unusual boost transfers, Baton Pass, and changes in effective typing.
An agent should retain useful history because older log lines are discarded.

No p2 private request, unrevealed Pokémon/set, exact opponent HP, RNG state, seed,
or referee component is put in the observation. Normal Showdown public information
(including rounded enemy HP) is retained. Observation values are strict JSON and
bounded below 200 kB for this singles format. At terminal states the request and
legal actions are cleared; the final protocol and public result remain available.

## Score and termination

For each side, HP fraction is `sum(current_hp / max_hp for all six slots) / 6`.
Fainted Pokémon contribute zero. `hp_margin = self_hp_fraction - opponent_hp_fraction`.
Outcome is +1 for a win, -1 for a loss, and 0 for a tie or an unfinished battle.
The scalar is `outcome + 0.25 * hp_margin`, with theoretical range [-1.25, 1.25].
An unfinished episode reports current HP margin with outcome zero. No HP
approximation from rounded client information is used for evaluator scoring.

`score_breakdown()` reports outcome, both HP fractions, margin, survivor counts,
turns, accepted decisions, and score as numbers. `info()` additionally contains
seed mappings, engine revision, validity, opponent, termination reason, and
`truncated`. Normal wins report `normal`; engine ties report `engine_tie`.
Crossing the configured turn limit or reaching the accepted decision limit
adjudicates a tie via the engine (`turn_limit` / `decision_limit`), preserving HP
margin and marking truncation. These resource limits are valid trials, unlike
infrastructure failures. Noops and malformed choices do not consume game decisions;
the evaluator remains responsible for limiting a policy that never acts.

## Verification

From the repository root, using the required evaluator interpreter:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/showdown/adaptor', expected_tier=1).to_markdown())"
```

From this directory, `.../runs/venvs/showdown/bin/python verify_adaptor.py` runs
complete real-engine episodes, deterministic action replays, malformed/noop
checks, truncation and bridge-failure handling, JSON size checks, and cleanup.
Results are saved to `runtime/verification.json`.

Verified on the supplied machine with Node 24.18.0 and evaluator Python 3.12.3:
the generic validator passed without warnings. Twenty complete battles plus a
full deterministic replay passed; all twenty ended normally. Against the default
heuristic on two dev and four train seeds, random and first-available policies
each won 0/6, while the type-aware scripted baseline won 4/6 (scores from -1.034
to +1.117). These are smoke-test results, not held-out performance estimates.
The largest observed JSON was 33,002 bytes. Malformed actions, noops, both resource
limits, a deliberately killed bridge, repeated reset, and idempotent cleanup passed.

### study/interfaces.md
# Pokémon Showdown random battles: programmatic interfaces

Target: local `gen9randombattle` singles in the supplied checkout (`~/gm-tools/pokemon-showdown`). The evaluator has no GUI and must work offline, so the local simulator is the practical choice.

## Interface comparison

| Interface | Maturity and latency | Observation and control | Legal/ToS, cost, limitations |
|---|---|---|---|
| **Showdown simulator `BattleStream` (chosen)** | First-party simulator API in the Showdown repository; deterministic seeded games and usually sub-millisecond to a few milliseconds per engine step on this host. | Node `BattleStream` emits protocol text; `getPlayerStreams` splits omniscient, spectator, and per-player streams. A player receives public updates plus its own `sideupdate` containing `|request|{JSON}`. Write `>p1 move 1`, `>p1 switch 3`, or `>p1 move 2 terastallize`. | Runs entirely offline, no account or server. Showdown is MIT-licensed code ([LICENSE](https://github.com/smogon/pokemon-showdown/blob/master/LICENSE)); retain attribution and do not redistribute proprietary data. Requires the checkout's Node version (package currently declares Node >=22.18). A bridge must parse protocol and must never feed the agent omniscient output. |
| **Showdown server WebSocket / HTTP protocol** | Mature production protocol used by the web client; network round trips commonly 20–300 ms and server queue/timer behavior applies. | Connect to a server WebSocket (for example the public endpoint advertised by [pokemonshowdown.com](https://pokemonshowdown.com/)); receive battle protocol and send `/choose move 1|<rqid>` or `/choose switch 3|<rqid>`. | Requires network and usually a guest or registered account; ladder battles have clocks and server policy. Automated play can violate server rules, trigger anti-bot controls, or risk an account. Costs no software license but is unusable in the offline evaluator. |
| **`poke-env` Python bot library** | Widely used open-source educational bot wrapper; event parsing and battle objects are convenient, but it is primarily a server client and adds Python/network overhead (tens of ms plus network). [GitHub](https://github.com/hsahovic/poke-env), [docs](https://poke-env.readthedocs.io/). | Subclass `Player`, inspect `Battle`/`Pokemon`/`Move` objects, return `choose_move`; the library translates to Showdown WebSocket commands. It exposes a structured, partial-information state rather than raw protocol. | Free (MIT). Needs a compatible `poke-env` install and a running Showdown server; package versions and server protocol can drift. Server automation still has ToS/anti-cheat risk. It does not directly wrap the local `BattleStream`, so using it offline would require a custom server adapter. |
| **Local Showdown server or custom simulator plugin** | The repository can run a local server and plugins can hook rooms/battles; mature engine but high integration and maintenance cost. | WebSocket client gets normal player-visible protocol; a plugin can expose extra telemetry or seed/reset hooks. | Offline and free, but plugin hooks can accidentally reveal hidden state and produce a nonstandard game. More processes and protocol work than `BattleStream`; use only if a needed server feature is missing. |
| **Built-in Node bot classes / random-player example** | `sim/tools/random-player-ai.ts` and `sim/examples/battle-stream-example.ts` are maintained smoke-test examples; latency is local-engine speed. | The class consumes player stream requests and writes legal random choices. Source state is protocol text. | Free and offline, but random policy is a baseline, not a strong agent. Reuse its stream lifecycle rather than its policy. [example](https://github.com/smogon/pokemon-showdown/blob/master/sim/examples/battle-stream-example.ts) |

…(truncated; full text in prior/interfaces.md)

6. On a move request, write exactly one legal command such as `move 1`, `move 2 terastallize`, or `switch 4` to the p1 stream. Choice syntax and errors are specified in [SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#possible-choices). On forced replacement, write `switch <slot>`; do not send a move. If a command is rejected, parse the following request and select again. Never use `default` as the strategic policy; it simply chooses the first legal option.
7. Read until the next `|request|` or `end`. At `end`, parse its JSON/log data and the referee's omniscient state only for score calculation. Report outcome, both sides' terminal HP fractions, HP margin, turn count, and whether the episode ended normally or was truncated.

The local text protocol is documented at [SIMULATOR.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md), [SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md), and the implementation of [`getPlayerStreams`](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-stream.ts).

## Credentials and environment

For the chosen local interface: **none**. Do not put credentials in source or study files. A server-backed alternative would need a server URL and, if using a registered account, username/password or a server-issued login token; those should be injected as environment variables at runtime and never committed.

## Practical risks

Keep the agent on its p1 stream: using `omniscient` or spectator output leaks hidden opponent sets and invalidates the task. Pin the checkout commit and format because random sets and protocol fields evolve. A Node/Python pipe must handle backpressure and blank-message framing. Seed values make starts reproducible but do not remove battle RNG; held-out seeds are still required. The standalone simulator has no wall-clock move timer, whereas a WebSocket server does.


## Measured model latency (at archive creation, 2026-10-30 03:16)
- System-2 (Astra), one-word answer: 74 s
- System-2, an answer of ~928 output tokens: 139 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).