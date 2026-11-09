Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Pokémon Showdown — Gen 9 Random Battle` — Play Pokémon Showdown Gen 9 Random Battle singles as p1 (Agent) against a
Foul Play MCTS opponent. It uses damage simulation, switch search and public-set inference. Both teams have six randomly generated Pokémon,
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
are populated at termination. The configured turn/decision limits and 480 seconds of cumulative
engine work adjudicate a tie. Your thinking time is excluded from that engine-work budget.

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

…(truncated; full text in prior/brief.md)

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
# Pokémon Showdown: offline Gen 9 Random Battle

`game.py` implements Tier 1 `make_game(**kwargs)` / TurnGame, reusing the engine
plumbing preserved in `game_v2.py`. It runs the real Pokémon Showdown simulator,
with freshly generated six-Pokémon teams, entirely offline. No GUI, account,
credentials, local server, or connection to a live game service is needed.

## Installed opponent and measured strength

The fixed opponent is **[Foul Play](https://github.com/pmariglia/foul-play)**,
revision `8f541f80f91f960603673796559f796b10ccf0fb`, using **poke-engine 0.0.48**
compiled with Gen 9 Terastallization support. Foul Play is an established open-source
search bot, substantially stronger here than the old adaptor's heuristic. Showdown
itself supplies only a random example player; that is not used as the opponent.
No trained policy or live service is required.

Each decision uses Foul Play's protocol parser and random-battle inference to
sample **four plausible hidden teams**, then runs Rust Monte Carlo tree search
for **200 ms per sample, one thread**. Search visits are aggregated by their share
of visits in each sample; the highest-ranked legal choice is selected. It searches
attacks, switches, setup, recovery, status and Tera, with public-set inference.
Every scenario uses the same opponent and budget; no weaker option is exposed.

### Hard compute bounds (this revision)

The saved worker already supplied 200 ms to the native engine, but had no external
search watchdog. poke-engine 0.0.48 checks elapsed time only after batches of 1,000
iterations, so that parameter is not a hard wall-clock limit. The reported
150–490-second stalls were not reproduced in this build session; the first three
old-worker steps measured about 0.91 seconds. This revision explicitly contains
native overruns and stalls elsewhere in the decision pipeline:

* Each decision has a **2.0-second budget**, starting before protocol parsing.
  Inference and MCTS run in a fresh Linux fork child. It publishes the cumulative
  policy after each completed sample. The parent stops/reaps the child at the
  deadline and selects from the completed MCTS results. Cleanup allows up to
  0.2 seconds beyond the search deadline. No random/heuristic fallback is used.
* A **3-second Node watchdog** covers worker startup, parsing, inference, native
  search and IPC. If no sample completes, or the worker stalls, the trial is
  marked invalid with an infrastructure error; the evaluator can retry it.
* A **6-second Python transport timeout** bounds an entire reset/step, including
  chained opponent choices. Infrastructure failure also closes the process group.
  Cleanup can take up to six additional seconds on an unresponsive process; it
  is not a successful gameplay step. The injected worker stall returned in 3.08 s.
* Defaults are **120 turns / 180 accepted player decisions**. An additional
  **480 seconds of cumulative engine work** forces a tie at the next completed
  step. This budget counts reset/step work, excludes the agent's thinking time,
  and can overshoot by the final bounded RPC. It keeps engine work below roughly
  nine minutes even for unusually long episodes, leaving time for model calls.

The per-sample target is unchanged from the prior bot, retaining its normal
search strength. The killable child is necessary because a Python thread or signal
handler cannot reliably interrupt a native extension holding the GIL. New referee
telemetry records search visits, bot decision count, deadline hits, total/max bot
compute seconds, and cumulative engine seconds. None is shown in observations.

### Measured latency and opponent strength

**Bounded Foul Play won 103 / 110 games (93.6%), with 7 losses and no ties**
against the exact old heuristic, approximately 95% Wilson interval **87.4–96.9%**.
Seeds were `1000003 + 7919*i`, `i=0..109`, disjoint from all evaluation splits,
with generated teams and the same fixed compute settings. Foul Play played p2;
the preserved heuristic played p1 from its own player stream, using unchanged
ratings and seeded tie-breaking. Its only generalization is finding the enemy by
opposite side ID. The benchmark-only `baseline` RPC is not an exposed action.
This is a small single-orientation strength estimate, not a ladder rating.

One initial game (seed 1799822) exposed an existing Illusion parser bug: `replace`
can omit HP. The bridge now preserves the previous public HP/effects in that case.
That seed was replayed after the fix and completed normally. The 110 results above
include that one replacement, not an extra game. Raw initial results, the failed
trial, and retry are retained in `runtime/opponent-benchmark-bounded.json` and
`runtime/benchmark-illusion-retry.json`; the merged results and summary are
`runtime/opponent-benchmark-final.json` and `runtime/benchmark-summary.json`.
All final games finished normally. The heuristic's best score was +1.11958,
below the maximum +1.25; it won seven games, leaving useful score headroom.

Measured on this Linux host with the evaluator Python, Node 24.18.0 and eight
concurrent benchmark games (some integration checks overlapped):

| Measurement | Bounded result |
| --- | ---: |
| Step calls across strength benchmark | 2,643 |
| Median / p95 / maximum step latency | 0.950 / 1.887 / 2.555 s |
| Mean / maximum full battle duration, including reset | 21.36 / 53.38 s |
| Mean turns / accepted player decisions | 19.09 / 24.03 |
| Maximum bot decision compute | 1.064 s |
| Natural search deadline hits | 0 |

Full random-player games on both dev seeds also completed normally:

| Seed | Turns / decisions | Reset | Whole battle | Median / p95 / max step |
| --- | ---: | ---: | ---: | ---: |
| 869431 | 24 / 29 | 2.017 s | 25.785 s | 0.955 / 1.000 / 1.883 s |
| 895003 | 21 / 27 | 1.886 s | 21.000 s | 0.943 / 1.002 / 1.007 s |

Across all six integration games there were 159 steps, median 0.945 s, p95 1.858 s,
maximum 1.913 s, and battle durations 19.79–25.79 seconds. All six baseline players
lost. A 40–60-decision game at the measured pace has ample room inside 15 minutes;
these random players generally lost sooner. Timings exclude model calls, include
real engine work, and can vary with load. The watchdogs bound pathological cases;
no natural stall was needed to verify them (see fault injection below).

The historical unguarded report (102 wins / 8 losses in 110 games, 92.7%) remains
in `runtime/opponent-benchmark.json` for comparison; it is not substituted for the
bounded measurement. Native MCTS randomness means individual replays differ.

## Setup and dependencies

Run `bash setup.sh` in this directory. Build time may use the network; play time
does not. Setup checks the supplied built Showdown checkout and installs
`requirements.txt` using `uv` into the evaluator's Python 3.12.3 environment:

`/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python`

It clones Foul Play if absent and checks its pinned revision. Rust/cargo is needed
only to build poke-engine. Defaults are `~/gm-tools/pokemon-showdown`,
`~/gm-tools/foul-play`, and `node`; optional overrides are in `env.example`.
No environment variable is mandatory. `SHOWDOWN_PYTHON` is internal and is set by
`game.py` to `sys.executable`, ensuring the bot uses the evaluator's environment.

Required Showdown revision: `a5df8274e85b0889bf2a9b3422a08b39732374fc`.
The supplied build uses Node 24.18.0; Node >=22.18 is required. Revision mismatches
fail explicitly. Keep both external checkouts and compiled dependencies unchanged;
git HEAD checks do not detect local modifications or stale build products.

Public random-battle inference data is bundled in `data/gen9randombattle.json`,
downloaded during this build from
`https://pkmn.github.io/randbats/data/full/gen9randombattle.json`.
SHA-256: `6a4b6fc8391d334e252001d8a48ec383159a7ccfdcab23830ab6def00aab793c`.
The worker replaces upstream's download-capable loader with a local-file-only

…(truncated; full text in prior/ADAPTOR.md)

requests/actions and preserve the public result and recent log. Repeated observe
calls are side-effect free; returned data can be mutated without changing state.

## Score and limits

`self_hp_fraction = sum(current_hp/max_hp over six Pokémon)/6`, likewise for the
opponent. `hp_margin = self_hp_fraction - opponent_hp_fraction`. Outcome is +1 for
a win, -1 for a loss, and 0 for a tie/unfinished battle.

**Scalar score = outcome + 0.25 * hp_margin**, range [-1.25, 1.25]. Winning remains
primary. `score_breakdown()` provides outcome, both fractions, margin, survivors,
turns, accepted decisions, bot search visits, timing/deadline telemetry and score. `info()` adds seed mappings,
validity, engine revision, opponent, truncation and termination reason. These are
evaluator-only. Intermediate scores reflect current HP with outcome zero.

Defaults are 120 turns and 180 accepted decisions, plus 480 seconds of cumulative
engine work. Limits force an engine tie, preserve HP margin and mark `turn_limit`,
`decision_limit` or `engine_time_limit` as truncated. Normal
wins use `normal`, natural ties `engine_tie`. Noops/bad actions do not consume
accepted decisions; the evaluator must bound agents that never act.

Foul Play/poke-engine is a separate mechanics model from authoritative Showdown,
so unusual mechanics or new data can differ. Public-set inference is uncertain;
MCTS has a finite budget and is not optimal play. This adapter uses upstream
Foul Play's parsing/sampling and poke-engine search, but integrates it over local
pipes instead of its normal WebSocket client and chooses the highest aggregated
legal visit share rather than upstream's random selection among close candidates.

## Verification

The requested generic validator prints **PASS with no warnings**. Run from the
repository root using the evaluator environment:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/showdown/adaptor', expected_tier=1).to_markdown())"
```

`runtime/validation.md` records the result. From this adaptor directory:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python verify_adaptor.py
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python verify_deadlines.py
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python benchmark_opponent.py --count 110 --workers 8
```

`verify_adaptor.py` plays six complete games (two each with random, first-available
and type-aware scripted actions). It checks JSON/privacy/size, score arithmetic,
malformed/noop handling, turn/decision limits, repeat reset, first-observation
reproducibility, a deliberately killed bridge, and idempotent cleanup. Per-step
latencies and complete episode results are in `runtime/verification.json`.

`verify_deadlines.py` injects search hangs with and without a completed result,
then suspends a real worker with SIGSTOP. The two child-search checks returned in
0.203 seconds for a 0.2-second test budget. The real worker watchdog invalidated
the trial in 3.08 seconds; all children were reaped. Details are in
`runtime/deadline-verification.json`. These tests exercise the stopping mechanism
without waiting for a naturally occurring multi-minute native stall.

A further complete random-player battle runs under `strace -f -e trace=network`.
The trace contains no network connect, listen or send calls. Its source-hash checks are in `runtime/network-trace-bounded.log` and
`runtime/offline-verification.json`. Runtime data stays under `runtime/`; playing
does not modify adaptor source or dependency manifests.

The Illusion regression is covered by `node verify_protocol.cjs` and the complete
replayed battle. An injected exhausted cumulative budget also produced a scored
tie in the real simulator (`runtime/engine-limit-verification.json`). The largest
observation in the strength benchmark was 26,846 bytes, below the 200 kB limit.

### study/interfaces.md
# Pokémon Showdown random battles: programmatic interfaces

Target: local `gen9randombattle` singles in the supplied checkout (`~/gm-tools/pokemon-showdown`). The evaluator has no GUI and must work offline, so the local simulator is the practical choice.

## Interface comparison

| Interface | Maturity and latency | Observation and control | Legal/ToS, cost, limitations |
|---|---|---|---|
| **Showdown simulator `BattleStream` (chosen)** | First-party simulator API in the Showdown repository; deterministic seeded games and usually sub-millisecond to a few milliseconds per engine step on this host. | Node `BattleStream` emits protocol text; `getPlayerStreams` splits omniscient, spectator, and per-player streams. A player receives public updates plus its own `sideupdate` containing `|request|{JSON}`. Write `>p1 move 1`, `>p1 switch 3`, or `>p1 move 2 terastallize`. | Runs entirely offline, no account or server. Showdown is MIT-licensed code ([LICENSE](https://github.com/smogon/pokemon-showdown/blob/master/LICENSE)); retain attribution and do not redistribute proprietary data. Requires the checkout's Node version (package currently declares Node >=22.18). A bridge must parse protocol and must never feed the agent omniscient output. |
| **Showdown server WebSocket / HTTP protocol** | Mature production protocol used by the web client; network round trips commonly 20–300 ms and server queue/timer behavior applies. | Connect to a server WebSocket (for example the public endpoint advertised by [pokemonshowdown.com](https://pokemonshowdown.com/)); receive battle protocol and send `/choose move 1|<rqid>` or `/choose switch 3|<rqid>`. | Requires network and usually a guest or registered account; ladder battles have clocks and server policy. Automated play can violate server rules, trigger anti-bot controls, or risk an account. Costs no software license but is unusable in the offline evaluator. |

…(truncated; full text in prior/interfaces.md)

The local text protocol is documented at [SIMULATOR.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md), [SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md), and the implementation of [`getPlayerStreams`](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-stream.ts).

## Credentials and environment

For the chosen local interface: **none**. Do not put credentials in source or study files. A server-backed alternative would need a server URL and, if using a registered account, username/password or a server-issued login token; those should be injected as environment variables at runtime and never committed.

## Practical risks

Keep the agent on its p1 stream: using `omniscient` or spectator output leaks hidden opponent sets and invalidates the task. Pin the checkout commit and format because random sets and protocol fields evolve. A Node/Python pipe must handle backpressure and blank-message framing. Seed values make starts reproducible but do not remove battle RNG; held-out seeds are still required. The standalone simulator has no wall-clock move timer, whereas a WebSocket server does.


## Measured model latency (at archive creation, 2026-10-31 12:11)
- System-2 (Astra), one-word answer: 8.0 s
- System-2, an answer of ~922 output tokens: 75 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).