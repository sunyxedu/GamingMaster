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
