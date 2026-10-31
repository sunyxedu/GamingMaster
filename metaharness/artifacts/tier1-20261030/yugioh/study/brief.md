# Yu-Gi-Oh! — headless EDOPro benchmark brief

## Scope and evidence

Study date: 2026-10-29. Target: Linux x86_64, Python **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python`, offline runtime, free software, no GUI. Recommend ordinary two-player Yu-Gi-Oh! Master Rule duels, not Rush Duel, Speed Duel, tag duels, or a live ranked service.

**Research limitation:** attempts to retrieve GitHub, raw.githubusercontent.com, and the official rulebook site failed with `curl: (6) Could not resolve host`. No browser/search connector is available. No relevant cached core, card database, scripts, or decks were found in the inspected local tool/cache directories. Source links below are reference targets from established documentation, **not pages retrieved in this session**. Rules are established game knowledge; current repository layouts, licenses, build recipes, and current tournament lists must be checked when sources are supplied. No installation or duel test was performed. `interfaces.md` distinguishes the proposed adaptor contract from upstream APIs.

## Objective and rules that affect decisions

- Each player normally begins at **8,000 Life Points (LP)** with five cards. Win by reducing the opponent to 0 LP, making them unable to draw when required, or satisfying a card's alternate victory condition. Simultaneous defeat can draw. Use the **engine's terminal result**, not an LP heuristic, to decide the winner. A Match usually consists of multiple Duels; this benchmark scores **single Duels**, with no side decking. [R1]
- Main Deck: 40–60 cards. Extra Deck: up to 15. Side Deck: up to 15 for matches. Normally at most three copies of a card name across all three, subject to a format's Forbidden/Limited list. The proposed initial benchmark uses exactly 40 Main Deck cards and empty Extra/Side Decks, with an explicitly custom three-copy format. It must not be described as current tournament legal. [R1, R4]
- Turns proceed through Draw, Standby, Main Phase 1, Battle, Main Phase 2, and End Phase, subject to rules/effects. The first player skips the first draw and cannot conduct a Battle Phase on their first turn. Main Phase 2 follows a conducted Battle Phase; do not invent a Main Phase 1 → Main Phase 2 transition. The normal hand limit is six at the end of the turn; effects can change rules. [R1]
- Normally one Normal Summon **or Normal Set** per turn. Level 5–6 monsters usually require one Tribute; Level 7+ usually require two. Special Summons have their own conditions and restrictions, not a universal once-per-turn limit. Attack-position monsters are face-up; face-down monsters are normally in Defense Position. Links cannot be in Defense Position. [R1]
- Extra Deck mechanics, if the pool is later expanded: Fusion uses its specified materials and a suitable summoning method; Ritual usually uses a Ritual effect and required Tributes to summon a Ritual Monster from the Main Deck/hand; Synchro generally combines a Tuner and other listed materials whose Levels total the summoned monster's Level; Xyz generally stacks the listed same-Level materials under a monster with a Rank; Link uses listed materials whose contribution satisfies its Link Rating. Pendulum Summoning uses scales and has location/zone restrictions, especially for face-up Extra Deck monsters. These are only summaries: card-specific methods, alternate materials, and the chosen Master Rule revision control legality. Current ordinary Master Rule placement differs between Fusion/Synchro/Xyz and Link or face-up Extra Deck Pendulum summons. Use the core's exact options rather than deriving material/zone legality from this paragraph. [R1, R6]
- Monsters, Spells, and Traps implement a much larger rules system than their printed ATK/DEF. Costs, activation conditions, targets, effects, continuous modifiers, restrictions, and once-per-turn clauses matter. The core and the matching Lua scripts are the authority. A language model must never validate legality from card prose alone. [R1, R3, R6]
- Battle: attacking an Attack Position monster normally compares ATK; the weaker monster is destroyed and its controller takes the difference. Attacking a Defense Position monster compares attacker ATK with defender DEF; destruction and damage differ, and piercing is an effect rather than a universal rule. Direct attacks usually require an empty opposing monster field. Effects and Damage Step rules can change these outcomes. Have code calculate projected damage, and the engine resolve it. [R1, R5]
- Effects form **Chains**, which resolve in reverse activation order. There is no opportunity to insert a new activation between resolving Chain Links. Players can respond on the other player's turn. Normal Spells and ignition effects generally require an appropriate open game state; fast effects and Counter Traps have different timing/Spell Speed rules. Trigger ordering, damage timing, and delayed triggers require the engine's prompts. “My turn” is not the same as “my decision.” [R1, R2, R3, R5]
- Public information includes face-up field cards, public Graveyards, face-up banished cards, counts, LP, and announced actions. Hands, face-down cards, Main Deck order, and unrevealed Extra Deck information are private. Effects can reveal particular cards temporarily. Even with a known fixed deck list, the opponent's current hand and shuffle are unknown. [R1]

## Controls and observation

The chosen interface supplies **semantic legal choices**, not mouse coordinates. Astra receives an acting-player view of the board, card text for known cards, public action history, and the current engine request. It chooses one opaque action ID. Requests can ask for:

- Summon, Set, Special Summon, change position, activate a particular effect, attack, or advance phase.
- Respond to a Chain, decline an optional response, select an effect or yes/no option.
- Select one or several cards as costs, targets, Tributes, or materials; select a zone, position, number, card name, Attribute, or Type; order simultaneous effects or other lists.

One “turn-by-turn” model decision **cannot** replace these interaction windows. Step until a decision for the agent, including during the opponent's turn; let the local opponent answer its own prompts. Never treat an optional “decline activation” as “end turn.” See the protocol and action contract in `interfaces.md`.

## Pace and tier

**Tier 1 is correct.** A locally embedded duel core advances when the host processes it and supplies requested responses. It does not require a real-time rendering loop or human-speed input. Automatic effect resolution can generate several events before the next request, but the duel waits for that response. This leaves seconds to a minute for Astra; Jev is optional and offers no reaction-time requirement. A native processing call can still hang on a bad script, so process supervision is necessary. [R6]

There is **no intrinsic finite thinking deadline** in the proposed untimed local core. For a finite operational value in `study.json`, the proposed benchmark sets an **external 300-second per agent decision deadline**, longer than the expected Astra call. At that deadline record `decision_timeout` and an agent forfeit, separately from a normal defeat. This is a proposed configurable evaluation policy, not a Yu-Gi-Oh! rule or measured engine limit. Engine processing timeouts are separate infrastructure failures. Online room clocks and tournament time procedures do not apply here.

## Score and progress

Report `outcome` (win/loss/draw), terminal reason, agent/opponent LP, turn count, decision count, and truncation/error flags. Define outcome value `w = +1` for a win, `-1` for a loss, `0` for a genuine draw. Define bounded LP margin `m = clamp((agent_LP - opponent_LP) / 8000, -1, +1)`. Recommended scalar for a completed duel: **`score = w + 0.1*m`**. Thus every win scores above every draw and every draw above every loss. A deck-out/alternate-condition win remains a win even with lower LP. Also report unmodified win rate; the small margin is only a tie-breaker, not a substitute objective.

No intermediate LP reward is required. Paying LP for a strong effect can be correct; LP gain alone is not strategic progress. Maintain separate counts of successful turns, legal responses, and terminal completions for adaptor diagnostics. Proposed deterministic safety caps are 200 turns or 10,000 response decisions; label exceeding either `truncated`, report a benchmark draw with score 0, and publish its rate separately. Never convert an unsupported prompt, missing script, or crash into a game draw. Benchmark timeouts may score as forfeits but must remain identifiable.

## Sources

- **R1 — Official rulebook landing page:** https://www.yugioh-card.com/en/rulebook/ — foundational rules, turn phases, zones, card types, summoning, battle, chains. Obtain a local copy before final rules validation.
- **R2 — Official Fast Effect Timing:** https://www.yugioh-card.com/en/play/fast-effect-timing/ — open game state and response opportunities.
- **R3 — Community Problem-Solving Card Text reference:** https://yugipedia.com/wiki/Problem-Solving_Card_Text — conditions, costs/targeting, resolution language; use official card text/rulings for a disputed interaction.
- **R4 — Official Forbidden & Limited Lists:** https://www.yugioh-card.com/en/limited/ — relevant only if a dated tournament format is adopted.
- **R5 — Community Damage Step reference:** https://yugipedia.com/wiki/Damage_Step — restricted activation windows and battle timing.
- **R6 — EDOPro native core:** https://github.com/edo9300/ygopro-core — authoritative native lifecycle, processing, query, and response interface once pinned and inspected.
- **R7 — Official card database:** https://www.db.yugioh-card.com/yugiohdb/ — card text and card-specific information; not an offline duel service.
