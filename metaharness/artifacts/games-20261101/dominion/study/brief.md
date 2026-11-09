# Dominion: study brief

Scope: two-player, offline **Dominion (deck-building card game)**, initially the Base set, second edition. Tier **1**. Written 2026-10-29. The target interpreter was checked directly: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python` is **Python 3.12.3**. No game package is installed.

**Evidence limitation:** this session has no web/browser tool and outbound HTTPS requests failed DNS resolution for PyPI, GitHub raw content, and Dominion Strategy Wiki. No local simulator source was found. Rules and strategy below are established game knowledge, accompanied by references for verification; these URLs were not successfully retrieved in this session. Software APIs, release support, licenses, and current service terms remain unverified. See `sources.md` for the evidence ledger. Do not interpret citations as claims of live browsing.

## Objective and setup

Build a deck with the greatest number of victory points (VP) **when the game ends**, not the most money or cards. Each player begins with 7 Coppers and 3 Estates, shuffles, and draws 5. Players take alternating turns; the shared Supply normally contains basic Treasures, basic Victory cards, Curses, and exactly 10 Kingdom piles. For two players, basic and Kingdom Victory piles normally contain 8 cards; ordinary Kingdom piles contain 10; the Curse pile contains 10. Starting Estates are additional to the Supply's 8 Estates. Initial Copper supply is 60 minus starting Coppers (46 remaining in a two-player game), Silver 40, Gold 30. Exclude expansions and special pile rules initially. [R1, R2]

| Basic card | Cost | Effect / final value |
|---|---:|---|
| Copper | $0 | $1 when played |
| Silver | $3 | $2 when played |
| Gold | $6 | $3 when played |
| Estate | $2 | 1 VP |
| Duchy | $5 | 3 VP |
| Province | $8 | 6 VP |
| Curse | $0 | −1 VP; Curse is its own type, not a Victory card |

Gardens is a Base Kingdom Victory card: **each copy** is worth floor(total cards owned / 10) VP at game end, counting itself and all owned cards. The engine must calculate variable VP. Cards in hand, deck, discard, play, and any other owned zone count; trashed cards do not. [R1, R3]

## A turn and all decision types

1. **Action:** start with 1 Action. Playing an Action consumes one Action, then resolve its printed instructions in order, as far as possible. Some cards draw, give more Actions, give money or Buys, gain cards, trash cards, or attack. The player may stop playing Actions even if more are legal. An Action that adds no Actions is commonly called a terminal Action.
2. **Buy:** start with 1 Buy. Play Treasures to produce money; Action-generated money also counts. Spend money and a Buy per purchase. Unused money/Actions/Buys normally disappear at turn end. Base-only Treasures can be resolved automatically once the player enters Buy; Merchant's first-Silver bonus must still be implemented by the engine. Do not assume arbitrary Treasure ordering is harmless when adding expansions.
3. **Clean-up:** discard hand and cards in play, draw 5 for the next turn. If a draw needs more cards than remain in the deck, draw those remaining, shuffle the discard to make a new deck, then continue drawing. Do not shuffle just because the deck became empty if no draw is requested. If both deck and discard run out, draw as many as possible. [R1, R2]

A gain normally places the card in the discard pile; a purchase is a gain, but gaining is not necessarily buying. Exceptions are on the card: Mine gains a Treasure to hand, Artisan gains to hand then topdecks a hand card. A gain without buying does not consume a Buy or money. Card costs, not produced money, govern Mine/Remodel/Workshop/Artisan limits. Trashing removes cards from ownership rather than discarding them. [R1, R3]

The adaptor needs prompts **within** a turn as well as for action and purchase choices: choose cards to discard, trash, gain or topdeck; order a sequence; pick a mode; optionally reveal a reaction; select an Action to play twice. Militia's victim discards to 3 during the opponent's turn. Moat's reaction is optional and does not play the Moat. Throne Room plays the chosen Action twice without spending another Action; mandatory effects happen twice, with intermediate choices resolved separately. Playing Throne Room again is not a new turn. [R1, R3]

**Edition boundary:** second-edition Base includes Artisan, Bandit, Harbinger, Merchant, Poacher, Sentry, Vassal. First-edition cards Adventurer, Chancellor, Feast, Spy, Thief, Woodcutter are not part of that second-edition pool. Match the manifest to actual simulator support instead of silently mixing editions. [R2]

## End, score, and progress

Check for end of game at the end of a player's turn: the Province pile is empty **or any 3 Supply piles are empty**. Curses and basic piles count toward the three. An emptied pile does not stop resolution mid-turn. Expansion rules (e.g. Colony) are outside the initial scope. Highest final VP wins; if VP is tied, the tied player who took fewer turns wins; if both VP and turns are tied, the game is tied. There is no equalizing extra turn after the ending turn. [R1, R4]

Required evaluator outputs: `agent_vp`, `opponent_vp`, integer `vp_margin = agent_vp - opponent_vp`, `outcome` (`win`, `draw`, `loss`), and `win_score` (1, 0.5, 0). Also record each player's turns, end reason, kingdom, seeds, and opponent policy. Margin can be zero in a win because of the turn-count tiebreak. Do not use margin alone to infer winner. Report win rate and mean VP margin separately across paired seats. A truncation or simulator failure is not a legitimate game end or a fabricated win. [R1]

Useful progress indicators are Province count, number of empty/nearly empty piles, public VP ledger, deck size/composition, deck cycling speed, payload (money/gains/attacks) per shuffle, and ability to buy multiple scoring cards. These are descriptive observations, not replacements for terminal score. Early VP is frequently a poor progress reward because buying green cards weakens future hands.

## Time and controls

This is **tier 1**: a synchronous offline simulator changes state only when a chosen action resolves or the simulator advances forced effects. There is no physics tick, built-in decision deadline, or deterioration while Astra thinks. Human online clocks and tournament slow-play rules are irrelevant to this chosen offline setup. Astra may take seconds to a minute on every nontrivial decision without in-game penalty; Jev is optional.

`study.json` uses `decision_latency_s = 86400.0` as a **conservative representable lower bound (one day allowed), not a rules-imposed timeout**. The actual rules-level limit is unbounded, which strict JSON cannot encode as Infinity. Any evaluator/model timeout must be tracked separately as infrastructure policy. Disable a game clock; permit suspension at every nested prompt.

Programmatic controls will be legal option IDs passed to the simulator's decision hooks through the adaptor, not keystrokes. `interfaces.md` defines the intended JSON contract and installation/verification path.

## References

- [R1] Rio Grande Games, official Dominion second-edition rulebook: https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion-2nd-Edition-Rules.pdf ; publisher product page if the PDF path has moved: https://www.riograndegames.com/games/dominion-2nd-edition/
- [R2] Dominion Strategy Wiki, Base set / editions: https://wiki.dominionstrategy.com/index.php/Dominion ; general rules: https://wiki.dominionstrategy.com/index.php/Gameplay
- [R3] Card reference index: https://wiki.dominionstrategy.com/index.php/Card_list ; example: https://wiki.dominionstrategy.com/index.php/Gardens
- [R4] End conditions: https://wiki.dominionstrategy.com/index.php/End_of_game
