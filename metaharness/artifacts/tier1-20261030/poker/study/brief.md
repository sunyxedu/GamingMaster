# Heads-up no-limit Texas Hold'em: game brief

**Tier 1 is correct.** For the selected offline simulator, time advances only when the adapter deals or applies an action. Astra may think for seconds or a minute without a poker penalty. There is no game clock, blind escalation, or opponent action while Astra is on turn. A harness may impose its own resource deadline, which is distinct from a game rule. `study.json` uses **`decision_latency_s: -1.0` as an explicit unlimited/no-game-timeout sentinel**, because standard JSON cannot represent infinity. Do not interpret this as a negative duration. [R2, L1]

Research status: Python and repository evidence were checked locally; web retrieval was blocked by DNS. Source URLs below are not live-verified. Details are in [research_log.md](research_log.md).

## Objective and match format

Two players compete for chips. Win a hand by making the other player fold, or by holding the best five-card poker hand at showdown. Maximize **expected net chips won per hand**, not number of pots won, survival time, or the chance of ending one short session ahead.

Recommended benchmark: no rake, no ante, small blind 50, big blind 100, both players start **each hand** with 20,000 chips (200 big blinds). Stacks reset after every hand; winnings accumulate in a separate score ledger. This matches the local ACPC-style reference configuration. It is a cash-game-like repeated-hand experiment, not an elimination tournament, even though the engine uses its `TOURNAMENT` rules mode to disallow folding when checking is free. No ICM, rebuy cost, increasing blinds, or tournament prize model applies. [L1, L2]

Use the frozen `heuristic-v1` opponent specified in `interfaces.md` unless the evaluator supplies another versioned policy. It runs entirely locally and has no model calls. Its thresholds are benchmark design choices, not claims of optimal play. PokerKit supplies the rules engine; the adaptor phase must implement this opponent and the private observation wrapper.

Fix the number of hands before evaluation; an initial development budget can be 200 hands, with larger final budgets as runtime permits. That number is a proposed evaluation setting, not a poker rule. Alternate seats and preferably use duplicate deals as specified in `interfaces.md`. Report:

- Primary score: total hero net chips divided by completed hands. Net chips include blinds and all later wagers; receiving the pot is not itself the profit.
- Secondary: `bb/hand = chips/hand / 100`; `bb/100 = 100 × chips/hand / 100` at these blinds. Always label units; do not confuse big blinds with fixed-limit “big bets.”
- Diagnostics: hands completed/required, total chips, opponent/version, seeds retained privately, illegal actions, timeouts, and uncertainty over independent deal pairs. An incomplete match must not masquerade as a full evaluation.

On a small-blind fold, the small blind loses 50 and the big blind wins 50, not 150. At showdown, net changes sum to zero. Resetting stacks must not add winnings to the score a second time. [L1, L2]

## Rules the agent must understand

Use one standard 52-card deck, no jokers. Each player has two private hole cards. Community cards appear in four betting rounds: **preflop** (no board), **flop** (three cards), **turn** (one more), **river** (one more). Only exposed board cards and one's own hole cards are available for a decision. An actual table burns a card before each board street; the proposed simulation fixes this deal schedule privately. [R1, R2]

In heads-up play the **button posts the small blind and acts first preflop**. The big blind acts first on every postflop street, so the button has postflop position. If the button only calls the big blind preflop, the big blind still gets a check-or-raise decision. Alternate which participant occupies each position. PokerKit's pinned reference uses engine seat 0 = BB and seat 1 = button/SB; do not assume seat 0 is the button. [R1, L1]

Legal decisions:

| Action | Meaning |
|---|---|
| Fold | Surrender the hand and money already invested; offer only when facing a wager in this benchmark. |
| Check | Pass when nothing is owed. |
| Call | Match the outstanding wager, or commit the remaining stack if that is insufficient. |
| Bet | First wager on a postflop street; ordinarily at least one big blind. |
| Raise | Increase an existing wager; the minimum increment is the size of the last full bet/raise on that street. |
| All-in | Commit the remaining stack. This can be a call, a full raise, or a legal short raise; it is not always a separate raising option. |

Example: with 50/100 blinds, the minimum open is **to 200**. If someone raises to 300 (a 200-chip increase over the big blind), a full reraise must be to at least 500. An all-in can be smaller than a full minimum raise. A short all-in does not normally reopen raising to a player who has already acted; heads-up, once the only opponent is all-in, there is no additional opponent against whom to raise. Use the engine's legal bounds rather than reconstructing these exceptions in the model. Return unmatched excess chips. There are no contested side pots between just two players. [R1, L1]

A round closes when all players able to act have matched the wager and had their required opportunity to act, or everyone checks. If a player folds, pay out immediately. If no further betting is possible after a called all-in, finish the board automatically and resolve the hand. Do not request dummy Astra actions for dealing or runouts. [R2, L1]

At showdown choose the best five cards from the seven available, using **zero, one, or two** hole cards. Ranking, strongest first: straight flush; four of a kind; full house; flush; straight; three of a kind; two pair; one pair; high card. A royal flush is the highest straight flush. Compare ranks and kickers within a category. Ace can be low in A-2-3-4-5, but straights do not wrap K-A-2. Suits have no strength order. Identical best five-card hands split the pot; if everyone plays the board, neither player's unused hole-card kicker breaks the tie. Let the engine handle odd chips. [R1, R2]

## Observation and pace

At each hero turn provide structured JSON: own cards, board, street, position, both public stacks and contributions, pot including outstanding wagers, amount to call, legal raise-to bounds, public action history, and a small set of exact legal candidate actions. Hide opponent cards, undealt cards, burn cards, RNG state, and deal seeds. The opponent gets the same information boundary from its own seat. Showdown reveals may become public only when the engine actually exposes them. [R2, L1]

There is no keyboard, mouse, screen capture, account, or realtime reaction requirement. The adapter applies one legal action, automatically advances through opponent/chance events, and stops at the next hero decision or terminal state. Jev is optional; code should perform arithmetic and legal checks, and Astra should choose strategy.

## Sources

- **R1:** Poker Tournament Directors Association rules, particularly heads-up button/blinds, full raises, and reopening betting: <https://www.pokertda.com/poker-tda-rules/>. General Hold'em explanation: <https://www.pokernews.com/poker-rules/texas-holdem.htm>. Physical-table procedures do not override the explicitly chosen simulator convention.
- **R2:** PokerKit documentation and implementation: <https://pokerkit.readthedocs.io/en/stable/>; <https://github.com/uoftcprg/pokerkit>.
- **L1:** Locally inspected reference integration: `/home/hemlix/GamingMaster-mh-tier1/scripts/reference/pokerkit_hunl.py`, pinned to PokerKit 0.7.6, and `tests/test_hunl_reference.py`. Public repository reference: <https://github.com/sunyxedu/GamingMaster> (local checkout is the verified evidence).
- **L2:** Local rules: `/home/hemlix/GamingMaster-mh-tier1/data_processing/rs/gmrules/src/games/hunl/mod.rs`; recorded fixture `tests/fixtures/hunl/pokerkit_reference.json`. ACPC data provenance: <https://doi.org/10.5281/zenodo.10796885>.
