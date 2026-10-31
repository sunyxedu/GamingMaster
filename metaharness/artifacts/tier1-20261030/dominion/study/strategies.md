# Dominion strategy prior for Astra and the harness

Scope: two-player Base-set Dominion, preferably second edition, with a tested card whitelist. This is strategic guidance, **not gameplay code**. Claims synthesize established community knowledge; network access was unavailable, so cited pages were not read live and no win-rate or simulator-performance claim is made. See `sources.md`. Exact card text and legality always outrank a heuristic.

## Division of responsibility

- **Code:** rules, legal option enumeration, shuffle/zone bookkeeping, card counts, cost/VP arithmetic, turn counts, public-event memory, exact visible-state endgame calculations, serialization/privacy, deterministic baseline policies, and optionally legal rollout/search. Never ask Jev to count cards, add money, determine legal costs, or compute a win from scores.
- **Astra prompts:** kingdom analysis, choosing money versus engine versus alternate scoring, selecting an opening, acquisition order, balancing components, opponent adaptation, deciding when to stop building and score, and revising the plan after draws/attacks. Tier 1 permits Astra at every meaningful choice.
- **Jev questions (optional):** recognize a situation or rank **already legal** options from supplied features and Astra's current plan. It does not invent a multi-turn plan, implement rules or calculate. Batch independent judgments about the same frozen observation; do not batch questions whose answers change what is subsequently drawn/revealed. Prefer no Jev dependency in the first working adaptor.

Astra should maintain a small plan record: chosen archetype; desired next acquisitions in order; draw/action/payload deficits; trash priorities and stopping rule; threats from opponent; likely ending; and conditions to switch to scoring. Update it at least each own turn and on a significant public attack/pile change. Give Astra current card text, legal options, counters, ownership ledger, remaining piles and prior plan. Never require it to reconstruct the game from raw prose alone.

## Read the kingdom before buying

Inventory the board's **trashing**, **draw**, **+Actions**, **+Buy/gaining**, **payload** (money, attacks, direct gains), **defense**, and **scoring/end conditions**. A Village is not an engine by itself; draw must reach the useful payload and enough Actions must let it be played. An engine can be weak if it draws well but buys only one card or cannot produce money. Conversely, an efficient money deck needs few moving parts and wins against slow unfocused construction. [S1, S2, S3]

**Code:** expose role tags and exact effects for implemented cards, terminal-action count, card counts and public pile pressure. **Astra:** choose a coherent build based on combinations and expected game speed. **Jev:** classify a supplied choice as supporting draw, action capacity, payload, thinning, or scoring; these tags can normally be static code instead.

Before committing to a long build, check: can Witch flood the deck with Curses; can the opponent race Provinces; can Workshops/Gardens empty three piles; is there enough +Buy/gaining to exploit a large turn? A board with Chapel + Village + Smithy + Market supports a different plan from a board with Smithy and little useful support.

## Openings and the first reshuffle

The initial two hands normally produce a 4/3 or 3/4 split, or a 5/2 or 2/5 split. This follows from seven Coppers distributed over two five-card hands. The *order* matters for timing and occasionally effects: two purchases usually first become drawable in the first reshuffle, but early effects can change that. Do not hard-code identical openings for every kingdom. [S1]

| Available tools / money | Practical opening prior | Why and caveat | Owner |
|---|---|---|---|
| Chapel, 4/3 | Chapel + Silver is a strong default | Early removal of Estates/Coppers improves future shuffles. A powerful attack or special synergy can change the companion buy. | Astra chooses; code executes legal buys |
| Chapel + Witch, 5/2 | Witch + Chapel is an attractive default | Early cursing plus thinning; not a universal theorem, assess pace/support. | Astra |
| Witch, 5/2 without Chapel | Witch on $5; consider useful $2 support or pass | Early Curse delivery is often decisive. Buying a weak $2 card just to spend money can hurt. | Astra |
| Militia, 4/3 | Militia + Silver is a solid attack/economy default | $2 payload with hand-size attack. Competes with Smithy and other terminals for the one starting Action. | Astra |
| Smithy money plan, 4/3 | Smithy + Silver | One early terminal draw card can improve money density; do not buy many Smithies without actions/cycling support. | Astra; explicit baseline in code |
| No useful action plan | Silver + Silver on 4/3 | Reliable economy; $5 opening often still buys Silver if no worthwhile $5 card. | Code baseline; Astra can improve |
| 5/2 with Sentry | Consider Sentry on $5 | Cantrip thinning/deck control; strength depends on support and competing attacks. | Astra |
| Any opening | Avoid an early Estate or speculative Village with no terminal demand | Unproductive cards slow cycling; cheap does not mean useful. | Astra; optional Jev flags |

A legal $5 hand need not spend $5. Likewise, no-buy is often correct on $2. Initial shuffle luck is not a justification for changing starting decks or supplying extra money. [S4–S8]

## Thinning: remove bad cards while preserving function

Chapel can trash **up to four** hand cards. Early priorities are typically Curses, Estates, then surplus Coppers, but preservation of buying power matters. Trashing all Coppers without a way to gain or buy useful economy can strand a deck. A Chapel opening aims to obtain stronger economy and repeatedly remove junk; trashing four weak cards can be worth missing a purchase when the deck will recover. The exact choice depends on remaining hand, deck composition and next shuffle—not a fixed "trash four" rule. Late in the game, keep needed VP if there is no time for thinning to repay its cost. [S4]

Sentry draws a card and gives +1 Action, then examines the top two cards for trash/discard/keep with ordering choices; it can improve a deck without using up the action chain. Remodel converts a trashed card into one costing **up to $2 more**; Mine optionally upgrades a Treasure to one costing **up to $3 more**, gained to hand. They are not identical to Chapel and are not mandatory upgrades whenever available. Removing Estates via Remodel to useful $4 cards can build; Gold-to-Province Remodel can score or finish; action cost and lost economy must be weighed. [S8; card pages via S12]

**Code:** enumerate valid trash sets/upgrade targets, calculate resulting composition, track known cards and prices. **Astra:** choose a target deck and how far to thin; adapt the priority near the finish. **Jev:** given the current plan and already computed resource-survival features, score legal trash candidates as keep/low/medium/high priority. Jev must not infer exact affordability after trashing from raw card names.

Common failure: treating all Coppers as permanently bad while still needing them, keeping every Estate until they clog repeated shuffles, or using a trashing action after the last useful reshuffle. Thin for future useful hands, not for its own sake.

## Draw, actions, and payload: build something that runs

A nonterminal draw card replaces itself without reducing available Actions; Village gives +1 Card, +2 Actions (net +1 Action when played). Smithy draws 3 but provides no Actions. A hand with Village then Smithy can continue playing Actions; Smithy first with only one Action can strand Village. Buying too many terminal Actions creates collisions where several useful cards share a hand but only one can be played. [S3, S5]

Prioritize enough action capacity to play the intended terminals, enough draw to find those cards, and enough payload to justify drawing. Market provides +1 Card, +1 Action, +1 Buy and $1; Festival provides +2 Actions, +1 Buy and $2 but no draw. An engine needs money and extra buys/gains to turn deck control into multiple Provinces or rapid pile depletion. Adding Gold is not automatically optimal when it displaces engine components or reliable Action-based money. [S3; S12]

Action ordering is situational:

- Play action-generating cards before terminal draw when necessary to avoid running out of Actions. Play cantrip filtering/trashing at a point that improves future draws. A universal fixed card-priority list is insufficient.
- Time Cellar discards to exchange dead cards for useful ones; do not discard a needed terminal merely because another action is presently better.
- Consider playing hand-reducing effects before Library, which draws until hand size 7 and can set aside Actions. Account for remaining Actions so the new hand is usable. [S13]
- Throne Room repeats a chosen Action; choose the target based on current resource constraints. Throne Room on Village can enable a chain; on Smithy gives substantial draw but requires a way to use drawn Actions; on Witch can deliver two attacks while Curses remain. Both resolutions can trigger distinct choices. [S9]
- Council Room draws 4 and adds a Buy, but also gives each other player a card. In Base setups, Council Room followed by Militia can make the extra enemy card disappear into the discard-to-three attack; the reverse order helps the opponent rebuild their hand. [S7, S14]
- Before an effect crosses a shuffle, consider what is currently in the discard. Future purchases usually enter discard after Action-phase draws; do not assume an upcoming buy will be in the imminent shuffle.

**Code:** exact sequencing legality, resource deltas, shuffle counts, optional search over legal visible effects. **Astra:** component ratios, acquisition order, nuanced sequencing and when to stop building. **Jev:** rank already legal Action plays under a plan such as "draw the deck while preserving action capacity" with resources provided; do not make it maintain the action stack.

Common failures: buying Villages without terminals, buying terminals without Villages, building draw with no money or gains, or building a complex engine whose first scoring turn happens after the opponent can end the game.

## Attacks and their counters

**Witch / Curse pressure.** Witch combines draw with junking opponents. Early Witches can matter more than an extra Gold because repeated Curses slow and reduce the opponent's score. Watch the Curse supply: when empty, Witch remains a terminal draw card but loses its junking payload; extra copies may no longer pay off. Responses include racing the attack, strong trashing such as Chapel/Sentry, drawing a resilient deck, or a fast ending. Moat blocks only when actually in the attacked hand and revealed; owning one does not grant permanent immunity. [S6, S4, S8]

**Militia / hand-size pressure.** Militia gives $2 and attacks opponents down to 3 cards. Preserve a hand that works: sometimes one Village + one draw card + payload is better than the three largest Treasures. With a simple money hand, keep the highest useful spending total; with multiple Actions but no Village, do not preserve unplayable terminals. Repeated Militia attacks against someone already at 3 usually add no further hand reduction. Library and robust draw can recover from small hands. [S7, S13]

**Bandit.** Gains Gold and attacks top-deck non-Copper Treasures according to its rules; Action-based money reduces vulnerability. Do not assume the attacker chooses which Treasure the victim trashes: follow the printed effect and engine prompt. Account for exposed cards without extrapolating the rest of the opponent's hidden deck. [S12]

**Code:** attack/reaction timing, legal victim selections, current Curse count, public counters. **Astra:** decide whether attacking, defending, thinning or racing best serves the kingdom plan. **Jev:** recognize "preserve engine starter" versus "preserve buying power" from supplied candidate hands; rank only legal candidate discard sets. Candidate generation and spend calculations belong in code.

Common failures: buying too much defense while falling behind economically, trying to block attacks with Moat in the discard pile, ignoring the Curse race, and paying for redundant attacks after their marginal effect vanishes.

## Greening and ending the game deliberately

Victory cards mostly do nothing in hand. Buy them when the points and ability to finish outweigh damage to future turns. Money decks often buy Provinces as soon as $8 is available; strong engines may delay single-Province purchases to improve eventual multiple-Province turns. This is a conditional decision, not "always buy Province" or "never green early." [S1–S3]

Always inspect remaining Provinces and **all** nearly empty Supply piles. Two empty piles plus an almost empty third can make a few cheap gains decisive. Gaining can end a game even without buying. Gardens/Workshop games can end through small piles before a Province race would finish. Each Gardens scores floor(total owned cards/10), so new cards can change the value of every owned Gardens at a threshold; code must compute it exactly. [S10, S11]

For each plausible ending line, code should calculate your resulting VP, opponent VP if publicly known, end condition, turn counts/tiebreak, and legality. Include Remodel, Workshop/Artisan gains, extra Buys and attacks in the line where applicable. A line may require nested decisions and must be recomputed if a draw reveals new information. It is not enough to compare "I bought 6 points" with the pre-turn lead.

If winning is primary, avoid an immediate legal ending that loses when a continuation offers a realistic chance. If both win and margin are evaluated, report both: do not sacrifice a certain win for speculative margin unless the evaluator explicitly defines such a preference. When losing with no recovery, sensible maximum-point play is a policy choice; do not implement a universal rule to prolong games forever.

**Code:** scoring arithmetic and bounded legal endgame enumeration; flag known forced wins/losses with evidence. **Astra:** choose timing under uncertainty, decide whether the engine can afford one more build turn, estimate opponent capability from public cards, revise plan when piles race. **Jev:** optional recognition of urgency from computed flags; never ask Jev to count scores or decide the tiebreak.

Common failures: automatic Duchies at an arbitrary turn number, greening until an engine stalls, buying the last Province while behind, missing a three-pile win, forgetting that the whole turn completes before the game ends, or treating VP ties as automatic draws.

## Concrete reproducible scripted opponents

These are **benchmark specifications proposed here**, not claims about a library's existing bot or optimal human play. Implement them only in the adaptor phase. Use the same rule legality and fair observations as the agent.

### BM-1: simple Big Money

No Kingdom-card purchases. Play all ordinary Treasures. On each buy opportunity, choose the first affordable, nonempty pile in this priority list:

1. Province.
2. Duchy if Provinces remaining is at most 4.
3. Estate if Provinces remaining is at most 2.
4. Gold.
5. Silver.
6. Otherwise stop; do not buy Copper or Curse.

This policy intentionally has crude endgame thresholds and can make losing endings. It is an easy, transparent comparator, not a claim of solved Big Money. On ordinary initial hands it buys two Silvers on 4/3; on 5/2 it buys Silver and passes. Handle forced choices legally: discard to retain the highest Treasure money first, then the higher printed VP among otherwise unusable cards; resolve equivalent choices by stable ID. When Bandit offers a choice of eligible revealed Treasures to trash, trash the one producing less money (Silver before Gold), with stable-ID ties. Reveal Moat if somehow present and blocking has no specified disadvantage. This baseline normally acquires no Actions or reactions in the initial scope. [Concept: S2]

### SM-1: one-Smithy money

Only use this variant when Smithy is in the kingdom. Same buys as BM-1 except, **after the Province and Duchy/Estate endgame tests, before Gold/Silver**, buy a Smithy if none is owned, at least 5 Provinces remain, and it is affordable. Thereafter buy no further Smithies. Play Smithy when able before Treasures. On a forced discard, if retaining one Smithy leaves it playable, keep one Smithy then fill the remaining slots with highest-value Treasures; otherwise use BM-1 discards. This is a precisely defined moderate baseline, not guaranteed stronger on every kingdom. Test it rather than assigning a win-rate claim. [Concept: S5]

### WM-1: one-Witch money

When Witch exists, use the same baseline with an early Witch acquisition replacing SM-1's Smithy rule: own at most one, buy it ahead of Gold/Silver only while Curses remain and at least 5 Provinces remain; retain/play it as the preferred terminal. Once Curses run out it still draws, so keep playing it when legal. This adds meaningful attack pressure but is not a general engine bot. Validate reactions and Curse allocation before adding it to evaluation. [Concept: S6]

For a stronger opponent, use an upstream action/engine bot **only after reading its policy**, or create a separately tested kingdom-specific engine policy. Name and version it, publish its rules, and compare actual results. Randomly selecting legal moves is useful for stress testing, not as the only measure of playing strength.

## Suggested scenario families (10 distinct Kingdom piles each)

These are proposed fixtures, not claims to reproduce publisher-recommended sets. Each card must first pass the engine's support tests. Use ordinary two-player counts and starting decks.

| ID | Ten piles | Purpose |
|---|---|---|
| `base_intro` | Cellar, Market, Merchant, Militia, Mine, Moat, Remodel, Smithy, Village, Workshop | Straightforward economy, action balance, attacks and reactions |
| `chapel_engine` | Chapel, Village, Smithy, Market, Festival, Laboratory, Militia, Moat, Remodel, Merchant | Thin/draw/payload planning, engine versus money |
| `curse_race` | Witch, Chapel, Sentry, Village, Smithy, Market, Moat, Militia, Festival, Laboratory | Early attacks, thinning and engine resilience |
| `gardens_rush` | Gardens, Workshop, Cellar, Village, Smithy, Market, Militia, Moat, Remodel, Merchant | Gains, deck-size scoring and three-pile endings |

Use `base_intro` for the first complete adaptor. Only expand to all supported Base cards after covering special prompts (Sentry, Vassal, Library, Bandit, Throne Room, Artisan, etc.). Do not replace unsupported cards randomly inside a named scenario: fail reset clearly or publish a new scenario ID. Include both seats and varied shuffle seeds for every fixture; hold out entire combinations in addition to seeds. Report per-opponent results so a BM exploit is not confused with general competence.

## Optional Jev typed questions

Inputs should include the current legal option list, Astra's plan, exact derived counts/resources, card descriptions, and ending flags. Candidate generation belongs to code. Examples of **independent questions at a frozen state**:

| Question | Typed output | Intended use |
|---|---|---|
| Which current bottleneck is most evident given these computed features? | choice: draw / actions / payload / scoring urgency / none | Feature suggestion to Astra, not an autonomous plan |
| How well does each legal acquisition support the stated plan? | one ordinal score per option: 0 / 1 / 2 / 3 | Cheap candidate ranking; Astra handles strategic exceptions |
| Which of these prevalidated discard sets best preserves the stated next-turn plan? | one candidate ID | Local choice; code has already checked size and supplied playable-card/spend features |
| Is this supplied pattern a terminal collision? | yes / no | Recognition only; code still enforces Actions |
| Which of these legal Action plays most supports “draw while retaining action capacity”? | one option ID | Local ordering suggestion, validated before execution |

Do not ask "How much money is in hand?", "Who wins if three piles empty?", "What is the legal Remodel gain limit?", or "What should we buy for the next five turns?" of Jev. The first three belong in code, the fourth in Astra. Jev failures must not affect rules legality or reveal hidden state.

## Community references

All links below are unverified live in this session; use the official rulebook for rules conflicts.

- [S1] Strategy overview: https://wiki.dominionstrategy.com/index.php/Strategy
- [S2] Big Money: https://wiki.dominionstrategy.com/index.php/Big_Money
- [S3] Engine construction: https://wiki.dominionstrategy.com/index.php/Engine
- [S4] Chapel: https://wiki.dominionstrategy.com/index.php/Chapel
- [S5] Smithy: https://wiki.dominionstrategy.com/index.php/Smithy
- [S6] Witch: https://wiki.dominionstrategy.com/index.php/Witch
- [S7] Militia: https://wiki.dominionstrategy.com/index.php/Militia
- [S8] Sentry: https://wiki.dominionstrategy.com/index.php/Sentry
- [S9] Throne Room: https://wiki.dominionstrategy.com/index.php/Throne_Room
- [S10] Gardens: https://wiki.dominionstrategy.com/index.php/Gardens
- [S11] Workshop: https://wiki.dominionstrategy.com/index.php/Workshop
- [S12] Card index and individual card FAQs: https://wiki.dominionstrategy.com/index.php/Card_list
- [S13] Library: https://wiki.dominionstrategy.com/index.php/Library
- [S14] Council Room: https://wiki.dominionstrategy.com/index.php/Council_Room
