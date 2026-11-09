Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Settlers of Catan (Catanatron)` — Standard four-player base-game Settlers of Catan, played to 10 true victory points. The controlled seat acts against Catanatron Random, WeightedRandom, and VictoryPoint built-in bots. The randomized standard island, tokens, ports, dice, development deck, and seat order are seeded by reset(seed). Each step applies exactly one legal action and then advances opponents until your next decision. Read legal_actions from the observation and submit its id (a0, a1, ...); ids expire after each step. The engine handles initial placement, rolls, production, seven/discards, robber theft, building, maritime trade, development cards, awards, and hidden information. Your resource and development identities are shown; opponent hands are represented only by counts. A hard cap reports termination='truncated' and is not a win.

Actions:
- roll: Roll dice (once per normal turn).
- end_turn: Finish the current turn.
- build_road: Build on an edge; use the legal action id or `build_road U V`.
- build_settlement: Build at a vertex; use the legal action id or `build_settlement N`.
- build_city: Upgrade your settlement; use the legal action id or `build_city N`.
- buy_development_card: Buy one development card.
- move_robber: Move the robber; use the legal action id (victim is engine validated).
- discard: Discard the engine-selected cards after a 7.
- play_knight_card: Play a Knight and move the robber.
- play_year_of_plenty: Take one or two resources; use the legal action id.
- play_monopoly: Name a resource to monopolize; use the legal action id.
- play_road_building: Play Road Building, then choose its roads.
- maritime_trade: Trade with the bank; use the legal action id.
Use legal_actions[].id from the observation. IDs are valid only for the current observation. The engine uses standard base-game rules and randomizes the standard island per seed.

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
- strategy_summary: Rank opening intersections by dice-pip production, resource diversity, grain/ore access, expansion routes, and useful ports; use the second settlement to complete a coherent build plan. Expand to productive legal vertices, city strong production, and buy development cards when they do not delay a winning build; pursue Longest Road or Largest Army only when the race changes the win path. Track resources, card age, pieces, awards, hidden VP, and immediate wins in code, while Astra plans several turns and adapts to opponent threats. Use Jev for fast rankings and yes/no tactical checks over a precomputed legal candidate list. Rob the leader's key production and protect a near-term win, but let the engine determine legality and exact award computation.
- scoring: Report terminal true victory points for the agent, win (1/0), winner, public VP, turns, action count, and termination reason. The primary benchmark is win rate against the specified built-in bots; mean terminal true VP is secondary. Include hidden VP cards and award points in true VP, and mark hard-cap episodes as truncated rather than wins.
- interface.observation: A JSON player view containing phase, current player, turn, pending decision, board topology/terrain/numbers/robber/buildings/roads/ports, the agent's resources and development-card identities, opponent resource and development-card counts, played Knights, pieces, public VP, awards, bank counts, last events, dice result, legal_actions, and terminal status. Hidden opponent cards and unplayed deck order must be filtered.
- interface.actions: Return one ID from the current legal list, mapped back to an engine action object. Depending on verified capabilities, decisions include initial placement, roll, discard, combined or separate robber/victim choices, maritime or supported domestic trade, builds, development purchase/play, and end turn. Randomized discards or bundled choices must retain the engine's actual granularity. Apply one action and advance bots/chance only until the next controlled-player decision or termination.

### study/strategies.md
# Strong CATAN play: rules of thumb and model responsibilities

The rules reference is the official CATAN rulebook/FAQ ([R1](https://www.catan.com/understand-catan/game-rules), [R2](https://www.catan.com/faq)). Community references for further verification are [S1](https://en.wikibooks.org/wiki/Settlers_of_Catan/Strategy), [S2](https://boardgamegeek.com/boardgame/13/catan/forums/67), and [S3](https://blog.colonist.io/). **None of these pages could be retrieved in this session because DNS failed.** The heuristics here are prior game knowledge and study recommendations, not quotations or claims about having read those pages. Treat them as a starting prior, not a guarantee under every map, bot, or house rule. The engine's legal-action list and award computation always win over a heuristic. Inspect [Catanatron](https://github.com/bcollazo/catanatron) for actual supported rules, especially domestic trading and discards.

## What to optimize

The objective is to be the first player to reach 10 true victory points on their own turn. Optimize win probability first, then expected terminal VP and turn length. Track hidden victory-point cards and award ownership explicitly; visible settlement/city points alone can mis-rank players. A settlement is one VP, a city adds one, and Largest Army and Longest Road are two each ([R1], [R2]).

For production, use dice ways out of 36 as weights: 6/8 = 5, 5/9 = 4, 4/10 = 3, 3/11 = 2, 2/12 = 1. A useful intersection score is the sum of its adjacent token weights, adjusted for robber exposure, resource scarcity, and whether the resources combine into a build. This is bookkeeping/code. Astra should reason about race position and opponents; Jev can make a contextual ranking from precomputed production and route features. Code, not Jev, finds the numerical maximum.

## Opening placement

1. During the first snake placement, rank legal intersections by expected production, then break ties using resource diversity, expansion direction, port value, and access to grain/ore. High-pip grain plus ore supports cities and development cards; wood plus brick supports the first settlement and road. A single spectacular number with no buildable complement is often worse than two good numbers with useful diversity.
2. Use the second placement to cover a missing build resource and claim a plausible expansion route. Prefer a second port only when the hand/resource plan can exploit it; do not sacrifice all production for a speculative 2:1 port. The second settlement's adjacent resources are the starting hand, so its immediate resources and the road's direction matter ([R1], [S1]).
3. Preserve at least one realistic route to the next settlement. A road that can be cut by an opponent or reaches no legal vertex is a poor opening spend. The adaptor should enumerate legal vertices/edges and score them; Astra should choose among the top candidates with the current opponent order in mind.

Code should calculate pips, legal distance/connectivity, resource costs, piece availability, ports, and shortest routes. Astra should select a coherent opening plan and revise it after seeing opponents. Jev should make fast pairwise or three-way rankings from a supplied candidate list, never generate coordinates or do unbounded search.

## Early and middle game build order

* Expand when the settlement cost is reachable and an uncontested high-value vertex exists. Settlements add a production source and VP; roads are primarily an investment in a settlement or a forced Longest Road race.
* Upgrade a productive settlement to a city when the extra production and VP are likely to arrive before an expansion, especially with strong ore/grain. Citying a weak or frequently robbed hex is less urgent.
* Buy development cards when ore/wool/grain are abundant, a city/settlement is temporarily blocked, or a Knight race and hidden VP can finish the game. Do not buy a card while it prevents an imminent winning city/settlement unless the card timing is decisive. A newly bought non-VP card cannot normally be played that turn ([R1], [R2]); code must track card age.
* Keep enough cards to act while avoiding a large exposed hand before a likely 7. On a 7, discard half (rounded down) when holding more than seven resources; development cards do not count. The exact discard and robber choices are tactical and should be legal-action questions.
* Treat domestic trades as negotiation and information. Accept a trade when it creates a near-term build or blocks an opponent's winning turn; do not hand the leader the exact card needed for a cheap finish. Maritime rates are deterministic and should be computed in code.

The code layer should maintain resource ledgers, card age, piece counts, build feasibility, port rates, road graph, settlement distance, award status, and one-development-card-per-turn legality. It should answer exact questions such as whether a build is affordable or an action wins immediately. Astra should plan two or three turns around likely production and opponent threats. Jev can classify which legal build best fits a supplied plan or whether a trade advances a particular strategic goal, using the affordability and win flags supplied by code.

## Awards and tactical races

Longest Road needs a continuous route of at least five roads and can be broken by an opponent settlement; branches and repeated edges must not be counted as a simple edge total. Largest Army starts at three played Knights, and an incumbent keeps the award on a tie ([R1], [R2]). Use the engine's award values and graph calculation rather than an LLM estimate.

Pursue Longest Road when the route is already close, the needed roads are cheap, and the opponent cannot cheaply interrupt it. Pursue Largest Army when Knights are available and the award can be secured before a rival's next turn. If an award race is too expensive, switch to cities, settlements, and hidden VP. A tactical road to take the award can be correct even when it produces no immediate board production, but only if it changes the win race.

When rolling a 7 or playing a Knight, place the robber on the leading opponent's highest expected-production hex, subject to blocking the opponent who can win soon and preserving one's own production. Prefer a legal victim with useful hand size when choosing whom to steal; the stolen resource is random. Avoid a retaliatory block that leaves the real leader untouched. Astra should explain the target and expected threat; Jev can choose among a short list of legal robber hex/victim pairs using supplied opponent scores and production.

## Endgame and counters

Count every path to 10: visible build, award transfer, a bought/revealed VP card, or a combination. Before ending a turn, search for a legal immediate win and protect the resource cards needed to execute it. If an opponent can win next turn, prioritize blocking their route, taking an award, robbing their key number, or making a trade that denies the needed resource; do not maximize personal expected production blindly.

Common mistakes are overvaluing a single 6/8, ignoring grain, building disconnected roads, racing Longest Road without a settlement endpoint, buying development cards that delay a guaranteed city, forgetting a newly bought card cannot be played, exposing eight-plus resources before a 7, and assuming a number is “due.” Dice are independent; use probabilities rather than streak narratives. These are prompt reminders for Astra and binary/choice checks for Jev, while exact legality and arithmetic remain code.

## Suggested harness division

**Code:** legal-action filtering; board graph and settlement distance; pip and expected-production features; resource/card/piece ledgers; dice probabilities; trade rates; award and VP bookkeeping; card-age and per-turn limits; terminal/win checks; deterministic seeds; opponent-visible information filtering; immediate-win enumeration.

**Astra prompts:** choose and explain an opening plan; weigh production versus diversity/ports; plan a route over several turns; adapt to robber placement, blocked intersections, revealed cards, and opponent win threats; decide whether an award race is worth its resource cost; negotiate or refuse a trade using the public race state.

**Jev typed questions:** rank a provided list of legal placements/builds/robber targets; classify whether a trade fits the current plan; choose discard resources from a reduced legal list while preserving a specified goal; identify the strongest strategic threat using supplied numerical features; answer yes/no on whether a legal Knight play, development purchase, or award contest advances Astra's plan. Jev receives precomputed candidates and arithmetic; it should not invent moves, infer hidden cards from engine internals, or plan a whole game. Immediate wins, numerical leaders, and affordability are exact code checks.

Test these priors against the selected Catanatron bots on fixed and held-out seeds. Record win rate, true VP, award transfers, and truncations by opponent mix; community heuristics can be wrong on maps with unusual ports or resource scarcity.

## Strategy families and explicit win paths

These are strategic hypotheses for evaluation, with [S1], [S2], and [S3] as community verification references. A fixed build order is unreliable because dice and competing placements change available resources.

| Plan | Opening and progression | Example route to 10 VP | Counter / failure condition | Owner |
|---|---|---|---|---|
| Ore–wheat–sheep (ore/grain/wool) | Prioritize strong ore and grain, enough wool, then cities and development cards; acquire one expansion via roads or port trades | 3 cities (6) + Largest Army (2) + 2 VP cards (2) | Heavy robber on one grain hex; poor ore access; a faster Knight rival; no third settlement route | Code computes production/card timing; Astra selects plan; Jev labels fit |
| Expansion / road economy | Strong lumber/brick plus grain/wool; settle useful intersections before upgrading or contesting road | 3 cities (6) + 2 settlements (2) + Longest Road (2) | Roads without reachable settlement space; city resources absent; road award easily stolen/cut | Code routes and pieces; Astra race choice; Jev ranks shortlisted routes |
| Port economy | Strong repeated resource plus accessible matching 2:1 port, or diversified surplus plus 3:1 | Use efficient trades to fund either plan above | Port sacrifices production; robber hits main resource; insufficient surplus; resource already needed for builds | Code exact conversions; Astra values dependency risk; Jev identifies overconcentration |
| Balanced flexible economy | Reliable coverage of all five resources, productive initial builds; adapt to draws and opponent races | Mixture of cities, settlements, an award, and hidden VP | Chasing equal resource counts instead of useful production; weaker total yield than specialized rivals | Code compares build deficits; Astra adapts; Jev recognizes bottlenecks |

…(truncated; full text in prior/strategies.md)

|---|---|---|
| Old Knight and robber on own high-yield hex | If pre-roll play is supported, remove it before rolling when production value outweighs saving the Knight; do not miss an immediate Largest Army win | Code checks timing, counts, award and legal targets; Astra weighs urgency; Jev recognizes “unblock key economy” |
| Road Building | Aim for a settlement route, a contested junction, or a decisive award; it supplies roads, not the settlement cards | Code checks two sequential road placements and remaining pieces; Astra route choice; Jev route-fit label |
| Year of Plenty | Take the precise missing cards for a winning or high-value immediate build, considering bank stock and port trades | Code enumerates resource pairs and build outcomes; Astra selects objective; Jev optional contextual rank |
| Monopoly | Use public production/trade history and uncertain hand beliefs to target a resource that completes a build or yields a large haul; waiting may improve yield but risks a rival finish | Code maintains estimates/bounds, never actual hidden hands; Astra selects timing and target; Jev recognizes recent production concentration |
| Development purchase | At the start of a fresh standard deck, Knights are 14/25 and VP cards 5/25; later probabilities depend on observed plays and one's own draws, with opponent holdings unknown | Code tracks publicly justified card beliefs; Astra values option/army potential; Jev optional plan fit |
| Forced discard with choice | Preserve bottleneck cards and an achievable near-term build; discard plentiful port-convertible surplus only after comparing its trade value | Code enumerates or shortlists legal discard multisets; Astra states goal; Jev selects among legal summaries |

Rules/card composition: [R1/R2]. Timing and strategic uses: prior knowledge, with [S1/S2/S3] for follow-up. If discards are random in the simulator, no discard policy is available; adapt hand-risk management instead. Only one non-VP development card is normally playable per turn: buying several Knights cannot immediately take Army. Road Building does not bypass the distance rule or opponent buildings. Victory-point cards count immediately toward a legal win and do not consume the normal played-development-card limit.

Do not treat hidden VP as zero when assessing opponents: a player with seven public VP and several old unplayed development cards may be close to winning. Equally, do not pretend those cards are known VP. Use conservative threat bounds and qualitative probabilities. Public production can support probabilistic resource tracking, but third-party theft and private discards introduce uncertainty. A larger hand does not by itself imply a higher chance of stealing the particular desired resource; code should estimate its proportion if history supports an estimate. [R1 information rules; study tactical recommendation.]

## Counters and opponent adaptation

Against ore/grain city builders, rob a critical grain/ore tile and deny their few reachable expansion sites when doing so also helps one's own plan. Against expansion players, secure contested vertices before their roads arrive and evaluate whether a legal settlement can break their award route. Against port specialization, block the high-yield input hex rather than an irrelevant scarce resource. Against an Army race, count played Knights and turn timing: catching up to an incumbent's count is insufficient. Against a road race, price the full cost of surpassing the rival, future road-piece limits, and cheap cuts; decline a costly bidding war if another path to 10 is faster. These are **Astra** decisions supported by **code** route/race features; **Jev** can identify the opponent's visible archetype from a provided set. [R1/R2 rules; S1/S2/S3 strategy references.]

Do not assume built-in bots bargain, retaliate, notice leaders, or implement human strategy. Measure each bot's robber targets, development use, and expansion tendency from public events, then adapt cautiously. Random bots are a smoke-test baseline, not evidence of competitive strength. Evaluate stronger included heuristic/search bots if available without heavy external model downloads. The same non-cheating observation boundary must hold during any policy search or model prompt. [C1 for actual bot implementations; study evaluation recommendation.]

## Compact prompt and typed-question contract

At each substantive decision Astra should receive: legal action IDs, our exact hand and playable old cards, normalized map/topology, production and deficit features computed in code, public opponents and threat estimates, verified engine capabilities, and the persistent plan. Ask for a **legal action ID** plus a short private plan update (target build/site, missing resources, award route, main opponent threat, abandonment condition). Replan after contested-site loss, major robber disruption, award transfer, or a new immediate-win threat; avoid rewriting the full plan on every forced roll. [Study design recommendation.]

Examples of Jev tasks are a choice among candidate IDs for “best matches the expansion plan,” a yes/no “does this route appear exposed to the listed rival settlement targets?”, and a categorical score `poor / acceptable / strong` for port fit using supplied surplus/rate features. Batch independent candidate questions in one call if used. These judgments must not replace exact legal/win checks and cannot authorize a move absent from the current legal list. Jev is optional for Tier 1; benchmark whether it improves latency/cost or merely adds noise. [Study design recommendation.]

### study/brief.md
# CATAN (Settlers of Catan): game brief

Scope: standard base-game CATAN, preferably four players, on a randomized standard island, played locally against simulator bots. No expansions, online account, GUI, or human turn timer. **Tier 1 is correct:** the simulator waits for a decision; thinking does not advance dice, opponents, or game time. Several seconds to a minute per Astra call is acceptable. Runtime quotas imposed by an evaluator are separate from game rules.

Evidence: official rule references are https://www.catan.com/understand-catan/game-rules and https://www.catan.com/faq. Network/DNS failed during this study; these sources and the simulator source could not be retrieved. Established tabletop rules below are the intended baseline. The selected simulator's exact fidelity must be checked as specified in `interfaces.md`. See `sources.md` for the evidence ledger.

## Objective and score

Be the first player with **at least 10 victory points on their own turn**. A settlement is 1 VP; a city is 2 VP total (upgrading adds 1); Longest Road and Largest Army each give 2 VP; each victory-point development card gives 1 hidden VP. Tabletop hidden VP cards are revealed to establish a win. They are not ordinary development-card actions. A player can reach a total above 10 with one award change. The engine's winner check must determine termination; a visible-score threshold alone is insufficient. [R1, R2]

Report terminal true VP as `score`, also `win` (0/1), winner, public VP, turns, and termination reason. Evaluate policy strength primarily by win rate, with mean terminal VP as a secondary measure. True VP includes the agent's hidden VP, not a sum over everyone's points. A hard episode cap is truncation, not a draw or an earned victory. Report current VP on truncation and mark it explicitly. Further details are in `interfaces.md`.

## Board, information, and setup

- Standard island: 19 land hexes (4 forest/lumber, 4 pasture/wool, 4 fields/grain, 3 hills/brick, 3 mountains/ore, 1 desert). Non-desert hexes carry number tokens. The robber starts in the desert. Terrain produces at adjacent buildings when its number is rolled unless the robber blocks it.
- Nine coastal harbors give 3:1 generic trades or 2:1 trades for one resource. Standard tokens have one each of 2 and 12 and two each of 3, 4, 5, 6, 8, 9, 10, 11. A simulator may use its own map randomization constraints; identify them rather than assuming arbitrary permutations are legal.
- Each player has 5 settlement pieces, 4 city pieces, and 15 roads. The base bank has 19 of each resource. The 25-card development deck contains 14 Knights, 5 Victory Points, 2 Road Building, 2 Year of Plenty, and 2 Monopoly.
- Initial placements follow a snake order: A, B, C, D, D, C, B, A. Each placement consists of a free settlement and an adjacent free road. Receive one resource from each producing hex adjacent to **the second settlement only**. Then A takes the first normal turn. Starting resources matter greatly for the first expansion/build.
- Buildings must satisfy the distance rule: no settlement/city at an adjacent vertex. During ordinary play a new settlement also needs connection to one's own road; initial setup settlements do not. Opponent buildings can interrupt road connectivity. Cities replace one's own settlements, returning the settlement piece to supply.
- Public information includes topology, numbers, ports, buildings, roads, robber, resource hand sizes, development-card counts, played Knights, awards, and public score. Exact opponent resources, unplayed development identities, and the order of the deck are hidden. Our own resource and development cards are private information available to us. Simulator internals may reveal more; the adaptor must filter them. [R1, R2]

## A normal turn and decision interruptions

1. Roll two dice, normally once per turn. A previously held development card can be played before the roll under standard rules (for example, a Knight to unblock production); check whether the chosen engine implements that timing.
2. On a non-7, matching unblocked tiles produce for **all** adjacent owners: 1 per settlement, 2 per city. Multiple buildings can collect from the same tile. Production is subject to available bank supply; use the engine for shortage resolution.
3. On a 7 there is no production. Every player with more than 7 resource cards discards half, rounded down (8→4, 9→4). Development cards do not count. The rolling player moves the robber to a different hex and steals one random resource card from one eligible adjacent opponent. Discard prompts may belong to players other than the turn owner. The precise choice of victim/hex and treatment of empty hands should come from engine legality.
4. The active player may trade and build repeatedly, and play at most one eligible development card for the turn. Standard game text presents trade then build phases; the common combined trade/build variant allows interleaving. Determine the simulator's sequence rather than enforcing a tabletop edition's phrasing externally.
5. End the turn when finished. Opponent decisions and chance resolution occur only as the adaptor explicitly advances the engine. [R1, R2]

## Costs and available actions

| Action | Resource cost | Effect / constraints |
|---|---|---|
| Build road | 1 lumber + 1 brick | One legal connected edge; no direct VP. Road continuity and opponent intersections matter. |
| Build settlement | 1 lumber + 1 brick + 1 wool + 1 grain | +1 VP and production; obey connectivity, spacing, and remaining pieces. |
| Upgrade to city | 3 ore + 2 grain | +1 VP and doubles that building's production; returns a settlement piece. |
| Buy development card | 1 ore + 1 wool + 1 grain | Draw secretly from the deck; no choice of identity. |
| Maritime trade | Usually 4 of one resource for 1 other | An owned harbor can improve to 3:1 or 2:1; honor bank availability. |
| Domestic trade | Negotiated with active player involved | Tabletop feature; commonly absent from Catanatron versions. Do not assume it is available. |
| Play Knight | Previously held Knight | Move robber and steal; contributes toward Largest Army. Does not trigger seven-discards. |
| Play Road Building | Previously held card | Build up to two legal free roads, subject to pieces/routes; engine may implement sequential choices. |
| Play Year of Plenty | Previously held card | Take two bank resources, including two of one kind if available; bank shortages can constrain this. |
| Play Monopoly | Previously held card | Choose a resource and take all cards of that resource from opponents. |
| Discard | Required after 7 with >7 resources | Choose exact cards if the engine exposes the choice; some implementations randomize this. |
| Move robber / choose victim | Required after 7 or Knight | Destination must change; theft is random, not an agent choice of stolen resource. |
| Roll / end turn | No cost | Phase-specific control actions, often the only legal choices. |

A non-VP development card normally cannot be played on the turn it was purchased, and only one such card may be played per turn. VP cards may contribute immediately to a winning total, including on purchase. Newly bought and previously held cards must therefore be distinguished. [R1, R2]

Longest Road requires a continuous eligible route of at least 5 roads; branches do not simply add and edges cannot be counted twice in the same route. Largest Army starts at 3 played Knights. An existing holder is not displaced by a tie. Road interruptions and ties after an interruption have subtleties: use engine award computation, not an LLM estimate or an ad hoc edge count. [R1, R2]

## Dice and pace

| Total | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ways out of 36 | 1 | 2 | 3 | 4 | 5 | 6 | 5 | 4 | 3 | 2 | 1 |

Production is expected **per roll**, not per personal turn. In a four-player round there are normally four rolls. A 7 happens with probability 1/6; holding eight cards is a risk, not an immediate penalty. Rolls are independent: a number is never “due.”

There is no finite game-imposed decision deadline in the chosen local interface: the engine waits for the next legal action. `study.json` uses **-1.0** as a documented numeric sentinel meaning **unlimited game-imposed decision time**, since strict JSON has no infinity literal. It is not a measured negative duration. A recommended operational Astra call budget is 60 seconds; that is a harness choice, not a CATAN rule. If the consumer requires a nonnegative value, its adaptor must explicitly translate the sentinel into its own timeout configuration rather than mistake it for a game deadline. Forced legality and routine bookkeeping should remain in code. Jev is optional and is not needed for real-time survival.

State changes at discrete actions and chance events. One game turn can involve many decisions: robber/steal, several trades/builds, a development card, then end turn. “Tier 1, turn by turn” should not be misinterpreted as exactly one model action for an entire turn. Number of actions and turns varies; no runtime or full-game benchmark was possible here.

### adaptor/ADAPTOR.md
# Settlers of Catan adaptor

This Tier 1 adaptor runs Catanatron **3.2.1** in process. It starts no server and
needs no credentials or network. `reset(seed)` constructs a fresh standard
four-player game. Catanatron seeds Python's random generator while constructing
the randomized standard map, ports, number tokens, development deck and seat
order; the dice, robber/discard choices and built-in bot choices then use the
same episode stream. The controlled seat is the RED player object and its
realized position is reported in `info().seat_order`.

The three opponents are the installed Catanatron `RandomPlayer`,
`WeightedRandomPlayer`, and `VictoryPointPlayer`. The engine enforces the base
game rules, including initial snake placement, rolls, production, discards,
robber/victim choices, maritime trade, development cards, awards, and hidden
victory points. The adaptor exposes the controlled player's private resources
and development identities, while opponent hands are counts only. Every
observation has a revision-local `legal_actions` list. Submit its `id` (`a0`,
`a1`, ...) to `step`; stale, malformed, or illegal actions are reported in the
next observation and do not raise. The convenience action grammar in
`describe()` handles simple build actions, but legal IDs are authoritative.

`score` is the controlled player's terminal true VP (including hidden VP cards
and awards). `info()` records true/public VP, winner, seat order, turns, action
count, engine version, and termination reason. The default hard cap is 1,000
completed turns or 20,000 actions; reaching it is `truncated`, never a win.
`score_breakdown()` exposes VP, win, public VP, turns, and actions. `close()` is
idempotent and releases the in-process game.

The study's scenario groups are declared in `game.py`; train, held-out, dev,
and validation seeds are disjoint. Catanatron's engine RNG is global, so a
single game instance should be used for one episode and planning code must not
call the engine or consume its random stream on a copy of the live episode.
The adaptor has no finite in-game decision timeout (Tier 1).

### study/interfaces.md
# CATAN interfaces and adaptor plan

This document assumes the standard base game and a local, offline evaluator. The references are linked in [`sources.md`](sources.md). In particular, [Catanatron](https://github.com/bcollazo/catanatron) is the preferred candidate, but the session could not resolve DNS, so its current release, exact method names, and Python 3.12 wheel availability still have to be checked in the adaptor build. Pin the inspected commit/release; do not silently track `main`.

## Choice

Use the **Catanatron Python engine directly** (a bot-library/core-engine interface), with a small adaptor that emits a filtered JSON observation and maps a JSON action ID back to an engine-owned action. It is the best fit for a Python evaluator; JSettlers2 and Pioneers are also plausible free headless engines but require separate processes and protocol integration. It avoids browser timing and image recognition and lets the adaptor ask for legal actions instead of guessing them. A Gym/Gymnasium wrapper can be added after the direct engine works, but it may alter decision granularity (discarding, robber victim, and initial placement) and its package/version contract is unverified.

The exact Catanatron symbols are release-sensitive. Historical APIs expose a `Game`/state object, player colors, typed actions, a legal-action list, and a way to apply one action. The lookup guide below is prior knowledge, **not a verified API contract for an installed version**. Before writing the adaptor, obtain one pinned version and inspect its source, then run a smoke probe that (1) constructs a four-player game with a fixed seed, (2) serializes the initial state, (3) enumerates legal actions, (4) applies one legal action, and (5) reaches a terminal state against built-in bots. If a symbol differs, use the pinned source's API rather than adding a compatibility layer based on guesses.

## Candidate interfaces

| Interface | Observation/control | Maturity and latency | Legal/cost risk | Decision |
|---|---|---|---|---|
| **Catanatron core Python engine** ([C1](https://github.com/bcollazo/catanatron), [C2](https://catanatron.readthedocs.io/)) | In-process board/state objects and typed legal actions; bots and seeded games | Purpose-built bot engine; local calls should be fast but neither action nor bot latency was measured | Free/open-source candidate; verify the pinned license and base-game fidelity | **Choose** |
| Catanatron Gym/Gymnasium wrapper ([C1](https://github.com/bcollazo/catanatron)) | `reset`/`step` observations and action integers or dictionaries, depending on release | Convenient RL loop, but wrapper may flatten or hide multi-step decisions; local latency | Free candidate; Gym/Gymnasium and NumPy version compatibility must be checked | Secondary, only after core probe |
| JSettlers2 ([A1](https://github.com/jdmonin/JSettlers2)) | Java server/client protocol, robot players, text/network messages | Mature open-source implementation; process/protocol latency is low but integration is substantial | Free; check GPL obligations and whether robot protocol exposes private/public state correctly | Fallback if Catanatron cannot install |
| Pioneers ([A2](https://pioneers.sourceforge.net/)) | C client/server, computer opponents, GUI-oriented protocol | Older project; headless setup and current build tooling are uncertain | Free; license/build and rule-variant compatibility need checking | Fallback only |

…(truncated; full text in prior/interfaces.md)

Seeding must cover board/ports/tokens, shuffled seats, dice, shuffled development deck, theft, randomized discards, and randomized bot choices. Inspect whether the engine uses global Python/NumPy RNGs; isolate episodes and planning rollouts so hypothetical search cannot consume the live game's randomness. Replaying a fixed seed **and action sequence** should reproduce events. Different actions may consume random draws differently; identical seeds alone do not guarantee identical dice sequences across policies. For fair comparisons, rotate the controlled color through all four seats, store the realized seat order, and use a fixed train/held-out seed split with the same bot mix and caps. Keep map randomness inside the engine's valid generator; do not create illegal token placements manually. Vary opponent seeds and policies independently of seat rotation. Compare win rate with uncertainty (for example Wilson intervals), report mean true VP and truncation rate, and include legal-action rejection/timeout counts. Audit whether built-in bots use hidden information and disclose it if unavoidable.

There is no game clock in a direct engine call. Runtime latency is unmeasured; benchmark setup, typical action application, and opponent thinking separately on this server. Allow roughly 60 seconds per substantive Astra decision as an operational starting point, automatically handle forced singleton actions, and reuse the current plan for routine choices. This is not an in-game deadline. `study.json` uses `-1.0` as a documented numeric sentinel for **no finite game-imposed decision timeout**.


## Measured model latency (at archive creation, 2026-10-30 03:28)
- System-2 (Astra), one-word answer: 30 s
- System-2, an answer of ~879 output tokens: 97 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).