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

The example paths are arithmetic, not promises about development draws. Five settlement pieces and four city pieces matter: a city returns a settlement piece, so expansion may require an upgrade first. Development VP cards are limited to five in the standard deck. Never build a strategy that assumes drawing a specific hidden card next. [Rules: R1/R2; strategic references: S1/S2.]

**Without domestic trading**, lower the value of producing a scarce resource purely to sell it to opponents. Raise the value of self-sufficiency, access to ore/grain, and a realistic harbor route; missing grain is especially painful because settlements, cities, and development cards all need it. A missing resource with only 4:1 access may make an apparently rich opening very slow. Do not prompt Astra to negotiate with built-in bots unless the pinned engine actually supports trade offers/responses. **Code:** engine capability flag and maritime deficit calculation. **Astra:** choose a feasible economy under that flag. **Jev:** recognize “missing critical resource without efficient conversion.” [C1 for capability verification; S1/S3 for strategy follow-up.]

## Placement as a paired, seat-dependent decision

The first and second settlements form one economy. In four-player setup, seat A picks first and last while seat D picks fourth and fifth consecutively. Early seats should value scarce strong ore/grain that may disappear and consider several feasible complements; the last seat can evaluate pairs directly. After each opponent placement, remove its vertex and neighboring vertices from candidates and recompute reachable expansion options. Do not lock onto a second site that another player can take before the return pick. Initial settlements need not connect to each other. [R1; strategic references S1/S2.]

High pip count is a baseline, not a complete value function. Two identical number tokens have the same expected total production as equivalent different-number weights, but concentrate gains into fewer rolls and can create hand overflow; a spread of numbers lowers some variance. Diversity is not a reason to reject a much higher expected yield. A port can bridge one missing resource, but a 2:1 port needs actual surplus of its specific resource and an owned coastal settlement; merely building a road beside it does nothing. Initial road direction should aim toward a vertex that remains legal after the distance rule and opponents' likely routes. Expansion at a distance of two edges costs two roads **plus** the settlement, not merely the settlement's four cards. [R1; S1/S2.]

**Code:** rank pairs with expected production by resource, starting hand from the second settlement, expected build deficits, legal paths, occupied/forbidden vertices, and port conversion. **Astra:** decide how much short-term tempo to trade for a durable economy and model contested return picks. **Jev:** classify supplied pairs by plan fit and route vulnerability; do not compute pip sums or shortest paths.

## Quantitative features and tactical ordering

Use `E[resource r per roll] = sum(building_multiplier × token_ways / 36)` over productive adjacent hexes of resource `r`, with multiplier 1 for a settlement and 2 for a city. Mark currently robbed production separately rather than permanently deleting it from long-term value. For a 6-ore/9-grain/3-wool settlement, the rates are 5/36 ore, 4/36 grain, and 2/36 wool per roll; the total is 11/36. A city upgrade adds another 11/36 expected cards per roll at that site, not a new independent number distribution. These are direct probability derivations, not sourced empirical performance estimates.

A deficit vector for each target build and the shortest maritime-trade sequence is more useful than total hand size. As a rough bottleneck estimate, code may use the largest `missing resource / expected income` ratio, but this ignores discrete dice, correlated production, robbers, and trading; do not call it an exact expected time. Prefer small legal-action searches or sampled rollouts for finish sequences. Ports must be applied to actual same-resource holdings and the remaining bank supply, not to a fractional expected hand. [R1 rules; study implementation recommendation.]

If there are four future rolls before spending a vulnerable hand, the probability of at least one 7 is `1 − (5/6)^4 ≈ 51.8%`. This is not the probability of losing a specific plan: hand size may change before the seven, and the active player may spend earlier. Spending down to seven now does not immunize a hand from later production. Keep rare bottleneck cards and discard redundant surplus when forced; do not preserve a five-card city plan while throwing away a guaranteed immediate win. [R1; probability derived here.]

Recommended within-turn priorities are: execute a guaranteed win; handle mandatory prompts; prevent an imminent rival win if possible; resolve contested useful builds; choose the best economical build or development purchase; trade only to enable a concrete objective; end the turn when no action improves the plan enough. A maritime trade loop should not consume all surplus just because a legal trade exists. Code should bound search, detect dominated/inverse trades, and avoid repeated observation/action loops. Astra resolves strategic tradeoffs; Jev ranks remaining candidates by the supplied objective. [Study recommendation; S1/S3 for contextual strategy.]

## Development cards, discards, and hidden information

| Tactical situation | Strong default and counter | Code / Astra / Jev |
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
