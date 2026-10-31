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
