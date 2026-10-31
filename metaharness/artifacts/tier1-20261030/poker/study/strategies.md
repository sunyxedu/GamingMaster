# Strategy prior: heads-up no-limit Texas Hold'em

This guide targets the specified **offline, no-rake, fixed-opponent, chips-per-hand** benchmark. The primary proposed game is 50/100 blinds and 20,000 starting chips per player, reset every hand (200bb). It is not a tournament survival problem. Tier 1 allows Astra to decide every turn; Jev is optional. All numerical ranges and sizing menus below are starting heuristics, **not claimed solver outputs**.

Evidence boundary: local rules and interface sources were inspected. Network requests failed DNS, so the external references below are relevant established documentation, strategy resources, and research locations, **not freshly retrieved pages**. Their current contents and URLs could not be verified. Mathematical examples below are derivations with stated assumptions; no paid charts or unavailable pretrained poker agent are required.

## 1. Division of work

| Topic | Code: exact or repeatable work | Astra prompts: judgment and planning | Optional Jev questions |
|---|---|---|---|
| Rules and accounting | Legal actions, integer sizing, actor/position, pot, call price, effective stacks, payoff, best five cards | Choose among already legal candidates | None; never outsource legality or arithmetic |
| Preflop | Card-to-class mapping, weighted range tables, size/depth buckets, random mixing | Choose a range family and adapt to opening/defense tendencies | Choice: which supplied range family fits this situation? |
| Range inference | Card removal, combination counts, likelihood updates, public history counters | Interpret action sequences and select plausible value/bluff ranges | Choice: likely range is capped / strong / polarized / uncertain |
| Postflop | Made hand, draws, board features, equity estimates and uncertainty, SPR | Compare checking, thin value, protection, bluffing, and future streets | Choice: primary purpose of a proposed bet |
| Exploitation | Opportunity-based statistics and uncertainty | Adjust to stable opponent mistakes; decide how much evidence is enough | Choice: observed behavior best fits overfolding / overcalling / overaggression / unknown |
| Evaluation | Hidden deals, independent RNG streams, duplicate isolation, chip ledger | Review recurring costly decisions and revise hypotheses | Optional flags for inconsistent explanations; not a second poker engine |

Strong poker combines a reasonable default strategy with selective exploitation. Equilibrium-inspired play is a robustness prior, not a requirement to stay balanced against a script that folds too much. Research systems such as Libratus and DeepStack use abstraction, opponent-range reasoning and subgame solving; those results do not imply that a few prompt instructions solve unrestricted Hold'em. [S1, S2, S3]

## 2. Preflop: enter more pots than in full-ring poker

**Position matters on every hand.** The button/SB acts first preflop and last postflop. The BB acts last preflop and first postflop. Button initiative plus postflop position supports a wide opening range; waiting for premium hands pays away too many blinds. Use heads-up ranges, never nine-handed early-position charts. [S4]

For a simple deep-stack starting strategy, use button raises around **2–2.5bb total**, with a wide range (roughly 70–90% is an exploratory prior, not a target independent of opponent). Include all pairs, most aces and kings, suited hands, connected hands, and many offsuit hands with high-card value. The weakest disconnected offsuit hands can fold. A mixed limping strategy is also legitimate: it keeps weak hands cheap and can contain traps. Choose and document a coherent raise/limp framework instead of claiming limping is always wrong. Raise more frequently if BB overfolds; prefer smaller pots with marginal hands if BB attacks raises strongly; value-raise larger against a caller insensitive to size.

**BB defense must depend on price.** Against a button min-raise to 2bb, the BB calls 1bb into the existing 3bb pot: the immediate equity threshold is 25%. That does not mean every hand with 25% equity can profitably call: out-of-position realization, domination, and future bets matter. Against larger opens, tighten weak offsuit calls. Against limps, check weak hands for free, and raise a wider value range if the limper calls too much. Do not auto-fold a free check or surrender BB to every raise. [S4, S5]

A deep-stack 3-bet starting size is roughly **3.5–4.5 times the opening total** from the BB (e.g. to 9–11bb versus 2.5bb). The button's positional advantage makes tiny reraises unattractive. Use a more linear range (strongest hands plus good broadways) against opponents who call many 3-bets, and allow selected blocker bluffs against players who fold enough. Low suited aces are candidates because they remove premium ace combinations and can make nut flushes; they are not mandatory bluffs against a calling station. A 4-bet might begin around 2.2–2.5 times the 3-bet total, but effective stacks and range strength govern whether that is appropriate. Use code to enforce the actual legal target and show the resulting SPR.

**At 200bb, one pair is not an automatic stack-off.** Suited nut potential and position gain value; dominated offsuit aces and weak kings carry reverse implied odds. Small pairs and suited connectors need enough future payoff and suitable opponent behavior to justify calls; “I can hit a set” is not sufficient. At 20–40bb, speculative calls lose implied odds, value rises for high cards/pairs, and reshove decisions become more common. Push/fold charts for short stacks are not a 200bb strategy. [S4, S6]

Implementation ownership: **code** stores explicit hand classes and size/depth buckets; **Astra** chooses/tunes the range and exceptions; **Jev** may identify the applicable already-defined situation. Do not ask Jev to enumerate 169 classes or calculate call odds. Represent each range with combination weights: a pocket pair has 6 combinations, a suited nonpair 4, an offsuit nonpair 12 before card removal. A percentage of the 169 labels is not the same percentage of the 1,326 two-card combinations.

## 3. Think in ranges, update without false certainty

Start with a range conditional on position and preflop action, then update after checks, bets, calls and raises. A check does not prove weakness; a large bet does not prove a monster. Keep plausible traps, draws and bluffs when the opponent's policy permits them. Remove combinations containing hero cards or visible board cards exactly. [S1, S2, S7]

**Code** can maintain weights proportional to prior weight × probability of the observed action for that hand class and context. **Astra** selects the likelihood model or a few alternatives, such as tight value-heavy, balanced, and loose aggressive. Preserve uncertainty rather than assigning one exact opponent hand. If the benchmark explicitly supplies the frozen opponent policy, its action rules can inform these likelihoods; access to its private cards, actual RNG draw, or future deck is still forbidden.

Equity is expected showdown share, including half of ties in heads-up. Compute it against a specified weighted range, not only a uniformly random hand. An opponent who raised three streets usually has a different range from the original BB defense range. Uniform-hand equity is a useful diagnostic and cheap baseline-opponent input, but often overestimates a bluff-catcher's equity against strong action.

**Code** samples legal opponent cards and future boards from the visible information set, independently of the simulator's true deck. Use stable per-decision randomness and a fixed sample budget; reuse sampled worlds when comparing candidate actions where appropriate. Report sample count and uncertainty. For independent bounded outcomes, worst-case equity standard error is at most `0.5/sqrt(N)`; at 2,000 samples it is about 1.12 percentage points, before range-model error. Increase work near a consequential threshold, not merely to make a number look precise.

For river equity against a uniform opponent, only `C(45,2)=990` possible hole-card combinations remain; exact enumeration is practical with a suitable evaluator. Turn full enumeration has 1,035 possible opponent combinations and 44 river cards per combination; flop has 1,081 opponent combinations and 990 unordered turn/river runouts per combination. These counts ignore any extra publicly known dead cards. Weighted ranges can be smaller. Hidden actual burn cards must not be excluded by the policy, because the policy does not know them.

Do not run a perfect-information search using the opponent's actual hand and call it poker reasoning. Sampling hidden states and choosing a different future action in each world can also overvalue lines through **strategy fusion**: future decisions must depend only on information then observable. A proper search tracks information sets/ranges and an opponent response model. Training CFR for the whole full-size game is unnecessary for this initial scripted-opponent benchmark. [S1, S2, S3]

## 4. Pot odds, bluff thresholds, and what the formulas assume

All calculations belong in **code**; give Astra the results with explicit definitions. Let `P` be the pot **currently visible including the opponent's outstanding wager**, and `C` the additional chips needed to call. With no further betting, no rake and no unmatched excess, call equity threshold is `C/(P+C)`, and call EV relative to folding is `e*(P+C)-C`. Count already-invested chips as sunk. If unmatched excess is possible, first remove refundable chips and use the actual contestable pot.

Example: pot was 600, opponent bets 300, so current pot is 900 and calling costs 300. Required equity is `300/1200=25%`. Do not divide by 900, and do not add the opponent's 300 twice. Before the river, raw equity meeting this threshold is not sufficient if future bets deny realization. Implied odds can sometimes justify a call below the immediate threshold, but require a plausible payoff when improving; reverse implied odds work the other way. [S5]

For a **new bet** of `B` into an uncontested existing pot `P0`, a pure bluff that always loses when called has EV `F*P0-(1-F)*B`. Break-even fold frequency is `B/(P0+B)`. This is a one-bet simplification; semibluffs retain equity and future streets change the EV. Use the correct formula for raises, whose call and raise costs differ from an opening bet.

| New bet size | Pure-bluff break-even folds | Call equity threshold facing it | Simplified minimum defense frequency |
|---|---:|---:|---:|
| 1/3 pot | 25% | 20% | 75% |
| 1/2 pot | 33.33% | 25% | 66.67% |
| 3/4 pot | 42.86% | 30% | 57.14% |
| 1 pot | 50% | 33.33% | 50% |
| 1.5 pot | 60% | 37.5% | 40% |

MDF here is `P0/(P0+B)`. It is a range-level anti-bluff reference in a simplified model, **not an instruction to call with every hand or always defend that fraction**. Defense includes raises. Range asymmetry, future streets and underbluffing can justify folding more. Pot odds and MDF answer different questions. [S5]

In a polarized river model where value always beats bluff-catchers and bluffs always lose, indifference implies **bluff:value ratio** `B/(P0+B)`, while **bluff fraction of the betting range** is `B/(P0+2B)`. A pot-sized bet gives 1 bluff per 2 value combinations, hence 1/3 bluffs, not 1/2. Code should count combinations after blockers; Astra decides whether the model assumptions fit the opponent. Against a script that calls every pair, reduce bluffs even if that is exploitable in theory.

Draw estimates: with 9 clean outs on a flop and 47 unseen cards, chance of hitting at least one out by the river is `1-(38/47)*(37/46)`, about 35.0%; with one card to come it is `9/46`, about 19.6%. The rule of 4 and 2 is a mental approximation, not a substitute for code. A made flush may still lose on paired boards or to a higher flush; do not double-count overlapping straight/flush outs. Paying for one card does not buy both remaining cards unless betting is finished. [S6]

## 5. Postflop plans and sizing

**Astra** should distinguish range advantage (who has more average strength) from nut advantage (who has more very strong hands). **Code** supplies board texture and weighted-range summaries; **Jev** can classify the strategic situation from those supplied facts. Position and actual preflop ranges are essential: the same board can favor different players after a 3-bet versus a limp/check pot. [S7]

| Situation | Starting plan, with exceptions | Ownership |
|---|---|---|
| Dry high-card board, e.g. A72 rainbow after button raise/BB call | Small continuation bets often efficiently pressure the BB's misses; protect checks with some good hands. A weak script that never folds makes pure air bets worse. | Astra range/line choice; code size and blockers; optional Jev classify |
| Connected middle board, e.g. 987 two-tone | BB's calling range can contain many two-pair, straight and strong-draw combinations. Check more of marginal range; strong value/draws may favor larger polarized sizes. | Astra; code made hands/draws |
| Paired or monotone board | Assess actual nut combinations and blockers; do not label every paired board dry or every monotone board an automatic large bet. | Code exact texture; Astra interpretation |
| Strong value on a draw-heavy board | Charge worse pairs and draws while building a pot; select a size they actually call. Do not “protect” by forcing every worse hand to fold when a smaller bet earns more. | Astra plus code EV candidates |
| Medium showdown value | Often check/call or check back. A bet needs worse calls, useful equity denial, or another concrete benefit. Avoid turning a hand with useful showdown value into a bluff without reason. | Astra |
| Strong draw, little showdown value | Semibluff when fold equity plus draw equity supports it. Prefer nut draws, useful blockers and good turn barrel opportunities. Calling can be better against a range that never folds. | Code equity/outs; Astra line |
| Turn changes nuts or range advantage | Reassess both ranges; barrel credible improvements, continue strong value, and stop some low-equity bluffs. A scary card is not automatically a good bluff card for hero. | Astra; optional Jev range-shift label |
| River | Value bet only when sufficient worse hands call; bluff with chosen low-showdown hands that block calls and leave folds available; use odds and opponent bluff frequency for bluff-catching. | Code combination/threshold work; Astra selection |

Use a manageable menu: check/call/fold plus legal bets/raises around 1/3, 1/2, 3/4, 1 and 1.5 pot, minimum raise and all-in. A smaller menu helps comparison; preserve exact engine sizing separately. Small bets can support a broad value range. Large bets and overbets usually need a more polarized range and a relevant nut advantage. Do not use one bet size for every board just because it is convenient. Against an inelastic caller, larger value bets may dominate “balanced” sizes. [S4, S7]

**SPR** is effective remaining stack divided by pot, preferably measured at the start of a street with matched bets. Code should state its convention when action is outstanding. At low SPR, strong top pair/overpairs can commit more naturally; at high SPR, one-pair stacks need much more caution. Plan future sizes: with no raises and matched bets, an equal fraction `f` of pot on each of `n` remaining streets gives `f=((1+2*S/P0)^(1/n)-1)/2` to invest effective stack `S` by the end. This is a sizing reference under fixed assumptions, not a reason to stack off. Code computes it; Astra decides whether that pot-building plan is desirable. [S6]

Bluff blockers are conditional. Holding an ace of a completed flush suit can remove nut-flush calls; holding a card from a missed draw may remove hands the opponent would have folded. For bluff-catching, prefer combinations blocking value and unblocking bluffs. A single “good blocker” never overrides a clearly value-heavy opponent history or makes a weak draw automatically profitable. [S7]

## 6. Exploit the frozen opponent, with evidence

| Observed tendency | Counter | Common overcorrection | Owner |
|---|---|---|---|
| BB folds too many opens | Open wider; use the smallest size that achieves profitable folds | Opening any two against a newly aggressive BB without updating | Code opportunities/fold rates; Astra ranges |
| Calls too widely, rarely raises | Value bet more thinly and often larger; reduce air bluffs | Never bluffing even where the script explicitly overfolds | Astra; code size-conditioned response counts |
| Excessive flop folds | More small continuation bets on suitable boards | Assuming the same fold rate on turn/river or after a raise | Code street-specific counts; Astra |
| Large turn/river bets overwhelmingly value | Fold more bluff-catchers, even beyond nominal MDF | Folding actual strong value or inventing certainty from two hands | Astra; code range equity and sample sizes |
| Too many bluffs/raises | Protect checks with strong hands, call down wider at suitable prices, reduce fragile bet/folds | Calling every runout with any pair or fighting with weak reraises | Astra |
| Passive checks frequently followed by folds | Probe turns and stab checked-back pots selectively | Treating every check as capped; ignoring traps | Astra/Jev labels; code conditional histories |
| Size choice tracks hand strength | Infer ranges separately by size and context | Treating an unobserved bucket as statistically certain | Code buckets/shrinkage; Astra hypotheses |

Track denominators: opportunities to open, fold-to-open opportunities, faced 3-bets, faced continuation bets, river bluff-catching opportunities. A frequency without its sample size is not actionable. Use shrinkage toward a stated prior (e.g. Beta(1,1) for a simple binary diagnostic, not an assertion of poker truth), and expose uncertainty. Showdown hands are a selected sample; unseen folds do not reveal what the opponent held. A lucky loss is not evidence a good call was wrong. [S5, S8]

Adapt from public action history, not from evaluator secrets. In duplicate evaluation, the second leg must not inherit the first leg's cards, board, results or strategic memory. If contexts cannot be isolated, balanced-seat independent deals are safer and more honest than a leaking duplicate score. Score all predeclared hands; never stop after a winning streak. [S8]

## 7. Suggested Astra decision prompt and optional Jev batch

The following is a prompt contract, not game-playing code:

> You play heads-up no-limit Hold'em for expected net chips per hand. Use only this private-view observation and public history. Blinds, stacks, legal candidates, pot odds and computed features are authoritative; do not reconstruct hidden cards. Identify position, effective stack, action sequence, plausible opponent range and confidence. Compare checking/calling with at most three meaningful legal bet/raise candidates. State whether a bet earns worse calls, folds better hands, or denies meaningful equity; name the likely opponent response and one future-street contingency. Prefer a robust default when evidence is weak and a specific exploit when evidence supports it. Output the selected legal candidate ID (and an optional short reason through a separate debug field). Never output an amount absent from the legal contract, chase past losses, or use a previous duplicate leg.

The observation should already include exact pot odds, candidate costs and post-call pot; made-hand and draw labels; equity against named range hypotheses with sample counts; position and SPR; and opponent statistics with denominators. A candidate score is estimated action quality, not a calibrated probability of winning the hand. Code alone performs any final probability mixing.

Optional **Jev** questions can be independent typed classifications in one batch:

| Question | Output type | Use |
|---|---|---|
| Given the supplied range summaries, who appears to have the nut advantage? | `hero / opponent / neither / uncertain` | Input to Astra's sizing review |
| What best describes hero's proposed bet? | `value / semibluff / pure_bluff / equity_denial / unclear` | Detect an incoherent rationale |
| Which supplied opponent tendency best fits these statistics? | `overfold / overcall / overaggressive / insufficient_evidence` | Suggest a hypothesis, not a forced adjustment |
| Does this river bluff remove more likely calls than likely folds, based on the listed combinations? | `yes / no / uncertain` | Qualitative cross-check; code counts combinations |
| Is this proposed plan consistent with the stated opponent model? | `yes / no / uncertain` | Optional second opinion |

Skip Jev when it adds no value: the world waits, and its faster response is not needed to prevent a timeout. Do not ask it to do equity arithmetic, find legal actions, remember hidden cards, or plan all remaining streets. Avoid duplicated model calls for automatic all-in runouts.

## 8. Failure modes and development priorities

1. **Code first in the adaptor phase:** legality, privacy, deterministic deals, exact hand ranking, payoff ledger and fixed opponent. A sophisticated strategy cannot repair a raise-to bug or leaked cards.
2. **Add exact features:** pot odds, SPR, draw labels, range-aware equity and a compact legal menu. Keep model prompts small enough to inspect; retain full public history in machine-readable form.
3. **Use a coherent preflop prior and a value-first postflop baseline.** Test exploit adjustments against the frozen opponent on development seeds; keep held-out deals untouched.
4. **Review recurring high-cost errors:** overly tight blind play, oversized deep-stack bluffs, paying off underbluffed rivers, missing thin value, and betting hands that neither get worse calls nor better folds.
5. **Measure chips/hand and uncertainty.** Win rate by hands or pots is not the objective. Pairwise deal averages are the unit for duplicate uncertainty; correlated adaptive sessions need independent block-level analysis. Many thousands of hands can be necessary for small win-rate differences, depending on variance; 200 development hands do not establish superiority. [S8]

Do not implement real-time clocks, GUI recognition, online account automation, tournament ICM, rake adjustments, GPU training, or a full-game equilibrium solver for this benchmark. They do not address the selected offline task. This study contains no game-playing implementation.

## Sources and reading targets

- **S1:** Brown and Sandholm, *Superhuman AI for heads-up no-limit poker: Libratus beats top professionals*, Science (2018): <https://doi.org/10.1126/science.aao1733>. Research motivation for abstraction, safe subgame reasoning and robust imperfect-information strategy; not a downloadable baseline policy promised here.
- **S2:** Moravčík et al., *DeepStack: Expert-level artificial intelligence in heads-up no-limit poker*, Science (2017): <https://doi.org/10.1126/science.aam6960>; project <https://www.deepstack.ai/>. Continual resolving and range-based reasoning.
- **S3:** Zinkevich et al., *Regret Minimization in Games with Incomplete Information* (2007), authors' poker research portal: <https://poker.cs.ualberta.ca/publications.html>. Practical framework reference: <https://openspiel.readthedocs.io/en/latest/algorithms.html>. CFR belongs in a deliberate research extension, not an assumed preinstalled dependency.
- **S4:** Upswing Poker, heads-up strategy/community instruction: <https://upswingpoker.com/heads-up-poker-strategy/>; article index <https://upswingpoker.com/blog/>. The range percentages and sizing examples in this guide are explicitly our starting priors, not copied charts.
- **S5:** Upswing Poker, *Minimum Defense Frequency vs. Pot Odds*: <https://upswingpoker.com/minimum-defense-frequency-vs-pot-odds/>. Supports distinguishing calling price from range defense; formulas here are derived explicitly.
- **S6:** PokerNews strategy library: <https://www.pokernews.com/strategy/>; rules <https://www.pokernews.com/poker-rules/texas-holdem.htm>. Reading topics: pot odds, implied/reverse implied odds, effective stacks, stack-to-pot ratio. These broad resource links do not certify a particular numeric chart.
- **S7:** GTO Wizard strategy library: <https://blog.gtowizard.com/>. Reading topics: range/nut advantage, equity realization, blockers, polarized versus merged betting. Public educational material is useful; a paid subscription or live solver is not required or assumed.
- **S8:** Burch et al., *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*: <https://arxiv.org/abs/1612.06915>. Duplicate deals are a simpler variance-control method, not an implementation of AIVAT. Local rule/scoring evidence: `/home/hemlix/GamingMaster-mh-tier1/data_processing/rs/gmrules/src/games/hunl/mod.rs` and `/home/hemlix/GamingMaster-mh-tier1/scripts/reference/pokerkit_hunl.py`; public repository <https://github.com/sunyxedu/GamingMaster>.
