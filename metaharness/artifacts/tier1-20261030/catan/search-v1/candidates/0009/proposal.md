parent: 0006

## Hypothesis

The remaining bottleneck is the unit of planning: individual actions or one
economic goal can miss a stronger whole-turn sequence. Candidate 0006 lost seed
108 at 7 VP. In its trace, steps 41–44 build roads [22,23] and [23,52], exchange
four sheep for wood, and buy development, while the recorded longest road stays
three. Step 75 trades four ore for sheep with one wheat already in hand. These
locally ranked choices destroy both useful build bundles and road continuity.

The best candidate, 0004, wins every training seed but still takes 107 controlled
actions on seed 106. Its steps 93 and 104 end turns holding the identical hand
WOOD=3, BRICK=2, SHEEP=2, WHEAT=1, ORE=2 at nine VP; the first is followed by a
discard. This is evidence of exposed waiting, not proof that a win was available.
Its one-goal forecast does not explicitly compare complete resulting turns.

There is also a concrete terminal-score opportunity in the leader's seed 101.
After step 72, `self.true_vp` is 9, `longest_road_owner` is null, and
`longest_road_length` is 4. Both road `a2` at [38,39] and settlement `a14` at 39
are legal. Step 73 chooses `a14` and ends at 10. The public road graph has the
trail 5–4–15–17–39; adding 39–38 predicts Longest Road and 11 VP. The new search
selects `a2` on this archived observation. This is an observation-supported
counterfactual, not a new engine evaluation of that training seed.

This candidate tests deterministic beam search over **action sequences**, using
the alternate 0006 parent. The search can see trade/build combinations, choose
between competing uses of the same resources, and put a one-point build before
a two-point award when that gives an immediately reachable eleven-point finish.
The latter is supported by 0003's seed 105: step 166 takes Longest Road from nine
true VP and the engine terminates at eleven. No winning action is postponed for
a speculative future eleven-point finish.

## Changes from the parent

Retain its complementary opening settlement policy. Replace normal-turn action
ranking with bounded public-state forward search, transposition pruning, exact
integer bank exchanges, piece and connectivity tracking, and an explicit leaf
evaluation of VP, production, resource reserves and useful expansion options.
Search uses at most eight action layers, beam width 28, a 5,500-expanded-state
target, and a 0.8-second search target checked between expansions. It ranks
terminal VP above all nonterminal utility and uses action count to break equal
terminal scores. This can choose settlement → trade → award instead of an
immediate lower-scoring award. On 0005's archived seed 104, step 171, it predicts
settlement 21 → four sheep for brick → road [15,17], finishing at 11 instead of
the available direct 10-point road. Again, this is an offline prediction.

The root actions always come from the current legal list. Hypothetical actions
only value a root choice; the executor recomputes from the next observation.
Unknown development draws, robber theft and Monopoly returns are not sampled
from engine state. They receive public-information estimates at search leaves.
No game imports, random-stream consumption, hidden information or model calls.
Production and resource prices are heuristic leaf values, not exact waiting
times. Root development availability is authoritative, including this build's
observed ability to expose a just-bought card for play. Robber scoring now
penalizes blocking our own production and weights opponents by public threats.

The road model enumerates edge-simple trails and uses the archive-supported
exclusion of edges incident to opponent buildings. A new settlement also
recomputes opponents' road lengths: in 0006 seed 106, step 99, settlement 14 cuts
WHITE's road and transfers the award to RED, raising true VP from 6 to 9. This
was the only transition discrepancy in the first audit and was repaired.

## Confirmation and refutation

Confirmation: useful VP/production per paid road improves, resource churn drops,
the weak parent episode improves, and the policy finishes with fewer actions or
occasionally eleven VP while retaining zero errors and zero model calls.
Refutation: the search's leaf values buy short-term VP at the expense of the
economy, prune essential preparatory trades, mispredict graph legality, or spend
too much time searching. Search is bounded and current legal actions remain the
authority; schema failures fall back to a legal action and are logged.

## Validation

* Final required preflight: `ok: true`, `game_done`, 10 VP, 66 controlled actions,
  zero model calls. The no-System-2 warning is intentional.
* First permitted dev invocation, `dev/20261030-065222`: seeds 301/302 both won
  at 10 VP, in 52/62 controlled actions. No errors or fallback events. The
  longest recorded search was 0.352 seconds. The 301 finish explicitly found
  road → settlement → city; 302 found two bank exchanges → settlement.
* Second and final dev invocation, `dev/20261030-065939`: both won at 10 VP,
  in 45/51 controlled actions (118/156 total actions including opponents).
  No errors, fallback events, or model calls; longest search 0.236 seconds.
  Both digests and step/search traces were inspected. Initial observations
  differ between runs for both seeds, so action-count differences are not a
  controlled causal estimate. Neither dev invocation produced an 11-point win.
* `audit.py` reads only archived traces. On 0003/0004/0006 it checks 1,559 normal
  observations: inferred road/settlement/city/trade action sets match the current
  engine lists exactly. All 707 observed deterministic transitions match hand,
  bank and true VP: 395 exchanges, 196 roads, 51 settlements, 59 cities and six
  Year of Plenty plays. No exceptions. The settlement-cut fix removed the only
  initially observed VP discrepancy.
* After the second dev run, a conservative empty-deck guard was added: when
  development is affordable but absent from a normal-turn legal list, search
  does not imagine buying a card after a later trade. The bank observation has
  no development-deck count. Replaying all 96 final-dev decisions after this
  guard changed zero selected IDs; the final preflight ran afterward.

No full training evaluation, held-out evaluation, engine-source inspection, or
model-backend call was performed. The training improvement remains unmeasured.

## Remaining limits

Beam pruning can miss long trade/road sequences. Opponents and dice are not
simulated; development value uses a prior, Monopoly yield uses public production
and hand sizes, and the robber discount is heuristic. Unowned generic ports
cannot be distinguished from ordinary `port: null` vertices, so hypothetical
settlements do not receive an unverified 3:1 benefit; the next real observation
corrects the rates. Root action sets are verified extensively, but that does
not prove every possible hypothetical future state matches the engine.
