parent: 0006

## Hypothesis

Whole-turn resource allocation is a better unit of decision than independent legal-action rankings. Parent 0006 loses seed 108 at 7 VP. At steps 90 and 91 it exchanges four sheep for wheat twice, leaving WHEAT=2, ORE=2. The second exchange could instead obtain ore and fund a city immediately. Earlier, steps 74–75 build an unproductive road and exchange four saved ore for sheep. The episode contains twelve road placements but only one city. These decisions are individually legal and collectively wasteful.

Leader 0004 has no training loss: it reaches exactly 10 on all eight seeds. Its longest training episode, seed 106, takes 107 controlled steps and 402 total engine actions. Its goal-at-a-time controller cannot explicitly compare a whole turn's alternative sequences. This experiment starts from 0006, not the leader's economic recipe, and tests bounded public-state search over complete bank-exchange/build macros. It also treats terminal true VP as the stated evaluation score: when a deterministic sequence can reach 11 by taking an award at nine, it may outrank a ten-point finish. No extra turn is deliberately spent after a known immediate win merely to hunt an extra point.

## Changes

Preserve 0006's opening placements and observation parsing. Replace normal-turn action priorities with a bounded search over complete build macros, each with integer maritime-payment sequences and graph-legal prerequisites. Each search state carries our hand, public bank, own pieces, roads, buildings, port rates and road-award ownership. Replan after every submitted current legal ID; hypothetical paths never authorize an unavailable action. Exact deterministic transitions stop at terminal VP; unknown development draws remain leaf estimates. A resource-reservation fallback handles turns with no complete macro, avoiding speculative roads without a settlement or award target. Public graph rules follow the observation-supported road convention documented in 0004, without importing its harness or any engine code.

## Confirmation / refutation

Confirm with more multi-build turns, fewer resource-destroying trades, fewer roads without productive endpoints, elimination of the parent's missed affordable cities, and equal or improved win/score with zero model latency. Eleven-point finishes are a secondary opportunity, not a reason to lose a guaranteed win. Refute if bounded search repeatedly prunes the best sequence, the heuristic overvalues development draws or reserves too long, or hypothetical graph/port predictions disagree with actual outcomes. Training outcomes remain for the evaluator; dev runs and archive replay only validate this candidate.

## Validation

- Pre-flight passed: a stand-in-model episode reached 10 VP in 56 controlled steps, with zero System-1/System-2 calls, no illegal-action errors, and no uncaught exceptions.
- One permitted dev evaluation (`dev/20261030-065535`) won on both seeds: seed 301 reached 10 VP in 37 steps and seed 302 reached 10 VP in 108 steps; mean score 10.0, zero model calls, zero failed calls, and both ended by `game_done`.
- The seed-301 trace shows the intended behavior: the search staged roads, a maritime conversion, settlement, and city across observations, then at 9 VP converted four brick for wheat and immediately city-built to 10. Seed 302 also completed a road/settlement macro and later city-built to 10 after a 7 discard.
- These dev episodes are functionality checks, not training evidence or a paired causal comparison with 0006/0004. Unknown development-card draws and robber theft remain approximate in hypothetical leaves; every posted action is still checked against the current legal list.
