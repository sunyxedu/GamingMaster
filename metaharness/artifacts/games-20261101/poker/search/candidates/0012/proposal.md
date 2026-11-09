parent: 0009

## Hypothesis

Preserving the complete opponent combination range and conditioning it on every public betting street will improve response-aware decisions more than a made-hand sizing gate. Candidate 0009's exact river computation discards flop/turn selection, while the leader 0005 resamples its range to about 200 particles. Both can consequently overvalue hands whose raw strength is high but whose opponents have repeatedly selected a stronger range. This candidate extends 0009's river rank-distribution method to board-completion batches, retaining every legal combination throughout the hand. It is not a refinement of the leader's sizing caps.

## Concrete archive evidence

The leader's weakest completed match, 0005 seed 4409, loses 20,000 on hand 64 (settlement at step 144): 7s9d on 6d7d2d3dKh bets 18,963 into 2,074; 8cTd calls. A nine-high flush passes the leader's strong-hand gate. This outcome does not by itself establish negative EV, but highlights the need to preserve higher flush combinations and account for earlier calls.

Parent 0009's weakest match is also 4409 (425.45 chips/hand). Its hand 66 settles at step 161 for -20,000. Ks6c on 4s7hKh4d2s bets 375 on the flop, calls a raise to 1,313, bets 2,345 on the turn, calls a raise to 8,207, then bets its remaining 10,230 on the river. Opponent shows Kc8h. The river solver sees neither earlier raise when constructing its range. Hand 64 loses another 5,000 after betting 3,750 on the four-diamond river and folding to 11,250. These are falsifiable information-loss examples, not proof that every earlier call was bad.

## Change from parent

Retain the standard-library exact evaluator and preflop policy. Build complete opponent equity tables by averaging exact river rank distributions over shared board completions. River is exact; turn enumerates all legal single-card completions; flop uses reproducible public-board-only sampled completions. Opponent uniform equity includes hero's private cards as possible unknown cards. Hero's conditional equity excludes both players' hypothetical holes from the runout. Replay all public opponent actions against those street-specific tables with softened likelihoods and nonzero support; never resample or delete unblocked combinations.

Use the resulting range for flop/turn as well as river EV decisions. Model the opponent's raise reply as an aggregate information set with a single fold/call continuation. Astra receives execution failures and low-probability response evidence, and can revise bounded likelihood uncertainty and realization margins. Model parsing and timeouts are caught, and every action is validated and checked through ctx.progress(). No game source, private state, evaluation seeds, or external libraries are accessed.

## Confirmation and refutation

Require complete, error-free independent matches, higher mean/weak-match chip scores, and evidence that repeated opponent strength changes later decisions. Replay should show that the parent hand-66 turn and river are no longer treated as ordinary uniform-equity value spots. Refute if extra computation prevents completion, if approximate likelihoods become overconfident, or if lost thin value and missed bluffs outweigh saved commitments. Shared runout samples induce correlated approximation errors; they are not independent observations or statistical confidence intervals. Match seeds, not hands, are the uncertainty unit.

## Verification

Preflight passes after 120 actions with arbitrary stand-in model output and no exception. A preliminary implementation completed development seed 901 in 428 seconds at +1,147.29 chips/hand, with two answered Astra calls, no failed calls, and zero illegal actions. This run preceded a correction to exclude hypothetical opponent holes from hero runouts, so it is exploratory rather than a direct evaluation of the final file.

A replay of parent 0009 seed 4409 hand 66 changes the decision at the turn: the full range assigns K6 equity 0.2235 after the opponent's raise, selects fold to 5,862, and checks the subsequent river rather than committing the remaining 10,230. Exact turn equity tables matched direct enumeration for K8, AA, and 53 examples. The first Astra review incorrectly treated previous-decision alternatives as current; the payload now labels their observation explicitly and reserves action overrides for execution recovery. Public-history counts now include deduplicated automatic settlement events.

The final implementation completed development seed 902 in 309 seconds: +1,294.30 chips/hand, 240 actions, two answered Astra calls, and no errors or illegal actions. This single final-code development seed is exploratory and does not substitute for the evaluator's training matches.
