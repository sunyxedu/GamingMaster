parent: 0010

## Hypothesis

A fixed opponent's exact river equity is a persistent latent fact. Multiplying independently softened call/check/raise likelihoods admits internally contradictory river sequences and makes posterior support look more reliable than it is. A small mixture of coherent threshold policies, each conditioned on the entire observed river sequence, should improve exploit decisions. Once responses are threshold partitions, cumulative weighted distributions also make every legal integer sizing cheap to evaluate. This is a different experiment from leader 0012's all-street shared-runout tables.

## Archive evidence

Leader 0012's weakest completed match (2207, 702.77 chips/hand) loses 20,000 on hand 82 at step 190: Kc3s on Kd9s5c4c check-shoves 17,500 after a 3,750 turn bet; Kh2c wins when the river is 2h. The leader also loses 12,500 on hand 64 of 4409: 7s9d on 6d7d2d3dKh bets 3,750 then calls a raise to 11,250, losing to 8cTd (step 151). These losses do not prove negative EV. They show why an exact rank table cannot certify an approximate action model. Two leader matches stop at the deadline, so new all-street computation is unattractive.

Candidate 0011 already tested a river check/raise/reraise tree and regressed to 390.645. Merely adding a tree is therefore not a novel hypothesis or established improvement. Its earlier-street range omitted selection. Parent 0010 carries earlier action likelihoods into full river support, completes all training matches, and provides a better controlled base for testing coherent river conditioning and continuous sizing.

## Changes from parent

Retain parent preflop, flop, and turn play, standard-library evaluator, complete river combination restoration, and approximate earlier-street likelihoods. On the river, maintain three persistent equity-bias hypotheses centered on the exact public-card uniform equity. Condition each on the complete river sequence with the documented deterministic thresholds and 8% mixed branch. Do not apply an independent likelihood floor at each river event. If all hypotheses are contradicted, report the contradiction to Astra and recover with the parent's softened river model.

Optimize a finite river response tree using sorted equity distributions and prefix sums. Test every integer raise total within the parent's ordinary sizing cap, plus its guarded maximum. Include out-of-position check/bet branches and aggregate fold/call/reraise continuations. Every hero choice is shared across the entire observed-response range; no hidden-hand clairvoyance is allowed. Opponent raise branches retain their once-per-street allowance. Re-solve after each observed action rather than executing stale future actions. Preserve conservative oversized-bet certification in internal branches as well as at the root.

Astra reviews model contradictions, unlikely responses, and execution failures. It can choose coherent versus softened river modeling and bounded threshold uncertainty, with strict parsing. Calls use 210-second timeouts, above the observed 146-second maximum. Every posted action is legalized and followed by ctx.progress(); local work checks the deadline. No other candidate, game implementation, hidden state, or held-out results are used.

## Confirmation and refutation

Confirm through complete, error-free independent matches, better net chips per hand, and fewer implausible river continuations without excessive missed value. Trace mechanism checks should show legal sizes outside the menu, different posterior support after repeated actions, and actual response frequencies compatible with predictions. Refute if coherent hypotheses repeatedly collapse, exact thresholds overfit outdated documentation, earlier-street posterior error still dominates, or new search delays completion. Candidate 0011's negative result is a reason to require measured improvement, not assume deeper search helps. Use matches as the uncertainty unit.

## Verification

Implementation and verification results will be appended after preflight and public-observation checks. Replay decisions are mechanism evidence only; later archived actions are not counterfactual outcomes of a new action.

## Verification results

Preflight passed after 74 actions with arbitrary stand-in output, zero answered calls, and no exception. Final-code development seed 901 completed in 161 seconds at +772.05 chips/hand, with one System-2 call (still in flight at terminal and canceled), no illegal actions, and no error. Final-code seed 902 completed in 216 seconds at +1,140.70 chips/hand, with one answered Astra review, no illegal actions, and no error. Astra left the coherent model unchanged because evidence was sparse. Two dev seeds are exploratory and do not establish an improvement over the leader.

Public replay on 0012's 4409 hand 64 river observation chooses a fold to the 11,250 raise (the archived call lost 12,500). The preceding river decision still selects a 3,125 value bet, so this solver does not claim the initial bet is wrong. Parent 0010's public hand-86 flop call is likewise left to its existing earlier-street policy; coherent river conditioning cannot fix it. Public replay of 0009 hand 66 at the river selects a minimum bet and plans to fold to a modeled raise; the parent 0012 already avoids the disastrous earlier turn call, so this replay is a mechanism check only.

Mechanism checks verified legal action output, threshold partitions, one-raise behavior, exception-safe review parsing, and menu-independent integer targets. In seed 902 the river search selected 16 raise sizes not present in the compact menu, all accepted by the engine, with 0 illegal actions. River search took 1.3–3.1 seconds in public replay and under 0.2 seconds in the development trace's measured local sections. One 11,824-chip development loss followed an earlier-street call/raise sequence; this remains a concrete failure mode for the inherited policy and a limit on what this river-specific hypothesis can repair.
