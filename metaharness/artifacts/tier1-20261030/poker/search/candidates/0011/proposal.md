parent: 0009

## Hypothesis and archive evidence

The river solver needs a contingent betting plan, not just a terminal action value. Candidate 0009 models hero betting, villain raising, and hero choosing fold or call, but omits hero reraising. It also values an out-of-position check as immediate showdown. These omissions can undervalue inducing a bet or raise with strong hands, and misvalue bluff-catching after a check. This candidate tests a finite river betting tree with range-consistent future decisions. It is a different parent and hypothesis from leader 0005's staged bet caps; flop/turn policy and the parent's range approximation are retained.

The leader's weakest training match is seed 4409, 675.08 chips/hand. At step 144, hand 64, 7s9d on 6d7d2d3dKh shoves 18,963 into 2,074, is called by 8cTd's higher flush, and loses 20,000. This is evidence that a made-hand gate does not establish caller-relative strength, not proof that one lost shove was negative EV. The alternative parent 0009 uses exact river ranks and avoids that shove, but its same match still loses 5,000 on hand 64 after betting 3,750 into 2,500 and folding to a raise to 11,250 (step 152). Its river log at that decision reports equity .7493; the subsequent raise-conditioned equity is .0315. This demonstrates how much information a response reveals and why contingent actions must be valued over the responding range.

A separate limitation is explicit: 0009's hand 66 bets and calls raises with Ks6c on 4s7hKh4d, then shoves 10,230 on 2s and loses 20,000 to Kc8h (step 161). The river posterior omits earlier street actions, so this experiment may still make that mistake. This candidate does not claim an exact opponent posterior or repair every observed loss.

## What changed

Reuse 0009's standard-library hand evaluator, exact river uniform-card equity table, approximate preflop/river range likelihoods, pre-river policy, and legal action menu. Replace the river EV calculation with a finite tree:

- Hero bet → opponent fold/call/raise → one aggregate hero fold/call/reraise decision → opponent fold/call.
- Hero checks in position → showdown.
- Hero checks out of position → opponent checks or makes a half-, three-quarter-, or full-pot bet → hero optimizes fold/call/raise conditional on the observed size.

Opponent responses use the documented .62/.85 value gates, .80 raise gate, .08 mixed branch, pot odds, and a smooth likelihood floor. The opponent's own river raise count limits its raising branch, as in the parent. All hypothetical hands sharing an observed action contribute to the same conditional range before a hero action is maximized. No per-hand clairvoyant action choices occur. Root bets are engine-supplied candidates; internal targets follow minimum full raises and effective stacks. Short all-in raises do not reopen raising. Dollar values include previously invested hypothetical chips so folding after a bet loses that bet.

The parent's oversized-bet guard (called equity at least .97 beyond 1.5 pots) and small cost penalty also apply inside the tree, preventing a continuation from bypassing the root guard. Those estimates remain approximate. A one-step observation update always supersedes the plan; no future action is posted without re-solving from the actual cards, history and bounds.

Astra receives public failure evidence and a compact current river plan, with an optional legal current-action override during recovery. Existing bounded posture reviews remain. Progress checks follow every action; rejected/stale actions, large public losses, or a response inconsistent with the predicted branch request review. Model calls use 180-second timeouts, above the recent 73-second maximum and the archive-wide 121-second maximum, and reserve 240 seconds. Arbitrary replies, invalid JSON, timeouts and failed calls preserve the legal local policy. Local tree work checks remaining wall time and yields before the grace deadline.

## Verification and falsification

An initial public-observation replay covered all 167 river decisions in the parent's four training traces: every emitted action was legal, total comparison runtime was 8.53 seconds, and 27 choices differed. This is a counterfactual decision audit, not a match evaluation: subsequent cards/actions in a trace are never treated as the result of a new choice. Final preflight and any development results are recorded below.

Confirm if independent completed matches improve net chips per hand and strong-hand value through check/raise or bet/reraise lines without excessive bluff-catching or missed direct value. Refute if the single-raise convention is inaccurate, the earlier-street range omission makes trapping expensive, or conditional sizing estimates systematically overvalue checks. Report match-level variation, not hand-level standard errors. No held-out results or implementation source were inspected.
