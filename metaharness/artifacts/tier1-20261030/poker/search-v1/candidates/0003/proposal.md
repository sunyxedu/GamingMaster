parent: none

## Hypothesis

The fixed opponent's random-hand equity is the exploitable abstraction. Its calls ignore hero's range even for 200bb bets, while its own checks, bet sizes, and raises disclose thresholds on that equity. A public-action Bayesian range plus response-aware sizing should both avoid paying strong ranges and extract much larger value than a generic made-hand policy. This is a fresh design, not a refinement of the leader's prompt or candidate 0001's category cutoffs.

The archived baseline's seed 2207 made only five actions, finished two hands, and ended at the 1800-second deadline. Its `raise_to 250` model call took 280.7 seconds; the following call was cancelled at the deadline. Thus full completion requires scripted routine play. Candidate 0001, now present in the archive, completes all four matches but averages -79.9925 chips/hand; seed 2207 scores -559.76. Completion alone does not solve the opponent. In its seed 2207 hand-9 trace, Qh5c reraises a premium 3-bet to 1250, then calls 18750 more against a shove and loses 20000 to AhKs. Hand-62 raises a nine-high straight on Td Js Ks Qs, then commits the rest on a paired river and loses 20000 to an ace-high straight. These are distinct failures to condition on the action sequence and relative hand strength.

## Implementation

Code owns exact best-five card ranking, legal current-street totals, reproducible simulation, and the entire action loop. No model calls or outside dependencies are required. All randomness is derived from public cards and a fixed algorithm label, never deal seeds or hidden state.

The harness maintains weights over all legal opponent hole combinations. Each observed opponent action updates these weights with a smoothed likelihood derived from the documented frozen policy. Preflop equity is estimated by 2048 independent samples per class; the opening range is the exact class union in prior/interfaces.md. Postflop, common public board completions supply both hero-versus-combo equity and the opponent's equity versus uniformly random opposing cards. River random-hand equity is exact: rank all legal combinations and subtract combinations sharing each opponent hole card. Turn integrates every possible river; flop samples 160 public completions. Hero card removal is applied to hero range equity but not to the information available to the opponent's uniform-equity calculation.

A one-street expected-value search compares checking/calling/folding with multiple bet sizes, including deep all-ins. It predicts the script's calls and raises, discounts uncertain future realization, and demands a larger edge for high exposure. A hypothetical future response is chosen against the entire posterior raising range, never separately for each sampled private hand. Preflop also explicitly tests value shoves against the script's price-dependent continuation range. This tests the particularly bold prediction that premium hands should often jam 200bb because the script calls using equity against random cards. No inference result commits multiple unvalidated actions; every turn recalculates from public history.

## Confirmation and refutation

Confirm: completed 100-hand matches with no illegal actions; positive independent-match mean; big value wagers called by materially dominated holdings; and range-conditioned folding that avoids the prior scripted policy's full-stack losses. Logs expose inferred equity, continuation probability, called equity, and action EV to audit these mechanisms. Refute: errors in observed action likelihood, poor calibration after strong action, negative value-jam outcomes across independent matches, or compute costs preventing completion. Two dev seeds are diagnostic, not evidence of low variance or established superiority. The one-street EV approximation and finite flop sampling remain explicit limitations.

## Validation

Initial pre-flight passed without exceptions. An independent five-card reference matched the exact evaluator on 6000 sampled five-, six-, and seven-card hands. Blocker-corrected river equities matched direct enumeration on four sampled combinations.

The first development evaluation (901 only) completed 100 hands in 116 seconds, 210 actions, zero model calls, zero illegal actions, and +576.5 chips/hand. One 200bb turn value shove with Jd7c on Jh Ks 5h 8c was called by Ah9h and lost to the river flush; this is consistent with a profitable value wager losing a single draw, not by itself a reason to abandon the hypothesis. After reading this digest and trace, the refinement replaces the approximate opening prior with the documented exact set, increases equity sample counts using the measured runtime headroom, and makes future raise responses explicitly range-wide. The final pre-flight passed (59 actions before its short dry-run deadline, no exceptions). A replay of the archived bad public observations now folds Qh5c to both the 500-chip reraise price and the 18750-chip shove price, and folds Kc9c to the 2813-chip turn raise on Td Js Ks Qs (range equity 3.04%). These are diagnostic replay checks, not additional evaluation episodes. The final development evaluation is in `dev/20261030-002838/`:

| Dev seed | Completed hands | Chips/hand | Actions | Wall seconds | Illegal actions / errors |
|---|---:|---:|---:|---:|---:|
| 901 | 100 | 574.50 | 207 | 719.8 | 0 / 0 |
| 902 | 100 | 214.25 | 181 | 695.6 | 0 / 0 |

Both ended with `game_done`, with zero S1/S2 calls. The mean is **394.375 chips/hand**; the sample standard error across the **two independent match scores** is **180.125 chips/hand**. This is noisy exploratory support, not a training score or reliable superiority claim. The initial one-seed result is not a third independent match and is excluded from this uncertainty calculation. No official training or held-out evaluation was run by this proposer.

The final seed 902 trace supplies direct mechanism evidence despite a losing result: hand-96 jams Ts8h on Jh 8c Th 5c, and the opponent calls 17750 with QcQs. Hero has exactly 72.727% equity against those legally revealed cards at the turn (32/44 winning rivers), but loses when 9d makes villain a straight. The policy estimated 76.03% against the full continuation range, which is a different quantity. This illustrates both the overcall exploit and the variance of 200bb value bets. Winning opponent holdings were often mucked; their exact cards are not inferred or claimed.

Future work should examine finite-sample flop range calibration, the correlation of repeated opponent decisions using cached equity, and the one-street approximation to future betting. Large value shoves intentionally create high match variance; selection must use completed independent match blocks.

Pre-flight repair: CPU-only preflop equity/range construction and bet-search loops could outlive the 60-second deadline because they lacked cooperative cancellation; added budget/done checkpoints inside expensive loops, immediate unwinding without caching partial tables, and clean handling of deadline races at the episode boundary while preserving the strategy. Repair pre-flight passed (`harness_return`, 57 actions); targeted cancellation and deadline-race checks also passed.
