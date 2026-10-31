parent: 0003

## Hypothesis

The leader models each opponent action as an independent noisy threshold test, although heuristic-v1 caches its random-hand equity for unchanged visible cards. A check constrains that same latent estimate below .62; a subsequent call further truncates the same estimate. Reapplying independent sigmoids and predicting another independent response discards this information. This can misprice the opponent's continuation and raising ranges, especially for deep value wagers. A persistent within-street posterior over the opponent's fixed equity estimate should improve public-range inference and response-aware sizing.

## Change under test

Build on the eligible leader 0003. Preserve its preflop strategy, one-street betting search, action sizes, public combination range, and zero model calls. Replace independent postflop action likelihoods with a latent-equity distribution per possible opponent holding. Public checks, calls, and exact bet sizes condition that distribution; future responses integrate against the conditioned distribution. On the river the opponent's equity is exact, so the latent distribution is a point mass. At a new street, carry forward holding weights and initialize a new equity estimate. Use the archived faster exact card evaluator as supporting engineering, with equivalence checks, and reserve time before entering potentially expensive game steps.

All calculations use public observations and hypothetical legal holdings. No deal seed, actual hidden hand, game implementation, or external model backend is accessed.

## Evidence and falsification

Leader 0003's weakest training match scored 83 chips/hand and completed all 100 hands. Its successor 0009's seed-2207 final decision was logged at clock 1799.4118, immediately before the 1800-second deadline; the game step outlived the grace period and made the candidate ineligible. Thus this experiment retains the leader's short planning horizon and adds a pre-action time reserve.

Confirm the inference hypothesis if analytic checks show that public history truncates future response probabilities consistently, observed policy actions have positive posterior likelihood, and completed independent development matches improve chips/hand without abandoning the leader's value-bet mechanism. Refute it if latent estimates repeatedly contradict public actions, if inference increases runtime enough to prevent completion, or if independent-match performance declines. Individual all-in outcomes do not establish calibration or superiority.

## Concrete trace diagnosis

In `0003/trace_seed2207.jsonl`, hand-1's flop observation holds 5sTh on 6d8hJh, pot 400, and ends with the opponent's free check. The documented policy makes that check only when its cached equity is below .62. Replaying the leader's posterior and response formula nevertheless assigns **0.0070864** probability to the later .80-equity raise branch. The new conditioned distribution assigns exactly zero. On the 2h turn, another opponent check leaves the same issue: old probability .00020595, new zero. The leader chooses a 600-chip bet there; the new posterior chooses 1000, estimating continuation probability .2204. This is a controlled replay of a public node, not a claim that the new action would win that counterfactual hand.

The inference also changes uncertain-value sizing before any current-street opponent action. In hand-82, Kc3s on Kd9s5c, the old policy jams 19750 with called equity .6173. The new uncertainty model selects 500 with called equity .7383. The archived jam lost 20000 to publicly shown Kh2c after a 2h river. That realized loss does not establish that the shove was bad: the experiment tests range calibration, not avoidance of one known runout. Hand-22's river jam remains a jam despite its archived loss; this is deliberately not a hand-specific avoidance rule.

## Numerical model and limits

For each possible holding, the latent equity estimate is a truncated normal with mean equal to its computed random-hand equity. Its variance combines the conservative Bernoulli bound `q*(1-q)/512` for the opponent's sampler with the measured variance of the harness's 160 public flop runouts divided by sample count. Turn integration has no harness sampling variance; river equity is a point mass. This replaces the parent's arbitrary logistic widths as well as its independent action likelihoods, so the experiment tests the coherent observation model as a whole, not correlation in isolation. The normal approximation, ignored tie-variance reduction, finite flop integration, and unchanged one-street future-value approximation remain limitations.

Public actions restrict a shared collection of equity intervals. Interval intersections retain the same sampled equity across repeated decisions. Normal tail integrals supply per-holding evidence and conditional call/raise probabilities. New streets retain holding weights but initialize a fresh latent estimate. Free-bet observations match the exact rounded total after minimum-size and stack clipping, merging branches when they emit the same public wager. The .08/.92 mixing probabilities are applied per observed action. Contradictory observations trigger explicit diagnostic counters and conservative recovery rather than a crash.

The archived 0009 bit-mask evaluator and strength-sorted CDF sweep are reused without its deeper river planning. Sample counts and exact table entries remain unchanged. This is an engineering control to make the new posterior affordable. Each action also checks a reserve of at least 20 seconds, enlarged to twice the slowest observed game-step duration plus 3 seconds. CPU loops retain cooperative cancellation.

## Validation

Preflight passed: 58 actions, zero model calls, no exceptions, clean `harness_return` at the short dry-run limit. `validate.py` compared 15,000 random five-, six-, and seven-card ranks against the archived exact evaluator and checked board-specialized ranking. Complete flop, turn, and river equity dictionaries matched exactly (3,385 combination entries). Analytic checks verify that a check eliminates the .80 raise branch, repeating the same check does not independently sharpen the posterior, check-then-call intersects thresholds, and river boundary tests are exact. The public-node replay and full output are saved in `replay.json` and `validation.txt`.

The action-semantics audit (`audit.py`, `audit.txt`) processed **926 opponent actions** from all archived training traces of 0003, 0004, and 0009, finding zero unsupported sizes or contradictory within-street action intervals. A full posterior replay of the weakest leader match processed **105 postflop observations and 85 opponent observations**, with no support conflicts, zero-evidence recovery, or unmodeled actions (`posterior_audit.json`). These checks consume only archived public observations and do not execute an evaluation game.

One allowed development evaluation completed both declared seeds, saved in `dev/20261030-032825/`:

| Dev seed | Chips/hand | Completed hands | Actions | Wall seconds | Illegal actions / errors |
|---|---:|---:|---:|---:|---:|
| 901 | 503.75 | 100 | 207 | 497.3 | 0 / 0 |
| 902 | 273.25 | 100 | 181 | 479.4 | 0 / 0 |

Both ended with `game_done`, using zero S1/S2 calls. The independent-match mean is **388.50 chips/hand**, with sample standard error **115.25 chips/hand across two matches**. The leader's archived scores on the same dev seeds were 574.50 and 214.25, mean 394.375. The changes are -70.75 and +59.00 chips/hand; the mean difference is -5.875. This is exploratory evidence of near-flat performance, **not evidence of strategic superiority**. Completion and numerical consistency support the implementation, while the score hypothesis remains unconfirmed. These are not training results, and no held-out results were accessed.

The digest and large settlements were inspected before finalizing. Dev-901 hand-9 retains the parent's turn value shove with Jd7c on JhKs5h8c after the opponent checks: the new estimate is .8332 equity against callers and the revealed Ah9h draws out on 8h. Dev-902 hand-81 loses the unchanged preflop AhKh shove to 5d5h. These show that large-pot variance and profitable-looking wagers that lose remain; the candidate does not encode avoidance of known cards or outcomes. In dev-901 hand-11, the checked-street posterior instead selects successive 300/750/2500 value bets and earns 3750 chips, illustrating retained moderate-size extraction.

No policy retuning followed the development result, and no second dev evaluation was used. Runtime was lower than the archived parent dev times but machine contention differs, so that comparison is not a controlled speed benchmark. The exact rank/table differential checks establish semantic equivalence of the supporting evaluator optimization. A future iteration should isolate correlation from uncertainty-width changes or test future-betting value without losing completion; these two dev matches do not choose between those explanations.
