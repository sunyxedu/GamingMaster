parent: 0014

## Hypothesis

The leader's remaining river edge is partly lost at its abstraction boundaries. Its solver knows heuristic-v1's exact river equity thresholds, but its observation updater smooths those thresholds and retains a .003 likelihood floor for impossible holdings. Its legal action search then considers only a coarse pot-fraction menu, although the game accepts every integer total. A more consistent terminal decision should condition exactly on observed river actions and evaluate prices immediately around the opponent's exact call/fold thresholds. This is a localized river improvement, preserving the leader's uncertain earlier-street model and completion safeguards.

The leader completes all four training matches and averages 799.362 chips/hand. Its weakest match is `0014/trace_seed2207.jsonl`, at 418.57. Hand-54 supplies a concrete pricing diagnosis: hero 6d8c on 9c3h4d7hJs, pot 500, checks available, selects `raise_to 750`, equity .0383, river EV 99.94. With exactly the same holding posterior, the new boundary candidates select `raise_to 644`, modeled EV 115.60. This reduces the price of a bluff by exploiting where the documented caller actually changes action; no actual hidden holding or future observation enters the calculation.

Hand-47 diagnoses the separate input inconsistency: Kd6c on TsKhAc5s4c faces a river bet of 2345 into 3126. That rounded 0.75-pot size identifies exact-equity [.62,.85), yet the parent assigns 1.9165% posterior mass outside that support. Exact conditioning narrows 990 holdings to 224 and changes estimated hero equity from .5354 to .5326. The chosen `raise_to 14069` is unchanged. The largest archived losses include shoves without a preceding opponent river action; this proposal does not claim to eliminate all large-pot losses.

## Changes from the parent

Observed river actions receive exact Bayesian likelihoods using the parent's existing exact equity table. Public wager history reconstructs stacks, current-street bets, minimum full-raise increments and whether the opponent has already raised. Bet evidence matches integer totals after minimum/stack clipping; indistinguishable branches contribute their summed probabilities. Calls retain high-equity holdings when another raise is unavailable or the opponent has already raised. Impossible holdings get zero weight. Unsupported or zero-evidence observations retain the board-compatible prior and increment explicit diagnostics rather than crashing.

Only the current river decision gets extra size candidates. For target t, hero's prior street wager h, villain's prior wager v, and current pot P, the opponent's call price is `(t-v)/(P+2*t-h-v)`. Its exact equity is k/1980: 990 uniform opposing holdings with half-credit ties. Integer arithmetic solves each call/fold equality, and the legal totals immediately below/above it join the menu. The existing three-pot exposure-gate boundary also joins the menu. Targets are bounded by reported legal limits and the effective matched stack. This is an expanded finite search, not a claim of globally solving every recursive raise-response breakpoint.

The inherited menu is evaluated first. Added sizes must improve its incumbent by the same existing risk penalty and action-change margin. Opponent reraises still use the parent's shared-information-set response solver; no action is selected separately for a hypothetical hidden holding. Hypothetical river searches inside turn lookahead keep their original menu to bound runtime. Preflop opens/jams, flop/turn inference, evaluator, resource controller, legal action conversion and wall-time reserve are retained. There are no model calls or background workers.

## Evidence and falsification

Confirm the mechanical hypothesis through exact public-action support, legal integer targets, nondecreasing modeled utility relative to the old menu under the same posterior, unchanged nonriver posteriors, clean preflight and full match completion. Confirm the scoring hypothesis only through terminal net chips per required hand across independent match blocks. Refute it if exact evidence conflicts with observations, finer prices exploit an inaccurate earlier-street posterior, larger action search loses completion, or independent scores decline. Better modeled EV is not proof of actual expected-chip improvement; earlier-street Monte Carlo and likelihood approximations remain.

## Validation

The initial exact-evidence-only version passed preflight and all four archived match replays: 686 nonriver posteriors were identical, and all 93 river actions remained unchanged. That flat action evidence motivated the boundary-pricing refinement, rather than any hand-specific outcome rule. The first permitted dev run completed both 100-hand matches with scores 1090.50/822.74, zero errors/illegal actions, and no inference conflicts (`dev/20261030-054802`). These results belong to the preliminary version, not the final pricing implementation. Its digest and large settlements were inspected; preflop and earlier-street large losses remain outside this experiment's scope.

`validate.py` checks threshold inclusivity, integer rounding, minimum-size aliases, stack clipping, all-in calls, prior raises and contradictory-evidence recovery. Across the leader's 317 available public settlement histories, all 80 opponent river actions had supported sizes and all 20 legally revealed holding/action pairs agreed with direct enumeration of 990 opposing hands. Automatic settlements mean these histories do not cover all 400 hands; missing cards were never reconstructed.

The final weakest-match replay preserves all 173 nonriver posteriors and changes seven of 22 river decisions, with zero unsupported actions or zero-evidence recoveries (`validation_final.txt`, `validation_seed2207.json`). `check_pricing.py` exhaustively checks every legal integer bet across 120 generated small-stack distributions with a passive opponent response. Expanded-menu utility never falls below the inherited menu; the remaining gap to the exhaustive optimum stays within the inherited action-change margin. This verifies the leaf pricing arithmetic, not global optimality when reraises are available (`pricing_validation.txt`). A preflight of the pricing version passed with 122 legal actions and zero model calls/errors; final verification and development results follow below.

## Final development results

The final file passed preflight again with **117 actions, zero model calls and no errors** (`preflight_final.json`). Its SHA-256 matches the second development evaluation's manifest. Both final episodes completed all 100 hands, with zero illegal actions, errors, unmodeled river actions or zero-evidence recoveries:

| Dev seed | Chips/hand | Actions | Wall seconds | End |
|---|---:|---:|---:|---|
| 901 | 1103.65 | 210 | 549.0 | game_done |
| 902 | 822.74 | 191 | 539.4 | game_done |

The mean is **963.195 chips/hand**, with a sample standard error of **140.455 across two independent match blocks**. These are development results, not training or held-out results. Compared with the preliminary exact-evidence-only version, changes were +13.15 and 0.00 chips/hand, but wall-dependent compute allocation prevents treating this as a perfectly controlled pricing ablation. The parent's archived dev-901 score was 1119.00, so this run does not establish superiority over the leader. No held-out games or full evaluation were run, and exactly two permitted development invocations were used.

Final traces contain 51 river decisions and 23 exact public-evidence updates, with at most 170 added root sizes per decision. The digest and settlements show both gains and costs: dev-901 hand-11 increases its river bet from 4000 to 4947, while hand-3's losing bluff grows from 500 to 536. Dev-902 still loses 20000 with the inherited preflop AhKh shove in hand-81 and 20000 in hand-96. Improved model pricing does not remove poker variance or earlier-street range error. All final artifacts are in `dev/20261030-060343/`; no strategy changes followed that result.
