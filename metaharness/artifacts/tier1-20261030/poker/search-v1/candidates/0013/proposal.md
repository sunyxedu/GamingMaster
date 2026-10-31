parent: 0012

## Hypothesis

Candidate 0012 is the strongest complete design, but its traces still show repeated oversized postflop commitments: 16–26 `raise_to` actions above 3 pots per 100-hand match. The recursive planner correctly models heuristic-v1's price-sensitive calls and fold branch, yet a medium-strength posterior can still select a stack-sized wager because fold equity is booked immediately while later variance is not represented by the fixed match denominator. Against a frozen caller, a smaller bet often extracts nearly the same calls and keeps losing outcomes bounded. A selective exposure gate should improve net chips per required hand by retaining large bets only when the posterior is clearly nut-heavy.

## What changed vs. parent

I copied 0012's completion-aware anytime turn-to-river planner, exact card evaluator, public range updates, shared-action response calculations, and legal action adapter. After each planner returns, `exposure_gate` examines the already selected current-street total wager. Bets costing at most three pots are untouched. A larger bet is preserved when called-equity (or, where unavailable, range equity) is at least 0.78. Otherwise it is clipped to the largest legal total no larger than 1.5 times the current pot plus the call price, with ordinary fold/check/call fallback if that bound cannot reach the minimum raise. The gate runs after planning and does not inspect seeds, score, or hidden cards; all hypothetical branches remain the parent's one-policy-per-public-node calculation.

This is deliberately a structural risk-control experiment rather than a parameter retune of the opponent model. It leaves premium preflop jams and strong-nut postflop value bets available, while limiting medium-equity all-ins that can dominate a match block. It also preserves 0012's wall-time reserve and anytime fallback, so the episode can complete even when optional lookahead is interrupted.

## Evidence that would confirm or refute it

Confirmation would be clean completion with fewer >3-pot actions, similar or better chips per hand on independent matches, and no increase in illegal actions. The strongest confirmation is improvement on seeds where the parent made a large bet with called equity below 0.78 and lost a large pot; one hand alone is not sufficient evidence. Refutation would be a substantial score decline caused by losing profitable fold-equity or thin-value opportunities, or a gate that clips legal raises too often and increases passive losses. Since the gate is intentionally conservative only above three pots, unchanged small and medium wagers provide a direct control for whether any change comes from exposure rather than the parent planner.

## Concrete trace motivation

The 0012 traces contain 16, 26, 18, and 21 actions above the three-pot cost threshold on seeds 2207, 1103, 4409, and 3301 respectively, while the candidate still averages only 548.84 chips/hand and completes all hands. The parent proposal's own runtime work establishes that completion is feasible; this proposal tests whether the remaining gap is variance from medium-strength oversized bets. A future iteration can inspect gated decisions and compare their modeled called equity with realized settlements.
