parent: 0001

## Hypothesis

Candidate 0001 fixed the leader's latency problem, but its worst training match
lost 559.76 chips/hand despite completing all 100 hands. The trace repeatedly
shows the policy treating a pair made solely by the board as a value hand, then
continuing in inflated pots (for example, it calls after a flop raise on a paired
board with no hole-card improvement). I hypothesize that this board-relative hand
strength error, combined with automatic aggression on marginal pairs, dominates
the negative result more than preflop coverage does.

## What changed vs. the parent

I kept the parent's fast, exact-candidate control loop and broad heads-up
preflop ranges, but replaced the postflop category test. The new policy tracks
whether a pair or two-pair is actually supplied by a hole card and treats a
board-only pair as a bluff-catcher. It folds one-pair hands to large prices,
calls them only at small pot odds, reserves raises for hole-improved strong
hands, and limits semi-bluffs to modest pots with genuine draws. Every target is
mapped through the observation's candidate list, so current-street raise totals
remain legal.

## Evidence that would confirm or refute it

Confirmation is a completed match with materially fewer large-pot losses on
paired boards, zero illegal actions, and an improved mean over the parent's
training seeds while retaining its completion speed. Refutation is a similar
loss profile even when board-only pairs are checked or folded, or a broad drop
from missing thin value against the heuristic's frequent calls. That would
motivate range-conditioned equity rather than further category thresholds.
