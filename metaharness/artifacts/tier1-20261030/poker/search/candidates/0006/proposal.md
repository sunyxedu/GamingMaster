parent: 0002

## Hypothesis

0002 showed that a very fast script can finish the match, but its coarse made-hand
and draw categories lost too much value (mean -81.3 chips/hand). The leader 0001
recovers value with a range solver, yet it spends substantial engineering effort
replaying a posterior and asks Astra for state audits. I test a different structural
claim: against this frozen opponent, a stable random-unseen-card equity policy is a
better baseline than coarse categories, while Astra is most useful as a sparse
high-level posture selector. Aggregate public action frequencies can distinguish a
calling posture from a folding posture without pretending that one showdown reveals
the opponent's private range.

## What changed

Candidate 0006 computes deterministic Monte Carlo equity from the visible cards on
every turn, uses position-aware heads-up preflop cutoffs, and chooses value sizes and
pot-odds calls from exact supplied legal candidates. It caps ordinary value bets at
pot-sized fractions before the river and requires much stronger equity before raising
a live wager, addressing the full-stack overbets visible in 0001's worst seed. It
tracks opponent actions from public history and asks Astra only at the beginning and
around hands 35 and 70 for bounded `balanced`, `pressure`, or `cautious` parameters.
Every model reply is optional, parsed defensively, and cannot post an unvalidated
action. Each scripted step checks progress; rejected or stale actions become explicit
failure evidence included in a later Astra replan, then the harness recovers to a
legal passive action.

## Evidence that would confirm or refute it

The hypothesis is supported if 0006 completes all 100 hands with no illegal actions
and improves the weak training seeds as well as the mean, while using at most three
model calls. It is refuted if random-card equity still produces systematically
negative results, if posture changes do not help across independent seeds, or if
the policy needs the posterior range inference and response-conditioned sizing used
by 0001 to avoid costly river calls and raises.
