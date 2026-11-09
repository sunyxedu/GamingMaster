parent: 0003

## Hypothesis

The public belief rollout controller is strong because it keeps all card and prompt bookkeeping local, but its Monte Carlo objective is too smooth near the end of a game. In the traces it sometimes spends a decisive $8 turn on another engine card, or chooses a low green card after a pile is already close to empty. Seed 1101 is the clearest example: the controller bought Market/Merchant/Militia through the early game and then repeatedly evaluated engine cards as competitive with scoring. A sampled hidden deck can make a delayed engine look slightly better even when the real game will end before that engine pays back. The leader also spent its only Astra call waiting for advice that never arrived, so the useful work is the local controller and exact public state.

## What changed vs. 0003

I retained the leader's public-only forward model, legal prompt handling, and bounded rollout search. Before search at every buy prompt, the harness now enumerates each currently legal gain, applies the exact Dominion end condition (Province empty or three piles empty), computes visible final VP including Gardens, Curses, and the turns-started tie-break, and immediately takes a purchase that ends the turn while winning. This deterministic gate cannot be overturned by hidden-deck sampling. I also increased the local scoring pressure for Province/Duchy and Gold, and keep Province in the rollout candidate set even when its heuristic score is low; the model still handles broad archetype selection when available, while code owns the arithmetic and finish decision.

## Evidence that would confirm or refute it

Confirmation would be a higher margin on games where the old controller lingered on engine purchases with four or fewer Provinces, with no increase in illegal actions or truncations, and traces showing `exact_finish` before a winning pile-ending buy. Refutation would be unchanged or lower margins despite the same openings, or cases where the gate ends on a visible pile depletion but loses because the public VP/tie calculation is wrong; those traces would identify a scoring or end-condition bug to fix before further strategy changes.
