parent: 0017

## Hypothesis

The recent candidates improve visible battle arithmetic and protect a 2000 DEF wall, but they still treat several profitable attacks as interchangeable. In this pool the opponent has a single Mirror Force, and the local horizon omits hidden back row by design. When two or more attackers are available, committing the strongest first can let Mirror Force erase the whole board before the agent learns that interaction exists. The archived traces repeatedly show the leader ending games with several face-up attackers and unknown back row; this is the remaining uncertainty boundary after the 0017 defense fix.

## Change

Starting from 0017, add a narrow hidden-back-row probe. At a current Battle Phase menu with at least two face-up attackers, at least one opponent face-down Spell/Trap, and no known-board lethal, choose the least valuable attacker that has a profitable visible attack. This creates a reversible first attack and preserves the stronger attackers if a mass-removal response appears. The probe is skipped when only one attacker is available, when all visible attacks are losing, or when the opponent has no back row. A known profitable effect or verified Rush expansion still takes priority.

Model actions are subjected to the same fresh-menu bait check, so a plan cannot bypass the uncertainty guard. No hidden card identity or resolution is inferred: after every attack the native observation decides whether the board changed, and the existing token, progress, cycle, position, Rush, and counterattack audits re-plan on interruption. The change is deliberately local for the final iteration and leaves tribute/deployment and chain rules untouched.

## Confirmation / refutation

Confirm if traces show low-value attacks probing unknown back row before high-value attacks, while lethal and single-attacker lines remain unchanged, with zero invalid actions or truncations. Strategic confirmation requires improved wins in training and held-out strata where Mirror Force or other hidden interaction matters. Refute if the probe spends an attacker that should have been preserved, delays a native win, or causes passivity against empty back row; also refute if the same mass-removal losses remain despite the changed ordering.
