parent: 0009

## Hypothesis

The two turn search is still spending probability mass on generic equally plausible replies after the opponent has shown its policy. In the parent trace for seed 181081, the heuristic repeats Goodra's Earthquake at steps 3, 4, 7 and 8, and repeats Gardevoir's Moonblast at steps 10 and 11. The parent `replies()` ranks by damage and then applies a fixed softmax; it does not use these public action observations, so it can value a switch or attack against an implausible branch while missing the repeated threat. This is a policy-model failure rather than a type-chart failure.

## What changed versus parent

The parent is 0009's bounded two-turn exchange search and legal fallback. I added an episode-local empirical opponent policy ledger. Every `|move|p2...|Move|` event increments a count keyed by the revealed opponent identity; `replies()` gives a capped (at most 35%) weight boost to candidate moves that this same active identity has already selected. The prior is deliberately weak, applies only to public revealed actions, and falls back to the parent's generic portfolio for unseen moves. The compact Astra prompt receives the observed counts as context, but Astra still cannot choose an action and cannot overwrite protocol facts. Switch and forced-request handling, stale-plan avoidance, damage approximations, and exception recovery remain unchanged.

## Confirmation / refutation

Confirm if repeated heuristic attacks cause search to choose safer counters, preserve fewer HP on the active replacement, and improve the weak seeds (especially the parent losses) without increasing illegal actions or episode errors. Refute if repetition is too sparse or the opponent changes moves after our action, making the prior overconfident; a flat score or more losses immediately after a policy repeat would show that generic uncertainty was the safer model. The capped boost and identity key make this falsifiable without assuming hidden sets or training-seed behavior.
