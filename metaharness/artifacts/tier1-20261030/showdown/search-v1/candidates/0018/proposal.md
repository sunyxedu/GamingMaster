parent: 0015

## Hypothesis

In `0012/trace_seed333667.jsonl`, the ledger records Boomburst removing 54.2% of Raichu and 64.5% of Chimecho, followed by a fatal switch into the same threat. This is direct evidence that the search needs to increase uncertainty after a demonstrated hit.

The search’s generic hidden STAB branch is useful before any attack is observed, but its fixed 8% weight remains too weak after the protocol demonstrates that the active opponent can remove a large fraction of a Pokémon’s HP. In that situation, a later replacement decision still averages mostly over optimistic revealed moves and can walk into the same threat. The hypothesis is that an evidence-triggered increase in uncertainty is safer than a globally larger hidden-move prior.

## What changed versus the parent

Candidate 0018 keeps the parent’s two-turn public-only search, legal-action filtering, model failure recovery, and empirical incoming-damage floor. It adds an episode-local calibration flag keyed by the exact revealed opponent identity. While processing each transition, the harness reads direct untagged `|-damage|` protocol lines against the active p1 Pokémon. If one hit removes at least 18% of that Pokémon’s exact HP, the opponent is marked calibrated. When the opponent still has unrevealed moves, its synthetic STAB reply receives weight 0.22 instead of 0.08; before evidence it remains 0.08. The branch is still a counterfactual search hypothesis and is never presented as a revealed fact.

This is deliberately narrow: a single substantial observed hit changes only future reply uncertainty for that opponent, while known move estimates and normal action scoring remain unchanged.

## Evidence that would confirm or refute it

Confirmation would show that after a large untagged hit, replacements avoid immediate repeat KOs and the search logs a higher-risk branch, especially on losses where the opponent later reveals a strong coverage move. Score should improve without increasing illegal actions or model failures. Refutation would be frequent over-switching after ordinary strong but manageable hits, or degradation on positions where the four revealed moves already explain the damage. The 18% threshold and 0.22 weight should then be reduced or conditioned on repeated evidence.

