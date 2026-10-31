parent: 0015

## Hypothesis

Parent 0015's first preflop branch can commit the whole 20,000-chip stack whenever its canonical hand equity is at least 0.60 and a raise is legal. That condition is evaluated before it checks whether the opponent has raised, so it also jams first-in on an unopened button or over a free BB check. The heuristic-v1 opponent is intentionally exploitable, but those public states provide no evidence of a premium opposing range; a 200bb preflop gamble makes one match block depend on a single deal. The same parent already has a separate premium response branch for observed reraises. I test whether restricting the all-in EV branch to histories containing an opponent raise reduces high-cost variance while retaining the value response that the heuristic's premium branch warrants.

The parent traces make this branch observable: seed 4409 records a first-in `raise_to 20000` at q=0.6729 (hand 7) with modeled jam equity 0.589; the same seed records q=0.6631 (hand 28) after an opponent raise, showing the response case that should remain. Seed 3301 has first-in jams at q=0.8550 and q=0.7917, while seed 1103 has one at q=0.6516. The lower-equity first-in examples risk nearly the full 200bb stack without an observed premium range.

## What changed vs. the parent

This harness is copied from 0015: it keeps the exact evaluator, public-action posterior, postflop sizing, raise-response fold gate, current-street legal target conversion, and zero-model-call loop. The only strategic change is the guard on the preflop value-jam calculation. It now runs only when `history` contains an opponent `raise_to`. Unopened button hands use the parent's ordinary 200-chip open/fold rule; a limped BB uses its check or 400-chip raise rule. When facing a documented opponent raise, the parent's sampled continuation-equity and EV test still decides whether a stack-sized raise is justified.

The guard depends only on public history and the current observation. It does not inspect seeds, scores, hidden cards, or terminal results, and all returned raises are clipped through the observation's legal bounds.

## Evidence that would confirm or refute it

Confirmation would show fewer first-in `raise_to max_raise_to` actions, fewer single-hand swings near 200bb, and equal or higher net chips per required hand across independent matches, with zero illegal actions and clean termination. The strongest confirmation is that the removed jams were mostly on unopened or checked streets and had lower realized value than the retained raises over a match block. Refutation would be a score decline caused by losing profitable first-in isolation against the heuristic, or by turning strong free-check hands into small pots; traces should compare the number and context of preflop all-ins before attributing a change to variance control. A single showdown cannot decide the hypothesis.

One permitted development run on seed 901 completed all 100 hands in 177 seconds, scored 835 chips/hand, used zero model calls, and had zero illegal actions. Its harness trace contains 43 preflop `raise_to 200` opens and no first-in 20,000-chip jam, while retaining stack-sized postflop value raises and the response branch. This is runtime and action-distribution evidence only; one independent seed does not establish a training improvement.
