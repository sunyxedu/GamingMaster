parent: 0000

## Hypothesis

The baseline's low score is caused less by a lack of Catan rules in Astra and more by asking a slow model to choose every low-level action from a huge, raw observation. The trace for seed 106 begins with a 50-option settlement list and then repeats long full-state prompts; across the archive the baseline averages 38.5 actions but only 1.88 VP, and seed 106 is the weakest at 1 VP. This encourages arbitrary setup sites and spends the 40-call allowance before a coherent build race develops.

## What changed

Candidate 0002 is a code-first harness. It computes pip-weighted production, diversity, wheat/ore and wood/brick complements from the public board and chooses initial settlements and roads directly. It also commits singleton, roll, discard, and affordable build actions without a model call, prioritizing city, settlement, then road. Astra receives a compact but race-aware observation only for robber, maritime/development choices, and genuinely ambiguous decisions. Replies are parsed for a legal `aN` ID and malformed/failed calls fall back safely to the current legal list. The persistent plan is carried through the Astra session.

## Evidence that would confirm or refute it

Confirmation would be fewer Astra calls and actions, higher early public VP, and especially improved weak seeds without an increase in illegal-action or timeout errors. If scores remain near 1–2 VP despite strong opening production, the failure is likely strategic (resource timing, opponent blocking, or development/robber handling) rather than prompt overhead; if setup improves but later games stall, the next harness should add explicit resource ledger and immediate-win checks.
