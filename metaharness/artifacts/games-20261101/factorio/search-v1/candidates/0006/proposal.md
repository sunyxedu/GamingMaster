parent: 0002

## Hypothesis

The leader's deterministic controller solves a real execution bottleneck and reaches a high score, but it leaves most of the attainable objective unused: every training episode's score is almost entirely automation science (mean 2243.25) and it never makes logistic science. The missing green chain is a structural opportunity, not a model-planning failure. A controller that treats the remote inventory as a deliberate transport layer can add logistic science without belts/inserters on the map, then consume it in labs for the higher logistic weight.

## What changed

This harness keeps the verified seed-relative bootstrap and closed-loop servicing, but adds a second production tier: two additional powered assemblers are assigned to `logistic-science-pack`, two additional labs receive both pack types, and poles extend the same electric network. The executor compiles and batches the exact inserter and transport-belt inputs from observed recipes, feeds green assemblers from stock, and changes research selection to permit Logistics once green packs exist. Recipe availability is checked before setting recipes, so a locked recipe cannot create an action-error loop. Astra remains an exception-only repair advisor with strict parsing and bounded asynchronous calls.

## Confirmation / refutation

Confirmation would be early automation production followed by nonzero engine-produced and lab-consumed logistic science, with no increase in action errors or stale turns. Refutation would be repeated placement/power collisions, inability to unlock or set the green recipe, or green input crafting starving the red chain enough to reduce total score. The key trace evidence is the score breakdown: candidate 0002 averages 2243.25 automation points and zero logistic points despite 128 charged turns, so a successful green tier should be visible directly in the production and lab-consumption ledgers.
