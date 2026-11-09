parent: 0007

## Hypothesis

The leader's remaining ceiling is partly caused by sending science to labs without
checking what the active technology can consume.  In the seed-202 trace, the late
factory repeatedly spends its transfer allowance on lab insertions while labs still
report `missing_science_packs`; the final ledger has 389 automation packs and 210
logistic packs produced, but only 362 and 208 consumed.  The leader's service loop
loads both pack types into every lab in fixed rounds.  During red-only research this
creates an avoidable remote transport queue, and during mixed research it can put the
wrong pack in front of the needed one.  A research-aware allocator should preserve
transfers for assembler inputs, fuel, and the currently required pack.

## What changed versus the parent

I copied the leader's proven map-relative construction, three coal sources, two red
and two green assemblers, four laboratories, recipe-derived worker scheduler, fuel
reserve, and strict asynchronous exception advisor.  The only policy experiment is
the lab router: it reads the live `research.current` and matching
`available_research` ingredients, and distributes only the science-pack types named
by that research.  It retains the parent's all-pack fallback whenever the adaptor
reports the `engine` sentinel, an unknown target, or malformed ingredients.  Packs
are still distributed evenly across labs and assembler outputs are still drained so
production cannot block.  No seed coordinates, IDs, grants, or model-generated
actions are added.

## Evidence that would confirm or refute it

Confirmation is fewer turns exhausting the 100-item transfer quota, fewer labs with
`missing_science_packs`, and equal or higher engine-produced and consumed science,
especially green consumption, with unchanged construction timing and no failed
operations.  Refutation is any increase in missing-pack time, lower lab consumption,
or green/automation production falling because the current research field is not a
technology name and the fallback is invoked too often.  The controller trace records
raw lab statuses, research state, production, consumption, transfers, and results so
the routing decision can be checked against those outcomes.

This is a structural transport-policy test rather than a recipe-count or coordinate
tweak.  Preflight is the required contract check; no full evaluation is run here.
