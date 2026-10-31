parent: 0014

## Hypothesis

The remaining loss is partly caused by the action-envelope limit rather than
missing recipes. Candidate 0014's coordinated bill improved the green branch,
but its seed-202 trace records 16 green `full_output` snapshots and the
controller repeatedly reaches 31 operations in one batch. Its service pass and
construction pass can consume operation slots before the producer feeds belts
and inserters to the logistics assemblers. Candidate 0007 reached the cap only
twice, while the 0014 change lowered green ingredient shortages at the cost of
more crowded batches. A late construction action is worth less than one turn
of scored science when green machines already exist.

## What changed versus the parent

I retain 0014's map-relative layout, fuel reserve, research and lab service,
joint affordability compiler, pending-output bookkeeping, and exception-only
asynchronous Astra repair. The only policy change is in the task executor:
before any logistics assembler is observed, it may use the parent's 28 task
operation slots; after one is commissioned, construction stops at 24 slots.
The four reserved slots give the subsequent production/service code room for
machine inputs, science output extraction, lab distribution, and fuel. The
limit is still below the 32-operation contract ceiling, so malformed or
overlong batches are not introduced and early bootstrap timing is unchanged.

## Evidence that would confirm or refute it

Confirmation is fewer 31-operation batches, fewer green `full_output` and
`item_ingredient_shortage` states, and equal or higher engine-produced and
lab-consumed logistic science, with no increase in failed operations or delayed
first green production. Refutation is a construction backlog that delays the
second green block enough to lower total science, or unchanged shortages and
operation-cap frequency despite the reserved slots. The controller trace
records goal progress, raw machine statuses, operation results, production and
lab-consumption ledgers so this single scheduling change can be separated from
recipe or geography effects.

Preflight passed at 64.93 points and 37 turns with no model calls, failed
operations, or contract exception. No full evaluation is run by the proposer.
