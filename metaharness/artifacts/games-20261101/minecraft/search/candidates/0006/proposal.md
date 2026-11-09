# Candidate 0006 — Code-owned milestone executor with tactical Jev interrupts

## Parent and hypothesis

Parent: `candidates/0003/` (formal rewrite parent); `candidates/0005/` is the
best-scoring comparator and supplies the concrete grounded-controller failure
trace.  The grounded-command lesson from 0003 is retained.  This is a
structural rewrite of the division of labour, not a confidence or latency
tweak.

The prior traces show that 0005 fixed the original bare-verb problem, but its
controller still lets a short-horizon selector choose the same local action
forever.  On seed 1 it posted `gather b:-3,64,-25 3` at 0.5 s and then repeated
that log's vertical neighbours (`b:-3,63,-25` and `b:-3,62,-25`) throughout the
first several minutes.  The final action histogram contains 195, 187, and 144
of those three gathers, but the episode retained an empty audited inventory and
scored 0.  In contrast, seed 0's early actions successfully gathered acacia,
crafted planks and a table, and seed 2 reached the same opening score.  Thus
parameterized commands are accepted; the failure is lack of progress memory,
phase completion, and recovery when a target is exposed but not practically
reachable.

The strongest concrete symptom is that 0005's code builds an option for a
resource whenever it is an exposed candidate, while its Jev state contains only
current options and no failed-target ledger or operation outcome.  A gather
job can therefore be selected again after a no-op/result cycle even when no
inventory change occurred.  Its Astra calls only change a `balanced`/`food`/
`iron` focus and cannot repair that loop.  The controller also asks Jev to
select ordinary crafting and navigation actions, spending fast calls on choices
that are exact recipe and lifecycle facts.

**Hypothesis:** making code the owner of a restartable milestone state machine
will convert valid actions into durable progress.  Jev should be used only for
an immediate, local threat interrupt where a visible hostile competes with work;
Astra should advise a bounded resource focus only after a measured stall.  An
operation ledger that records an inventory/progress/position change and
quarantines an unchanged command should prevent the seed-1 exposed-log loop.
This tests a different model split from 0005 rather than adding more menu
options.

## What the harness changes

`harness.py` has three cooperating layers:

1. **MilestoneExecutor (code):** computes the current phase from audited
   inventory and permanent progress (`wood/table`, wooden tools, stone pick,
   food, raw iron/furnace, shield, bed).  It performs exact recipe arithmetic,
   binds every target to a current `b:`/`s:` candidate, approaches a resource
   with a safe stand before gathering when it is not in reach, places a table or
   furnace only on an observed support, and explores new safe stands when the
   desired resource is not currently exposed.  It remembers visited stands and
   quarantines an exact action for 45 seconds after its bounded deadline passes
   without an inventory, advancement, score, or meaningful position change.
   `active` remains set across the adaptor's noop ticks, so a completed one-shot
   cannot be retried merely because the next snapshot is temporarily unchanged.

2. **ThreatGate (Jev):** code prefilters visible hostiles, conditional threats,
   lava/fire, drowning, and self hazard into at most eight complete immediate
   responses.  Jev selects among those already-legal strings, such as `shield
   on`, `attack 42`, `equip bread hand`, `eat bread`, a listed retreat
   navigation, or `noop`.  It never invents coordinates, counts, recipes, or a
   long route.  With no immediate threat there is no Jev call: the executor's
   verified build order is more reliable than a generic action Choice.

3. **Astra adviser:** at a long no-progress interval, and no more often than
   every 180 seconds, a background Astra call chooses only the token
   `balanced`, `food`, `iron`, `bed`, or `explore`.  The token is a tie-breaker
   among implemented branches; Astra cannot emit a command.  It is started
   only with at least 260 wall seconds remaining and a short prompt so a late
   answer cannot block survival.  Invalid or late text leaves the previous
   focus unchanged.

The executor retains the useful 0005 protocol: full whitespace commands are
compiled from current observations, no unobserved coordinates are embedded,
only one bounded job is allowed to own the hands, and each loop waits for a
real tick.  It additionally distinguishes an active operation from the noop
lease used to keep that operation alive.  A stale observation cancels a job;
a death uses ordinary `respawn`; no command, seed, server or privileged state
is used.

## System-1 state and question

Jev receives a deliberately small JSON state on a threat change or a three
second threat-lease expiry:

```json
{
  "self": {
    "health_band": "stable|critical",
    "food_band": "fed|low",
    "dimension": "overworld|the_nether|the_end",
    "on_fire_or_lava": false
  },
  "threats": [
    {"kind": "creeper", "distance_band": "touching"}
  ],
  "legal_responses": ["shield on", "attack 42", "navigate s:...", "noop"]
}
```

The actual state uses only derived bands and threat names/distance bands, not a
large terrain dump or arithmetic.  Candidate descriptions are encoded in the
Choice criteria, and the exact command strings are the option keys.  The one
question is:

* `response`: a `Choice` asking for the best immediate survival response from
  `legal_responses`; shield/retreat is preferred for uncertainty and hazards
  must never be attacked.

The answer bridge checks that the returned value is a current option.  Jev
probability below `0.55` vetoes an attack or retreat in favour of `shield on`
when available, otherwise `noop`.  Missing, non-string, stale, or malformed
answers use the same conservative fallback.  A Jev timeout has a 2-second
budget; it does not interrupt an existing bounded job because threat decisions
are requested only while idle.  Budget and model-unavailable exceptions retain
the platform's normal propagation semantics.

The remaining action decisions are deliberately not Jev questions: code owns
recipe prerequisites, counts, target IDs, phase transitions, cooldowns,
operation results and exploration.  This prevents Jev's known counting/math
failure mode and removes the option-order bias from routine crafting.

## Astra scheduling and fallback

Astra is launched only when the executor has had no audited inventory or
progress change for roughly 45 seconds, has no adviser already running, and
has enough remaining wall and System-2 budget.  It sees the current phase,
small inventory map, health/food, dimension, and nearby useful block summary.
The parser searches for an allowed focus token and ignores all other text.
While it runs, code continues with the current phase and its safe fallback.
If no answer arrives, the focus remains `balanced`; this cannot create a
malformed action or stale target.

## Confirmation and refutation

The hypothesis is confirmed by traces showing:

* no repeated exact gather command after its bounded job fails without an
  inventory change;
* task/result progress and audited logs, planks, table, stone, food, raw iron,
  furnace/ingots, and shield appearing in the expected phase order;
* seed 1 moving away from the water-adjacent exposed oak rather than spending
  hundreds of gathers on its vertical stack;
* positive opening and iron/shield components on more seeds, with no harness
  errors and low stale ticks; and
* Jev calls concentrated around actual hostile/hazard events, while Astra
  focus changes follow measurable stalls.

It is refuted if the executor still cycles through the same target despite the
ledger, if safe navigation never reaches the exposed resource and no alternate
stands are found, or if deterministic recipe/placement assumptions are wrong
for this adaptor.  A zero score with accepted, progressing early operations
would instead indicate that the next bottleneck is the missing Nether/End
route, not the proposed division of labour.  This candidate intentionally
makes no claim that the short 900-second evaluation can defeat the dragon; its
first test is whether robust opening and iron/shield preparation replace the
flat 0–3 opening plateau.
