parent: 0006

## Hypothesis

Candidate 0006 improves legality with short transaction sequences, but its executor still treats development purchases and Longest Road projects as ordinary portfolio options. The archive’s losing traces show the cost of that choice: candidate 0002 seed 102 bought four development cards and finished at 3 true VP after 43 turns; seed 103 bought three and finished at 4 after 84 turns. In the same family of traces, roads and a Knight were spent while the agent had no completed settlement/city conversion. The failure is therefore capital allocation, rather than stale IDs or model syntax.

This candidate tests a structural **VP conversion firewall**. Until the agent is near the finish, it commits spare resources to a live settlement or city project and suppresses development-card purchases and award-only roads when those actions would consume resources needed for that project. Development is allowed when it immediately secures Largest Army, supplies a missing one-point win, or no viable board conversion exists. The firewall is recomputed from every observation and is released on explicit route loss, a race change, or sustained progress stagnation, so it can recover on boards where expansion is blocked.

## What changed vs. the parent

The live-ID compiler, graph route search, maritime-trade construction, short Astra sequences, postcondition checks, and `ctx.progress()` failure detector are inherited from 0006. This harness adds a conversion-first gate to the project ranking and executor:

* settlement/city projects receive a protected-capital bonus and become the active commitment whenever a legal route exists;
* buying a development card is rejected while a funded or reachable settlement/city route exists, except for an immediate Army/VP finish or after the route has been invalidated;
* award-only road projects are similarly disabled before the agent has a concrete conversion, unless the award itself reaches 10 VP;
* after many turns without score or observation progress, the gate opens and Astra is asked to choose a recovery mode.

All checks use current legal actions and resource ledgers; no coordinates or seed-specific sequence is hard-coded.

## Evidence that would confirm or refute it

Confirmation would be fewer development purchases before the first expansion, more completed settlements/cities, higher minimum terminal VP, and fewer losses in which the opponents win while our public VP remains 3–5. Refutation would be repeated end-turn loops because the gate blocks the only viable action, or lower VP on seeds where a Knight/hidden VP race was actually the shortest path. Traces should show every suppressed purchase paired with a live conversion project, and the gate should open after route loss or `ctx.progress()` stagnation rather than locking permanently.
