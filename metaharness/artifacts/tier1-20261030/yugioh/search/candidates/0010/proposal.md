parent: 0008

## Hypothesis

The leader's chain arithmetic and target transactions address tactical effect errors, but its local policy still makes most ordinary deployment and battle decisions with independent static scores. The remaining losses likely come from sequencing: a choice that looks locally positive can consume a body, clear a useful response, or end a turn before converting board advantage into pressure. Candidate 0008 called Astra only at narrow exchange disadvantages (at most two ordinary reviews per duel), so it does not routinely compare the whole current turn's remaining options.

The worst training seed remains 1111 at -0.37, despite candidate 0008's 0.54 mean. Its three duel scores include both a win and losses, showing the chain-aware script is not scenario-independent. In `candidates/0008/trace_seed1111.jsonl`, step 38 is an own Main Phase 1 menu with “set spell/trap Book of Moon” and “enter Battle Phase,” but the harness chooses “end turn”; the next own main phase at step 42 again ends with Book of Moon available. These are concrete unused options, not proof that setting Book would have won. Candidate 0008 also uses only eight System-2 calls per episode, while native duels commonly span many own turns. The evidence supports testing broader turn sequencing, not another combat weight adjustment.

## Change under test

Retain candidate 0008's legal-action validation, chain projection, target binding, consequence audits and public combat allocator. Change Astra's review gate from only reactive disadvantage windows to one review at the start of each agent-controlled Main Phase 1 when meaningful strategic choices exist, plus the existing failure repairs. Astra returns one exact current action; there is no future token queue or persisted recommendation. The next native observation returns control to the local policy, which can replan from the fresh menu. The model receives the local action, projected chain consequences, battle estimate, remaining Normal Summon status when observable, and recent failure evidence. Local execution remains authoritative for legal menus and target subsets.

The additional calls are bounded by the existing per-episode model budget and wall-clock guards. Opponent-turn responses and mandatory prompts remain local unless a detected failure requires review; slow inference must not block routine responses.

## Confirmation and refutation

Confirm if Astra changes a meaningful fraction of main-phase deployments or battle sequencing, the live-menu priorities remain legal after opponent interaction, and training improves the weakest seeds without increasing truncations or technical errors. Refute if repeated phase reviews consume calls without changing actions, priorities become stale after responses, or the leader's scores fall because routine turn planning delays native actions. Trace review should compare each reviewed recommendation with the applied action and the next observed board; only native evaluation can establish a win-rate gain.

## Preflight

The stand-in-model preflight passed: native three-duel episode completed at
0.3667 with 407 actions and no exception. It answered no Astra calls, so this
verifies contract and garbage-output safety only; it does not test whether the
new review gate improves play. The requested training evaluation remains the
first strength measurement.
