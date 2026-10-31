parent: 0007

## Hypothesis

The leader's main late-game failure is the shared 100-item transfer quota, rather than a shortage of engine capacity. Its turn-127 ledger has more than 2,500 engine-produced iron plates but essentially no immediate iron stock, and 57 turns spend all 100 transfers. The service loop repeatedly extracts furnace results into stock even when no machine or construction step needs those plates. That creates a needless two-hop buffer: furnace -> stock -> handcraft/machine. Result inventories are finite but large enough to hold several turns of smelting, so a burst policy should preserve quota for science packs and assembler inputs while keeping furnaces below `full_output`.

## What changed versus the parent

I retain candidate 0007's verified seed-relative layout, coal expansion, two green assemblers, laboratory distribution, recipe-derived worker scheduler, and exception-only Astra repair. The structural change is the material-transfer policy. Furnace results stay in their native result inventories while stock is above a demand threshold; when stock is genuinely low, the controller withdraws a larger refill, and when a result slot approaches capacity it drains only the excess. Ore insertion and fuel remain demand-critical. This tests whether fewer, larger plate withdrawals reduce quota exhaustion and increase real lab/assembler feeding without changing machine count or research choices.

## Evidence that would confirm or refute it

Confirmation is a lower fraction of quota-exhausted turns, fewer `full_output` furnace statuses, and higher consumed automation/logistics science with the same or higher bootstrap and engine production. Stock should show larger plate bursts while furnace result buffers oscillate below capacity. Refutation is any sustained furnace `full_output`, assembler starvation caused by delayed withdrawals, or lower science consumption despite preserved transfers. The controller logs raw inventories, transfer remaining, production, lab consumption, and failures so this distinction is observable.

## Diagnostic dev result

A single dev episode on seed 9001 completed all 128 turns without an episode error at **4076.0** (one late Astra call was cancelled after game completion). The result is close to, but above, the parent's 4030 development trace on seed 101; this is not a controlled same-seed comparison. The raw ledger reached 2,591 engine iron plates, 1,025 copper plates, 396 automation packs and 215 logistics packs produced, with 365/211 consumed at turn 127. Immediate stock still reached zero iron on several late turns, so the policy did not remove every quota bottleneck; its value should be judged by transfer exhaustion and furnace statuses on the evaluator's seeds rather than by the scalar alone.
