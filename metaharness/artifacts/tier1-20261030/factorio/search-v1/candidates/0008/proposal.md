parent: 0002

## Hypothesis

Specialize slow assembler capacity in green science and use spare serial-worker time for red supplementation. A green pack requires six seconds and earns two engine-production points, while red requires five seconds and earns one. The expensive engine lane should therefore make green, with the worker helping balance matched red/green consumption. This changes the boundary between manual work and engine production rather than refining leader 0005's two-red/one-green architecture.

Archive evidence: parent 0002's worst training seed 202 ends at 2232.06, with 232 red packs in stock at turn 127. Leader 0005's worst seed 202 ends at 3728.06. At turn 127 its stock has 181 iron plates, 86 red packs and zero copper plates; two of three labs report `missing_science_packs`, while its green assembler reports `full_output` with two packs. Counters show 117 engine green packs and 181 consumed green packs. This confirms both green scarcity and a productive role for the paid manual worker, but its specialization may be backwards. Evidence comes from `candidates/0002/trace_seed202.jsonl` and `candidates/0005/trace_seed202.jsonl`.

## Change from parent

Retain verified map-relative burner smelting, pump-relative power, runtime-recipe compilation, complete action parsing and bounded asynchronous exception advice. Replace three red assemblers with one red and two green assemblers, backed by five iron smelters, two copper smelters, two coal drills and five labs. Locations depend only on the public current map.

The worker scheduler reserves paid inputs and up to 29 seconds of already-affordable jobs, never spending pending outputs. It maintains green intermediates and supplements red packs when laboratory supply requires them. The next construction recipe reserves its direct ingredients, preventing production from repeatedly spending a lab's gears and circuits. Construction and fueled production continue concurrently.

Science extraction, fuel and evenly distributed laboratory supplies take priority over surplus plate stockpiles under the shared 100-item transfer quota. Research uses observed availability and science costs, prioritizing required unlocks and then weighted consumption per laboratory second. Full production, consumption, worker-time, transfer and research telemetry is recorded. Astra is an asynchronous exception advisor; arbitrary model output is parsed strictly, errors are caught, and stale plans are discarded.

No manual product is counted as engine production. Success depends on real powered laboratories consuming the packs. No seed-specific IDs or coordinates are encoded.

## Confirmation / refutation

Confirm if green pack production and paired laboratory consumption increase, red surplus falls, and the overall score exceeds the parent without errors. Refute if manual red supplementation crowds out green intermediates, construction delays commissioning, the 100-item quota cannot support balanced raw materials, laboratories lack power, or green production outpaces usable red supply. Dev and exact-file preflight evidence will be appended after testing.

## First dev episode and evidence-based refinement

The initial harness completed dev seed 9001 (`dev/20261030-015754`) in 128 turns / 216.9 wall seconds with **3493.0** points, zero System-2 calls, zero failed operations and no episode error. Components were **1601 red**, **1872 green**, **20 bootstrap**. All five labs, seven smelting pairs and three assemblers were built by turn 80. At turn 127, real production was 237 red / 192 green packs and laboratory consumption was 336 red / 181 green packs. Final measurements include the last charged turn.

The trace exposed two correctable constraints within this hypothesis. With only one red assembler and no early manual supplement, green research did not complete until turn 75. At turn 60 the worker used only 210 ticks, leaving time for paid science. The supplement now begins with the first red assembler instead of waiting for green commissioning. Meanwhile, 32 of the 53 turns from turn 75 onward exhausted the transfer quota. Several assemblers had empty inputs despite healthy ingredient stock because surplus plates were collected first. Assembler feeding now precedes plate collection, and labs receive multi-turn refills rather than single-pack top-ups. A bounded eight-turn commissioning grace avoids starting a long red-only project just as the first green packs arrive; if they do not arrive, research selection resumes.

These changes keep the same machine/worker specialization. They are not a claim that the first dev score measures the refined file. The first preflight passed at 64.77 in 36 actions. Offline checks cover arbitrary malformed replies, four archived layouts, input reservations, and all 128 first-dev economic snapshots.

## Final dev evidence

The refined harness completed the second allowed dev episode on seed 9002 (`dev/20261030-020752`) in 128 turns / 219.0 wall seconds at **4020.0**, with zero System-2 calls, zero failed operations and no episode error. The score trace shows sustained gains through the final turn. The new assembler-feed ordering and earlier red supplement preserved production while green commissioning progressed; green production began by turn 73 in this seed. This is evidence for the hypothesis on a different seed, not a training-score claim.

Both dev runs are deliberately retained: the first unrefined split scored 3493.0 on seed 9001, while the refined file scored 4020.0 on seed 9002. The seed change prevents treating the difference as a controlled ablation. The evaluator should test whether the specialization generalizes across its training and unseen seeds.
