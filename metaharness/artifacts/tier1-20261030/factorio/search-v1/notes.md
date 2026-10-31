
## Candidate 0002 — declarative executor (initial findings)

- Baseline traces distinguish model intent from submitted action: `place stone-furnace 43 34` became bare `place`, and the first multicommand answer lost its building operations. Test parsing against complete game grammar, including hyphenated identifiers and batches.
- Baseline seed 404 exhausted 1800 wall seconds at charged turn 1; turn utilization is a major independent bottleneck even before strategic optimization.
- Public observations verify burner drill at integer `(x,y)` facing east directly feeds a furnace at `(x+2,y)`. Seed-relative patch-center selection worked in the first dev episode.
- Runtime recipes differ from familiar prior recipes: the observed offshore pump needs gears and pipes, while poles produce two per recipe. Compile actual enabled recipe data rather than assuming historical costs.
- Preflight passed with score 5.76 and 36 actions without a model response. Long prerequisite crafting chains are normal progress; a six-turn unchanged top-level goal falsely triggered an expensive model consultation.
- Final 0002 findings: the pre-fix dev completed all 128 turns (18.85 bootstrap, no errors), but `boiler: full_output` plus `engine: no_input_fluid` exposed a one-tile steam gap. Adding the pipe and batching construction yielded 64.77 in the revised 36-action preflight. Prefer machine-state evidence over stored joules; clear observed trees in planned footprints. Synchronous advice stalled the first dev attempt, so repairs now run asynchronously with stale-plan rejection.

### Candidate 0003: persistent production and construction goals

- Compare model responses to actual submitted strings before attributing failure to strategy. Baseline seed 101 loses a four-operation batch and strips `place` arguments; its final game turn is only 2.
- Public recipe observations suffice for a dependency-frontier scheduler: reserve existing ingredients immediately and wait for completed intermediate outputs before submitting parent crafts.
- Direct east-facing burner/furnace pairs at adjacent integer centers produced genuine iron and copper on dev seed 9001. Three pairs reached score 2.13 by turn 16 without model calls.
- Pre-flight passed, but its short budget admitted no model answer; independent garbage-JSON and reservation checks supplement that limited coverage.

### Candidate 0001 — command boundary and factory supervisor
- Parent traces show model commands were not the submitted actions: `place burner-mining-drill 55 26 8` became bare `place`, and a four-operation opening became only its final mining operation. Seed 101 remained at turn 2 through repeated grammar errors. Preserve arguments and semicolon order before judging strategy.
- Exact initial observations support a seed-relative burner drill facing east into a furnace two tiles east. Public-action dev opening produced iron by turn 2 and both plate types by turn 5 (score 0.38), without a model call.
- The observation uses `entities[].inventories[role]` as item-count maps, `status_name`, and per-job `ticks/count/remaining`; compact these economic facts and sample resource IDs rather than repeating every ore tile.
- Reported model generation latency can substantially understate total synchronous waiting. The first dev call recorded 47.3 seconds but returned only after several minutes overall. Reserve wall time independently using a ctx.spawn future, then finish useful simulation before the deadline.
- Routine policy execution needs emergency fuel priority and strict finite numeric validation. Native batch operations must take priority over automatic chores at the 32-operation boundary; silently truncating the combined list would recreate the parent's failure.

- Candidate 0003 latency finding: its dev trace stopped at turn 16 / score 2.13 while the first 220-second-requested model call remained pending; the CLI later exited with KeyboardInterrupt without a digest/eval result. Independently bound waits around a single ctx.spawn inference future and reserve wall time for simulation; do not describe incomplete traces as completed science evaluations.

Candidate 0001 completion: original dev revision ended normally at score 3.9, turn 32, six S2 calls; it placed power components but boiler/engine `no_input_fluid` and lab `no_power` meant zero science. Final preflight after timeout reserves, prerequisite repair, transfer clamping and trigger-research normalization passed at score 4.48 / 37 actions. The final revision has not had a full real-model dev episode; do not attribute the original dev score to it.

## Candidate 0005 — green-science branch

- In leader 0002 seed 202, turn 80 has green science researched and three stocked labs reporting `no_research_in_progress`; turn 127 holds 232 unused red packs. Late score can be constrained by useful research demand, not pack production alone.
- First dev seed 9001 validates one green assembler plus manual intermediates: 128 turns, score 2685.06, green contribution 886, no episode error. That run is a development revision, not the final harness or a training result.
- Cheap handcraft intermediates avoid the extra inventory transfers of an automated circuit/inserter/belt chain. A scheduler must reserve paid inputs, honor output multiplicity, and keep pending outputs unavailable until observed.
- Larger fuel top-ups when inventory is low preserve operation slots; even when fewer than 100 items move, the 32-operation limit can otherwise choke construction and production feeds.
- Spare manual-worker time can make green packs for legitimate lab-consumption credit when red stock is waiting; manual packs themselves never earn engine-production points. Exact recipe duration bounds the supplement.

## Candidate 0006 — green tier hypothesis

- Candidate 0002's high score is almost entirely automation science (2243 mean points) with zero logistic science, leaving a clear structural target.
- Extending a deterministic controller is safer than asking Astra to discover green logistics under 40 slow calls; recipe and technology availability must gate every green placement and research action.
- Green science needs inserters and transport belts as real inputs; compile them from observed recipes and reserve stock before feeding assemblers.
- A second block must share the verified steam network while preserving red assemblers and labs; placement collisions and pole reach are the main refutation signals.

- Candidate 0005 final evidence: dev seed 9002 scored 3741.06 in 128 turns with zero model calls and zero failed operations; green contributed 1788. Final preflight and 128-snapshot bookkeeping checks passed. Copper reached zero near the horizon while iron/red stock remained, identifying the next supply bottleneck; these are dev results, not training scores.

### Candidate 0004 — consumption capacity and research demand

- The leader's seed-202 trace ends with 232 red packs in stock and nine in each of three labs; late research advances only three units per turn while red assemblers produce nine packs. Factory balance must include research seconds per unit, not just science production rate.
- Nine-lab dev revision (9001) completed 128 turns at 2709.01 with no error, consumed 521 red packs, and exhausted red-only research by turn 120. More labs recovered score but exposed finite research demand; production stock alone cannot identify the next useful investment.
- Green dev revision (9002) completed at 2694.61 with no error and 632 green-science points. Its lab expansion never completed: inserter production repeatedly spent the same gears/circuits the construction compiler was trying to accumulate. A shared inventory requires explicit construction barriers or material reservations; alternating worker turns alone did not solve this contention.
- Coal mining occupied 10 of the last 49 turns in the green dev and 14 in the nine-lab dev. Six nine-lab batches reached the controller's 31-operation limit. The final revision adds a second verified coal drill/chest pair and batches burner refills, alongside completing the labs before green commissioning.
- Both allowed dev evaluations are used. Their scores belong to preliminary revisions, not the final ordering/fuel refinements. Final offline checks cover four layouts, the construction barrier, malformed/failed advisor responses, and 136 observed-economy ledger replays; final preflight evidence is recorded in the proposal. Preserve full economic telemetry because ordinary step observations truncate before machine inventories.

### Candidate 0009 — copper balance experiment
- The strongest green branch exposed a late resource imbalance: copper reached zero while iron and red packs remained, so additional red capacity did not translate into green consumption.
- A second seed-relative copper drill/furnace pair is a structural throughput test; it is scheduled after bootstrap power and first science to protect the early milestone.
- Existing service order (extract, refuel, feed furnaces, then feed assemblers/labs) should make the new pair compatible with the 100-item transfer quota without IDs tied to a seed.
- Extra smelting can backfire if coal or transfer slots become the bottleneck; inspect copper, coal and operation-limit traces together rather than attributing every green stall to recipe demand.

## Candidate 0007 — demand-driven transfer scheduling

- Leader 0005 seed 202 ends with red surplus and empty copper/green supply: turn 125 planning stock has 143 iron plates, 0 copper plates and 84 red packs; two labs are missing science despite holding red. Its transfer ledger spends about 37.5 items/turn merely collecting plates. Furnace output inventories can hold surplus until needed.
- Prioritizing science output, lab consumption and assembler inputs before bounded raw-material collection is feasible under the 100-item quota. Even distribution of scarce packs should allocate in code, then issue one transfer per lab; emitting one operation for every distribution round wastes the separate 32-operation allowance.
- Small fuel buffers were a dead end: leaving four coal after a refill triggered inherited emergency mining at stock <8. Initial dev 9001 scored 2971.75 and mined manually on 27 turns from turn 80 onward, starving the shared craft worker. Reserve fuel relative to the emergency trigger and provision burner expansion completely.
- Final 0007 dev on the same seed scored 4102.00, 128 turns, no calls/errors/failed operations. Earlier first-green commissioning, a third coal pair and a restored fuel buffer moved first green production from turn 86 to 67 and eliminated manual mining after turn 79. These linked changes are not individually ablated, and the score is not a training result.
- The new limit is still transport: final dev exhausts 100 transfers on 57 turns, and reaches 31 operations in one batch. At turn 127 it has 214 engine green packs produced / 210 consumed, but the post-planning iron buffer is empty. More nominal machines alone will not solve this limit. Raw observations and post-reservation controller stock must be distinguished in trace analysis.

### Candidate 0008 — manual/engine science specialization

- Leader 0005's worst trace had a green assembler `full_output` while labs reported `missing_science_packs`; matching science consumption matters more than nominal assembler count.
- Assigning two assemblers to green and using paid manual crafts for red produced 3493 on dev seed 9001 and 4020 on dev seed 9002, with no model calls or failed operations. Green consumption remained real lab consumption; manual products were not counted as engine production.
- A construction reservation for the next building prevented the worker from spending gears/circuits needed for labs and assemblers. Pending manual outputs must remain unavailable until observed in stock.
- Feeding machine inputs before extracting surplus plates reduced transfer-quota starvation. Science and fuel transfers should be batched; 32 late turns still hit the 100-item quota in the first revision.
- A short grace period before selecting long red-only research helps the first green packs commission, while research selection must fall back if green never appears.
- Preflight stand-in models return arbitrary text; strict action parsing and caught advisor failures are contract requirements even when the normal path uses zero model calls.

## Candidate 0011 — late laboratory capacity

- Leader 0007's turn-127 ledgers show two green assemblers producing about 210 packs and four labs consuming about 210; this suggests consumption capacity is a plausible ceiling after the green chain is stable.
- A fifth lab is a structural test with a narrow change: it is placed only after the proven bootstrap, power, smelting, green assemblers, and fourth lab, so early-science regressions can be separated from late demand effects.
- The existing service loop iterates observed labs and distributes both science-pack types, so adding a lab requires no seed-specific IDs or transfer logic.
- A late lab can still lose to the shared 100-item transfer quota; score and per-turn transfer exhaustion must be interpreted together rather than assuming nominal lab count raises consumption.

## Candidate 0012 — furnace-buffer transfer experiment

- Leader 0007's turn-127 raw ledger shows thousands of engine plates produced but near-zero immediate iron stock; 57 turns exhaust the 100-item transfer quota, confirming transport rather than smelting capacity as the late constraint.
- Furnace result inventories are a lossless intermediate buffer. Extracting every result each turn spends quota on stock that may sit unused while labs and assemblers wait for transfers.
- Candidate 0012 tests burst extraction on low stock or near-full result slots, leaving a 20-item furnace buffer; this is a structural transfer policy change with the same machine layout.
- The critical refutation signal is `full_output`: buffering must never allow result slots to remain full long enough to stop production, and assembler input starvation must be checked separately from stock size.

### Candidate 0010 — physical inserter route
- Leader 0007's seed-202 trace exhausts all 100 transfers on 39 late turns; science extraction/loading competes directly with plate and fuel service.
- A first local-inserter layout failed live: all four inserters reported `waiting_for_space_in_destination` because assembler/lab centers were only three tiles apart; score fell to 1274. Preserve this geometry lesson and gate reservations on observed `working` status.
- Long-handed inserters need four-tile center separation for assembler-to-lab delivery in this layout; coordinates must rotate with the pump site.
- Failed local logistics must fall back to remote transfers. Reservation based only on placement or electric network silently starves labs.
- Preflight remains robust to arbitrary model output; no model call is required on the happy path.
- The four-tile integer midpoint still did not align long-inserter ports in dev; use the exact half-tile midpoint and ensure route lookup follows moved lab coordinates.

### Candidate 0013 — deficit-ranked transfer scheduling

- Candidate 0012's late states expose a quota allocator failure: iron stock can be zero while copper remains 25–60, and assemblers alternate `full_output` with `item_ingredient_shortage`; production counters keep increasing, so nominal furnace capacity is not the immediate limit.
- A fixed furnace threshold is too seed-sensitive. Raw extraction should be ranked from observed downstream deficits and result-slot pressure, with an explicit iron tie-break when red/green assemblers lack gears or plates.
- Keep the leader's proven layout and fuel reserve while changing one structural policy; compare green produced/consumed packs and status frequencies against 0007/0012.
- Contract traces are planning snapshots: controller stock is already reservation-adjusted and can differ from the next raw observation. Use raw entity inventories/statuses to detect true blockers.

### Candidate 0015 — research-aware lab routing

- The leader's late ledger is close to balanced production/consumption for green packs, but its fixed lab loop still spends transfers on both science types regardless of the active technology.
- `research.current` can be an opaque `engine` sentinel; policy code must fall back to conservative all-pack routing whenever it cannot match a live technology and its ingredient list.
- Science should be treated as a demand queue: distribute a scarce required pack evenly across labs, while preserving transfer capacity for assembler inputs and fuel.
- Raw `status_name`, production, lab consumption, and transfer counters are needed together; a high pack stock alone does not prove that the selected research can consume it.

### Candidate 0014 — green input priority

- Leader traces show a logistics assembler can hold inserters while reporting `item_ingredient_shortage` because belts are absent; production counters alone hide this machine-level starvation.
- Inspect exact craft costs before diagnosing worker-time starvation: the leader can queue cable, inserters, and belts in one turn. Scarce shared iron and independent inventory targets are the stronger allocation hypothesis.
- Candidate 0014 compiles a complete affordable target vector against one shared stock ledger, then queues only jobs whose ingredients already exist; pending output may inform the plan but cannot be spent.
- Keep the proven remote transfer policy and layout fixed when testing worker ordering, so changes in green production or lab consumption are attributable to the hypothesis.
- Final 0014 dev: 9001 scores 4121 versus archived same-seed parent 4102, with first green at turn 64 versus 67 and late green shortage snapshots 20 versus 26. Red output falls slightly and 31-operation batches rise from 2 to 14; allocation helps modestly but shifts pressure to operation count. Dev 9002 scores 4078, with no failed operations/errors and one advisor cancelled at normal completion. These are dev results, not training scores.

### Candidate 0016 — operation-slot reservation

- Candidate 0014's coordinated worker bill reduced green ingredient shortages, but its trace hit the 31-operation batch cap 14 times; operation pressure is a separate bottleneck from the shared 100-item transfer quota.
- Reserve operation slots only after a logistics assembler is observed. Preserving the early 28-slot construction budget avoids delaying bootstrap while leaving late slots for assembler inputs, lab packs, and fuel.
- A batch can contain 32 operations, yet a controller's internal task cap can silently crowd out later `add()` calls; inspect submitted action strings and operation counts, not just engine `results` failures.
- Validate the policy against green `full_output` and `item_ingredient_shortage` frequencies together with first-green turn and produced/consumed science; fewer construction operations alone is not evidence of a score gain.

### Candidate 0018 — assembler-first transfer ordering

- Candidate 0015's research-aware routing selected the right pack types but still ran before assembler input moves; a transfer policy can be semantically correct and temporally wrong.
- The shared 100-item quota couples fuel, furnace extraction, assembler ingredients, and lab packs. Reserve scored assembler inputs before science delivery when machine statuses show ingredient shortages.
- Keep the opaque `research.current == "engine"` fallback and the existing parser/operation caps; narrow policy changes are safer in the final iteration.
- Evaluate `item_ingredient_shortage` and `missing_science_packs` together with produced/consumed science, since improving one queue by starving the other is not a win.

### Candidate 0017 — buffered assembler outputs
- Late 0015 traces repeatedly extract one to three science packs from every assembler before feeding labs or assembler inputs; those small moves consume the same 100-item transfer quota as scored inputs.
- Assembler output inventories are a lossless intermediate buffer.  Conditional extraction should preserve production while leaving quota for inputs, provided a near-full flush prevents `full_output`.
- Keep the demand target explicit (combined stock plus lab inventory) and flush on output pressure; stock alone is misleading because labs can already hold packets.
- This is a narrow policy experiment distinct from 0018's phase ordering: it changes when output is moved, while retaining 0015's research-aware lab route and construction.
