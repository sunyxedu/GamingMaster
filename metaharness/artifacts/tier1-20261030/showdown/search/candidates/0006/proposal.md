parent: 0002

# Hypothesis: useful conditional policies preserve reasoning for the whole battle

The prior controllers spend most of their reasoning budget re-deciding short tactical sequences, while their continuation predicates reject even expected effects. This leaves closing play to a weak numerical fallback. A compact policy with explicit outcome bounds should save calls without suppressing genuine surprises. This tests model-to-controller compilation, not a stronger damage estimate or a larger adversarial matrix.

Evidence from the leader's worst training episode, `0001/trace_seed232003.jsonl` (score -1.1249): after step 2 brings in Salamence, the planned follow-up is canceled with **“Changed mechanics/reveal: |-ability|p1a: Salamence|Intimidate|boost”**. This is the expected entry ability, not an adverse surprise. Steps 6–9 repeatedly select Earthquake into the same Mamoswine, paying for a fresh call each time. The trace also marks Rotom's Thunder Wave failing into Lanturn's Volt Absorb as **“Move failed/missed/met immunity; do not blindly continue”**, although that is evidence that the defensive switch worked. These examples isolate a mismatch between the model's plan and the controller's failure detector. The parent 0002's seed 232003 used 18 Astra-controlled actions and 16 fallback actions; its dev run authorized no guarded repeats. More arithmetic alone did not solve allocation.

# Change from parent

Retain 0002's public-observation parsing, type/mechanics helpers and emergency fallback. Replace its control loop, combat-card prompt and repeat mechanism with a small declarative policy compiler. Each fresh Astra call supplies one immediate action, strategic memory, and up to three conditional future attack/replacement rows. Rows bind species and move IDs, remap current request slots, explicitly bound HP and incoming/outgoing damage, and enumerate expected enemy replies. An unlisted reply is accepted only when observed direct damage is within the envelope (or it hits our immunity), with no disruptive effects. Unlisted non-damaging replies still trigger replanning. Policies last at most four subsequent decisions. Setup, recovery and Tera remain immediate model choices; they are never blindly looped.

The verifier reads each resulting protocol and `ctx.progress()`. It distinguishes our attacks failing from the opponent hitting our immunity, and accepts expected Intimidate and verified own entry terrain/weather abilities. Any opponent switch forces a new model decision, even when that species was previously revealed. Uncovered matchups, unexpected non-damaging moves, damage above the envelope, insufficient attack progress, changed status/field, enemy boosts/Tera, rejected commands and unusual mechanics discard the policy and carry evidence to Astra. The prompt preserves the complete request, all public sets, move/species descriptions, recent events and measured outcomes without a generic cutoff. Opponent hidden data and game implementation are never read.

Use 240-second model timeouts, above the archive's 178-second observed maximum, with 270 seconds required to begin a call. Catch timeout/backend failures and validate arbitrary output before compiling. Retain one call unspent so the hard call limit does not end the episode before a fallback can finish.

# Confirmation and refutation

Confirm if compiled policies actually execute, especially after forced replacements and expected defensive switches; fewer calls should leave model decisions available in late game, with improved completed wins and weak-seed HP margins. Inspect policy cancellation reasons, damage surprises and actions per successful call. Refute if Astra still emits empty policies, expected events cancel most rows, policy output costs more than it saves, or permissive guards allow bad repeated exchanges. Saving calls without better outcomes is not sufficient evidence of stronger play.

Validation and dev results will be recorded below. No seed-specific policies are implemented.

# Early development findings

The first dev invocation loaded the initial complete revision. Early trace inspection found two additional brittle cases: expected Rillaboom Grassy Surge canceled a follow-up, and a harmless unlisted Gunk Shot (17.8% to Gengar, below the 43% envelope) canceled another. Final verification accepts the explicitly observed own entry environment effect and treats measured, non-disruptive damage within the bound as compatible evidence. These are policy-semantics fixes, not species prescriptions.

The same initial dev build exposed a dangerous branch: Astra supplied a Sludge Wave rule against a *switching* Wyrdeer, assuming its Body Slam immunity with max_loss=0. Gengar repeated Sludge Wave, missed the KO after recovery, and died to newly revealed Psychic Noise. The final compiler accepts only rules for the current opponent; every enemy switch forces fresh reasoning. This constrains policy scope instead of trusting an unevaluated new matchup.

# Local validation

Preflight passed twice with natural game completion (21 actions), with the explicit warning that its short virtual budget did not admit an Astra call. Separate mock-context full loops covered valid JSON, arbitrary text, TimeoutError, RuntimeError and null terminal requests. `check_policy.py` also builds prompts and checks fallback legality on 198 complete archived states; truncated trace observations are skipped.

Targeted regressions verify expected Intimidate, our immunity, own terrain activation, low-HP and damage-envelope cancellation, remapping both moves and forced-switch identities, opponent-switch invalidation, failed damage promises, all three relevant `ctx.progress()` signals, and persistent publicly revealed ability immunities. The fallback now ranks known-immune attacks below legal switches; this is a bounded inherited-fallback repair rather than the experimental policy hypothesis. The parent’s approximate calculator remains an emergency fallback only; its damage numbers are not supplied as authoritative predictions to Astra.

A third interface issue appeared in the initial dev trace: one Astra response wrote nickname `Arceus` while the observed species details were `Arceus-Steel`, so that row did not compile. Final prompts expose stable private identity strings and use the opponent selector `current`. The compiler accepts observed identity/species aliases without changing observed species or types. This is checked by replay; no hidden set data are used.

# Completed dev evaluation: negative strength evidence

One dev invocation completed both seeds in `dev/20261031-153902/`, using initial revision SHA-256 `7511590715666f96a1b3df739219b234b1508edec61d3109b6ee2a6fe3830295`:

| Seed | Score | Result | Decisions | Astra calls | Compiled actions | Fallback actions | Wall seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| 869431 | -1.119378 | loss | 38 | 24 | 5 | 8 | 1557 |
| 895003 | -1.084848 | loss | 35 | 19 | 3 | 12 | 1578 |

Mean score **-1.102113**; zero episode errors and zero failed calls. Each episode also had one unique-legal-action choice. The longest successful Astra call took **208.3 seconds**, above the previous archive maximum but below this harness's 240-second timeout.

The eight compiled actions establish that the policy mechanism executes, including a U-turn replacement into Glimmora and a later faint replacement. They do **not** establish improved decision quality: Gengar died to unmodeled Psychic Noise, Sunflora barely survived Ogerpon's unmodeled Superpower, and the Munkidori line was disrupted by opposing Tera and Knock Off. A Giratina attack made no net progress into Recover and correctly canceled. Several useful-looking continuations were lost to expected entry effects or naming; these are covered by the final local repairs described above. Twenty late fallback actions demonstrate that the initial intervention did not eliminate the budget problem.

The completed digest and traces therefore **refute a claim of strength improvement for the initial revision**. Final changes narrow the policy scope, accept verified benign entry/damage effects, verify damage predictions on the initial action, use stable identities and repair known-immunity fallback. Their mechanical behavior passes archive/regression/preflight checks, but their playing strength has not received another real model evaluation. No second dev invocation or full training evaluation was run. The remaining research risk is that a short policy merely carries a mistaken survival/coverage belief forward; measured damage cannot certify unseen alternative attacks.

Final preflight and source hash are recorded in `preflight.json` and `validation.json`; the short dry run still exercises fallback only, and separate mocked model-path tests cover the missing path. The design contains no seed-specific rules or cross-episode memory.
