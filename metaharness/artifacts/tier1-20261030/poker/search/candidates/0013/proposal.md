parent: 0012

## Hypothesis

Candidate 0012's public-history range model is promising, but its exact-support board tables are too expensive for the fixed 1,800-second episode. The evidence is structural: seeds 3301 and 4409 both reached only 70 completed hands at the deadline, while the harness was still doing full range work; the leader's 1021.8 mean therefore includes provisional denominator scores. A lower-workload approximation that preserves all legal opponent combinations should complete more hands and may increase the actual score despite noisier equity.

## Evidence

The 0012 evaluation reports 100 hands in seeds 1103 and 2207, but only 70 in seeds 3301 and 4409, both ending at the deadline. Seed 4409 had a cancelled Astra review at the deadline. Its trace repeatedly logs full-support `complete_range_search` decisions, with 1,081 opponent combinations at flop and 1,035 at turn, and 64 sampled flop runouts per table. This is a throughput failure, not an action-legality failure.

## What changed

This candidate retains 0012's full-support public-history likelihoods, exact hand evaluator, capped sizing, response verification, and exception-safe Astra review. It reduces board-completion work: turn tables use a deterministic 16-river subset instead of all legal rivers, and flop tables use 24 deterministic runouts instead of 64. Tables remain cached by visible cards and board, and deterministic sampling does not touch the deal RNG. Astra reviews are capped at four and require a 240-second reserve, leaving more wall time for hands. Every action remains candidate-validated and checked with `ctx.progress()`; model text can only adjust bounded posture parameters.

## Confirmation and refutation

Confirm if independent matches complete substantially more hands, especially on the two deadline seeds, while preserving positive net chips and avoiding illegal actions. Refute if smaller runout samples cause systematically poor value sizing or if compute remains the bottleneck. The shared deterministic subsets are approximation evidence, not independent statistical samples; use independent match scores to judge the result.

## Verification

The module is derived from the preflighted 0012 harness and keeps its standard-library-only imports and robust garbage-response parsing. `python3 -m py_compile` passes. Contract preflight passes with `ok: true`, 173 actions, no errors or illegal actions, and no System-2 answer delivered by the stand-in. No full evaluation was run here.
