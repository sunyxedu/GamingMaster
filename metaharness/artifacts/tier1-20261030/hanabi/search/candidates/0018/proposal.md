parent: 0012

# Hypothesis

The inverse-clue encoder conveys relationships between cards, but parent 0012
unions those relationships into independent per-slot support sets. Its subsequent
without-replacement posterior consequently assigns weight to hands that could
never have produced the observed clue. Retaining the canonical clue-choice
constraints should improve uncertain discard choices and protect prerequisites,
without changing the communication convention or adding rollout search.

# Evidence

The best candidate 0013's worst training trace, `trace_seed14.jsonl`, discards
R5/card 35 at turn 53 with zero clues and eleven deck cards. It finishes at 21
with three lives, R4/Y4/G3/W5/B5. Its residual problem includes card loss, not
just unsafe endgame plays.

The distinct parent 0012's same trace loses W5/card 1 at turn 31 (one clue,
25 deck cards) on a `private posterior discard` valued 1.4242. It then loses
the last G3/card 25 at turn 37 (zero clues, 21 deck cards), again on a
`private posterior discard`, valued 1.585144. The earlier G3 was discarded
at turn 26. Parent 0012 ends at 20 with all lives. Its proposal explicitly
acknowledges that unary support unions discard encoder-induced correlations.
These losses motivate testing the approximation; they do not prove that exact
conditioning will reverse either particular decision.

# Change from parent

Keep the canonical rank/play and color/discard encoder, sound public support
propagation, action scoring, separate seat sessions, bounded Astra advice and
legal-action transport. Retain every accepted clue as a public factor on stable
card IDs. The factor records only public board utilities, clue symbol and touched
IDs; it never records a visible private face. Later public play/discard faces
substitute for removed cards, so relationships with still-held cards survive.

For the acting seat only, enumerate finite-population joint hand assignments
subject to literal masks, propagated supports, copy capacities and all retained
canonical-choice factors. Prune a partial assignment when a better truthful
message is already forced. Accumulate complete assignment weights and marginals.
Only a completely finished enumeration can eliminate identities. The final
version intersects each private domain with identities that have a complete
compatible assignment, then recomputes the parent's without-replacement prior
on those refined domains. It does not use the enumeration's relative weights
as a calibrated behavioral posterior. Node and elapsed-time limits fall back
to the parent's sound unary-support approximation. No partial enumeration can
assert certainty or supply a probability.

The shared tracker is reconstructible from public events. Private enumeration
uses only the current observer's visible partner hand and is discarded after
the decision. No sampled decks, engine access or training-seed policy inputs
are introduced.

# Confirmation and refutation

Confirm the mechanism if exact conditioning removes positive-weight impossible
hands, materially changes discard probabilities, and reduces live last-copy
losses while completing games. Log complete/fallback inference, retained
factors, enumerated nodes and the largest marginal change. Refute the scoring
hypothesis if changed estimates make worse discards or rarely differ. A factor
rejecting a later publicly revealed true assignment refutes correctness.

The enumerated support is exact conditional on the explicit factors, existing
support sets and physical counts. The final probabilities again discard
correlations after using them for sound support deduction. Neither calculation
models every strategic decision in the transcript. Unmodeled action-selection
effects remain. Runtime bounds may select the parent's approximation.

# Validation

Harness and proposal are written before experiments. Required preflight,
observation-only factor audits, exhaustive small-hand comparisons, model-error
fallback checks and permitted dev results will be recorded here.

## First dev result and conservative refinement

The initial version used the full weighted conditioned marginals. Preflight
passed. Dev `20261030-021954` finished at **21/25**, 69 actions, all three lives,
two successful Astra calls, 109 seconds and no error, below the parent's 24.
All 67 factor enumerations completed. At turn 16 the first divergence discarded
R2 instead of the recommended spare R4: the new weighting raised R2's dead
probability from 0.50 to 0.60, enough to override the instruction. Although that
immediate discard was harmless, the changed continuation later discarded the
last B4 at turn 35 and W4 at turn 40. This is negative scoring evidence; it
does not establish a factor correctness failure or a causal attribution of
all downstream loss to the first divergence.

The final refinement therefore uses complete joint enumeration only to remove
unsupported identities, followed by the parent's multiplicity-weighted prior.
This isolates the benefit of logical deductions from the more speculative
relative probabilities induced by an incomplete model of policy choices. It
retains the canonical encoder and all action scores unchanged; it introduces
no seed-specific threshold, card rule or exception.

Observation-only audits checked 450 randomized small hands, including public
removal substitutions, against exhaustive encoder reconstruction (1,466
compatible weighted assignments); weights and supports matched to 1e-12.
Forced node exhaustion returned no estimate. Stateful replay of 347 archived
parent actions preserved every actual observed/revealed face. Of 322 inference
attempts, 320 completed and two fell back, with no contradictory factor sets.
On the fixed parent seed-14 history, W5's slot at turn 31 became certainly
critical (previously 0.5), and Y5's slot at turn 55 became certainly critical
(previously 1/3). These are fixed-history inference results, not counterfactual
episode scores. The last G3 at turn 37 remained ambiguous.

The inherited protocol checks also passed 819 archived views and 400 small
exhaustive fixtures, plus garbage replies, invalid selections, backend failure,
outer queue timeout and separate resumed sessions. Preflight itself answered
zero model calls.
