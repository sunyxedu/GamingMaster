parent: 0003

## Hypothesis

The leader's communication protocol executes safe plays reliably, but its one-step card-loss utility cannot distinguish affordable draws from discards that make an entire color inaccessible. Evaluate its existing legal actions by sampled terminal score instead. A short, bounded information-respecting rollout should improve choices without needing another communication convention or a publicly constrained discard policy.

## Evidence

Leader 0003's worst training trace (`trace_seed14.jsonl`) discards W5/card 1 on turn 31, reporting `own discard E_loss=0.542, p_dead=0.286`, and the last G3/card 25 on turn 37 with zero clues, reporting `E_loss=0.605, p_dead=0.349`. The earlier G3/card 13 was discarded at turn 26. It finishes at 20 with all lives. Turns 9–18 also contain five clue/discard pairs with little progress. The problem is not safe instruction execution but the long-term consequences of locally cheap discards and clue expenditure. Candidates 0004–0006 did not improve the training leader despite more elaborate save protocols, motivating search over the successful existing protocol.

## Change

Retain the leader's rotating rank/play and color/discard channel, exact own-hand beliefs, legal action validation, final-round risk rule, and isolated advisory sessions. Add bounded Monte Carlo policy evaluation: sample current own cards and the unseen deck from literal masks and remaining multiplicities; simulate the exact public rules; give every future seat a correctly redacted observation; run the existing protocol policy to termination. Compare root actions using common sampled worlds and expected terminal benchmark score, including zero on bomb-out. Simulated private hands and decks are local to the current decision and are never retained or shown to another actual seat or its model session. Astra can arbitrate only already validated options with computed rollout summaries.

Own-hand samples are drawn sequentially from exact conditional joint marginals, without replacement; the remaining unseen multiset is shuffled as the hypothetical deck. This is exact for the literal-clue prior, not for policy-dependent action selection. Future rollout decisions use the parent's protocol and utility with cheaper conservative per-card marginals, while actual decisions retain exact joint inference. This approximation can miss correlated deductions but does not reveal a simulated actor's own cards. No actual hidden state is read.

## Confirmation and refutation

Support: fewer live last-copy losses, fewer low-progress clue/discard cycles, improved terminal scores, and natural game completion within budget. Refute: simulation mistakes, systematic heuristic-policy bias, literal posterior selection bias, excessive compute, or worse training score. Record rollout counts, projected terminal values, chosen action and fallback value. No seed, engine implementation, actual hidden hand or deck order is a policy input.

## Validation

Implementation and proposal are written before dev evaluation. Preflight, rule-transition fixtures, redaction checks and permitted dev findings will be appended below. The proposer will not run training or held-out evaluations.

### Simulator and contract checks

The observation-derived simulator matched **278 archived nonterminal transitions** from the leader's training traces, including counters, legal actions, visible hands, positive and negative clue masks, and final-round transitions. **41 posterior checks** verified own-mask consistency, exact conservation of the unseen multiset, and actor redaction. Four additional fixtures cover a five restoring a clue, no draw on bomb-out, exactly two actions after the final draw, and no draw after reaching 25. Six fake-context cases cover garbage, out-of-range and valid model choices, backend failure, model timeout, caller queue timeout, and separate initial seat sessions. These tests read observations only, never game implementation source.

The first preflight completed normally at a stand-in score of 22, 69 actions and zero model calls. This is a contract check rather than a performance estimate.

### First dev result and uncertainty refinement

The first permitted dev episode (`dev/20261030-004733`) finished **19/25**, 72 actions, all three lives, and no error in 122 seconds. Its single adviser request hit the caught 90-second caller timeout; automatic play completed. Search did not resolve the proposed loss mechanism: it discarded the last W3 on turn 14 and the last R3 on turn 44. In particular, turn 14's 24-sample estimates favored an uncertain own discard over a visible partner discard, while several other switches were based on only a fraction of a point.

The refinement widens search to all protocol-valid root options, raises the target to 96 common-world samples, and bounds each decision to eight real seconds. It requires at least eight complete paired samples; unfinished rollout batches are discarded. An alternative replaces the parent's action only when its paired mean score gain exceeds `0.10 + 1.25 * paired_standard_error`. This conservative comparison addresses maximization over noisy estimates; it is not a guarantee against rollout-policy or posterior bias. Astra is offered only options that satisfy the same evidence threshold, with at most two seat-isolated calls. The second dev episode tests this single refinement, not a new convention.

### Second dev result

The second and final permitted dev episode (`dev/20261030-005149`) completed at **22/25**, 70 actions, all three lives, two information tokens, two submitted System-2 calls and no episode error, in **311 seconds**. One adviser answer selected the already top-ranked option; the other exceeded the outer 90-second wait, was ignored, and arrived later during automatic play. Search completed 66–96 paired samples per evaluated decision and recommended seven deviations from the parent's local choice. Final stacks were R5/Y4/G4/W4/B5.

No lower-rank last copy was discarded in this trajectory, but unique W5 was discarded at turn 22 and Y5 at turn 55. Thus the change improved the paired dev result from 19 to 22 and reduced prerequisite losses, while failing to preserve the full physical ceiling. It does not prove an improvement over leader 0003's training mean of 21.5. Both allowed dev evaluations are used; no training or held-out evaluation was launched.

### Limits and follow-up

The search prior omits information conveyed by the partner's choice not to issue a play command. Cheap rollout beliefs omit within-hand correlation, and continuation policies do not themselves search. These can bias estimates even with small Monte Carlo standard errors; the paired uncertainty margin measures sampling error only. The eight-second bound also means sample counts can vary with machine load. Keep these effects separate from communication correctness when inspecting training results. The exact rules, slot decoder and posterior-conservation audits passed; the only change after the second dev run began was public last-copy-loss logging and documentation.

Final preflight after the logging change passed with natural completion, 70 legal actions, a 22-point stand-in score and no model calls. The observation-transition, posterior/redaction, terminal and adviser fallback tests all passed again. `validate.py` contains those observation-only checks for later iterations; it is not imported by the runtime harness.
