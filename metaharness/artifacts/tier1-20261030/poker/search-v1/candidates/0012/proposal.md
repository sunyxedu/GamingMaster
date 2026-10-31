parent: 0009

## Hypothesis

The promising deeper planner fails because it treats computation as free and only checks the episode deadline immediately before acting. Candidate 0009's worst trace, `trace_seed2207.jsonl`, logs hand-69 `raise_to 200` at clock **1799.411830**, then a deadline at 1800 and an episode-end error at 1813.24: **“game hung: a call into the game (tick/step/observe) did not return”**. Its completed steps had median 8.36 s, 95th percentile 18.90 s, and maximum 22.99 s between decision log and returned observation. Its 0.5-second reserve was plainly insufficient. The archive reports no completed training match for this parent; its stronger provisional scores do not establish terminal superiority.

The eligible leader 0003 is a useful cheaper fallback, but its weakest match shows a possible benefit from preserving optional lookahead: hand-22 with Js8c on Td4h9dJc chose `raise_to 250`, valuing passive play at 335.4 chips. It subsequently jammed 19500 on the Ts river and lost 20000 to publicly revealed JhQc. Parent 0009 instead valued the turn check at 1588.18, checked, and eventually folded to a river raise. This is a concrete diagnostic, not proof from one losing hand.

**Test:** budget computation across the whole required match, keeping a complete one-street action ready while optional turn-to-river search improves it. Deep search should be spent only when measured game-step latency and remaining hands leave room. This is a completion/resource-allocation experiment from 0009, distinct from the parallel inference and exposure-cap proposals.

## Changes from the parent

Retain the parent's exact bit-mask evaluator, public-action holding posterior, preflop policy, exact turn/river card tables, terminal river solver, legal integer action bounds, betting thresholds and exposure costs. No model calls or outside dependencies are added.

Before each turn, a controller uses remaining wall time, remaining required hands, observed actions per hand, and an exponential average of actual `ctx.act` duration. It allocates 32/64/96/160 public flop runouts and at most ten seconds to optional turn lookahead. The controller never consults score, hidden information, or a seed. A turn first obtains a complete one-street fallback. Deeper search publishes only a fully evaluated passive baseline or fully evaluated betting candidate; a local timeout returns the last complete choice. No partially evaluated river branch is treated as a finished EV. The all-in endpoint is searched first after the baseline so early interruption does not systematically exclude the parent's strongest value exploit.

Reserve at least 30 seconds before entering a game step, growing to twice the largest measured step plus three seconds (bounded by 20% of the episode's initial budget for short preflight episodes). Both computation checkpoints and the final pre-action check respect this reserve. Local search expiry is distinct from episode expiry and caught without ending the match. There are no background workers.

## Confirmation and refutation

Confirm the engineering hypothesis if all training matches finish or return cleanly without deadline-grace errors, the scheduler reduces optional work under slow-step replay, and completed independent development matches preserve substantial value extraction. The primary objective remains terminal net chips / required hands; early clean return is safety, not success. Refute it if reduced sampling destabilizes the posterior, partial action search loses significant value, the controller still cannot complete, or its game-step reserve is insufficient. Development seeds are exploratory match blocks, not individual-hand significance tests. Runtime dependence means different machine load may produce different legal policies; this is a deliberate limitation to measure.

Validation: preflight passed with 91 legal actions, zero model calls and no exceptions. One permitted development match on seed 901 completed all 100 hands in 135.6 seconds, scored 1105.50 chips/hand, and had zero illegal actions or errors. The trace records the controller's selected 96/160/32 flop samples and turn lookahead budgets; no partial turn branch was submitted. This is an exploratory single match, not a training score or uncertainty estimate. No official evaluation or held-out games are run by this proposer.
