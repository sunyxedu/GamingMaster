parent: 0011

## Hypothesis

A two-turn search cannot compensate for treating an observed but undefined enemy
attack as harmless. Candidate 0011's `replies()` skips moves without a reference,
even after the opponent has used them and exact own HP loss is public. A damage
ledger should fit uncertain empirical move models and put those threats into the
search immediately, without waiting for another Astra answer. This takes the
0011 search branch, not 0012's action-firewall/continuation recipe.

The leader's worst episode, `0012/trace_seed333667.jsonl`, shows the same missing
reference problem: after `Boomburst` reduces Raichu from 249/249 to 114/249, it
switches Chimecho into another Boomburst for 187/290 damage, then spends Poison
Tera and faints to the next hit. Boomburst is absent from the 115 archived public
`move_dex` definitions. The model's second call was still pending; code made all
remaining decisions after the wall reserve activated.

More direct parent evidence is in `0011/trace_seed333667.jsonl`: step 9 has
`|move|p2a: Typhlosion|Scorching Sands|p1a: Volcanion`, followed by
`|-supereffective|p1a: Volcanion` and a KO from 125/256. Step 11 then shows the
same move super-effective against Raichu, leaving 129/249. These are public
constraints on the move's mechanics. Nevertheless the parent selects another
Thunderbolt at step 12, taking Raichu down to 21/249 before switching Chimecho
into the attack. Scorching Sands is also absent from the public reference.

## What changed versus the parent

Retain 0011's two-turn simultaneous-action search, empirical action-frequency
prior, sparse Astra threat portfolios, legality checks and failure recovery.
Add an episode-local empirical attack model consumed by the actual rollouts:

- Replay public events at the moment of each enemy hit, including switches,
  earlier attacks, HP, boosts, Tera, field changes and item/ability revelations.
  Sum direct multi-hit losses once. Exclude tagged residual damage and criticals;
  a KO is a censored lower bound. Collector scaffolding comes from 0007's public
  damage ledger, but evidence now changes the simulated exchanges themselves.
- For a move with a definition, shrink bounded damage ratios toward the base
  mechanics estimate. Transfer only across compatible attacker/field contexts
  and modeled defensive abilities. Changed stat stages remain in the raw math.
- For a revealed move without a definition, consider type/category combinations
  consistent with the observed effectiveness messages. Fit a damage multiplier
  from the observed targets' exact stats and HP loss, downweight combinations
  whose predictions disagree across samples, and use a weighted estimate with a
  small upper-tail premium for other targets. A second super-effective hit on a
  different target can sharply narrow the possible types. These remain uncertain
  models, not asserted move facts. A subsequently supplied move definition
  supersedes the surrogate immediately.
- Keep a same-target empirical damage floor in compatible contexts. Log sample
  provenance and the surviving unknown-move type/category hypotheses. Astra sees
  a compact evidence report and still supplies hypotheses rather than commands.

There are no seed-conditioned rules, opponent-set tables, hidden-state reads or
runtime archive reads. The harness imports only the standard library.

## Confirmation and refutation

Confirm if unknown revealed attacks remain represented in reply portfolios,
subsequent measurements narrow their hypotheses, replacements survive threats
already demonstrated in the protocol, and weak episodes improve without more
errors. Refute if broad type uncertainty induces harmful switches, censored hits
inflate danger, or the simplified mechanics cannot transfer observed damage
between targets. The ordinary rollouts remain approximate: unusual abilities,
Transform/Illusion, nonlinear/fixed damage, multi-hit distributions and unmodeled
field effects can invalidate the fitted mechanics. Context checks and bounded
estimates reduce, but do not eliminate, that risk.

## Development evidence

The initial known-move-only calibration changed **zero of 296** archived parent
choices. This was evidence against merely adding a multiplier. The final design
also addresses the omitted unknown-move branch. A replay of 100 decisions from
three parent loss traces changed two first actions while keeping every action
legal. In the Scorching Sands position after Raichu's first hit, it selects the
Chimecho switch one turn earlier; the parent trace later verifies that this
replacement is immune to the same attack. This is an off-policy mechanism check,
not a counterfactual win or a measured improvement in score.

Final validation results are recorded below. No full training evaluation was run.

## Final validation

`python3 candidates/0014/validate.py` passed. It replayed **296** parent decision
observations, producing only finite values and currently legal actions. It
retained 86 clean samples (42 collected on switches and 14 censored by KOs) and
changed nine first choices relative to the same search with its sample ledger
removed. The replay took 31.8 seconds including the baseline comparisons.
`validation.json` records each changed choice and its unknown-move hypotheses.
These comparisons reuse archived observations and model hypotheses; they are
neither live games nor a win-rate estimate.

Focused checks verified hit-time damage before Leftovers, critical exclusion,
multi-hit aggregation, KO censoring, and the Ground-only inference from the two
Scorching Sands targets. The inferred attack respects Chimecho's observed own
Levitate ability. Five stand-in control-loop runs each completed five legal
steps under arbitrary prose, malformed structured data, TimeoutError,
RuntimeError, and an indefinitely pending model call with the wall reserve.

The final evaluator pre-flight passed: `game_done`, 18 accepted actions, no
exception. Its short budget admitted no System-2 answer, which is why the
separate stand-in checks matter. No real-model dev evaluation or full training
evaluation was run. Queue/deadline handling and the sparse call policy are
inherited from 0011, so this candidate isolates empirical threat modeling rather
than claiming a model-latency improvement.
