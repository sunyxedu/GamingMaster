# Lab notebook

## Candidate 0002

- Both archived baselines die at tick 5 with zero food. Jev takes 0.559–0.625 s for a 0.200 s tick, while Astra takes 5.125–5.628 s. The next-action model loop is the dead end to fix before tuning prompts.
- A fresh model answer is not necessarily a fresh action: seed 2's first Jev answer describes tick 0 but arrives after tick 3, and its second arrives after termination. Compile reusable control and re-evaluate it on each current observation.
- Generality requires runtime semantic binding as well as generic scheduling. Candidate 0002 discovers field roles from public evidence and asks Astra to generate the durable policy; its conditional geometry bootstrap is temporary and cannot guarantee survival on every interface.
- Jev can lease a strategy ID while code recomputes the actual action. Version, regime, request age, confidence and current safety all matter. Log actual changes from the local default, not just model-call counts.
- Garbage output, malformed code, runaway Python loops and prediction contradictions need an executable fallback. Synthetic checks passed; safety labels from generated code still need empirical validation and are not an independent proof.
- Respect the exact contract: read typed answers directly by question ID, use `.value`/`.probabilities`/`.confidence`, forward the usage guide to interface-design prompts, propagate ModelUnavailable, and drain every spawned future. The 900 s wall budget also matters even when the game's tick cap is much larger.

## Candidate 0003

- `0002` never exercised its compiled-controller design: all nine Astra calls timed out at 45 seconds, with zero installations. Its 0.41237 mean belongs to the temporary reflex. Diagnose artifact delivery before drawing conclusions about a proposed strategy.
- An inference timeout was followed by an empty-source parse error. Keep transport/no-artifact failures separate from invalid generated code; otherwise subsequent prompts repair a compiler problem the model never caused.
- Replaying seed 0's last public snapshot and recorded actions reaches head `[0,9]` at tick 460, retained body on the right, neck above, walls left/below; tick 461 requests `DOWN`. Zero stale ticks did not prevent a greedy geometric trap.
- The new hypothesis is staged compilation with a compact return helper and temporary reward avoidance during acquisition, then a stronger replacement while local control runs. Longer 120/180-second allowances are experimental, not measured speedups; successful deployment alone is insufficient without improved completion.
- Synthetic checks caught a stale fallback-action bug: invalidating a generated policy must also clear its cached action. Failed upgrades should retain the working policy, whereas contradicted predictions must disable it. Join outstanding workers even after termination and propagate either model's `ModelUnavailable`.
