# 0005 — Grounded opening controller with a safe idle lease

## Parents and hypothesis

Crossover parents: `0000-tier2_jev_only` and `0002-tier2_idle`.
Neither parent earned progress: the supplied leaderboard reports zero for all three training episodes of each. The useful inheritance is Jev's inexpensive tactical classification and idle's non-interference with the live world, not a claimed successful parent strategy.

The concrete action-bridge failure is visible in the Jev parent's source: its Choice contains `game.action_spec.names`, then it posts `ctx.act(ans["act"].value)`. That can choose `gather` but cannot supply a block candidate and count. The public action contract explicitly rejects bare parameterized names. In the inspected seed-0 initial trace, `b:-23,68,31` is an exposed `acacia_log`, visible and harvestable at distance 8.69, but not in reach. Thus there is a grounded opening opportunity even though the baseline earns no opening points. This is an initial-observation quote, not a claim to have exhaustively audited every rejected tick. No root `lessons.md` was present in the conditional read.

**Hypothesis:** the zero-score plateau is substantially an action compilation and job-lifecycle problem, rather than a need for more model calls. Compile full commands from observations and inventory, allow bounded jobs to finish, and ask Jev to choose only among executable, explained alternatives. Preserve idle as an explicit control lease while a model or existing job is active.

## Change

This is a new controller, not a copy of either parent's loop. Code builds a small dependency-driven opening: logs, planks, table, pick, exposed stone, stone pick, exposed iron, furnace, and shield. Food hunting/harvesting and observed beds are opportunistic branches. Placement requires an observed in-reach support with locally passable space. No hidden resources, seed-specific coordinates, privileged calls, or game implementation access are used. Inventory accounting excludes containers and previews.

All model-selected commands are complete strings, not bare verbs. Every operation receives an intervening noop, which clears one-shot latching without cancelling a job. An observed busy task owns the hands. Repeated attempts have cooldowns. The loop posts once and waits for the game tick, including while background inference runs. Safe idle does not mean survival is guaranteed; this partial controller has limited combat and no comprehensive lava escape policy.

Astra is asynchronous and only requested after forty seconds without inventory change, at least 180 seconds apart and with sufficient remaining wall time. Its output is one validated priority token: balanced, food, or iron. These modest biases cannot invent actions or bypass prerequisites. This is deliberately a bounded macro adviser rather than an unvalidated policy generator.

## System-1 interface

State contains the explicit goal, Astra focus, authoritative inventory, code-derived health/hunger bands, visible nearby hostile descriptions with range buckets, a recommended action, and at most five full commands with purposes and code priorities. It excludes the raw terrain dump.

One request batches two independent questions:
- `action`: Choice over those exact command strings. Its winning probability must be at least 0.60; confidence is not substituted for probability.
- `unsafe`: Noul asking whether the supplied nearby hostile evidence makes resource work unsafe. At 0.85 and with current enemies still present, resource work is vetoed in favor of shield or noop.

There is at most one outstanding Jev request, with a three-second timeout. Answers expire after two seconds and must match the current dimension and inventory; chosen commands must still occur in the newly generated legal shortlist. Background requests have no custom polling loops. Arbitrary output, missing answers, and transient errors cannot become commands. Budget exhaustion and unavailable-model exceptions propagate according to contract.

While waiting, post noop and keep bounded jobs alive. After ambiguity or a failed request, execute the top code-grounded option on the short decision cooldown; after exhausted model budget, use that controller alone. Stale observations cause cancellation of a busy job followed by noop. Respawn is explicitly supported. No deliberate controls movement is renewed while waiting for reasoning.

## Confirmation, refutation, and limits

Confirm with valid parameterized actions, logs and crafting-table possession, positive opening score, actual food/iron acquisition, low stale ticks, and no runtime errors. Trace events distinguish selected Jev actions, hazard vetoes, Astra focus changes, and deterministic actions. Refute the hypothesis if well-formed commands still fail to gather the initially exposed resources, if inventory normalization does not match authoritative observations, or if the controller cycles through unproductive crafting/navigation attempts.

This is a deadline-limited partial harness, not a dragon-completion policy: there is no complete Nether/stronghold/End sequence. Later iterations should first verify placement, inventory auditing delay, smelting retrieval and movement outcomes, then extend the same grounded planner. Repeated vetoes or stagnant legal options are reasons to revise the Jev interface in a later candidate; runtime Astra here only revises macro priority, not question text.

No live evaluation was run. Only syntax compilation is performed at submission; scores and latency improvements remain hypotheses.
