# Candidate 0002: compile guarded options, then recognize and execute

Parent: `0001-tier2_astra_only`; supporting comparison: `0000-tier2_jev_only`.

## Hypothesis and archive evidence

The dominant demonstrated failure is a lossy action bridge, before strategic skill can matter. A second failure is incomplete observation delivery. Slow sequential inference and unhandled timeouts compound these defects. If Astra compiles reusable, fully parameterized options from the public interface, and Jev selects among options instantiated from current observations, the agent should finally acquire resources and complete milestones while retaining a responsive execution loop.

Concrete evidence (JSONL line numbers are one-based):

* In `0001-tier2_astra_only/trace_seed0.jsonl:17`, Astra returns **`gather b:-34,67,41 4`**, with latency **7.4286 s**. Line **19**, tick **15**, applies **`gather`**, marked fresh. Line 32 returns another complete gather command; the tick action census contains 110 bare `gather` ticks and 24 bare `navigate` ticks. The model supplied the argument in the first response; the bridge lost it. The parent's instruction asks for one action name and its `parse_action(..., names, ...)` call cannot preserve this richer interface reliably.
* The same worst episode finishes at **221.48 s**, score **0**, with **401 stale ticks out of 431**. At line 484, clock 211.46, the prompt's health is **13.333333015441895**, down from 20 at clock 204.89. This supports late survival failure, but the truncated prompts do not establish its exact cause. They visibly end midway through candidate records with `…(truncated)`, omitting later observations needed for inventory, outcomes, and progress.
* Parent seeds 1 and 2 consume **40 Astra calls**, finish their harness runs at **283.18 s** and **288.11 s**, and report `game_done: false`. The parent breaks out on `RuntimeError`. Spending the full reasoning budget is therefore also costing the alive-at-time-limit reward; it is not an ordinary completed survival episode.
* `0000-tier2_jev_only/trace_seed2.jsonl:19`, clock **3.8516**, contains `last_result: {id:1, action:"gather", status:"failed", error:"bad_arguments"}`, empty authoritative inventory, and zero progress points. At line **749**, clock **168.8124**, it still has empty inventory, zero progress points, and failed bare gather id 16. Faster verb selection does not repair missing arguments.
* That trace's line **760** records a failed Jev request taking **5.1600 s**, `request deadline exceeded`. The episode ends at **173.97 s**, score **5.7993**, because `TimeoutError` escapes. Successful Jev calls average **0.8229 s**, median **0.7734 s**, p90 **0.8699 s** in this episode: they already exceed the 0.5 s tick. Parent Astra seed 0 averages **7.1438 s** per call. These measured latencies motivate asynchronous execution rather than assuming either model is instantaneous.

The experiment changes the bridge and its scheduling together, so it cannot isolate the marginal benefit of Jev from correct parameterization. The first diagnostic is useful completed actions, not the score alone.

## What changed

`harness.py` contains no Minecraft item names, action verbs, observation paths, recipe table, scoring formula, or terrain solver. Each episode starts from `game.describe()`, the documented action specification attributes, and `ctx.observe()`.

Astra returns a JSON policy with a strategic goal, compact field/rule meanings, selection and danger guidance, observation expressions for task/outcome/time handling, and guarded action rules. The first request asks for only 4–7 opening rules under 1800 output tokens; subsequent requests may extend the dependency chain. The complete bundled `usage_guide()` is forwarded to Astra. Each call uses a fresh prompt with current evidence instead of an indefinitely growing conversation.

A bounded interpreter evaluates this policy as data. It supports exact arithmetic, comparisons, dictionary access, filtered sums, list iteration, distance calculations, and token construction. No generated Python, `eval`, `exec`, file access, or backend access is available. A rule binds candidates from the full latest observation, checks guards, ranks targets, and generates a complete whitespace-separated command. Public textual signatures are used to infer minimum argument counts when their syntax permits it. Unknown verbs, missing inferred arguments, non-scalar tokens, whitespace inside tokens, invalid expressions, and missing required paths are rejected. This is syntactic and policy-guard validation, **not proof of all game legality**; adapter feedback remains authoritative.

Known arithmetic belongs in these executable expressions; Astra supplies the game-specific constants only after reading the description and observations. Prompts encourage feasible-recipe checks, authoritative completion, prerequisite-first resource use, delayed-process retrieval, and avoiding repeated one-time rewards. No fixed Minecraft opening is embedded in the module. This operationalizes the study's separation of exact bookkeeping, strategic planning, and fast recognition without violating held-out generality.

Long observation arrays are sampled into valid `{sample_of, values}` objects **only for model summaries**. Every dictionary field is retained, including fields occurring after large arrays. Astra receives up to 128 entries per array; Jev receives up to 32, plus the complete selected candidate descriptions. The interpreter always sees the full observations. Sampling can still hide relevant details, and the model is explicitly told omitted facts are unknown.

## Jev interface and answer-to-action bridge

Every request carries a named state object:

* `rules`, `interface`: the actual game description, action meanings, neutral action, and notes.
* `plan`: Astra's goal, field/mechanic semantics, selection hint, and danger hint.
* `observation`, `summary_note`: current structured observation and sampling caveat.
* `options`: at most 12 concrete commands, each with a key, rule ID, explanation, priority, target cost, target record, and emergency flag. The interpreter retains at most three targets per rule before sorting by urgency, priority, and cost.
* `busy`, `recent`, `version`: interpreted job status, the last eight dispatch/outcome events, and policy version.

Three independent questions share one `ctx.decide` call:

1. **`option` / Choice:** recognize the safest useful concrete alternative for the current subgoal. Options include explicit `unknown` and `hold` choices. Jev never invents a target ID or numerical action mapping. Dispatch requires the selected option's probability ≥0.50 and, when confidence is present, confidence ≥0.15. Missing confidence does not become certainty; the probability gate still applies.
2. **`danger` / Choice:** assess whether continuing the current activity is clear, requires urgent interruption, or is unknown. Normal actions require `clear` probability ≥0.60. An observed urgent interruption with `stop` probability ≥0.75 can post Astra's validated stop command while busy. An emergency option can be selected despite uncertain general safety if its own current guards hold.
3. **`revise` / Noul:** estimate whether failures, changed threats, missing prerequisites, or lack of productive idle options require a new strategic interface. Probability ≥0.80 requests Astra revision. Safe ongoing bounded work is not itself a reason to replan.

These thresholds are conservative initial design choices, not calibrated or fitted from these traces. The archive supports the latency and syntax diagnoses, not numerical confidence calibration.

Responses carry their request time, option menu, and policy version in harness bookkeeping. A reply older than **2.5 s** or from a replaced policy is discarded. The selected command and rule must still occur in a newly instantiated current menu; immediately before posting, the harness obtains another observation and checks again. Disappeared targets, changed guards, cooldowns, or a newly busy job therefore invalidate the action. A policy is reusable semantic context, not a reusable authorization to execute an old command.

`jev_control` events record choice probability, confidence, response age, revalidation, danger, resulting command, and revision requests. `command_posted`, `option_failed`, `jev_discarded`, and policy events make the actual division of labor inspectable. Routine neutral heartbeats are not presented as new semantic decisions.

## Scheduling, fallback, and lifecycle

There is at most one outstanding Astra request and one outstanding Jev request, both through `ctx.spawn`. The main loop posts an action and uses `ctx.wait_for_tick()`; it never calls `game.tick()` or spins. All spawned futures are joined in `finally`, even if another one fails. `ModelUnavailable` propagates; broad `RuntimeError` swallowing is removed. Only transient `TimeoutError` is converted into a recoverable result.

While inference is pending, the documented neutral action continues bounded jobs or idles. A bounded command is posted through a tick boundary, followed by a neutral pulse so deliberate later repeats are not trapped by command latching. While the policy's task expression indicates busy, only an interruption can replace the job; other options wait for observed idle. The optional continuous-control mode renews a last action for at most one second without another approved choice.

Astra refreshes about 35 s after successful installation, or sooner for requested revisions, with at least 10 s between revision launches. Parse failures back off eight seconds. There are at most **12** planner launches, with a budget reserve and checks on reported game time and wall budget. Planner calls have up to 40 s; Jev calls up to 2 s, capped by available time. Budget exhaustion continues the control loop rather than prematurely ending the episode. A failed concrete command is suppressed for 25 s and forwarded to the next planner. Per-command cooldowns default to three seconds to avoid feedback-lag replay.

The initial policy still costs real time, and neutral fallback is not a universal emergency reflex. This candidate does not claim to solve sub-tick hazards before Astra discovers the relevant rules. Safety depends on the compiled guards and observed alternatives; fabricated semantics, inaccessible targets, or slow policy compilation can still lose an episode.

## Confirmation and refutation

Confirm the hypothesis if matched evaluator seeds show complete parameterized commands reaching the adapter, disappearance of the systematic `bad_arguments` loop, positive authoritative resource/milestone progress, and better mean score than both baselines. Require episodes to reach the true terminal condition without timeout-induced harness errors. Check the new model traces for actual inference latency; the harness summary's elapsed latency also includes tick-boundary polling delay.

Refute or narrow it if complete legal commands produce little progress, if the rule compiler repeatedly fails or exhausts its timeout, or if terrain/tool/dependency errors dominate after syntax is fixed. Frequent `jev_discarded` events would show the larger state or freshness limit is ineffective. Many high-confidence choices rejected by current guards would indicate the interface is too stale or poorly scoped. A low stale fraction alone does not confirm success because neutral heartbeats can reduce it without improving play.

Likely next branches should be selected from those observations: simplify the compiler if bootstrapping dominates; improve learned guards if invalid attempts dominate; improve target comparison if fully legal work is unsafe; optimize inference frequency only after productive execution is established.

## Validation performed

No real game episodes or model experiments were run. `test_contract.py` uses synthetic observations and stand-in typed models only. All **13** short offline checks pass under the supplied Python environment: garbage/first-option output, full argument preservation and neutral pulses, vanished targets, transient timeouts, both ordinary and timeout-subclass `ModelUnavailable`, exact expression arithmetic, preservation of late observation fields, exhausted budgets, low-probability rejection, busy-job exclusion, stale-response rejection with future joining, and explicit-failure cooldown. This is a local contract check, not the evaluator's official pre-flight or an estimated score.
