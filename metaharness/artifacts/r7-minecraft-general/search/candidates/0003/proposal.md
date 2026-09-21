# Candidate 0003: short grounded menus instead of policy compilation

Parent: **0002**. Supporting comparisons: **0000-tier2_jev_only** and **0001-tier2_astra_only**.

## Hypothesis and diagnosis

The best candidate's immediate bottleneck is **time to first executable output**, not demonstrated weakness in Jev's decisions. Asking Astra to compile a reusable guarded policy makes the entire action loop depend on a large response that never arrives. Replace that compilation task with a small menu of complete, grounded commands. Let Jev recognize readiness and choose among those commands, while code enforces simple exact guards, target presence, one-shot execution, and job exclusivity. This should produce resource/milestone progress before the episode ends.

Evidence from the archive (one-based JSONL line numbers):

- All three 0002 episodes tie at **20.000**. Across their traces, **all 24 Astra calls fail**, taking **40.007–40.034 s**. There are no successful Astra responses and no Jev calls. This is not evidence that a compiled policy played badly: no policy was installed.
- `0002/trace_seed0.jsonl:84` records `ok:false`, `error:"CLI invocation exceeded 40.000s"`, and `response:null`. Line **86** records `policy_rejected`, `reason:"planner timeout"`. The same pattern repeats eight times per episode; only seven rejections are processed during active play because the last future is joined afterward.
- The same trace's line **633** reports `plan_attempts:8`, `s1_observed_latency_s:[]`, and **`commands:0`**. Its entire tick-action census is **600 `noop` ticks**. Line **634** reports score 20, zero stale ticks, and wall time **323.525 s**. Healthy idling explains the score; neutral heartbeats explain the freshness. Neither establishes useful control.
- The first 0002 planner prompts are **31,259–36,697 characters**. They request 4–7 initial rules, up to 1,800 output tokens, with a substantial expression language, policy metadata and the full Jev guide. The trace cannot separate reasoning complexity from output length or other backend latency, so the hypothesis concerns the combined compilation burden.
- Short concrete output has previously arrived: `0001-tier2_astra_only/trace_seed0.jsonl:17` returns **`gather b:-34,67,41 4`** in **7.4286 s**. Line **19** applies only **`gather`**. This candidate preserves the whole command. That episode eventually scores zero; it is evidence about response latency and the broken bridge, not evidence of successful strategy.
- `0000-tier2_jev_only/trace_seed2.jsonl:19` and **749** show bare `gather` failing with `bad_arguments`, empty authoritative inventory, and zero progress. Line **760** records a **5.1600 s** timeout that terminates that baseline early. Successful Jev calls in that episode average **0.8229 s**, already longer than one tick. The new loop remains asynchronous and treats ordinary inference deadlines as recoverable.

## Experimental change

The large expression interpreter and runtime policy compiler are removed. Astra now returns one short goal, essential semantics, three discovered observation paths, a control mode/interrupt command, and **2–4 concrete options** (parser accepts 1–5). The prompt targets **under 450 output tokens**. Options include the complete command, explicit observable readiness conditions, current object IDs that must remain present, and optional simple numeric/value guards.

This is a receding-horizon action menu, not a generated program. Later steps may be offered before prerequisites finish, but they are never presumed ready. A command is used at most once within its menu. Commands sent after an outstanding planner's snapshot are marked used when that delayed menu arrives, preventing its replay of newly completed work. A command also has a 12-second cross-menu cooldown. The previous menu can continue while a replacement is being prepared.

Code checks conjunctions of `[path, comparison, value, optional_missing_default]`. Comparisons are exact `eq/ne/ge/le/gt/lt/contains`; absent fields fail closed unless Astra explicitly supplies a justified default. These checks handle resource thresholds and observed prerequisites without regenerating an expression language. The interpreter does not execute generated Python. Astra remains responsible for choosing quantities and dependency order; incorrect model arithmetic is a remaining risk, not magically solved by the guard checker. Public feasible-action/recipe data and subsequent authoritative outcomes are the grounding sources.

Command validation preserves all whitespace-separated arguments, checks the verb against the public action specification, and checks minimum argument counts when a signature is present in its descriptions. Code rechecks guards, object-ID presence and exclusive-job status against a new observation immediately before posting. These are partial legality checks, not a replacement for the adaptor's authority.

No game name, item, recipe, coordinate convention, score formula, action verb, or observation field is embedded in the harness. Paths, commands, quantities, semantics and interruption behavior come from `game.describe()`, `game.action_spec` and runtime observations. The synthetic tests deliberately use a different vocabulary. The study informs the prompts' prerequisite-first planning, authoritative confirmation, exact quantity guards and asynchronous delayed-work handling; no Minecraft solver is compiled into the module.

## Jev state and bridge

Each request explicitly carries:

- `rules` and `interface`: public rules and complete action descriptions.
- `plan`: Astra's current goal and essential field/mechanic meanings.
- `observation`: structured current facts; all dictionary fields are retained, long arrays sampled with original indices.
- `options`: complete commands, readiness descriptions, reference IDs and exact guards that currently pass.
- `target_evidence`: original records containing selected reference IDs, independently retained even when the general array sample omits them.
- `busy`, `recent`, `version`, and `summary_note`: job status, recent dispatch/outcome evidence, menu version and explicit sampling caveat.

One `ctx.decide` batches independent questions:

1. **`choose` / Choice** compares the few offered commands for the current subgoal. It includes `hold` and `unknown`; Jev does not invent a command or action mapping.
2. **`ready_aN` / Choice**, one per offered option, judges whether *all* its observable prerequisites, safety and usefulness hold now. Choices are `unknown`, `ready`, and `blocked`. This assessment does not depend on the selection answer, and must not predict that an unfinished job has completed.
3. **`danger` / Choice** independently detects a need to interrupt a running job. Supported immediate danger can post Astra's validated interrupt command; any replacement waits for observed idle.
4. **`revise` / Noul** requests a new menu when facts contradict the goal, a target/prerequisite is missing, or failures require adaptation. Safe ongoing work is not itself a revision trigger.

A normal dispatch requires selection probability **≥0.45**, independently assessed ready probability **≥0.70**, and confidence **≥0.10 when supplied**. Missing confidence is not replaced with certainty; probability gates still apply. Interruption requires `stop` probability **≥0.75**; revision requires Noul **≥0.80**. These are initial thresholds, not calibrated estimates derived from the archive.

A judgment must be at most **3 seconds old**, from the current menu version, and still match an available option after current guards/reference/job checks. Exact guard checks do not fully cover every natural-language readiness fact; rapid environmental changes within that window remain a risk. Menu authorization expires after **40 seconds**, even if refreshes fail. Jev receives relevant rules every time. Astra receives the relevant local Jev contract in the menu prompt; it is not asked to generate question API objects or return prose from Jev.

## Scheduling and revision

There is at most one outstanding Astra future and one Jev future. Astra calls have a **28-second maximum deadline**, capped by discovered remaining episode time and wall budget. Smaller output is the substantive change; simply shortening the old compiler's deadline would not fix it. Requests are at least eight seconds apart, limited to 30 per episode, and leave a model-call budget reserve. Revisions trigger on an 18-second menu horizon, depletion to one unused option, or Jev's evidence-based request. Parse/deadline failures back off six seconds. The old menu remains usable within its expiry while replacement is pending.

Jev has a 2.5-second request deadline, capped by available time, and runs again after a short idle delay or approximately one second while a job is busy. Actual request duration is measured with `time.monotonic()` and logged separately from observation age, which includes tick polling. Standard model traces remain authoritative for provider latency.

The executor posts through `ctx.act`, then uses `ctx.wait_for_tick`; it never ticks the game itself. Bounded commands receive a neutral pulse on the following tick, preserving ongoing work and clearing the latch for deliberate later reuse. Continuous controls, if discovered, have at most a one-second renewal lease. Neutral fallback runs while inference is pending, unavailable under budget, rejected or expired. Initial startup still idles until the first menu; there is no game-specific emergency controller before semantics are discovered.

Ordinary `TimeoutError` is recoverable; `ModelUnavailable`, including a timeout subclass with that name in its MRO, propagates. All outstanding futures are joined in `finally`, even when another fails. Exhausting inference allowances does not terminate an unfinished episode. No backend access, shared cross-episode cache or privileged game operations are used.

Events `menu_requested`, `menu_installed`, `menu_rejected`, `jev_control`, `jev_discarded` and `command_posted` expose the causal bridge. `jev_control` logs probabilities, optional confidence, request latency, age and proposed dispatch; `command_posted` records actual execution. Neutral posts must not be interpreted as productive model-informed decisions merely because the evaluator reports low stale ticks.

## Confirmation, refutation and limitations

Confirm the mechanism if successful small menus arrive materially before the parent's 40-second deadline, produce fresh Jev judgments, and post complete commands that yield authoritative resource/milestone progress. Confirm the practical claim if matched evaluator seeds exceed **20 mean score** while finishing valid episodes. Inspect time to first productive command and completed useful actions, alongside the official score/freshness metrics.

Refute the latency hypothesis if even these short menus consistently time out. If menus arrive but Jev authorizes nothing, examine readiness disagreement, omitted evidence, and overly restrictive guards before changing thresholds. If commands execute but fail or earn no credit, the next bottleneck is grounding/dependency planning or target selection. Frequent replacement-version discards or repeated replanning would indicate that short snapshot menus are too transient. Any survival loss is part of the score tradeoff and must be measured.

This iteration jointly reduces output complexity and changes the representation, so it does not isolate a pure token-count effect or Jev's marginal contribution. Snapshot targets can disappear, sampling can omit useful terrain, guards can be wrong, menus can become obsolete, and interrupting a job alone may not evade a threat. A positive result would justify improving menu coverage; it would not prove general optimality.

## Validation

No real episodes or model experiments were run. `test_contract.py` performs **13 passing offline checks** with synthetic observations and typed stand-ins: garbage output, arbitrary first options, full command preservation and neutral pulses, exclusive-job ordering, vanished targets, exact/missing guards, transient timeout recovery, both forms of `ModelUnavailable`, exhausted budgets, stale judgments, malformed metadata/arity, changed prerequisites, low probabilities and late-field preservation. Futures are verified joined. These are local contract checks, not an evaluator score or the official pre-flight.
