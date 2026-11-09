# 0005 — Crossover with an explicit motor clock

Parents: `0000-tier2_jev_only` and `0001-tier2_astra_only`.

## Diagnosis and hypothesis

The plateau is principally an execution-interface failure, not evidence that models cannot recognize UP. In parent 0000, seed 101, all 464 fresh ticks apply UP; the first at tick 13 yields score 1 at tick 14, but later fresh UP postings do not release the joystick. The model's state first records a death at tick 32, a sidewalk reset at tick 95, and subsequent timeouts at ticks 1986, 4018 and 5936, while every answer remains UP. All six seeds finish at one point. Parent 0001 seed 103 has zero posted actions: its only model call is cancelled after 162.152 seconds when the game ends. Seed 101's first Astra UP takes 59.112 seconds and only reaches the first road row. No `lessons.md` exists in this archive at proposal time.

Jev's useful mechanism is cheap repeated typed action recognition and robust exception handling. Astra's is strategic rule interpretation with a persistent session and a legal-action fallback; it has not demonstrated successful routing in its baseline. This crossover keeps both model interfaces but removes either from the motor timing critical path. The hypothesis is that a per-frame executor with verified direction/release cycles and geometric short-horizon proposals will break the one-point plateau, while asynchronous Jev choices and durable Astra policy changes improve route selection without buying idle game time.

## Architecture

- Code observes every tick, tracks wrapped object displacements, checks landing/carrier clearance and searches short sequences. Only the first proposed hop is executed; all geometry is rechecked at the next decision boundary. No game implementation, emulator cloning, private state or seed-specific route is accessed.
- A direction is held until a public observation confirms movement (at most four ticks), then explicit NOOP frames release the input. Initial ten-tick hop spacing and eight-pixel horizontal spacing are provisional; observed lateral displacement calibrates the latter. The executor posts even NOOP every tick and uses `ctx.wait_for_tick`, not spin waiting.
- Jev runs in one bounded outstanding future. Its state contains explicit rules, Astra's current policy, zone, a named timer-urgency bucket, three code-ranked feasible candidates (action, landing row/x, reachability, safety bucket, forward/towards-home predicates), supported-wait and home uncertainty. It never receives raw RAM or irrelevant metadata.
- The batched questions are two Choices with reversed option ordering selecting the next feasible action, plus a Noul recognizing urgent carrier escape. Code accepts matching action Choices only when both winning probabilities are at least 0.65; missing probabilities imply rejection. UNKNOWN is explicit. A result must be no more than 30 ticks old with unchanged life/home/level, plan version and row, and at most ten pixels of drift; it is then revalidated against current feasible near-best candidates. Semantic escape is diagnostic; geometric safety cannot be overruled by it. Events distinguish accepted advice, discarded advice and an actual action changed by Jev.
- Astra runs concurrently and emits only validated enum JSON: open-home ordering, pace and focus. The allowed focus selects forward progress, wide platforms or home alignment. It sees recent deaths/homes and candidate facts, not a request for an immediate joystick direction. Initial timeout is 120 seconds, above the archived 97-second tail. A new call is considered after 60 seconds when failure/stall evidence is available; policy is invalidated on incompatible level change. Arbitrary text or invalid keys preserve the previous executable policy.
- Model calls use only `ctx.think` and `ctx.decide`. Transient inference failures preserve the executable controller; lifecycle and unavailable-model exceptions propagate normally. There are no background Python loops to outlive the episode.

## Fallbacks and uncertainty

While either model is late, code executes the freshest safe leading proposal, or NOOP when no supported landing exists. This is not a cached lethal direction. Diving and unknown carriers are excluded, occupied homes are excluded, and an unknown crocodile is never called absent. Current interval widths are estimates, not an asserted exact simulator; live training tries will test hop cadence and collision margins before refinement.

## Evidence criteria

Confirmation: distinct repeated UP hops with intervening observed NOOP, multiple forward score changes, reaching median/river or homes, fewer timeout-only lives, and useful `jev_changes_action`/`astra_policy_adopted` events without stalled motor output. Refutation: repeated missed pulses despite releases, geometry-predicted safe hops repeatedly causing immediate deaths, or persistent failure to complete homes despite increased movement. Record trial outcomes and corrections below; distinguish measured results from conjecture.

## Initial status

Complete module written before any live try. Parent trace analysis establishes debounce failure; geometry and action timing remain provisional pending live observation.
