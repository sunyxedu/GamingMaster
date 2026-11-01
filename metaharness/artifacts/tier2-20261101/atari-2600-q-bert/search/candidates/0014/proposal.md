# Candidate 0014 — temporal shelter and referee bridge

**Mode:** rewrite. **Parent:** `candidates/0009/`. The best comparison is `0007`; `0011` is the most informative attempted repair of the same late-round failure.

## Hypothesis

The late ceiling is not primarily a landing-decoder problem. It is a **temporal route commitment** problem: the controller sees a legal edge, but it chooses a bottom endpoint or immediately reverses into one while a Purple Ball is in a cartridge phase in which the RGB decoder suppresses it. The next observation can still say `hazards=[]` and give several `safe_moves`, even though the hop/return path has already become lethal.

The evidence is unusually consistent:

* In `0009/trace_seed4.jsonl`, after the fourth inferred clear, tick **1783** is at node 14 with an adjacent Coily in the compact radar and an occluded Purple state. The controller nevertheless commits `RIGHT` toward endpoint 20. It verifies the endpoint at tick 1792, tries to return, and loses a life at tick **1802**.
* In `0009/trace_seed3.jsonl`, tick **2014** waits at endpoint node 15 because the radar predicts a dangerous exit. It then leaves to node 10 at tick 2025 but immediately posts `DOWN` again at tick **2027**, returning toward the same endpoint. At tick 2033 the observation exposes Coily at node 16 only after the reverse hop is already in progress; the life is lost at **2044**.
* The loss observations in these windows repeatedly show `hazards=[]` together with `timers.purple_ball_state=36` (or another high state). Thus the parent's `recent_shadow`/`occluded` label is useful recognition but only a soft cost; it does not control the route mode. `0011` changed corner ranking, but its first divergence already chose a different unsafe route on board one, and it logged no useful corner deferrals.
* Verified landings did not fix either death: `0009` had no meaningful mismatch in the late windows. The issue is what is selected **after** a correct landing and how long the controller remains committed to a corner, not merely whether the intended node was decoded correctly.

The testable prediction is that a controller that treats a high hidden Purple timer as a bounded shelter/recovery mode, forbids an immediate reverse when a roomier edge exists, and learns finite bad-edge memories after a loss will preserve more lives after the fourth clear without sacrificing the reliable first-board traversal. A correct temporal policy should show fewer endpoint entries during latent-purple intervals, fewer immediate reverse pairs, and later scoring epochs. If it instead stalls on the first board or still dies at central nodes with a quiet timer, then the timer is not a useful hazard phase and the missing mechanism is likely an escape-disc/Coily policy.

## Structural change from 0009

The parent has one planner that ranks coverage edges and a radar that adds penalties. This candidate has three explicit layers:

1. **`PaletteBoard` is the only progress authority.** It learns a destination from public scored colour transitions, quarantines whole-board flashing, and treats inferred clears as diagnostics. It does not trust a stale all-zero `needed_landings` vector during a handoff. The official score, life and terminal signals remain evaluator-owned.
2. **`HazardField` produces two kinds of fact:** finite spatial shadows for visible/recent lethal objects, and a separate `latent` temporal fact when the named Purple Ball is absent but the public `timers.purple_ball_state` is high. The latent fact is not fabricated as a node; it changes the route objective to a shelter objective and gives endpoint/downward commitments a large cost. All shadows expire.
3. **`RoutePlanner` owns a short, closed-loop route, not a model-selected action.** It selects a target and ranks only current `safe_moves`. During quarantine or life recovery it targets central, multi-exit shelter nodes. It conditionally filters immediate reversals, bottom endpoints, and edges recently implicated in a life loss, but retains a required endpoint if no alternative can reach the current objective. A hop transaction posts a direction at most twice, sends `NOOP` during animation, verifies the observed landing, and records source/destination edge memory. A bounded `NOOP` is used only when the current cube is safer than all exits.

This is deliberately different from 0009's soft radar tie-breaker: hidden-timer recognition changes the **subgoal and route class**, while the motor remains deterministic. It is also different from 0011's unconditional corner penalty: the endpoint filter is conditional on quarantine, recent edge evidence, and availability of a roomier alternative.

### Loss memory

On a life decrease, the active transaction and last observed edge are marked in both directions for 170 ticks. The target is discarded, the motor waits for respawn, and a 120-tick recovery posture favors shelter. This is finite adaptive bookkeeping, not a seed-specific route or a permanent ban: a required edge can become available again after the memory expires.

## System-1 division and state schema

Jev is no longer asked to choose a graph edge on every request. Code has already computed the graph, distances, candidate risks, escape counts, endpoint/reverse flags, the latent timer phase, and the proposed edge. Jev judges only the timing role of that code plan. A request carries a compact strict-JSON object of this shape:

```json
{
  "game": "Atari 2600 Q*bert; six-row 21-node board",
  "position": {"node": 15, "standing": true, "last_posted": "NOOP"},
  "progress": {"epoch": 4, "unfinished": 8, "inferred_clears": 4, "target": 7},
  "temporal": {
    "purple_timer": 36, "latent_purple": true,
    "quarantine_age": 3, "survival_recovery": false
  },
  "candidates": [
    {"role":"escape", "action":"UP", "landing":"transit",
     "danger":"clear", "escape_space":"multiple",
     "temporal":"ordinary", "code_rank":0},
    {"role":"coverage", "action":"DOWN", "landing":"unfinished",
     "danger":"near", "escape_space":"none",
     "temporal":"reverse", "code_rank":1}
  ],
  "proposed_action": "UP",
  "hazards": [{"kind":"purple_timer", "visibility":"occluded",
               "proximity":"uncertain", "motion":"active"}]
}
```

The actual implementation includes only derived semantic buckets, not raw RAM, RGB values, or arbitrary model text. The candidate action list is generated by code and revalidated at execution.

Each asynchronous Jev request batches three independent typed judgments:

* `timing_a`: `Choice` over `commit | hold | retreat | unknown`.
* `timing_b`: the same choice with reversed criteria order, to reduce first-option bias.
* `immediate`: `Noul` asking whether the supplied proposed edge has concrete immediate lethal risk.

The answer-to-action bridge is intentionally indirect:

* The two Choices must agree and have a minimum selected-option probability of **0.60**. Otherwise the deterministic route remains in control.
* `hold` can select only a code-created `NOOP` candidate and is bounded by the planner's 5–8 tick wait window. It cannot create indefinite idle behavior.
* `retreat` can select only a code-created alternative with no bad-edge mark and risk no more than 1.5 above the fallback. Jev cannot name a direction or override `safe_moves`.
* An `immediate` probability of **0.82** can veto the proposed edge only in favor of a supplied comparable alternative; if no such alternative exists, the code route (or a deliberate `NOOP`) is retained.
* Answers are tagged with tick, node, palette epoch, life epoch, route-policy version, and a hazard signature including the latent timer bucket. They expire after **5 ticks**, a node/life/epoch/policy change, a hazard signature change, or loss of standing posture. Missing probabilities, garbage stand-in answers, disagreement, timeout, and all other model failures are safe fallbacks.

Requests launch at most every **8 ticks**, normally during a hop or between decisions, with a **0.95 s** timeout. The world is never blocked for Jev. Every game tick still receives a legal action or deliberate `NOOP`.

## Astra division

Astra is an off-clock author of a small durable contract, not a joystick planner. At initialization, a palette epoch, and a life loss (subject to a 300-tick cooldown), it receives score/lives, inferred progress, recent edges, public hazard facts, timer state, and the current policy. It is asked for only:

```json
{"route":"sweep|nearest|central", "temporal":"quarantine|balanced"}
```

`route` changes coverage tie-breaking; `temporal` changes only the severity of ordinary (non-latent) radar costs. The latent high-timer gate and endpoint protection cannot be disabled by Astra. Prose, malformed JSON, unsupported values, or a slow/cancelled call leave the current policy untouched. Astra is capped at three asynchronous calls with a 105-second timeout and is never awaited by the motor.

This puts long-horizon interpretation where it belongs: Astra can decide whether a later-board stall calls for a central route or a balanced shadow, while code enforces the graph/colour/timer invariants and Jev performs a cheap local timing check.

## Fallback and confirmation

The executable hierarchy is:

1. `NOOP` during global flashing, falling, an unanchored sprite, the post-landing neutral gate, or a bounded shelter wait;
2. complete/verify an existing one-hop transaction, never holding it as a macro;
3. choose the lowest-risk current `safe_moves` edge from the shelter or coverage planner;
4. use a finite loss-memory detour or roomier alternative; and
5. if every edge is exposed and the wait bound expires, take the least-bad required legal edge rather than deadlocking.

The candidate never uses FIRE, RAM writes, an arcade 28-cube map, guessed disc exits, seed-specific action strings, or the reported life counter as a termination condition.

Evidence supporting the hypothesis would be: (a) no endpoint entry/return pair during `latent_purple=true` when a central alternative exists, (b) fewer `source→destination→source` reversals immediately before losses, (c) fewer repeated losses on the same directed edge after `edge_memory_after_loss`, and (d) more than the parent's four inferred clears in most training episodes while stale ticks and model failures remain near zero. Refutation would be a large first-board score loss from over-quarantine, high-timer waits that never resolve while Q*bert is safe, or deaths that occur with no latent timer, no recent reverse, and no spatially reachable lethal object. The latter would motivate a future disc/Coily lure experiment rather than another route penalty.
