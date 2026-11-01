# Candidate 0019 — crossover of corridor control and rolling-ball memory

**Mode:** crossover. **Parents:** `candidates/0016/` and `candidates/0015/`.

## Hypothesis

The parents solve complementary parts of the same failure. 0016's neutralized, one-hop handshake with a landing/escape corridor is the best demonstrated motor and coverage baseline (21,808 mean), while 0015 demonstrates that a localized memory of a recently rendered rolling ball can alter candidate routing after the sprite disappears. Neither explains the final deaths: 0016 reached six inferred clears in 4/6 seeds, but all four deaths in its seeds 0 and 5 after the first occurred with no visible hazards and `purple_ball_state=36`; 0015 seed 2 had the same pattern (losses at 2432/2514/2942, empty hazards/state 36) even though its 5–60 tick shadow was exercised at ticks 1974/2014/2024.

The working hypothesis is therefore narrower than “a longer shadow prevents the hidden ball”: retain 0016's protocol and corridor scoring, add 0015's soft, spatially localized descent memory, and move Jev's decision from *this* hop to the **next junction while the current hop animates**. A normal Jev call takes about 0.18–0.24 s, or roughly three game ticks, while a hop generally lasts eight or nine ticks. In 0016 Jev advice was often stale before it could affect the current hop (86/100 replies stale in representative extremes; only 8.3% of actions were preceded by fresh Jev). Asking one hop ahead should convert latency into useful control time without blocking the motor. The response still has to survive freshness and code-safety gates at arrival.

This does not assume `purple_ball_state=36` localizes a ball. No state-36 direction bias, invented sprite position, learned disc exit, or guaranteed Coily behavior is introduced. Unseen collisions may instead be Coily or another interaction.

## Evidence from parents

* **0016 motor/corridor mechanism:** It reached 5–6 scored clears, with six in four seeds. Its verified one-hop transaction reduced uncontrolled held-edge risks. However, seed 0's first life loss at tick 1722 occurred while a Purple Ball was rendered (node 3, Q*bert node 15), and subsequent losses at ticks 2148, 2972, and 3122 were in empty-hazard/state-36 observations. Across seeds 0–5, repeated late losses likewise overwhelmingly have no rendered hazard. Its handshake and corridor logic thus supports progression but does not cover occluded-object time windows.
* **0015 memory mechanism:** A recent rendered ball's downward-only descendant cone is a plausible localized risk cue, activated after four clears. It changed several late edge rankings, but seed 2 still lost at 2432/2514/2942 with empty hazards/state 36. This is evidence the memory is worth retaining as a *soft* cost, not evidence that the cone predicts the collision reliably. Its first seed-2 loss at tick 1161 also occurred before shadow activation.
* **Adviser timing:** 0016 opens Jev calls while standing, but a response arriving during or after a hop can be stale by the next current-node decision. The 0.18–0.24 s observed latency is below a hop's ~0.53–0.60 s elapsed time, so speculative next-junction choices are a testable latency-pipeline improvement.

## What changed

### Keep 0016's motor, simplify its advisory interface

The controller retains the 21-node graph, palette/score flash epochs, deterministic coverage order, corridor risk/escape count, single-hop neutral handshake, landing verification, bounded recovery after life loss, and asynchronous Astra. It never queues multiple hops and only posts a directional action if the adaptor's current `safe_moves` permits it. Model output is never an action string or a route; `FIRE` is absent.

### Import only a bounded 0015-style spatial memory

Whenever the RGB-verified observation exposes a Purple or Red Ball, code stores its observed node and tick. For at most 60 ticks, it computes possible descendants only along the board's DOWN/RIGHT rolling directions, with a small time-banded depth limit. Candidate landing risk gets a soft penalty if it lies in this cone. It does not mark a move illegal, does not convert timer state 36 into an object, and clears the history on a scoring board epoch or life loss. This component can influence both the route cost and compact Jev facts. Its purpose is local tie-breaking in the documented occlusion gap, not a sole safety mechanism.

### Ask Jev at the next junction, then revalidate

When the controller commits one hop from node A to B, code speculatively computes the legal outgoing corridors from B and starts one asynchronous Jev request. The compact state schema is:

```json
{
  "game": "Atari 2600 Q*bert, six-row 21-cube pyramid",
  "decision_timing": "current hop underway; judge only next junction",
  "current_hop": {"action": "DOWN", "projected_landing": 7},
  "next_position": {
    "standing_after_hop": 7,
    "candidates": [
      {"key": "RIGHT", "landing": "unfinished",
       "lethal_exposure": "possible_branch",
       "escape_after_landing": "several", "route": "near", "code_rank": "alternate"}
    ]
  },
  "threats_now": [{"kind": "purple_ball", "visibility": "rendered", "relation": "distant"}],
  "progress": {"inferred_board_clears": 4,
               "incomplete_bucket": "several",
               "priority": "survival and required phase progress"}
}
```

The batched request contains two reordered `Choice`s over supplied legal candidate directions plus `keep_code_choice`. Jev judges (1) the safest useful next corridor, not whether to change the in-flight action; option text distinguishes concrete from possible rolling exposure, number of escape routes, and useful/transit landings; and (2) a repeated reordered judgment to reduce first-option bias. Code calculates geometry, distances, candidate risk, and target progress.

At the arrival node the motor waits for a neutral tick, recomputes candidates from the latest observation, and accepts advice only if both Choices agreed, the winning option has at least **0.60** probability in both, the node/life/board/policy epochs match, and the response is at most **12 ticks** old. Jev is a safety specialist, never a generic tie-breaker for coverage: it can override if the latest corridor recomputation shows at least **0.45** less risk (or **0.80** less if the rendered hazard signature changed), while remaining within **2.0** route cost and no more than one fewer certified escape. Alternatively, a same-risk choice is allowed only with at least **0.72** confidence, at least one additional certified exit, no more than **0.5** extra route cost, and no loss of landing usefulness. Equal-risk choices without that concrete escape gain remain with deterministic code. Otherwise deterministic corridor ranking wins. This arrival-time gate prevents model choice from becoming a directionally arbitrary tie-breaker.

### Astra designs durable posture, not actions

Astra is asynchronous, limited to two calls, and never on the motor clock. It chooses between center/upper/frontier tie posture and balanced/branch-aware sensitivity to the soft rolling cone. The initial call establishes a mode; one subsequent call may review after a life loss or board transition. JSON is strictly whitelisted and malformed/prose answers leave the policy unchanged. Astra cannot supply a board route, action, hidden hazard location, or disc exit.

## Timing, fallback, and terminal policy

The fallback is always a fresh deterministic legal corridor from the 0016-style ranker. During a hop, neutral pulses and the existing bounded retry/landing handshake proceed independent of Jev/Astra. Missing, late, failed, malformed, disagreed, low-probability, stale, or newly contradicted Jev answers are ignored. A life loss invalidates pending advice, clears trajectory memory, and runs a short neutral recovery gate. The controller stops only on `game.done` or `ctx.done`, never on the reported `lives == 0` field.

## Confirmation and refutation

**Support** would be: retaining 0016-like early clears while Jev has a materially higher rate of fresh, accepted next-junction reviews than its 8.3% action freshness; some accepted choices should differ from the deterministic fallback because of exit/shadow facts; and the runs should show more fifth/sixth clears or later survival, especially fewer repeat losses shortly after a rendered ball disappears. The trace should link each accepted choice to its projected landing, fresh observed node, recomputed risk, and subsequent landing/life event.

**Refutation** would be: answers are still mostly late or invalidated, Jev rarely changes an action, the shadow does not change viable routes, or repeated empty-hazard/state-36 losses occur at the same rates despite the combined controller. If the route decisions are unchanged but performance improves, the motor/corridor baseline or ordinary seed variation—not these new mechanisms—would be the likely explanation. If safety gates reduce throughput without reducing losses, the soft memory or advice margins are too conservative.

The targeted diagnostic is the conditional distribution of losses after a visible rolling-ball sighting and the accepted next-junction decisions. Neither lower stale-tick percentage nor raw S1 call count alone confirms the hypothesis.
