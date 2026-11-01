# Candidate 0020 — conditional endcap deferral with in-flight next-hop advice

**Mode:** explore. **Parent:** `candidates/0016/` (best controller; `0007/` is a trace-level corroboration, not a code parent).

## Hypothesis

The remaining losses may be amplified by repeatedly treating the two one-exit bottom tips as ordinary coverage targets. This candidate tests a precisely conditional alternative: while any unfinished non-tip cube remains, nodes 15 and 20 are deferred as destinations; once the interior is complete, the tips become the active targets and are completed normally. This is not a permanent corner prohibition and does not add a generic numeric penalty to every route. The aim is to reduce exposure to an endcap while useful alternatives still exist, then finish the unavoidable tips as a compact final phase.

The evidence is suggestive, not causal. In `0016/trace_seed0.jsonl`, the third inferred scoring clear is logged at tick 1282. On the still-incomplete subsequent board, the route lands at node 20 at tick 1593, leaves via LEFT, and loses a life at tick 1603. Later the same run again crosses node 20 before losing a life at tick 2109; overall it ends at 23,125 after six inferred clears and four reported losses. These records do not prove the bottom tip caused either collision: hazards may be missed by the sparse observation trace, and the first death can occur during the escape hop. They do show the route compiler deliberately selected a one-exit destination while ordinary coverage remained. In `0007/trace_seed0.jsonl`, after the fourth inferred clear, the route repeatedly revisits the 14↔20 corner corridor; at tick 2219 it selects DOWN from node 14 and loses at 2222 with Coily estimated at node 9, then later loses in an empty-hazard/state-36 window. This is consistent with unnecessary endcap traffic being an exposure amplifier, not a complete explanation for the hidden-hazard deaths.

This exact conditional experiment has not been completed. `0011` proposed a corner cost but its +40 penalty remained unconditional; its first divergence from 0007 occurred on board one and traded away a route labelled clear for one labelled intercept-possible. The lessons record **zero** `corner_commitment_deferred` events, so its nominal condition did not test a strict postponement rule. Candidate 0020 instead excludes a one-exit tip from the active target set only while non-tip demand exists. No cost is added to an alternative route, and an emergency no-options fallback remains available.

## What changed

1. **Target-set scheduling, not local edge scoring.** The ledger continues to learn destination colour from a scored +25 landing (including a possible +300 capture) and recognizes the global palette-flash/score tally as a new board. The coverage policy forms two sets: unfinished interior cubes and unfinished degree-one tips. If the interior set is nonempty, only it can be selected as a target. Once empty, the controller switches to the remaining tips. The directional edge is still checked against the observation's `safe_moves`; endpoints are not made illegal by the harness, and a no-route emergency can choose an available edge.
2. **Jev judges the state after the action currently underway.** Candidate 0016 often asked Jev about the current node, then rejected the answer because Q*bert had moved by the time its roughly 180 ms response arrived; the trace/leaderboard report only 8.3% Jev-to-action freshness. Here, immediately after arming one hop, code computes legal-geometry and risk descriptions for the *expected landing node* and sends Jev those next-hop choices while the animation is running. At the observed landing it rechecks the node, board epoch, life epoch, target, age, current legality and freshly recomputed risk/cost. A model recommendation cannot authorize FIRE, an off-board edge, a deferred tip, or a materially more dangerous move.
3. **Astra is an asynchronous conditional reviewer, not the action clock.** After two inferred board clears, Astra receives a compact snapshot and may choose either `keep_conditional_deferral` or `relax_only_after_stall`. The default is the tested strict rule. A valid relaxation can only add a safe tip target after at least 90 ticks without interior score progress, and only if a currently safe move reaches a needed tip with no represented near lethal hazard. Arbitrary/prose output, timeout, or invalid JSON leaves strict deferral unchanged. At most one S2 request is made. Astra cannot choose joystick actions, routes, discs, or colors.
4. **The game clock is never blocked.** The motor pulses one legal direction, neutralizes during animation, verifies the observed landing, retries only briefly if no hop began, and uses bounded recovery after a loss. Jev/Astra calls run in background tasks; late or malformed replies are inert. The loop ends only on game/context completion, not on reported lives reaching zero.

## Jev interface and answer bridge

Each Jev request contains a strict JSON summary with:

```json
{
  "decision_stage": "These alternatives are for the expected landing cube; the current hop is already underway.",
  "position": {"observed_node": 14, "expected_landing": 20, "motion": "hopping"},
  "coverage": {
    "board_epoch": 3, "inferred_scoring_clears": 3,
    "unfinished_cubes": 8, "interior_unfinished": 6,
    "deferred_one_exit_tips": [20], "target": 9
  },
  "hazards": [{"kind": "coily", "visibility": "rendered",
                "proximity": "far", "motion": "pursuing"}],
  "choices": [{"action": "UP", "landing": "unfinished_cube",
                "lethal_risk": "no_supported_contact", "escape": "open",
                "route_to_target": "one_hop", "code_preferred": true}],
  "fallback": "UP"
}
```

Code supplies graph actions, target membership, risk buckets, and escape categories; Jev does not count, decode RAM, infer a route, or create an action. One request batches three independent questions:

- `route_a`: a `Choice` among up to four supplied action names plus `unknown`.
- `route_b`: the same semantic Choice with option order reversed; the bridge requires agreement to limit first-option bias.
- `fallback_danger`: a `Noul` asking whether the code fallback has *concrete* immediate danger; distant visibility or uncertainty is explicitly not a yes.

A choice is accepted only if both Choice answers agree on an offered action, both selected-option probabilities are at least **0.62**, and the action still exists in the latest recomputed option set. It must remain within **0.35** risk and **2.0** route-cost units of the deterministic fallback and pass a hard tip-deferral check. A Noul probability of at least **0.82** may veto the fallback only when another current option is at least **0.2** lower risk. Advice is attached to expected landing, board epoch, life epoch, target and request tick; it expires after **14 ticks** and is dropped on a position mismatch, death, or epoch change. The previous target is retained as provenance, not a veto: if reaching the expected landing completed it, current options and costs are recomputed for the new target before applying Jev. A choice is still subject to current legality, risk, cost, and endpoint gates. Missing probabilities and malformed answers produce code fallback.

The call is started with a **0.9 s** timeout. Archived Jev latency is about 0.18 s median and 0.4 s maximum in 0016, while an ordinary hop spans several ticks; requesting during the action therefore gives a plausible inference window without stalling the game. The executable fallback is always available. Only one Jev request can be in flight.

## Confirmation and refutation

The policy hypothesis is supported if training episodes repeatedly log `endcap_deferred` while interior work remains, show fewer actual landings on nodes 15/20 before interior completion, and preserve the 0016-like first clears while delaying or reducing life losses and ultimately reaching more scoring clears. The opening is an important negative control: because the tip deferral adds no global numeric route penalty, a large early score regression would count against the premise or the implementation.

A capped live smoke trial is already a caution, not confirmation: seed 0 ran to game over at tick **3,195**, scoring **19,050** with **five** inferred clears and life losses at **1777/2229/2562/3047**. It did exercise conditional deferral (19 `endcap_deferred` events); the first loss came later in absolute ticks than 0016 seed 0's tick **1603**, but the candidate also progressed more slowly and still lost all four lives. The 0016 official seed-0 episode scored **23,125** with six inferred clears. This is a single non-matched smoke/evaluation comparison and cannot estimate the training mean, but it is a concrete early warning that any survival benefit may not compensate for delayed coverage. The full evaluation must determine whether other seeds or longer play reverse that tradeoff.

It is refuted if strict deferral executes but life-loss timing per clear and total clear count do not improve, if avoiding tips causes repeated progress stalls or increases exposure on remaining interior routes, or if deaths continue in comparable numbers without tip entries. If the explicit relaxation policy is used, its stall trigger and resulting tip trips must be analyzed separately from strict episodes. A score gain without actual `endcap_deferred` events would not confirm the hypothesis.

The in-flight Jev design is independently confirmed if answers arrive before landing and make occasional safe, legal choices that survive the freshness and risk gates; it is refuted if model answers are still mostly late/stale or never alter an action. This is an interface test, not a claim that model-call volume improves survival. Astra's advisory value remains observational: a valid relaxation must be logged and applied only after the stated stall gate. Given the archive's weak evidence that S2 changes outcomes, strict code behavior is the fallback and the S2 call is capped at one.
