# Candidate 0023 — risk-dominance gate for threatened tips

**Mode:** refine. **Parent:** `candidates/0016/` (the strongest candidate). This keeps its corridor planner, motor handshake, and asynchronous advisers; only the endpoint admission condition changes.

## Hypothesis

After the third inferred board clear, the parent sometimes chooses a one-exit bottom tip while Coily is freshly visible and close to the takeoff cube. This candidate tests a narrow hypothesis: **defer that tip only when code finds a legal non-tip corridor whose computed immediate risk is within 0.35 units of the tip's risk and whose usable exit count is at least as large.** The tip's immediate cube progress is then postponed in favor of an option that is close on both of the parent's own risk/escape measures.

It is not a global corner penalty or a strict tip ban. Stale/absent enemies do not trigger it. If no single alternative is within **0.35 risk units** and preserves at least as many usable exits, the parent ranking is retained. The rule activates after three inferred clears only, preserving the opening as a negative control.

## Trace evidence and diagnosis

In the worst `0016` seed 5 trace, at tick **1584** the harness logged `node=14`, `action=RIGHT`, `target=20`, and rendered Coily with `proximity=near`. The hop landed at node **20** at tick **1593**, and `handshake_recovery` reported `lives=3` at tick **1603**. Later, tick **2099** showed Q*bert on node 20 choosing `LEFT` while rendered Coily was still `near`; it landed back at node 14 at **2108**, with a life loss recorded at **2109**. This does not prove that the tip caused either collision, but it shows a repeated endcap entry/escape sequence under the exact visible-threat condition the gate can inspect.

The common first loss at tick 1603 also appeared in representative 0016 seeds 0 and 2. However, 0016 and earlier candidates also lose in empty-hazard/state-36 windows; this candidate does not claim the visible-tip pattern explains all deaths. Lessons reject broad conservatism: 0020's strict endcap deferral scored 20,362.5 mean versus 0016's 21,808.3, and 0014's shelter rewrite scored lower still.

A two-minute smoke test of an initial, blunt version of this gate gave useful counter-evidence: at seed 5 tick **1584**, it deferred 14→20, selected 14→9, and lost a life at **1592**. Its score at the 120-second cap was **14,975**, with one life lost. That version removed the tip whenever any non-tip move existed, even if the substitute was worse. The final rule is narrower: it defers only when a non-tip option is within **0.35 risk units** of the tip and retains at least the same number of usable exits. If no such option exists, it makes no change. The smoke is not evidence of improvement; it motivates this dominance threshold and makes the hypothesis falsifiable.

A second two-minute smoke used the final risk-dominance gate. It again ended at **14,975** with one life lost, but at tick **1584** logged `action=RIGHT` and no `tip_entry_deferred`; the loss followed at **1603**, as in the parent. Thus the final condition did **not** alter the observed parent-like choice in this smoke window. It is intentionally not claimed as a positive result: the evaluation must show whether other qualifying situations occur, and whether those situations change survival.

## What changed

The board graph derives its one-exit tips from the fixed 21-node graph (nodes 15 and 20). The gate is activated when `inferred_clears >= 3`. A tip is marked threatened only if the Coily record is currently rendered, at most two ticks old, and within two graph edges of Q*bert's origin. For each such candidate, code checks all other currently legal corridors. It filters that tip only if at least one non-tip corridor has:

- computed risk no greater than the tip's risk plus **0.35**, and
- a computed usable-exit count greater than or equal to the tip's.

The event `tip_entry_deferred` records the source node, deferred tip nodes, and reason. The `deferred_tip` fact is carried in Jev's compact state. This code does not infer hazards from `purple_ball_state`, the scheduler, or absent sprites. The existing `safe_moves` and motor handshake remain the hard legality/execution controls. All other route costs, coverage order, target scheduling, and recovery logic are from 0016.

## Jev state schema and answer bridge

Jev receives a compact JSON corridor state, not raw RAM. Relevant fields are:

```json
{
  "position": {"node": 14, "standing": true,
                "safe_moves": ["LEFT", "RIGHT", "DOWN"]},
  "progress": {"epoch": 3, "inferred_clears": 3,
                "unfinished": 2, "target": 20, "posture": "spine"},
  "corridors": [
    {"key": "edge_DOWN", "action": "DOWN", "landing": "transit",
     "danger": "concrete", "escape": "closed", "endpoint": false,
     "tip_gate": "open", "code_preferred": true}
  ],
  "hazards": [{"kind": "coily", "visibility": "rendered",
                "proximity": "near", "motion": "pursuit"}],
  "deferred_tip": {"origin": 14, "tips": [20],
                   "reason": "rendered_coily_tip_dominated_by_non_tip_corridor"},
  "fallback": "edge_DOWN"
}
```

The `corridors` array contains only the candidates remaining after the gate. `deferred_tip` explains why a candidate may be absent. Code computes graph legality, risk, exits, target membership, freshness, and filtering; Jev is not asked to count, decode, or invent an action.

The parent's four batched questions remain unchanged:

1. `corridor_a`: Choice over supplied corridor keys plus `wait`.
2. `corridor_b`: same Choice with reversed option order to reduce first-option bias.
3. `proposed_safe`: Noul about concrete immediate lethal danger on the proposed corridor.
4. `keep_escape`: Noul about whether it has a meaningful legal exit.

Both Choices must agree, and each selected-option probability must be at least **0.60**. A Jev selection must still be present and legal, non-closed, within **0.35** risk and **2.0** cost units of the deterministic fallback; the safety and escape Noul thresholds remain **0.62** and **0.55**. Jev cannot reintroduce a filtered tip. Calls remain asynchronous, one in flight, no more than once per eight standing ticks, with **0.9 s** timeout. Advice is rejected when older than eight ticks, executed more than three ticks after request, or invalidated by node/life/epoch/policy/hazard-signature changes. Failure, malformed stand-in output, low confidence, or lateness uses the fresh deterministic fallback.

Astra remains unchanged: at most two background calls, returning only durable `posture` and `safety` values. It cannot select actions, authorize endpoints, or override the gate. The action loop continues until the game or context ends; `FIRE` is never offered.

## Confirmation and refutation

Support requires actual `tip_entry_deferred` events under rendered/fresh/near-Coily conditions, with the selected non-tip corridor meeting both numeric dominance checks. Compare those events to 0016 for:

- life-loss timing after the third clear and scored clear count;
- actual landings on nodes 15/20 while another certified alternative exists;
- progress pace, mismatch count, stale ticks, and early-board score.

The hypothesis is refuted if the gate is exercised but losses and clear counts do not improve, if candidate filtering still forces an equally or more dangerous route, or if score regresses without a survival gain. If no gate event occurs, the evaluation is inconclusive for this specific mechanism; a score change alone would not demonstrate that the endpoint rule caused it. Continued empty-hazard/state-36 deaths without deferred-tip events would confirm that this mechanism does not explain the remaining loss mode.
