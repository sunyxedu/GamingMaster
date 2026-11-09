# Candidate 0011 — hazard-exposed corner gate

**Mode:** refine. **Parent:** `candidates/0007/`.

## Hypothesis

The parent has learned the palette epochs and can keep Q*bert moving for four to six inferred clears, but its escape-space feature is only a soft tie-breaker. In particular, it can commit to a bottom corner with one geometric exit while a lethal character is already close enough to reach that corner during the ten-tick hop. The targeted hypothesis is:

> Late deaths are partly caused by treating a one-exit landing as an ordinary useful/transit edge. A short, code-owned threat horizon that defers such a commitment when a turnable edge exists will preserve lives without changing the parent’s palette interpretation or hop timing.

This is intentionally narrower than a new enemy predictor. It does not guess disc exits, chase Coily, or alter the adaptor’s observations.

## Trace evidence

The parent’s worst seed-4 trace shows the exact commitment the change addresses:

* At tick **3694**, `coverage_status` is at node **14**, with `action="RIGHT"`, `lives=1`, `unfinished=16`, and `threats=[["coily",18]]`.
* The hop lands at node **20** at tick **3703**. In the public 21-node graph, node 20 has only the return edge to node 14.
* At tick **3714**, the parent is back at node 14 and again selects `RIGHT`, while its threat memory reports Coily at node **13**. It repeats the same corner commitment and loses the last life at tick **3734**.

The parent’s existing extra penalty only applies when Coily is within graph distance two *and* the estimated safe-exit count is zero. It therefore does not reject the one-exit node-20 commitment at the observed distances (three graph steps for the second commitment and four for the first), even though the hop itself consumes most of the useful reaction horizon. This is a concrete gap in the stated “preserve escape space” strategy, not an inference from the final score alone.

The seed-0 worst trace also records a limitation rather than evidence to overfit: at tick **2312** the parent selected `UP` from node 4 with `threats=[]`, and a life was lost at tick **2322**. The corner gate cannot solve a hazard that is not visible; that episode will help distinguish the proposed visible-threat improvement from a need for a separate spawn/visibility model.

## One targeted change

The parent’s deterministic controller, round-local palette tracker, neutralized one-hop motor, asynchronous Jev bridge, and Astra policy audit are retained. `Controller.candidates()` now additionally:

1. identifies a destination with exactly one geometric exit;
2. checks the parent’s recent lethal-threat memory against a fixed short horizon (three graph steps for either Coily or a rolling red/purple ball); and
3. marks the edge `corner_blocked` when that threat can reach the one-exit destination within the hop horizon.

When a turnable candidate is no more dangerous than the blocked corner (within a small code margin), `corner_blocked` edges are removed before fallback ranking. A visibly more dangerous turnable edge is not preferred merely because it has more exits. If every available edge is such a commitment, the controller inserts a bounded neutral wait (at most 24 ticks while current danger is below the existing immediate-danger bound), then permits the required corner as a last resort. Thus the change cannot strand the controller indefinitely or trade a clear landing for a known collision. The parent’s existing descending-ball wait remains unchanged.

The new arithmetic is local and deterministic; Jev is not asked to calculate graph distances. The code still recomputes legality and threat facts on the latest observation before posting an action.

## System-1 state and bridge

Each asynchronous Jev request retains the parent’s compact JSON state; the new gate is deliberately upstream of it. If a hazard-exposed one-exit edge is rejected, it is not offered to Jev. This prevents the targeted code experiment from changing Jev’s ordinary recognition distribution:

```json
{
  "game": "Atari 2600 Q*bert; 21 cubes; code owns graph and arithmetic",
  "policy": {"style": "nearest", "review": "collision_first"},
  "progress": {"board_epoch": 6, "decoder_labels_may_be_stale": true},
  "position": {"node": 14, "motion": "standing"},
  "candidates": [
    {"action": "LEFT", "landing": "unfinished", "danger": "clear",
     "escape_space": "multiple", "route": "one_hop_away", "code_preferred": true},
    {"action": "DOWN", "landing": "transit", "danger": "clear",
     "escape_space": "narrow", "route": "farther", "code_preferred": false}
  ],
  "proposed_action": "LEFT",
  "threats": [{"kind": "coily", "proximity": "farther"}]
}
```

Code supplies candidates, graph-derived escape counts, danger buckets, and the deterministic preferred edge. A blocked corner is handled by code before this state is formed; Jev is never asked to calculate graph distances, decide whether a guessed disc exit is valid, or invent an action. The parent’s state schema and question wording are otherwise unchanged.

One request batches three independent typed questions:

* `a`: `Choice` over the supplied edge names, `NOOP` when code supplied a bounded wait, and `unknown`. It chooses among already-filtered, legal alternatives.
* `b`: the same `Choice` with the option order reversed. The code uses agreement to reduce the provider’s first-option bias.
* `danger`: `Noul` asking whether the proposed candidate has immediate supported lethal danger. A distant threat alone is not enough for a veto; the code-owned corner gate has already encoded the hop-horizon commitment separately.

The bridge accepts a Choice only when both answers agree, name a currently supplied action, and each selected probability is at least **0.60**. A danger Noul at **0.82** can veto the fallback only under the parent’s existing no-worse-danger condition. Answers are tagged with node, life epoch, palette epoch, policy version, threat signature, and source tick; they expire after **8 ticks** or any mismatch. Current legality, candidate membership, and corner blocking are always rechecked. Invalid values, garbage stand-in answers, disagreement, low probability, timeout, or a stale response leave the code route in charge.

## Timing, Astra, and fallback

The request remains asynchronous, with a one-second Jev timeout and at least a five-tick launch interval. Every real-time tick posts a legal action or deliberate `NOOP`; no inference call blocks the emulator. The parent’s Astra adviser remains off the action clock, capped at three asynchronous policy-only calls with a 100-second timeout. Astra may select coverage style and Jev review wording, but cannot authorize an edge, suppress the single-exit gate, or emit code. The harness catches malformed JSON/text and model failures and continues.

The executable fallback is:

1. `NOOP` during palette flashing, falling, or unanchored motion;
2. the current one-hop transaction’s short launch pulse followed by neutral release;
3. a turnable, legal candidate ranked by deterministic progress and danger;
4. a bounded neutral wait when all legal candidates are hazard-exposed one-exit commitments; and
5. the least-bad legal commitment after the wait bound, so required bottom-cube coverage cannot deadlock.

No FIRE, guessed disc exit, arcade 28-node map, seed-specific route, or life-counter termination shortcut is used.

## Confirmation and refutation

The hypothesis is supported if 0011 retains the parent’s low stale fraction and early inferred clears while reducing deaths immediately following a visible Coily/ball near a one-exit destination—especially the seed-4 pattern above—and reaches later boards or a longer wall-clock survival time. A useful diagnostic is fewer `corner_commitment_deferred`-equivalent decisions followed by life losses, not merely more Jev calls.

It is refuted if the gate does not change the repeated node-14/node-20 pattern, if visible-threat deaths occur elsewhere at the same rate, or if waiting/route detours materially reduce first-board scoring without improving survival. If seed-0-like deaths with an empty hazard list remain dominant, that would motivate a separate future hypothesis about scheduler/visibility prediction; this candidate deliberately does not combine that change with the corner experiment.
