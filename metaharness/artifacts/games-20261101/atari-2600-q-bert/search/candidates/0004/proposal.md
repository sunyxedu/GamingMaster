# Candidate 0004 — guarded graph execution with asynchronous advisers

**Parent:** `candidates/0001-tier2_astra_only/` (with `0000-tier2_jev_only/` as the
strongest comparison).

## Hypothesis

The low scores are primarily an execution-structure failure, not a lack of
Q*bert rules. The parent proves that Astra cannot be the clock: in seed 1 it
made one 66.86-second call while all 1,000 ticks were stale and the score stayed
0. The Jev-only trace is faster but still sends an edge as the current action
while its answer is in flight. In seed 4, Jev repeatedly selected `DOWN` at the
opening (`DOWN` at ticks 3, 7, 10, 12, ...), reached the bottom-left node 15 for
125 points, and then four lives were lost without another round. Its end state
has `qbert_node: 15`, `falling: true`, `lives: 0`, and `life_losses: 4`; 216 of 334
ticks were stale. Thus a reasonable local choice becomes lethal when the edge
is held for several real-time ticks and then repeated at a boundary.

I expect a code-owned, one-hop-at-a-time controller to beat both baselines:
only code can enforce the 21-node graph and stop posting a direction once a hop
has begun. Jev should recognize local danger and choose among already-generated
alternatives, while Astra should plan only an unfinished-cube order and never
be on the action-critical path.

## What changed

* The harness builds the six-row, 21-node graph from the documented Atari action
  semantics. It computes BFS routes, tracks `cube_phase`/`needed_landings`,
  applies the level-specific reversible-cycle caution, prioritizes the upper
  opening, and resynchronizes target state after life and round changes.
* A directional action is posted for one decision tick, with at most one
  standing-tick retry to survive a sticky-action realization that replaced it
  with NOOP. Every subsequent hop tick posts `NOOP`, and the first standing
  tick after a landing is also neutralized. This prevents a 0.15–0.20 s Jev
  call or a stale edge from becoming a second boundary hop. `safe_moves` is
  used before `legal_moves`; the code rejects FIRE and any answer outside the
  current candidate set.
* Jev is asynchronous and local. At a standing cube one request batches two
  independently reordered `Choice`s for the immediate edge, a `Noul` lethal-risk
  veto for the proposed edge, and two reordered `Choice`s for the next target.
  The controller uses a Jev edge only when the two choices agree and the answer
  is fresh; otherwise it follows its deterministic BFS edge. A late answer is
  tagged with epoch/tick/node/life/round and discarded.
* Astra is also asynchronous. The initial request and a bounded number of
  life-loss/round-transition revisions ask for a JSON target order, not an
  action. Invalid, arbitrary, or late text is ignored. A deterministic route
  remains executable while Astra reasons, and an accepted plan cannot interrupt
  a hop.

## System-1 contract and safety policy

The state sent to Jev is a strict, compact JSON object:

```json
{
  "position": {
    "node": 0,
    "standing": true,
    "legal_moves": ["DOWN", "RIGHT"],
    "safe_moves": ["DOWN", "RIGHT"],
    "proposed_edge": "DOWN"
  },
  "progress": {
    "level": 1, "round": 1, "lives": 4,
    "unfinished_nodes": [0, 1, 2],
    "phases": ["S", "S", "D"]
  },
  "route": {
    "target": 1,
    "next_edges": ["DOWN"],
    "candidate_targets": [1, 2, 3]
  },
  "hazards": [{"kind": "purple_ball", "node": 3,
                "risk": "near", "distance": "adjacent", "motion": "rolling"}],
  "recent": {"last_action": "DOWN", "last_event": "cube", "life_losses": 0}
}
```

The actual implementation also supplies the small game/progress fields and
candidate descriptions. It excludes raw RAM and sprite placeholders. The
questions are:

1. `move_a` and `move_b`: reordered `Choice`s over the code-supplied legal
   edges plus `NOOP`. They recognize which already-legal edge is least risky;
   the code accepts only agreement, current legality, and a top probability of
   at least 0.42.
2. `danger`: `Noul` asking whether the proposed edge has an immediate lethal
   edge-fall or enemy-contact risk. A probability at least 0.64 vetoes that
   proposed edge; code then chooses another safe route edge or waits.
3. `target_a` and `target_b`: reordered `Choice`s over at most ten unfinished
   nodes, with descriptions containing the code-computed route. Agreement and
   probability at least 0.50 may replace the current target. Jev never invents
   a node or calculates cube arithmetic.

The two orderings address the provider-documented first-option bias. All model
answers are advisory: a candidate that is not in `safe_moves`/`legal_moves` is
ignored, FIRE is never offered, and an answer from a different node, epoch,
life count, round, or more than a few ticks old is discarded. During a Jev call
the fallback is `NOOP` while still standing; after six ticks code takes one
BFS-legal edge. During an animation/fall the fallback is always `NOOP`. This is
intentionally conservative because a missed 25-point landing costs less than a
life and the 3,100-point round bonus.

## Astra interface and revision trigger

Astra receives a symbolic snapshot with level/round/lives, all phases, the
unfinished node list, current node, recent event, and semantic hazard summaries.
It is asked for only `{"targets": [node, ...], "risk": "..."}`. The plan is
accepted only after strict JSON extraction and integer/range validation; every
other answer is harmless. The initial call runs in the background. A further
call is started when a life is lost or the level/round changes (up to six total
calls), using the prior session when available. A plan never overrides the
current hop or bypasses local code checks. If Astra is absent, times out, or
returns prose, the graph route and Jev/local fallback continue unchanged.

This division directly tests the hypothesis: if stale held edges caused the
baseline deaths, the new trace should show short directional posts followed by
NOOP hop guards, fewer boundary falls, more than five completed cubes without
life loss, and a lower stale-tick fraction even when Astra has not answered.
The hypothesis is refuted if the controller still dies at a legal boundary
while `NOOP` is posted during every hop, or if score remains near the idle/low
baseline despite surviving substantially longer; that would point to a wrong
ROM action-to-edge mapping or a route/colour-cycle error, triggering Astra
revision of the target schema rather than looser action thresholds.
