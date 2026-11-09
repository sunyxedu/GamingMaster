# Candidate 0017 — adaptive edge postmortems with exact round demand

**Mode:** crossover. **Parents:** `candidates/0007/` and `candidates/0006/`.

## Hypothesis

The parents have solved the first structural problem: a round is visible as a
whole-board flash and a large score tally even though the adaptor's
`completed_rounds`/`green_ball_points` fields do not recognize it.  They also
have the right real-time motor shape: graph-legal one-hop pulses, neutral input
during animation, and model calls off the action clock.

The remaining failure is more specific than “the sprite decoder is stale.” In
all six `0007` episodes, the life-loss observations have
`timers.purple_ball_state=36` and `hazards=[]`; the controller therefore sees
all or most `safe_moves` as clear.  Yet the losses are not arbitrary.  Across
`0007`, the same planned edges recur immediately before deaths: `13 LEFT`
caused four losses, `1 DOWN` caused four, and `6 RIGHT` caused two.  For
example, seeds 1, 4, and 5 all lose their first post-fourth-board life at tick
2019 after the route commits `LEFT` from node 13 with no rendered hazard.  The
parent's `update()` clears its pulse and recent route state on a life loss, so it
throws away exactly the evidence needed to take a different branch after the
respawn.  Candidate 0013's timer-only directional nudge did not repair this:
its traces still lose after `13 DOWN`, `10 RIGHT`, and `13 DOWN` while the same
state 36 is active.  Candidate 0014's broad shelter/edge rewrite regressed the
opening, so a generic high-timer quarantine is not a supported solution.

**Hypothesis:** an unrendered enemy is best handled as a *closed-loop action
failure*, not as a globally interpreted timer.  Recording the actually
executed source→destination edge and the public scheduler signature at the
moment of a real life loss should let the controller select an alternate route
on the next life.  A short recovery posture should protect the respawn without
turning the agent into the previously refuted long shelter wait.  Combining
this with 0006's exact finite colour-cycle demand should also prevent the
boolean “one landing is enough” trap as boards advance beyond the four Level-1
rounds.

This is not a hard-coded seed route: the ledger is populated only by an
observed life decrease in the current episode, and its entries expire.  A
failed edge is penalized in both directions for a bounded period; a repeated
failure or matching scheduler signature strengthens that local penalty.  If
all legal exits are marked, the least-bad legal edge remains available.

## What changed

1. **Adaptive edge postmortem (new structural mechanism).** The motor keeps a
   transaction record until the edge is confirmed or an actual life decrease
   is observed.  Before clearing the transaction it records the source,
   destination, failure count, board epoch, and the public Purple scheduler
   value as a signature.  The route planner adds a large but finite cost to
   that edge and its reverse, a smaller same-epoch cost, and an additional
   cost when the current public signature matches.  The record expires after
   260 ticks.  A 42-tick central/recovery bias is applied after a loss.  This
   is deliberately postmortem-only: it cannot perturb the reliable opening
   before there is evidence in the current run.

2. **Exact round-local demand.** The 0007 palette-epoch detector is retained,
   but its boolean `color != goal` ledger is replaced by 0006-style role
   transitions.  Raw colours are mapped locally to S/I/D from rewarded local
   transitions.  The code applies the documented transition table for the
   inferred level, including D→I/D→S reversibility, so the planner targets a
   required landing rather than treating an intermediate landing as completion.
   Whole-board flashing remains quarantined and all clear counts remain
   explicitly *inferred*; no observation or evaluator counter is changed.

3. **Closed-loop lookahead and motor.** Candidate costs combine nearest useful
   demand, current named-hazard risk, escape-space, recent route visits, and the
   postmortem.  A one-hop transaction is committed only from a current
   `safe_moves` edge, released to NOOP during flight, and revalidated at the
   observed landing.  A Jev answer is pipelined for the expected next node and
   is accepted only if node, life epoch, palette epoch, policy version, hazard
   signature, failed-edge signature, freshness and probability checks still
   match.  Jev can break a close tie or veto toward an equally safe supplied
   option; it cannot invent a direction, erase postmortem evidence, or use
   FIRE.

4. **Astra remains off-clock.** Astra writes only a small durable tie policy
   (`nearest|frontier|center` and `evidence_first|conservative`).  It receives
   the score, inferred progress, recent palette evidence and postmortem edges,
   and may be requested initially, after an epoch, after a life loss, or after a
   long score stall.  Malformed/late output leaves code's policy intact.

## Evidence and falsification

The hypothesis is supported if the candidate preserves the parent’s first four
inferred clears, then shows fewer repeated source→destination edges after the
first loss, fewer subsequent losses in the same signature, and more fifth/later
scored flashes.  The important causal trace events are `edge_postmortem`,
`edge_commit`, `edge_landed`, and the later `failed_edges` list—not Jev call
counts.  Exact-demand success should appear as useful transitions on the
second and later level cycles without a false zero-point handoff.

It is refuted if the opening loses the parent’s early clears, if the same edge
still kills repeatedly despite fresh postmortem penalties, if recovery waits
consume lives or create an apex/central orbit, or if life losses occur on
unrelated edges with no improvement in survival.  It is also refuted if the
role ledger repeatedly declares all cubes complete before a scored flash or
fails to route the Level-2 intermediate colour.  In that case the remaining
problem is likely a ROM-specific hazard trajectory or a validated disc/Coily
escape policy, not more edge memory.

## Jev contract and tier-2 bridge

Every request contains a strict-JSON derived state, never raw RAM or RGB:

```json
{
  "position": {"node": 13, "motion": "standing"},
  "board": {"inferred_epoch": 4, "inferred_clears": 4,
            "unfinished_cubes": 8, "level_cycle": 2},
  "recovery": {"active": true, "loss_count": 1},
  "hazards": [["coily", 9, "fresh"]],
  "recent_failed_edges": [[13, 8, 1, "same_phase"]],
  "candidates": [
    {"option":"DOWN", "role":"useful_landing", "landing":"clear",
     "escape_space":"many", "route":"adjacent_target",
     "postmortem":"no_loss_evidence", "code_rank":0},
    {"option":"LEFT", "role":"transit", "landing":"clear",
     "escape_space":"one", "route":"farther",
     "postmortem":"avoid_after_recent_loss", "code_rank":1}
  ],
  "fallback": "DOWN", "projected_decision": true
}
```

Code computes graph legality, distances, role demand, hazard distance,
postmortem age/penalty, and escape counts. Jev answers three independent
questions in one request:

* `route_a`: a `Choice` over the supplied edges, bounded `NOOP` when code
  supplied it, and `unknown`; prefer a clear, roomier, useful option and treat
  a recent postmortem as evidence against an option.
* `route_b`: the same `Choice` with criteria order reversed, reducing the
  documented first-option bias.
* `immediate`: a `Noul` asking whether the fallback has concrete immediate
  lethal risk. A timer-only or distant fact is explicitly not enough.

Both Choices must agree and have selected-option probability at least `0.60`.
The chosen candidate must be within 3.0 computed cost units of code's fallback,
within 0.45 danger, and may not discard a postmortem-safe fallback for a
postmortem-marked edge. A Noul probability at least `0.84` may veto only toward
an equally safe, non-postmortem candidate. Missing probabilities, disagreement,
`unknown`, malformed stand-in answers, timeout, or any exception retain the
fresh deterministic fallback.

Requests launch at most every six ticks and at most one is outstanding. The
request is normally made during a committed hop for the *projected destination*
(the 0006 parent mechanism), then consumed only if that exact node is observed
standing within nine ticks. Metadata also binds life/board/policy versions,
hazard signature and failed-edge signature. A late Jev answer never blocks a
tick. Each loop posts a legal action or deliberate NOOP and calls
`ctx.wait_for_tick()`.

Astra's policy prompt includes the same division of labour and restricts its
output to the two enumerated tie policies. It is asynchronous, capped at
three calls with a 105-second timeout, and cannot alter the postmortem gate,
colour bookkeeping, graph, or action. The harness catches every ordinary model
failure and is designed to survive arbitrary pre-flight model text and first
option answers.

## Why this is a crossover rather than a parameter tweak

0007 contributes the high-scoring observation-calibrated palette epochs,
short guarded motor, and off-clock adviser bridge. 0006 contributes
round-local finite-cycle bookkeeping and projected-next-node Jev decisions.
The new core mechanism is neither parent's: **the controller learns from an
actual life loss at the exact edge transaction and changes its future route
without interpreting an invisible timer as a location**.  The evaluation
should distinguish it by the postmortem events and by later-life edge
diversity, even if model freshness remains near zero stale ticks.
