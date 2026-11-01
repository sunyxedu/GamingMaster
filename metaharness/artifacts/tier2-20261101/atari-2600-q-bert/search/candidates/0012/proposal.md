# Candidate 0012 — exact phase-cycle scheduling

**Mode:** explore. **Parent:** `candidates/0006/`. **Comparison:** best observed controller `candidates/0007/`; the late-life evidence also includes `0011` as a negative refinement. This candidate does not modify any other candidate.

## Hypothesis

The remaining ceiling is not only enemy visibility. The controller that currently scores best has a **boolean destination tracker**: it learns one raw destination colour, marks every cube with that colour complete, and only after a 22-tick timeout starts another pass. That is a useful repair for the first palette inversion, but it is not the documented cartridge state machine. It loses the distinction between a cube that needs two landings and one that needs one, and it treats a later `D -> I` or `D -> S` transition as if it were an ordinary new-board colour.

My hypothesis is:

> Once the first four inferred boards have been crossed, exact board-local phase demand (`S/I/D`, including the reversible levels) is the missing structural constraint. Scheduling the next useful landing from the number of transitions still needed, rather than from a single learned destination and a timeout, will reduce false completion/oscillation, shorten exposure to hazards, and let the controller make real progress on later boards without sacrificing the robust first four clears.

This is deliberately different from the prior hazard-only refinements. The 0007 trace shows that the controller reaches board five with a raw `148 -> 214` landing at tick 1932, calls `214` its destination, then has to infer the next pass after treating many `214` cubes as complete. In the same run, the board is still in the public `level=1, round=1` decoder view and the tracker has no representation of the two-landings phase. Candidate 0006 had a two-stage heuristic (`goal` 1 or 2), but not a board-local phase cycle for all five schedules; 0007 reverted to a single goal plus `additional_pass`. No prior candidate tests an exact phase-demand planner through the inferred transition from board 4 to level 2 and the reversible cycles.

Late deaths remain important evidence but are not silently attributed to this change. In 0007 seed 0, the first post-fourth-clear life losses occur at ticks **2222**, **2322**, and **2422**; the last public observations before the latter two show no lethal hazard. In seed 4, the first late loss at **1987** likewise follows an observation with no lethal object. That means the phase experiment must preserve the parent’s guarded motor and conservative hazard fallback; it does not claim that phase bookkeeping alone predicts an unseen spawn.

## What changed

### 1. A cycle tracker replaces the single-goal tracker

`PhaseBook` keeps the raw start, intermediate, and destination colours for the current inferred board, recent scored transitions, and an exact integer `need[21]`:

* Level 1: `S -> D`; `S` needs 1 and `D` needs 0.
* Level 2: `S -> I -> D`; `S` needs 2, `I` needs 1, `D` needs 0.
* Level 3: `S -> D -> S`; `D` is protected as complete and any non-D state needs one useful landing.
* Level 4: `S -> I -> D -> I`; `S` needs 2, `I` needs 1, `D` needs 0.
* Level 5: `S -> I -> D -> S`; `S` needs 2, `I` needs 1, `D` needs 0.

The level is inferred only from observed whole-board award/flash epochs (`four boards per level`, with level 5 clamped/repeated); the public decoder counters are retained as evidence but do not control execution. A local positive 25-point component and a small changed-cube set teach transitions. Sam/Green events cannot teach forward phase roles. Global recolouring is quarantined until the board is stable and Q*bert is back at node 0, then a new board-local palette is learned. The stable flash frame is not trusted as the next board’s start colour: later boards anchor `S` from the first scored local transition, because the archived trace shows a post-flash frame at 148 followed by the actual first landing 26 -> 214. Once `S -> I` is observed, untouched S cubes already expose demand 2 and I cubes demand 1, even before I -> D identifies the destination.

This is not a hard-coded raw colour table: `148`, `26`, `214`, and every other byte remain observations. If a role is unknown, the tracker uses a conservative one-landing demand until a scored transition identifies it (with the partial S/I distinction above as soon as it is observed). It never writes RAM, alters score, or substitutes for the adaptor’s counters.

### 2. Short continuation search uses phase demand

For each legal immediate edge, code expands a small two-hop geometry beam. It does not queue those future moves. It ranks the first edge using immediate hazard facts, a conservative future branch envelope, exact landing demand (a two-landing `S` cube is useful but does not outrank an occupied edge), route distance, escape space, and recent repetition. The existing one-hop transaction remains the only executable macro: the directional edge is pulsed briefly, then released to `NOOP`, and the next observation revalidates the rendered anchor.

This continuation search is the structural change being tested: a nearest boolean target can no longer make the controller oscillate over an already-complete cube merely because the next palette role has not yet been discovered. The code still uses the public `safe_moves`, legal 21-node graph, enemy objects, hazard distances represented as conservative node/position facts, and a short threat shadow. It does not guess a disc exit or use the arcade 28-node map.

### 3. Astra and Jev remain off the action clock

Astra receives inferred cycle roles, demand counts, recent transitions, score/lives and threat summaries. It can select only `phase_first|nearest` ordering and `predictive|visible_only` hazard wording. Garbage or unsupported JSON leaves the current policy unchanged. At most three asynchronous reviews are launched, with a 100-second timeout; no response can authorize an edge or override legality.

Jev is asked about the *next supplied edge* only after code has generated it. The request is pipelined for the expected landing so its 1-second call can overlap the hop. Two reordered Choices reduce first-option bias; a Noul is speculative and can only veto a concrete immediate danger. The current observation, board cycle and code-generated alternatives are sent on every request because Jev has no inherited context.

## Tier-2 System-1 contract

The state sent to Jev is compact JSON with this shape:

```json
{
  "game": "Atari 2600 Q*bert, six-row 21-cube graph",
  "rules": "Code owns graph, arithmetic, phase demand, legality; survival first",
  "cycle": {
    "inferred_clears": 4,
    "level_by_cycle": 2,
    "cycle": ["S", "I", "D"],
    "roles": {"start_known": true, "intermediate_known": true,
              "destination_known": false},
    "demand_counts": {"zero": 5, "one": 2, "two": 14},
    "recent_transitions": [{"old": 148, "new": 214, "node": 7}]
  },
  "position": {"node": 13, "motion": "standing"},
  "alternatives": {
    "LEFT": {"landing": "two_landing_phase", "danger": "clear",
              "escape": "multiple", "phase_demand": 2,
              "continuation": "open", "preview": ["LEFT", "UP"]},
    "DOWN": {"landing": "transit", "danger": "intercept_possible",
             "escape": "narrow", "phase_demand": 0,
             "continuation": "risk_present", "preview": ["DOWN", "LEFT"]},
    "unknown": "No supported preference; retain code fallback."
  },
  "fallback": "LEFT",
  "hazards": [{"kind": "coily", "proximity": "farther"}],
  "timers": {"scheduler": "mid_cycle", "purple_state": "active_or_transition"}
}
```

The actual state contains only finite buckets and code-generated values; it does not send raw RAM or RGB. The `cycle` fields let Jev recognize whether an option advances a two-landing phase without asking it to count or infer palette arithmetic. `preview` is a short code-owned explanation of the continuation, not a command queue.

Every request batches:

* `choice_a`: a `Choice` over supplied `UP/RIGHT/LEFT/DOWN`, optional code-generated `NOOP`, and `unknown`;
* `choice_b`: the same Choice with reversed option order;
* `danger`: a `Noul` about immediate lethal danger on the code fallback, where distant visibility is explicitly not immediate.

The bridge accepts a movement recommendation only when both Choices agree, the selected key is in the current alternatives, and both selected-option probabilities are at least **0.60**. It additionally requires the selected candidate’s computed danger to be no worse than the fallback by more than **0.25**, and its total code cost to be within **2.5**. The danger Noul threshold is **0.84** and only permits a switch to a similarly safe supplied alternative. Missing fields, arbitrary stand-in values, disagreement, timeout, low probability, changed life/phase/policy/threat signature, or age greater than **8 ticks** fall back to deterministic code. Current `safe_moves` is checked again immediately before posting.

The fallback hierarchy is:

1. `NOOP` while flashing, falling, not visible, unanchored, or in the neutral release of a hop;
2. a bounded deliberate wait only when a rolling ball makes every candidate more dangerous than the current cube and Coily is not close;
3. the least-dangerous legal candidate whose exact phase demand and two-hop continuation make progress;
4. a short legal pulse, followed by neutral release and re-observation.

No life counter, model answer, or inferred clear ends the episode; only `game.done`/`ctx.done` does. Malformed actions are never posted, and FIRE is never used.

## Confirmation and refutation

The hypothesis is supported if, relative to 0007’s approximately 20,708 mean, the candidate:

* retains the first four inferred flash/3100 epochs with similarly low stale ticks;
* shows `phase_intermediate_learned` and `phase_destination_learned` on later boards, with no 22-tick boolean `additional_pass` stall;
* reduces repeated landings on already-D cubes and converts more of the board-five/board-six interval into scored landings or another stable flash; and
* reaches a later inferred board or survives longer before the same number of losses, without increasing model failures.

The cleanest trace evidence is a board with demand values 2→1→0 on the same cube and a subsequent `D->I` or `D->S` reversal that is routed correctly. The phase hypothesis is refuted if the learned roles fail to predict the next scored transition, if exact demand causes a first-board score regression without later progress, if it still falsely declares all phases complete, or if the same no-visible-hazard life-loss pattern dominates while phase traces are correct. The latter result would motivate a separate scheduler/hidden-spawn or disc-escape experiment; it is intentionally not folded into this candidate.
