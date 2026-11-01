# Candidate 0026 — observed phase-demand routing

**Mode:** rewrite. **Parent:** `candidates/0010/`; `0016` is the score/survival reference. This candidate changes the model/code division rather than copying either controller: code now plans from observed per-cube phase demand, Jev selects a durable cube goal (not a joystick edge), and Astra may alter only a bounded target tie-break preference.

## Hypothesis

A source of wasted late-board movement is treating cube progress as a binary “destination color reached” flag when the cartridge exposes `cube_phase` and `needed_landings` for every cube. The level cycles are not interchangeable: levels 2/4/5 require intermediate phases, and levels 3–5 can undo a destination landing. A target that was useful one landing ago may no longer be useful; a completed cube may be actively harmful as a transit square on a reversible level.

Candidate 0010 is evidence against expecting a learned failed-edge cost to solve this: seed 3 attributed losses to **15→10 at ticks 607 and 1237**, despite the route-style changes and growing penalty. The best reference, 0016, reaches five or six scored clears, but its phase book learns a goal color from local score and treats non-goal cubes as binary unfinished; it uses a 24-tick all-equal recovery to reopen a second pass. Its worst-run trace seed 5 lost a life at tick **1722** while landing **10→15** with a visible purple ball estimated at node 3 (`risk=distant`), then lost at **2148** on **14→20** with `hazards=[]` and `purple_ball_state=36`. Those observations do not support a reliable hidden-enemy location, and the losses are not claimed to be fixed by phase accounting. They do show that hazard-only micro-routing and edge postmortems have not removed the progression ceiling.

The specific, testable claim is narrower: **routing from the currently observed per-cube landing demand and finite phase cycle will reduce wasted/reversing landings and enable more complete boards on later multi-stage/reversible rounds, without sacrificing the parent's early-board coverage pace.** This does not claim that more accurate cube bookkeeping predicts invisible enemies.

## What changed

### Code: phase state, goal, and execution

* Read `needed_landings` directly, which is the adaptor's reported number of transitions to the next D phase; use the exposed `cube_phase` labels and correct level cycle as a fallback only when that counter is missing/invalid. This avoids turning a current `D` into “finished forever” on reversible levels 4/5. For a phase fallback, the landing demand to D is: Level 1 S=1/D=0; Level 2 S=2/I=1/D=0; Level 3 S=1/D=0; Level 4 S=2/I=1/D=2; Level 5 S=2/I=1/D=3. A Sam reversion or a later-cycle landing is reflected from the next observation rather than a permanent visited bit. Unknown entries retain prior demand instead of being silently treated as complete.
* Use the demonstrated 21-node triangular graph. A persistent target is selected from the stable coverage order and only from cubes that still need landings; a weighted shortest-path calculation makes unnecessary transit over `D` costly on reversible levels 3–5. The stable opening coverage order is only a target scheduler, not a route macro. All target and route calculations are recomputed from the current observation.
* Treat a global palette flash and a scored tally as board-clear evidence; `completed_rounds=0` is not used as a terminal/progress authority. Quarantine the transition briefly. After a scored clear, if the adaptor temporarily exposes the previous board's all-zero demand vector, keep all cubes eligible until a positive demand vector or a local scored color transition calibrates the new board. No action is ever taken from the reported lives counter alone.
* Retain a simple one-hop transaction: only current `safe_moves` may be posted, post one directional edge, then use NOOP through the hop and verify the observed destination. A bounded single retry is permitted only if no departure was observed. Visible lethal hazards add a soft local cost; no timer, absent sprite, or stale object is converted into a fictional enemy position.

### Jev: choose goals, not immediate edges

Each request contains a compact, strict JSON summary:

```json
{
  "game": "Atari 2600 Q*bert, six-row 21-cube board; never use FIRE",
  "board": {
    "level": 2, "epoch": 1,
    "cycle": "S>I>D", "unfinished_count": 16,
    "route_mode": "stable"
  },
  "position": {"node": 7, "standing": true},
  "target_options": [
    {"key": "cube_4", "location": "upper", "phase_work": "one landing",
     "route": "near", "destination_phase": "I", "code_rank": "best"}
  ],
  "constraint": "Choose a useful next cube; code rechecks demand, route and hazards before any hop."
}
```

Code computes legal targets, route cost, phase work, and descriptions. One batched request asks two identical `Choice` questions with reversed option order (`goal_a`, `goal_b`) over the top four generated cube goals plus `defer`. Jev does not count, infer the map, or invent an action. A goal is accepted only if the two choices agree, the selected key is code-generated, and both selected-option probabilities are at least **0.60**. The answer is a strategic preference, not an edge instruction: code adopts it only if that cube is still unfinished, the board/life epoch is unchanged, it is at most **12 ticks** old, and its recomputed target cost is within **1.0** of the current deterministic coverage goal. All hop legality and local hazard checks are rerun at execution.

Requests are asynchronous, one outstanding at a time, with a **1.0 s** timeout and a **12-tick** minimum interval. Timeout, API failure, malformed stand-in answers, disagreement, low confidence, stale epochs, or an invalid goal use the code target. There is no synchronous wait for Jev; while it is pending the motor continues its verified one-hop routine. `FIRE` is never a candidate.

### Astra: durable operating preference

Astra receives level/epoch, phase counts, lives/losses, and the current route mode. It can return only `route_mode=stable|center|upper|escape`. This changes a tie-break among near-equal unfinished goals, never graph legality, the chosen movement edge, a hazard veto, or a target more than one route-cost unit from the code best. The initial review is asynchronous; a second review is requested after the first observed life loss if the first call has finished. The prompt includes the Atari cycle rules and explicitly excludes the 28-node arcade board. Invalid or prose-only output is ignored; calls are capped at two with a 100-second timeout and never block a tick.

## Fallback and end conditions

The control priority is: palette flash/fall/recovery → NOOP; active hop → NOOP until observed landing or bounded recovery; otherwise choose a code-ranked unfinished target and shortest legal safe hop. If `safe_moves` is empty, stay neutral. Visible hazards only affect a soft ranking of current legal edges. No advice may add an edge absent from `safe_moves`; no FIRE is emitted. Continue until `game.done` or `ctx.done`, including after `lives` reaches zero.

## Confirmation and refutation

Support requires the evaluation traces to show that phase-demand planning is actually active (including intermediate phases where exposed), fewer completed-`D` re-landings on levels 3–5, fewer unproductive revisits / less time per scored clear, and more scored board clears or score than the 0016 reference without losing early clears. Review `phase_route`, `phase_board_epoch`, `hop_arrival`, and `jev_goal_adopted` together; a changed policy label or model answer alone is not evidence of benefit.

The hypothesis is refuted if valid observed demand/phase values do not change route choices, if phase-aware routing slows or regresses the opening before later cycles are reached, or if it produces no additional clears while life losses remain at the same progression point. If every training episode ends before reversible cycles, the later-cycle claim is inconclusive; a score change alone would not establish it. Similar empty-hazard losses would confirm that this phase planner is not a hidden-enemy solution.
