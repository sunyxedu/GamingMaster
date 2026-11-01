# Candidate 0025 — spend one score-lane life, then probe the upper route

## Parent and targeted hypothesis

Parent: `candidates/0018/`.  I retain its observation-confirmed connector graph,
RAM-synchronized jump lease, sparse transition FIRE pulses, defensive model bridge,
and score-shell endpoint.

The parent’s worst episodes show a very specific policy failure.  In
`candidates/0018/trace_seed3.jsonl`, the controller logs
`score_fallback_enabled` with `phase=1` and reason `first nominal climb confirmed`,
then `ramp_advance` only from phase 0 to phase 1 at `x=105,y=154`.  It never
attempts the next connector.  Its final harness event is
`score_shell_end` at tick 3236 with score 4000, followed by `finished` with
`phase=0`, `jumps=41`, and `screens=0`.  The same trace family repeatedly shows
Mario around `x=98–99,y=148` while each hazard crossing earns another 100 points.
Seed 0 is the same failure at a higher local score: `finished` reports
`phase=0`, `jumps=42`, `score=4100`, and `screens=0`.  Thus the parent’s durable
score is a lower-ramp jump loop, not progress toward the goal.

The opposite experiment in `0023` demonstrates why simply deleting the fallback
is too risky: it reaches phase 0→1→2 on the first Mario, but the first upper
hazard jump at tick 157 consumes that life in several traces and its mean falls
to 1516.67.  The targeted hypothesis here is therefore a *life-budget* one:

> The first Mario can be used for the parent’s proven score lane, but the first
> life boundary should authorize exactly one nominal upper-route attempt on the
> next Mario.  If that route attempt also consumes a life, return to the
> parent’s tight score fallback rather than repeating the experiment.

This tests when to spend the route risk, not a new jump threshold, direction,
connector coordinate, or model policy.

## What changed

The parent’s architecture and action path are unchanged except for one
risk-scheduling state machine:

* The first active Mario still enters `farm_mode` after the observed bottom
  climb, exactly as in 0018.  This preserves the parent’s roughly 2000-point
  first-life score floor rather than paying the 0023 cost immediately.
* On the first real life boundary while no screen has been completed, the
  controller sets `farm_mode=False` and marks a one-time `route_probe_used`.
  The next Mario is then allowed to use the existing nominal connector graph,
  with all existing y-confirmation and jump-lease rules.
* The phase-0→1 climb only arms the score lane when the one-time probe has not
  been authorized.  This prevents the route probe from being immediately
  converted back into farming at its first climb.
* On a later life boundary, or after any already-used probe, the existing
  score fallback is restored.  No connector, hazard, rivet, lifecycle, or
  Astra/Jev rule is changed.

The result isolates one experiment: can the parent’s score-preserving opening
buy a later, intact route attempt without sacrificing the score floor?  A
successful route can now reach the rivet screen; a failed route is bounded to
one life and still leaves the final score lane available.

## Confirmation and falsification

Confirmation is a trace with the parent’s first-life phase-1 score lane,
followed by a life-boundary event such as `route_probe_enabled`, then
`ramp_advance` beyond phase 1 (ideally phase 2/3), a `stage=rivet` transition,
or positive `screens_completed`/`rivets_removed`.  A useful partial confirmation
is a second-life route attempt with fewer repeated phase-1 jumps and a score at
least near the parent’s floor, followed by `score_fallback_enabled` only after
the probe fails.

The hypothesis is refuted if the second active life immediately logs the
parent’s `score_fallback_enabled`/phase-1 loop, if it never gets beyond phase 1
while the route lease is active, or if the added probe lowers scores without
producing any additional vertical or screen progress.  A high score made only
of repeated jumps at `y≈148` is not route confirmation.  It is also refuted if
one failed probe leaves the controller retrying the same nominal route on every
remaining life.  Model completions or garbage Astra text are not evidence: the
same deterministic fallback must be safe with no usable model answer.

## Tier-2 System-1 state and bridge

I keep the parent’s event-triggered asynchronous Jev role.  The actor posts a
code-owned action before any inference can matter and maintains at most one
outstanding request.  The strict JSON state is a compact derived view:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
  "event": "connector_stall|hazard",
  "stage": "ramp|rivet",
  "player": {
    "horizontal_band": "left|middle|right",
    "vertical_band": "bottom|lower|middle|top",
    "motion": "up|down|steady",
    "jump_committed": false
  },
  "route": {
    "phase": 2,
    "selected_connector": 0,
    "connector_options": 2,
    "alternative_available": true,
    "climb_open": false,
    "fallback_action": "right",
    "rivet_row": 0,
    "rivet_rows": [0, 6, 12, 18],
    "farm_mode": false,
    "route_probe_used": true
  },
  "hazard": {
    "present": true,
    "slot": 3,
    "side": "right",
    "motion": "left",
    "distance_bucket": "near",
    "closing": true,
    "urgency": "commit"
  },
  "progress": {
    "lives_remaining": 1,
    "screens_completed": 0,
    "rivets_removed": 0
  },
  "last_action": "right"
}
```

Code computes coordinates, same-row/closing tests, semantic distance buckets,
row-byte decoding, connector legality, jump state, life-budget state, and
literal actions.  Jev is never asked to count, compare raw coordinates, invent
a ladder, or calculate a direction.

One request batches two independent typed judgments:

1. `route`: `Choice` over `hold`, `nominal`, and `alternate`.  `alternate` is
   meaningful only when the supplied route has a split connector and measured
   no-progress; `hold` is the safe label during an active climb or hazard.
2. `escape`: `Choice` over `keep_climbing`, `jump_left`, and `jump_right`.
   This is recognition of a supplied immediate same-row hazard, not a
   replacement for the hard code-owned collision guard.

A result is accepted only if its value is in the declared set, its selected
option probability is at least `0.60` when a probability map is present, and
its event signature still matches stage, phase, branch, rivet row, hazard slot,
hazard urgency/motion, and route-failure count.  It expires after three
control ticks.  Missing probabilities are tolerated only for conservative
`hold` or `keep_climbing`; all other missing, low-confidence, stale, timed-out,
failed, or malformed answers are discarded.  The deterministic fallback
continues immediately in every case.  The one-time route-probe state is never
changed by Jev.

Astra remains a nonblocking macro reviewer, spawned after repeated connector
deadlocks or rivet entry.  It receives the same compact derived state and is
asked for optional JSON labels `connector=nominal|alternate|adaptive`,
`hazard=climb|jump`, and `first_side=left|right`.  Plain, fenced, malformed,
or arbitrary stand-in text is ignored.  A valid label is applied only at a
later revalidated route boundary; Astra cannot spend or cancel the route probe,
interrupt a jump, suppress a restart, or post an action.  The actor continues
ticking while it thinks.

## Runtime safety

The loop uses `ctx.wait_for_tick()`, posts only names in the public action set,
starts idle with one FIRE edge, continues through a nonterminal zero-spare
transition, and stops only on context/game terminal state or the parent’s
exhausted-bonus transition shell.  All model/future parsing is defensive, and
background work is finite and checks the episode context through the normal
model calls.  No seed-specific trace, ROM data, or private emulator state is
used.
