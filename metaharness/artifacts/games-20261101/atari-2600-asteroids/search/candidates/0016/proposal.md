# Candidate 0016 — semantic intent arbiter with code-owned microprograms

## Parent and hypothesis

Parent: `candidates/0006/`. The best comparison is 0011, which kept 0007's
frame-local clock ownership and low-duty firing while removing most thrust.

0006 proved two useful things: native-frame ownership and bounded FIRE/escape
execution are much more important than waiting for a model, and a route should
be recomputed from the current snapshot rather than held as a plain action. Its
failure is visible in its traces: seed 2 used 561 thrust/diagonal-thrust ticks
and scored only 980, while seed 4 used 1,079 and still lost ships at frames
1,108, 4,432, 5,176, and 6,364. Jev was asked to choose a whole route packet,
although its answers frequently disagreed (the early route choices include
`escape_left` versus `hold`). The local fallback then treated approximate
clearance as a reason to keep moving. This made the route selector both too
powerful (it could create thrust) and too weak (it could not understand target
value or field transitions).

The later 0011 traces provide a sharper counterexample. Its high-scoring runs
used almost no thrust (seed 4: only a few dozen thrust actions, 14,720 points),
but the controller still lost ships when a small/medium rock appeared close to
the stationary firing lane: seed 6 lost at frames 572, 1,528, and 3,000, and
seed 8 lost its first ship at 410 after `immediate` labels and an `UPLEFTFIRE`
followed by more turning. The code already knew candidate lanes, but Jev's
posture/target advice was mixed into the same tactical interface, and there was
no explicit distinction between a long-lived firing posture and a short
emergency relocation.

**Hypothesis:** the division of labour, rather than another collision threshold,
is the remaining bottleneck. Jev should recognize *what kind of field and shot
opportunity is present* (a semantic intent), while code should select and
revalidate every movement microprogram and every native action. Astra should
only choose a durable risk/mobility/fire posture from aggregate evidence. If
this separation is right, the controller will retain the parent’s central
FIRE score while adding displacement only when a current code forecast shows a
real margin, avoiding both 0006's thrust-heavy behavior and 0011's stationary
collision traps.

## What changed

* `Arbiter` computes wrap-aware approximate rock tracks, uncertainty penalties,
  swept clearances for `hold/left/right` short paths, target value/alignment,
  fire availability, life epochs, and the hard emergency/defense gates. It
  never asks a model to calculate a distance, velocity, intercept, or action.
* The native bridge is a receding microprogram. Normal play is an anchored
  aim-and-FIRE pulse with an explicit release gap. A relocation is a short
  turn/one-frame-thrust/coast program, rechecked every frame; a combined
  turn-thrust pulse is permitted only in a close window. No plain thrust or
  DOWN answer can remain latched while a model thinks.
* Jev no longer chooses among raw action routes. One batched request supplies
  two reversed-order field-regime judgments, two reversed-order target/sweep
  judgments, a side preference for a possible code-generated relocation, and a
  last-resort defense Noul. A returned label can affect only a short-lived
  target/regime preference or break a code tie. It cannot create movement,
  override path clearance, or invoke hyperspace.
* Astra is a slow policy reviewer, started asynchronously and allowed at most
  two reviews. It receives aggregate risk, route labels, score, lives, and the
  current policy. It may return only `priority`, `mobility`, `fire`, `defense`,
  and `aim` from allow-lists. It cannot send an action, target slot, route,
  timestamp, or emergency instruction. The default is survival, rare movement,
  normal pulse fire, and last-resort defense.
* A life transition invalidates model labels, targets, movement goals, and
  defense state. Unknown enemy craft/projectiles remain unknown, and a missing
  velocity is charged as uncertainty rather than treated as zero evidence of
  safety. All model failures, malformed stand-in output, stale answers, and
  timeouts fall back to the code-owned policy.

This is a structural rewrite of 0006: models describe the tactical *intent*
and long-term posture; deterministic code arbitrates safety and executes the
intent. It is not a change to one cooldown or one lane threshold.

## Evidence that would confirm or refute it

Support would be zero stale/error episodes with a higher mean than 0011, or at
least later first/last life losses at similar score rate. The useful trace
signatures are: substantially fewer thrust ticks than 0006/0008, movement
starts followed by rechecks rather than repeated route starts, more accepted
small/medium target choices, and fewer `immediate` windows that end with only
turning or a stale goal. A strong result would preserve long runs such as 0011
seed 4 while improving the seed-6/seed-8 early losses.

Refutation would be a return to 0006-like thrust counts and early game-over,
frequent side oscillation caused by model labels, or no score despite long
survival because target selection suppresses firing. It would also refute the
hypothesis if life-loss frames are unchanged while the new semantic labels
are accepted: that would indicate the code geometry, not the division of
labour, needs a different observation model. Official score remains unclipped
ALE reward; survival and life changes are diagnostics only.

## Jev state, questions, thresholds, and bridge

Every request is strict JSON with these parts:

```json
{
  "rules": {"mode": 0, "fire_requires_release": true,
             "noop_does_not_brake": true, "down_is_hyperspace": true,
             "model_selects_intent_not_action": true},
  "snapshot": {"frame": 1234, "lives": 3, "score": 5020,
                "player_present": true, "position": "middle",
                "motion": "moving", "danger": "soon",
                "impact_window": "soon", "rocks": {"count": 6,
                "small": 2, "unknown_motion": 1}, "missiles_visible": 1},
  "paths": [{"side": "hold", "clearance": "open", "contact": "distant"},
            {"side": "left", "clearance": "wide", "contact": "distant"},
            {"side": "right", "clearance": "tight", "contact": "soon"}],
  "targets": [{"id": "R12_small", "size": "small",
                "bearing": "right_above", "alignment": "near_axis",
                "distance": "middle", "motion": "known",
                "contact": "later"}],
  "controller": {"code_side": "left", "current_target": "auto",
                  "defense_used": false},
  "policy": {"priority": "survival", "mobility": "rare"}
}
```

The batched questions are:

1. `regime_a` and `regime_b` are reversed-order `Choice`s over
   `turret | fragmenting | escape | transition | uncertain`. They classify the
   field, not an action. They must agree and each selected probability must be
   at least 0.18; then the regime is valid for 54 frames.
2. `target_a` and `target_b` are reversed-order `Choice`s over the code-made
   target IDs plus `sweep`. They must agree, be present in the latest object
   list, and have minimum selected probability at least 0.16. The source slot
   is immediately revalidated with current size/geometry and discarded after a
   life epoch or stale snapshot.
3. `side` is a `Choice` over `hold | left | right`, with a 0.25 probability
   gate and a 42-frame expiry. It is only a tie breaker: code can ignore it if
   the side's swept clearance is worse, and code still decides whether a goal
   is started.
4. `defense` is a `Noul`. Its positive result is accepted only at probability
   0.80, and only if code independently sees overlap/immediate danger within
   two frames, both bounded paths have negative clearance, and defense has not
   been used in the life. Thus Jev cannot turn a late or stale answer into a
   hyperspace loop.

Answers older than 42 observed frames, from a different life epoch, or with
invalid/missing values are ignored. Disagreement leaves the previous short
intent untouched. Local safety is never model-gated: `overlap/immediate/soon`
selects the best current code path, and a stale/malformed Jev response simply
falls back to the anchored turret or bounded escape. During respawn or missing
player geometry the action is `NOOP`. Astra policy is also allow-listed and
cannot override these gates.

## Scheduling

The controller posts an initial code action before waiting for the first tick,
then calls `wait_for_tick()` and posts a fresh action on every observed native
frame. Jev has at most one outstanding request, starts at most every 24 frames,
and has a 2.4-second timeout; this is long enough for the measured Jev tail but
never blocks the clock. Astra runs in the background, with an initial review
and one later scheduled review. A late Astra result changes policy only after
parsing and never posts an action. `BudgetExceeded`, `EpisodeOver`, provider
timeouts, and arbitrary pre-flight model text are contained as normal fallback
conditions; `ModelUnavailable` ends safely through the contract path.
