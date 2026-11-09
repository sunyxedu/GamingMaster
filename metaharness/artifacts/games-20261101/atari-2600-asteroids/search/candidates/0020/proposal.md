# Candidate 0020 — phase-shifted turret with flicker ghosts

## Parent and hypothesis

Parent: `candidates/0017/` (with `0011` as the comparison controller).

The strongest result is still the low-duty turret architecture, not model-driven
control.  Candidate 0017 reached the 18,000-frame budget on seeds 0 and 4 with
17,220 and 16,540 points, and its frame ownership was excellent (one stale tick
in the whole evaluation).  Its weak result is concrete rather than a lack of
Jev reasoning: seed 10 scored only 3,100 and lost lives at frames 1,566, 3,018,
and 3,338.  In the first loss window the trace shows the ship stationary at
approximately `(121,128)` while the local controller progresses from
`ttc=41` (frame 1516), to `25` (1531), to `5` (1551), and then invokes DOWN at
1554.  It does not issue an escape pulse in that window; the life epoch arrives
at 1566.  The object list is also visibly unstable: at frame 1524 several
objects are present, some have null velocity, and at frames 1553–1555 a small
or split object appears/disappears while the ship remains in place.  Thus the
late defense is not evidence that a second model answer was needed; the ship
was allowed to remain correlated with one screen corridor until the approximate
box estimate became too late.

The same trace repeats the pattern at frames 2972–2999 and 3292–3321: the
controller mostly emits FIRE/NOOP, then spends DOWN at contact 0/1 and loses a
life 19 or 17 frames later.  0017's own radar log has only 11 NOOP/one-frame
turn controls and no `escape_pulse` in this worst episode.  Conversely, the
successful seed-0 trace used only six escape pulses and a mostly stationary
turret.  This suggests that thrust-heavy roaming is not the answer, but a
completely static target is also not robust to flicker, splits, and newly
occupied vertical tracks.

**Hypothesis:** a low-duty, feedback-controlled horizontal phase shift will
preserve the turret's score while breaking the stationary collision pattern.
The controller should move between broad central waypoints even before a
forecast says `immediate`, but only with one-frame horizontal thrust pulses and
fresh position feedback.  It should retain short-lived asteroid tracks across
one to three missing frames, so a flickering object is a temporary ghost rather
than a false clear field.  This is structurally different from the prior
controllers' five-lane/urgent-escape policy: the phase patrol is the normal
posture, its waypoints are generated from the current field, and its pulse
budget is intentionally small.  Movement is stopped or redirected when the
new risk map is worse; it is never a held model action.

## What changed

* The controller builds a wrap-aware **vertical-crossing risk map** for a
  continuous set of central x candidates.  It scores where each observed rock
  would cross the ship's y band, rather than making the decision only from a
  current nearest distance or a fixed lane label.
* A short-lived slot/size continuity tracker infers a velocity when the
  adaptor's velocity is null and retains a maximum three-frame ghost when an
  otherwise continuous slot flickers.  A size change, large jump, life change,
  or player discontinuity clears the track.  Source slots remain temporary
  observations, never physical identities.
* A normal patrol alternates left and right central phase waypoints (roughly
  x=58 and x=102, subject to the live risk map).  Each leg permits at most six
  one-frame horizontal thrust pulses with a 14-frame spacing.  Emergency legs
  may use at most eight pulses with a six-frame spacing.  The ship first turns
  toward a horizontal heading, then pulses and observes the result; measured
  drift near a waypoint can cause one counter-thrust pulse.  The patrol has no
  seed-indexed schedule and no open-loop thrust timer.
* The turret keeps a real FIRE release cadence, favors small and readable
  targets, and uses a bounded target aim or alternating scan between movement
  pulses.  Collision response outranks aiming.  One hyperspace pulse per life
  remains a last resort only at a computed contact of at most two frames when a
  candidate waypoint is not materially safe; Jev cannot create a defense.
* Astra receives aggregate evidence only and can select among allow-listed
  patrol/aim/fire/risk/defense postures.  Jev classifies the current semantic
  regime and a side tie-break, but code owns all arithmetic and actions.  Both
  answers are freshness-checked and harmless when late, malformed, or absent.

This tests the untested middle ground between 0017's static risk firewall and
0008's thrust-heavy proactive movement.  It also tests whether explicit flicker
memory improves the false-`clear` problem without importing an unverified wrap
period or game mechanic.

## Evidence that would confirm or refute it

Confirmation would be a higher mean unclipped ALE reward with no harness
errors, especially if the seed-10-style losses occur later or disappear, while
`phase_pulse` counts remain small and most frames remain FIRE/turn/coast.  Useful
trace evidence is a pulse followed by measured x displacement and a later or
absent life epoch, fewer late DOWN actions, and ghost/transition guards that
correspond to later observations rather than permanent paralysis.  A run that
reaches 18,000 frames while retaining the turret score would support the
hypothesis more strongly than a short high-score burst.

Refutation would be earlier losses caused by accumulated horizontal inertia, a
large score drop despite similar survival, repeated patrol reversals without a
measured x change, or ghost tracks causing the controller to stop firing for
long periods.  If the phase waypoints are repeatedly predicted safe but still
produce contact, that would indicate that the RAM geometry is too incomplete
for this map and the phase intervention should be abandoned.  Selection remains
based only on original unclipped ALE reward; survival, lives and life-loss
frames are diagnostics.

## Jev state, questions, and bridge

Every request is one strict JSON state derived from the latest snapshot.  Its
schema is:

```json
{
  "rules": {
    "mode": 0,
    "difficulty": 0,
    "fire": "press_then_release",
    "noop": "coasts_not_brakes",
    "down": "mode_0_hyperspace",
    "heading": "left_increments_right_decrements",
    "enemy_coverage": "unknown_not_absent",
    "model_role": "semantic_posture_only"
  },
  "snapshot": {
    "frame": 1234,
    "lives": 3,
    "score": 530,
    "player": "present",
    "position": "middle",
    "motion": "still",
    "danger": "watch",
    "impact_window": "later",
    "rocks": 7,
    "unknown_or_ghost": 2,
    "missiles": 1,
    "transition_guard": 0
  },
  "patrol": {
    "goal": "left",
    "goal_distance": "middle",
    "waypoints": [
      {"id": "X58", "relation": "left", "risk": "safe",
       "impact": "distant", "clearance": "open"}
    ],
    "pulses_this_leg": 2,
    "phase": 1
  },
  "target": {"size": "small", "bearing": "right_above",
             "motion": "known", "distance": "middle"},
  "controller": {"mode": "patrol", "aim": "target",
                  "defenses_this_life": 0, "frame_local": true},
  "plan": {"patrol": "steady", "aim": "target", "fire": "normal",
           "risk": "cautious", "defense": "rare"}
}
```

The batched request contains five independent judgments:

1. `regime_a` and `regime_b` are reversed-order Choices over
   `turret | patrol | escape | recover | uncertain`.  They are accepted only
   when they agree and the selected probability is at least 0.18.  The result
   changes a 60-frame semantic mode lease; it cannot directly choose thrust.
2. `aim` is a Choice over `target | sweep | hold`, accepted at probability
   0.18 for a 60-frame lease.  The local target calculation and FIRE timer
   still revalidate it every frame.
3. `side` is a Choice over `hold | left | right`, accepted at probability
   0.22.  It is only a tie-break if the next code-generated risk map is being
   replanned; it cannot overturn an unsafe waypoint.
4. `defense` is a Noul.  A true probability of at least 0.84 sets a temporary
   hint, but code additionally requires a present player, at most two frames to
   the local contact, no materially safer waypoint, and one unused defense in
   the current life.

A response is discarded if it is from another life epoch, is future-dated, or
is more than 48 observed frames old.  Disagreement, low probability, timeout,
provider errors, arbitrary stand-in values, and missing fields leave the local
policy unchanged.  While Jev is pending, the controller continues its
frame-local action and `wait_for_tick()` loop; no posted action waits for a
model.

## Astra scheduling and fallback

The first Astra review is launched after frame 300, once the launch family has
produced score/risk evidence.  A second review is allowed after a life
transition or a later scheduled interval.  Its prompt includes frame, score,
lives, contact bucket, current risk/clearance, number of observed and ghost
rocks, goal, phase-pulse count, defense count, and the current policy.  It must
return a compact JSON object with only:

* `patrol`: `steady`, `rare`, or `responsive`;
* `aim`: `target` or `sweep`;
* `fire`: `normal` or `spaced`;
* `risk`: `cautious` or `balanced`;
* `defense`: `rare` or `never`.

Malformed or prose output is ignored.  Astra cannot post an action, change the
pulse limits, bypass a discontinuity quarantine, or spend DOWN.

The deterministic fallback is executable without either model: if the player
is absent, post `NOOP`; after a discontinuity or life change, coast/fire for a
short recovery interval while rebuilding tracks; otherwise select a live
central waypoint, turn toward horizontal, and issue only a budgeted one-frame
pulse.  When no pulse is justified, aim from the current observation and emit
only a release-separated FIRE/turn/NOOP action.  Immediate danger uses the
fresh risk map for an urgent bounded pulse and reserves DOWN for the explicit
last-resort gate.  Thus arbitrary model output during pre-flight cannot crash
the harness or turn into a stale held thrust.
