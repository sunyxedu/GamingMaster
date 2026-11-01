# Candidate 0018 — kinematic anchor and threat-first occupancy control

## Parents and diagnosis

Parent: `candidates/0014/` (rewrite).  The main reference is the current best,
`candidates/0017/`, with `0011` as the useful scoring reference.

The confirmed part of the archive is that the emulator clock must remain local:
0017 reached 17,220 and 16,540 on seeds 0 and 4 with essentially no stale ticks,
while the remaining failures were deaths, not model latency.  I therefore keep
one-frame ticking, short FIRE presses, and asynchronous advisory models.

The concrete failure I am targeting is momentum being mistaken for a stationary
posture.  In 0017 seed 10 the launch observation at frame 100 has player velocity
`[0.0, 0.5]`.  The controller then uses no thrust after the prelude; the ship is
observed around x=93 at frame 101, x=111 at frame 274, and x=121 at frame 1484.
At frames 1545–1553 the local radar says `ttc` 11 then 5, but the actions are
mostly `FIRE`/`NOOP`; the only `DOWN` is frame 1554 with `ttc` 0, and the life
change is frame 1566.  The Jev request judged frame 1525 and chose `posture=coast`,
which kept the aiming loop from turning while the threat approached.  The same
trace repeats the pattern at frames 2999, 3321, and 3793: defense follows a
contact that is already at `ttc` 0–1.  Thus the issue is not simply that the
radar needs another label.  It needs an explicit momentum-damping phase and an
offensive loop that a semantic “coast” answer cannot freeze.

The parent 0014 also shows why a fixed lane reservation is insufficient: it
starts many reservations but records no completed/cancelled trajectory and its
trace never exposes a meaningful numeric TTC.  Conversely, 0017's idle-derived
fire posture is valuable when the ship has actually settled.  The hypothesis is
that a short, observed-velocity anchor phase followed by a full two-dimensional
threat map will preserve that turret advantage without 0008's indiscriminate
thrust.

## Hypothesis and structural change

A ship that inherits velocity from the disclosed UPRIGHT launch prelude should
first be treated as a moving body.  Code estimates velocity from the adaptor's
finite differences plus consecutive decoded positions, rotates toward the
opposite velocity, and uses isolated thrust pulses to damp it.  Once speed is
small, the controller returns to target firing.  This is not a brake shortcut:
NOOP remains inertia-preserving, and every damping pulse is a real UP action.

For threats, code now builds an occupancy forecast in the relative x/y plane for
coast, current-nose thrust, anti-velocity braking, and eight calibrated compass
headings.  It scores each short path with wrap-aware swept boxes, uncertainty
inflation, earliest contact and clearance.  A selected escape is a two- or
three-pulse *tranche*: rotate toward the selected code, thrust at most once every
few frames, and revalidate it on every new observation.  It is cancelled on a
life/discontinuity, a cleared state, or a forecast that is worse than coasting.
The new danger buckets treat a contact within 18 frames as caution rather than
waiting for 0017's 5-frame immediate window.  Hyperspace remains one bounded
per-life last resort, but the local gate may issue it by TTC 5 when every
reachable short path still intersects; it no longer waits for TTC zero.

The model division is deliberately different from the parent's posture/lane
bridge:

* **Code** owns kinematics, target threat ranking, occupancy arithmetic, heading
  conversion, path reachability, momentum damping, FIRE release, defense budget,
  life epochs, discontinuity quarantine, and every posted action.  A Jev answer
  can never veto aiming during caution/critical danger or create a thrust/down
  action.
* **Jev** receives derived semantic buckets and chooses between `anchor`,
  `evade`, `hunt`, and `hold`, plus a revalidated observed target.  Two reversed
  choices must agree.  `evade` can lower the local caution threshold modestly;
  `anchor`/`hold` are only hints.  Code still performs the exact safety test and
  ignores a stale target id.  Its defense Noul is only a supporting signal for
  the local TTC/path gate.
* **Astra** receives one aggregate post-launch review (score, losses, speed,
  anchor/escape counts, defense use, and danger).  It returns an allow-listed
  durable preference for threat/value/sweep aiming, anchor/reactive mobility,
  normal/spaced FIRE, and last-resort/never defense.  It never returns an
  action, heading, sequence, or timestamp; malformed or late output is ignored.

This tests a structural explanation rather than changing a firing interval or a
single threshold.

## Jev state, questions, and bridge

Every request is strict JSON made from the latest snapshot.  Its important
schema is:

```json
{
  "rules": {
    "mode": 0,
    "fire": "press_then_release",
    "noop": "inertia_not_brake",
    "down": "mode_0_hyperspace_last_resort",
    "orientation": "LEFT_increments_RIGHT_decrements",
    "enemy_coverage": "unknown_not_absent"
  },
  "snapshot": {
    "frame": 1550, "lives": 4, "score": 1370,
    "player": "present", "position": "right", "motion": "still",
    "danger": "caution", "impact": "soon", "clearance": "tight",
    "asteroids": 8, "missiles": 1, "uncertain": true
  },
  "threats": [
    {"id": "r13_small_5", "size": "small", "bearing": "center_above",
     "impact": "soon", "clearance": "tight", "motion": "uncertain"}
  ],
  "paths": [
    {"id": "d0", "direction": "0", "clearance": "open",
     "impact": "none", "kind": "escape"},
    {"id": "brake", "direction": "0", "clearance": "limited",
     "impact": "soon", "kind": "brake"}
  ],
  "controller": {"phase": "anchor", "maneuver": "none",
                  "defenses_this_life": 0, "frame_local": true},
  "plan": {"aim": "threat_first", "mobility": "anchor_then_coast",
           "fire": "normal", "defense": "last_resort"}
}
```

The actual implementation sends the current records and no invented enemy
objects; the example is illustrative of the compact fields and named buckets.
Numeric arithmetic is done in code.  `mode_a` and `mode_b` are reversed-order
Choices over:

* `anchor`: damp inherited speed with a code-selected anti-velocity pulse;
* `evade`: favor a named, code-generated escape path if its latest forecast is
  safer;
* `hunt`: aim at the supplied threat/value target and use code-gated FIRE;
* `hold`: preserve a clear-field firing posture, never override a hard contact.

They are accepted only when the values agree and both probabilities are at least
0.30.  The accepted mode lasts 54 frames, but only changes local thresholds; it
cannot issue a sequence.  `target_a` and `target_b` are also reversed Choices
over `none` and at most six current source-slot/size/index ids.  Agreement and a
minimum probability of 0.28 are required, and the id must still be present in
the newest snapshot.  Source slots are not treated as permanent identities.

The `defense` Noul is accepted as a hint only at probability 0.80.  Code still
requires the player to be present, an imminent local coast collision (TTC at
most 6), an unused defense in this life, and no path whose contact is removed
with at least the local gain threshold.  Thus arbitrary stand-in answers cannot
cause hyperspace.  A late answer, a life-epoch mismatch, a negative age, or an
unknown option is discarded.  If Jev times out or returns garbage, the local
controller continues with the anchor/occupancy fallback.

Requests are launched at most every 36 observed frames, with one outstanding
request and a 2.25-second timeout.  `wait_for_tick()` is used on every native
frame while a request is pending, so no model latency becomes a stale action.
The first choice/action is code-owned before the first model response.

## Fallback and evidence

The fallback hierarchy is:

1. missing player, discontinuity, life change, or post-defense recovery: release
   controls and quarantine old tracks;
2. local TTC/path gate: use a revalidated finite escape tranche, or one
   per-life DOWN pulse if all short paths still collide;
3. high observed speed outside an immediate contact: rotate toward anti-velocity
   and issue a single damping pulse when its cooldown permits;
4. settled turret: turn one code step toward the threat-first target (or a slow
   sweep) and press FIRE only on a release/cooldown slot.

Every `ctx.act` is followed by a tick, including while Jev/Astra futures run.
All model parsing is defensive, and model failures are advisory misses rather
than escaping exceptions.  Only standard library and `metaharness.*` imports
are used.

Evidence that would confirm the hypothesis is a higher mean score with later
life epochs, especially for seed-10-style launches: `anchor_pulse` should appear
near nonzero launch velocity, `escape_start` should precede the 18-frame caution
window, and defenses should no longer cluster at TTC 0.  The controller should
retain the high-score horizon runs while using substantially fewer thrust frames
than 0008.  Evidence against it would be early deaths caused by damping, repeated
escape tranches with no changed position/clearance, or a clear-field score loss
without later life epochs.  The official comparison remains unclipped ALE reward;
survival and lives are diagnostics only.
