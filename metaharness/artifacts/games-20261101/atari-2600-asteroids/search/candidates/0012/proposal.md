# Candidate 0012 — threat-shadow escape and post-defense verification

## Parent and diagnosis

Parent: `candidates/0011/`; reference best: `0011` (mean 10,531.7).

The parent establishes that frame-local control and a very low thrust duty cycle are
valuable.  Its remaining losses are not stale-model losses: it posts a fresh action on
every tick, yet its local urgent branch can thrust in the current nose direction before
that nose is aligned with the selected lane.  In the seed-8 trace, the first life has a
`DOWN` at frame 387, then at frame 398 the ship is at x=133 with an immediate small rock
at x=130/y=88; it posts `UPLEFTFIRE`/turns until the life loss at frame 410.  In seed-10,
frame 279 defense is followed by immediate geometry and the first loss at frame 304.
Later losses likewise show the controller firing or turning while the nearest small
fragment is already in the ship corridor (seed-8 around frames 3734–3748 and seed-4
around 5963–5986).  A lane score at one fixed y does not say that the *current nose*
will move toward that lane, and a hyperspace pulse is currently treated as spent without
checking whether its landing is still critical.

## Hypothesis

The missing mechanism is a two-stage **threat shadow**: first choose a safe *escape
vector* from the current relative trajectories, then rotate until the nose is aligned
with that vector before thrusting.  The parent’s urgent combined turn/thrust can amplify
an incorrectly oriented escape and make a lane crossing worse.  A short rotate-only
stage, followed by at most one-frame aligned thrust pulses, should preserve the parent’s
low movement scale while making each thrust geometrically meaningful.  A defense pulse
should also enter a short verification state: after a discontinuity or a new landing,
clear the old threat shadow and permit a second pulse only when the *new* observation
still predicts contact and the landing did not improve clearance.  This tests landing
verification, not a generic retry macro.

This is deliberately not the rejected 0008 proactive patrol or 0010's open-ended
receding-horizon/retry policy.  Normal control remains the 0011 low-duty lane turret;
only the urgent bridge changes to an aligned vector state machine, with a two-pulse
per-life cap and a post-pulse observation gate.  Unknown/flickering tracks are inflated
and never treated as proof of safety.

## What changed

* `ThreatShadow` samples wrap-aware relative trajectories for the current position and
  the five lane positions, plus four cardinal/intermediate escape headings.  It ranks
  candidates by minimum clearance, contact time, centrality, and turn cost, retaining a
  named vector rather than asking Jev to do arithmetic.
* The local action bridge has explicit `aim -> align -> pulse -> settle` phases.  During
  urgent danger it turns only until the chosen nose heading is within one orientation
  code of the escape heading; then it emits one bounded thrust frame and rechecks the
  next observation.  No urgent action can thrust while the nose points elsewhere.
  Normal danger uses the parent’s short lane pulses and target/firing policy.
* `DOWN` is allowed only for a current critical shadow, at most twice per life, with a
  14-frame gap.  Each pulse records position/clearance and opens a 12-frame verification
  window.  A landing/discontinuity invalidates the old vector; a second pulse is legal
  only if fresh geometry remains critical and the measured clearance did not improve.
  This prevents repeated stale `DOWN` while testing whether defense actually helped.
* Jev still receives derived buckets and chooses semantic posture, but it now judges
  `align_escape` versus `lane_escape` versus `target_fire`; it cannot request a raw
  thrust, defense, or duration.  Astra remains a delayed allow-listed policy reviewer
  and never owns the clock.

## Evidence that would confirm or refute it

Confirmation would be later first/final life-loss frames on seed-8/10-like openings,
with urgent thrust count still close to 0011's roughly 40–120 per run, and with second
DOWN pulses concentrated after a post-pulse critical landing.  It is especially
supportive if the frame-410 and frame-304 patterns become rotate/verify events rather
than a thrust in a misaligned nose.  Refutation would be lower score with no survival
gain, many rotate-only stalls while contact is imminent, repeated defense after a
successful landing, or thrust counts approaching 0008/0010.  Useful logs are shadow
kind/heading, alignment error, minimum clearance, phase, defense verification outcome,
and life-loss frame; official selection remains unclipped ALE reward.

## Tier-2 System-1 interface

Every request is strict JSON with no raw RAM arithmetic:

```json
{
  "rules": {"mode": 0, "fire": "press_then_release", "noop": "inertia",
             "down": "mode_0_hyperspace", "heading": "left_increments",
             "enemy_coverage": "unknown_not_absent"},
  "snapshot": {"frame": 1234, "lives": 3, "danger": "watch|critical|clear",
                "impact": "none|soon|immediate", "position": "middle",
                "motion": "still|moving|unknown", "rocks": 7, "missiles": 1},
  "shadow": {"phase": "align|pulse|settle|none", "kind": "lane|vector|none",
             "heading": "left_up", "clearance": "tight", "contact": "soon"},
  "candidates": ["target_fire", "sweep_left", "sweep_right", "hold",
                 "lane_escape", "align_escape"],
  "defense": {"pulses": 1, "verify": "pending|improved|still_critical|none"},
  "plan": {"aim": "target", "risk": "cautious", "fire": "normal"}
}
```

One batched request has two reversed-order `Choice`s over the currently generated
candidate ids, two reversed-order risk `Choice`s, and a `Noul` about whether a pending
critical landing deserves a bounded retry.  Choices must agree and be no more than 48
observed frames old.  `align_escape` needs probability at least 0.40; ordinary posture
needs 0.17.  Defense advice needs yes probability 0.78 and is still subordinate to the
current local critical gate and post-pulse verification.  Low confidence, disagreement,
malformed stand-ins, timeout, life change, or discontinuity leaves code-owned control
unchanged.

The fallback is deterministic: absent player -> `NOOP`; clear field -> target-aware
turn plus one-frame FIRE press/release; watch danger -> the least-cost lane with bounded
motion; critical danger -> a verified shadow alignment/pulse or bounded hyperspace.
Astra may alter only target/sweep aim, cautious/balanced thresholds, normal/spaced fire,
and never/retry defense posture.  The harness waits for each native tick and posts one
fresh legal action, so all model latency is advisory and cannot hold thrust or FIRE.
