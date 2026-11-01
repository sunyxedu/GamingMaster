# Candidate 0011 — urgent relocation without the normal thrust cooldown

## Parent and hypothesis

Parent: `candidates/0007/`.

0007's frame-local predictive lane controller is the right architecture: it improved the
mean to 5343.3 with zero stale ticks, so this candidate keeps its five-lane forecast,
pulsed FIRE, asynchronous Jev/Astra supervision, and local clock ownership. Its remaining
failure is more specific than the general movement problem. The controller can correctly
label a collision `immediate`, but its normal lane-thrust cooldown can then suppress the
escape that the label calls for.

The seed-8 trace is the clearest example: near the final loss it reports `immediate` at
frames 9400, 9405, 9410, 9415, 9420, and 9430, while posting `RIGHTFIRE`, `FIRE`,
`FIRE`, `FIRE`, `FIRE`, and `FIRE`; it replans lane `65 -> 85` at frames 9409/9421 and
loses the last ship at frame 9434. In seed 10, the first defense at frame 279 is followed
by `danger=immediate` controls through frame 303, but the only visible thrust response in
that window is one `UPLEFT` at frame 297; the ship changes from 4 to 3 lives at frame
304. These are not stale-model failures: 0007 posted every frame and the local forecast
was already urgent.

**Hypothesis:** once code has an `immediate` swept-contact forecast and a different lane
has been selected, waiting for the ordinary 22/28-frame thrust timer is more dangerous
than one extra bounded thrust pulse. Allowing an urgent, one-frame relocation pulse to
bypass that normal cooldown (but throttling urgent pulses to a short emergency interval)
will preserve more ships without turning the normal central firing policy into the
thrust-heavy policy rejected by 0008.

## One targeted change

`0011` is 0007 with one action-bridge change in `LanePilot._lane_motion`: for an urgent
state with a nontrivial lane goal, the existing turn/thrust pulse remains bounded to one
frame, but an urgent pulse may be issued even while `next_thrust` is in the normal
cooldown. After that pulse, the next urgent pulse is held for six observed frames. The
ordinary non-urgent two-frame pulse and its 22/28-frame cooldown are unchanged; no
proactive movement is added, and defense, aiming, FIRE cadence, model prompts, freshness,
and Astra scheduling are unchanged.

The six-frame interval is only an execution guard against repeatedly holding UP during a
single forecast. It is not a claim about cartridge physics: it is local scheduling, and
it is reset on a life transition along with the existing thrust state. The local geometry
and lane-cost checks still decide whether an urgent pulse exists. Jev cannot request a
raw thrust or bypass that gate.

## Evidence that would decide the hypothesis

Support would be zero stale/error episodes with longer survival and higher mean score,
especially if the seed-8 frame-9434 and seed-10 frame-304/3622-style losses occur later,
while thrust counts remain far below 0008's thrust-heavy runs. Refutation would be earlier
losses caused by emergency drift, a large drop in score despite similar survival, or
urgent pulses that repeat without a changing forecast. The useful diagnostics are
life-loss frames, `danger`/`threat_time`, lane-goal transitions, urgent thrust count,
and score per life; official selection remains unclipped ALE reward.

## Jev interface retained from the parent

Every request is strict JSON containing derived facts, not raw RAM arithmetic:

```json
{
  "rules": {"mode": 0, "fire": "level_press_then_release",
             "noop": "does_not_brake", "down": "mode_0_hyperspace",
             "heading": "left_increments_right_decrements",
             "enemy_objects": "unknown_not_absent"},
  "snapshot": {"frame": 1234, "lives": 3, "score": 530,
                "player_present": true, "position": "middle",
                "motion": "moving", "danger": "immediate",
                "impact_window": "immediate", "rocks": 7, "missiles": 1},
  "lanes": [{"id": "L105", "relation": "right_safe"}],
  "target": {"slot": 13, "size": "small", "bearing": "right_above",
             "motion": "known", "distance": "middle"},
  "controller": {"posture": "avoid", "lane_goal": "L105",
                  "defense_used": false, "frame_local": true},
  "plan": {"aim": "target", "patrol": "adaptive_lane",
           "risk": "cautious", "defense": "last_resort", "fire": "normal"}
}
```

The one batched Jev request asks for: two reversed-order `Choice`s classifying
`clear|distant|watch|immediate|unknown`; two reversed-order `Choice`s selecting the
short-lived posture `focus|sweep_left|sweep_right|hold|avoid`; a `Choice` over `stay` and
the five precomputed lane IDs; and a `Noul` asking whether a mode-0 hyperspace pulse
should be considered. Danger agreement requires the selected probability to be at least
0.45 for `immediate` (0.16 for lower-stakes labels); posture/lane advice requires 0.16.
The emergency thrust change does **not** use Jev's answer: it is gated by the latest
local swept forecast and lane goal. A Jev answer is discarded after 48 frames or on a
life-epoch change, and disagreement/garbage/timeout leaves local control unchanged.

The fallback is the same code-owned policy: if the player is absent, post `NOOP`; if
there is no urgent contact, retain the lane, aim from current geometry, and emit
one-frame FIRE presses on the local release schedule; if there is an urgent contact,
use only the bounded local lane pulse/one-per-life defense gates. Astra remains a
background reviewer of aggregate score, lives, danger, lane cost, and current plan; its
allow-listed plan can never post an action or override the urgent local gate.
