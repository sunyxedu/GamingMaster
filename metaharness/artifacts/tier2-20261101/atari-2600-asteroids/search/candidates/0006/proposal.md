# Candidate 0006 — receding-horizon escape lanes

## Parent and diagnosis

Parent: `candidates/0003/` (the FIRE press/release bridge). The best reference is
`candidates/0005/`, whose frame-local controller is retained as the comparison
point.

The archive shows that frame ownership and bounded FIRE pulses work: 0005 posted
fresh actions on every tick, scored 1,320 on average, and its seed-6 run scored
2,250 while the ship remained still for long scoring periods. The remaining failure
is not stale input. It is that the safety fallback does not actually leave the
collision lane.

The clearest evidence is 0005 seed 8. Just before the first loss, Jev judged
frames 1460, 1484, 1508, 1532, and 1652 as `emergency`; at frame 1652 its compact
state said `center_level`, `near`, and `imminent`, and both risk questions agreed.
Nevertheless, the local bridge's emergency branch only returned `LEFT` or `RIGHT`
until its heading matched an away code. It did not thrust while turning. The ship
therefore stayed around the same centre position and lost lives at ticks 1506,
1755, 2083, and 2094, ending at score 730. The action window immediately before
the second loss contains 23 `RIGHT` ticks, and the final loss contains 31 `LEFT`
ticks. Rotating in place is not an escape when the fragment is crossing the
ship's current lane.

This is a different bottleneck from 0003's long held `UPRIGHT` actions and from
0005's scheduling problem. It calls for changing what Jev selects and what code
executes, not for making the existing posture thresholds more permissive.

## Hypothesis

A short, model-selected **trajectory packet** will survive better than a
model-selected posture plus a rotation-only emergency rule. Code can cheaply
simulate several conservative 20–28-frame lanes using the observed object
positions and velocities. Jev should select among those already-scored packets;
it should not decide an action, calculate an intercept, or infer a direction from
raw RAM. In an emergency, code still has an unconditional safety override, but
it now uses bounded thrust while turning into the best predicted open lane.

The controller should retain 0005's central firing/sweep behaviour in clear
fields. It should add thrust only when the receding-horizon estimate says that
stationary or rotation-only control is materially worse. A one-use, mode-0
hyperspace pulse is a last resort when both bounded thrust lanes are predicted to
overlap the threat.

This is confirmed if the score remains materially above idle while the first and
second life losses move later, especially in the 0005 seed-8 style state where an
imminent centre-level fragment is visible. It is refuted if trajectory packets
cause immediate losses in otherwise clear fields, or if escape thrust moves the
ship but score and survival remain at 0005's 730/early-game-over pattern. A useful
negative result would be a safe but score-poor run: that would indicate that the
lane simulator is too conservative rather than that Jev's route recognition is
wrong.

## What changed

* `LanePlanner` owns a fresh observation, wrap-aware relative geometry, velocity
  uncertainty, approximate swept clearances, and a finite-horizon simulation of
  `hold`, left/right scans, left/right thrust escapes, counter-drift, and defense.
  Unknown velocity receives a conservative penalty; source slots are not treated
  as permanent identities.
* The native action bridge is packetized but frame-local. Scan packets fire only
  one-frame FIRE presses with release gaps. Escape packets are capped at ten
  frames and use `UPLEFT`/`UPRIGHT` followed by a short `UP` tail; they cannot
  leave thrust latched for the duration of a slow model call. Every current frame
  rechecks the danger facts and can abandon the packet.
* Jev no longer chooses a 14-action name, a target slot, or a vague posture. It
  gets two independently ordered Choices over the code-generated route IDs.
  Agreement is required, and high-consequence routes have a higher probability
  gate. Jev's selected route is only an intent for a short freshness window; code
  rejects it after a life change or when current safety facts invalidate it.
* The deterministic emergency selector is stronger than the model: it compares
  the two thrust lanes every frame, locks the better lane only briefly, and uses
  hyperspace only when both lanes predict overlap, the threat is imminent, and
  defense has not been used recently in that life. A late or malformed Jev answer
  therefore cannot suppress an escape.
* Astra is an asynchronous objective author, not a delayed action supervisor. It
  receives the compact route state once at launch and may return only
  `objective`, `sweep`, and `defense` policy values. Those values adjust route
  preference and firing/defense posture; Astra never supplies actions, route IDs,
  timestamps, or arithmetic. If its roughly minute-long call is cancelled by
  game-over, the local default remains fully executable.

## System-1 state and questions

Every Jev request carries strict JSON with these top-level parts:

```json
{
  "rules": {"fire_requires_release": true, "noop_does_not_brake": true,
             "down_is_hyperspace": true, "packets_are_bounded": true},
  "snapshot": {
    "frame": 1234, "lives": 3, "score": 580,
    "player_present": true,
    "position": {"x_band": "middle", "y_band": "middle", "centrality": "central"},
    "heading_code": 9, "motion": "still|moving|unknown",
    "danger": "clear|approaching|emergency|unknown",
    "rocks": {"count": 6, "immediate": 0, "approaching": 1},
    "missiles_visible": 1, "motion_discontinuity": false,
    "enemy_coverage": "unknown_not_absent"
  },
  "objective": {"objective": "safety_first", "sweep": "alternate",
                 "defense": "last_resort"},
  "current_intent": "scan_left",
  "routes": [
    {"id": "escape_left", "purpose": "...", "clearance": "open",
     "contact": "later", "valid": true}
  ]
}
```

The actual route list contains seven named options: `hold`, `scan_left`,
`scan_right`, `escape_left`, `escape_right`, `counterdrift`, and `defend`.
Descriptions explain the bounded semantics and code-derived clearance/time
buckets. They do not ask Jev to compare raw distances or count objects.

The single request batches:

1. `route_a`: choose the safest useful bounded route, with options in a
   frame-alternating order.
2. `route_b`: independently choose the route again with the reverse order.

An answer is accepted only when both values agree, the selected option is a
current route, and the minimum selected-option probability is at least **0.48**
for `escape_left`, `escape_right`, `counterdrift`, or `defend`, or **0.18** for
ordinary hold/scan routes. A route is also rejected if it is no longer valid in
the current frame. Accepted escape intents expire after ten frames; other
intents expire after eighteen, and all intents expire on a life change or after
24 frames of observation age. These thresholds make cheap firing advice easy to
use while requiring strong agreement for thrust or hyperspace.

The local safety layer does not ask Jev whether an emergency exists. It computes
that from inflated approximate clearances and closest approach. `emergency`
means present overlap/tight clearance or a predicted near crossing in about 14
frames; `approaching` is the wider 22/32-frame band. This keeps arithmetic and
consequence-sensitive gating in code, as required by the Jev contract.

## Timing and fallback

A Jev request is spawned at most once per 18 observed frames, with a three-second
request timeout and one outstanding request. The game is always advanced with
`wait_for_tick()` and a newly posted local action; model latency cannot make an
escape or FIRE pulse stale. Responses are tagged with their judged frame and
life. Timeout, runtime failure, malformed stand-in answers, disagreement, low
probability, or stale state simply leaves the receding-horizon local route in
place.

The first local route is posted before the first tick. With no player, during a
36-frame respawn/discontinuity grace window, or after a decoder error, the
fallback is `NOOP`. In a clear field it alternates short rotation sweeps and
one-frame FIRE presses. In an approaching field it prefers a predicted lane only
when that lane improves clearance. In an emergency it chooses the better bounded
thrust lane without waiting for Jev. If both thrust lanes predict overlap, the
threat's time-to-contact is at most ten frames, and defense is permitted, it
uses one `DOWN` pulse; otherwise it continues the best bounded escape. No model
answer can repeat `DOWN` indefinitely.

Astra is started once in the background and may revise only the durable objective
policy if it returns before the episode ends. Its timeout, cancellation, prose,
unknown field values, or invalid JSON leaves `safety_first / alternate /
last_resort` unchanged. `ModelUnavailable` and budget/deadline conditions follow
the harness contract; ordinary inference failures are contained so arbitrary
pre-flight stand-ins cannot terminate the episode.
