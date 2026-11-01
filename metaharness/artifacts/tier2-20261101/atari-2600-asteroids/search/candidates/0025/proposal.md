# Candidate 0025 — let validated escapes reach their thrust pulse

## Parent and targeted diagnosis

Parent: `candidates/0022/`. I keep its calibrated per-heading gun, bounded
occupancy paths, one-frame clock owner, short FIRE presses, Jev gun-choice
interface, and advisory Astra review. This is a careful refine, not a new
movement policy.

The parent’s worst official training episode is seed 2: native reward
**24,960**, two life losses, with life epochs at frames **14,376** and
**16,912**. The trace shows the exact gap between its advertised escape layer
and its executed controls:

* An escape starts at frame **14,202** (`ttc=12`, path `d2`), but the
  14,190–14,360 collision window contains no `UP`, `UPLEFT`, or `UPRIGHT`
  action before the defense at **14,355** (`ttc=5`). The ship is lost at
  **14,376**.
* In the second loss window, the local radar sees `ttc=15` at **16,701**,
  then the decoded forecast disappears at frames **16,706–16,714**. The
  controller defends at **16,715** with `ttc=0`; escape starts occur only
  afterward at **16,782`, `16,810`, `16,875`, and `16,897`, and the life epoch
  follows at **16,912**.
* The parent’s `gun_summary` for this episode reports **15 escape starts** but
  only **4 thrust frames**. The action records around the later starts are
  mostly `RIGHT`, `LEFT`, and `NOOP`; only `UPRIGHT`/`UP` appears at
  **16,889–16,895**, after the critical window.

This is not evidence that more movement or more hyperspace is generally good;
those changes were tested and regressed in 0021/0023. It is evidence of a
specific executor deadlock. In 0022’s `KinematicPilot.update`, an active
escape is discarded either when measured player speed is below `0.10` or when
the forecast is labeled clear only three frames after the start. The turret is
usually stationary (`speed=0.0` in the collision-window control records), and
one-frame decoder gaps can make a real warning look clear. A newly started
escape is therefore canceled before it can finish turning and issue its
existing bounded thrust pulse. The path forecast can be correct while the
action bridge silently reduces it to rotation and coasting.

## Hypothesis

If a code-generated, revalidated escape is allowed a bounded **first-pulse
commitment** while the ship is initially stationary or a forecast briefly
flickers clear, it will reach its already planned finite thrust pulse early
enough to change some collision outcomes, without turning every warning into
roaming. The parent’s path gain thresholds, warning gates, expiry, defense
gate, and turret fallback remain unchanged. Thus this isolates **premature
pre-thrust escape cancellation**, rather than changing a risk threshold,
firing cadence, or defense budget.

## One code change

In `KinematicPilot.update`, an escape now records whether it has issued a real
thrust pulse. Before that first pulse, the old low-speed cancellation is
excluded and the dynamic `forecast_clear` / `path_worse` revalidation checks
wait; the existing expiry, missing-path, life-change, and discontinuity
quarantines still cancel it. After the first pulse, the parent’s original
per-observation clear/path-cost revalidation resumes. This gives a selected
path only its existing **16- or 24-frame** bounded maneuver window to get
through the turn and issue its 2–3-frame tranche; it does not add a movement
schedule, new action name, seed branch, imported trajectory, or extra defense.

## Evidence that would confirm or refute it

Confirming evidence is not merely more `escape_start` events. In traces, starts
should be followed by bounded `UP`, `UPLEFT`, or `UPRIGHT` pulses while danger
is still `watch`/`caution`, followed by revalidation/cancellation or a finite
completion. Seed-2-style defense should move earlier or disappear, life epochs
should be later/fewer, and mean unclipped ALE reward should improve without a
large increase in thrust frames or stale ticks.

Refuting evidence is any systematic score loss on clear fields, repeated
thrust into a path whose next observations worsen, more deaths immediately
after a now-executed pulse, or no increase in thrust/completed escapes despite
active starts. A result with the same high survival but lower reward would mean
that the parent's stationary calibrated turret was the better tradeoff. Score
remains the only primary objective; lives and escape diagnostics explain the
result and are not bonuses.

## System-1 interface and bridge (unchanged from 0022)

Jev gets one compact strict-JSON state built from the newest observation, not
raw RAM arithmetic:

* `rules`: mode 0/difficulty 0, FIRE press-then-release and two observed
  missile slots, NOOP coasting rather than braking, LEFT/RIGHT orientation
  semantics, mode-0 hyperspace, and unknown enemy coverage;
* `snapshot`: absolute observation frame, player presence, missile count,
  danger/impact buckets, score/lives, and uncertainty;
* `gun.opportunities`: at most three code-ranked opportunities described by
  `calibrated|provisional` trajectory, `clean|marginal` intercept,
  `aligned|short|long` turn, `shootable_before_contact|crossing|noncrossing`,
  and `known|uncertain` motion;
* `gun.calibration`, recent score progress, the local emergency override, and
  Astra’s current durable focus/goal.

Every request batches three independent typed judgments:

1. `focus_a`: `Choice` among `threat`, `clean`, `value`, and `unknown`;
2. `focus_b`: the same semantic choice with reversed option order, to reduce
   Jev’s option-order bias;
3. `clearance`: a `Noul` asking whether the supplied opportunities explicitly
   contain a known-motion, clean `shootable_before_contact` opportunity.

The two focus choices are accepted only when they agree, their selected-option
probabilities are each at least **0.60**, the origin life epoch still matches,
and the answer is no more than **36 frames** old. Accepted gun roles last at
most **54 frames**, while the concrete target and intercept are recomputed on
every observation. `clearance >= 0.80` is logged as semantic support only; it
never disables local escape or defense. The Noul cannot create a movement or
hyperspace action.

Requests are started at most every **36 observed frames**, one outstanding,
with a **2.25 s** timeout. A late, malformed, disagreeing, low-probability, or
life-stale answer is dropped. While it is pending, the local controller owns
every native frame. The fallback is, in order: discontinuity/life recovery;
local occupancy and revalidated escape; code-gated one-per-life hyperspace
when no short path is safe; observed-momentum damping; then calibrated
threat/value aiming with released FIRE pulses. The only changed bridge is that
an accepted escape is not discarded merely because its current speed is still
below `0.10`.

Astra remains one asynchronous aggregate review after launch (150 s timeout).
It may return only allow-listed `aim` preferences (`threat_first`,
`value_first`, or `sweep`), which affect gun ranking. It cannot return actions,
movement thresholds, defense timing, or trajectories; malformed text is
ignored. All inference goes through `ctx.decide`/`ctx.think`, and all model
failures leave the deterministic fallback running.
