# Candidate 0021 — contact-rescue turret

## Parents and concrete diagnosis

Parents: `candidates/0018/` and `candidates/0017/`.

The two parents earn their score for different, compatible reasons:

* 0017 has the strongest robustness mechanism from its idle crossover: the
  emulator clock is owned locally, FIRE is pulsed, and uncertain geometry does
  not turn into an open-ended thrust command. That produced 17,220 on seed 0
  and 16,540 on seed 4 with essentially no stale ticks.
* 0018 adds useful kinematic bookkeeping and a two-dimensional, wrap-aware
  occupancy ranking. It raised the mean to 12,815 and reached the 18,000-frame
  budget on seeds 0, 4, 8, and 10. Its launch-velocity estimate and one-pulse
  paths are useful when they actually produce a bounded movement.

The common failure is visible in the collision windows, not in model latency.
In 0018 seed 6, the controller posts `defense` at frame 1657 and then posts
`NOOP` through the critical window at frames 1659–1677; the life epoch is at
1678. The same pattern occurs after defense at 3149 and 4805. In seed 2,
0018 has a small/medium corridor at frames 6918, 7303, 8671 and 8943; the
controller repeatedly reports critical TTC 0 and then uses NOOP recovery until
life loss (for example 8671 → 8688 and 8943 → 8968). The request state itself
contains paths whose impact is immediate/overlap, but the action bridge treats
“all paths overlap” as a reason to wait. In 0017 seed 10, defenses at 1554,
2999, 3321 and 3793 are followed by life epochs at 1566, 3018, 3338 and later;
its conservative coast firewall is safe in clear fields but becomes passive
when an actually-close rock is beside the ship. This explains both the high
long runs and the catastrophic seed-sensitive tail.

The hypothesis is therefore **a close-contact state needs a distinct bounded
escape protocol**. More Jev labels or more late hyperspace retries will not
help if the deterministic bridge posts NOOP while the collision persists.
Conversely, broad proactive movement is refuted by 0008 and 0020. I retain the
parents' stationary turret for clear fields, but when a close crossing is
observed I select a short separation pulse, revalidate it every frame, and only
then use one hyperspace pulse. A defense no longer starts a long coast
quarantine: the next observed snapshot can start a fresh contact rescue.

## What changed

This is a new contact-state architecture rather than a threshold-only merge.

1. **Short flicker bridge.** Current asteroid records are augmented by at most
   two frames of same-size, same-slot predicted tracks. A large jump or size
   change rejects the old track. Ghosts are widened and are used as uncertain
   threats, never as permanent identities or proof that an enemy exists.
2. **Impulse-aware occupancy.** Candidate paths model one actual thrust pulse
   after a turn delay, rather than treating an escape as a long consecutive
   thrust reservation. The code compares coast, nose and eight cardinal/
   diagonal headings using relative velocity and swept approximate boxes.
3. **Contact rescue.** A near corridor is detected from current axis overlap,
   closing motion, or close unknown geometry. For a caution/critical contact,
   code chooses the path that improves the *nearest* object's first crossing or
   local clearance, with a direct-away tie break. It executes at most one or two
   thrust pulses over a 16–20-frame expiry. If the heading is not yet aligned,
   it turns; at the last few TTC frames it uses a thrust-plus-turn pulse instead
   of waiting for a perfect angle. Every fresh observation can cancel or
   replace the rescue.
4. **Defense is a last resort, not recovery.** At most one DOWN pulse is used
   per life and only for a close unresolved contact. A subsequent snapshot is
   allowed to start a two-pulse rescue if the player is still present. A
   life/teleport/discontinuity clears old tracks and routes, but does not impose
   the parents' long NOOP window.
5. **Preserved turret and anchor.** Outside the contact alarm, the controller
   uses the low-duty target/sweep FIRE loop and only one forecast-supported
   preemptive pulse. An observed launch velocity can receive isolated
   anti-velocity anchor pulses, but uncertainty and a clear field never cause
   patrol movement. All controls are posted once per native frame while model
   calls run in `ctx.spawn`.

Astra only chooses allow-listed durable preferences (`aim`, `mobility`, FIRE
spacing, and defense posture). Jev chooses semantic regime and a currently
observed target. Neither can issue a direction, action, defense, or sequence.
Garbage text, arbitrary stand-in answers, stale answers, and low-confidence
answers are ignored.

## Jev state and typed bridge

Each request is strict JSON made from the newest snapshot. The main schema is:

```json
{
  "rules": {
    "mode": 0,
    "fire": "press then release; holding is not rapid fire",
    "noop": "coasts; it is not a brake",
    "down": "mode-0 hyperspace; one bounded last-resort pulse",
    "enemy_coverage": "unknown; null is not absence"
  },
  "snapshot": {
    "frame": 4800,
    "lives": 2,
    "score": 4370,
    "danger": "critical",
    "near_corridor": true,
    "impact": "immediate",
    "motion": "still",
    "uncertain": false,
    "asteroids": 6,
    "missiles": 1
  },
  "threats": [
    {"id": "r5_medium_2", "size": "medium",
     "bearing": "center_below", "impact": "soon",
     "clearance": "overlap", "motion": "known", "ghost": false}
  ],
  "paths": [
    {"id": "d4", "direction": "4", "impact": "soon", "clearance": "tight"},
    {"id": "d12", "direction": "12", "impact": "immediate", "clearance": "overlap"}
  ],
  "controller": {
    "phase": "rescue",
    "rescue_path": "d4",
    "rescue_pulses": 1,
    "defenses_this_life": 0,
    "frame_local": true
  },
  "plan": {"aim": "threat_first", "mobility": "contact_rescue",
           "fire": "normal", "defense": "last_resort"}
}
```

The request batches five independent judgments:

* `mode_a` and `mode_b` are reversed-order `Choice`s over `rescue`, `hunt`,
  `sweep`, and `hold`. Agreement with probability at least 0.30 lasts 54
  frames. `rescue` is only an advisory confirmation; it cannot create thrust.
  `hold` can suppress clear-field aiming but cannot suppress local rescue.
* `target_a` and `target_b` are reversed-order `Choice`s over `none` and at
  most seven current slot/size/index ids. Agreement with probability at least
  0.28 is accepted for 54 frames, and the id must still be present. Source
  slots are not treated as identities across size changes or discontinuities.
* `defense` is a `Noul`. A true answer is recorded only at probability 0.80,
  but the code still requires a close unresolved crossing, an unused defense,
  and no materially safer local pulse. It is never an independent DOWN command.

A request is launched at most every 36 observed frames, with one outstanding
request and a 2.25-second timeout. Its origin frame and life epoch are stored;
answers older than 42 frames, from another life, or with malformed values are
dropped. The local controller remains executable before, during, and after the
request.

## Fallback and success test

The deterministic priority is:

1. absent player or discontinuity: NOOP and clear old tracks;
2. active, revalidated contact rescue: turn or issue one bounded thrust pulse;
3. close unresolved contact with no remaining useful pulse: one DOWN pulse,
   followed by a fresh rescue attempt if the player remains visible;
4. a narrow forecast-supported preemptive pulse;
5. isolated launch-velocity damping;
6. sparse target/sweep FIRE turret, with NOOP as the uncertainty fallback.

A stale or low-confidence answer never bypasses this hierarchy. Each `ctx.act`
is followed by `wait_for_tick`; no model call owns the emulator clock.

Confirmation would be a higher mean unclipped ALE score than 0018, with the
same or lower stale fraction, and especially later life epochs in the 0018
seed-6/seed-2 windows. Trace evidence should show `rescue_start` and
`rescue_pulse` before the life epoch, fewer `defense` events at TTC 0, and no
long run of critical `NOOP` records after a defense. It should retain the
long-horizon 0018 turret runs without approaching the thrust-heavy behavior of
0008.

The hypothesis is refuted if contact rescues cause early drift deaths in clear
fields, if ghost records produce repeated false alarms, if rescue pulses do not
change the next observed position/clearance, or if the turret loses much more
score without later life epochs. The official objective remains the original
unclipped ALE reward; survival and lives are diagnostics only.
