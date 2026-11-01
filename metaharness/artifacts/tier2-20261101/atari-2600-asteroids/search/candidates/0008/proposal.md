# Candidate 0008: proactive safe-sector movement

## Parents and diagnosis

Parent: `candidates/0005/` (the best evaluated candidate). I also compared its
worst trace with `0004/`, whose emergency thrust was only a late reaction.

The strongest remaining failure is not stale control. Candidate 0005 posts a
fresh local action on every native frame (`0%` stale ticks), but its normal
policy is effectively a stationary turret: it rotates and fires, and its
escape branch rotates away without thrust. In the worst 0005 episode
(`trace_seed8.jsonl`), the ship is at approximately `(84,100)` with zero
velocity at the observations after losses at frames **1598, 1847, 2175**, and
it ends at frame **2186** with score **730**. The same pattern is visible in
seed 10: the ship remains near `(85,100)` and stationary until its last life
at frame **2605**. It therefore cannot leave a vertical crossing lane before a
fragment reaches the center. Candidate 0005 also never posts `DOWN` in seed 8;
its model-gated defense is too conservative to rescue this failure.

Candidate 0004 does use bounded emergency thrust, but its seed-8 post-loss
positions are still `(84,100)`, `(85,98)`, and `(84,92)` and it reaches game
over at tick 2000. That is evidence against merely making the existing
emergency pulse longer. It does not test deliberate movement while clearance
is still recoverable.

## Hypothesis

The score/survival ceiling is caused by center-lane commitment. Controlled
stationary firing is useful for the opening field, but once several rocks or
fragments are present, waiting for `emergency` is too late: turning changes the
nose, not the ship's position, and NOOP does not brake. A controller that
proactively selects a short safe sector from swept relative trajectories,
rotates to it, and applies only a bounded thrust pulse should reduce repeated
center collisions without returning to the long `UP` latches of the early
candidates.

This is a structural test, not a firing-gap or prompt tweak. The controller
keeps stationary/pulsed firing as its fallback, but adds a low-duty-cycle lane
change whenever a field is dense or a cautionary crossing is predicted. A
critical collision can independently invoke one one-frame mode-0 hyperspace
pulse per life; Jev cannot suppress that safety gate. There is no assumed
hyperspace probability or score bonus.

The hypothesis is supported if 0008 keeps the `0%` stale-tick property while
raising score and/or the number of frames before each life loss, especially in
seed-8/seed-10-like dense fragment fields. It is refuted if proactive pulses
produce earlier deaths than 0005, mostly push the ship to an edge without
clearing threats, or score no better than the stationary policy while spending
more actions on thrust. Per-episode diagnostics to inspect are move-pulse
count, minimum derived clearance, life-loss frames, defense pulses, and final
score; official score remains unclipped ALE reward only.

## What changed

* `SafeSectorController` derives wrap-aware relative positions, relative
  velocities, approximate swept clearance, target heading, and a danger bucket
  locally. It treats object slots as short-lived observations rather than
  identities and invalidates plans on life changes or motion discontinuities.
* For eight coarse heading sectors, code projects the relative rock position
  over 0--46 frames under a conservative acceleration estimate. It chooses the
  highest-clearance sector with a soft edge penalty. The exact arithmetic stays
  in code; Jev only chooses among named candidates.
* A normal lane change first turns without thrust, then uses at most five
  thrust frames and waits 150 frames before another proactive move. An
  emergency plan uses at most seven thrust/turn frames. Thus a delayed answer
  cannot leave UP held for hundreds of frames. The current frame can always
  veto a stale move and start a new safety plan.
* FIRE remains a one-frame press followed by a four-frame release gap. The
  normal fallback aims at a code-ranked target or performs a short sweep; no
  fire or turn answer is allowed to become an unbounded action latch.
* `DOWN` is a deterministic, one-frame defense only when current derived danger
  is emergency and the nearest swept clearance is critical (clearance <= 3 or
  time-to-contact <= 7 frames), at most once per life. Jev may approve the
  semantic defense but cannot create it from a stale observation.
* Astra receives aggregate score/life/clearance evidence after 240 observed
  frames and returns an allow-listed durable policy (`safe_lane` versus
  `stationary`, risk, fire cadence, and defense posture). Its response changes
  only future local thresholds; it never blocks a tick or posts an action.

## Jev state, questions, and bridge

Every request sends a strict JSON state with:

* verified mode-0 rules (`FIRE` requires release, `NOOP` preserves inertia,
  `DOWN` is hyperspace, and unknown enemy coverage is not absence);
* current frame, lives, score, player presence, coarse centrality and speed,
  danger, nearest-clearance and nearest-TTC buckets, asteroid/size counts, and
  motion-discontinuity status;
* one code-generated target summary (size, cardinal bearing, distance and
  clearance buckets, velocity-known flag);
* at most five named `candidate_maneuvers`: `move` (with a code-selected safe
  heading), `aim`, `sweep_left`, `sweep_right`, `coast`, plus `defend` only for
  an emergency. Candidate descriptions explicitly say that movement is a
  bounded pulse, not a hold; no raw RAM or arithmetic is sent.

The single Jev request batches four independent typed judgments:

1. `maneuver_a`: choose a candidate from the normal order;
2. `maneuver_b`: repeat the same choice with reversed option order to reduce
   first-option bias;
3. `risk`: classify `clear`, `caution`, `emergency`, or `unknown` from the
   already-derived buckets;
4. `defense`: a Noul whose yes probability is considered only for a current
   emergency.

Both maneuver answers must agree. The selected option must have probability at
least **0.38** for `move`/`defend`, or **0.16** for cheap aim/sweep/coast
choices. The answer is discarded after **42 observed frames**, on a life change,
or if its candidate is no longer present. Accepted tactical advice expires
after **36 frames**. A defense yes needs probability at least **0.78**, but the
local critical-clearance gate remains authoritative. The model can never
weaken an emergency move or force a defense outside that gate.

## Fallback and timing

The first action and every subsequent action are computed from the latest
observation. While Jev (2.5 s timeout, one outstanding request every 30
frames) or Astra (100 s timeout, one scheduled request) runs, the harness calls
`wait_for_tick()` and posts the local result. A missing player yields NOOP;
a malformed/late model result is ignored. Emergency geometry immediately
chooses safe-sector movement or the one-shot defense. At normal risk, a
currently due lane change wins over model advice; otherwise the controller
uses the bounded aim/fire fallback. Budget/deadline and model-unavailable
exceptions follow the contract, and no custom background loop is left behind.
