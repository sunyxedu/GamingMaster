# Candidate 0024 — score-closed-loop firing

## Parent and diagnosis

Parent: **0012**, as directed for this explore generation; **0022** is the best working
reference (28,303.3 mean native reward), and 0018 is the mobility/safety reference. I
retain the useful plumbing shared by those candidates—native-frame ownership, cold
projectile calibration, sparse thrust, short FIRE presses, and local hyperspace gates—rather
than attempting another escape rewrite.
The assigned parent 0012 is useful as a negative control: its trace records zero
`shadow_plan`/`shadow_pulse` events in the failing seed-2 and seed-10 episodes, so adding
another shadow or defense retry would not test a new mechanism. Its zero-stale local loop
also confirms that this candidate should keep the clock plumbing and put the experiment in
weapon outcome selection instead.

The 0022 traces show a persistent distinction between a predicted shot and a rewarded shot.
For example, seed 0 posts 1,867 FIRE actions but observes 994 missile tracks and ends at
30,850; seed 2 posts 1,873, observes 967, and ends at 24,960. The final summaries report
roughly 990 calibrated samples in both cases, so the difference is not simply failure to
learn the 16 projectile vectors. The shot records also label almost every shot
`clearable:false`: in the complete trace, seed 0 has 0/187 clearable records in the sampled
shot log, while seeds 2, 4, 6, 8, and 10 have only 0–2. This field is a forecast, not a hit
oracle, but it is evidence that the solver's target/intercept test is not a useful proxy for
what actually produces ALE reward.

The strongest direct clue is the score stream itself. In the late part of seed 0, rewards
continue after shots whose predicted `clean`/`clearable` field is often false, while many
shots with low geometric miss do not have a matching immediate reward. The controller has
no feedback path from those outcomes: Astra only chooses a durable `value_first`/`threat`
style and Jev chooses a generic gun rule. Thus 0022 spends the same five-frame firing
budget on a mathematically plausible but empirically unproductive bearing. It can survive
(the six 0022 episodes all reach 18,000 frames), but its score remains about 25k–31k.

## Hypothesis

A short, code-owned reward attribution loop can improve the official score without adding
movement or late defense: when a native score increase follows a recent FIRE press, assign
that increase softly to the press's observed heading, retain that bearing for a bounded
number of frames, and let it compete with—not override—fresh collision and intercept
facts. This tests a mechanism that no previous candidate tried: using the actual public
reward stream to close the weapon-selection loop, rather than trusting approximate boxes as
hit labels.

This is intentionally conservative. It does not claim that the last shot caused a reward,
does not modify ALE score, and does not persist knowledge across episodes. Attribution is
only an exponential recency weight over uncredited presses within 52 frames. A reward
bearing expires after 26–34 frames, decays, is cleared on a life/discontinuity reset, and
is never allowed to cancel escape or defense. A score delta remains the native score and
is used only as a policy signal.

## What changed

* **Reward-bearing memory.** `RewardTurret._feedback` watches the authoritative cumulative
  score. On a positive delta, it selects the most recent uncredited FIRE press in a short
  window, updates a 16-entry heading table, and emits `reward_bearing`. The table is
  confidence-weighted by evidence and periodically decayed; it is not a seed schedule.
* **Reward-aware candidate ranking.** The harness retains 0022-style online missile-vector
  learning and moving-rock interception, but each candidate receives a small measured
  bearing bonus. Target ranking is cached for at most four native frames while the
  collision/danger derivation remains fresh every frame; this is a scheduler margin, not
  an action latch. The moving-rock closest-approach estimate checks neighboring integer
  frames instead of scanning a long projectile ray, preserving the same bounded geometric
  candidates with less CPU time. An Astra/JeV-selected `reward_lock` can prefer the active bearing only
  in clear/distant states. A current clean pre-contact opportunity and all safety logic
  outrank that preference. When no candidate exists, the turret performs a short sweep.
* **No additional motion budget.** Escape uses at most one thrust pulse after rechecking
  a 30-frame swept forecast. Hyperspace remains capped at two pulses per life and only at
  a TTC of three frames or less with no demonstrably better local pulse. This keeps the
  experiment focused on score attribution and avoids the refuted repeated-rescue and
  thrust-heavy changes from 0021/0023.
* **Model boundary.** Jev now judges four semantic offensive preferences (`reward_lock`,
  `intercept`, `small_first`, `sweep`) plus a derived danger class and a bounded lock
  confirmation. It cannot invent a heading or action. Astra sees reward-bearing counts,
  calibrated heading summaries, score, lives, and shot statistics and may return only
  `focus`, `lock`, and `fire` preferences. Garbage Astra text and arbitrary typed stand-in
  answers are ignored safely.

## Evidence that would confirm or refute it

Confirming evidence is not merely a higher score in one run. I would look for:

1. `reward_bearing` events with plausible recent ages, repeatable heading-rate separation,
   and a short dwell that is followed by higher native score rate than the pre-lock period;
2. no increase in thrust, stale ticks, or defense frequency, and preferably no increase in
   life losses relative to 0022;
3. a mean reward above 28,303.3 across training seeds, with gains not concentrated in one
   launch seed; and
4. reward locks surviving only while fresh score evidence exists, rather than becoming a
   permanent heading bias.

Refuting evidence would be zero reward events, reward-bearing headings indistinguishable
from the baseline, reward locks that suppress score, or score/life deterioration while the
calibrated geometry is unchanged. If clearable/intercept predictions remain almost always
false but reward feedback improves score, that would refute 0022's geometric proxy as a
hit label while supporting this narrower empirical controller. If reward attribution
improves only after a life or during an unobserved slot reuse, it should be treated as
ambiguous, not proof of causality.

## Tier-2 schedule and fallback

The loop posts one local action before waiting and one after every `wait_for_tick`; no model
call owns the emulator clock. At most one Jev request is in flight. It is launched every 36
frames (or 48 after a failed launch), with `timeout_s=2.45`; the local controller continues
on every native frame while it is pending. A result is accepted only if its source life epoch
matches and it is at most 48 observed frames old. A malformed result, timeout, disagreement
between the duplicated high-stakes preference choices, low winning probability, or stale
source is a logged miss and leaves the deterministic preference unchanged. The system-1
budget is large enough for this cadence; failures do not block ticking.

One Astra review is spawned after frame 360 with a 125-second timeout. This is deliberately
advisory because the archived System-2 latency can exceed a minute. Its JSON is parsed with
an object scanner and a closed allow-list; no Astra output is ever used as a raw action or
as a safety threshold. All in-flight work is allowed to be cancelled at episode close.

### Jev state schema

Every request receives a strict JSON object containing:

* `rules`: FIRE press/release, inertia/NOOP, mode-0 hyperspace, orientation increments,
  and the fact that unknown enemy coverage is not absence;
* `snapshot`: absolute observation frame, lives, native score, player presence, derived
  danger/impact buckets, motion, asteroid count, and observed missile count;
* `opportunities`: at most four code-generated rocks described semantically as
  `precontact|crossing|noncrossing`, `clean|marginal`, `aligned|short|long`, and
  `known|uncertain`, with no request for Jev arithmetic;
* `learning`: whether projectile calibration is established, whether score is recent,
  whether a reward-bearing heading is active, and a coarse rate-strength bucket;
* `candidates`, `local_override`, and the durable Astra `plan`.

The four questions are independent and share that state:

* `focus_a` and reversed-order `focus_b` are a `Choice` over
  `reward_lock`, `intercept`, `small_first`, `sweep`. Both must select the same option;
  the minimum winning probability is 0.40 and the accepted preference lasts 60 frames.
  A recent score-bearing preference is only applied in the clear/distant regime.
* `risk` is a `Choice` over already-derived `clear`, `watch`, `immediate`, `critical`,
  and `unknown`. It is diagnostic only; local code owns all emergency actions.
* `lock` is a `Noul` asking whether explicitly supplied recent reward evidence supports a
  short dwell. Its probability is used only to set the `short`/`ignore` offensive preference
  (`>=0.82` to enable, `<=0.25` to disable), never to authorize hyperspace.

The answer-to-action bridge is deterministic. `RewardTurret._rank_shots` enumerates
headings, uses measured missile vectors when available, scores short moving-rock
intersections, then adds a small online reward-bearing bonus. `choose` first checks a
local defense gate, then revalidates a one-pulse escape, then turns one native frame toward
the selected heading and emits a press/release FIRE pulse subject to observed missile
slots and a five/six-frame cadence. It never holds a model-selected action open. A missing
or low-confidence model answer falls through to the geometric threat/value ranking and the
safe coast/fire fallback.

## Why this is a final-generation careful change

The archive already established that native-frame ownership, sparse FIRE, and cold gun
calibration are valuable; it also established that more rescue pulses, earlier patrols,
and coupled thrust are not reliable. 0022 reached the horizon in every episode, so another
survival rewrite is a risky regression. This candidate changes one untested link—the
controller's absence of outcome feedback—while preserving the best candidate's overall
survival shape and containing every new preference by freshness, confidence, danger, and
life-epoch checks.
