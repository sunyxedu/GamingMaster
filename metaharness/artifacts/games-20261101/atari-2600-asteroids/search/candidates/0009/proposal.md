# Candidate 0009 — lane-first escape arbiter

## Parents and concrete diagnosis

Parents: `candidates/0007/` and `candidates/0008/`.

`0007` is the strongest parent (mean 5343.3), and its useful mechanism is clear:
it owns every native frame, fires with short press/release pulses, turns locally toward
code-ranked targets, and uses very little thrust.  Its traces show that this is much
better than either a stale model action or broad roaming.  For example, the best
seed-4 run reached frame 12926 and score 9610 with only 81 `UP` ticks, while most
control was `NOOP`, turn, and `FIRE`.

Its remaining failure is also concrete.  In seed 10, the controller labeled an
impact `immediate` at frame 279 with TTC 2 and posted `DOWN`; the first ship was lost
at frame 304.  In seed 4, it posted defenses at frames 2677, 3465, and 3769, yet
lost ships at 3210, 3482, and 3960.  Thus a last-frame hyperspace pulse is not a
reliable collision solution.  More importantly, `0007` evaluates five fixed
horizontal lanes at the current vertical coordinate.  When the dangerous crossing
is vertical or every horizontal lane is compromised, its lane choice can merely
turn the ship in place until the contact arrives.  Its best run still ended in
`game_done` at frame 13002 rather than the 18000-frame budget.

`0008` supplies the missing structural idea: score a set of swept escape sectors,
not only x-lanes, and make movement proactive while there is still clearance.  But
its implementation applies that idea too broadly.  The worst seed-4 run spent 1760
thrust/diagonal-thrust ticks and scored only 680; its action counts include 802 `UP`,
509 `UPRIGHT`, and 449 `UPLEFT`.  The lessons explicitly reject equating proactive
movement with predictive relocation.  It moved often in fields where `0007`'s low-
duty turret was scoring, and its four once-per-life defenses did not compensate.

## Hypothesis

The plateau is caused by a missing **cross-axis escape only at the failure boundary**,
not by insufficient model calls.  `0007`'s lane-first, low-thrust turret should be
preserved.  When a current observation predicts a collision, the controller should
first compare its horizontal alternatives.  It should use `0008`-style sector
projection only when the horizontal alternatives are all hot, the current lane is
already tight, or persistent drift makes an up/down cardinal sector materially
safer.  The selected movement must be a two- or three-frame pulse, rechecked every
frame, with a cooldown—not a periodic patrol.  Hyperspace should be a last resort
when an immediate current-nose thrust forecast cannot improve an overlapping contact,
rather than the default response at TTC 0–2.

This is a structural crossover: two-resolution safety planning plus an explicit
movement arbiter, while retaining the parent’s demonstrated firing mechanism.  It
uses only current decoded observations and does not identify or memorize launch
seeds.

The hypothesis is supported if the candidate keeps near-zero stale ticks, preserves
`0007`-level score on calm seeds, and extends the first/final life-loss frames on
seed-8/10-like crossings without approaching `0008`'s thrust count.  It is refuted
if the cross-axis branch is rarely reached and deaths remain at the same TTC, if it
causes high thrust counts or edge commitment, or if suppressing early hyperspace
reduces survival without increasing score.  Useful diagnostics are `escape_start`
kind (`horizontal` versus `sector`), warning streak, current/best lane cost,
sector clearance, pulse count, defense count, life-loss frames, and score per life.
The official score remains the unclipped ALE reward only.

## What changed

* `EscapeArbiter` maintains wrap-aware relative trajectories, approximate swept
  rectangle clearance, and uncertainty inflation for null/discontinuous velocity.
  It treats source slots as current observations, not permanent identities.
* A lane forecast evaluates six central x positions.  A separate cardinal forecast
  evaluates up/left/down/right with a short thrust pulse and includes both a short
  contact horizon and a longer drift horizon.  The arbiter chooses the lane result
  when it is materially better; it chooses the cardinal result only when lanes are
  unsafe or the cross-axis gain is real.
* Normal `aim`/`sweep` control remains a code-owned target and FIRE cadence.  Movement
  rotates toward a cardinal code, then applies at most two normal thrust frames or
  three urgent frames.  The plan expires quickly and the next observation can cancel
  it on a discontinuity, life change, or changed risk.  There is no timer that moves
  the ship merely because many rocks exist.
* The emergency branch is deterministic and cannot be vetoed by Jev.  It first tests
  whether thrust along the current nose improves the near forecast.  A single `DOWN`
  is allowed only for a critical overlap/near-contact for which that immediate escape
  does not improve the forecast, at most once per life.  `DOWN` is never treated as a
  brake.
* Astra is a delayed policy reviewer after an aggregate observation window.  Its
  allow-listed choices are `hybrid|turret`, `conservative|early` escape timing,
  normal/spaced fire, and hyperspace-last/never.  These only adjust future local
  thresholds; Astra cannot post actions or make a safe gate unsafe.

## Jev state and questions

Each request is strict JSON and contains:

* `rules`: mode 0, FIRE press/release semantics, NOOP inertia, DOWN hyperspace,
  orientation direction, and the fact that unknown enemy coverage is not absence;
* `snapshot`: absolute observation frame, lives, score, player presence, position and
  speed buckets, danger bucket, clearance bucket, TTC bucket, asteroid size counts,
  missile count, warning streak, and discontinuity flag;
* `lanes`: each code-generated lane id with relation and clearance-risk bucket;
* `sectors`: the four code-generated cardinal sectors with direction and risk bucket;
* `target`: a code-selected target’s size, bearing, distance, and velocity-known flag;
* `candidates`: `aim`, `sweep_left`, `sweep_right`, `hold`, and, only when locally
  generated, one bounded escape or `defend` candidate;
* `controller`: tactical posture, active-escape flag, defense-used flag, policy, and
  frame-local ownership.

One request batches four independent typed judgments:

1. `choice_a`: choose a candidate from the current code-generated set;
2. `choice_b`: the same choice with reversed option order, reducing Jev’s listed-first
   bias;
3. `risk`: classify `clear`, `watch`, `caution`, `emergency`, or `unknown` from
   already-bucketed facts;
4. `defense`: a Noul asking whether the locally present critical defense candidate is
   worth considering.

The bridge requires the two Choices to agree.  A selected escape or defense needs
minimum probability **0.42** in both answers; cheap aiming/posture choices need
**0.17**.  A valid answer is tagged with its source frame and life epoch, discarded
if older than 46 observed frames or after a life transition, and never creates an
unlisted action.  A Jev escape is only a hint: the local lane/sector safety tests,
movement cooldown, pulse limit, and current observation remain authoritative.  A
Noul cannot create a defense outside the deterministic critical gate.  Unknown,
malformed, low-confidence, disagreeing, or late answers leave the local policy
unchanged.

The request is started every 30 observed frames when no earlier request is pending;
its 2.4-second timeout is longer than the measured Jev tail while the local loop
continues ticking.  The fallback is always executable: target-aware turn/fire on a
clear field, lane-first bounded movement for a current warning, sector movement when
all lanes are unsafe, and release/NOOP when the player is absent.

## Astra timing and revision rule

Astra receives one aggregate review after frame 520 (if the episode and System-2
budget remain).  The prompt includes score, lives, warning persistence, clearance,
impact, lane gain, cross-axis gain, active escape, and current policy.  A response is
parsed as an allow-listed JSON object; arbitrary prose or stand-in output is ignored.
The initial policy is valid without Astra, so a slow, timed-out, or cancelled answer
cannot stop the clock.  In a longer protocol, a later review would be warranted only
after a life-loss cluster or a persistent mismatch between lane and sector forecasts;
this candidate deliberately limits itself to one review so the intervention is
measurable and does not spend calls on an unresponsive action channel.

Astra should revise the interface only if the aggregate evidence shows a systematic
mismatch: repeated losses while horizontal lane cost is high but sector gain is
positive suggests earlier cross-axis escape; high thrust with no life extension
suggests `turret`/`conservative`; good survival but low shot rate suggests normal
rather than spaced fire.  It is not asked to perform frame-by-frame arithmetic.

## Tier-2 fallback and accounting

The harness posts an initial local action immediately, then waits for each native
`wait_for_tick()`, observes the resulting immutable frame, consumes any finished
model result, and posts exactly one fresh action.  Model latency therefore never
turns into a held thrust or stale defense.  Model exceptions, malformed JSON, and
arbitrary preflight answers are treated as advisory misses; budget, episode-over,
and model-unavailable conditions return cleanly under the contract.  No custom
background loop remains after return, and the game’s own score/lives/end conditions
remain authoritative.
