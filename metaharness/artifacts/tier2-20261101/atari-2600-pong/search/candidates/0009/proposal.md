# Candidate 0009: actuator-aware mixed returns with a measured contact feature

## Parents and concrete diagnosis

Parents: `candidates/0006/` and `candidates/0008/`.  The best reference is 0006;
0008 is the parent that actually exercised the button path.

0006 supplies the mechanism that is already strongly supported: code-owned reflected
intercept control, visible-band/settling-aware braking, and an occasional small edge
return.  It returned at least 20 points in all three training episodes (`20, 20, 21`),
with only 1, 2, and 0 stale ticks.  Its remaining weakness was a single long `20:0`
rally on seed 1 that reached the 53,999-tick deadline.  Its Astra responses explicitly
kept `use_feature:false`, and every recorded Jev feature judgment was `NO`, so the
feature mechanism was not tested.

0008 provides useful robustness evidence but not a winning default: it ran a local
frame loop and did produce four, three, and twenty-two contact-timed pulses on the
three traces.  However, its always-attack/always-feature fallback scored only `1, 2,
20`; the seed-0 trace ended `2:1` and seed 1 `2:0` at the deadline.  Thus a feature
pulse is legal and can coexist with a win, but making it the first-return/default
policy is refuted.  The traces also show that Jev's `USE` answers were available near
contacts, so absence of pulses was not the reason for 0008's poor episodes.

## Hypothesis

0006's mixed edge schedule is the right collision controller, but a permanently
feature-disabled policy can leave a particularly favorable robot/configuration in a
long draw.  A **single, cautious Speed/WHAMMY trial after several clean right contacts**
may convert that draw without paying 0008's repeated-pulse cost.  The key safety
hypothesis is that FIRE should be an infrequent experimental return, not a property of
every attacking return: require one ordinary contact first, then one Jev-authorized
pulse per rally, and put the feature on only an already-overlapped contact selected by
0006's visible-band servo.  A negative point after a feature trial disables the feature
for a short recovery period; ordinary mixed edge control remains active.

This is a careful crossover rather than a rewrite: 0006's actuator-aware control and
mixed/edge schedule stay in charge of every frame.  0008 contributes only the
contact-timed, variation-gated, one-pulse bridge and its freshness checks.

## What changed

* Keep 0006's local reflected projection, wall-sign correction, paddle velocity/NOOP
  braking, visible collision band, score/serve phase boundaries, and mixed edge fallback.
* Add a feature state machine with an explicit delayed-trial state, one pulse per
  rally, a post-miss cooldown, and a **contact-count gate**.  FIRE cannot occur during
  automatic serve delay, on an occluded/unknown trajectory, outside Robot Pong 1/2,
  or without current ball/paddle overlap.
* Ask Jev for two reordered authorization Choices, but ask for a future contact
  reservation rather than a raw action.  Code rechecks the latest phase, approach,
  ETA, x-window, paddle overlap, rally generation, and trial state before emitting
  FIRE.  A late, malformed, disagreeing, or low-probability answer is harmless.
* Keep Astra as a slow policy author.  Its initial plan remains 0006's conservative
  mixed schedule and feature disabled, so an arbitrary/late Astra answer cannot turn a
  stable controller into 0008's risky always-feature policy.  The code arms exactly
  one trial only after eight verified right contacts without a point; Astra can tune
  the surrounding edge schedule, but cannot put FIRE on the critical loop or arm an
  early pulse.
* On a feature-related negative point, use a short feature cooldown and return to the
  established mixed/edge policy.  A successful point does not authorize repeated
  pulses automatically; the next rally must pass the same gate.

## System-1 state schema and questions

Each request is strict JSON and contains only decision-relevant derived facts:

* `scenario`: mode, difficulty, variation, feature kind, and right-paddle nominal size;
* `score` and `observation`: reward counters, frame, phase, serve timer, and rally id;
* `ball`: visible/short-gap position, signed velocity, approach, reliability, and an
  ETA bucket computed by code;
* `right_paddle`: center, visible collision band, measured/settled position, and the
  code direction;
* `control`: safe and edge predicted contact, target relation, contact window,
  feature legality, feature preapproval, current contact count, and one-pulse/cooldown
  flags; and
* a short list of recent point/feature evidence.

The batched questions are:

1. `feature_a` and `feature_b` independently choose `USE` or `NO` for
   the **next** right-paddle contact.  Their option orders are reversed to reduce
   first-option bias.  `USE_FEATURE` means reserve one legal pulse; it never means
   press FIRE on the request snapshot.
2. `style_a` and `style_b` retain 0006's typed recognition of the supplied center/edge
   return request.  They are useful for identifying an unsafe edge request, but Jev
   never computes the trajectory or raw action.

A feature vote is accepted only if both answers agree, each selected option has
probability at least `.62`, and the request is from the current rally and contact
horizon.  A style hint requires agreement at `.56`; code arithmetic remains primary.
A vote expires at the projected contact plus a small margin.  The code never treats
Jev confidence alone as probability and handles missing confidence/stand-in answers
as no vote.

## Astra revision trigger and fallback

Astra receives the public game description, compact observation, current mixed policy,
recent point records, feature-use records, and reason for revision.  It may return a
short JSON policy with `mode`, edge schedule, braking, attack cadence, and an advisory
`use_feature`; all values are clamped.  Initial malformed output leaves the tested
0006 fallback.  Code—not Astra—arms the one feature trial after eight verified right
contacts with no point; an Astra call is allowed to revise the surrounding policy but
never becomes a frame-control dependency.  A replan starts after eight verified right
contacts with no point or four point outcomes.

If Jev, Astra, or a future fails, the controller still posts one legal action before
waiting for each tick.  During serve delay and uncertain/away travel it covers center;
near a reliable rightward contact it uses the actuator-aware visible band.  System-1
and System-2 calls are spawned, never awaited by frame control, and all results are
checked for freshness.  The loop catches normal timeout/budget/episode/model errors
and stops when `ctx.done` or the game ends.

## Evidence that would confirm or refute the hypothesis

Confirmation would preserve 0006's near-zero stale rate and at least its `20,20,21`
return profile while converting the long scoreless rally, with a small number of
`contact_feature_pulse` events (not a pulse on every contact).  A useful trace would
show the pulse after clean ordinary contacts, at a live overlap, followed by either a
positive point or a safe return; a negative trial should be followed by the documented
cooldown.

Refutation would be repeated clean feature trials followed by negative rewards, an
increase in opponent points relative to 0006, or no score improvement despite a
scoreless rally in which the feature gate really fires.  If the score remains at least
as strong with no accepted feature votes, the feature is unnecessary and the next
iteration should stay with 0006's mixed/edge controller.  If pulses occur outside the
contact/variation gates, the bridge is incorrect even if the score improves.
