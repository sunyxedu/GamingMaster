# Candidate 0007 — temporal hazard loop with recovery-column pulses

## Parents and diagnosis

Parents: `candidates/0003/` and `candidates/0005/`.

I retain the mechanism that made `0003` the best candidate: a directional Jev
answer is an edge, consumed for one emulator tick and immediately released to
`NOOP`. The trace evidence is strong: `0003` changed the parent's 1,239-frame
held-`UP` run into isolated one-tick pulses and raised mean score from 6 to 12.
It still never reached a home because every life used the same x column and the
controller did not observe the dangerous transition while Jev was thinking.
For example, in the seed-109 trace the frog reached candidate row 7, its
screen-x moved from 72 to 56, and RAM `0xE7` was `0x80` at tick 905; the next
visible request was about nine emulator frames later and the frog had died.
The road deaths in the other traces occurred at rows 0–3 while the frog stayed
at screen-x 40. The candidate therefore earned forward-hop points without
changing the traffic/river survival problem.

`0005` supplies the other parent mechanism worth keeping: an advisory Astra
supervisor can run out of band while a fast loop keeps the game alive. Its
failure is equally concrete. Jev was asked to judge an unverified row candidate
and repeatedly answered `START`/`NOOP` (seed 109 had an 8,153-tick final NOOP
run); its strong wait veto turned uncertainty into starvation. `0007` keeps
Astra advisory but removes its ability to veto progress, and does not treat the
hypothesis row byte as a complete object decoder.

## Hypothesis

The remaining failure is structural rather than a need for more Astra prose:
`0003` only samples state at Jev-response cadence, so a short-lived river hazard
or a life reset can be missed, and it never changes the column after repeated
same-column road collisions. A controller that (1) observes every live frame,
(2) exposes verified *temporal transitions* rather than opaque RAM to Jev, (3)
uses a one-step lateral recovery experiment after a death, restoring the offset
at the median, and (4) uses an E7 hazard transition to trigger an immediate
forward escape should convert hop credit into river/home progress. The column
change is adaptive to a death and alternates sides; it is not keyed to a seed or
a fixed phase route.

Evidence that would confirm the hypothesis is a trace with one-frame direction
pulses, a lateral pulse after a road death, no long held directional run, and
nonzero `frogs_home` or at least a later river row with fewer deaths. A useful
intermediate confirmation is that the frame loop sees `0xE7` become nonzero and
changes the next pulse rather than waiting through Jev latency. It is refuted
if the recovery pulses cause immediate deaths without extending the maximum
row, or if river hazards remain missed despite frame-by-frame observation; then
the candidate row/E7 hypotheses need ROM calibration rather than more model
policy.

## What changed

* The main loop waits for one tick, observes the new RAM, updates a compact
  temporal tracker, and posts an action for the next tick. It never blocks the
  game on a synchronous Jev or Astra call, so every tick has a fresh legal
  action and a late model answer cannot hold a direction.
* The released-edge bridge from `0003` is structural: after any direction is
  consumed, the next action is `NOOP`, with a short deterministic debounce.
  A late, malformed, failed, or stale answer is discarded.
* After a death, code alternates a single `LEFT`/`RIGHT` recovery pulse on the
  next life, then sends the opposite pulse at the median/river boundary. This
  changes the repeatedly tested road column while preserving the central home
  approach. A fresh life is forced to make eventual `UP` progress; Jev cannot
  recreate `0005`'s indefinite wait.
* Code uses the observed row transition `255 -> 0 -> 1 ...` only as broad phase
  evidence. It exposes `ROAD_OR_MEDIAN`, `RIVER`, and `HOME_APPROACH`, current x
  motion, row motion, score/row progress, life id, and recent action outcomes.
  It also exposes `0xE7` current/rising, `0xE5`, and `0xE3` as unnamed cues. A
  rising/nonzero E7 in the river causes a one-pulse `UP` escape; it is not
  claimed to identify a particular turtle or object.
* One concise Astra call is launched at opening in the background and parsed
  only into a bounded advisory card (`focus`, `rule`, `lateral_rule`). It is
  included in later Jev state but never directly executed and never blocks the
  frame loop. Arbitrary Astra output is harmless.

## Jev state and questions

Every Jev request receives strict JSON, rebuilt from the current observation:

* `rules`: one-hop action semantics and survival priority;
* `time`: tick/frame, timer, ticks since progress, and ticks since a direction;
* `life`: death/reserve counts, life index, last death phase/row, pending
  recovery shift and its direction, and whether center restoration is pending;
* `frog`: candidate row/x/screen-x, broad phase, screen motion, row motion, and
  left-edge/interior/right-edge bucket;
* `progress`: score, homes, level and terminal flag;
* `verified_temporal_evidence`: current event, current/rising E7, E5/E3, home
  counter, and a bounded list of changed RAM addresses;
* the last eight frame summaries, the bounded Astra card, and the one-frame
  control contract.

The request batches three independent typed judgments:

1. `action`: a rotated-order `Choice` over literal `NOOP`, `UP`, `RIGHT`,
   `LEFT`, and `DOWN`, asking for the next single frame;
2. `tactic`: a rotated-order `Choice` over `ADVANCE`, `WAIT`, `SHIFT_LEFT`,
   `SHIFT_RIGHT`, and `RETREAT`, making a lateral/recovery judgment explicit
   instead of relying on Jev to invent a direction from raw bytes;
3. `forward_safe`: a `Noul` about one immediate `UP` hop, conditioned on phase,
   E7 rising, motion and the recent event. It is informative; it is not the
   starvation veto used by `0005`.

A lateral tactic is accepted at probability at least 0.30; otherwise the direct
literal action is used if legal. No confidence value is treated as certainty.
The deterministic bridge then applies these safety/progress gates:

* terminal maps to `NOOP`;
* during the five-frame debounce, `NOOP` wins;
* a rising river E7 or a live carrier-edge bucket has priority over a model
  `NOOP` and selects `UP` or an inward lateral pulse;
* a pending post-death recovery shift and median restoration have priority;
* a model `NOOP` at `START`, or after 22 ticks without score/row progress, is
  replaced by one `UP` pulse;
* otherwise a legal Jev direction is posted for one frame, and all malformed,
  stale (>16 ticks), wrong-life, timed-out, or missing answers fall back to
  `NOOP` (with the start/no-progress exceptions above).

Thus Jev supplies recognition and immediate posture, while code owns exact
freshness, action legality, the one-frame edge, temporal bookkeeping, and the
small adaptive recovery search. Astra revises neither action timing nor these
thresholds; later evidence can justify changing the E7 interpretation or the
recovery geometry.
