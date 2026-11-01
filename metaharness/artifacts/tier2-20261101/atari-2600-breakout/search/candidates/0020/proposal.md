# Candidate 0020 — late side-bounce continuity lock

## Parent and diagnosis

Direct parent: `candidates/0016/`; strongest control reference: `candidates/0017/`
(mean 420.167), with `0014` (411.5) and `0015` (420.167) as supporting
references. I retained the 0017 temporal flight tracker, latched overlap
interval, leased motor, measured paddle motion, asynchronous model bridge, and
defensive episode plumbing. The new experiment is deliberately isolated to the
collision transition where the archive has concrete evidence of a late miss.

`0017` was already real-time safe: its six episodes had only 1–3 stale ticks
and no System-1 failures. Nevertheless, seed 3 lost its last life at frame
9412 with score 354. The loss window is diagnostic:

* after a brick at frame 9377, the ball was `(91,79)` with decoded
  `(dx,dy)=(2,4)` and the paddle was at `x=73`;
* the controller rebuilt contracts at frames 9382–9402, with landing values
  alternating approximately `140,147,141,149,...`;
* it locked the return at frame 9404, then called a horizontal sign change at
  frame 9408 `side_bounce`, cleared its velocity history, and reset the
  contract;
* it immediately formed a new `eta=0.67` contract from the post-bounce sample,
  posted `NOOP, NOOP, RIGHT, RIGHT`, and lost the ball at frame 9412.

`0015` seed 5 shows the same failure shape: landing estimates oscillate around
141/147 through frames 9204–9229, a side-bounce reset occurs at frame 9230
with only 0.67 predicted frames left, and the ball is lost at frame 9234.
`0014` also rebuilds point-like targets during its final side-wall approaches.
The evidence points to a transition/actuator failure, not late Jev or Astra:
waiting for a new post-bounce estimate is too late.

`0016` explored a different structural direction—uncertainty intersection,
frequent corridor replanning, and endpoint shaping—and fell to mean 312.333.
This candidate does not repeat that geometry or delegate paddle arithmetic to a
model.

## Hypothesis

A horizontal side-wall bounce near the paddle is a coordinate-frame transition,
not authorization to throw away a viable catch corridor. The reflected
projection and the existing paddle overlap set already contain the useful
information. Replacing them with a fresh estimate after a near-zero-time wall
bounce causes a direction reset exactly when the paddle cannot travel.

The controller therefore distinguishes two side-bounce cases in code:

* **early side bounce:** it genuinely changes the future landing, so the
  existing quarantine path resets velocity history and replans after clean
  samples;
* **late side bounce:** the current corridor is locked or its prior ETA is at
  most eight frames, the new vertical motion is still descending, and a ball is
  present. The corridor is retained, widened by only two pixels at each safe
  edge, and kept locked through the transition. A paddle bounce, brick event,
  life loss, serve, wall transition, or ball loss still invalidates it.

This is a targeted continuity lock, not a hard-coded paddle angle or a new
physics table. It changes no normal landing projection and leaves the proven
interval motor in charge of every action.

Confirmation would be `side_bounce_continuity` events in the old loss windows,
no immediate post-bounce contract reset, fewer late direction reversals, and a
higher score or later loss than 0017 without stale ticks or errors. Strong
confirmation would be clearing the few remaining first-wall bricks in one of
the 0017 episodes that ended at 426–430 or below. Refutation would be a miss
while the retained corridor is physically overlapped, an early miss caused by
retaining a corridor through an upward flight, or no score improvement despite
removing the reset. Those outcomes would implicate paddle collision timing or
RAM geometry rather than model authority.

## Code-owned controller and fallback

The loop posts a legal action before waiting for the next native tick. It uses
only the public JSON rectangles, decoded velocity/displacement history, and
reflected side-wall bounds. The existing parent controller:

1. recognizes serve/life/brick/paddle/side-wall boundaries without treating a
   missing ball as an episode reset;
2. averages short clean displacement history for diagnostics while using the
   current decoded velocity for responsive projection;
3. computes a complete paddle-left overlap interval after the ball leaves the
   brick band;
4. latches that interval, locks it in the last four frames, and drives it with
   short directional leases and a measured one-pulse brake;
5. now retains only a **late descending** interval across a side bounce, with a
   small edge expansion. Early side bounces still use quarantine and replan.

The deterministic fallback is always available: FIRE while
`serve_required`, a broad home interval while ascending or uncertain, the
latched reflected overlap interval during descent, and a cooled counter-pulse
only when measured paddle drift would leave the interval. A timeout,
malformed answer, contradictory answer, or missing model cannot block the
emulator or generate an illegal action. No model emits `NOOP`, `FIRE`,
`LEFT`, or `RIGHT` for a frame.

## Astra role

Astra is launched asynchronously once a live ball is visible. Its short prompt
contains the current derived flight/urgency, speed regime, interval width and
landing lane, outer-edge bitmap buckets, wall progress, lives, and rally
contract. It returns an untrusted `PRESERVE`, `LEFT`, or `RIGHT` home-anchor
preference. The code gates that preference by plentiful lives, normal/known
speed, a non-imminent phase, and edge occupancy; it can affect only the broad
home interval during an upward flight. It cannot alter a catch interval, the
continuity lock, or the motor. A timeout or unrecognized prose is ignored.

## Jev state, questions, thresholds, and bridge

Jev is requested asynchronously at a serve, clean paddle bounce, life loss,
wall transition, or the other parent-defined strategic boundaries, with a
36-frame cooldown. It receives one strict JSON state of derived facts, not the
raw six-by-18 grid and not arithmetic:

```text
{
  frame, event, lives, score, serve_required,
  phase, urgency, speed, estimate, catch_interval,
  landing_lane, paddle_motion, eta,
  rally_contract, astra_plan,
  left_edge, right_edge, left_count, right_count,
  bricks_remaining, wall_index,
  contract_epoch, rally_version,
  side_bounce_hold, side_bounce_count,
  fallback_action
}
```

The two batched `Choice` questions, `contract_a` and `contract_b`, use the same
four options in rotated/reversed order:

* `PRESERVE`: centered, catch-first rally;
* `LEFT` / `RIGHT`: an early, plentiful-life home-anchor preference only;
* `NO_CHANGE`: retain the current contract and let code act.

The bridge accepts only matching choices whose weaker selected-option
probability is at least `0.55`, whose rally version is current, and whose
safety facts still hold. Low lives, fast or unsettled flight, a locked/near
return, or an open requested edge forces preservation. The answer can never
change the continuity decision or choose a frame action. Late Jev results are
ignored while the code motor continues.

The new `side_bounce_hold` and `side_bounce_count` fields make the tested
transition visible to Jev without asking it to calculate whether retention is
safe; that decision remains an exact code predicate. A side bounce does not
change the strategic rally version by itself, while a life or wall transition
does invalidate old model advice.

## Revision trigger

If late continuity improves survival but the wall still stalls, the next Astra
interface should propose a verified, bounded contact-section or tunnel trial
using bitmap progress. If retained corridors are overlapped yet misses persist,
remove the continuity exception and test paddle collision timing or a small
verified transition search. If early side bounces cause damage, tighten the
late predicate rather than increasing model authority.
