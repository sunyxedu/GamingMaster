# Candidate 0012 — use shield erosion as a weak hidden-bomb sensor

Parent: `candidates/0006/` (best prior mean, 280.0); implementation also retains the broad-fire sweep structure validated by 0010.

## Hypothesis

The repeated failure is not stale control or lack of firing throughput.  The
best candidates post fresh actions, hold FIRE, and still lose lives while the
symbolic bomb list is empty.  In 0006 seed 2 the trace records `life_loss` at
frames 508 and 896 with `blind=true`, and the controller records no
`bomb_override`; between those losses it enters repeated zero-distance blocked
routes.  0010 similarly reaches 115 in its worst seed-0 trace and records
`last_bomb_frame=-100000` at both losses.  Thus the bomb decoder is not merely
sparse in time: a controller needs a second observable consequence of enemy
fire.

The public observation includes nine-row shield occupancy masks.  A bomb that
hits a shield can remove occupancy even when the projectile itself is absent
from the sampled JSON.  My test is therefore: compare masks across observations,
attribute erosion to a recent player-shot lane when possible, and treat
unexplained erosion as weak, short-lived evidence for an unsafe horizontal lane.
This is deliberately not a claim that every mask change is a bomb; it is a
sensor-fusion experiment.  The controller keeps the proven broad left/right
sweep and continuous combined FIRE, but temporarily escapes away from the
inferred lane.  Visible bomb collision prediction always outranks this cue,
and ordinary empty-bomb observations never count as evidence of safety.

A further blind death stores its cannon lane as the same weak evidence.  The
signal expires, so a noisy or misattributed shield change cannot permanently
pin the controller.  Astra is used once, off the actuator path, to choose only
the retention duration (`brief`, `normal`, `extended`) after actual evidence.
Invalid Astra output leaves the default duration unchanged.

## What changed

* Added shield-mask differencing using the documented `health_mask` bit layout.
* Added conservative attribution: erosion within 4.5 pixels of a recent shot
  lane is labelled `shot_aligned_erosion`, not enemy evidence.
* Added short-lived lane records for unexplained erosion and decoder-hidden life
  loss.  These records influence only a bounded lateral escape predicate.
* Preserved full-rate `right_fire`/`left_fire` broad sweeps and visible-bomb
  trajectory checks from the stronger structural candidates.
* Added one batched asynchronous Jev request at every 20-frame interval, with
  urgent refresh after erosion/life loss.  Jev can confirm only a code-computed
  left/right escape; it cannot invent coordinates or turn an empty bomb list
  into safety.
* Added one background Astra revision after the first real erosion or blind
  life signal.  It selects the expiry window, not individual actions.

## Confirmation and refutation

Confirmation would show `shield_lane_inferred` events before some otherwise
blind losses, fewer deaths while the cannon is inside or moving toward those
lanes, later survival, and a higher kill/score mean without a large increase in
zero-progress oscillation.  It would be especially persuasive if runs with
many ordinary `shot_aligned_erosion` events do not trigger unnecessary escapes,
while unexplained events do.

Refutation would be unchanged score and death timing despite frequent inferred
lanes, a large score loss from escaping harmless shield damage, or evidence that
mask changes are mostly sprite/decoder noise.  It would also refute the
hypothesis if episodes with no erosion remain the same as 0006 and all gains
come only from the unchanged sweep.

## System-1 interface

Every request carries strict JSON with schema `space_invaders_shield_erosion_v1`:

* `frame`, `lives`, `wave`, categorical player lane, committed sweep, and the
  one-shot rule;
* `signals.shield_change` (`none`, `shot_aligned_erosion`, or
  `unexplained_erosion`), whether a recent player shot explains it, whether a
  lane signal is fresh/aging, and whether a directly predicted bomb is high
  risk;
* `signals.lane_side` and `safe.left`/`safe.right`, all computed in code;
* `safe.fallback_escape`, cover at the current lane, and no raw arithmetic
  task for Jev.

The single request batches two reordered `Choice`s over
`DODGE_LEFT`, `DODGE_RIGHT`, `CONTINUE`, and `UNKNOWN`.  A choice is accepted
only when both answers agree, each assigns at least 0.60 probability to that
choice, confidence (when supplied) is at least 0.25, and the answer is no more
than 36 frames old.  An escape is additionally accepted only if the selected
side is currently code-marked safe and the corresponding lane evidence is
still within 24 frames.  Any timeout, malformed stand-in answer, disagreement,
life/wave change, or unsafe side is ignored.  The deterministic fallback posts
fresh combined FIRE every tick and uses visible-bomb escape, then the recent
shield/death lane, then the committed sweep.

Astra receives a compact evidence record after the first unexplained erosion or
life signal.  Its only allowed output is `{"avoid_window":"brief"}` (or
`normal`/`extended`), parsed defensively.  Defaults are 96, 180, and 300
emulator frames respectively; default is 180.  Astra never blocks an action
and cannot override visible collision safety or firing.

## Evaluation diagnostic

The important trace events are `shield_lane_inferred`, `life_loss_signal`,
`astra_window_updated`, and `shield_controller_summary`.  Compare these with
score jumps, action direction, and subsequent life-loss frames.  The decisive
result is not simply more model calls: it is whether a public side effect
(shield erosion) supplies useful information in the exact episodes where the
prior bomb field was empty.
