# Candidate 0019 — reversible pulses, not permanent lane reversals

**Mode:** crossover. **Parents:** `candidates/0015/` and `candidates/0016/`.

## Hypothesis

The two useful parent mechanisms operate at different timescales:

* 0015's score and survival came from the uninterrupted `right_fire`/`left_fire`
  sweep, with shield-mask changes occasionally causing an escape.
* 0016 made the sensor more disciplined with temporal projectile evidence, but its
  evidence reviewer never affected control: every episode ended with
  `down_in_shot_field=0`, `jev_promotions=0`, `strong_dodges=0`, and
  `jev_changed_actions=0`.

The failure is therefore not evidence that movement should stop, nor evidence that
one shield chip identifies a bomb. It is a control-timescale problem: 0015 turns
weak, isolated erosion into a full sweep-direction reversal, while 0016 requires
strong corroboration that the decoder never supplies. A weak cue needs a small,
reversible experiment that does not destroy firing coverage.

**Tested hypothesis:** require temporal corroboration for deterministic shield
avoidance, preserve the broad combined-FIRE sweep otherwise, and make an accepted
avoidance a short pulse away from the cue followed by automatic return to the old
sweep phase. Explicit bombs and a coherent descending track remain code-owned
emergencies. Jev may promote one weak cue to the same bounded pulse, but cannot
choose a target, invent a projectile, or alter FIRE.

## Archive evidence this acts on

0015 seed 4 scored 145 and lost lives at frames 612 and 932 even though both
signals said `recent_bomb=false`; it had already applied shield escapes at 212,
260, and 468. Its final event still reported `jev_changed_actions=0`. Thus more
permanent reversals were not demonstrated to prevent the blind losses. In its
better seed 3, the repeated early escapes coexisted with 535 points and a final
loss at frame 2068, while the controller continued to report no effective Jev
change.

0016 seed 1 scored 145 with life evidence at 1096 and 1380, and seed 3 scored
625 with life evidence at 1128 and 2496. The temporal ledger did correctly obtain
several `OUTGOING` verdicts, but no `INCOMING` verdict, no projectile escape, and
no changed action. The same traces show the productive baseline plainly:
0016 seed 3 has long regular combined-FIRE runs (about 40 ticks per sweep leg),
while 0015's extra lane reversals break those runs. The shield observations also
show that ordinary outgoing fire erodes masks (for example, occupancy changes
while `player_shot` is active), so a single chip must not be treated as a bomb.

These facts refute neither the possibility of hidden incoming fire nor the value
of shield evidence; they motivate a bounded, corroborated intervention that can
be audited rather than another permanent lane memory.

## What changed

1. **Productive baseline retained.** Every normal action is `left_fire` or
   `right_fire`; the controller uses a broad 36..118 screen-x sweep. It does not
   aim at model-selected targets or suppress FIRE because a shot is active.
2. **Temporal evidence ledger retained in a smaller form.** Public bomb and
   player-shot points are associated by nearby x and physically plausible y
   motion. A new track is marked as an own launch only at the muzzle while FIRE
   is posted. An explicit bomb or two downward increments without an own-launch
   match creates a strong code-owned cue. An upward track is never made hostile.
3. **Shield corroboration gate.** The mask decoder records only top-entry pixels
   not explained by a recent own launch. A single `shield_chip` is reviewable but
   not automatically actionable. A second nearby unexplained chip within a
   bounded 80-frame window creates `shield_repeat`; this can cause one pulse.
   Bulk clear/refill and life/wave discontinuities are discarded.
4. **Reversible pulse controller.** A strong incoming cue can trigger an
   emergency pulse. A repeated shield lane can trigger one short pulse away from
   its x-coordinate. The pulse lasts 36 emulator frames by default, respects the
   current corridor and live path-safety check, then resumes the prior sweep
   direction. It cannot become a permanent wall or accumulate one action per
   observation. After a life loss, the phase reverses for a short central
   recovery corridor, without storing the respawn x as a death lane. Late waves
   modestly tighten the corridor and keep pulse horizons short.
5. **No early, ineffective Astra call.** The archive's early Astra calls were
   canceled at game over (0015's call lasted about 51 seconds; 0016's about 51
   seconds) and produced zero effective updates. This harness defers its one
   possible System-2 call until at least 720 fresh actions and a late/low-life
   state. If that unusually long episode remains alive, Astra chooses only
   `SHORT`, `MEDIUM`, or `LONG` pulse duration (24/36/56 frames). Garbage,
   timeout, cancellation, or a late answer leaves the compiled policy unchanged.
   This is an escalation for genuinely late-wave evidence, not a blocker on the
   actuator.

## System-1 state and questions

Each request is a compact JSON object with schema
`space_invaders_reversible_pulse_v1`:

* `frame`, `lives`, `wave`, categorical player lane, and `phase` (`SWEEP`,
  `PULSE`, or `RECOVERY`);
* formation phase (`MID_OR_OPENING` or `LATE`) and no raw invader list;
* `cue.kind` (`single shield chip`, `repeated shield lane`, `descending track`,
  `explicit bomb`, or `none`), qualitative cue age, side relative to the
  cannon, strength, and whether the cue has an own-shot explanation;
* `paths.sweep`, code-derived `safe_left`/`safe_right`, qualitative left/right
  risk, and the rule that direct collision prediction has priority;
* explicit rules: empty bombs are not safety, a chip is not a surviving bomb,
  Jev cannot invent objects, and every resulting action still includes FIRE.

A request batches two independent `Choice` questions, `judgment_a` and
`judgment_b`, with the same four options in reverse order:

* `PULSE_LEFT` — only a safe short pulse when the supplied cue supports moving
  left;
* `PULSE_RIGHT` — the symmetric safe pulse;
* `KEEP_SWEEP` — weak, contradictory, or non-actionable evidence;
* `UNSURE` — no justified pulse choice.

The wording gives structured inclusion/exclusion descriptions and explicitly
forbids coordinate arithmetic, target selection, projectile invention, or action
mapping. The two answers must agree, the selected option probability must be at
least **0.60** in both, and any supplied confidence must be at least **0.25**;
missing confidence is not treated as certainty. A response is accepted only if
its epoch and cue identity are still current, it is at most 32 emulator frames
old, and the cue has not been consumed. Execution rechecks the current path,
requires the proposed side to be the code-computed away side and below the high
risk threshold, and expires advice after 20 frames. Disagreement, low
probability, stale identity, malformed stand-in output, or timeout is ignored.

There is at most one asynchronous Jev request. The main loop never waits for it;
while it is in flight the code-owned sweep or emergency fallback posts the next
combined-FIRE action. The 0.85-second timeout is above the archived Jev tail and
its failure is non-fatal.

## Answer-to-action bridge and fallback

The priority order on every fresh observation is:

1. **Explicit/coherent incoming evidence:** choose the safer legal horizontal
   direction with a current collision check and start a short pulse.
2. **Corroborated shield lane:** if the repeated cue is relevant and its away
   side is safe, consume it once and pulse for the current duration.
3. **Fresh, agreed Jev promotion:** apply only the same bounded pulse and only
   when it matches the code-computed away side. Jev cannot override a direct
   collision choice.
4. **Normal sweep:** reverse at the code-owned corridor limits and continue
   combined FIRE.

No action waits for a model. The pulse is bounded, the cue is consumed once, and
life/wave resets invalidate tracks, cues, and advice. If `ctx.decide` fails or
returns arbitrary preflight text, the deterministic path remains legal. If Astra
returns arbitrary prose, the duration remains 36 frames.

## Confirmation and refutation

Confirmation would require more than a score increase: traces should show
`shield_repeat` observations, pulses that are shorter than a full sweep leg,
continued combined-FIRE throughput, and either fewer blind life losses or longer
survival at comparable kill progress. A stronger confirmation is a direct or
coherent descending cue for which the selected pulse has lower current path risk
and is followed by survival through the predicted contact interval. Jev
promotion should be rare and auditable, not a source of random reversals.

Refutation would be any of: no repeated cues and no direct trajectories ever
appearing; pulses firing on self-inflicted shield damage; repeated pulses
reducing kill rate without improving survival; or a continued unchanged death
pattern after the controller has made genuine pulse interventions. If the score is
high while pulse counters remain zero, that confirms only the inherited sweep,
not the evidence hypothesis. A wave clear and retained lives remain the main
headroom beyond the parents' all-lives-lost, zero-wave outcomes.
