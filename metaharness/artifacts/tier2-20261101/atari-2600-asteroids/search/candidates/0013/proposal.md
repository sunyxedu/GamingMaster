# Candidate 0013 — bounded verified hyperspace retries

## Parent and hypothesis

Parent: `candidates/0011/` (the best candidate, mean score 10531.7). I kept its
frame-local five-lane forecast, urgent relocation pulse, pulsed FIRE, asynchronous
Jev/Astra supervision, freshness checks, and clock ownership.

The parent’s remaining failure is visible in its worst trace, not a stale-tick
problem. In `trace_seed8.jsonl`, it emits `defense` at frame **387** with
`threat_time: 2`, and the ship changes from 4 to 3 lives at frame **410**. It
later spends its one defense at frame **4631**. Near the end, the model state at
frame **4988** says `danger: "immediate"`, five rocks, and
`controller.defense_used: true`; the game ends at tick **4902** / frame **4994**
with `lives: 0` and score **4520**, after repeated `UP` pulses could not clear the
contact. The same pattern is earlier and more compact in seed 10: a defense at
frame **279** (`threat_time: 2`) is followed by a life loss at **304**, and a
defense at **7063** is followed by another loss at **7094**. Seed 4 also loses a
ship at frame **3690** in an immediate window after the one-shot defense gate has
been used.

These traces justify testing the parent’s resource assumption: a hyperspace pulse
is an escape attempt, not a disclosed once-per-life cartridge resource. A failed
or unlucky landing can leave a fresh immediate contact, while the parent’s
`defense_used` boolean permanently disables DOWN for that life. The hypothesis is
that allowing a **small, observable retry budget** will rescue some of those
contacts without turning defense into a held action or disturbing the successful
normal firing/lane policy.

This is a single targeted bridge change. It does not add proactive movement,
change the forecast, or retune aiming and FIRE. The earlier candidate 0010 also
explored retries, but it changed the whole movement architecture; this candidate
isolates retries on the proven 0011 controller.

## One change

`defense_used` is retained only as a diagnostic flag. The action gate now tracks
`defense_count` and `last_defense` per life. It permits at most **three** one-frame
`DOWN` pulses in a life, requires **16 observed frames** between pulses, and still
requires the current local swept-box forecast to be immediate (`threat_time <= 2`,
or a fresh accepted Jev emergency hint under the same local immediate gate). The
counter and timer reset on a native life change. After every pulse the existing
controller clears thrust state and clears the model hint; it does not hold DOWN.
The plan value `defense: "never"` remains an explicit Astra veto.

All other parent constants and scheduling are unchanged. In particular, the
urgent lane pulse remains bounded by its six-frame gate, normal thrust retains its
cooldown, FIRE remains a press/release cadence, and stale or malformed model
answers cannot create an action.

## Confirmation or refutation

The hypothesis is supported if retries are rare and occur only after a current
immediate forecast, and if seed-8/seed-10-style life-loss frames move later or
disappear while the parent’s score-producing fire loop and near-zero stale ticks
remain intact. A strong confirming trace has a second `DOWN`, a resulting player
motion discontinuity/relocation, and continued life.

It is refuted if retries commonly land into another contact, cause earlier losses,
interrupt useful firing enough to lower score, or do not improve survival over
0011. The diagnostic events are `defense_count`, `last_defense`, immediate
`threat_time`, life-change frames, and score per life; official selection remains
unclipped ALE reward only.

## System-1 state, questions, thresholds, and bridge

The Jev state is the parent’s compact, derived JSON. It contains:

* `rules`: mode 0; FIRE is level press-then-release; NOOP does not brake; DOWN is
  mode-0 hyperspace; LEFT/RIGHT orientation semantics; enemy coverage is unknown,
  not absent.
* `snapshot`: absolute frame, score, lives, player presence and left/middle/right
  position buckets, still/moving/unknown motion, code-derived danger and impact
  buckets, asteroid count, and missile count.
* `lanes`: the five code-generated lane IDs with safe/watch/hot side labels.
* `target`: one code-selected rock summary (slot, size, bearing, known/uncertain
  motion, distance bucket).
* `controller`: tactical posture, lane goal, `defense_count`,
  `defense_retry_ready`, and `frame_local: true`.
* `plan`: Astra’s allow-listed aim, patrol, risk, defense, and fire preferences.

One request batches six independent judgments:

1. `danger_a` and `danger_b` are reversed-order Choices over
   `clear|distant|watch|immediate|unknown`.
2. `posture_a` and `posture_b` are reversed-order Choices over
   `focus|sweep_left|sweep_right|hold|avoid`.
3. `lane` is a Choice over `stay` and only the current five precomputed lane IDs.
4. `defense` is a Noul asking whether one bounded retry should be considered given
   the supplied immediate-contact and retry facts.

Jev never calculates distances, chooses a raw joystick action, or owns timing.
Danger advice is accepted only when the two reversed Choices agree and the
selected probability is at least **0.45** for `immediate` (**0.16** otherwise).
Posture and lane advice need agreement/legality and probability **0.16**. The
Noul can set a short-lived emergency hint only at yes probability **0.78**. An
answer is discarded after **48 frames**, on a life-epoch change, or if it refers
to a future frame. These are unchanged parent freshness/confidence gates except
that the defense question describes the retry budget.

The hard action bridge is deterministic. A `DOWN` is legal only when the player
is present, the plan does not say `never`, `defense_count < 3`, at least 16 frames
have elapsed since `last_defense`, and the latest local geometry reports an
impact within four sampled frames. The local first condition means the ordinary
path is `threat_time <= 2`; Jev can support the parent’s existing immediate hint
path but cannot bypass the local gate. If no model answer is usable, the fallback
is unchanged: code-owned lane/aim/fire control, bounded urgent thrust, and at
most the verified retry pulse when its hard conditions hold. Missing player data
posts NOOP. Model latency never stops native ticking.

Astra remains delayed aggregate advice and can only revise the existing
allow-listed plan. It never posts DOWN or any other action; its output is parsed
strictly and ignored if malformed.
