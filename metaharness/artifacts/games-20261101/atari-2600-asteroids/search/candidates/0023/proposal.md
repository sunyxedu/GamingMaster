# Candidate 0023 — coupled escape thrust for reachable tranches

## Parent and targeted hypothesis

Parent: `candidates/0018/` (kinematic anchor/threat-first occupancy).

I keep its frame-local clock ownership, short FIRE presses, velocity anchor, wrap-aware
occupancy forecast, revalidation, bounded hyperspace gate, Jev interface, and asynchronous
Astra review.  This is one targeted change to the parent’s escape executor.

The parent’s worst episode is seed 6.  Its trace reports `escape_start` at frames 566,
1178, 1643, 2366, 2967, 2990, 2998, 3102, 4230, 4454, 4543, 4559, 4571,
4578, 4585, 4742, 4791, 5165, 5179, 5207, and 5339, but no escape-pulse
records and only **14** native `UP` plus **3** `UPLEFT` ticks in the entire
5,292-tick episode (the other controls were 2,990 `NOOP`, 951 `LEFT`, 808 `RIGHT`,
293 `FIRE`, 135 `LEFTFIRE`, 94 `RIGHTFIRE`, and 4 `DOWN`; there were no `UPRIGHT`
ticks).  The four local defenses then arrive at frames 1657 (`ttc: 0`), 3149
(`ttc: 0`), 4805 (`ttc: 1`), and 5181 (`ttc: 0`), followed by life epochs at 1678,
4592, 4840, and terminal loss at 5376.  Seed 2 shows the same sparse-displacement
symptom: 34 escape starts but only 5 `UP`, 3 `UPRIGHT`, and 1 `UPLEFT` native ticks,
with defenses at frames 6918 (`ttc: 2`), 7303 (`ttc: 5`), 8671 (`ttc: 0`), 8943
(`ttc: 3`), 10929 (`ttc: 0`), 11758 (`ttc: 3`), and 12145 (`ttc: 1`).

The likely failure is not that the occupancy candidates are never selected.  It is that
`_maneuver_action` spends the whole short tranche doing pure rotation: it only thrusts
when the current heading is already within one orientation code of the reserved heading,
and the maneuver can expire or be invalidated before that happens.  Thus an apparently
reachable escape is a reservation with no displacement, and hyperspace is requested only
when the contact is already overlapping.  The isolated test is: make an active escape
tranche commit one bounded thrust impulse while it is still turning, before a later
observation can invalidate it, while retaining the parent’s per-frame path
revalidation and never allowing a model to create thrust.

## Change

In `_maneuver_action`, an escape no longer requires exact heading alignment for every
thrust pulse.  On the first active frame it uses the legal `UPLEFT`/`UPRIGHT`
combination toward the desired heading for one bounded pulse even while turning, then
waits five frames as before; subsequent pulses remain subject to the same local
revalidation and tranche expiry.  The existing forecast, remaining-pulse count,
cancellation, life/discontinuity reset, and local defense check remain unchanged.  The
action is still one native frame.  This tests whether *executing* the already selected
escape before a transient revalidation cancels it, rather than adding more labels,
prevents the ttc-zero cluster.  The normal turret and anchor paths are unchanged.

Evidence for confirmation is a nonzero, modest count of thrust actions attached to
`escape_start`s, earlier improved clearance/later life epochs, and fewer defenses at
TTC 0–1 without the indiscriminate thrust volume that hurt 0008.  Evidence against it
is an early score/life drop caused by thrust during a bad heading, escape pulses whose
clearance does not improve, or the same zero-thrust escape starts and TTC-zero defenses.
The official metric remains unclipped ALE reward; lives, life epochs, and survival are
diagnostics only.

## Jev state, questions, thresholds, and fallback

The parent’s strict JSON state is retained: derived snapshot buckets (`frame`, `lives`,
`score`, player presence/position/motion, danger/impact/clearance, rock and missile
counts), up to six threat records with bearing/size/impact/known-motion, revalidated
path records (`id`, direction, clearance, impact, kind), controller phase/maneuver and
per-life defense count, and the allow-listed Astra plan.  Raw RAM is not sent to Jev.

One batched request carries five independent typed judgments.  Two reversed `Choice`
questions (`mode_a`, `mode_b`) choose `anchor`, `evade`, `hunt`, or `hold`; agreement
and per-option probability at least `0.30` are required.  Two reversed target `Choice`s
choose `none` or one of the currently observed source-slot/size/index ids; agreement,
probability at least `0.28`, and newest-snapshot presence are required.  A `defense`
`Noul` is only a hint and is accepted at probability at least `0.80`; code additionally
requires an imminent coast contact, no safer revalidated path, a present player, and an
unused defense in the life.  These thresholds and questions are unchanged so the only
experimental variable is coupled escape thrust.

Answers are tagged with the judged frame and life epoch, dropped after 42 frames or on a
life/discontinuity, and never directly name an action or heading.  Jev is launched at
most every 36 frames with a single outstanding request and a 2.25-second timeout; the
last local controller keeps ticking while it is pending.  Garbage, disagreement,
timeouts, unavailable target ids, and low confidence are advisory misses.  The fallback
is, in order: release/quarantine after absence or reset; local revalidated escape or
per-life `DOWN` only under the parent TTC/path gate; the coupled bounded maneuver/anchor
pulse; then short target turns and press/release FIRE.  Astra remains a single aggregate
review and may only select the parent’s four durable preferences; malformed or late text
is ignored.

## Evaluation

A successful result should preserve the parent’s long horizon turret runs (seeds 0, 4, 8,
and 10) while improving the two collision-sensitive episodes (especially seed 6) with
fewer late defenses.  A regression on clear-field score or increased thrust-heavy deaths
would refute the reachability hypothesis and indicate that the parent’s conservative
rotation-only behavior, rather than inert escape reservations, was the safer choice.
