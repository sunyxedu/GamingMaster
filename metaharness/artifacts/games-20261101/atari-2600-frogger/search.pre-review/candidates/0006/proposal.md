# Candidate 0006 — candidate-track perception with typed maneuver judging

## Parent and diagnosis

Parent: `candidates/0001-tier2_astra_only/`, as requested by the rewrite
selection.  I retained its useful observation-driven framing and safe handling
of model failures, but not its blocking control loop.  The best comparator is
`candidates/0003/`.

The archive makes the failure concrete.  In the parent, seed 109's first Astra
call returned no useful action and the episode posted no action until it lost
all frogs; across the parent there were 19 System-2 calls, 99.98% stale ticks,
and mean score 1.  Its 49.7 s plus output-length latency cannot drive a 60 Hz
emulator.  Candidate 0003 fixed the action latch and reached mean 12, but its
six episodes still had `frogs_home=0` and `deaths=4`.  Its seed-103 choices
remain nearly the same after the release fix: at the start Jev sees only a raw
RAM dump and chooses `NOOP`/`UP`; once rows 0–6 are reached it selects `UP` at
roughly 0.4–0.5 probability, with LEFT and RIGHT around 0.1.  The resulting
trace has isolated UP pulses, scores 14, and then loses the last frog without a
home.  Thus the release mechanism was confirmed, while the raw-RAM direct
question was not a perception or route policy.

## Hypothesis

**The next bottleneck is not action freshness but an unstructured perception
interface.**  Jev is being asked to infer traffic gaps, carrier support, and
horizontal route choice from 128 unrelated-looking bytes.  It defaults to the
obvious forward action.  A useful rewrite is to make code compute only
verifiable, coarse candidate facts from consecutive RAM observations, then ask
Jev for independent typed judgments of the immediate hazard and maneuver.  A
small amount of forced progress and exploration is needed so Jev uncertainty
cannot turn into the 0005 failure mode (almost all NOOP), while the one-frame
release remains mandatory.

The code tests this with a deliberately explicit candidate decoder: five groups
of four moving bytes are treated as road-lane candidates; the faster-changing
run beginning at `$96` supplies five carrier-row groups, with the intervening
candidate row treated as the safe median.  For the frog's candidate x, it reports semantic
buckets (`OPEN/TIGHT/BLOCKED` traffic, `SUPPORTED/EDGE/MISSING` carrier,
approach/recede motion, and a left/right seek direction).  These are hypotheses,
not hidden game source: the raw bytes and decoder warning are sent to Astra for
review, and the Jev state never pretends that an object identity is certain.
The controller only uses these facts to generate a candidate; code then applies
Jev's hazard/maneuver answer with probability gates and a stall escape.

Astra is now a background route-policy reviewer rather than a frame driver.  It
returns an allow-listed JSON card (`bias`, road rule, river rule, fallback),
which changes the candidate generation and is included in the next Jev state.
It is launched initially and after life loss, while NOOP/one-frame pulses and
Jev continue.  Malformed prose is inert.  This directly tests whether slow
System-2 reasoning is valuable as a structural policy revision when it does not
hold the joystick.

## What changed and why

* **Code:** tracks the relevant RAM groups, computes circular proximity and
  coarse motion/support buckets, derives a next-row candidate, and owns score/
  death/row-stall bookkeeping.  It never asks Jev to count byte distances.
* **System-1:** receives a compact JSON state containing zone, horizontal band,
  current/target semantic buckets, progress, last action, and the parsed Astra
  card—not the full 128-byte dump.  One request batches three questions: two
  independently reordered `Choice`s over `FORWARD/WAIT/LEFT/RIGHT/BACK/UNSURE`,
  and one hazard `Choice` over `CLEAR/TRAFFIC/WATER/EDGE/UNKNOWN`.
* **Bridge:** code vetoes a high-probability traffic/water/edge hazard, accepts
  a duplicate maneuver only when both typed choices agree with a meaningful
  selected-option probability, otherwise uses the computed candidate.  After a
  short stall it forces a legal progress/steer candidate rather than idling.
  Every UP/LEFT/RIGHT/DOWN is posted for one tick and followed by a fresh NOOP;
  a late Jev call follows the same pulse bridge.
* **System-2 scheduling:** Astra is spawned with a compact snapshot and at most
  five reviews per episode, with a short JSON output target and a 78 s timeout.
  Reviews are never awaited by the realtime loop.  A revision prompt includes
  recent zone/target/action/outcome evidence and can retain the session.  Only
  allow-listed fields can affect control, so arbitrary stand-in output cannot
  crash pre-flight or invent a seed-specific schedule.

## Confirmation and refutation evidence

Confirmation would be a held-out/training trace with meaningful LEFT/RIGHT
pulses, fewer deaths per forward progress, at least one `frogs_home` increment,
and a score above 0003 without a large increase in stale ticks.  The strongest
specific signal is that target `SUPPORTED`/road `OPEN` states lead to forward
pulses while `BLOCKED`/`MISSING` states lead to waits or steering, rather than
0003's unconditional UP preference.

Refutation would be six clean episodes that still have zero homes and four
lives lost, or evidence that the grouped RAM candidates do not correlate with
road/carrier outcomes: e.g. repeated deaths in `OPEN` road targets and safe
crossings in `MISSING` river targets.  If scores improve only because of more
forward-hop points but no home, the remaining failure is the candidate decoder
or horizontal target, not Astra latency.  A failure with 0% stale ticks but
very few directional pulses would refute the progress escape and indicate that
Jev gates remain too conservative.

## Jev contract, thresholds, and fallback

The state sent to Jev is strict JSON with:

* `zone`, `horizontal_band`, coarse `homes`, `level`, `timer`, `row_change`,
  `progress`, `stall_bucket`, `last_action`, and `decoder`;
* `current`: `kind`, `status`, `side`, `breadth`, `motion`, `escape`, `seek`;
* `target`: the same fields plus `target_zone`; and
* `plan`: the allow-listed Astra card and `candidate_action`.

`current` describes the row the frog occupies.  `target` describes the next
row.  A traffic target's `OPEN/TIGHT/BLOCKED` means no nearby/near/overlapping
RAM candidate; a carrier target's `SUPPORTED/EDGE/MISSING` means overlap/near/no
candidate.  `UNKNOWN` is a valid answer.  The numeric RAM and raw x/row are
kept for Astra and auditability, not put in Jev's decision state.

The two maneuver Choices use the same criteria with different rotating option
orders, addressing Jev's documented first-option bias.  A consensus is
executable when the selected-option probability is at least **0.34** in both
answers.  A single answer can preserve the code candidate at **0.42**.  A
`TRAFFIC`, `WATER`, or `EDGE` hazard veto applies at **0.58** probability when
its corresponding target/current precondition is present.  A weak or malformed
answer falls back to the code candidate.  After six consecutive decision
observations without row progress, a NOOP is replaced by a directional or UP
candidate; this is a liveness guard, not a held command.

The fallback hierarchy is: explicit lethal-hazard veto; current typed consensus;
code candidate from the RAM lens; then NOOP if the lens is unknown.  At the
starting sidewalk and after a reset the candidate is UP, and after every
accepted direction the bridge waits for exactly one tick and posts NOOP.  Model
timeouts, malformed values, arbitrary Astra text, and absent optional fields
are all recoverable; budget/episode closure returns normally.  No module or
cross-episode state is required.
