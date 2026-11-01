# Candidate 0007 — refreshed policy leases over a persistent event mailbox

## Parents and diagnosis

Parents: `candidates/0003/` and `candidates/0005/`.

Both parents established the important positive mechanism: Jev must approve a
predefined semantic scheduling policy, while code expands that policy into
legal timestamped transitions.  `0003` was the stronger scorer (mean
984018.333; five of six training episodes reached 1,000,000), and its exact
chord/hold expansion produced no hold breaks.  `0005` was the stronger realtime
plumbing experiment: it kept an approved plan in a mailbox and reposted the
same timestamp batch, reducing stale ticks from about 93% to 35%.

The remaining failure is visible in the supplied worst traces rather than a
rules misunderstanding.  In `0003/trace_seed103.jsonl`, the first request is
made at chart time 0, the first scheduling record arrives around wall clock
1.064 s, and it reports only 40 scheduled heads.  The first non-noop batch
contains already-passed times such as `888 2 down` and `1014 ...`; the final
observation is 214 MAX, 4 GREAT, 2 OK, 2 MEH, and 5 MISS (score 904110).
The lessons summarize the same episode as 743 stale ticks out of 792 and a
minimum head error of -168 ms.  The problem is not that a chord or hold was
expanded incorrectly: the scheduler approved the *request-time* candidate IDs,
then the chart advanced while Jev was thinking, so newly visible notes had to
wait for another request.

`0005` fixes the durable-mailbox half, but its `trace_seed104.jsonl` shows the
other edge: its first Jev request failed at 1.201 s, and the first successful
approval was only around 1.862 s.  The mailbox then preserved the plan, but
10 objects were already lost (73 MAX and 10 MISS in the lesson's summary), for
775932.  Thus simply making a request timeout longer does not solve a slow
first lease; simply asking again more often does not solve a stale approved
set.

## Hypothesis

A Jev answer should be treated as a short **policy lease**, not as approval of
the exact observation that was present when the request began.  If the answer
selects `arm_visible`, code can safely admit the complete, newly refreshed
visible horizon to a timestamp mailbox.  Re-reading immediately after the
answer should prevent the request/response gap from creating an unapproved
hole.  The mailbox must remain persistent so already-approved hold tails and
future transitions continue while Jev is unavailable.

The combined hypothesis is therefore:

> A compact policy question plus a post-answer full-horizon lease, backed by
> the persistent event ledger and an executable timeout fallback, will retain
> `0003`'s high timing score while removing its dense-chart late gaps and
> `0005`'s first-timeout damage.

This is deliberately not a per-note Jev controller.  The model chooses no
lane, timestamp, or raw action string.  Exact chart times, chord grouping,
tap releases, hold releases, event ordering, and JSON/action validation remain
in the harness.

## What changed

* **Lease boundary instead of stale candidate IDs.** The request contains a
  compact derived summary of rows, kinds, active-hold risk, queue coverage, and
  telemetry.  On either a successful answer or a normal timeout, the harness
  calls `game.observe()` again and uses that newest observation as the lease
  input.  `arm_visible` admits every validated object in that refreshed
  horizon, including objects that appeared while Jev was answering.  This is
  the structural change from `0003`'s `decided`/`approved` request snapshot.
* **Persistent physical-event ledger.** Every object ID has retained metadata,
  head/tail posted sets, expired-head set, and physical transition keys.  A
  leased hold remains in `records` after it leaves `lanes[*].upcoming`, so its
  tail can be queued later.  Reposting the last batch is a heartbeat and does
  not duplicate a transition.
* **Adaptive lease cadence.** A normal lease is renewed about 700 chart ms
  after the newest observation, well before the fixed 2,000 ms planning
  horizon runs out.  If Jev times out or returns malformed data, the retry is
  shortened to 220 ms while the prior mailbox continues.  `keep_queue` is
  accepted only with a 0.60 probability for that option; if a near-term
  unleased object would otherwise be uncovered, code overrides it with a new
  admission check.
* **Compact model input.** The model does not spend tokens rereading every
  lane-local duplicate.  It receives up to 48 temporal rows with decoded lane
  sets and pattern names, rather than being asked to count or calculate.  The
  scheduler retains the complete observation locally.  This targets the
  long first response seen in the parents without asking Jev for arithmetic.
* **Careful timeout behavior.** The initial executable fallback is the same
  `arm_visible` lease policy; a timeout refreshes the observation and admits
  its valid objects rather than leaving an empty mailbox.  This is a policy
  fallback, not a fabricated model action.  Ordinary timeout/runtime and
  garbage typed answers are contained; `BudgetExceeded` and
  `ModelUnavailable` retain their required termination semantics.
* **Exact stable timing.** Heads and tails are scheduled at their authoritative
  chart times, with a one-millisecond tap release.  Events are limited to the
  current planning horizon, just-passed heads are retimed to the current clock
  only while still inside the public MISS window, and same-time releases sort
  before same-lane presses.  No timing-bias correction is introduced because
  the parents' best traces already show 0 ms head/tail errors.

No Astra call is made.  The archive's direct Astra trial produced no
Astra-to-action events, and this game has no strategic subgoal that justifies
spending a slow call in the clock-critical path.  The System-1 interface is
fully specified in code, and the durable mailbox is the fallback while any
future interface revision would be too slow.

## Confirmation and falsification

Confirmation would be visible in training traces as follows:

1. The first successful or timeout path logs a refreshed lease and an
   `events` batch whose target times extend beyond the request-time visible
   set; early notes should have non-MISS results rather than the 0003/103 or
   0005/104 late cluster.
2. Dense seed 103 should have substantially fewer than five MISS and fewer
   negative head errors near the MISS boundary, while seed 104 should not lose
   its first visible rows after a slow first Jev call.
3. Hold tails should still appear after their heads leave the visible lists,
   with hold-break rate zero.  Chord rows should receive simultaneous lane
   transitions.
4. Mean score should approach the 1,000,000 ceiling without increasing model
   calls to per-tick/per-note levels.  Stale ticks should be no worse than
   0003, and ideally closer to 0005 because the event batch is reposted while
   lease decisions run.

The hypothesis is weakened if the refreshed batch is valid but only objects
present in the old request score: that would indicate a simulator restriction
on the lease interpretation or a bookkeeping error.  It is refuted if the
first timeout path has a nonempty, timely fallback batch but all of its notes
MISS, which would point to an event timestamp/queue assumption rather than
model recognition.  A rising hold-break rate specifically falsifies the
persistent-tail ledger; repeated `keep_queue` starvation with a full queue
would falsify the confidence threshold or admission predicate.

## System-1 state, questions, thresholds, and bridge

Each Jev request is strict JSON with this derived schema:

```text
{
  ruleset, now_ms, key_count, od, windows_ms, horizon_ms,
  visible: {
    object_count, tap_count, hold_count, first_start_ms, last_start_ms,
    rows: [{start_ms, lanes, patterns, holds, is_chord}]
  },
  active_holds: [{id, lane, release_due, hold_break_risk}],
  queue: {
    leased_objects, heads_posted, tails_posted, coverage_until_ms,
    policy, lease_rule
  },
  telemetry: {
    score, accuracy, combo, miss_rate, hold_break_rate, counts,
    recent_results
  }
}
```

There is one batched typed question, `mailbox`, a `Choice`:

* `arm_visible`: the newest post-answer visible horizon is approved for the
  controller to admit.  Code then uses the refreshed observation, not values
  invented by Jev, to queue all validated heads, chord lanes, tap ups, and
  known hold tails.
* `keep_queue`: no new admission is necessary because the existing timestamp
  mailbox covers the visible section.

The answer bridge accepts only those exact keys.  A `keep_queue` answer must
have probability at least `0.60`; otherwise it falls back to `arm_visible`.
Even a high-probability keep is overridden when an unleased object is within
roughly the next 1,200 ms (or within its MISS recovery interval), so a semantic
answer cannot starve a chart.  Missing, malformed, or arbitrary preflight
answers select `arm_visible`.  A Jev timeout/runtime error leaves the previous
lease running, refreshes the observation, admits the refreshed objects under
the same executable fallback, and retries after 220 ms.  The normal Jev
request timeout is 1.8 s, above the observed 0.4--1.2 s tail; the old mailbox
continues during that wait.

The event bridge retains `records`, `leased`, `head_posted`, `tail_posted`,
`expired_heads`, `event_keys`, and pending future transitions.  It never
releases a hold before its recorded tail, never uses an already-held key as a
new tap down, and never emits outside the fixed future planning horizon.  A
new policy lease is sought by chart time, not by number of notes, so the same
interface generalizes across key counts, streams, anchors, chords, and holds.
