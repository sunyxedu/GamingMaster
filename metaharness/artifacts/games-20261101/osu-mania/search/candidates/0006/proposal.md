# Candidate 0006 — compact semantic leases with a deterministic mailbox

## Parent and concrete diagnosis

Parent: `candidates/0003/` (with the 0005 mailbox trace as a timing
cross-check).

The parent proved that the important low-level bridge is correct: a Jev answer
can authorize a window, and code can expand it into exact future transitions.
Five of six 0003 episodes were perfect.  Its remaining seed is a useful
failure, not random score noise.  In `trace_seed103.jsonl`, the first request
contains a 5,161-character state and takes 1.055 seconds.  During that request
the realtime game applies about 65 noop ticks.  The first batch therefore
arrives after the earliest heads have passed; the episode ends with 214 MAX,
4 GREAT, 2 OK, 2 MEH, and 5 MISS for 904,110.  Later requests have 8,000–9,000
character states and repeatedly re-decide a policy even when the mailbox is
already doing the work.

The crossover exposes the same boundary in a different form.  In
`candidates/0005/trace_seed104.jsonl`, the first request times out at 1.201
seconds; the first ten objects are unrecoverable MISS results, although every
later approved event is a MAX.  Its persistent mailbox is sound, but the
candidate-note state and two-question request did not protect the bootstrap.
The archive's latency measurement also says Jev is about 0.434 seconds on a
warm request, so request size and the cold tail are the actionable target.

## Hypothesis

The remaining score loss is primarily **decision-start latency**, not pattern
recognition or event arithmetic.  A model does not need the full per-note
observation to decide whether a verified visible window is safe to execute.
If code constructs a small execution lease from the public observation, and Jev
only grants that lease plus labels its broad shape, then:

1. the first request is small enough to return before the first normal head;
2. a grant can cover all currently visible IDs without asking Jev to count,
   calculate milliseconds, or generate an action; and
3. the deterministic ledger can keep holds and timestamped events alive while
   Jev is thinking or temporarily late.

This is a structural change from 0003's repeated full-observation policy choice:
System-1 becomes a semantic gate/recognizer, code becomes the persistent plan
and event compiler, and the action loop is driven by lease coverage rather than
by every model completion.

## What changed

* `_shape_digest` sends Jev a strict, compact JSON projection: current clock,
  horizon, pending object IDs, counts of taps/holds/chord groups, a six-object
  preview, active-hold risk, queue coverage, and small progress telemetry.  It
  deliberately omits the repeated lane-local note dump and irrelevant raw
  fields.  Exact note metadata remains in a local catalog populated only from
  the public observation.
* A single batched request contains three typed judgments.  `lease` chooses
  `activate_visible`, `preserve_lease`, or `pause_uncertain`; `shape` is a
  recognition label used only to choose the next review interval; and
  `hold_attention` is a diagnostic Noul.  Jev never writes timestamps, lanes,
  or command strings.  A lease applies only to the concrete `pending_ids`
  captured in that request.
* Once granted, code owns the rest: it deduplicates objects, preserves exact
  chart times, emits simultaneous chord transitions in one nondecreasing
  batch, gives taps a real down/up pair, and retains approved hold tails after
  heads leave visibility.  `head_sent` and `finish_sent` are separate so a
  delayed tail is never forgotten or double-posted.
* Reviews happen at roughly 900 ms (650 ms for Jev's dense/hold-transition
  label), rather than immediately looping into another synchronous request.
  Every intervening tick posts the most recent timestamp mailbox; the adaptor
  specifies that this repetition is idempotent.  This makes the controller
  responsive to the realtime evaluator without sacrificing future events.
* The latest observation is reread after every model call before constructing
  events.  A response that arrives late can still schedule a head inside the
  public MISS window, while a stale head is not forged into a perfect future
  event.  A timeout grants the already decoded request snapshot as a bounded
  fallback and retries after 120 ms; it never parses model prose.
* The fallback is defensive against arbitrary pre-flight answers: an unknown
  lease value activates the prevalidated snapshot, while a high-confidence
  `preserve_lease` is honored only when an existing mailbox's latest timestamp
  reaches the complete pending section.  Thus option-order bias cannot starve
  a newly visible note.  Budget/deadline and model-unavailable exceptions
  retain their contract meaning.

Astra is intentionally not placed in the realtime hot path.  The public
observation already supplies the rules, coordinates, windows, and pattern
facts, while Astra's measured 21+ second response is longer than these charts.
In this design the verified adaptor code is the authoritative rule summary and
lease planner; there is no useful unresolved strategic subgoal to hand to
Astra during a ten-second episode.  If traces instead show repeated late Jev
leases after this compression, the next revision's escalation prompt would
send Astra the compact digest plus recent miss/hold-break telemetry and ask it
to author only a new lease question/fallback policy, never event timestamps.
This candidate therefore spends zero System-2 calls unless that evidence makes
such a revision worthwhile, preserving the leaderboard's System-2 tie-break.

## Confirmation and refutation

The hypothesis is confirmed by an initial request whose state is materially
smaller than 0003/0005, a first event batch before the first heads' MISS
boundaries, and a reduction of seed-103-style early MISS results without
increasing failed calls.  The strongest result is six perfect training
episodes, or at least non-MISS results for the objects that were visible in
that first lease, while subsequent event batches retain 0003's zero-error
MAX/hold behavior.

It is refuted if compact requests still arrive too late while a mailbox is
empty, if the lease grant does not produce legal `events` batches, or if score
loss appears after the first lease despite all head/tail IDs being retained.
A low score with correct early batches would point to an event-order or
simultaneous-key assumption; a low score with no batches would point to the
activation answer or its timeout fallback.  A hold-break increase would
specifically falsify the catalog/tail persistence invariant.

## System-1 state, questions, thresholds, and fallback

The state sent on each review is:

```text
{
  ruleset, clock_ms, horizon_ms, key_count, od,
  pending_visible, pending_ids,
  shape: {
    object_count, head_groups, chord_groups, tap_count, hold_count,
    patterns, earliest_head_in_ms, latest_head_in_ms
  },
  first_objects: [{id, lane, kind, head_in_ms, tail_in_ms}],
  active_holds: [{lane, release_due, hold_break_risk}],
  coverage: {granted_objects, heads_sent, finishes_sent, last_mailbox_exists},
  telemetry: {score, accuracy, combo, miss_rate, hold_break_rate, counts}
}
```

All values are derived from the current structured observation and are strict
JSON.  `pending_ids` is the model's selection scope, not a request to generate
IDs; the code maps a grant back to its exact catalog records.

* `lease` is a `Choice`.  `activate_visible` grants every pending object in
  the captured window.  `preserve_lease` grants none only when its probability
  is at least 0.65, an existing mailbox's latest timestamp reaches the whole
  pending section (including a hold tail), and the parser has found no uncovered
  event.  `pause_uncertain` and invalid/absent values use the activation
  default; the observation parser has already rejected malformed records, and
  losing a valid visible head is worse than a semantic no-op.
* `shape` is an independent `Choice` over `ordinary`, `chords`,
  `repeated_lane`, `hold_transition`, and `mixed_dense`.  It does not calculate
  timing.  A valid `mixed_dense` or `hold_transition` answer shortens the next
  review to 650 ms; other labels use 900 ms.  An invalid label keeps the last
  interval.
* `hold_attention` is a recognition-only `Noul` about explicitly due/high-risk
  active holds.  It is logged for later diagnosis and cannot veto a
  prevalidated hold: the 0005 trace already showed that a low-confidence hold
  guard could incorrectly erase good holds.

A request timeout or ordinary Jev runtime failure leaves all prior grants and
mailbox events intact, grants the captured snapshot as the executable late
fallback, and schedules the next review 120 ms after the newest observation.
`BudgetExceeded`, `EpisodeOver`, and `ModelUnavailable` are not swallowed.
The controller posts a fresh idempotent mailbox on every tick, so model latency
changes freshness of the decision but does not stop already approved play.
