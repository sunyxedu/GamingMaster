# Candidate 0008 — latency-hedged asynchronous approval reservoir

## Parent and diagnosis

Parent: `candidates/0005/` (crossover of the literal Jev controller and the idle
mailbox); the reference comparator is `candidates/0003/`, the current best.

Both useful parents establish that the mechanical bridge is now correct: a
typed approval of concrete object ids can be expanded into exact timestamped
lane transitions, including chords and hold tails.  The remaining failures are
not a pattern-recognition problem.  They occur before the approval reaches the
mailbox.

The clearest evidence is the worst 0005 trace, seed 104.  Its first decision
starts at chart time 0, times out at 1.201 s, and the retry is not requested
until chart time 1392 ms.  The first successful answer arrives about 466 ms
later, around 1.86 s.  The first event batch begins at chart time 2286 ms and
the final result is 73 MAX and 10 MISS (775932), even though every event that
was eventually scheduled has zero head/tail error.  Thus ten early objects
were lost to an approval gap, not to bad event arithmetic.  0003 shows the
same tail sensitivity from another angle: its seed-103 first Jev call took
1.0547 s, and the first post-answer batch was submitted at chart time 1056 ms;
that episode ended with five MISS and a minimum signed head error of -168 ms.

The strong 0005 episodes are a useful control: once the first answer arrives
in time, seeds 100, 101, 102, and 105 reach 1,000,000 with zero hold breaks.
The existing 820 ms refresh and persistent hold metadata are therefore worth
keeping.  The untested part is the transport latency tail.

## Hypothesis

A single synchronous Jev request creates a preventable first-response outage.
If the request is asynchronous, the game can keep applying the already approved
mailbox while it waits; if it has not completed by a short measured latency
target, a second identical Jev request can be *hedged* against it.  The first
usable typed answer should arrive near the faster of the two response times,
reducing the probability that the first visible notes pass before approval.

This is a structural latency hypothesis, not a timing or parameter calibration
hypothesis.  It predicts the largest gain on dense/high-key seeds with an early
long-tail Jev request (especially the 0005 seed-104 failure), while leaving
already-perfect episodes mechanically unchanged.  It also predicts that
parallel hedges should be rare in normal latency and consume far fewer calls
than a per-tick policy.

## What changed

* Jev is no longer called synchronously in the game loop.  A round starts one
  `ctx.spawn` worker containing the ordinary `ctx.decide` call.  The main loop
  continues to wait for realtime ticks and executes the current timestamp
  reservoir while that worker runs.
* At 270 chart milliseconds after a round starts, if its primary answer is
  still pending, the harness starts one second Jev request over the exact same
  strict-JSON state and typed Choice.  The first usable result wins; late hedge
  results are ignored and cancelled where possible.  At most four background
  calls are retained, and every call has a five-second provider timeout.
* A successful answer approves `execute_all`, `execute_taps`,
  `execute_holds`, or `execute_none` over the concrete object ids in that
  round.  Only approved ids can enter the scheduler.  The scheduler itself is
  deterministic: it groups equal timestamps through one nondecreasing batch,
  creates a real down/up tap, preserves key state, and adds an approved hold's
  tail later when that tail enters the horizon.
* The observation is read again after polling answers and before posting an
  event batch.  This keeps wall-clock Jev latency from turning an otherwise
  future chart time into an invalid stale event.
* The current approved reservoir remains valid across rounds.  A refresh is
  attempted every 700 ms after a winning answer, while empty rounds poll for
  new notes without spending a Jev call.  A timeout, malformed result, failed
  hedge, or garbage pre-flight answer leaves existing events intact and retries
  later; it never parses prose into an action or invents an unapproved note.
* Timestamp batches are reposted as an idempotent heartbeat (or `noop` before
  the first approval) so the realtime loop does not wait on Jev and stale ticks
  are not needlessly introduced.  Background futures are cancelled in the
  `finally` path.

Astra is intentionally not used in this candidate.  0001 and 0004 demonstrate
that a slow Astra answer did not precede a useful action, whereas the current
problem has a precise typed interface and a sub-second recognition task.  No
System-2 plan is needed to convert this public, validated chart representation
into events; adding Astra would increase the first-response outage the
hypothesis is designed to remove.

## Confirmation and refutation

Confirmation would be:

1. the first successful approval in seed 104 (or an analogous dense seed)
   arrives before the first dense object group rather than after chart time
   1.3–1.8 s;
2. the trace shows a hedge only on slow primary calls, and the first event batch
   contains early object times;
3. the 0005 seed-104 miss cluster disappears or shrinks while the zero-hold-
   break behavior remains; and
4. mean score improves over 955261 without a large increase in Jev calls.

The hypothesis is refuted if the hedge produces no earlier usable answers,
causes provider/contract errors, or the early objects still miss despite a
legal batch being posted before their times.  If scores improve on early
objects but later sections miss, the latency diagnosis is supported and the
remaining issue is refresh coverage or event-reservoir bookkeeping.  If all
models answer quickly but the score falls, that would indicate that concurrent
calls or future-event reposting is unsafe and the hedge should be removed.

## Tier-2 System-1 interface

### State schema

Each round sends only strict JSON made from the current public observation:

```text
{
  ruleset, current_time_ms, planning_horizon_ms, key_count, od,
  judgement_windows_ms,
  lanes: [{lane, key_down}],
  candidate_notes: [{id, lane, kind, start_ms, end_ms,
                     time_to_head_ms, pattern, chord_lanes, chord_size,
                     same_lane_run_length}],
  active_holds: [{id, lane, head_time_ms, tail_time_ms, key_down,
                  release_due, hold_break_risk}],
  recent_results: [{id, lane, kind, result, head_error_ms,
                    tail_error_ms, hold_break}],
  telemetry: {score, accuracy, combo, miss_rate, hold_break_rate,
              remaining_objects, counts},
  execution_contract: {approval_unit, tap, hold, chord, timing,
                       invalid_records},
  reservoir: {already_approved_objects, already_decided_objects, policy}
}
```

`candidate_notes` is the complete set of newly observed ids for the round,
not a request for Jev to count, calculate, or generate timestamps.  Metadata is
retained in code after visibility ends.  The state is rebuilt for each round;
a hedge receives the same snapshot so it is a latency duplicate, not a second
opinion about a changed clock.

### Question and bridge

The one batched question is a `Choice` named `selection`:

* `execute_all`: approve every listed complete object;
* `execute_taps`: approve only listed taps;
* `execute_holds`: approve only listed holds;
* `execute_none`: approve no listed object because the state is malformed or
  explicitly unsafe.

The instructions explicitly say that the bridge supplies lanes and chart
milliseconds and that all simultaneous chord lanes and both hold transitions
must be preserved.  Code accepts only one of those exact values.  `execute_all`
is accepted as the safe validated-chart default even if a confidence field is
absent.  A non-all refusal with a reported winning probability below 0.50 is
classified as undecided and leaves the round retryable; an unknown value or
missing answer is never converted into an action.  The answer does not itself
post an event: the bridge rereads the observation, checks the fixed planning
horizon, and posts only approved ids.

### Freshness, hedge, and fallback policy

* The primary future is tagged with a round id and its candidate snapshot.
  A hedge begins only after 270 ms and only while that round has no winner.
  A result from a resolved/old round cannot mutate the current approval sets.
* A winner schedules the next round 700 ms later.  If both calls fail or are
  unusable, the retry delay is 180 ms; current queued timestamps remain in
  force.  Model/provider budget and unavailability exceptions are not hidden.
* Before every event post, the code uses the newest `current_time_ms` and
  rejects irrecoverably old heads, out-of-horizon timestamps, malformed holds,
  duplicate transitions, and same-lane key-state conflicts.  A past head still
  inside the public MISS window is moved to a near-current legal future event,
  rather than pretending it was posted at its old time.
* While no fresh Jev result exists, the fallback is the last approved event
  batch (or `noop` before the first approval).  It never approves unseen ids.
  An approved hold remains in the reservoir until its exact tail transition is
  posted, so a transient Jev outage cannot break a hold that was already
  selected.
* The heartbeat is not a new semantic decision.  Repeated timestamp batches
  are documented idempotent actions; it is used only to keep the realtime
  mailbox active while `ctx.wait_for_tick()` advances the game.

The useful measurement is not merely the number of Jev calls.  Traces should
compare the wall/chart time of the first accepted round, the time of its first
event batch, hedge frequency, early-object result counts, and the final
miss/hold-break distribution.  A score increase concentrated in the previous
approval gap would support the stated latency-tail hypothesis rather than a
chance change in pattern selection.
