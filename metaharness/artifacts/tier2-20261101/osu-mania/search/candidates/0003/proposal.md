# Candidate 0003 — replace abstract Jev commands with a timestamped scheduler

## Parent and diagnosis

Parent: `candidates/0000-tier2_jev_only/` (also the current best candidate).
There is no `lessons.md` in this archive.

The parent’s worst episodes are all tied at zero, and the trace makes the
failure mechanical rather than strategic.  In `trace_seed100.jsonl`, the first
request is at chart time `0` while the observation already exposes lane 0,
object 1, at `start_ms: 1030` (the visible planning horizon is 2000 ms).  Jev
returns the choice `noop` with probability `0.74`; the next fresh action is
literally `noop`.  Later answers are labels such as `tap <lane>`,
`event <time_ms> <lane> <down|up>`, and `events <time_ms> <lane> <down|up> ...`.
Those labels contain neither an actual lane nor a timestamp, so the adaptor
cannot hit a note.  The same pattern appears in every training trace: for
example seed 101 posts 47 actions and seed 103 posts 48, but their final
observations contain only MISS results (all 76 and 227 objects respectively).
The resulting aggregate is `accuracy: 0.00`, `miss_rate: 1.00`, `native_score_v1:
0.0`, despite Jev calls succeeding.

## Narrow hypothesis

The zero is caused primarily by an unexecutable answer-to-action bridge, not
by Jev being unable to recognize mania patterns.  If Jev chooses between a
named *precomputed scheduling policy* and the harness converts that policy to
legal future events from the structured observation, the same Jev-only
architecture should score substantially above idle.  The bridge must refresh
the observation after the synchronous Jev call: a 0.2–0.5 s call otherwise
makes an action generated from the request-time clock stale, while the game
continues at 16 ms ticks.

## What changed (one targeted change)

I kept the parent’s synchronous, Jev-only loop and its no-Astra allocation.  I
changed only the action interface and low-level bridge:

* Jev now receives a compact JSON state with ruleset, current time, horizon,
  visible notes, chord lanes, patterns, active-hold risk, recent results, and
  progress.  It chooses `execute_window` or `keep_existing`, never an abstract
  command grammar.
* On `execute_window`, code flattens/deduplicates lane-local notes, groups the
  implied simultaneous timestamps through one nondecreasing `events` batch,
  emits a tap as `down` then `up`, and emits a hold head and tail.  It keeps
  per-object head/tail sets so repeated observations do not duplicate events.
* After Jev returns, the harness calls `game.observe()` again and schedules
  from that newest state.  Only events in the current planning horizon (or a
  still-recoverable just-passed judgement window) are emitted.  Future hold
  tails can be added later; active-hold fields are also accepted as a fallback.
* Invalid/garbage typed answers default to `execute_window`; a transient Jev
  timeout uses the last executable policy and continues.  Contract budget and
  model-unavailable exceptions still propagate normally.

This is code for the public, verified action grammar and observation fields; it
does not inspect a chart file or use hidden seed data.  The model still
approves the scheduling policy, while code performs exact timestamp arithmetic
and legality checks.

## Falsification and confirmation

Confirmation would be a nonzero score and nonzero MAX/GREAT/GOOD counts in
training traces, with queued event actions visible before the corresponding
note times and a large reduction in MISS rate.  A score near full marks would
also support the stale-observation part of the hypothesis.

The hypothesis is refuted if the new traces contain valid, timely `events`
batches but remain all MISS, which would indicate a simulator/action-grammar
or event-order assumption instead.  It is weakened if only the first horizon
scores and later notes miss: that would point to queue deduplication or hold
release tracking rather than Jev recognition.  If Jev repeatedly selects
`keep_existing` while there are unscheduled notes, the policy question needs a
later revision; the empty-queue safeguard should still make the first visible
window executable.

## System-1 contract and fallback

**State schema.** Each request is strict JSON:

```text
{
  ruleset, current_time_ms, planning_horizon_ms, key_count,
  judgement_windows_ms,
  visible_notes: [{id, lane, kind, start_ms, end_ms,
                   time_to_start_ms, is_chord, chord_lanes, pattern}],
  visible_patterns: [{start_ms, time_to_start_ms, lanes, pattern, is_chord}],
  active_holds: [{id, lane, release_due, hold_break_risk, release_window_ms}],
  recent_results: last 8,
  progress: {score, accuracy, combo, miss_rate, hold_break_rate},
  scheduler: {already_scheduled_objects, action_rule}
}
```

The sole question is a `Choice` over:

* `execute_window`: code queues every visible, not-yet-scheduled object,
  including simultaneous chord lanes, tap up transitions, and hold tails.
* `keep_existing`: code submits `noop` and trusts already queued events.

The first option is intentionally the executable default when notes are
visible.  A returned `keep_existing` is honored only after at least one head
has been posted **and** its returned option probability is at least `0.60`.
The newest observation is then rechecked; if a note entered the horizon while
Jev was thinking, the bridge overrides `keep_existing` and queues it.  An
unknown value, absent answer, low-confidence/garbage stand-in, or transient
call failure falls back to `execute_window`.  No raw model text is parsed as
an action.  The event scheduler is idempotent through object/event sets, and
it never waits for a new decision when an event is already queued.  The model
timeout is 5 s (above the measured Jev tail); the fallback catches ordinary
timeout/runtime failures while allowing normal budget/deadline and
unavailable-model termination.
