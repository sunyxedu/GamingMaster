# Candidate 0005: approved timestamp-plan mailbox

## Parents and diagnosis

Crossover parents: `candidates/0000-tier2_jev_only/` and
`candidates/0002-tier2_idle/`.

The Jev parent is robust as a contract baseline (6/6 episodes ended `ok`, with
no failed calls), but it asks Jev to choose one of the literal grammar strings
`noop`, `event <time_ms> <lane> <down|up>`, etc.  The traces show the concrete
failure: Jev repeatedly selected `noop`, then selected incomplete strings such
as `tap <lane>` and `event <time_ms> <lane> <down|up>`.  Those are action-schema
placeholders, not legal lane events.  For example, seed 100's first answers
were `noop` at about 0.375 s, 0.297 s, 0.276 s, then `tap <lane>` and the
literal event placeholder; the final observation had 24 MISS results, no timing
errors, and score 0.  Seed 101 and seed 102 show the same pattern.  The parent
therefore made 330 successful Jev calls but scored exactly like the idle parent:
mean score 0, miss rate 1.0.  The idle parent contributes the complementary
lesson: waiting for the real-time clock is safe and survives every episode, but
it cannot score anything.

## Hypothesis

The limiting mechanism is not recognition of the chart.  It is the missing
bridge between a semantic Jev answer and the parameterized event grammar,
combined with trying to make a 0.2--0.5 second model call decide the current
tick.  A useful harness should ask Jev to approve a *visible object plan* once,
then let deterministic, verified code do the exact timing, chord grouping,
key-state bookkeeping, and hold release.  Existing approved events must keep
running while Jev is thinking; the idle behavior is the fallback whenever no
fresh approval is available.

This is a structural change rather than a prompt-only change: Jev no longer
chooses an action-string template.  It chooses a subset policy over concrete
object IDs (`execute_all`, `execute_taps`, `execute_holds`, or `execute_none`),
and the harness expands only the approved IDs into `events TIME LANE ...`.
Nothing outside the model-approved subset is auto-played.

## What changed

* The state sent to Jev is a compact strict-JSON projection of the public
  observation: ruleset, current chart time, OD/windows, key states, active
  holds, recent timing/error telemetry, and only newly visible candidate notes.
  Each candidate includes its object ID, lane, kind, absolute head/tail times,
  chord lanes, pattern cue, and same-lane run length.
* One request batches two independent typed judgments.  `selection` is a
  `Choice` over the four subset policies.  `hold_guard` is a `Noul` asking only
  whether the supplied hold records are complete and safe to hold through their
  stated tails.  The controller performs all arithmetic and does not ask Jev to
  calculate timestamps.
* A selected `execute_all` plan is expanded in code into down at each tap/hold
  head, an up one millisecond later for a tap, and an up at the hold tail.
  Equal-time lane transitions are sorted into a legal batch, so chords remain
  simultaneous.  The code keeps a virtual key state and a pending-event list,
  preventing duplicate down transitions while an approved hold is active.
* Approved hold IDs remain in the local plan after their heads leave the
  visible list.  Their release is posted when the tail enters the current
  planning horizon.  A refresh is requested roughly 820 ms apart, leaving the
  old timestamped mailbox in front of measured Jev latency.
* After the first event batch, the last timestamped batch is re-posted as a
  heartbeat on each loop.  The adaptor documents repeated Tier-2 timestamp
  submissions as idempotent, so this reduces stale ticks without changing the
  queued transitions; if no batch exists, the harness simply waits like the
  idle parent.
* Exceptions, missing/garbage stand-in values, malformed observations, and
  transient Jev failures do not escape the episode.  They leave existing
  timestamped events untouched and wait for the next tick/refresh.  Unknown
  selection output approves nothing; the idle parent is the safe fallback.  A
  timeout or malformed answer shortens the next retry interval to 180 ms while
  the prior batch keeps running through the idempotent heartbeat.
  Budget/episode closure and model-unavailability exceptions retain their
  normal contract semantics.

## Confirmation and refutation evidence

The hypothesis is confirmed if training traces show legal event batches whose
object IDs receive non-MISS results, with holds receiving a tail error and
chords receiving simultaneous lane transitions.  The first two live trials
already supplied this signal: seeds 100 and 101 each reached ScoreV1
1,000,000, including their holds; the heartbeat variant retained both scores
and reduced stale ticks to 271/849 and 129/650.  The strongest confirmation is
an accuracy/ScoreV1 score materially above zero while the number of Jev calls
stays near one per planning chunk rather than one per note.  A useful secondary
signal is that most actions are timestamp batches posted before their target
notes, despite model latency.

It is refuted if Jev consistently selects `execute_none`/incomplete values even
when the candidate plan is explicitly prevalidated, if event batches are
accepted but notes still have null timing errors, or if holds systematically
break because the persistent approved-plan bookkeeping is wrong.  A zero score
with no event batches would specifically refute the approval interface, not the
low-level scheduling arithmetic.  A zero score with correctly formed batches
would point to an unverified event-time or simulator-queue assumption and
should be investigated from observations before adding more model calls.

## Jev interface, thresholds, and fallback

The state schema is:

```text
{
  ruleset, now_ms, key_count, od, judgement_windows_ms,
  planning_horizon_ms,
  lanes: [{lane, key_down}],
  candidate_notes: [{id, lane, kind, start_ms, end_ms, pattern,
                     chord_lanes, chord_size, same_lane_run_length}],
  candidate_plan: {selection_unit, tap_events, hold_events, chord_rule,
                   timing_rule, invalid_plan},
  active_holds: [{lane, head_time_ms, tail_time_ms, key_down,
                  release_due, hold_break_risk}],
  recent_results: [...last eight...],
  telemetry: {score, accuracy, combo, miss_rate, hold_break_rate,
              remaining_objects, counts}
}
```

`selection` is a `Choice`:

* `execute_all` means every listed candidate ID is model-approved;
* `execute_taps` or `execute_holds` selects only that named kind;
* `execute_none` is the explicit unknown/unsafe fallback.

The code accepts only an exact known choice.  It does not parse prose or
extract times from a model response.  `hold_guard` is a batched `Noul` used as a
recognition/diagnostic signal, but it cannot veto `execute_all`: the local chart
validation has already rejected malformed or overlapping holds.  This matters
because the seed-101 trial produced `execute_all` together with a low guard
probability (0.28); using a 0.35 veto there caused three otherwise schedulable
holds to be rejected.  An explicit `execute_taps` choice remains the model's
hold fallback.  An absent or malformed guard does not crash or synthesize a
plan; the explicit selection still controls.  A malformed selection approves
no candidates and retries on a later refresh.  Timeout and transient errors
similarly preserve the already posted mailbox and do not post new events; the
runtime retry is 180 ms rather than a full 820 ms refresh interval.  The
measured Jev p90 is below the 1.2 s call timeout, and future events are
scheduled inside the fixed 2,000 ms horizon before the next refresh.

This split gives Astra no per-note clock duty and uses no System-2 calls: the
parents establish that slow/grammar-level control cannot act in this game, while
the deterministic adaptor already has the public note times and rules needed for
exact expansion.  If a later iteration adds Astra, its appropriate role is to
revise the subset policy/hold fallback after observed errors, not to emit live
keyboard strings.
