# Candidate 0003: timestamped Jev policy batches

**Parent:** `candidates/0000-tier2_jev_only/` (the Jev-only direct-action
controller).

## Hypothesis

The parent is failing primarily because it asks Jev for a primitive action at
the current tick even though the answer arrives about 0.18 s later. That is
four to five 40 ms ticks, so the observation and the action are stale before
the action is posted. The trace makes this concrete:

- On seed 105, the request at chart time 0 ms saw the first note at 1219 ms;
  Jev returned `noop` after 0.213 s. The next fresh action was at tick 6,
  while ticks 1--5 were already stale. The same pattern repeats: at request
  time 1760 ms Jev selected `press 2` for a note only 4.7 ms away, but the
  fresh action was not applied until tick 50 (about 2000 ms).
- The worst parent episode, seed 102, scored 0.075 with 27 misses and 11
  broken holds. Its request at 1400 ms chose `press 2` while the relevant
  hold head was already 44.8 ms in the past; the action arrived at tick 41.
  Later, the trace shows `release 2` requested at 4800 ms while the hold was
  already 100.2 ms past its displayed start, and the final score reports
  `broken_holds: 11`.
- Across the parent evaluation, Jev latency was about 0.17--0.18 s and
  `stale_fraction` was 0.7796. The final observations show 22 misses on
  average (33.33 heads) and only 3.67 max combo. This is a timing/bridge
  failure, not evidence that Jev cannot read the structured chart. Seed 105
  even produced a correct `press 2+3` choice for a chord, but it was made
  from a stale horizon and did not solve the next stream.

## One change

I keep the parent's Jev-only architecture and remove only its current-tick
action bridge. Jev now chooses a bounded global timing policy (`target_exactly`,
`slightly_early`, or `slightly_late`) from a compact state containing recent
signed judgments and the upcoming structured horizon. Code then converts the
already exposed note horizon into one timestamp-sorted JSON `batch`: one press
at every note's `start_ms`, and one release just after every visible hold's
`end_ms` when that tail is safe to include in the visible timeline. Hold tails
are remembered and deferred rather than allowing a batch to jump over heads
that have not entered the 2.2 s observation yet. Note IDs prevent duplicate
scheduling as the observation window slides. The batch is posted before its
events are due, so model latency is no longer added to the hit time. No Astra
call or seed-specific chart data is used.

Exact timestamping, lane legality, chord grouping (equal timestamps), and hold
bookkeeping remain deterministic code, as the study recommends. If Jev is
late, times out, or returns garbage, the safe policy is exact timing; a stale
model answer cannot turn into an invalid action. Jev is refreshed only at the
start and on a long interval; the rolling note window is advanced entirely by
code, so waiting for a policy refresh cannot create a chart-time gap. A fresh
noop is posted between batches, both to prevent a previous batch from being
replayed and to make the realtime fallback explicit.

## Falsifiable test

The hypothesis is supported if the candidate's misses and broken holds fall
sharply on both the 6K hold seed and the 5K alternation seed, with fresh batch
events producing signed offsets near zero, while Jev latency remains similar.
It is refuted if score stays near 0.17 despite future batches, or if batches
are replayed/ignored (duplicate judgments, malformed-action feedback, or no
reduction in stale timing); that would point to an adaptor scheduling semantic
rather than a Jev recognition problem. The comparison is intentionally to the
parent, not to a new chart-specific parser.

## Jev interface and realtime fallback

The state sent to Jev is strict JSON and contains:

- `episode`: current chart time/tick, key count, judge/mod, and judgment
  windows;
- `chart`: pattern family, offset, BPM segments, and snap resolution;
- `upcoming`: up to 64 notes with ID, effective column, absolute start/end,
  kind/pattern, group metadata, and time-to-receptor;
- `holds`: active hold columns/tails and head judgments;
- `recent`: the last eight signed judgments; and
- `stats`: head/tail histograms, misses, broken holds, accuracy, and pending
  counts.

There is one typed `Choice` question. `target_exactly` means use the reported
absolute timestamps; the two other choices apply only an 8 ms global shift and
are bounded so a questionable answer cannot miss by a whole tick. The exact
option is the safe default for malformed stand-in answers and model timeouts;
`BudgetExceeded`/`ModelUnavailable` are allowed to end the episode normally.
The harness never asks Jev to calculate beats, count notes, choose a lane, or
emit prose/JSON.

A batch is posted only for new future note IDs, while remembered hold tails are
added once the visible frontier makes them safe. While a policy call is in
flight, already-posted events cover the latency; after posting, the loop waits
on `ctx.wait_for_tick()` rather than spinning. If the optional batch call is
rejected by a pre-flight stand-in, it falls back to a legal `noop`, and no
exception escapes. This preserves the parent's no-Astra control path while
testing whether absolute-time batching, rather than another Jev prompt, fixes
the worst episodes.
