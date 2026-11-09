parent: 0005

## Hypothesis

The leader's remaining losses come from route selection, rather than from the
parser transport or from Astra being unable to interpret clues. In the 0005
enchanter seed-2 trace, Astra spends the episode on `west`, `look`, `east`, and
another `look` while the Long Road response is unchanged. The same controller
can preserve a stale suffix through several ordinary observations. Candidate
0009 detects some cycles, but its response-conditioned programs still spend
many calls replanning the same hallway and its 120-second calls frequently
timeout. A transcript-derived frontier should make unexplored exits explicit,
and a short model deadline should leave time to act after a slow request.

## What changed vs. the parent

0011 keeps 0005's repaired command transport, compact provisional notebook,
positive-score route continuation, and total model-output decoding. It starts a
fresh Astra session with each planning call, asks for one objective plus a
3--8-command route, and gives Astra a directed frontier of exits extracted from
observed prose. Code records state signatures, attempted edges, repeated
responses, score changes, parser/clarification/menu/hazard text, and a bounded
exact transcript. A route is interrupted on parser context failure, score loss,
or a repeated edge in the same textual state. If Astra times out or returns
garbage, code explores only an exit explicitly named in the current response,
then at most one `look` and one `inventory` at that state; it never turns
arbitrary prose fragments into noun probes. Each call has a 75-second cap so a
single slow inference cannot consume the whole 1,800-second episode.

This is a structural frontier-and-edge experiment, distinct from 0005's
longer batching and 0009's model-authored conditional program. It contains no
title, seed, walkthrough, or VM-state assumptions.

## Evidence that would confirm or refute it

Confirmation would show fewer alternating direction/observation loops, more
unique state edges per Astra call, and useful fallback movement after a model
timeout, with no increase in deaths or parser errors. The enchanter seed-2
pattern should trigger a replan or a fresh observed exit instead of repeating
the Long Road. Similar or higher final native score with fewer failed model
calls would support the latency part of the hypothesis.

Refutation would be lower scores because legitimate return trips are blocked by
the textual state key, or because the exit frontier spends turns on exploration
when a puzzle action was required. If traces show novel rooms but no score,
the next experiment should improve object/provenance memory rather than add
more cycle protection.
