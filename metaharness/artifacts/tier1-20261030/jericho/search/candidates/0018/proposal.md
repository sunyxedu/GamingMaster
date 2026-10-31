parent: 0011

## Hypothesis

The leader's useful route decisions are being diluted by an evidence packet that
repeats the same prose and leaves too little wall time for later calls. In the
0011 Enchanter seed-2 trace, the controller reaches `Outside Gate`, tries
`open gate`, `unlock gate`, `examine chain`, `rezrov gate`, and then returns to
roads and the trail without a score event; the episode ends at 43 actions and
score 0. In the related 0015 seed-10 trace, 31 actions are mostly direction and
observation commands, four Astra calls time out, and the final call is cut off
with only about five seconds left. The 0011 packet sends the opening response,
current response, recent rows containing both, and up to fourteen long edge
records on every call (the trace records roughly 15k input tokens on the first
call). Repeated context increases latency and encourages route continuation
without a fresh puzzle diagnosis.

## What changed vs. the parent

0018 keeps 0011's repaired parser transport, directed frontier, exact response
ledger, fresh Astra sessions, bounded fallback, and total output decoder. It
compresses only the model-visible packet: the current response and opening
mission remain verbatim within fixed limits; recent history is four short rows;
score-changing rows and failures are retained separately; frontier and repeated
edge records carry short state/response excerpts. The game description and
belief notebook are also capped. Astra is asked for a 2--4-command commitment,
and the controller executes at most four commands before requiring a new
packet. Each model request is capped at 70 seconds and leaves eight seconds of
wall-clock headroom for acting and returning.

This is a latency/evidence-duplication experiment, not a new game strategy or
seed-specific route. The packet still includes exact latest parser text,
opening clues, score events, failed commands, and directed exits so compression
should not erase the information needed to interpret a puzzle.

## Evidence that would confirm or refute it

Confirmation would be lower input sizes and call latencies, more completed Astra
calls before the 1,800-second deadline, fewer stale direction suffixes, and
higher final native score on the weak Enchanter-like seeds at comparable action
counts. A useful trace should show a new packet after the gate/trail evidence
rather than another long route continuation.

Refutation would be unchanged or lower score despite materially shorter packets,
or a rise in parser errors because the omitted history was needed to resolve an
object. If score events disappear while calls increase, the packet needs richer
object provenance rather than further compression.
