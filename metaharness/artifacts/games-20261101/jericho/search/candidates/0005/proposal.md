parent: 0003

## Hypothesis

Candidate 0003 fixed the action transport bug and reached real points, but its
route execution is still too conservative. The controller clears a pending
batch whenever the native score changes. In the seed 1 trace, successful actions
such as `read paper`, `west`, `east`, `take wood`, and `take note` produced
positive score changes, yet the episode used 18 Astra calls for only 22 turns.
Seed 2 similarly spent all 40 calls for 57 turns. A positive score event is
usually evidence that the current dependency chain is working; it does not by
itself make the next movement or manipulation stale. The scarce resource here
is slow System-2 calls, rather than parser turns.

## What changed from the parent

0005 asks Astra for a longer, sequential route (up to 12 commands) with
optional literal response checks and explicit stop points. It retains the route
through positive score changes, while cutting it on parser errors, clarification
prompts, hazards, negative score changes, failed expected-response checks,
repeated identical command/response pairs, or terminal flags. It keeps a
timestamped exact transcript and a compact notebook, and sends both back to
Astra at each replan. Command decoding is total over strict JSON, fenced JSON,
tagged lines, bare inputs, and arbitrary pre-flight garbage; lifecycle and
compound inputs are rejected. Model and action failures are caught so deadline
or stand-in behavior cannot escape as an episode error.

This is a structural batching experiment, distinct from 0003's score-change
invalidation and 0002's branching policy compiler. It uses no game-specific
route, VM state, walkthrough, or seed logic.

## Evidence that would confirm or refute it

Confirmation would show more actions per Astra call, fewer calls ending with a
successful score change, and equal or higher final native score without a rise
in deaths or stale parser loops. Traces should show a route continuing across
positive rewards and replanning only after actual evidence changes. Refutation
would be long suffixes that become unsafe after ordinary rewards, repeated
unchanged responses, or lower scores from carrying stale plans. If so, score
changes need finer classification (for example, room-entry rewards versus
object transformations) or single-step replanning should return.
