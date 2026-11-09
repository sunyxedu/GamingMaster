parent: 0002

## Hypothesis

Candidate 0002 had the right idea (local push search) but its state compression likely
discarded useful information. It canonicalized every player reachable region to the minimum
coordinate, then restored the player to the pushed box square only when expanding a successor.
That representation can make walking cost and the side from which a later box is approached
inconsistent. Its traces show the practical symptom: despite 121 Astra calls, seeds 100, 105,
106, and 107 stopped with only one or two boxes on goals. Candidate 0003 showed the related
failure mode after a model mistake: repeated walking consumed all 2,000 actions without a new
push. I hypothesize that retaining the actual player square in the push-state search, while
using Astra only after a bounded search, will produce complete local plans on more layouts and
avoid handing the model a state already trapped in a walk loop.

## What changed vs. the parent

The search state is now `(sorted boxes, actual player coordinate)` throughout. A successor
stores the real post-push player square (the box's old square), and its cost counts every walk
and push action. The reverse-push matching heuristic and static corner/dead-cell pruning remain,
but the parent's dynamic 2x2 pruning is removed from successor generation because it can reject
valid corridor arrangements. The search gets a bounded 14 seconds and 45,000 expansions; hard
levels fall back to Astra macros. Model output is parsed only from an explicit `ACTIONS:` line,
validated against each observation and a local transition predictor, and cleared after pushes.
Timeouts and garbage responses use a legal static-safe fallback, so pre-flight stand-ins cannot
escape as errors.

## Evidence that would confirm or refute it

Confirmation would be more levels solved by the local phase, fewer Astra calls than 0002, and no
2,000-action episodes dominated by repeated walking. Traces should show the first actions before
any model request on layouts that fit the search, and successor player coordinates matching the
returned observations. If search time rises while the weak seeds still fall back to long model
macros, or if preserving exact player positions makes the frontier too large, the hypothesis is
refuted; that would point toward component-level canonicalization with a richer side/access
summary rather than full-coordinate states.
