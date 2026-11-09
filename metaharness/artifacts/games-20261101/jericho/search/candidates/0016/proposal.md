parent: 0011

## Hypothesis

The leader's useful puzzle reasoning is being throttled by oversized Astra
packets and low value observation turns.  In the worst Enchanter trace (seed 2)
the controller repeatedly planned `... look ... look` while the score stayed at
zero, and after learning/attempting `rezrov gate` it spent the remaining wall
time on three 75-second model timeouts.  In Detective seed 9 it alternated
hallway and closet transitions (`east`, `look`, `east`, `west`, `look`) while
the response and score remained unchanged.  The frontier guard catches exact
same-state edges, but it does not recognize this two-room cycle and the large
recent transcript makes later calls slower.

## What changed vs. the parent

0016 retains 0011's fresh Astra sessions, transcript-derived directed exit
frontier, positive-score suffixes, parser/hazard interruption, total command
decoder, and bounded timeout.  It shrinks each evidence packet to the current
text, opening clue, five short recent rows, ten frontier entries, and five
failure summaries.  The prompt asks for at most five commands and explicitly
discourages `look`/`inventory` repetitions when the latest movement already
describes the room.  The per-call cap is 65 seconds, leaving more wall time for
subsequent decisions after a slow answer.  A short exact transition history
detects A-B-A-B movement cycles and clears the stale plan so Astra must select
an unexplored exit or puzzle action.

This is a conservative throughput and recurrence fix: it changes no game rules,
seed mapping, title knowledge, or VM state and keeps one parser command per
turn.

## Evidence that would confirm or refute it

Confirmation would show fewer repeated `look` actions and fewer hallway/closet
or road oscillations, shorter model input/latency, and more score events before
the deadline at similar or lower call count.  The Enchanter seed-2 pattern
should trigger a cycle replan and leave enough time for a fresh spell/objective
decision after a timeout.

Refutation would be lower scores because a legitimate return trip is cut, or
because compact context omits a prerequisite clue and causes speculative
commands.  If traces still show long calls with compact packets, the remaining
limitation is Astra latency rather than transcript growth; if they show novel
rooms but no score, object/provenance memory needs a separate experiment.
