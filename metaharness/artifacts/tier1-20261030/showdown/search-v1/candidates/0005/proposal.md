parent: 0001

## Hypothesis

After candidate 0001 repaired the fatal parser, the remaining losses are mostly a
decision quality and latency problem.  Its traces show Astra spending about a
minute per request on a large raw ledger, while the resulting free-form choice
can ignore the defensive value of the six revealed teammates.  In seed 282617,
after the first model timeout the policy moved Empoleon into Braviary and then
Chesnaught while Zebstrika's Supercell Slam was already revealed; in seed
333667, candidate 0003 selected a Clefable/Volcanion oscillation while Noivern
and Tinkaton were revealed.  A controller that computes a small tactical
shortlist should reduce these avoidable switches and leave Astra to resolve the
uncertain matchup.

## What changed vs. the parent

This is a structural candidate-ranking design.  Code builds current-request
options from legal attacks and switches, using the full type chart, STAB,
accuracy, priority, recovery/status/hazard descriptions, HP, Tera cost, and
the revealed foe's defensive typing.  Astra receives that shortlist plus a
bounded team/event ledger and chooses one exact current legal token.  Calls are
made at a new active/reveal checkpoint or every six accepted decisions, capped
at 18 calls, instead of paying for a large model call on every request.  The
local ranking remains the fallback for malformed, timed-out, unavailable, or
stand-in responses.  Forced replacement, hidden-trap rejection, reordered
switch slots, and arbitrary model prose are handled by re-observing and
re-ranking.

## Evidence that would confirm or refute it

Confirmation would be clean episodes with fewer Astra calls than 0001, traces
showing a meaningful attack and a defensive switch among the model's options,
and improved weak-seed HP margins or wins.  In particular, the model should
stop selecting a switch whose incoming type is worse than the best legal attack
when a revealed foe has already shown its damaging move.  Refutation would be
losses where the shortlist omits the actual winning move (especially hazards,
setup, or Tera), or where the compact prompt makes Astra repeatedly choose the
same locally dominated switch despite correct options; that would indicate the
next iteration needs a more exact damage/speed ledger rather than more call
frequency.
