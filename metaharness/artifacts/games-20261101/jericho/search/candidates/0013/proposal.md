parent: 0011

## Hypothesis

The leader loses score when Astra reaches a puzzle but has forgotten the
precondition it learned earlier. In the enchanter seed 2 trace, the controller
spends 35 turns reaching Outside Gate and then submits `open gate`, `unlock
gate`, and `rezrov gate`; the response says “The gate is secure; it cannot be
unlocked” and the next packet still treats rezrov as the likely solution. The
packet only retains eight recent responses, while each new model notebook
replaces the previous one. Library seed 11 shows the same information loss in a
different form: after unlocking and opening the rare-books door, Astra repeats
`enter door`, `go through wooden door`, and `take book` without preserving the
observed directional clue that `south` enters the room.

## What changed vs. the parent

0013 keeps 0011's repaired command transport, directed frontier, fresh Astra
sessions, bounded transcript, total output decoding, and route interruption on
parser errors or hazards. It adds three structural controls:

* A safe opening reconnaissance action (`inventory`) is posted before the first
  plan unless the opening text is a menu or clarification. This exposes starting
  tools and spells before route selection.
* Code maintains an append-only, bounded durable-evidence ledger. Inventory,
  look/examine/read/manipulation responses, score events, and explicit failures
  remain visible across calls. Previous model notebooks are retained as a short
  history instead of silently discarded.
* A no-reward counter is included in every packet. After ten actions without a
  score increase, the current suffix is cut and Astra is explicitly asked for a
  named prerequisite or information-gathering interaction. Failure text also
  recognizes locked objects and unknown spells; code refuses to
  repost the exact same command while its failure response is still current.
* When a response explicitly presents the Inform help screen and says `Q =
  resume game`, code submits that contextual answer once before replanning.

The ledger is transcript-derived and bounded; it does not assume a title,
solution, seed, VM state, or walkthrough. Inventory is skipped when a live
menu/clarification is visible so the parser's prompt mode is preserved.

## Evidence that would confirm or refute it

Confirmation would show inventory or starting-tool facts appearing before the
first major route, fewer repeats of a failed puzzle command after a replan, and
higher score on the zero-score enchanter/detective seeds without a rise in
parser failures. The enchanter gate should cause a new prerequisite search after
its secure-gate response rather than another blind `rezrov gate`; the library
route should retain the successful `south` transition.

Refutation would be a score drop caused by the extra inventory turn, or traces
where the ledger grows the prompt and increases timeouts while Astra still
repeats stale commands. If durable evidence is present but model choices remain
unchanged, the failure is strategic puzzle reasoning rather than memory and the
next experiment should test a stronger code-side action relevance filter.
