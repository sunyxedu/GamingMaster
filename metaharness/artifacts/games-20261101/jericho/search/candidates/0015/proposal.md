parent: 0005

## Hypothesis

The plateau is partly caused by an impoverished recurrence key.  The frontier
controller treats a room's first lines plus native score as the state.  In
interactive fiction, learning a spell, opening a container, or changing the
inventory can leave both values identical while making a previously impossible
command valid.  A permanent `(room, command)` ban therefore risks suppressing
the needed retry.  The evidence motivating this is the repeated zero-score
probing in the 0011 Enchanter trace: after reaching `Outside Gate`, Astra tried
`rezrov gate`, `open gate`, and later returned to the same gate text while the
spell book still reported that rezrov was not committed.  In the 0011 seed-8
trace, the controller similarly spends turns at the temple (`in`, `open`,
`rezrov temple`, `enter`) while the room prose stays almost unchanged; the
useful change is in learned spells and objects, which the old key cannot see.

## What changed vs. the parent

0015 keeps the parent's bounded transcript, fresh Astra calls, directed exit
frontier, total output decoder, short model timeout, and conservative parser
handling.  It adds a code-side world version and inventory hint.  Every
substantive command (movement, manipulation, learning, giving, or a score
change) advances the version; observations such as `look`, `inventory`, and
`examine` do not.  Attempted edges are keyed by the versioned context, so a
failed `open door` can be reconsidered after a key is acquired or a spell is
learned.  A separate transition counter `(base room, command, base result)`
cuts a genuine three-way oscillation even when versions keep changing.  The
prompt tells Astra that old failures may become valid after a prerequisite
change and includes the compact inventory evidence.  A one-shot diagonal
direction alias repair (`northeast` to `ne`, etc.) remains available only after
an explicit parser rejection.

This is a state bookkeeping experiment, rather than another route-batching or
session-context change.  It uses no title-specific walkthrough, seed mapping,
or VM inspection.

## Evidence that would confirm or refute it

Confirmation would show a command repeated after a real prerequisite change,
with fewer “already tried” route drops, more score events after spell or
container transitions, and no increase in west/east loops.  Enchanter-like
episodes should revisit a gate or temple only after the inventory/spell hint
changes and then progress.  A higher weak-seed score at similar calls and
actions would support the hypothesis.

Refutation would be zero useful retries, or a rise in repeated movement because
the version is too permissive.  If traces show three-transition caps firing on
legitimate maze traversals, the state key needs richer observed room identity;
if retries remain absent while Astra keeps issuing speculative object verbs,
the limiting problem is puzzle interpretation rather than recurrence memory.
