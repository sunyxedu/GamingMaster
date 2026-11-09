parent: 0000

## Hypothesis

The incumbent fails before it can reason about the story because it confuses the
action channel with the parser command.  In the seed 0 trace, Astra first
returned `{"command":"east"}`, but the harness's generic `parse_action` reduced
that to the literal action `command`; the game answered “That's not a verb I
recognise.”  Subsequent answers (`command east`, `command("east")`, and plain
`east`) were also repeatedly submitted as `command`.  The same loop consumed
17–27 calls and ended at score 0 on every evaluated seed.  The seed 2 trace
spent the entire deadline in its first model call, despite the opening text
already stating the complete three-part objective.

## What changed vs. the parent

This harness separates parser-command extraction from action-channel names and
accepts JSON objects/lists, fenced JSON, wrappers, and plain command lines.  It
first compiles an explicit opening objective from the observed transcript into
short, dependency-ordered parser commands: movement, taking an uncarried item,
placing it in its target, and closing the target.  Each command is still
submitted and observed individually.  If the transcript does not expose a
compilable objective, or if a command produces a parser failure, Astra receives
a compact current-state request for a fresh JSON command plan.  A malformed or
failed model call falls back to bounded `look` steps and cannot escape the
episode contract.

## Evidence that would confirm or refute it

Confirmation would be a nonzero score with zero or very few Astra calls, and no
“not a verb” responses, especially on seeds whose opening text contains the
observed objective.  A score of zero after the compiled commands, or parser
responses showing that the task requires a different take/put/close sequence,
would refute the inferred plan and indicate that Astra recovery needs to be
engaged earlier.  A successful command plan on training seeds but failures on
unseen story variants would show that command extraction is fixed while the
objective compiler is too narrow.
