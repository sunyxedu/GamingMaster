parent: 0005

## Hypothesis

The leader's long routes fail because the route contract is too weak. Astra
usually emits several ordinary commands with empty `expect` fields, so the
executor has no machine-checkable boundary between a command that worked and a
command whose assumptions are now stale. In the 0005 enchanter seed 2 trace,
the controller submits `west`, `look`, `west`, `look`, then `east` while the
Long Road prose and score remain unchanged. In the 0008 detective seed 9
trace, a route reaches `north` from the street and immediately produces the
death text about two men attacking; the preceding route had no response guard
that could have forced a replan at the dangerous location. These failures are
consistent with committing a plausible suffix rather than verifying each
transition.

## What changed vs. the parent

0012 keeps 0005's repaired parser-command transport, exact transcript, score
bookkeeping, bounded model calls, and total handling of model/action failures.
It changes the planning protocol to a transactional route. Astra returns a
short list of command records with literal `expect` anchors, optional `avoid`
anchors, and a risk/stop flag. Code allows a suffix command only when its
record has an evidence anchor, then checks the immediate parser response before
retaining the next command. A missing anchor, parser error, clarification or
menu, hazard, negative score, explicit avoid text, or repeated textual state
cuts the route and sends a fresh evidence packet to Astra. Thus a bare list
degrades to its first command instead of becoming an unchecked macro. Sessions
are fresh, and the prompt carries opening text, recent exact observations,
failed guard transitions, score, and a provisional notebook. The contextual
`q` answer is handled only after a live help/menu response explicitly asks to
resume.

This is a transition-verification experiment, distinct from 0010's broad
cycle counter and 0011's exit frontier. It does not use a walkthrough, title or
seed branch, VM state, snapshots, or hidden game data.

## Evidence that would confirm or refute it

Confirmation would show `guard_ok` transitions, fewer stale suffixes after
room or score changes, and fewer alternating movement/`look` loops or deaths at
similar actions per Astra call. The enchanter Long Road pattern should cut a
route when its expected text is absent or the state repeats, while ordinary
multi-step routes with stable literal anchors should continue. A score lift on
the weak training seeds without increased parser failures would support the
hypothesis.

Refutation would be lower scores because Astra cannot supply useful literal
anchors, or because legitimate sparse-text transitions are cut too often. If
guards are mostly empty or mismatched despite better prompts, the remaining
problem is clue/object planning rather than stale route commitment; if guarded
routes still die, the risk labels need a stronger code-side policy.

Pre-flight passed with arbitrary stand-in replies and two posted recovery
actions; all model parsing and menu handling returned cleanly.
