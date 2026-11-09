parent: 0002

## Hypothesis

After command transport was repaired, a remaining source of lost score is
state cycling and speculative commitments. Candidate 0005's enchanter seed 2
trace alternates `west`, `look`, and `east` while the response remains the
same Long Road description; it finishes at native score 0 after 15 turns.
Candidate 0005's library seed 11 trace also spends turns on `enter rare books
door`, `go through door`, `enter door`, `west`, and repeated inspections after
the door state has already changed. Candidate 0002 demonstrates the other
side of the problem: its conditional program followed the model's unverified
`bozbar snake` guess and the story explicitly says the dragon knocks the
player over the edge, ending the episode at 10/51.

The experiment is whether a directed room ledger and conservative program
interruption can preserve useful response-conditioned batching while preventing
alternating movement loops and unsubstantiated irreversible actions. The room
ledger is transcript-derived and treats a repeated room description at the
same score as a cycle signal; it does not assume that a room name is globally
unique.

## What changed vs. the parent

0009 keeps 0002's model-as-policy-compiler structure, but adds an
episode-wide edge ledger. Each observed transition is keyed by the current
room signature, command, and normalized response. Repeated blocked edges are
shown to Astra and interrupt a program. A state signature excludes native move
count, so returning to the same textual state at the same score is detected
even when the model alternates directions. The ledger is used as a replanning
signal rather than a permanent ban: score or descriptive changes create a new
state key.

Programs are short and response-conditioned. Parser errors, menus,
clarifications, hazards, empty responses, score losses, and repeated outcomes
return control to Astra. Commands that can discard, surrender, consume, or
fight over a unique object require an explicit `proceed:true` gate in the
model output; otherwise the harness pauses for a safer plan. Exact evidence,
opening text, inventory, score events, failed commands, and provisional memory
are carried in a bounded packet, and each call starts a fresh session. The
decoder and all model/action paths are total for arbitrary stand-in output.

## Evidence that would confirm or refute it

Confirmation would show fewer alternating movement sequences, no repeated
room-state cycles, and fewer deaths from speculative operations, while still
executing several useful program nodes per Astra call. The enchanter seed 2
pattern should trigger a cycle replan after the first return rather than spend
the remaining route on `look`/direction churn. A score increase over 0005 on
weak seeds without a rise in parser failures would support the hypothesis.

Refutation would be lower scores because legitimate return trips are mistaken
for cycles, or because the proceed gate suppresses necessary puzzle actions.
Repeated replans that still choose the same edge would indicate that response
text alone is too weak a state key and that object/inventory provenance needs
to be represented more explicitly.
