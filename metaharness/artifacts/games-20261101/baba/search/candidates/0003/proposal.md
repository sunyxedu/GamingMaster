parent: 0000

## Hypothesis

The leader’s lost puzzles are primarily a control-loop failure rather than a
lack of puzzle information. It paid one slow Astra call for every move: seed
103 used 34 calls and seed 107 used 27 before the 1800-second deadline, while
the successful episodes also spent one call per action. The model often had
the right local direction but the per-turn latency (up to 208 seconds) left no
time to recover when it wandered.

The trace makes the failure explicit: seed 107 ends with `"ended_by":
"deadline"`, `"acts": 27`, and a cancelled call whose error is
`ModelCallCancelled: model call cancelled: episode deadline`; seed 103 has the
same deadline ending after 34 acts. There is no harness exception, so the
available wall time was consumed by otherwise valid sequential calls.

## What changed

Candidate 0003 asks Astra for a queue of at most eight primitive actions. The
harness submits those actions individually, preserving the turn contract, but
checks the returned observation and `ctx.progress()` after every action. A
blocked directional action, an illegal action, or two unchanged observations
flushes the queue and sends the exact current board back for replanning. The
prompt makes rule edits and pushes short-horizon commitments, so a plan can be
stopped at a changed rule instead of blindly replayed. Action extraction is
labelled and bounded, with a safe fallback for arbitrary preflight output and
model-call failures.

## Evidence that would confirm or refute it

Confirmation would be fewer Astra calls and substantially less wall time on
the difficult training seeds, while retaining or improving solved puzzles;
the traces should show replans after real stalls rather than deadline endings.
Refutation would be plans repeatedly becoming wrong after one or two actions,
or a lower score despite the call reduction, indicating that the model needs
single-step feedback or that queue parsing is capturing rationale text.
