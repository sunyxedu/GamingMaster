parent: none

Hypothesis: the incumbent's zero score is primarily an action-translation failure,
not deficient game reasoning. Its initial command vocabulary is frozen, so new
valid commands become unrelated defaults. Seed 100's second answer is `go north`
but the submitted action is `drop key`; the model repeats `go north` while the
harness repeats rejected drops. Seed 102 answers `insert spork into dresser` but
submits `drop spork`. Seed 105 answers `drop golf ball` but submits `examine
blackberry`. All nine training scores are zero, but this does not establish that
Astra cannot solve the quests. Seed 106 also loses its entire deadline before
executing a command, highlighting latency risk.

This fresh architecture tests a distinct hypothesis beyond a dynamic-vocabulary
patch: one semantic compilation of the literal objective should suffice for many
generic quests, while online admissibility acts as a constraint solver for exact
names, matching keys, and local prerequisites. Astra emits an ordered program of
command templates, optionally guarded by explicit room names. The interpreter
binds each template against the CURRENT commands. It may resolve an unspecified
key only when there is exactly one admissible matching command, and may open or
unlock a named source/destination. It verifies each resulting state before moving
the program counter. It records rooms and directed observed edges, and uses BFS
only on those edges when satisfying room guards. There is no reverse-edge guess,
seed lookup, hidden state, or generic collect-all policy.

Model calls are reserved for compilation, missing information, incorrect
postconditions, exhausted plans without native victory, and loop/progress alarms.
Each repair includes exact objective/current commands, observed room snapshots,
verified and failed action history, remaining program, and ctx.progress(). A
fresh self-contained session avoids accumulated erroneous conversation history.
Eating requires an exact objective quote naming the food and a consumption verb.
Invalid output never defaults to an unrelated command. Timeouts are caught;
240-second call timeouts exceed the observed 132-second maximum, subject to the
remaining episode deadline. There are no worker threads.

Confirmation: high native score with roughly one call per literal multi-step
quest, and logs showing new commands bound after state changes. On deviations,
the interpreter should stop at the first unverified effect and resume after an
Astra repair. Refutation: interpretation errors or underspecified objectives
routinely require many repairs; admissibility does not establish assumed matching
key/postcondition semantics; or guards incorrectly skip necessary quest events.
Compare calls, actions, rejected commands, recovery count and actual score, not
only apparent progress. A successful training result still requires validation
on unseen objectives and names; no fixed action sequence is encoded.

Validation: preflight passed (garbage/arbitrary model output returned safely with
no action). `py_compile` passed. One development evaluation on seeds 300–302
scored 1.0 on all three, ended all episodes by native victory, and used exactly
one Astra call each (2, 4, and 10 actions respectively), with zero invalid
commands or model failures. Seed 301 confirms that eating is allowed when an
exact objective quote authorizes it; seed 302 exercises a multi-room ordered
program and an object retrieved through nested lock/open prerequisites. These
seeds confirm initial compilation and current-command binding, but do not test
repair after a bad postcondition.
