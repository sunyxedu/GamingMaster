parent: 0008

# Hypothesis

The leader's communication protocol is effective, but its one-turn utility
cannot price the future consequences of saving, drawing, and spending the last
clue. Preserve that protocol and compare choices by sampled terminal score.
Unlike candidate 0007, build the continuation on the current leader and carry
its persistent public play promises into every hypothetical game. Search must
respect each simulated actor's information boundary and compare actions on the
same sampled worlds.

# Evidence from the leader

In `0008/trace_seed14.jsonl`, turn 51 spends the last clue on a G2 color signal;
turn 52 plays R4, and turn 53 discards R5/card 35 at zero clues. The final red
height is four. Separately, Y2/card 48 is only played on turn 68, the final draw;
the last two actions play Y3 and Y4, stranding the saved Y5. W5, Y5 and G5 had
occupied one hand together since turn 8. The game ends 21/25 with all lives.
These are sequencing and access failures, not malformed actions or misplays.

Across the other training traces, W5 is discarded on turn 26 of seed 11, B5
on turn 10 and last B4 on turn 31 of seed 12, and G5 on turn 50 of seed 13.
Thus the remaining score gap combines lost physical ceilings and insufficient
time to execute preserved chains. No seed or hidden engine state is a runtime
policy input.

# Change from parent

Keep the literal joint posterior, color play promises, literal rank/save clues,
discard-index channel, stable IDs, legal dictionaries and isolated model
sessions. Add bounded paired rollout evaluation of prevalidated root actions.
Sample own hands without replacement conditional on literal AND public protocol
domains, shuffle the residual unseen multiset, and run the same protocol through
redacted future views. Copy only public promise state between simulated seats;
sampled private faces remain local to this one decision. No actual engine
implementation is read.

Retain the parent's action unless a paired advantage exceeds a sampling-noise
margin. Keep automatic fallbacks for low wall budget, incomplete search batches,
invalid model text and model/queue timeouts. Astra sees computed alternatives
and may arbitrate evidence-supported choices within the acting seat's session.

# Confirmation and refutation

Support would be fewer live-last-copy losses, earlier execution of prerequisite
chains, more completed colors, and improved terminal scores without protocol
misplays. Refute if continuation approximation, literal-posterior selection bias,
or noisy maximization leads to worse choices; also refute operationally if
search prevents natural completion. Log paired terminal estimates, uncertainty,
sample count and the parent alternative. Compare actual losses and endings,
not only optimistic rollout predictions.

# Validation

Harness and proposal were written before experiments. Observation-only rule,
promise-replay and redaction audits, preflight, and at most two permitted dev
evaluations will be recorded below. No training or held-out evaluation is run.

## Implementation and initial checks

The rollout continuation uses the parent's exact beliefs and action policy,
including the exact public discard codebook. To make this practical, the
matching dynamic program is replaced by an equivalent inclusion-exclusion sum
on subsets of at most five slots. Five hundred randomized posterior comparisons
matched the parent. This is an arithmetic optimization, not an independent-card
approximation. The candidate menus matched the parent on all 258 archived
training decisions before applying search.

The observation-derived simulator matched 254 nonterminal transitions, including
legal menus, masks, counters and scores. Separate fixtures passed for the final
draw/countdown, a five restoring a token, no replacement on perfect completion,
and zero-score bomb-out. A previous public snapshot remains unchanged after a
simulated discard, so decoding uses counts before the action. Sampling audits
checked 114 worlds against masks, promises, card conservation and actor masking.
Public codebook face-erasure and persistent promise replay matched the parent.

Initial preflight passed: natural game completion, 63 legal actions and no
errors. Its 23-point stand-in result is a contract check, not a performance
estimate; its low budget skipped rollout/advice branches. Explicit fake contexts
also passed garbage and out-of-range replies, valid choice parsing, model errors,
backend and outer queue timeouts, and initial/resumed separate seat sessions.

Search compares up to two choices per action family, at most 64 paired complete
worlds, with bounded per-decision compute and at least 12 complete pairs. An
alternative must exceed the parent by `0.15 + 1.5 * paired_standard_error`.
Incomplete batches are discarded. This margin measures sampling noise only;
policy-selection bias in the posterior and continuation-policy limitations remain.

## Completed dev result

The first permitted dev evaluation, `dev/20261030-011947`, completed **24/25**
in **63 actions and 443 seconds**, with all three lives, seven clues and no
error. Parent 0008's archived final dev result on this same deal was 22/25.
The final stacks were R4/Y5/G5/W5/B5: Y5 was played on turn 55 and G5 on turn 60
(the final draw). Thus the previously stranded yellow/green sequences were
executed before the countdown expired. This is paired evidence on one dev
seed, not evidence of a training or held-out improvement.

Search evaluated 46 decisions with 12–64 complete paired worlds (median 33.5)
and changed five parent choices, at turns 14, 15, 17, 35 and 49. The changes
were literal preparation clues, not new communication rules. Six discard play
signals were transmitted. Twenty additional hypothetical continuation games
from an archived observation produced 1,254 legal steps; all public promises
contained the simulated true card and every claimed-certain play was correct.
These are simulator/protocol tests, not additional real dev evaluations.

One Astra request exceeded the outer 120-second wait, was safely ignored, and
returned much later during automatic play. The backend eventually reported a
10.6-second latency, illustrating that service queue time differs from reported
model latency. No model contribution to the decisions is claimed; the harness
completed automatically after the timeout. The evaluation records no failed
backend call and no episode error.

## Residual failure and limits

The single live last-copy loss was R5/card 23 on turn 28. On turn 27 the search
estimated 23.00 terminal points for a rank-5 clue versus 22.54 for the parent's
own play, but its paired gain was only 0.458 with standard error 0.376 over
24 samples. The evidence margin correctly refused to treat that noisy estimate
as decisive; the following seat then discarded R5 from a domain estimated 80%
dead. This exposes an unresolved compute-allocation question, rather than a
protocol failure. A future iteration can allocate more samples to a promising
critical save without weakening the uncertainty margin.

The current search also skips an own certain play when its local value leads
the next alternative by more than 0.8. The archive contains cases where such a
play coexists with a threatened partner critical card, so this shortcut can
miss preventive sequencing. It is retained as part of the tested version.
Future simulations use the parent policy without recursively searching, and
literal-plus-protocol beliefs omit information in policy selection. Neither
sampling standard errors nor this single positive dev result establish that
those biases are small. Compute caps can also change sample counts with host
load. Automatic play remains available below the search wall-budget reserve.

No decision-policy changes followed the dev result. Only observation-snapshot
formatting and live-last-copy logging were added after that run was started;
the rule-transition and posterior audits passed again. One of the two allowed
dev evaluations was used. No training or held-out evaluation was launched.

Final preflight after the formatting/logging changes passed again: natural
completion, 63 legal actions and no error. A replay of the completed dev trace
validated all 63 posted actions against the protocol menu and independently
reconstructed all six discard signals from public pre-action snapshots. Every
visible promised card satisfied its decoded domain. `validate.py` retains the
observation-only inference, transition, posterior, terminal and adviser checks.
