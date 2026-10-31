parent: 0006

## Hypothesis

The leader's weak outcomes are uncertainty, not an execution defect. Its
`trace_seed13.jsonl` records `[5 3 2 4]` receiving `0 black peg(s), 3 white
peg(s)`, followed by `[5 2 3 6]` receiving `(1,2)`. The final `[4 2 3 5]`
has 18 surviving codes, expected score 0.5416667, and realized score 0.5.
Every `verified_step` reports `issue: null`. Seed 11 similarly scores 0.625
against a final conditional expectation of 0.725. The public-rule archive
already establishes that the leader attains the uniform expected-reward
optimum, 1915/2880. Retuning the objective cannot improve that optimum.

Final-iteration hypothesis: the leader does unnecessary exact work. Its 359
second probes fall into only 16 equivalence classes under the symmetries
fixing the opening. Searching the earliest member of each class should preserve
the leader's complete contingent policy, including solve and lexical tie breaks,
while reducing planning work and deadline exposure. Unlike prior objective
changes, this predicts identical per-code actions and scores when coupled to
the same private digit permutation, not merely equal mean reward.

## Change from parent

Keep the leader's uniformly sampled root digit permutation, numeric-reward
objective, complete terminal action set, exact TicTacToe negamax and single-step
controller. Construct the 48-element opening stabilizer from the public rules:
24 position permutations with compensating digit relabelings, times two swaps
of the absent digits. Partition canonical probes into orbits, validate coverage
and disjointness, and map each earliest representative through the sampled digit
map. Search 16 representatives instead of 359 probes. The first-clue posterior,
feedback partitions, total reward, solve count and immediate-solve indicator
are invariant under this group. Earliest representatives preserve the parent's
final tie priority; the selected representative's terminal search is unchanged.

This transfers the symmetry reduction explored in 0011 to the current leader
without transferring its different floor objective or deterministic opening.
No action table, training seed, hidden code or environment source is used.

Strengthen the continuation boundary: a missing predicted feedback branch,
changed branch belief, inconsistent solve certificate, or failure to consume
exactly one legal guess disables local control. A missing continuation must not
silently become a freshly optimized move. As in the parent, ctx.progress(),
native plies, persistent clues/board marks, reward and horizon verify every
action. Astra receives failure evidence and public observations for one legal
repair action; repaired actions are checked too. Model exceptions and arbitrary
responses use the parent's bounded legal fallback. No background work or
normal-path model calls are introduced.

## Confirmation and refutation

Independently run the original 359-probe search and the quotient search for
all ten non-solving opening-feedback branches. Couple digit maps and compare
selected probes, complete branch certificates and terminal actions. Replay all
360 public abstract secrets. Confirm losslessness only if every action and
outcome matches, not just the aggregate mean. Repeat with nontrivial digit maps
to detect an incorrect relationship between symmetry and lexical priority.
Measure cold and branch-search runtime separately from native dev latency.

Fault-inject missing/corrupt continuation certificates, rejected/stalled moves,
changed feedback, garbage model replies and model exceptions. Required preflight
and one native dev run check integration. Refute correctness if any ordinary
public branch differs from the leader, if native effect checks falsely fail, or
if bad certificates avoid the Astra handoff. Refute the runtime hypothesis if
orbit construction costs more than it saves.

No higher expected score, fewer game actions, or improved realized reward floor
is claimed. The same random policy can score differently on another independent
four-game sample. The gain sought is less redundant planning with explicit
certificate failure detection. Training, validation and held-out evaluation are
left to the evaluator. The complete harness and proposal precede validation;
measured results will be appended below.

## Results

The lossless-search hypothesis is supported. `check_equivalence.py` compares
the unchanged parent implementation against this candidate on all ten first-clue
branches under identity, a digit swap and a six-cycle. All **30 selected second
probes** and **210 terminal branch certificates/actions** match exactly. The
runtime always evaluates **16** representatives covering **359** legal probes.
Across these paired local searches, parent branch search took **7.013 seconds**
and candidate search **0.286 seconds**, about **24.5x faster**. Cold model
construction was approximately **0.187 seconds** for each. This is a local timing
sample, with concurrent checks running, not a guaranteed speed ratio or a
comparison of native episodes on different random maps.

The same audit replays **360 complete synthetic controller episodes** with
**1,078 clean effect checks** and no recovery calls. It reuses the independently
compared public branch plans in the test fixture to avoid repeated searches;
the deployed module does not share any cross-episode cache. Synthetic successful
solves omit the final ordinary clue, matching the documented native behavior.
The exact score counts remain **4 at 0.25, 32 at 0.375, 77 at 0.5, 85 at 0.625,
109 at 0.75 and 53 solves**. Mean stays **1915/2880 = 0.6649305556**, and mean
actions stay **1078/360 = 2.9944444**. There is no score-distribution gain.

Additional completed checks:

- `check_policy.py` independently checks all **129,600** feedback/payoff pairs,
  210 terminal certificates and the two digit-equivariance generators. All
  **720** digit permutations take each fixed secret to every abstract code
  exactly twice. Thus the parent's per-fixed-secret expectation and full reward
  distribution over private randomness are preserved.
- `check_recovery.py` covers all **787** terminal TicTacToe opponent paths with
  no losses and seven injected recovery classes: rejection, stagnation, raised
  action, altered feedback format, model timeout, backend error and garbage
  response. Each invokes Astra's context interface and finishes within four
  attempts. These are stand-in calls, not real-model recovery measurements.
- `check_certificates.py` crosses seven faults with four reply modes, giving
  **28** successful recovery cases. Missing branches, wrong beliefs, illegal
  final guesses, incorrect totals, incorrect solve counts, lost followups and
  unconsumed guesses each trigger exactly one model call and complete in three
  actions. Reply modes are legal text, garbage, TimeoutError and RuntimeError.
  A separate check ensures singleton posteriors do not bypass certificate
  validation, and ordinary omitted-feedback solves remain accepted.
- Required preflight passed: TicTacToe draw, four actions, zero calls/errors.
  Its warning about unexercised model answers is addressed by the fault tests.
- One native dev evaluation, `dev/20261031-133424`, scored **0.5 on 210** and
  **0.75 on 211**, mean **0.625**, with zero model calls, errors or rejected
  moves. The digest and both traces were read; all seven `verified_step`
  events have `issue: null`. Native Mastermind completed in **0.492 seconds**,
  with all 16 probe classes searched. The single sampled dev reward does not
  imply improvement over the parent's equal theoretical expectation. No second
  dev run, full evaluation, validation or held-out inspection was performed.

This is a careful computational and recovery improvement on the current leader,
not a new objective. The expected suite score remains **0.5824652778** under
the public model; leader 0006's larger observed training mean is a finite random
realization. Search interrupted by the deadline and Astra repairs prioritize
legal completion and lie outside the exact-equivalence guarantee. Under the
generous normal deadline the speed improvement may have no selection benefit,
since the evaluator breaks score ties by model calls and actions, not CPU time.
