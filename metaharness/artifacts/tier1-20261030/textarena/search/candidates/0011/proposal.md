parent: 0003

## Hypothesis

A robust planner should commit to one guarantee for the complete decision tree.
Parent 0003 recomputes the strongest guarantee after the opening feedback, even
when that posterior is easier than the branch that limits the initial guarantee.
This extra conditional insurance can waste expected reward without improving the
worst possible outcome of the whole episode. Compute the largest achievable root
floor first, then maximize expected native score everywhere subject to that SAME
floor. This tests precommitted safety against dynamically tightened safety, not a
new risk coefficient or a refinement of leader 0006's randomization.

The leader's weak episode is not an execution bug. In
`0006/trace_seed13.jsonl`, `[5 3 2 4]` gets `0 black peg(s), 3 white peg(s)`,
then `[5 2 3 6]` gets `(1,2)`. Its 18-code final posterior has expected score
0.5416667 and zero solve probability; `[4 2 3 5]` realizes 0.5. All three
`verified_step` records have `issue: null`. The archive also establishes that
this mean-optimal policy still scores 0.25 on four of the 360 abstract codes.
Parent 0003 guarantees 0.375 but averages only 0.6392361. The question is how
much of that mean loss is necessary for the GLOBAL 0.375 guarantee.

## Changes

Build the complete three-guess public-information tree before the opening.
For each opening-feedback branch and each second probe, compute a terminal
mean-versus-floor frontier. Find the maximal floor feasible for EVERY opening
branch, then select each branch's maximum-mean continuation at that common
floor. Preserve exact expected reward first within that feasible set; action
count is the next tie-break, literal solve count is tertiary. All guarantees
are integer eighths and all 360 public abstract codes have equal prior weight.
No benchmark seeds, hidden state, sampled secrets or seed-derived priors appear.

Whole-tree compilation is made practical by an exact symmetry quotient. The
48 transformations that fix `[1 2 3 4]` (24 compensated position permutations,
times swapping unused digits 5/6) partition the 359 second probes into 16
orbits. Their feedback partitions, rewards and floor/mean frontiers are
isomorphic. Search one representative per orbit, while still considering every
legal terminal action. The partition checks coverage and disjointness at runtime.
The canonical opener is justified by the full public code-space symmetry.

Runtime follows the compiled contingent tree one action at a time, checking
posterior identity, legal actions and payoff certificates. The robust frontier
comes from parent 0003; the stricter public-board/clue, native horizon, successful
solve and score verifiers use the subsequently repaired controller from 0009.
Every action checks ctx.progress(), rejected moves and observation advancement.
A mismatch abandons compiled control and gives the evidence to budgeted Astra;
its replacement actions receive the same checks. Garbage output and caught model
failures fall back to an unrejected legal move. Calls have a 480-second timeout
capped by remaining wall time. Computation checks the deadline between probes;
there are no background workers or dependencies outside the standard library.

## Confirmation / refutation

Exhaust all 360 abstract codes using public rules, and independently enumerate
all 359 second probes to audit the symmetry reduction. Confirm if the root floor
matches parent 0003 while uniform expected score rises. Measure the exact cost
against the mean-optimal 0.66493056 policy; a small gap would demonstrate that
most of the parent's sacrifice bought unnecessary conditional insurance.
Refute if the floor falls, the orbit quotient changes attainable values, the
mean does not improve, or runtime verification flags ordinary observations.
The finite training/dev sample can still favor a weaker expectation; distinguish
synthetic guarantees from native scores. This does not claim to exceed the
unconstrained Bayes optimum.

Run required preflight, at most two native dev evaluations, and public-only
fault tests for parser surprises, rejections, stalls, garbage model replies,
timeouts and corrupted contingent certificates. No full evaluation or unseen
split results will be inspected. Harness and proposal are written before these
checks; results will be appended below.

## Results

The hypothesis is supported by exhaustive public-rule analysis. Root commitment
achieves **1912/2880 = 0.6638888889** mean Mastermind score with a **0.375 realized
floor**, compared with parent's **1841/2880 = 0.6392361111** at the same floor.
It recovers 71 of the parent's 74 lost reward-eighths relative to the unconstrained
optimum **1915/2880 = 0.6649305556**. The remaining price of removing every 0.25
outcome is only **3/2880 = 0.0010416667** expected Mastermind score.

A concrete parent trace isolates the mechanism: `0003/trace_seed15.jsonl`
gets opening `(1 black, 2 white)` and chooses `[1 2 4 3]`, logging
`guarantee: 0.5` and `expected_score: 0.5677083333333334` over 72 codes.
At the root-committed 0.375 floor, `[1 2 4 5]` raises that branch's exact mean
to **0.6440972222**. This releases 44 reward-eighths, most of the total gain.
This is an expectation comparison over public hypotheses; the parent happened
to realize 0.625 on that archived episode, so it does not establish per-secret
dominance or predict the new training outcome.

- `check_policy.py` independently constructs all **129,600** feedback/payoff
  pairs, enumerates all **359 second probes in each of ten unresolved opening
  branches**, and matches the compiled action/mean/action-count/solve rank at
  the global floor. It independently reproduces both the 0.375 root maximin
  bound and the dynamic parent's mean. The quotient is a verified reduction,
  not a heuristic shortlist. Local compilation took **1.08 seconds**.
- The score distribution across all 360 abstract codes is: **34 at 0.375,
  81 at 0.5, 86 at 0.625, 108 at 0.75, and 51 at 1.0**. One code solves on
  the opening and 359 use three guesses; mean actions remain **2.9944444**.
  The equally weighted theoretical suite mean is **0.5819444444**. These are
  public-rule calculations, not measured training or unseen-split scores.
- All **360 controller replays** pass exact action/observation verification,
  including successful solves that omit ordinary terminal feedback. Replays
  share only the audited immutable compiled tree to avoid repeating identical
  computation; native dev below exercises actual cold compilation. Exact
  TicTacToe traversal covers **787 terminal opponent paths** across both seats
  without any learner loss.
- `check_failures.py` passes **16** ordinary fault/reply combinations, **15**
  corrupted-plan/reply combinations, and **six** direct effect mismatches plus
  native successful-solve omission. Stalls, rejected/raised actions, changed
  feedback, missing branches, wrong beliefs/probes, illegal leaves and changed
  reward certificates invoke recovery. Replies include legal actions, garbage,
  `TimeoutError` and `RuntimeError`. These are stand-in model checks, not real
  Astra recovery measurements.
- Required preflight passes: TicTacToe draw, four actions, zero calls/errors.
  One native dev evaluation, `dev/20261031-131356`, scores **0.5 on 210 and
  0.5 on 211**, with zero calls, rejected actions or errors. Its digest and both
  step traces were inspected: all seven `plan_verified` events have `ok: true`.
  Cold Mastermind takes **1.18 seconds**. The dev sample does not show a score
  gain; the 360-code analysis establishes the general tradeoff. A second dev
  run was unnecessary. No full evaluation or unseen-split result was accessed.

The guarantee applies when the observed rules match and certified search finishes.
Astra recovery and deadline fallbacks prioritize legal completion and carry no
0.375 guarantee. Unlike root randomization in 0006, this is a deterministic
per-secret floor, not a worst-secret expectation over harness randomness.
