parent: 0004

# Hypothesis

Mean-optimal Mastermind policies remain exposed to low realized rewards under a
three-guess horizon. Parent 0004 only insures tied mean-optimal actions; the
leader 0006 merely randomizes a mean-optimal policy. This experiment instead
maximizes the probability of reaching the public draw benchmark, score >= 0.5,
and then maximizes exact native mean among equally reliable policies. The
hypothesis is that the parent gives up a substantial reduction in weak outcomes
for a small gain in expected score. This is an explicitly different objective;
it does not claim to beat the proved uniform-mean optimum.

# Change from parent

Retain parent 0004's public posterior, full probe enumeration, contingent leaf
certificates, exact TicTacToe minimax, one-action execution and Astra recovery.
Replace its mean-first ranking by lexicographic (-count(score < 0.5), total
numeric reward, lower-tail counts, solve count). Evaluate ALL legal terminal
actions for this ranking, including eliminated secrets. Branch histograms add,
so optimizing the same tuple in every leaf and at the probe gives the exact
finite-horizon reliability optimum. The fixed 0.5 threshold is the game's
public draw utility, not a number selected from seed outcomes. The canonical
opening and deterministic tie order are inherited from parent 0004; there is
no random remapping, fitted seed prior, or lookup of training codes.

Every executed action checks ctx.progress(), old board/clue persistence, native
move count, legality, posterior consistency, and score/horizon. Any mismatch
hands public evidence to Astra through ctx.think for one-action replanning;
subsequent actions receive the same checks. Garbage replies and model timeouts
fall back to an unrejected legal action. No background tasks or external model
calls are used.

# Confirmation and refutation

Exhaust the 360 abstract distinct-digit codes using only the stated rules.
Compare probability below 0.5, exact mean, minimum score and solve rate against
parent 0004. Confirm a meaningful reliability/mean tradeoff if weak outcomes
fall substantially with a modest mean loss. Refute practical value if the mean
loss is large, if the safety rate barely changes, or if ordinary observations
violate the model. Native dev runs and required preflight check integration;
fault injection checks the recovery path. Keep synthetic all-code metrics
separate from evaluation outcomes. No full evaluation, held-out or validation
results will be inspected.

# Evidence and validation

Implementation and proposal written before running validation. Detailed trace
evidence and measured checks will be added below.

# Validation performed after implementation

The public-rule exhaustive check (`check_search.py`) covered all 129,600
feedback pairs and all 360 possible secrets. Parent 0004's distributional
policy has mean **0.66493056**, minimum **0.25**, and 33/360 outcomes below
0.5. The reliability-first policy has mean **0.63576389**, minimum **0.375**,
and only 11/360 below 0.5 (46/360 solves versus 50 for the parent). Mean
loss is therefore 0.0291667 while the low-tail count falls by 22 codes. Mean
actions are 2.99167 versus 2.99444 because some branches solve on the second
move. This confirms the proposed tradeoff and refutes any claim of improving
frozen expected score.

Preflight passed on a four-action TicTacToe draw with zero errors and zero
model calls. The inherited synthetic recovery suite passed 16 fault/reply
combinations; 15 additional contingent-plan corruption cases (prefix, missing
branch, belief, illegal leaf, histogram crossed with garbage, legal reply and
TimeoutError) invoked Astra recovery and terminated in three actions. Four
public effect mismatches were detected. No full evaluation or held-out result
was run.
