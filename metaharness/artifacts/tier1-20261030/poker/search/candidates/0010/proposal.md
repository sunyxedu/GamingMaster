parent: 0005

## Hypothesis and trace evidence

The leader's particle compression can remove rare stronger hands before the river, while sampled opponent equity introduces further caller-selection error. Its strong-hand exception then magnifies that error into stack-sized bets. Candidate 0010 tests a complete-support river reconstruction attached to the leader's successful preflop/flop/turn policy.

In the leader's weakest training match, `candidates/0005/trace_seed4409.jsonl`, step 144 settles hand 64 for -20,000. Hero 7s9d on 6d7d2d3dKh bets `raise_to 18963` into 2,074. The logged posterior has only 122 combinations; the shove is ranked at EV 7,784.2 with called equity .837, above EV 3,207.7 for a raise to 2,592. Opponent shows 8cTd: a ten-high flush beats hero's nine-high flush. The observation before that decision is the `obs` at step 143. This is evidence of exposure to posterior error, not proof that any particular losing shove had negative expectation.

Candidate 0009 establishes a useful exact river-equity computation, but its proposal explicitly omits flop/turn action selection. This candidate borrows the rank-distribution calculation while retaining the leader's likelihoods for the complete public betting sequence. No hidden game state or implementation source is used.

## What changed

On the river, reconstruct all 990 opponent-hole combinations compatible with hero and the board. Replay the full hand's public history from unit prior weights. Earlier-street likelihoods retain the parent's deterministic Monte Carlo estimates and soft thresholds, so the posterior remains approximate. Rebuilding support prevents an absent flop particle from remaining impossible forever.

Compute each hypothetical opponent hand's uniform river equity exactly using the rank distribution of 1,081 board-compatible combinations, subtracting combinations sharing either hypothetical hole card. Hero cards remain possible unknown cards in the opponent's calculation, respecting its information boundary. Actual hero range weights exclude hero-blocked opponent combinations. Exact showdown results and uniform equity do not make the action likelihoods exact.

River EV includes fold, call, check and legal bet sizes. A modeled opponent raise has one aggregate fold/call continuation across its entire range; no per-hidden-hand clairvoyant response is allowed. Ordinary sizing retains the parent's 1.25-pot cap. The maximum target is additionally considered only for a flush of king-high rank or better, or a full house/quads/straight flush, improving the public board, with posterior equity at least .93. An over-cap target must also have estimated called equity at least .97. Flush rank can partly come from the board; the complete caller-distribution check is therefore the stronger protection. Preflop and flop/turn sizing and calling remain inherited from 0005, including its early monster exception. There is no separate pair cap or arbitrary river calling floor in the final design.

Checking out of position is still approximated by immediate showdown value; later opponent bets trigger a new solve. The raise model includes only one raise and an approximate minimum target. These are deliberate limitations, not a complete game solver.

The inherited loop validates every action and checks `ctx.progress()` afterward. Rejections, no progress, range collapse, computation failures, repeated opponent reraises and large revealed showdowns request Astra audits. Asynchronous advice changes reusable posture; stale action advice is discarded. At most seven audits have a 180-second timeout and a 240-second wall reserve, exceeding the latest 73-second latency and the leader's 125-second timeout. Inference and parsing failures retain the local action. Search checks the deadline and returns with a 15-second reserve.

## Verification and falsification

Required preflight passed (`ok: true`, 76 actions, harness_return, no exception). The short dry run delivered no System-2 answers and does not establish match completion or real-model adaptation. Syntax compilation also passed.

Replaying only the public observation before leader step 144 selects legal `raise_to 2592`, not 18963. Full support identifies 211 stronger combinations, posterior equity .8858, and called equity .848 for the selected bet. The solve took approximately 2.4 seconds locally. Ten hypothetical opponent river equities agreed with direct enumeration over all 990 legal opposing hands. Garbage model text and a simulated TimeoutError both retained the legal action without exception.

Confirm if complete, error-free independent matches improve mean and weak-seed net chips while preserving strong-hand value. Refute if reconstruction's added CPU prevents completion, earlier likelihood errors dominate despite complete support, or tighter overbet eligibility forfeits more value than it saves. Examine river loss concentration, called equity, support size and model recovery as mechanisms; do not use hand-level variation as independent-match uncertainty. No full training evaluation or held-out results were accessed.
