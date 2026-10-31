parent: 0001

## Hypothesis and trace evidence

The leader completes matches, but its approximate terminal EV search can magnify errors in estimated caller composition into a 200bb commitment. This candidate tests **staged pot construction**: restrict the action set for ordinary hands, preserve the option to reassess on the next street, and permit immediate stack pressure only for qualifying strong made hands. This is a different hypothesis from the parent's throughput and unconstrained response-conditioned sizing recipe; the existing evaluator, range inference, and asynchronous control loop are reused to isolate the sizing intervention.

Concrete evidence from `candidates/0001/trace_seed2207.jsonl`, its weakest completed training match (80.95 chips/hand):

- Step 40: J♠8♣ on T♦4♥9♦J♣, pot 1,250, chooses `raise_to 19375`. The logged alternative has `ev: 3438.3`, `fold: 0.308`, `called_equity: 0.669`, versus EV 1,480.7 for a raise to 3,125. The opponent shows Q♣J♥ and hero loses 20,000.
- Step 136: Q♥2♣ on 8♦4♥2♠T♦Q♣, pot 700, chooses `raise_to 19650`, with estimated called equity 0.848. The opponent shows J♠9♦ for a straight and hero loses 20,000.
- Step 152: K♣3♠ on K♦9♠5♣, pot 500, chooses `raise_to 19750`, with estimated called equity 0.646. K♥2♣ calls and improves on the river; hero loses 20,000. Hero was ahead at commitment here, so this loss alone is not evidence that the shove was negative EV.

These outcomes motivate testing uncertainty exposure; they do not prove the parent's EV estimates were wrong. The potentially costly approximation includes caller selection, finite particles, sampled runouts, and omitted future betting. Reducing variance is not the objective by itself: the cap must preserve or increase mean net chips across independent matches to be useful. Candidate 0003 also suffered an independent throughput problem: seed 3301 ended at the deadline with zero steps, and only one of its four matches completed. Its synchronous first audit is not adopted.

## Implementation

Ordinary postflop targets are bounded by `max(street_bets) + fraction * (pot + to_call)`, with fractions 0.75 on flop, 1.0 on turn, and 1.25 on river, clipped to the legal interval. Thus a high mandatory minimum raise can exceed the nominal fraction. This is a per-decision bound, not a cumulative hand loss limit; repeated raises and calls can still commit a stack.

The legal maximum is additionally considered when posterior equity is at least 0.90 and either (a) hero has trips or better before the river, or (b) hero has a flush or better on the river and the complete hand outranks the public board. A board-only full house cannot qualify. These are strong-hand eligibility tests, not proof of the nuts. The river rule deliberately leaves straights and two pair in the ordinary sizing branch. Preflop behavior and call decisions are inherited unchanged. The existing EV search chooses within this restricted set; the cap is also checked after ranking.

The rest of the parent is retained: standard-library exact best-five evaluation, public-history likelihoods, deterministic local sampling, legal action validation, and checking `ctx.progress()` after each action. Rejections, stalled observations, computation failures, range collapse, repeated opponent reraises, and large revealed showdowns supply evidence for Astra audits. Only reusable posture and lesson survive an asynchronous response; a stale action is never executed. Astra can choose `balanced`, `pressure`, or `cautious`, affecting entry/call behavior. It does not remove the sizing restriction. Model parsing and calls are exception-safe; timeouts are 125 seconds, above the archive's 80-second observed maximum. At most seven audits are attempted, with wall-time and call reserves. A legal passive fallback remains available.

## Validation and development evidence

- Final required preflight: `ok: true`, 126 actions, no exception, ended at the short preflight deadline. No System-2 answer was delivered; this is contract/local-policy evidence, not a completed benchmark.
- A focused replay using the public observation preceding archived step 40 selected legal `raise_to 1250` instead of the former shove. Garbage model text and a simulated `TimeoutError` both retained the legal proposed action without crashing.
- Dev call 1, seed 901: 100/100 hands, 746.79 chips/hand, 244 actions, zero illegal actions, no episode error. This used an earlier eligibility rule allowing very high posterior equity alone; it is not a score for the final strategy.
- Dev call 2, seed 902: 100/100 hands, 891.22 chips/hand, 233 actions, zero illegal actions, no episode error. It used the final strategic rule; only defensive local variable initialization changed afterward. Both dev runs started one asynchronous audit, which did not yield an applied answer before completion. Consequently real-model adaptation remains unverified by these runs.

Both digests and traces were inspected. The second dev trace includes a full-stack loss through a river call (step 219), despite the betting restriction. This confirms the cap is not a hand loss limit, and the unchanged call model remains a possible weakness. The positive scores are encouraging completion/value evidence, not a controlled estimate of improvement: the two runs used different revisions, and the first parent's seed-901 dev episode was incomplete. No full training evaluation or held-out deals were run.

## Confirmation or refutation

Confirm the hypothesis if the final harness completes all training matches without error and improves mean net chips and the weak-match scores while retaining strong-hand value. Inspect full-stack commitments, large losses, and calls to staged value bets as mechanisms, not substitute objectives. Refute it if the opponent would have paid much larger bets and the restriction consistently loses value, or if correlated posterior errors simply reappear as repeated capped raises and expensive calls. Use independent match seeds as the uncertainty unit; neither a single bad beat nor one profitable development match establishes the hypothesis.

Pre-flight repair: local range/equity loops could keep running across the 60-second deadline without a wall-time guard, so the harness now yields before the five-second grace period.
