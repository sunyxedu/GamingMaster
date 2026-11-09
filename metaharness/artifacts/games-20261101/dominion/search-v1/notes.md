
### Candidate 0003 — public-belief rollout experiment
- The naive candidate never finished a training game: its worst trace stopped on turn 13 with six Provinces left after 30 steps and 1,800 seconds. Completion rate must be checked before interpreting its -4.5 mean as playing strength.
- A structured, bounded policy can be requested from Astra while local code handles repeated prompt kinds. Legal selections must be rebuilt from current occurrence IDs; never cache action IDs across prompts or episodes.
- A separate rollout model can use ownership minus visible zones to sample the agent's unrevealed deck. Opponent zones require a belief distribution, not a claim that public ownership reveals the hand. Search evaluates final VP margin, not proxy money or a custom win-weighted score.
- The supplied preflight passed with 49 steps but no System-2 call (its short wall budget activates this candidate's reserve). Therefore arbitrary-output and timeout handling need explicit local checks as well; preflight alone does not exercise that path.

### Candidate 0001 — reusable plans and lossless prompt execution
- The parent has a second failure beyond latency: `trace_seed1301.jsonl` lines 15/16 reduce Astra's four-card Chapel selection to one posted ID; lines 30/31 reject valid JSON by posting bare `choose`. Preserve complete occurrence-ID lists, then validate min/max locally.
- All six parent games hit deadline before completion. A Sentry discard pass alone took 224.9 seconds (seed1301 line 87). Routine nested prompts should not consume strategic model calls.
- Fresh, compact planning snapshots can carry the previous explicit deck plan without accumulating a conversation full of repeated rules. Recheck ownership targets and affordability after every purchase; cached option IDs are unsafe.
- The short pre-flight passed but its wall budget disabled planning. Separate offline checks exercised malformed JSON, arbitrary text, TimeoutError/RuntimeError, and all 140 archived start/step prompts; all produced legal-sized distinct selections. The four-card Chapel regression now selects all four IDs.
- VP endings require whole-turn basket arithmetic, including Gardens thresholds and turns-started tiebreaks. A bounded public-state search can handle this without guessing the unrevealed deck order; it must explicitly disclose search caps.
- Requested model timeouts did not bound the first dev attempt's observed wait: both first calls remained pending for over eight minutes. A separate 150-second wait on a ctx.spawn model future, followed by disabling planning and finishing locally, completed both final dev games without errors/pending tasks (+17, -11; mean +3). No model answers arrived, so this validates recovery, not planner quality; retain that distinction in later comparisons.

### Candidate 0002 — specialist controller with selective tactical criticism
- The worst baseline's turn-13 deck had Chapel, Festival, Laboratory, Militia, two Sentries and Witch but only one Gold, versus five opposing Golds. That motivates a restricted economy specialist, but does not prove engines fail: the baseline never completed and its parser also corrupted thinning.
- A Gardens strategy needs an explicit third depletion pile. After Gardens and Estates, continuing to prefer generic economy can miss the rush ending; this candidate targets Workshops and checks exact public VP before ending.
- Requested model timeout and total waiting time need not coincide. The first dev traces remained at the initial model request beyond 150 seconds; a separate main-thread wait limit around `ctx.spawn` preserves the ability to finish locally.
- Tactical critiques can return a current legal purchase without requiring a reusable target deck or model-generated action code. Validate the whole ID list against current bounds, and disable further reviews when a pending call has been abandoned.
- Preflight's short wall budget bypassed all model reviews. Direct fake-context checks are needed to cover malformed JSON, duplicate/unknown IDs, TimeoutError/RuntimeError and pending-call reserve handling; archived public prompts also catch multi-card selection regressions.
- The completed 0002 dev run refuted automatic Gardens specialization: scores -21/-12 despite 51 agent VP in both games, zero illegal choices/errors, and completion in 180.6 seconds. Gardens emptied early but deck dilution delayed Workshop depletion until the opponent was ahead. Final fallback is money/attack on every kingdom; Gardens requires a valid critic opt-in. No additional dev run remains to measure that final default change, and neither completed dev episode received model advice before the independent wait cutoff.

### Candidate 0007 — lead-aware race timing
- A race lock at Province count five can be premature: candidate 0006's seed-1101 trace had only two opponent Provinces, while seed-1300 bought a Duchy while already far ahead.
- Public VP and Province ownership support a smaller urgent set: opponent three Provinces, opponent two with three left, a four-VP deficit with four left, or two Provinces remaining.
- Preserve rollout control for ordinary green timing; forcing Duchy whenever three Provinces remain conflates pile pressure with an actual end threat.
- Restrict WM Moat defense to turns with no affordable green card so protection does not displace scoring.
- Pre-flight stand-in models can finish a legal episode without exercising Astra; strategic validation still requires evaluator traces.

### Candidate 0005 — opponent-aware nonblocking planner
- Public opponent identity is useful strategic state: WM-1 justifies Witch/defense, SM-1 makes one Smithy a coherent mirror, and BM-1 rewards payload and economy. This is a structural test rather than a threshold tweak.
- A model future must never gate a turn-game loop. Candidate 0001/0003 traces show planner calls pending for 60–150 seconds; polling while local code acts preserves completion and lets a fast answer improve later purchases.
- Keep model output declarative and bounded. The executor owns occurrence IDs, exact multi-card choices, and ending baskets; this avoids the parent’s parser loss of four-card Chapel selections.
- Candidate 0003’s rollout policy completed quickly and scored well, but its trace still had the agent at 9 VP versus an opponent at 15 VP with five Golds versus two. Completion alone does not establish economy quality; inspect acquisition mix and opponent matchup.

### Candidate 0004 — exact finish gate
- The strongest prior controller's rollout values can be strategically wrong near termination: sampled hidden deck payback competes with a deterministic winning Province or three-pile ending.
- Public ownership, pile counts, VP (including Gardens/Curses), and turns-started are sufficient to certify a winning current-turn ending without a model call.
- Keep model advice optional and bounded; the leader's only Astra call often remained pending and did not change play, while local search drove the score.
- Increasing Province/Gold heuristic values alone is weaker than an explicit end-condition gate because rollouts can still overturn a heuristic choice.

### Candidate 0006 — opponent race lock
- The leader's only loss on `base_intro` was a clean timing loss: BM-1 owned four Provinces when the agent continued buying terminals, then the Province pile ended at a one-VP deficit.
- Public opponent ownership and remaining pile counts are enough to trigger a narrow scoring mode; no hidden deck inference is needed for this intervention.
- Race thresholds must stay conditional: forcing green too early can starve the deck's $8 payload, so the lock fires at five Provinces, two opposing Provinces, or a close VP lead.
- A single Moat against a publicly owned WM Witch is a separate defensive test; permanent Moat buying would conflate attack defense with race timing.
- Preflight confirms the race branch coexists with the rollout controller and arbitrary stand-in model output without contract errors.

### Candidate 0009 — complete-policy comparison
- A purchase rollout evaluates a move relative to its assumed future policy. Candidate 0003's worst trace preferred Village to Gold by about 1.27 projected VP and Market to Province by about 1.81; those small advantages motivate comparing coherent continuations, not treating rollout rankings as ground truth.
- Candidate 0005's Chapel economy made no purchase on turns 6–8 in its worst SM game. A portfolio should include both thinning and unthinned economies so it can test thinning payback instead of assuming it universally wins.
- Successive-halving search must select only surviving finalists. Our first dev trace exposed eliminated 32-sample policies re-entering against 192-sample finalists; the final implementation fixes this unequal-comparison bug and shares real/simulated buy tie-breaking.
- Final dev results were +10/+9 with zero illegal actions/errors in about 14 seconds. Both Astra calls were pending at completion and cleanly cancelled, so this is evidence for the local portfolio only. Fast local completion can make an asynchronous planner irrelevant.
- Seven direct checks cover 1,822 archived prompts, model garbage/failures, acquisition consistency, and core simulation invariants. Preflight alone skipped the model branch. Synthetic public-belief forecasts are diagnostics, not additional evaluated game scores; dev coverage on one kingdom does not establish Chapel/Witch strength.

### Candidate 0008 — complete-policy portfolio
- Inspect the whole turn before diagnosing a missed scoring purchase. Leader 0003 decision 40 buys Silver and then Province with its remaining money/buy; the Silver-vs-Province rollout ranking alone does not prove a missed Province.
- A portfolio needs strategically different continuations. The first version mostly agreed on when to score and produced +9/-10 in dev; adding build-to-$16 and earlier-scoring policies yielded +12/+4 in the final permitted dev run. These are two development games, not evidence across the training kingdoms.
- Comparing complete policies on common initial samples and then an independent validation batch tests continuation-policy dependence directly. Paired uncertainty and a small switching threshold reduce changes driven by sampling noise; bounded searches must disclose when only screening samples fit.
- Astra can influence a fast local controller without blocking: one final dev answer arrived and selected a close Smithy-money policy; the other game completed without an answer. No-answer completion and model-assisted play are different evidence.
- Simulator/executor consistency needs source-specific gain checks. A generic acquisition priority could trash Silver with Mine and then gain Silver back; explicitly choosing the Treasure upgrade matches Mine's modeled gain-to-hand behavior.
- Final preflight passed; 2,939 archived prompts validated and 3,558 sampled turns conserved ownership/zones with nonnegative piles. Preflight skipped model calls, so garbage output, TimeoutError, RuntimeError and pending-future paths were checked separately.


### Candidate 0010 — Province harvesting inside build policies
- In leader 0008's worst trace, turns 11–14 spend $8/$8/$9/$11 on Laboratory/Markets while SM-1 already owns three Provinces and the agent owns none. These were actual full-turn Province passes; the game ends 30–50.
- Complete-policy search can couple a current scoring choice to an entirely different future acquisition plan. Candidate 0010 retains engine construction but adds a public opponent-Province-lead condition for harvesting an affordable Province, consistently in root selection and continuation buys.
- Do not infer optimality from a bad realized outcome. Frozen public-state forecasts favor the parent's investment over harvesting by roughly 1–5 VP; the new rule is an empirical guard against possibly optimistic forecasts, and should be rejected if actual margins regress.
- One dev evaluation completed at +12/+4 with no errors or illegal actions, but selected money policies throughout. This validates execution, not the engine-race intervention; both Astra requests were still pending at completion.
- Preflight skipped System-2. Separate checks covered garbage/failing/pending model futures, all 561 parent archived prompts, four concrete race interventions, and 883 sampled turns with conserved card zones and nonnegative piles.

### Candidate 0012 — realized payload feedback
- Candidate 0008's worst episode repeatedly bought Laboratory/Market while owning no Gold and only about $12 of payload for a 13-card deck; public VP and Province counts alone do not explain the draw/action overbuild.
- A feedback mode can use public owned-card totals to detect when engine size has outrun terminal money, then force Gold/Silver until payload catches up; this is structurally different from a fixed opponent-race trigger.
- Keep the mode local and hysteretic: recompute after every prompt, clear it only after a payload floor is met, and never cache prompt occurrence IDs.
- Existing complete-policy rollouts remain useful when the deck is balanced; bypassing them during repair avoids letting noisy hidden-world forecasts reintroduce the same cantrip bias.
- Aggregate payload is only a proxy for hand-level draw quality; a trace showing repair purchases without improved VP would suggest conditioning on observed turn income or shuffle timing.

### Candidate 0011 — payload/action-capacity invariant
- The -20 Chapel/SM trace spent $8–$11 on Laboratory/Market while owning no Gold; action draw alone did not produce a reliable Province payload.
- A race lock and a payload lock test different failure layers: Province ownership reacts to the opponent, while Gold reservation protects the agent's own conversion capacity.
- Action prompts can expose terminal collisions even when buy plans look coherent; gate terminal play when remaining Actions cannot consume the terminal hand.
- Keep the payload heuristic shared by rollouts and live execution, otherwise forecast improvements can be an executor artifact.

### Candidate 0014 — realized-income turn commitment
- Aggregate owned payload can lag the actual question: whether the current shuffle converted actions into enough money to score.
- Commit a whole turn to scoring or repair from the live buy income and prior-turn income; buy-by-buy policy switching can reintroduce engine drift after a failed draw.
- Keep the lock public and hysteretic, and rebuild labels each prompt because a Market or gain can create a second buy with new IDs.
- A strong draw can justify Province immediately when the opponent has a public lead; a weak draw should purchase Treasure before another cantrip.

### Candidate 0015 — public VP-pace race clock
- Candidate 0010’s Province harvest waits for a three-Province/two-card lead; this can be too late when SM-1 has only two Provinces but a six-VP lead and the Province pile is already at five.
- A public VP-plus-pile-pressure predicate is a distinct race signal from aggregate payload repair: it interrupts construction only when an affordable Province exists and the opponent is visibly ahead.
- Keep the predicate identical in rollout, policy-root, and live acquisition; otherwise forecast and executor can disagree about the intervention.
- The trigger must stay narrow because visible VP includes early green cards and does not reveal the opponent’s next hand; test for reduced overbuilding without assuming hidden-state certainty.

### Candidate 0013 — margin-valued complete-turn endgames
- A win-only gate can contradict the evaluation objective. Leader 0008 rejected an available -12 finish at turn 17 and a -10 finish at turn 19 in its worst trace, then ended -20. The observations prove better-than-realized alternatives, not their expected superiority.
- Compare complete purchase baskets: Province plus Estate can improve the exact finish, and buying after a pile empties still counts because termination waits for turn cleanup. Gardens can change value when later purchases cross a deck-size threshold.
- Exact ending margins and sampled continuation margins belong on the same scale. The new comparison permits losing finishes and profitable delays of wins, while requiring a sampled advantage over an exact finish to exceed uncertainty.
- The one dev evaluation improved to +18/+14 with zero errors or illegal actions, largely through late Gardens purchases rather than a losing finish. This validates the broader scoring search on those two games; it does not yet confirm the central loss-ending hypothesis or training performance.
- Frozen-state continuation forecasts at the first rejected finish remain close: two Duchies project -11.854 ±0.502 standard error versus exact -12. At the later state, the best continuation projects -12.833 versus exact -10. Keep counterevidence instead of treating the realized -20 as proof the model is wrong.
- Preflight skipped model inference. Separate checks covered 561 legal archived prompt responses, 173 exact feasible baskets, Gardens and post-ending purchases, fresh-ID basket commitments, and malformed/failed/pending Astra futures; neither dev episode received its requested Astra answer.

### Candidate 0017 — buy-conservation fallback
- A legal empty buy can still be a serious harness failure: candidate 0013 repeatedly passed with $11–$13 and multiple Buys before buying weaker cards later in the same turn.
- Keep the repair at the executor boundary. Search and Astra remain advisory; a fallback must inspect only the current prompt and rebuild occurrence IDs.
- Province/Gold/Silver are safer emergency choices than inventing a plan, but the fallback must run only after a policy/search miss so it does not erase deliberate engine or Gardens decisions.
- Trace evidence distinguishes “no illegal action” from “good control”: silent passes are valid simulator actions yet can leave the objective under-optimized.

### Candidate 0016 — parity-gated public race clock
- The broad VP/pace trigger in 0015 fired in the −21 trace before the agent owned any Province; visible deficit alone can turn a recoverable game into premature green buying.
- In the −11 trace, the same public deficit occurred with one owned Province and $13, while the controller still bought Cellar. Requiring one own Province and a one-card opponent lead targets this concrete conversion race.
- Keep the predicate in live acquisition, policy roots, and sampled worlds; otherwise complete-policy forecasts can recommend construction immediately after the executor starts scoring.
- Race interrupts must clear any queued basket because a fresh Province changes both pile pressure and the validity of later queued IDs.
- The inherited margin basket search remains responsible for the final two Provinces; the new clock is deliberately narrow and should be judged by whether it removes delayed scoring without repeating the early −21 failure.

### Candidate 0018 — public opponent-zone conditioning
- Public event deltas can constrain hidden-world rollouts: Militia discards reveal cards that remain in discard until a shuffle, and a just-bought card cannot be in the current hand without a draw.
- A conservative ledger must replay only disclosed draws and cleanup boundaries; on count mismatch it should discard constraints and fall back to uniform sampling rather than infer a hidden order.
- The disclosed opponent discard rule turns Militia cards into a useful retained-rank bound, raising the sampled immediate scoring threat in the archived failure state from 2.8% to 40.6%.
- Repeated observations are stable and must not replay log deltas; an episode-local decision ID guard prevents double counting while preserving fresh constraints.
- Conditioning changed the two-game dev result to +10/+9 with zero errors, but this is execution evidence only; impossible-world removal still needs training-seed score confirmation.
