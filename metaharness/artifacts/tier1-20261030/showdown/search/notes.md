
### Candidate 0002 — counterfactual combat cards

- The worst baseline loss (333667, -1.1845) was not a deadline issue: Volcanion lost 239/256 HP chipping Calm Mind Giratina, and Clefable later selected Fire Blast but died to Gigaton Hammer before moving. Explicit survival and check-preservation audits are a separate hypothesis from latency optimization.
- Protocol damage and move-order observations should be retained as contextual evidence. Move order alone does not prove raw Speed: priority, Trick Room and activation effects can explain it.
- Public move_dex contains own-team moves, not necessarily revealed enemy moves. Hypothetical STAB probes must be labeled as probes and must not be presented as inferred enemy sets or damage bounds.
- A fresh stateless prompt with a short strategic memory avoids accumulating duplicate full observations. On 204 archived actionable states the new combat-card prompts stayed below 16.4k characters; no loss of the latest request or recent protocol to generic truncation.
- Preflight passed but its short budget caused zero Astra calls. Separate archive replay verified legal fallback selection and malformed/JSON/plain-action parsing; a passing fallback-only preflight does not establish model-path coverage.

### Candidate 0003 — adversarial response matrix

- A distinct intervention from supplying combat calculations is to ask Astra for counterfactual values over several opponent responses, then let code select a downside-aware action. This makes the opponent-response assumptions inspectable; it does not make them correct.
- The first dev matrix omitted the actual physical attack from an unrevealed opponent. After Body Slam was observed, the next matrix explicitly compared repeating it, switching and defensive Tera. Keep observations capable of overturning the model's initial set hypothesis.
- Public Tera is stored as `teraTypeRevealed` in the observed public summary; private requests use `terastallized`. Verify field names against traces before computing type matchups.
- Terminal observations can have `request: null`. The initial preflight caught this ledger edge case; the corrected preflight passed with one answered System-2 call and no warnings. A short preflight otherwise exercises only fallback if model admission requires a large remaining wall budget.
- Direct-damage samples need to exclude `[from]` residual/recoil events, flag critical hits and faint censoring, and distinguish pre-turn context from effects gained during the turn. Archive replay covered 193 available transitions; 108 serialized observations were truncated and were skipped rather than treated as complete states.

## Candidate 0001 — evidence-first reports (iteration 1)

- The leader's generic renderer truncates *inside* `move_dex`: seed 333667's first three prompts are 10,286 characters and omit `species_dex` and `recent_events`. Fix information delivery before diagnosing every bad action as model strategy failure.
- Recent protocol must precede bulky static data. Keep public damage/move-order evidence and a short strategic memo across fresh model sessions; a reused session cannot recover events never supplied.
- Observation schemas need direct checks: terminal `request` can be null; hazard keys include `move: `; opponent Tera appears as `teraTypeRevealed`. The first pre-flight caught the terminal-null bug, and the repair passed.
- Plans should bind Pokémon identities and move IDs, then remap slots and verify effects after each action. New opponent moves, mechanics, failed attacks, unsafe HP and no-progress signals invalidate continuations and trigger Astra when budget allows.
- A 150-second call reserve avoids beginning a late call near the deadline, but also means the short pre-flight may exercise only fallback. Separate archived-observation and malformed-answer checks are necessary to cover report construction and parsing.
- Dev 0001 finished one win (+1.051724) and one loss (-1.129911), no errors; a 180-second timeout recovered. No checked continuation executed, so call savings remain unproven. The losing late fallback repeated Choice-locked Earth Power into observed Levitate; recognizing public immunity ability tags and penalizing immune attacks fixed that local failure in replay, but its score impact is unmeasured.

- Candidate 0002 timing follow-up: account for elapsed episode-clock time around ctx.think, not just result.latency_s. The first dev call consumed about 177 wall seconds while the trace reported 55.8 seconds of model latency; admission now reserves the largest observed end-to-end call plus finishing time.

Candidate 0002 dev result: seed 869431 won +1.105867 with four survivors, 30 actions, 18 Astra calls, no errors, 1639 s. The last 12 actions used budget fallback; no guarded repeats were authorized. Tauros converted the endgame, but fallback wasted Sunflora into Blissey recovery first. This single initial-revision dev win does not isolate the survival-audit hypothesis; final item/protocol/timing/parser safeguards were locally checked, not re-evaluated with models. Details: candidates/0002/proposal.md.
- The completed 0003 dev episode lost (-1.1041, 33 actions, no episode error); four of twelve 180-second calls timed out, so matrix complexity was too costly in this run. A malformed three-scenario/two-value matrix and repeated immune fallback attacks were concrete failures. Final revisions reject incomplete matrices, expose bench type matchups, shorten output, flag unmodeled opponent responses and switch out when all damaging attacks are immune; seven dev observations now take a legal switch. Final preflight passes, but final strength is unmeasured.


## Candidate 0005 — bounded tactical policies (iteration 2)

- Trace counts matter more than the presence of a repeat API: 0001 executed only 14 checked continuations out of 355 actions and used 85 fallbacks; 0003 executed zero verified repeats and 101 fallbacks. Temporal abstraction was largely unrealized.
- A predicted first enemy move reveal can be compatible with a plan. A real Piloswine switch into an explicitly anticipated Icicle Crash passes the new guard; an unanticipated coverage move cancels it. Unknown moves are never blanket-approved.
- Bind a policy to Pokemon identities and move IDs, then resolve current slots; archive-based permutation checks catch stale-slot execution. Gross hit damage, field/effect fingerprints, expiry and ctx.progress() independently bound the policy.
- Passive item effects may reveal items without an explicit -item line. Preserve those public facts in policy fingerprints; an expected attack does not make a newly revealed item automatically safe.
- Preflight passed with one model-path call; separate offline checks covered malformed outputs and 198 complete archived states. The mechanism passes these checks, but call savings and playing strength still require the real dev/training traces.

## Candidate 0006 — short policies with outcome contracts (iteration 2)

- In leader 0001's worst training episode (232003), expected Salamence Intimidate canceled a follow-up, and an enemy Thunder Wave hitting Volt Absorb was labeled plan failure. Failure detection must distinguish opponent failure from our failure and account for planned entry effects.
- Parent 0002 used 18 Astra actions and 16 fallback actions on seed 232003. Its combat arithmetic did not establish useful action batching; test declarative policies with HP/damage bounds as a separate intervention.
- The first 0006 dev build found expected Grassy Surge causing the same false cancellation. Final guards accept a verified own entry terrain/weather ability and harmless observed damage within the explicit policy envelope; non-damaging surprises still trigger a replan.
- A policy for a *different*, previously revealed opponent was unsafe: Gengar's Body Slam immunity did not protect against Wyrdeer's unrevealed Psychic Noise. Final compilation is restricted to the current opponent and every enemy switch triggers fresh Astra reasoning.
- Initial 0006 dev revision lost both seeds (mean -1.102113), with eight compiled actions, twenty fallback actions and no errors: useful execution is not evidence of stronger play. One call took 208.3 seconds and succeeded under the 240-second timeout. Final guard/identity repairs are locally validated but not model-evaluated; see the proposal. Preflight exercised fallback only, so separate archive and mock-context checks cover the model path.

## Candidate 0004 — empirical exchange controller (iteration 2)

- Parent 0001's ten traces contain 14 checked-plan actions versus 85 budget/parse fallback actions in 355 decisions. In seed 181081, three successive Drain Punches cost separate calls, then a boosted Zapdos endgame falls to fallback. This motivates allocating calls by changed tactical state; it does not make every repeated move safe.
- A strict empirical gate approved only one of 188 readable parent transitions. Accounting for observed order and our explicitly intended Tera raises this to three. Hidden priority/coverage and speed ties remain uncertainties; damage samples do not establish guaranteed wins.
- Early dev answers vetoed every continuation when asked for an opt-in `hold` field. Final 0004 removes opt-in and lets a bounded code gate own the decision, with HP/context/progress checks after every action. Dev-loaded versions precede this change, so their scores cannot establish the final gate's strength.
- Public move descriptions can be compiled from the archive without copying sets: 115 move records had zero conflicting observations. A revealed-coverage table exposes Icicle Crash's 2x multiplier against Scyther and 0.5x against Thick Fat Piloswine. Unknown moves and unsupported dynamic typing stay unknown.
- Pre-flight passes but covers fallback only. Separate checks exercise 206 observations, 188 transitions, real checked-action sequencing, effect-based invalidation, malformed replies, model failures and action rejection. Passing pre-flight alone remains insufficient evidence for model-path correctness.
- Dev 0004 seed 869431 won +1.14693 (20 actions, one caught timeout) and seed 895003 won +1.02775 (29 actions, no failures). Both loaded pre-final revisions and executed no empirical continuation; favorable Tera/critical/endgame events make these operational evidence only.
- The 895003 trace exposed a new generic failure detector: repeated Bronzong Iron Head exchanges with Malamar included Rest and HP loss, so observation changed every turn and the old progress counters did not fire. Candidate 0004 adds a four-step recovery-cycle audit.

Candidate 0005 dev follow-up:

- Dev 869431 lost at -1.062977 (35 actions, 28 calls, no errors). Twenty-seven replies had null policies; the sole compiled rule was canceled by its own Rillaboom Grassy Surge. Final elicitation requests narrow conditional next steps, and guards accept only protocol-confirmed expected entry terrain; unexpected/suppressed field effects still replan. Final preflight and 26 offline checks pass, but final strength/call savings remain unmeasured. Full evidence and limitations are in candidates/0005/proposal.md.

### Candidate 0007 — verified same-move queues (iteration 3)

- Candidate 0001's seed 232003 trace repeatedly paid 24–101 seconds for the same Earthquake/Mamoswine exchange; its `repeat` field was usually zero, so the intended temporal abstraction did not execute.
- A queue should contain one damaging move repeated at most twice, bound to current active identities and remapped through the latest request. Queuing a switch, setup, recovery, Tera, or two different moves carries unverified assumptions.
- Queue guards must treat any new opponent move, status/boost/field/item/form event, rejection, immunity/miss, low damage, or low HP as evidence for immediate replanning; `ctx.progress()` alone cannot detect every changed matchup.
- A fresh Astra answer must clear stale failure flags before installing a queue. Otherwise a prior cancellation can invalidate the new plan on its first continuation.
- Raising the late-call reserve and constraining output length trades some late reasoning for fewer long model calls; this is useful only if queued actions actually appear in evaluation traces.

- Candidate 0008 tests information ordering as a structural hypothesis: block early setup/status until at least one opposing damaging move is revealed, rather than adding more arithmetic to the prompt.
- Ranked multi-action JSON gives malformed or unsafe Astra output a legal backup; safety checks must happen after parsing because arbitrary stand-in text is used in preflight.
- A repeat is earned from observed same-matchup HP progress and canceled on changed identities, low HP, stale progress, or rejected actions; this is narrower than trusting model-declared continuation policies.
- Opening barriers can be harmful when a guaranteed immunity/setup line exists, so held-out traces should count both prevented greedy setup and blocked safe opportunities.

## Candidate 0009 — recurrent tactical consultations (iteration 3)

- A complete report does not guarantee correct mechanics: 0001's worst trace calls Ground into Ice/Ground super-effective and later switches Scyther into revealed Icicle Crash. Its 181081 trace calls Skarmory resistant to Fighting before boosted Close Combat KOs it. Expose type facts without presenting approximate damage as certainty.
- Test communication architecture separately from policy compilation: 0009 uses the naive parent, persistent bounded conversations, full dynamic snapshots, incremental static rules, and a short action/risk answer. It emits no strategic memo or multi-turn policy; reducing output is not itself proof of better decisions.
- A session checkpoint must re-send all static rules, including when a backend returns no session ID. Dynamic slots, HP, effects, legality and request flags must never depend on an old delta. Public move records compiled from this archive had 115 consistent entries and zero conflicting records.
- Offline replay covered 1,446 complete archived observations and 1,376 transitions; the largest initial checkpoint was 12,503 characters. Mock full loops cover valid replies, garbage, TimeoutError and RuntimeError, plus session restarts and progress failures. Preflight passed after fixing an initial file-assembly omission, but its short allowance exercised fallback only.
- A 300-second timeout exceeds the latest 236-second observed maximum. Call admission and timeout are separate decisions: 0009 admits calls with more than 120 seconds left to retain tactical reasoning late, accepting that an unusually slow late call can hit the normal episode deadline. Terminal reasons must be reported alongside score.
- Dev 0009 completed one win (+1.030260) and one loss (-1.126842), mean -0.048291, with no errors/timeouts. Each used 24 Astra calls; 16 total fallback actions show the wall-time problem persists. End-to-end call medians were 64/73 seconds despite reported model medians near 39/42 seconds. Checkpoint prompts reached 20.4k/26.8k. A critical hit and newly revealed Earthquake complicate the losing Gengar setup example; Toxic Debris from Glimmora's faint helped the winning endgame. These are diagnostic outcomes, not evidence of improved training strength.
