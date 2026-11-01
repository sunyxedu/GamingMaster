
## Generation 1 (automatic summary: the reflection did not finish)
### candidates/0003 — refine of 0000-tier2_jev_only
status: fatal: proposer failed: ModelTimeout: timeout: claude -p exceeded 3600s (session 3032c895-bb20-4b83-bc48-532314b8c381); mean score None; episodes: –
calls S1/S2 0/0, failed 0/0, stale None, Jev→act None, Astra→act None
proposal (head):
(none)
### candidates/0004 — rewrite of 0002-tier2_idle
status: fatal: proposer failed: ModelTimeout: timeout: claude -p exceeded 3600s (session ce52209f-346d-4e12-8d3a-911765c78df0); mean score None; episodes: –
calls S1/S2 0/0, failed 0/0, stale None, Jev→act None, Astra→act None
proposal (head):
(none)
### candidates/0005 — crossover of 0000-tier2_jev_only, 0002-tier2_idle
status: ok; mean score 288.0; episodes: seed 0: 382.0 game_done · seed 1: 353.0 game_done · seed 2: 227.0 game_done · seed 3: 361.0 game_done · seed 4: 143.0 game_done · seed 5: 262.0 game_done
calls S1/S2 2660/6, failed 0/5, stale 0.00012514078338130396, Jev→act 0.05546516478931998, Astra→act 2.0859407592824365e-05
proposal (head):
# Candidate 0005 — asynchronous interception with advisory models

## Parents

- `candidates/0000-tier2_jev_only/`: the best scored parent, mean 3.333, because Jev occasionally selected a direction that happened to contact bricks.
- `candidates/0002-tier2_idle/`: the robustness baseline, which never posted a harmful direction and therefore ran until the deadline with five lives, but scored 0.

## 


## Generation 2
- `0006` (rewrite of `0001-tier2_astra_only`) replaced synchronous Astra steering with an asynchronous temporal-velocity interceptor and STABLE/QUIET contracts. It scored 200.833 (160, 77, 292, 198, 318, 160), with 0.000248 stale ticks, 572 S1/8 S2 calls, and 1 failed S2 call; all 6 episodes ended `game_done`. In worst seed 1, fresh control still lost lives at frames 2264, 4111, 4491, and 5536, then ended at 6258 with score 77 and 64 bricks; all 87 accepted Jev contracts were STABLE. Best seed 4 reached 318 and 27 bricks remaining at frame 7360, but lost lives at 936, 3230, 3973, and 5060.
- `0007` (crossover of `0005` and `0002-tier2_idle`) added measured paddle braking, uncertain-return handling, and contact-offset shaping over `0005`'s interceptor. It scored 281.000 (291, 371, 177, 157, 330, 360), with 0.000407 stale ticks, 1912 S1/6 S2 calls, and 6/6 failed S2 calls. Worst seed 3 ended at frame 7155 with score 157 and 47 bricks after losses at 3318, 4845, 5888, and 6635; its 267 accepted modes were 255 TRACK, 7 BRAKE, and 5 CENTER, while 56 calibrations split outgoing RIGHT/LEFT 31/25 with offsets from -10 to +9. Best seed 1 reached 371 and 10 bricks remaining at frame 9373 after losses at 2202, 4731, 5560, and 7182; 320 accepted modes were mostly TRACK (305), so shaping did not become a persistent tunnel policy.
- `0008` (explore of `0005`) made the measured-velocity servo and empirical contact mode the primary change, with Astra advisory only. It scored 234.000 (143, 98, 218, 400, 350, 195), with 0.000204 stale ticks, 1008 S1/6 S2 calls, and 6/6 failed S2 calls (5 timeouts); every accepted Jev mode in the selected traces was SAFE and neither trace emitted `contact_calibrated`. Worst seed 1 ended at 6741 with score 98 and 58 bricks after losses at 1062, 1732, 2686, and 5338, including fresh LEFT/RIGHT alternation around the first three losses. Best seed 3 reached 400 and 5 bricks remaining, but still lost lives at 1369, 4923, 8825, and 9013 and ended at 9191 without clearing the wall.
- Asynchronous code-owned action posting is confirmed: `0001` had 100% stale ticks and score 0, whereas `0005`–`0008` had stale fractions 0.000125–0.000407 and no episode errors; the improvement came from the fallback loop, not from slow-model control.
- More stateful contracts did not improve the parent: means were 200.833, 281.000, and 234.000 for `0006`–`0008` versus 288.000 for `0005`; Jev-informed action fractions were 1.4%, 4.1%, and 2.3%, and Astra-informed fractions were 0% at leaderboard precision.
- Contact shaping is not isolated: `0007` measured 138 contacts across its selected worst/best traces but offsets changed sign and magnitude, while `0008` measured 0; a safer servo and a calibrated angle policy were not separately tested.
- Fresh actions still reverse at the impact window: `0007` seed 3 alternated RIGHT/LEFT at frames 3313–3322, and `0008` seed 1 alternated LEFT/RIGHT at 1057–1066; the traces support collision-timing/actuator-state work rather than more Jev prose. Astra plumbing supplied no control evidence: `0007` and `0008` timed out on 6/6 calls, and `0006` had 1 failed call.
- Remaining headroom is wall progress: best overall `0005` averaged score 288.0/864, 28.5 bricks remaining, 7991.0 frames, 0 lives, `wall_index=0`, `walls_cleared=0`, and `both_walls_cleared=0`; all 6 episodes ended `game_done`. The furthest gap is completing even the first of the 2 walls before the five lives are exhausted, not stale-tick or deadline plumbing.

## Generation 3
- `0009` (crossover of `0005` and `0008`) kept the reflected interceptor and added measured paddle velocity, target hysteresis, committed pulses, cooldown braking, and optional Jev side shaping. It fell to mean 183.167 (89, 280, 220, 110, 144, 256) with 0.000237 stale ticks, 1003/6 S1/S2 calls, and 6/6 Astra timeouts. Worst seed 0 lost at frames 1881, 4708, 4957, 5569 and ended at 5611 with 89; frames 1865–1881 still contain fresh LEFT/NOOP/RIGHT reversals, while frames 4941–4957 are all NOOP. Best seed 1 reached 280 but repeated the same reversals before losses at 4043, 4552, 6529 and 7196; every accepted Jev preference in these traces was SAFE and Astra was unavailable.
- `0010` (explore of `0008`) replaced point chasing with a brick-band phase gate, latched landing-overlap interval, and measured neutral/counter-steer, while deliberately removing Astra. It scored only 5.500 (5, 5, 6, 5, 7, 5), with 0.001577 stale ticks, 81/0 calls, no call failures, and 6/6 `game_done` episodes. Worst seed 0 posted NOOP for frames 323–339 before the first loss at score 2 and repeated this through losses at 550, 761 and 976; the trace records landing plans but no useful directional action. Best seed 4 likewise posted NOOP for frames 713–729, 924–940, 1135–1151 and 1350–1366, ending at 7, refuting the claim that the committed interval preserved catches.
- `0011` (refine of `0005`) added filtered/committed landing lanes and measured counter-steering near contact without changing the advisory model architecture. It scored 127.333 (150, 135, 130, 151, 92, 106), with 0.000315 stale ticks, 1587/6 calls, and 6/6 Astra timeouts. Worst seed 4 ended at 4176 with 92 after losses at 965, 2099 and 2845; the 949–965 stream cycles RIGHT/NOOP, LEFT, RIGHT, then LEFT immediately before the first loss. Best seed 3 reached 151 but still showed the same three-frame direction cycles at 1713–1729 and a 17-frame NOOP run at 2099–2115; accepted Jev modes were 80/84 and 149/154 TRACK, so the counter-steering hypothesis was not isolated and did not improve the parent.
- Fresh code-owned posting remains necessary: all three candidates had stale fractions below 0.0016 and no episode errors, but this alone did not preserve lives or clear a wall.
- Hysteresis/commitment was not confirmed: `0009` and `0011` still reversed in fresh-action windows, while `0010` overcorrected to NOOP; the next test needs collision-transition/contact evidence, not more mode labels.
- Astra supplied no control evidence: `0009` and `0011` timed out on 6/6 calls, and `0010` made 0 calls; Jev mostly returned SAFE/TRACK, so model-call counts did not imply action influence.
- Serve and termination plumbing was legal (`game_done` in all 18 evaluated generation-3 episodes); the costly contract detail was low-level action behavior after plan creation, not stale ticks or the 900-second deadline.
- Remaining headroom is the first wall: best overall `0005` averaged `ale_reward=cartridge_score=288.0` of the achievable 864, with 28.5 bricks remaining, 0 lives, `wall_index=0`, `walls_cleared=0`, and `both_walls_cleared=0`; all six episodes ended `game_done`, so completing two six-row walls—not model latency—is the furthest missing component.

## Generation 4 (automatic summary: the reflection did not finish)
### candidates/0012 — explore of 0006
status: ok; mean score 313.3333333333333; episodes: seed 0: 372.0 game_done (8074 steps, 135 s) · seed 1: 383.0 game_done (7224 steps, 120 s) · seed 2: 400.0 game_done (7801 steps, 130 s) · seed 3: 286.0 game_done (8027 steps, 134 s) · seed 4: 153.0 game_done (7153 steps, 119 s) · seed 5: 286.0 game_done (6430 steps, 107 s)
calls S1/S2 627/6, failed 0/5, stale 0.0004473372251671923, Jev→act 0.014007921412428114, Astra→act 2.2376871265859607e-05
proposal (head):
# Candidate 0012 — receding-horizon capture-set control

## Parent and hypothesis

Parent: `candidates/0006/`.  I also used the best evaluated implementation,
`candidates/0005`, as the performance comparison.  The parent and its
successors already established that the real-time loop must be code-owned;
this candidate does not return to synchronous model steering.

**Hypothesis:** the remaining los
### candidates/0013 — refine of 0005
status: ok; mean score 224.5; episodes: seed 0: 344.0 game_done (8309 steps, 138 s) · seed 1: 106.0 game_done (6543 steps, 109 s) · seed 2: 335.0 game_done (7803 steps, 130 s) · seed 3: 329.0 game_done (7003 steps, 117 s) · seed 4: 78.0 game_done (4765 steps, 79 s) · seed 5: 155.0 game_done (6076 steps, 101 s)
calls S1/S2 2116/6, failed 0/1, stale 0.00014815180621743747, Jev→act 0.0522559454720569, Astra→act 0.00012347813202281876
proposal (head):
# Candidate 0013 — turn-gated interception

## Parent and hypothesis

Parent: `candidates/0005/`, the asynchronous code-owned reflected-interceptor with Jev steering advice and Astra tunnel advice (mean 288.0).

The parent is generally fresh, but its point-target servo changes direction at the worst possible time.  In the seed-4 trace, the tick stream immediately before the first replacement serve
### candidates/0014 — rewrite of 0007
status: ok; mean score 411.5; episodes: seed 0: 436.0 game_done (12635 steps, 211 s) · seed 1: 388.0 game_done (11199 steps, 187 s) · seed 2: 418.0 game_done (14224 steps, 237 s) · seed 3: 388.0 game_done (9441 steps, 157 s) · seed 4: 421.0 game_done (12305 steps, 205 s) · seed 5: 418.0 game_done (11123 steps, 185 s)
calls S1/S2 718/6, failed 0/6, stale 0.00029607906720994825, Jev→act 0.010125939611039813, Astra→act 0.0
proposal (head):
# Candidate 0014 — event contracts and an interval motor

## Parent and diagnosis

Parent: `candidates/0007/`, with `0005` as the best-scoring reference.  I kept
0005/0007's important property: code posts a legal action before waiting for a
model, and the model is never on the 60-Hz critical path.

The traces identify a lower-level failure rather than a stale-tick failure.
0007 seed 3 had only 2 s


## Generation 5 (automatic summary: the reflection did not finish)
### candidates/0015 — refine of 0014
status: ok; mean score 420.1666666666667; episodes: seed 0: 421.0 game_done (13346 steps, 222 s) · seed 1: 464.0 game_done (31027 steps, 517 s) · seed 2: 418.0 game_done (9061 steps, 151 s) · seed 3: 410.0 game_done (11575 steps, 193 s) · seed 4: 426.0 game_done (12245 steps, 204 s) · seed 5: 382.0 game_done (9241 steps, 154 s)
calls S1/S2 769/6, failed 0/6, stale 0.0001502977050696572, Jev→act 0.008892023773733262, Astra→act 0.0
proposal (head):
# Candidate 0015 — filtered receding-horizon interception

## Parent and targeted hypothesis

Parent: `candidates/0014/`. I keep its event-contract architecture,
interval motor, asynchronous code-owned action loop, and advisory Jev/Astra
bridge. This candidate makes one low-level change: replace the parent's
instantaneous per-frame velocity with a same-sign two-sample RAM estimate and
refresh the 
### candidates/0016 — rewrite of 0008
status: ok; mean score 312.3333333333333; episodes: seed 0: 417.0 game_done (9509 steps, 158 s) · seed 1: 99.0 game_done (6089 steps, 101 s) · seed 2: 371.0 game_done (8811 steps, 147 s) · seed 3: 295.0 game_done (7565 steps, 126 s) · seed 4: 355.0 game_done (9374 steps, 156 s) · seed 5: 337.0 game_done (7457 steps, 124 s)
calls S1/S2 467/6, failed 0/0, stale 0.00024587644708533963, Jev→act 0.009571045026950588, Astra→act 0.00012296845859037156
proposal (head):
# Candidate 0016 — endpoint contracts for safe tunnel shaping

## Parent and diagnosis

The direct parent is `candidates/0008/`.  I used `0014` as the strongest control
reference because it is the first archive candidate to make substantial wall
progress.

The parent established two things that must not be given back: a deterministic
60-Hz controller is essential, and ball/paddle geometry belongs 
### candidates/0017 — crossover of 0014, 0006
status: ok; mean score 420.1666666666667; episodes: seed 0: 430.0 game_done (12087 steps, 201 s) · seed 1: 426.0 game_done (9321 steps, 155 s) · seed 2: 468.0 game_done (12584 steps, 210 s) · seed 3: 354.0 game_done (9419 steps, 157 s) · seed 4: 440.0 game_done (11638 steps, 194 s) · seed 5: 403.0 game_done (14297 steps, 238 s)
calls S1/S2 688/6, failed 0/6, stale 0.00015862486661090763, Jev→act 0.009922838393307853, Astra→act 0.0
proposal (head):
# Candidate 0017 — temporally latched interval control

## Parents and hypothesis

This is a crossover of `candidates/0014/` and `candidates/0006/`.

`0014` is the strongest parent (mean 411.5): its score comes from an
asynchronous, code-owned reflected landing interval and a leased motor that
posts an action every native frame. It correctly keeps Astra/Jev off the
critical path. `0006` supplies t


## Generation 6
- `0018` (rewrite of `0015`) replaced the parent's descending motor with an interior crossing certificate, identified actuator-lag regression, short sequence search, and Astra/Je​​v evidence audits. It scored 703.667 (852, 506, 840, 512, 864, 648), with 7.182e-5 stale ticks, 206/18 S1/S2 calls, and 0/0 failures. Worst seed 1 reached wall 1 but died at 506 with 70 bricks left: its five failed contact records (frames 6495, 7608, 7832, 9377, 12510) had paddle drift 3–5 pixels/frame, including an apparent rectangle overlap at 6495, and all 11 classifications were MOTOR. Best seed 4 reached 864; it recorded 164 stable catches and 5 failed contacts, while three Astra reviews and CORE→MEASURED certificate changes accompanied the two-wall run.
- `0019` (crossover of `0015` and `0016`) made the endpoint objective one-per-flight, fitted queued/sticky actuator motion, and added measured cleanup rays. It scored 743.500 (864, 801, 428, 864 deadline, 796, 708), with 1.043e-4 stale ticks, 1835/18 calls, and 0/0 failures. Worst seed 2 ended at 428 with one brick left on wall 0: five miss windows ended at frames 5510, 6834, 12834, 17424, and 24581; of 31 resolved trials, 25 were `endpoint_not_reached`, and contacts 11270–16488 repeatedly requested LEFT but measured outgoing RIGHT. Best game-done seed 0 cleared both walls at 864 despite five resolved trials containing only two confirmed RIGHT contacts; the objective bridge still changed 151 actions, so the objective was live but its causal score benefit was not isolated.
- `0020` (explore of `0016`) tested a code-only late side-bounce continuity lock over the interval motor. It scored 396.500 (427, 598, 425, 361, 190, 378), with 1.714e-4 stale ticks, 692/6 calls, 0 failed S1 and 6/6 timed-out S2 calls; all episodes were `game_done`. Worst seed 4 logged 9 continuity events, including frame 6838 at ETA 1.67 after a lock, then reset at a paddle bounce (6841) and side bounce (6844); it lost the ball at 6913 after the next lock and finished at 190 with 41 bricks remaining. Best seed 1 logged 18 such events and reached 598/50 bricks after clearing wall 0, but its final loss at 19883 followed a lock at ETA 3.25 without a side bounce, so the tested exception did not cover the remaining failure.
- Code-owned native-frame posting remains necessary: all three candidates stayed below 1.714e-4 stale ticks and had no episode errors; model calls were not the reflex loop. `0020` demonstrates that six S2 timeouts can be harmless to eligibility but provide zero Astra-informed actions, while `0018` and `0019`'s successful S2 calls still informed only 0.016% and 0.009% of posted actions.
- Contact telemetry confirmed a moving-paddle failure mode, not a sufficient catch certificate: `0018` logged 5 failed contacts in each inspected trace, with drift 3–6 on the failures; `0019` measured 31 trials in its worst trace but repeatedly failed to reach the requested endpoint. Exact collision order, queued input at impact, and a controlled model-off/model-on ablation remain untested.
- Flight-scoped bookkeeping fixed `0016`-style repeated trial starts: `0019` resolved 31 trials rather than consuming a quota on alternating geometry samples, and its objective bridge changed 219 actions in seed 2 and 151 in seed 0. That did not make endpoint direction reliable: 24 worst-trace trials requested LEFT and observed RIGHT, so mapping quality—not more trial labels—remains the open shaping hypothesis.
- The serve/life/termination contract held: every inspected episode used FIRE without resetting on a lost ball, scores matched cartridge score, and all but `0019` seed 3 ended `game_done`; that seed reached 864 at the 900-second deadline with one life, showing deadline survival after completion rather than a model or adaptor error.
- Remaining headroom is second-wall completion: best `0019` averages `ale_reward=cartridge_score=743.5/864`, `bricks_remaining=13`, `lives_remaining=0.17`, `wall_index=0.83`, `walls_cleared=1.17`, and `both_walls_cleared=0.33`. Only seeds 0 and 3 reached 864; seeds 1, 2, 4, and 5 ended with 24, 1, 14, and 39 bricks left, respectively. The furthest gap is converting first-wall survival into the second wall's final bricks before lives run out, not stale ticks or deadline plumbing.

## Generation 7
- `0021` (crossover of `0019` and `0018`) tried a flight-level collision reservation: a short lease near ETA 10, one emergency reversal, and a hard catch interval around the parent’s queued/sticky motor. It fell to mean 310.667 (362, 290, 138, 369, 305, 400), with stale fraction 0.000402, 408/6 S1/S2 calls, and 0/0 failures; all 6 episodes ended `game_done`. In worst seed 2, 57/62 contact records were catches but the five misses ended at frames 3882, 5233, 5455, 5914, and 6353; the last window held the paddle at x=72 through ETA 6.6, then issued LEFT at ETA 5.7/3.6/1.5 and RIGHT at ETA 0 while x moved to 64, producing drift 5 and a miss. Best seed 5 still had 5 misses among 73 records; its final frame-7675 miss had predicted landing 91.79, residual 0.21, drift 0, and a paddle near x=79, so a reservation and a stationary predicted overlap did not certify the cartridge collision.
- `0022` (explore of `0019`) tried an impact-latched brake/adaptive phase and model-selected return profiles. It reached mean 763.333 (785, 421, 848, 857 deadline, 854, 815), stale fraction 0.000106, 1672/19 calls, and one timed-out S2 call; 5 episodes were `game_done` and 1 reached the deadline. Worst seed 1 established 140 locks and observed 135 contacts, but had 5 miss windows: at frame 7221 the lock began at ETA 6.6 and the paddle velocity changed from 0.26 to -4.69 while actions changed LEFT/NOOP/RIGHT, and frame 7786 similarly drove x=117 to 136 to 124 against interval 112–124. Best seed 3 established 472 locks and observed 468 contacts, reached score 857 at 53,999 frames with wall 1, 1 brick, and 1 life, but still had 4 miss windows (11888, 14803, 29273, 37630) and ended by `deadline`, not `game_done`.
- `0023` (refine of `0019`) shortened the motor rollout from `ceil(eta)+2` to the collision horizon for ETA <= 6, hypothesizing that the post-collision tail caused reversals. It scored 559.167 (428, 786, 421, 428, 428, 864), with stale fraction 0.000105, 1310/14 calls, 0/0 failures, and 6/6 `game_done` episodes. Worst seed 2 still had 5 miss windows; at frame 13969 the target was about [20, 33], the paddle moved x=22 to 15, and fresh actions alternated RIGHT, NOOP, LEFT, RIGHT before loss at score 421 with 2 bricks left. Best seed 5 cleared both walls at 864, but also logged 5 miss windows; its 34 resolved shaping trials were only 19 confirmed versus 15 `endpoint_not_reached`, so the one completion does not isolate the shorter horizon from the inherited shaping and cleanup code.

- Native-frame fallback remains confirmed: all three harnesses had stale fractions 0.000105–0.000402 and no episode errors; only 0.912%–1.053% of actions followed fresh S1 answers and 0.010%–0.013% followed S2 answers. The 0022 S2 timeout did not stop the loop.
- The terminal-action hypothesis remains unconfirmed by three different lifetimes: the reservation traces had 5/5 miss windows, the impact-lock traces 5/4, and the shorter-horizon traces 5/5 in their inspected worst/best episodes. Every listed action was fresh; collision ordering, the actual collision mask, and queued input at impact remain the untested plumbing.
- Impact locks can extend survival without proving completion: 0022 seed 3 ran 53,999 frames with 1 life and 1 brick left, while its mean `both_walls_cleared` was still 0.00 and no episode reached 864. A model-on/model-off or lock-on/lock-off ablation is needed before assigning the 763.333 mean to the new lock.
- Endpoint shaping remains uneven and must reject `endpoint_not_reached` as evidence: 0023 seed 2 confirmed 2/14 trials, while seed 5 confirmed 19/34; late model choices changed at most about 0.01% of posted actions. More labels or profile prose do not address the 12/14 and 15/34 trials that failed to reach the requested endpoint.
- Serve/life/termination plumbing was legal: scores matched cartridge score, lost balls were followed by FIRE without an episode reset, and 17/18 inspected episodes ended `game_done`; the one deadline episode had a live ball and an unfinished brick. The episode cost is lost lives or unfinished bricks, not stale ticks, model failure, or an illegal reset.
- Remaining headroom is second-wall cleanup. The best candidate `0022` averages `ale_reward=cartridge_score=763.333/864`, `bricks_remaining=7.67`, `lives_remaining=0.17`, `wall_index=0.83`, `walls_cleared=0.83`, `both_walls_cleared=0.00`, and 29,900.17 emulated frames; its endings were scores 785/421/848/857/854/815, with five `game_done` episodes and one deadline, and none cleared both walls. The furthest gap is the completion component—`both_walls_cleared` stayed 0/6 while the game can achieve 864—caused by final second-wall bricks before lives or the deadline run out.

## Generation 8
- `0024` (explore of `0013`; trajectory-epoch fencing plus an observed-position capture invariant, replacing adaptive terminal braking with move-to-edge/NOOP-inside) averaged **715.333** versus parent `0022`'s 763.333; seeds 0 and 5 hit deadline at 864, while worst seed 2 ended `game_done` at 421 with 2 first-wall bricks left. Its worst and best traces each log 5 miss windows: seed 2's final miss at frame 22689 leaves paddle x=130 inside the reported [127,139] interval with residual velocity, and seed 4's final miss at 40222 has x=61 just outside [63,75]. Fencing and a fixed interval therefore did not certify the cartridge collision, despite seed 4 clearing both walls for 864.
- `0025` (refine of `0022`; lag-aware terminal motor selection rolls NOOP/LEFT/RIGHT through queued/sticky paddle response) averaged **516.000**, a 247.333-point drop from its parent; all 6 episodes ended `game_done`. Worst seed 1 scored 323 and died on wall 0 with 19 bricks left; best seed 3 scored 775 after clearing wall 0 but died with 32 bricks on wall 1. Both selected traces still show 5 miss windows: the worst ends at frame 12981 with paddle x=107 versus interval [127,139], and the best at frame 18934 with x=79 versus [85,97]. The rollout changed terminal requests (including NOOP/LEFT/RIGHT switches in seed 1) but did not prevent misses or improve aggregate score.
- `0026` (rewrite of `0010`; retains `0022`'s catch motor and adds observed-native-step side-wall projection, shadow checks, and scoped advisory) scored **792.833**, the new best (+29.500 versus `0022`); seeds 0/1/3/4/5 ended `game_done`, seed 2 reached the 900-s deadline at 864. Worst seed 0 died at 608 with 58 bricks on wall 1; its frame-24211 miss has paddle x=73 moving left at about 4.2 px/frame while the interval is [79,91]. Best seed 2 cleared both walls with one life at the deadline, despite 4 miss windows. Geometry traces support the mechanism: seed 0 logs 81 native wall checks, including observed x=7 matching native x=7 while continuous predicts 9; seed 2 logs 159 checks, including observed/native x=152 against continuous x=148. This supports better discrete prediction, not a causal score attribution.
- Native-step projection is the generation's directly confirmed mechanism: observed threshold crossings produce 2–4 px differences from continuous reflection, with native predictions matching the checked observation in the cited traces. Its effect on catches/completion remains unisolated; `0026` also rewrites model schedules and evidence policy and has no predictor-off ablation.
- The terminal-capture hypothesis remains incomplete: `0024`'s inside-interval miss and `0025`'s repeated interval misses show that lock lifetime, queued-input rollout, and nominal overlap alone do not certify collision. The `0026` misses still show motor/landing residuals even where native projection corrects geometry.
- Model advice is not demonstrated as the score driver: S1→act was 0.95%, 0.96%, and 0.20% for `0024`–`0026`; S2→act was 0.008%, 0.012%, and 0.004%. There were respectively 18, 11, and 11 S1 timeouts, but all three remained `ok` with stale fractions 0.000109, 0.000174, and 0.000114; asynchronous fallback kept posting actions.
- Plumbing/termination stayed valid: all three candidates completed 6 episodes with no evaluation errors and used legal replacement serves. A deadline is the 900 s/53,999-tick cap, not a failure: `0026` seed 2 reached 864 and cleared both walls before that cap; conversely, five `0026` episodes ended `game_done` after exhausting lives. No model timeout or stale-tick burden explains the remaining lost bricks.
- Remaining headroom is completion reliability: best `0026` averages `ale_reward=cartridge_score=792.833/864`, `bricks_remaining=17.67`, `lives_remaining=0.17`, `wall_index=1.00`, `walls_cleared=1.17`, and `both_walls_cleared=0.17`. The game description gives 864 for clearing both 432-point walls; only 1/6 episodes achieved both-wall completion (that seed ended at the deadline with 1 life and a live ball), while the other 5 ended `game_done`. The furthest score component is `both_walls_cleared` at 1/6, driven by life losses before clearing the last bricks in most runs.
