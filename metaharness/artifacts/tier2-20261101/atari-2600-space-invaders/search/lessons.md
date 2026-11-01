## Generation 5

- **0015** — refine of 0012; removed the exposed-only shield gate so weak recent erosion could trigger the existing bounded escape, while retaining broad combined-FIRE sweeps. Mean **392.500** (**25.38** kills, **723.00** survival ticks), up from 346.250, but all **8/8** episodes still had **0** lives and **0** waves. Traces logged **89** `shield_lane_inferred` and **63** `shield_escape_applied` events: worst seed 4 scored **145** (**15** kills, **367** ticks) and lost lives at frames **612/932** with `recent_bomb=false`; best seed 3 scored **535** (**34** kills, **811** ticks) and still lost lives at **608/2068**. Every final trace reported `jev_changed_actions=0` and `astra_effective=0`; more escapes did not establish prevented deaths.
- **0016** — rewrite of 0008; replaced decoder-label assumptions with a temporal projectile ledger, early one-token Astra rubric, and Jev evidence review while preserving full-rate FIRE. It led the generation at **405.625** (**25.12** kills, **665.25** ticks; **0.25** UFO hits), but the proposed threat sensor did not fire: every trace ended with `down_in_shot_field=0`, `jev_promotions=0`, `strong_dodges=0`, and `jev_changed_actions=0`. Worst seed 1 scored **145** (**14** kills, **422** ticks); an identity conflict at frame **964** preceded life evidence at **1096/1380**. Best seed 3 scored **625** (**28** kills, **683** ticks, **1** UFO), yet life evidence still appeared at **1128/2496**. The ledger recorded **81** unattributed chips and **18** identity conflicts, not a confirmed incoming track; the score came from the preserved sweep.
- **0017** — crossover of 0012/0003; added bounded late column intercepts (short leases, continuous FIRE, shield fallback) to the broad sweep. Mean fell to **270.625** (**20.25** kills, **602.38** ticks). Worst seed 3 scored **110** (**12** kills, **418** ticks) with **0** accepted intercepts and **0** changed actions; best seed 4 scored **515** (**33** kills, **814** ticks), with **6** accepted intercepts and **14** changed actions, but its six `intercept_outcome`s all had `score_delta=0.0` and it still lost two lives. Across the generation there were **21** accepted intercepts, **59** changed actions, and **21** zero-delta outcomes; Astra changed policy **0** times.

### General lessons

- Continuous fresh lateral movement with combined FIRE remains the supported scoring mechanism: 0015/0016 reached **392.500/405.625** points and **25.38/25.12** kills without demonstrated model changes, while 0017’s extra target selection fell to **270.625/20.25**.
- Removing the shield exposure gate increased activity (**89** inferred, **63** applied escapes) but did not validate the sensor: 0015 still logged **16** blind `life_loss_signal` events with `recent_bomb=false`.
- The temporal-ledger hypothesis was not confirmed: **0** downward shot-field tracks, **0** promotions, and **0** changed actions appeared despite **81** chips and **18** identity conflicts; incoming-threat evidence remains untested rather than solved.
- Model-selected target interception was operational but unproductive: **21** accepted leases and **59** changed actions produced **21** outcomes with **0.0** score delta; no seed-specific benefit was visible.
- Pacing was no longer the main cost (stale fractions **0.094%**, **0.156%**, **0.125%**), but plumbing remained unused: Astra produced **0** updates/effective actions, with **2** S2 timeouts in 0015 and **1** S1 timeout in 0016.
- Remaining headroom is first-wave completion/survival: best 0016 averages **405.625** Atari points, **25.12/36** invaders, **0/3** lives remaining, **665.25** survival ticks, **0.25** UFO hits, and **0/1** waves cleared; the game allows **3** lives and describes **630** points for one cleared wave, so `all_lives_lost` before wave completion is furthest from the achievable result.

## Generation 6 (automatic summary: the reflection did not finish)
### candidates/0018 — rewrite of 0015
status: ok; mean score 223.125; episodes: seed 0: 75.0 game_done · seed 1: 220.0 game_done · seed 2: 80.0 game_done · seed 3: 220.0 game_done · seed 4: 475.0 game_done · seed 5: 280.0 game_done · seed 6: 260.0 game_done · seed 7: 175.0 game_done
calls S1/S2 24/8, failed 0/0, stale 0.0018315018315018315, Jev→act 0.005504587155963303, Astra→act 0.00022935779816513763
proposal (head):
# 0018 — break the patrol/bomb phase, not the firing mechanism

**Parent:** `candidates/0015/` (rewrite as requested).  The strongest comparison is
`0016`; its continuous-fire fallback is retained, but its temporal-projectile
reviewer is not.

## Hypothesis

The useful part of the recent candidates is the actuator, not their model
interventions: a fresh combined left/right action with FIRE on ever
### candidates/0019 — crossover of 0016, 0015
status: ok; mean score 110.625; episodes: seed 0: 160.0 game_done · seed 1: 95.0 game_done · seed 2: 105.0 game_done · seed 3: 105.0 game_done · seed 4: 105.0 game_done · seed 5: 105.0 game_done · seed 6: 105.0 game_done · seed 7: 105.0 game_done
calls S1/S2 44/0, failed 0/0, stale 0.00036589828027808267, Jev→act 0.016105417276720352, Astra→act 0.0
proposal (head):
# Candidate 0019 — reversible pulses, not permanent lane reversals

**Mode:** crossover. **Parents:** `candidates/0015/` and `candidates/0016/`.

## Hypothesis

The two useful parent mechanisms operate at different timescales:

* 0015's score and survival came from the uninterrupted `right_fire`/`left_fire`
  sweep, with shield-mask changes occasionally causing an escape.
* 0016 made the sensor mo
### candidates/0020 — explore of 0016
status: ok; mean score 280.0; episodes: seed 0: 345.0 game_done · seed 1: 250.0 game_done · seed 2: 305.0 game_done · seed 3: 110.0 game_done · seed 4: 365.0 game_done · seed 5: 285.0 game_done · seed 6: 335.0 game_done · seed 7: 245.0 game_done
calls S1/S2 564/8, failed 1/0, stale 0.0005923000987166831, Jev→act 0.11062821019359936, Astra→act 0.0003951007506914263
proposal (head):
# 0020 — event-driven formation-relative patrol

Parent: **0016**.  This is a structural rewrite of its deterministic sweep, retaining only the empirically useful combined-FIRE fallback.  I did not modify any sibling candidate.

## Hypothesis

The best controller is scoring well because continuous combined FIRE and lateral motion work, but its lateral policy is a metronome rather than a real-time 


## Generation 1

- **0003 — refine of 0000:** compressed state plus typed dodge/target bridge and code fire-lock, but kept synchronous Jev. Mean **188.125** with **55.54% stale**; worst seed 4 was **95/11 kills/517 ticks**, with **207 left, 173 right, 135 fire** and life losses at frames **1104/1572/2065** while bombs were empty; best seed 0 reached **485/19/759** but still lost all lives. The representation helped, not the latency or safety hypothesis.
- **0004 — rewrite of 0002:** code-owned patrol/fire safety with pipelined Jev and Astra policy. Mean **95.625** and effectively **0.02% stale**; worst seed 3 was **0/6/355**, with **149 left, 174 right, 17 fire** and all lives lost at frame **1417**; best **335/12/752** had only **1** combined-FIRE action. Fresh ticks without productive firing did not validate the loop.
- **0005 — crossover of 0000/0002:** asynchronous tick actuator, bounded quietness, target geometry, and shoot-and-relocate. Mean **198.125** (**16.62 kills**); worst seed 3 **130/14/381** accepted **93** relocate intents and changed control **14** times, but had **0** safety overrides and **2** unobserved losses; best seed 7 **315/25/751** had **182/47** accepted/changed intents and the same two blind-loss pattern. Pacing and firing improved score, not survival attribution.
- **0006 — rewrite of 0003; proposal timed out:** the evaluated controller used committed broad traverses with continuous `*_fire`, typed Jev, and off-path Astra. Mean **280.000**; worst seed 2 was **15/3/349**, with routes 3–5 blocked at **0 pixels**, blind losses at frames **508/896**, **58** Jev accepts, **0** effective Jev/Astra changes; best seed 4 was **460/20/821** with **1 UFO** and the same zero effective model plumbing. Broad FIRE was real; adaptation was not.
- **0007 — crossover of 0005/0004:** replaced local relocation with full-lane sweep and death-conditioned blind mode. Mean **49.375**; worst seed 3 was **0/0/315** with **2** unseen life-loss events and **24** accepted intents; best seed 4 was **135/13/765** with **2** unseen losses and **85** accepted intents. The wider survival sweep preserved fresh control but lost the parent’s shot throughput; no effective safety intervention appears.
- **0008 — explore of 0005:** inferred movement limits from no-motion and retained feedback bounce/shot calibration. Mean **123.125**; worst seed 6 was **10/2/289**, with **26 fire, 185 right, 78 left**, two recoveries, and limits still **32/147**; best seed 2 was **315/24/936** with **18** calibrations but no inferred boundary. The boundary hypothesis was not exercised, while the alignment gate suppressed firing.
- **0009 — crossover of 0006/0008:** restored broad continuous-FIRE sweeps and added boundary/stall feedback plus visible-bomb overrides. Mean **208.750**; worst seed 2 was **115/10/426**, inferred only the left limit **39**, had **14 blocked finishes**, **2** unseen losses, and **4** accepted/0 effective Jev policies; best seed 0 was **540/27/753** with **2** visible-bomb overrides but **0** effective policy changes. Score came from the sweep, not feedback or Jev.
- **0010 — explore of 0008:** kept feedback limits but restored broad sweeps and FIRE on every tick. Mean **264.375** (**20.25 kills**); worst seed 0 was **115/12/680**, with **15 hits/33 misses** and two losses at `last_bomb_frame=-100000`; best seed 7 was **505/33/873** with **45 hits/37 misses**. Throughput was confirmed; bomb sensing and target-order changes were not.
- **0011 — refine of 0006:** continued the heading after a stalled route instead of immediately reversing. Mean **256.250**; worst seed 4 was **45/7/348** and still produced zero-distance blocked routes 3–5 before the blind loss at frame **508**; best seed 1 was **500/34/1093**, but Jev/Astra remained **0 effective**. Same-heading recovery did not lift the low tail and lowered the mean.
- **0012 — explore of 0006:** used shield-mask erosion and blind deaths as short-lived lane evidence while retaining broad FIRE. Mean **346.250**; worst seed 7 had **190/17/529**, **7** inferred lanes and **3** escapes, then losses at **1128/1744** with `recent_bomb=false`; best seed 0 had **620/28/675**, **11/5** inferred/applied events and losses at **1532/2092**, plus **1 UFO**. Both ended with **0** model-changed actions; the sensor remained unvalidated.
- **0013 — refine of 0006:** applied one bounded continuation after a scoreless reached traverse. Mean **204.375**; the worst seed 1 applied the extension at frame **288** after **2** arms and ended **90/11/531** after a loss at **976**; the best seed 3 applied the same extension and ended **390/27/874** after losses at **2308** and later. The intervention executed, but no score/survival gain or effective model change followed.
- **0014 — rewrite of 0009:** made Jev choose outcome-accounted reachable corridors and Astra choose a rubric. Mean **223.750**; worst seed 1 had **8** outcomes, a life loss at frame **588**, and **0** Jev changes/0 divergent actions, ending **30/8/414**; best seed 6 had **18** outcomes and **1** Jev plan change but **0** divergent actions, with **395** of **415** points from `SWEEP`; Astra’s rubric was rejected. Mechanical corridors, not model choice, produced the score.
- **0015 — refine of 0012:** removed the exposed-only shield gate so earlier erosion could trigger bounded escapes. Mean **392.500**; worst seed 4 had **6/4** inferred/applied events and losses at **612/932** with `recent_bomb=false`, ending **145/15/367**; best seed 3 had **13/9**, ended **535/34/811**, and still lost lives at **608/2068**. More escapes did not demonstrate prevented deaths.
- **0016 — rewrite of 0008:** replaced label-based aiming with a temporal projectile ledger and evidence review while preserving full-rate FIRE. Mean **405.625**; worst seed 1 ended **145/14/422** after identity conflict at frame **964** and life evidence at **1096/1380**, with **0** downward tracks, promotions, strong dodges, or changed actions; best seed 3 ended **625/28/683** with **1 UFO**, two life signals, and the same **0** threat promotions/changed actions. The sweep, not the ledger, produced the lead.
- **0017 — crossover of 0012/0003:** added short Jev-selected column intercepts inside the broad sweep. Mean **270.625**; worst seed 3 had **0** accepted intercepts and ended **110/12/418**; best seed 4 accepted **6**, changed **14** actions, but all **6** outcomes had **0.0** score delta and it still ended **515/33/814** after two losses. Target interception was operational but unproductive.
- **0018 — rewrite of 0015:** replaced the fixed triangle with event-jittered formation-relative patrol legs and life-phase resets. Mean **223.125**; worst seed 0 logged **116** route starts, **113** finishes, **2** resets, **0** visible dodges, and **3** applied Jev postures with **0** changed routes, ending **75/10/377**; best seed 4 logged **127/124**, **2** resets, **0** visible dodges, and **3/0** applied/changed routes, ending **475/32/924**. The phase change did not improve the sweep.
- **0019 — crossover of 0015/0016:** required repeated shield chips for reversible 36-frame pulses and kept combined FIRE otherwise. Mean **110.625**; worst seed 1 ended **95/10/310** after **2** pulses, **2** repeated lanes, **0** model pulses, and two losses; best seed 0 ended **160/15/394** after **2** pulses and **1** model pulse/change, still with two losses and no explicit tracks. Pulses disrupted the productive baseline without validating hidden-threat evidence.
- **0020 — explore of 0016:** shortened and jittered patrol legs by formation phase and reset after life loss. Mean **280.000**; worst seed 3 changed **OPENING→PRESSURE**, made **10** reversals, had **2** life transitions and **0** visible escapes, ending **110/13/386**; best seed 6 reached **PRESSURE→CRITICAL** with **37** reversals and **26** recoveries but a Jev timeout, ending **335/25/807**. Phase plumbing ran, yet score and survival remained below 0016.

### General lessons

- Continuous combined FIRE plus a fresh-tick broad sweep is the confirmed mechanism: **0010/0012/0015/0016 = 264.375/346.250/392.500/405.625**, while selective/aim-heavy **0007/0008/0019 = 49.375/123.125/110.625**; **0004** shows that near-zero stale ticks alone are insufficient when only **1** combined-FIRE action occurs.
- Target, route, and sensor interventions were not causally demonstrated: **0006** had **58** Jev accepts and **0** effective changes, **0012** had **0** changed actions, **0014** had **1** plan change but **0** divergent actions, and **0017** had **21** zero-delta intercept outcomes across the generation’s intercepts.
- Hidden-threat/boundary hypotheses remain untested or refuted: shield escapes (**0012/0015**) coexist with blind losses and `recent_bomb=false`; **0016** saw **0** downward tracks/promotions; **0008** inferred **0** boundaries; and **0009** learned only the left limit while the right stayed unknown.
- Pacing moved from a contract cost to a plumbing cost: synchronous **0003** spent **55.54%** stale ticks, but later candidates were near zero stale while Astra’s **32–50 s** calls mostly arrived too late; timed-out S1/S2 calls and malformed/rejected policy answers left the code fallback in control.
- The score distribution still rewards risky partial waves: the high **620/625** traces include a UFO and end after two life losses, while route extensions/intercepts/phase changes lowered means; no candidate cleared a wave.
- Remaining headroom: best **0016** averages **405.625** Atari score, **25.12/36** kills, **0/3** lives remaining, **665.25** survival ticks, **0.25** UFO hits, and **0/1** waves cleared; all **8/8** episodes end `all_lives_lost`. The description says a complete wave is **630** points and the objective is to preserve lives while clearing it, so the largest failure is terminal all-lives loss before the first wave clear (also leaving **10.88** kills and **224.375** points short of one wave).
## Generation 2

- **0006** — Rewrite of 0003; proposer timed out before producing `proposal.md`. Mean **280.0** (977 S1, 10 S2, 3 failed S2). Scores ranged 15–460. Without the proposal or traces, the mechanism and causal explanation cannot be established.
- **0007** — Crossover of 0005 and 0004. Combined code-owned aiming/fire locking and shoot-relocate with categorical Jev judgments, strict freshness gates, and empirical full-lane sweeping after deaths. Hypothesis: separate firing alignment from survival movement and avoid repeatedly dwelling in dangerous lanes. Mean **49.375**, versus 198.125 for 0005 and 95.625 for 0004. Four failed S1 calls, 905 S1 calls total. The proposed boundary/death-memory mechanism did not produce a robust improvement. Detailed trace evidence unavailable in the supplied context.
- **0008** — Explore of 0005. Tested feedback-calibrated movement boundaries and interior bouncing, hypothesizing that the parent wasted escape opportunities by issuing movement into a clipped boundary. Mean **123.125**, versus 198.125 for 0005. Scores ranged 10–315; two failed S1 calls. The proposed boundary correction did not improve the parent. Detailed trace evidence unavailable in the supplied context.
- **0009** — Crossover of 0006 and 0008. Mean **208.750**, 394 S1 calls, zero failures. Best episode 540, worst 65. Mechanism and trace-level explanation require the missing proposal and traces.
- **0010** — Explore of 0008. Mean **264.375**, 734 S1 calls, zero failures. Best episode 505, worst 115. Improved over 0008, but the specific mechanism cannot be verified from the supplied proposal evidence.
- **0011** — Refine of 0006. Mean **256.250**, 977 S1 and 9 S2 calls, two failed S2 calls. Scores 45–500. Mechanism and trace explanation unavailable.
- **0012** — Explore of 0006. Mean **346.250**, 152 S1 and 8 S2 calls, zero failures. Scores 190–620. Strong improvement with substantially fewer S1 calls. Mechanism and trace explanation unavailable.
- **0013** — Refine of 0006. Mean **204.375**, 941 S1 and 11 S2 calls, three failed S2 calls. Scores 60–390. Did not preserve 0006's performance.
- **0014** — Rewrite of 0009. Mean **223.750**, 131 S1 and 10 S2 calls, zero failures. Scores 30–415. Mechanism and trace explanation unavailable.
- **0015** — Refine of 0012. Mean **392.500**, 33 S1 and 8 S2 calls, two failed S2 calls. Scores 145–535. Improved over 0012 while reducing S1 calls to 33.
- **0016** — Rewrite of 0008. Mean **405.625**, the current best, with 104 S1 and 8 S2 calls. One failed S1 call. Scores 145–625. The rewrite substantially improved on its parent despite retaining the same ancestry.
- **0017** — Crossover of 0012 and 0003. Mean **270.625**, 40 S1 and 8 S2 calls, zero failures. Scores 105–515. Did not retain 0012's performance.
- **0018** — Rewrite of 0015. Mean **223.125**, 24 S1 and 8 S2 calls, zero failures. Scores 75–475. Lower than its parent despite fewer S1 calls.
- **0019** — Crossover of 0016 and 0015. Mean **110.625**, 44 S1 calls, zero S2 calls. Seven of eight episodes scored exactly 105; the remaining episode scored 160. Strong evidence of a consistently poor behavioral regime compared with either parent.
- **0020** — Explore of 0016. Mean **280.000**, 564 S1 and 8 S2 calls, one failed S1 call. Scores 110–365. Did not improve on its parent.

### General lessons for next proposers

- The best results increasingly came from reducing dependence on model calls: 0015 achieved 392.5 with 33 S1 calls; 0016 achieved 405.625 with 104. Both had only 8 S2 calls.
- Exploration and rewriting can outperform refinement: 0012 reached 346.25 from 0006; 0016 reached 405.625 from 0008. Parent lineage alone does not predict performance.
- Crossover is not reliably beneficial: 0017 scored 270.625 and 0019 scored 110.625, substantially below their stronger parents.
- S2 failures are not automatically fatal to evaluation: 0015 scored 392.5 despite two failed S2 calls. However, 0016's single S1 failure and 405.625 score show that reducing failures remains worthwhile.
- Contract/plumbing costs include failed calls, freshness constraints, and long S2 latency. The observed S2 latency model is approximately 32.1 seconds plus 24.5 ms per output token; design timeouts accordingly. The exact trace-level causes and untested mechanisms require the missing candidate artifacts.

- **Remaining headroom:** 0016's mean score composition is 405.62 Atari points, 25.12 invaders killed, 665.25 survival ticks, 0.25 UFO hits, zero remaining lives, and zero waves cleared. All eight episodes end with `game_done`, with scores ranging from 145 to 625. Survival and kills are already substantial, but zero waves cleared across all eight episodes identifies wave completion as the clearest unachieved milestone. The game description and episode traces were not accessible in the supplied context, so the precise achievable wave target and terminal failure mechanism cannot be established.


## Generation 3 (automatic summary: no reflection was recorded)
### candidates/0009 — crossover of 0006, 0008
status: ok; mean score 208.75; episodes: seed 0: 540.0 game_done · seed 1: 325.0 game_done · seed 2: 115.0 game_done · seed 3: 165.0 game_done · seed 4: 115.0 game_done · seed 5: 230.0 game_done · seed 6: 65.0 game_done · seed 7: 115.0 game_done
calls S1/S2 394/0, failed 0/0, stale 0.00043280675178532783, Jev→act 0.08443386014288808, Astra→act 0.0
proposal (head):
# Candidate 0009 proposal

## Parents and hypothesis

Parents: `candidates/0006/` and `candidates/0008/` (crossover). Candidate 0006 is the leaderboard leader and supplies the primary mechanism: code-owned committed broad sweeps with FIRE held during movement. Candidate 0008 contributes feedback-based boundary inference, shot/life outcome tracking, conservative model integration, and a latest-only
### candidates/0010 — explore of 0008
status: ok; mean score 264.375; episodes: seed 0: 115.0 game_done · seed 1: 190.0 game_done · seed 2: 345.0 game_done · seed 3: 195.0 game_done · seed 4: 145.0 game_done · seed 5: 330.0 game_done · seed 6: 290.0 game_done · seed 7: 505.0 game_done
calls S1/S2 734/0, failed 0/0, stale 0.0006845798391237378, Jev→act 0.12536393218016784, Astra→act 0.0
proposal (head):
# Candidate 0010 proposal

## Parent and hypothesis

Parent: `candidates/0008/`, with the sustained-sweep mechanism from the best
candidate `0006` as a comparison.  I read the parent code/proposal and the
public worst traces; I did not inspect game implementation source or the
unfinished sibling directories.

**Hypothesis:** the biggest remaining failure is not only the learned movement
boundary. 
### candidates/0011 — refine of 0006
status: ok; mean score 256.25; episodes: seed 0: 110.0 game_done · seed 1: 500.0 game_done · seed 2: 100.0 game_done · seed 3: 225.0 game_done · seed 4: 45.0 game_done · seed 5: 330.0 game_done · seed 6: 470.0 game_done · seed 7: 270.0 game_done
calls S1/S2 977/9, failed 0/2, stale 0.0012062726176115801, Jev→act 0.1678743961352657, Astra→act 0.0003450655624568668
proposal (head):
# Candidate 0011 — continue through a stalled route instead of reversing

Parent: `candidates/0006/` (the current best candidate, mean 280.0).

## Hypothesis

The useful part of 0006 is its sustained, firing lateral traverse, but its
stall recovery can turn that traverse back into a short oscillation.  When the
controller has not seen the cannon move for 24 frames, `action()` marks the
route `bloc


## Generation 4 (automatic summary: no reflection was recorded)
### candidates/0012 — explore of 0006
status: ok; mean score 346.25; episodes: seed 0: 620.0 game_done · seed 1: 430.0 game_done · seed 2: 315.0 game_done · seed 3: 215.0 game_done · seed 4: 360.0 game_done · seed 5: 310.0 game_done · seed 6: 330.0 game_done · seed 7: 190.0 game_done
calls S1/S2 152/8, failed 0/0, stale 0.00020016012810248197, Jev→act 0.03043043043043043, Astra→act 0.0
proposal (head):
# Candidate 0012 — use shield erosion as a weak hidden-bomb sensor

Parent: `candidates/0006/` (best prior mean, 280.0); implementation also retains the broad-fire sweep structure validated by 0010.

## Hypothesis

The repeated failure is not stale control or lack of firing throughput.  The
best candidates post fresh actions, hold FIRE, and still lose lives while the
symbolic bomb list is empty.  
### candidates/0013 — refine of 0006
status: ok; mean score 204.375; episodes: seed 0: 265.0 game_done · seed 1: 90.0 game_done · seed 2: 225.0 game_done · seed 3: 390.0 game_done · seed 4: 90.0 game_done · seed 5: 295.0 game_done · seed 6: 60.0 game_done · seed 7: 220.0 game_done
calls S1/S2 941/11, failed 0/3, stale 0.0016108824055843924, Jev→act 0.16798135532448907, Astra→act 0.00017927572606669058
proposal (head):
# Candidate 0013: Bounded Scoreless Traverse Extension

## Parent and Hypothesis

Parent: `candidates/0006/` (mean score 280.0). This is one targeted change to
its route-boundary transition. The tick-local controller, persistent latest-only
Jev worker, four typed questions, firing policy, bomb override, and off-path
Astra policy revision are preserved.

**Hypothesis:** immediately reversing after 
### candidates/0014 — rewrite of 0009
status: ok; mean score 223.75; episodes: seed 0: 145.0 game_done · seed 1: 30.0 game_done · seed 2: 170.0 game_done · seed 3: 330.0 game_done · seed 4: 160.0 game_done · seed 5: 375.0 game_done · seed 6: 415.0 game_done · seed 7: 165.0 game_done
calls S1/S2 131/10, failed 0/0, stale 0.0, Jev→act 0.026915964659954796, Astra→act 0.0004109307581672488
proposal (head):
# 0014 — Outcome-accounted corridors, not repeated strategy labels

## Parent and hypothesis

**Parent:** `0009` (rewrite). Comparisons: best candidate `0006`, and `0010`/`0011` as recorded in the archive. Only this candidate directory is modified.

**Hypothesis:** the successful mechanism is continuous FIRE during lateral travel, but the current hierarchy cannot respond to evidence that a particu

## Generation 7

I couldn't access the specified working directory or candidate files in the available execution environment. The supplied leaderboard and proposal excerpts support the following preliminary findings, but the trace-specific evidence remains unverified.

- **0021 — crossover (0016, 0018):** Proposer timed out after 3,600 seconds. No proposal, evaluation, episodes, or model calls. Fatal status; unusable regardless of potential performance.
- **0022 — explore (0018):** Mean 284.375 over 8 episodes; 152 S1 calls, 8 S2 calls, 0% stale ticks, 2 failed S2 calls. Scores: 155, 100, 120, 550, 115, 90, 600, 545. Hypothesis: acquire only when the laser is empty while retaining continuous FIRE and lateral movement. The highly polarized results suggest the firing schedule did not consistently improve performance. Worst/best trace evidence requires verification.
- **0023 — refine (0016):** Mean 405.625 over 8 episodes; 104 S1 calls, 8 S2 calls, zero failures, 0% stale ticks. Scores: 520, 145, 395, 625, 560, 210, 360, 430. Hypothesis: calibrate joystick polarity before temporal safety and projectile-path prediction. It tied 0016's mean rather than improving it. The identical score components and episode scores indicate no measurable aggregate benefit from the calibration change. Trace-level causal attribution requires verification.
- **General lessons:**
  - Continuous-FIRE sweep architectures remain the strongest demonstrated family: 0016 and 0023 both average 405.625.
  - Empty-laser acquisition remains unproven as an improvement: 0022 trails the best by 121.25 points.
  - Joystick-polarity calibration did not improve the parent across the 8-seed aggregate.
  - Contract/plumbing reliability matters: 0021 lost the entire generation evaluation to a 3,600-second proposer timeout; 0022 also had 2 failed S2 calls.
  - Temporal safety, projectile interception, and the actual contribution of Astra review remain insufficiently isolated by these aggregate results.
- **Remaining headroom:** Candidate 0023 averages 405.625 Atari points, 25.12 invaders killed, 665.25 survival ticks, 0.25 UFO hits, zero remaining lives, and zero waves cleared. All 8 episodes end with `game_done`. The game description's achievable outcome must be checked against the actual task specification before asserting a numerical gap. The clearest measurable shortfall is wave completion: **0 waves cleared**, despite 25.12 kills per episode on average.

## Generation 8

- **0024 — explore of 0011:** Added late outer-column excursions to the x=38..116 combined-FIRE sweep. Mean **274.375** (**19.38 kills**, **660.25 survival ticks**, **0.12 UFO hits**, **0/8 waves**, **0 lives**). Worst seed 0 scored **215/20/688**; gaps appeared at frames 0, 844, 1888, and 2316, but only two extensions armed (136, 972), one finished (1652), and life evidence arrived at 748/1760 before 16 invaders remained. Best seed 4 scored **555/25/850** with 1 UFO; it also armed two and finished one, while 23 accepted Jev answers were all `KEEP` with 0 changed extensions, then ended all-lives-lost with 11 invaders alive. The gap detector ran, but no late-kill or survival benefit was shown.
- **0025 — refine of 0023:** Reversed the fresh combined-FIRE sweep only after a confirmed life decrement, retaining the temporal ledger and model reviews. Mean **394.375** (**24.38 kills**, **663.50 survival ticks**, **0.38 UFO hits**, **0/8 waves**, **0 lives**), below parent 0023 by **11.25**; 117 S1/8 S2 calls included 1 S2 timeout. Worst seed 1 scored **260/19/526**: reversals occurred at frames 1092 and 1372, but the identity conflict at 964 was followed by **0** downward tracks, strong dodges, chip escapes, or Jev-changed actions, and all lives were lost. Best seed 4 reached **775/31/799** with 2 UFOs, one reversal and one projectile escape; the trace still saw 2 lives at its last evidence summary, but the terminal observation was all-lives-lost and 0 waves. The reversal executed without demonstrating prevention.
- **0026 — rewrite of 0009:** Replaced target/evidence decisions with a 180-frame outer-half recovery after each observed life decrement, while retaining broad combined FIRE. Mean **389.375** (**26.25 kills**, **678.88 survival ticks**, **0.12 UFO hits**, **0/8 waves**, **0 lives**), with 16 S1/8 S2 calls, 1 S2 timeout, and 1 stale tick. Worst seed 1 scored **255/21/547**: left recoveries started at frames 1412 and 1700 and both completed, yet `life_losses=2`, `model_recoveries=0`, and the episode ended all-lives-lost. Best seed 4 scored **565/27/693** with 1 UFO; both recoveries again chose LEFT at 1556/2556, Astra's `COVER` arrived after the second start, and the final result still had 0 lives and 0 waves. The recovery plumbing ran, but the death-zone heuristic did not improve the baseline.

### General lessons for next proposers

- Fresh, code-owned lateral movement with combined FIRE remains the only confirmed scoring mechanism: 0024/0025/0026 averaged **274.375/394.375/389.375**, all below the **405.625** best, while their new interventions never changed the terminal outcome class.
- The proposed mechanisms executed but were not causally validated: 0024 logged **32** gaps, **18** arms, and **7** finishes; 0025 logged **15** reversals; 0026 logged **16/16** recovery starts/finishes. Every episode in all three candidates still ended with all lives lost and 0 waves.
- Model plumbing remained mostly advisory or unused: 0024 had **119** accepted `KEEP` answers and **0** changed extensions; 0025 had **35** evidence verdicts and **0** changed actions; 0026 had **0** model recoveries. Each candidate incurred **1** S2 timeout.
- Empty/late threat evidence is still the contract gap: 0025 recorded **85** unattributed chips but **0** downward tracks, and 0026's observed life transition triggered recovery only after the delayed event; terminal observations could report 0 lives after the controller's last evidence still reported 2.

- **Remaining headroom:** The best candidate, 0023, averages **405.625** Atari points, **25.12/36** invaders, **665.25** survival ticks, **0.25** UFO hits, **0/3** lives remaining, and **0/1** waves cleared. The description says a wave is achievable at **630** points with lives to preserve; all 8 episodes end `all_lives_lost`, so the furthest failure is the **0/1 wave-clear and 0/3-life** terminal state (also **10.88** kills and **224.375** points short of one wave).

