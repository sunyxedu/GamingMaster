# Strategy prior for Hanabi self-play

Optimize **expected terminal team score, 0–25, with zero on life exhaustion**, under each acting seat's information boundary. One shared policy can agree on conventions before the game, but cannot carry another seat's private view into the current seat's reasoning. There is no opponent to defeat and no timing pressure. Astra can reason at every turn; most gains should come from accurate code bookkeeping and a coherent communication policy. Jev is optional, not the engine for arithmetic or multi-turn planning. [S1, S2]

## Evidence status and source map

External access failed during this study; the following are prior-known references, not newly retrieved pages. No claimed score, threshold, or latency below is a measured result. Exact community conventions vary by group and level. The simple protocol proposed below is an **engineering recommendation**, not a claim to reproduce H-Group or an optimal published agent.

* **S1 — Hanabi rules:** https://www.ultraboardgames.com/hanabi/game-rules.php . Actions, information restrictions, multiplicities and ending.
* **S2 — Bard et al., The Hanabi Challenge: A New Frontier for AI Research:** https://arxiv.org/abs/1902.00506 . Cooperative partial observability, established rule/convention approaches, learned agents, and the difference between self-play and ad-hoc play.
* **S3 — H-Group community conventions:** https://hanabi.github.io/ . Play/save clues, focus, chop, finesse, prompts, tempo and more advanced convention systems. Treat the site as a community reference, not mandatory rules. Deep page locations and current wording were not verified.
* **S4 — Foerster et al., Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning:** https://arxiv.org/abs/1811.01458 . Action choice communicates information under a common policy; public belief reasoning and conventions.
* **S5 — HLE reference engine:** https://github.com/google-deepmind/hanabi-learning-environment . Legal moves, public/hidden state separation, benchmark mechanics. Its examples are useful baselines, not evidence of optimal strategy.

## Division of work

| Capability | Code | Astra prompts | Jev questions |
|---|---|---|---|
| Rules and legality | Authoritative legal actions, counters, masks, score, final-round horizon | Explain a proposed legal move in terms of objective | Never decide legality from intuition |
| Beliefs | Remaining copies, literal clue masks, correlated hand beliefs, feasibility, probabilities | Interpret intentional clues under the fixed protocol; identify competing explanations | Optional classify intent when given already computed features |
| Communication | Stable IDs, shared protocol and separate seat memory, focus/commitment tracking, recipient-perspective simulation | Choose an informative clue or multi-seat sequence and discuss ambiguity | Optional choose among prevalidated clue candidates |
| Tactics | Enumerate legal moves; simulate revealed outcomes; search with information constraints | Trade score, life, clues, deck consumption and turn order | Optional preference among short summaries, never arithmetic |
| Long horizon | Deadline and reachable-score bounds; Monte Carlo estimates if worthwhile | Plan which color dependencies to unlock and which hand to free | Unnecessary |

The harness should receive an observation, calculate exact facts, ask Astra only when there is a meaningful choice, validate the returned action against the current legal list, and execute one turn. Avoid repeated model calls to re-count cards or re-derive token rules. Do not create a hidden fast real-time control loop for this tier-1 task.

## 1. Exact card accounting and literal deduction

**Code; Astra consumes summaries.** Track copies for each identity, successful stacks, public discards including failed plays, visible hands, and own literal constraints. An unplayed card `(c,r)` is immediately playable iff `r == height[c] + 1`. It is already obsolete iff `r <= height[c]`. A higher card is also dead if a required lower rank has no surviving copies anywhere outside the public played/discarded zones. Keep “dead” distinct from “currently unplayable”: a red 4 above a red 1 is often valuable. [S1, S2]

For current seat i, define unseen count `U(c,r) = original copies - successful copies - discarded copies - copies visible in other hands`. Those unseen cards are distributed between i's own hand and the unknown deck. A completed stack contains one successful copy of every rank up to its height. Avoid counting those successful cards again as discarded. Negative U indicates an accounting bug, not a surprising belief.

For each own card maintain the set of identities consistent with **all positive and negative clues** it received while held. Intersect that with physical availability. A clue naming blue removes blue from every untouched card of the target at that moment. It does not constrain future replacements. Track touched IDs and hand shifts exactly. A card need not be fully identified to be safe: if every possible identity is next on its respective stack, it can be played with certainty. Conversely, “rank 1” ceases to imply safe after some colors' 1s have already been played.

Single-card approximation: weight compatible identities by U. For a better belief, enumerate/sample **joint assignments of the entire own hand without replacement**, respecting all masks, with weights proportional to remaining-copy counts at each assignment. Slots are correlated: two cards cannot both be the sole remaining blue 5. An identity feasible for one slot alone may become impossible given the other slots. Deal histories and remembered observations may add legitimate seat-local constraints. Prune impossible assignments before calculating marginals.

For each own slot, calculate `p_playable`, `p_dead`, `p_critical`, likely identities, and guaranteed classifications. These are exact only under an explicitly specified belief model; copy-count priors conditioned solely on clues do not account for all policy-dependent signaling. Do not call a pragmatic inference mathematical certainty. [S2, S4]

**Useful exact examples:**

* Empty stacks and a rank-1 clue: every touched card is playable **now**, but duplicate same-color 1s may make the second one dead after the first play. Recompute after every action.
* Red stack at 2 and both red 3s discarded: red 4/5 are dead and the color's achievable ceiling is 2.
* Red stack at 0 and one red 2 discarded: the other red 2 is critical, even though it cannot yet be played.
* A blue clue touches one of three cards: that card is blue; the two untouched cards are not blue, regardless of whether the intended message was play or save.

## 2. Critical cards, reachable score, and safe discards

**Code for classification and bounds; Astra for tradeoffs.** An unplayed reachable identity is critical when only one physical copy remains outside played/discarded zones. “Only one currently visible” is not the same thing: another may be in the deck or one's own hand. All live 5s begin unique; 2s/3s/4s become unique after one copy is discarded; 1s after two. A unique higher card may stop being valuable after its color is already blocked. [S1, S3]

For each color, find the first missing rank above its height with no surviving copies. Its maximum eventual height is one less than that rank, or 5 if none is missing. Sum these heights for a physical upper bound. This is optimistic: insufficient final turns, inaccessible hands, lives and clue resources can reduce the realizable score further. Discarding the last red 2 can forfeit red 2,3,4,5, not just one point.

Discard preference, subject to the convention's protected cards:

1. Certainly dead/obsolete card, freeing hand space and replenishing a clue.
2. A redundant card when another copy is known to be protected or imminently playable; account for both seats making the same discard decision.
3. Low expected marginal score loss among unprotected cards, considering critical probability, rank dependencies and draw benefit.

Never discard solely because a card is high rank. A unique 5 can wait safely if there is no deadline, and another clue may be needed to preserve it. Also do not reflexively save every 5 at any cost: a save that exhausts clues or blocks immediate progress can be inferior to securing several achievable lower ranks. Save urgency is determined by likely discard timing, not rarity alone. [S3]

## 3. Opening and early-game communication

**Code enumerates truthful clue candidates and touched IDs; Astra chooses the productive sequence.** With empty stacks, rank-1 clues are often efficient because every touched 1 is immediately playable. Prefer clues that create several **usable** plays, unlock different colors, and leave the next seat an informed move. At the initial eight clues, discarding is forbidden in this profile; a clue is often the natural opening. A blind play is a risk, not the default opening. [S1–S3]

Evaluate actionability, not raw number of touched cards. Three 1s containing duplicates are not three guaranteed points. A color clue touching both red 1 and red 4 is ambiguous without a focus convention. Giving information to a later seat may leave the immediate next seat with no safe move and no good token action. In 2-player games, each clue is delivered to the next actor, making simple protocols particularly effective; in 3–5 players, intervening turns and clue urgency matter much more.

Default opening protocol: start with literal clues and a published discard convention; add play/save signaling only once both clue giver and receiver use the same decoder. Do not introduce a finesse spontaneously because Astra notices an attractive line.

## 4. Clue value: actionability, saving, tempo, and negative information

**Code scores concrete consequences and constructs recipient inputs; Astra plans.** A clue costs both one token and a whole turn, while play/discard can draw fresh cards. A high-quality clue can enable a safe play, save a threatened critical card, establish a sequence, or let the recipient discard the remainder safely. Often the negative information is as useful as the touched card. Count what the recipient can infer after the clue, not what the giver knows from seeing the hand. [S2–S4]

Useful candidate features include safe plays newly enabled, number of distinct needed identities involved, protected critical cards, ambiguity of focus, effect on the next seat's move, whether other touched cards become falsely marked playable, clue balance after acting, and whether the deck/final-round clock makes the information timely. A clue with many touches may create no actionable information; a single-touch clue may unlock a 5 and refund its cost on play.

Simulate the **recipient perspective** when testing a clue. The giver cannot directly know every recipient-visible fact because the giver cannot see its own cards, which the recipient does see. Use appropriate belief samples or conservative reasoning for that uncertainty. Do not hand the recipient simulation the omniscient engine state or transplant the giver's view as if it were the recipient's view.

At low clues, discarding a known dead card often supports the entire team better than playing another safe low card immediately. At full clues, a legal informative or redundant clue may be necessary when no safe play is known. Repeated clues can deliberately pass tempo or avoid a harmful discard, but every clue still consumes a turn; there is no free “wait.”

## 5. A safe initial self-play protocol

**Code should define this protocol; Astra must follow it rather than inventing rules midgame.** This is a proposed baseline protocol, inspired by human conventions [S3], designed for reliability rather than maximum compression:

* Store literal knowledge separately from inferred intent. Literal knowledge is never overridden by a convention. Mark deductions with provenance: `literal`, `physical-count`, or `protocol`.
* Define **chop** as the oldest currently unprotected own card. Protection initially means received a clue; optional extensions can release cards proven dead. The rule must be deterministic, common to all seats, and based on information available to the discarding seat. A clue giver can predict the recipient's chop from public touch/protection history.
* A clue touching chop is conservatively treated as a **save/protection clue**, with no automatic promise that the touched cards are playable. This deliberately sacrifices some play-clue efficiency to reduce convention ambiguity. All touched cards retain their literal information and are protected. Repeated clues to already protected cards do not automatically order a play.
* First version: play only by literal/count certainty; use clues to refine candidates and discard known dead cards or unprotected chop. This is a safe baseline, not an expectation of top scores. Clue-starved hands with no unprotected cards need an explicit fallback based on smallest expected discard loss, not an undefined chop.
* A later, separately evaluated version may add a **direct play clue**: when a clue does not touch chop and newly touches exactly one card, the giver promises that focus card is playable at the receiver's next turn. Require the promise to survive planned intervening moves; incidental previously touched cards receive no new play instruction. Multiple newly touched non-chop cards are literal-only in this simple version. Every actor must use the same interpretation, and inferred promises must be checked against later public changes before execution.
* Preserve card IDs for pending plays and saves; cancel or revise commitments when stacks advance, the card moves, or a preceding dependency fails. There is no automatic “all clued cards are plays.”

The direct-play extension is reliable only if the giver enforces its semantic promise and the receiver knows it is in use. It is **not the complete H-Group focus/save system**. Compare baseline versus extension on held-out deals. More communication is useful only when the recipient can decode it unambiguously.

## 6. Stronger conventions: finesse, prompts and indirect messages

**Astra can plan; code must implement and validate the shared protocol. Jev should not improvise it.** Strong human groups use focus rules, save clues, finesse, prompts, order conventions, and other signals to extract more than literal color/rank information. These achieve coordinated chains with fewer clues but require a precise common interpretation. Community definitions differ; importing one term without its prerequisite rules is dangerous. [S3, S4]

A representative **finesse**: A gives C a clue interpreted as a play on a red 2 while red needs 1; the agreed convention can tell intervening B to play a particular otherwise unknown red 1 first. A valid line requires B really to hold the prerequisite, the correct blind-play position to be unambiguous, C to wait until B acts, and no competing interpretation. This is illustrative, not a complete finesse rule or a license to blind-play any unclued card. A **prompt** similarly asks an intervening player to use an already clued card under an agreed protocol; distinguishing a prompt from a finesse is protocol-specific.

Implement advanced conventions only after defining: clue focus; save priority; which seat acts first; exact card selection order; how waiting is represented; what happens if the prerequisite fails; how unrelated clues cancel or preserve a pending line; and how to disambiguate multiple possible chains. All fields in this commitment tracker must be inferable from allowed observations and public actions. An incorrect finesse can cost multiple lives across a chain, so one-model “confidence” is not adequate validation.

Convention agents and learned self-play agents can also use action choice as a code. Hints need not maximize literal entropy reduction; the choice among legal hints communicates under a fixed policy. Bayesian Action Decoder provides the research basis for this perspective [S4]. A compact signaling protocol is an optional research direction, **not an initial dependency**: legal hint availability, slot shifts, final hands and all token counts must be covered before using it. Never add an out-of-band message action or use arbitrary timing to communicate.

## 7. Resource and risk management

**Code computes probabilities and candidate values; Astra chooses risk posture under the actual score rule.** Lives are a limited budget for uncertain plays. Two lives left can justify a carefully assessed chance to unlock valuable cards; one life left makes failure a loss of the entire accumulated score. Use expected terminal value, not a fixed “play if confidence > 0.8” policy. [S1, S2]

For a candidate uncertain play with estimated success probability p, compare `p * expected_value_after_success + (1-p) * expected_value_after_failure` against the best clue/discard alternative. Values include the loss of the card on failure, future draws, future information and remaining lives. At one life, failure value is zero under this benchmark. If the alternative preserves a final score S and success adds exactly one point with no other effects, the illustrative threshold is `p > S/(S+1)`; at S=20 this exceeds 95%. This formula is for that restricted scenario, not a universal threshold. Arithmetic belongs in code.

Avoid consuming a life on a card that is merely likely to be a duplicate; blindly discarded critical cards can also damage the final score without losing lives. Resource management includes keeping clues available for urgent saves, avoiding eight-token waste when a 5 could recover a token, and using dead-card discards to refresh the deck. Maximizing life preservation alone is not the objective, but a bomb-out destroys the terminal score in this profile.

## 8. Deck clock and final-round planning

**Code maintains horizon and runs small search; Astra selects the line.** Every nonterminal play/discard draws while cards remain; clues slow deck consumption but still take turns. Sometimes a discard is valuable to expose a needed card; sometimes a clue preserves the deck long enough for a sequence. Count future turns before a player will next act, not just remaining cards. [S1, S2]

After the last draw there are exactly N further player actions. Plan backward from that order: which cards can be played before their holders lose their last turn? A clue delivered to a seat that will never act again produces no playable point. A card visible in a partner's hand is only useful if they can identify the right slot and have a turn. A new 5 drawn on the last draw can still be played by its drawer on that drawer's one extra turn if information reaches it in time.

Final-round discards do not draw. Their only direct benefit is token generation, potentially enabling a later seat's clue and then an even later play. With two seats, many such sequences no longer fit. The next guaranteed point is often more valuable than saving a card that cannot be played before the end. Stop protecting unachievable future dependencies, but do not knowingly bomb out merely because the game is nearly over.

Use branch-and-bound or a short rollout over legal actions when the horizon is small. Terminal scores and move counts are exact; uncertain own identities still need beliefs. Return executable **contingent** plans that reevaluate after observed draws and revealed plays, not an inflexible action sequence that assumes one hidden deal.

## 9. Search that respects information

**Code for inference/search; Astra for candidate generation and qualitative assessment.** Small hands make enumeration of clue-compatible own assignments practical with pruning, while simulating an entire unseen deck and every policy branch can be expensive. Sample hidden worlds according to multiplicity-weighted beliefs and compare candidate moves by expected terminal score. Give each simulated actor only its legal view, plus its own protocol memory. A simulated partner must not act optimally on a card face it cannot see. [S2, S4]

Beware **strategy fusion**: a plan may look excellent when it chooses different future actions in sampled worlds that are observationally identical to the actor. Such a policy cannot be executed. Likewise, maximizing over unknown draw outcomes is clairvoyance, not search. Re-determinizing independently for each seat can break consistency with public history; validate all samples and commitments. A simpler honest rule policy is preferable to an optimistic but invalid rollout oracle.

Beliefs may incorporate the likelihood of earlier actions under the agreed policy: a player choosing a particular clue can exclude interpretations it would never intentionally signal. Only do this when the actual sender follows that policy. Do not assume every missed opportunity proves a card unplayable or every discard perfectly encodes a message; uncertain model choices make action-likelihood assumptions brittle. Store protocol version and distinguish deliberate signals from fallback actions.

## 10. Astra prompt content and optional Jev questions

**Astra input should include:** current-seat redacted observation; ordered legal action IDs/objects; public protocol version; candidate identities and certainty provenance; computed play/dead/critical probabilities; clue effects evaluated from recipient perspective; predicted chop and urgent saves; achievable color ceilings; final-turn order; and current-seat pending commitments. Keep full public event history available but summarize it into relevant facts for short prompts.

**Astra instruction:** choose one legal action to maximize expected terminal score. Treat literal facts and engine counts as authoritative. Apply only the declared conventions. Consider whether the recipient can actually decode a clue. Prefer a concise action plus intent/reason; put persistent information only in the current seat's memory or the explicitly public tracker. Do not ask for or infer deck seed from evaluation metadata. Let code parse/validate the choice; handle malformed output without taking an unintended move.

**Jev is optional** and may batch independent typed judgments over precomputed candidates. Appropriate examples:

| Typed question | Choices / input requirement | Use |
|---|---|---|
| Which intent best matches this clue under protocol version X? | `play`, `save`, `literal_only`, `ambiguous`; include focus/chop/touched facts | Flag ambiguity for Astra; cannot override deterministic protocol rules |
| Which of these already legal clue candidates is clearest? | Candidate IDs and computed recipient deductions | Advisory selection when no multi-turn planning is needed |
| Does this plan rely on an undeclared convention? | yes/no, with protocol and concise plan | Additional review, not a proof |
| Which candidate needs deeper planning? | Candidate IDs with resource/horizon summaries | Prioritize Astra attention |

Do not ask Jev to count copies, calculate probability, choose legal actions from raw text, maintain convention state, or solve a multi-seat chain. Code should answer exact questions; Astra should handle planning. The tier-1 world waits, so no latency-based reason exists to force a Jev call.

## 11. Common failures and their remedies

There are no adversarial counters in ordinary cooperative Hanabi. The relevant “counters” are responses to resource bottlenecks, ambiguity and inconsistent teammates.

| Failure / bottleneck | Remedy | Owner |
|---|---|---|
| Same agent remembers a card face from another seat | Isolate private histories/model sessions by seat; share public transcript only | **Code** |
| Every touched card treated as playable | Separate literal clues, save status and play promises; recompute after each play | **Code + Astra** |
| Contradictory convention interpretations | Freeze protocol and priority rules; downgrade ambiguous clues to literal information | **Code + Astra** |
| Card remembered by stale slot | Track stable IDs, remap action index at decision time | **Code** |
| Duplicate plays after a multi-card 1 clue | Recompute each card against updated stacks and copy beliefs | **Code** |
| Last copy discarded, blocking all higher ranks | Compute critical/dead status and color ceilings before discards; send timely saves | **Code + Astra** |
| Two players discard different copies of the same needed identity | Track imminent partner discards and common protection commitments | **Code + Astra** |
| Zero clues and no known play | Use best legal discard to restore information; avoid arriving here with threatened critical cards | **Astra using code values** |
| Eight clues and no safe play | Give a useful truthful clue, or a safe redundant clue if needed; discard is illegal | **Code + Astra** |
| Over-saving makes every hand protected and stalls draws | Release proven dead cards; define fallback discard by expected loss; balance saves against tempo | **Code + Astra** |
| Misplay based on “high confidence” at one life | Compare expected terminal score including zero on failure | **Code + Astra** |
| Clever finesse recipient cannot decode | Reject unless predeclared rules and turn order justify the same unique interpretation | **Code + Astra; optional Jev review** |
| Late clue reaches a seat after its final turn | Compute exact remaining actor sequence before assessing clue benefit | **Code** |
| Raw own-card unknowns confused with rank zero/one | Normalize external rank encodings and unknown sentinels at the adaptor boundary | **Code** |
| Search plays partners' hidden cards with omniscient knowledge | Restrict each simulated policy to its view; group indistinguishable decision nodes | **Code** |
| Strong training seeds but poor held-out results | Evaluate paired policies on disjoint random deals; report bomb-outs and variance | **Evaluation code** |

## 12. Recommended development order

1. Implement and verify the simulator/interface contract before optimizing play. Reproduce deterministic legal episodes and targeted ending/clue fixtures.
2. Add literal candidate masks, card accounting, dead/critical detection, legal-action validation and seat isolation. Start with guaranteed plays, useful literal clues, and conservative discards.
3. Establish a single common save/chop protocol and endgame countdown; compare behavior over many random seeds. This should eliminate obvious accidental discards and useless late clues.
4. Give Astra concise candidate summaries and recipient-perspective clue assessments. Tune resource/risk choices against terminal score, not clue entropy alone.
5. Test the explicit direct-play signaling extension and short belief rollouts. Add advanced conventions only when failure traces show the baseline's communication limits and all seats can decode the richer protocol.
6. Keep improvements only when paired held-out evaluation supports them; report mean score, spread, bomb-out rate and perfect rate, separated by player count. Evaluate bomb-out handling and information isolation as correctness requirements, not adjustable performance tricks.

No empirical performance target is asserted here. The challenge paper discusses strong established agents [S2], but matching their scores depends on player count, rule profile, convention implementation, inference budget and evaluation seeds.
