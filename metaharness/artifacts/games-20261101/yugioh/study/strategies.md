# Strategy prior for Astra, optional Jev, and exact-rule code

## Scope, evidence, and division of work

This is a strategy prior for ordinary Yu-Gi-Oh! duels and the **three fixed beginner/intermediate decks in `interfaces.md`**. It is not a claim about the best tournament decks in October 2026. The initial pool deliberately omits Extra Deck combos, hand traps, and archetype search engines. Modern-game advice below is marked as an expansion path, not assumed to apply to the baseline.

Web retrieval failed in this session. The bibliography contains official and community reference URLs that must be retrieved and archived when access is available; no current article, tournament result, deck statistic, or source quotation is claimed to have been read. Rule-sensitive statements should be checked against the pinned card text and engine. Tactical recommendations are a synthesis of established game concepts and the proposed pool, not measured performance guarantees.

**Code owns exactness:** legal moves, state/knowledge bookkeeping, all arithmetic, constraints, costs/materials, effect-use tracking where observable, legal response encoding, seed handling, timeout policy, and score. The engine and matching scripts are the final rules authority. Search code, if added, must reason only from the agent's information set.

**Astra owns plans:** choose a win condition, identify the useful resource exchange, sequence plays around interaction, infer plausible opposing threats, choose which interruption matters, and preserve follow-up. It should explain a short plan to itself and return an action from the actual prompt rather than inventing a complete turn.

**Jev is optional:** this is Tier 1, with no need for a reflex controller. Use it only for independent typed judgments that reduce context cost, such as classifying a card's role, recognizing an exposed threat, or rating the relevance of a candidate interruption. Do not ask Jev to calculate lethal, solve material subsets, infer exact hidden cards, or plan a combo. All of its outputs are advisory and restricted to known cards and supplied legal candidates.

## Priority order and evaluation

1. **Code: obey the current request.** A mandatory cost/target/trigger request interrupts every broader plan. A legal optional decline can be best, but an illegal “pass” is never acceptable. Preserve effect IDs, source instance, selected targets, and prompt token. [S1, S2, S4]
2. **Code + Astra: take a genuinely established win.** Code evaluates legal damage or alternate-win opportunities from public/owned information, with explicit assumptions about unknown interaction. Astra chooses whether extra protection or an initial bait is worth delaying the attack. “Known board lethal” is not guaranteed lethal through an unrevealed Trap. [S1, S3]
3. **Astra: prevent a likely immediate loss.** Identify the opponent's actual route to lethal or a decisive lock, then use the least costly sufficient answer. Do not spend every interruption on the first effect simply because it is activatable. Code supplies projected damage and restrictions. [S2, S5, S6]
4. **Astra: improve usable resources and position.** Count what remains useful after the exchange: cards in hand/field, live Graveyard effects or revival targets, an unused Normal Summon, available zones, threats, defenses, and next-turn plays. Raw card count is informative but not decisive. [S5]
5. **Astra: preserve follow-up once the board is adequate.** An extra summon or effect activation that does not improve the expected outcome may expose another card to mass removal or consume next turn's starter. LP are a resource, not a reward to hoard at the expense of winning. [S5, S6]

A useful evaluation order is: terminal outcome → imminent survival/lethal → live threats and interaction → resource/follow-up quality → LP margin. Do not turn the benchmark's small LP tie-breaker into an instruction to heal, avoid all costs, or stall a winning game.

## Opening plans for the fixed pool

The first player starts with five cards, skips their first draw, and has no first-turn Battle Phase. Going first usually favors setting up useful interaction; going second offers an additional normal draw and a chance to attack. Which is better depends on the deck matchup and actual hand. The benchmark randomizes starting position, so Astra must handle both. **Code** supplies turn and first-turn restrictions; **Astra** chooses the plan. [S1]

| Deck | Opening and follow-up plan — Astra | Exact support — code | Optional Jev judgment |
|---|---|---|---|
| `normal_pressure` | Establish a strong low-level attacker; back it with one or more useful defensive interactions according to the hand and threat of mass removal. Going second, use removal to turn bodies into damage. Keep another summon in reserve if committing it adds little. | Compare actual ATK/DEF, calculate battle results and lethal assumptions, track Normal Summon and attack availability. | Which known opposing card most threatens the current attacker? Is this legal move pressure, protection, or unnecessary commitment? |
| `flip_control` | Against a stronger attacker, a useful flip Set can generate removal or recovery instead of losing a weak face-up monster. Protect it if doing so preserves a valuable next turn. Recover the Spell that supports the present plan, not automatically the flashiest Spell. Transition to attackers after trading resources. | Track public targets, own Graveyard Spell availability, whether a Set/position change is legal, effects already used, and cards pending destruction. | Does this known flip effect offer removal, bounce, recovery, or no immediate value? Which already-legal target is strategically valuable? |
| `tribute_pressure` | High-level monsters are not openers without bodies or an alternate legal summon. Keep low-level monsters alive for Summoned Skull or Blue-Eyes, and exploit legal revival when a large monster reaches a Graveyard. A Tribute Summon is worthwhile when its resulting threat is worth the bodies spent. | Count actual required Tributes and legal tribute candidates; compare before/after battle damage and resources. Do not assume every Level 7+ card has identical special conditions. | Which candidate tribute body is least useful next turn? Is a big monster a decisive threat or just a small ATK improvement? |

Concrete baseline examples, subject to the engine's actual offered choices:

- **Astra:** do not Tribute two productive attackers for Blue-Eyes merely because it has the highest printed ATK. **Code:** compare the existing legal attacks with the 3,000-ATK body, and account for lost cards, blockers, and remaining battle opportunities. If the existing monsters can attack profitably, attacking before a Main Phase 2 Tribute can preserve damage; if they would die or expose a fatal response, that line may be worse. [S1, S5, C1]
- **Astra:** protect a low-level body when it enables a strong next-turn Tribute, but do not spend multiple cards to preserve a body whose payoff is easily answered. **Jev:** rate the body's strategic role from a supplied short list; no resource arithmetic. [S5]
- **Astra:** a face-down monster in `flip_control` is plausibly a flip effect, not certainly Man-Eater Bug. Attack sequencing and non-battle removal can reduce the risk, but hidden identities remain unknown. **Code:** retain a distribution/remaining-card count derived only from public observations if a belief model is used. [S1, S3]
- **Astra:** setting every Quick-Play Spell immediately is a mistake. A Quick-Play Spell in hand may be usable on your own turn; after it is Set, it normally cannot be activated that turn. Set it for future opponent-turn use only when giving up present use is acceptable. **Code:** expose only legal activation windows. [S1, S2]

## Tactical interactions and counters in the initial pool

Each example names who should own the work. Card text can be errata-dependent; validate against the pinned CDB/scripts and official database before using it as a hard rule. [C1]

| Interaction / common mistake | Strong response and counterplay | Assignment |
|---|---|---|
| **Mystical Space Typhoon destroys; it does not negate.** | Destroying an already activated ordinary Spell/Trap does not generally stop its effect from resolving. Chaining MST to Mirror Force normally does not save the attackers. Continuous/equip/field-card dependencies can behave differently: inspect the card, not a universal slogan. | **Code:** engine resolution, no homemade destruction=negation shortcut. **Astra:** choose meaningful back-row removal timing. **Jev:** identify “destroy” versus “negate” from known text. [S4, C1] |
| **Mirror Force punishes an attack commitment.** | It responds to an attack declaration and destroys the opponent's Attack Position monsters; it does not target each one. Keep a reserve, remove unknown back row when worthwhile, or use a legal position-changing response to preserve an important body. Do not assume “cannot be targeted” defeats non-targeting removal. | **Code:** legal timing and positions. **Astra:** commit only the attackers needed and weigh unknown Trap risk. **Jev:** recognize exposed concentration of attackers. [S1, S4, C1] |
| **Book of Moon can interrupt a face-up monster's role.** | Use a legal flip-down to stop an attack, protect a body from some effects, or interfere with an exposed monster's ability to serve a future role. It does not automatically negate an effect already activated; removing/flipping a source often leaves its Chain Link intact. It cannot flip a Link Monster face-down if modern decks are added. | **Code:** legality and resulting battle/target state. **Astra:** choose the valuable timing and target. **Jev:** classify the target's immediate role. [S1, S2, S4, C1] |
| **Trap Hole is not a universal summon answer.** | Its ordinary text covers an opponent's Normal/Flip Summon with the required ATK; it does not cover arbitrary Special Summons. Preserve a different answer for Monster Reborn. Summon-response timing is governed by the engine, not a free reaction to any change of state. | **Code:** exact offered windows and candidates. **Astra:** ration interaction according to threats. **Jev:** identify summon categories from supplied events. [S1, C1] |
| **Fissure does not let you pick any target.** | It destroys the opponent's face-up monster with the lowest ATK under its text. Adjust the board first if a particular removal result matters; do not submit an illegal desired target. Current stats and resolution state matter. | **Code:** determine eligible outcomes from engine constraints and current state. **Astra:** sequence removal/battle to make the effect useful. [S4, C1] |
| **Compulsory Evacuation Device trades tempo for a card.** | Returning a Tribute monster removes a larger investment than returning an easily resummoned low-level monster. Bouncing a threat before it produces value can be strong; bouncing its source after effect activation may not stop that effect. A future Extra Deck monster normally returns to its appropriate Extra Deck destination, as enforced by the core. | **Code:** target eligibility and actual destination. **Astra:** value tempo versus permanent resource advantage. **Jev:** rank supplied visible threats. [S5, C1] |
| **Dark Hole is symmetrical.** | Use it to recover from a worse board or before committing your own new threats. Count friendly losses and the opponent's replacements. If a valuable friendly monster survives by some effect, establish that from actual text/core state. | **Code:** tally the known affected field and damage after clearing. **Astra:** choose whether recovery is worth the exchange. [S5, C1] |
| **Monster Reborn can use either Graveyard.** | Select the body that advances the current win/survival plan, including an opposing large monster, if legal. Do not automatically revive the highest printed ATK when a different effect or board position wins. Modern special-summon restrictions/proper-summon history can make attractive targets illegal. | **Code:** legal target list and battle arithmetic. **Astra:** choose role and sequencing. **Jev:** role classification only. [S1, C1] |
| **Call of the Haunted is continuous and tied to its target.** | Treat its ongoing field dependency differently from a one-shot revival Spell. Removal of the Trap can affect the revived monster, and disrupting the Trap before its revival resolves can matter. Let the exact script handle detailed exceptions. | **Code:** track the visible relationship and actual resolution. **Astra:** plan around vulnerable revived bodies and enemy removal. [S4, C1] |
| **Man-Eater Bug is a mandatory flip effect.** | Do not casually flip it on an empty opposing field: its required target may be your own monster, including itself when eligible. Removing a flipped Bug does not by itself erase an already activated effect. | **Code:** legal mandatory targets. **Astra:** choose when to expose it. **Jev:** flag potential friendly-target cost from the shown candidates. [S4, C1] |
| **Penguin Soldier offers bounce value.** | Use its available targets to remove invested threats or recover a useful friendly body when legal. Do not assume a monster already determined destroyed by battle can be saved by an arbitrary self-bounce; follow the actual Damage Step candidates. | **Code:** target list/limits and battle state. **Astra:** select a meaningful exchange. [S6, C1] |
| **Magician of Faith recovers a Spell.** | Plan around useful Spells actually in your Graveyard; recovering removal, revival, or protection depends on the board and next turn. Setting it with no recovery target may still defend, but should not be counted as guaranteed card advantage. | **Code:** legal Graveyard candidates and known spell availability. **Astra:** select the plan-supporting recovery. **Jev:** classify candidate roles. [S5, C1] |
| **Rush Recklessly is a combat modifier, not a universal defensive answer.** | Code calculates whether its actual boost changes destruction or lethal. Some ATK-changing effects can be used in particular Damage Step windows; the engine must decide whether the current prompt allows it. Spending it for extra LP damage may be worse than keeping a body alive later. | **Code:** all arithmetic and timing. **Astra:** resource value. **Jev:** no arithmetic request. [S1, S6, C1] |

## Resource use, sequencing, and interaction

### Card advantage and tempo

**Astra:** favor exchanges that leave more useful resources than the opponent: remove a monster they spent multiple cards to establish, get repeat value from an effect, or force an answer with a card you were willing to lose. But one-card removal on the last attacker preventing lethal is correct even if the nominal exchange is poor. A hand full of unusable high-level monsters is not equivalent to a hand of live plays. A temporary bounce can win through tempo despite not permanently removing a card. [S5]

**Code:** provide separate features for visible card counts, live legal actions, graveyard resources that are publicly known, available normal summon/attacks, known restrictions, and computed damage. Do not flatten these into a purported universal “card advantage” formula. Cards moved to the Graveyard can remain resources; Tributes, costs, and materials are not always pure waste.

**Jev:** score a supplied known card's usefulness under a supplied plan (`low`, `medium`, `high`, `unknown`). It should not invent unseen extenders or calculate net material/LP totals.

### Play order and commitment

**Astra:** start with the play that gives useful information or demands a response without losing access to the stronger line. Preserve flexible removal until the opponent commits a target if waiting is safe. Avoid using a Normal Summon on a redundant body before checking whether another hand card needs it. Protect the play that makes the rest of the hand functional; accept losing a less valuable bait when that opens a winning route. [S2, S5]

**Code:** track consumed once-per-turn effects, normal-summon availability, summon/type/attribute locks, battle restrictions, and selected targets using public events/core state. Where the engine does not expose a restriction directly, annotate it as a text-derived/inferred constraint and still use engine legality. Restrictions that apply after activating an effect can invalidate a planned line even when each earlier step was legal.

**Jev:** independently label candidate actions `starter`, `extension`, `interaction`, `protection`, `follow_up`, or `unclear` from supplied definitions and card text. Astra uses these judgments; Jev does not arrange their order.

### Chains and conserving interruption

**Astra:** ask what happens if this effect resolves, whether the opponent can replace it, and whether a later commitment is a better interruption point. Sometimes stopping the initial search is best; sometimes the dangerous effect is the summoned monster or the opponent's last useful resource. Do not “always negate the first search” or always wait for the largest monster. In the initial pool this becomes choosing which summon/attack/removal/revival deserves your Trap or Quick-Play Spell. [S2, S4, S5]

**Code:** preserve Chain Link order, source/effect identity, announced costs and targets, available responses, and activation-versus-effect-negation distinctions. The core controls resolution and simultaneous trigger ordering. Quick Effects cannot be inserted halfway through a resolving Chain. Costs already paid do not normally return simply because an effect is negated; use actual card/core rules for exceptions. [S2, S4]

**Jev:** score each legal response for relevance to a named threat using known text and board features. Include an `unknown` response rather than turning uncertainty into false confidence. Recognizing a threatening effect is useful; proving the best chain line is Astra/code work.

### Battle and incomplete information

**Code:** calculate current battle outcomes, defense/piercing details, available attackers, attack restrictions, legal direct attacks, and reachable LP totals under explicitly stated interaction assumptions. Enumerate short attack orders when useful. Use current modified stats and engine flags rather than printed stats alone. Do not claim ordinary arithmetic resolves all damage-modification effects. [S1, S6]

**Astra:** decide whether to enter battle, which attacker to risk against a Set monster, whether to remove a blocker before battle, and whether a damage boost is better saved. Holding an attacker back or leaving it in Defense Position can preserve resources against a plausible Mirror Force. Attack for lethal when the alternative gives the opponent another draw to recover; do not play indefinitely around every possible card. [S5, S6]

**Jev:** recognize possible mass-removal exposure, a revealed battle-protection effect, or a known “must remove before attacking” threat. Supply yes/no/unknown labels, not invented hidden-card identities.

### Information discipline

**Code:** maintain revealed-card memory, public used-card counts, known effects and expiry, and distinct “known,” “inferred,” and “unknown” flags. Opponent hand size and known public deck list are available; hand identities and deck order are not. After a shuffle or hidden randomization, invalidate location-specific tracking that the player could no longer follow. [S1, S3]

**Astra:** make a short list of plausible opposing answers from the fixed pool and observed play, and select a line robust to likely answers. Do not equate a bot declining a response with proof that its hand lacks one unless the exact opponent policy justifies that inference. Keep inference separate from fact.

**Code/search:** full true-state simulator rollouts would be cheating if their evaluations exploit the opponent's actual hidden hand or deck order. An optional information-set search can sample **hypothetical** hidden worlds consistent with the agent's information and the public deck manifest. Do not replay the evaluator's true secret seed to discover the hidden deal. The core may not support state cloning; any replay/snapshot scheme needs separate engineering and validation. Search is an optional upgrade, not a prerequisite for the baseline.

## Modern-deck expansion: principles and counters

These are general ideas for a future expanded pool. The named cards below are **not in the initial three decks**, may be forbidden/limited in a chosen format, and are not necessarily present in a given opponent list. Do not build a live-format metagame prior without dated sources and frozen deck lists. [S7, S8, C1]

| Pattern | Strong-player plan / counter | Owner |
|---|---|---|
| Combo deck going first | Convert starters into useful interaction plus follow-up. Choose an end board that covers different threats instead of counting negates alone. Sequence around the actual disruption present in the fixed format. | **Astra:** plan/end-board choice. **Code:** materials, legal paths, locks, hard/soft once-per-turn bookkeeping. **Jev:** recognize starter/extender/interaction roles. |
| Hand traps such as Ash Blossom, Effect Veiler, Infinite Impermanence | Identify the engine bottleneck, likely extenders, and which answer's text matches the effect. Ash does not negate every effect; targeting field negation has different conditions and failure modes. | **Astra:** interruption point and baiting. **Code:** exact legality, windows, costs/targets, residual effects. **Jev:** classify the known effect being used. |
| Nibiru exposure | Count the relevant monsters Normal/Special Summoned this turn from public events, not merely summon events. Plan whether a useful protective interaction or adequate stopping point is available before exposure becomes relevant. It is not a universal instruction to stop after four monsters. | **Code:** count and legality. **Astra:** risk versus payoff. **Jev:** recognize “exposed” only from a precomputed count/flag. |
| Board breakers and Tribute-based removal | Several negates may still lose to non-activation removal, non-targeting removal, or a breaker with response restrictions. Preserve follow-up or diversify threats; do not describe every boss as protected just because it has targeting immunity. | **Code:** target/destruction/tribute restrictions. **Astra:** resource allocation and breaker sequence. **Jev:** map threat protection categories from known text. |
| Graveyard-centric decks | Disrupt the actual revival/recycling target or payoff at a legal point, while anticipating redundant resources. Some cards benefit from being sent to the Graveyard; removal type matters. | **Astra:** choose choke point. **Code:** destinations, target eligibility, delayed triggers. **Jev:** identify Graveyard reliance from revealed effects. |
| Simultaneous trigger ordering / “chain blocking” | When legal simultaneous triggers can be ordered, an order can change which directly responding effect is available. This does not make a whole chain unanswerable. Order only the effects the engine actually offers; region/ruleset differences matter. | **Code:** offered permutations and rule compliance. **Astra:** choose protective order. **Jev:** not responsible for solving chain order. [S4, S9] |
| “When … you can” timing and once-per-turn text | Optional timing and activation/use restrictions differ between cards. Leaving and returning to the field does not reset every named once-per-turn restriction. Do not memorize one rule for all cards. | **Code:** actual engine enforcement, observable usage tracking. **Astra:** avoid plans depending on an illegal second use. **Jev:** text categorization with uncertainty. [S10, S11] |

Opening scripts should be stored as **conditional plans with checkpoints**, not a list of action indices. Check the hand, selected variant, remaining deck targets, allowed Extra Deck, restrictions, and observed opponent response at every checkpoint. Interrupted combos must replan. A canned line is unsafe if its search target is already in hand or its required material is no longer available.

## Proposed Astra prompt and memory contract

This is a prompt design specification, not game-playing implementation.

Give Astra:

1. Current acting-player observation, prompt token, legal choices with effect text, and exact constraints.
2. Public events since its last decision; current chain and announced targets/costs.
3. Code-calculated battle/LP summaries with assumptions, observed used effects/restrictions, and a short persistent plan.
4. The fixed-format deck manifest or a compact role map; only publicly justified information about the opponent's live cards.

Instruction: “Choose only a supplied legal action/selection for this prompt. Prefer winning and avoiding immediate loss; then improve usable resources and preserve follow-up. Respect unknown information. Explain the immediate purpose and the condition that would make you change the plan. Do not recalculate arithmetic already supplied by code or assume an offered activation must be used.”

Persist: plan objective, intended next checkpoint, reserved interaction and what it answers, observed opponent strategy, uncertainty, and why the plan may need revision. Avoid a long unbounded transcript; preserve important public reveals and restrictions in code-maintained state. Invalidate planned action IDs after every engine response; retain semantic intentions instead.

### Optional Jev typed questions

Each question must be answerable independently from its included JSON context; provide candidates and an explicit unknown/none option. Batch them if helpful, but do not let one answer depend on another within the same call.

| Typed task | Example question | Allowed answer | What code/Astra does with it |
|---|---|---|---|
| Choice | “Which supplied role best describes this known effect in the stated plan?” | One of `removal`, `protection`, `recovery`, `pressure`, `setup`, `unclear` | Cache the annotation with card/effect revision; Astra interprets it. |
| Score | “How relevant is legal response A to the named visible threat?” | Integer 0–3 plus an independent unknown flag, if the typed protocol permits | Rank for Astra review; never bypass legality. |
| Yes/no/unknown | “Does this shown effect text explicitly negate an activation or effect?” | `yes`, `no`, `unknown` | Help distinguish destruction from negation; do not replace core resolution. |
| Choice | “Which supplied known target is the strongest visible source of recurring advantage?” | One supplied target ID or `unknown` | Suggest target prioritization, subject to plan and exact constraints. |
| Yes/no/unknown | “Given the provided precomputed exposure flag and board, is committing another body strategically redundant?” | `yes`, `no`, `unknown` | Astra decides whether to stop extending. |

Do **not** ask Jev “find lethal,” “choose a whole turn,” “sum these materials,” “track how many summons occurred,” or “what is in the opponent's hand.” Those require arithmetic, stateful bookkeeping, planning, or unavailable information.

## Baselines and strategy evaluation

**Code baseline:** a seeded random-legal policy establishes engine completeness, not competitive strength. A sensible scripted policy can prioritize forced responses; verified public-board lethal; preventing projected lethal; good-value removal; strong low-level summons; useful defensive Sets; profitable attacks; then phase progression. It must avoid infinite/redundant reversible action loops, must distinguish optional activation from a forced choice, and must not use hidden opponent information. For the tribute deck, compare investment with the resulting threat rather than always summoning the largest card.

**Astra evaluation:** compare win rate, completed-duel rate, illegal/stale response rate, truncation rate, and per-deck/seat results against the baseline over the same held-out seeds. Inspect failures such as MST-as-negation, spending a removal on a low-value target, missing known-board lethal, wasting a Normal Summon, losing flip value, or overextending into Mirror Force. Improvements should survive multiple opponent seeds and both first/second positions. LP margin alone is a poor measure of strategy quality. The correct deck plan may differ when the opponent changes from random to scripted.

## Reference URLs and how to use them

**None were fetched in this session.** These are stable reference targets from prior knowledge; availability and current content remain to be checked. Use official rules/card text and the pinned engine for rulings, community pages for explanations, and the explicit benchmark proposals above for engineering priorities.

- **S1:** Official rulebook — https://www.yugioh-card.com/en/rulebook/ . Basic rules, summon/battle/turn structure, Spell/Trap types.
- **S2:** Official Fast Effect Timing — https://www.yugioh-card.com/en/play/fast-effect-timing/ . Turn-player priority and response windows.
- **S3:** Community public/private knowledge overview — https://yugipedia.com/wiki/Public_knowledge . Verify detailed visibility against reference-client behavior.
- **S4:** Community Chain and PSCT references — https://yugipedia.com/wiki/Chain and https://yugipedia.com/wiki/Problem-Solving_Card_Text . Resolution order, costs/targets, conjunctions, and activation versus effect distinctions.
- **S5:** Community card-advantage concept — https://yugipedia.com/wiki/Card_advantage . A vocabulary for exchanges, not a complete evaluation function.
- **S6:** Community Damage Step reference — https://yugipedia.com/wiki/Damage_Step . Detailed battle timing and restricted activations.
- **S7:** Community hand-trap concept — https://yugipedia.com/wiki/Hand_trap . General interaction vocabulary, not a current tier list.
- **S8:** Official Forbidden/Limited lists — https://www.yugioh-card.com/en/limited/ . Freeze a dated list if adopting a tournament format.
- **S9:** Simultaneous effects — https://yugipedia.com/wiki/Simultaneous_Effects_Go_On_Chain . Check regional/version-specific details rather than hard-coding a remembered ordering rule.
- **S10:** Optional timing — https://yugipedia.com/wiki/Missing_the_timing . Confirm each actual card and trigger window.
- **S11:** Once-per-turn wording — https://yugipedia.com/wiki/Once_per_turn . Use the pinned actual text and core behavior.
- **C1:** Official card database — https://www.db.yugioh-card.com/yugiohdb/ . Search the exact card names in the tables, including MST, Mirror Force, Book of Moon, Trap Hole, Fissure, Compulsory Evacuation Device, Dark Hole, Monster Reborn, Call of the Haunted, Man-Eater Bug, Penguin Soldier, Magician of Faith, Rush Recklessly, and any expansion cards.
- **C2:** Script implementations — https://github.com/ProjectIgnis/CardScripts . Pin alongside the core; use implementation and regression fixtures to validate the intended rules rather than relying on prose alone.
- **C3:** Scripted-opponent source — https://github.com/ProjectIgnis/windbot . Once available, inspect a matching executor for concrete policy sequencing and supported decks; no executor was inspected here.
