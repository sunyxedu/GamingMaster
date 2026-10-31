
### Candidate 0001 — semantic state and feedback navigation
- All eight baseline opening model prompts truncate inside `glyphs`; neither `stats` nor `message` reaches Astra. Serialize selected semantic fields explicitly before any bulky arrays. A small prompt can contain more usable information.
- The baseline takes 6–31 commands per 1,800-second episode. Turn-based play removes in-game latency penalties, but model-per-keypress control still exhausts the wall budget before exploration produces score.
- The 23-action set is materially smaller than full NetHack. Inventory letters are only expressible when they coincide with an available command key; general quaff/wield/pickup strategies cannot simply be copied from the study guide.
- Step traces are themselves truncated strings, so candidate 0001 logs compact named-state checkpoints and supervisor plans. This makes future failure analysis possible without relying on raw-array serialization.

### Candidate 0002 — autonomous execution with sparse advice
- Baseline final observations expose prompt stalls that scores alone hide: seed 1106 ends at `In what direction?`, game time 5 after 11 actions; seed 1104 ends with `Unknown command '^M'.` Prompt handling needs its own state machine, not unconditional Enter.
- Distinguish commands submitted from game turns advanced. Invalid movement into stone may cost a command but no game time; remember failed edges to keep those probes bounded.
- The archived `specials` array consistently marks initial pets with bit 8, while an apparent dog statue has bit 32. Character-only threat detection can attack pets or misclassify statues; preserve those flags and verify behavior from messages.
- A sparse adviser can consume compact outcome logs while code handles routes and verifies each action. Candidate 0002 tests this division rather than asking Astra to produce every movement.
- Pre-flight passed, but its short virtual budget allowed no adviser call. Test parser garbage and timeout behavior explicitly as well; a passing pre-flight alone does not establish coverage of model-response handling.

### Candidate 0003 — model-led macro transactions
- A useful contrast to sparse-adviser agents is to keep goal selection in Astra while compiling explicit bounded goals into feedback-checked actions. This separates the benefit of action throughput from the benefit of a hand-written strategic policy.
- The baseline's seed 1101 submits eight commands but finishes at its starting square, game time 3. Commands, elapsed turns, displacement and score are different progress measures; log them together.
- A plan should become invalid when coordinates change floors, an unapproved creature reaches melee range, or HP assumptions fail. Pet movement and ordinary newly revealed floor should not invalidate every action batch.
- Pre-flight can pass without exercising inference: candidate 0003's initial run made 1,062 actions and zero model calls. Separate mocked garbage-reply and TimeoutError checks exercised the missing paths; malformed JSON field types also need validation.
- Do not update bookkeeping twice when reconsidering a completed plan without taking an action. Otherwise visits/search coverage acquire fictitious evidence. Candidate 0003 now updates its belief state once per returned action observation.
- Dev 3305 exposed a false permanent wall from `Your kitten is in the way!` after the pet moved between observations. Movement failure must distinguish transient occupants from geometry. Dev 3304/3305 also spent about 599/959 seconds awaiting opening answers whose reported inference times were only 41.6/11.4 seconds; total call blocking can dominate reported model latency, motivating a short guarded setup transaction before the first call.

### Candidate 0004 — event-driven exploration and tactical advice
- Leader traces repeatedly spend a whole model transaction on `wait` or on a fight that has already ended; ordinary exploration, gold collection, and staircase descent should be deterministic so a slow adviser cannot stall game-time progress.
- `run <dir>` is useful only when several already mapped squares are clear; requiring a verified corridor gives action throughput without crossing unknown monsters or doors.
- A tactical answer should authorize observed glyphs only. Keeping `mode == fight` as a permanent permission would attack newly spawned or peaceful creatures, so the controller resets authorization to the visible set.
- Prompt handling and movement feedback remain state-machine work: pet-block messages are transient, while unchanged position plus unchanged turn is geometry evidence for a blocked edge.
- Delay the first Astra call and require substantial progress between calls because archived traces show a request can wait hundreds of wall-clock seconds before its reported inference begins.

### Candidate 0005 — outcome-driven risk bounds
- NetHack's displayed score can disappear at terminal death: candidate 0002 seed 1106 showed 125 points in the death screen but episode_end reported score 0. Preserve health before chasing another room or monster.
- A failed movement command is geometric evidence only when the returned message and time support it. “Your kitten is in the way” and “Really attack...” are transient occupancy or prompt states; “It's a wall” at unchanged time is a blocked edge.
- Door state must be repaired from outcome text even when a creature occupies the doorway. “The door crashes open” means the remembered `+` is now traversable; repeatedly kicking it can create attack prompts and waste turns.
- Route planning should treat creature proximity as a cost and apply a health reserve before every move. A model classification can authorize a species, but it cannot bypass the local reserve, pet, prompt, or terrain gates.
- A pre-flight with no model calls validates the loop but does not exercise reply parsing or timeout handling; keep separate malformed JSON and caught-TimeoutError checks.

### Candidate 0006 — cycle-aware novelty routing
- Position changes are not progress: candidate 0003 seed 1108 repeats a three-cell `se,se,w` route for roughly 150 turns while score stays 12, so a stale counter based only on unchanged coordinates misses the failure.
- A useful watchdog compares short windows of observed positions and temporarily bans both directions of a repeated edge; the ban must expire because an apparent loop edge can later be the only route to stairs.
- Progress signatures should use unique cells or newly revealed terrain, not the length of a visit counter that increments on every action.
- Small visit penalties in path cost preserve necessary backtracking while making a least-visited reachable tile a deterministic escape when frontier and door candidates are exhausted.
- Cycle detection is observation-only and therefore remains safe under garbage Astra replies; model calls should be reserved for tactical classification after deterministic novelty routing fails.

### Candidate 0007 — latency-free coverage fallback
- A System-2 request can consume roughly 1,173 virtual seconds before returning even when its reported inference latency is under two minutes; tier-1 harnesses must treat model calls as a scarce wall-clock resource.
- A live, healthy run should not return merely because every currently mapped tile has been visited twice; hidden doors and stairs require continued wall search and coverage walks.
- Persistent directed-edge revisit costs complement short-cycle bans: position changes can still form long shuttles after a temporary ban expires.
- A verified early-monster character whitelist can replace a tactical model call when the single-adjacent-creature and HP-reserve gates remain in force; unknown or multi-creature encounters still retreat.
- Candidate 0006 still stopped healthy at depth 1 when its visible frontier was exhausted; map completion is not evidence that stairs or secret doors are unreachable.
- A bounded continuation phase can probe distant low-visit cells and search safely before returning, preserving the parent’s HP and prompt gates.
- A `run` is a useful throughput primitive only when every square in a straight path is already mapped and creature-free; compile such paths after the tactical gate, never for an unknown corridor.
- Treating a model target as advisory requires adding its mode to the target validity check; otherwise the harness silently ignores newly introduced fallback targets.
- Keep fallback duration finite: persistent wandering can consume the turn budget while score and depth remain unchanged (as seen in seed 1108).

### Candidate 0008 — asynchronous advisory latency
- A turn-based harness can keep stepping while Astra thinks; blocking `ctx.think` consumed most of the wall deadline on seeds 1108 and 1101 despite no game-time cost.
- `ctx.spawn` futures need bounded ownership: one pending request, polling from the main loop, and no new work near deadline prevent orphaned background calls.
- Apply late advice only after reparsing against the current visible glyph set; deterministic safety gates must remain valid while the map changes.
- Preserve score before model calls: an `EpisodeOver` from a background request should be treated as stale advice, not an episode error.

### Candidate 0010 — health-aware hunger continuation
- Candidate 0009's hunger guard stopped at state 3 (“Weak”) even when HP was full; the trace's `weak hunger without accessible safe food` reason therefore conflated a warning with imminent death.
- Treat hunger state 3 (“Weak”) as a soft continuation condition when HP exceeds the measured damage reserve; reserve a hard stop for state 4 or a fragile-health state.
- Keep the action-spec limitation in mind: inventory letters are not generally legal actions, so “no accessible food” can be true even with food text in inventory.
- The parent’s map/cycle/prompt machinery is valuable; this proposal changes one safety gate so the result isolates hunger policy rather than introducing route or model confounds.

### Candidate 0012 — bounded unknown-melee probe
- A healthy `no safe continuation` stop can mean a single unfamiliar creature seals the only mapped route; HP and weapon checks can authorize one feedback-checked attack without opening the general combat gate.
- Attack permissions must be observation-scoped: clear them after every `ctx.act` result because a monster can move or a prompt can replace the intended melee transaction.
- The existing action alphabet cannot express arbitrary inventory letters reliably; hunger handling should distinguish “food exists” from “food can be selected by an available action” before relying on an eat prompt.

### Candidate 0013 — declined attack prompt memory
- A yes/no answer can be syntactically valid yet strategically repeat forever: after `Really attack ...? [yn] (n)`, the next observation may still present the same creature and retrigger the probe.
- Prompt answers need a belief update just like movement outcomes; temporarily marking the declined glyph as avoided lets routing choose another edge.
- Keep prompt avoidance observation-scoped and separate from model-provided danger labels, so a creature that moves away can be reconsidered later.
- Preflight stand-in models may never exercise inference or prompt branches; inspect archived traces for exact terminal messages when designing these state transitions.

### Candidate 0011 — pet-swap deadlock breaker
- A position-changing loop can still be a failure: seed 1105 repeatedly logged `You swap places with your little dog.` while score stayed flat and hunger worsened.
- Treat transient pet occupancy as a navigation event, not as ordinary map progress; clear the stale target and let the pet take a turn.
- Temporary edge bans should cover both directions of the swap and expire, because a pet-blocked square may later be the only route.
- Keep the intervention observation-driven and bounded; a permanent pet avoidance rule could strand the player or waste turns near monsters.

### Candidate 0015 — screen-level prompt transactions
- A healthy run can end with `unresolved prompt` after hundreds of actions; inspect prompt and tty state, not only HP and score.
- NetHack prompt text may be present in tty output while the latest-message field is empty or stale; prompt detection should use a compact screen projection.
- Returning `wait` to a directional prompt cannot make progress. Consume the remembered direction, then fall back to a recent cardinal move if observation truncation lost it.
- Do not clear a pending prompt transaction before the prompt action is selected; bookkeeping order can turn a valid kick response into an apparent unresolved prompt.

### Candidate 0014 — semantic quarantine for declined attacks
- A movement into a peaceful creature is a transaction: `Really attack` can leave both position and game time unchanged, so generic stale-edge logic alone repeatedly retries it.
- Quarantine the exact level/coordinate after that prompt; quarantining a glyph globally would incorrectly suppress other creatures of the same species.
- Prompt refusal handling must happen on the world action observation, before the follow-up `n` answer overwrites the message.
- Temporary edge bans and a short wait let pets or peaceful monsters move while preserving a later reconsideration path.

### Candidate 0016 — screen prompt plus coordinate quarantine
- A prompt detector and a tactical belief update are separate requirements: tty-only visibility can hide `Really attack`, while detection without quarantine retries the same probe.
- Coordinate-scoped refusal memory must be retained when adding screen projections; a broad glyph blacklist would suppress unrelated monsters.
- Directional prompts should consume the pending kick direction, with a recent cardinal fallback when observation text is truncated; `wait` leaves NetHack waiting for input.
- Prompt transactions should be recorded before posting the answer so the next observation cannot re-enter the same unknown-melee branch.

### Candidate 0017 — low-score plateau safety gate
- A terminal death can erase an otherwise real score: seed 1104 reached score 8, then died on a late first-floor move and reported terminal score 0.
- Position novelty and healthy HP are insufficient evidence that continued wandering is worthwhile; score-only progress is a useful independent signal.
- A plateau cutoff must be narrow (low score, first floor, no visible threat, healthy reserve) so it does not suppress deeper exploration or runs still earning gold/XP.
- Returning from a turn-based episode preserves the current score and avoids spending hundreds of actions on a flat route.

### Candidate 0018 — bounded stagnation recovery
- A stable map can still be a false completion signal: seed 1105 held `known: 278` and score 21 for hundreds of turns while hunger advanced to 4.
- Search and frontier counters are evidence, not permanent truth; a small, health-gated reset can retest hidden doors without discarding terrain or blocked-edge evidence.
- Progress watchdogs should use score, revealed terrain, and unique cells together; position changes alone can hide a no-progress shuttle.
- Recovery loops need a strict per-episode cap and threat/HP gates, since repeated search consumes hunger and can turn a safe run into starvation.
