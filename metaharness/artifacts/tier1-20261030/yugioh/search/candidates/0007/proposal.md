parent: 0004

## Hypothesis and trace evidence

The leader's battle search and reaction policy disagree about what is achievable.
The search considers only current ATK, while Rush Recklessly is activated only
once a battle is already in progress. Consequently the agent never initiates
some attacks that its available combat trick would win.

Concrete evidence from the leader's worst training episode, seed 1111 (score
-0.373333): step 13 chooses `choose r1p13 2`, “enter Main Phase 2”, while the
actual menu offers Rush Recklessly from hand and an attack by 1900-ATK Luster
Dragon into a 2000-ATK Gene-Warped Warwolf. The +700 card would make Luster 2600
ATK before unknown interaction. Step 166 repeats the same choice with Rush
legally activatable from the Spell/Trap zone (`choose r3p39 0`) and the same
1900-versus-2000 comparison. Instead Rush is spent defensively on the following
opponent turns (steps 24 and 176). These are observed missed opportunities,
not claims that the counterfactual games necessarily win. In duel 1, waiting
allows another enemy Luster Dragon to join the board; both own monsters are
gone by the next own turn.

An initial draft investigated premature model end-turn intentions, based on the
parent's dev write-up. Direct inspection of that dev trace refuted the stronger
claim: the model omitted Battle Phase, but an intervening response invalidated
the plan and the fallback did attack. That draft was replaced before evaluation.
The submitted candidate tests only the evidenced Rush/combat disconnect.

## Change under test

Preserve 0004's transport, position-aware decisions, exact subset validation,
visible combat assignment search, bounded Astra calls and failure detection.
At an own battle-command prompt, expand the search with one currently legal
Rush activation and each currently legal attacker as its prospective target.
Compare the existing combat search at observed ATK with the same search at
ATK +700. Spend the card only when the improvement exceeds its explicit resource
cost (900 evaluation units); ordinary nonlethal +700 LP damage does not qualify.
Do not spend it solely to probe an unknown defender. If unboosted combat already
has visible lethal, preserve the card.

Execute only the current legal spell action. Bind its target by current public
zone reference, and verify the native observation actually reports the +700
before allowing the ordinary attack search to proceed. The engine may interrupt,
bounce, destroy or otherwise change the intended line; no hypothetical attack
is queued. A failed observed effect invalidates the plan, records concrete
expected/actual evidence and requests bounded Astra repair. Every action retains
the parent's token/progress/illegal-action checks. Astra receives the possible
Rush expansion and outstanding consequence check in its compact observation.

The implementation uses only public observations, current menus, card text and
standard-library code. No game implementation, hidden identity, scenario mapping
or seed is read. Existing 235-second model timeout and 210-second wait guard
exceed the archive's 120-second worst reported inference latency while preserving
room under the native 300-second decision deadline.

## Confirmation and refutation

Confirm mechanistically if the two archived battle exits instead select legal
Rush, select Luster as the target, observe 2600 ATK, and re-evaluate combat.
Confirm strategically only if native wins and weakest-seed results improve
without errors, forfeits or truncation. Refute if early Rush exposes the card to
avoidable removal, increases losses to Mirror Force, or wins no more games.
The main tradeoff is activating before battle rather than holding Rush until a
later damage window; this intentionally buys verified execution over speculative
attack commitments. Hidden responses remain unknown. Replay establishes legality
and the changed decision, not a counterfactual win.
