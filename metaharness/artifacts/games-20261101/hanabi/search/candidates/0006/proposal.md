parent: 0001

## Hypothesis
The remaining score gap is partly a coordination failure between private discard selection and public saves. A giver cannot reliably protect the receiver's next discard if its prediction uses public beliefs but the receiver chooses from different private beliefs. The replacement is a publicly predictable fallback discard and a durable, mechanically decoded critical-save promise. This tests a different parent and mechanism from leader 0003's rotating immediate instruction channel.

## Trace evidence
Leader 0003 seed 14: turn 31 discards W5 (card 1) with one clue available, reporting `own discard E_loss=0.542, p_dead=0.286`. Turn 37 discards G3 (card 25) at zero clues, reporting `E_loss=0.605, p_dead=0.349`; G3 card 13 had already been discarded on turn 26. The game finishes 20 with three lives and green at 2. These are irreversible lost ceilings, not misplays or deadline truncation.
Parent 0001 seed 14 discards R3 card 0 on turn 6, then the last R3 card 28 on turn 39 with one token. It finishes 18, red at 2, all three lives. The parent's partner-discard prediction uses public marginals while its actual discard scores use private marginals and soft protection penalties. A save can therefore target the wrong slot or lose priority to an immediate own play.

## Changes from parent
- Retain exact joint finite-copy beliefs, public copy accounting, color play-focus protocol, seat-isolated model sessions and legal-action dictionaries.
- Choose the fallback discard using public-only marginals, public promises and stable oldest-slot tie breaks. Promised and publicly certain critical/playable cards are excluded while alternatives exist. A private certainly-dead card can override the schedule; an uncertain private guess cannot.
- Below eight clues, a rank 2..5 clue touching the public discard slot guarantees that slot is a reachable last copy at clue time. The sender validates this from its visible hand. The receiver reconstructs the domain of possible critical identities from public counts, masks and touched IDs. This domain persists until the card leaves. Other rank clues are literal, including all rank clues at eight tokens, so a truthful legal filler always exists at full tokens.
- Evaluate imminent partner danger before taking an own safe play. A verified new play for the partner can substitute for a save; otherwise a critical save takes priority. Clue flags alone no longer permanently protect every incidental touch.
- At most two short Astra option selections, each in a separate seat session with an 80-second outer wait and 75-second model timeout. Garbage replies fall back to the scripted choice; any exception disables further advice. No hidden state, seed inputs or cross-seat private memory.

## Predictions and falsification
Confirm if threatened last-copy discards fall and reachable ceilings improve without excessive save clues, while complete episodes remain legal and safe. Refute if public discard selection throws away useful duplicates faster than private inference, saving fills both hands and stalls progress, or zero-token forced discards still dominate. Track public scheduled discards, save promises, certain-dead overrides and final ceilings. A single dev seed is diagnostic rather than a generalization estimate.

## Validation
Implementation and this proposal were written before evaluation. Preflight and dev findings will be appended below.

### Completed validation
Preflight passed with a natural game ending, 65 actions, no errors and zero model calls. Its 20-point smoke-test score is not a performance estimate. A focused public-information check covered all 72 nonterminal observations of parent 0001's weakest trace: removing every visible face left the public discard schedule and touched-ID-based decoder unchanged. The offered clues included 27 verified truthful promises, ten critical saves. Separate fake-context checks passed for malformed JSON, out-of-range choices, model timeout, backend failure, outer queue timeout, and independent initial sessions for both seats. The joint-belief algorithm is inherited unchanged from the parent's tested implementation.

The single dev evaluation, `dev/20261030-002624`, finished **23/25**, 70 actions, 80.57 seconds, three lives, two clue tokens, and no episode error. There were 23 certain plays, 19 scheduled discards, 14 play clues, seven save clues and seven literal clues. Offline public-event accounting found **zero live last-copy discards**. Saves at turns 13, 21 and 35 protected future W5, W3 and B4; later rank saves also directly identified playable critical prerequisites. The remaining stacks were R5/Y4/G5/W5/B4. The missing fives were held until the final countdown, so this run's residual loss is tempo/access rather than a lost physical ceiling. Do not infer superiority over the training leader from this single diagnostic episode.

The sole adviser request reached the 80-second caller wait without returning; the harness logged `adviser_fallback` with `TimeoutError`, disabled advice and completed automatically. The evaluation records one requested call and zero failed backend calls because the pending request was cancelled at episode close; no successful adviser contribution is claimed. No second dev evaluation or training/held-out evaluation was run.

After this run, only diagnostic/cleanup changes were made: remove an unused inherited protection helper, align the constructor's initial adviser spacing with episode reset, and log live last-copy losses using the newly public discard event. The decision policy is unchanged. Final preflight after these edits passed again: natural completion, 65 legal actions, no error, and the same 20-point smoke-test result.

### Limitations for the next iteration
A critical-only rank-save promise can reject an otherwise useful literal clue to a noncritical discard slot. Public scheduling also deliberately gives up some private marginal discard optimization. These are real costs, not established wins; inspect held prerequisites and unnecessary duplicate discards if training scores regress. This proposal tests predictable save coordination, not optimal multi-turn clue planning. The dev game also suggests a separate future hypothesis: allocate more effort to finishing held dependency chains before consuming the last few draws. Do not solve that here by borrowing another seat's hidden hand or changing conventions midgame.
