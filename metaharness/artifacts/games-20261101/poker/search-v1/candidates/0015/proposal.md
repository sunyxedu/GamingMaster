parent: 0007

## Hypothesis

Candidate 0007 controls the first wager but still treats a postflop raise as a normal continuation branch. The heuristic-v1 opponent's raise branch is deliberately value-heavy: it only raises after calling when its cached random-hand equity is at least .80, and it gets one raise per street. A medium-strength hero hand can therefore pass the four-pot cap, call a raise, and later commit the stack against a range that is much stronger than the original caller range. The archived 0007 development trace exhibits this exposure: seed 902 hand 96 eventually commits 20,000 with Ts8h after a turn raise from QcQs. The river outdraw is variance, but the public raise is a strong signal that the parent does not price separately.

I test whether explicitly separating “facing a raise” from ordinary calls improves chips per hand. Once an opponent raise is visible on the current street, the harness folds one-pair and draw bluff-catchers unless their range equity clears a higher street-specific threshold (0.72 on flop/turn, 0.66 on river), and it disallows a hero re-raise below 0.82 equity. Strong made hands can still value raise. This is a structural response policy, independent of seed and hidden cards.

## What changed vs. the parent

The exact evaluator, public-action range update, preflop ranges, bounded pot-relative sizing, legal total-raise mapping, and cancellation-safe loop are inherited from 0007. In `postflop`, a current-street opponent raise is detected from public history. Before the normal EV search, the policy applies a raise-response gate: it chooses a legal fold/call only when the posterior equity is above the street threshold, and it filters candidate hero raises below the stronger threshold. Ordinary checks, bets, and calls against non-raising bets retain the parent policy. No model calls or hidden engine state are used.

## Evidence that would confirm or refute it

Confirmation would show fewer calls or re-raises after opponent raises, no illegal actions, complete matches, and improved mean net chips across independent seeds while retaining value extraction against passive calls. Refutation would be a score drop caused by folding profitable raises, or traces showing that the opponent raises frequently with weak holdings despite the documented .80 threshold. The hypothesis is falsified by public-action calibration and completed match scores, not by one river result.
