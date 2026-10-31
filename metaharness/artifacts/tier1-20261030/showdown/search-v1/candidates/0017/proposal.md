parent: 0012

## Hypothesis

The leader's replacement and damage ledger is useful, but its local scorer still
has a bad terminal tradeoff: it treats a fast, revealed attack as a reason to
switch even when the opposing active is already nearly fainted.  In the worst
leader episode (`0012/trace_seed333667.jsonl`), Noivern's Boomburst reduced
Raichu to 46% but left Noivern at 9%; the fallback then chose `switch 5`,
exposing Chimecho to another Boomburst.  Chimecho fainted, and the same pattern
continued until the team was gone.  The failure is visible in the protocol and
does not require guessing a hidden set: a low-HP enemy with a reliable KO line
should usually be finished even if it moves first.

## What changed

I retain 0012's public-only ledger, empirical damage tracking, policy
continuation, Astra firewall, timeout recovery, and current-request legality.
The fallback ranker now gives a narrow bonus to an accurate damaging move when
the foe is at most 18% HP and the move's estimated damage has a 20% margin over
the foe's remaining HP.  The bonus is applied after the generic slower-than-a-
lethal-hit penalty, so it specifically prevents fleeing from a near-certain
finish while leaving healthy-foe switching unchanged.  It is still gated by
the existing damage/type model and accuracy estimate.

I also add verified public mechanics for several high-impact moves that are
often first revealed on the opponent side (`Boomburst`, `Gigaton Hammer`,
`Play Rough`, `Dragon Pulse`, and `Eruption`).  Once the protocol names one,
the same threat and replacement calculations can use its real type, category,
and base power.  These entries do not expose hidden moves or sets and remain
overridden by the observation's move dex.

## Evidence that would confirm or refute it

Confirmation would show fewer voluntary switches immediately after an opponent
falls below the finish threshold, fewer replacement KOs, and a better result in
losses resembling the Noivern sequence, without increasing illegal actions.
Rank traces should show `move` outranking `switch` only when the foe is both low
HP and in the modeled KO range.  Refutation would be a score drop caused by
staying into a priority move, inaccurate estimate, or a forced double-switch;
then the threshold or accuracy margin should be tightened rather than making
the rule global.
