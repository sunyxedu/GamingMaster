parent: 0012

## Hypothesis

The leader is losing in two coupled ways after its first slow Astra answer. Its
fallback does not know several high-impact attacks until Astra supplies a fact,
and it can spend the remaining wall budget waiting for a second answer instead
of executing the already-tested local controller. The worst leader trace makes
both failures visible. In `trace_seed333667.jsonl`, the first answer is accepted
at virtual clock 1374 s. The next call is queued, reaches the 100 s reserve
cutoff at 1700 s, and the episode then proceeds with fallback choices. The
protocol reveals Noivern's `Boomburst` (140 BP Normal): it removes Volcanion,
then 135/249 of Raichu and 187/290 of Chimecho. The fallback switches into that
same threat and spends Poison Tera before Chimecho faints. Later `Gigaton Hammer`
removes Clefable. These are public mechanics and scheduling failures, rather
than a need for a longer model prompt.

## What changed versus the parent

I retain 0012's public-only ledger, empirical damage calibration, action firewall,
short policy continuation, legality recovery, and deterministic ranker. I add:

* Verified public entries for `Boomburst`, `Gigaton Hammer`, `Play Rough`,
  `Dragon Pulse`, and `Eruption`. They are consulted only after a protocol
  reveal and are still overridden by the observation's move dex; no hidden set
  is inferred.
* A narrow terminal-finish bonus. When a reliable damaging move has at least
  an 80% estimated damage margin against a foe below 18% HP, the ranker restores
  that move after the generic faster-foe penalty. This prevents fleeing from a
  nearly guaranteed finish while leaving healthy-foe switch decisions intact.
* A late-call reserve. After one Astra call, a second call is allowed only
  while more than 850 virtual seconds remain. Once the first answer consumed
  the leader's observed 1374-second queue delay, code immediately continues
  with legal actions and preserves the episode's remaining time.

## Evidence that would confirm or refute it

Confirmation would show fewer replacement KOs after a revealed high-power move,
more finishes against low-HP opposing actives, and no episodes stalled by a
second queued call. The harness should retain the leader's wins and zero-error
contract while accepting more decisions after the first model answer. Refutation
would be a lower win rate because the finish gate trades away a needed switch,
or because skipping a second Astra answer loses valuable setup/switch planning
when the first call was unusually cheap. The trace should then show whether the
problem is the 18%/80% gate or the 850-second reserve.
