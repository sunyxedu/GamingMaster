parent: 0009

## Hypothesis

The bounded two-turn exchange search in 0009 is structurally sound, but its
opponent portfolio becomes too optimistic when only weak moves have been
revealed. Random-battle sets are curated four-move sets, and a species can keep
its decisive STAB or coverage move hidden for several turns. The parent then
scores a voluntary switch or setup against a narrow set of replies and can
walk into a catastrophic first hit. This is a hidden-action uncertainty
failure, distinct from the leader's model-action firewall and from 0011's
repeat-policy prior.

The leader's worst archived episode is concrete evidence for this gap: in
`0012/trace_seed333667.jsonl` the ledger records repeated unexpected attacks,
including `"move": "boomburst"` for 54.2% of Raichu and 64.5% of Chimecho,
then `"move": "gigatonhammer"` for 96.2% of Clefable. The fallback had to
react only after each move was revealed, and the episode finished at score
`-1.12817`. Those attacks were absent from the public own-move reference at
the earlier decisions, so a narrow reply portfolio could not price the risk.

## What changed versus the parent

I retain 0009's public-only ledger, two-turn search, legal filtering, timeout
recovery, and Astra's role as a source of explicitly uncertain move facts. In
`SearchBoard.replies`, whenever fewer than four opponent moves are revealed, I
add one low-weight adversarial branch for the strongest unrevealed STAB type
that can damage the current active. Its category is inferred only from the
opponent's observed/estimated attacking stats, its base power is a generic 85,
and the branch is marked synthetic in the existing search path. It receives an
8% prior, while revealed and Astra-supported moves retain the parent weights.
The branch is regenerated for every current request and never exposed as a
fact in the prompt or ledger. Known four-move opponents are unchanged.

This makes the root value risk-sensitive: a line that wins against all known
replies but loses outright to a plausible hidden STAB pays a controlled penalty,
while robust attacks, defensive Tera, and replacements remain competitive.

## Evidence that would confirm or refute it

Confirmation would show fewer replacement KOs and fewer immediate losses after
the first reveal, especially in traces where the opponent later unveils a
high-power move absent from `move_dex`; root logs should show `surprise<Type>`
branches changing the selected action. The score should improve on weak seeds
without increasing illegal actions or model failures. Refutation would be a
drop on positions where the revealed move list is already representative, or
repeated over-switching/setup avoidance caused by the 8% branch. If refuted,
later work should lower the branch prior or condition it on observed damage and
speed evidence rather than removing the ledger and search entirely.
