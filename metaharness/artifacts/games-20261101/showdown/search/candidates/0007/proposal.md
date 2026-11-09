parent: 0001

# Hypothesis

The best candidate loses useful reasoning budget because it treats every turn as a fresh Astra decision, while its own continuation mechanism almost never executes. The seed 232003 trace shows repeated Earthquake decisions into the same Mamoswine (calls 28, 33, 38, 43) and then repeated calls to finish the same low HP threat (calls 49, 54, 61). Each answer re-derives the same matchup, with 24–101 seconds of model latency, and the episode eventually falls back to weak play. The model was often directionally right, but paying for repeated choices leaves too little budget for late threats.

A call-free continuation is only safe when the preceding exchange verifies the assumptions. The same trace also shows why an unrestricted script is dangerous: Mamoswine's hidden priority and coverage were uncertain, and later Mismagius setup/status changed the matchup. The hypothesis is therefore that a *short, same-move queue* with strict observation guards can capture genuine repeated attacks while preserving Astra calls for switches, setup, Tera, new reveals and endgames.

# What changed from the parent

Candidate 0007 keeps the parent's complete recent protocol report, public reveal ledger, rough type/power fallback, robust JSON parsing, legal-action recovery and `ctx.progress()` checks. Its model contract is smaller: Astra returns one immediate action and an optional queue of zero to two copies of the same damaging move. Queue entries are compiled only when that move is the current legal action, is non-status, is non-Tera, and targets the current active identities.

After each queued attack, code verifies the next request. It cancels the queue on an opponent switch or newly revealed move, status/boost/field/item/form event, forced replacement, action rejection, low HP, a missed/immune/failed attack, no measurable foe HP progress, or an observed incoming hit that makes the active unsafe. Move slots are remapped from the current request each time. Thus the queue cannot carry a stale slot or a plan across a hidden coverage reveal. A fresh Astra decision clears stale failure evidence and can replace the queue. The prompt explicitly forbids queuing setup, recovery, switching, Tera, or moves justified only by an unrevealed coverage assumption.

The call admission threshold is raised to 290 seconds of remaining wall budget and the request timeout is 260 seconds, above the archive's 236-second worst observed call. Short answers (under 100 words) reduce latency. One call remains reserved; timeout, backend failure and malformed output fall back to a locally ranked legal action without escaping the harness.

# Evidence that would confirm or refute it

Confirmation would show queued actions in traces, fewer Astra calls and more calls available after the first opponent team reveals, with improved weak-seed outcomes or HP margins. The key safety evidence is queue cancellation on an unexpected move, status, immunity or low-damage exchange, followed by a new Astra plan. Refutation would be queues rarely being accepted, queues carrying mistaken assumptions despite the guards, or saved calls failing to improve wins because the fallback still decides too many turns. A queue that increases damage but loses late-game checks would also refute the premise. No seed-specific move or opponent rule is compiled.
