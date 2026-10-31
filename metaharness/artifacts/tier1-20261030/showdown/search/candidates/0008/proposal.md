parent: 0001

# Hypothesis

The strongest recurring failure is an information-ordering error: Astra spends an early turn on setup or status while the opposing set is still almost completely unknown. In the worst 0001 trace (seed 333667), Clefable used Calm Mind on turns 2 and 3 into a Giratina that had only revealed Pressure; the model did not yet know whether a physical or special coverage move could punish it. In another loss (seed 232003), the eventual endgame revealed Psychic Noise only after the active Munkidori was committed. The long evidence reports improved visibility but still leave the model free to make an irreversible setup bet before a damaging probe.

# What changed vs. the parent

Candidate 0008 keeps the parent's evidence-first state delivery and public-only bookkeeping, but adds a structural threat barrier. During turns 1–2, if the opponent has revealed no damaging move, code rejects Astra's setup, status, or other non-recovery status action and chooses a legal damaging move or switch. This forces one low-cost information/progress exchange before setup. Astra now returns a ranked list of up to three actions; malformed or unsafe first choices fall through to a backup, and the harness validates known type immunities and stale slots. A single extra attack is authorized only after the previous same-matchup attack visibly removed at least 15% foe HP while costing under 20% own HP. The repeat is canceled on changed identities, low HP, rejected actions, stalled progress, or any new request. The fallback remains a bounded type-aware scorer and never repeats an unverified move.

The call reserve is 285 seconds, above the archive's 236-second worst observed System-2 latency, so a late call does not strand the episode. Timeouts and arbitrary model text are caught; terminal null requests and hidden trapping are handled by rereading the observation.

# Confirmation / refutation

The hypothesis is supported if early setup/status actions decrease, first damaging reveals happen sooner, and weak seeds show fewer losses caused by an unmodeled coverage move, while the verified repeat saves calls only in genuinely favorable exchanges. It is refuted if the barrier sacrifices useful safe setup (for example, a guaranteed Unaware or immunity line), if fallback damage ranking is too weak when Astra fails, or if delaying setup gives MCTS enough free turns to overwhelm the lead. The key evidence is held-out win rate and HP margin, with traces checking barrier activations, repeat cancellations, and model-fallback frequency.
