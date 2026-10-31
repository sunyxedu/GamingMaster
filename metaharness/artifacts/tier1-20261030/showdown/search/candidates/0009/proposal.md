parent: 0000-tier1_astra_naive

# Hypothesis: retain tactical reasoning, remove the memo/policy translation layer

The leader and its descendants spend time generating strategic memos and proposed continuations, then often discard those continuations and finish with heuristic actions. Candidate 0001 has 250 model calls for 355 decisions; its mean -0.641 masks a median -1.035 and worst -1.125. Later policy candidates remain near -0.9 to -1.1. This candidate tests a different parent and a different communication architecture: a bounded continuing Astra conversation, one fresh decision per consultation, very short output, complete dynamic snapshots and incremental static rules. It intentionally generates no model-written memo or compiled multi-turn policy.

Concrete evidence from the leader's worst episode (`0001/trace_seed232003.jsonl`): turn 1 predicts "hazards or Lava Plume" and loses full-health Swampert to Solar Beam. The Piloswine replacement rationale calls Earthquake "super-effective" against Mamoswine (it is neutral). At turn 15 it switches Scyther into already revealed Icicle Crash and loses it immediately. These are failures of tactical assumptions, not simply absent observations. In `0001/trace_seed181081.jsonl`, the switch into Skarmory is justified as handling Flying and Fighting "via resistances"; Fighting is neutral to Steel/Flying. After two Bulk Ups, Close Combat KOs Skarmory from 89.8%, and the rest of the battle is largely fallback. More verbose state/memo output did not reliably preserve grounded reasoning.

The naive parent's worst episode (`0000-tier1_astra_naive/trace_seed333667.jsonl`) also shows why merely restoring its renderer is insufficient: its first prompts truncate inside move_dex and never deliver recent_events/species_dex. Its existing generic, cumulative sessions repeat the incomplete input. Candidate 0009 preserves the continuing conversation but replaces information delivery and action validation completely.

# Changes

- Each consultation supplies every current team slot, exact private stats/HP/items/ability/moves, active request flags, all revealed opponents, field/effects, legal commands and protocol since the previous consultation. Nothing is generically truncated. Move and species rules are sent once per session, then as explicit additions/replacements. Ten consultations trigger a new checkpoint with observed history and damage/order evidence; failed replies/calls and rejected actions also restart the session.
- A direct type-chart audit exposes all own weaknesses/resistances and active attacks' type factors, including defensive Tera. These are explicitly only type-chart factors, never damage estimates or asserted ability/field behavior. Neutral interactions are not described as resistance. Dynamic move types are left for protocol/mechanics reasoning.
- Public move descriptions for already revealed enemy moves come from a static dictionary compiled from this archive's public move_dex observations; it contains no teams, seed rules or hidden sets. Observation move data override these records.
- Astra returns only the legal action and an eighteen-word strongest-response risk. The conversation supplies reasoning continuity; no regenerated strategic memo, damage matrix or policy schema is required. Unique legal choices execute directly. All other ordinary choices receive fresh reasoning while budget remains.
- Every action checks ctx.progress(), rejection, failed effects, actual damage, fainting, changed enemy boosts and four-action recovery cycles. Facts and failures go to the next consultation, including after every emergency action. Damage/order evidence is labeled contextual and never upgraded into a universal guarantee.
- Timeout is 300 seconds, above the archive's 236-second worst call, with caught timeout/backend errors and robust parsing of arbitrary responses. A 120-second call-admission reserve tests retaining reasoning deeper into the battle; a long late call can still reach the normal episode deadline. One call stays unspent to avoid premature closure at the call limit. The emergency fallback borrows 0002's reference arithmetic, adds known revealed coverage to switch evaluation, and is never presented to Astra as exact calculations.

# Confirmation and refutation

Confirm if the compact continuing conversations deliver more model-informed late decisions with shorter latency/output, reduce the fallback share, and improve weak-seed completed outcomes. Inspect factual matchup errors and whether failures actually change the next decision. A reduction in fallback or prompt length without improved play is not sufficient. Refute if conversation history amplifies stale assumptions, the short answer loses useful scrutiny, session checkpoints erase important context, or the late call-admission policy merely converts played losses to deadline ties. We report terminal reason as well as score.

This is one candidate testing a communication/control architecture, not a claim that type-chart cards alone solve hidden-information search. No training-seed prescriptions, game implementation access, hidden data or unaccounted inference are used.

# Validation

Preflight, archive replay, model-path fault checks and one dev evaluation are completed below.

Offline validation completed: 1,446 complete archived observations rendered without exceptions and produced legal emergency actions; 1,376 recorded transitions passed effect verification. Largest initial checkpoint: 12,503 characters. Mocked complete loops passed valid model output, arbitrary text, TimeoutError and RuntimeError; checks also cover ten-call session rollover, lost session IDs, static-rule replay, current-slot delivery, known type interactions and all three progress failure signals. One initial preflight ran before a failed file-assembly step was repaired and found a missing import. After restoring the complete helper/import section, preflight passed (21 legal actions, natural terminal loss, no error). Its short budget did not admit a model call, hence the separate model-path checks.

One dev invocation completed both allowed dev seeds. Its loaded harness includes the missing-session static-data repair. Results and any later changes are documented below.

# Dev evaluation (two diagnostic seeds)

The completed invocation `dev/20261031-173846/` had no episode errors or failed model calls:

| Seed | Score | Decisions | Astra calls | Fallback / only-legal | Wall seconds |
|---|---:|---:|---:|---:|---:|
| 869431 | -1.126842 | 39 | 24 | 14 / 1 | 1743 |
| 895003 | 1.030260 | 26 | 24 | 2 / 0 | 1691 |

Mean score was **-0.048291**. The win had three survivors and removed all opponents; its final two actions were fallback. The loss crossed the 120-second call-admission threshold after 24 calls, leaving 16 calls unused; it then played 14 fallback actions and ended with all six own Pokémon fainted. Median end-to-end model-call intervals were approximately 64 s and 73 s for the two seeds, with maxima around 131 s and 136 s; median prompt sizes were 6.6k and 7.1k characters, but checkpoint prompts reached 20.4k and 26.8k. This confirms the incremental renderer is usually compact while checkpoint state can still be large.

The win demonstrates working protocol handling and late model decisions: Malamar used Rest and Sleep Talk before a critical Dragon Pulse finished it, two Toxic Spikes layers from Glimmora taking physical hits supported the endgame, and Crawdaunt finished the poisoned Baxcalibur. The loss shows the strategic limitation clearly. Astra repeatedly selected attacks into a recovering Arceus line and eventually tried Nasty Plot with Gengar at 86% while Arceus was unrevealed beyond Will-O-Wisp/Recover; a critical Judgment then reduced Gengar to 15%, and newly revealed Earthquake finished it. The harness recorded the critical incoming packet, and an earlier four-decision recovery cycle. Judgment was disabled by Cursed Body after the critical hit, so the second setup was a gamble on unrevealed coverage, not proof that Astra ignored a known usable lethal move. This is evidence against claiming that a short risk field alone fixes survival errors.

The dev result is mixed and does not establish stronger play. The same two dev seeds gave 0001 a mean of -0.039094, close to this result; these stochastic single runs are not a controlled strength comparison, and neither can be compared directly to the training leaderboard. Compact continuing prompts and robust recovery completed both battles without infrastructure errors, but sixteen fallback actions across the two games refute a claim that this architecture solves the late-budget problem. The next iteration should test stronger action gating or counterfactual survival checks, informed by these traces.

# Final revision

Dev source SHA-256: `f1e55c00e4d37fde45e270ae072d6aeb538c24c1004196eb1345a5c14f0c3bc7`. Final source SHA-256: `bc7cbeac02a6dbde0c67a62103bf7970e004391eed894aa0e6263baa10691e59`. The only post-dev code change repairs terminal logging: when `request` is null, a surviving own Pokemon's HP is read from the terminal public record instead of incorrectly defaulting to zero. It changes no action selection. A targeted null-request/surviving-HP regression and final preflight both pass. Final preflight again completed 21 legal actions with no error and exercised fallback only; the earlier mock tests cover model paths. Validation details are in `validation.json` and `check_harness.py`. No second dev invocation or full training evaluation was run.
