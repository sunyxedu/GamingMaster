parent: 0004

## Hypothesis

The remaining failure is not attack assignment: the controller evaluates a
reactive card before knowing whether its intended target is valid, and evaluates
each response as though earlier chain links did not exist. A transaction joining
activation, target, pending resolution and observed consequence should conserve
interaction and prevent self-harm. This tests a different hypothesis from the
parent's position semantics and public combat allocator.

Concrete evidence in the leader's losing episodes:

- `trace_seed1101.jsonl`, steps 171–172: `choose r2p49 0` activates Rush
  Recklessly while the friendly Luster Dragon is face-down. The mandatory
  face-up selection only contains the opponent's Gemini Elf, and
  `select r2p50 0` buffs that enemy. Seed 1105 repeats this at steps 17–18.
- `trace_seed1105.jsonl`, steps 180–183: two Rush Recklessly copies are chained
  onto the same 1700 ATK Battle Ox against 2000 ATK Alexandrite Dragon. The
  second activation sees the first link and its target in the public chain,
  but still uses the unmodified 1700 ATK. One +700 is already sufficient.
  This duel eventually loses with the opponent at 400 LP. This is observed
  waste, not a claim that conserving the card would necessarily win.
- The worst seed 1111 finishes at about -0.373 with two native losses, so
  completion and legal actions alone do not establish tactical correctness.

## Change under test

Retain the parent's transport, legal subset validation, position semantics and
bounded public combat calculation. Replace the reactive controller with a
public-chain projection and effect transaction: score an activation together
with its target, project visible pending links in reverse order, bind that
target to the fresh native selection, and audit the result after resolution.
Only pinned pool-card semantics and public information are used. Unknown
interaction remains uncertainty, and the native menu always decides legality.

Remove future main-command queues. Astra instead reviews a current difficult
exchange or failure, receives the local proposal, chain projection and failure
evidence, and returns one exact current action plus optional target advice.
Calls have a 235-second timeout (above the observed 120-second maximum) and a
separate 210-second wait guard; same-prompt waits remain below 240 seconds.
Every action checks ctx.progress(), prompt movement and semantic repetitions.
Resolution discrepancies invalidate the transaction and trigger review.

## Confirmation and refutation

Confirm that archived self-buffing/duplicate-response cases now decline, useful
single-copy responses still activate, target binding stays legal, and native
completion remains error-free. Actual improvement requires more native wins
across training seeds and seats, particularly weak episodes; replay checks
alone cannot show that. Refute if conserving interaction misses survival,
projection mistakes force bad targets, or deliberation causes forfeits.

## Validation results

Both permitted development evaluations completed natively:

| Seed | Score | Wins / losses | Steps | Wall seconds | Astra requests |
|---|---:|---:|---:|---:|---:|
| 3301 | 0.3785417 | 2 / 1 | 448 | 307.5 | 2 |
| 3302 | 0.3754167 | 2 / 1 | 244 | 215.1 | 1 |

Mean development score is **0.3769792**, with **4/6 native wins**, zero
forfeits, truncations, invalid actions or technical errors. This is the same
development win count as the parent; the small LP difference is not evidence
of a playing-strength improvement. Training evaluation must test that claim.

Only the first 3301 Astra request returned in time to be applied (83.67 seconds
reported model latency). The second 3301 request and the 3302 request reached
the 210-second wait guard and were canceled when their episodes completed.
Thus these runs mostly measure the local transaction policy; they do not
isolate the benefit of Astra's exchange review. All inference went through ctx.

The development traces exposed two audit defects. A monster bounced to hand
was immediately Set into the same slot before the next observation; the audit
mistook the unidentified replacement for the old target. Also, a newly chosen
Book response had to be included in the predicted chain, not just the incoming
links. Final audits require an observed matching card code before asserting
continued identity and include the submitted activation/target selection.
These final audit repairs have replay/fixture and preflight coverage, but no
third real-model run (the development allowance is exhausted).

Final checks in `checks.py` / `checks.json`:

- 3,458 available structured leader observations, zero invalid generated
  actions. Truncated string observations are skipped rather than reconstructed.
- Both archived enemy-buffing Rush activations decline; the useful first Rush
  remains active and the redundant second copy declines.
- Garbage/malformed model replies, empty text, timeout, backend failure, stale
  tokens, duplicate subsets, weighted Tributes and incomplete orders recover.
- New-response chain projection and hidden-slot replacement audit fixtures pass.
- Final preflight: **ok**, native `game_done`, 407 actions, no exception. It
  returned no stand-in model answer; parser/error paths were checked with a stub.

No full training evaluation, private game implementation, hidden state,
held-out results, or another candidate's unfinished proposal was used.
