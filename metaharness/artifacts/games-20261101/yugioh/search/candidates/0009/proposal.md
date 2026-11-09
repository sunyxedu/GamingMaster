parent: 0004

## Hypothesis

The remaining weakness is the division of reasoning: opening-turn action queues
spend scarce slow inference on routine summons while fixed rankings decide
context-sensitive interruption and resource exchanges. Replace that division,
not the leader's position or combat recipe. Astra should review critical current
decisions, including whether to preserve interaction, and commit at most one
effect plus its targets. Routine command execution stays cheap.

Concrete evidence: 0004 completes every training duel and wins 24/36, yet its
worst seed 1111 scores -0.372708 with two losses. It uses only 31 System-2 calls
across 12 episodes and only about 0.54% of actions directly follow fresh Astra
answers. In trace_seed1111.jsonl, duel 1 step 37 (`choose r1p37 0`) spends
Compulsory Evacuation Device on a just-summoned Vorse Raider while the agent has
8000 LP and an empty monster field. Step 39 (`choose r1p39 2`) ends turn 8 with
Book of Moon in hand and no monsters; step 43 again ends turn 10 with Book in
hand, now at 6000 LP. Those are observable resource/timing decisions, not claims
that a different decision would certainly win. The parent has no selective
review of these choices; its generic quick-play hold rule suppresses setting
when Battle Phase is offered, even when no attacker makes that phase useful.
The parent's proposal also reports an Astra turn queue that omitted Battle
Phase after summoning its intended attacker. Long future queues can override
sound local play without purchasing better interruption decisions.

## What changed

Remove normal opening/turn planning and its future action queue. Keep the
parent's local rules, legal validator, position semantics and visible battle
allocator as a controlled baseline. Review only a live decision with potential
high regret: an opponent chain we can answer, a defensive battle involving
flexible interaction, an end turn that leaves a quick-play in hand, a contested
Dark Hole, a Tribute investment, or competing removal/revival options. At most
four ordinary reviews per duel and fourteen calls including repairs protect
completion; identical tactical states do not cause repeat consultations.

Astra returns one current legal action and optional exact visible target
references for the effect activated now. That small target contract is compiled
only when the same own effect appears in the current chain; native subset
constraints are validated. No future tokens, turn queue, hidden identities,
or simulator are used. Plans expire across turns/duels, absent targets trigger
repair, and every action is checked with ctx.progress(), prompt/error changes,
semantic repetition and observed position consequences. Replanning receives the
failure evidence. All model failures fall back to local legal play.

Timeout 235 seconds and independent 210-second wait guard exceed the archive's
120-second observed latency while leaving time before the native 300-second
pending-decision deadline. One context-managed call may remain in flight, but
late actions are discarded and no concurrent model workers are created.

## Confirmation and refutation

Confirm if reviews actually change critical interaction/deployment decisions,
the target contracts bind correctly, native completion remains reliable, and
wins/weakest-seed results improve beyond the parent across decks and seats.
Refute if sparse forks miss the important choices, Astra spends too much time
on routine effects, contracts mis-target, or local execution with fewer model
turn queues is no better. Replay/preflight can test legality and contracts;
only native evaluation can establish strength. This is a different orchestration
hypothesis using the leader as an unchanged tactical baseline, not a further
position-score or combat-weight refinement.

## Validation

Pending preflight and permitted development evaluation.

Replay and preflight validation: candidate 0009 replayed 3,458 archived leader
observations with zero locally generated invalid actions. A context stub tested
garbage, malformed fields, empty replies, timeout, backend failure, exact action
extraction, target contracts, wrong chain ownership/effect/turn, unrelated attack
target subsets, vanished targets, weighted Tribute selection and full ordering.
All passed. Preflight also completed the stand-in three-duel episode with no
exception. It made no Astra call, so the stub/replay checks cover response parsing
and critical-fork detection separately. No real-model dev evaluation was run.
