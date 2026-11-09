parent: none

Hypothesis: a bounded semantic plan, compiled against each fresh native legal
menu, can amortize Astra across a turn and preserve completion under slow
inference. Plans need two distinct checkpoints: action legality and actual
board progress. This is a fresh design rather than a refinement of 0001's
parser repair or 0002's delegation of most effect decisions to Astra.

The archive's flat leader is not evidence about playing strength. Its score
breakdown contains 36 forfeits and zero native completions. In
`trace_seed1101.jsonl`, the first successful call reports 53.93 seconds of
latency, but the next step submits bare `choose` and advances to a forfeited
duel. The generic parser has discarded the token and arguments. All candidates
need exact transport; this proposal instead tests how to obtain useful play
within the latency budget after transport is fixed.

The final harness asks Astra for one exact current action and up to six future
main-command intentions described by label and card name. It never predicts
future tokens. Each intention is rebound to the observed legal menu; impossible
intentions are discarded and sent back with failure evidence. Observations are
self-contained compact JSON with the complete live prompt, current stats,
public/own zones, chain, the last 12 events and effect-card text. No accumulated
session transcript or external state is used. Code handles zone placement,
legal subset/weighted tribute enumeration, target ranking, basic combat and
conservative reaction timing. Card references in menus are joined to current
visible zones/catalogue rather than assumed to carry their own stats.

Every action checks `ctx.progress()` deltas, prompt advancement and action
errors. A fingerprint excluding the prompt token detects repeated decisions
with the same board, chain, recent events and legal menu. Repetition triggers
Astra review and then a legal phase exit if the repaired line still cycles.
Unexpected friendly losses and turn/round boundaries invalidate intentions.
The planner has an 18-call soft budget, normally six reviews per duel; unused
calls permit failure repair. Local execution remains available near the wall
limit. Native rules and observed legality remain authoritative.

Inference runs via `ctx.spawn`/`ctx.think`, with a 155-second backend timeout
(above the archive's observed 106-second maximum). A separate 135-second
wall/game-clock wait guard resumes local execution while the request is still
queued or running. Only one request may be in flight. A late answer is discarded,
never applied to a newer token. Same-prompt retry waits are also bounded. All
model failures and action rejections are caught, and no free-running worker is
created beyond the context-managed inference future.

Development evidence and repairs:

- First dev run, seed 3301: score -0.672083, two forfeits and one truncation,
  no technical errors. A call requested at trace time 1793403692.180 returned
  at 1793403995.399: 303.22 seconds elapsed despite reported model latency of
  63.61 seconds. This caused a response-deadline forfeit. The separate wall
  guard addresses queueing/overhead not represented by model latency.
- Astra's first plan tried a Normal Set after using its Normal Summon and
  described Set Fissure as opponent-turn removal. Live-menu matching rejected
  the impossible Set. The prompt now explicitly states the single Normal
  Summon/Set limit and Normal versus Quick-Play Spell timing.
- Both dev runs exposed an attack/cancel cycle. A `select_cards` prompt has
  `mode: subset`, candidates and minimum/maximum, AND one `choices` entry:
  `cancel / select none`. Treating that entry as a forced singleton silently
  cancels the attack. Tokens keep advancing, so token-only monitoring misses
  the failure. The final version prioritizes compound candidates over that
  cancel button and adds semantic cycle detection.
- Second dev run, seed 3302: score 0, three truncations, zero forfeits or
  technical errors, 359.5 seconds, one in-flight planner call. It ran the
  wall-guard revision but was launched before the cancel-selection fix; it
  validates recovery from a queued call, NOT final playing strength. Its zero
  score is a failure to finish natively and is not claimed as an improvement.
- After repair, local action validation against more than 16,000 archived dev
  observations found no invalid generated actions. Preflight changed from
  10,000+ actions/deadline to native game_done in 316 actions, score -0.385,
  with no exception. Preflight made no model calls, so a separate context stub
  checked four garbage/model-error cases, stale tokens, duplicate selections,
  incomplete orders, weighted Tributes and semantic rebinding. The final
  action policy passed all 193 distinct archived-observation checks with zero
  invalid responses (`checks.json`). These are contract and regression checks,
  not a training or held-out result.

Confirmation requires three native completions without forfeits/truncations,
more native decisions per Astra call, and then improved native win rate across
training matchups/seats. Refutation includes renewed semantic cycles,
frequent unavailable intentions, large opportunity costs from local reaction
choices, or dependency on queued inference never completing. Both permitted
dev evaluations were consumed before final repairs; final playing strength
remains unmeasured. No seed-specific decisions, hidden deck identities,
private game implementation, or held-out results were used.
