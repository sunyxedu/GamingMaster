- The baseline's `parse_action` was given `action_spec.names` (`choose`, `select`, `order`, `noop`) rather than prompt choices, so `choose r1p1 1` became bare `choose` and stalled on a stale token.
- Exact prompt actions must be validated against the current token; action names alone are insufficient for this adaptor.
- Singleton engine prompts can advance without Astra calls, preserving the 40-call budget for strategic choices.
- Compacting repeated static card catalogues should reduce Astra latency while retaining prompt, board, chain, and recent events.
- A harness must contain TimeoutError/RuntimeError/BudgetExceeded and recover with a legal fallback; arbitrary stand-in model text is part of pre-flight.

### Candidate 0003 — code-first tactical controller

- The baseline's score is a transport failure, not a strategy measurement: seed 1101 returns `choose r1p1 1` from Astra but posts bare `choose`; the adaptor reports `Expected current prompt token and arguments`, then forfeits all three duels.
- Preserve and validate the entire action, including the current prompt token. A model retry cannot fix a downstream parser that discards arguments.
- Native prompt counts greatly exceed strategic decision counts. Summon placement and irrelevant optional response windows should not each consume a model call; reserve calls for meaningful exchanges and retain call-budget headroom.
- First dev observations already show a plan/execution mismatch: an adviser recommends setting Penguin Soldier while preserving removal, but a generic positive Fissure score immediately spends the removal. Natural-language intent needs a small executable contract if subsequent actions are scripted.
- Candidate 0003's dev observations show `mode: subset` prompts can also have a `choices` array containing only `cancel / select none`. Dispatching by nonempty `choices` creates legal attack/cancel and summon/cancel loops with fresh tokens. Dispatch by mode and detect repeated public states, not just repeated tokens. Fixing this changed preflight from deadline termination to game completion; the final check used 435 actions (score 1.090833).
- A requested model timeout did not bound the observed request-to-return stall. A locally bounded wait on a single `ctx.spawn` inference future kept both second-dev episodes playing without forfeits. Discard late token-bound answers and test garbage/failure fallback paths separately when preflight does not exercise a model call.

- Candidate 0003's full native completion shows that exact action serialization plus a public-state controller can solve the contract; model frequency is a separate design variable.
- A System-2 call can remain in flight after the engine has already ended the duel (the leader trace logs cancellation after score 1.1), so late arbitration has measurable cost without strategic benefit.
- Replanning from fresh observations after every `ctx.act` is safer than executing a multi-step natural-language plan; prompt tokens and zone slots are transient.
- One opening policy call per duel preserves a place for deck/seat reasoning while preventing chain-response latency from consuming the 1800-second episode budget.
