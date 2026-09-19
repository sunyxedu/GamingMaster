# 0002 — "compiled bridge": Astra writes the Astra→Jev bridge as code

Parent: `0000-tier2_jev_only` (best so far, mean 0.000). Also informed by `0001-tier2_astra_only`.

## Diagnosis of the parent (evidence from traces)

`0000` seed 0: five Jev calls, **every one answers `keep`** (p = 0.68, 0.58, 0.60, 0.58, 0.54) while
the head walks x = 6 → 10 along y = 6 with `danger.right` never consulted; death by wall at tick 6.
Seeds 1 and 2 are identical (6 ticks, `keep`, wall). Two independent failure modes:

1. **No signal for the classifier.** Jev, handed the raw observation and "which action is best?",
   has nothing a zero-shot classifier can use — it just picks the status-quo option. Nothing in the
   state says *which* options are fatal or *which* move gains score; Jev would have to simulate the
   game to know, and it does not reason.
2. **Latency.** Jev p50 ≈ 0.57–0.60 s > 0.5 s tick, so one decision per tick is impossible
   (33 % stale). `0001` is the other extreme: Astra at ~2.1 s/call, 81 % stale, same wall death.

The seed harnesses therefore conflate two different jobs: *understanding the game* (which needs
reasoning, once) and *reacting every 0.5 s* (which must be instant and cannot wait for either model).

## Hypothesis

The right bridge between Astra and Jev is **executable code, not prose**. If Astra (once per
evaluation) turns `game.describe()` + `action_spec` + one sample observation into a function
`bridge(obs) → {options, utility, state}`, then

* every tick gets a **fresh, non-fatal** action in microseconds (options exclude moves that end the
  game on the next tick; the best utility is posted *before* any model call) → stale fraction ≈ 0
  regardless of model latency;
* Jev is finally asked a question it can answer: "among these 2–3 options, each labelled with its
  concrete consequence (`'safe, 2 cells closer to food'`), which is best?" — with Astra's short
  coach rules in the state;
* the code is cached in `ctx.memory`, so seeds ≥ 1 cost 0–1 Astra calls, and Astra can *repair* it
  from tracebacks and *improve* it from a death post-mortem (last 4 observations + actions), which is
  a cross-seed learning loop the seeds lacked;
* it is game-general by construction: nothing in the harness names an observation key or an action;
  the held-out game gets its own generated bridge (verified offline: a lanerunner-shaped bridge is
  adopted and drives `left/right/jump/stay` correctly).

## What the harness does

| moment | Astra (System-2) | Jev (System-1) | action posted |
|---|---|---|---|
| episode start, no cached code | 2 background calls: *guidance* (≤5 coach rules, stateless) and *code* (the bridge module; own session) | generic path: per-action `Noul` "would this end the game?" + `Choice` "best for score"; actions with p(end) > 0.5 excluded | Jev's pick (after the ~0.6 s call) |
| code arrives | validated on the live observation; on failure a *repair* call with the traceback + the module (≤ 2 rounds); same if it raises 3× at runtime | — | — |
| every tick with a bridge | — | `Choice` over the bridge's labelled options; state = coach rules + bridge `state` + last action | **bridge argmax posted first**, then Jev called synchronously; when it returns, re-observe, re-run the bridge and post `argmax(utility_norm + 0.5·p_jev/max p_jev)` (Jev only settles near-ties; ≤ 2 ticks old; restricted to live options) |
| late decision | the tick that elapses during a Jev call already runs on the fresh bridge action; the loop only `wait_for_tick`s when no tick elapsed, so it never spins and never races the clock | | |
| next seed | cached bridge; one *improve* call if the episode did not reach the tick limit (≤ 2 revisions) | | |

Expected Astra usage over 3 seeds: 2 + (0–2 repairs) + ≤ 2 improves ≈ 4–6 calls (budget 20/episode).

## Offline verification (virtual + wall clock, stand-in models)

Wall clock, tick 0.1 s, 80 ticks, scripted Astra returning a toy greedy bridge:
`snake` 5 and 9 food (vs 0), **0 stale ticks**, S2 = 3 over 2 seeds; the repair path recovers from
"bad code → reply without code → good code"; `lanerunner` adopts its own bridge (7.6 / 21.8 points).
The first virtual-clock attempt exposed a real design bug (a *background* Jev future can never be
observed by a driver that advances the virtual clock without yielding) — hence the synchronous
post-then-ask scheme above, which is deterministic on both clocks.

## What would confirm / refute

* **Confirm:** mean score ≫ 0 with stale fraction near 0; trace shows `bridge_adopted` early in seed 0
  and 0–1 Astra calls in seeds 1–2; `decision` events show `options` never containing a fatal move.
* **Refute / next steps:**
  - if Astra's code is rejected repeatedly (`bridge_reject` × 3) → the contract prompt is too
    demanding; simplify to `options` only and let the harness compute utilities from Jev alone;
  - if seed 0 dies during the generic phase before the code lands (Astra > 10–20 s) → invest in the
    generic phase (e.g. ask Astra for a JSON "danger-key / reward-key" digest first, which is a much
    shorter call than code) or reuse guidance across seeds more aggressively;
  - compare `jev_changed` (how often Jev overturned the bridge argmax) against the score: if
    overrides correlate with deaths/lower score, lower `JEV_WEIGHT` or restrict Jev to ties; if they
    help, raise it — this is the one knob that measures whether System-1 adds value over the code.
