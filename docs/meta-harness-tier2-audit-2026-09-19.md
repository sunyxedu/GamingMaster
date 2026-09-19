# Tier 2 live audit — 2026-09-19

**The outer loop improved cold-start Snake performance, but transfer remains
poor and the runtime Astra benefit is small on this benchmark.** All 48 episodes
completed without evaluator errors. Both proposer and slow player were real
`gpt-6-astra` at high effort; the fast model resolved to `jev-1.13.0`.

The [complete archive](../metaharness/artifacts/audit-20260919-tier2/) contains
candidate code, proposals, manifests, every evaluation, compressed request/response
and action traces, and checksums. `tool_audit.json` found no external tool calls
in the 31 recorded player sessions. Proposer tool calls are retained separately;
the proposer genuinely read archived failures and rewrote the harness.

## Protocol

Wall clock, no simulated latency; Snake tick 0.5 s, at most 120 ticks, four Astra
and 240 Jev calls per episode. Training seeds 0/1/2; frozen test seeds 700/701/702.
The old Astra champion was re-evaluated as a strong control. Independent episode
memory is the default; the shared-memory condition is explicit. Test results
were never fed back into search. Search and final tests used different runner
revisions as documented in `provenance.json` and `test_provenance.json`; all test
controls used the same final inference restrictions and runner.

## Search

| Candidate | Training mean | Total Jev / Astra calls |
|---|---:|---:|
| Jev-only baseline | 0.00 | 18 / 0 |
| Astra-only baseline | 0.00 | 0 / 3 |
| Prior Astra champion | 8.67 | 180 / 6 |
| New proposal 0003 | 1.33 | 180 / 6 |
| New proposal 0004, selected | **13.00** | **3 / 3** |

Proposal 0003 failed because Jev would not confidently infer numeric action
effects and the generated compiler used positional feature indices that its
executor rejected. Proposal 0004 diagnosed both from traces: it repaired the
feature contract, used Jev for semantic binding, and learned effects from actual
transitions while Astra compiled a reactive policy. It did not receive a
manually supplied game-specific mapping. One training episode died at tick 118;
the others reached 120, so greater reward did not mean uniformly better survival.

## Frozen tests

| Condition | Jev-only baseline | Prior champion | New 0004 |
|---|---:|---:|---:|
| Snake, cold starts | 0.00 | 7.33 | **11.67** |
| Snake, shared episode memory | 0.00 | 11.33 | **11.67** |
| Lane Runner, cold starts | **2.47** | 1.13 | 1.13 |

New 0004's cold scores were **12, 12, 11**, versus **7, 6, 9** for the prior
champion. The shared-memory control exposes the incumbent's large amortization
benefit. The new candidate does not cache its policy across episodes.

Lane Runner episodes for both generated harnesses lasted only **3, 4, 10 ticks**.
The early Snake procedure depends on finding useful safety masks and a goal-error
vector. That procedure did not transfer; Astra's compiled policy generally
arrived after these short episodes ended. A schema-agnostic wrapper alone is
therefore insufficient evidence of game generality.

| Runtime ablation of frozen 0004 | Snake scores | Mean |
|---|---|---:|
| Astra + Jev | 12, 12, 11 | 11.67 |
| Jev, runtime Astra disabled | 12, 11, 11 | 11.33 |
| Astra, runtime Jev disabled | 0, 0, 0 | 0.00 |

The proposer is **still Astra** in both ablations: only runtime access is removed.
The useful mechanism here is largely Astra-designed code plus Jev semantic
calibration. The 0.33-point runtime Astra difference over three seeds does not
establish a meaningful planning benefit. This is not equivalent to the naive
Jev-only baseline, whose harness has no learned transition controller.

These are small symbolic-observation experiments. Snake supplies exact positions
and danger indicators; raw-screen control, Minecraft and Genshin are untested.
The next generality experiment needs a diverse training-game suite, a genuinely
new held-out game, and matched cold-start and model-ablation controls. Lane Runner
has now been exposed and must not be reused as a fresh blind test.

## Reproduction

```bash
uv run python -m scripts.metaharness_audit search --tier 2 --iterations 2 --root /tmp/my-tier2-search
uv run python -m scripts.metaharness_audit test --tier 2 --root /tmp/my-tier2-search
```

Use the existing Codex login and `TYPESAFE_API_KEY` in the gitignored `.env`.
Inference is stochastic; seeds fix the environment, not the model's output.
To re-evaluate the exact frozen program, use `python -m metaharness eval` with
`candidates/0004/harness.py` from the archive, seeds `700,701,702`, game kwargs
`{"tick_s":0.5,"max_ticks":120}`, and the budgets and memory mode above.
