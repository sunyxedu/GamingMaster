# Training Guide

Two stages: imitation learning (IL) from human replays, then PPO self-play
fine-tuning (RL). This document covers the invariants that make training
*correct* — several of them were the subject of a full audit and are easy to
break again if you don't know they exist.

## The one rule that matters most

**The policy that acts must be the policy that is trained.** In this codebase
that means three things must match between data collection and gradient
computation:

1. **TRM pondering depth** (`act_steps`). The TRM can refine its answer over
   multiple ACT steps with a carry. Depth changes the policy, so:
   - `train_rl.py` uses a **fixed** depth `--act_steps` (default 1) for both
     the rollout (`step_rl(act_steps=K)`) and the PPO update
     (`forward(act_steps=K)`) and the IL auxiliary update. Never let these
     diverge — the PPO ratio becomes meaningless (this was the single worst
     bug in the original implementation).
   - `train_il.py` samples a random depth in `[1, --max_act_steps]` per batch
     (deep-supervision style), so that inference-time carry states are
     in-distribution and `q_halt` is trained at every depth.
   - Evaluation scripts (`play_model.py`, `simulate_model.py`) take
     `--act_steps`; when evaluating an RL checkpoint, pass the same value the
     checkpoint was trained with. Without it, the q_halt-gated adaptive loop
     runs instead (fine for IL checkpoints, wrong for RL ones — RL never
     updates `q_halt`).

2. **Train/eval mode.** Rollouts run in eval mode, so the PPO update also
   runs in eval mode (`ppo_update` calls `model.eval()`): dropout off,
   BatchNorm running statistics. Gradients flow fine in eval mode. If you
   ever switch the update to train mode, old and new log-probs come from
   different networks and the ratio is invalid again.

3. **Recurrent state.** The spatiotemporal KV cache (and GRU hidden, if you
   use `temporal_type="gru"`) is threaded through the whole rollout. The
   equivalence `step_rl(act_steps=K)` ≡ `forward(act_steps=K)` is asserted by
   `tests/test_audit_fixes.py::test_step_vs_full_consistency_act_steps` and
   should hold to ~1e-6 in the ratio.

## Imitation learning

```bash
uv run python train_il.py --data_dir data/output/training/<player> \
    --batch_size 4 --tf_start 0.4 --tf_end 0.0
```

Loss components (`model/losses.py::ImitationLoss`):

| Component | Weight | Mask | Notes |
|-----------|--------|------|-------|
| Source focal CE | 3.0 | `count > 0` | focal γ=1.5 over 625 positions |
| Direction CE | 1.5 | `count > 0` ∧ conditioned-source correct | see below |
| Split CE | 0.3 | same as direction | 0 = all-but-1, 1 = 50% |
| Pass BCE | 1.0 | `count >= 0` | `count == -1` frames are excluded (see data doc) |
| Value MSE | 0.2 | sequence | target ∈ [−1, 1] (heuristic ⊕ outcome blend) |
| Q-halt BCE | 0.2 | `count >= 0` | target = "is the full answer correct at this ACT depth" (whole move on move frames, correct pass on pass frames — paper-faithful) |

**Direction/split conditioning.** The direction and split heads are
conditioned on the embedding of the source tile the network actually used:
the ground-truth source on teacher-forced steps, a sampled source otherwise.
Their labels (the human's direction for the *human's* source) are only
meaningful when the conditioning source equals the ground truth, so those
loss terms are weighted by the indicator `pred["src_idx"] == gt_src`. With
`--tf_start 0` an untrained model rarely samples the right source, so
direction/split receive little signal at first — start with `--tf_start 0.4`
(decayed to 0) as in `train_job.sh`.

**Padded frames and BatchNorm.** Variable-length sequences are zero-padded;
`frame_mask` keeps padded frames out of the spatial encoder so they cannot
pollute BatchNorm statistics.

**Distributed.** Standard DDP via `torchrun`. An OOM on one rank raises
instead of skipping — a rank-local skip would desynchronize gradient
allreduce and silently corrupt every later step of the epoch.

## RL self-play

```bash
uv run python train_rl.py \
    --il_checkpoint checkpoints/checkpoint_epochN.pt \
    --il_data_dir data/output/training/<player> \
    --act_steps 1
```

Per iteration: collect `--episodes_per_gpu` self-play episodes against an
opponent sampled from the pool → normalize advantages globally (across
buffers and ranks) → `--ppo_epochs` PPO updates with gradient accumulation →
one IL auxiliary step (policy heads only; its value term is disabled so it
cannot fight the PPO critic) → decay schedules.

Reward design (`env/generals_env.py`):

| Term | Value | Notes |
|------|-------|-------|
| Win / loss | ±1.0 | turn-limit truncation gives neither, and bootstraps |
| Dense shaping | 0.6 × Δheuristic × scale | potential-based, telescopes over cycles |
| Enemy city capture / loss | +0.16 / −0.16 | symmetric — city trading is zero-sum |
| Neutral city capture | +0.04 … +0.12 | efficiency bonus |
| Invalid move | −0.05 | RL runs without a legal mask; paired with the invalidity aux loss |

There is deliberately **no pass penalty** (it was trivially dodged by
shuffling armies between owned tiles) and **no failed-city-attack penalty**
(multi-hit sieges are normal play). The `ValueHead` is unbounded — discounted
returns exceed [−1, 1] and a tanh critic saturates on them.

**Truncation vs termination.** `GeneralsEnv(max_turns=...)` truncates with
`info["truncated"]=True`; `collect_selfplay_episode` then bootstraps the last
value instead of treating the state as terminal (and stores `done=False` for
GAE). The same logic covers `--rollout_steps` running out.

**Resume.** `--resume` restores weights *before* the KL reference model and
opponent pool are captured, recomputes every decayed quantity
(LR, entropy weight, reward scales) from the iteration count, keeps the
opponent pool on CPU, and survives resuming across the encoder-unfreeze
boundary (optimizer state is skipped with a warning when the trainable set
changed). Checkpoint loading is strict: missing keys raise instead of leaving
modules randomly initialized.

**Distributed.** `train_rl.py` does not use DDP (manual gradient all-reduce
instead); rank 0's parameters and buffers are broadcast at startup so all
ranks start identical.

## Older checkpoints

Checkpoints trained before the 2026-07 audit fixes load cleanly (no key
changes), but two head semantics changed underneath them:

- `q_halt` is now read from the HL decision token instead of the board's
  top-left tile — an old checkpoint's q-gated ACT loop is uncalibrated.
  Evaluate such checkpoints with a fixed `--act_steps`.
- `ValueHead` lost its tanh — old value outputs will no longer be squashed.

Both recalibrate quickly under continued training; just don't trust their
raw q_halt/value outputs at inference before that.
