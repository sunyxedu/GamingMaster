# TRM recursive dynamics in TWISTER

Experiment: replace the next-state predictor of stock TWISTER (a 4-block causal
transformer) with a Tiny Recursive Model (TRM) loop, and compare on Atari-100k.
Two rounds so far. Round 1 has finished. Round 2 is running.

## Layout of this directory

| Path | Content |
|---|---|
| `README.md` | This file. |
| `recursive_dynamics.py` | The `RecursiveDynamics` module. Copy to `nnet/modules/twister/` in a TWISTER checkout. |
| `test_recursive_dynamics.py`, `test_trm_fidelity.py` | Tests. Copy next to the module. |
| `patches/stock_twister.patch` | Changes to stock TWISTER files. Apply with `git apply` at commit `e6d8cde` of `burchim/TWISTER`. |
| `slurm/` | The SLURM arrays used for the runs. |

## The part we replaced

In stock TWISTER the dynamics model predicts the next latent code from the past
codes and actions:

```
x_t = mix(z_{t-1}, a_{t-1})
x_t -> attention block 1 -> block 2 -> block 3 -> block 4 -> deter_t -> next code
```

- Each block is a causal transformer block: self-attention over the last 8 steps, then feed-forward.
- Each block keeps a KV cache of its own past outputs. So block 3 at step t can look at block 3 outputs from steps t-1 to t-8. Processed history is available at every depth.
- 4 blocks, 9,465,856 parameters.

Everything else (encoder, decoder, contrastive loss, actor, critic, hyperparameters) is stock in all runs.

## The three architectures, side by side

| | Stock | Round 1: MLP loop | Round 2: attention loop |
|---|---|---|---|
| Temporal mixing (looks at past steps) | 4 attention blocks, stacked | 1 attention block, runs once, before the loop | 1 attention block, inside the loop |
| What is looped | nothing | 2 MLP layers | the attention block |
| Calls of the looped part per step | 0 | 15 (H=3, L=4) | 15 (H=3, L=4) or 8 (H=2, L=3) |
| Caches of past steps | 4, one per block | 1, before the loop | 15 or 8, one per call |
| Can later steps see the loop's output? | not applicable | no, only through the sampled code | yes, through the per-call caches |
| Dynamics parameters | 9,465,856 | 4,469,760 (47%) | 2,367,488 (25%) |
| Status | baseline | done | running |

The TRM loop is the same in both rounds:

```
z_H, z_L = fixed start values
repeat H times:
    repeat L times:
        z_L = net(z_L, z_H + x_t)
    z_H = net(z_H, z_L)
deter_t = LayerNorm(z_H)
```

`net(z, inj)` adds `inj` to `z`, then runs the network. The same weights are used in every call. Only the last outer cycle keeps gradients.

## Round 1: MLP loop (done)

### Architecture

1. `x_t = mix(z_{t-1}, a_{t-1})` (stock action mixer).
2. One attention block looks at the last 8 steps. Output `c_t`. This runs once. It is the only part that sees the past. It plays the role of the input `x` in the TRM paper.
3. The loop above, with `c_t` as the injected input. `net` is 2 MLP layers: LayerNorm, Linear 512 to 1024, SiLU, Linear 1024 to 512, with a residual connection. No attention inside the loop. Each timestep is refined alone.
4. `deter_t = LayerNorm(z_H)`. The stock head predicts the next code.

| Part | Layers | In the loop? | Parameters |
|---|---|---|---|
| Attention block | 1 | No, runs once | 2.4M |
| `net` | 2 MLP layers | Yes, 15 calls | 2.1M |

This is the MLP-only variant of the TRM paper. Config: `dynamics_core = "trm"`, `trm_loop = "mlp"`, `trm_H_cycles = 3`, `trm_L_cycles = 4`, `trm_reason_layers = 2`, `trm_reason_ff_ratio = 2`.

### What ran

| Item | Value |
|---|---|
| Arms | `transformer` (stock) and `trm` (MLP loop). Same seed in each pair. |
| Games, seeds | Ms. Pac-Man seeds 0, 1, 2. Breakout seed 0. |
| Benchmark | Atari-100k. 50 epochs of 2,000 gradient steps. 100k agent steps. |
| Model size | Stock "S". Hidden 512. Context window 8 steps. Batch 16, sequence 64. |
| Evaluation | Every 5 epochs. Mean score over 100 episodes. |
| Precision, hardware | float32. One A100 80GB per run. 24 hour limit. |
| Jobs | 2365373 (seed 0), 2366485 (seeds 1, 2) |

Speed: stock 2.19 to 2.24 it/s. MLP loop 2.11 to 2.13 it/s.

### Results

"Last-5" is the mean of the last five evaluations. "Final" is the epoch 50 score.

| Game | Seed | Stock last-5 (final) | MLP loop last-5 (final) | Loop / stock |
|---|---|---|---|---|
| Ms. Pac-Man | 0 | 3,331 (4,313) | 1,921 (1,937) | 58% |
| Ms. Pac-Man | 1 | 1,738 (1,888) | 1,625 (1,598) | 93% |
| Ms. Pac-Man | 2 | 1,903 at epoch 30 | 1,823 at epoch 25 | running |
| Breakout | 0 | 20.4 (25.5) | 1.3 (0.8 at epoch 20) | 6% |

All evaluation scores, epochs 5 to 50:

```
Ms. Pac-Man seed 0   stock: 1071  645 1552 1757 2799 2027 3454 3223 3636 4313
                     loop : 1016  973 1292 1830 1799 1941 1965 1901 1861 1937
Ms. Pac-Man seed 1   stock:  659 1120 1107 1325 1338 1725 1656 1701 1721 1888
                     loop :  778 1254 1310 1308 1422 1626 1680 1650 1572 1598
Ms. Pac-Man seed 2   stock:  726 1622 1220 1716 2013 1903   (running)
                     loop :  806  964 1163 1118 1823
Breakout seed 0      stock:  2.1  6.1  8.8 14.9 15.9 12.4 22.5 22.2 19.4 25.5
                     loop :  1.4  1.1  1.8  0.8   (24 hour limit at epoch 25)
```

Findings:

- Ms. Pac-Man: both arms are equal for about 25 epochs. Then the stock model keeps improving and the loop stops at about 1,600 to 1,950.
- Breakout: the loop agent scores about 1 for the whole run. It does not learn to hit the ball. Its episodes are very long, so evaluations took hours and the run hit the 24 hour limit.
- Cause: the loop cannot see processed history. Only one attention block looks at the past, and the loop output is not stored for later steps. Ball speed and direction need two processed past frames. The model cannot build them.

## Round 2: attention loop (running)

### Architecture

Same loop, different `net`, and no block before the loop:

1. `x_t = mix(z_{t-1}, a_{t-1})`.
2. The loop above, with `x_t` injected directly, as in the TRM paper. `net` is one TWISTER attention block: causal self-attention over the last 8 steps, then feed-forward. This is the HRM block.
3. Every call keeps its own KV cache of the past steps at that loop position. Call 7 at step t can look at what call 7 computed for steps t-1 to t-8. This is the processed history that round 1 could not see.
4. `deter_t = LayerNorm(z_H)`.

| Part | Layers | In the loop? | Parameters |
|---|---|---|---|
| `net` | 1 attention block | Yes, 15 or 8 calls | 2.37M |

Config: `dynamics_core = "trm"`, `trm_loop = "attention"`, `trm_loop_blocks = 1`. Arm `attn15`: `trm_H_cycles = 3`, `trm_L_cycles = 4`. Arm `attn8`: `trm_H_cycles = 2`, `trm_L_cycles = 3`.

We did not carry `z_H` and `z_L` from one timestep to the next. In HRM the carry goes across steps on the same input. Across timesteps it would make training sequential over the 64-step sequence, about 60 times slower. The per-call caches already pass processed history forward.

### What is running

| Item | Value |
|---|---|
| Arms | `attn15`, `attn8`, and `transformer` (stock) for the missing Breakout seed 1 |
| Games, seeds | Ms. Pac-Man and Breakout, seeds 0 and 1 |
| Runs | 9 |
| Job | 2369891 |
| Everything else | As in round 1 |

Speed: `attn15` 1.47 it/s, `attn8` 1.87 it/s (stock 2.2). At 1.47 it/s a Breakout run needs about 25 hours, so the two `attn15` Breakout runs have a dependent resume job (`slurm/isca_trm_resume.slurm`, `--load_last`) that continues from the last checkpoint after the 24 hour limit.

Results will be added when the runs finish. They compare against the round 1 stock scores for the same game and seed.

## How the loop is trained (both rounds)

1. Only the last outer cycle keeps gradients. The first H-1 outer cycles run under `torch.no_grad()`. Their outputs enter the last cycle as constants. This is the TRM one-step gradient rule.
2. No weights are frozen during training. All weights train for the full run.
3. Gradients reach the parts before the loop (round 1: the attention block; both rounds: the action mixer and the encoder) through the injected input in the last cycle.
4. The start values `z_H` and `z_L` are fixed random buffers. They are not trained.
5. Losses are the stock TWISTER losses. The actor and critic train inside imagination, through the loop.
6. Not used yet: halting (ACT), training at random depths, carrying `z_H` and `z_L` across timesteps.

## Tests

All pass on the laptop and on the cluster, for both loop types.

- The module output is bit-exact equal to a plain copy of the paper loop, with the same weights.
- The loop makes exactly H*L + H calls per step, and keeps one cache per call (round 2).
- A full sequence gives the same output as the same sequence fed in chunks or one step at a time, through the caches.
- Every parameter and the input get a gradient. The start buffers do not.
- The stock arm has exactly the stock parameter names. No loop parts leak in.

## Codebase changes

Relative to stock TWISTER, commit `e6d8cde`.

| File | Change |
|---|---|
| `nnet/modules/twister/recursive_dynamics.py` | New. `RecursiveDynamics` with two loop types: `mlp` (round 1) and `attention` (round 2). Same interface as `TransformerNetwork`. |
| `nnet/modules/twister/tssm.py` | Builds `RecursiveDynamics` when `dynamics_core == "trm"`. Stock path unchanged. |
| `nnet/models/twister.py` | Config keys `dynamics_core`, `trm_loop`, `trm_loop_blocks`, `trm_context_blocks`, `trm_H_cycles`, `trm_L_cycles`, `trm_reason_layers`, `trm_reason_ff_ratio`. Passed to the TSSM. |
| `main.py` | `--seed` argument. |
| `nnet/envs/__init__.py` | `dm_control` import is optional. |
| `nnet/envs/atari/atari_env.py` | `gym 0.19` and `atari-py` replaced by `gymnasium` and `ale-py`. Same wrapper settings. All arms use it. |
| `nnet/modules/twister/test_recursive_dynamics.py`, `test_trm_fidelity.py` | Tests listed above. |
| `experiments/isca_trm_pilot.slurm`, `isca_trm_seeds.slurm`, `isca_trm_loop.slurm`, `isca_trm_resume.slurm` | SLURM arrays for round 1 (first two) and round 2, and the resume job. |

## How to run

```bash
# round 1 arm (MLP loop)
env_name=atari100k-ms_pacman run_name=twister_trm \
  override_config='{"dynamics_core":"trm"}' python main.py --wandb --seed 0

# round 2 arm (attention loop, 15 calls)
env_name=atari100k-ms_pacman run_name=twister_attn15 \
  override_config='{"dynamics_core":"trm","trm_loop":"attention"}' python main.py --wandb --seed 0

# stock arm: omit override_config

# tests
python nnet/modules/twister/test_recursive_dynamics.py
python nnet/modules/twister/test_trm_fidelity.py
```
