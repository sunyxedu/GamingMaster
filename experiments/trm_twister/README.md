# TRM recursive dynamics in TWISTER

Experiment: replace the next-state predictor of stock TWISTER (a 4-block causal
transformer) with a Tiny Recursive Model (TRM) core, and compare on Atari-100k.

## Layout of this directory

| Path | Content |
|---|---|
| `README.md` | This file. |
| `recursive_dynamics.py` | The `RecursiveDynamics` module. Copy to `nnet/modules/twister/` in a TWISTER checkout. |
| `test_recursive_dynamics.py`, `test_trm_fidelity.py` | Tests. Copy next to the module. |
| `patches/stock_twister.patch` | Changes to stock TWISTER files. Apply with `git apply` at commit `e6d8cde` of `burchim/TWISTER`. |
| `slurm/` | The two SLURM arrays used for the runs. |

## What we ran

| Item | Value |
|---|---|
| Codebase | Stock `github.com/burchim/TWISTER`, plus the changes listed below |
| Benchmark | Atari-100k. 50 epochs of 2,000 gradient steps. 100k agent steps. |
| Model size | Stock "S". Hidden 512. Context window 8 steps. Batch 16, sequence 64. |
| Arms | `transformer` (stock) and `trm` (recursive core). Same seed in each pair. |
| Games, seeds | Ms. Pac-Man seeds 0, 1, 2. Breakout seed 0. |
| Evaluation | Every 5 epochs. Mean score over 100 episodes. |
| Precision, hardware | float32. One A100 80GB per run. 24 hour limit. |

Parameters of the dynamics part: stock 9,465,856. TRM 4,469,760 (47%).
Speed: stock 2.19 to 2.24 it/s. TRM 2.11 to 2.13 it/s.

## Results

"Last-5" is the mean of the last five evaluations. "Final" is the epoch 50 score.

| Game | Seed | Stock last-5 (final) | TRM last-5 (final) | TRM / stock |
|---|---|---|---|---|
| Ms. Pac-Man | 0 | 3,331 (4,313) | 1,921 (1,937) | 58% |
| Ms. Pac-Man | 1 | 1,738 (1,888) | 1,625 (1,598) | 93% |
| Ms. Pac-Man | 2 | 1,903 at epoch 30 | 1,823 at epoch 25 | running |
| Breakout | 0 | 20.4 (25.5) | 1.3 (0.8 at epoch 20) | 6% |

All evaluation scores, epochs 5 to 50:

```
Ms. Pac-Man seed 0   stock: 1071  645 1552 1757 2799 2027 3454 3223 3636 4313
                     trm  : 1016  973 1292 1830 1799 1941 1965 1901 1861 1937
Ms. Pac-Man seed 1   stock:  659 1120 1107 1325 1338 1725 1656 1701 1721 1888
                     trm  :  778 1254 1310 1308 1422 1626 1680 1650 1572 1598
Ms. Pac-Man seed 2   stock:  726 1622 1220 1716 2013 1903   (running)
                     trm  :  806  964 1163 1118 1823
Breakout seed 0      stock:  2.1  6.1  8.8 14.9 15.9 12.4 22.5 22.2 19.4 25.5
                     trm  :  1.4  1.1  1.8  0.8   (24 hour limit at epoch 25)
```

Findings:

- Ms. Pac-Man: both arms are equal for about 25 epochs. Then the stock model keeps improving and the TRM model stops at about 1,600 to 1,950.
- Breakout: the TRM agent scores about 1 for the whole run. It does not learn to hit the ball. Its episodes are very long, so evaluations took hours and the run hit the 24 hour limit.
- Cause: the loop cannot see processed history. Only one attention block looks at the past, and the loop output is not stored for later steps. Ball speed and direction need two processed past frames. The model cannot build them.

Next step: put the attention block inside the loop, so that each loop cycle can look at the previous cycle's history. Control: two attention blocks with the same MLP loop.

## How the TRM core works

At each timestep:

1. Mix the last code and the last action into `x_t` (stock action mixer).
2. One attention block looks at the last 8 steps. Output: `c_t`. This runs once. We call it the context block. It plays the role of the input `x` in the TRM paper.
3. Set two states to fixed start values: `z_H` and `z_L`.
4. Run the loop:
   ```
   repeat 3 times (H):
       repeat 4 times (L):
           z_L = net(z_L, z_H + c_t)
       z_H = net(z_H, z_L)
   ```
5. Output `deter_t = LayerNorm(z_H)`. The stock head predicts the next code from it.

`net` is 2 MLP layers (LayerNorm, Linear 512 to 1024, SiLU, Linear 1024 to 512, residual). It first adds the injection to the state, then runs the 2 layers. The same 2 layers are used in all 15 calls.

| Part | Layers | In the loop? | Parameters |
|---|---|---|---|
| Context block | 1 attention block | No, runs once | 2.4M |
| `net` | 2 MLP layers | Yes, 15 calls | 2.1M |
| Output norm | 1 LayerNorm | No | small |

Only the 2 MLP layers are in the loop. This is the MLP-only variant of the TRM paper. The variant with attention inside the loop was not run yet.

## How it is trained

1. Only the last outer cycle keeps gradients. The first 2 outer cycles run under `torch.no_grad()`. Their outputs enter the last cycle as constants. This is the TRM one-step gradient rule.
2. No weights are frozen during training. All weights train for the full run.
3. Gradients reach the context block and the encoder through `c_t` in the last cycle.
4. The start values `z_H` and `z_L` are fixed random buffers. They are not trained.
5. Losses are the stock TWISTER losses. The actor and critic train inside imagination, through the loop.
6. Not used in this phase: halting (ACT), training at random depths, carrying `z_H` and `z_L` across timesteps.

## Codebase changes

Relative to stock TWISTER.

| File | Change |
|---|---|
| `nnet/modules/twister/recursive_dynamics.py` | New. `RecursiveDynamics` (context block + `ReasoningCore`). Same interface as `TransformerNetwork`. |
| `nnet/modules/twister/tssm.py` | Builds `RecursiveDynamics` when `dynamics_core == "trm"`. Stock path unchanged. |
| `nnet/models/twister.py` | Config keys `dynamics_core`, `trm_context_blocks`, `trm_H_cycles`, `trm_L_cycles`, `trm_reason_layers`, `trm_reason_ff_ratio`. Passed to the TSSM. |
| `main.py` | `--seed` argument. |
| `nnet/envs/__init__.py` | `dm_control` import is optional. |
| `nnet/envs/atari/atari_env.py` | `gym 0.19` and `atari-py` replaced by `gymnasium` and `ale-py`. Same wrapper settings. Both arms use it. |
| `nnet/modules/twister/test_recursive_dynamics.py` | Interface tests: shapes, streaming equality, gradients, parameter count, observe and imagine. |
| `nnet/modules/twister/test_trm_fidelity.py` | Semantics tests: bit-exact match to the paper loop, 15 calls per step, gradient boundary, stock state dict unchanged. |
| `experiments/isca_trm_pilot.slurm`, `experiments/isca_trm_seeds.slurm` | SLURM arrays for the runs above. |

Defaults used in all runs: `trm_context_blocks = 1`, `trm_H_cycles = 3`,
`trm_L_cycles = 4`, `trm_reason_layers = 2`, `trm_reason_ff_ratio = 2`.

## How to run

```bash
# one run
env_name=atari100k-ms_pacman run_name=twister_trm \
  override_config='{"dynamics_core":"trm"}' python main.py --wandb --seed 0

# stock arm: omit override_config

# tests
python nnet/modules/twister/test_recursive_dynamics.py
python nnet/modules/twister/test_trm_fidelity.py
```
