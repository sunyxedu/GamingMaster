# Known Limitations

Deliberate trade-offs and unfixed issues, so nobody rediscovers them the
hard way.

## Within-tick move priority is a heuristic (deliberate)

The replay simulator and the RL env order same-tick moves with a
chase/defense/army-size heuristic instead of reproducing the server's exact
(historically changing) priority rule. Decision: not worth chasing — the rule
only settles same-tick races, and self-play RL adapts to whatever convention
it trains under. See `docs/data-pipeline.md`.

## KV cache memory grows O(T) with O(T²) copying

`SpatiotemporalBlock.forward_step` re-concatenates the temporal cache every
step (`torch.cat`), so a 1000-step self-play episode costs roughly
`2 players × n_st_layers × [627, T, d]` floats (a few GB at the default
model size) plus quadratic copy traffic. Fine on training GPUs at
`rollout_steps=1000`; will hurt for much longer games. A ring/doubling
buffer would fix the copying if it ever matters.

## q_halt-gated ACT is only meaningful for IL checkpoints

`q_halt` is trained during IL (BCE against source-argmax correctness at the
sampled ACT depth) but **not** during RL (RL uses a fixed `--act_steps` and
has no q_halt objective). Adaptive pondering at inference is therefore only
calibrated for IL checkpoints; evaluate RL checkpoints at their fixed
training depth. Checkpoints from before the 2026-07 audit have an
uncalibrated q_halt entirely (it used to read a board-corner token).

## Modern map features are unsupported

Swamps, deserts, lookouts, observatories, and game modifiers are neither
simulated nor learned — such replays are rejected during data prep, and the
RL env never generates them. The bot targets classic 1v1 maps.

## Miscellaneous

- The RL env's random maps place mountains/cities uniformly (with a
  connectivity check and 50% side swap); they don't reproduce the real
  generator's fairness constraints (mirrored distances, city fairness).
- `train_rl.py` trains player 0 only per episode; opponents come from a
  frozen pool, so there is no gradient through the opponent's seat.
- The world-model objective (`predict_next_obs`, `--world_model_weight`)
  trains only during IL; RL uses shaped environment rewards with no
  intrinsic-curiosity term.
