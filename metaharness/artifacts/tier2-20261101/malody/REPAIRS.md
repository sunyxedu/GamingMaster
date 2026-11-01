# Repairs

- 2026-10-31 12:25: all three generation-1 candidates (0003-0005) scored the maximum 1.0 on every training seed
  (weighted accuracy 1.0, completion 1.0, 0 misses, full combo); no later generation can raise the training score.
  Finalized like Flappy Bird: games.flags `--iters 1 --no-reflect`, restarted during the generation-1 reflection, so
  validation, held-out and the no-models ablation run now and the agent slots go to games with headroom.
