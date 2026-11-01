# Repairs

- 2026-10-31 16:10: 0020 (generation 6) filled the 12x12 board on all six training seeds (141 food, length 144,
  no collision); no later generation can raise the training score. Finalized like Malody: games.flags
  `--iters 6 --no-reflect`, restarted before the generation-6 reflection, so validation, held-out and the
  no-models ablation run now.
