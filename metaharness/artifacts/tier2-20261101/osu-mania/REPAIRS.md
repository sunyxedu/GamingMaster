# Repairs

- 2026-10-31 17:35: 0006 and 0008 (generation 2) scored the ScoreV1 maximum 1,000,000 on all six training charts; no
  later generation can raise the training score. Finalized like Malody and Snake: games.flags `--iters 2 --no-reflect`,
  restarted before the generation-2 reflection, so validation, held-out and the no-models ablation run now.
- 2026-10-31 18:18: the first no-models ablation of 0008 (0.0) set System-1 to None, and 0008 returns at once when
  ctx.system1 is None (it reads that as a tier-1 task), so its code was never measured. Kept as
  `candidates/0008/ablation-no-models.system1-none/`; re-run on 77004c8 (silent stand-in models, every call fails).
  Result: 0.0 again, now measured: 0008 plays every chart to the end (game_done 6) but all 351 Jev calls fail and it
  presses nothing, because each scheduled batch of key presses waits for a Jev approval. The champion depends on Jev.
