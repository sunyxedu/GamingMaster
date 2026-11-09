# Repairs

- 2026-10-31 04:00: the search (2 generations, best 0008 = 14.5, 0 homes) was moved to `search.pre-review/`.
  Its adaptor predates the general observation review (commit 0d33f4b): the observation carried raw RAM and
  guessed frog fields but no decoded hazards or platforms. The relaunched pipeline reuses study and adaptor,
  runs the review (adaptor/review.md), then starts a fresh search on the reviewed observation.
- 2026-10-31 07:15: the first review (03:57-07:08, pre-1af656a code) was one repeated sentence after a 503 and changed
  nothing; its stub review.md let a search start at 07:08 on the unreviewed observation. That search (baselines only)
  moved to `search.unreviewed-0708/`, the stub removed, and the pipeline relaunched on 2748d89 to redo the review.
