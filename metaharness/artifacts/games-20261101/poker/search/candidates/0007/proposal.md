Append these bullets to `notes.md`:

- Candidate 0005's cap was not sufficient: its “private nuts” comparison used a partial four-card board and allowed a 200bb river shove with a non-nut flush.
- Action-conditioned posterior equity and uniform-card showdown equity must not be multiplied into precise fold equity; both can be badly wrong on coordinated rivers.
- A river response rule needs explicit board-danger and uncertainty margins; one-pair or non-nut flush bluff-catchers should not inherit a generic equity threshold.
- Legal candidate validation and `ctx.progress()` recovery are essential because a fallback or model override can otherwise reintroduce an illegal or stale action.
- Model calls are optional in tier 1; a cheap policy that completes all 100 hands can beat a slower policy whose provisional score is dominated by deadline truncation.