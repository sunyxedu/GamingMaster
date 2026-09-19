Design a harness for a **turn-based, hidden-rule** game (tier 1, ARC-AGI-3 style).

Game: `rulepuzzle` — A 8x8 colour-grid puzzle with 2 levels and a budget of 40 actions per level. You control the white cell. Six actions ACTION1..ACTION6 exist; their effects are NOT given and must be discovered by experiment (they are fixed within an episode). Reaching the yellow cell completes a level. Some colours may be walls, keys, doors, toggles or teleports. Score = levels completed.

Actions:
- ACTION1: unknown — discover by trying
- ACTION2: unknown — discover by trying
- ACTION3: unknown — discover by trying
- ACTION4: unknown — discover by trying
- ACTION5: unknown — discover by trying
- ACTION6: unknown — discover by trying
ACTION5/ACTION6 may be context-dependent. Effects are fixed for the whole episode.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 15
calls per episode; every call may return *several* actions to save budget. `ctx.act(action)` steps the
game and returns the next observation. `ctx.memory` persists across the seeds of one evaluation, so
the harness may carry over what it learned about the action semantics (they are re-drawn per seed —
so store *how to discover*, not the answer).

Evaluation metric: mean levels completed over seeds (primary), then System-2 calls (fewer is better).

What the harness must design: what to store (action→effect hypotheses, visited states, level maps),
what to retrieve and show Astra each turn (full history? a diff? an ASCII map with a legend?), how to
parse Astra's reply into actions robustly, how many actions to commit per call, when to run cheap
scripted exploration instead of asking Astra, and how to detect that a hypothesis was refuted.

**Generality requirement.** The best harness is re-evaluated on a *held-out game* with the same
interface (`game.describe()`, `game.action_spec`, dict observations, `ctx`) but different rules and
different observation keys. Do not hard-code this game's rules or observation fields; discover them
from `game.describe()` / the observation, or ask the System-2 model. Game-specific solvers score 0
on the held-out game.