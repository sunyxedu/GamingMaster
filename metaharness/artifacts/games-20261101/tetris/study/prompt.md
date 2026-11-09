# Study the game: **Tetris**

Working directory: `/home/hemlix/GamingMaster-mh-tier2/runs/tetris/study` (write everything here). The evaluator's Python is `/home/hemlix/GamingMaster-mh-tier2/runs/tetris/.venv/bin/python`
(Python 3.12.3); judge software requirements against it, not the system python.
User notes (platform, account, constraints — may be empty): Offline headless Linux server (no display, no GPU, no accounts). The evaluator's Python is the venv printed in the prompt; pip works there. Use a faithful local implementation or environment of the classic falling-blocks game with real-time gravity, exposing the board state symbolically.

We are building an agent for this game out of two foundation models plus a generated harness:
* Astra — a slow, strong reasoning model (seconds to a minute per call). Plans, explains rules, writes code.
* Jev — a fast (~0.5 s), cheap model that answers *typed* questions (choice / score / yes-no) about a
  JSON state, many independent questions per call. Recognises situations; does not plan or do arithmetic.
Games split into tiers: **tier 1** — the world waits (turn-based, puzzles, most card games): Astra plays
through a harness. **tier 2** — the world keeps moving (action, real-time strategy with seconds of slack):
Astra thinks while Jev acts. The user says the game is tier 2; verify and keep it unless clearly wrong.

## Deliverables (all in this directory)
1. `brief.md` — rules, objective, how score/progress is measured, controls/actions, pace: how often the
   game state changes and how long a decision may take without penalty. Cite sources.
2. `interfaces.md` — every practical way to observe and control this game programmatically (official
   API or environment, mods/plugins, bot libraries, emulator hooks, screen capture + input injection...).
   For each: maturity, latency, what it exposes, legal/ToS risk, cost. Pick one and say why. List the
   software to install and the credentials / environment variables (server address, account, API key)
   the user must provide. Never write credential values into files.
3. `strategies.md` — what strong players do: openings/build orders/rotations, heuristics, common
   mistakes, known counters. For each, say whether it belongs in *code* (exact rules, search, bookkeeping),
   in *Astra prompts* (planning, adaptation), or in *Jev questions* (fast recognition/choice). Cite sources.
4. `study.json` — exactly this schema (JSON, no comments):
```
{
  "game": "<canonical game name>",
  "tier": 1 | 2,                       // 1: the world waits for the agent; 2: it ticks while we think
  "tier_rationale": "<one paragraph: how fast decisions must be and why>",
  "decision_latency_s": <float>,       // how long a decision may take before the game punishes it
  "interface": {
    "chosen": "<api | mod | bot library | emulator | screen+input | official env ...>",
    "why": "<why this beats the alternatives>",
    "software": ["<package or program to install>", ...],
    "env": [{"var": "ENV_VAR_NAME", "description": "<what to put there>", "required": true}],
    "setup_steps": ["<step the adaptor phase must perform>", ...],
    "observation": "<what the adaptor can observe and in what form (json state / text / screenshot)>",
    "actions": "<what the adaptor can do and how>"
  },
  "scoring": "<what the adaptor should report as score and why it reflects playing well>",
  "scenarios": "<how reset(seed) should vary starts / levels / opponents so held-out seeds are meaningful>",
  "strategy_summary": "<3-8 sentences: how strong players play and what to implement>",
  "risks": ["<anything that can block the adaptor: anti-cheat, ToS, cost, latency, GUI-only>", ...]
}
```
Spend most effort on 2 and 3: the next phase must be able to build a working adaptor from `interfaces.md`
alone, and the harness search will read `strategies.md` as its prior knowledge.
Time: about 90 minutes. Write a first complete version of all four files within the first third of it, then deepen them: the files on disk are what the next phase gets.