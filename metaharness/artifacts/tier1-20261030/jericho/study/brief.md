# Jericho interactive fiction: study brief

Research status, 2026-10-30: the evaluator interpreter was checked locally and is Python 3.12.3. Jericho is absent. Fresh attempts to fetch GitHub raw source, Read the Docs, PyPI, IF Archive, Brass Lantern, and Eristic all failed DNS resolution in this sandbox. Consequently, the links below are source references for verification, **not pages successfully read in this session**. The mechanics and API descriptions are based on established Jericho/Z-machine knowledge; release details, licenses, exact supported ROMs, and Python 3.12 behavior remain unverified. No game-playing code has been written. See `interfaces.md` for a concrete implementation and acceptance plan and `research-log.md` for local evidence.

## What this game is

Jericho is a Python research environment around a modified Frotz Z-machine interpreter, rather than one adventure with universal rules. It runs supported compiled interactive-fiction stories. Zork I, II, and III are separate games; other stories differ in parser vocabulary, maximum score, hazards, victory conditions, and even whether scoring exists. Use a fixed manifest of individually approved story binaries. Do not mistake support for many Z-machine files for full instrumentation support for every release of every story. [J1–J4]

The player reads prose, types a command, and receives the narrator's response. Rooms form a graph with directional exits; movable objects can be carried, manipulated, combined, or given to characters. Puzzles impose dependencies: obtain a key before unlocking a door, find illumination before entering darkness, learn a word before casting a spell. The world is only partially observed, and text can omit relevant state. Parser errors, ambiguity questions, scripted events, and changing room descriptions all matter. [J5, I1, Z1]

## Objective and score

The evaluation objective is **normalized in-game score**, as requested: final native score divided by the verified positive maximum for that exact story build. Report raw final score, maximum, normalized score, win/loss, and truncation separately. Do not replace this with exploration bonuses, the sum of only positive rewards, a language-model estimate, or the highest score ever seen. Death can cost points; final score must reflect that. A successful ending and maximum score are not necessarily equivalent. `interfaces.md` specifies edge cases and aggregation. [J2, J5]

Zork I illustrates the genre: find treasures, solve obstacles, and put treasures in the living-room trophy case. Discovering/acquiring a treasure and depositing it can contribute different points; merely carrying everything does not finish the task. Its familiar maximum is 350 points, but the manifest must verify the exact loaded release rather than assuming all games share this scale. Light lifetime, carrying capacity, hostile creatures, and irreversible mistakes constrain routes. The 350 figure is a known Zork I reference value, not a locally tested ROM measurement. [Z1–Z3]

## Commands and parser behavior

| Purpose | Typical inputs | Caveat |
|---|---|---|
| Movement | `north`, `n`, `south`, `east`, `west`, `up`, `down`, `in`, `out` | Exits may be one-way, conditional, or non-geometric. Never infer the reverse exit is valid. |
| Observe | `look`, `inventory`, `examine lamp`, `read leaflet` | Abbreviations such as `l`, `i`, and `x` depend on the parser; even observation commands must not be assumed turn-free across stories. |
| Manipulate | `take lamp`, `drop rope`, `open window`, `unlock door with key`, `put jewel in case` | Use names and verbs from the game's prose. Containers and supporters differ. |
| Operate/fight | `turn on lamp`, `tie rope to railing`, `attack troll with sword` | Preconditions and wording are game-specific. These can consume scarce turns or provoke death. |
| Communicate | `ask person about topic`, `give item to person`, a quoted word | Conversation grammars vary. An answer to a clarification prompt may be just a noun. |
| Wait | `wait` or often `z` | Explicitly advances game time; waiting for the model does not. |
| Meta | `score`, `save`, `restore`, `restart`, `quit`, sometimes `undo` | Evaluator should control persistence and lifecycle; do not expose reset/restore/undo as ordinary scored player actions. |

Submit one parser input per step. A compound line such as `n.then open door` can execute several turns or behave differently between parsers. Preserve clarification prompts: after “Which key do you mean?” the next input should answer that question rather than launch a fresh plan. [I1, Z1]

## Pace and tier

**Tier 1 is correct for the selected untimed stories.** Simulation normally advances when input is processed, not while the player thinks. Combat, lamp drain, flooding, NPC movement, and timed puzzles are usually measured in game turns. Astra may deliberate for seconds or a minute without taking an in-game turn. Repeated inspection, failed actions, or `wait` may still advance game time depending on the story. [J2, I1, Z1]

The Z-machine specification does allow timed input, so “all interactive fiction waits forever” is too broad. Exclude stories using wall-clock timed input from this Tier 1 pool, or prove the selected interpreter handles them in a deterministic, turn-based way. Evaluation service wall-clock limits are separate from game mechanics and were not supplied. [V1]

There is **no finite game-imposed decision deadline** for an admitted untimed game. In `study.json`, `decision_latency_s = -1.0` is an explicitly documented sentinel meaning unbounded, not a negative actual timeout. If a downstream schema requires a nonnegative number, the adaptor owner must define an unbounded convention; do not silently invent a 60-second game deadline.

## Sources to verify

- [J1] Jericho project and installation overview: https://github.com/microsoft/jericho
- [J2] Jericho API documentation: https://jericho-py.readthedocs.io/en/latest/api.html
- [J3] Python implementation, exact signatures and bindings: https://github.com/microsoft/jericho/blob/master/jericho/jericho.py
- [J4] Game-specific data/support: https://github.com/microsoft/jericho/tree/master/jericho
- [J5] Hausknecht et al., *Interactive Fiction Games: A Colossal Adventure* (Jericho paper): https://arxiv.org/abs/1909.05398
- [I1] Brass Lantern, *A Beginner's Guide to Playing Interactive Fiction*: https://brasslantern.org/beginners/playif.html
- [Z1] Infocom documentation archive; Zork manuals and reference cards: https://infodoc.plover.net/
- [Z2] Zork I community hints (spoilers; verify wording against selected release): https://www.eristic.net/games/infocom/zork1.html
- [Z3] Zork I source reference (source availability alone does not establish redistribution permission): https://github.com/historicalsource/zork1
- [V1] Z-machine standard, input/output and timed input: https://inform-fiction.org/zmachine/standards/z1point1/sect15.html
