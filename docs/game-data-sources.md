# Training Data and Environment Infrastructure Survey (2026-07)

**Context**: A game-agnostic policy network (spatiotemporal encoder + TRM pondering core), with an IL (imitation of human/expert play) → PPO self-play RL pipeline, a discrete action space, and existing grid encoder / entity encoder / pointer decoder adapters (see [multi-game.md](multi-game.md)).

---

## Overall Ranking Table

| Rank | Game | IL Data | RL Environment | Overall | Biggest Bottleneck |
|---|---|---|---|---|---|
| 1 | **Chess** | S (CC0, ~8 billion games) | S (millisecond-level headless) | **S** | Almost none — minimal engineering effort |
| 2 | **Go** | A (OGS 56M + Tygem 1.5M high-dan) | S (PettingZoo/OpenSpiel/KataGo) | **S-** | High-quality data licensing murkier than Lichess |
| 3 | **Lux AI (Kaggle S2/S3)** | A (full step-by-step state of Kaggle episodes) | S (official JAX GPU parallel engine) | **A** | Imitation target is a top bot rather than humans |
| 4 | **Othello** | B (WTHOR ~120K games) | S (rules trivial, Edax/eGaroucid usable as sparring partners and evaluators) | **A-** | Small data volume (but small board, well-suited to validating the pipeline) |
| 5 | **Shogi** | B+ (floodgate ~110K games/year, engine games) | A (cshogi/python-shogi + USI engines) | **A-** | No large public repository of human professional game records (copyright) |
| 6 | **StarCraft II** | A- (Blizzard official replay pack, >1M games, includes ML license) | A- (PySC2 headless, faster than real-time) | **B+** | Client/replay stuck at 4.9/4.10 (2019); games are long, compute cost extremely high |
| 7 | **Poker (NLHE)** | B- (ACPC hundreds of millions of hands are machine games; human hand histories restricted by ToS) | A (RLCard/OpenSpiel, extremely fast) | **B-** | Public "human" data almost nonexistent; imperfect information better suited to CFR-type methods |
| 8 | **Hearthstone** | C (no public bulk game data) | B- (Fireplace still active but incomplete card coverage; SabberStone unmaintained since 2022) | **C+** | IL data can only be self-collected (Deck Tracker gray area); simulator rule fidelity questionable |
| 9 | **Clash Royale** | D (official API has only post-match metadata, no frame-by-frame) | D (no precise rules simulator; can only run live on an emulator in real time) | **D** | Frame-level imitation data does not exist; self-play cannot be headless-accelerated |
| — | **generals.io** (baseline) | A | A | Done | — |

**Recommended conclusion**: The immediately actionable choices are **Chess and Go** (data, environment, and evaluator all off-the-shelf; Maia has already proven the feasibility of IL on Lichess data); **Lux AI** is the best new "non-board-game, grid+entity hybrid" target, with extremely high affinity to the grid/entity adapters; SC2 is doable but requires accepting high compute costs and being locked to an old version; Hearthstone and Clash Royale are **not recommended** with the current infrastructure (the former lacks data, the latter lacks both data and environment).

---

## 1. Chess — Overall S

**Human Game Data (S)**
- [Lichess open database](https://database.lichess.org/): **~7.95 billion standard rated games**, packaged monthly as PGN (.pgn.zst, ~20-30GB per monthly pack, >2TB compressed in total), **CC0 license, explicitly permitting commercial and research use**. Includes clock information from April 2017 onward, plus 390M+ Stockfish-evaluated positions (jsonl). Variants (Chess960, Crazyhouse, Atomic, etc., ~30M games each) are also fully open.
- Curated subset: [Lichess Elite Database](https://database.nikonoel.fr/) (2400+ curated games).
- PGN is a complete move sequence with deterministic rules → use [python-chess](https://github.com/niklasf/python-chess) to reconstruct the full state at each turn step by step, **no need to write a custom simulator**.
- Precedent: [Maia Chess](https://www.maiachess.com/) used 100M+ Lichess games for pure IL, achieving 53% human move prediction; [Maia-2 (NeurIPS 2024)](https://arxiv.org/abs/2409.20553) covers all skill levels with a single model — directly validating that the IL stage is feasible.

**API/Environment (S)**
- The [Lichess API](https://lichess.org/api) has an official bot interface (can be deployed for live testing).
- RL environments: [PettingZoo classic `chess_v6`](https://pettingzoo.farama.org/) (AEC + legal action mask), OpenSpiel; python-chess is itself a precise rules engine, headless, tens of thousands of steps per second, with self-play far faster than real-time.
- Evaluation: Stockfish (free, UCI) for skill calibration and move-quality analysis.

**Affinity**: The 8×8 grid encoder is a natural fit; a from-square/to-square dual pointer decoder (already implemented as `FromToDecoder`) or an AlphaZero-style 8×8×73 action plane are both mature options.
**Bottleneck**: No substantive bottleneck; it is the cheapest candidate in engineering terms.

---

## 2. Go — Overall S-

**Human Game Data (A)**
- [OGS full dump](https://forums.online-go.com/t/can-we-get-an-sgf-database-dump/38837): **~56 million games** (SGF/JSON, ~11GB compressed, mirrored on the Internet Archive); a summary is collected in [yenw/computer-go-dataset](https://github.com/yenw/computer-go-dataset).
- High-dan data: Tygem **1,516,031 games of 9D vs 9D** (2005–2016), 50K TOM 9D games, 73K professional game records (1940–2017), [KGS high-dan game records (u-go.net)](https://www.u-go.net/gamerecords/). Fox has no official dump.
- Engine data: [KataGo distributed training](https://katagotraining.org/) publishes **95M+ self-play games** of training data (usable for pre-training or distillation); KataGo itself is MIT-licensed.
- SGF is a pure move sequence, and state reconstruction is trivial. Licensing note: OGS/KGS data is "usable by community convention," with **no CC0 statement like Lichess's**, so commercial use requires caution.

**API/Environment (S)**
- PettingZoo classic `go_v5`, OpenSpiel, GTP protocol to connect any engine; KataGo as evaluator/sparring partner (its analysis engine gives per-move win rates). All headless and faster than real-time.

**Affinity**: 19×19 grid encoder + 361+pass single pointer (already implemented as `PlacementDecoder`), even simpler than chess.
**Bottleneck**: Data licensing not as clean as Lichess; the KataGo baseline is too strong, creating pressure when comparing results.

---

## 3. StarCraft II — Overall B+

**Human Game Data (A-)**
- [Blizzard replay packs](https://github.com/Blizzard/s2client-proto/blob/master/samples/replay-api/README.md): download anonymized ladder 1v1 replays by version via the Blizzard Game Data API (requires a developer account + `download_replays.py`). **Per the AlphaStar Unplugged figures: ~1TB and over 1 million games across all versions** ([google-deepmind/alphastar data docs](https://github.com/google-deepmind/alphastar/blob/main/alphastar/unplugged/data/README.md)). **Still downloadable in 2026**, but **versions are stuck at 3.16–4.9.x, with no new packs after 2019**.
- License: requires agreeing to the **AI and Machine Learning License** — research use is explicitly permitted.
- Replays are engine-level action streams, so **states must be reconstructed by replaying through the game client** (the PySC2 replay observer pipeline), and the replay version must strictly match the client.
- Additional: [SC2EGSet esports dataset](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset), [sc2-serializer](https://5had3z.github.io/sc2-serializer/replay_data.html).

**API/Environment (A-)**
- [s2client-proto](https://github.com/Blizzard/s2client-proto) + [PySC2](https://github.com/google-deepmind/pysc2) (still maintained); the **Linux headless client goes up to 4.10 at most**, runs many times faster than real-time without rendering; [SMAC/SMACv2](https://github.com/oxwhirl/smacv2) provides micromanagement subtasks. Self-play is feasible but each game is extremely long.

**Affinity**: The adapters are already in place (`StarCraftEncoder`/`RTSDecoder`): feature layers go through grid, units through entity, select/target through pointer (isomorphic to AlphaStar).
**Bottleneck**: **Compute**. Recommend starting from SMACv2 or a restricted race/map subset.

---

## 4. Hearthstone — Overall C+

**Human Game Data (C)**
- [HSReplay.net](https://hsreplay.net/): data comes from user-uploaded Deck Tracker power.log. **No public bulk download**; the [API](https://github.com/HearthSim/hsreplaynet-api-docs) exposes only account-level data, and [Premium](https://hsreplay.net/premium/) sells **aggregate statistics** rather than raw games. Bulk raw data requires negotiating with HearthSim or building your own Deck Tracker collection (a Blizzard ToS gray area).
- [HearthstoneJSON](https://hearthstonejson.com/): full static card data — just card attributes, not games.
- Conclusion: **the human game data required for IL does not exist through public channels**.

**API/Environment (B-)**
- [Fireplace](https://github.com/jleclanche/fireplace) (Python simulator): still active, but card coverage is incomplete and rule fidelity is unofficial.
- [SabberStone](https://github.com/HearthSim/SabberStone) (C#): **effectively unmaintained since 2022-12**.
- No maintained standard RL environment; the official Blizzard API has only card/metadata.

**Affinity**: Good fit for entity encoder + pointer decoder (`HearthstoneEncoder`/`EntityActionDecoder` already in place) — but unfortunately no data.
**Bottleneck**: **No public human game data**, plus simulator rule drift. Could be downgraded to "pure RL self-play + a fixed small card pool on Fireplace."

---

## 5. Clash Royale — Overall D

**Human Game Data (D)**
- [Official Supercell API](https://developer.clashroyale.com/) (requires registration + IP allowlist): `/players/{tag}/battlelog` returns only **post-match metadata for the last 25 matches** — deck, crown count, final tower HP, trophies, `elixirLeaked`, etc. (see [go-clash fields](https://pkg.go.dev/github.com/overwolfmobile/go-clash)). **There is no frame-by-frame / per-card-play temporal data whatsoever**.
- [RoyaleAPI](https://royaleapi.com/) is merely a statistics site on top of the official API.
- Corroborating evidence: [KataCR](https://github.com/wty-yy/KataCR) had to **reconstruct state via CV from screen video with YOLOv8** just to do RL; [ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot) is likewise emulator screenshots + detection. Frame-level imitation data **does not exist** in the public domain.

**API/Environment (D)**
- **There is no community simulator with precise rules** (unit interactions are closed-source). All bots run live on an Android emulator at real-time 1x speed, cannot be headless-accelerated for self-play, and violate the Supercell ToS.

**Affinity**: On paper an 18×32 grid + hand entity fit (`ClashRoyaleEncoder`/`CardPlacementDecoder` already in place).
**Bottleneck**: **Both data and environment are missing**, so neither end of IL→RL holds up.

> **2026-07-05 deep-dive update**: A dedicated in-depth survey (see [clash-royale.md](clash-royale.md)) confirms:
> Frame-level data exists but is extremely scarce (KataCR Replay-Dataset: 105 games / 113,981 frames / 2 decks, no license);
> KataCR's 154-class YOLO detection dataset + synthesis pipeline is reusable (reconstructing state from YouTube recordings has already been demonstrated in practice);
> Kaggle has 481 million post-match battlelogs (metadata only); private servers "support battles" but server-side fidelity is unverified.
> Actionable path: C = battlelog-assisted pre-training (1-2 weeks) → A = KataCR stack + recording expansion for offline IL (2-4 person-months, no online PPO possible)
> → B = private server to complete the battle logic (6+ person-months, the only path capable of self-play, with the highest legal risk).

---

## 6. generals.io (baseline, done)

- Official replay interface and format: [dev.generals.io/replays](https://dev.generals.io/replays) (.gior → .gioreplay JSON, requires a simulator to reconstruct per-turn state — already implemented); the bot API (socket.io) can be deployed for live testing. The community has [strakam/generals-bots](https://github.com/strakam/generals-bots) (a high-speed JAX simulator).

---

## 7. New Candidates

### 7a. Lux AI Challenge (Kaggle, S2/S3) — Overall A (strongly recommended)

- **Data (A)**: Kaggle stores **complete step-by-step JSON for all ladder episodes** (full state + both sides' actions at each step), downloadable in bulk by team via Meta Kaggle / the Kaggle API ([RoboEden/Luxai-s2-Baseline](https://github.com/RoboEden/Luxai-s2-Baseline) has ready-made scripts); the S1 champion Toad Brigade took the IL route, and the community has validated the full IL→RL pipeline.
- **Environment (S)**: [Lux-Design-S3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) (NeurIPS 2024) is an **official JAX GPU parallel engine**, enabling large-scale self-play on a single GPU; S2 also has a gym interface.
- **Affinity**: 2D grid resource map + multiple units + unit-level discrete actions — almost isomorphic to the generals.io architecture, making it the most natural second "native battlefield."
- **Bottleneck**: The imitation target is a bot rather than human hands; the rules differ each season.

### 7b. Shogi — Overall A-

- **Data (B+)**: The [floodgate (UTokyo wdoor)](http://wdoor.c.u-tokyo.ac.jp/shogi/index-e.html) game-record repository is packaged by year (CSA format), with **110,043 games in 2024**, cumulatively on the order of a million games; these are **engine games** (high-quality demonstrations rather than human data). Human professional game records carry copyright-precedent risk in Japan; [lishogi.org](https://lishogi.org/) has no full monthly dump.
- **Environment (A)**: python-shogi / cshogi, USI protocol to connect YaneuraOu/dlshogi for evaluation.
- **Affinity**: 9×9 grid + hand pieces (a small entity set) + dual pointer, highly reusable.

### 7c. Othello — Overall A-

- **Data (B)**: [WTHOR database (French Othello Federation)](https://www.ffothello.org/informatique/la-base-wthor/): **~120K games** from official tournaments (1980 to present, .wtb binary, with a [parser](https://github.com/LimeEng/wthor)), free. All are games between human experts.
- **Environment (S)**: The rules can be implemented precisely in a few dozen lines of code (or via OpenSpiel), with Edax/eGaroucid for evaluation; self-play is extremely fast.
- **Affinity**: 8×8 grid + 64+pass single pointer — **the cheapest smoke-test target for validating the entire multi-game pipeline**.

### 7d. Poker (NLHE) — Overall B- (cautious)

- **Data (B-)**: [uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset) ([Zenodo mirror](https://zenodo.org/records/13997158)): ACPC hundreds of millions of hands (machine games), all 10K Pluribus hands, etc.; **large-scale real human hand histories are prohibited from redistribution by platform ToS**.
- **Environment (A)**: RLCard, OpenSpiel.
- **Bottleneck**: Missing human data + methodological mismatch (the optimal-solution framework for imperfect information is CFR/deep CFR, not PPO self-play).

### 7e. Examined and Rejected

- **Dota 2**: [OpenDota](https://www.opendota.com/) replay-parsed data is grade A, but there is **no headless self-play environment faster than real-time**, so the RL side is grade D.
- **Rocket League**: RLGym + ballchasing.com are both excellent, but the **continuous control action space** does not fit the discrete pointer decoder system.
- **MicroRTS / Battlesnake / Screeps**: Good environments but no decent demonstration data, so the IL stage falls through.

---

## Recommended Rollout Order

1. **Othello** (~1 week): WTHOR + a self-written 80-line simulator, to smoke-test the full "new-game onboarding → IL → RL → evaluation" pipeline (reusing `BoardEncoder` + `PlacementDecoder`).
2. **Chess** (flagship): Lichess CC0 monthly packs + python-chess reconstruction + PettingZoo/custom env + Stockfish evaluation, with Maia as a public IL benchmark (`FromToDecoder`).
3. **Go**: OGS/Tygem data + KataGo evaluation, reusing the board-game pipeline.
4. **Lux AI S3**: Kaggle episode IL + official JAX engine RL, as a modern battlefield isomorphic to generals.io, directly validatable on the leaderboard.
5. **SC2**: Only when compute allows, do SMACv2 or a restricted subset; Hearthstone and Clash Royale are shelved (the model adapters are ready, awaiting improvements in the data/simulator ecosystem).
