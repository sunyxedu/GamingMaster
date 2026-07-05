# 训练数据与环境基础设施调研（2026-07）

**适用背景**：游戏无关策略网络（时空编码器 + TRM pondering core），IL（人类/高手对局模仿）→ PPO 自博弈 RL 管线，离散动作空间，已有 grid encoder / entity encoder / pointer decoder 适配器（见 [multi-game.md](multi-game.md)）。

---

## 总排名表

| 排名 | 游戏 | IL 数据 | RL 环境 | 综合 | 最大瓶颈 |
|---|---|---|---|---|---|
| 1 | **Chess** | S（CC0，~80 亿局） | S（毫秒级 headless） | **S** | 几乎没有——工程量最小 |
| 2 | **Go** | A（OGS 5600 万 + Tygem 150 万高段） | S（PettingZoo/OpenSpiel/KataGo） | **S-** | 高质量数据授权比 Lichess 模糊 |
| 3 | **Lux AI (Kaggle S2/S3)** | A（Kaggle episode 全量逐步状态） | S（官方 JAX GPU 并行引擎） | **A** | 模仿对象是顶级 bot 而非人类 |
| 4 | **Othello/黑白棋** | B（WTHOR 约 12 万局） | S（规则平凡，Edax/eGaroucid 可做陪练与评估） | **A-** | 数据量小（但棋盘小、适合验证管线） |
| 5 | **Shogi/将棋** | B+（floodgate 年约 11 万局，为引擎对局） | A（cshogi/python-shogi + USI 引擎） | **A-** | 公开人类职业棋谱大库缺失（版权） |
| 6 | **StarCraft II** | A-（Blizzard 官方 replay pack，>100 万局，含 ML 许可） | A-（PySC2 headless、快于实时） | **B+** | 客户端/replay 停在 4.9/4.10（2019）；单局长、算力成本极高 |
| 7 | **Poker (NLHE)** | B-（ACPC 数亿手为机器对局；人类牌谱受 ToS 限制） | A（RLCard/OpenSpiel，极快） | **B-** | 公开"人类"数据几乎没有；非完美信息更适合 CFR 类方法 |
| 8 | **Hearthstone** | C（无公开批量对局数据） | B-（Fireplace 尚活跃但卡牌覆盖不全；SabberStone 停更于 2022） | **C+** | IL 数据只能自采（Deck Tracker 灰色地带）；模拟器规则保真度存疑 |
| 9 | **Clash Royale** | D（官方 API 只有赛后元数据，无逐帧） | D（无精确规则模拟器，只能模拟器实机实时跑） | **D** | 逐帧模仿数据不存在，自博弈无法 headless 加速 |
| — | **generals.io**（基线） | A | A | 已完成 | — |

**结论建议**：立即可上的是 **Chess 和 Go**（数据、环境、评估器全部现成，Maia 已证明 Lichess 数据上 IL 的可行性）；**Lux AI** 是"非棋类、网格+实体混合"的最佳新增标的，与 grid/entity 适配器契合度极高；SC2 可做但要接受高昂算力与旧版本锁定；Hearthstone 和 Clash Royale 以现有基础设施**不建议投入**（前者缺数据、后者数据与环境双缺）。

---

## 1. Chess — 综合 S

**人类对局数据（S）**
- [Lichess open database](https://database.lichess.org/)：**约 79.5 亿盘标准 rated 对局**，按月打包 PGN（.pgn.zst，月包 ~20-30GB，全量 >2TB 压缩），**CC0 许可，明确允许商用与研究**。2017 年 4 月起含时钟信息，另有 3.9 亿+ Stockfish 评估位置（jsonl）。变体（Chess960、Crazyhouse、Atomic 等各约 3000 万局）也全量开放。
- 精选子集：[Lichess Elite Database](https://database.nikonoel.fr/)（2400+ 对局筛选）。
- PGN 是完整着法序列，规则确定 → 用 [python-chess](https://github.com/niklasf/python-chess) 逐步重建每回合完整状态，**无需自写模拟器**。
- 先例：[Maia Chess](https://www.maiachess.com/) 用 1 亿+ Lichess 对局做纯 IL，人类着法预测 53%；[Maia-2 (NeurIPS 2024)](https://arxiv.org/abs/2409.20553) 单模型覆盖全棋力段——直接验证了 IL 阶段可行性。

**API/环境（S）**
- [Lichess API](https://lichess.org/api) 有官方 bot 接口（可上线实测）。
- RL 环境：[PettingZoo classic `chess_v6`](https://pettingzoo.farama.org/)（AEC + 合法动作 mask）、OpenSpiel；python-chess 本身即精确规则引擎，headless、每秒数万步，自博弈远快于实时。
- 评估：Stockfish（免费、UCI）做棋力标定和 move-quality 分析。

**契合度**：8×8 grid encoder 天然适配；from-square/to-square 双 pointer decoder（已实现 `FromToDecoder`）或 AlphaZero 式 8×8×73 动作平面均成熟。
**瓶颈**：无实质性瓶颈，是所有候选中工程上最便宜的。

---

## 2. Go — 综合 S-

**人类对局数据（A）**
- [OGS 全量 dump](https://forums.online-go.com/t/can-we-get-an-sgf-database-dump/38837)：**约 5600 万局**（SGF/JSON，压缩约 11GB，Internet Archive 有镜像），汇总见 [yenw/computer-go-dataset](https://github.com/yenw/computer-go-dataset)。
- 高段数据：Tygem **9D vs 9D 共 1,516,031 局**（2005–2016）、TOM 9D 5 万局、职业棋谱 7.3 万局（1940–2017）、[KGS 高段棋谱（u-go.net）](https://www.u-go.net/gamerecords/)。Fox 无官方 dump。
- 引擎数据：[KataGo distributed training](https://katagotraining.org/) 公开 **9500 万+ 自博弈对局**训练数据（可做预训练或蒸馏），KataGo 本体 MIT。
- SGF 是纯着法序列，状态重建平凡。授权注意：OGS/KGS 数据是"社区惯例可用"，**没有 Lichess 那样的 CC0 明示**，商用需谨慎。

**API/环境（S）**
- PettingZoo classic `go_v5`、OpenSpiel、GTP 协议接任意引擎；KataGo 做评估器/陪练（分析引擎给逐手胜率）。全部 headless、快于实时。

**契合度**：19×19 grid encoder + 361+pass 单 pointer（已实现 `PlacementDecoder`），比 chess 还简单。
**瓶颈**：数据许可不如 Lichess 干净；KataGo baseline 过强，成果对比压力大。

---

## 3. StarCraft II — 综合 B+

**人类对局数据（A-）**
- [Blizzard replay packs](https://github.com/Blizzard/s2client-proto/blob/master/samples/replay-api/README.md)：通过 Blizzard Game Data API（需开发者账号 + `download_replays.py`）按版本下载匿名化天梯 1v1 replay。**AlphaStar Unplugged 口径：全版本合计约 1TB、超 100 万局**（[google-deepmind/alphastar 数据文档](https://github.com/google-deepmind/alphastar/blob/main/alphastar/unplugged/data/README.md)）。**2026 年仍可下载**，但**版本停留在 3.16–4.9.x，2019 年后无新 pack**。
- 许可：需同意 **AI and Machine Learning License**——研究用途明确允许。
- Replay 是引擎级动作流，**必须用游戏客户端回放重建状态**（PySC2 replay observer 管线），replay 版本必须与客户端严格一致。
- 补充：[SC2EGSet 电竞数据集](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset)、[sc2-serializer](https://5had3z.github.io/sc2-serializer/replay_data.html)。

**API/环境（A-）**
- [s2client-proto](https://github.com/Blizzard/s2client-proto) + [PySC2](https://github.com/google-deepmind/pysc2)（仍在维护）；**Linux headless 客户端最高 4.10**，无渲染可数倍于实时；[SMAC/SMACv2](https://github.com/oxwhirl/smacv2) 提供微操子任务。自博弈可行但单局极长。

**契合度**：适配器已就绪（`StarCraftEncoder`/`RTSDecoder`）：feature layer 走 grid、units 走 entity、select/target 走 pointer（AlphaStar 同构）。
**瓶颈**：**算力**。建议从 SMACv2 或限定种族/地图子集切入。

---

## 4. Hearthstone — 综合 C+

**人类对局数据（C）**
- [HSReplay.net](https://hsreplay.net/)：数据来自用户上传的 Deck Tracker power.log。**没有公开批量下载**；[API](https://github.com/HearthSim/hsreplaynet-api-docs) 只开放账号级数据，[Premium](https://hsreplay.net/premium/) 卖**聚合统计**而非原始对局。批量原始数据需与 HearthSim 商谈或自建 Deck Tracker 采集（暴雪 ToS 灰色地带）。
- [HearthstoneJSON](https://hearthstonejson.com/)：全量卡牌静态数据——只是卡牌属性，不是对局。
- 结论：**IL 所需人类对局数据在公开渠道不存在**。

**API/环境（B-）**
- [Fireplace](https://github.com/jleclanche/fireplace)（Python 模拟器）：仍活跃，但卡牌覆盖不全、非官方规则保真。
- [SabberStone](https://github.com/HearthSim/SabberStone)（C#）：**2022-12 后实质停更**。
- 无维护中的标准 RL 环境；暴雪官方 API 只有卡牌/元数据。

**契合度**：entity encoder + pointer decoder 契合度好（`HearthstoneEncoder`/`EntityActionDecoder` 已就绪）——可惜没数据。
**瓶颈**：**无公开人类对局数据**、模拟器规则漂移。可退化为"纯 RL 自博弈 + 固定小卡池 on Fireplace"。

---

## 5. Clash Royale — 综合 D

**人类对局数据（D）**
- [官方 Supercell API](https://developer.clashroyale.com/)（需注册+IP 白名单）：`/players/{tag}/battlelog` 仅返回**最近 25 场的赛后元数据**——卡组、皇冠数、终局塔血量、奖杯、`elixirLeaked` 等（见 [go-clash 字段](https://pkg.go.dev/github.com/overwolfmobile/go-clash)）。**没有任何逐帧/逐次出牌的时序数据**。
- [RoyaleAPI](https://royaleapi.com/) 只是官方 API 之上的统计站。
- 旁证：[KataCR](https://github.com/wty-yy/KataCR) 为做 RL 只能用 YOLOv8 从**屏幕视频 CV 重建状态**；[ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot) 同样是模拟器截屏+检测。帧级模仿数据在公开世界**不存在**。

**API/环境（D）**
- **没有精确规则的社区模拟器**（单位交互闭源）。所有 bot 均走 Android 模拟器实机、实时 1x 速度，无法 headless 加速自博弈；且违反 Supercell ToS。

**契合度**：18×32 grid + 手牌 entity 纸面契合（`ClashRoyaleEncoder`/`CardPlacementDecoder` 已就绪）。
**瓶颈**：**数据与环境双重缺失**，IL→RL 两端都不成立。建议搁置，或仅作 CV-驱动展示项目（KataCR 路线，成本极高）。

---

## 6. generals.io（基线，已完成）

- 官方 replay 接口与格式：[dev.generals.io/replays](https://dev.generals.io/replays)（.gior → .gioreplay JSON，需模拟器重建逐回合状态——已实现）；bot API（socket.io）可上线实测。社区有 [strakam/generals-bots](https://github.com/strakam/generals-bots)（JAX 高速模拟器）。

---

## 7. 新增候选

### 7a. Lux AI Challenge (Kaggle, S2/S3) — 综合 A（强烈推荐）

- **数据（A）**：Kaggle 保存**全部天梯 episode 的完整逐步 JSON**（每步全量状态+双方动作），Meta Kaggle / Kaggle API 可按队伍批量下载（[RoboEden/Luxai-s2-Baseline](https://github.com/RoboEden/Luxai-s2-Baseline) 有现成脚本）；S1 冠军 Toad Brigade 即 IL 路线，社区已验证 IL→RL 全管线。
- **环境（S）**：[Lux-Design-S3](https://github.com/Lux-AI-Challenge/Lux-Design-S3)（NeurIPS 2024）**官方 JAX GPU 并行引擎**，单卡即可大规模自博弈；S2 亦有 gym 接口。
- **契合度**：2D 网格资源图 + 多单位 + 单位级离散动作——与 generals.io 架构几乎同构，是最自然的第二个"原生战场"。
- **瓶颈**：模仿对象是 bot 而非人类手操；每季规则不同。

### 7b. Shogi/将棋 — 综合 A-

- **数据（B+）**：[floodgate（东大 wdoor）](http://wdoor.c.u-tokyo.ac.jp/shogi/index-e.html) 棋谱倉庫按年打包（CSA 格式），**2024 年 110,043 局**，累计约百万局级；为**引擎对局**（高质量示教而非人类数据）。人类职业棋谱在日本有版权判例风险；[lishogi.org](https://lishogi.org/) 无月度全量 dump。
- **环境（A）**：python-shogi / cshogi，USI 协议接 YaneuraOu/dlshogi 做评估。
- **契合度**：9×9 grid + 持驹（小 entity 集合）+ 双 pointer，复用度高。

### 7c. Othello/黑白棋 — 综合 A-

- **数据（B）**：[WTHOR 数据库（法国黑白棋联合会）](https://www.ffothello.org/informatique/la-base-wthor/)：**约 12 万局**正式比赛（1980 至今，.wtb 二进制，有 [parser](https://github.com/LimeEng/wthor)），免费。全是人类高手对局。
- **环境（S）**：规则数十行代码精确实现（或 OpenSpiel），Edax/eGaroucid 做评估；自博弈极快。
- **契合度**：8×8 grid + 64+pass 单 pointer——**验证整个多游戏管线最便宜的冒烟测试标的**。

### 7d. Poker (NLHE) — 综合 B-（谨慎）

- **数据（B-）**：[uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset)（[Zenodo 镜像](https://zenodo.org/records/13997158)）：ACPC 数亿手（机器对局）、Pluribus 全部 1 万手等；**大规模真实人类牌谱受平台 ToS 禁止分发**。
- **环境（A）**：RLCard、OpenSpiel。
- **瓶颈**：人类数据缺失 + 方法论错配（非完美信息的最优解框架是 CFR/深度 CFR 而非 PPO 自博弈）。

### 7e. 已考察并否决

- **Dota 2**：[OpenDota](https://www.opendota.com/) replay 解析数据 A 级，但**没有 headless 快于实时的自博弈环境**，RL 端 D 级。
- **Rocket League**：RLGym + ballchasing.com 俱佳，但**连续控制动作空间**与离散 pointer decoder 体系不契合。
- **MicroRTS / Battlesnake / Screeps**：环境好但缺像样的示教数据，IL 阶段落空。

---

## 落地顺序建议

1. **Othello**（1 周级）：WTHOR + 自写 80 行模拟器，冒烟测试"新游戏接入→IL→RL→评估"全管线（复用 `BoardEncoder` + `PlacementDecoder`）。
2. **Chess**（主打）：Lichess CC0 月包 + python-chess 重建 + PettingZoo/自写 env + Stockfish 评估，Maia 作为 IL 公开对标（`FromToDecoder`）。
3. **Go**：OGS/Tygem 数据 + KataGo 评估，复用棋类管线。
4. **Lux AI S3**：Kaggle episode IL + 官方 JAX 引擎 RL，作为 generals.io 同构的现代战场，可直接打榜验证。
5. **SC2**：仅在算力允许时做 SMACv2 或限定子集；Hearthstone、Clash Royale 搁置（模型适配器已备好，等数据/模拟器生态改善）。
