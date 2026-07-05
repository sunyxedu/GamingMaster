# Clash Royale AI 基础设施深度调研（2026-07-05）

深度调研（103 agent、多源对抗核实，全部关键论断 3-0 通过）的结论。目标：为 IL→PPO
管线找 CR 的数据与环境方案（模型侧适配器 `model/games/clash_royale.py` 已就绪）。

## 核心结论

1. **不存在任何可加速、headless 的高保真 CR 模拟器。** 唯一发表的端到端 CR AI 工作
   [KataCR (arXiv 2504.04783)](https://arxiv.org/abs/2504.04783v1) 明确因"非嵌入式
   agent 与真实环境交互太慢、无法在线训练"而放弃在线 RL，改用**离线 RL**
   （Decision Transformer / StARformer），评估直接在真机（手机 + Scrcpy）进行。
2. **帧级数据存在但极少。** 公开的唯一帧级时序对局数据是 KataCR 配套的
   [Clash-Royale-Replay-Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset)：
   作者手动对战 8000 分内置 AI 采集 **105 局 / 113,981 帧**（动作帧仅 4.01%），
   仓库 347 个 `.npy.xz`（约 2 GB），只覆盖 2 套卡组，**无 license**。
3. **CV 感知栈可以直接复用。**
   [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)：
   154 类（150 检测类，2023-10 时点基本覆盖全卡池）、4,654 张单位切片 +
   合成生成管线 + **6,939 张人工标注真实帧（116,878 个框，YOLO 格式）**；
   视频源是 YouTube 录像/自录——"从录像重建状态"这条路已被实践。
   另有 [Roboflow 社区集](https://universe.roboflow.com/clashroyale/clash-royale-of3d3/dataset/3)
   972 张（CC BY 4.0，质量社区级）。
4. **大规模数据只有赛后元数据。** Kaggle
   [s1m0n38/clash-royale-games](https://www.kaggle.com/datasets/s1m0n38/clash-royale-games)
   **4.81 亿场**（2022-09~2023-12，~92.6 GB CSV，每行 24 列：卡组/奖杯/皇冠，无任何局内动作）
   + [bwandowando S18](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset)
   3,790 万场。官方 API（developer.clashroyale.com）上限就是这类赛后摘要
   （已核实：官网 JS bundle 零次出现 replay 通道；RoyaleAPI 的"replay"只是深链回客户端观战）。
5. **KataCR 最终效果很弱**：最佳变体 StARformer-3L 对 8000 分内置 AI 胜率仅
   **10%**（多数变体 5%/0%，20 局评估）——纯视觉 + 105 局离线 RL 的上限证明。
6. **私服战斗逻辑未经证实。** [retroroyale/ClashRoyale](https://github.com/retroroyale/ClashRoyale)
   及其 fork（ZrdRoyale 等）声明"支持战斗"但需打补丁客户端，战斗很可能是客户端驱动，
   服务端权威模拟的保真度需要读码验证；IP/法律风险显著。

## 三条路线

| 路线 | 内容 | 量级 | 风险 | 能否支撑 PPO 自博弈 |
|---|---|---|---|---|
| **C** battlelog 辅助预训练 | Kaggle 4.8 亿场 → 卡组表征 / 胜率预测 / value 先验 | 1-2 周 | 极低 | 否（无放置策略） |
| **A** KataCR 栈 + 录像扩充（主线） | 复用检测数据集+感知管线冷启动；YouTube/自录扩充帧级 IL 数据；真机 Scrcpy 小规模评估 | 2-4 人月 | 标注成本、真机吞吐极低、录制/自动化违反 Supercell ToS | 否（只能离线） |
| **B** 私服补全战斗逻辑 | 以 retroroyale 骨架 + 社区解包数值自建 headless 战斗引擎 | 6+ 人月 | 保真度需逐单位校验；法律风险最高 | **是（唯一路径）** |

**建议**：C 立即启动（与我们 CR 适配器的 value head 兼容，作辅助任务）；A 为主线
（第一步：拉 Replay-Dataset，检查 `.npy` schema 能否映射到我们的
`arena_seq/hand_*` obs 契约，写转换器把 105 局灌进 `train_il_game.py` 做冒烟）；
B 仅在 A 数据瓶颈无法突破且愿意长期投入时评估（第一步是读 retroroyale 的
battle 相关源码，确认服务端到底有没有权威模拟）。

## 未决问题（下一轮调研/动手验证）

- retroroyale 系服务端源码里战斗结算是权威模拟还是客户端上报？（决定路线 B 起点）
- KataCR replay `.npy` 的内部 schema（网格分辨率/单位编码/动作表示）与我们适配器的
  兼容度？（决定路线 A 第一步的转换成本）
- ClashRoyaleBuildABot / py-clash-bot 等工程的环境方案与吞吐（本轮未获 3-0 证实的
  claim，不下结论）。
- Supercell 官方是否发表过内部 AI/bot/模拟器资料。

## 注意事项

- Replay-Dataset 无 license，复用有法律不确定性；154 类标签会随卡池更新过时
  （论文 2025-04 时点）。
- 所有真机自动化（Scrcpy 操控、录制采集）均处于 Supercell ToS 灰色/违规地带，
  仅适合研究用途、小规模、不公开对战。
