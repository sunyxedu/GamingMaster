# Clash Royale AI Infrastructure Deep Dive (2026-07-05)

Conclusions from a deep-dive investigation (103 agents, multi-source adversarial verification; every key claim passed 3-0). Goal: find a data and environment solution for Clash Royale to feed our IL→PPO pipeline (the model-side adapter `model/games/clash_royale.py` is already in place).

## Core Findings

1. **No accelerated, headless, high-fidelity CR simulator exists.** The only published end-to-end CR AI work,
   [KataCR (arXiv 2504.04783)](https://arxiv.org/abs/2504.04783v1), explicitly abandoned online RL because "a non-embedded
   agent interacts with the real environment too slowly to train online," switching instead to **offline RL**
   (Decision Transformer / StARformer), with evaluation run directly on a physical device (phone + Scrcpy).
2. **Frame-level data exists but is extremely scarce.** The only public frame-level time-series match data is the
   [Clash-Royale-Replay-Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset) that ships with KataCR: the
   author manually played against the 8000-trophy built-in AI to collect **105 matches / 113,981 frames** (action frames
   are only 4.01%), with 347 `.npy.xz` files in the repo (about 2 GB), covering only 2 decks, and **no license**.
3. **The CV perception stack can be reused directly.**
   [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset): 154 classes
   (150 detection classes, essentially covering the full card pool as of 2023-10), 4,654 unit crops + a synthetic
   generation pipeline + **6,939 hand-annotated real frames (116,878 boxes, YOLO format)**; the video sources are
   YouTube recordings / self-recorded footage — the "reconstruct state from recordings" path has already been proven in
   practice. There is also a [Roboflow community set](https://universe.roboflow.com/clashroyale/clash-royale-of3d3/dataset/3)
   of 972 images (CC BY 4.0, community-grade quality).
4. **Large-scale data is post-match metadata only.** Kaggle
   [s1m0n38/clash-royale-games](https://www.kaggle.com/datasets/s1m0n38/clash-royale-games)
   holds **481 million matches** (2022-09 to 2023-12, ~92.6 GB CSV, 24 columns per row: deck / trophies / crowns, without
   any in-match actions), plus [bwandowando S18](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset)
   with 37.9 million matches. The official API (developer.clashroyale.com) caps out at exactly this kind of post-match
   summary (verified: the official site's JS bundle has zero occurrences of a replay channel; RoyaleAPI's "replay" is
   merely a deep link back into the client to spectate).
5. **KataCR's final performance is very weak**: the best variant, StARformer-3L, achieves only a **10%** win rate against
   the 8000-trophy built-in AI (most variants at 5% / 0%, over a 20-match evaluation) — proof of the ceiling for pure
   vision + 105-match offline RL.
6. **Private-server battle logic is unverified.** [retroroyale/ClashRoyale](https://github.com/retroroyale/ClashRoyale)
   and its forks (ZrdRoyale, etc.) claim "battle support" but require a patched client; the battle is very likely
   client-driven, and the fidelity of authoritative server-side simulation would need a code read to verify; the IP /
   legal risk is significant.

## Three Routes

| Route | Content | Magnitude | Risk | Can it support PPO self-play |
|---|---|---|---|---|
| **C** battlelog-assisted pretraining | Kaggle 481M matches → deck representation / win-rate prediction / value prior | 1-2 weeks | Very low | No (no placement policy) |
| **A** KataCR stack + recording augmentation (main line) | Reuse detection dataset + perception pipeline for cold start; augment frame-level IL data with YouTube / self-recording; small-scale evaluation via physical-device Scrcpy | 2-4 person-months | Annotation cost, extremely low physical-device throughput, recording/automation violates Supercell ToS | No (offline only) |
| **B** private-server battle-logic completion | Build a headless battle engine on the retroroyale skeleton + community-datamined values | 6+ person-months | Fidelity must be validated unit by unit; highest legal risk | **Yes (the only path)** |

**Recommendation**: Start C immediately (compatible with the value head of our CR adapter, as an auxiliary task); A is the
main line (first step: pull the Replay-Dataset, check whether the `.npy` schema can map to our `arena_seq/hand_*` obs
contract, and write a converter to feed the 105 matches into `training/train_il_game.py` for a smoke test); evaluate B only if A's
data bottleneck proves impossible to break through and there is willingness to invest long-term (the first step is to read
retroroyale's battle-related source code and confirm whether the server actually does authoritative simulation).

## Open Questions (next round of investigation / hands-on verification)

- In the retroroyale-family server source, is battle resolution authoritative simulation or client-reported? (Decides the
  starting point for Route B.)
- How compatible is the internal schema of KataCR's replay `.npy` (grid resolution / unit encoding / action representation)
  with our adapter? (Decides the conversion cost of Route A's first step.)
- The environment approach and throughput of projects like ClashRoyaleBuildABot / py-clash-bot (a claim not verified 3-0
  this round; no conclusion drawn).
- Whether Supercell has officially published any internal AI/bot/simulator material.

## Caveats

- The Replay-Dataset has no license, so reuse carries legal uncertainty; the 154-class labels will go stale as the card
  pool updates (as of the 2025-04 paper).
- All physical-device automation (Scrcpy control, recording collection) sits in Supercell ToS gray/violating territory,
  and is suitable only for research purposes, at small scale, without public matches.
