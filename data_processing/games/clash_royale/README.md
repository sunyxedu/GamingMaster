# Clash Royale data

Tier-3 game: the trained network plays it (`model/games/clash_royale.py`), no
Meta-Harness. This directory turns public **action replays** into training
shards for that adapter.

```
catalog/cards.json      122 cards: dense id, RoyaleAPI card_key, cost, kind, stats
catalog/timeline.json   elixir regeneration schedule, match sections, tick length
crdata/                 Rust converter (fetch / stats / convert)
loader.py               cr-frames.v1 shards -> .pt shards (GameShardDataset)
```

## Source data

Anonymised RoyaleAPI replays on Hugging Face (`VanguardX101/IL_Replay`):
252,238 matches, 17.8 M actions, Parquet. Each replay carries both decks
(with evolution / hero forms and levels), tower troops, the result, and every
card play / champion ability at 20 Hz with an 18×32 arena cell and side. It
does **not** carry unit positions, tower HP over time, or hands — those are
either reconstructed here (elixir, hand cycle) or absent (units).

Sampled facts (first 5000 replays): 89 % `Ranked` (Path of Legend), card
level 16 in 90 % of slots, all 122 catalog cards present, 2054 distinct decks
per 10 000 player-sides; team plays land in rows 0–15 in 86 % of cases (the
rest are spells cast into the enemy half), which fixes the side convention.

## Pipeline

Runs on macOS and Linux: the converter is pure Rust (stable toolchain, TLS via
rustls, no system libraries beyond a C compiler for the zstd codec); the
loader needs only numpy + torch. `tests/test_clash_royale_data.py` builds the
crate and runs it end-to-end when `cargo` is on the path or in `~/.cargo/bin`.

```sh
cd data_processing/games/clash_royale/crdata && cargo build --release
B=target/release/crdata
$B fetch   --out data/cr/raw --limit 3                 # SHA-256 verified, resumable
$B stats   --input data/cr/raw --limit 1
$B convert --input data/cr/raw --out data/cr/frames    # 5000 replays ≈ 2 s on a laptop
uv run python -m data_processing.games.clash_royale.loader --frames data/cr/frames --out data/cr/shards
```

`convert` emits **two samples per replay** (one per player, in that player's
own frame: own back line = row 0, x mirrored for the top player) at one frame
every `--step-ticks` (10 = 0.5 s). Per frame:

| field | how it is obtained |
|---|---|
| `elixir`, `opp_elixir` | exact: regeneration schedule from `timeline.json` integrated between plays, minus card costs; violations (a play the player could not afford under the reconstruction) are counted per sample and sequences with more than `--max-elixir-violations` are dropped |
| `hand_ids`, `next_card` | hand-cycle inference: enumerate the 70 × 24 initial (hand, queue) states and keep those under which every recorded play was in hand; the canonical one is used and `hand_consistent_states` records the ambiguity |
| `recent_*` | the last `--recent` plays of both sides with card, cell (own frame) and age in ticks |
| `waited`, `card_slot`, `place_x/y`, `ability` | the first own action inside the frame window |
| `phase`, `overtime` | single / double / triple elixir and overtime flags |

`loader.py` builds the adapter's observation: deployment planes (where each
side committed elixir and how long ago, exponential decay), own/enemy half and
river planes, broadcast scalars, hand features, legality masks (affordable
cards; own half, whole arena when a spell is in hand), and targets
(`waited`, `card_slot`, `place_idx`, `value`). Shard `meta` carries
`n_cards`, `arena_channels`, `step_ticks` etc. so `GameShardDataset` refuses to
mix generations.

Known limits: no unit state (an IL model trained here learns *when and where
people deploy given plays and elixir*, not reactions to unit positions);
placement legality ignores destroyed towers; champion abilities are labelled
as `ability` frames but the adapter has no ability head yet.
