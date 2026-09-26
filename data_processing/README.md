# Data pipelines

All data pipelines — fetch, filter, replay/simulate, convert — are Rust. They run
on macOS and Linux with a stable Rust toolchain (`https://rustup.rs`); nothing
else is needed at conversion time. Training reads the output from Python.

```
data_processing/
├── Cargo.toml                 workspace
├── rs/gmshard/                gm-shard.v1 writer / reader / overlay writer (.npy + index.json)
├── rs/gmdata/                 `gmdata <game> fetch|convert|…` for every game below,
│                              `gmdata label <game>` engine teacher labels (src/label/)
├── games/clash_royale/crdata/ `crdata` (Clash Royale replays)
├── games/dataset.py           GameShardDataset: reads gm-shard directories (and .pt)
├── games/clash_royale/        card catalog + load-time observation expansion
├── dataset_seq.py             SeqReplayDataset for generals gio records
├── download/ processing/ parsers/   generals reference decoder/simulator/converter
│                                    (viewers, play_model; the Rust pipeline is
│                                    tested bit-identical to them)
└── games/othello|gomoku/rules.py    rules used by the RL self-play environment
```

```sh
cd data_processing && cargo build --release
G=target/release/gmdata        # add --help to any subcommand
```

## Output format

`gm-shard.v1` directories: one `.npy` per tensor (`obs.*`, `tgt.*`, frame-major,
samples are frame ranges) plus `index.json` (game, converter version, shard meta,
per-sample meta). One-hot planes are stored bit-packed and unpacked on load.
`GameShardDataset(dir, game=...)` memory-maps them directly — no conversion to
`.pt`. It refuses to mix converter versions, shard meta or key sets. generals.io
keeps its historical per-game gio JSON (`SeqReplayDataset`, `training/train_il.py`).
For the multi-game trainers generals reads gm-shards made from them —
bit-packed, memory-mapped, ~10x smaller, ~200x faster to load, records shuffled
across shards (`--seed`, `--keep-order` for path order), and pinned
tensor-for-tensor to `SeqReplayDataset` by `tests/test_rust_generals_parity.py`:

```sh
$G generals shard --input data/generals/training --out data/generals/shards
```

Small corpora can skip the conversion: the gio directory itself is read by
`games/generals.py::GeneralsGioDataset` (same decoding, but a whole-record JSON
parse per draw and a first-run length scan over every record).

## Games, sources and the high-quality recipe

Imitation learning here uses only high-quality play. Every converter has quality
filters and a `--hq` preset; the defaults reproduce the previous (unfiltered)
behaviour, which is what the golden parity tests pin.

| game | source (fetch) | high-quality recipe | notes |
|---|---|---|---|
| generals.io | `$G generals ladder --out D --min-stars 70` (global duel ladder) | winner ≥ 70★ (or `--both`), human accounts only (`isBot` check), ≥ 50 turns | per-account `generals pipeline` (= `scripts/prepare_data.py`) still exists; its default account Human.exe is a bot |
| chess | `$G chess fetch --month 2021-06 --elite` (Lichess Elite DB) or `--month` (full dump) | `convert --hq`: both Elo ≥ 2200, rated, normal termination, no bullet, ≥ 20 plies | elite months are pre-filtered 2400+/2200+ (2500+/2300+ since 2021-12) |
| go | `$G go fetch --url https://dl.u-go.net/gamerecords/KGS-…tar.bz2` | `convert --hq`: both ≥ 6d, no time-loss/forfeit, no handicap, ≥ 50 moves | KGS archives already need one 7d or both 6d |
| shogi | `$G shogi fetch --day 2025/06/01 --ratings 20250601` or `--year 2023` | `convert --hq --ratings D/ratings-….json`: both engines ≥ 3500 on the floodgate table, floodgate-300-10F, no time-up | engine play |
| othello | `$G othello fetch --first-year 2000` (WTHOR) | `convert --hq`: \|recorded − perfect-play score\| ≤ 4 discs, year ≥ 2000 | human tournament games; the score bound measures endgame error |
| gomoku | Gomocup results zips (`https://gomocup.org/results/`) | `convert --top-engines 8` on one league (e.g. Freestyle1) | strength = win rate within the input |
| riichi mahjong | `$G mahjong fetch --shards 0 1 2` (Tenhou Houou mjai, 2.1M games) | `convert --hq`: hanchan only; optional `--top-placements 2` | phoenix room = top players already |
| Pokémon | `$G pokemon fetch --format gen9ou --min-rating 1500` | `convert --hq`: rating ≥ 1500, ≥ 5 turns, no inactivity losses; optional `--winner-only` | forfeits are kept (concessions affect only the last turn) |
| Lux AI S3 | `$G lux fetch --top-teams 10` (Kaggle API) | only episodes where both agents are top-10 leaderboard teams | needs `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` |
| Clash Royale | `crdata fetch --out D` (RoyaleAPI replays, HF) | `crdata convert --hq`: Ranked + Grand Challenge, ≥ 60 s, ≥ 5 own plays; optional `--winner-only` | anonymised (no trophies); see games/clash_royale/README.md |

Measured on real data (kept / seen): chess elite 2021-06 601k / 781k games;
go KGS 2019-04 202 / 1255; shogi 2025-06-01 158 / 447; othello 2019–21
1502 / 3559; gomoku Freestyle1 2023 658 / 2880; Clash Royale 5000 replays
8456 / 9392 player sequences (4186 with `--winner-only`).

## Engine teacher labels

`gmdata label <game>` adds engine soft targets (teacher move distributions per
decoder head) and engine values to existing gm-shards, as overlay shards that
hard-link the base tensors (the base is never modified) and that the trainers
read unchanged (`--soft_target_weight`, `--engine_value_weight`). Engines are
external processes installed outside the repo by
`scripts/engines/fetch_engines.sh` (Stockfish, Egaroucid, YaneuraOu, KataGo;
licences in its header). Columns, options and the per-game extension point are
documented in `rs/gmdata/src/label/mod.rs`; `gmdata label selftest --out DIR`
exercises the whole pipeline with a built-in fake engine.

```sh
scripts/engines/fetch_engines.sh stockfish
$G label chess --input data/chess/train --out data/chess_sf/train \
   --engine ~/gm-engines/bin/stockfish --workers 32 --limit nodes=1000000
scripts/engines/fetch_engines.sh egaroucid
$G label othello --input data/othello/train --out data/othello_ega/train \
   --engine ~/gm-engines/bin/egaroucid --workers 64   # Egaroucid level 15, exact <= 20 empties
```

## Tests

```sh
cargo test --release                                   # Rust unit tests
uv run pytest tests/test_rust_converters_parity.py \
              tests/test_rust_generals_parity.py tests/test_clash_royale_data.py \
              tests/test_label_overlay.py tests/test_fetch_engines.py \
              tests/test_label_othello.py
```

Real-engine tests run only when the engine is installed (e.g. `STOCKFISH=path`
or `~/gm-engines/bin/stockfish`, `GM_EGAROUCID=path` or `~/gm-engines/bin/egaroucid`);
the label tests otherwise use fake engines.

The parity tests pin every converter to golden SHA-256 digests of the retired
Python reference outputs on committed real-data fixtures (tests/fixtures/).
