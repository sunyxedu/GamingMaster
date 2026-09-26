# Clash Royale data

Tier-3 game: the trained network plays it (`model/games/clash_royale.py`), no
Meta-Harness. This directory turns public **action replays** into training
shards for that adapter.

```
catalog/cards.json      122 cards: dense id, RoyaleAPI card_key, cost, kind, stats
catalog/timeline.json   elixir regeneration schedule, match sections, tick length
crdata/                 Rust converter (fetch / stats / convert) -> gm-shard.v1
expand.py               load-time expansion of compact raw.* fields into the adapter obs
belief.py               the causal hand belief in Python (what a live bot computes)
```

## Source data

Anonymised RoyaleAPI replays on Hugging Face (`VanguardX101/IL_Replay`):
252,238 matches, 17.8 M actions, Parquet. Each replay carries both decks
(with evolution / hero forms and levels), tower troops, the result, and every
card play / champion ability at 20 Hz with an 18×32 arena cell and side. It
does **not** carry unit positions, tower HP over time, or hands — those are
either reconstructed here (elixir, a belief over the hand) or absent (units).

Sampled facts (first 5000 replays): 89 % `Ranked` (Path of Legend), card
level 16 in 90 % of slots, all 122 catalog cards present, 2054 distinct decks
per 10 000 player-sides; team plays land in rows 0–15 in 86 % of cases (the
rest are spells cast into the enemy half), which fixes the side convention.

## Pipeline

Runs on macOS and Linux: the converter is pure Rust (stable toolchain, TLS via
rustls, no system libraries beyond a C compiler for the zstd codec); loading
needs only numpy + torch. `tests/test_clash_royale_data.py` builds the
crate and runs it end-to-end when `cargo` is on the path or in `~/.cargo/bin`.

```sh
cd data_processing && cargo build --release -p crdata
B=target/release/crdata
$B fetch   --out data/cr/raw --limit 1                 # SHA-256 verified, resumable (17 MB / 5000 replays)
$B stats   --input data/cr/raw --limit 1
$B convert --input data/cr/raw --out data/cr/shards    # 5000 replays ≈ 4 s, ~1 GB
```

`convert` also reads JSON-lines files (`{"tag", "payload"}` per line, e.g.
`tests/fixtures/clash_royale/replays.jsonl`).
`GameShardDataset("data/cr/shards", game="clash_royale")` reads the shards
directly; the dense arena planes are built per sample at load time (~10 ms for
a 3-minute match), so disk holds only compact per-frame fields.

`convert` emits **two samples per replay** (one per player, in that player's
own frame: own back line = row 0, x mirrored for the top player) at one frame
every `--step-ticks` (10 = 0.5 s). Every input column of a frame is information
the player has at that moment (`crdata/src/leak_audit.rs` truncates replays
and rebuilds, perturbs the opponent's private information, and checks the
belief against true cycles; `tests/test_clash_royale_leaks.py`). Per frame:

| field | how it is obtained |
|---|---|
| `elixir`, `opp_elixir` | exact: regeneration schedule from `timeline.json` integrated between plays, minus card costs (the opponent's from their visible plays only); violations (a play the player could not afford under the reconstruction) are counted per sample and sequences with more than `--max-elixir-violations` are dropped |
| `deck_ids`, `deck_form`, `deck_kind`, `deck_cost` | the 8 own deck cards — the **deck tokens**, ordered by card id (the order carries no meaning) |
| `deck_in_hand`, `deck_next` | the causal hand belief: of the 70 × 24 equally likely initial (hand set, queue order) states, those under which every own play **before the frame** was in hand; per deck card the fraction with it in hand / next (`crdata/src/hand.rs::CycleBelief`). After m ≤ 3 in-deck own plays: played cards 0, the rest 4/(8 − m) resp. 1/(8 − m); from the 4th on exactly 0/1 |
| `recent_*` | the last `--recent` plays of both sides with card, cell (own frame) and age in ticks; ability events are attributed to the unique ability-capable card their side has already played, else unknown |
| `waited`, `card_slot`, `place_x/y`, `ability` | the first own action inside the frame window; `card_slot` = **deck index** of the played card |
| `phase`, `overtime` | single / double / triple elixir and overtime flags |

`expand.py` builds the adapter's observation at load time: deployment planes (where each
side committed elixir and how long ago, exponential decay), own/enemy half and
river planes, broadcast scalars, deck-token features (cost, kind, form,
affordable, P(in hand), P(next)), legality masks (a deck card is playable iff
P(in hand) > 0 and it is affordable at the frame; placements on the own half,
the whole arena when a playable card is a spell), and targets (`waited`,
`card_slot`, `place_idx`, `value`). The card head of the network points over
the deck tokens (`ClashRoyaleConfig.cr_card_tokens = "deck"`). The IL loss is
not masked: the label is the first action inside the 0.5 s window, and a card
can become affordable during it (0.2 % of real play labels). Shard `meta`
carries `n_cards`, `arena_channels`, `card_tokens`, `deck_size`,
`card_feature_dim` etc. so `GameShardDataset` refuses to mix generations and
the trainers refuse a mismatching network.

**converter_version 3** (current). Version 2 shards (`raw.hand_*`) inferred the
hand cycle from the WHOLE match and picked a canonical hand: the card about to
be played was marked "certainly in hand" (99.7 % of the decisions before the
4th own play on real replays) — a future leak — and ability events named the
first ability-source candidate, possibly an opponent card not played yet.
`expand.py` still expands them, unchanged, for legacy
`cr_card_tokens = "hand"` checkpoints; the trainers warn on them (refuse with
`--require_frozen_schema`), and a fresh default network refuses them.

### Deployment (live bot)

The bot must feed the network the same inputs it was trained on: one
`belief.CycleBelief(deck ids in belief.deck_token_order)` per match, updated
with the bot's **own plays** (`observe(card_id)`), gives `deck_in_hand` /
`deck_next`; do **not** substitute the client's true hand or next card for them
(a different input distribution). The true hand (and elixir) may be used as a
legality mask on the card head: it always lies inside the belief's support.
`belief.py` is bit-identical to the converter
(`tests/test_clash_royale_belief.py`).

### Full re-conversion (after a converter_version bump)

Converting needs `cargo` (the cluster has none): convert on a machine with the
Rust toolchain and copy the result to the cluster data root (`$DATA_ROOT` of
`scripts/slurm/multi/submit_core_study.sh`). `GM_DATA` below is your local data
root. From the repository root:

```sh
cd data_processing && cargo build --release -p crdata && cd ..
CR=data_processing/target/release/crdata
$CR fetch   --out "$GM_DATA/raw/cr"                      # every replay part of VanguardX101/IL_Replay (resumable)
$CR convert --input "$GM_DATA/raw/cr" --out "$GM_DATA/shards_v3/clash_royale" --hq
# train / val split as before: shard directories with index % 5 == 4 → val, the rest → train
for d in "$GM_DATA"/shards_v3/clash_royale/clash_royale_*; do
  i=$((10#${d##*_})); s=train; [ $((i % 5)) -eq 4 ] && s=val
  mkdir -p "$GM_DATA/split_v3/clash_royale/$s" && ln -sfn "$d" "$GM_DATA/split_v3/clash_royale/$s/"
done
rsync -aL "$GM_DATA/split_v3/clash_royale/" <cluster>:<DATA_ROOT>/clash_royale/   # -L: copy, not the links
```

Replace the cluster's version-2 `clash_royale/{train,val}` directories
entirely: `GameShardDataset` refuses a directory that mixes converter versions.
(`--hq` keeps the same samples as version 2: 8456 of 9392 player sequences per
5000 replays.)

Known limits: no unit state (an IL model trained here learns *when and where
people deploy given plays and elixir*, not reactions to unit positions);
placement legality ignores destroyed towers; champion abilities are labelled
as `ability` frames but the adapter has no ability head yet.
