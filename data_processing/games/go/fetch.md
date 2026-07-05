# Getting Go game records (SGF)

Three good public sources, roughly in order of volume-per-effort. After
downloading, point the converter at the unpacked directory:

```sh
uv run python -m data_processing.games.go.convert \
    --input raw_sgf/ --output_dir data/go/train \
    --shard_size 256 --board_size 19 --min_moves 30
```

## 1. OGS (online-go.com) dump — Internet Archive

Tens of millions of games (all board sizes — the converter's
`--board_size` filter handles the mix). Mirrored on the Internet Archive:

```sh
wget https://archive.org/download/ogs-sgf-2021/ogs-sgf-2021.tar.bz2
```

(Search archive.org for "OGS SGF" for the latest snapshot name.)

## 2. u-go.net KGS archives

Curated high-dan (4d+) KGS games, one tarball per month, ~100k games
total. Clean records, strong amateur/pro play — the classic AlphaGo-era
supervised set:

```sh
wget https://u-go.net/gamerecords/KGS-2019_04-19-1255-.tar.bz2
```

(Browse https://u-go.net/gamerecords/ for the full monthly list.)

## 3. katagotraining.org

KataGo distributed-training selfplay games — machine games, but billions
of positions with very strong play; SGFs are bundled inside the training
data packages:

```sh
wget https://katagotraining.org/uploaded/networks/zips/kata1/kata1-b18c384nbt-s9996604416-d4316597426.zip
```

(See https://katagotraining.org/games/ for browsable selfplay SGF sets.)

Notes:
- Handicap games are skipped by default; pass `--allow_handicap` to keep
  them (AB stones are pre-placed before frame 0).
- Games without a decisive `RE` tag are skipped unless
  `--allow_no_result` (their value target becomes 0).
