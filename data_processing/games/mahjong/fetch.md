# Getting riichi mahjong game records (mjai .mjson)

The converter consumes **mjai-format** event streams (one JSON event per
line: `start_game` / `start_kyoku` / `tsumo` / `dahai` / `pon` / `chi` /
`reach` / `hora` / ...), the community lingua franca for Tenhou logs.
After downloading, point the converter at the archive (zips are read
directly, gzipped members included):

```sh
uv run python -m data_processing.games.mahjong.convert \
    --input mjai_2009.zip --output_dir data/mahjong/train \
    --shard_size 128 --max_games 50000
```

## 1. Bulk: tenhou-to-mjai community mirror (17 years, pre-converted)

The `tenhou-to-mjai` project publishes yearly zips of Tenhou phoenix/
houou-table logs already converted to mjai `.mjson` (≈7k games/year in
the early years, far more later). One command per year:

```sh
gh release download logs -R gimite/tenhou-to-mjai -p 'mjai_2009.zip'
```

(Browse the project's releases page for the full year list; a widely
mirrored alternative is the `mjai-logs` dataset on archive.org — search
"tenhou mjai logs".)

## 2. Incremental: Tenhou raw feed via list.cgi + conversion

Tenhou publishes a rolling index of recent raw logs. Fetch the index
(one command), then convert new `.mjlog` XMLs to mjai with the
`mjai-converter` tool before running our converter:

```sh
curl -s 'https://tenhou.net/sc/raw/list.cgi?old' -o list_old.cgi.txt
```

Each listed archive lives under `https://tenhou.net/sc/raw/dat/` (e.g.
`scraw2025.zip` for a full year, `sca<YYYYMMDD>.log.gz` for daily
indexes of game IDs). Raw `.mjlog` (gzipped Tenhou XML) must be
converted to mjai `.mjson` first — e.g. with
[`mjai`](https://github.com/gimite/mjai)'s `mjai convert in.mjlog
out.mjson` — our pipeline intentionally consumes only the mjai format.

Notes:
- Red fives appear as `5mr` / `5pr` / `5sr` (the `0m`-style notation is
  also accepted by the converter).
- Games with hidden tiles (`"?"`, spectator-view logs) are skipped —
  only full game records are trainable.
- Sanma (3-player) logs are not supported; the converter skips any game
  whose `tehais` don't hold 4×13 tiles.
