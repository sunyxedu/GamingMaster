# Getting Pokémon Showdown replays

One-liner (public replay search API, cursor-paginated, resumable):

```sh
uv run python -m data_processing.games.pokemon.fetch_replays \
    --format gen9ou --min_rating 1100 --max_replays 5000 \
    --sleep 1.0 --output_dir replays/gen9ou
```

then convert to shards:

```sh
uv run python -m data_processing.games.pokemon.convert \
    --input replays/gen9ou --output_dir data/pokemon/train \
    --shard_size 256 --min_rating 1100
```

Notes:

- The search API returns ~51 replays per page and only goes back so far
  for very active formats; for bulk historical data, the
  **metamon parsed replay dataset** on Hugging Face
  (`jakegrigsby/metamon-parsed-replays`, from the Metamon paper) has
  millions of pre-parsed Showdown battles — a converter from that format
  is a straightforward alternative to scraping if v1 needs more data.
- The `rating` field on a replay is the loser's post-game Elo; `null`
  for unrated games (`--min_rating 0` keeps them).
- Replays are observer-view (no `|request|` lines): hidden information
  such as the opponent's full moveset is inferred per-turn by
  `protocol.py`, which is expected.
