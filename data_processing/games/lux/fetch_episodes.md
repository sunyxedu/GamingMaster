# Getting Lux AI Season 3 episode replays

Competition: `lux-ai-season-3` (NeurIPS 2024/25 Lux AI Challenge).
All routes need Kaggle auth: put your API token in
`~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token) and
`pip install -U kaggle` — the `episodes`/`replay` subcommands used below
require a recent CLI.

After downloading, point the converter at the file or directory:

```sh
uv run python -m data_processing.games.lux.convert \
    --input replays/ --output_dir data/lux/train \
    --shard_size 64 --max_episodes 1000
```

## 1. Kaggle CLI chain: leaderboard → team submissions → episodes → replay

Pick strong teams off the leaderboard, list their submissions, list each
submission's episodes, then download replays by episode id:

```sh
# 1. Who is worth imitating (top of the leaderboard):
kaggle competitions leaderboard lux-ai-season-3 --show | head -30

# 2. A team's submissions (team id from the leaderboard listing):
kaggle competitions team-submissions lux-ai-season-3 <TEAM_ID>

# 3. Episodes played by one submission (submission id from step 2):
kaggle competitions episodes lux-ai-season-3 --submission <SUBMISSION_ID>

# 4. Download one replay JSON by episode id:
kaggle competitions replay <EPISODE_ID> > episode-<EPISODE_ID>-replay.json
```

Loop step 4 over the episode ids from step 3 for a per-team scrape.
(Replays are also served directly at
`https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay`
with body `{"episodeId": <id>}` — what the CLI calls under the hood —
useful if your CLI version predates the `replay` subcommand.)

## 2. Bulk: Meta Kaggle `Episodes.csv`

For large-scale scraping don't paginate the API — the
[Meta Kaggle dataset](https://www.kaggle.com/datasets/kaggle/meta-kaggle)
ships `Episodes.csv` + `EpisodeAgents.csv` covering every simulation
episode on the platform (refreshed daily):

```sh
kaggle datasets download kaggle/meta-kaggle -f Episodes.csv
kaggle datasets download kaggle/meta-kaggle -f EpisodeAgents.csv
```

Filter `Episodes.csv` by the Lux S3 `CompetitionId`, join
`EpisodeAgents.csv` for per-agent scores (pick high-`UpdatedScore`
agents ≈ high-Elo games), then fetch each surviving `Id` with step 4
above. Rate-limit the replay endpoint (~1 req/s) — episodes are ~5–15 MB
each.

## 3. Prepared replay datasets

Community-uploaded replay dumps appear under Kaggle Datasets (search
"lux ai season 3 episodes"); when available they are one
`kaggle datasets download <owner>/<slug>` instead of thousands of API
calls. Verify the JSON shape matches the converter's expectation
(top-level `steps`, per-agent `observation.obs` JSON string) — that is
the raw `GetEpisodeReplay` format.
