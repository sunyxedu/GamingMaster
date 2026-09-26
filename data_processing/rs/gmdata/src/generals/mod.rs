//! generals.io: metadata API → `.gior` replays → per-player fog-of-war
//! simulation → `gio` training records (→ `shard`: gm-shards for the
//! multi-game trainers).

pub mod api;
pub mod filter;
pub mod gio;
pub mod gior;
pub mod ladder;
pub mod lz;
pub mod pipeline;
pub mod shard;
pub mod sim;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Cmd {
    /// Fetch replay metadata for a player → JSON list
    FetchList {
        #[arg(long)]
        player: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10000)]
        total: usize,
    },
    /// Download `.gior` replays by id → `<out>/<id>.json`
    Download {
        /// replay ids, or a metadata JSON list file (`--ids @file.json`)
        #[arg(long, num_args = 1..)]
        ids: Vec<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "na")]
        server: String,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
    },
    /// Decode one `.gior` file to replay JSON (stdout)
    Decode { path: PathBuf },
    /// Simulate one replay JSON from a player's perspective → simulator JSON
    Simulate {
        #[arg(long)]
        replay: PathBuf,
        #[arg(long)]
        player: String,
        #[arg(long)]
        out: PathBuf,
        /// keep replays the simulator cannot reproduce faithfully (viewers only)
        #[arg(long)]
        lenient: bool,
    },
    /// Simulate + convert one replay JSON → gio training JSON
    Convert {
        #[arg(long)]
        replay: PathBuf,
        #[arg(long)]
        player: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_actions_per_tick: usize,
    },
    /// High-quality IL data from the global ladder: strong human players only
    Ladder {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "duel")]
        ladder: String,
        #[arg(long, default_value = "na")]
        server: String,
        /// stars required of the imitated side (winner, or both with --both)
        #[arg(long, default_value_t = 70)]
        min_stars: i64,
        /// imitate both players (both must reach --min-stars)
        #[arg(long)]
        both: bool,
        #[arg(long)]
        allow_bots: bool,
        #[arg(long, default_value_t = 50)]
        min_turns: i64,
        #[arg(long, default_value_t = 1000)]
        max_games: usize,
        #[arg(long, default_value_t = 10000)]
        max_pages: usize,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
    },
    /// End-to-end resumable pipeline per player (fetch → filter → download → simulate → convert)
    Pipeline(PipelineCli),
    /// gio training records → gm-shard directories (GameShardDataset, train_multi_game)
    Shard {
        /// gio record directories (`*.json`, searched recursively) or files, in order
        #[arg(long, num_args = 1..)]
        input: Vec<PathBuf>,
        /// output directory (use a fresh one: shard numbering continues after existing shards)
        #[arg(long)]
        out: PathBuf,
        /// records per shard directory
        #[arg(long, default_value_t = 128)]
        shard_size: usize,
        /// seed of the record shuffle before writing (the trainers read a shard at a time)
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// write records in path order instead of shuffled
        #[arg(long)]
        keep_order: bool,
    },
}

#[derive(Args)]
pub struct PipelineCli {
    #[arg(long, short = 'p', num_args = 1..)]
    players: Vec<String>,
    #[arg(long, short = 'o')]
    output: PathBuf,
    #[arg(long, default_value_t = 1000)]
    target: usize,
    #[arg(long, default_value_t = 80)]
    min_turns: i64,
    #[arg(long)]
    min_opponent_stars: Option<i64>,
    /// ISO date; default 2025-04-01T00:00:00Z (the reference pipeline's cutoff)
    #[arg(long)]
    start_date: Option<String>,
    #[arg(long)]
    workers: Option<usize>,
    #[arg(long, default_value_t = 8)]
    max_concurrent: usize,
    #[arg(long, default_value = "na")]
    server: String,
    #[arg(long, default_value_t = 4)]
    max_actions_per_tick: usize,
    /// "k/n": this job's shard (Slurm array: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT)
    #[arg(long, default_value = "0/1")]
    shard: String,
    /// merge per-shard filtered lists into filtered/<player>.json and exit
    #[arg(long)]
    merge: bool,
}

fn write_json<T: serde::Serialize>(path: &std::path::Path, v: &T) -> Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    serde_json::to_writer(&mut w, v)?;
    w.flush()?;
    Ok(())
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::FetchList { player, out, total } => {
            let list = api::fetch_list(&api::client()?, &player, total, 200)?;
            std::fs::write(&out, serde_json::to_vec_pretty(&list)?)?;
            eprintln!("saved {} replays for {player} to {}", list.len(), out.display());
        }
        Cmd::Download { ids, out, server, concurrency } => {
            let mut all = Vec::new();
            for id in ids {
                if let Some(f) = id.strip_prefix('@') {
                    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(f)?)?;
                    for e in v.as_array().context("metadata list")? {
                        if let Some(i) = e.get("id").and_then(|x| x.as_str()) {
                            all.push(i.to_string());
                        }
                    }
                } else {
                    all.push(id);
                }
            }
            let failed = api::download_replays(&all, &out, &server, concurrency)?;
            eprintln!("downloaded {} / {} (failed {})", all.len() - failed.len(), all.len(), failed.len());
        }
        Cmd::Decode { path } => {
            let r = gior::parse_gior(&std::fs::read(&path)?)?;
            println!("{}", serde_json::to_string(&r)?);
        }
        Cmd::Simulate { replay, player, out, lenient } => {
            let r: gior::Replay = serde_json::from_slice(&std::fs::read(&replay)?)?;
            let s = sim::Simulator::new(&r, &player, !lenient)?.simulate();
            write_json(&out, &s)?;
        }
        Cmd::Convert { replay, player, out, max_actions_per_tick } => {
            let r: gior::Replay = serde_json::from_slice(&std::fs::read(&replay)?)?;
            let s = sim::Simulator::new(&r, &player, true)?.simulate();
            let g = gio::convert(&s, max_actions_per_tick, true);
            write_json(&out, &g)?;
        }
        Cmd::Ladder { out, ladder, server, min_stars, both, allow_bots, min_turns, max_games, max_pages, concurrency } => {
            let a = ladder::LadderArgs { out, ladder, server, min_stars, both, allow_bots, min_turns, max_games, max_pages, concurrency };
            println!("{}", ladder::run(&a)?);
        }
        Cmd::Shard { input, out, shard_size, seed, keep_order } => {
            let order = if keep_order { None } else { Some(seed) };
            println!("{}", shard::shard(&input, &out, shard_size, order)?);
        }
        Cmd::Pipeline(p) => {
            if p.merge {
                for player in &p.players {
                    let n = pipeline::merge_shards(player, &p.output)?;
                    eprintln!("{player}: merged {n} trainable games");
                }
                return Ok(());
            }
            let (k, n) = p.shard.split_once('/').and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
                .filter(|(k, n)| *n >= 1 && k < n).ok_or_else(|| anyhow::anyhow!("--shard must be k/n with 0 <= k < n"))?;
            let args = pipeline::PipelineArgs {
                target: p.target,
                min_turns: p.min_turns,
                min_opponent_stars: p.min_opponent_stars,
                start_date: p.start_date,
                workers: p.workers.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)),
                max_concurrent: p.max_concurrent,
                server: p.server,
                max_actions_per_tick: p.max_actions_per_tick,
                shard: (k, n),
            };
            for player in &p.players {
                pipeline::run_player(player, &p.output, &args)?;
            }
        }
    }
    Ok(())
}
