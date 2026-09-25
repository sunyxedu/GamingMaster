//! `crdata` — Clash Royale replay data pipeline.
//!
//!   crdata fetch   --out data/cr/raw [--limit 3]
//!   crdata convert --input data/cr/raw --out data/cr/frames [--step-ticks 10] [--recent 8]
//!   crdata stats   --input data/cr/raw [--limit 1]
//!
//! Input: RoyaleAPI action replays (`royaleapi-battle-replay.v1` payloads in
//! Parquet). Output: per-player decision frames (`cr-frames.v1`, `.npy` shards)
//! that `data_processing/games/clash_royale/loader.py` turns into the
//! `model/games/clash_royale.py` observation / target dict.

#![allow(dead_code)]
mod catalog;
mod elixir;
mod fetch;
mod frames;
mod hand;
mod npy;
mod replay;
mod shard;

use anyhow::{Context, Result};
use catalog::{Catalog, Timeline};
use clap::{Parser, Subcommand};
use frames::FrameConfig;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "crdata", version, about = "Clash Royale replay → training frames")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the replay Parquet parts from Hugging Face (SHA-256 verified, resumable)
    Fetch {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = fetch::DEFAULT_REPO)]
        repo: String,
        /// also download the expanded `actions/` table (not needed for conversion)
        #[arg(long)]
        actions: bool,
        /// only the first N part files
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Convert Parquet parts to `cr-frames.v1` shards
    Convert {
        /// a directory (all `**/part-*.parquet` under it) or one Parquet file
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        step_ticks: u32,
        #[arg(long, default_value_t = 8)]
        recent: usize,
        #[arg(long, default_value_t = 512)]
        samples_per_shard: usize,
        /// catalog directory (cards.json + timeline.json); default: ../catalog next to this crate
        #[arg(long)]
        catalog: Option<PathBuf>,
        /// only the first N replays (smoke tests)
        #[arg(long)]
        limit: Option<usize>,
        /// keep only these game modes (comma separated, e.g. "Ranked,Ladder"); default: all
        #[arg(long)]
        game_modes: Option<String>,
        /// drop a player's sequence when its elixir reconstruction violates the rules more than this many times
        #[arg(long, default_value_t = 3)]
        max_elixir_violations: u32,
    },
    /// Summarise a Parquet part: modes, results, cards, event kinds, coordinate convention
    Stats {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long)]
        limit: Option<usize>,
    },
}

fn default_catalog_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../catalog")
}

fn load_catalog(dir: Option<PathBuf>) -> Result<(Catalog, Timeline)> {
    let dir = dir.unwrap_or_else(default_catalog_dir);
    Ok((Catalog::load(&dir.join("cards.json"))?, Timeline::load(&dir.join("timeline.json"))?))
}

fn parquet_parts(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if p.is_dir() {
                if p.file_name().map(|n| n == "actions").unwrap_or(false) {
                    continue;
                }
                walk(&p, out)?;
            } else if p.extension().map(|e| e == "parquet").unwrap_or(false) {
                out.push(p);
            }
        }
        Ok(())
    }
    walk(input, &mut out)?;
    out.sort();
    anyhow::ensure!(!out.is_empty(), "no .parquet files under {}", input.display());
    Ok(out)
}

fn cmd_convert(
    input: PathBuf, out: PathBuf, step_ticks: u32, recent: usize, samples_per_shard: usize, catalog: Option<PathBuf>,
    limit: Option<usize>, game_modes: Option<String>, max_viol: u32,
) -> Result<()> {
    let (cat, tl) = load_catalog(catalog)?;
    let cfg = FrameConfig { step_ticks, recent };
    let modes: Option<Vec<String>> = game_modes.map(|s| s.split(',').map(|m| m.trim().to_string()).collect());
    let parts = parquet_parts(&input)?;
    let mut writer = shard::ShardWriter::new(&out, samples_per_shard, step_ticks, recent, &cat.version, 0)?;
    let mut n_rows = 0usize;
    let mut n_parse_err = 0usize;
    let mut n_mode_skip = 0usize;
    let mut n_hand_fail = 0usize;
    let mut n_viol_drop = 0usize;
    let mut remaining = limit.unwrap_or(usize::MAX);
    let t0 = std::time::Instant::now();
    for (pi, part) in parts.iter().enumerate() {
        if remaining == 0 {
            break;
        }
        let rows = replay::read_rows(part).with_context(|| format!("read {}", part.display()))?;
        let rows: Vec<_> = rows.into_iter().take(remaining).collect();
        remaining -= rows.len();
        n_rows += rows.len();
        // parse + build in parallel, keep file order for deterministic shards
        let built: Vec<(Result<Vec<frames::Sample>>, Option<String>)> = rows
            .par_iter()
            .map(|(tag, payload)| match replay::parse_payload(tag, payload, &cat) {
                Ok(r) => {
                    if let Some(m) = &modes {
                        if !m.contains(&r.game_mode) {
                            return (Ok(Vec::new()), Some("mode".into()));
                        }
                    }
                    let s = frames::build_replay(&r, &cat, &tl, &cfg);
                    let n_hand_fail = 2 - s.len();
                    (Ok(s), if n_hand_fail > 0 { Some(format!("hand:{n_hand_fail}")) } else { None })
                }
                Err(e) => (Err(e), None),
            })
            .collect();
        for (res, note) in built {
            match note.as_deref() {
                Some("mode") => n_mode_skip += 1,
                Some(h) if h.starts_with("hand:") => n_hand_fail += h[5..].parse::<usize>().unwrap_or(1),
                _ => {}
            }
            match res {
                Ok(samples) => {
                    for s in samples {
                        if s.meta.elixir_violations > max_viol {
                            n_viol_drop += 1;
                            continue;
                        }
                        writer.push(s)?;
                    }
                }
                Err(_) => n_parse_err += 1,
            }
        }
        eprintln!(
            "[{}/{}] {} → {} samples, {} frames so far ({:.0}s)",
            pi + 1, parts.len(), part.file_name().unwrap().to_string_lossy(),
            writer.written_samples + 0, writer.written_frames, t0.elapsed().as_secs_f64()
        );
    }
    writer.flush()?;
    let summary = serde_json::json!({
        "format": "cr-frames.v1", "input": input, "parts": parts.len(), "rows": n_rows,
        "parse_errors": n_parse_err, "mode_skipped": n_mode_skip, "hand_inference_failed": n_hand_fail,
        "elixir_violation_dropped": n_viol_drop, "shards": writer.written_shards,
        "samples": writer.written_samples, "frames": writer.written_frames,
        "step_ticks": step_ticks, "recent": recent, "catalog_version": cat.version,
        "seconds": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("convert_summary.json"), serde_json::to_vec_pretty(&summary)?)?;
    eprintln!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn cmd_stats(input: PathBuf, catalog: Option<PathBuf>, limit: Option<usize>) -> Result<()> {
    let (cat, _tl) = load_catalog(catalog)?;
    let parts = parquet_parts(&input)?;
    let mut modes: BTreeMap<String, usize> = BTreeMap::new();
    let mut results: BTreeMap<i8, usize> = BTreeMap::new();
    let mut cards: BTreeMap<u16, usize> = BTreeMap::new();
    let (mut plays, mut abilities, mut unknown_cards) = (0usize, 0usize, 0usize);
    let (mut team_low, mut team_high, mut opp_low, mut opp_high) = (0usize, 0usize, 0usize, 0usize);
    let mut durations = Vec::new();
    let mut n = 0usize;
    for part in parts.iter().take(limit.unwrap_or(usize::MAX)) {
        for (tag, payload) in replay::read_rows(part)? {
            let Ok(r) = replay::parse_payload(&tag, &payload, &cat) else { continue };
            n += 1;
            *modes.entry(r.game_mode.clone()).or_default() += 1;
            *results.entry(r.result).or_default() += 1;
            durations.push(r.duration_s);
            for c in r.team.deck.iter().chain(r.opp.deck.iter()) {
                *cards.entry(c.id).or_default() += 1;
                if c.id == cat.unknown_id {
                    unknown_cards += 1;
                }
            }
            for e in &r.events {
                match e.kind {
                    replay::EventKind::Play { gy, .. } => {
                        plays += 1;
                        match (e.side, gy < 16) {
                            (replay::SideId::Team, true) => team_low += 1,
                            (replay::SideId::Team, false) => team_high += 1,
                            (replay::SideId::Opp, true) => opp_low += 1,
                            (replay::SideId::Opp, false) => opp_high += 1,
                        }
                    }
                    replay::EventKind::Ability { .. } => abilities += 1,
                }
            }
        }
    }
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = durations.get(durations.len() / 2).copied().unwrap_or(0.0);
    let out = serde_json::json!({
        "replays": n, "game_modes": modes, "results_team": results,
        "distinct_cards": cards.len(), "unknown_card_slots": unknown_cards,
        "plays": plays, "abilities": abilities,
        "team_plays_in_rows_0_15": team_low, "team_plays_in_rows_16_31": team_high,
        "opp_plays_in_rows_0_15": opp_low, "opp_plays_in_rows_16_31": opp_high,
        "duration_s": {"min": durations.first(), "median": med, "max": durations.last()},
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Fetch { out, repo, actions, limit } => fetch::fetch(&repo, &out, actions, limit),
        Cmd::Convert { input, out, step_ticks, recent, samples_per_shard, catalog, limit, game_modes, max_elixir_violations } => {
            cmd_convert(input, out, step_ticks, recent, samples_per_shard, catalog, limit, game_modes, max_elixir_violations)
        }
        Cmd::Stats { input, catalog, limit } => cmd_stats(input, catalog, limit),
    }
}
