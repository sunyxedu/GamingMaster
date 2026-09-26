//! `gmdata rules …` — generic tools for EVERY game registered in gmrules (no per-game
//! code): list / conformance check / random-play shards / trajectory-JSONL conversion.
//!
//! ```sh
//! gmdata rules list
//! gmdata rules check   --game kuhn_poker --config '{"players":3}' --games 500
//! gmdata rules random  --game goofspiel --episodes 1000 --out data/goofspiel/random
//! gmdata rules convert --game kuhn_poker --input trajectories.jsonl --out data/kuhn_poker/x
//! ```
//!
//! Trajectory JSONL (one record per line, `.jsonl` / `.jsonl.gz`): `{"id": "...",
//! "events": [[seat, id], ...] (seat −1 = chance), "returns": [...] (optional, checked
//! against the replay), "seats": [...] (optional: only these seats' streams),
//! "meta": {...} (optional), "soft": [[event index, [[action, p], ...]], ...] (optional
//! teacher distributions of the Act events, written with `--soft-k K`)}` — e.g. what a
//! reference implementation, a solver or a scripted
//! bot writes; real sources get their own parser module (see `kuhn_poker.rs`).

use crate::cand::{self, ConvertOptions, Record};
use crate::common;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmrules::registry;
use gmrules::{Event, Rng};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Cmd {
    /// Registered games and their frame specs
    List,
    /// Run the conformance battery (causality, privacy, frame checks, simulator replays)
    Check {
        #[arg(long)]
        game: String,
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long, default_value_t = 500)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Uniformly random playouts → gm-shards (bootstrap data, tests)
    Random {
        #[arg(long)]
        game: String,
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long, default_value_t = 1000)]
        episodes: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        #[arg(long)]
        store_events: bool,
    },
    /// Trajectory JSONL → gm-shards (replayed and verified by the game's rules)
    Convert {
        #[arg(long)]
        game: String,
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        #[arg(long)]
        store_events: bool,
        /// keep records that stop before the end (tgt.value_mask false on them)
        #[arg(long)]
        allow_truncated: bool,
        /// write the records' "soft" teacher distributions as tgt.action_probs_idx/val
        /// with the K most probable legal actions per labelled decision (0 = none)
        #[arg(long, default_value_t = 0)]
        soft_k: usize,
    },
}

/// Parse one trajectory-JSONL line.
pub fn parse_jsonl_record(line: &str, fallback_id: &str) -> Result<Record> {
    let v: Value = serde_json::from_str(line)?;
    let events = v["events"]
        .as_array()
        .context("\"events\" must be an array of [seat, id] pairs")?
        .iter()
        .map(|p| {
            let (s, i) = (p.get(0).and_then(Value::as_i64), p.get(1).and_then(Value::as_i64));
            match (s, i) {
                (Some(s), Some(i)) => Event::from_pair(s, i).map_err(anyhow::Error::msg),
                _ => bail!("event {p} is not [seat, id]"),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let floats = |k: &str| v.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>());
    Ok(Record {
        id: v.get("id").and_then(Value::as_str).map_or_else(|| fallback_id.to_string(), String::from),
        events,
        meta: v.get("meta").cloned().unwrap_or(Value::Null),
        seats: v.get("seats").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|s| s as usize).collect()),
        expected_returns: floats("returns"),
        soft_targets: match v.get("soft") {
            None | Some(Value::Null) => Vec::new(),
            Some(soft) => soft
                .as_array()
                .context("\"soft\" must be [[event index, [[action, p], ...]], ...]")?
                .iter()
                .map(|e| {
                    let ev = e.get(0).and_then(Value::as_u64).context("soft entry: event index")?;
                    let dist = e
                        .get(1)
                        .and_then(Value::as_array)
                        .context("soft entry: [[action, p], ...]")?
                        .iter()
                        .map(|ap| match (ap.get(0).and_then(Value::as_u64), ap.get(1).and_then(Value::as_f64)) {
                            (Some(a), Some(p)) => Ok((a as u32, p as f32)),
                            _ => bail!("soft entry {ap}: [action, p] expected"),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok((ev as usize, dist))
                })
                .collect::<Result<Vec<_>>>()?,
        },
    })
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::List => {
            for name in registry::game_names() {
                let g = registry::lookup(name).map_err(anyhow::Error::msg)?;
                let spec = g.spec("").map_err(anyhow::Error::msg)?;
                println!("{}", json!({"game": name, "version": g.version(), "default_config": g.normalize_config("").map_err(anyhow::Error::msg)?, "spec": spec}));
            }
            Ok(())
        }
        Cmd::Check { game, config, games, seed } => {
            let g = registry::lookup(&game).map_err(anyhow::Error::msg)?;
            let rep = g.check(&config, games, seed).map_err(anyhow::Error::msg)?;
            println!("{game}: ok {rep}");
            Ok(())
        }
        Cmd::Random { game, config, episodes, seed, out, shard_size, store_events } => {
            let g = registry::lookup(&game).map_err(anyhow::Error::msg)?;
            let mut rng = Rng::new(seed);
            let recs = (0..episodes).map(|i| {
                let ev = g.random_trajectory(&config, rng.next_u64(), usize::MAX).map_err(anyhow::Error::msg)?;
                Ok(Record { id: format!("random:{seed}:{i}"), events: ev, ..Default::default() })
            });
            let opts = ConvertOptions { shard_size, store_events, ..Default::default() };
            let st = cand::convert_dyn(&game, &config, recs, &out, &opts)?;
            println!("{}", serde_json::to_string(&st)?);
            Ok(())
        }
        Cmd::Convert { game, config, input, out, shard_size, store_events, allow_truncated, soft_k } => {
            let files = common::find_files(&input, &[".jsonl", ".jsonl.gz"])?;
            anyhow::ensure!(!files.is_empty(), "no .jsonl files under {}", input.display());
            // streamed file by file, line by line
            let recs = files.into_iter().flat_map(|path| -> Box<dyn Iterator<Item = Result<Record>>> {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                let lines = match std::fs::File::open(&path) {
                    Ok(f) => {
                        let gz = name.to_lowercase().ends_with(".gz");
                        let r: Box<dyn Read> = if gz { Box::new(flate2::read::MultiGzDecoder::new(f)) } else { Box::new(f) };
                        BufReader::new(r).lines()
                    }
                    Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::from(e).context(name)))),
                };
                Box::new(lines.enumerate().filter_map(move |(ln, line)| {
                    let id = format!("{name}:{}", ln + 1);
                    match line {
                        Ok(l) if l.trim().is_empty() => None,
                        Ok(l) => Some(parse_jsonl_record(&l, &id).with_context(|| id.clone())),
                        Err(e) => Some(Err(anyhow::Error::from(e).context(id))),
                    }
                }))
            });
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, soft_k };
            let st = cand::convert_dyn(&game, &config, recs, &out, &opts)?;
            println!("{}", serde_json::to_string(&st)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_records_convert_by_name() {
        let r = parse_jsonl_record(r#"{"id":"x","events":[[-1,2],[-1,0],[0,1],[1,0]],"returns":[1,-1],"soft":[[2,[[1,0.9],[0,0.1]]]]}"#, "f:1").unwrap();
        assert_eq!(r.soft_targets, vec![(2, vec![(1, 0.9), (0, 0.1)])]);
        assert!(parse_jsonl_record(r#"{"events":[],"soft":[[2,[[1]]]]}"#, "f").is_err());
        assert_eq!(r.events[2], Event::Act(0, 1));
        assert_eq!(r.expected_returns, Some(vec![1.0, -1.0]));
        assert!(parse_jsonl_record(r#"{"events":[[0]]}"#, "f").is_err());
        let dir = std::env::temp_dir().join(format!("gmdata-rules-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let st = cand::convert_dyn("kuhn_poker", "", vec![Ok(r)].into_iter(), &dir, &ConvertOptions::default()).unwrap();
        assert_eq!((st.converted, st.samples, st.frames), (1, 2, 2));
        let sh = gmshard::ShardReader::open(&dir.join("kuhn_poker_00000")).unwrap();
        assert_eq!(sh.meta()["config"]["players"], 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
