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
//! "meta": {...} (optional)}` — e.g. what a reference implementation or a scripted
//! bot writes; real sources get their own parser module (see `kuhn_poker.rs`).

use crate::cand::{self, ConvertOptions, Record};
use crate::common;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmrules::registry;
use gmrules::{Event, Rng};
use serde_json::{json, Value};
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
            let mut recs = Vec::with_capacity(episodes);
            for i in 0..episodes {
                let ev = g.random_trajectory(&config, rng.next_u64(), usize::MAX).map_err(anyhow::Error::msg)?;
                recs.push(Record { id: format!("random:{seed}:{i}"), events: ev, meta: Value::Null, seats: None, expected_returns: None });
            }
            let opts = ConvertOptions { shard_size, store_events, ..Default::default() };
            let st = cand::convert_dyn(&game, &config, recs.into_iter(), &out, &opts)?;
            println!("{}", serde_json::to_string(&st)?);
            Ok(())
        }
        Cmd::Convert { game, config, input, out, shard_size, store_events, allow_truncated } => {
            let files = common::find_files(&input, &[".jsonl", ".jsonl.gz"])?;
            anyhow::ensure!(!files.is_empty(), "no .jsonl files under {}", input.display());
            let mut recs = Vec::new();
            for path in files {
                let text = String::from_utf8(common::read_maybe_gz(std::fs::read(&path)?)?)?;
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                for (ln, line) in text.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let id = format!("{name}:{}", ln + 1);
                    recs.push(parse_jsonl_record(line, &id).with_context(|| id.clone())?);
                }
            }
            let opts = ConvertOptions { shard_size, store_events, allow_truncated };
            let st = cand::convert_dyn(&game, &config, recs.into_iter(), &out, &opts)?;
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
        let r = parse_jsonl_record(r#"{"id":"x","events":[[-1,2],[-1,0],[0,1],[1,0]],"returns":[1,-1]}"#, "f:1").unwrap();
        assert_eq!(r.events[2], Event::Act(0, 1));
        assert_eq!(r.expected_returns, Some(vec![1.0, -1.0]));
        assert!(parse_jsonl_record(r#"{"events":[[0]]}"#, "f").is_err());
        let dir = std::env::temp_dir().join(format!("gmdata-rules-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let st = cand::convert_dyn("kuhn_poker", "", vec![r].into_iter(), &dir, &ConvertOptions::default()).unwrap();
        assert_eq!((st.converted, st.samples, st.frames), (1, 2, 2));
        let sh = gmshard::ShardReader::open(&dir.join("kuhn_poker_00000")).unwrap();
        assert_eq!(sh.meta()["config"]["players"], 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
