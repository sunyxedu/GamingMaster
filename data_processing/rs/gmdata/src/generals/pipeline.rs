//! End-to-end per-player pipeline: fetch metadata → filter → download →
//! simulate → convert, until `target` trainable games exist. Same layout,
//! manifest and statuses as the Python `scripts/prepare_data.py`, so runs
//! are idempotent and resumable and `filtered/<player>.json` keeps feeding
//! `train_rl --filtered_file`.
//!
//!     output/
//!       manifest/<player>.json        {"fetched_offset": N, "games": {id: {status, meta, reason?}}}
//!       replays/<id>.json             raw replays
//!       filtered/<player>.json        trainable-game list
//!       training/<player>/<id>.json   gio training records

use super::{api, filter, gio, gior::Replay, sim};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const FILTER_START_DATE: &str = "2025-04-01T00:00:00Z";
const PAGE: usize = 200;

pub struct PipelineArgs {
    pub target: usize,
    pub min_turns: i64,
    pub min_opponent_stars: Option<i64>,
    pub start_date: Option<String>,
    pub workers: usize,
    pub max_concurrent: usize,
    pub server: String,
    pub max_actions_per_tick: usize,
}

struct Manifest {
    path: PathBuf,
    offset: usize,
    games: Map<String, Value>,
}

impl Manifest {
    fn load(path: PathBuf) -> Result<Self> {
        if path.exists() {
            let v: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            Ok(Self {
                path,
                offset: v.get("fetched_offset").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                games: v.get("games").and_then(|g| g.as_object()).cloned().unwrap_or_default(),
            })
        } else {
            Ok(Self { path, offset: 0, games: Map::new() })
        }
    }

    fn save(&self) -> Result<()> {
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&json!({"fetched_offset": self.offset, "games": self.games}))?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    fn set(&mut self, id: &str, status: &str, reason: Option<String>) {
        if let Some(g) = self.games.get_mut(id).and_then(|g| g.as_object_mut()) {
            g.insert("status".into(), Value::from(status));
            match reason {
                Some(r) => {
                    g.insert("reason".into(), Value::from(r));
                }
                None => {
                    g.remove("reason");
                }
            }
        }
    }

    fn trainable(&self) -> usize {
        self.games.values().filter(|g| matches!(g.get("status").and_then(|s| s.as_str()), Some("converted") | Some("simulated"))).count()
    }

    fn by_status(&self, statuses: &[&str]) -> Vec<String> {
        self.games.iter().filter(|(_, g)| statuses.contains(&g.get("status").and_then(|s| s.as_str()).unwrap_or(""))).map(|(k, _)| k.clone()).collect()
    }
}

/// simulate + convert one game; Err(permanent) → skipped, Err(transient) → error.
pub enum GameOutcome {
    Converted,
    Skipped(String),
    Error(String),
}

pub fn process_game(replay_path: &Path, player: &str, training_path: &Path, k: usize) -> GameOutcome {
    let replay: Replay = match std::fs::read(replay_path).context("read replay").and_then(|b| serde_json::from_slice(&b).context("parse replay")) {
        Ok(r) => r,
        Err(e) => return GameOutcome::Error(e.to_string()),
    };
    let simulator = match sim::Simulator::new(&replay, player, true) {
        Ok(s) => s,
        Err(e) => return GameOutcome::Skipped(e.to_string()),
    };
    let out = simulator.simulate();
    let g = gio::convert(&out, k, true);
    let tmp = training_path.with_extension("tmp");
    let write = || -> std::result::Result<(), String> {
        use std::io::Write;
        let f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        serde_json::to_writer(&mut w, &g).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())
    };
    match write() {
        Ok(()) => match std::fs::rename(&tmp, training_path) {
            Ok(()) => GameOutcome::Converted,
            Err(e) => GameOutcome::Error(e.to_string()),
        },
        Err(e) => GameOutcome::Error(e),
    }
}

pub fn run_player(player: &str, output: &Path, a: &PipelineArgs) -> Result<()> {
    let manifest_dir = output.join("manifest");
    let replays_dir = output.join("replays");
    let training_dir = output.join("training").join(player);
    let filtered_dir = output.join("filtered");
    for d in [&manifest_dir, &replays_dir, &training_dir, &filtered_dir] {
        std::fs::create_dir_all(d)?;
    }
    let mut m = Manifest::load(manifest_dir.join(format!("{player}.json")))?;
    let flt = filter::Filter {
        min_turns: a.min_turns,
        player_name: Some(player.to_string()),
        min_opponent_stars: a.min_opponent_stars,
        start_date: Some(a.start_date.clone().unwrap_or_else(|| FILTER_START_DATE.to_string())),
        ..Default::default()
    };
    eprintln!("\n=== {player}: {}/{} trainable ===", m.trainable(), a.target);
    for id in m.by_status(&["error"]) {
        m.set(&id, "pending", None);
    }
    let client = api::client()?;
    let mut exhausted = false;
    loop {
        if m.trainable() >= a.target {
            break;
        }
        let mut pending = m.by_status(&["pending"]);
        if pending.is_empty() {
            if exhausted {
                break;
            }
            let page = match api::fetch_page(&client, player, m.offset, PAGE) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  fetch failed ({e}) — continuing without new games");
                    exhausted = true;
                    continue;
                }
            };
            if page.is_empty() {
                exhausted = true;
                continue;
            }
            let n = page.len();
            m.offset += n;
            for entry in page.iter().filter(|e| filter::passes(e, &flt)) {
                if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                    if !m.games.contains_key(id) {
                        m.games.insert(id.to_string(), json!({"status": "pending", "meta": entry}));
                    }
                }
            }
            if n < PAGE {
                exhausted = true;
            }
            eprintln!("  fetched {} games, candidates: {}", m.offset, m.by_status(&["pending"]).len());
            m.save()?;
            continue;
        }
        // take a batch: at most what is still needed (plus slack for rejections)
        let need = a.target - m.trainable();
        // just what is still needed (+ slack for simulator rejections), capped at 2×workers
        pending.truncate((need + need / 4 + 1).min(2 * a.workers).max(1));
        let missing: Vec<String> = pending.iter().filter(|g| !replays_dir.join(format!("{g}.json")).exists()).cloned().collect();
        let failed = if missing.is_empty() { Vec::new() } else { api::download_replays(&missing, &replays_dir, &a.server, a.max_concurrent)? };
        for id in &failed {
            m.set(id, "error", Some("download failed".into()));
        }
        let ready: Vec<String> = pending.into_iter().filter(|g| !failed.contains(g)).collect();
        let k = a.max_actions_per_tick;
        let results: Vec<(String, GameOutcome)> = ready
            .par_iter()
            .map(|id| {
                let out = process_game(&replays_dir.join(format!("{id}.json")), player, &training_dir.join(format!("{id}.json")), k);
                (id.clone(), out)
            })
            .collect();
        for (id, out) in results {
            match out {
                GameOutcome::Converted => m.set(&id, "converted", None),
                GameOutcome::Skipped(r) => {
                    eprintln!("  skipped {id}: {r}");
                    m.set(&id, "skipped", Some(r));
                }
                GameOutcome::Error(r) => {
                    eprintln!("  error {id}: {r}");
                    m.set(&id, "error", Some(r));
                }
            }
        }
        m.save()?;
        eprintln!("  trainable {}/{}", m.trainable(), a.target);
    }
    m.save()?;
    if m.trainable() < a.target {
        eprintln!("  history exhausted: {}/{} trainable", m.trainable(), a.target);
    }
    let trainable_meta: Vec<Value> = m
        .games
        .values()
        .filter(|g| matches!(g.get("status").and_then(|s| s.as_str()), Some("converted") | Some("simulated")))
        .filter_map(|g| g.get("meta").cloned())
        .collect();
    std::fs::write(filtered_dir.join(format!("{player}.json")), serde_json::to_vec_pretty(&trainable_meta)?)?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for g in m.games.values() {
        *counts.entry(g.get("status").and_then(|s| s.as_str()).unwrap_or("?").to_string()).or_default() += 1;
    }
    eprintln!("  done: {counts:?}");
    Ok(())
}
