//! High-quality generals.io IL data from the global ladder index (not per account):
//!
//!   GET https://generals.io/api/replays?count=200&offset=N&l=duel
//!
//! lists every ladder game newest-first with both players' stars. Games are kept when
//! they are 1v1, long enough, and the imitated player(s) are strong humans:
//!   * `--min-stars`: stars of the imitated side (winner, or both with `--both`)
//!   * bots excluded via /api/starsAndRanks `isBot` (cached), unless `--allow-bots`
//! Each kept game is downloaded, simulated from the imitated perspective(s) with the
//! strict simulator, and written as a gio training record
//! `training/<id>__<player>.json` (one flat directory for SeqReplayDataset).

use super::{api, gio, gior::Replay, sim};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

const UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";

pub struct LadderArgs {
    pub out: PathBuf,
    pub ladder: String,
    pub server: String,
    pub min_stars: i64,
    pub both: bool,
    pub allow_bots: bool,
    pub min_turns: i64,
    pub max_games: usize,
    pub max_pages: usize,
    pub concurrency: usize,
}

fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder().user_agent(UA).timeout(std::time::Duration::from_secs(30)).build()?)
}

struct BotCache {
    path: PathBuf,
    map: HashMap<String, bool>,
}

impl BotCache {
    fn load(path: PathBuf) -> Self {
        let map = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        BotCache { path, map }
    }
    fn is_bot(&mut self, c: &reqwest::blocking::Client, name: &str) -> bool {
        if let Some(&b) = self.map.get(name) {
            return b;
        }
        let url = format!("https://generals.io/api/starsAndRanks?u={}", urlenc(name));
        let b = c.get(&url).send().ok().and_then(|r| r.text().ok()).and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["isBot"].as_bool()).unwrap_or(false);
        self.map.insert(name.to_string(), b);
        b
    }
    fn save(&self) {
        let _ = std::fs::write(&self.path, serde_json::to_vec_pretty(&self.map).unwrap_or_default());
    }
}

fn urlenc(s: &str) -> String {
    s.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
        _ => format!("%{b:02X}"),
    }).collect()
}

/// Names of the players to imitate in this game (empty = reject).
fn imitated(g: &Value, a: &LadderArgs, c: &reqwest::blocking::Client, bots: &mut BotCache) -> Vec<String> {
    let ranking = g["ranking"].as_array().cloned().unwrap_or_default();
    if g["type"] != "1v1" || ranking.len() != 2 || g["turns"].as_i64().unwrap_or(0) < a.min_turns {
        return Vec::new();
    }
    let stars = |p: &Value| p["stars"].as_i64().unwrap_or(0);
    let name = |p: &Value| p["name"].as_str().unwrap_or("").to_string();
    let winner = ranking.iter().find(|p| p["kills"].as_i64() == Some(1)).unwrap_or(&ranking[0]).clone();
    let cands: Vec<Value> = if a.both { ranking.clone() } else { vec![winner] };
    if cands.iter().any(|p| stars(p) < a.min_stars) {
        return Vec::new();
    }
    if !a.allow_bots && ranking.iter().any(|p| bots.is_bot(c, &name(p))) {
        return Vec::new();
    }
    cands.iter().map(name).collect()
}

pub fn run(a: &LadderArgs) -> Result<Value> {
    let c = client()?;
    let (replays, training) = (a.out.join("replays"), a.out.join("training"));
    std::fs::create_dir_all(&replays)?;
    std::fs::create_dir_all(&training)?;
    let mut bots = BotCache::load(a.out.join("botcache.json"));
    let mut manifest = std::fs::OpenOptions::new().create(true).append(true).open(a.out.join("manifest.jsonl"))?;
    let (mut kept, mut seen, mut rejected, mut records, mut sim_skip) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for page in 0..a.max_pages {
        if kept >= a.max_games {
            break;
        }
        let url = format!("https://generals.io/api/replays?count=200&offset={}&l={}", page * 200, a.ladder);
        let games: Vec<Value> = match c.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.text()) {
            Ok(t) => serde_json::from_str(&t).unwrap_or_default(),
            Err(e) => {
                eprintln!("  index page {page} failed: {e}");
                break;
            }
        };
        if games.is_empty() {
            break;
        }
        let mut batch: Vec<(String, Vec<String>, Value)> = Vec::new();
        for g in games {
            seen += 1;
            let who = imitated(&g, a, &c, &mut bots);
            if who.is_empty() {
                rejected += 1;
                continue;
            }
            if let Some(id) = g["id"].as_str() {
                batch.push((id.to_string(), who, g.clone()));
            }
            if kept + batch.len() >= a.max_games {
                break;
            }
        }
        bots.save();
        let ids: Vec<String> = batch.iter().map(|b| b.0.clone()).collect();
        let failed = api::download_replays(&ids, &replays, &a.server, a.concurrency)?;
        for (id, who, meta) in batch {
            if failed.contains(&id) {
                continue;
            }
            let Ok(r) = std::fs::read(replays.join(format!("{id}.json"))).map_err(anyhow::Error::from)
                .and_then(|b| serde_json::from_slice::<Replay>(&b).map_err(anyhow::Error::from)) else { continue };
            let mut ok_any = false;
            for player in &who {
                match sim::Simulator::new(&r, player, true) {
                    Ok(s) => {
                        let g = gio::convert(&s.simulate(), 4, true);
                        let safe: String = player.chars().map(|ch| if ch.is_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' }).collect();
                        let path = training.join(format!("{id}__{safe}.json"));
                        let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&path)?);
                        serde_json::to_writer(&mut w, &g)?;
                        w.flush()?;
                        records += 1;
                        ok_any = true;
                    }
                    Err(e) => {
                        sim_skip += 1;
                        eprintln!("  skipped {id} ({player}): {e}");
                        break;
                    }
                }
            }
            if ok_any {
                kept += 1;
                writeln!(manifest, "{}", json!({"id": id, "imitated": who, "meta": meta}))?;
            }
        }
        eprintln!("  page {page}: seen {seen}, kept {kept}, training records {records}");
    }
    Ok(json!({"seen": seen, "rejected": rejected, "kept": kept, "simulator_skipped": sim_skip, "training_records": records}))
}
