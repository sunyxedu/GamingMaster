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
//!
//! [`index`] (`gmdata generals index`) saves the whole index instead — every game's
//! metadata, ~1.75 M duel games from 2024-08 on (2026-09) — for the star statistics that
//! choose the thresholds; `ladder --index` then selects from that file.

use super::{api, gio, gior::Replay, sim};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

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
    /// the games of a saved index ([`index`]) instead of the live one
    pub index: Option<PathBuf>,
    /// imitate every player at `min_stars` or more whose opponent has
    /// `min_opponent_stars` or more — the winner and the loser alike
    pub each: bool,
    pub min_opponent_stars: i64,
    /// 1 game in N (by id) is written to `training_val/` instead of `training/`
    pub val_every: u64,
    /// write gm-shards (`<dir>/train`, `<dir>/val`) instead of gio JSON records — a gio
    /// record is ~15 MB of JSON, its shard sample ~1.5 MB; a game already in the manifest
    /// is skipped (a resumed run appends shards)
    pub shards: Option<PathBuf>,
}

/// What one game produced: gio files written / already there, or shard samples.
enum Made {
    Files { written: usize, existing: usize },
    Samples(Vec<(gmshard::Sample, Value, usize)>),
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
    let cands: Vec<Value> = if a.each {
        (0..2).filter(|&i| stars(&ranking[i]) >= a.min_stars && stars(&ranking[1 - i]) >= a.min_opponent_stars)
            .map(|i| ranking[i].clone()).collect()
    } else if a.both {
        ranking.clone()
    } else {
        vec![winner]
    };
    if cands.is_empty() || cands.iter().any(|p| stars(p) < a.min_stars) {
        return Vec::new();
    }
    if !a.allow_bots && ranking.iter().any(|p| bots.is_bot(c, &name(p))) {
        return Vec::new();
    }
    cands.iter().map(name).collect()
}

/// The imitated player's land and army at ticks 50 / 100 / 200 (turns 25 / 50 / 100) and
/// the game's length: the opening's efficiency, for star-level comparisons.
fn summary(s: &sim::SimOutput) -> Value {
    let at = |v: &[i64]| [50usize, 100, 200].map(|t| v.get(t).copied());
    let (land, army) = (at(&s.info_my_land), at(&s.info_my_army));
    json!({"land": land, "army": army, "ticks": s.info_my_land.len(), "won": s.winner.as_deref() == Some(s.player.as_str())})
}

/// The games of a saved index, deduplicated by id, in pages of 200 (the file's order:
/// newest first).
fn index_pages(path: &Path) -> Result<Vec<Vec<Value>>> {
    let mut seen = HashSet::new();
    let mut games = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
        let Ok(g) = serde_json::from_str::<Value>(&line?) else { continue };
        if g["id"].as_str().is_some_and(|id| seen.insert(id.to_string())) {
            games.push(g);
        }
    }
    Ok(games.chunks(200).map(|c| c.to_vec()).collect())
}

pub fn run(a: &LadderArgs) -> Result<Value> {
    use rayon::prelude::*;
    let c = client()?;
    let (replays, training, training_val) = (a.out.join("replays"), a.out.join("training"), a.out.join("training_val"));
    std::fs::create_dir_all(&replays)?;
    if a.shards.is_none() {
        std::fs::create_dir_all(&training)?;
        if a.val_every > 0 {
            std::fs::create_dir_all(&training_val)?;
        }
    }
    let mut bots = BotCache::load(a.out.join("botcache.json"));
    let manifest_path = a.out.join("manifest.jsonl");
    // shards mode resumes by the manifest: its games are in the shards already
    let mut done_ids: HashSet<String> = HashSet::new();
    if a.shards.is_some() && manifest_path.exists() {
        for line in std::io::BufReader::new(std::fs::File::open(&manifest_path)?).lines() {
            if let Some(id) = serde_json::from_str::<Value>(&line?).ok().and_then(|v| v["id"].as_str().map(String::from)) {
                done_ids.insert(id);
            }
        }
    }
    let mut manifest = std::fs::OpenOptions::new().create(true).append(true).open(&manifest_path)?;
    let mut writers: [Option<gmshard::ShardWriter>; 2] = [None, None];
    let mut shard_meta: Option<Value> = None;
    let saved = a.index.as_deref().map(index_pages).transpose()?;
    let (mut kept, mut seen, mut rejected, mut records, mut sim_skip, mut existing) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut frames, mut records_val, mut meta_mismatch, mut convert_failed) = (0usize, 0usize, 0usize, 0usize);
    for page in 0..a.max_pages {
        if kept >= a.max_games {
            break;
        }
        let games: Vec<Value> = match &saved {
            Some(pages) => match pages.get(page) {
                Some(p) => p.clone(),
                None => break,
            },
            None => {
                let url = format!("https://generals.io/api/replays?count=200&offset={}&l={}", page * 200, a.ladder);
                match c.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.text()) {
                    Ok(t) => serde_json::from_str(&t).unwrap_or_default(),
                    Err(e) => {
                        eprintln!("  index page {page} failed: {e}");
                        break;
                    }
                }
            }
        };
        if games.is_empty() {
            break;
        }
        let mut batch: Vec<(String, Vec<String>, Value)> = Vec::new();
        for g in games {
            seen += 1;
            if g["id"].as_str().is_some_and(|id| done_ids.contains(id)) {
                existing += 1;
                kept += 1;
                continue;
            }
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
        // simulate + convert the batch in parallel; manifest lines and shard samples in batch order
        let done: Vec<(String, Vec<String>, Value, Made, usize, Value)> = batch.into_par_iter()
            .filter(|(id, _, _)| !failed.contains(id))
            .filter_map(|(id, who, meta)| {
                let r = std::fs::read(replays.join(format!("{id}.json"))).ok()
                    .and_then(|b| serde_json::from_slice::<Replay>(&b).ok())?;
                let dir = if a.val_every > 0 && crate::common::hash_bucket(&id, a.val_every) { &training_val } else { &training };
                let (mut written, mut have, mut skip) = (0usize, 0usize, 0usize);
                let mut samples = Vec::new();
                let mut sums = serde_json::Map::new();
                for player in &who {
                    let safe: String = player.chars().map(|ch| if ch.is_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' }).collect();
                    let name = format!("{id}__{safe}.json");
                    let path = dir.join(&name);
                    match sim::Simulator::new(&r, player, true) {
                        Ok(s) => {
                            let out = s.simulate();
                            sums.insert(player.clone(), summary(&out));
                            if a.shards.is_none() && path.exists() {
                                have += 1;
                                continue;
                            }
                            let g = gio::convert(&out, 4, true);
                            if a.shards.is_some() {
                                match super::shard::gio_to_sample(&g, &name) {
                                    Ok(x) => samples.push(x),
                                    Err(e) => eprintln!("  {name}: not converted: {e:#}"),
                                }
                                continue;
                            }
                            let tmp = path.with_extension("json.tmp");
                            let ok = std::fs::File::create(&tmp).map_err(anyhow::Error::from).and_then(|f| {
                                let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
                                serde_json::to_writer(&mut w, &g)?;
                                w.flush()?;
                                Ok(())
                            }).and_then(|_| std::fs::rename(&tmp, &path).map_err(anyhow::Error::from));
                            if ok.is_ok() {
                                written += 1;
                            }
                        }
                        Err(e) => {
                            skip += 1;
                            eprintln!("  skipped {id} ({player}): {e}");
                            break;
                        }
                    }
                }
                let made = if a.shards.is_some() { Made::Samples(samples) } else { Made::Files { written, existing: have } };
                Some((id, who, meta, made, skip, Value::Object(sums)))
            })
            .collect();
        for (id, who, meta, made, skip, sums) in done {
            sim_skip += skip;
            let n = match made {
                Made::Files { written, existing: have } => {
                    records += written;
                    existing += have;
                    if written + have > 0 {
                        kept += 1;
                    }
                    written
                }
                Made::Samples(samples) => {
                    let val = a.val_every > 0 && crate::common::hash_bucket(&id, a.val_every);
                    let mut n = 0;
                    for (sample, m, t) in samples {
                        if shard_meta.is_none() {
                            let root = a.shards.as_ref().unwrap();
                            writers[0] = Some(gmshard::ShardWriter::new(&root.join("train"), "generals",
                                                                        super::shard::CONVERTER_VERSION, m.clone(), 128)?);
                            writers[1] = Some(gmshard::ShardWriter::new(&root.join("val"), "generals",
                                                                        super::shard::CONVERTER_VERSION, m.clone(), 128)?);
                            shard_meta = Some(m.clone());
                        }
                        if Some(&m) != shard_meta.as_ref() {
                            meta_mismatch += 1;
                            continue;
                        }
                        writers[val as usize].as_mut().unwrap().push(sample)?;
                        n += 1;
                        frames += t;
                        if val {
                            records_val += 1;
                        } else {
                            records += 1;
                        }
                    }
                    if n == 0 {
                        convert_failed += 1;
                    } else {
                        kept += 1;
                    }
                    n
                }
            };
            if n > 0 {
                writeln!(manifest, "{}", json!({"id": id, "imitated": who, "meta": meta, "summary": sums}))?;
            }
        }
        manifest.flush()?;
        if page % 20 == 0 {
            eprintln!("  page {page}: seen {seen}, kept {kept}, training records {records} (+{records_val} val, {existing} existing)");
        }
    }
    let mut shards = [0usize; 2];
    for (i, w) in writers.iter_mut().enumerate() {
        if let Some(w) = w {
            w.flush()?;
            shards[i] = w.written_shards;
        }
    }
    Ok(json!({"seen": seen, "rejected": rejected, "kept": kept, "simulator_skipped": sim_skip, "training_records": records,
              "existing": existing, "val_records": records_val, "frames": frames, "shards": shards,
              "meta_mismatch": meta_mismatch, "convert_failed": convert_failed, "shard_meta": shard_meta}))
}

/// One page of a ladder's global index (newest first), retried with a growing pause.
fn index_page(c: &reqwest::blocking::Client, ladder: &str, page: usize) -> Result<Vec<Value>> {
    let url = format!("https://generals.io/api/replays?count=200&offset={}&l={ladder}", page * 200);
    let mut last = None;
    for attempt in 0..10u64 {
        match c.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.text()) {
            Ok(t) => match serde_json::from_str::<Value>(&t) {
                Ok(v) => return Ok(v.as_array().cloned().unwrap_or_default()),
                Err(e) => last = Some(anyhow::anyhow!("page {page}: JSON: {e}")),
            },
            Err(e) => last = Some(anyhow::anyhow!("page {page}: {e}")),
        }
        std::thread::sleep(std::time::Duration::from_secs(5 * (attempt + 1)));
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("page {page}: failed")))
}

/// Save the whole ladder index: every game's metadata (id, type, started, turns, ranking
/// with each player's stars at the time and kills), one JSON object per line in
/// `out/index.jsonl`. Pages are read in order, `parallel` at a time (the index grows at its
/// head while it is read — ~1 game a minute — which repeats games across page boundaries,
/// written once by id, and could skip only the few games that arrive between the reads of
/// two neighbouring pages of one batch). Resumable: `out/index.state.json` holds the next
/// page; `max_pages` bounds this run.
pub fn index(out: &Path, ladder: &str, max_pages: usize, parallel: usize) -> Result<Value> {
    std::fs::create_dir_all(out)?;
    let c = client()?;
    let (path, state) = (out.join("index.jsonl"), out.join("index.state.json"));
    let mut next = std::fs::read(&state).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["next_page"].as_u64()).unwrap_or(0) as usize;
    let mut ids: HashSet<String> = HashSet::new();
    if path.exists() {
        for line in std::io::BufReader::new(std::fs::File::open(&path)?).lines() {
            if let Some(id) = serde_json::from_str::<Value>(&line?).ok().and_then(|v| v["id"].as_str().map(String::from)) {
                ids.insert(id);
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(&path)?);
    let (mut written, mut repeated, mut pages, mut done) = (0usize, 0usize, 0usize, false);
    let k = parallel.max(1);
    'scan: while pages < max_pages {
        let n = k.min(max_pages - pages);
        let got: Vec<Result<Vec<Value>>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n).map(|i| {
                let c = &c;
                sc.spawn(move || index_page(c, ladder, next + i))
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("page thread panicked")))).collect()
        });
        for res in got {
            let mut games = res?;
            if games.is_empty() {
                // the end, unless the API hiccuped: ask once more
                std::thread::sleep(std::time::Duration::from_secs(5));
                games = index_page(&c, ladder, next)?;
                if games.is_empty() {
                    done = true;
                    break 'scan;
                }
            }
            for g in games {
                let Some(id) = g["id"].as_str() else { continue };
                if !ids.insert(id.to_string()) {
                    repeated += 1;
                    continue;
                }
                serde_json::to_writer(&mut f, &g)?;
                f.write_all(b"\n")?;
                written += 1;
            }
            f.flush()?;
            next += 1;
            pages += 1;
            std::fs::write(&state, serde_json::to_vec(&json!({"next_page": next, "done": false}))?)?;
            if pages % 200 == 0 {
                eprintln!("  index page {next}: {} games", ids.len());
            }
        }
    }
    std::fs::write(&state, serde_json::to_vec(&json!({"next_page": next, "done": done}))?)?;
    Ok(json!({"games": ids.len(), "written": written, "repeated": repeated, "pages": pages, "next_page": next, "done": done}))
}
