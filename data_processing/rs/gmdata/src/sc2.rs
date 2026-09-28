//! StarCraft II replays: the replay packs Blizzard publishes for machine learning (the
//! anonymised 1v1 ladder replays of a client build — 3.16.1 Pack 1 / Pack 2 on the
//! cluster, `~/gm-data/sc2/`; the Blizzard AI and Machine Learning licence: research, not
//! commercial).  A replay replays only on the client build that recorded it, so every
//! command runs headless SC2 processes of that build (the `sc2api` crate).
//!
//! `scan`: each replay's metadata through the game itself (`RequestReplayInfo`) — map,
//! duration, build, and per player the race, the result, the MMR and the APM the ladder
//! recorded — as JSON lines, the input of the quality selection (MMR is the ladder's skill
//! rating at the game; AlphaStar's supervised data kept replays whose players had MMR >
//! 3500, the top ~22% of the ladder then).
//!
//! `data`: the build's vocabulary (gmsc2::Vocab) from the game data of a running replay.
//!
//! `convert`: the selected replays (the scan: both players' MMR, APM, a decided game of some
//! length) from each player's view → gm-shards of gmsc2's decisions (its module docs: one
//! frame per 8-loop window, every command of the window a decision on it, then "done").
//! Storage bounds the frames kept: per player-game `--chunks` runs of `--chunk-windows`
//! consecutive windows, one in each equal part of the game (early … late; a stable hash picks
//! the start), each run one sample; between runs the replay steps `--skip-windows` windows at
//! a time, observing only to keep the unit slots.  Every run of a replay goes to `train` or
//! (every `--val-every`-th replay by a hash of its file) `val`.  The replays go best first
//! (by the lower MMR of the two players).

use anyhow::{Context, Result};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use sc2api::{proto, Install, Sc2};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Shard schema version of `convert` (gmsc2's frames).
pub const CONVERTER_VERSION: u32 = 1;

#[derive(Subcommand)]
pub enum Cmd {
    /// Replay metadata (map, duration, races, results, MMR, APM) as JSON lines
    Scan {
        /// the SC2 install (the directory with Versions/, Libs/)
        #[arg(long)]
        install: PathBuf,
        /// replay files or directories (recursive)
        #[arg(long, required = true)]
        replays: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// SC2 processes in parallel
        #[arg(long, default_value_t = 4)]
        workers: usize,
        /// first websocket port (worker i listens on port + i)
        #[arg(long, default_value_t = 23000)]
        port: u16,
        #[arg(long)]
        max_replays: Option<usize>,
    },
    /// The client build's vocabulary (unit types, abilities, buffs, upgrades) as JSON: the
    /// game data of a running replay
    Data {
        #[arg(long)]
        install: PathBuf,
        /// any replay of the build
        #[arg(long)]
        replay: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 23400)]
        port: u16,
    },
    /// Replays → gm-shards of gmsc2's decisions (module docs)
    Convert {
        #[arg(long)]
        install: PathBuf,
        /// the scan's JSON lines (`sc2 scan`)
        #[arg(long)]
        scan: PathBuf,
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// both players' MMR at least
        #[arg(long, default_value_t = 5000)]
        min_mmr: i64,
        /// both players' APM at least (players who left or idled)
        #[arg(long, default_value_t = 30)]
        min_apm: i64,
        /// game time at least (seconds)
        #[arg(long, default_value_t = 120.0)]
        min_seconds: f64,
        /// runs of windows kept per player-game (0 = the whole game in one sample)
        #[arg(long, default_value_t = 3)]
        chunks: usize,
        #[arg(long, default_value_t = 48)]
        chunk_windows: usize,
        /// windows stepped at once between runs
        #[arg(long, default_value_t = 8)]
        skip_windows: usize,
        #[arg(long, default_value_t = 4)]
        workers: usize,
        #[arg(long, default_value_t = 23600)]
        port: u16,
        #[arg(long)]
        max_replays: Option<usize>,
        #[arg(long, default_value_t = 50)]
        val_every: u64,
        #[arg(long, default_value_t = 16)]
        shard_size: usize,
        /// convert only the selected replays i with i % parts == part (jobs side by side)
        #[arg(long, default_value_t = 0)]
        part: usize,
        #[arg(long, default_value_t = 1)]
        parts: usize,
    },
    /// Measure one replay through the game: observation cost, the actions it reports (kinds,
    /// loops, gaps), the frames and labels of gmsc2 — the numbers behind the frame design
    Probe {
        #[arg(long)]
        install: PathBuf,
        #[arg(long)]
        replay: PathBuf,
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long, default_value_t = 1)]
        player: i32,
        /// loops observed one by one from the start
        #[arg(long, default_value_t = 2000)]
        fine_loops: u32,
        /// the step after them
        #[arg(long, default_value_t = 8)]
        step: u32,
        #[arg(long, default_value_t = 40000)]
        max_loops: u32,
        #[arg(long, default_value_t = 23500)]
        port: u16,
    },
}

fn race(r: proto::Race) -> &'static str {
    match r {
        proto::Race::Terran => "terran",
        proto::Race::Zerg => "zerg",
        proto::Race::Protoss => "protoss",
        proto::Race::Random => "random",
        _ => "none",
    }
}

fn result(r: proto::Result) -> &'static str {
    match r {
        proto::Result::Victory => "victory",
        proto::Result::Defeat => "defeat",
        proto::Result::Tie => "tie",
        _ => "undecided",
    }
}

/// One replay's metadata record.
pub fn info_json(path: &Path, r: &proto::ResponseReplayInfo) -> Value {
    let players: Vec<Value> = r.player_info.iter().map(|p| {
        let pi = p.player_info.as_ref();
        json!({
            "player_id": pi.and_then(|x| x.player_id),
            "race_requested": pi.map(|x| race(x.race_requested())),
            "race": pi.map(|x| race(x.race_actual())),
            "result": p.player_result.as_ref().map(|x| result(x.result())),
            "mmr": p.player_mmr,
            "apm": p.player_apm,
        })
    }).collect();
    json!({
        "replay": path.display().to_string(),
        "map": r.map_name,
        "loops": r.game_duration_loops,
        "seconds": r.game_duration_seconds,
        "game_version": r.game_version,
        "base_build": r.base_build,
        "data_version": r.data_version,
        "players": players,
    })
}

pub fn scan(install: &Path, replays: &[PathBuf], out: &Path, workers: usize, port: u16, max: Option<usize>) -> Result<Value> {
    let mut files = Vec::new();
    for r in replays {
        files.extend(crate::common::find_files(r, &[".SC2Replay"])?);
    }
    if let Some(m) = max {
        files.truncate(m);
    }
    let inst = Install::find(install, None)?;
    let files = Arc::new(files);
    let next = Arc::new(AtomicUsize::new(0));
    let out_f = Arc::new(Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(out).with_context(|| format!("create {}", out.display()))?)));
    let (ok, bad) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let tmp = std::env::temp_dir().join(format!("gm-sc2-scan-{}", std::process::id()));
    std::thread::scope(|sc| -> Result<()> {
        let mut hs = Vec::new();
        for w in 0..workers.max(1) {
            let (files, next, out_f, ok, bad, inst, tmp) =
                (files.clone(), next.clone(), out_f.clone(), ok.clone(), bad.clone(), inst.clone(), tmp.join(w.to_string()));
            hs.push(sc.spawn(move || -> Result<()> {
                let mut sc2: Option<Sc2> = None;
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(path) = files.get(i) else { return Ok(()) };
                    if sc2.is_none() {
                        sc2 = Some(Sc2::launch(&inst, port + w as u16, &tmp, None)?);
                    }
                    let line = match sc2.as_mut().unwrap().replay_info(path) {
                        Ok(r) => {
                            ok.fetch_add(1, Ordering::SeqCst);
                            info_json(path, &r)
                        }
                        Err(e) => {
                            bad.fetch_add(1, Ordering::SeqCst);
                            // a broken replay can leave the process unusable: start a fresh one
                            sc2 = None;
                            json!({"replay": path.display().to_string(), "error": e.to_string()})
                        }
                    };
                    let mut f = out_f.lock().unwrap();
                    writeln!(f, "{line}")?;
                    let n = ok.load(Ordering::SeqCst) + bad.load(Ordering::SeqCst);
                    if n % 1000 == 0 {
                        f.flush()?;
                        eprintln!("[sc2 scan] {n}/{}", files.len());
                    }
                }
            }));
        }
        for h in hs {
            h.join().expect("a scan thread panicked")?;
        }
        Ok(())
    })?;
    out_f.lock().unwrap().flush()?;
    Ok(json!({"replays": files.len(), "ok": ok.load(Ordering::SeqCst), "errors": bad.load(Ordering::SeqCst),
              "build": inst.build}))
}

fn tmp_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("gm-sc2-{tag}-{}", std::process::id()))
}

pub fn data(install: &Path, replay: &Path, out: &Path, port: u16) -> Result<Value> {
    let inst = Install::find(install, None)?;
    let mut sc2 = Sc2::launch(&inst, port, &tmp_dir("data"), None)?;
    sc2.start_replay(replay, 1, None)?;
    let voc = gmsc2::Vocab::from_data(&sc2.data()?, inst.build);
    std::fs::write(out, serde_json::to_string(&voc.to_json())?)?;
    Ok(json!({"build": inst.build, "units": voc.units.len(), "actions": voc.actions.len(), "buffs": voc.buffs.len(),
              "upgrades": voc.upgrades.len(), "abilities": voc.general.len()}))
}

#[allow(clippy::too_many_arguments)]
pub fn probe(install: &Path, replay: &Path, vocab: &Path, player: i32, fine_loops: u32, step: u32, max_loops: u32, port: u16)
    -> Result<Value> {
    use gmsc2::{label, unit_commands, Encoder, MapInfo, Skip, Vocab};
    use std::collections::BTreeMap;
    use std::time::Instant;
    let voc = Vocab::load(vocab)?;
    let inst = Install::find(install, None)?;
    let mut sc2 = Sc2::launch(&inst, port, &tmp_dir("probe"), None)?;
    let info = sc2.replay_info(replay)?;
    let t0 = Instant::now();
    sc2.start_replay(replay, player, None)?;
    let start_s = t0.elapsed().as_secs_f64();
    let mut enc = Encoder::new(MapInfo::from_game_info(&sc2.game_info()?), player as u32);
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut with_loop, mut without_loop) = (0usize, 0usize);
    let mut per_obs: BTreeMap<usize, usize> = BTreeMap::new();
    let mut abilities: BTreeMap<String, usize> = BTreeMap::new();
    let mut labels: BTreeMap<&str, usize> = BTreeMap::new();
    let mut targets: BTreeMap<i64, usize> = BTreeMap::new();
    let (mut obs_s, mut step_s, mut enc_s) = (0f64, 0f64, 0f64);
    let (mut n_obs, mut units_max, mut units_sum, mut slotted_max) = (0usize, 0usize, 0usize, 0usize);
    let mut prev: Option<gmsc2::Frame> = None;
    let mut last_loop = 0u32;
    let mut result = Value::Null;
    loop {
        let t = Instant::now();
        let o = sc2.observe()?;
        obs_s += t.elapsed().as_secs_f64();
        n_obs += 1;
        let Some(obs) = o.observation.as_ref() else { break };
        let gl = obs.game_loop.unwrap_or(0);
        last_loop = gl;
        for a in &o.actions {
            use proto::action_raw::Action as R;
            let kind = match a.action_raw.as_ref().and_then(|r| r.action.as_ref()) {
                Some(R::UnitCommand(uc)) => {
                    let id = voc.action_id(uc.ability_id.unwrap_or(0).max(0) as u32);
                    let name = voc.ability_of(id).and_then(|g| voc.actions.iter().find(|x| x.id == g)).map(|x| x.name.clone())
                        .unwrap_or_else(|| format!("unknown {}", uc.ability_id.unwrap_or(0)));
                    *abilities.entry(name).or_default() += 1;
                    "unit_command"
                }
                Some(R::CameraMove(_)) => "camera_move",
                Some(R::ToggleAutocast(_)) => "toggle_autocast",
                None if a.action_chat.is_some() => "chat",
                None => "other",
            };
            *kinds.entry(kind).or_default() += 1;
            if a.game_loop.is_some() { with_loop += 1 } else { without_loop += 1 }
        }
        let cmds = unit_commands(&o.actions);
        *per_obs.entry(cmds.len()).or_default() += 1;
        let t = Instant::now();
        if let Some(pf) = &prev {
            for uc in &cmds {
                match label(uc, pf, &enc, &voc) {
                    Ok(l) => {
                        *labels.entry("command").or_default() += 1;
                        *targets.entry(l.target_kind).or_default() += 1;
                    }
                    Err(Skip::UnknownAbility) => *labels.entry("unknown_ability").or_default() += 1,
                    Err(Skip::NoActor) => *labels.entry("no_actor").or_default() += 1,
                }
            }
        }
        let f = enc.encode(obs, &voc);
        enc_s += t.elapsed().as_secs_f64();
        let n_units = obs.raw_data.as_ref().map(|r| r.units.len()).unwrap_or(0);
        units_max = units_max.max(n_units);
        units_sum += n_units;
        slotted_max = slotted_max.max(f.unit_mask.iter().filter(|&&m| m).count());
        prev = Some(f);
        if !o.player_result.is_empty() {
            result = json!(o.player_result.iter().map(|r| (r.player_id, r.result)).collect::<Vec<_>>());
            break;
        }
        if gl >= max_loops {
            break;
        }
        let t = Instant::now();
        if sc2.step(if gl < fine_loops { 1 } else { step }).is_err() {
            break;
        }
        step_s += t.elapsed().as_secs_f64();
    }
    let mut top: Vec<(String, usize)> = abilities.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    top.truncate(40);
    Ok(json!({
        "replay": replay.display().to_string(), "map": info.map_name, "loops": info.game_duration_loops,
        "players": info.player_info.iter().map(|p| json!({"mmr": p.player_mmr, "apm": p.player_apm})).collect::<Vec<_>>(),
        "start_replay_s": start_s, "observations": n_obs, "last_loop": last_loop, "result": result,
        "observe_ms": 1000.0 * obs_s / n_obs.max(1) as f64, "step_ms": 1000.0 * step_s / n_obs.max(1) as f64,
        "encode_ms": 1000.0 * enc_s / n_obs.max(1) as f64,
        "units_mean": units_sum as f64 / n_obs.max(1) as f64, "units_max": units_max, "slotted_max": slotted_max,
        "units_without_slot": enc.over_cap,
        "action_kinds": kinds, "actions_with_loop": with_loop, "actions_without_loop": without_loop,
        "commands_per_observation": per_obs, "labels": labels, "label_target_kinds": targets, "top_abilities": top,
    }))
}

/// A replay the scan selected.
#[derive(Clone, Debug)]
pub struct Pick {
    pub path: PathBuf,
    pub loops: u32,
    pub map: String,
    /// per player id − 1: (race, result, mmr, apm)
    pub players: Vec<(String, String, i64, i64)>,
}

impl Pick {
    fn min_mmr(&self) -> i64 {
        self.players.iter().map(|p| p.2).min().unwrap_or(0)
    }
}

/// The scan's replays of the build that pass the bars, best first (the lower MMR of the two).
pub fn select(scan: &Path, build: u32, min_mmr: i64, min_apm: i64, min_seconds: f64) -> Result<(Vec<Pick>, Value)> {
    use std::io::BufRead;
    let mut st: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut picks = vec![];
    for line in std::io::BufReader::new(std::fs::File::open(scan).with_context(|| format!("open {}", scan.display()))?).lines() {
        let line = line?;
        let Ok(r) = serde_json::from_str::<Value>(&line) else {
            *st.entry("unreadable").or_default() += 1;
            continue;
        };
        *st.entry("scanned").or_default() += 1;
        let why = (|| {
            if r.get("error").is_some() {
                return Some("error");
            }
            if r["base_build"].as_u64() != Some(build as u64) {
                return Some("other_build");
            }
            let ps = r["players"].as_array()?;
            if ps.len() != 2 {
                return Some("not_1v1");
            }
            let mut res: Vec<&str> = ps.iter().map(|p| p["result"].as_str().unwrap_or("")).collect();
            res.sort();
            if res != ["defeat", "victory"] {
                return Some("undecided");
            }
            if (r["loops"].as_f64().unwrap_or(0.0)) < min_seconds * 22.4 {
                return Some("short");
            }
            if ps.iter().any(|p| p["mmr"].as_i64().map(|m| m < min_mmr).unwrap_or(true)) {
                return Some("low_mmr");
            }
            if ps.iter().any(|p| p["apm"].as_i64().unwrap_or(0) < min_apm) {
                return Some("low_apm");
            }
            None
        })();
        if let Some(w) = why {
            *st.entry(w).or_default() += 1;
            continue;
        }
        let mut players = vec![(String::new(), String::new(), 0, 0); 2];
        for p in r["players"].as_array().unwrap() {
            let i = p["player_id"].as_u64().unwrap_or(0) as usize;
            if (1..=2).contains(&i) {
                players[i - 1] = (p["race"].as_str().unwrap_or("").to_string(), p["result"].as_str().unwrap_or("").to_string(),
                                  p["mmr"].as_i64().unwrap_or(0), p["apm"].as_i64().unwrap_or(0));
            }
        }
        picks.push(Pick { path: PathBuf::from(r["replay"].as_str().unwrap_or("")), loops: r["loops"].as_u64().unwrap_or(0) as u32,
                          map: r["map"].as_str().unwrap_or("").to_string(), players });
        *st.entry("selected").or_default() += 1;
    }
    picks.sort_by(|a, b| b.min_mmr().cmp(&a.min_mmr()).then(a.path.cmp(&b.path)));
    Ok((picks, json!(st)))
}

fn fnv(key: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The kept runs of windows [start, start + len) of a game of `windows` windows: one in each of
/// `chunks` equal parts (a stable hash of `key` picks its start); the whole game for 0.
pub fn runs(windows: usize, chunks: usize, len: usize, key: &str) -> Vec<(usize, usize)> {
    if chunks == 0 || windows <= chunks * len {
        return vec![(0, windows)];
    }
    (0..chunks).map(|c| {
        let (lo, hi) = (c * windows / chunks, (c + 1) * windows / chunks);
        let span = (hi - lo).saturating_sub(len);
        let start = lo + if span > 0 { (fnv(&format!("{key}#{c}")) % (span as u64 + 1)) as usize } else { 0 };
        (start, len.min(hi - start))
    }).collect()
}

/// The decisions of one run of windows (gmsc2 module docs), as they accumulate.
#[derive(Default)]
struct Run {
    units: Vec<u8>,
    unit_xy: Vec<u16>,
    unit_type: Vec<u16>,
    unit_order: Vec<u16>,
    unit_buffs: Vec<u16>,
    unit_mask: Vec<bool>,
    unit_own: Vec<bool>,
    unit_commanded: Vec<f32>,
    map_bits: Vec<f32>,
    globals: Vec<f32>,
    upgrades: Vec<f32>,
    last_action: Vec<i64>,
    window_start: Vec<bool>,
    action_type: Vec<i64>,
    actors: Vec<f32>,
    target_kind: Vec<i64>,
    target_unit: Vec<i64>,
    target_point: Vec<i64>,
    queued: Vec<bool>,
    windows: (usize, usize),
}

impl Run {
    fn push(&mut self, f: &gmsc2::Frame, w: &gmsc2::Window, l: &gmsc2::Label) {
        self.units.extend_from_slice(&f.units);
        self.unit_xy.extend_from_slice(&f.unit_xy);
        self.unit_type.extend_from_slice(&f.unit_type);
        self.unit_order.extend_from_slice(&f.unit_order);
        self.unit_buffs.extend_from_slice(&f.unit_buffs);
        self.unit_mask.extend_from_slice(&f.unit_mask);
        self.unit_own.extend_from_slice(&f.unit_own);
        self.unit_commanded.extend(w.commanded.iter().map(|&c| c as u8 as f32));
        self.map_bits.extend(f.map_bits.iter().map(|&b| b as f32));
        let mut g = f.globals.clone();
        g[gmsc2::global::WINDOW_CMDS] = w.window_cmds();
        self.globals.extend(g);
        self.upgrades.extend(f.upgrades.iter().map(|&b| b as f32));
        self.last_action.push(w.last_action);
        self.window_start.push(w.n == 0);
        self.action_type.push(l.action_type);
        self.actors.extend(l.actors.iter().map(|&a| a as u8 as f32));
        self.target_kind.push(l.target_kind);
        self.target_unit.push(l.target_unit);
        self.target_point.push(l.target_point);
        self.queued.push(l.queued);
    }

    fn frames(&self) -> usize {
        self.action_type.len()
    }

    fn sample(self, meta: Value, value: f32, n_up: usize) -> Sample {
        use gmsc2::*;
        let t = self.frames();
        Sample::new(meta)
            .u8("obs.units", &[U, F], self.units)
            .u16("obs.unit_xy", &[U, 2], self.unit_xy)
            .u16("obs.unit_type", &[U], self.unit_type)
            .u16("obs.unit_order", &[U], self.unit_order)
            .u16("obs.unit_buffs", &[U, BUFFS], self.unit_buffs)
            .bool("obs.unit_mask", &[U], self.unit_mask)
            .bool("obs.unit_own", &[U], self.unit_own)
            .bits("obs.unit_commanded", &[U], &self.unit_commanded)
            .bits("obs.map_bits", &[MAP_BITS, MAP, MAP], &self.map_bits)
            .f32("obs.globals", &[G], self.globals)
            .bits("obs.upgrades", &[n_up.max(1)], &if n_up == 0 { vec![0.0; t] } else { self.upgrades })
            .i64("obs.last_action", &[], self.last_action)
            .bool("obs.window_start", &[], self.window_start)
            .i64("tgt.action_type", &[], self.action_type)
            .bits("tgt.actors", &[U], &self.actors)
            .i64("tgt.target_kind", &[], self.target_kind)
            .i64("tgt.target_unit", &[], self.target_unit)
            .i64("tgt.target_point", &[], self.target_point)
            .bool("tgt.queued", &[], self.queued)
            .f32("tgt.value", &[], vec![value; t])
    }
}

#[derive(Default)]
struct GameStats {
    windows: usize,
    commands: usize,
    done: usize,
    unknown: std::collections::BTreeMap<i32, usize>,
    no_actor: usize,
    over_k: usize,
    over_cap: usize,
    observations: usize,
}

/// One player's view of a replay → its runs (samples) and counts.
#[allow(clippy::too_many_arguments)]
fn convert_view(sc2: &mut Sc2, pick: &Pick, player: usize, voc: &gmsc2::Vocab, chunks: usize, chunk_windows: usize,
                skip_windows: usize, build: u32) -> Result<(Vec<Sample>, GameStats)> {
    use gmsc2::*;
    sc2.start_replay(&pick.path, player as i32, None)?;
    let mut enc = Encoder::new(MapInfo::from_game_info(&sc2.game_info()?), player as u32);
    let total = pick.loops as usize / STEP_MUL as usize;
    let key = format!("{}#{player}", pick.path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default());
    let plan = runs(total, chunks, chunk_windows, &key);
    let run_of = |w: usize| plan.iter().position(|&(a, n)| (a..a + n).contains(&w));
    let mut st = GameStats::default();
    let mut done_runs: Vec<Run> = vec![];
    let mut cur: Option<(usize, Run)> = None;
    let o = sc2.observe()?;
    st.observations += 1;
    let mut frame = enc.encode(o.observation.as_ref().context("no observation")?, voc);
    let mut w = 0usize;
    loop {
        if let Some(r) = run_of(w) {
            if cur.as_ref().map(|c| c.0) != Some(r) {
                if let Some((_, run)) = cur.take() {
                    done_runs.push(run);
                }
                cur = Some((r, Run { windows: (w, w), ..Default::default() }));
            }
            if sc2.step(STEP_MUL).is_err() {
                break;
            }
            let o = sc2.observe()?;
            st.observations += 1;
            st.windows += 1;
            let run = &mut cur.as_mut().unwrap().1;
            let mut ctx = Window::default();
            for uc in unit_commands(&o.actions) {
                if ctx.n >= K_MAX {
                    st.over_k += 1;
                    continue;
                }
                match label(uc, &frame, &enc, voc) {
                    Ok(l) => {
                        run.push(&frame, &ctx, &l);
                        ctx = ctx.after(&l);
                        st.commands += 1;
                    }
                    Err(Skip::UnknownAbility) => *st.unknown.entry(uc.ability_id.unwrap_or(0)).or_default() += 1,
                    Err(Skip::NoActor) => st.no_actor += 1,
                }
            }
            if ctx.n < K_MAX {
                run.push(&frame, &ctx, &Label::done());
                st.done += 1;
            }
            run.windows.1 = w + 1;
            if !o.player_result.is_empty() {
                break;
            }
            let Some(obs) = o.observation.as_ref() else { break };
            frame = enc.encode(obs, voc);
            w += 1;
        } else {
            // between runs: to the next run's first window, a few windows at a time
            let Some(next) = plan.iter().map(|&(a, _)| a).filter(|&a| a > w).min() else { break };
            let target = next.min(w + skip_windows.max(1));
            if sc2.step(STEP_MUL * (target - w) as u32).is_err() {
                break;
            }
            let o = sc2.observe()?;
            st.observations += 1;
            if !o.player_result.is_empty() {
                break;
            }
            let Some(obs) = o.observation.as_ref() else { break };
            frame = enc.encode(obs, voc);
            w = target;
        }
    }
    if let Some((_, run)) = cur.take() {
        done_runs.push(run);
    }
    // play the replay out, so the process takes the next one from a finished game
    for _ in 0..16 {
        if sc2.step(4096).is_err() {
            break;
        }
        match sc2.observe() {
            Ok(o) if o.player_result.is_empty() => {}
            _ => break,
        }
    }
    st.over_cap = enc.over_cap;
    let me = &pick.players[player - 1];
    let foe = &pick.players[2 - player];
    let value = match me.1.as_str() {
        "victory" => 1.0,
        "defeat" => -1.0,
        _ => 0.0,
    };
    let n_up = voc.upgrades.len();
    let samples = done_runs.into_iter().filter(|r| r.frames() > 0).enumerate().map(|(i, r)| {
        let meta = json!({"replay": pick.path.file_name().map(|f| f.to_string_lossy().to_string()), "player": player,
                          "race": me.0, "result": me.1, "mmr": me.2, "apm": me.3, "enemy_race": foe.0, "enemy_mmr": foe.2,
                          "map": pick.map, "loops": pick.loops, "run": i, "windows": [r.windows.0, r.windows.1], "build": build});
        r.sample(meta, value, n_up)
    }).collect();
    Ok((samples, st))
}

#[allow(clippy::too_many_arguments)]
pub fn convert(install: &Path, scan: &Path, vocab: &Path, out: &Path, min_mmr: i64, min_apm: i64, min_seconds: f64,
               chunks: usize, chunk_windows: usize, skip_windows: usize, workers: usize, port: u16, max: Option<usize>,
               val_every: u64, shard_size: usize, part: usize, parts: usize) -> Result<Value> {
    use std::sync::mpsc;
    let voc = Arc::new(gmsc2::Vocab::load(vocab)?);
    let inst = Install::find(install, Some(voc.build))?;
    let (mut picks, sel) = select(scan, inst.build, min_mmr, min_apm, min_seconds)?;
    if let Some(m) = max {
        picks.truncate(m);
    }
    if parts > 1 {
        picks = picks.into_iter().enumerate().filter(|(i, _)| i % parts == part % parts).map(|(_, p)| p).collect();
    }
    let n_picks = picks.len();
    let meta = json!({"units": gmsc2::U, "own": gmsc2::OWN, "enemy": gmsc2::ENEMY, "unit_features": gmsc2::F,
                      "flags": gmsc2::feature::N_FLAGS, "map": gmsc2::MAP, "map_bits": gmsc2::MAP_BITS,
                      "globals": gmsc2::G, "loc": gmsc2::LOC, "k_max": gmsc2::K_MAX, "step_mul": gmsc2::STEP_MUL,
                      "buffs": gmsc2::BUFFS, "build": voc.build, "unit_vocab": voc.n_units(), "action_vocab": voc.n_actions(),
                      "buff_vocab": voc.n_buffs(), "upgrades": voc.upgrades.len(), "min_mmr": min_mmr});
    let mut train = ShardWriter::new(&out.join("train"), "sc2", CONVERTER_VERSION, meta.clone(), shard_size)?;
    let mut val = ShardWriter::new(&out.join("val"), "sc2", CONVERTER_VERSION, meta, shard_size)?;
    let jobs: Arc<Vec<(usize, usize)>> = Arc::new((0..picks.len()).flat_map(|i| [(i, 1), (i, 2)]).collect());
    let picks = Arc::new(picks);
    let next = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::sync_channel::<(usize, Result<(Vec<Sample>, GameStats)>)>(workers * 2);
    let tmp = tmp_dir("convert");
    let t0 = std::time::Instant::now();
    let mut tot: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut unknown: std::collections::BTreeMap<i32, usize> = Default::default();
    let mut errors: Vec<String> = vec![];
    std::thread::scope(|sc| -> Result<()> {
        for wk in 0..workers.max(1) {
            let (jobs, picks, next, tx, voc, inst, tmp) =
                (jobs.clone(), picks.clone(), next.clone(), tx.clone(), voc.clone(), inst.clone(), tmp.join(wk.to_string()));
            sc.spawn(move || {
                let mut sc2: Option<Sc2> = None;
                loop {
                    let j = next.fetch_add(1, Ordering::SeqCst);
                    let Some(&(i, player)) = jobs.get(j) else { return };
                    if sc2.is_none() {
                        match Sc2::launch(&inst, port + wk as u16, &tmp, None) {
                            Ok(s) => sc2 = Some(s),
                            Err(e) => {
                                let _ = tx.send((i, Err(e)));
                                continue;
                            }
                        }
                    }
                    let r = convert_view(sc2.as_mut().unwrap(), &picks[i], player, &voc, chunks, chunk_windows, skip_windows,
                                         inst.build);
                    if r.is_err() {
                        sc2 = None; // a fresh process for the next view
                    }
                    if tx.send((i, r)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        for (i, r) in rx.iter() {
            *tot.entry("views").or_default() += 1;
            match r {
                Ok((samples, st)) => {
                    let name = picks[i].path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                    let held = crate::common::hash_bucket(&name, val_every);
                    for s in samples {
                        *tot.entry("samples").or_default() += 1;
                        *tot.entry(if held { "val_samples" } else { "train_samples" }).or_default() += 1;
                        *tot.entry("frames").or_default() += s.frames()?;
                        if held { val.push(s)? } else { train.push(s)? }
                    }
                    for (k, v) in [("windows", st.windows), ("commands", st.commands), ("done", st.done), ("no_actor", st.no_actor),
                                   ("over_k", st.over_k), ("units_without_slot", st.over_cap), ("observations", st.observations)] {
                        *tot.entry(k).or_default() += v;
                    }
                    for (k, v) in st.unknown {
                        *unknown.entry(k).or_default() += v;
                    }
                }
                Err(e) => {
                    *tot.entry("failed_views").or_default() += 1;
                    if errors.len() < 20 {
                        errors.push(format!("{}: {e:#}", picks[i].path.display()));
                    }
                }
            }
            let n = tot["views"];
            if n % 100 == 0 {
                eprintln!("[sc2 convert] {n}/{} views, {:.0} s", jobs.len(), t0.elapsed().as_secs_f64());
            }
        }
        Ok(())
    })?;
    train.flush()?;
    val.flush()?;
    Ok(json!({"selection": sel, "replays": n_picks, "counts": tot, "unknown_abilities": unknown, "errors": errors,
              "train_shards": train.written_shards, "val_shards": val.written_shards,
              "seconds": t0.elapsed().as_secs_f64(), "build": inst.build}))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Scan { install, replays, out, workers, port, max_replays } => {
            println!("{}", scan(&install, &replays, &out, workers, port, max_replays)?);
            Ok(())
        }
        Cmd::Data { install, replay, out, port } => {
            println!("{}", data(&install, &replay, &out, port)?);
            Ok(())
        }
        Cmd::Convert { install, scan, vocab, out, min_mmr, min_apm, min_seconds, chunks, chunk_windows, skip_windows, workers,
                       port, max_replays, val_every, shard_size, part, parts } => {
            println!("{}", convert(&install, &scan, &vocab, &out, min_mmr, min_apm, min_seconds, chunks, chunk_windows,
                                   skip_windows, workers, port, max_replays, val_every, shard_size, part, parts)?);
            Ok(())
        }
        Cmd::Probe { install, replay, vocab, player, fine_loops, step, max_loops, port } => {
            println!("{}", probe(&install, &replay, &vocab, player, fine_loops, step, max_loops, port)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_cover_each_part_once_and_stay_put() {
        let r = runs(1000, 3, 48, "a.SC2Replay#1");
        assert_eq!(r.len(), 3);
        for (c, &(a, n)) in r.iter().enumerate() {
            assert_eq!(n, 48);
            assert!(a >= c * 1000 / 3 && a + n <= (c + 1) * 1000 / 3, "{r:?}");
        }
        assert_eq!(r, runs(1000, 3, 48, "a.SC2Replay#1"), "a stable hash");
        assert_ne!(r, runs(1000, 3, 48, "a.SC2Replay#2"));
        assert_eq!(runs(100, 3, 48, "x"), vec![(0, 100)], "a short game is kept whole");
        assert_eq!(runs(1000, 0, 48, "x"), vec![(0, 1000)]);
    }
}
