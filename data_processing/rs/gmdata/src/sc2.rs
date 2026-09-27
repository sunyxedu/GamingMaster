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

use anyhow::{Context, Result};
use clap::Subcommand;
use sc2api::{proto, Install, Sc2};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Scan { install, replays, out, workers, port, max_replays } => {
            println!("{}", scan(&install, &replays, &out, workers, port, max_replays)?);
            Ok(())
        }
    }
}
