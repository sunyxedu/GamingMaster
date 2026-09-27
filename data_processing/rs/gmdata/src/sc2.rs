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

fn bucket(gap: i64) -> &'static str {
    match gap {
        i64::MIN..=0 => "0",
        1 => "1",
        2 => "2",
        3..=4 => "3-4",
        5..=8 => "5-8",
        9..=16 => "9-16",
        17..=32 => "17-32",
        33..=64 => "33-64",
        65..=128 => "65-128",
        _ => ">128",
    }
}

#[allow(clippy::too_many_arguments)]
pub fn probe(install: &Path, replay: &Path, vocab: &Path, player: i32, fine_loops: u32, step: u32, max_loops: u32, port: u16)
    -> Result<Value> {
    use gmsc2::{encode, label, MapInfo, Vocab};
    use std::collections::BTreeMap;
    use std::time::Instant;
    let voc = Vocab::load(vocab)?;
    let inst = Install::find(install, None)?;
    let mut sc2 = Sc2::launch(&inst, port, &tmp_dir("probe"), None)?;
    let info = sc2.replay_info(replay)?;
    let t0 = Instant::now();
    sc2.start_replay(replay, player, None)?;
    let start_s = t0.elapsed().as_secs_f64();
    let map = MapInfo::from_game_info(&sc2.game_info()?);
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut with_loop, mut without_loop) = (0usize, 0usize);
    let mut lag: BTreeMap<i64, usize> = BTreeMap::new();
    let mut per_obs: BTreeMap<usize, usize> = BTreeMap::new();
    let mut gaps: BTreeMap<&str, usize> = BTreeMap::new();
    let mut abilities: BTreeMap<String, usize> = BTreeMap::new();
    let mut labels: BTreeMap<&str, usize> = BTreeMap::new();
    let mut targets: BTreeMap<i64, usize> = BTreeMap::new();
    let mut cmd_loops: Vec<i64> = vec![];
    let (mut obs_s, mut step_s, mut enc_s) = (0f64, 0f64, 0f64);
    let (mut n_obs, mut units_max, mut units_sum, mut own_max, mut over_cap) = (0usize, 0usize, 0usize, 0usize, 0usize);
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
        let mut n_cmd = 0;
        for a in &o.actions {
            use proto::action_raw::Action as R;
            let kind = match a.action_raw.as_ref().and_then(|r| r.action.as_ref()) {
                Some(R::UnitCommand(uc)) => {
                    n_cmd += 1;
                    let name = voc.ability_of(voc.action_id(uc.ability_id.unwrap_or(0).max(0) as u32))
                        .and_then(|g| voc.actions.iter().find(|x| x.id == g)).map(|x| x.name.clone())
                        .unwrap_or_else(|| format!("unknown {}", uc.ability_id.unwrap_or(0)));
                    *abilities.entry(name).or_default() += 1;
                    if let Some(l) = a.game_loop {
                        cmd_loops.push(l as i64);
                    }
                    "unit_command"
                }
                Some(R::CameraMove(_)) => "camera_move",
                Some(R::ToggleAutocast(_)) => "toggle_autocast",
                None if a.action_chat.is_some() => "chat",
                None => "other",
            };
            *kinds.entry(kind).or_default() += 1;
            match a.game_loop {
                Some(l) => {
                    with_loop += 1;
                    *lag.entry(gl as i64 - l as i64).or_default() += 1;
                }
                None => without_loop += 1,
            }
        }
        *per_obs.entry(n_cmd).or_default() += 1;
        let t = Instant::now();
        if let Some(pf) = &prev {
            let l = label(&o.actions, pf, &map, &voc);
            let has_cmd = o.actions.iter().any(|a| matches!(a.action_raw.as_ref().and_then(|r| r.action.as_ref()),
                                                            Some(proto::action_raw::Action::UnitCommand(_))));
            *labels.entry(match (l.action_type, has_cmd) {
                (0, false) => "noop",
                (0, true) => "command_without_token_actors",
                (1, _) => "unknown_ability",
                _ => "command",
            }).or_default() += 1;
            if l.action_type > 1 {
                *targets.entry(l.target_kind).or_default() += 1;
            }
        }
        let f = encode(obs, &map, player as u32, &voc);
        enc_s += t.elapsed().as_secs_f64();
        let n_units = obs.raw_data.as_ref().map(|r| r.units.len()).unwrap_or(0);
        units_max = units_max.max(n_units);
        units_sum += n_units;
        own_max = own_max.max(f.own_slots);
        over_cap += (n_units > gmsc2::U) as usize;
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
    cmd_loops.sort();
    for w in cmd_loops.windows(2) {
        *gaps.entry(bucket(w[1] - w[0])).or_default() += 1;
    }
    let mut top: Vec<(String, usize)> = abilities.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    top.truncate(40);
    let minutes = last_loop as f64 / 22.4 / 60.0;
    Ok(json!({
        "replay": replay.display().to_string(), "map": info.map_name, "loops": info.game_duration_loops,
        "players": info.player_info.iter().map(|p| json!({"mmr": p.player_mmr, "apm": p.player_apm})).collect::<Vec<_>>(),
        "start_replay_s": start_s, "observations": n_obs, "last_loop": last_loop, "result": result,
        "observe_ms": 1000.0 * obs_s / n_obs.max(1) as f64, "step_ms": 1000.0 * step_s / n_obs.max(1) as f64,
        "encode_ms": 1000.0 * enc_s / n_obs.max(1) as f64,
        "units_mean": units_sum as f64 / n_obs.max(1) as f64, "units_max": units_max, "own_tokens_max": own_max,
        "observations_over_cap": over_cap,
        "action_kinds": kinds, "actions_with_loop": with_loop, "actions_without_loop": without_loop,
        "obs_loop_minus_action_loop": lag, "commands_per_observation": per_obs,
        "commands_per_minute": cmd_loops.len() as f64 / minutes.max(1e-9), "command_gaps_loops": gaps,
        "labels": labels, "label_target_kinds": targets, "top_abilities": top,
    }))
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
        Cmd::Probe { install, replay, vocab, player, fine_loops, step, max_loops, port } => {
            println!("{}", probe(&install, &replay, &vocab, player, fine_loops, step, max_loops, port)?);
            Ok(())
        }
    }
}
