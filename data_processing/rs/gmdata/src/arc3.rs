//! ARC-AGI-3 recordings → shards for the arc3 adapter (model/games/arc3.py).
//!
//! Input: the ARC-AGI-3 toolkit's `*.recording.jsonl` — one play session (one `guid`) per
//! file, one line per API response: `{"timestamp", "data": {game_id, frame, state,
//! levels_completed, win_levels, action_input: {id, data: {x, y}}, guid, full_reset,
//! available_actions}}`, plus a trailing summary line without a frame.  The same format
//! holds ARC Prize's human baseline recordings (`arc_agi_3_public_demo_human_testing.zip`:
//! 340 first-time sessions of the 25 public games) and agents' recordings (the toolkit's
//! recorder; e.g. the ARA Labs trajectories on Hugging Face).  Directories are searched
//! recursively; zip archives are read in place (macOS `__MACOSX/` / `._*` entries skipped).
//!
//! One sample per session: the whole play, resets and game overs included, as one stream —
//! what a player learns from a failed attempt stays in its memory.  Frames are the frames
//! env/vec_arc3.py gives the network for the same actions (tests/test_arc3_recordings.py
//! replays sessions through the env and compares):
//!
//!   obs.grid    [T,64,64] u8   the frame's last layer (the settled state after the action's
//!                              animation); x = column, y = row
//!   obs.delta   [T,64,64]      cells that differ from the previous frame (none on frame 0);
//!                              bit-packed
//!   obs.globals [T,22] f32     0-7 available actions (RESET always), 8 levels / win_levels,
//!                              9 min(actions / max_actions, 1), 10-17 the previous action
//!                              one-hot (none on frame 0), 18-19 its click (x / 63, y / 63),
//!                              20 it completed a level, 21 GAME_OVER
//!   tgt.type_idx [T] i64       the next action (0 RESET, k ACTIONk), click_idx y·64 + x (ACTION6)
//!   tgt.legal_type [T,8]       RESET + the available actions; a GAME_OVER frame RESET alone
//!   tgt.legal_click [T,4096]   every cell when ACTION6 is legal (bit-packed)
//!   tgt.value [T] f32          the return still to come: levels the session completes after
//!                              the frame / win_levels (the env's reward units)
//!
//! Row semantics (checked on all 340 human sessions): the first row is the RESET that starts
//! the game; row i's `action_input` is the action taken at row i − 1's frame; a RESET row
//! after a GAME_OVER restarts the level (the env's forced RESET) and one mid-play restarts it
//! by choice; after a GAME_OVER the rows up to the next RESET are bookkeeping the player could
//! not act in (dropped, stat `noise_rows`); a WIN row ends the session.  A row with an empty
//! frame keeps the previous grid (stat `empty_frames`).  Action ids are integers or the enum
//! names ("ACTION6", "RESET") depending on the recorder, and some recorders miss the starting
//! RESET row (23 of the 340 human sessions): such a session starts at its first row, the
//! state after the first action, with that action counted (stat `missing_start`).
//!
//! Quality: first-time human play explores and errs — which is what solving an unseen game
//! is — so no session is dropped for inefficiency by default; `--min-levels` keeps sessions
//! that completed at least that many levels.  `--games` / `--exclude-games` select games (the
//! held-out transfer games), `--split train|val --val-every N` holds out every N-th session
//! by a hash of its guid.

use crate::common;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const CONVERTER_VERSION: u32 = 2;
const GRID: usize = 64;
const CELLS: usize = GRID * GRID;
const N_TYPES: usize = 8;
const CLICK: usize = 6;
pub const N_GLOBALS: usize = 22;

#[derive(Subcommand)]
pub enum Cmd {
    /// Convert recording files (.jsonl, directories, zips) to gm-shards; `--out` omitted =
    /// statistics only
    Convert {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 16)]
        shard_size: usize,
        /// the env's action budget (global 9 = actions / max_actions, clipped at 1); RL must
        /// use the same (shard meta `max_actions`)
        #[arg(long, default_value_t = 2048)]
        max_actions: usize,
        /// games to keep (names: ls20, vc33, ...; comma-separated); none = all
        #[arg(long, value_delimiter = ',')]
        games: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        exclude_games: Vec<String>,
        /// keep sessions that completed at least this many levels
        #[arg(long, default_value_t = 0)]
        min_levels: i64,
        /// train = every session but each `--val-every`-th (by a hash of its guid), val = those
        #[arg(long)]
        split: Option<String>,
        #[arg(long, default_value_t = 10)]
        val_every: u64,
        /// a label for the samples' meta (e.g. human, agent:<name>)
        #[arg(long, default_value = "recording")]
        source: String,
    },
}

/// One recording line with a frame.
struct Row {
    game_id: String,
    guid: String,
    /// the frame's last layer; None = an empty frame
    grid: Option<Vec<u8>>,
    state: String,
    levels: i64,
    win_levels: i64,
    available: Vec<i64>,
    /// (type, x, y) of the action that produced this row
    action: (i64, i64, i64),
}

fn action_id(v: &Value) -> Option<i64> {
    if let Some(i) = v.as_i64() {
        return Some(i);
    }
    let s = v.as_str()?;
    if s == "RESET" {
        return Some(0);
    }
    s.strip_prefix("ACTION")?.parse().ok()
}

fn parse_row(line: &str) -> Result<Option<Row>> {
    let v: Value = serde_json::from_str(line).context("recording line is not JSON")?;
    let d = &v["data"];
    let Some(frame) = d.get("frame").and_then(|f| f.as_array()) else { return Ok(None) };
    let Some(ai) = d.get("action_input") else { return Ok(None) };
    let grid = match frame.last() {
        None => None,
        Some(layer) => {
            let rows = layer.as_array().context("frame layer is not a list")?;
            if rows.len() != GRID {
                bail!("frame layer has {} rows", rows.len());
            }
            let mut g = Vec::with_capacity(CELLS);
            for r in rows {
                let r = r.as_array().context("frame row is not a list")?;
                if r.len() != GRID {
                    bail!("frame row has {} cells", r.len());
                }
                for c in r {
                    let x = c.as_u64().context("frame cell is not an integer")?;
                    if x > 15 {
                        bail!("palette index {x}");
                    }
                    g.push(x as u8);
                }
            }
            Some(g)
        }
    };
    let typ = action_id(&ai["id"]).context("action_input.id")?;
    let data = &ai["data"];
    let (x, y) = (data["x"].as_i64().unwrap_or(0), data["y"].as_i64().unwrap_or(0));
    Ok(Some(Row {
        game_id: d["game_id"].as_str().unwrap_or("").to_string(),
        guid: d["guid"].as_str().unwrap_or("").to_string(),
        grid,
        state: d["state"].as_str().unwrap_or("").to_string(),
        levels: d["levels_completed"].as_i64().unwrap_or(0),
        win_levels: d["win_levels"].as_i64().unwrap_or(0),
        available: d["available_actions"].as_array().map(|a| a.iter().filter_map(action_id).collect()).unwrap_or_default(),
        action: (typ, x, y),
    }))
}

/// A session's frames and labels (module docs).
struct Session {
    game: String,
    game_id: String,
    guid: String,
    grid: Vec<u8>,
    delta: Vec<f32>,
    globals: Vec<f32>,
    legal_type: Vec<bool>,
    legal_click: Vec<f32>,
    type_idx: Vec<i64>,
    click_idx: Vec<i64>,
    value: Vec<f32>,
    levels: i64,
    win_levels: i64,
    won: bool,
    rows: usize,
    noise_rows: usize,
    empty_frames: usize,
    game_overs: usize,
    resets: usize,
    /// the recording starts after the first action (no starting RESET row)
    missing_start: bool,
}

impl Session {
    fn frames(&self) -> usize {
        self.type_idx.len()
    }
}

/// The env state a frame shows (env/vec_arc3.py's arrays for one env).
struct Env {
    grid: Vec<u8>,
    available: [bool; N_TYPES],
    levels: i64,
    win_levels: i64,
    actions: usize,
    last_type: i64,
    last_click: (f32, f32),
    just_leveled: bool,
    game_over: bool,
}

impl Env {
    fn load(&mut self, r: &Row) -> Vec<f32> {
        let g = r.grid.clone().unwrap_or_else(|| self.grid.clone());
        let delta: Vec<f32> = g.iter().zip(&self.grid).map(|(a, b)| (a != b) as u8 as f32).collect();
        self.grid = g;
        self.available = [false; N_TYPES];
        self.available[0] = true;
        for &a in &r.available {
            if (1..N_TYPES as i64).contains(&a) {
                self.available[a as usize] = true;
            }
        }
        self.levels = r.levels;
        self.win_levels = r.win_levels.max(1);
        self.game_over = r.state == "GAME_OVER";
        delta
    }
    fn globals(&self, max_actions: usize) -> [f32; N_GLOBALS] {
        let mut g = [0f32; N_GLOBALS];
        for (i, &a) in self.available.iter().enumerate() {
            g[i] = a as u8 as f32;
        }
        g[8] = self.levels as f32 / self.win_levels as f32;
        g[9] = (self.actions as f32 / max_actions as f32).min(1.0);
        if self.last_type >= 0 {
            g[10 + self.last_type as usize] = 1.0;
        }
        g[18] = self.last_click.0;
        g[19] = self.last_click.1;
        g[20] = self.just_leveled as u8 as f32;
        g[21] = self.game_over as u8 as f32;
        g
    }
    fn legal(&self) -> [bool; N_TYPES] {
        let mut l = self.available;
        if self.game_over {
            l = [false; N_TYPES];
            l[0] = true;
        }
        l
    }
}

/// The frames of one recording (module docs); None = no frame to learn from.
fn session(lines: &[String], max_actions: usize) -> Result<Option<Session>> {
    let mut rows = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        if let Some(r) = parse_row(l).with_context(|| format!("line {}", i + 1))? {
            rows.push(r);
        }
    }
    let Some(first) = rows.first() else { return Ok(None) };
    let game_id = first.game_id.clone();
    let game = game_id.split('-').next().unwrap_or("").to_string();
    let mut s = Session {
        game, game_id, guid: first.guid.clone(), grid: vec![], delta: vec![], globals: vec![], legal_type: vec![],
        legal_click: vec![], type_idx: vec![], click_idx: vec![], value: vec![], levels: 0, win_levels: 0, won: false,
        rows: rows.len(), noise_rows: 0, empty_frames: 0, game_overs: 0, resets: 0, missing_start: false,
    };
    let mut env = Env { grid: vec![0; CELLS], available: [false; N_TYPES], levels: 0, win_levels: 1, actions: 0, last_type: -1,
                        last_click: (0.0, 0.0), just_leveled: false, game_over: false };
    if first.grid.is_none() {
        bail!("the first frame is empty");
    }
    env.load(first);
    if first.action.0 != 0 {
        // some recorders miss the RESET that starts the game: the first row is the state
        // after the first action — the session starts there, that action already taken
        let (typ, x, y) = first.action;
        if !(1..N_TYPES as i64).contains(&typ) {
            bail!("the first row is action {typ}");
        }
        s.missing_start = true;
        env.actions = 1;
        env.last_type = typ;
        if typ as usize == CLICK {
            env.last_click = (x.clamp(0, 63) as f32 / 63.0, y.clamp(0, 63) as f32 / 63.0);
        }
        env.just_leveled = first.levels > 0;
    }
    let mut delta = vec![0f32; CELLS]; // frame 0: no change
    let mut levels_at: Vec<i64> = Vec::new();
    let mut i = 1;
    loop {
        // the frame the player acts on
        let g = env.globals(max_actions);
        let legal = env.legal();
        // the action taken at it: the next row, past the bookkeeping rows after a GAME_OVER
        let mut j = i;
        if env.game_over {
            while j < rows.len() && rows[j].action.0 != 0 {
                s.noise_rows += 1;
                j += 1;
            }
        }
        let Some(r) = rows.get(j) else { break };
        let (typ, x, y) = r.action;
        if !(0..N_TYPES as i64).contains(&typ) {
            bail!("row {j}: action type {typ}");
        }
        if !legal[typ as usize] {
            // an action the frame did not allow (the API refuses those): not a decision
            s.noise_rows += 1;
            i = j + 1;
            continue;
        }
        s.grid.extend_from_slice(&env.grid);
        s.delta.extend_from_slice(&delta);
        s.globals.extend_from_slice(&g);
        s.legal_type.extend_from_slice(&legal);
        s.legal_click.extend(std::iter::repeat(legal[CLICK] as u8 as f32).take(CELLS));
        s.type_idx.push(typ);
        s.click_idx.push(if typ as usize == CLICK { (y.clamp(0, 63) * GRID as i64) + x.clamp(0, 63) } else { 0 });
        levels_at.push(env.levels);
        // the env's step
        if r.grid.is_none() {
            s.empty_frames += 1;
        }
        if typ == 0 {
            s.resets += 1;
        }
        let before = env.levels;
        env.actions += 1;
        env.last_type = typ;
        env.last_click = if typ as usize == CLICK { (x.clamp(0, 63) as f32 / 63.0, y.clamp(0, 63) as f32 / 63.0) } else { (0.0, 0.0) };
        delta = env.load(r);
        env.just_leveled = env.levels > before;
        if env.game_over {
            s.game_overs += 1;
        }
        if r.state == "WIN" {
            s.won = true;
            break;
        }
        i = j + 1;
    }
    if s.type_idx.is_empty() {
        return Ok(None);
    }
    s.levels = env.levels;
    s.win_levels = env.win_levels;
    let w = env.win_levels.max(1) as f32;
    s.value = levels_at.iter().map(|&l| ((env.levels - l).max(0)) as f32 / w).collect();
    Ok(Some(s))
}

/// Every recording of the inputs: (name, lines).
fn for_each_recording(inputs: &[PathBuf], f: &mut dyn FnMut(&str, Vec<String>) -> Result<()>) -> Result<usize> {
    let mut n = 0;
    let is_rec = |name: &str| {
        let base = name.rsplit('/').next().unwrap_or(name);
        !name.contains("__MACOSX/") && !base.starts_with("._") && (name.ends_with(".jsonl") || name.ends_with(".jsonl.gz"))
    };
    for input in inputs {
        for p in common::find_files(input, &[".jsonl", ".jsonl.gz", ".zip"])? {
            let name = p.to_string_lossy().to_string();
            if name.ends_with(".zip") {
                let mut z = zip::ZipArchive::new(std::fs::File::open(&p)?)?;
                let mut names: Vec<String> = (0..z.len()).filter_map(|i| z.by_index(i).ok().map(|m| m.name().to_string())).collect();
                names.sort();
                for m in names {
                    if !is_rec(&m) {
                        continue;
                    }
                    let mut buf = Vec::new();
                    z.by_name(&m)?.read_to_end(&mut buf)?;
                    let text = String::from_utf8(common::read_maybe_gz(buf)?)?;
                    f(&format!("{name}!{m}"), text.lines().map(str::to_string).collect())?;
                    n += 1;
                }
            } else if is_rec(&name) {
                let text = String::from_utf8(common::read_maybe_gz(std::fs::read(&p)?)?)?;
                f(&name, text.lines().map(str::to_string).collect())?;
                n += 1;
            }
        }
    }
    Ok(n)
}

fn guid_bucket(guid: &str, every: u64) -> bool {
    // FNV-1a: stable across runs and platforms
    let mut h: u64 = 0xcbf29ce484222325;
    for b in guid.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    every > 0 && h % every == 0
}

#[allow(clippy::too_many_arguments)]
pub fn convert(inputs: &[PathBuf], out: Option<&Path>, shard_size: usize, max_actions: usize, games: &[String],
               exclude: &[String], min_levels: i64, split: Option<&str>, val_every: u64, source: &str) -> Result<Value> {
    if let Some(sp) = split {
        if sp != "train" && sp != "val" {
            bail!("--split is train or val, got {sp}");
        }
    }
    let meta = json!({"global_features": N_GLOBALS, "grid": GRID, "max_actions": max_actions});
    let mut w = match out {
        Some(o) => Some(ShardWriter::new(o, "arc3", CONVERTER_VERSION, meta, shard_size)?),
        None => None,
    };
    let mut st: BTreeMap<&str, usize> = ["sessions", "kept", "frames", "other_game", "few_levels", "other_split", "empty", "errors",
        "won", "noise_rows", "empty_frames", "game_overs", "resets", "missing_start"].into_iter().map(|k| (k, 0)).collect();
    let mut per_game: BTreeMap<String, (usize, usize, usize, i64)> = BTreeMap::new(); // sessions, frames, wins, levels
    let mut versions: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let n = for_each_recording(inputs, &mut |name, lines| {
        *st.get_mut("sessions").unwrap() += 1;
        let s = match session(&lines, max_actions) {
            Ok(Some(s)) => s,
            Ok(None) => {
                *st.get_mut("empty").unwrap() += 1;
                return Ok(());
            }
            Err(e) => {
                *st.get_mut("errors").unwrap() += 1;
                eprintln!("[arc3] {name}: {e:#}");
                return Ok(());
            }
        };
        if !s.guid.is_empty() && !seen.insert(s.guid.clone()) {
            *st.get_mut("empty").unwrap() += 1;
            return Ok(());
        }
        if (!games.is_empty() && !games.contains(&s.game)) || exclude.contains(&s.game) {
            *st.get_mut("other_game").unwrap() += 1;
            return Ok(());
        }
        if s.levels < min_levels {
            *st.get_mut("few_levels").unwrap() += 1;
            return Ok(());
        }
        if let Some(sp) = split {
            if guid_bucket(&s.guid, val_every) != (sp == "val") {
                *st.get_mut("other_split").unwrap() += 1;
                return Ok(());
            }
        }
        *st.get_mut("kept").unwrap() += 1;
        *st.get_mut("frames").unwrap() += s.frames();
        *st.get_mut("won").unwrap() += s.won as usize;
        for (k, v) in [("noise_rows", s.noise_rows), ("empty_frames", s.empty_frames), ("game_overs", s.game_overs),
                       ("resets", s.resets), ("missing_start", s.missing_start as usize)] {
            *st.get_mut(k).unwrap() += v;
        }
        let e = per_game.entry(s.game.clone()).or_default();
        e.0 += 1;
        e.1 += s.frames();
        e.2 += s.won as usize;
        e.3 += s.levels;
        *versions.entry(s.game_id.clone()).or_default() += 1;
        if let Some(w) = w.as_mut() {
            let t = s.frames();
            let sample = Sample::new(json!({"game": s.game, "game_id": s.game_id, "guid": s.guid, "source": source,
                                            "levels": s.levels, "win_levels": s.win_levels, "won": s.won, "rows": s.rows,
                                            "missing_start": s.missing_start, "file": name}))
                .u8("obs.grid", &[GRID, GRID], s.grid)
                .bits("obs.delta", &[GRID, GRID], &s.delta)
                .f32("obs.globals", &[N_GLOBALS], s.globals)
                .i64("tgt.type_idx", &[], s.type_idx)
                .i64("tgt.click_idx", &[], s.click_idx)
                .bool("tgt.legal_type", &[N_TYPES], s.legal_type)
                .bits("tgt.legal_click", &[CELLS], &s.legal_click)
                .f32("tgt.value", &[], s.value);
            debug_assert_eq!(sample.frames().unwrap(), t);
            w.push(sample)?;
        }
        Ok(())
    })?;
    let shards = match w {
        Some(mut w) => {
            w.flush()?;
            w.written_shards
        }
        None => 0,
    };
    let mut v = json!(st);
    v["recordings"] = json!(n);
    v["shards"] = json!(shards);
    v["per_game"] = json!(per_game.iter().map(|(g, e)| (g.clone(), json!({"sessions": e.0, "frames": e.1, "won": e.2,
                                                                           "levels": e.3}))).collect::<BTreeMap<_, _>>());
    v["versions"] = json!(versions);
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Convert { input, out, shard_size, max_actions, games, exclude_games, min_levels, split, val_every, source } => {
            println!("{}", convert(&input, out.as_deref(), shard_size, max_actions, &games, &exclude_games, min_levels,
                                   split.as_deref(), val_every, &source)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(fill: u8, marks: &[(usize, usize, u8)]) -> Value {
        let mut g = vec![vec![fill; GRID]; GRID];
        for &(x, y, c) in marks {
            g[y][x] = c;
        }
        json!([g])
    }

    fn line(id: Value, xy: Option<(i64, i64)>, fr: Value, state: &str, levels: i64, avail: &[i64]) -> String {
        let mut data = json!({"game_id": "tt01-abc"});
        if let Some((x, y)) = xy {
            data["x"] = json!(x);
            data["y"] = json!(y);
        }
        json!({"timestamp": "t", "data": {"game_id": "tt01-abc", "frame": fr, "state": state, "levels_completed": levels,
               "win_levels": 2, "available_actions": avail, "guid": "g1", "full_reset": false,
               "action_input": {"id": id, "data": data, "reasoning": null}}}).to_string()
    }

    #[test]
    fn a_session_becomes_the_envs_frames() {
        let lines = vec![
            line(json!(0), None, frame(9, &[]), "NOT_FINISHED", 0, &[1, 6]),
            line(json!(6), Some((3, 2)), frame(9, &[(3, 2, 4)]), "NOT_FINISHED", 0, &[1, 6]),
            line(json!("ACTION1"), None, frame(9, &[(3, 2, 4)]), "GAME_OVER", 0, &[1, 6]),
            // bookkeeping after the game over, then the (forced) RESET
            line(json!(1), None, frame(9, &[(3, 2, 4)]), "GAME_OVER", 0, &[1, 6]),
            line(json!("RESET"), None, frame(9, &[]), "NOT_FINISHED", 0, &[1, 6]),
            line(json!(6), Some((10, 20)), json!([]), "NOT_FINISHED", 1, &[1, 6]),
            line(json!(1), None, frame(9, &[(0, 0, 1)]), "WIN", 2, &[1, 6]),
            json!({"data": {"levels_completed": 2, "won": 1}}).to_string(),
        ];
        let s = session(&lines, 4).unwrap().unwrap();
        assert_eq!(s.type_idx, [6, 1, 0, 6, 1]);
        assert_eq!(s.click_idx, [2 * 64 + 3, 0, 0, 20 * 64 + 10, 0]);
        assert_eq!((s.noise_rows, s.empty_frames, s.game_overs, s.resets, s.won), (1, 1, 1, 1, true));
        let g = |t: usize| &s.globals[t * N_GLOBALS..(t + 1) * N_GLOBALS];
        // frame 0: available RESET + 1 + 6, nothing done yet
        assert_eq!(&g(0)[..8], &[1., 1., 0., 0., 0., 0., 1., 0.]);
        assert_eq!(&g(0)[9..], &[0.; 13]);
        // frame 1: after the click at (3, 2): its one-hot, its coordinates, the changed cell
        assert_eq!((g(1)[16], g(1)[18], g(1)[19], g(1)[9]), (1.0, 3.0 / 63.0, 2.0 / 63.0, 0.25));
        assert_eq!(s.delta[CELLS + 2 * 64 + 3], 1.0);
        assert_eq!(s.delta[CELLS..2 * CELLS].iter().sum::<f32>(), 1.0);
        // frame 2: GAME_OVER — RESET alone is legal, no click cell
        assert_eq!(g(2)[21], 1.0);
        assert_eq!(&s.legal_type[16..24], &[true, false, false, false, false, false, false, false]);
        assert!(s.legal_click[2 * CELLS..3 * CELLS].iter().all(|&x| x == 0.0));
        // frame 3: after the RESET; the empty frame of row 5 keeps the grid, levels 1 → 2
        assert_eq!(g(3)[10], 1.0);
        assert_eq!((g(4)[8], g(4)[20]), (0.5, 1.0));
        // the return still to come: 2 levels over win_levels 2 before frame 3's click
        assert_eq!(s.value, [1.0, 1.0, 1.0, 1.0, 0.5]);
        assert_eq!(&s.grid[..4], &[9, 9, 9, 9]);
    }

    #[test]
    fn illegal_rows_and_bad_first_rows() {
        // ACTION2 is not available: the row is no decision (the API refuses it)
        let lines = vec![
            line(json!(0), None, frame(9, &[]), "NOT_FINISHED", 0, &[1]),
            line(json!(2), None, frame(9, &[]), "NOT_FINISHED", 0, &[1]),
            line(json!(1), None, frame(8, &[]), "NOT_FINISHED", 0, &[1]),
        ];
        let s = session(&lines, 100).unwrap().unwrap();
        assert_eq!((s.type_idx.clone(), s.noise_rows), (vec![1], 1));
        // no starting RESET row: the session starts after its first action
        let lines = vec![line(json!("ACTION6"), Some((5, 6)), frame(9, &[]), "NOT_FINISHED", 0, &[1, 6]),
                         line(json!(1), None, frame(8, &[]), "NOT_FINISHED", 0, &[1, 6])];
        let s = session(&lines, 100).unwrap().unwrap();
        assert!(s.missing_start);
        assert_eq!(s.type_idx, [1]);
        assert_eq!((s.globals[9], s.globals[16], s.globals[18], s.globals[19]), (0.01, 1.0, 5.0 / 63.0, 6.0 / 63.0));
        assert!(guid_bucket("x", 1) && !guid_bucket("x", 0));
    }
}
