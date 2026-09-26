//! Backgammon: gnubg self-play records → verified candidate-action gm-shards, and the
//! differential tools pinning the rules to gnubg. Rules and frames:
//! `gmrules::games::backgammon` (the simulator's code).
//!
//! ```sh
//! # records: gnubg (2-ply) plays both seats; OUR engine runs every match (dice from the
//! # match's seed) and checks gnubg's full legal-move list against ours at every checker
//! # play and gnubg's dead-cube refusals against the rules; one gz JSONL per worker
//! gmdata backgammon generate --out data/backgammon/raw --matches 1000 --workers 16 --seed 1
//! # transcripts → gm-shards: every record replayed and verified by the rules; gnubg's
//! # equities → soft targets, its match-winning chances → engine values
//! gmdata backgammon convert --input data/backgammon/raw --out data/backgammon/train \
//!     --soft-k 8 --engine-values
//! # any transcript — also gnubg's OWN match loop (scripts/reference/gnubg_selfplay_dump.py:
//! # gnubg's dice, cube, Crawford, scoring) — replayed game by game: legality, results,
//! # scores, Crawford flags; exit 1 on a mismatch
//! gmdata backgammon verify --input tests/fixtures/backgammon/gnubg_selfplay.jsonl.gz
//! # legal-move differential test against gnubg: live (playout positions + random boards),
//! # optionally recorded as a fixture, or on a recorded fixture
//! gmdata backgammon diff --live --positions 20000 --seed 7 --random-frac 0.4
//! gmdata backgammon diff --fixture tests/fixtures/backgammon/gnubg_legal.jsonl.gz
//! ```
//!
//! gnubg runs as an external engine process — `scripts/reference/gnubg_bridge.py`
//! (embedded in this binary; `--bridge` overrides) under a Python with GNU Backgammon's
//! bindings, by default a throw-away uv environment (`uv run --no-project --python 3.12
//! --with gnubg==1.1.0a16 python`, `--python` overrides; wheels for macOS arm64 and
//! Linux x86-64 with glibc >= 2.28). gnubg is GPL-3.0-or-later and is never linked or
//! vendored; the games it plays are our data.
//!
//! Why our engine runs the games: gnubg's own match loop resigns hopeless games (the
//! rules have no resignation: their results are no rules outcome), rolls its own dice
//! and cannot be checked position by position. Driving the matches ourselves gives
//! complete games, reproducible dice (a match is a pure function of its seed: the RL
//! env replays a record exactly, tests/test_backgammon.py) and a differential test at
//! every position gnubg plays. gnubg's own match loop is the independent reference for
//! the match flow (`verify` on its transcripts).
//!
//! Transcript format (`bgmatch.v1`, one match per JSONL line — one money game per line
//! in money play; seats 0 / 1; moves are gnubg's flat (from, to) lists from the mover's
//! side, 25 = bar, 0 = off; a pair may span several dice):
//!
//! ```text
//! {"format": "bgmatch.v1", "id": "...", "config": {gmrules BackgammonConfig},
//!  "players": [...], "meta": {"seed": ..., ...},
//!  "games": [{"score": [s0, s1], "crawford": false,
//!             "actions": [{"p": 0, "a": "move", "dice": [4, 3], "move": [13, 9, 24, 21],
//!                          "eq": [[[13, 9, 24, 21], 0.1234], ...], "mwc": 0.53},  (analysis: optional)
//!                         {"p": 1, "a": "no_double", "cube": {"nd": .., "dt": .., "dp": ..,
//!                                                              "mwc": [nd, dt, dp]}},
//!                         {"p": 1, "a": "double", "cube": {...}}, {"p": 0, "a": "take"},
//!                         {"p": 0, "a": "drop"}, {"p": 0, "a": "resign", "value": 1}, ...],
//!             "winner": 1, "points": 2, "resigned": false}, ...],
//!  "winner": 1, "score": [3, 7]}
//! ```
//!
//! A turn without a cube action where the rules offer the cube is an implicit "no
//! double"; the opening roll is the first move's dice (the higher die belongs to the
//! mover). Every move is resolved by its resulting position to the rules' candidate (a
//! split turn — more than 64 distinct moves — becomes its checker moves, the smallest
//! id first among those that still reach the position, and then the completion).
//! Checked: every action legal in its phase, dice, per-game results (winner, points),
//! scores and Crawford flags at every game start, the match result.
//!
//! Teacher labels (`convert`): soft targets (`--soft-k`) π ∝ exp(Q/τ) (`--soft-tau`,
//! equity units) over gnubg's analysed moves at its top ply (whole-move decisions), the
//! doubler's Q = ND vs min(DT, DP) and the taker's −DT vs −DP; engine values
//! (`--engine-values`, the acting seat's expected outcome in the rules' units): match
//! play 2·MWC − 1 of gnubg's best action (move: the best move; cube: max(ND, min(DT, DP));
//! take: 1 − min(DT, DP)), money play clamp(equity × cube / money_scale); every step of
//! a split turn gets its move's value. gnubg gives no analysis for a dead cube or a
//! roll without a legal move: those frames have no teacher (mask false).
//!
//! Full scale (cluster): `scripts/fetch_backgammon.sh OUT MATCHES WORKERS` (generate +
//! convert + a val split; the exact commands and the expected corpus — ~360 frames per
//! 7-point match, ~2.2 KB per frame, ~13 s per match per worker at 2-ply — are in the
//! script header and in `model/games/gmrules/backgammon.py`).

use crate::common;
use crate::label::engine::{Engine, EngineCommand};
use crate::traj::{self, ConvertOptions, Record};
use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::Subcommand;
use gmrules::games::backgammon::{self as bg, apply_steps, legal_moves, off, Backgammon, BackgammonConfig, Dice, Phase, Pos, CHECKERS};
use gmrules::{Event, Game, Rng, Seat, Turn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub const FORMAT: &str = "bgmatch.v1";
/// The gnubg bridge (engine process), embedded so the binary works from anywhere.
pub const BRIDGE_PY: &str = include_str!("../../../../scripts/reference/gnubg_bridge.py");
/// Default Python running the bridge: a throw-away uv environment with gnubg.
pub const DEFAULT_PYTHON: &str = "uv run --no-project --python 3.12 --with gnubg==1.1.0a16 python";

// ---------------------------------------------------------------------------------------
// transcripts
// ---------------------------------------------------------------------------------------

/// gnubg's cube analysis of a decision (the doubler's view, EMG).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CubeEq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proper: Option<String>,
    /// match play: the same three as the doubler's match-winning chances
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mwc: Option<[f64; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Action {
    /// seat
    pub p: usize,
    /// "move" | "double" | "no_double" | "take" | "drop" | "resign" | "accept" | "reject"
    pub a: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dice: Option<[u8; 2]>,
    /// flat (from, to) pairs
    #[serde(default, rename = "move", skip_serializing_if = "Option::is_none")]
    pub mv: Option<Vec<u8>>,
    /// analysed moves: (flat move, equity), best first
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eq: Vec<(Vec<u8>, f64)>,
    /// match play: the mover's match-winning chance after the best move
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mwc: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cube: Option<CubeEq>,
    /// resignation value (1 single, 2 gammon, 3 backgammon)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<u8>,
}

impl Action {
    fn new(p: usize, a: &str) -> Action {
        Action { p, a: a.into(), dice: None, mv: None, eq: Vec::new(), mwc: None, cube: None, value: None }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GameRec {
    /// score at the start of the game (seat 0, seat 1)
    #[serde(default)]
    pub score: [u32; 2],
    /// the game is the Crawford game
    #[serde(default)]
    pub crawford: bool,
    pub actions: Vec<Action>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<u32>,
    #[serde(default)]
    pub resigned: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Transcript {
    #[serde(default)]
    pub format: String,
    pub id: String,
    /// the rules variant the match was played under
    pub config: BackgammonConfig,
    #[serde(default)]
    pub players: Vec<String>,
    #[serde(default)]
    pub meta: Value,
    pub games: Vec<GameRec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<[u32; 2]>,
}

/// (from, to) steps of a flat gnubg move.
pub fn steps_of(flat: &[u8]) -> Result<Vec<(usize, usize)>> {
    ensure!(flat.len() % 2 == 0 && flat.len() <= 8, "move {flat:?}: expected up to four (from, to) pairs");
    Ok(flat.chunks(2).map(|c| (c[0] as usize, c[1] as usize)).collect())
}

/// Chance id of the roll (a, b).
pub fn roll_id(a: u8, b: u8) -> u32 {
    let (hi, lo) = (a.max(b) as u32, a.min(b) as u32);
    6 * (hi - 1) + (lo - 1)
}

/// Chance id of the opening roll won by `mover` with the dice (a, b).
pub fn opening_id(mover: Seat, a: u8, b: u8) -> u32 {
    let (hi, lo) = (a.max(b) as u32, a.min(b) as u32);
    let (d0, d1) = if mover == 0 { (hi, lo) } else { (lo, hi) };
    bg::OPENING_BASE + 6 * (d0 - 1) + (d1 - 1)
}

/// Teacher distribution π ∝ exp((Q − max Q) / τ) over (id, Q) pairs.
pub fn softmax(q: &[(u32, f64)], tau: f64) -> Vec<(u32, f32)> {
    let m = q.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = q.iter().map(|x| ((x.1 - m) / tau).exp()).collect();
    let z: f64 = w.iter().sum();
    q.iter().zip(w).map(|(x, w)| (x.0, (w / z) as f32)).collect()
}

/// What walking a transcript produced.
#[derive(Debug)]
pub struct Walked {
    pub events: Vec<Event>,
    /// teacher distributions: (event index, [(action id, p)])
    pub soft: Vec<(usize, Vec<(u32, f32)>)>,
    /// engine values: (event index, the actor's expected outcome in [-1, 1])
    pub values: Vec<(usize, f32)>,
    pub state: Backgammon,
    /// games that ended in the engine (and matched the record)
    pub games: usize,
    /// the record stops at a resignation (the episode is unfinished)
    pub resigned: bool,
    /// the record stops inside a game (accepted with `WalkOptions::allow_truncated`)
    pub truncated: bool,
    pub moves: usize,
    pub cube_actions: usize,
    pub split_turns: usize,
    /// analysed moves that are no legal candidate (should stay 0)
    pub bad_analysis: usize,
}

/// How [`walk`] maps a transcript.
#[derive(Clone, Copy, Debug, Default)]
pub struct WalkOptions {
    /// soft-target temperature τ (equity units); 0 = no soft targets
    pub tau: f64,
    /// accept a record that stops inside a game (the causality tests' truncations)
    pub allow_truncated: bool,
}

/// The acting seat's expected outcome in the rules' outcome units ([-1, 1]) from gnubg's
/// numbers: match play 2·MWC − 1, money play clamp(equity × cube / money_scale) (equities
/// are EMG, normalised to the current cube value `cube`).
pub fn value_of(cfg: &BackgammonConfig, cube: u32, eq: Option<f64>, mwc: Option<f64>) -> Option<f32> {
    let v = if cfg.match_length > 0 { 2.0 * mwc? - 1.0 } else { eq? * cube as f64 / cfg.money_scale };
    v.is_finite().then(|| v.clamp(-1.0, 1.0) as f32)
}

/// Map a transcript's games onto the rules engine (`cfg`, fresh state): events, soft
/// targets (`opt.tau > 0`), engine values, and every check listed in the module docs.
pub fn walk(cfg: &BackgammonConfig, games: &[GameRec], opt: WalkOptions) -> Result<Walked> {
    let tau = opt.tau;
    let mut g = Backgammon::new_game(cfg);
    let mut w = Walked {
        events: Vec::new(),
        soft: Vec::new(),
        values: Vec::new(),
        state: g.clone(),
        games: 0,
        resigned: false,
        truncated: false,
        moves: 0,
        cube_actions: 0,
        split_turns: 0,
        bad_analysis: 0,
    };
    let money = cfg.match_length == 0;
    let act = |g: &mut Backgammon, w: &mut Walked, seat: Seat, id: u32| -> Result<()> {
        g.apply_action(seat, id).map_err(|e| anyhow!("{e}"))?;
        w.events.push(Event::Act(seat, id));
        Ok(())
    };
    'games: for (gi, game) in games.iter().enumerate() {
        ensure!(g.phase() == Phase::Opening, "game {gi}: the previous game is not over ({:?})", g.phase());
        if !money {
            ensure!(game.score == g.score(), "game {gi}: recorded start score {:?}, the rules say {:?}", game.score, g.score());
            ensure!(game.crawford == g.crawford().0, "game {gi}: recorded Crawford flag {}, the rules say {}", game.crawford,
                    g.crawford().0);
        }
        let n_results = g.results().len();
        let mut last_cube: Option<CubeEq> = None;
        let mut i = 0;
        while i < game.actions.len() {
            let a = &game.actions[i];
            let at = || format!("game {gi} action {i} ({} by seat {})", a.a, a.p);
            ensure!(a.p < 2, "{}: seat out of range", at());
            match (g.phase(), a.a.as_str()) {
                (_, "resign") => {
                    ensure!(game.resigned, "{}: a resignation in a game not marked resigned", at());
                    w.resigned = true;
                    break 'games;
                }
                (Phase::Opening, "move") => {
                    let [d1, d2] = a.dice.with_context(|| format!("{}: no dice", at()))?;
                    ensure!(d1 != d2 && (1..=6).contains(&d1) && (1..=6).contains(&d2), "{}: opening roll {d1}-{d2}", at());
                    let o = opening_id(a.p, d1, d2);
                    g.apply_chance(o).map_err(|e| anyhow!("{}: {e}", at()))?;
                    w.events.push(Event::Chance(o));
                    // the move itself is handled in the Move phase (same action)
                }
                (Phase::Cube, "double" | "no_double") => {
                    ensure!(a.p == g.on_roll(), "{}: seat {} is on roll", at(), g.on_roll());
                    let id = if a.a == "double" { bg::DOUBLE } else { bg::NO_DOUBLE };
                    if let Some(c) = &a.cube {
                        if let (Some(nd), Some(dt), Some(dp)) = (c.nd, c.dt, c.dp) {
                            if tau > 0.0 {
                                w.soft.push((w.events.len(), softmax(&[(bg::NO_DOUBLE, nd), (bg::DOUBLE, dt.min(dp))], tau)));
                            }
                            // the position's value under gnubg's best cube action
                            let mwc = c.mwc.map(|[n, t, p]| n.max(t.min(p)));
                            if let Some(v) = value_of(cfg, g.cube().0, Some(nd.max(dt.min(dp))), mwc) {
                                w.values.push((w.events.len(), v));
                            }
                        }
                    }
                    act(&mut g, &mut w, a.p, id).with_context(at)?;
                    last_cube = a.cube.clone();
                    w.cube_actions += 1;
                    i += 1;
                }
                (Phase::Cube, "move") => {
                    // the player rolled without doubling (the cube was available)
                    ensure!(a.p == g.on_roll(), "{}: seat {} is on roll", at(), g.on_roll());
                    act(&mut g, &mut w, a.p, bg::NO_DOUBLE).with_context(at)?;
                    w.cube_actions += 1;
                }
                (Phase::Take, "take" | "drop") => {
                    ensure!(a.p != g.on_roll(), "{}: the doubler cannot answer its own double", at());
                    let id = if a.a == "take" { bg::TAKE } else { bg::DROP };
                    if let Some(c) = &last_cube {
                        if let (Some(dt), Some(dp)) = (c.dt, c.dp) {
                            if tau > 0.0 {
                                w.soft.push((w.events.len(), softmax(&[(bg::TAKE, -dt), (bg::DROP, -dp)], tau)));
                            }
                            // the taker's value: its best answer (the doubler's min)
                            let mwc = c.mwc.map(|[_, t, p]| 1.0 - t.min(p));
                            if let Some(v) = value_of(cfg, g.cube().0, Some(-dt.min(dp)), mwc) {
                                w.values.push((w.events.len(), v));
                            }
                        }
                    }
                    act(&mut g, &mut w, a.p, id).with_context(at)?;
                    w.cube_actions += 1;
                    i += 1;
                }
                (Phase::Roll, "move") => {
                    ensure!(a.p == g.on_roll(), "{}: seat {} is on roll", at(), g.on_roll());
                    let [d1, d2] = a.dice.with_context(|| format!("{}: no dice", at()))?;
                    ensure!((1..=6).contains(&d1) && (1..=6).contains(&d2), "{}: dice {d1}-{d2}", at());
                    let o = roll_id(d1, d2);
                    g.apply_chance(o).map_err(|e| anyhow!("{}: {e}", at()))?;
                    w.events.push(Event::Chance(o));
                }
                (Phase::Move, "move") => {
                    ensure!(a.p == g.on_roll(), "{}: seat {} is on roll", at(), g.on_roll());
                    if let Some([d1, d2]) = a.dice {
                        let (hi, lo) = g.roll();
                        ensure!((d1.max(d2), d1.min(d2)) == (hi, lo), "{}: dice {d1}-{d2}, the roll is {hi}-{lo}", at());
                    }
                    let pos = g.pos(a.p);
                    let steps = steps_of(a.mv.as_deref().unwrap_or(&[]))?;
                    let target = apply_steps(&pos, &steps).map_err(|e| anyhow!("{}: {e}", at()))?;
                    let plan = g.plan_move(&target).map_err(|e| anyhow!("{}: move {:?}: {e}", at(), a.mv))?;
                    if plan.len() == 1 && tau > 0.0 && !a.eq.is_empty() {
                        let mut q = Vec::with_capacity(a.eq.len());
                        for (flat, eq) in &a.eq {
                            let r = steps_of(flat).and_then(|s| apply_steps(&pos, &s).map_err(|e| anyhow!(e)));
                            match r.ok().and_then(|r| g.candidates().iter().find(|c| c.result == r)) {
                                Some(c) if !q.iter().any(|&(id, _)| id == c.id) => q.push((c.id, *eq)),
                                Some(_) => {}
                                None => w.bad_analysis += 1,
                            }
                        }
                        if !q.is_empty() {
                            w.soft.push((w.events.len(), softmax(&q, tau)));
                        }
                    }
                    if plan.len() > 1 {
                        w.split_turns += 1;
                    }
                    // the best move's value (every step of a split turn follows it)
                    let value = value_of(cfg, g.cube().0, a.eq.first().map(|e| e.1), a.mwc);
                    for id in plan {
                        if let Some(v) = value {
                            w.values.push((w.events.len(), v));
                        }
                        act(&mut g, &mut w, a.p, id).with_context(at)?;
                    }
                    w.moves += 1;
                    i += 1;
                }
                (ph, _) => bail!("{}: not possible in phase {ph:?}", at()),
            }
        }
        if g.results().len() == n_results + 1 {
            let r = *g.results().last().unwrap();
            if let Some(win) = game.winner {
                ensure!(win == r.winner, "game {gi}: recorded winner {win}, the rules say {}", r.winner);
            }
            if let Some(p) = game.points {
                ensure!(p == r.points, "game {gi}: recorded {p} points, the rules say {} (cube {} x {}{})", r.points, r.cube,
                        r.multiplier, if r.dropped { ", dropped" } else { "" });
            }
            ensure!(!game.resigned, "game {gi}: marked resigned, but the recorded play finished it");
            w.games += 1;
        } else if game.resigned {
            w.resigned = true;
            break;
        } else if opt.allow_truncated {
            w.truncated = true;
            break;
        } else {
            bail!("game {gi}: the recorded actions stop before the game is over ({:?})", g.phase());
        }
    }
    w.state = g;
    Ok(w)
}

/// A transcript as a converter record (whole match = one episode).
pub fn to_record(t: &Transcript, cfg: &BackgammonConfig, opt: WalkOptions) -> Result<(Record, Walked)> {
    ensure!(t.format.is_empty() || t.format == FORMAT, "{}: format {:?}, expected {FORMAT}", t.id, t.format);
    let w = walk(cfg, &t.games, opt).with_context(|| t.id.clone())?;
    let over = w.state.phase() == Phase::Over;
    let returns = if over {
        if let Some(win) = t.winner {
            ensure!(Some(win) == w.state.match_winner(), "{}: recorded winner {win}, the rules say {:?}", t.id, w.state.match_winner());
        }
        if let (Some(s), false) = (t.score, cfg.match_length == 0) {
            ensure!(s == w.state.score(), "{}: recorded final score {s:?}, the rules say {:?}", t.id, w.state.score());
        }
        Some(w.state.returns())
    } else {
        ensure!(w.resigned || opt.allow_truncated, "{}: the match is not over after its last game", t.id);
        None
    };
    let meta = json!({
        "players": t.players,
        "games": w.games,
        "final_score": if over { json!(w.state.score()) } else { Value::Null },
        "resigned": w.resigned,
        "source": t.meta,
    });
    let rec = Record {
        id: t.id.clone(),
        events: w.events.clone(),
        meta,
        seats: None,
        expected_returns: returns,
        soft_targets: w.soft.clone(),
        engine_values: w.values.clone(),
        ..Default::default()
    };
    Ok((rec, w))
}

/// Lines of the transcript files under `input` (`.jsonl` / `.jsonl.gz`, sorted), as
/// (id for errors, line).
fn transcript_lines(input: &Path) -> Result<impl Iterator<Item = Result<(String, String)>>> {
    let files = common::find_files(input, &[".jsonl", ".jsonl.gz"])?;
    ensure!(!files.is_empty(), "no .jsonl / .jsonl.gz transcripts under {}", input.display());
    Ok(files.into_iter().flat_map(|path| -> Box<dyn Iterator<Item = Result<(String, String)>>> {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::from(e).context(name)))),
        };
        let r: Box<dyn Read> = if name.to_lowercase().ends_with(".gz") { Box::new(flate2::read::MultiGzDecoder::new(f)) } else { Box::new(f) };
        Box::new(BufReader::new(r).lines().enumerate().filter_map(move |(ln, l)| {
            let id = format!("{name}:{}", ln + 1);
            match l {
                Ok(l) if l.trim().is_empty() => None,
                Ok(l) => Some(Ok((id, l))),
                Err(e) => Some(Err(anyhow::Error::from(e).context(id))),
            }
        }))
    }))
}

pub fn parse_transcript(line: &str) -> Result<Transcript> {
    Ok(serde_json::from_str(line)?)
}

// ---------------------------------------------------------------------------------------
// the gnubg engine process
// ---------------------------------------------------------------------------------------

/// How to start the gnubg bridge.
#[derive(Clone, Debug)]
pub struct EngineSpec {
    /// the Python command (split on whitespace)
    pub python: String,
    /// the bridge script (None = the embedded copy)
    pub bridge: Option<PathBuf>,
    pub plies: u32,
    pub timeout: Duration,
}

impl EngineSpec {
    fn command(&self, script: &Path) -> Result<EngineCommand> {
        let mut parts: Vec<String> = self.python.split_whitespace().map(String::from).collect();
        ensure!(!parts.is_empty(), "empty --python command");
        let program = parts.remove(0);
        parts.push(script.display().to_string());
        parts.push("--plies".into());
        parts.push(self.plies.to_string());
        Ok(EngineCommand::process(program, &parts))
    }
}

/// The bridge script on disk (the embedded copy written to a temp file when needed).
struct BridgeFile {
    path: PathBuf,
    temp: bool,
}

impl BridgeFile {
    fn new(spec: &EngineSpec) -> Result<BridgeFile> {
        if let Some(p) = &spec.bridge {
            return Ok(BridgeFile { path: p.clone(), temp: false });
        }
        let path = std::env::temp_dir().join(format!("gmdata-gnubg-bridge-{}.py", std::process::id()));
        std::fs::write(&path, BRIDGE_PY).with_context(|| format!("write {}", path.display()))?;
        Ok(BridgeFile { path, temp: true })
    }
}

impl Drop for BridgeFile {
    fn drop(&mut self) {
        if self.temp {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// One gnubg bridge process.
pub struct Gnubg {
    engine: Engine,
    timeout: Duration,
    pub calls: usize,
    pub busy: Duration,
}

impl Gnubg {
    fn start(spec: &EngineSpec, script: &Path) -> Result<Gnubg> {
        let engine = Engine::spawn(spec.command(script)?)?;
        let mut g = Gnubg { engine, timeout: spec.timeout, calls: 0, busy: Duration::ZERO };
        // the first start may build the throw-away environment
        let hello = g.request_with(&json!({"op": "hello"}), Duration::from_secs(900))?;
        ensure!(hello["ok"] == true, "gnubg bridge: unexpected hello reply {hello}");
        Ok(g)
    }

    fn request_with(&mut self, req: &Value, timeout: Duration) -> Result<Value> {
        let t = Instant::now();
        let line = self.engine.request(&req.to_string(), timeout, |_| true)?.pop().context("no reply")?;
        self.calls += 1;
        self.busy += t.elapsed();
        let v: Value = serde_json::from_str(&line).with_context(|| format!("bridge reply {line:?}"))?;
        if let Some(e) = v.get("error") {
            bail!("gnubg bridge error: {e} (request {req})");
        }
        Ok(v)
    }

    fn request(&mut self, req: &Value) -> Result<Value> {
        let t = self.timeout;
        self.request_with(req, t)
    }
}

/// gnubg's board for the position from the mover's view: [opponent, mover].
fn board_json(p: &Pos) -> Value {
    json!([p.opp.to_vec(), p.me.to_vec()])
}

/// gnubg's cube / match context of the player on roll.
fn cube_json(g: &Backgammon, mover: Seat) -> Value {
    let (value, owner) = g.cube();
    let rel = if owner < 0 { -1 } else if owner as usize == mover { 0 } else { 1 };
    let s = g.score();
    let cfg = g.config();
    json!({"value": value, "owner": rel, "match": cfg.match_length, "score": [s[mover], s[1 - mover]],
           "crawford": g.crawford().0, "jacoby": cfg.match_length == 0 && cfg.jacoby})
}

fn flat_move(m: &Value) -> Result<Vec<u8>> {
    m.as_array().context("move: array expected")?.iter().map(|x| x.as_u64().map(|x| x as u8).context("move entry")).collect()
}

fn flat_moves(v: &Value) -> Result<Vec<Vec<u8>>> {
    v.as_array().context("moves: array expected")?.iter().map(flat_move).collect()
}

/// The set of resulting positions of gnubg's moves (flat lists) from `pos` — an empty
/// list (no legal move) is the position itself.
fn gnubg_results(pos: &Pos, moves: &[Vec<u8>]) -> Result<HashSet<Pos>> {
    if moves.is_empty() || moves.iter().all(|m| m.is_empty()) {
        return Ok(HashSet::from([*pos]));
    }
    moves.iter().map(|m| apply_steps(pos, &steps_of(m)?).map_err(|e| anyhow!("gnubg move {m:?}: {e}"))).collect()
}

/// Compare our legal moves of (pos, roll) with gnubg's list; Err describes the difference.
fn compare_legal(pos: &Pos, hi: u8, lo: u8, gnubg: &[Vec<u8>]) -> Result<usize> {
    let theirs = gnubg_results(pos, gnubg)?;
    let (ours, _) = legal_moves(pos, Dice::roll(hi, lo));
    let ours: HashSet<Pos> = ours.iter().map(|c| c.result).collect();
    if ours != theirs {
        let only_ours = ours.difference(&theirs).count();
        let only_theirs = theirs.difference(&ours).count();
        bail!("legal moves differ for roll {hi}-{lo} at {}: {} ours, {} gnubg's, {only_ours} only ours, {only_theirs} only gnubg's",
              json!(board_json(pos)), ours.len(), theirs.len());
    }
    Ok(ours.len())
}

// ---------------------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
pub struct GenStats {
    pub matches: usize,
    pub games: usize,
    pub decisions: usize,
    pub checker_plays: usize,
    pub cube_decisions: usize,
    pub split_turns: usize,
    pub max_candidates: usize,
    pub legal_checked: usize,
    pub engine_calls: usize,
    pub engine_seconds: f64,
    pub failed: usize,
    /// matches skipped for a gnubg / gmrules disagreement (--keep-going; see the
    /// discrepancies-*.ndjson file)
    pub discrepancies: usize,
    pub errors: Vec<String>,
}

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Play one match with gnubg deciding both seats (dice from `seed`).
#[allow(clippy::too_many_arguments)]
pub fn play_match(eng: &mut Gnubg, cfg: &BackgammonConfig, seed: u64, id: &str, plies: u32, top: usize, verify_legal: bool,
                  st: &mut GenStats) -> Result<Transcript> {
    let mut g = Backgammon::new_game(cfg);
    let mut rng = Rng::new(seed);
    let mut games: Vec<GameRec> = Vec::new();
    let mut cur = GameRec { score: g.score(), crawford: g.crawford().0, actions: Vec::new(), winner: None, points: None, resigned: false };
    let mut take_answer: Option<bool> = None;
    let mut outcomes = Vec::new();
    loop {
        match g.turn() {
            Turn::Terminal => break,
            Turn::Chance => {
                outcomes.clear();
                g.chance_outcomes(&mut outcomes);
                let w: Vec<f64> = outcomes.iter().map(|&(_, p)| p).collect();
                let o = outcomes[rng.weighted(&w)].0;
                g.apply_chance(o).map_err(|e| anyhow!("{e}"))?;
            }
            Turn::Act(seats) => {
                let seat = seats.iter().next().unwrap();
                st.decisions += 1;
                match g.phase() {
                    Phase::Cube => {
                        let r = eng.request(&json!({"op": "cube", "board": board_json(&g.pos(seat)), "cube": cube_json(&g, seat)}))?;
                        // gnubg refuses to analyse a dead cube (a post-Crawford leader, or
                        // score + cube reaching the match length): it never doubles there,
                        // and our rules must agree that the cube is dead for the mover
                        let cannot = r["cannot_double"].as_bool().unwrap_or(false);
                        if cannot != g.cube_dead_for(seat) {
                            bail!("{id}: gnubg vs gmrules: gnubg says the cube is {} for the player on roll, the rules \
                                   say it is {} (cube {:?}, score {:?}, Crawford {:?})",
                                  if cannot { "dead" } else { "live" }, if cannot { "live" } else { "dead" }, g.cube(), g.score(),
                                  g.crawford());
                        }
                        let double = r["double"].as_bool().context("cube reply: double")?;
                        take_answer = Some(r["take"].as_bool().context("cube reply: take")?);
                        let mut a = Action::new(seat, if double { "double" } else { "no_double" });
                        let m = &r["mwc"];
                        a.cube = Some(CubeEq {
                            nd: r["nd"].as_f64().map(round4),
                            dt: r["dt"].as_f64().map(round4),
                            dp: r["dp"].as_f64().map(round4),
                            proper: r["proper"].as_str().map(String::from),
                            mwc: match (m["nd"].as_f64(), m["dt"].as_f64(), m["dp"].as_f64()) {
                                (Some(n), Some(t), Some(p)) => Some([round6(n), round6(t), round6(p)]),
                                _ => None,
                            },
                        });
                        g.apply_action(seat, if double { bg::DOUBLE } else { bg::NO_DOUBLE }).map_err(|e| anyhow!("{e}"))?;
                        cur.actions.push(a);
                        st.cube_decisions += 1;
                    }
                    Phase::Take => {
                        let take = take_answer.take().context("a take decision without the double's analysis")?;
                        g.apply_action(seat, if take { bg::TAKE } else { bg::DROP }).map_err(|e| anyhow!("{e}"))?;
                        cur.actions.push(Action::new(seat, if take { "take" } else { "drop" }));
                        st.cube_decisions += 1;
                    }
                    Phase::Move => {
                        let (hi, lo) = g.roll();
                        let pos = g.pos(seat);
                        st.max_candidates = st.max_candidates.max(if g.is_split() { 0 } else { g.candidates().len() });
                        let r = eng.request(&json!({"op": "move", "board": board_json(&pos), "dice": [hi, lo],
                                                    "cube": cube_json(&g, seat)}))?;
                        let arr = r["moves"].as_array().context("move reply: moves")?;
                        let mut listed = Vec::with_capacity(arr.len());
                        let mut eqs = Vec::with_capacity(arr.len());
                        let mut mply = Vec::with_capacity(arr.len());
                        for m in arr {
                            listed.push(flat_move(&m[0])?);
                            eqs.push(m[1].as_f64().context("move reply: equity")?);
                            mply.push(m[2].as_u64().context("move reply: ply")?);
                        }
                        // analysis = the moves evaluated at full depth (equities of the moves
                        // gnubg's filters cut at lower plies are not comparable)
                        let top_ply = mply.iter().copied().max().unwrap_or(0);
                        if verify_legal {
                            compare_legal(&pos, hi, lo, &listed).with_context(|| format!("{id}: gnubg vs gmrules"))?;
                            st.legal_checked += 1;
                        }
                        let best = listed.first().cloned().unwrap_or_default();
                        let target = apply_steps(&pos, &steps_of(&best)?).map_err(|e| anyhow!("{id}: gnubg's move {best:?}: {e}"))?;
                        let plan = g.plan_move(&target).map_err(|e| anyhow!("{id}: gnubg's move {best:?}: {e}"))?;
                        if plan.len() > 1 {
                            st.split_turns += 1;
                        }
                        for a in plan {
                            g.apply_action(seat, a).map_err(|e| anyhow!("{e}"))?;
                        }
                        let mut a = Action::new(seat, "move");
                        a.dice = Some([hi, lo]);
                        a.mv = Some(best);
                        a.mwc = r["mwc"].as_f64().map(round6);
                        a.eq = listed
                            .into_iter()
                            .zip(eqs)
                            .zip(mply)
                            .filter(|&(_, p)| p == top_ply)
                            .take(top)
                            .map(|((m, e), _)| (m, round4(e)))
                            .collect();
                        cur.actions.push(a);
                        st.checker_plays += 1;
                    }
                    ph => bail!("{id}: a decision in phase {ph:?}"),
                }
            }
        }
        if g.results().len() > games.len() {
            let r = *g.results().last().unwrap();
            cur.winner = Some(r.winner);
            cur.points = Some(r.points);
            games.push(std::mem::replace(&mut cur, GameRec {
                score: g.score(),
                crawford: g.crawford().0,
                actions: Vec::new(),
                winner: None,
                points: None,
                resigned: false,
            }));
            st.games += 1;
        }
    }
    st.matches += 1;
    let winner = g.match_winner();
    Ok(Transcript {
        format: FORMAT.into(),
        id: id.into(),
        config: cfg.clone(),
        players: vec![format!("gnubg {plies}-ply"); 2],
        meta: json!({"engine": "gnubg (GNU Backgammon, PyPI gnubg 1.1.0a16)", "seed": seed,
                     "generator": "gmdata backgammon generate"}),
        games,
        winner,
        score: (cfg.match_length > 0).then(|| g.score()),
    })
}

/// Seed of match `i` of a run.
pub fn match_seed(seed: u64, i: usize) -> u64 {
    Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)).next_u64()
}

/// What [`generate`] plays.
#[derive(Clone, Debug)]
pub struct GenOptions {
    /// matches `first .. first + matches` of the run `seed` (ids and dice seeds come from
    /// the global match index, so disjoint ranges — e.g. Slurm array tasks — never overlap
    /// and a range can be regenerated exactly)
    pub first: usize,
    pub matches: usize,
    pub workers: usize,
    pub seed: u64,
    /// analysed moves kept per checker play
    pub top: usize,
    /// compare gnubg's legal move list with ours at every checker play
    pub verify_legal: bool,
    /// log rules discrepancies (gnubg vs gmrules) to `discrepancies-*.ndjson` and go on
    /// instead of aborting the run
    pub keep_going: bool,
    pub force: bool,
}

/// Matches per gzip member of the output files: a killed run loses at most this many
/// matches per worker (the files stay readable: concatenated gzip members).
const MEMBER_MATCHES: usize = 8;

fn is_discrepancy(msg: &str) -> bool {
    msg.contains("gnubg vs gmrules") || msg.contains("gnubg's move")
}

/// gnubg self-play: `workers` bridge processes, one output file per worker
/// (`bgmatch-s<seed>-m<first>-w<k>.jsonl.gz`).
pub fn generate(out: &Path, cfg: &BackgammonConfig, spec: &EngineSpec, o: &GenOptions) -> Result<GenStats> {
    Backgammon::check_config(cfg).map_err(|e| anyhow!("config: {e}"))?;
    std::fs::create_dir_all(out)?;
    let (seed, first, matches) = (o.seed, o.first, o.matches);
    let workers = o.workers.max(1).min(matches.max(1));
    let paths: Vec<PathBuf> = (0..workers).map(|k| out.join(format!("bgmatch-s{seed}-m{first}-w{k:03}.jsonl.gz"))).collect();
    if !o.force {
        if let Some(p) = paths.iter().find(|p| p.exists()) {
            bail!("{} exists (use --force, another --seed or another --first-match)", p.display());
        }
    }
    let bad_path = out.join(format!("discrepancies-s{seed}-m{first}.ndjson"));
    let bad_file = std::sync::Mutex::new(None::<std::fs::File>);
    let bridge = BridgeFile::new(spec)?;
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    let results: Vec<Result<GenStats>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|k| {
                let (path, bridge, done, bad_path, bad_file) = (&paths[k], &bridge.path, &done, &bad_path, &bad_file);
                s.spawn(move || -> Result<GenStats> {
                    let mut st = GenStats::default();
                    let mut eng = Gnubg::start(spec, bridge)?;
                    let mut file = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?);
                    let mut pending = String::new();
                    let mut in_member = 0usize;
                    let flush = |file: &mut std::io::BufWriter<std::fs::File>, pending: &mut String| -> Result<()> {
                        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                        gz.write_all(pending.as_bytes())?;
                        file.write_all(&gz.finish()?)?;
                        file.flush()?;
                        pending.clear();
                        Ok(())
                    };
                    for i in (first + k..first + matches).step_by(workers) {
                        let id = format!("gnubg{}-s{seed}-{i:07}", spec.plies);
                        let ms = match_seed(seed, i);
                        let mut tries = 0;
                        loop {
                            let mut local = GenStats::default();
                            match play_match(&mut eng, cfg, ms, &id, spec.plies, o.top, o.verify_legal, &mut local) {
                                Ok(t) => {
                                    pending.push_str(&serde_json::to_string(&t)?);
                                    pending.push('\n');
                                    in_member += 1;
                                    if in_member == MEMBER_MATCHES {
                                        flush(&mut file, &mut pending)?;
                                        in_member = 0;
                                    }
                                    merge(&mut st, &local);
                                    break;
                                }
                                Err(e) => {
                                    let msg = format!("{e:#}");
                                    if is_discrepancy(&msg) {
                                        // a rules discrepancy must be fixed, never skipped silently
                                        if !o.keep_going {
                                            return Err(e);
                                        }
                                        st.discrepancies += 1;
                                        let mut f = bad_file.lock().unwrap();
                                        if f.is_none() {
                                            *f = Some(std::fs::File::create(bad_path)?);
                                        }
                                        writeln!(f.as_mut().unwrap(), "{}", json!({"id": id, "seed": ms, "error": msg}))?;
                                        break;
                                    }
                                    tries += 1;
                                    if st.errors.len() < 10 {
                                        st.errors.push(format!("{id}: {msg}"));
                                    }
                                    eng.engine.restart()?;
                                    let _ = eng.request_with(&json!({"op": "hello"}), Duration::from_secs(900))?;
                                    if tries >= 3 {
                                        st.failed += 1;
                                        break;
                                    }
                                }
                            }
                        }
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if k == 0 && (n % 10 == 0 || n == matches) {
                            let el = t0.elapsed().as_secs_f64();
                            eprintln!("[backgammon generate] {n}/{matches} matches, {:.1} s/match/worker", el * workers as f64 / n as f64);
                        }
                    }
                    if !pending.is_empty() {
                        flush(&mut file, &mut pending)?;
                    }
                    st.engine_calls = eng.calls;
                    st.engine_seconds = eng.busy.as_secs_f64();
                    Ok(st)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("worker panicked")))).collect()
    });
    let mut total = GenStats::default();
    for r in results {
        let st = r?;
        merge(&mut total, &st);
        total.engine_calls += st.engine_calls;
        total.engine_seconds += st.engine_seconds;
    }
    Ok(total)
}

fn merge(a: &mut GenStats, b: &GenStats) {
    a.matches += b.matches;
    a.games += b.games;
    a.decisions += b.decisions;
    a.checker_plays += b.checker_plays;
    a.cube_decisions += b.cube_decisions;
    a.split_turns += b.split_turns;
    a.max_candidates = a.max_candidates.max(b.max_candidates);
    a.legal_checked += b.legal_checked;
    a.failed += b.failed;
    a.discrepancies += b.discrepancies;
    for e in &b.errors {
        if a.errors.len() < 10 {
            a.errors.push(e.clone());
        }
    }
}

// ---------------------------------------------------------------------------------------
// convert
// ---------------------------------------------------------------------------------------

/// Conversion counters of the transcript → record step (the rest: traj::Stats).
#[derive(Clone, Debug, Default, Serialize)]
pub struct MapStats {
    pub transcripts: usize,
    pub other_variant: usize,
    pub rejected: usize,
    pub resigned: usize,
    pub truncated: usize,
    pub split_turns: usize,
    /// decisions with a teacher distribution / an engine value
    pub soft_decisions: usize,
    pub engine_values: usize,
    pub bad_analysis: usize,
    pub errors: Vec<String>,
}

/// The records of every transcript under `input` whose rules are `cfg` (others are
/// counted in `other_variant`): parsed and mapped in parallel chunks, yielded in input
/// order; transcripts that fail the checks come out as `Err` (counted by the converter).
pub fn records<'a>(input: &Path, cfg: &'a BackgammonConfig, opt: WalkOptions, stats: &'a std::sync::Mutex<MapStats>)
    -> Result<impl Iterator<Item = Result<Record>> + 'a> {
    let mut lines = transcript_lines(input)?.peekable();
    let mut buf: std::collections::VecDeque<Result<Record>> = std::collections::VecDeque::new();
    Ok(std::iter::from_fn(move || loop {
        if let Some(r) = buf.pop_front() {
            return Some(r);
        }
        lines.peek()?;
        let chunk: Vec<Result<(String, String)>> = lines.by_ref().take(1024).collect();
        let mapped: Vec<Option<Result<Record>>> = chunk
            .into_par_iter()
            .map(|l| {
                let (lid, line) = match l {
                    Ok(x) => x,
                    Err(e) => return Some(Err(e)),
                };
                let t = match parse_transcript(&line) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.context(lid))),
                };
                let mut st = stats.lock().unwrap();
                st.transcripts += 1;
                if t.config != *cfg {
                    st.other_variant += 1;
                    return None;
                }
                drop(st);
                match to_record(&t, cfg, opt) {
                    Ok((rec, w)) => {
                        let mut st = stats.lock().unwrap();
                        st.split_turns += w.split_turns;
                        st.bad_analysis += w.bad_analysis;
                        st.resigned += w.resigned as usize;
                        st.truncated += w.truncated as usize;
                        st.soft_decisions += w.soft.len();
                        st.engine_values += w.values.len();
                        Some(Ok(rec))
                    }
                    Err(e) => {
                        let mut st = stats.lock().unwrap();
                        st.rejected += 1;
                        if st.errors.len() < 10 {
                            st.errors.push(format!("{e:#}"));
                        }
                        Some(Err(e.context(lid)))
                    }
                }
            })
            .collect();
        buf.extend(mapped.into_iter().flatten());
    }))
}

// ---------------------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
pub struct VerifyStats {
    pub transcripts: usize,
    pub games: usize,
    pub finished: usize,
    pub resigned: usize,
    pub moves: usize,
    pub cube_actions: usize,
    pub split_turns: usize,
    pub gammons: usize,
    pub backgammons: usize,
    pub drops: usize,
    pub crawford_games: usize,
    pub matches_checked: usize,
    pub mismatches: usize,
    pub errors: Vec<String>,
}

/// Replay every game of `t` from its own start score: legality, results; plus the
/// score / Crawford progression between games, and (no resignation) the whole match.
pub fn verify_transcript(t: &Transcript, st: &mut VerifyStats) {
    st.transcripts += 1;
    let base = &t.config;
    let n = base.match_length;
    let fail = |st: &mut VerifyStats, msg: String| {
        st.mismatches += 1;
        if st.errors.len() < 20 {
            st.errors.push(format!("{}: {msg}", t.id));
        }
    };
    let mut crawford_done = base.post_crawford;
    for (gi, game) in t.games.iter().enumerate() {
        st.games += 1;
        st.crawford_games += game.crawford as usize;
        let mut cfg = base.clone();
        if n > 0 {
            // the rules' Crawford flag for this game, from the scores and the history
            let want = base.crawford && n > 1 && !crawford_done && game.score.contains(&(n - 1));
            if want != game.crawford {
                fail(st, format!("game {gi}: recorded Crawford flag {}, the rules say {want}", game.crawford));
            }
            crawford_done |= game.crawford;
            cfg.score = game.score;
            cfg.post_crawford = base.crawford && n > 1 && !game.crawford && game.score.contains(&(n - 1));
            if gi > 0 {
                let prev = &t.games[gi - 1];
                if let (Some(w), Some(p)) = (prev.winner, prev.points) {
                    let mut s = prev.score;
                    s[w] += p;
                    if s != game.score {
                        fail(st, format!("game {gi}: start score {:?}, the previous game ended {s:?}", game.score));
                    }
                }
            }
        }
        if let Err(e) = Backgammon::check_config(&cfg) {
            fail(st, format!("game {gi}: {e}"));
            continue;
        }
        match walk(&cfg, std::slice::from_ref(game), WalkOptions::default()) {
            Ok(w) => {
                st.moves += w.moves;
                st.cube_actions += w.cube_actions;
                st.split_turns += w.split_turns;
                if w.resigned {
                    st.resigned += 1;
                } else {
                    st.finished += 1;
                    if let Some(r) = w.state.results().last() {
                        st.drops += r.dropped as usize;
                        st.gammons += (r.multiplier == 2) as usize;
                        st.backgammons += (r.multiplier == 3) as usize;
                    }
                }
            }
            Err(e) => fail(st, format!("game {gi}: {e:#}")),
        }
    }
    if !t.games.iter().any(|g| g.resigned) {
        st.matches_checked += 1;
        if let Err(e) = to_record(t, base, WalkOptions::default()) {
            fail(st, format!("whole match: {e:#}"));
        }
    }
}

// ---------------------------------------------------------------------------------------
// diff
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
pub struct DiffStats {
    pub positions: usize,
    pub agree: usize,
    pub mismatches: usize,
    pub doubles: usize,
    pub passes: usize,
    pub bar: usize,
    pub bearoff: usize,
    pub max_moves: usize,
    /// random boards (the rest: playout positions)
    pub random: usize,
    pub errors: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct DiffLine {
    board: [Vec<u8>; 2],
    dice: [u8; 2],
    moves: Vec<Vec<u8>>,
}

fn side_of(v: &[u8]) -> Result<bg::Side> {
    v.try_into().map_err(|_| anyhow!("a side has 25 entries, got {}", v.len()))
}

fn diff_one(st: &mut DiffStats, pos: &Pos, d: [u8; 2], moves: &[Vec<u8>]) {
    st.positions += 1;
    st.doubles += (d[0] == d[1]) as usize;
    st.bar += (pos.me[24] > 0) as usize;
    st.bearoff += pos.me[6..25].iter().all(|&c| c == 0) as usize;
    match compare_legal(pos, d[0].max(d[1]), d[0].min(d[1]), moves) {
        Ok(n) => {
            st.agree += 1;
            st.max_moves = st.max_moves.max(n);
            st.passes += (moves.is_empty() || moves.iter().all(|m| m.is_empty())) as usize;
        }
        Err(e) => {
            st.mismatches += 1;
            if st.errors.len() < 20 {
                st.errors.push(format!("{e:#}"));
            }
        }
    }
}

/// Differential test on a recorded fixture of gnubg legal-move lists.
pub fn diff_fixture(path: &Path) -> Result<DiffStats> {
    let mut st = DiffStats::default();
    for l in transcript_lines(path)? {
        let (_, line) = l?;
        let d: DiffLine = serde_json::from_str(&line)?;
        let pos = Pos { me: side_of(&d.board[1])?, opp: side_of(&d.board[0])? };
        pos.validate().map_err(|e| anyhow!(e))?;
        diff_one(&mut st, &pos, d.dice, &d.moves);
    }
    Ok(st)
}

/// A random legal position (my view) for the differential test: blocking points,
/// checkers on the bar, bear-offs, partial and forced moves — situations playouts reach
/// rarely. Both sides keep at least one checker in play.
pub fn random_position(rng: &mut Rng) -> Pos {
    'retry: loop {
        let mut p = Pos { me: [0; 25], opp: [0; 25] };
        for side in 0..2 {
            let off = if rng.next_f64() < 0.3 { rng.below(15) as u8 } else { 0 };
            let bar = if rng.next_f64() < 0.3 { 1 + rng.below(3) as u8 } else { 0 };
            let mut left = CHECKERS - off;
            let b = bar.min(left);
            left -= b;
            // home-only (bear-off) boards, or stacks anywhere
            let home = rng.next_f64() < 0.3;
            let mut tries = 0;
            while left > 0 {
                tries += 1;
                if tries > 200 {
                    continue 'retry;
                }
                let q = if home { 1 + rng.below(6) as usize } else { 1 + rng.below(24) as usize };
                // side 0 = me (point q), side 1 = the opponent (its point q = my 25 − q)
                let blocked = if side == 0 { false } else { p.me[24 - q] > 0 };
                if blocked {
                    continue;
                }
                let n = (1 + rng.below(4) as u8).min(left);
                if side == 0 {
                    p.me[q - 1] += n;
                } else {
                    p.opp[q - 1] += n;
                }
                left -= n;
            }
            if side == 0 {
                p.me[24] = b;
            } else {
                p.opp[24] = b;
            }
        }
        if off(&p.me) < CHECKERS && off(&p.opp) < CHECKERS && p.validate().is_ok() {
            return p;
        }
    }
}

/// Live differential test: positions from playouts of our engine (each move gnubg's
/// 0-ply choice with probability `gnubg_frac`, else a uniformly random legal move) and,
/// with probability `random_frac` per position, random legal boards
/// ([`random_position`], random dice); gnubg's legal list compared with ours at each;
/// optionally recorded as a fixture.
pub fn diff_live(spec: &EngineSpec, positions: usize, seed: u64, gnubg_frac: f64, random_frac: f64, write: Option<&Path>)
    -> Result<DiffStats> {
    let bridge = BridgeFile::new(spec)?;
    let mut eng = Gnubg::start(&EngineSpec { plies: 0, ..spec.clone() }, &bridge.path)?;
    let mut out = match write {
        Some(p) => {
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d)?;
            }
            let f = std::io::BufWriter::new(std::fs::File::create(p)?);
            Some(flate2::write::GzEncoder::new(f, flate2::Compression::best()))
        }
        None => None,
    };
    let mut st = DiffStats::default();
    let mut rng = Rng::new(seed);
    let mut query = |st: &mut DiffStats, pos: &Pos, hi: u8, lo: u8| -> Result<Vec<Vec<u8>>> {
        let r = eng.request(&json!({"op": "legal", "board": board_json(pos), "dice": [hi, lo]}))?;
        let moves = flat_moves(&r["moves"])?;
        diff_one(st, pos, [hi, lo], &moves);
        if let Some(o) = out.as_mut() {
            let line = DiffLine { board: [pos.opp.to_vec(), pos.me.to_vec()], dice: [hi, lo], moves: moves.clone() };
            writeln!(o, "{}", serde_json::to_string(&line)?)?;
        }
        Ok(moves)
    };
    let cfg = BackgammonConfig { match_length: 0, cube: false, ..Default::default() };
    let mut g = Backgammon::new_game(&cfg);
    let (mut outcomes, mut legal) = (Vec::new(), Vec::new());
    while st.positions < positions {
        if rng.next_f64() < random_frac {
            let pos = random_position(&mut rng);
            let (a, b) = (1 + rng.below(6) as u8, 1 + rng.below(6) as u8);
            query(&mut st, &pos, a.max(b), a.min(b))?;
            st.random += 1;
            continue;
        }
        match g.turn() {
            Turn::Terminal => g = Backgammon::new_game(&cfg),
            Turn::Chance => {
                outcomes.clear();
                g.chance_outcomes(&mut outcomes);
                let w: Vec<f64> = outcomes.iter().map(|&(_, p)| p).collect();
                let o = outcomes[rng.weighted(&w)].0;
                g.apply_chance(o).map_err(|e| anyhow!("{e}"))?;
            }
            Turn::Act(seats) => {
                let seat = seats.iter().next().unwrap();
                let (hi, lo) = g.roll();
                if g.dice_left() != Dice::roll(hi, lo) {
                    // inside a split turn: a uniformly random next step
                    legal.clear();
                    g.legal_actions(seat, &mut legal);
                    let a = legal[rng.below(legal.len() as u64) as usize];
                    g.apply_action(seat, a).map_err(|e| anyhow!("{e}"))?;
                    continue;
                }
                let pos = g.pos(seat);
                let moves = query(&mut st, &pos, hi, lo)?;
                if rng.next_f64() < gnubg_frac && !moves.is_empty() {
                    // gnubg's 0-ply choice (its list is sorted best first)
                    let t = apply_steps(&pos, &steps_of(&moves[0])?).map_err(|e| anyhow!(e))?;
                    for a in g.plan_move(&t).map_err(|e| anyhow!("gnubg's 0-ply move: {e}"))? {
                        g.apply_action(seat, a).map_err(|e| anyhow!("{e}"))?;
                    }
                } else {
                    legal.clear();
                    g.legal_actions(seat, &mut legal);
                    let a = legal[rng.below(legal.len() as u64) as usize];
                    g.apply_action(seat, a).map_err(|e| anyhow!("{e}"))?;
                }
            }
        }
    }
    drop(query);
    if let Some(o) = out {
        o.finish()?.flush()?;
    }
    Ok(st)
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

fn parse_cfg(config: &str, match_length: Option<u32>) -> Result<BackgammonConfig> {
    let mut cfg: BackgammonConfig = if config.trim().is_empty() { BackgammonConfig::default() } else { serde_json::from_str(config)? };
    if let Some(n) = match_length {
        cfg.match_length = n;
    }
    Backgammon::check_config(&cfg).map_err(|e| anyhow!("config: {e}"))?;
    Ok(cfg)
}

#[derive(clap::Args, Debug, Clone)]
pub struct EngineArgs {
    /// Python command running the gnubg bridge (split on whitespace)
    #[arg(long, default_value = DEFAULT_PYTHON)]
    pub python: String,
    /// bridge script (default: the copy embedded in this binary)
    #[arg(long)]
    pub bridge: Option<PathBuf>,
    /// per-request timeout, seconds
    #[arg(long, default_value_t = 120)]
    pub timeout: u64,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// gnubg self-play matches (our engine runs the games; gnubg decides) → transcripts
    Generate {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100)]
        matches: usize,
        /// global index of the first match (disjoint ranges for parallel jobs)
        #[arg(long, default_value_t = 0)]
        first_match: usize,
        /// points per match (0 = money games); overrides --config's
        #[arg(long)]
        match_length: Option<u32>,
        /// gmrules BackgammonConfig JSON (default: 7-point match, Crawford)
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long, default_value_t = 4)]
        workers: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// gnubg evaluation plies (2 = gnubg's "world class" play)
        #[arg(long, default_value_t = 2)]
        plies: u32,
        /// analysed moves kept per checker play (soft targets)
        #[arg(long, default_value_t = 8)]
        top: usize,
        /// do not compare gnubg's legal move lists with ours
        #[arg(long)]
        no_verify_legal: bool,
        /// log gnubg / gmrules disagreements to discrepancies-*.ndjson and go on (default:
        /// abort the run)
        #[arg(long)]
        keep_going: bool,
        /// overwrite existing output files
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Transcripts (.jsonl / .jsonl.gz) → candidate-action gm-shards (every record verified)
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// convert the transcripts of this points-per-match (0 = money); overrides --config's
        #[arg(long)]
        match_length: Option<u32>,
        /// gmrules BackgammonConfig JSON the transcripts must have been played under
        #[arg(long, default_value = "")]
        config: String,
        /// soft targets: the K most probable analysed candidates per labelled decision
        #[arg(long, default_value_t = 0)]
        soft_k: usize,
        /// soft-target temperature τ (equity units): π ∝ exp(Q / τ)
        #[arg(long, default_value_t = 0.02)]
        soft_tau: f64,
        /// records (whole matches, ~430 frames each) per shard
        #[arg(long, default_value_t = 64)]
        shard_size: usize,
        #[arg(long)]
        max_records: Option<usize>,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// keep unfinished records (resigned games; truncations in the causality tests)
        #[arg(long)]
        allow_truncated: bool,
        /// write gnubg's evaluations as engine values (tgt.engine_value / engine_value_mask:
        /// match play 2 MWC - 1, money play clamp(equity x cube / money_scale))
        #[arg(long)]
        engine_values: bool,
    },
    /// Replay transcripts game by game (legality, results, scores, Crawford); exit 1 on a mismatch
    Verify {
        #[arg(long)]
        input: PathBuf,
    },
    /// Legal-move differential test against gnubg (live bridge or recorded fixture)
    Diff {
        /// a fixture of gnubg legal-move lists (.jsonl / .jsonl.gz)
        #[arg(long)]
        fixture: Option<PathBuf>,
        /// query a live gnubg bridge on positions from playouts
        #[arg(long)]
        live: bool,
        #[arg(long, default_value_t = 1000)]
        positions: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// share of playout moves chosen by gnubg (0-ply) instead of uniformly at random
        #[arg(long, default_value_t = 0.5)]
        gnubg_frac: f64,
        /// share of positions that are random legal boards instead of playout positions
        #[arg(long, default_value_t = 0.3)]
        random_frac: f64,
        /// record the live positions and gnubg's lists as a fixture (.jsonl.gz)
        #[arg(long)]
        write: Option<PathBuf>,
        #[command(flatten)]
        engine: EngineArgs,
    },
}

fn spec_of(e: &EngineArgs, plies: u32) -> EngineSpec {
    EngineSpec { python: e.python.clone(), bridge: e.bridge.clone(), plies, timeout: Duration::from_secs(e.timeout) }
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Generate { out, matches, first_match, match_length, config, workers, seed, plies, top, no_verify_legal, keep_going,
                        force, engine } => {
            let cfg = parse_cfg(&config, match_length)?;
            let o = GenOptions { first: first_match, matches, workers, seed, top, verify_legal: !no_verify_legal, keep_going, force };
            let st = generate(&out, &cfg, &spec_of(&engine, plies), &o)?;
            println!("{}", serde_json::to_string(&st)?);
            Ok(())
        }
        Cmd::Convert { input, out, match_length, config, soft_k, soft_tau, shard_size, max_records, store_events, allow_truncated,
                       engine_values } => {
            let cfg = parse_cfg(&config, match_length)?;
            ensure!(soft_tau > 0.0, "--soft-tau must be > 0");
            let ms = std::sync::Mutex::new(MapStats::default());
            let wopt = WalkOptions { tau: if soft_k > 0 { soft_tau } else { 0.0 }, allow_truncated };
            let recs = records(&input, &cfg, wopt, &ms)?.take(max_records.unwrap_or(usize::MAX));
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, soft_k, engine_values };
            let st = traj::convert::<Backgammon>(&cfg, recs, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            v["transcripts"] = serde_json::to_value(ms.into_inner().unwrap())?;
            println!("{v}");
            Ok(())
        }
        Cmd::Verify { input } => {
            let mut st = VerifyStats::default();
            for l in transcript_lines(&input)? {
                let (id, line) = l?;
                match parse_transcript(&line) {
                    Ok(t) => verify_transcript(&t, &mut st),
                    Err(e) => {
                        st.mismatches += 1;
                        st.errors.push(format!("{id}: {e:#}"));
                    }
                }
            }
            println!("{}", serde_json::to_string(&st)?);
            if st.mismatches > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
        Cmd::Diff { fixture, live, positions, seed, gnubg_frac, random_frac, write, engine } => {
            let st = match (fixture, live) {
                (Some(f), false) => diff_fixture(&f)?,
                (None, true) => diff_live(&spec_of(&engine, 0), positions, seed, gnubg_frac, random_frac, write.as_deref())?,
                _ => bail!("give exactly one of --fixture FILE / --live"),
            };
            println!("{}", serde_json::to_string(&st)?);
            if st.mismatches > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opening_game() -> GameRec {
        GameRec { score: [0, 0], crawford: false, actions: Vec::new(), winner: None, points: None, resigned: false }
    }

    fn opts(tau: f64) -> WalkOptions {
        WalkOptions { tau, allow_truncated: false }
    }

    #[test]
    fn ids_and_softmax() {
        assert_eq!(roll_id(3, 5), roll_id(5, 3));
        assert_eq!(roll_id(6, 6), 35);
        assert_eq!(opening_id(1, 6, 2), bg::OPENING_BASE + 6 + 5);
        let p = softmax(&[(1, 0.10), (2, 0.08), (3, -0.5)], 0.02);
        assert!((p.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[0].1 > p[1].1 && p[2].1 < 1e-6);
        // engine values: match play 2·MWC − 1, money play equity × cube / scale, clamped
        let m7 = BackgammonConfig::default();
        let money = BackgammonConfig { match_length: 0, ..Default::default() };
        assert_eq!(value_of(&m7, 1, Some(0.3), Some(0.75)), Some(0.5));
        assert_eq!(value_of(&m7, 1, Some(0.3), None), None);
        assert_eq!(value_of(&money, 2, Some(0.5), None), Some(0.25));
        assert_eq!(value_of(&money, 8, Some(1.0), Some(0.9)), Some(1.0));
    }

    /// A transcript written by hand: opening, implicit and explicit no-doubles, a
    /// double / take, a drop; the walk maps it and the checks catch tampering.
    #[test]
    fn walk_maps_and_checks() {
        let cfg = BackgammonConfig { match_length: 3, ..Default::default() };
        let mv = |p: usize, d: [u8; 2], m: &[u8]| Action { dice: Some(d), mv: Some(m.to_vec()), ..Action::new(p, "move") };
        let mut game = opening_game();
        let cube = CubeEq { nd: Some(0.2), dt: Some(0.1), dp: Some(1.0), proper: None, mwc: Some([0.6, 0.55, 0.7]) };
        game.actions = vec![
            Action { mwc: Some(0.52), ..mv(0, [3, 1], &[8, 5, 6, 5]) }, // opening 3-1: 8/5 6/5
            mv(1, [6, 4], &[24, 18, 13, 9]),                             // implicit no double
            Action { cube: Some(cube), ..Action::new(0, "double") },
            Action::new(1, "take"),
            mv(0, [2, 1], &[13, 11, 11, 10]),
            Action::new(1, "double"),
            Action::new(0, "drop"),
        ];
        game.winner = Some(1);
        game.points = Some(2);
        let w = walk(&cfg, std::slice::from_ref(&game), opts(0.02)).unwrap();
        assert_eq!((w.games, w.moves, w.cube_actions, w.resigned, w.truncated), (1, 3, 5, false, false));
        assert_eq!(w.state.score(), [0, 2]);
        assert_eq!(w.events[0], Event::Chance(opening_id(0, 3, 1)));
        assert_eq!(w.events[2], Event::Act(1, bg::NO_DOUBLE));
        assert_eq!(w.soft.len(), 2, "the explicit double and its take carry teacher targets");
        // values: the opening move (2·0.52 − 1), the double (best action: nd 0.6 vs
        // min(dt, dp) 0.55 → 0.6), the taker (1 − min(dt, dp) = 0.45)
        let v: Vec<(usize, f32)> = w.values.clone();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].0, 1);
        assert!((v[0].1 - 0.04).abs() < 1e-6 && (v[1].1 - 0.2).abs() < 1e-6 && (v[2].1 + 0.1).abs() < 1e-6, "{v:?}");
        assert!(v.iter().all(|&(e, _)| matches!(w.events[e], Event::Act(..))));
        // the whole record replays through the generic driver
        let t = Transcript { format: FORMAT.into(), id: "t".into(), config: cfg.clone(), players: vec![], meta: Value::Null,
                             games: vec![game.clone()], winner: None, score: None };
        assert!(to_record(&t, &cfg, opts(0.0)).is_err(), "the match is not over after one 2-point game");
        let (rec, _) = to_record(&t, &cfg, WalkOptions { tau: 0.0, allow_truncated: true }).unwrap();
        assert!(rec.expected_returns.is_none() && rec.engine_values.len() == 3);
        // tampering is caught: wrong points, an illegal move, wrong dice, a missing end
        let mut bad = game.clone();
        bad.points = Some(1);
        assert!(walk(&cfg, &[bad], opts(0.0)).unwrap_err().to_string().contains("points"));
        let mut bad = game.clone();
        bad.actions[1] = mv(1, [6, 4], &[24, 18, 13, 8]);
        assert!(format!("{:#}", walk(&cfg, &[bad], opts(0.0)).unwrap_err()).contains("reaches"));
        let mut bad = game.clone();
        bad.actions[4] = mv(0, [3, 1], &[13, 11, 11, 10]); // the dice do not fit the move
        assert!(format!("{:#}", walk(&cfg, &[bad], opts(0.0)).unwrap_err()).contains("roll 3-1 reaches"));
        let mut bad = game.clone();
        bad.actions.truncate(5);
        assert!(walk(&cfg, std::slice::from_ref(&bad), opts(0.0)).unwrap_err().to_string().contains("stop before"));
        // ... unless truncated records are allowed: then the events are a prefix
        let cut = walk(&cfg, &[bad], WalkOptions { tau: 0.0, allow_truncated: true }).unwrap();
        assert!(cut.truncated && w.events.starts_with(&cut.events));
        let mut bad = game;
        bad.resigned = true;
        bad.actions.truncate(4);
        bad.actions.push(Action { value: Some(1), ..Action::new(0, "resign") });
        let w = walk(&cfg, &[bad], opts(0.0)).unwrap();
        assert!(w.resigned && w.games == 0);
    }

    /// The embedded bridge is the repository's script.
    #[test]
    fn bridge_is_embedded() {
        assert!(BRIDGE_PY.contains("def handle(req, plies"));
    }
}
