//! Gomoku: Gomocup `.psq` records → shards for the "gomoku" adapter
//! (BoardEncoder + PlacementDecoder, 3 planes, no pass).
//!
//!   obs.board_seq [T,3,N,N] f32  mover stones / opponent stones / empty
//!   tgt.place_idx [T] i64        row*N+col
//!   tgt.passed    [T] bool       all false
//!   tgt.value     [T] f32        ±1 for the frame's mover
//!
//! Rules: freestyle (five or more) or standard (exactly five). Only games whose
//! last move completes a win are decisive; others are skipped unless
//! `--keep-undecided` (winner = last mover). Shard meta records `board_size`.

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const EMPTY: u8 = 0;
const BLACK: u8 = 1;
const WHITE: u8 = 2;

fn opponent(p: u8) -> u8 {
    BLACK + WHITE - p
}

pub fn wins(b: &[u8], n: i32, player: u8, row: i32, col: i32, exact_five: bool) -> bool {
    for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
        let mut run = 1;
        for sign in [1, -1] {
            let (mut r, mut c) = (row + sign * dr, col + sign * dc);
            while (0..n).contains(&r) && (0..n).contains(&c) && b[(r * n + c) as usize] == player {
                run += 1;
                r += sign * dr;
                c += sign * dc;
            }
        }
        if run == 5 || (run > 5 && !exact_five) {
            return true;
        }
    }
    false
}

/// (width, height, moves as 0-based (row, col)); Err(reason) on unusable files.
pub fn parse_psq(text: &str) -> std::result::Result<(i32, i32, Vec<(i32, i32)>), String> {
    let mut lines = text.lines();
    let first = lines.next().ok_or("empty file")?;
    // header: first "<digits>x<digits>" (ASCII)
    let bytes = first.as_bytes();
    let mut dims = None;
    'scan: for i in 0..bytes.len() {
        if bytes[i].is_ascii_digit() && (i == 0 || !bytes[i - 1].is_ascii_digit()) {
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'x' {
                let mut k = j + 1;
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                if k > j + 1 {
                    dims = Some((first[i..j].parse::<i32>().unwrap_or(0), first[j + 1..k].parse::<i32>().unwrap_or(0)));
                    break 'scan;
                }
            }
        }
    }
    let (w, h) = dims.ok_or_else(|| format!("unrecognized header {first:?}"))?;
    let mut moves = Vec::new();
    for line in lines {
        match parse_move_line(line) {
            Some((x, y)) => moves.push((y - 1, x - 1)),
            None => {
                if !moves.is_empty() {
                    break;
                }
            }
        }
    }
    Ok((w, h, moves))
}

/// Engine names from the trailer (the two lines ending in `.zip` after the moves).
pub fn trailer_engines(text: &str) -> Option<(String, String)> {
    let names: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| l.to_lowercase().ends_with(".zip")).collect();
    (names.len() >= 2).then(|| (names[0].to_string(), names[1].to_string()))
}

/// `^\s*(\d+),(\d+)(?:,(-?\d+))?\s*$`
fn parse_move_line(line: &str) -> Option<(i32, i32)> {
    let t = line.trim_matches(|c: char| c.is_whitespace());
    let parts: Vec<&str> = t.split(',').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(parts[0]) || !digits(parts[1]) {
        return None;
    }
    if parts.len() == 3 {
        let p = parts[2].strip_prefix('-').unwrap_or(parts[2]);
        if !digits(p) {
            return None;
        }
    }
    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
}

fn game_to_sample(moves: &[(i32, i32)], n: i32, exact_five: bool, keep_undecided: bool, min_moves: usize, source: &str)
    -> std::result::Result<Sample, String> {
    if moves.len() < min_moves.max(1) {
        return Err("too_short".into());
    }
    let mut b = vec![EMPTY; (n * n) as usize];
    let mut player = BLACK;
    let (mut planes, mut movers, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    let mut winner: Option<u8> = None;
    for (k, &(r, c)) in moves.iter().enumerate() {
        if winner.is_some() {
            return Err(format!("moves_after_win (move {k})"));
        }
        let opp = opponent(player);
        for want in [player, opp, EMPTY] {
            planes.extend(b.iter().map(|&v| (v == want) as u8 as f32));
        }
        movers.push(player);
        if !(0..n).contains(&r) || !(0..n).contains(&c) {
            return Err(format!("illegal_move (({r}, {c}) off a {n}x{n} board)"));
        }
        let i = (r * n + c) as usize;
        if b[i] != EMPTY {
            return Err(format!("illegal_move (({r}, {c}) already occupied)"));
        }
        b[i] = player;
        idx.push((r * n + c) as i64);
        if wins(&b, n, player, r, c, exact_five) {
            winner = Some(player);
        }
        player = opponent(player);
    }
    let winner = match winner {
        Some(w) => w,
        None if keep_undecided => *movers.last().unwrap(),
        None => return Err("undecided".into()),
    };
    let t = movers.len();
    let value: Vec<f32> = movers.iter().map(|&m| if m == winner { 1.0 } else { -1.0 }).collect();
    let nn = n as usize;
    Ok(Sample::new(json!({"source": source, "plies": t, "winner": if winner == BLACK { "black" } else { "white" }}))
        .bits("obs.board_seq", &[3, nn, nn], &planes)
        .i64("tgt.place_idx", &[], idx)
        .bool("tgt.passed", &[], vec![false; t])
        .f32("tgt.value", &[], value))
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Convert .psq files (recursive) to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 512)]
        shard_size: usize,
        #[arg(long)]
        max_games: Option<usize>,
        #[arg(long, default_value_t = 9)]
        min_moves: usize,
        #[arg(long, default_value_t = 15)]
        board_size: i32,
        /// freestyle (five or more) | standard (exactly five)
        #[arg(long, default_value = "freestyle")]
        rule: String,
        #[arg(long)]
        keep_undecided: bool,
        /// keep only games where both engines are among the N best by win rate over the
        /// input directory (computed in a first pass; engine names from the psq trailer)
        #[arg(long)]
        top_engines: Option<usize>,
    },
}

#[allow(clippy::too_many_arguments)]
/// Engine → win rate over decisive games in `files`; engines with fewer than 10 games are unranked.
fn engine_ranking(files: &[PathBuf], n: i32, exact_five: bool) -> Vec<(String, f64, usize)> {
    let mut rec: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for f in files {
        let Ok(bytes) = std::fs::read(f) else { continue };
        let text = String::from_utf8_lossy(&bytes);
        let (Ok((w, h, moves)), Some((e1, e2))) = (parse_psq(&text), trailer_engines(&text)) else { continue };
        if (w, h) != (n, n) || moves.is_empty() {
            continue;
        }
        let mut b = vec![EMPTY; (n * n) as usize];
        let mut player = BLACK;
        let mut winner = None;
        for &(r, c) in &moves {
            if !(0..n).contains(&r) || !(0..n).contains(&c) || b[(r * n + c) as usize] != EMPTY {
                break;
            }
            b[(r * n + c) as usize] = player;
            if wins(&b, n, player, r, c, exact_five) {
                winner = Some(player);
                break;
            }
            player = opponent(player);
        }
        let Some(wp) = winner else { continue };
        let (we, le) = if wp == BLACK { (&e1, &e2) } else { (&e2, &e1) };
        rec.entry(we.clone()).or_default().0 += 1;
        rec.entry(we.clone()).or_default().1 += 1;
        rec.entry(le.clone()).or_default().1 += 1;
    }
    let mut v: Vec<(String, f64, usize)> = rec.into_iter().filter(|(_, (_, g))| *g >= 10).map(|(e, (w, g))| (e, w as f64 / g as f64, g)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

#[allow(clippy::too_many_arguments)]
pub fn convert(input: &Path, out: &Path, shard_size: usize, max_games: Option<usize>, min_moves: usize, board_size: i32,
               rule: &str, keep_undecided: bool, top_engines: Option<usize>) -> Result<serde_json::Value> {
    anyhow::ensure!((5..=25).contains(&board_size), "board_size must be within [5, 25], got {board_size}");
    anyhow::ensure!(rule == "freestyle" || rule == "standard", "unknown rule {rule:?}");
    let exact_five = rule == "standard";
    let mut w = ShardWriter::new(out, "gomoku", 1, json!({"board_size": board_size}), shard_size)?;
    let mut stats: BTreeMap<String, usize> = BTreeMap::new();
    let (mut seen, mut kept) = (0usize, 0usize);
    let files = common::find_files(input, &[".psq"])?;
    let top: Option<std::collections::HashSet<String>> = top_engines.map(|k| {
        let r = engine_ranking(&files, board_size, exact_five);
        eprintln!("engine ranking (win rate, games): {}", r.iter().take(k).map(|(e, w, g)| format!("{e} {w:.2}/{g}")).collect::<Vec<_>>().join(", "));
        r.into_iter().take(k).map(|x| x.0).collect()
    });
    for path in files {
        seen += 1;
        let text = String::from_utf8_lossy(&std::fs::read(&path)?).to_string();
        if let Some(t) = &top {
            match trailer_engines(&text) {
                Some((a, b)) if t.contains(&a) && t.contains(&b) => {}
                _ => {
                    *stats.entry("low_quality".into()).or_default() += 1;
                    continue;
                }
            }
        }
        let res = parse_psq(&text).and_then(|(wd, ht, moves)| {
            if (wd, ht) != (board_size, board_size) {
                return Err("wrong_board_size".to_string());
            }
            game_to_sample(&moves, board_size, exact_five, keep_undecided, min_moves,
                           &path.file_name().unwrap().to_string_lossy())
        });
        match res {
            Ok(s) => {
                w.push(s)?;
                kept += 1;
                if max_games.map(|m| kept >= m).unwrap_or(false) {
                    break;
                }
            }
            Err(e) => *stats.entry(e.split(' ').next().unwrap_or("").to_string()).or_default() += 1,
        }
    }
    w.flush()?;
    let mut v = json!({"seen": seen, "kept": kept, "shards": w.written_shards});
    for (k, n) in stats {
        v[k] = json!(n);
    }
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Convert { input, out, shard_size, max_games, min_moves, board_size, rule, keep_undecided, top_engines } => {
            println!("{}", convert(&input, &out, shard_size, max_games, min_moves, board_size, &rule, keep_undecided, top_engines)?);
            Ok(())
        }
    }
}
