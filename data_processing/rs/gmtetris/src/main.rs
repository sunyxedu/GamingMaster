//! `gmtetris selfplay` — versus-Tetris games between two Cold Clear bots, played on
//! the gmrules `tetris` game (its rules decide everything: pieces, garbage, the end),
//! written as trajectory JSONL for `gmdata rules convert --game tetris`.
//!
//! ```sh
//! gmtetris selfplay --games 1000 --nodes 20000 --workers 8 --seed 1 --out raw/sp-1.jsonl.gz
//! gmdata rules convert --game tetris --input raw/sp-1.jsonl.gz --out shards/train
//! ```
//!
//! Each decision: the bot of the seat (its own tree, kept across turns: Cold Clear's
//! normal mode, `--nodes` search nodes per move, 1 thread, no opening book) is told the
//! pieces it has not seen (the shared sequence up to its current piece + 5), reset to
//! our field when garbage arrived, and asked for a move; the move's placement (hold
//! flag + final position + T-spin status, canonicalised) MUST be one of our legal
//! actions — a mismatch drops the game and is counted (`rules_mismatch`). Both seats
//! decide on the same state (a simultaneous turn); chance events (pieces from the 7-bag,
//! hole columns) come from `--seed`. Cold Clear samples its tree with thread_rng, so the
//! games are not reproducible from the seed — the JSONL records every event.
//!
//! When a bot's tree sees only losing moves it gives no move; the seat then plays its
//! lowest legal placement (counted in `meta.fallback_moves`, the game is lost anyway).
//!
//! Per game `meta`: the nodes of each seat, turns, pieces / attack / lines per seat
//! (attack per piece is the strength measure of versus play), the result.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use cold_clear::evaluation::Standard;
use cold_clear::{BotState, Options};
use gmrules::games::tetris::{action_id, Falling, Piece as GPiece, Tetris, TetrisConfig, NEXT, TSPIN_FULL, TSPIN_MINI,
                             TSPIN_NONE};
use gmrules::{Game, Rng, Turn};
use libtetris::{Board, FallingPiece, Piece, RotationState, TspinStatus};
use rayon::prelude::*;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Parser)]
#[command(about = "Versus-Tetris self-play with Cold Clear on the gmrules tetris game")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Cold Clear vs Cold Clear → trajectory JSONL(.gz)
    Selfplay {
        #[arg(long, default_value_t = 100)]
        games: usize,
        /// search nodes per move (both seats, unless --nodes-b)
        #[arg(long, default_value_t = 20000)]
        nodes: u32,
        /// search nodes per move of seat 1 (strength comparisons)
        #[arg(long)]
        nodes_b: Option<u32>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        /// the gmrules tetris config (JSON, "" = default)
        #[arg(long, default_value = "")]
        config: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn cc_piece(p: GPiece) -> Piece {
    match p {
        GPiece::I => Piece::I,
        GPiece::O => Piece::O,
        GPiece::T => Piece::T,
        GPiece::L => Piece::L,
        GPiece::J => Piece::J,
        GPiece::S => Piece::S,
        GPiece::Z => Piece::Z,
    }
}

fn our_piece(p: Piece) -> GPiece {
    match p {
        Piece::I => GPiece::I,
        Piece::O => GPiece::O,
        Piece::T => GPiece::T,
        Piece::L => GPiece::L,
        Piece::J => GPiece::J,
        Piece::S => GPiece::S,
        Piece::Z => GPiece::Z,
    }
}

/// A Cold Clear placement as our canonical falling piece.
fn our_falling(f: &FallingPiece) -> Falling {
    let rot = match f.kind.1 {
        RotationState::North => 0,
        RotationState::East => 1,
        RotationState::South => 2,
        RotationState::West => 3,
    };
    let tspin = match f.tspin {
        TspinStatus::None => TSPIN_NONE,
        TspinStatus::Mini => TSPIN_MINI,
        TspinStatus::Full => TSPIN_FULL,
    };
    Falling { piece: our_piece(f.kind.0), rot, x: f.x, y: f.y, tspin }.canonical()
}

fn field_of(g: &Tetris, seat: usize) -> [[bool; 10]; 40] {
    let mut out = [[false; 10]; 40];
    for (y, row) in out.iter_mut().enumerate() {
        for (x, c) in row.iter_mut().enumerate() {
            *c = g.players[seat].field.rows[y] & (1 << x) != 0;
        }
    }
    out
}

struct Seat {
    bot: BotState<Standard>,
    /// sequence pieces given to the bot so far
    given: usize,
}

fn cc_options(nodes: u32) -> Options {
    Options { max_nodes: nodes, min_nodes: 0, threads: 1, ..Options::default() }
}

/// One game; Err = a rules mismatch (the message).
fn play(cfg: &TetrisConfig, nodes: [u32; 2], seed: u64, id: &str) -> Result<serde_json::Value, String> {
    let mut g = Tetris::new_game(cfg);
    let mut rng = Rng::new(seed);
    let mut events: Vec<[i64; 2]> = Vec::new();
    let eval = Standard::default();
    let mut seats: Vec<Option<Seat>> = vec![None, None];
    let mut ch = Vec::new();
    let mut legal = Vec::new();
    let mut garbage_seen = [0usize; 2];
    let mut fallback = [0u32; 2];
    loop {
        match g.turn() {
            Turn::Terminal => break,
            Turn::Chance => {
                ch.clear();
                g.chance_outcomes(&mut ch);
                let w: Vec<f64> = ch.iter().map(|c| c.1).collect();
                let o = ch[rng.weighted(&w)].0;
                g.apply_chance(o).map_err(|e| e.0)?;
                events.push([-1, o as i64]);
            }
            Turn::Act(set) => {
                let mut chosen = [0u32; 2];
                for s in set.iter() {
                    let p = &g.players[s];
                    let upto = p.pos + 1 + NEXT;
                    let seat = seats[s].get_or_insert_with(|| {
                        // the bot starts from a board holding its first pieces
                        let mut b = Board::new();
                        for &q in &g.seq[..upto] {
                            b.add_next_piece(cc_piece(q));
                        }
                        Seat { bot: BotState::new(b, cc_options(nodes[s])), given: upto }
                    });
                    while seat.given < upto {
                        seat.bot.add_next_piece(cc_piece(g.seq[seat.given]));
                        seat.given += 1;
                    }
                    // garbage arrived (or any change the bot cannot know): our field is the truth
                    if garbage_seen[s] != garbage_count(&g, s) {
                        garbage_seen[s] = garbage_count(&g, s);
                        seat.bot.reset(field_of(&g, s), p.b2b, p.combo);
                    }
                    let mv = loop {
                        match seat.bot.think() {
                            Ok(t) => {
                                let r = t.think(&eval);
                                seat.bot.finish_thinking(r);
                            }
                            Err(_) => {
                                if let Some((mv, _)) = seat.bot.suggest_move(&eval, None, p.pending_lines()) {
                                    break Some(mv);
                                }
                                break None; // its tree sees only losing moves
                            }
                        }
                    };
                    legal.clear();
                    g.legal_actions(s, &mut legal);
                    let Some(mv) = mv else {
                        // the bot gave up (a lost position): our lowest legal placement,
                        // the bot reset to our field next turn
                        fallback[s] += 1;
                        chosen[s] = legal[0];
                        garbage_seen[s] = usize::MAX;
                        continue;
                    };
                    let f = our_falling(&mv.expected_location);
                    let a = action_id(mv.hold, &f);
                    if legal.binary_search(&a).is_err() {
                        return Err(format!("{id}: seat {s} turn {}: Cold Clear's move {:?} hold {} ({}) is not legal here\n{}",
                                           g.turn_no, mv.expected_location, mv.hold, g.action_string(s, a), g.state_string()));
                    }
                    seat.bot.advance_move(mv.expected_location);
                    chosen[s] = a;
                }
                for s in set.iter() {
                    g.apply_action(s, chosen[s]).map_err(|e| e.0)?;
                    events.push([s as i64, chosen[s] as i64]);
                }
            }
        }
    }
    let ret = g.returns();
    let stats: Vec<_> = g.players.iter().map(|p| json!({"pieces": p.pieces, "attack": p.attack, "lines": p.lines,
        "app": if p.pieces > 0 { p.attack as f64 / p.pieces as f64 } else { 0.0 }})).collect();
    Ok(json!({"id": id, "events": events, "returns": ret,
              "meta": {"teacher": "cold-clear", "nodes": nodes, "turns": g.turn_no, "fallback_moves": fallback, "players": stats,
                       "result": match (ret[0] > 0.0, ret[1] > 0.0) { (true, _) => "p0", (_, true) => "p1", _ => "draw" }}}))
}

/// Garbage lines a player has received so far.
fn garbage_count(g: &Tetris, seat: usize) -> usize {
    g.players[seat].received as usize
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Selfplay { games, nodes, nodes_b, seed, workers, config, out } => {
            let cfg: TetrisConfig = if config.trim().is_empty() { TetrisConfig::default() } else { serde_json::from_str(&config)? };
            Tetris::check_config(&cfg).map_err(|e| anyhow::anyhow!(e.0))?;
            let nodes = [nodes, nodes_b.unwrap_or(nodes)];
            let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
            let done = AtomicUsize::new(0);
            let t0 = std::time::Instant::now();
            let results: Vec<Result<serde_json::Value, String>> = pool.install(|| {
                (0..games).into_par_iter().map(|i| {
                    let r = play(&cfg, nodes, seed.wrapping_mul(1_000_003).wrapping_add(i as u64), &format!("s{seed}-g{i}"));
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 10 == 0 || n == games {
                        eprintln!("[{:.0}s] {n}/{games} games", t0.elapsed().as_secs_f64());
                    }
                    r
                }).collect()
            });
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let file = std::fs::File::create(&out).with_context(|| format!("{}", out.display()))?;
            let mut w: Box<dyn Write> = if out.extension().is_some_and(|e| e == "gz") {
                Box::new(flate2::write::GzEncoder::new(file, flate2::Compression::default()))
            } else {
                Box::new(file)
            };
            let (mut ok, mut bad, mut p0, mut p1, mut draw, mut turns, mut app) = (0, 0, 0, 0, 0, 0u64, [0f64; 2]);
            for r in &results {
                match r {
                    Ok(v) => {
                        ok += 1;
                        match v["meta"]["result"].as_str() {
                            Some("p0") => p0 += 1,
                            Some("p1") => p1 += 1,
                            _ => draw += 1,
                        }
                        turns += v["meta"]["turns"].as_u64().unwrap_or(0);
                        for s in 0..2 {
                            app[s] += v["meta"]["players"][s]["app"].as_f64().unwrap_or(0.0);
                        }
                        writeln!(w, "{v}")?;
                    }
                    Err(e) => {
                        bad += 1;
                        if bad <= 3 {
                            eprintln!("rules mismatch: {e}");
                        }
                    }
                }
            }
            w.flush()?;
            drop(w);
            let k = ok.max(1) as f64;
            println!("{}", json!({"games": games, "written": ok, "rules_mismatch": bad, "p0_wins": p0, "p1_wins": p1, "draws": draw,
                                  "mean_turns": turns as f64 / k, "app": [app[0] / k, app[1] / k], "nodes": nodes,
                                  "seconds": t0.elapsed().as_secs_f64()}));
            if ok == 0 {
                bail!("no game written");
            }
        }
    }
    Ok(())
}
