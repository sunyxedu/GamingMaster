//! Skat — the International Skat Server's game archive → candidate-action gm-shards
//! (rules and frames: `gmrules::games::skat`, the simulator's code).
//!
//! Source: <https://skatgame.net/iss/> "DOWNLOAD ISS GAMES": `iss-games-07-2024.sgf.bz2`
//! (545 MiB bzip2, 9.1M games 2007-10 .. 2024-07, humans and programs — kermit, zoot,
//! theCount, xskat, bernie, ... — "All games are stored and made available for free to
//! everyone"; the server and its data are by Michael Buro, University of Alberta; the
//! GPLv3 client sources document the record format, `doc/skat-game-format`).
//! `gmdata skat fetch` downloads it (plus the ISS ratings page); the full-scale
//! pipeline is `scripts/fetch_skat.sh` (fetch → verify → stats → convert --hq).
//!
//! Record format: one game per line,
//!
//! ```text
//! (;GM[Skat]PC[Internet Skat Server]CO[]SE[53]ID[9]DT[2007-10-29/04:58:00/UTC]P0[jeff]P1[Montana]P2[vaun]
//!   R0[0.0]R1[0.0]R2[0.0]MV[w S7.ST.CA.DT....SJ.D8 1 18 0 p 2 20 1 p 2 CH 0 SQ 1 H9 2 SA ... ]
//!   R[d:2 loss v:-96 m:2 bidok p:56 t:5 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)
//! ```
//!
//! `MV` = `<who> <move>` pairs, who `w` (the server) or a seat (0 forehand, 1 middlehand,
//! 2 rearhand = `P0..P2`): the deal (32 cards: 10 per seat in seat order, then the
//! skat), bids (numbers) / `y` / `p`, `s` (pick up; the server answers with the skat
//! `w XX.YY`) or a hand declaration (`GH`, `CHS`, `GO`, `NOH`, ...), after the pickup
//! `<game>.<discard>.<discard>` (`C.DK.D8`, `NO.CT.CK`), cards (`SQ`), and the ISS-only
//! moves `RE` (resign), `SC` (the declarer shows his cards), `w TI.<seat>` (timeout),
//! `w LE.<seat>` (a player left). `R0..R2` carry no ratings (0.0 / empty).
//!
//! Every record is REPLAYED by the gmrules skat rules — each move checked against
//! the legal actions (bids, answers, declarations with their bid limits, discards,
//! follow-suit) — and the recorded result `R[...]` must be reproduced EXACTLY by
//! `Skat::iss_result` (declarer, win / loss, value, matadors, overbid, card points,
//! tricks, schneider, schwarz, penalties, leave / timeout, resignation): the parity
//! proof of the rules on real data (`gmdata skat verify` reports it for any corpus;
//! `--dump-rejected FILE` keeps the rejected lines). Moves are read as the ISS server
//! reads them: a pickup declaration may come as a half-move (`D`, then `DK.D8`); the
//! cards shown with an ouvert declaration are not checked (the server shows the
//! defenders the declarer's actual hand, as the frames do; 43 records of 2024 show
//! wrong ones).
//!
//! The full archive (`verify --input https://skatgame.net/iss/iss-games-07-2024.sgf.bz2`,
//! streamed, ~2.5 min, bound by the bzip2 stream): 9,094,699 records, 9,094,590
//! verified — 6,139,697 finished, 354,574 passed in, 2,492,924 resigned, 107,395 aborted
//! (timeout / a player leaving) — and 0 result mismatches; 109 rejected: 75 bids off
//! the bid list (the ISS code takes any number, real Skat does not), 23 truncated
//! lines, 7 deals of 29 cards, 1 line holding two records, 3 card moves (`DH`) the
//! server's lossy card parser took for some card. The rules are also pinned to the ISS
//! engine itself by randomized differential tests (tests/test_skat_reference.py).
//!
//! Conversion (one sample per seat stream, per-seat layout; see gmrules skat for the
//! frames): games that end normally (incl. games passed in: three pass frames, value
//! 0) and games ended by resignations (the declarer's, or both defenders': scored by
//! ISS's rules on the final state — `traj::Record::conceded`) are converted; a single
//! defender's `RE` and `SC` are not moves of the game (dropped; a defender's frames
//! after an `SC` do not show the declarer's cards). Games ended by a timeout / a
//! player leaving (`TI` / `LE`, ~1%) are verified but skipped. A jump bid becomes
//! MORE sub-decisions + the bid; a hand declaration its type + modifier; a pickup
//! declaration the game + both discards in the recorded order.
//!
//! Quality filters (per seat: the other seats' streams are dropped, the value target
//! stays the game's result): `--player NAME` (repeatable), `--ratings FILE
//! --min-rating R [--max-rc C]` (the ISS ratings page: Elo taking the opposition into
//! account, a recent snapshot), `--players FILE --min-games N --min-money M` (per-player
//! statistics of the corpus itself, `gmdata skat stats`: average zero-sum "money"
//! score per game, covers every year); a seat is kept when ANY given filter keeps it;
//! program instances `name:2` count as `name`. `--hq` = `--min-games 2000 --min-money
//! 5 --min-rating 1050 --max-rc 60` with the given `--players` / `--ratings` files.
//! `--split train|val` takes the games with ISS id % `--val-mod` (100) != 0 / == 0;
//! `--count-only` reports what a conversion would write (same filters) without writing.
//! Every `--input` may also be the archive's URL (streamed, nothing stored).
//!
//! ```sh
//! gmdata skat fetch   --out data/skat/raw                    # 545 MiB + iss-ratings.html
//! gmdata skat fetch   --out ~/gm-data/skat --sample 20000    # the first 20k games only
//! gmdata skat verify  --input data/skat/raw --dump-rejected data/skat/rejected.sgf
//! gmdata skat stats   --input data/skat/raw --out data/skat/players.json
//! gmdata skat convert --input data/skat/raw --out data/skat/train --hq --split train \
//!                     --players data/skat/players.json --ratings data/skat/raw/iss-ratings.html
//! gmdata skat render  --input trajectories.jsonl --out games.sgf   # gmrules → ISS records
//! ```
//!
//! Full scale (the cluster; Mac or Linux with cargo): `scripts/fetch_skat.sh data/skat`
//! — download (545 MiB + the ratings page), verify (~3 min, the parity report
//! `verify.json` + `rejected.sgf`), corpus player stats (~3 min), `convert --hq` into
//! `data/skat/train` (~20 min at ~8k games/s) and `data/skat/val` (~3 min). Measured
//! with `convert --count-only` on the full archive (ratings page of 2026-09-20; ~418
//! bytes per frame on disk, bit-packed; 35.0 frames per game, 11.7 per seat stream):
//!
//! | corpus                                 | games     | seat streams | frames      | disk     |
//! |----------------------------------------|-----------|--------------|-------------|----------|
//! | every seat (`HQ=0`, train + val)       | 8,987,195 | 26,961,585   | 314,830,960 | ~132 GB  |
//! | `--hq --split train` (the recipe)      | 6,757,574 | 11,168,014   | 137,182,532 | ~57 GB   |
//! | `--hq --split val` (id % 100 == 0)     | 68,225    | 113,023      | 1,388,058   | ~0.6 GB  |
//!
//! The recipe keeps 69 players by corpus strength (10.4M seat-games; the programs
//! theCount, kermit and zoot hold 7.6M of them) plus the humans rated ≥ 1050 on the
//! ratings page; the most frequent players, xskat (7.1M seat-games, −9.8 money per
//! game) and bernie (2.9M, −8.2), are left out.

use crate::common;
use crate::traj::{self, Concession, ConvertOptions, Record};
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmrules::games::skat::{
    bid_index, card_str, next_bid_index, parse_card, Contract, IssEnding, IssResult, Phase, Skat, SkatConfig, BIDS,
    BID0, BID_WINDOW, DECL_NULL_OUVERT, DECLARE0, DISCARD0, HAND0, MOD0, MOD_OUVERT, MOD_PLAIN, MONEY_SCALE, MORE,
    NULL, PASS, PICKUP, PLAY0, YES,
};
use gmrules::{Event, Game, Seat, Turn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

pub const ISS_URL: &str = "https://skatgame.net/iss/iss-games-07-2024.sgf.bz2";
pub const RATINGS_URL: &str = "https://skatgame.net/mburo/iss-ratings.html";

// ---------------------------------------------------------------------------
// records
// ---------------------------------------------------------------------------

/// One ISS game record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IssRecord {
    pub id: String,
    pub date: String,
    pub players: [String; 3],
    /// `(who, move)`, who −1 = the server
    pub moves: Vec<(i8, String)>,
    pub result: String,
}

/// Parse one archive line (None: blank / comment).
pub fn parse_line(line: &str) -> Result<Option<IssRecord>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let body = line.strip_prefix("(;").context("an ISS record starts with \"(;\"")?;
    let mut rec = IssRecord::default();
    let (mut gm, mut mv, mut res) = (false, false, false);
    let mut rest = body;
    loop {
        let rest_t = rest.trim_start();
        if rest_t.is_empty() || rest_t.starts_with(";)") || rest_t.starts_with(')') {
            break;
        }
        let open = rest_t.find('[').with_context(|| format!("tag without value near {:?}", &rest_t[..rest_t.len().min(20)]))?;
        let close = rest_t[open..].find(']').context("unterminated tag value")? + open;
        let (tag, val) = (&rest_t[..open], &rest_t[open + 1..close]);
        match tag {
            "GM" => {
                if val != "Skat" {
                    bail!("GM[{val}] is not Skat");
                }
                gm = true;
            }
            "ID" => rec.id = val.to_string(),
            "DT" => rec.date = val.to_string(),
            "P0" | "P1" | "P2" => rec.players[(tag.as_bytes()[1] - b'0') as usize] = val.to_string(),
            "MV" => {
                let toks: Vec<&str> = val.split_whitespace().collect();
                if toks.len() % 2 != 0 {
                    bail!("MV has an odd number of tokens");
                }
                for p in toks.chunks(2) {
                    let who = match p[0] {
                        "w" => -1,
                        "0" => 0,
                        "1" => 1,
                        "2" => 2,
                        w => bail!("MV: mover {w:?}"),
                    };
                    rec.moves.push((who, p[1].to_string()));
                }
                mv = true;
            }
            "R" => {
                rec.result = val.to_string();
                res = true;
            }
            _ => {}
        }
        rest = &rest_t[close + 1..];
    }
    if !(gm && mv && res) {
        bail!("missing GM / MV / R tags");
    }
    Ok(Some(rec))
}

/// How a replayed record ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndKind {
    /// played to the end (10 tricks or an early end of the rules)
    Finished,
    /// everybody passed
    Passed,
    /// the declarer / both defenders resigned
    Resigned,
    /// a timeout or a player leaving (`TI` / `LE`)
    Aborted,
    /// the record stops before the end without an ending event
    Truncated,
}

/// A record replayed by the rules.
pub struct Replayed {
    pub events: Vec<Event>,
    pub state: Skat,
    pub ending: IssEnding,
    pub kind: EndKind,
    /// ISS result computed by the rules (None for truncated records)
    pub result: Option<IssResult>,
    /// jump bids (more than one bid value up)
    pub jump_bids: usize,
    /// `SC` moves / non-final resignations dropped
    pub dropped: usize,
    /// pickup declarations sent as half-moves (declaration, then the discards)
    pub half_moves: usize,
    /// declarations whose shown cards are not the declarer's hand (unchecked by ISS)
    pub shown_mismatch: usize,
}

fn seat_of(s: &str) -> std::result::Result<Seat, String> {
    match s {
        "0" => Ok(0),
        "1" => Ok(1),
        "2" => Ok(2),
        _ => Err(format!("seat {s:?}")),
    }
}

fn cards_mask(parts: &[&str]) -> std::result::Result<u32, String> {
    parts.iter().try_fold(0u32, |m, p| parse_card(p).map(|c| m | (1 << c)).ok_or(format!("card {p:?}")))
}

/// An ISS move as gmrules actions (+ its oddities, counted by `verify`).
struct MoveActions {
    actions: Vec<u32>,
    /// a declaration after the pickup without its discards (they follow as a move)
    half: bool,
    /// cards shown with a declaration that are not the declarer's hand: the ISS server
    /// does not check them (it shows the defenders the declarer's actual hand, as the
    /// frames do)
    shown_mismatch: bool,
}

fn acts(actions: Vec<u32>) -> MoveActions {
    MoveActions { actions, half: false, shown_mismatch: false }
}

/// The two discards `parts[0..2]` of a declaration.
fn discards(parts: &[&str]) -> std::result::Result<(u32, u32), String> {
    let d1 = parse_card(parts[0]).ok_or(format!("discard {:?}", parts[0]))?;
    let d2 = parse_card(parts[1]).ok_or(format!("discard {:?}", parts[1]))?;
    Ok((d1, d2))
}

/// Do the cards shown with a declaration (`shown`, maybe none) differ from `hand`?
fn shown_mismatch(shown: &[&str], ouvert: bool, hand: u32) -> bool {
    !shown.is_empty() && (!ouvert || shown.len() != 10 || cards_mask(shown).ok() != Some(hand))
}

/// The gmrules actions of ISS move `mv` of `seat` in state `g` — the ISS server's
/// acceptance rules (`SimpleState.makeMove`): bid-list values only (real Skat; the ISS
/// code takes any number, 75 archive records hold other values), declarations with
/// their bid limits, the pickup declaration with both discards or as a half-move
/// (declaration, then `<card>.<card>`), shown cards unchecked.
fn move_actions(g: &Skat, seat: Seat, mv: &str) -> std::result::Result<MoveActions, String> {
    match g.phase() {
        Phase::Bid => {
            if mv == "p" {
                return Ok(acts(vec![PASS]));
            }
            let v: u32 = mv.parse().map_err(|_| format!("bid expected, got {mv:?}"))?;
            let i = bid_index(v).ok_or(format!("bid {v} is not in the bid list"))?;
            let n = next_bid_index(g.max_bid());
            if i < n {
                return Err(format!("bid {v} does not exceed {}", g.max_bid()));
            }
            let mut out = vec![MORE; (i - n) / BID_WINDOW];
            out.push(BID0 + i as u32);
            Ok(acts(out))
        }
        Phase::Answer => match mv {
            "y" => Ok(acts(vec![YES])),
            "p" => Ok(acts(vec![PASS])),
            _ => Err(format!("answer y / p expected, got {mv:?}")),
        },
        Phase::SkatOrHand => {
            if mv == "s" {
                return Ok(acts(vec![PICKUP]));
            }
            let parts: Vec<&str> = mv.split('.').collect();
            let c = Contract::parse_iss(parts[0]).ok_or(format!("game declaration {mv:?}"))?;
            if !c.hand {
                return Err(format!("{mv:?} before the pickup must be a hand game"));
            }
            let m = if c.ty == NULL {
                if c.ouvert { MOD_OUVERT } else { MOD_PLAIN }
            } else {
                c.hand_modifier()
            };
            Ok(MoveActions {
                actions: vec![HAND0 + c.ty as u32, MOD0 + m],
                half: false,
                shown_mismatch: shown_mismatch(&parts[1..], c.ouvert, g.hand(seat)),
            })
        }
        Phase::Declare => {
            let parts: Vec<&str> = mv.split('.').collect();
            let c = Contract::parse_iss(parts[0]).ok_or(format!("game declaration {mv:?}"))?;
            if c.hand {
                return Err(format!("{mv:?}: hand game after the pickup"));
            }
            let d = DECLARE0 + if c.ty == NULL && c.ouvert { DECL_NULL_OUVERT } else { c.ty as u32 };
            if parts.len() == 1 {
                return Ok(MoveActions { actions: vec![d], half: true, shown_mismatch: false });
            }
            if parts.len() < 3 {
                return Err(format!("<game>.<card>.<card> expected after the pickup, got {mv:?}"));
            }
            let (d1, d2) = discards(&parts[1..3])?;
            let rest = g.hand(seat) & !(1 << d1) & !(1 << d2);
            Ok(MoveActions {
                actions: vec![d, DISCARD0 + d1, DISCARD0 + d2],
                half: false,
                shown_mismatch: shown_mismatch(&parts[3..], c.ouvert, rest),
            })
        }
        Phase::Discard1 => {
            // the discards after a half-move declaration
            let parts: Vec<&str> = mv.split('.').collect();
            if parts.len() < 2 {
                return Err(format!("<card>.<card> expected after the declaration, got {mv:?}"));
            }
            let (d1, d2) = discards(&parts[..2])?;
            let rest = g.hand(seat) & !(1 << d1) & !(1 << d2);
            let ouvert = g.contract().is_some_and(|c| c.ouvert);
            Ok(MoveActions {
                actions: vec![DISCARD0 + d1, DISCARD0 + d2],
                half: false,
                shown_mismatch: shown_mismatch(&parts[2..], ouvert, rest),
            })
        }
        Phase::Play => Ok(acts(vec![PLAY0 + parse_card(mv).ok_or(format!("card expected, got {mv:?}"))?])),
        p => Err(format!("move {mv:?} in phase {p:?}")),
    }
}

/// Replay a record with the rules: the trajectory, how it ended and the ISS result
/// the rules compute (not yet compared with the record's). Err = a move the rules
/// reject (or a malformed record).
pub fn replay_record(rec: &IssRecord) -> std::result::Result<Replayed, String> {
    let cfg = SkatConfig {};
    let mut g = Skat::new_game(&cfg);
    let mut events = Vec::with_capacity(80);
    let mut it = rec.moves.iter();
    let deal = match it.next() {
        Some((-1, d)) => d,
        _ => return Err("the first move must be the deal (w ...)".into()),
    };
    let cards: Vec<&str> = deal.split('.').collect();
    if cards.len() != 32 {
        return Err(format!("the deal has {} cards", cards.len()));
    }
    for p in cards {
        let c = parse_card(p).ok_or(format!("deal card {p:?}"))?;
        g.apply_chance(c).map_err(|e| e.0)?;
        events.push(Event::Chance(c));
    }
    let mut ending = IssEnding::default();
    let (mut over, mut resigned_end, mut awaiting_skat) = (false, false, false);
    let (mut jump_bids, mut dropped, mut half_moves, mut shown_bad) = (0, 0, 0, 0);
    for (k, (who, mv)) in it.enumerate() {
        let at = |m: String| format!("move {} ({who} {mv}): {m}", k + 1);
        if over {
            return Err(at("after the end of the game".into()));
        }
        if *who < 0 {
            if let Some(s) = mv.strip_prefix("TI.") {
                ending.timeout = Some(seat_of(s).map_err(at)?);
                over = true;
            } else if let Some(s) = mv.strip_prefix("LE.") {
                ending.left = Some(seat_of(s).map_err(at)?);
                over = true;
            } else if awaiting_skat {
                let parts: Vec<&str> = mv.split('.').collect();
                if parts.len() != 2 || cards_mask(&parts).map_err(at)? != g.dealt_skat() {
                    return Err(at(format!("the revealed skat is not the dealt one ({})", card_str_mask(g.dealt_skat()))));
                }
                awaiting_skat = false;
            } else {
                return Err(at("unexpected server move".into()));
            }
            continue;
        }
        if awaiting_skat {
            return Err(at("the server did not reveal the skat after the pickup".into()));
        }
        let s = *who as Seat;
        if mv == "RE" {
            if g.phase() != Phase::Play || ending.resigned[s] {
                return Err(at("resignation outside card play / twice".into()));
            }
            ending.resigned[s] = true;
            if ending.resignation_ends(g.declarer().expect("declarer in card play")) {
                over = true;
                resigned_end = true;
            } else {
                dropped += 1;
            }
            continue;
        }
        if mv == "SC" || mv.starts_with("SC.") {
            if g.phase() != Phase::Play || g.declarer() != Some(s) {
                return Err(at("only the declarer shows his cards, in card play".into()));
            }
            dropped += 1;
            continue;
        }
        if g.turn() != Turn::Act(gmrules::SeatSet::single(s)) {
            return Err(at(format!("seat {s} is not to move ({:?}, to move {})", g.phase(), g.to_move())));
        }
        let m = move_actions(&g, s, mv).map_err(at)?;
        half_moves += m.half as usize;
        shown_bad += m.shown_mismatch as usize;
        // a bid above the next bid value
        let bid = m.actions.last().copied().filter(|a| (BID0..MORE).contains(a));
        jump_bids += bid.is_some_and(|a| (a - BID0) as usize > next_bid_index(g.max_bid())) as usize;
        for a in m.actions {
            g.apply_action(s, a).map_err(|e| at(e.0))?;
            events.push(Event::Act(s, a));
        }
        if mv == "s" {
            awaiting_skat = true;
        }
    }
    let kind = if over {
        if resigned_end { EndKind::Resigned } else { EndKind::Aborted }
    } else if g.turn() == Turn::Terminal {
        if g.declarer().is_none() { EndKind::Passed } else { EndKind::Finished }
    } else {
        EndKind::Truncated
    };
    let result = (kind != EndKind::Truncated).then(|| g.iss_result(&ending));
    Ok(Replayed { events, state: g, ending, kind, result, jump_bids, dropped, half_moves, shown_mismatch: shown_bad })
}

fn card_str_mask(m: u32) -> String {
    (0..32).filter(|c| m & (1 << c) != 0).map(card_str).collect::<Vec<_>>().join(".")
}

// ---------------------------------------------------------------------------
// player filters
// ---------------------------------------------------------------------------

/// Player name without a program-instance suffix (`xskat:2` → `xskat`).
pub fn base_name(name: &str) -> &str {
    match name.rsplit_once(':') {
        Some((b, n)) if !b.is_empty() && !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) => b,
        _ => name,
    }
}

/// One player's corpus statistics (`gmdata skat stats`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlayerStats {
    pub games: u64,
    pub declarer: u64,
    pub declarer_won: u64,
    /// mean zero-sum ISS money score per game
    pub money: f64,
    /// mean Seeger-Fabian (tournament) points per game
    pub tpg: f64,
    pub first: String,
    pub last: String,
}

/// ISS ratings page rows: name → (Elo R, confidence RC, games).
pub fn parse_ratings(html: &str) -> HashMap<String, (f64, f64, u64)> {
    let mut text = String::with_capacity(html.len());
    let mut tag = false;
    for ch in html.chars() {
        match ch {
            '<' => tag = true,
            '>' => {
                tag = false;
                text.push(' ');
            }
            c if !tag => text.push(c),
            _ => {}
        }
    }
    let mut out = HashMap::new();
    for line in text.lines() {
        // " 3 theCount 40733 Decl[ 34 83 17 ] Def[ 66 21 79 ] 27.66 10.85 1118.1 23.6"
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 17 || toks[0].parse::<u64>().is_err() || toks[3] != "Decl[" {
            continue;
        }
        let Ok(games) = toks[2].parse::<u64>() else { continue };
        let Some(close) = toks.iter().rposition(|t| *t == "]") else { continue };
        let nums: Vec<f64> = toks[close + 1..].iter().filter_map(|t| t.parse().ok()).collect();
        if nums.len() >= 4 {
            out.entry(toks[1].to_string()).or_insert((nums[2], nums[3], games));
        }
    }
    out
}

/// Which seats to keep (see the module docs).
#[derive(Default)]
pub struct Filter {
    pub names: Vec<String>,
    pub ratings: Option<HashMap<String, (f64, f64, u64)>>,
    pub min_rating: f64,
    pub max_rc: f64,
    pub stats: Option<HashMap<String, PlayerStats>>,
    pub min_games: u64,
    pub min_money: f64,
}

impl Filter {
    pub fn active(&self) -> bool {
        !self.names.is_empty() || self.ratings.is_some() || self.stats.is_some()
    }

    pub fn keep(&self, name: &str) -> bool {
        if !self.active() {
            return true;
        }
        let b = base_name(name);
        self.names.iter().any(|n| n == name || n == b)
            || self.ratings.as_ref().is_some_and(|r| r.get(b).is_some_and(|&(elo, rc, _)| elo >= self.min_rating && rc <= self.max_rc))
            || self.stats.as_ref().is_some_and(|s| s.get(b).is_some_and(|p| p.games >= self.min_games && p.money >= self.min_money))
    }
}

// ---------------------------------------------------------------------------
// streaming
// ---------------------------------------------------------------------------

/// The lines of an archive file (`.bz2` / `.gz` decompressed on the fly; invalid UTF-8
/// replaced).
fn open_lines(path: &Path) -> Result<Box<dyn Iterator<Item = std::io::Result<String>>>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let name = path.to_string_lossy().to_lowercase();
    let r: Box<dyn Read> = if name.ends_with(".bz2") {
        Box::new(bzip2::read::MultiBzDecoder::new(f))
    } else if name.ends_with(".gz") {
        Box::new(flate2::read::MultiGzDecoder::new(f))
    } else {
        Box::new(f)
    };
    let mut br = BufReader::with_capacity(1 << 20, r);
    Ok(Box::new(std::iter::from_fn(move || {
        let mut buf = Vec::new();
        match br.read_until(b'\n', &mut buf) {
            Ok(0) => None,
            Ok(_) => Some(Ok(String::from_utf8_lossy(&buf).into_owned())),
            Err(e) => Some(Err(e)),
        }
    })))
}

/// An HTTP client for long downloads / streams (no total timeout; the archive is
/// 545 MiB).
fn stream_client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("GamingMaster-research/0.1 (imitation-learning data pipeline)")
        .connect_timeout(std::time::Duration::from_secs(60))
        .timeout(None::<std::time::Duration>)
        .build()?)
}

/// The lines of a remote archive, streamed (`.bz2` decompressed): nothing is stored.
fn url_lines(url: &str) -> Result<Box<dyn Iterator<Item = std::io::Result<String>>>> {
    let resp = stream_client()?.get(url).send()?.error_for_status()?;
    let r: Box<dyn Read> = if url.to_lowercase().ends_with(".bz2") {
        Box::new(bzip2::read::MultiBzDecoder::new(resp))
    } else {
        Box::new(resp)
    };
    let mut br = BufReader::with_capacity(1 << 20, r);
    Ok(Box::new(std::iter::from_fn(move || {
        let mut buf = Vec::new();
        match br.read_until(b'\n', &mut buf) {
            Ok(0) => None,
            Ok(_) => Some(Ok(String::from_utf8_lossy(&buf).into_owned())),
            Err(e) => Some(Err(e)),
        }
    })))
}

/// The ISS archive files under `input` (sorted).
pub fn archive_files(input: &Path) -> Result<Vec<PathBuf>> {
    let files = common::find_files(input, &[".sgf", ".sgf.bz2", ".sgf.gz"])?;
    anyhow::ensure!(!files.is_empty(), "no .sgf / .sgf.bz2 / .sgf.gz files under {}", input.display());
    Ok(files)
}

/// Every record line of the archive files, `(provenance id, line)`; `input` may also
/// be an http(s) URL of an archive, streamed without storing it.
pub fn record_lines(input: &Path) -> Result<Box<dyn Iterator<Item = Result<(String, String)>>>> {
    let s = input.to_string_lossy();
    if s.starts_with("http://") || s.starts_with("https://") {
        let name = s.rsplit('/').next().unwrap_or("url").to_string();
        let lines = url_lines(&s)?;
        return Ok(Box::new(lines.enumerate().filter_map(move |(ln, l)| match l {
            Ok(l) if l.trim().is_empty() => None,
            Ok(l) => Some(Ok((format!("{name}:{}", ln + 1), l))),
            Err(e) => Some(Err(anyhow::Error::from(e).context(format!("{name}:{}", ln + 1)))),
        })));
    }
    let files = archive_files(input)?;
    Ok(Box::new(files.into_iter().flat_map(|path| -> Box<dyn Iterator<Item = Result<(String, String)>>> {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        match open_lines(&path) {
            Ok(lines) => Box::new(lines.enumerate().filter_map(move |(ln, l)| match l {
                Ok(l) if l.trim().is_empty() => None,
                Ok(l) => Some(Ok((format!("{name}:{}", ln + 1), l))),
                Err(e) => Some(Err(anyhow::Error::from(e).context(format!("{name}:{}", ln + 1)))),
            })),
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    })))
}

/// Per-record verification counters.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VerifyStats {
    pub records: usize,
    pub unparseable: usize,
    /// a move the rules reject
    pub illegal: usize,
    /// the rules' result differs from the recorded one
    pub result_mismatch: usize,
    pub finished: usize,
    pub passed: usize,
    pub resigned: usize,
    pub aborted: usize,
    pub truncated: usize,
    /// verified = replayed legally AND the recorded result reproduced (truncated excluded)
    pub verified: usize,
    pub jump_bids: usize,
    pub dropped_moves: usize,
    pub half_moves: usize,
    pub shown_mismatch: usize,
    pub games: BTreeMap<String, usize>,
    pub errors: Vec<String>,
}

impl VerifyStats {
    fn note(&mut self, msg: String, show: usize) {
        if self.errors.len() < show {
            self.errors.push(msg);
        }
    }
}

/// A verified record: its replay, or why it was rejected.
pub enum Checked {
    Ok(IssRecord, Replayed),
    Unparseable(String),
    Illegal(String),
    Mismatch(String),
}

pub fn check_line(id: &str, line: &str) -> Option<Checked> {
    let rec = match parse_line(line) {
        Ok(Some(r)) => r,
        Ok(None) => return None,
        Err(e) => return Some(Checked::Unparseable(format!("{id}: {e:#}"))),
    };
    let r = match replay_record(&rec) {
        Ok(r) => r,
        Err(e) => return Some(Checked::Illegal(format!("{id} (ISS {}): {e}", rec.id))),
    };
    if let Some(res) = &r.result {
        let got = res.iss_string();
        if got != rec.result {
            return Some(Checked::Mismatch(format!("{id} (ISS {}): recorded R[{}], the rules give R[{got}]", rec.id, rec.result)));
        }
    }
    Some(Checked::Ok(rec, r))
}

fn tally(st: &mut VerifyStats, c: &Checked, show: usize) {
    st.records += 1;
    match c {
        Checked::Ok(_, r) => {
            match r.kind {
                EndKind::Finished => st.finished += 1,
                EndKind::Passed => st.passed += 1,
                EndKind::Resigned => st.resigned += 1,
                EndKind::Aborted => st.aborted += 1,
                EndKind::Truncated => st.truncated += 1,
            }
            if r.kind != EndKind::Truncated {
                st.verified += 1;
            }
            st.jump_bids += r.jump_bids;
            st.dropped_moves += r.dropped;
            st.half_moves += r.half_moves;
            st.shown_mismatch += r.shown_mismatch;
            let g = r.state.contract().map_or("-".to_string(), |c| c.iss_string());
            *st.games.entry(g).or_insert(0) += 1;
        }
        Checked::Unparseable(e) => {
            st.unparseable += 1;
            st.note(format!("unparseable {e}"), show);
        }
        Checked::Illegal(e) => {
            st.illegal += 1;
            st.note(format!("illegal {e}"), show);
        }
        Checked::Mismatch(e) => {
            st.result_mismatch += 1;
            st.note(format!("result {e}"), show);
        }
    }
}

/// Verify every record (parallel chunks); `rejected`: write the rejected lines there.
pub fn verify(input: &Path, max_records: Option<usize>, show: usize, rejected: Option<&Path>) -> Result<VerifyStats> {
    let mut st = VerifyStats::default();
    let mut dump = match rejected {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p).with_context(|| format!("create {}", p.display()))?)),
        None => None,
    };
    let mut lines = record_lines(input)?.take(max_records.unwrap_or(usize::MAX)).peekable();
    while lines.peek().is_some() {
        let chunk: Vec<(String, String)> = lines.by_ref().take(1 << 14).collect::<Result<_>>()?;
        let checked: Vec<Option<Checked>> = chunk.par_iter().map(|(id, l)| check_line(id, l)).collect();
        for ((_, line), c) in chunk.iter().zip(&checked) {
            let Some(c) = c else { continue };
            tally(&mut st, c, show);
            if let (Some(w), false) = (dump.as_mut(), matches!(c, Checked::Ok(..))) {
                writeln!(w, "{}", line.trim_end())?;
            }
        }
    }
    if let Some(mut w) = dump {
        w.flush()?;
    }
    Ok(st)
}

/// Seeger-Fabian list points of the three seats (declarer won v + 50, lost v − 50 and
/// 40 per defender).
fn tournament_points(r: &IssResult) -> [f64; 3] {
    if r.passed || r.penalty || r.declarer < 0 {
        return [0.0; 3];
    }
    let d = r.declarer as usize;
    let v = r.value as f64;
    let mut s = if v > 0.0 { [0.0; 3] } else { [40.0; 3] };
    s[d] = if v > 0.0 { v + 50.0 } else { v - 50.0 };
    s
}

/// Per-player statistics of every verified finished / resigned / passed game.
pub fn player_stats(input: &Path, max_records: Option<usize>) -> Result<(BTreeMap<String, PlayerStats>, VerifyStats)> {
    let mut st = VerifyStats::default();
    let mut acc: BTreeMap<String, PlayerStats> = BTreeMap::new();
    let mut lines = record_lines(input)?.take(max_records.unwrap_or(usize::MAX)).peekable();
    while lines.peek().is_some() {
        let chunk: Vec<(String, String)> = lines.by_ref().take(1 << 14).collect::<Result<_>>()?;
        let checked: Vec<Option<Checked>> = chunk.par_iter().map(|(id, l)| check_line(id, l)).collect();
        for c in checked.iter().flatten() {
            tally(&mut st, c, 10);
            let Checked::Ok(rec, r) = c else { continue };
            if !matches!(r.kind, EndKind::Finished | EndKind::Resigned | EndKind::Passed) {
                continue;
            }
            let res = r.result.as_ref().expect("result of a finished game");
            let money = res.money();
            let tp = tournament_points(res);
            let date = rec.date.split('/').next().unwrap_or("").to_string();
            for s in 0..3 {
                let p = acc.entry(base_name(&rec.players[s]).to_string()).or_default();
                p.games += 1;
                p.money += money[s];
                p.tpg += tp[s];
                if res.declarer == s as i32 {
                    p.declarer += 1;
                    p.declarer_won += (res.value > 0) as u64;
                }
                if p.first.is_empty() || date < p.first {
                    p.first = date.clone();
                }
                if date > p.last {
                    p.last = date.clone();
                }
            }
        }
    }
    for p in acc.values_mut() {
        p.money /= p.games.max(1) as f64;
        p.tpg /= p.games.max(1) as f64;
    }
    Ok((acc, st))
}

// ---------------------------------------------------------------------------
// conversion
// ---------------------------------------------------------------------------

/// Which records a conversion takes (deterministic by ISS game id).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Split {
    #[default]
    All,
    /// ids not divisible by `val_mod`
    Train,
    /// ids divisible by `val_mod`
    Val,
}

/// The conversion's ISS-side counters (the traj driver counts the rest).
#[derive(Default)]
pub struct ConvertCtx {
    pub filter: Filter,
    pub split: Split,
    pub val_mod: u64,
    pub allow_truncated: bool,
    pub st: RefCell<VerifyStats>,
    /// finished / resigned games whose seats were all filtered out
    pub filtered: RefCell<usize>,
}

/// The traj record of a verified replay (None: not converted — aborted, truncated
/// without --allow-truncated, every seat filtered out).
pub fn to_record(ctx: &ConvertCtx, id: &str, rec: &IssRecord, r: Replayed) -> Option<Record> {
    if ctx.split != Split::All {
        // ids are integers on ISS; anything else is hashed
        let n = rec.id.parse::<u64>().unwrap_or_else(|_| rec.id.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)));
        if (n % ctx.val_mod.max(1) == 0) != (ctx.split == Split::Val) {
            return None;
        }
    }
    match r.kind {
        EndKind::Aborted => return None,
        EndKind::Truncated if !ctx.allow_truncated => return None,
        _ => {}
    }
    let seats: Vec<Seat> = (0..3).filter(|&s| ctx.filter.keep(&rec.players[s])).collect();
    if seats.is_empty() {
        *ctx.filtered.borrow_mut() += 1;
        return None;
    }
    let money = r.result.as_ref().map(|x| x.money().to_vec());
    let meta = json!({
        "iss_id": rec.id,
        "date": rec.date,
        "players": rec.players,
        "declarer": r.state.declarer().map_or(-1, |d| d as i64),
        "game": r.state.contract().map(|c| c.iss_string()),
        "result": rec.result,
        "ending": r.kind,
        "resigned": (0..3).filter(|&s| r.ending.resigned[s]).collect::<Vec<_>>(),
    });
    Some(Record {
        id: format!("{id}#{}", rec.id),
        events: r.events,
        meta,
        seats: ctx.filter.active().then_some(seats),
        expected_returns: money.clone().filter(|_| r.kind != EndKind::Truncated),
        conceded: (r.kind == EndKind::Resigned).then(|| {
            let m = money.expect("result of a resigned game");
            Concession { outcome: m.iter().map(|&x| (x / MONEY_SCALE) as f32).collect(), returns: m }
        }),
        ..Default::default()
    })
}

/// The converter's records, streamed in input order (lines verified in parallel
/// chunks; a record the rules reject comes out as `Err`, counted by the traj driver
/// as unparseable and in `ctx.st` by reason).
pub fn records<'a>(input: &Path, ctx: &'a ConvertCtx, max_records: Option<usize>)
    -> Result<impl Iterator<Item = Result<Record>> + 'a> {
    let mut lines = record_lines(input)?.take(max_records.unwrap_or(usize::MAX)).peekable();
    let mut pending: std::collections::VecDeque<Result<Record>> = Default::default();
    Ok(std::iter::from_fn(move || loop {
        if let Some(r) = pending.pop_front() {
            return Some(r);
        }
        lines.peek()?;
        let chunk: Vec<Result<(String, String)>> = lines.by_ref().take(1 << 14).collect();
        let checked: Vec<Result<Option<(String, Checked)>>> = chunk
            .into_par_iter()
            .map(|l| l.map(|(id, line)| check_line(&id, &line).map(|c| (id, c))))
            .collect();
        for c in checked {
            let (id, c) = match c {
                Ok(Some(x)) => x,
                Ok(None) => continue,
                Err(e) => {
                    pending.push_back(Err(e));
                    continue;
                }
            };
            tally(&mut ctx.st.borrow_mut(), &c, 10);
            match c {
                Checked::Ok(rec, r) => pending.extend(to_record(ctx, &id, &rec, r).map(Ok)),
                Checked::Unparseable(e) | Checked::Illegal(e) | Checked::Mismatch(e) => {
                    pending.push_back(Err(anyhow::anyhow!(e)))
                }
            }
        }
    }))
}

/// What a conversion of the records would write (`convert --count-only`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Counts {
    pub converted: usize,
    pub conceded: usize,
    pub samples: usize,
    pub frames: usize,
    /// records the rules reject (by reason in the ISS counters)
    pub rejected: usize,
}

pub fn count(records: impl Iterator<Item = Result<Record>>) -> Counts {
    let mut c = Counts::default();
    for r in records {
        let Ok(rec) = r else {
            c.rejected += 1;
            continue;
        };
        let keep = |s: Seat| rec.seats.as_ref().is_none_or(|v| v.contains(&s));
        c.converted += 1;
        c.conceded += rec.conceded.is_some() as usize;
        c.samples += (0..3).filter(|&s| keep(s)).count();
        c.frames += rec.events.iter().filter(|e| matches!(e, Event::Act(s, _) if keep(*s))).count();
    }
    c
}

// ---------------------------------------------------------------------------
// rendering (gmrules trajectory → ISS record)
// ---------------------------------------------------------------------------

/// An ISS record line of a gmrules skat trajectory (the inverse of the parser: the
/// round trip is exact). Unfinished trajectories get `R[unknown]`.
pub fn render(events: &[Event], id: &str, players: &[String; 3]) -> std::result::Result<String, String> {
    let mut g = Skat::new_game(&SkatConfig {});
    if events.len() < 32 {
        return Err("a trajectory starts with the 32 deal events".into());
    }
    let mut deal = Vec::with_capacity(32);
    for e in &events[..32] {
        let Event::Chance(c) = *e else { return Err("the first 32 events must be the deal".into()) };
        g.apply_chance(c).map_err(|e| e.0)?;
        deal.push(card_str(c));
    }
    let mut mv = vec![format!("w {}", deal.join("."))];
    let (mut hand_ty, mut decl, mut discards) = (None, None, Vec::new());
    for e in &events[32..] {
        let Event::Act(s, a) = *e else { return Err("chance event after the deal".into()) };
        match a {
            PASS => mv.push(format!("{s} p")),
            YES => mv.push(format!("{s} y")),
            MORE => {}
            PICKUP => {
                mv.push(format!("{s} s"));
                mv.push(format!("w {}.{}", deal[30], deal[31]));
            }
            _ if (BID0..MORE).contains(&a) => mv.push(format!("{s} {}", BIDS[(a - BID0) as usize])),
            _ if (HAND0..MOD0).contains(&a) => hand_ty = Some((a - HAND0) as u8),
            _ if (MOD0..DECLARE0).contains(&a) => {
                let ty = hand_ty.take().ok_or("modifier without a hand game")?;
                let m = a - MOD0;
                let c = Contract {
                    ty,
                    hand: true,
                    ouvert: m == MOD_OUVERT,
                    schneider: ty != NULL && m >= 1,
                    schwarz: ty != NULL && m >= 2,
                };
                mv.push(format!("{s} {}", c.iss_string()));
            }
            _ if a >= DECLARE0 => {
                let d = a - DECLARE0;
                decl = Some(if d == DECL_NULL_OUVERT { "NO".to_string() } else { gmrules::games::skat::GAME_CHARS[d as usize].to_string() });
            }
            _ if a >= DISCARD0 => {
                discards.push(card_str(a - DISCARD0));
                if discards.len() == 2 {
                    let d = decl.take().ok_or("discards without a game")?;
                    mv.push(format!("{s} {d}.{}.{}", discards[0], discards[1]));
                    discards.clear();
                }
            }
            _ => mv.push(format!("{s} {}", card_str(a - PLAY0))),
        }
        g.apply_action(s, a).map_err(|e| e.0)?;
    }
    let result = if g.turn() == Turn::Terminal { g.final_result().iss_string() } else { "unknown".into() };
    Ok(format!(
        "(;GM[Skat]PC[gmsim]CO[]SE[0]ID[{id}]DT[2000-01-01/00:00:00/UTC]P0[{}]P1[{}]P2[{}]R0[]R1[]R2[]MV[{} ]R[{result}] ;)",
        players[0],
        players[1],
        players[2],
        mv.join(" ")
    ))
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Subcommand)]
pub enum Cmd {
    /// Download the ISS game archive (and the ISS ratings page)
    Fetch {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = ISS_URL)]
        url: String,
        /// stream only the first N games into <out>/iss-sample.sgf (no archive kept)
        #[arg(long)]
        sample: Option<usize>,
        /// skip the ratings page
        #[arg(long)]
        no_ratings: bool,
    },
    /// Replay and verify every record (moves + recorded results); no shards
    Verify {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        max_records: Option<usize>,
        /// rejection reasons to print
        #[arg(long, default_value_t = 20)]
        show: usize,
        /// write the rejected record lines to this file
        #[arg(long)]
        dump_rejected: Option<PathBuf>,
    },
    /// Per-player statistics of the corpus (the --players file of convert)
    Stats {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        max_records: Option<usize>,
    },
    /// Convert ISS records to candidate-action gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// keep only the seats of these players (repeatable)
        #[arg(long)]
        player: Vec<String>,
        /// ISS ratings page (iss-ratings.html) for --min-rating
        #[arg(long)]
        ratings: Option<PathBuf>,
        #[arg(long)]
        min_rating: Option<f64>,
        /// largest rating confidence interval accepted with --min-rating
        #[arg(long)]
        max_rc: Option<f64>,
        /// player statistics (gmdata skat stats) for --min-games / --min-money
        #[arg(long)]
        players: Option<PathBuf>,
        #[arg(long)]
        min_games: Option<u64>,
        #[arg(long)]
        min_money: Option<f64>,
        /// high-quality preset (see the module docs); needs --players and/or --ratings
        #[arg(long)]
        hq: bool,
        /// all records, or the train / val split by ISS game id (id % --val-mod == 0 → val)
        #[arg(long, value_enum, default_value_t = Split::All)]
        split: Split,
        #[arg(long, default_value_t = 100)]
        val_mod: u64,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        #[arg(long)]
        max_records: Option<usize>,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// keep records that stop before the end (tgt.value_mask false on them)
        #[arg(long)]
        allow_truncated: bool,
        /// count the records / samples / frames the conversion would write (same
        /// filters and split), write nothing — sizes a full-scale run
        #[arg(long)]
        count_only: bool,
    },
    /// gmrules skat trajectories (JSONL, as `gmdata rules convert` reads) → ISS records
    Render {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

/// Stream the first `n` games of the remote archive into `dest`.
fn fetch_sample(url: &str, dest: &Path, n: usize) -> Result<usize> {
    let resp = stream_client()?.get(url).send()?.error_for_status()?;
    let mut lines = BufReader::new(bzip2::read::MultiBzDecoder::new(resp));
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = dest.with_extension("part");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    let mut k = 0;
    let mut buf = Vec::new();
    while k < n {
        buf.clear();
        if lines.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        f.write_all(&buf)?;
        k += 1;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, dest)?;
    Ok(k)
}

fn build_filter(player: Vec<String>, ratings: Option<PathBuf>, min_rating: Option<f64>, max_rc: Option<f64>,
                players: Option<PathBuf>, min_games: Option<u64>, min_money: Option<f64>, hq: bool) -> Result<Filter> {
    if hq && ratings.is_none() && players.is_none() {
        bail!("--hq needs --players (gmdata skat stats) and/or --ratings (iss-ratings.html)");
    }
    let mut f = Filter { names: player, ..Default::default() };
    if let Some(p) = ratings {
        let html = String::from_utf8_lossy(&std::fs::read(&p).with_context(|| format!("read {}", p.display()))?).into_owned();
        let r = parse_ratings(&html);
        anyhow::ensure!(!r.is_empty(), "no rating rows in {}", p.display());
        f.min_rating = min_rating.unwrap_or(if hq { 1050.0 } else { f64::NEG_INFINITY });
        f.max_rc = max_rc.unwrap_or(if hq { 60.0 } else { f64::INFINITY });
        f.ratings = Some(r);
    }
    if let Some(p) = players {
        let v: Value = serde_json::from_slice(&std::fs::read(&p).with_context(|| format!("read {}", p.display()))?)?;
        let map: HashMap<String, PlayerStats> = serde_json::from_value(v["players"].clone()).context("players.json: \"players\"")?;
        f.min_games = min_games.unwrap_or(if hq { 2000 } else { 0 });
        f.min_money = min_money.unwrap_or(if hq { 5.0 } else { f64::NEG_INFINITY });
        f.stats = Some(map);
    }
    Ok(f)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, url, sample, no_ratings } => {
            std::fs::create_dir_all(&out)?;
            let mut info = json!({});
            if let Some(n) = sample {
                let dest = out.join("iss-sample.sgf");
                info["sample_games"] = json!(fetch_sample(&url, &dest, n)?);
                info["sample"] = json!(dest);
            } else {
                let name = url.rsplit('/').next().unwrap_or("iss-games.sgf.bz2");
                let dest = out.join(name);
                anyhow::ensure!(common::download(&stream_client()?, &url, &dest)?, "404: {url}");
                info["archive"] = json!(dest);
            }
            if !no_ratings {
                let dest = out.join("iss-ratings.html");
                if common::download(&common::http()?, RATINGS_URL, &dest)? {
                    info["ratings"] = json!(dest);
                }
            }
            println!("{info}");
            Ok(())
        }
        Cmd::Verify { input, max_records, show, dump_rejected } => {
            let st = verify(&input, max_records, show, dump_rejected.as_deref())?;
            println!("{}", serde_json::to_string(&st)?);
            Ok(())
        }
        Cmd::Stats { input, out, max_records } => {
            let (players, st) = player_stats(&input, max_records)?;
            if let Some(p) = out.parent() {
                std::fs::create_dir_all(p)?;
            }
            let n = players.len();
            std::fs::write(&out, serde_json::to_vec_pretty(&json!({"verify": st, "players": players}))?)?;
            println!("{}", json!({"players": n, "records": st.records, "verified": st.verified, "out": out}));
            Ok(())
        }
        Cmd::Convert { input, out, player, ratings, min_rating, max_rc, players, min_games, min_money, hq, split,
                       val_mod, shard_size, max_records, store_events, allow_truncated, count_only } => {
            let filter = build_filter(player, ratings, min_rating, max_rc, players, min_games, min_money, hq)?;
            let ctx = ConvertCtx { filter, split, val_mod, allow_truncated, ..Default::default() };
            if count_only {
                let c = count(records(&input, &ctx, max_records)?);
                let mut v = serde_json::to_value(&c)?;
                v["iss"] = serde_json::to_value(&*ctx.st.borrow())?;
                v["filtered_out"] = json!(*ctx.filtered.borrow());
                println!("{v}");
                return Ok(());
            }
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, ..Default::default() };
            let st = traj::convert::<Skat>(&SkatConfig {}, records(&input, &ctx, max_records)?, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            v["iss"] = serde_json::to_value(&*ctx.st.borrow())?;
            v["filtered_out"] = json!(*ctx.filtered.borrow());
            println!("{v}");
            Ok(())
        }
        Cmd::Render { input, out } => {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&out)?);
            let mut n = 0;
            for path in common::find_files(&input, &[".jsonl", ".jsonl.gz"])? {
                for (ln, line) in open_lines(&path)?.enumerate() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let rec = crate::rules::parse_jsonl_record(&line, &format!("{}:{}", path.display(), ln + 1))?;
                    let names: Vec<String> = rec.meta.get("players").and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    let players: [String; 3] = std::array::from_fn(|i| names.get(i).cloned().unwrap_or_else(|| format!("p{i}")));
                    let l = render(&rec.events, &rec.id, &players).map_err(|e| anyhow::anyhow!("{}: {e}", rec.id))?;
                    writeln!(w, "{l}")?;
                    n += 1;
                }
            }
            w.flush()?;
            println!("{}", json!({"games": n, "out": out}));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrules::trajectory::replay;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/skat/iss_sample.sgf")
    }

    #[test]
    fn parses_iss_lines() {
        let l = "(;GM[Skat]PC[Internet Skat Server]CO[]SE[53]ID[9]DT[2007-10-29/04:58:00/UTC]P0[jeff]P1[Montana]P2[vaun]R0[0.0]R1[0.0]R2[0.0]MV[w S7.ST.CA.DT.CQ.S8.C8.D9.HT.SQ.HQ.DK.H9.DJ.HK.HJ.C9.DQ.HA.H8.SA.CJ.CK.DA.H7.C7.S9.CT.SK.D7.SJ.D8 1 18 0 p 2 20 1 p 2 CH 0 SQ 1 H9 2 SA 2 SK 0 ST 1 HK 0 D9 1 DQ 2 DA 2 D7 0 DT 1 DK 0 HT 1 H8 2 H7 0 S8 1 HQ 2 S9 2 C7 0 CA 1 DJ 1 C9 2 CT 0 C8 2 CK 0 CQ 1 HJ 1 HA 2 CJ 0 S7 ]R[d:2 loss v:-96 m:2 bidok p:56 t:5 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)";
        let rec = parse_line(l).unwrap().unwrap();
        assert_eq!((rec.id.as_str(), rec.players[2].as_str(), rec.moves.len()), ("9", "vaun", 1 + 5 + 30));
        let r = replay_record(&rec).unwrap();
        assert_eq!(r.kind, EndKind::Finished);
        assert_eq!(r.result.unwrap().iss_string(), rec.result);
        // the round trip through render is exact
        let back = parse_line(&render(&r.events, "9", &rec.players).unwrap()).unwrap().unwrap();
        assert_eq!(back.moves, rec.moves);
        assert_eq!(back.result, rec.result);
        assert!(parse_line("(;GM[Chess]MV[]R[] ;)").is_err());
        assert!(parse_line("").unwrap().is_none());
        assert_eq!(base_name("xskat:2"), "xskat");
        assert_eq!(base_name("a:b"), "a:b");
    }

    #[test]
    fn rejects_broken_records() {
        let deal = "w S7.ST.CA.DT.CQ.S8.C8.D9.HT.SQ.HQ.DK.H9.DJ.HK.HJ.C9.DQ.HA.H8.SA.CJ.CK.DA.H7.C7.S9.CT.SK.D7.SJ.D8";
        let rec = |mv: &str| parse_line(&format!("(;GM[Skat]ID[1]P0[a]P1[b]P2[c]MV[{deal} {mv} ]R[x] ;)")).unwrap().unwrap();
        // out of turn / not a bid list value / hand after pickup / discard not held / wrong skat
        assert!(replay_record(&rec("0 18")).is_err());
        assert!(replay_record(&rec("1 19")).is_err());
        assert!(replay_record(&rec("1 18 0 p 2 p 1 s w SJ.D8 1 GH.CA.DT")).is_err());
        assert!(replay_record(&rec("1 18 0 p 2 p 1 s w SJ.D8 1 G.CA.SA")).is_err());
        assert!(replay_record(&rec("1 18 0 p 2 p 1 s w SJ.D7 1 G.HQ.DK")).is_err());
        // null after pickup above 23 is illegal; null ouvert up to 46 is fine
        assert!(replay_record(&rec("1 18 0 y 1 20 0 y 1 22 0 y 1 23 0 y 1 24 0 p 2 p 1 s w SJ.D8 1 N.HQ.DK")).is_err());
        let r = replay_record(&rec("1 18 0 y 1 20 0 y 1 22 0 y 1 23 0 y 1 24 0 p 2 p 1 s w SJ.D8 1 NO.HQ.DK")).unwrap();
        assert_eq!((r.kind, r.state.contract().unwrap().iss_string()), (EndKind::Truncated, "NO".to_string()));
        // a jump bid is MORE sub-decisions + the bid
        let r = replay_record(&rec("1 77 0 y 1 p 2 p")).unwrap();
        assert_eq!(r.jump_bids, 1);
        assert_eq!(r.events[32..35], [Event::Act(1, MORE), Event::Act(1, MORE), Event::Act(1, BID0 + 24)]);
        // moves after a game-ending resignation are rejected
        let base = "1 18 0 p 2 p 1 GH 0 CQ";
        let r = replay_record(&rec(&format!("{base} 1 RE"))).unwrap();
        assert_eq!(r.kind, EndKind::Resigned);
        assert!(replay_record(&rec(&format!("{base} 1 RE 1 HQ"))).is_err());
        // one defender's resignation does not end the game
        let r = replay_record(&rec(&format!("{base} 0 RE"))).unwrap();
        assert_eq!((r.kind, r.dropped), (EndKind::Truncated, 1));
        // a pickup declaration as a half-move (declaration, then the discards); the
        // discards alone need two cards
        let pick = "1 18 0 p 2 p 1 s w SJ.D8";
        let r = replay_record(&rec(&format!("{pick} 1 G 1 HQ.DK"))).unwrap();
        let full = replay_record(&rec(&format!("{pick} 1 G.HQ.DK"))).unwrap();
        assert_eq!((r.half_moves, full.half_moves, r.events.clone()), (1, 0, full.events));
        assert!(replay_record(&rec(&format!("{pick} 1 G 1 HQ"))).is_err());
        assert!(replay_record(&rec(&format!("{pick} 1 G 1 CA.DT"))).is_err());
        // shown cards are not checked (as on the ISS server), only counted
        let r = replay_record(&rec(&format!("{pick} 1 NO.HQ.DK.SJ.D8.H9.DJ.HK.HJ.C9.DQ.HA.H8"))).unwrap();
        assert_eq!((r.shown_mismatch, r.state.contract().unwrap().iss_string()), (0, "NO".to_string()));
        let r = replay_record(&rec(&format!("{pick} 1 NO.HQ.DK.HQ.DK.H9.DJ.HK.HJ.C9.DQ.HA.H8"))).unwrap();
        assert_eq!(r.shown_mismatch, 1);
        let r = replay_record(&rec("1 18 0 p 2 p 1 NOH.HQ.DK.H9.DJ.HK.HJ.C9.DQ.HA.H8")).unwrap();
        assert_eq!((r.shown_mismatch, r.state.contract().unwrap().iss_string()), (0, "NOH".to_string()));
    }

    /// Every record of the committed real-data fixture replays and reproduces its
    /// recorded result exactly.
    #[test]
    fn fixture_records_verify() {
        let st = verify(&fixture(), None, 5, None).unwrap();
        assert_eq!((st.unparseable, st.illegal, st.result_mismatch), (0, 0, 0), "{:?}", st.errors);
        assert_eq!(st.verified, st.records);
        assert!(st.finished > 0 && st.passed > 0 && st.resigned > 0 && st.aborted > 0, "{st:?}");
        // pinned oddities: half-move declarations, unchecked shown cards
        assert!(st.half_moves >= 5 && st.shown_mismatch >= 2, "{st:?}");
        // every rendered trajectory parses back to the same trajectory
        for (id, line) in record_lines(&fixture()).unwrap().map(Result::unwrap) {
            let rec = parse_line(&line).unwrap().unwrap();
            let r = replay_record(&rec).unwrap();
            if matches!(r.kind, EndKind::Finished | EndKind::Passed) {
                let back = replay_record(&parse_line(&render(&r.events, &rec.id, &rec.players).unwrap()).unwrap().unwrap()).unwrap();
                assert_eq!(back.events, r.events, "{id}");
            }
            // the gmrules replay (the converter's) agrees on terminal state and returns
            let rr = replay::<Skat>(&SkatConfig {}, &r.events).unwrap();
            assert_eq!(rr.terminal(), matches!(r.kind, EndKind::Finished | EndKind::Passed), "{id}");
        }
    }

    #[test]
    fn converts_the_fixture() {
        let dir = std::env::temp_dir().join(format!("gmdata-skat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ctx = ConvertCtx::default();
        let opts = ConvertOptions { store_events: true, ..Default::default() };
        let st = traj::convert::<Skat>(&SkatConfig {}, records(&fixture(), &ctx, None).unwrap(), &dir, &opts).unwrap();
        let iss = ctx.st.borrow().clone();
        assert_eq!((st.illegal, st.returns_mismatch, st.unparseable), (0, 0, 0), "{st:?}");
        assert_eq!(st.converted, iss.finished + iss.passed + iss.resigned);
        assert_eq!(st.conceded, iss.resigned);
        // every seat decides at least once (bids or passes)
        assert_eq!(st.samples, 3 * st.converted);
        let sh = gmshard::ShardReader::open(&dir.join("skat_00000")).unwrap();
        assert_eq!(sh.meta()["schema"], "candidate.v1");
        assert_eq!(sh.meta()["max_actions"], 12);
        std::fs::remove_dir_all(&dir).unwrap();
        // --count-only predicts what the conversion writes, also with a seat filter
        let c = count(records(&fixture(), &ConvertCtx::default(), None).unwrap());
        assert_eq!((c.converted, c.conceded, c.samples, c.frames), (st.converted, st.conceded, st.samples, st.frames));
        let ctx = ConvertCtx { filter: Filter { names: vec!["kermit".into()], ..Default::default() }, ..Default::default() };
        let c = count(records(&fixture(), &ctx, None).unwrap());
        let ctx = ConvertCtx { filter: Filter { names: vec!["kermit".into()], ..Default::default() }, ..Default::default() };
        let s = traj::convert::<Skat>(&SkatConfig {}, records(&fixture(), &ctx, None).unwrap(), &dir, &ConvertOptions::default()).unwrap();
        assert!(c.samples > 0 && c.samples < c.converted * 3);
        assert_eq!((c.converted, c.samples, c.frames), (s.converted, s.samples, s.frames));
        std::fs::remove_dir_all(&dir).unwrap();
        // the train / val split partitions the converted games
        let mut n = 0;
        for split in [Split::Train, Split::Val] {
            let ctx = ConvertCtx { split, val_mod: 7, ..Default::default() };
            let d = dir.with_extension(format!("{split:?}"));
            let s = traj::convert::<Skat>(&SkatConfig {}, records(&fixture(), &ctx, None).unwrap(), &d, &ConvertOptions::default()).unwrap();
            assert!(s.converted > 0);
            n += s.converted;
            std::fs::remove_dir_all(&d).unwrap();
        }
        assert_eq!(n, st.converted);
    }

    #[test]
    fn ratings_page_rows() {
        let html = "<pre> Nr Name        Games      % A  B  C       % D  E  F     T      M     R     RC\n  3 theCount    40733 Decl[ 34 83 17 ] Def[ 66 21 79 ] 27.66  10.85 1118.1  23.6\n 53 xskat      150990 Decl[ 34 79 21 ] Def[ 66 18 82 ] 22.72  -4.30 1000.0   0.0 Reference\n</pre>";
        let r = parse_ratings(html);
        assert_eq!(r["theCount"], (1118.1, 23.6, 40733));
        assert_eq!(r["xskat"], (1000.0, 0.0, 150990));
        let f = Filter { ratings: Some(r), min_rating: 1050.0, max_rc: 60.0, ..Default::default() };
        assert!(f.keep("theCount") && f.keep("theCount:2") && !f.keep("xskat") && !f.keep("nobody"));
        assert!(Filter::default().keep("anybody"));
    }
}
