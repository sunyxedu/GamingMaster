//! Contract bridge: deal records → candidate-action gm-shards. Rules and frames are
//! `gmrules::games::bridge` — the simulator's own code — and every record is REPLAYED
//! and verified by them (`traj.rs`: every chance outcome, seat, call and card; the
//! recorded contract and result where the source states them).
//!
//! Sources (`--format`):
//!
//! ```text
//! openspiel  the OpenSpiel WBridge5 dataset (gs://openspiel-data/bridge, train / valid /
//!            test .txt; ~1.16M deals bid in SAYC and played by WBridge5): one OpenSpiel
//!            history per line — 52 deal ids (card k to seat k mod 4), then calls 52..89
//!            and cards 0..51. Dealer N, nobody vulnerable (board 1) for every deal.
//! jack       Ben's WBC play data (github.com/lorserker/ben release "training-data",
//!            play_jackformat.zip → play.txt; ~1.9M boards played by bridge robots, every
//!            dealer / vulnerability): 4 lines per board —
//!              W:<W hand> <N hand> <E hand> <S hand>      (hands S.H.D.C, clockwise)
//!              <dealer> <vul ALL|NS|EW|-> <contract level+strain>.<result =|+k|-k>.<seat>
//!              <calls from the dealer: PP DD RR 1C .. 7N>
//!              <cards in play order: SQS2SA...>
//!            (the trailing seat letter is not the declarer — only level and strain are
//!            checked; the declarer and the doubling come from the auction)
//! pbn        Portable Bridge Notation: [Deal] [Dealer] [Vulnerable] [Auction] (+ AP),
//!            [Play] (tricks as seat columns from the opening leader), [Contract],
//!            [Declarer], [Result]
//! lin        BBO LIN (hand records, vugraph files with several boards): md (dealer digit
//!            1=S 2=W 3=N 4=E + hands S,W,N,E), sv (o / n / e / b), mb calls (alerts `!`
//!            ignored), pc cards, mc claims; boards split at qx / md
//! ```
//!
//! `.gz` and `.zip` inputs are read streaming (play_jackformat.zip needs no unpacking);
//! `--input` may be a file or a directory. Records are trajectories of the DEFAULT
//! config (board drawn by chance, IMP outcome): the board event of the record's
//! dealer / vulnerability, the deal, the calls and cards, so every source lands in one
//! shard family (same meta, same tensors). Records that stop early — claims, auctions
//! without play — are kept with `tgt.value_mask` false (all bridge shards carry
//! value_mask; `--complete-only` drops them). The same deal never straddles a split:
//! `--split train|val` routes by a hash of the deal (`--val-mod`, default 1 in 100).
//!
//! ```sh
//! gmdata bridge fetch --source openspiel --out data/bridge/raw/openspiel
//! gmdata bridge fetch --source ben --out data/bridge/raw/ben
//! gmdata bridge convert --format openspiel --input data/bridge/raw/openspiel/train.txt --out data/bridge/train/openspiel
//! gmdata bridge convert --format jack --input data/bridge/raw/ben/play_jackformat.zip --split train --out data/bridge/train/ben
//! ```
//!
//! Outcome (value target): the IMPs of the seat's side's duplicate score against 0,
//! / 24. BBO vugraph archives are NOT fetched (pending a decision); the LIN reader is
//! ready for them.

use crate::common;
use crate::traj::{self, ConvertOptions, Record};
use anyhow::{bail, ensure, Context, Result};
use clap::{Subcommand, ValueEnum};
use gmrules::games::bridge::{self as br, Bridge, BridgeConfig};
use gmrules::{Event, Game, Turn};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const OPENSPIEL_URL: &str = "https://storage.googleapis.com/openspiel-data/bridge";
pub const BEN_PLAY_URL: &str = "https://github.com/lorserker/ben/releases/download/training-data/play_jackformat.zip";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    Openspiel,
    Jack,
    Pbn,
    Lin,
}

impl Format {
    fn extensions(self) -> &'static [&'static str] {
        match self {
            Format::Openspiel | Format::Jack => &[".txt", ".txt.gz", ".zip"],
            Format::Pbn => &[".pbn", ".pbn.gz", ".zip"],
            Format::Lin => &[".lin", ".lin.gz", ".zip"],
        }
    }

    fn name(self) -> &'static str {
        match self {
            Format::Openspiel => "openspiel",
            Format::Jack => "jack",
            Format::Pbn => "pbn",
            Format::Lin => "lin",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Source {
    /// the OpenSpiel WBridge5 dataset: train.txt, valid.txt, test.txt (+ README.txt)
    Openspiel,
    /// Ben's WBC play data: play_jackformat.zip
    Ben,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Split {
    All,
    Train,
    Val,
}

// ---------------------------------------------------------------------------
// deals → trajectories
// ---------------------------------------------------------------------------

/// A deal as a source describes it, before it becomes a trajectory.
#[derive(Clone, Debug, Default)]
pub struct Deal {
    pub id: String,
    pub dealer: u8,
    /// vulnerable by partnership (N/S, E/W)
    pub vul: [bool; 2],
    /// the card ids of each seat N E S W (13 each)
    pub hands: [Vec<u8>; 4],
    pub calls: Vec<u32>,
    /// the cards in play order
    pub plays: Vec<u8>,
    /// PBN: tricks as seat columns (column 0 = `.0`), `None` = no card recorded
    pub trick_rows: Option<(u8, Vec<[Option<u8>; 4]>)>,
    /// recorded contract level / strain (None: not stated; Some(None): passed out)
    pub contract: Option<Option<(u8, u8)>>,
    pub declarer: Option<u8>,
    pub doubled: Option<u8>,
    /// tricks the declarer took (recorded result)
    pub tricks: Option<u8>,
    /// a claim ends the play: declarer's total tricks
    pub claim: Option<u8>,
    pub meta: Value,
}

/// FNV-1a over the holder of every card: the deal's identity for the splits.
pub fn deal_hash(hands: &[Vec<u8>; 4]) -> u64 {
    let mut holder = [0u8; 52];
    for (p, h) in hands.iter().enumerate() {
        for &c in h {
            if (c as usize) < 52 {
                holder[c as usize] = p as u8;
            }
        }
    }
    holder.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn check_hands(hands: &[Vec<u8>; 4]) -> Result<()> {
    let mut seen = [false; 52];
    for (p, h) in hands.iter().enumerate() {
        ensure!(h.len() == 13, "{} holds {} cards", br::SEAT_CHARS[p], h.len());
        for &c in h {
            ensure!(c < 52 && !seen[c as usize], "card {c} dealt twice");
            seen[c as usize] = true;
        }
    }
    Ok(())
}

/// Apply the next decision of the record: the acting seat comes from the rules (the
/// declarer on dummy's plays). Returns false when the action is illegal (the event is
/// kept: the converter's replay reports the record as illegal) or the deal is over.
fn push_action(g: &mut Bridge, events: &mut Vec<Event>, a: u32) -> Result<bool> {
    let Turn::Act(s) = g.turn() else {
        bail!("{} after the end of the deal", g.action_string(0, a));
    };
    let seat = s.iter().next().expect("one acting seat");
    events.push(Event::Act(seat, a));
    Ok(g.apply_action(seat, a).is_ok())
}

/// The scores of a finished deal (per seat) for a contract and declarer tricks.
fn returns_of(c: Option<br::Contract>, tricks: u8, vul: [bool; 2]) -> Vec<f64> {
    let Some(c) = c else { return vec![0.0; 4] };
    let s = br::score(c.level, c.denom, c.dbl, tricks, vul[br::partnership(c.declarer) as usize]) as f64;
    (0..4u8).map(|p| if br::partnership(p) == br::partnership(c.declarer) { s } else { -s }).collect()
}

/// A deal as a verified-on-replay record of the default config: the board event, the
/// deal (each hand ascending, card k to seat k mod 4), the calls and the cards (seats
/// from the rules). The recorded contract is checked here, the recorded result through
/// `expected_returns` (checked by the converter against the replayed returns).
pub fn deal_record(d: Deal) -> Result<Record> {
    check_hands(&d.hands)?;
    let board = br::board_of(d.dealer, d.vul);
    let mut hands = d.hands.clone();
    for h in hands.iter_mut() {
        h.sort_unstable();
    }
    let mut events = vec![Event::Chance(board as u32 - 1)];
    events.extend((0..52).map(|k| Event::Chance(hands[k % 4][k / 4] as u32)));
    let mut g = Bridge::new_game(&BridgeConfig::default());
    for e in &events {
        let Event::Chance(o) = *e else { unreachable!() };
        g.apply_chance(o).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    // a call after the end of the auction is an illegal action like any other (the
    // replay rejects the record)
    let mut legal = true;
    for &a in &d.calls {
        if !push_action(&mut g, &mut events, a)? {
            legal = false;
            break;
        }
    }
    let auction_over = legal && !g.in_auction();
    if auction_over {
        // the recorded contract must be the auction's (passed out: no contract)
        let got = g.contract();
        if let Some(want) = d.contract {
            ensure!(want == got.map(|c| (c.level, c.denom)), "recorded contract {} but the auction makes {}",
                    contract_name(want), got.map_or("pass".into(), |c| c.to_string_short()));
        }
        if let (Some(want), Some(c)) = (d.declarer, got) {
            ensure!(want == c.declarer, "recorded declarer {} but the auction makes {}", br::SEAT_CHARS[want as usize],
                    c.to_string_short());
        }
        if let (Some(want), Some(c)) = (d.doubled, got) {
            ensure!(want == c.dbl, "recorded doubling {want} but the auction makes {}", c.to_string_short());
        }
    }
    if legal && g.in_play() {
        match &d.trick_rows {
            Some((first, rows)) => {
                'tricks: for row in rows {
                    for _ in 0..4 {
                        let Some(hand) = g.hand_to_play() else { break 'tricks };
                        let Some(c) = row[((hand + 4 - first) % 4) as usize] else { break 'tricks };
                        if !push_action(&mut g, &mut events, c as u32)? {
                            break 'tricks;
                        }
                    }
                }
            }
            None => {
                for &c in &d.plays {
                    if !push_action(&mut g, &mut events, c as u32)? {
                        break;
                    }
                }
            }
        }
    }
    // a finished deal with a recorded result: its returns (checked by the converter
    // against the replay's); passed out: 0
    let finished = g.turn() == Turn::Terminal;
    let expected_returns = if !finished {
        None
    } else if g.contract().is_none() {
        Some(vec![0.0; 4])
    } else {
        d.tricks.map(|t| returns_of(g.contract(), t, d.vul))
    };
    let mut meta = json!({
        "board": board,
        "dealer": br::SEAT_CHARS[d.dealer as usize].to_string(),
        "vul": [d.vul[0], d.vul[1]],
        "contract": if auction_over { g.contract().map(|c| c.to_string_short()) } else { None },
    });
    if let Some(t) = d.claim {
        meta["claim"] = json!(t);
    }
    if let Value::Object(m) = d.meta {
        for (k, v) in m {
            meta[k] = v;
        }
    }
    Ok(Record { id: d.id, events, meta, seats: None, expected_returns, soft_targets: Vec::new() })
}

fn contract_name(c: Option<(u8, u8)>) -> String {
    c.map_or("pass".into(), |(l, s)| format!("{l}{}", br::DENOM_NAMES[s as usize]))
}

// ---------------------------------------------------------------------------
// source formats
// ---------------------------------------------------------------------------

/// One OpenSpiel WBridge5 line: board 1 (dealer N, nobody vulnerable), the deal in the
/// line's own order, then the actions with the rules' acting seats.
pub fn openspiel_record(line: &str, id: &str) -> Result<Option<Record>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let ids: Vec<u32> = line.split_whitespace().map(|t| t.parse::<u32>().with_context(|| format!("token {t:?}"))).collect::<Result<_>>()?;
    ensure!(ids.len() >= 52 && ids[..52].iter().all(|&c| c < 52), "a line starts with the 52 deal ids (0..51)");
    let mut g = Bridge::new_game(&BridgeConfig::default());
    let mut events = vec![Event::Chance(0)];
    g.apply_chance(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    for &c in &ids[..52] {
        events.push(Event::Chance(c));
        g.apply_chance(c).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    for &a in &ids[52..] {
        ensure!(a < br::N_ACTIONS, "action id {a} (0..90)");
        if !push_action(&mut g, &mut events, a)? {
            break;
        }
    }
    let contract = if g.in_auction() { None } else { g.contract().map(|c| c.to_string_short()) };
    Ok(Some(Record {
        id: id.to_string(),
        events,
        meta: json!({"source": "openspiel", "board": 1, "contract": contract}),
        seats: None,
        expected_returns: None,
        soft_targets: Vec::new(),
    }))
}

/// Hands "AKQ.JT9.876.5432" (spades, hearts, diamonds, clubs; `-` / empty = void).
pub fn parse_pbn_hand(s: &str) -> Result<Vec<u8>> {
    let parts: Vec<&str> = s.trim().split('.').collect();
    ensure!(parts.len() == 4, "hand {s:?}: 4 suits S.H.D.C expected");
    let mut out = Vec::with_capacity(13);
    for (k, part) in parts.iter().enumerate() {
        let suit = 3 - k as u8;
        if *part == "-" {
            continue;
        }
        let mut chars = part.chars().peekable();
        while let Some(ch) = chars.next() {
            let r = if ch == '1' && chars.peek() == Some(&'0') {
                chars.next();
                8
            } else {
                br::parse_rank(&ch.to_string()).with_context(|| format!("rank {ch:?} in hand {s:?}"))?
            };
            out.push(br::card_id(suit, r));
        }
    }
    Ok(out)
}

/// Hands clockwise from `first` ("N:... ... ... ..." without the prefix) → by seat.
fn hands_from(first: u8, hands: &[&str]) -> Result<[Vec<u8>; 4]> {
    ensure!(hands.len() == 4, "4 hands expected, got {}", hands.len());
    let mut out: [Vec<u8>; 4] = Default::default();
    for (k, h) in hands.iter().enumerate() {
        out[((first as usize) + k) % 4] = parse_pbn_hand(h)?;
    }
    Ok(out)
}

fn parse_vul(s: &str) -> Result<[bool; 2]> {
    Ok(match s.trim().to_ascii_uppercase().as_str() {
        "-" | "NONE" | "LOVE" | "O" | "0" | "Z" | "" => [false, false],
        "NS" | "N-S" | "N" => [true, false],
        "EW" | "E-W" | "E" => [false, true],
        "ALL" | "BOTH" | "B" => [true, true],
        v => bail!("vulnerability {v:?}"),
    })
}

/// One Jack-format board (4 lines, see the module docs).
pub fn jack_deal(lines: &[String], id: &str) -> Result<Deal> {
    ensure!(lines.len() == 4, "4 lines per board");
    let deal = lines[0].trim();
    let (first, rest) = deal.split_once(':').context("deal line <seat>:<hands>")?;
    let first = br::parse_seat(first).with_context(|| format!("seat {first:?}"))?;
    let hands = hands_from(first, &rest.split_whitespace().collect::<Vec<_>>())?;
    let info: Vec<&str> = lines[1].split_whitespace().collect();
    ensure!(info.len() >= 2, "info line <dealer> <vul> <contract>.<result>.<seat>");
    let dealer = br::parse_seat(info[0]).with_context(|| format!("dealer {:?}", info[0]))?;
    let vul = parse_vul(info[1])?;
    let mut d = Deal { id: id.to_string(), dealer, vul, hands, ..Default::default() };
    let calls: Vec<&str> = lines[2].split_whitespace().collect();
    d.calls = calls.iter().map(|c| br::parse_call(c).with_context(|| format!("call {c:?}"))).collect::<Result<_>>()?;
    let passed_out = d.calls.len() == 4 && d.calls.iter().all(|&c| c == br::PASS);
    if let Some(cr) = info.get(2) {
        let f: Vec<&str> = cr.split('.').collect();
        let head = f[0].to_ascii_uppercase();
        if passed_out || head.starts_with('P') || head == "0" {
            d.contract = Some(None);
        } else {
            let b = br::parse_call(head.trim_end_matches(['X', '*'])).and_then(br::bid_of).with_context(|| format!("contract {cr:?}"))?;
            d.contract = Some(Some(b));
            if let Some(res) = f.get(1) {
                // relative to the contract (=, +k, -k); a bare number (18 of the 1.9M
                // boards) is the declarer's trick count
                let t: i32 = match *res {
                    "=" => 6 + b.0 as i32,
                    x if x.starts_with(['+', '-']) => 6 + b.0 as i32 + x.parse::<i32>().with_context(|| format!("result {x:?}"))?,
                    x => x.parse().with_context(|| format!("result {x:?}"))?,
                };
                ensure!((0..=13).contains(&t), "result {res:?} of {cr:?}");
                d.tricks = Some(t as u8);
            }
        }
    }
    let play = lines[3].trim();
    ensure!(play.len() % 2 == 0, "cards are 2 characters each: {play:?}");
    d.plays = (0..play.len() / 2).map(|k| br::parse_card(&play[2 * k..2 * k + 2]).with_context(|| format!("card {:?}", &play[2 * k..2 * k + 2]))).collect::<Result<_>>()?;
    d.meta = json!({"source": "jack", "result": info.get(2)});
    Ok(d)
}

/// PBN games (one per `[Event` ... blank line), parsed into deals.
pub fn pbn_deal(text: &str, id: &str) -> Result<Option<Deal>> {
    let mut tags: Vec<(String, String)> = Vec::new();
    let mut sections: std::collections::HashMap<String, Vec<String>> = Default::default();
    let mut cur: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('%') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') {
            let inner = line.trim_start_matches('[').trim_end_matches(']');
            let (name, value) = inner.split_once(' ').unwrap_or((inner, ""));
            let value = value.trim().trim_matches('"').to_string();
            cur = Some(name.to_string());
            tags.push((name.to_string(), value));
            continue;
        }
        if let Some(c) = &cur {
            if c == "Auction" || c == "Play" {
                // strip notes (=1=), NAGs ($1) and brace comments
                let clean: String = line.split('{').next().unwrap_or("").to_string();
                sections.entry(c.clone()).or_default().extend(clean.split_whitespace().map(String::from));
            }
        }
    }
    let tag = |n: &str| tags.iter().find(|(k, _)| k == n).map(|(_, v)| v.as_str());
    let Some(deal) = tag("Deal") else { return Ok(None) };
    let (first, rest) = deal.split_once(':').context("[Deal \"<seat>:<hands>\"]")?;
    let first = br::parse_seat(first).with_context(|| format!("deal seat {first:?}"))?;
    let hands = hands_from(first, &rest.split_whitespace().collect::<Vec<_>>())?;
    let dealer = br::parse_seat(tag("Dealer").context("[Dealer]")?).context("dealer")?;
    let vul = parse_vul(tag("Vulnerable").unwrap_or("None"))?;
    let mut d = Deal { id: id.to_string(), dealer, vul, hands, ..Default::default() };
    // the auction, AP = pass out the rest
    if let Some(caller) = tag("Auction") {
        let caller = br::parse_seat(caller).context("[Auction] seat")?;
        ensure!(caller == dealer, "the auction starts with {} but {} deals", br::SEAT_CHARS[caller as usize], br::SEAT_CHARS[dealer as usize]);
        for t in sections.get("Auction").into_iter().flatten() {
            if t.starts_with('=') || t.starts_with('$') || t == "*" || t == "+" || t == "-" {
                continue;
            }
            if t.eq_ignore_ascii_case("AP") {
                // passes until the auction ends: 3 after a bid, else 4 in all
                let bid = d.calls.iter().any(|&c| br::bid_of(c).is_some());
                let trailing = d.calls.iter().rev().take_while(|&&c| c == br::PASS).count();
                let need: usize = if bid { 3 } else { 4 };
                d.calls.extend(std::iter::repeat_n(br::PASS, need.saturating_sub(trailing)));
                break;
            }
            d.calls.push(br::parse_call(t).with_context(|| format!("call {t:?}"))?);
        }
    }
    let contract = tag("Contract").unwrap_or("").to_ascii_uppercase();
    if contract.starts_with('P') {
        d.contract = Some(None);
    } else if !contract.is_empty() {
        let dbl = contract.matches('X').count() as u8;
        let b = br::parse_call(contract.trim_end_matches('X')).and_then(br::bid_of).with_context(|| format!("contract {contract:?}"))?;
        d.contract = Some(Some(b));
        d.doubled = Some(dbl);
        if let Some(s) = tag("Declarer").filter(|s| !s.is_empty()) {
            d.declarer = Some(br::parse_seat(s).context("declarer")?);
        }
        if let Some(r) = tag("Result").filter(|s| !s.is_empty()) {
            d.tricks = Some(r.parse().with_context(|| format!("result {r:?}"))?);
        }
    }
    if let Some(leader) = tag("Play").filter(|s| !s.is_empty() && *s != "-") {
        let first = br::parse_seat(leader).context("[Play] seat")?;
        let toks: Vec<&String> = sections
            .get("Play")
            .into_iter()
            .flatten()
            .filter(|t| !(t.starts_with('=') || t.starts_with('$') || t.as_str() == "*"))
            .collect();
        let rows = toks
            .chunks(4)
            .map(|ch| {
                let mut row = [None; 4];
                for (k, t) in ch.iter().enumerate() {
                    row[k] = if t.as_str() == "-" { None } else { Some(br::parse_card(t).with_context(|| format!("card {t:?}"))?) };
                }
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        d.trick_rows = Some((first, rows));
    }
    let players: Vec<Option<&str>> = ["North", "East", "South", "West"].iter().map(|s| tag(s)).collect();
    d.meta = json!({"source": "pbn", "board_tag": tag("Board"), "event": tag("Event"), "players": players});
    Ok(Some(d))
}

/// A LIN hand "SAK4HQ9DK962CAQ32" (suits in any order; ranks after each suit letter).
fn parse_lin_hand(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut suit: Option<u8> = None;
    let mut chars = s.trim().chars().peekable();
    while let Some(ch) = chars.next() {
        if let Some(su) = br::parse_suit(ch).filter(|_| ch.is_ascii_alphabetic() && "CDHScdhs".contains(ch)) {
            suit = Some(su);
            continue;
        }
        let su = suit.with_context(|| format!("rank before a suit in LIN hand {s:?}"))?;
        let r = if ch == '1' && chars.peek() == Some(&'0') {
            chars.next();
            8
        } else {
            br::parse_rank(&ch.to_string()).with_context(|| format!("rank {ch:?} in LIN hand {s:?}"))?
        };
        out.push(br::card_id(su, r));
    }
    Ok(out)
}

/// The boards of one LIN text (`key|value|` pairs; a board starts at `qx`, or at `md`
/// when the board already has a deal).
pub fn lin_deals(text: &str, id: &str) -> Vec<Result<Deal>> {
    let flat: String = text.lines().map(|l| l.trim_end_matches('\r')).collect::<Vec<_>>().join("");
    let toks: Vec<&str> = flat.split('|').collect();
    let mut boards: Vec<Vec<(String, String)>> = Vec::new();
    let mut cur: Vec<(String, String)> = Vec::new();
    let mut names: Option<String> = None;
    let mut i = 0;
    while i + 1 < toks.len() {
        let key = toks[i].trim().to_ascii_lowercase();
        let val = toks[i + 1].to_string();
        i += 2;
        let starts = (key == "qx" && cur.iter().any(|(k, _)| k == "md" || k == "mb"))
            || (key == "md" && cur.iter().any(|(k, _)| k == "md"));
        if starts {
            boards.push(std::mem::take(&mut cur));
        }
        if key == "pn" {
            names = Some(val.clone());
        }
        if key == "md" && !cur.iter().any(|(k, _)| k == "pn") {
            if let Some(n) = &names {
                cur.push(("pn".into(), n.clone()));
            }
        }
        cur.push((key, val));
    }
    if cur.iter().any(|(k, _)| k == "md") {
        boards.push(cur);
    }
    boards
        .into_iter()
        .filter(|b| b.iter().any(|(k, _)| k == "md"))
        .enumerate()
        .map(|(n, b)| lin_board(&b, &format!("{id}#{}", n + 1)))
        .collect()
}

fn lin_board(kv: &[(String, String)], id: &str) -> Result<Deal> {
    let get = |k: &str| kv.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
    let md = get("md").context("no md|...| deal")?;
    let mut chars = md.chars();
    let d0 = chars.next().context("empty md")?;
    let dealer = match d0 {
        '1' => 2u8,
        '2' => 3,
        '3' => 0,
        '4' => 1,
        c => bail!("md dealer digit {c:?} (1=S 2=W 3=N 4=E)"),
    };
    let parts: Vec<&str> = chars.as_str().split(',').collect();
    // hands in the order S, W, N, E; the last one may be left out
    let order = [2u8, 3, 0, 1];
    let mut hands: [Vec<u8>; 4] = Default::default();
    for (k, &seat) in order.iter().enumerate() {
        hands[seat as usize] = match parts.get(k) {
            Some(p) if !p.trim().is_empty() => parse_lin_hand(p)?,
            _ => Vec::new(),
        };
    }
    let missing: Vec<usize> = (0..4).filter(|&s| hands[s].is_empty()).collect();
    if missing.len() == 1 {
        let taken: Vec<u8> = hands.iter().flatten().copied().collect();
        hands[missing[0]] = (0..52u8).filter(|c| !taken.contains(c)).collect();
    }
    let vul = parse_vul(get("sv").unwrap_or("o"))?;
    let mut d = Deal { id: id.to_string(), dealer, vul, hands, ..Default::default() };
    for (k, v) in kv {
        match k.as_str() {
            "mb" => {
                let c = v.trim().trim_end_matches('!');
                if c.is_empty() {
                    continue;
                }
                d.calls.push(br::parse_call(c).with_context(|| format!("LIN call {v:?}"))?);
            }
            "pc" => d.plays.push(br::parse_card(v).with_context(|| format!("LIN card {v:?}"))?),
            "mc" => d.claim = Some(v.trim().parse().with_context(|| format!("claim {v:?}"))?),
            _ => {}
        }
    }
    let board = get("ah").map(|s| s.trim().trim_start_matches("Board").trim().to_string()).or_else(|| get("qx").map(String::from));
    let players = get("pn").map(|p| p.split(',').map(String::from).collect::<Vec<_>>());
    d.meta = json!({"source": "lin", "board_label": board, "players_swne": players});
    Ok(d)
}

// ---------------------------------------------------------------------------
// input streams
// ---------------------------------------------------------------------------

type Lines = Box<dyn Iterator<Item = std::io::Result<String>> + Send>;

/// The text lines of a file: plain, `.gz`, or every entry of a `.zip` (streamed by a
/// reader thread — a large archive is never unpacked to disk).
pub fn open_lines(path: &Path) -> Result<Lines> {
    let name = path.to_string_lossy().to_lowercase();
    if name.ends_with(".zip") {
        let path = path.to_path_buf();
        let (tx, rx) = std::sync::mpsc::sync_channel::<std::io::Result<String>>(8192);
        std::thread::spawn(move || {
            let run = || -> std::io::Result<()> {
                let f = std::fs::File::open(&path)?;
                let mut z = zip::ZipArchive::new(f).map_err(std::io::Error::other)?;
                for i in 0..z.len() {
                    let e = z.by_index(i).map_err(std::io::Error::other)?;
                    if e.is_dir() {
                        continue;
                    }
                    for line in BufReader::new(e).lines() {
                        if tx.send(line).is_err() {
                            return Ok(());
                        }
                    }
                }
                Ok(())
            };
            if let Err(e) = run() {
                let _ = tx.send(Err(e));
            }
        });
        return Ok(Box::new(rx.into_iter()));
    }
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r: Box<dyn Read + Send> = if name.ends_with(".gz") { Box::new(flate2::read::MultiGzDecoder::new(f)) } else { Box::new(f) };
    Ok(Box::new(BufReader::new(r).lines()))
}

/// Deals (or ready records: OpenSpiel lines) of one input file, in order.
enum Item {
    Record(Record),
    Deal(Deal),
}

fn items(format: Format, path: &Path) -> Result<Box<dyn Iterator<Item = Result<Item>> + Send>> {
    let fname = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let lines = open_lines(path)?;
    Ok(match format {
        Format::Openspiel => Box::new(lines.enumerate().filter_map(move |(ln, l)| {
            let id = format!("{fname}:{}", ln + 1);
            match l {
                Ok(l) => openspiel_record(&l, &id).with_context(|| id.clone()).transpose().map(|r| r.map(Item::Record)),
                Err(e) => Some(Err(anyhow::Error::from(e).context(id))),
            }
        })),
        Format::Jack => {
            // 4 lines per board; resync on the next deal line after a broken board
            let mut buf: Vec<String> = Vec::with_capacity(4);
            let mut start = 0usize;
            let mut lines = lines.enumerate();
            let fname2 = fname.clone();
            Box::new(std::iter::from_fn(move || loop {
                let Some((ln, l)) = lines.next() else {
                    return (!buf.is_empty()).then(|| {
                        let n = buf.len();
                        buf.clear();
                        Err(anyhow::anyhow!("{fname2}:{}: a board of {n} lines at the end of the file", start + 1))
                    });
                };
                let l = match l {
                    Ok(l) => l.trim_end_matches('\r').to_string(),
                    Err(e) => return Some(Err(e.into())),
                };
                if l.trim().is_empty() {
                    continue;
                }
                let is_deal = l.len() > 2 && l.as_bytes()[1] == b':' && "NESW".contains(l.as_bytes()[0] as char);
                if is_deal && !buf.is_empty() {
                    let n = buf.len();
                    buf.clear();
                    buf.push(l);
                    let at = start + 1;
                    start = ln;
                    return Some(Err(anyhow::anyhow!("{fname2}:{at}: a board of {n} lines")));
                }
                if buf.is_empty() {
                    start = ln;
                }
                buf.push(l);
                if buf.len() == 4 {
                    let id = format!("{fname2}:{}", start + 1);
                    let lines4 = std::mem::take(&mut buf);
                    return Some(jack_deal(&lines4, &id).with_context(|| id.clone()).map(Item::Deal));
                }
            }))
        }
        Format::Pbn => {
            // games are separated by blank lines (a game = lines up to a blank line after a tag)
            let mut buf = String::new();
            let mut start = 0usize;
            let mut have_tag = false;
            let mut lines = lines.enumerate();
            let mut done = false;
            let fname2 = fname.clone();
            Box::new(std::iter::from_fn(move || loop {
                if done {
                    return None;
                }
                let next = lines.next();
                let end_of_game = match &next {
                    None => true,
                    Some((_, Ok(l))) => l.trim().is_empty() && have_tag,
                    Some((_, Err(_))) => false,
                };
                if end_of_game {
                    if next.is_none() {
                        done = true;
                    }
                    if !have_tag {
                        continue;
                    }
                    let text = std::mem::take(&mut buf);
                    have_tag = false;
                    let id = format!("{fname2}:{}", start + 1);
                    match pbn_deal(&text, &id).with_context(|| id.clone()) {
                        Ok(Some(d)) => return Some(Ok(Item::Deal(d))),
                        Ok(None) => continue,
                        Err(e) => return Some(Err(e)),
                    }
                }
                match next.expect("not the end") {
                    (_, Err(e)) => return Some(Err(e.into())),
                    (ln, Ok(l)) => {
                        if !have_tag && l.trim_start().starts_with('[') {
                            have_tag = true;
                            start = ln;
                        }
                        buf.push_str(&l);
                        buf.push('\n');
                    }
                }
            }))
        }
        Format::Lin => {
            let text: String = lines.collect::<std::io::Result<Vec<_>>>()?.join("\n");
            Box::new(lin_deals(&text, &fname).into_iter().map(|r| r.map(Item::Deal)))
        }
    })
}

/// Counters of the source side (printed with the converter's stats).
#[derive(Default)]
pub struct Selection {
    pub other_split: std::sync::atomic::AtomicUsize,
}

/// Records of every input file of `format` under `input`, routed by `split` (deal
/// hash modulo `val_mod`: 0 = val), after skipping `skip` and at most `max` of them.
pub fn records<'a>(format: Format, input: &Path, split: Split, val_mod: u64, sel: &'a Selection)
    -> Result<impl Iterator<Item = Result<Record>> + 'a> {
    let files: Vec<PathBuf> = common::find_files(input, format.extensions())?
        .into_iter()
        .filter(|p| input.is_file() || !p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("readme")))
        .collect();
    ensure!(!files.is_empty(), "no {} files under {}", format.name(), input.display());
    let it = files.into_iter().flat_map(move |path| -> Box<dyn Iterator<Item = Result<Item>> + Send> {
        match items(format, &path) {
            Ok(it) => it,
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    });
    Ok(it.filter_map(move |item| {
        let rec = match item {
            Err(e) => return Some(Err(e)),
            Ok(Item::Record(r)) => r,
            Ok(Item::Deal(d)) => {
                let id = d.id.clone();
                match deal_record(d).with_context(|| id) {
                    Ok(r) => r,
                    Err(e) => return Some(Err(e)),
                }
            }
        };
        if split != Split::All {
            let hands = hands_of(&rec.events);
            let val = deal_hash(&hands) % val_mod.max(1) == 0;
            if val != (split == Split::Val) {
                sel.other_split.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
        }
        Some(Ok(rec))
    }))
}

/// The hands of a record (its 52 deal events after the board event).
fn hands_of(events: &[Event]) -> [Vec<u8>; 4] {
    let mut hands: [Vec<u8>; 4] = Default::default();
    for (k, e) in events.iter().skip(1).take(52).enumerate() {
        if let Event::Chance(c) = *e {
            hands[k % 4].push(c as u8);
        }
    }
    hands
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Subcommand)]
pub enum Cmd {
    /// Download raw data (openspiel: the WBridge5 dataset train / valid / test from GCS;
    /// ben: Ben's WBC play data, play_jackformat.zip). BBO vugraph is not fetched.
    Fetch {
        #[arg(long, value_enum)]
        source: Source,
        #[arg(long)]
        out: PathBuf,
    },
    /// Replay and verify every record without writing shards (a whole corpus in
    /// minutes): counts of legal / illegal / truncated / result-mismatch records
    Check {
        #[arg(long, value_enum)]
        format: Format,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_enum, default_value = "all")]
        split: Split,
        #[arg(long, default_value_t = 100)]
        val_mod: u64,
        #[arg(long)]
        max_records: Option<usize>,
        /// list at most this many failing records in "errors"
        #[arg(long, default_value_t = 10)]
        max_errors: usize,
    },
    /// Convert deal records to candidate-action gm-shards (every record replayed and
    /// verified by the rules)
    Convert {
        #[arg(long, value_enum)]
        format: Format,
        /// a file or a directory (.txt / .pbn / .lin, also .gz and .zip)
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// all | train | val — by deal hash (the same deal never lands in both)
        #[arg(long, value_enum, default_value = "all")]
        split: Split,
        /// 1 in N deals go to val
        #[arg(long, default_value_t = 100)]
        val_mod: u64,
        #[arg(long, default_value_t = 0)]
        skip: usize,
        #[arg(long)]
        max_records: Option<usize>,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// drop records that stop before the end of the deal (claims, auctions only)
        #[arg(long)]
        complete_only: bool,
        /// game config JSON (only "reward" may change: records carry their own board)
        #[arg(long, default_value = "")]
        config: String,
    },
}

/// Counters of [`check`].
#[derive(Debug, Default, serde::Serialize)]
pub struct CheckStats {
    pub records: usize,
    pub complete: usize,
    pub truncated: usize,
    pub illegal: usize,
    pub returns_mismatch: usize,
    pub unparseable: usize,
    pub frames: usize,
    pub errors: Vec<String>,
}

/// Replay every record (in parallel chunks) with the rules, as the converter does,
/// without encoding shards to disk.
pub fn check(recs: impl Iterator<Item = Result<Record>>, max_errors: usize) -> CheckStats {
    use rayon::prelude::*;
    let cfg = BridgeConfig::default();
    let mut st = CheckStats::default();
    let mut recs = recs.peekable();
    while recs.peek().is_some() {
        let chunk: Vec<Result<Record>> = recs.by_ref().take(4096).collect();
        let res: Vec<(String, std::result::Result<(bool, bool, usize), String>)> = chunk
            .par_iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|rec| {
                let out = gmrules::replay::<Bridge>(&cfg, &rec.events).map_err(|e| e.to_string()).map(|r| {
                    let mismatch = match (&rec.expected_returns, r.terminal()) {
                        (Some(want), true) => want.iter().zip(r.state.returns()).any(|(a, b)| (a - b).abs() > 1e-6),
                        _ => false,
                    };
                    (r.terminal(), mismatch, r.decisions.len())
                });
                (rec.id.clone(), out)
            })
            .collect();
        for r in &chunk {
            if let Err(e) = r {
                st.records += 1;
                st.unparseable += 1;
                if st.errors.len() < max_errors {
                    st.errors.push(format!("unparseable: {e:#}"));
                }
            }
        }
        for (id, r) in res {
            st.records += 1;
            let note = match r {
                Ok((_, true, _)) => {
                    st.returns_mismatch += 1;
                    format!("{id}: recorded result differs from the replay")
                }
                Ok((terminal, false, frames)) => {
                    st.frames += frames;
                    if terminal { st.complete += 1 } else { st.truncated += 1 }
                    continue;
                }
                Err(e) => {
                    st.illegal += 1;
                    format!("{id}: illegal: {e}")
                }
            };
            if st.errors.len() < max_errors {
                st.errors.push(note);
            }
        }
    }
    st
}

fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("GamingMaster-research/0.1 (imitation-learning data pipeline)")
        .timeout(Duration::from_secs(6 * 3600))
        .build()?)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { source, out } => {
            let c = client()?;
            let files: Vec<(String, PathBuf)> = match source {
                Source::Openspiel => ["README.txt", "train.txt", "valid.txt", "test.txt"]
                    .iter()
                    .map(|f| (format!("{OPENSPIEL_URL}/{f}"), out.join(f)))
                    .collect(),
                Source::Ben => vec![(BEN_PLAY_URL.to_string(), out.join("play_jackformat.zip"))],
            };
            for (url, dest) in &files {
                ensure!(common::download(&c, url, dest)?, "{url}: not found");
            }
            println!("{}", json!({"source": format!("{source:?}").to_lowercase(), "files": files.iter().map(|f| f.1.clone()).collect::<Vec<_>>()}));
            Ok(())
        }
        Cmd::Check { format, input, split, val_mod, max_records, max_errors } => {
            let sel = Selection::default();
            let recs = records(format, &input, split, val_mod, &sel)?.take(max_records.unwrap_or(usize::MAX));
            let mut v = serde_json::to_value(check(recs, max_errors))?;
            v["format"] = json!(format.name());
            v["other_split"] = json!(sel.other_split.load(std::sync::atomic::Ordering::Relaxed));
            println!("{v}");
            Ok(())
        }
        Cmd::Convert { format, input, out, split, val_mod, skip, max_records, shard_size, store_events, complete_only, config } => {
            let cfg: BridgeConfig = if config.trim().is_empty() { BridgeConfig::default() } else { serde_json::from_str(&config)? };
            Bridge::check_config(&cfg).map_err(|e| anyhow::anyhow!("{e}"))?;
            ensure!(cfg.board == 0 && cfg.deal.is_empty(), "records carry their own board and deal: config may only set \"reward\"");
            let sel = Selection::default();
            let recs = records(format, &input, split, val_mod, &sel)?.skip(skip).take(max_records.unwrap_or(usize::MAX));
            let opts = ConvertOptions { shard_size, store_events, allow_truncated: !complete_only, soft_k: 0 };
            let st = traj::convert::<Bridge>(&cfg, recs, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            v["format"] = json!(format.name());
            v["other_split"] = json!(sel.other_split.load(std::sync::atomic::Ordering::Relaxed));
            println!("{v}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrules::replay;

    const HANDS: [&str; 4] = ["AKQJ.AKQ.AKQ.AKQ", "T98.JT9.JT9.JT98", "765.876.876.7654", "432.5432.5432.32"];

    fn hands() -> [Vec<u8>; 4] {
        let mut h: [Vec<u8>; 4] = Default::default();
        for (p, s) in HANDS.iter().enumerate() {
            h[p] = parse_pbn_hand(s).unwrap();
        }
        h
    }

    #[test]
    fn hands_and_notation() {
        let h = hands();
        check_hands(&h).unwrap();
        assert!(h[0].contains(&br::card_id(3, 12)) && h[3].contains(&br::card_id(0, 1)));
        assert_eq!(parse_pbn_hand("-.AKQJT98765432.-.-").unwrap().len(), 13);
        assert_eq!(parse_pbn_hand("10.2.3.4").unwrap()[0], br::card_id(3, 8));
        assert!(parse_pbn_hand("AK.Q").is_err());
        assert_eq!(parse_lin_hand("SAK4HQ9DK962CAQ32").unwrap().len(), 13);
        assert_eq!(parse_lin_hand("hT9s2").unwrap(), vec![br::card_id(2, 8), br::card_id(2, 7), br::card_id(3, 0)]);
        assert_eq!(parse_vul("Both").unwrap(), [true, true]);
        assert_eq!(parse_vul("-").unwrap(), [false, false]);
        assert!(parse_vul("xx").is_err());
        let mut bad = h.clone();
        bad[0][0] = bad[1][0];
        assert!(check_hands(&bad).is_err());
        // the split never separates a deal from itself
        assert_eq!(deal_hash(&h), deal_hash(&h.clone()));
    }

    #[test]
    fn deal_records_verify_contract_and_result() {
        // N opens 7NT, all pass, N takes all 13 tricks (highest legal card every time)
        let mut d = Deal { id: "x".into(), dealer: 0, vul: [false, false], hands: hands(), ..Default::default() };
        d.calls = ["7NT", "Pass", "Pass", "Pass"].iter().map(|c| br::parse_call(c).unwrap()).collect();
        let r = deal_record(d.clone()).unwrap();
        let mut g = replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap().state;
        let mut plays = Vec::new();
        while let Turn::Act(s) = g.turn() {
            let seat = s.iter().next().unwrap();
            let mut l = Vec::new();
            g.legal_actions(seat, &mut l);
            let a = *l.last().unwrap();
            plays.push(a as u8);
            g.apply_action(seat, a).unwrap();
        }
        d.plays = plays;
        d.contract = Some(Some((7, 4)));
        d.tricks = Some(13);
        let r = deal_record(d.clone()).unwrap();
        assert_eq!(r.events.len(), 1 + 52 + 4 + 52);
        assert_eq!(r.events[0], Event::Chance(0)); // board 1
        assert_eq!(r.expected_returns, Some(vec![1520.0, -1520.0, 1520.0, -1520.0]));
        let st = replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap();
        assert_eq!(st.state.returns(), r.expected_returns.clone().unwrap());
        // a wrong recorded result is carried to the converter's returns check
        let mut wrong = d.clone();
        wrong.tricks = Some(12);
        assert_eq!(deal_record(wrong).unwrap().expected_returns.unwrap()[0], -50.0);
        // a wrong recorded contract / declarer is rejected here
        let mut wrong = d.clone();
        wrong.contract = Some(Some((7, 3)));
        assert!(deal_record(wrong).unwrap_err().to_string().contains("recorded contract"));
        let mut wrong = d.clone();
        wrong.declarer = Some(2);
        assert!(deal_record(wrong).is_err());
        // an illegal card stops the record there (the replay reports it as illegal)
        let mut bad = d.clone();
        bad.plays.swap(0, 1);
        let r = deal_record(bad).unwrap();
        assert!(replay::<Bridge>(&BridgeConfig::default(), &r.events).is_err());
        // board of the dealer / vulnerability: E deals, N/S vulnerable = board 2
        let mut e = d.clone();
        e.dealer = 1;
        e.vul = [true, false];
        e.calls = vec![br::PASS; 4];
        e.plays.clear();
        e.contract = Some(None);
        e.tricks = None;
        let r = deal_record(e).unwrap();
        assert_eq!(r.events[0], Event::Chance(1));
        assert_eq!(r.expected_returns, Some(vec![0.0; 4]));
    }

    #[test]
    fn openspiel_lines() {
        let line = "27 46 36 43 18 22 0 20 24 2 40 41 28 16 21 10 42 32 48 47 13 17 3 5 12 25 34 8 29 38 23 4 30 26 35 19 9 44 7 51 14 1 45 15 49 33 11 50 39 31 6 37 52 52 58 59 52 61 52 62 52 64 52 52 52 39 31 3 47 41 13 1 45 21 5 49 17 27 16 7 51 50 18 2 6 15 12 22 23 48 4 24 32 0 20 28 44 25 34 37 9 43 14 26 11 10 30 46 36 33 40 8 29 38 35 19 42";
        let r = openspiel_record(line, "t:1").unwrap().unwrap();
        assert_eq!(r.events.len(), 1 + 117);
        let st = replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap();
        assert!(st.terminal());
        assert!(openspiel_record("# c", "t").unwrap().is_none());
        assert!(openspiel_record("1 2 3", "t").is_err());
    }

    #[test]
    fn jack_boards() {
        let lines: Vec<String> = [
            "W:AT73.JT4.92.A984 K4.AK76.KQT85.T2 QJ9865.Q53.64.65 2.982.AJ73.KQJ73",
            "W ALL 3N.-1.N",
            "PP 1N PP 3N PP PP PP",
            "SQS2SAS4S3SKS5H2D5D4DJD2H9H4HAH3DTD6D3D9D8S6D7HTDQC5DAC4CKCAC2C6STH6S8C3S7H7S9C7SJH8C8CTHQCJHJHKDKH5CQC9",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let d = jack_deal(&lines, "j:1").unwrap();
        assert_eq!((d.dealer, d.vul, d.contract, d.tricks), (3, [true, true], Some(Some((3, 4))), Some(8)));
        let r = deal_record(d).unwrap();
        assert_eq!(r.events[0], Event::Chance(3)); // W deals, all vulnerable: board 4
        let st = replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap();
        assert!(st.terminal());
        assert_eq!(st.state.returns(), r.expected_returns.unwrap());
        assert_eq!(st.state.returns()[0], -100.0);
    }

    #[test]
    fn pbn_games() {
        let text = "[Event \"t\"]\n[Board \"1\"]\n[Dealer \"N\"]\n[Vulnerable \"None\"]\n\
                    [Deal \"N:AKQJ.AKQ.AKQ.AKQ T98.JT9.JT9.JT98 765.876.876.7654 432.5432.5432.32\"]\n\
                    [Declarer \"N\"]\n[Contract \"1NT\"]\n[Result \"13\"]\n[Auction \"N\"]\n1NT AP\n[Play \"E\"]\n\
                    HJ H8 H5 HA\n";
        let d = pbn_deal(text, "p:1").unwrap().unwrap();
        assert_eq!(d.calls, vec![br::bid_action(1, 4), br::PASS, br::PASS, br::PASS]);
        let (first, rows) = d.trick_rows.clone().unwrap();
        assert_eq!((first, rows.len()), (1, 1));
        let r = deal_record(d).unwrap();
        // one trick recorded: E leads HJ, S H8, W H5, N HA; then the record stops
        let st = replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap();
        assert!(!st.terminal());
        assert_eq!(st.state.cards_played(), 4);
        assert_eq!(r.expected_returns, None);
        assert!(pbn_deal("[Event \"x\"]\n", "p").unwrap().is_none());
    }

    #[test]
    fn lin_boards() {
        // S deals (1), E/W vulnerable; the E hand is left out
        let text = "pn|Sam,Wes,Nor,Eve|st||md|1SAKQJHAKQDAKQCAKQ,S432H5432D5432C32,S765H876D876C7654,|rh||ah|Board 3|sv|e|\
                    mb|1N!|an|15-17|mb|p|mb|p|mb|p|pg||pc|D2|pc|D6|pc|DJ|pc|DA|pg||mc|13|";
        let deals = lin_deals(text, "l");
        assert_eq!(deals.len(), 1);
        let d = deals.into_iter().next().unwrap().unwrap();
        assert_eq!((d.dealer, d.vul, d.claim), (2, [false, true], Some(13)));
        assert_eq!(d.hands[1].len(), 13);
        let r = deal_record(d).unwrap();
        let st = replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap();
        assert_eq!((st.terminal(), st.state.cards_played()), (false, 4));
        assert_eq!(r.meta["claim"], 13);
        // two boards (qx), names carried over
        let two = format!("vg|Test,,I,1,2,A,,B,|pn|a,b,c,d|qx|o1|{}qx|o2|{}",
                          "md|3HAKQJT98765432,SAKQJT98765432,DAKQJT98765432,|sv|o|mb|p|mb|p|mb|p|mb|p|",
                          "md|4HAKQJT98765432,SAKQJT98765432,DAKQJT98765432,|sv|b|mb|p|mb|p|mb|p|mb|p|");
        let deals: Vec<Deal> = lin_deals(&two, "v").into_iter().map(|d| d.unwrap()).collect();
        assert_eq!(deals.len(), 2);
        assert_eq!((deals[0].dealer, deals[1].dealer, deals[1].vul), (0, 1, [true, true]));
        assert_eq!(deals[1].meta["players_swne"][0], "a");
        for d in deals {
            let r = deal_record(d).unwrap();
            assert!(replay::<Bridge>(&BridgeConfig::default(), &r.events).unwrap().terminal());
        }
    }
}
