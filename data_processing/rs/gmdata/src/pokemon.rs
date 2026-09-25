//! Pokémon Showdown replays (observer-view sim protocol) → shards for the pokemon adapter.
//!
//! One sample per (replay, side): a frame at every `|turn|N` boundary (state before
//! the turn) labelled with the side's visible action that turn — its first `|move|`
//! (action_type 0, move_idx = slot in the active mon's revealed-move list, −1 for a
//! 5th+ revealed move) or chosen `|switch|` (action_type 1, switch_idx = team slot).
//! Unusable turns are skipped (struggle, called moves, cant, forced/mid-turn switches).
//!
//!   obs.species_ids [T,12] i64 (crc32 hash buckets)   obs.move_ids [T,4] i64
//!   obs.poke_feats [T,12,17] f32   obs.slot_mask [T,12] bool   obs.field_seq [T,20] f32
//!   tgt.action_type / move_idx / switch_idx / tera / active_idx [T] i64, tgt.value [T] f32
//! Replays without a `|win|` yield no samples.

use crate::common;
use anyhow::Result;
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const TEAM: usize = 6;
const FEAT: usize = 17;
const FIELD: usize = 20;
const MOVES: usize = 4;
const F_HP: usize = 0;
const F_ACTIVE: usize = 1;
const F_FAINTED: usize = 2;
const F_STATUS: usize = 3;
const F_BOOSTS: usize = 10;
const F_REVEALED: usize = 15;
const F_TERA: usize = 16;
const BOOSTS: [&str; 5] = ["atk", "def", "spa", "spd", "spe"];
const SCREENS: [&str; 3] = ["reflect", "lightscreen", "auroraveil"];

fn norm(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

fn bucket(name: &str, buckets: u32) -> i64 {
    if name.is_empty() {
        return 0;
    }
    (crc32fast::hash(norm(name).as_bytes()) % (buckets - 1) + 1) as i64
}

fn status_idx(s: &str) -> Option<usize> {
    ["", "brn", "par", "psn", "tox", "slp", "frz"].iter().position(|&x| x == s)
}

fn weather_idx(w: &str) -> usize {
    match w {
        "sunnyday" | "desolateland" => 0,
        "raindance" | "primordialsea" => 1,
        "sandstorm" => 2,
        "snow" | "snowscape" | "hail" => 3,
        _ => 4,
    }
}

fn terrain_idx(t: &str) -> Option<usize> {
    ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"].iter().position(|&x| x == t)
}

fn base_species(details: &str) -> String {
    details.split(',').next().unwrap_or("").trim().to_string()
}

/// Python `float(s)` for the forms seen in HP strings.
fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok().filter(|x| x.is_finite() || t.to_lowercase().contains("inf"))
}

fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let t2 = t.strip_prefix('+').unwrap_or(t);
    if t2.is_empty() || t2.starts_with('+') {
        return None;
    }
    t2.parse::<i64>().ok()
}

/// "78/100" / "0 fnt" / "45/100 par" → (frac, status, fainted)
fn parse_hp(hp: &str) -> (f64, Option<usize>, bool) {
    let parts: Vec<&str> = hp.split_whitespace().collect();
    if parts.is_empty() {
        return (1.0, None, false);
    }
    let (fs, rest) = (parts[0], &parts[1..]);
    if rest.first() == Some(&"fnt") {
        return (0.0, None, true);
    }
    let frac = if let Some((a, b)) = fs.split_once('/') {
        match (py_float(a), py_float(b)) {
            (Some(n), Some(d)) => (n / d.max(1.0)).clamp(0.0, 1.0),
            _ => 1.0,
        }
    } else {
        py_float(fs).map(|x| (x / 100.0).clamp(0.0, 1.0)).unwrap_or(1.0)
    };
    let st = rest.first().and_then(|r| status_idx(r));
    if frac == 0.0 {
        return (0.0, st, true);
    }
    (frac, st, false)
}

#[derive(Clone, Default)]
struct Mon {
    species: String,
    hp: f64,
    status: usize,
    boosts: [i64; 5],
    fainted: bool,
    revealed: bool,
    tera: bool,
    moves: Vec<String>,
}

impl Mon {
    fn new(species: String) -> Self {
        Mon { species, hp: 1.0, ..Default::default() }
    }
}

#[derive(Clone, Default)]
struct SideSt {
    name: String,
    team: Vec<Mon>,
    active: Option<usize>,
    nick2slot: HashMap<String, usize>,
    nick_order: Vec<String>,
    hazards: HashMap<String, i64>,
}

impl SideSt {
    fn claimed(&self, i: usize) -> bool {
        self.nick2slot.values().any(|&v| v == i)
    }
    fn assign(&mut self, nick: &str, i: usize) -> usize {
        if !self.nick2slot.contains_key(nick) {
            self.nick_order.push(nick.to_string());
        }
        self.nick2slot.insert(nick.to_string(), i);
        i
    }
    fn slot_for(&mut self, nick: &str, details: &str) -> usize {
        if let Some(&s) = self.nick2slot.get(nick) {
            return s;
        }
        let sp = base_species(details);
        let n = norm(&sp);
        for i in 0..self.team.len() {
            if norm(&self.team[i].species) == n && !self.claimed(i) {
                return self.assign(nick, i);
            }
        }
        for i in 0..self.team.len() {
            let pn = norm(&self.team[i].species);
            if !pn.is_empty() && (n.starts_with(&pn) || pn.starts_with(&n)) && !self.claimed(i) {
                return self.assign(nick, i);
            }
        }
        if self.team.len() < TEAM {
            self.team.push(Mon::new(sp));
            let i = self.team.len() - 1;
            return self.assign(nick, i);
        }
        let i = self.active.unwrap_or(0);
        self.assign(nick, i)
    }
    fn slot_of(&self, nick: &str) -> Option<usize> {
        self.nick2slot.get(nick).copied().or(self.active).filter(|&s| s < self.team.len())
    }
}

#[derive(Clone)]
struct Frame {
    species: Vec<String>,
    moves: Vec<String>,
    feats: Vec<f32>,
    mask: Vec<bool>,
    field: Vec<f32>,
    active: i64,
}

struct Rec {
    frame: Frame,
    action_type: i64,
    move_idx: i64,
    switch_idx: i64,
    tera: i64,
}

struct Battle {
    sides: [SideSt; 2],
    weather: Option<usize>,
    terrain: Option<usize>,
    turn: i64,
    winner: Option<usize>,
}

fn side_idx(pos: &str) -> Option<usize> {
    match pos {
        "p1" => Some(0),
        "p2" => Some(1),
        _ => None,
    }
}

fn ident(tok: &str) -> (String, String) {
    let (pos, nick) = tok.split_once(':').unwrap_or((tok, ""));
    (pos.trim().chars().take(2).collect(), nick.trim().to_string())
}

fn encode(b: &Battle, side: usize) -> Frame {
    let other = 1 - side;
    let mut feats = vec![0f32; 2 * TEAM * FEAT];
    let mut mask = vec![false; 2 * TEAM];
    for (base, s) in [(0, &b.sides[side]), (TEAM, &b.sides[other])] {
        for (i, p) in s.team.iter().take(TEAM).enumerate() {
            let j = base + i;
            mask[j] = !p.species.is_empty();
            let f = &mut feats[j * FEAT..(j + 1) * FEAT];
            f[F_HP] = p.hp as f32;
            f[F_ACTIVE] = (s.active == Some(i)) as u8 as f32;
            f[F_FAINTED] = p.fainted as u8 as f32;
            f[F_STATUS + p.status] = 1.0;
            for (k, &bv) in p.boosts.iter().enumerate() {
                f[F_BOOSTS + k] = ((bv as f64 / 6.0).clamp(-1.0, 1.0)) as f32;
            }
            f[F_REVEALED] = p.revealed as u8 as f32;
            f[F_TERA] = p.tera as u8 as f32;
        }
    }
    let mut fld = vec![0f32; FIELD];
    if let Some(w) = b.weather {
        fld[w] = 1.0;
    }
    if let Some(t) = b.terrain {
        fld[5 + t] = 1.0;
    }
    fld[9] = (b.turn as f64 / 50.0).min(1.0) as f32;
    for (off, s) in [(10, &b.sides[side]), (15, &b.sides[other])] {
        let h = |k: &str| *s.hazards.get(k).unwrap_or(&0);
        fld[off] = (h("spikes").min(3) as f64 / 3.0) as f32;
        fld[off + 1] = (h("toxicspikes").min(2) as f64 / 2.0) as f32;
        fld[off + 2] = (h("stealthrock") > 0) as u8 as f32;
        fld[off + 3] = (h("stickyweb") > 0) as u8 as f32;
        fld[off + 4] = (SCREENS.iter().filter(|&&sc| h(sc) > 0).count().min(3) as f64 / 3.0) as f32;
    }
    let own = &b.sides[side];
    let mut moves = vec![String::new(); MOVES];
    if let Some(a) = own.active.filter(|&a| a < own.team.len()) {
        for (k, m) in own.team[a].moves.iter().take(MOVES).enumerate() {
            moves[k] = m.clone();
        }
    }
    let mut species: Vec<String> = own.team.iter().map(|p| p.species.clone()).collect();
    species.resize(TEAM, String::new());
    let mut os: Vec<String> = b.sides[other].team.iter().map(|p| p.species.clone()).collect();
    os.resize(TEAM, String::new());
    species.extend(os);
    Frame { species, moves, feats, mask, field: fld, active: own.active.unwrap_or(0) as i64 }
}

const IGNORED: [&str; 34] = ["", "j", "l", "c", "c:", "chat", "raw", "html", "uhtml", "inactive", "inactiveoff", "t:", "gametype", "gen",
    "tier", "rated", "rule", "clearpoke", "teampreview", "start", "upkeep", "teamsize", "player", "-message", "message", "debug",
    "error", "seed", "choice", "unlink", "-hint", "-anim", "n", "join"];

struct Parse {
    records: [Vec<Rec>; 2],
    winner: Option<usize>,
    names: [String; 2],
    skipped: BTreeMap<String, usize>,
    unknown: BTreeMap<String, usize>,
}

fn parse_log(log: &str) -> Parse {
    let mut b = Battle { sides: Default::default(), weather: None, terrain: None, turn: 0, winner: None };
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut records: [Vec<Rec>; 2] = [Vec::new(), Vec::new()];
    let mut frames: [Option<Frame>; 2] = [None, None];
    let mut actions: [Option<Rec>; 2] = [None, None];
    let mut consumed = [false; 2];
    let mut forced_pivot = [false; 2];
    let mut tera_pending = [false; 2];
    let mut active_fainted = [false; 2];
    macro_rules! skip {
        ($r:expr) => {
            *skipped.entry($r.to_string()).or_default() += 1
        };
    }
    macro_rules! finalize {
        () => {
            for side in 0..2 {
                if frames[side].is_none() {
                    continue;
                }
                match actions[side].take() {
                    Some(r) => records[side].push(r),
                    None => skip!("no_visible_action"),
                }
            }
            for side in 0..2 {
                frames[side] = None;
                actions[side] = None;
                consumed[side] = false;
                tera_pending[side] = false;
                active_fainted[side] = false;
                forced_pivot[side] = false;
            }
        };
    }
    for raw in log.split('\n') {
        if !raw.starts_with('|') {
            continue;
        }
        let parts: Vec<&str> = raw.split('|').collect();
        let cmd = parts.get(1).copied().unwrap_or("");
        let args: &[&str] = if parts.len() > 2 { &parts[2..] } else { &[] };
        let a = |i: usize| args.get(i).copied().unwrap_or("");
        match cmd {
            "player" if args.len() >= 2 && side_idx(a(0)).is_some() => {
                if !a(1).is_empty() {
                    b.sides[side_idx(a(0)).unwrap()].name = a(1).to_string();
                }
                continue;
            }
            "poke" if args.len() >= 2 => {
                if let Some(si) = side_idx(a(0)) {
                    if b.sides[si].team.len() < TEAM {
                        b.sides[si].team.push(Mon::new(base_species(a(1))));
                    }
                }
                continue;
            }
            "switch" | "drag" | "replace" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let slot = b.sides[pi].slot_for(&nick, a(1));
                let is_choice = cmd == "switch" && b.turn > 0 && actions[pi].is_none() && !consumed[pi] && !active_fainted[pi]
                    && !forced_pivot[pi] && frames[pi].is_some();
                if cmd == "switch" && b.turn > 0 && !is_choice {
                    if forced_pivot[pi] {
                        forced_pivot[pi] = false;
                        consumed[pi] = true;
                        skip!("forced_pivot_switch");
                    } else if active_fainted[pi] {
                        skip!("forced_replacement_switch");
                    } else if actions[pi].is_some() || consumed[pi] {
                        skip!("mid_turn_switch");
                    }
                }
                let s = &mut b.sides[pi];
                if let Some(prev) = s.active.filter(|&x| x < s.team.len()) {
                    s.team[prev].boosts = [0; 5];
                }
                s.active = Some(slot);
                let p = &mut s.team[slot];
                p.revealed = true;
                p.boosts = [0; 5];
                if args.len() >= 3 && !a(2).is_empty() {
                    let (hp, st, fnt) = parse_hp(a(2));
                    p.hp = hp;
                    p.fainted = fnt;
                    if let Some(st) = st {
                        p.status = st;
                    }
                }
                if is_choice {
                    actions[pi] = Some(Rec { frame: frames[pi].clone().unwrap(), action_type: 1, move_idx: -1, switch_idx: slot as i64, tera: 0 });
                }
                continue;
            }
            "move" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if args[2..].iter().any(|x| x.contains("[from]")) {
                    continue;
                }
                let Some(slot) = b.sides[pi].slot_of(&nick) else { continue };
                let mv = a(1);
                if norm(mv) == "struggle" {
                    if actions[pi].is_none() && !consumed[pi] {
                        consumed[pi] = true;
                        skip!("struggle");
                    }
                    continue;
                }
                let p = &mut b.sides[pi].team[slot];
                let nm: Vec<String> = p.moves.iter().map(|m| norm(m)).collect();
                let idx = match nm.iter().position(|m| *m == norm(mv)) {
                    Some(i) => i,
                    None => {
                        p.moves.push(mv.to_string());
                        p.moves.len() - 1
                    }
                };
                if actions[pi].is_none() && !consumed[pi] && frames[pi].is_some() {
                    actions[pi] = Some(Rec {
                        frame: frames[pi].clone().unwrap(),
                        action_type: 0,
                        move_idx: if idx < MOVES { idx as i64 } else { -1 },
                        switch_idx: -1,
                        tera: tera_pending[pi] as i64,
                    });
                }
                continue;
            }
            "cant" if !args.is_empty() => {
                let (pos, _) = ident(a(0));
                if let Some(pi) = side_idx(&pos) {
                    if actions[pi].is_none() {
                        consumed[pi] = true;
                        skip!("cant");
                    }
                }
                continue;
            }
            "-activate" | "-enditem" if args.len() >= 2 => {
                let (pos, _) = ident(a(0));
                if let Some(pi) = side_idx(&pos) {
                    if matches!(a(1), "ability: Emergency Exit" | "ability: Wimp Out" | "Eject Button" | "Eject Pack") {
                        forced_pivot[pi] = true;
                    }
                }
                continue;
            }
            "-damage" | "-heal" | "-sethp" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let Some(slot) = b.sides[pi].slot_of(&nick) else { continue };
                let p = &mut b.sides[pi].team[slot];
                let (hp, st, fnt) = parse_hp(a(1));
                p.hp = hp;
                if fnt {
                    p.fainted = true;
                }
                if let Some(st) = st {
                    p.status = st;
                }
                continue;
            }
            "-status" if args.len() >= 2 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].status = status_idx(a(1)).unwrap_or(0);
                }
                continue;
            }
            "-curestatus" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].status = 0;
                }
                continue;
            }
            "-boost" | "-unboost" | "-setboost" if args.len() >= 3 => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let Some(k) = BOOSTS.iter().position(|&x| x == a(1)) else { continue };
                let Some(slot) = b.sides[pi].slot_of(&nick) else { continue };
                let Some(amt) = py_int(a(2)) else { continue };
                let bv = &mut b.sides[pi].team[slot].boosts[k];
                *bv = match cmd {
                    "-boost" => (*bv + amt).clamp(-6, 6),
                    "-unboost" => (*bv - amt).clamp(-6, 6),
                    _ => amt.clamp(-6, 6),
                };
                continue;
            }
            "-clearboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].boosts = [0; 5];
                }
                continue;
            }
            "-clearnegativeboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    for x in b.sides[pi].team[slot].boosts.iter_mut() {
                        *x = (*x).max(0);
                    }
                }
                continue;
            }
            "-clearallboost" => {
                for s in b.sides.iter_mut() {
                    if let Some(act) = s.active.filter(|&x| x < s.team.len()) {
                        s.team[act].boosts = [0; 5];
                    }
                }
                continue;
            }
            "-copyboost" | "-invertboost" | "-swapboost" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    let bs = &mut b.sides[pi].team[slot].boosts;
                    if cmd == "-invertboost" {
                        for x in bs.iter_mut() {
                            *x = -*x;
                        }
                    } else {
                        *bs = [0; 5];
                    }
                }
                continue;
            }
            "faint" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    let p = &mut b.sides[pi].team[slot];
                    p.fainted = true;
                    p.hp = 0.0;
                    p.boosts = [0; 5];
                    if b.sides[pi].active == Some(slot) {
                        active_fainted[pi] = true;
                    }
                }
                continue;
            }
            "-weather" if !args.is_empty() => {
                let w = norm(a(0));
                b.weather = if w == "none" || w.is_empty() { None } else { Some(weather_idx(&w)) };
                continue;
            }
            "-fieldstart" if !args.is_empty() => {
                if let Some(t) = terrain_idx(&norm(a(0).rsplit(':').next().unwrap_or(""))) {
                    b.terrain = Some(t);
                }
                continue;
            }
            "-fieldend" if !args.is_empty() => {
                if let Some(t) = terrain_idx(&norm(a(0).rsplit(':').next().unwrap_or(""))) {
                    if b.terrain == Some(t) {
                        b.terrain = None;
                    }
                }
                continue;
            }
            "-sidestart" | "-sideend" if args.len() >= 2 => {
                let (pos, _) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                let name = norm(a(1).rsplit(':').next().unwrap_or(""));
                let known = matches!(name.as_str(), "spikes" | "toxicspikes" | "stealthrock" | "stickyweb") || SCREENS.contains(&name.as_str());
                if known {
                    let h = b.sides[pi].hazards.entry(name.clone()).or_insert(0);
                    if cmd == "-sidestart" {
                        let cap = match name.as_str() {
                            "spikes" => 3,
                            "toxicspikes" => 2,
                            _ => 1,
                        };
                        *h = (*h + 1).min(cap);
                    } else {
                        *h = 0;
                    }
                }
                continue;
            }
            "-terastallize" if !args.is_empty() => {
                let (pos, nick) = ident(a(0));
                let Some(pi) = side_idx(&pos) else { continue };
                if let Some(slot) = b.sides[pi].slot_of(&nick) {
                    b.sides[pi].team[slot].tera = true;
                }
                tera_pending[pi] = true;
                continue;
            }
            "turn" if !args.is_empty() => {
                let Some(n) = py_int(a(0)) else { continue };
                finalize!();
                b.turn = n;
                frames = [Some(encode(&b, 0)), Some(encode(&b, 1))];
                continue;
            }
            "win" if !args.is_empty() => {
                finalize!();
                let name = a(0).trim();
                for (i, s) in b.sides.iter().enumerate() {
                    if s.name == name {
                        b.winner = Some(i);
                    }
                }
                continue;
            }
            "tie" => {
                finalize!();
                continue;
            }
            _ => {}
        }
        if IGNORED.contains(&cmd) || cmd == "leave" || cmd == "showteam" || cmd.starts_with("c:") {
            continue;
        }
        *unknown.entry(cmd.to_string()).or_default() += 1;
    }
    finalize!();
    Parse { records, winner: b.winner, names: [b.sides[0].name.clone(), b.sides[1].name.clone()], skipped, unknown }
}

fn replay_samples(r: &Value, sb: u32, mb: u32, p: &Parse) -> Vec<Sample> {
    let mut out = Vec::new();
    let Some(w) = p.winner else { return out };
    for side in 0..2 {
        let recs = &p.records[side];
        if recs.is_empty() {
            continue;
        }
        let t = recs.len();
        let (mut sp, mut mv, mut feats, mut mask, mut field) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut at, mut mi, mut si, mut te, mut ai) = (vec![], vec![], vec![], vec![], vec![]);
        for rec in recs {
            let f = &rec.frame;
            sp.extend(f.species.iter().map(|s| bucket(s, sb)));
            mv.extend(f.moves.iter().map(|m| bucket(m, mb)));
            feats.extend_from_slice(&f.feats);
            mask.extend_from_slice(&f.mask);
            field.extend_from_slice(&f.field);
            ai.push(f.active);
            at.push(rec.action_type);
            mi.push(rec.move_idx);
            si.push(rec.switch_idx);
            te.push(rec.tera);
        }
        let v = if w == side { 1.0 } else { -1.0 };
        let fmt = r.get("formatid").or_else(|| r.get("format")).cloned().unwrap_or_else(|| json!("?"));
        out.push(Sample::new(json!({"replay_id": r.get("id").cloned().unwrap_or(json!("?")), "side": (if side == 0 { "p1" } else { "p2" }),
                                     "player": p.names[side], "rating": r.get("rating").cloned().unwrap_or(Value::Null),
                                     "format": fmt, "won": w == side}))
            .i64("obs.species_ids", &[2 * TEAM], sp)
            .i64("obs.move_ids", &[MOVES], mv)
            .f32("obs.poke_feats", &[2 * TEAM, FEAT], feats)
            .bool("obs.slot_mask", &[2 * TEAM], mask)
            .f32("obs.field_seq", &[FIELD], field)
            .i64("tgt.action_type", &[], at)
            .i64("tgt.move_idx", &[], mi)
            .i64("tgt.switch_idx", &[], si)
            .i64("tgt.tera", &[], te)
            .i64("tgt.active_idx", &[], ai)
            .f32("tgt.value", &[], vec![v; t]));
    }
    out
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download public replays of a format via the replay search API (resumable, polite)
    Fetch {
        #[arg(long, default_value = "gen9ou")]
        format: String,
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        #[arg(long, default_value_t = 1000)]
        max_replays: usize,
        #[arg(long, default_value_t = 1000)]
        sleep_ms: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Convert replay .json files to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        shard_size: usize,
        #[arg(long, default_value_t = 0)]
        min_rating: i64,
        #[arg(long)]
        max_replays: Option<usize>,
        #[arg(long, default_value_t = 512)]
        species_buckets: u32,
        #[arg(long, default_value_t = 256)]
        move_buckets: u32,
    },
}

const BASE: &str = "https://replay.pokemonshowdown.com";

fn fetch(format: &str, min_rating: i64, max: usize, sleep_ms: u64, out: &Path) -> Result<Value> {
    let client = common::http()?;
    std::fs::create_dir_all(out)?;
    let (mut downloaded, mut listed, mut low, mut existing, mut errors) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut before: Option<i64> = None;
    let pause = || std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
    'pages: while downloaded < max {
        let mut url = format!("{BASE}/search.json?format={format}");
        if let Some(b) = before {
            url.push_str(&format!("&before={b}"));
        }
        let page: Vec<Value> = match client.get(&url).send().and_then(|r| r.error_for_status()).and_then(|r| r.json()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[fetch] search page failed ({e}); retrying in 10s");
                std::thread::sleep(std::time::Duration::from_secs(10));
                continue;
            }
        };
        if page.is_empty() {
            break;
        }
        let has_more = page.len() >= 51;
        before = page.last().and_then(|x| x["uploadtime"].as_i64());
        for item in page.iter().take(if has_more { 50 } else { page.len() }) {
            if downloaded >= max {
                break 'pages;
            }
            listed += 1;
            let Some(id) = item["id"].as_str() else { continue };
            if min_rating > 0 && item["rating"].as_i64().unwrap_or(0) < min_rating {
                low += 1;
                continue;
            }
            let dest = out.join(format!("{id}.json"));
            if dest.exists() {
                existing += 1;
                continue;
            }
            pause();
            match client.get(format!("{BASE}/{id}.json")).send().and_then(|r| r.error_for_status()).and_then(|r| r.bytes()) {
                Ok(bts) => {
                    let tmp = dest.with_extension("tmp");
                    std::fs::write(&tmp, &bts)?;
                    std::fs::rename(&tmp, &dest)?;
                    downloaded += 1;
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("[fetch] {id} failed: {e}");
                }
            }
        }
        if !has_more || before.is_none() {
            break;
        }
        pause();
    }
    Ok(json!({"listed": listed, "downloaded": downloaded, "skipped_rating": low, "skipped_existing": existing, "errors": errors}))
}

#[allow(clippy::too_many_arguments)]
pub fn convert(input: &Path, out: &Path, shard_size: usize, min_rating: i64, max_replays: Option<usize>, sb: u32, mb: u32) -> Result<Value> {
    let mut w = ShardWriter::new(out, "pokemon", 1, json!({"move_buckets": mb, "species_buckets": sb}), shard_size)?;
    let mut st: BTreeMap<&str, usize> = ["seen", "kept", "samples", "frames", "no_winner", "low_rating", "not_replay"].into_iter().map(|k| (k, 0)).collect();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().map(|e| e == "json").unwrap_or(false)).collect();
    files.sort();
    for path in files {
        if max_replays.map(|m| st["kept"] >= m).unwrap_or(false) {
            break;
        }
        let r: Value = match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(v @ Value::Object(_)) if v.get("log").is_some() => v,
            _ => {
                *st.get_mut("not_replay").unwrap() += 1;
                continue;
            }
        };
        *st.get_mut("seen").unwrap() += 1;
        if min_rating > 0 && r["rating"].as_f64().unwrap_or(0.0) < min_rating as f64 {
            *st.get_mut("low_rating").unwrap() += 1;
            continue;
        }
        let p = parse_log(r["log"].as_str().unwrap_or(""));
        for (k, v) in &p.skipped {
            *skipped.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &p.unknown {
            *unknown.entry(k.clone()).or_default() += v;
        }
        if p.winner.is_none() {
            *st.get_mut("no_winner").unwrap() += 1;
            continue;
        }
        let ss = replay_samples(&r, sb, mb, &p);
        if ss.is_empty() {
            continue;
        }
        *st.get_mut("kept").unwrap() += 1;
        for s in ss {
            *st.get_mut("samples").unwrap() += 1;
            *st.get_mut("frames").unwrap() += s.frames()?;
            w.push(s)?;
        }
    }
    w.flush()?;
    let mut v = json!(st);
    v["shards"] = json!(w.written_shards);
    v["skipped"] = json!(skipped);
    v["unknown"] = json!(unknown);
    Ok(v)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { format, min_rating, max_replays, sleep_ms, out } => {
            println!("{}", fetch(&format, min_rating, max_replays, sleep_ms, &out)?);
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, min_rating, max_replays, species_buckets, move_buckets } => {
            println!("{}", convert(&input, &out, shard_size, min_rating, max_replays, species_buckets, move_buckets)?);
            Ok(())
        }
    }
}
