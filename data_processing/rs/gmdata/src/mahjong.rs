//! Riichi mahjong: mjai event streams (Tenhou logs) → shards for the mahjong adapter.
//!
//! One sample per (game, seat): that seat's decision-point frames over the game.
//! Own-turn frames after each own draw (label: dahai / riichi / kan / hora /
//! ryukyoku by lookahead) and after a chi/pon (forced discard); call-opportunity
//! frames after another player's discard when the seat called on it or was
//! cheaply call-eligible (pon/kan: ≥2 of the kind; chi: shimocha with a partner
//! pair) and passed. State: hands, reds, melds, ponds, riichi, dora, scores, wall.
//!
//!   obs.tiles_seq   [T,34,16] f32   obs.players_seq [T,4,12] f32   obs.table_seq [T,12] f32
//!   tgt.action_type [T] i64         tgt.tile_idx [T] i64           tgt.value [T] f32
//!   tgt.legal_action_type [T,8]     tgt.legal_tile_of [T,34]       (packed bits → f32 {0,1})
//! value = final placement 1st +1, 2nd +1/3, 3rd −1/3, 4th −1 (ties → lower seat).
//!
//! Legality (converter_version 2): `legal_action_type` = the action types legal at the
//! frame, `legal_tile_of` = the legal tile arguments of the GROUND-TRUTH action type
//! (all true when that type takes no tile: hora / ryukyoku / pass).
//!   own draw      dahai always (in riichi: only the drawn kind); riichi when closed, not
//!                 yet in riichi, score ≥ 1000, wall ≥ 4 and some discard leaves tenpai
//!                 (tiles = those discards); kan = ankan (4 in hand) / kakan (own pon + 1
//!                 in hand) kinds while the wall is not empty; hora = the concealed tiles
//!                 plus the fixed melds form a winning shape; ryukyoku = kyushu kyuhai
//!                 (first own discard not made, no call yet, ≥ 9 distinct terminal/honor
//!                 kinds).
//!   after chi/pon dahai only, minus kuikae (the called kind; for chi also the suji
//!                 kind at the other end of the run).
//!   claim         pass always; chi (shimocha: run bases with both partners in hand),
//!                 pon (≥ 2), kan (≥ 3) when not in riichi and the wall is not empty;
//!                 hora = concealed + the discard form a winning shape.
//! Exact for dahai / riichi (tenpai) / chi / pon / kan / pass; hora and ryukyoku are
//! supersets (yaku, furiten and kan limits are not checked). A frame whose ground truth
//! falls outside its masks keeps all-true masks and is counted as `mask_mismatch`.
//!
//! Input: a .zip of .mjson(.gz) files, a directory of .mjson(.gz), one file, or a
//! Parquet file/directory with an `events` column (one mjai game per row).

use crate::common;
use anyhow::{Context, Result};
use arrow_array::{Array, LargeStringArray, StringArray};
use clap::Subcommand;
use gmshard::{Sample, ShardWriter};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

const DAHAI: i64 = 0;
const RIICHI: i64 = 1;
const CHI: i64 = 2;
const PON: i64 = 3;
const KAN: i64 = 4;
const HORA: i64 = 5;
const RYUKYOKU: i64 = 6;
const PASS: i64 = 7;
const ACTION_NAMES: [&str; 8] = ["dahai", "riichi", "chi", "pon", "kan", "hora", "ryukyoku", "pass"];
/// Whether an action type takes a tile argument (the decoder's ARG_USES table).
const USES_TILE: [bool; 8] = [true, true, true, true, true, false, false, false];
const N_TYPES: usize = 8;
const CONVERTER_VERSION: u32 = 2;
const NK: usize = 34;
const WALL: i64 = 70;
const PLACEMENT: [f32; 4] = [1.0, (1.0f64 / 3.0) as f32, (-1.0f64 / 3.0) as f32, -1.0];

struct Skip;
type R<T> = std::result::Result<T, Skip>;

fn tile_kind(p: &str) -> R<usize> {
    let b = p.as_bytes();
    let honors = ["E", "S", "W", "N", "P", "F", "C"];
    if let Some(i) = honors.iter().position(|&h| h == p) {
        return Ok(27 + i);
    }
    let suit = |c: u8| match c {
        b'm' => Some(0),
        b'p' => Some(1),
        b's' => Some(2),
        _ => None,
    };
    match b.len() {
        2 => {
            let s = suit(b[1]).ok_or(Skip)?;
            match b[0] {
                b'1'..=b'9' => Ok(s * 9 + (b[0] - b'1') as usize),
                b'0' => Ok(s * 9 + 4),
                _ => Err(Skip),
            }
        }
        3 if b[0] == b'5' && b[2] == b'r' => Ok(suit(b[1]).ok_or(Skip)? * 9 + 4),
        _ => Err(Skip),
    }
}

fn is_red(p: &str) -> bool {
    p.ends_with('r') || p.starts_with('0')
}

fn dora_from_marker(k: usize) -> usize {
    if k < 27 {
        let base = (k / 9) * 9;
        base + (k - base + 1) % 9
    } else if k < 31 {
        27 + (k - 27 + 1) % 4
    } else {
        31 + (k - 31 + 1) % 3
    }
}

fn is_terminal_or_honor(k: usize) -> bool {
    k >= 27 || k % 9 == 0 || k % 9 == 8
}

/// Remove sets (triplets, runs inside one suit) from `c` until empty: the lowest
/// remaining kind must start a triplet or a run, so the search only branches there.
fn only_sets(c: &mut [i64; NK]) -> bool {
    let Some(i) = c.iter().position(|&x| x > 0) else { return true };
    if c[i] >= 3 {
        c[i] -= 3;
        let ok = only_sets(c);
        c[i] += 3;
        if ok {
            return true;
        }
    }
    if i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        for d in 0..3 {
            c[i + d] -= 1;
        }
        let ok = only_sets(c);
        for d in 0..3 {
            c[i + d] += 1;
        }
        if ok {
            return true;
        }
    }
    false
}

/// `c` = `sets` sets + one pair.
fn regular_agari(c: &[i64; NK], sets: i64) -> bool {
    if sets < 0 || c.iter().sum::<i64>() != 3 * sets + 2 {
        return false;
    }
    let mut w = *c;
    for p in 0..NK {
        if w[p] >= 2 {
            w[p] -= 2;
            if only_sets(&mut w) {
                return true;
            }
            w[p] += 2;
        }
    }
    false
}

/// Seven distinct pairs (a quad is not two pairs).
fn chiitoitsu(c: &[i64; NK]) -> bool {
    c.iter().all(|&x| x == 0 || x == 2) && c.iter().filter(|&&x| x == 2).count() == 7
}

/// All 13 terminal/honor kinds, one of them paired, nothing else.
fn kokushi(c: &[i64; NK]) -> bool {
    (0..NK).all(|k| if is_terminal_or_honor(k) { c[k] >= 1 } else { c[k] == 0 }) && c.iter().sum::<i64>() == 14
}

/// Winning shape of the concealed tiles `c` next to `meld_n` fixed melds (yaku not checked).
fn is_agari(c: &[i64; NK], meld_n: i64) -> bool {
    if c.iter().any(|&x| x < 0) || c.iter().sum::<i64>() != 14 - 3 * meld_n {
        return false;
    }
    regular_agari(c, 4 - meld_n) || (meld_n == 0 && (chiitoitsu(c) || kokushi(c)))
}

/// Some tile completes the 13-tile-equivalent concealed hand `c` (any kind, so a wait
/// on kinds the hand already holds four of — karaten — still counts: a superset).
fn is_tenpai(c: &[i64; NK], meld_n: i64) -> bool {
    let mut w = *c;
    (0..NK).any(|k| {
        w[k] += 1;
        let ok = is_agari(&w, meld_n);
        w[k] -= 1;
        ok
    })
}

/// Decision context of a frame (what just happened to the acting seat).
#[derive(Clone, Debug)]
enum Ctx {
    /// own draw of kind `drawn`
    Draw { drawn: usize },
    /// forced discard after a chi/pon of `called` with the `consumed` hand kinds
    AfterCall { called: usize, consumed: Vec<usize>, is_chi: bool },
    /// another seat's discard `kind` by `discarder`
    Claim { kind: usize, discarder: usize },
}

/// Legal action types and, per type, legal tile arguments.
struct Legal {
    types: [bool; N_TYPES],
    tiles: [[bool; NK]; N_TYPES],
}

impl Legal {
    fn none() -> Self {
        Legal { types: [false; N_TYPES], tiles: [[false; NK]; N_TYPES] }
    }
    fn allow(&mut self, t: i64, tiles: [bool; NK]) {
        self.types[t as usize] = true;
        self.tiles[t as usize] = tiles;
    }
    /// (type mask, tile mask of type `t`, whether (t, tile) is inside them)
    fn frame(&self, t: i64, tile: i64) -> ([bool; N_TYPES], [bool; NK], bool) {
        let ti = t as usize;
        let row = if USES_TILE[ti] { self.tiles[ti] } else { [true; NK] };
        let ok = self.types[ti] && (!USES_TILE[ti] || (0..NK as i64).contains(&tile) && row[tile as usize]);
        (self.types, row, ok)
    }
}

fn s<'a>(ev: &'a Value, k: &str) -> R<&'a str> {
    ev.get(k).and_then(|v| v.as_str()).ok_or(Skip)
}
fn n(ev: &Value, k: &str) -> R<i64> {
    ev.get(k).and_then(|v| v.as_i64()).ok_or(Skip)
}
fn seat(ev: &Value, k: &str) -> R<usize> {
    let v = n(ev, k)?;
    if (0..4).contains(&v) {
        Ok(v as usize)
    } else {
        Err(Skip)
    }
}

struct Kyoku {
    oya: usize,
    bakaze: String,
    kyoku: i64,
    honba: i64,
    kyotaku: i64,
    scores: [i64; 4],
    dora: Vec<usize>,
    hands: [[i64; NK]; 4],
    reds: [HashSet<usize>; 4],
    melds: [[i64; NK]; 4],
    meld_n: [i64; 4],
    kan_n: [i64; 4],
    disc: [[i64; NK]; 4],
    disc_n: [i64; 4],
    riichi: [bool; 4],
    pending: [bool; 4],
    wall: i64,
    /// called chi / pon / daiminkan (an ankan keeps the hand closed)
    open: [bool; 4],
    /// kinds ponned (kakan candidates)
    pon_kinds: [[bool; NK]; 4],
    /// any chi / pon / kan this kyoku (ends the kyushu kyuhai window)
    any_call: bool,
}

impl Kyoku {
    fn new(ev: &Value) -> R<Self> {
        let scores_v = ev.get("scores").and_then(|v| v.as_array()).ok_or(Skip)?;
        let mut scores = [0i64; 4];
        for i in 0..4 {
            scores[i] = scores_v.get(i).and_then(|v| v.as_i64()).ok_or(Skip)?;
        }
        let mut k = Kyoku {
            oya: seat(ev, "oya")?,
            bakaze: s(ev, "bakaze")?.to_string(),
            kyoku: ev.get("kyoku").and_then(|v| v.as_i64()).unwrap_or(1),
            honba: ev.get("honba").and_then(|v| v.as_i64()).unwrap_or(0),
            kyotaku: ev.get("kyotaku").and_then(|v| v.as_i64()).unwrap_or(0),
            scores,
            dora: vec![tile_kind(s(ev, "dora_marker")?)?],
            hands: [[0; NK]; 4],
            reds: Default::default(),
            melds: [[0; NK]; 4],
            meld_n: [0; 4],
            kan_n: [0; 4],
            disc: [[0; NK]; 4],
            disc_n: [0; 4],
            riichi: [false; 4],
            pending: [false; 4],
            wall: WALL,
            open: [false; 4],
            pon_kinds: [[false; NK]; 4],
            any_call: false,
        };
        let tehais = ev.get("tehais").and_then(|v| v.as_array()).ok_or(Skip)?;
        for (si, t) in tehais.iter().enumerate() {
            let t = t.as_array().ok_or(Skip)?;
            if t.len() != 13 || si >= 4 {
                return Err(Skip);
            }
            for p in t {
                let p = p.as_str().ok_or(Skip)?;
                if p == "?" {
                    return Err(Skip);
                }
                k.add(si, p)?;
            }
        }
        Ok(k)
    }
    fn add(&mut self, s: usize, p: &str) -> R<()> {
        let k = tile_kind(p)?;
        self.hands[s][k] += 1;
        if is_red(p) {
            self.reds[s].insert(k);
        }
        Ok(())
    }
    fn remove(&mut self, s: usize, p: &str) -> R<usize> {
        let k = tile_kind(p)?;
        if self.hands[s][k] <= 0 {
            return Err(Skip);
        }
        self.hands[s][k] -= 1;
        if is_red(p) || self.hands[s][k] == 0 {
            self.reds[s].remove(&k);
        }
        Ok(k)
    }
    fn draw(&mut self, s: usize, p: &str) -> R<usize> {
        self.wall -= 1;
        self.add(s, p)?;
        tile_kind(p)
    }
    fn discard(&mut self, s: usize, p: &str) -> R<usize> {
        let k = self.remove(s, p)?;
        self.disc[s][k] += 1;
        self.disc_n[s] += 1;
        if self.pending[s] {
            self.pending[s] = false;
            self.riichi[s] = true;
        }
        Ok(k)
    }
    fn consumed(ev: &Value) -> R<Vec<String>> {
        ev.get("consumed").and_then(|v| v.as_array()).ok_or(Skip)?.iter().map(|x| x.as_str().map(String::from).ok_or(Skip)).collect()
    }
    fn call(&mut self, ev: &Value) -> R<()> {
        let (c, t) = (seat(ev, "actor")?, seat(ev, "target")?);
        let k = tile_kind(s(ev, "pai")?)?;
        if self.disc[t][k] <= 0 {
            return Err(Skip);
        }
        self.disc[t][k] -= 1;
        self.disc_n[t] -= 1;
        self.melds[c][k] += 1;
        for p in Self::consumed(ev)? {
            let kk = self.remove(c, &p)?;
            self.melds[c][kk] += 1;
        }
        self.meld_n[c] += 1;
        match s(ev, "type")? {
            "daiminkan" => self.kan_n[c] += 1,
            "pon" => self.pon_kinds[c][k] = true,
            _ => {}
        }
        self.open[c] = true;
        self.any_call = true;
        Ok(())
    }
    fn ankan(&mut self, ev: &Value) -> R<()> {
        let c = seat(ev, "actor")?;
        for p in Self::consumed(ev)? {
            let kk = self.remove(c, &p)?;
            self.melds[c][kk] += 1;
        }
        self.meld_n[c] += 1;
        self.kan_n[c] += 1;
        self.any_call = true;
        Ok(())
    }
    fn kakan(&mut self, ev: &Value) -> R<()> {
        let c = seat(ev, "actor")?;
        let kk = self.remove(c, s(ev, "pai")?)?;
        self.melds[c][kk] += 1;
        self.kan_n[c] += 1;
        self.pon_kinds[c][kk] = false;
        self.any_call = true;
        Ok(())
    }
    fn can_pon(&self, s: usize, k: usize) -> bool {
        self.hands[s][k] >= 2
    }
    fn can_chi(&self, s: usize, discarder: usize, k: usize) -> bool {
        if s != (discarder + 1) % 4 || k >= 27 {
            return false;
        }
        let (r, base) = ((k % 9) as i64, k - k % 9);
        let has = |x: i64| self.hands[s][base + x as usize] > 0;
        (r >= 2 && has(r - 2) && has(r - 1)) || (r > 0 && r < 8 && has(r - 1) && has(r + 1)) || (r <= 6 && has(r + 1) && has(r + 2))
    }

    /// Legal action types / tile arguments of `seat` in context `ctx` (see the module doc).
    fn legal(&self, seat: usize, ctx: &Ctx) -> Legal {
        let h = &self.hands[seat];
        let meld_n = self.meld_n[seat];
        let in_hand: [bool; NK] = std::array::from_fn(|k| h[k] > 0);
        let mut l = Legal::none();
        match ctx {
            Ctx::Draw { drawn } => {
                if self.riichi[seat] {
                    l.allow(DAHAI, std::array::from_fn(|k| k == *drawn && h[k] > 0));
                } else {
                    l.allow(DAHAI, in_hand);
                }
                if !self.riichi[seat] && !self.open[seat] && self.scores[seat] >= 1000 && self.wall >= 4 {
                    let mut w = *h;
                    let tenpai_discards: [bool; NK] = std::array::from_fn(|d| {
                        if w[d] == 0 {
                            return false;
                        }
                        w[d] -= 1;
                        let ok = is_tenpai(&w, meld_n);
                        w[d] += 1;
                        ok
                    });
                    if tenpai_discards.iter().any(|&x| x) {
                        l.allow(RIICHI, tenpai_discards);
                    }
                }
                if self.wall > 0 {
                    let kans: [bool; NK] = std::array::from_fn(|k| h[k] == 4 || (self.pon_kinds[seat][k] && h[k] >= 1));
                    if kans.iter().any(|&x| x) {
                        l.allow(KAN, kans);
                    }
                }
                if is_agari(h, meld_n) {
                    l.allow(HORA, [true; NK]);
                }
                let yaochu = (0..NK).filter(|&k| is_terminal_or_honor(k) && h[k] > 0).count();
                if self.disc_n[seat] == 0 && !self.any_call && yaochu >= 9 {
                    l.allow(RYUKYOKU, [true; NK]);
                }
            }
            Ctx::AfterCall { called, consumed, is_chi } => {
                let mut tiles = in_hand;
                tiles[*called] = false;
                if *is_chi {
                    let r = consumed.iter().copied().chain(std::iter::once(*called)).min().unwrap_or(*called);
                    if *called == r && r % 9 + 3 <= 8 {
                        tiles[r + 3] = false;
                    } else if *called == r + 2 && r % 9 >= 1 {
                        tiles[r - 1] = false;
                    }
                }
                l.allow(DAHAI, tiles);
            }
            Ctx::Claim { kind, discarder } => {
                let c = *kind;
                l.allow(PASS, [true; NK]);
                if !self.riichi[seat] && self.wall > 0 {
                    if self.can_chi(seat, *discarder, c) {
                        let bases: [bool; NK] = std::array::from_fn(|b| {
                            b < 27 && b / 9 == c / 9 && b % 9 <= 6 && b <= c && c <= b + 2
                                && (b..b + 3).filter(|&x| x != c).all(|x| h[x] > 0)
                        });
                        if bases.iter().any(|&x| x) {
                            l.allow(CHI, bases);
                        }
                    }
                    let only_c: [bool; NK] = std::array::from_fn(|k| k == c);
                    if h[c] >= 2 {
                        l.allow(PON, only_c);
                    }
                    if h[c] >= 3 {
                        l.allow(KAN, only_c);
                    }
                }
                let mut w = *h;
                w[c] += 1;
                if is_agari(&w, meld_n) {
                    l.allow(HORA, [true; NK]);
                }
            }
        }
        l
    }

    fn encode(&self, seat: usize, focus: Option<usize>, own_turn: bool, focus_actor: Option<usize>, out: &mut Frames) {
        let (shimo, toi, kami) = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4);
        let dora_kinds: HashSet<usize> = self.dora.iter().map(|&m| dora_from_marker(m)).collect();
        let mut marker = [0i64; NK];
        for &m in &self.dora {
            marker[m] += 1;
        }
        for k in 0..NK {
            let mut row = [0f64; 16];
            row[self.hands[seat][k].min(4) as usize] = 1.0;
            row[5] = self.melds[seat][k] as f64 / 4.0;
            row[6] = (self.melds[shimo][k] + self.melds[toi][k] + self.melds[kami][k]) as f64 / 4.0;
            row[7] = self.disc[seat][k] as f64 / 4.0;
            row[8] = self.disc[shimo][k] as f64 / 4.0;
            row[9] = self.disc[toi][k] as f64 / 4.0;
            row[10] = self.disc[kami][k] as f64 / 4.0;
            row[11] = dora_kinds.contains(&k) as u8 as f64;
            let visible = self.hands[seat][k] + marker[k] + (0..4).map(|p| self.melds[p][k] + self.disc[p][k]).sum::<i64>();
            row[12] = visible.min(4) as f64 / 4.0;
            row[13] = is_terminal_or_honor(k) as u8 as f64;
            row[14] = self.reds[seat].contains(&k) as u8 as f64;
            row[15] = (Some(k) == focus) as u8 as f64;
            out.tiles.extend(row.iter().map(|&x| x as f32));
        }
        for rel in 0..4 {
            let p = (seat + rel) % 4;
            let mut row = [0f64; 12];
            row[0] = self.scores[p] as f64 / 100000.0;
            row[1] = self.riichi[p] as u8 as f64;
            row[2 + (p + 4 - self.oya) % 4] = 1.0;
            row[6] = (p == self.oya) as u8 as f64;
            row[7] = self.meld_n[p] as f64 / 4.0;
            row[8] = self.disc_n[p] as f64 / 24.0;
            row[9] = self.kan_n[p] as f64 / 4.0;
            row[10] = (Some(p) == focus_actor) as u8 as f64;
            row[11] = 1.0;
            out.players.extend(row.iter().map(|&x| x as f32));
        }
        let mut t = [0f64; 12];
        if let Some(i) = "ESWN".find(self.bakaze.as_str()).filter(|_| self.bakaze.len() == 1) {
            t[i] = 1.0;
        }
        t[4] = (self.kyoku - 1) as f64 / 4.0;
        t[5] = self.honba as f64 / 8.0;
        t[6] = self.kyotaku as f64 / 4.0;
        t[7] = self.wall.max(0) as f64 / WALL as f64;
        t[8] = self.dora.len() as f64 / 5.0;
        t[9] = own_turn as u8 as f64;
        t[10] = self.riichi.iter().filter(|&&r| r).count() as f64 / 4.0;
        t[11] = 1.0;
        out.table.extend(t.iter().map(|&x| x as f32));
    }
}

#[derive(Default)]
struct Frames {
    tiles: Vec<f32>,
    players: Vec<f32>,
    table: Vec<f32>,
    atype: Vec<i64>,
    tile: Vec<i64>,
    legal_type: Vec<f32>,
    legal_tile: Vec<f32>,
    mismatch: usize,
}

fn ty(ev: &Value) -> &str {
    ev.get("type").and_then(|v| v.as_str()).unwrap_or("")
}
fn actor(ev: &Value) -> Option<i64> {
    ev.get("actor").and_then(|v| v.as_i64())
}

fn own_turn_label(evs: &[Value], i: usize, a: usize) -> R<Option<(i64, i64)>> {
    let mut j = i + 1;
    while j < evs.len() && ty(&evs[j]) == "dora" {
        j += 1;
    }
    let Some(nx) = evs.get(j) else { return Ok(None) };
    let is_me = actor(nx) == Some(a as i64);
    Ok(match ty(nx) {
        "dahai" if is_me => Some((DAHAI, tile_kind(s(nx, "pai")?)? as i64)),
        "reach" if is_me => match evs.get(j + 1) {
            Some(d) if ty(d) == "dahai" && actor(d) == Some(a as i64) => Some((RIICHI, tile_kind(s(d, "pai")?)? as i64)),
            _ => None,
        },
        "ankan" if is_me => Some((KAN, tile_kind(Kyoku::consumed(nx)?.first().ok_or(Skip)?)? as i64)),
        "kakan" if is_me => Some((KAN, tile_kind(s(nx, "pai")?)? as i64)),
        "hora" if is_me && nx.get("target").and_then(|v| v.as_i64()) == Some(a as i64) => Some((HORA, 0)),
        "ryukyoku" => Some((RYUKYOKU, 0)),
        _ => None,
    })
}

fn discard_calls(evs: &[Value], i: usize, discarder: usize) -> R<HashMap<usize, (i64, i64)>> {
    let mut calls = HashMap::new();
    let mut j = i + 1;
    while j < evs.len() && matches!(ty(&evs[j]), "reach_accepted" | "dora") {
        j += 1;
    }
    while j < evs.len() && ty(&evs[j]) == "hora" && evs[j].get("target").and_then(|v| v.as_i64()) == Some(discarder as i64)
        && actor(&evs[j]) != Some(discarder as i64) {
        calls.insert(seat(&evs[j], "actor")?, (HORA, 0));
        j += 1;
    }
    if calls.is_empty() {
        if let Some(nx) = evs.get(j) {
            let t = ty(nx);
            if matches!(t, "chi" | "pon" | "daiminkan") && nx.get("target").and_then(|v| v.as_i64()) == Some(discarder as i64) {
                let k = tile_kind(s(nx, "pai")?)?;
                let a = seat(nx, "actor")?;
                let v = match t {
                    "chi" => {
                        let mut base = k;
                        for p in Kyoku::consumed(nx)? {
                            base = base.min(tile_kind(&p)?);
                        }
                        (CHI, base as i64)
                    }
                    "pon" => (PON, k as i64),
                    _ => (KAN, k as i64),
                };
                calls.insert(a, v);
            }
        }
    }
    Ok(calls)
}

fn bookkeeping(st: &mut Kyoku, ev: &Value) -> R<bool> {
    match ty(ev) {
        "dora" => {
            st.dora.push(tile_kind(s(ev, "dora_marker")?)?);
            Ok(true)
        }
        "reach_accepted" => {
            st.scores[seat(ev, "actor")?] -= 1000;
            st.kyotaku += 1;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn add_deltas(st: &mut Kyoku, ev: &Value) -> R<()> {
    if let Some(d) = ev.get("deltas") {
        let d = d.as_array().ok_or(Skip)?;
        for (i, x) in d.iter().take(4).enumerate() {
            st.scores[i] += x.as_i64().ok_or(Skip)?;
        }
    }
    Ok(())
}

/// (sample, frames whose ground truth fell outside the computed legality masks)
fn convert_game(evs: &[Value], name: &str, min_frames: usize) -> R<Vec<(Sample, usize)>> {
    let mut frames: Vec<Frames> = (0..4).map(|_| Frames::default()).collect();
    let mut st: Option<Kyoku> = None;
    let mut n_kyoku = 0;
    let mut consumed: HashSet<usize> = HashSet::new();
    let emit = |st: &Kyoku, frames: &mut Vec<Frames>, seat: usize, label: (i64, i64), focus: Option<usize>, own: bool, fa: Option<usize>,
                ctx: Ctx| {
        let f = &mut frames[seat];
        st.encode(seat, focus, own, fa, f);
        f.atype.push(label.0);
        f.tile.push(label.1);
        let (types, tiles, ok) = st.legal(seat, &ctx).frame(label.0, label.1);
        if ok {
            f.legal_type.extend(types.iter().map(|&b| b as u8 as f32));
            f.legal_tile.extend(tiles.iter().map(|&b| b as u8 as f32));
        } else {
            // never drop data silently: keep the frame unconstrained and count it
            f.legal_type.extend([1.0f32; N_TYPES]);
            f.legal_tile.extend([1.0f32; NK]);
            f.mismatch += 1;
        }
    };
    for (i, ev) in evs.iter().enumerate() {
        if consumed.contains(&i) {
            continue;
        }
        let t = ty(ev);
        if t == "start_kyoku" {
            st = Some(Kyoku::new(ev)?);
            n_kyoku += 1;
            continue;
        }
        let Some(k) = st.as_mut() else { continue };
        match t {
            "tsumo" => {
                let a = seat(ev, "actor")?;
                let drawn = k.draw(a, s(ev, "pai")?)?;
                let mut j = i + 1;
                while j < evs.len() && bookkeeping(k, &evs[j])? {
                    consumed.insert(j);
                    j += 1;
                }
                if let Some(label) = own_turn_label(evs, i, a)? {
                    emit(k, &mut frames, a, label, Some(drawn), true, None, Ctx::Draw { drawn });
                }
            }
            "dahai" => {
                let a = seat(ev, "actor")?;
                let kk = k.discard(a, s(ev, "pai")?)?;
                let mut j = i + 1;
                while j < evs.len() && bookkeeping(k, &evs[j])? {
                    consumed.insert(j);
                    j += 1;
                }
                let calls = discard_calls(evs, i, a)?;
                for rel in 1..4 {
                    let sidx = (a + rel) % 4;
                    let ctx = Ctx::Claim { kind: kk, discarder: a };
                    if let Some(&c) = calls.get(&sidx) {
                        emit(k, &mut frames, sidx, c, Some(kk), false, Some(a), ctx);
                    } else if !k.riichi[sidx] && k.wall > 0 && (k.can_pon(sidx, kk) || k.can_chi(sidx, a, kk)) {
                        emit(k, &mut frames, sidx, (PASS, 0), Some(kk), false, Some(a), ctx);
                    }
                }
            }
            "chi" | "pon" => {
                k.call(ev)?;
                let a = seat(ev, "actor")?;
                if let Some(nx) = evs.get(i + 1) {
                    if ty(nx) == "dahai" && actor(nx) == Some(a as i64) {
                        let lab = (DAHAI, tile_kind(s(nx, "pai")?)? as i64);
                        let ctx = Ctx::AfterCall {
                            called: tile_kind(s(ev, "pai")?)?,
                            consumed: Kyoku::consumed(ev)?.iter().map(|p| tile_kind(p)).collect::<R<Vec<_>>>()?,
                            is_chi: t == "chi",
                        };
                        emit(k, &mut frames, a, lab, None, true, None, ctx);
                    }
                }
            }
            "daiminkan" => k.call(ev)?,
            "ankan" => k.ankan(ev)?,
            "kakan" => k.kakan(ev)?,
            "reach" => k.pending[seat(ev, "actor")?] = true,
            "reach_accepted" | "dora" => {
                bookkeeping(k, ev)?;
            }
            "hora" => {
                add_deltas(k, ev)?;
                k.kyotaku = 0;
            }
            "ryukyoku" => add_deltas(k, ev)?,
            _ => {}
        }
    }
    let Some(st) = st else { return Err(Skip) };
    if n_kyoku == 0 {
        return Err(Skip);
    }
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by_key(|&s| (-st.scores[s], s));
    let mut placement = [0usize; 4];
    for (r, &s) in order.iter().enumerate() {
        placement[s] = r;
    }
    let mut out = Vec::new();
    for (sidx, f) in frames.into_iter().enumerate() {
        let t = f.atype.len();
        if t < min_frames {
            continue;
        }
        let v = PLACEMENT[placement[sidx]];
        let smp = Sample::new(json!({"name": name, "seat": sidx, "kyokus": n_kyoku, "placement": placement[sidx] + 1,
                                     "final_score": st.scores[sidx]}))
            .f32("obs.tiles_seq", &[NK, 16], f.tiles)
            .f32("obs.players_seq", &[4, 12], f.players)
            .f32("obs.table_seq", &[12], f.table)
            .i64("tgt.action_type", &[], f.atype)
            .i64("tgt.tile_idx", &[], f.tile)
            .f32("tgt.value", &[], vec![v; t])
            .bits("tgt.legal_action_type", &[N_TYPES], &f.legal_type)
            .bits("tgt.legal_tile_of", &[NK], &f.legal_tile);
        out.push((smp, f.mismatch));
    }
    Ok(out)
}

/// (name, mjson text) for every game in `input`.
fn iter_games(input: &Path, mut f: impl FnMut(&str, &str, Option<bool>) -> Result<bool>) -> Result<()> {
    let is_parquet = |p: &Path| p.extension().map(|e| e == "parquet").unwrap_or(false);
    if input.is_file() && !is_parquet(input) {
        let bytes = std::fs::read(input)?;
        if bytes.starts_with(b"PK") {
            let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
            for i in 0..z.len() {
                let mut m = z.by_index(i)?;
                let name = m.name().to_string();
                if m.is_dir() || !(name.ends_with(".mjson") || name.ends_with(".mjson.gz")) {
                    continue;
                }
                let mut buf = Vec::new();
                m.read_to_end(&mut buf)?;
                let text = String::from_utf8_lossy(&common::read_maybe_gz(buf)?).to_string();
                if !f(&name, &text, None)? {
                    return Ok(());
                }
            }
        } else {
            let text = String::from_utf8_lossy(&common::read_maybe_gz(bytes)?).to_string();
            f(&input.file_name().unwrap().to_string_lossy(), &text, None)?;
        }
        return Ok(());
    }
    let parquet = common::find_files(input, &[".parquet"])?;
    if !parquet.is_empty() {
        for p in parquet {
            let file = std::fs::File::open(&p)?;
            let b = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
            let schema = b.schema().clone();
            let ev_idx = schema.index_of("events").context("parquet needs an `events` column")?;
            let id_idx = schema.index_of("game_id").ok();
            let kf_idx = schema.index_of("kyoku_first").ok();
            let cols: Vec<usize> = std::iter::once(ev_idx).chain(id_idx).chain(kf_idx).collect();
            let mask = parquet::arrow::ProjectionMask::roots(b.parquet_schema(), cols);
            for batch in b.with_projection(mask).with_batch_size(512).build()? {
                let batch = batch?;
                let get = |name: &str| -> Option<Vec<Option<String>>> {
                    let c = batch.column_by_name(name)?;
                    if let Some(a) = c.as_any().downcast_ref::<StringArray>() {
                        return Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect());
                    }
                    let a = c.as_any().downcast_ref::<LargeStringArray>()?;
                    Some((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect())
                };
                let evs = get("events").context("events must be a string column")?;
                let ids = get("game_id");
                let kf: Option<Vec<Option<i64>>> = batch.column_by_name("kyoku_first").map(|c| {
                    let c = arrow_cast::cast(c, &arrow_schema::DataType::Int64).ok();
                    match c {
                        Some(c) => {
                            let a = c.as_any().downcast_ref::<arrow_array::Int64Array>().unwrap().clone();
                            (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect()
                        }
                        None => vec![None; batch.num_rows()],
                    }
                });
                for (r, e) in evs.into_iter().enumerate() {
                    let Some(e) = e else { continue };
                    let name = ids.as_ref().and_then(|v| v[r].clone()).unwrap_or_else(|| format!("{}#{r}", p.display()));
                    let hanchan = kf.as_ref().and_then(|v| v[r]).map(|k| k == 0);
                    if !f(&name, &e, hanchan)? {
                        return Ok(());
                    }
                }
            }
        }
        return Ok(());
    }
    for p in common::find_files(input, &[".mjson", ".mjson.gz"])? {
        let text = String::from_utf8_lossy(&common::read_maybe_gz(std::fs::read(&p)?)?).to_string();
        if !f(&p.file_name().unwrap().to_string_lossy(), &text, None)? {
            return Ok(());
        }
    }
    Ok(())
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Download Tenhou Houou (phoenix room) mjai logs: parquet shards of the public HF mirror
    Fetch {
        #[arg(long, default_value = "data/mahjong/raw")]
        out: PathBuf,
        /// shard indices 0..23 (each ~88k games, ~300 MB)
        #[arg(long, num_args = 1.., default_values_t = [0])]
        shards: Vec<usize>,
    },
    /// Convert mjai logs to gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 128)]
        shard_size: usize,
        #[arg(long)]
        max_games: Option<usize>,
        #[arg(long, default_value_t = 8)]
        min_frames: usize,
        /// keep only hanchan (East-South) games: the parquet `kyoku_first` column, or for
        /// mjson input the presence of a South-round kyoku
        #[arg(long)]
        hanchan_only: bool,
        /// imitate only seats that finished in the top N (1..4)
        #[arg(long)]
        top_placements: Option<usize>,
        /// high-quality preset: hanchan only (all seats are phoenix-room players)
        #[arg(long)]
        hq: bool,
    },
}

pub fn convert(input: &Path, out: &Path, shard_size: usize, max_games: Option<usize>, min_frames: usize, hanchan_only: bool,
               top_placements: Option<usize>) -> Result<Value> {
    let mut w = ShardWriter::new(out, "mahjong", CONVERTER_VERSION, json!({}), shard_size)?;
    let (mut games, mut skipped, mut samples, mut frames, mut mismatch) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut hist: BTreeMap<&str, usize> = ACTION_NAMES.iter().map(|&n| (n, 0)).collect();
    let mut low_quality = 0usize;
    iter_games(input, |name, text, hanchan| {
        if max_games.map(|m| games >= m).unwrap_or(false) {
            return Ok(false);
        }
        if hanchan_only {
            let is_hanchan = hanchan.unwrap_or_else(|| text.contains("\"bakaze\":\"S\""));
            if !is_hanchan {
                low_quality += 1;
                return Ok(true);
            }
        }
        let evs: std::result::Result<Vec<Value>, _> = text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect();
        match evs.ok().and_then(|e| convert_game(&e, name, min_frames).ok()) {
            Some(ss) => {
                games += 1;
                for (smp, mm) in ss {
                    if let Some(k) = top_placements {
                        if smp.meta["placement"].as_u64().unwrap_or(4) as usize > k {
                            continue;
                        }
                    }
                    samples += 1;
                    mismatch += mm;
                    if let Some(t) = smp.tensors.iter().find(|t| t.name == "tgt.action_type") {
                        if let gmshard::Column::I64(v) = &t.col {
                            frames += v.len();
                            for &a in v {
                                *hist.get_mut(ACTION_NAMES[a as usize]).unwrap() += 1;
                            }
                        }
                    }
                    w.push(smp)?;
                }
            }
            None => skipped += 1,
        }
        Ok(true)
    })?;
    w.flush()?;
    Ok(json!({"games": games, "skipped": skipped, "low_quality": low_quality, "samples": samples, "frames": frames,
              "shards": w.written_shards, "action_hist": hist, "mask_mismatch": mismatch}))
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Fetch { out, shards } => {
            let client = common::http()?;
            for i in shards {
                let name = format!("tenhou-{i:05}.parquet");
                let url = format!("https://huggingface.co/datasets/hhim8826/tenhou-houou-mjai/resolve/main/data/{name}");
                anyhow::ensure!(common::download(&client, &url, &out.join(&name))?, "404: {url}");
            }
            Ok(())
        }
        Cmd::Convert { input, out, shard_size, max_games, min_frames, hanchan_only, top_placements, hq } => {
            println!("{}", convert(&input, &out, shard_size, max_games, min_frames, hanchan_only || hq, top_placements)?);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "123m456p789s1122z" → kind counts (z: 1..7 = E S W N P F C).
    fn hand(s: &str) -> [i64; NK] {
        let mut c = [0i64; NK];
        let mut digits: Vec<usize> = Vec::new();
        for ch in s.chars() {
            match ch {
                '0'..='9' => digits.push(ch.to_digit(10).unwrap() as usize),
                'm' | 'p' | 's' | 'z' => {
                    let base = match ch {
                        'm' => 0,
                        'p' => 9,
                        's' => 18,
                        _ => 27,
                    };
                    for d in digits.drain(..) {
                        c[base + if d == 0 { 4 } else { d - 1 }] += 1;
                    }
                }
                _ => panic!("bad hand {s}"),
            }
        }
        c
    }

    fn waits(c: &[i64; NK], meld_n: i64) -> Vec<usize> {
        let mut w = *c;
        (0..NK)
            .filter(|&k| {
                w[k] += 1;
                let ok = is_agari(&w, meld_n);
                w[k] -= 1;
                ok
            })
            .collect()
    }

    /// A kyoku whose seat 0 holds `h0` (the other seats: filler hands).
    fn kyoku_with(h0: [i64; NK]) -> Kyoku {
        let ev = json!({"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "kyoku": 1, "honba": 0,
                        "kyotaku": 0, "oya": 0, "scores": [25000, 25000, 25000, 25000],
                        "tehais": [["1m","1m","1m","1m","2m","2m","2m","2m","3m","3m","3m","3m","4m"],
                                   ["1p","1p","1p","1p","2p","2p","2p","2p","3p","3p","3p","3p","4p"],
                                   ["1s","1s","1s","1s","2s","2s","2s","2s","3s","3s","3s","3s","4s"],
                                   ["E","E","E","E","S","S","S","S","W","W","W","W","N"]]});
        let mut k = Kyoku::new(&ev).ok().unwrap();
        k.hands[0] = h0;
        k
    }

    fn kinds(m: &[bool; NK]) -> Vec<usize> {
        (0..NK).filter(|&k| m[k]).collect()
    }

    #[test]
    fn agari_shapes() {
        assert!(is_agari(&hand("123m456p789s11122z"), 0));
        assert!(is_agari(&hand("11122233344455m"), 0)); // needs the triplet/run backtrack
        assert!(is_agari(&hand("11123456789999m"), 0)); // chuuren shape
        assert!(!is_agari(&hand("123m456p789s11123z"), 0));
        // chiitoitsu: seven distinct pairs; a quad is not two pairs
        assert!(is_agari(&hand("1199m2288p3377s11z"), 0));
        assert!(!is_agari(&hand("1111m2288p3377s11z"), 0));
        // kokushi (closed only)
        assert!(is_agari(&hand("19m19p19s12345677z"), 0));
        assert!(!is_agari(&hand("19m19p19s12345566z"), 0)); // no chun
        // open melds: 2 melds → 8 concealed tiles = 2 sets + pair; no chiitoitsu with melds
        assert!(is_agari(&hand("123m456p11z"), 2));
        assert!(!is_agari(&hand("123m456p11z"), 0));
        assert!(is_agari(&hand("55z"), 4));
        assert!(!is_agari(&hand("99m2288p3377s11z"), 1));
        // honors never form runs
        assert!(!is_agari(&hand("123z456p789s11122z"), 0));
        // wrong tile count for the meld count
        assert!(!is_agari(&hand("123m456p789s11122z"), 1));
    }

    #[test]
    fn tenpai_waits() {
        // ryanmen 23m → 1m / 4m
        assert_eq!(waits(&hand("23m456p789s11122z"), 0), vec![0, 3]);
        // shanpon EE SS → E / S; EEE + S → S tanki only
        assert_eq!(waits(&hand("123m456p789s1122z"), 0), vec![27, 28]);
        assert_eq!(waits(&hand("123m456p789s1112z"), 0), vec![28]);
        // nobetan 1234m → 1m / 4m
        assert_eq!(waits(&hand("1234m456p789s111z"), 0), vec![0, 3]);
        // chuuren: all nine kinds of the suit
        assert_eq!(waits(&hand("1112345678999m"), 0), (0..9).collect::<Vec<_>>());
        // chiitoitsu tanki
        assert_eq!(waits(&hand("1199m2288p3377s1z"), 0), vec![27]);
        // kokushi 13-sided
        let all13: Vec<usize> = (0..NK).filter(|&k| is_terminal_or_honor(k)).collect();
        assert_eq!(waits(&hand("19m19p19s1234567z"), 0), all13);
        // 4 melds: a tanki on the last concealed tile
        assert_eq!(waits(&hand("5z"), 4), vec![31]);
        assert!(!is_tenpai(&hand("13579m2468p1357s"), 0));
    }

    #[test]
    fn draw_frame_masks() {
        // closed 14 tiles: 23m456p789s11122z + drawn 9m
        let mut k = kyoku_with(hand("239m456p789s11122z"));
        let l = k.legal(0, &Ctx::Draw { drawn: 8 });
        assert!(l.types[DAHAI as usize] && l.types[RIICHI as usize] && !l.types[HORA as usize]);
        assert!(!l.types[CHI as usize] && !l.types[PON as usize] && !l.types[PASS as usize]);
        assert_eq!(kinds(&l.tiles[RIICHI as usize]), vec![8]); // only the 9m keeps tenpai
        assert_eq!(kinds(&l.tiles[DAHAI as usize]).len(), 11);
        // not enough points / too few wall tiles / open hand / already in riichi → no riichi
        k.scores[0] = 900;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 8 }).types[RIICHI as usize]);
        k.scores[0] = 1000;
        k.wall = 3;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 8 }).types[RIICHI as usize]);
        k.wall = 4;
        assert!(k.legal(0, &Ctx::Draw { drawn: 8 }).types[RIICHI as usize]);
        k.open[0] = true;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 8 }).types[RIICHI as usize]);
        k.open[0] = false;
        k.riichi[0] = true;
        let l = k.legal(0, &Ctx::Draw { drawn: 8 });
        assert!(!l.types[RIICHI as usize]);
        assert_eq!(kinds(&l.tiles[DAHAI as usize]), vec![8]); // tsumogiri only
        // tsumo shape → hora
        let k = kyoku_with(hand("123m456p789s11122z"));
        assert!(k.legal(0, &Ctx::Draw { drawn: 27 }).types[HORA as usize]);
        // kyushu kyuhai: ≥ 9 distinct terminal/honor kinds before the first discard / any call
        let mut k = kyoku_with(hand("19m19p19s1234z258m"));
        assert!(k.legal(0, &Ctx::Draw { drawn: 0 }).types[RYUKYOKU as usize]);
        k.any_call = true;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 0 }).types[RYUKYOKU as usize]);
        let k = kyoku_with(hand("19m19p19s12z225588m"));
        assert!(!k.legal(0, &Ctx::Draw { drawn: 0 }).types[RYUKYOKU as usize]); // 8 kinds
    }

    #[test]
    fn kan_kinds() {
        let mut k = kyoku_with(hand("1111m456p789s1122z"));
        let l = k.legal(0, &Ctx::Draw { drawn: 0 });
        assert_eq!(kinds(&l.tiles[KAN as usize]), vec![0]); // ankan
        // kakan: an own pon of E + the 4th E in hand (11 concealed tiles)
        k.hands[0] = hand("123m456p789s15z");
        k.meld_n[0] = 1;
        k.open[0] = true;
        k.pon_kinds[0][27] = true;
        let l = k.legal(0, &Ctx::Draw { drawn: 27 });
        assert_eq!(kinds(&l.tiles[KAN as usize]), vec![27]);
        k.wall = 0; // no kan off the last tile
        assert!(!k.legal(0, &Ctx::Draw { drawn: 27 }).types[KAN as usize]);
        // daiminkan claim needs three in hand
        let k = kyoku_with(hand("111m456p789s1122z"));
        let l = k.legal(0, &Ctx::Claim { kind: 0, discarder: 2 });
        assert!(l.types[KAN as usize] && l.types[PON as usize] && l.types[PASS as usize]);
        assert!(!l.types[CHI as usize] && !l.types[DAHAI as usize]); // seat 0 is not shimocha of 2
        assert_eq!(kinds(&l.tiles[KAN as usize]), vec![0]);
        assert_eq!(kinds(&l.tiles[PON as usize]), vec![0]);
    }

    #[test]
    fn claim_masks() {
        // seat 0 is shimocha of seat 3; discard 4m with 2m 3m 5m 6m in hand
        let k = kyoku_with(hand("2356m456p789s112z"));
        let l = k.legal(0, &Ctx::Claim { kind: 3, discarder: 3 });
        assert!(l.types[CHI as usize] && !l.types[PON as usize] && !l.types[KAN as usize]);
        assert_eq!(kinds(&l.tiles[CHI as usize]), vec![1, 2, 3]); // 234 / 345 / 456
        // a riichi seat: pass (or ron) only
        let mut k2 = kyoku_with(hand("2356m456p789s112z"));
        k2.riichi[0] = true;
        let l = k2.legal(0, &Ctx::Claim { kind: 3, discarder: 3 });
        assert!(!l.types[CHI as usize] && l.types[PASS as usize]);
        // ron shape
        let k = kyoku_with(hand("23m456p789s11122z"));
        let l = k.legal(0, &Ctx::Claim { kind: 0, discarder: 1 });
        assert!(l.types[HORA as usize] && !l.types[CHI as usize]);
    }

    #[test]
    fn kuikae() {
        let k = kyoku_with(hand("1236789m456p11z"));
        let dahai = |called: usize, consumed: Vec<usize>, is_chi: bool| {
            kinds(&k.legal(0, &Ctx::AfterCall { called, consumed, is_chi }).tiles[DAHAI as usize])
        };
        let held = |skip: &[usize]| -> Vec<usize> { (0..NK).filter(|&x| k.hands[0][x] > 0 && !skip.contains(&x)).collect() };
        // chi 3m with 45m (run 345, called at the low end) → no 3m, no 6m (suji)
        assert_eq!(dahai(2, vec![3, 4], true), held(&[2, 5]));
        // chi 6m with 45m (run 456, called at the high end) → no 6m, no 3m
        assert_eq!(dahai(5, vec![3, 4], true), held(&[5, 2]));
        // kanchan chi 5m with 46m → only 5m
        assert_eq!(dahai(4, vec![3, 5], true), held(&[4]));
        // run 789m called 7m: no suji beyond the suit edge
        assert_eq!(dahai(6, vec![7, 8], true), held(&[6]));
        // run 123m called 3m: no suji below the suit edge
        assert_eq!(dahai(2, vec![0, 1], true), held(&[2]));
        // pon: only the called kind
        assert_eq!(dahai(27, vec![27, 27], false), held(&[27]));
        let l = k.legal(0, &Ctx::AfterCall { called: 27, consumed: vec![27, 27], is_chi: false });
        assert_eq!(l.types, [true, false, false, false, false, false, false, false]);
    }

    #[test]
    fn frame_mask_fallback() {
        let mut l = Legal::none();
        l.allow(DAHAI, std::array::from_fn(|k| k == 3));
        assert!(l.frame(DAHAI, 3).2);
        assert!(!l.frame(DAHAI, 4).2);
        assert!(!l.frame(HORA, 0).2);
        l.allow(PASS, [true; NK]);
        let (_, row, ok) = l.frame(PASS, 0);
        assert!(ok && row.iter().all(|&x| x));
    }

    #[test]
    fn fixture_game_has_no_mask_mismatch() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/mahjong/000fad71bb735a027861961e.mjson");
        let text = std::fs::read_to_string(&path).unwrap();
        let evs: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect();
        let out = convert_game(&evs, "fixture", 1).ok().unwrap();
        assert_eq!(out.len(), 4);
        for (smp, mismatch) in &out {
            assert_eq!(*mismatch, 0);
            assert!(smp.frames().unwrap() > 0);
            let names: Vec<&str> = smp.tensors.iter().map(|x| x.name.as_str()).collect();
            assert!(names.contains(&"tgt.legal_action_type") && names.contains(&"tgt.legal_tile_of"));
        }
    }
}
