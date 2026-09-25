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
//! value = final placement 1st +1, 2nd +1/3, 3rd −1/3, 4th −1 (ties → lower seat).
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
        if s(ev, "type")? == "daiminkan" {
            self.kan_n[c] += 1;
        }
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
        Ok(())
    }
    fn kakan(&mut self, ev: &Value) -> R<()> {
        let c = seat(ev, "actor")?;
        let kk = self.remove(c, s(ev, "pai")?)?;
        self.melds[c][kk] += 1;
        self.kan_n[c] += 1;
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

fn convert_game(evs: &[Value], name: &str, min_frames: usize) -> R<Vec<Sample>> {
    let mut frames: Vec<Frames> = (0..4).map(|_| Frames::default()).collect();
    let mut st: Option<Kyoku> = None;
    let mut n_kyoku = 0;
    let mut consumed: HashSet<usize> = HashSet::new();
    let emit = |st: &Kyoku, frames: &mut Vec<Frames>, seat: usize, label: (i64, i64), focus: Option<usize>, own: bool, fa: Option<usize>| {
        st.encode(seat, focus, own, fa, &mut frames[seat]);
        frames[seat].atype.push(label.0);
        frames[seat].tile.push(label.1);
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
                    emit(k, &mut frames, a, label, Some(drawn), true, None);
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
                    if let Some(&c) = calls.get(&sidx) {
                        emit(k, &mut frames, sidx, c, Some(kk), false, Some(a));
                    } else if !k.riichi[sidx] && k.wall > 0 && (k.can_pon(sidx, kk) || k.can_chi(sidx, a, kk)) {
                        emit(k, &mut frames, sidx, (PASS, 0), Some(kk), false, Some(a));
                    }
                }
            }
            "chi" | "pon" => {
                k.call(ev)?;
                let a = seat(ev, "actor")?;
                if let Some(nx) = evs.get(i + 1) {
                    if ty(nx) == "dahai" && actor(nx) == Some(a as i64) {
                        let lab = (DAHAI, tile_kind(s(nx, "pai")?)? as i64);
                        emit(k, &mut frames, a, lab, None, true, None);
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
        out.push(Sample::new(json!({"name": name, "seat": sidx, "kyokus": n_kyoku, "placement": placement[sidx] + 1,
                                     "final_score": st.scores[sidx]}))
            .f32("obs.tiles_seq", &[NK, 16], f.tiles)
            .f32("obs.players_seq", &[4, 12], f.players)
            .f32("obs.table_seq", &[12], f.table)
            .i64("tgt.action_type", &[], f.atype)
            .i64("tgt.tile_idx", &[], f.tile)
            .f32("tgt.value", &[], vec![v; t]));
    }
    Ok(out)
}

/// (name, mjson text) for every game in `input`.
fn iter_games(input: &Path, mut f: impl FnMut(&str, &str) -> Result<bool>) -> Result<()> {
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
                if !f(&name, &text)? {
                    return Ok(());
                }
            }
        } else {
            let text = String::from_utf8_lossy(&common::read_maybe_gz(bytes)?).to_string();
            f(&input.file_name().unwrap().to_string_lossy(), &text)?;
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
            let cols: Vec<usize> = std::iter::once(ev_idx).chain(id_idx).collect();
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
                for (r, e) in evs.into_iter().enumerate() {
                    let Some(e) = e else { continue };
                    let name = ids.as_ref().and_then(|v| v[r].clone()).unwrap_or_else(|| format!("{}#{r}", p.display()));
                    if !f(&name, &e)? {
                        return Ok(());
                    }
                }
            }
        }
        return Ok(());
    }
    for p in common::find_files(input, &[".mjson", ".mjson.gz"])? {
        let text = String::from_utf8_lossy(&common::read_maybe_gz(std::fs::read(&p)?)?).to_string();
        if !f(&p.file_name().unwrap().to_string_lossy(), &text)? {
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
    },
}

pub fn convert(input: &Path, out: &Path, shard_size: usize, max_games: Option<usize>, min_frames: usize) -> Result<Value> {
    let mut w = ShardWriter::new(out, "mahjong", 1, json!({}), shard_size)?;
    let (mut games, mut skipped, mut samples, mut frames) = (0usize, 0usize, 0usize, 0usize);
    let mut hist: BTreeMap<&str, usize> = ACTION_NAMES.iter().map(|&n| (n, 0)).collect();
    iter_games(input, |name, text| {
        if max_games.map(|m| games >= m).unwrap_or(false) {
            return Ok(false);
        }
        let evs: std::result::Result<Vec<Value>, _> = text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect();
        match evs.ok().and_then(|e| convert_game(&e, name, min_frames).ok()) {
            Some(ss) => {
                games += 1;
                for smp in ss {
                    samples += 1;
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
    Ok(json!({"games": games, "skipped": skipped, "samples": samples, "frames": frames, "shards": w.written_shards, "action_hist": hist}))
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
        Cmd::Convert { input, out, shard_size, max_games, min_frames } => {
            println!("{}", convert(&input, &out, shard_size, max_games, min_frames)?);
            Ok(())
        }
    }
}
