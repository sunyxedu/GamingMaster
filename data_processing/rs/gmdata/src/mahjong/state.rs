//! Per-kyoku game state tracked from the mjai events, the observation encoding, and the
//! exact legality of every decision under Tenhou's rules (see the module doc in `mod.rs`).

use super::rules::*;
use serde_json::Value;
use std::collections::HashSet;

pub(super) const DAHAI: i64 = 0;
pub(super) const RIICHI: i64 = 1;
pub(super) const CHI: i64 = 2;
pub(super) const PON: i64 = 3;
pub(super) const KAN: i64 = 4;
pub(super) const HORA: i64 = 5;
pub(super) const RYUKYOKU: i64 = 6;
pub(super) const PASS: i64 = 7;
pub(super) const ACTION_NAMES: [&str; 8] = ["dahai", "riichi", "chi", "pon", "kan", "hora", "ryukyoku", "pass"];
/// Whether an action type takes a tile argument (the decoder's ARG_USES table).
pub(super) const USES_TILE: [bool; 8] = [true, true, true, true, true, false, false, false];
pub(super) const N_TYPES: usize = 8;
/// Live-wall draws of a kyoku (136 − 52 dealt − 14 dead wall); rinshan draws count too.
pub(super) const WALL: i64 = 70;

/// Decision context of a frame (what just happened to the acting seat).
#[derive(Clone, Debug)]
pub(super) enum Ctx {
    /// own draw of kind `drawn`
    Draw { drawn: usize },
    /// forced discard after a chi/pon of `called` with the `consumed` hand kinds
    AfterCall { called: usize, consumed: Vec<usize>, is_chi: bool },
    /// another seat's discard (or, with `chankan`, its kakan) of `kind` by `discarder`
    Claim { kind: usize, discarder: usize, chankan: bool },
}

/// Legal action types and, per type, legal tile arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Legal {
    pub types: [bool; N_TYPES],
    pub tiles: [[bool; NK]; N_TYPES],
}

impl Legal {
    pub fn none() -> Self {
        Legal { types: [false; N_TYPES], tiles: [[false; NK]; N_TYPES] }
    }
    pub fn allow(&mut self, t: i64, tiles: [bool; NK]) {
        self.types[t as usize] = true;
        self.tiles[t as usize] = tiles;
    }
    /// Some action other than pass is legal (a claim decision exists).
    pub fn has_claim(&self) -> bool {
        (0..N_TYPES).any(|t| t != PASS as usize && self.types[t])
    }
    /// (type mask, tile mask of type `t`, whether (t, tile) is inside them)
    pub fn frame(&self, t: i64, tile: i64) -> ([bool; N_TYPES], [bool; NK], bool) {
        let ti = t as usize;
        let row = if USES_TILE[ti] { self.tiles[ti] } else { [true; NK] };
        let ok = self.types[ti] && (!USES_TILE[ti] || (0..NK as i64).contains(&tile) && row[tile as usize]);
        (self.types, row, ok)
    }
}

pub(super) fn s<'a>(ev: &'a Value, k: &str) -> R<&'a str> {
    ev.get(k).and_then(|v| v.as_str()).ok_or(Skip)
}
pub(super) fn n(ev: &Value, k: &str) -> R<i64> {
    ev.get(k).and_then(|v| v.as_i64()).ok_or(Skip)
}
pub(super) fn seat(ev: &Value, k: &str) -> R<usize> {
    let v = n(ev, k)?;
    if (0..4).contains(&v) {
        Ok(v as usize)
    } else {
        Err(Skip)
    }
}

/// Observation buffers of one seat's frames (layouts in `Kyoku::encode`).
#[derive(Default)]
pub(super) struct ObsBuf {
    /// tile planes [T, 34, 16] as u8 quarters (value × 4)
    pub tiles_q: Vec<u8>,
    pub players: Vec<f32>,
    pub table: Vec<f32>,
}

pub(super) struct Kyoku {
    pub oya: usize,
    pub bakaze: String,
    pub kyoku: i64,
    pub honba: i64,
    pub kyotaku: i64,
    pub scores: [i64; 4],
    /// revealed dora markers (kinds)
    pub dora: Vec<usize>,
    pub hands: [[i64; NK]; 4],
    pub reds: [HashSet<usize>; 4],
    /// tile counts of each seat's melds (the observation's meld columns)
    pub melds: [[i64; NK]; 4],
    /// the melds themselves (yaku evaluation)
    pub meld_list: [Vec<Meld>; 4],
    pub meld_n: [i64; 4],
    pub kan_n: [i64; 4],
    /// ponds: discards still on the table (called tiles leave them)
    pub disc: [[i64; NK]; 4],
    pub disc_n: [i64; 4],
    /// kinds each seat ever discarded, called ones included (furiten: Tenhou judges it on
    /// the seat's own discards, including those called away)
    pub river: [[bool; NK]; 4],
    /// discards made (never decremented by calls)
    pub discards_made: [i64; 4],
    pub first_discard: [Option<usize>; 4],
    /// riichi declared (from the declaration discard on; the stick follows reach_accepted)
    pub riichi: [bool; 4],
    /// reach announced, declaration discard not yet made
    pub pending: [bool; 4],
    /// same-turn furiten: passed on a winning tile since the seat's own last discard
    pub furiten_temp: [bool; 4],
    /// passed on a winning tile while in riichi: furiten for the rest of the kyoku
    pub furiten_riichi: [bool; 4],
    /// live-wall draws left
    pub wall: i64,
    /// called chi / pon / daiminkan (an ankan keeps the hand closed)
    pub open: [bool; 4],
    /// kinds ponned (kakan candidates)
    pub pon_kinds: [[bool; NK]; 4],
    /// any chi / pon / kan this kyoku (ends the first uninterrupted go-around)
    pub any_call: bool,
    /// a kan was declared: the next draw is a rinshan draw
    pub after_kan: bool,
    /// the current own-turn draw is a rinshan draw
    pub rinshan_draw: bool,
}

impl Kyoku {
    pub fn new(ev: &Value) -> R<Self> {
        let scores_v = ev.get("scores").and_then(|v| v.as_array()).ok_or(Skip)?;
        let mut scores = [0i64; 4];
        for (i, sc) in scores.iter_mut().enumerate() {
            *sc = scores_v.get(i).and_then(|v| v.as_i64()).ok_or(Skip)?;
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
            meld_list: Default::default(),
            meld_n: [0; 4],
            kan_n: [0; 4],
            disc: [[0; NK]; 4],
            disc_n: [0; 4],
            river: [[false; NK]; 4],
            discards_made: [0; 4],
            first_discard: [None; 4],
            riichi: [false; 4],
            pending: [false; 4],
            furiten_temp: [false; 4],
            furiten_riichi: [false; 4],
            wall: WALL,
            open: [false; 4],
            pon_kinds: [[false; NK]; 4],
            any_call: false,
            after_kan: false,
            rinshan_draw: false,
        };
        let tehais = ev.get("tehais").and_then(|v| v.as_array()).ok_or(Skip)?;
        if tehais.len() != 4 {
            return Err(Skip);
        }
        for (si, t) in tehais.iter().enumerate() {
            let t = t.as_array().ok_or(Skip)?;
            if t.len() != 13 {
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

    pub fn draw(&mut self, s: usize, p: &str) -> R<usize> {
        self.wall -= 1;
        self.rinshan_draw = std::mem::take(&mut self.after_kan);
        self.add(s, p)?;
        tile_kind(p)
    }

    pub fn discard(&mut self, s: usize, p: &str) -> R<usize> {
        let k = self.remove(s, p)?;
        self.disc[s][k] += 1;
        self.disc_n[s] += 1;
        self.river[s][k] = true;
        self.discards_made[s] += 1;
        if self.first_discard[s].is_none() {
            self.first_discard[s] = Some(k);
        }
        if self.pending[s] {
            self.pending[s] = false;
            self.riichi[s] = true;
        }
        // the seat's own discard ends same-turn furiten (同巡振聴の解消は打牌時), a discard after
        // its own call too, even a pon called out of turn order: the stricter reading (no
        // release after a 逆巡 call) contradicts 16 recorded rons in 165k Houou games; other
        // seats' calls never release it
        self.furiten_temp[s] = false;
        self.rinshan_draw = false;
        Ok(k)
    }

    pub fn consumed(ev: &Value) -> R<Vec<String>> {
        ev.get("consumed").and_then(|v| v.as_array()).ok_or(Skip)?.iter().map(|x| x.as_str().map(String::from).ok_or(Skip)).collect()
    }

    /// chi / pon / daiminkan of the target's last discard
    pub fn call(&mut self, ev: &Value) -> R<()> {
        let (c, t) = (seat(ev, "actor")?, seat(ev, "target")?);
        let k = tile_kind(s(ev, "pai")?)?;
        if c == t || self.disc[t][k] <= 0 {
            return Err(Skip);
        }
        self.disc[t][k] -= 1;
        self.disc_n[t] -= 1;
        self.melds[c][k] += 1;
        let mut tiles = vec![k];
        for p in Self::consumed(ev)? {
            let kk = self.remove(c, &p)?;
            self.melds[c][kk] += 1;
            tiles.push(kk);
        }
        tiles.sort_unstable();
        let same = tiles.iter().all(|&x| x == tiles[0]);
        let kind = match s(ev, "type")? {
            "daiminkan" if tiles.len() == 4 && same => {
                self.kan_n[c] += 1;
                self.after_kan = true;
                MeldKind::Daiminkan
            }
            "pon" if tiles.len() == 3 && same => {
                self.pon_kinds[c][k] = true;
                MeldKind::Pon
            }
            "chi" if tiles.len() == 3 && tiles[0] < 27 && tiles[0] % 9 <= 6 && tiles[1] == tiles[0] + 1 && tiles[2] == tiles[0] + 2
                && (c + 3) % 4 == t => MeldKind::Chi,
            _ => return Err(Skip),
        };
        self.meld_list[c].push(Meld { kind, tiles });
        self.meld_n[c] += 1;
        self.open[c] = true;
        self.any_call = true;
        Ok(())
    }

    pub fn ankan(&mut self, ev: &Value) -> R<usize> {
        let c = seat(ev, "actor")?;
        let mut tiles = Vec::with_capacity(4);
        for p in Self::consumed(ev)? {
            let kk = self.remove(c, &p)?;
            self.melds[c][kk] += 1;
            tiles.push(kk);
        }
        if tiles.len() != 4 || tiles.iter().any(|&x| x != tiles[0]) {
            return Err(Skip);
        }
        let k = tiles[0];
        self.meld_list[c].push(Meld { kind: MeldKind::Ankan, tiles });
        self.meld_n[c] += 1;
        self.kan_n[c] += 1;
        self.any_call = true;
        self.after_kan = true;
        Ok(k)
    }

    pub fn kakan(&mut self, ev: &Value) -> R<usize> {
        let c = seat(ev, "actor")?;
        let kk = tile_kind(s(ev, "pai")?)?;
        if !self.pon_kinds[c][kk] {
            return Err(Skip);
        }
        let kk = self.remove(c, s(ev, "pai")?)?;
        let m = self.meld_list[c].iter_mut().find(|m| m.kind == MeldKind::Pon && m.tiles[0] == kk).ok_or(Skip)?;
        m.kind = MeldKind::Kakan;
        m.tiles.push(kk);
        self.melds[c][kk] += 1;
        self.kan_n[c] += 1;
        self.pon_kinds[c][kk] = false;
        self.any_call = true;
        self.after_kan = true;
        Ok(kk)
    }

    pub fn accept_riichi(&mut self, a: usize) {
        self.scores[a] -= 1000;
        self.kyotaku += 1;
    }

    pub fn add_dora(&mut self, ev: &Value) -> R<()> {
        self.dora.push(tile_kind(s(ev, "dora_marker")?)?);
        Ok(())
    }

    pub fn add_deltas(&mut self, ev: &Value) -> R<()> {
        if let Some(d) = ev.get("deltas") {
            let d = d.as_array().ok_or(Skip)?;
            for (i, x) in d.iter().take(4).enumerate() {
                self.scores[i] += x.as_i64().ok_or(Skip)?;
            }
        }
        Ok(())
    }

    pub fn kans_total(&self) -> i64 {
        self.kan_n.iter().sum()
    }

    fn round_wind(&self) -> u8 {
        match self.bakaze.as_str() {
            "S" => 1,
            "W" => 2,
            "N" => 3,
            _ => 0,
        }
    }

    /// An abortive draw is pending if the discard just made passes without a ron, so the
    /// discard allows a ron only: four winds on the first go-around, four kans by more
    /// than one seat, four riichi.
    pub fn abortive_pending(&self) -> bool {
        let winds = !self.any_call
            && self.discards_made.iter().all(|&d| d == 1)
            && matches!(self.first_discard[0], Some(w) if (27..31).contains(&w))
            && self.first_discard.iter().all(|&d| d == self.first_discard[0]);
        let kans = self.kans_total() >= 4 && self.kan_n.iter().filter(|&&x| x > 0).count() > 1;
        winds || kans || self.riichi.iter().all(|&r| r)
    }

    fn win_ctx(&self, s: usize, tsumo: bool, chankan: bool) -> WinCtx {
        WinCtx {
            tsumo,
            riichi: self.riichi[s],
            haitei: tsumo && self.wall == 0 && !self.rinshan_draw,
            houtei: !tsumo && !chankan && self.wall == 0,
            rinshan: tsumo && self.rinshan_draw,
            chankan,
            first_draw: tsumo && self.discards_made[s] == 0 && !self.any_call,
            seat_wind: ((s + 4 - self.oya) % 4) as u8,
            round_wind: self.round_wind(),
        }
    }

    /// Furiten: a wait among the seat's own discards, or a winning tile passed since its
    /// last discard (or since its riichi).
    pub fn is_furiten(&self, s: usize) -> bool {
        if self.furiten_temp[s] || self.furiten_riichi[s] {
            return true;
        }
        let w = waits(&self.hands[s], self.meld_n[s]);
        (0..NK).any(|k| w[k] && self.river[s][k])
    }

    /// Ron on another seat's discard (or kakan, `chankan`) of kind `k`: a winning shape,
    /// no furiten, a yaku.
    pub fn ron_legal(&self, s: usize, k: usize, chankan: bool) -> bool {
        let h = &self.hands[s];
        if h[k] >= 4 {
            return false;
        }
        let mut w = *h;
        w[k] += 1;
        is_agari(&w, self.meld_n[s]) && !self.is_furiten(s) && has_yaku(h, &self.meld_list[s], k, &self.win_ctx(s, false, chankan))
    }

    /// Tsumo after drawing `drawn`: a winning shape with a yaku (furiten never blocks it).
    pub fn tsumo_legal(&self, s: usize, drawn: usize) -> bool {
        let h = &self.hands[s];
        if h[drawn] <= 0 || !is_agari(h, self.meld_n[s]) {
            return false;
        }
        let mut c = *h;
        c[drawn] -= 1;
        has_yaku(&c, &self.meld_list[s], drawn, &self.win_ctx(s, true, false))
    }

    /// Seat `s` let a tile of kind `k` pass (no ron on it): if it completes the hand's
    /// shape — with or without a yaku — the seat is furiten until its next discard, and for
    /// good once in riichi (Tenhou: 和了放棄(搶槓を含む)後は同巡フリテン; リーチ後の当たり牌見逃しは以降振聴).
    pub fn note_pass(&mut self, s: usize, k: usize) {
        let mut w = self.hands[s];
        if w[k] >= 4 {
            return;
        }
        w[k] += 1;
        if is_agari(&w, self.meld_n[s]) {
            self.furiten_temp[s] = true;
            if self.riichi[s] {
                self.furiten_riichi[s] = true;
            }
        }
    }

    /// Legal action types / tile arguments of `seat` in context `ctx` (see the module doc).
    pub fn legal(&self, seat: usize, ctx: &Ctx) -> Legal {
        let h = &self.hands[seat];
        let meld_n = self.meld_n[seat];
        let in_hand: [bool; NK] = std::array::from_fn(|k| h[k] > 0);
        let mut l = Legal::none();
        match ctx {
            Ctx::Draw { drawn } => {
                let drawn = *drawn;
                if self.riichi[seat] {
                    l.allow(DAHAI, std::array::from_fn(|k| k == drawn && h[k] > 0));
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
                if self.wall > 0 && self.kans_total() < 4 {
                    let kans: [bool; NK] = if self.riichi[seat] {
                        // ankan of the drawn tile only (no 送り槓), and only if the waits stay
                        // the same (牌姿や役の増減は不問)
                        let mut ok = false;
                        if h[drawn] == 4 {
                            let mut pre = *h;
                            pre[drawn] -= 1;
                            let mut post = *h;
                            post[drawn] = 0;
                            let (wp, wq) = (waits(&pre, meld_n), waits(&post, meld_n + 1));
                            ok = wp.iter().any(|&x| x) && wp == wq;
                        }
                        std::array::from_fn(|k| ok && k == drawn)
                    } else {
                        std::array::from_fn(|k| h[k] == 4 || (self.pon_kinds[seat][k] && h[k] >= 1))
                    };
                    if kans.iter().any(|&x| x) {
                        l.allow(KAN, kans);
                    }
                }
                if self.tsumo_legal(seat, drawn) {
                    l.allow(HORA, [true; NK]);
                }
                let yaochu = (0..NK).filter(|&k| is_terminal_or_honor(k) && h[k] > 0).count();
                if self.discards_made[seat] == 0 && !self.any_call && yaochu >= 9 {
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
            Ctx::Claim { kind, discarder, chankan } => {
                let c = *kind;
                l.allow(PASS, [true; NK]);
                if !*chankan && !self.riichi[seat] && self.wall > 0 && !self.abortive_pending() {
                    let n: i64 = h.iter().sum();
                    if seat == (*discarder + 1) % 4 && c < 27 && n >= 3 {
                        let bases: [bool; NK] = std::array::from_fn(|b| {
                            if !(b < 27 && b / 9 == c / 9 && b % 9 <= 6 && b <= c && c <= b + 2) {
                                return false;
                            }
                            let mut rest = *h;
                            for x in (b..b + 3).filter(|&x| x != c) {
                                if rest[x] <= 0 {
                                    return false;
                                }
                                rest[x] -= 1;
                            }
                            // 喰い替え: the called kind and, for a run called at an end, the
                            // kind beyond the other end may not be discarded next; some other
                            // tile must be left to discard
                            rest[c] = 0;
                            if b == c && c % 9 <= 5 {
                                rest[c + 3] = 0;
                            } else if b + 2 == c && c % 9 >= 3 {
                                rest[c - 3] = 0;
                            }
                            rest.iter().sum::<i64>() > 0
                        });
                        if bases.iter().any(|&x| x) {
                            l.allow(CHI, bases);
                        }
                    }
                    let only_c: [bool; NK] = std::array::from_fn(|k| k == c);
                    if h[c] >= 2 && n >= 3 && n - h[c] > 0 {
                        l.allow(PON, only_c);
                    }
                    if h[c] >= 3 && self.kans_total() < 4 {
                        l.allow(KAN, only_c);
                    }
                }
                if self.ron_legal(seat, c, *chankan) {
                    l.allow(HORA, [true; NK]);
                }
            }
        }
        l
    }

    /// The observation of `seat` (tiles [34,16], players [4,12], table [12]; layouts in
    /// model/games/mahjong.py) appended to the three buffers.  Every tile feature is a
    /// multiple of 1/4 (counts / 4 and 0/1 flags), stored exactly as u8 quarters (value × 4):
    /// the loader expands them to the float32 `tiles_seq` (data_processing/games/mahjong.py).
    pub fn encode(&self, seat: usize, focus: Option<usize>, own_turn: bool, focus_actor: Option<usize>, out: &mut ObsBuf) {
        let (shimo, toi, kami) = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4);
        let dora_kinds: HashSet<usize> = self.dora.iter().map(|&m| dora_from_marker(m)).collect();
        let mut marker = [0i64; NK];
        for &m in &self.dora {
            marker[m] += 1;
        }
        let q = |x: i64| x.clamp(0, 255) as u8;
        for k in 0..NK {
            let mut row = [0u8; 16];
            row[self.hands[seat][k].clamp(0, 4) as usize] = 4;
            row[5] = q(self.melds[seat][k]);
            row[6] = q(self.melds[shimo][k] + self.melds[toi][k] + self.melds[kami][k]);
            row[7] = q(self.disc[seat][k]);
            row[8] = q(self.disc[shimo][k]);
            row[9] = q(self.disc[toi][k]);
            row[10] = q(self.disc[kami][k]);
            row[11] = 4 * dora_kinds.contains(&k) as u8;
            let visible = self.hands[seat][k] + marker[k] + (0..4).map(|p| self.melds[p][k] + self.disc[p][k]).sum::<i64>();
            row[12] = q(visible.min(4));
            row[13] = 4 * is_terminal_or_honor(k) as u8;
            row[14] = 4 * self.reds[seat].contains(&k) as u8;
            row[15] = 4 * (Some(k) == focus) as u8;
            out.tiles_q.extend_from_slice(&row);
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

#[cfg(test)]
mod tests {
    use super::super::rules::tests::hand;
    use super::*;
    use serde_json::json;

    /// A kyoku whose seat 0 holds `h0` (the other seats: filler hands).
    pub(crate) fn kyoku_with(h0: [i64; NK]) -> Kyoku {
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

    fn claim(kind: usize, discarder: usize) -> Ctx {
        Ctx::Claim { kind, discarder, chankan: false }
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
        // closed tsumo shape → hora (menzen tsumo is a yaku)
        let k = kyoku_with(hand("123m456p789s11122z"));
        assert!(k.legal(0, &Ctx::Draw { drawn: 27 }).types[HORA as usize]);
        // kyushu kyuhai: ≥ 9 distinct terminal/honor kinds before the first discard / any call
        let mut k = kyoku_with(hand("19m19p19s1234z258m"));
        assert!(k.legal(0, &Ctx::Draw { drawn: 0 }).types[RYUKYOKU as usize]);
        k.any_call = true;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 0 }).types[RYUKYOKU as usize]);
        let k = kyoku_with(hand("19m19p19s12z225588m"));
        assert!(!k.legal(0, &Ctx::Draw { drawn: 0 }).types[RYUKYOKU as usize]); // 8 kinds
        // riichi needs a real wait: discarding the drawn chun from 1111m 456p 789s 123s C
        // would wait on a fifth 1m only (純カラ); discarding a 1m (tanki on C) is tenpai
        let k = kyoku_with(hand("1111m456p789s123s7z"));
        assert_eq!(kinds(&k.legal(0, &Ctx::Draw { drawn: 33 }).tiles[RIICHI as usize]), vec![0]);
    }

    #[test]
    fn open_tsumo_needs_a_yaku() {
        // chi 456p; 13m 678s 111p 99p + drawn 2m: no yaku by tsumo on an open hand
        let mut k = kyoku_with(hand("123m678s11199p"));
        k.meld_list[0].push(Meld { kind: MeldKind::Chi, tiles: vec![12, 13, 14] });
        k.meld_n[0] = 1;
        k.open[0] = true;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 1 }).types[HORA as usize]);
        k.wall = 0; // haitei
        assert!(k.legal(0, &Ctx::Draw { drawn: 1 }).types[HORA as usize]);
        k.wall = 5;
        k.rinshan_draw = true; // rinshan kaihou
        assert!(k.legal(0, &Ctx::Draw { drawn: 1 }).types[HORA as usize]);
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
        k.wall = 10;
        k.kan_n = [1, 1, 1, 1]; // four kans on the table: no fifth
        assert!(!k.legal(0, &Ctx::Draw { drawn: 27 }).types[KAN as usize]);
        // daiminkan claim needs three in hand
        let k = kyoku_with(hand("111m456p789s1122z"));
        let l = k.legal(0, &claim(0, 2));
        assert!(l.types[KAN as usize] && l.types[PON as usize] && l.types[PASS as usize]);
        assert!(!l.types[CHI as usize] && !l.types[DAHAI as usize]); // seat 0 is not shimocha of 2
        assert_eq!(kinds(&l.tiles[KAN as usize]), vec![0]);
        assert_eq!(kinds(&l.tiles[PON as usize]), vec![0]);
    }

    #[test]
    fn riichi_ankan_keeps_the_waits() {
        // riichi on 111m 456p 23s 789s EE (waits 1s/4s): drawing the 4th 1m, the ankan
        // leaves 23s + EE → waits 1s/4s: allowed
        let mut k = kyoku_with(hand("111m456p23789s11z"));
        k.riichi[0] = true;
        k.hands[0][0] += 1;
        assert_eq!(kinds(&k.legal(0, &Ctx::Draw { drawn: 0 }).tiles[KAN as usize]), vec![0]);
        // 111m 23m EE: waits 1m 4m E; the ankan would drop the E wait: refused
        let mut k = kyoku_with(hand("11123m456p789s11z"));
        k.riichi[0] = true;
        k.hands[0][0] += 1;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 0 }).types[KAN as usize]);
        // 1112345m 456p 789s: waits 2m 3m 5m 6m; ankan of 1m leaves 2345m → waits 2m 5m: refused
        let mut k = kyoku_with(hand("1112345m456p789s"));
        k.riichi[0] = true;
        k.hands[0][0] += 1;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 0 }).types[KAN as usize]);
        // no okurikan: four 9s in hand but the drawn tile is something else
        let mut k = kyoku_with(hand("9999s23m456p789m1z"));
        k.riichi[0] = true;
        k.hands[0][27] += 1;
        assert!(!k.legal(0, &Ctx::Draw { drawn: 27 }).types[KAN as usize]);
    }

    #[test]
    fn claim_masks() {
        // seat 0 is shimocha of seat 3; discard 4m with 2m 3m 5m 6m in hand
        let k = kyoku_with(hand("2356m456p789s112z"));
        let l = k.legal(0, &claim(3, 3));
        assert!(l.types[CHI as usize] && !l.types[PON as usize] && !l.types[KAN as usize]);
        assert_eq!(kinds(&l.tiles[CHI as usize]), vec![1, 2, 3]); // 234 / 345 / 456
        // a riichi seat: pass (or ron) only
        let mut k2 = kyoku_with(hand("2356m456p789s112z"));
        k2.riichi[0] = true;
        let l = k2.legal(0, &claim(3, 3));
        assert!(!l.types[CHI as usize] && l.types[PASS as usize] && !l.has_claim());
        // ron: 23m 456p 789s 111z 22z (yakuhai-less? E triplet is the round wind: a yaku)
        let k = kyoku_with(hand("23m456p789s11122z"));
        let l = k.legal(0, &claim(0, 1));
        assert!(l.types[HORA as usize] && !l.types[CHI as usize] && l.has_claim());
        // houtei: no calls on the last discard, a ron is still possible
        let mut k = kyoku_with(hand("2356m456p789s112z"));
        k.wall = 0;
        assert!(!k.legal(0, &claim(3, 3)).has_claim());
    }

    #[test]
    fn calls_leave_a_legal_discard() {
        // 4566m + three melds: chi 3m with 45m would leave 66m, all 喰い替え (6m is the suji
        // of a run called at its low end) → no chi
        let mut k = kyoku_with(hand("4566m"));
        k.meld_n[0] = 3;
        let l = k.legal(0, &claim(2, 3));
        assert!(!l.types[CHI as usize], "{:?}", kinds(&l.tiles[CHI as usize]));
        // pon of 5m from 5555m would leave only 5m, the called kind: refused (the kan is fine)
        let mut k = kyoku_with(hand("5555m"));
        k.meld_n[0] = 3;
        let l = k.legal(0, &claim(4, 2));
        assert!(!l.types[PON as usize] && l.types[KAN as usize]);
        // a lone tanki tile (four melds) cannot call at all
        let mut k = kyoku_with(hand("5m"));
        k.meld_n[0] = 4;
        assert!(!k.legal(0, &claim(4, 3)).types[CHI as usize]);
    }

    #[test]
    fn furiten_blocks_ron() {
        // 23m 456p 789s 111z 22z waits 1m / 4m; East triplet = round-wind yakuhai
        let mut k = kyoku_with(hand("23m456p789s11122z"));
        assert!(k.ron_legal(0, 0, false));
        k.river[0][3] = true; // 4m among the own discards → furiten on both waits
        assert!(!k.ron_legal(0, 0, false) && !k.ron_legal(0, 3, false));
        k.river[0][3] = false;
        k.note_pass(0, 3); // passed on 4m → same-turn furiten
        assert!(k.furiten_temp[0] && !k.ron_legal(0, 0, false));
        k.hands[0][9] += 1;
        k.discard(0, "1p").ok().unwrap(); // own discard lifts it
        assert!(k.ron_legal(0, 0, false));
        // in riichi a passed winning tile is furiten for good
        k.riichi[0] = true;
        k.note_pass(0, 0);
        k.hands[0][9] += 1;
        k.discard(0, "1p").ok().unwrap();
        assert!(!k.ron_legal(0, 3, false));
        // a tile that does not complete the hand changes nothing
        let mut k = kyoku_with(hand("23m456p789s11122z"));
        k.note_pass(0, 5);
        assert!(!k.furiten_temp[0]);
    }

    #[test]
    fn yakuless_ron_is_illegal_but_still_furiten() {
        // closed 13m 456p 678s 111p 99p waits 2m: no yaku by ron
        let mut k = kyoku_with(hand("13m456p678s11199p"));
        k.bakaze = "S".into();
        assert!(!k.ron_legal(0, 1, false));
        assert!(!k.legal(0, &claim(1, 2)).has_claim());
        // chankan is a yaku
        assert!(k.ron_legal(0, 1, true));
        // houtei too
        k.wall = 0;
        assert!(k.ron_legal(0, 1, false));
        k.wall = 10;
        // letting the 2m pass (no yaku) still makes the seat furiten until its discard
        k.note_pass(0, 1);
        assert!(k.furiten_temp[0] && !k.ron_legal(0, 1, true));
    }

    #[test]
    fn abortive_draws_leave_only_ron() {
        let mut k = kyoku_with(hand("2356m456p789s112z"));
        k.kan_n = [2, 2, 0, 0]; // four kans by two seats: the next discard allows a ron only
        assert!(!k.legal(0, &claim(3, 3)).has_claim());
        k.kan_n = [4, 0, 0, 0]; // one seat's four kans (suukantsu tenpai): calls stay legal
        assert!(k.legal(0, &claim(3, 3)).types[CHI as usize]);
        // but no fifth kan
        let mut k = kyoku_with(hand("111m456p789s1122z"));
        k.kan_n = [0, 4, 0, 0];
        let l = k.legal(0, &claim(0, 2));
        assert!(l.types[PON as usize] && !l.types[KAN as usize]);
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
}
