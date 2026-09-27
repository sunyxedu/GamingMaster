//! HUNL frames: a custom schema with the CANDIDATE layout (`candidate_abstract.v1`) —
//! the same obs tensors, legality key and meta dims as `FrameSpec::candidate`, so the
//! generic candidate adapter (model/games/candidate.py) and train_rl_joint's
//! `CandidateSpec` play it, but the candidates are an ABSTRACTION of the exact legal
//! amounts instead of the legal ids themselves.
//!
//! Candidates (slot order; a slot is available when its action is legal and its amount
//! differs from every lower available slot's):
//!
//! ```text
//! 0 fold            (facing a bet)
//! 1 check / call
//! 2 all-in          raise to the stack
//! 3 min-raise       raise to min_raise_to (when below the stack)
//! 4.. pot fractions raise to bet + f * (pot after calling), f in RAISE_FRACTIONS,
//!                   rounded to a chip; strictly between min-raise and all-in
//! ```
//!
//! The simulator plays a candidate's amount ([`Candidates::decode`]); the IL target of a
//! recorded action ([`Candidates::nearest`]) is its candidate — a raise maps to the
//! available raise slot nearest in log(raise increment) (ties: the lower slot), so
//! every candidate's own amount maps back to itself (the conformance battery's round
//! trip). The exact action is kept for a later continuous size head:
//!
//! ```text
//! target  action_idx [] i64   candidate index (the action key)
//!         raise_to   [] i64   exact raise-to amount (total commitment, chips); 0 = no raise
//!         raise_frac [] f32   exact raise increment / pot after calling; 0 = no raise
//! ```
//!
//! Observation (the acting seat's view; `L(x) = ln(1 + x / big blind) / ln(1001)`,
//! amounts otherwise relative to the pot or the stack — nothing depends on the chip
//! scale, so one network plays every blind / stack config):
//!
//! ```text
//! entities [9, 18]  slot 0-1 own hole cards (type 1), sorted high to low
//!                   slot 2-4 flop (sorted), 5 turn, 6 river (type 2; absent until dealt)
//!                     card: rank one-hot (13), suit one-hot (4), rank / 12
//!                   slot 7 me (type 3), 8 opponent (type 4): see PLAYER_* below
//! globals  [48]     round, pot, price, stack-to-pot, raise bounds, per-round
//!                   investments and raise counts, own made-hand category and draws
//!                   (from own + board cards only), board texture — see GLOBAL_*
//! actions  [15, 12] per candidate: kind, all-in, raise fraction, chips added, ...
//! action_mask [15]  available candidates (= legal.action)
//! ```
//!
//! Never encoded: the opponent's hole cards (the ACPC logs record them; the
//! privacy battery re-deals them), the board before it is dealt (causality battery).

use super::eval::{self, rank_of, suit_of};
use super::{Hunl, HunlConfig, Kind, CALL, FOLD};
use crate::frame::FrameMut;
use crate::game::{FrameSpec, Seat};
use crate::schema::{BufMut, TensorSpec, TensorsMut};
use serde_json::json;

/// Pot fractions of the fractional raise candidates.
pub const RAISE_FRACTIONS: [f64; 11] = [0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0];
pub const SLOT_FOLD: usize = 0;
pub const SLOT_CALL: usize = 1;
pub const SLOT_ALL_IN: usize = 2;
pub const SLOT_MIN_RAISE: usize = 3;
pub const SLOT_FRAC0: usize = 4;
pub const N_CANDIDATES: usize = SLOT_FRAC0 + RAISE_FRACTIONS.len();

pub const MAX_ENTITIES: usize = 9;
pub const ENTITY_FEATURES: usize = 18;
/// Entity type ids: 1 own hole card, 2 board card, 3 me, 4 opponent (0 = absent).
pub const ENTITY_TYPES: usize = 5;
pub const GLOBAL_FEATURES: usize = 48;
pub const ACTION_FEATURES: usize = 12;
pub const SCHEMA: &str = "candidate_abstract.v1";

const SLOT_ME: usize = 7;
const SLOT_OPP: usize = 8;
const T_HOLE: i32 = 1;
const T_BOARD: i32 = 2;
const T_ME: i32 = 3;
const T_OPP: i32 = 4;

// obs / legal / target tensor indices (see `spec`; entities, entity_mask and
// entity_type are 0-2)
const OBS_GLOBALS: usize = 3;
const OBS_ACTIONS: usize = 4;
const OBS_ACTION_MASK: usize = 5;
const LEGAL_ACTION: usize = 0;
const TGT_ACTION_IDX: usize = 0;
const TGT_RAISE_TO: usize = 1;
const TGT_RAISE_FRAC: usize = 2;

// player entity features
const PLAYER_BUTTON: usize = 0;
const PLAYER_STACK: usize = 1;
const PLAYER_SPENT: usize = 2;
const PLAYER_L_STACK: usize = 3;
const PLAYER_L_SPENT: usize = 4;
const PLAYER_STREET_BET: usize = 5;
const PLAYER_L_STREET_BET: usize = 6;
const PLAYER_ALL_IN: usize = 7;
const PLAYER_ACTED: usize = 8;
const PLAYER_LAST_CHECK: usize = 9;
const PLAYER_LAST_CALL: usize = 10;
const PLAYER_LAST_RAISE: usize = 11;
const PLAYER_ROUND_RAISES: usize = 12;
const PLAYER_HAND_RAISES: usize = 13;
const PLAYER_AGGRESSOR: usize = 14;
const PLAYER_PREV_AGGRESSOR: usize = 15;
const PLAYER_VPIP: usize = 16;
const PLAYER_LAST_RAISE_FRAC: usize = 17;

// globals
const GLOBAL_ROUND: usize = 0; // 4 one-hot
const GLOBAL_POT: usize = 4;
const GLOBAL_L_POT: usize = 5;
const GLOBAL_PRICE: usize = 6;
const GLOBAL_L_TO_CALL: usize = 7;
const GLOBAL_SPR: usize = 8;
const GLOBAL_L_EFFECTIVE: usize = 9;
const GLOBAL_MIN_RAISE_FRAC: usize = 10;
const GLOBAL_CAN_RAISE: usize = 11;
const GLOBAL_CAN_FOLD: usize = 12;
const GLOBAL_BUTTON: usize = 13;
const GLOBAL_L_STACK: usize = 14;
const GLOBAL_ROUND_ACTIONS: usize = 15;
const GLOBAL_ROUND_RAISES: usize = 16;
const GLOBAL_OPP_ALL_IN: usize = 17;
const GLOBAL_INVESTED: usize = 18; // 8: (me, opp) per round, / pot
const GLOBAL_RAISES_BY_ROUND: usize = 26; // 4
const GLOBAL_CATEGORY: usize = 30; // 9 one-hot
const GLOBAL_POCKET_PAIR: usize = 39;
const GLOBAL_SUITED: usize = 40;
const GLOBAL_GAP: usize = 41;
const GLOBAL_HIGH: usize = 42;
const GLOBAL_LOW: usize = 43;
const GLOBAL_FLUSH_DRAW: usize = 44;
const GLOBAL_STRAIGHT_DRAW: usize = 45;
const GLOBAL_BOARD_PAIRED: usize = 46;
const GLOBAL_BOARD_SUITED: usize = 47;

// candidate features
const ACT_FOLD: usize = 0;
const ACT_CHECK: usize = 1;
const ACT_CALL: usize = 2;
const ACT_RAISE: usize = 3;
const ACT_ALL_IN: usize = 4;
const ACT_RAISE_FRAC: usize = 5;
const ACT_ADDED_SHARE: usize = 6;
const ACT_L_ADDED: usize = 7;
const ACT_SPENT_AFTER: usize = 8;
const ACT_SPR_AFTER: usize = 9;
const ACT_MIN_RAISE: usize = 10;
const ACT_L_RAISE_TO: usize = 11;

/// Names of the candidate slots (shard meta, logs).
pub fn candidate_names() -> Vec<String> {
    let mut v: Vec<String> = ["fold", "check_call", "all_in", "min_raise"].iter().map(|s| s.to_string()).collect();
    v.extend(RAISE_FRACTIONS.iter().map(|f| format!("pot_{f:.2}")));
    v
}

/// The frame schema (independent of the config: one network plays every variant).
pub fn spec(_cfg: &HunlConfig) -> FrameSpec {
    FrameSpec {
        n_seats: 2,
        hidden_info: true,
        simultaneous: false,
        max_decisions: 4 * super::MAX_ROUND_ACTIONS,
        schema: SCHEMA.into(),
        obs: vec![
            TensorSpec::f32("entities", &[MAX_ENTITIES, ENTITY_FEATURES]),
            TensorSpec::bool("entity_mask", &[MAX_ENTITIES]),
            TensorSpec::i32("entity_type", &[MAX_ENTITIES]),
            TensorSpec::f32("globals", &[GLOBAL_FEATURES]),
            TensorSpec::f32("actions", &[N_CANDIDATES, ACTION_FEATURES]),
            TensorSpec::bool("action_mask", &[N_CANDIDATES]),
        ],
        legal: vec![TensorSpec::bool("action", &[N_CANDIDATES])],
        target: vec![TensorSpec::i64("action_idx", &[]), TensorSpec::i64("raise_to", &[]), TensorSpec::f32("raise_frac", &[])],
        action_keys: vec!["action_idx".into()],
        meta: json!({
            "max_entities": MAX_ENTITIES,
            "entity_features": ENTITY_FEATURES,
            "entity_types": ENTITY_TYPES,
            "global_features": GLOBAL_FEATURES,
            "max_actions": N_CANDIDATES,
            "action_features": ACTION_FEATURES,
            "candidates": candidate_names(),
            "raise_fractions": RAISE_FRACTIONS.to_vec(),
        }),
        candidate: None,
    }
}

/// The candidate table of the player to act: availability and the canonical action of
/// every slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidates {
    pub avail: [bool; N_CANDIDATES],
    pub action: [u32; N_CANDIDATES],
    /// The bet to match (raise increments are measured from it).
    pub faced: u32,
}

impl Candidates {
    pub fn of(g: &Hunl, seat: Seat) -> Candidates {
        let mut c = Candidates { avail: [false; N_CANDIDATES], action: [0; N_CANDIDATES], faced: g.max_spent() };
        c.avail[SLOT_FOLD] = g.fold_legal(seat);
        c.action[SLOT_FOLD] = FOLD;
        c.avail[SLOT_CALL] = true;
        c.action[SLOT_CALL] = CALL;
        if let Some((lo, hi)) = g.raise_range(seat) {
            c.avail[SLOT_ALL_IN] = true;
            c.action[SLOT_ALL_IN] = hi;
            if lo < hi {
                c.avail[SLOT_MIN_RAISE] = true;
                c.action[SLOT_MIN_RAISE] = lo;
            }
            let pot_after_call = (g.pot() + g.to_call(seat)) as f64;
            let mut last = lo;
            for (k, &f) in RAISE_FRACTIONS.iter().enumerate() {
                let r = (g.max_spent() as f64 + f * pot_after_call).round();
                if r > last as f64 && r < hi as f64 {
                    c.avail[SLOT_FRAC0 + k] = true;
                    c.action[SLOT_FRAC0 + k] = r as u32;
                    last = r as u32;
                }
            }
        }
        c
    }

    /// The canonical action of slot `k` (None: not an available slot).
    pub fn decode(&self, k: i64) -> Option<u32> {
        let k = usize::try_from(k).ok()?;
        (k < N_CANDIDATES && self.avail[k]).then(|| self.action[k])
    }

    /// The candidate of a legal action: fold / call to their slots, a raise to the
    /// available raise slot nearest in log(raise increment) (ties: the lower slot).
    pub fn nearest(&self, action: u32) -> usize {
        match action {
            FOLD => SLOT_FOLD,
            CALL => SLOT_CALL,
            r => {
                let x = ((r - self.faced) as f64).ln();
                let mut best = (f64::INFINITY, SLOT_CALL);
                for k in SLOT_ALL_IN..N_CANDIDATES {
                    if self.avail[k] {
                        let d = (((self.action[k] - self.faced) as f64).ln() - x).abs();
                        if d < best.0 {
                            best = (d, k);
                        }
                    }
                }
                best.1
            }
        }
    }
}

/// `ln(1 + chips / big blind) / ln(1001)`: amounts on a log big-blind scale (1 at 1000
/// big blinds).
fn lbb(chips: u32, bb: u32) -> f32 {
    ((1.0 + chips as f64 / bb as f64).ln() / 1001f64.ln()) as f32
}

fn ratio(a: u32, b: u32) -> f32 {
    if b == 0 { 0.0 } else { (a as f64 / b as f64) as f32 }
}

/// Mark entity `slot` present with type `t`; its feature slice.
fn entity<'a>(ents: &'a mut [f32], mask: &mut [u8], types: &mut [i32], slot: usize, t: i32) -> &'a mut [f32] {
    mask[slot] = 1;
    types[slot] = t;
    &mut ents[slot * ENTITY_FEATURES..(slot + 1) * ENTITY_FEATURES]
}

fn write_card(f: &mut [f32], card: u8) {
    f[rank_of(card) as usize] = 1.0;
    f[13 + suit_of(card) as usize] = 1.0;
    f[17] = rank_of(card) as f32 / 12.0;
}

/// Cards sorted high to low (rank, then suit).
fn sorted_desc(cards: &[u8]) -> Vec<u8> {
    let mut v = cards.to_vec();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

/// Per-player facts from the public betting history.
#[derive(Default)]
struct PlayerHist {
    acted: bool,
    last: Option<(Kind, bool)>, // (kind, it was a check)
    round_raises: u32,
    hand_raises: u32,
    vpip: bool,
    last_raise_frac: f32,
}

fn history(g: &Hunl) -> ([PlayerHist; 2], Option<u8>, Option<u8>) {
    let mut h: [PlayerHist; 2] = Default::default();
    let round = g.round() as u8;
    let mut aggressor = None;
    let mut prev_aggressor = None;
    for a in g.acts() {
        let p = &mut h[a.seat as usize];
        let check = a.kind == Kind::Call && a.after == a.before;
        if a.round == round {
            p.acted = true;
            p.last = Some((a.kind, check));
            if a.kind == Kind::Raise {
                p.round_raises += 1;
            }
        }
        if a.kind == Kind::Raise {
            p.hand_raises += 1;
            aggressor = Some(a.seat);
            if a.round + 1 == round {
                prev_aggressor = Some(a.seat);
            }
            let pot_after_call = a.pot + (a.faced - a.before);
            p.last_raise_frac = ratio(a.after - a.faced, pot_after_call);
        }
        if a.round == 0 && a.after > a.before {
            p.vpip = true;
        }
    }
    (h, aggressor, prev_aggressor)
}

/// Write `seat`'s frame (obs + legal) — see the module docs.
pub fn encode_frame(g: &Hunl, seat: Seat, f: &mut FrameMut<'_>) {
    let cfg = g.config();
    let (bb, s0) = (cfg.big_blind, cfg.stack);
    let opp = 1 - seat;
    let spent = g.spent();
    let pot = g.pot();
    let (hist, aggressor, prev_aggressor) = history(g);
    let cands = Candidates::of(g, seat);
    let hole = sorted_desc(g.hole(seat));
    let board = g.board();

    // ---- entities
    {
        let [BufMut::F32(ents), BufMut::Bool(mask), BufMut::I32(types), ..] = f.obs.bufs_mut() else {
            unreachable!("hunl obs layout")
        };
        for (i, &c) in hole.iter().enumerate() {
            write_card(entity(ents, mask, types, i, T_HOLE), c);
        }
        let mut shown = sorted_desc(&board[..board.len().min(3)]);
        shown.extend_from_slice(&board[board.len().min(3)..]);
        for (i, &c) in shown.iter().enumerate() {
            write_card(entity(ents, mask, types, 2 + i, T_BOARD), c);
        }
        for (slot, t, p) in [(SLOT_ME, T_ME, seat), (SLOT_OPP, T_OPP, opp)] {
            let street = spent[p] - g.round_base().min(spent[p]);
            let h = &hist[p];
            let e = entity(ents, mask, types, slot, t);
            e[PLAYER_BUTTON] = (p == 1) as u8 as f32;
            e[PLAYER_STACK] = ratio(s0 - spent[p], s0);
            e[PLAYER_SPENT] = ratio(spent[p], s0);
            e[PLAYER_L_STACK] = lbb(s0 - spent[p], bb);
            e[PLAYER_L_SPENT] = lbb(spent[p], bb);
            e[PLAYER_STREET_BET] = ratio(street, pot);
            e[PLAYER_L_STREET_BET] = lbb(street, bb);
            e[PLAYER_ALL_IN] = (spent[p] >= s0) as u8 as f32;
            e[PLAYER_ACTED] = h.acted as u8 as f32;
            match h.last {
                Some((Kind::Call, true)) => e[PLAYER_LAST_CHECK] = 1.0,
                Some((Kind::Call, false)) => e[PLAYER_LAST_CALL] = 1.0,
                Some((Kind::Raise, _)) => e[PLAYER_LAST_RAISE] = 1.0,
                _ => {}
            }
            e[PLAYER_ROUND_RAISES] = h.round_raises.min(4) as f32 / 4.0;
            e[PLAYER_HAND_RAISES] = h.hand_raises.min(8) as f32 / 8.0;
            e[PLAYER_AGGRESSOR] = (aggressor == Some(p as u8)) as u8 as f32;
            e[PLAYER_PREV_AGGRESSOR] = (prev_aggressor == Some(p as u8)) as u8 as f32;
            e[PLAYER_VPIP] = h.vpip as u8 as f32;
            e[PLAYER_LAST_RAISE_FRAC] = h.last_raise_frac;
        }
    }

    // ---- globals
    {
        let gl = f.obs.f32(OBS_GLOBALS);
        let round = g.round();
        gl[GLOBAL_ROUND + round] = 1.0;
        let to_call = g.to_call(seat);
        let eff = (s0 - spent[seat]).min(s0 - spent[opp]);
        gl[GLOBAL_POT] = ratio(pot, 2 * s0);
        gl[GLOBAL_L_POT] = lbb(pot, bb);
        gl[GLOBAL_PRICE] = ratio(to_call, pot + to_call);
        gl[GLOBAL_L_TO_CALL] = lbb(to_call, bb);
        gl[GLOBAL_SPR] = ratio(eff, eff + pot);
        gl[GLOBAL_L_EFFECTIVE] = lbb(eff, bb);
        if let Some((lo, _)) = g.raise_range(seat) {
            gl[GLOBAL_MIN_RAISE_FRAC] = ratio(lo - g.max_spent(), pot + to_call);
            gl[GLOBAL_CAN_RAISE] = 1.0;
        }
        gl[GLOBAL_CAN_FOLD] = g.fold_legal(seat) as u8 as f32;
        gl[GLOBAL_BUTTON] = (seat == 1) as u8 as f32;
        gl[GLOBAL_L_STACK] = lbb(s0, bb);
        let mut per_round = [[0u32; 2]; 4];
        let mut raises = [0u32; 4];
        for a in g.acts() {
            per_round[a.round as usize][a.seat as usize] += a.after - a.before;
            raises[a.round as usize] += (a.kind == Kind::Raise) as u32;
        }
        // the blinds count as preflop investment
        per_round[0][0] += cfg.big_blind;
        per_round[0][1] += cfg.small_blind;
        let round_acts = g.acts().iter().filter(|a| a.round as usize == round).count();
        gl[GLOBAL_ROUND_ACTIONS] = round_acts.min(8) as f32 / 8.0;
        gl[GLOBAL_ROUND_RAISES] = raises[round].min(4) as f32 / 4.0;
        gl[GLOBAL_OPP_ALL_IN] = (spent[opp] >= s0) as u8 as f32;
        for r in 0..4 {
            gl[GLOBAL_INVESTED + 2 * r] = ratio(per_round[r][seat], pot);
            gl[GLOBAL_INVESTED + 2 * r + 1] = ratio(per_round[r][opp], pot);
            gl[GLOBAL_RAISES_BY_ROUND + r] = raises[r].min(4) as f32 / 4.0;
        }
        // own cards (+ the public board) only
        if hole.len() == 2 {
            let (hi, lo) = (rank_of(hole[0]), rank_of(hole[1]));
            gl[GLOBAL_POCKET_PAIR] = (hi == lo) as u8 as f32;
            gl[GLOBAL_SUITED] = (suit_of(hole[0]) == suit_of(hole[1])) as u8 as f32;
            gl[GLOBAL_GAP] = (hi - lo) as f32 / 12.0;
            gl[GLOBAL_HIGH] = hi as f32 / 12.0;
            gl[GLOBAL_LOW] = lo as f32 / 12.0;
            if board.len() >= 3 {
                let mut cards = hole.clone();
                cards.extend_from_slice(board);
                let v = eval::eval(&cards);
                let cat = eval::category(v);
                gl[GLOBAL_CATEGORY + cat as usize] = 1.0;
                if board.len() < 5 {
                    let mut suits = [0u32; 4];
                    let mut ranks = 0u16;
                    for &c in &cards {
                        suits[suit_of(c) as usize] += 1;
                        ranks |= 1 << rank_of(c);
                    }
                    let flush_draw = cat < eval::FLUSH && suits.contains(&4);
                    gl[GLOBAL_FLUSH_DRAW] = flush_draw as u8 as f32;
                    gl[GLOBAL_STRAIGHT_DRAW] = (cat < eval::STRAIGHT && straight_draw(ranks)) as u8 as f32;
                }
            }
        }
        if board.len() >= 2 {
            let mut rk = [0u8; 13];
            let mut su = [0u8; 4];
            for &c in board {
                rk[rank_of(c) as usize] += 1;
                su[suit_of(c) as usize] += 1;
            }
            gl[GLOBAL_BOARD_PAIRED] = rk.iter().any(|&n| n >= 2) as u8 as f32;
            gl[GLOBAL_BOARD_SUITED] = su.iter().any(|&n| n >= 3) as u8 as f32;
        }
    }

    // ---- candidates
    let pot_after_call = pot + g.to_call(seat);
    for k in 0..N_CANDIDATES {
        if !cands.avail[k] {
            continue;
        }
        let a = cands.action[k];
        let added = g.chips_added(seat, a);
        let spent_after = spent[seat] + added;
        {
            let feat = &mut f.obs.f32(OBS_ACTIONS)[k * ACTION_FEATURES..(k + 1) * ACTION_FEATURES];
            match a {
                FOLD => feat[ACT_FOLD] = 1.0,
                CALL if added == 0 => feat[ACT_CHECK] = 1.0,
                CALL => feat[ACT_CALL] = 1.0,
                r => {
                    feat[ACT_RAISE] = 1.0;
                    feat[ACT_RAISE_FRAC] = ratio(r - g.max_spent(), pot_after_call);
                    feat[ACT_MIN_RAISE] = (g.raise_range(seat).map(|x| x.0) == Some(r)) as u8 as f32;
                    feat[ACT_L_RAISE_TO] = lbb(r, bb);
                }
            }
            feat[ACT_ALL_IN] = (a != FOLD && spent_after >= s0) as u8 as f32;
            feat[ACT_ADDED_SHARE] = ratio(added, pot + added);
            feat[ACT_L_ADDED] = lbb(added, bb);
            feat[ACT_SPENT_AFTER] = ratio(spent_after, s0);
            if a != FOLD {
                let eff_after = (s0 - spent_after).min(s0 - spent[opp]);
                feat[ACT_SPR_AFTER] = ratio(eff_after, eff_after + pot + added);
            }
        }
        f.obs.bool(OBS_ACTION_MASK)[k] = 1;
        f.legal.bool(LEGAL_ACTION)[k] = 1;
    }
}

/// Some window of five consecutive ranks (the ace also low) holds exactly four of the
/// ranks in `mask` (an open-ended or gutshot straight draw).
fn straight_draw(mask: u16) -> bool {
    let m = mask as u32;
    let ext = (m << 1) | ((m >> 12) & 1);
    (0..=9).any(|lo| (ext >> lo & 0b11111).count_ones() == 4)
}

/// The targets of `seat` taking the legal `action`: its candidate + the exact amount.
pub fn encode_target(g: &Hunl, seat: Seat, action: u32, t: &mut TensorsMut<'_>) {
    let c = Candidates::of(g, seat);
    t.i64(TGT_ACTION_IDX)[0] = c.nearest(action) as i64;
    if action >= 2 {
        t.i64(TGT_RAISE_TO)[0] = action as i64;
        let pot_after_call = g.pot() + g.to_call(seat);
        t.f32(TGT_RAISE_FRAC)[0] = ratio(action - g.max_spent(), pot_after_call);
    }
}
