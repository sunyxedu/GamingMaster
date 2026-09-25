//! Turn one replay into two decision-frame sequences (one per player), in
//! each player's own frame of reference (own half = rows 0..16, own back line
//! = row 0, x mirrored for the opponent so left/right stay egocentric).
//!
//! A frame every `step_ticks` (default 10 = 0.5 s) carries everything the
//! player could know without seeing units: exact own elixir, the opponent's
//! elixir (also exact — it is a function of their plays), the hand and next
//! card (reconstructed), the last `recent` plays of both sides with ages, the
//! match phase, and the label: the first own action inside the frame window
//! (wait / card slot + cell / champion ability).

use crate::catalog::{Catalog, Timeline};
use crate::elixir::ElixirSim;
use crate::hand::{infer_initial_hand, HandState};
use crate::replay::{Event, EventKind, Replay, SideId};
use serde::Serialize;

pub const HAND: usize = 4;
pub const GRID_W: i8 = 18;
pub const GRID_H: i8 = 32;

#[derive(Debug, Clone, Copy)]
pub struct FrameConfig {
    pub step_ticks: u32,
    pub recent: usize,
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self { step_ticks: 10, recent: 8 }
    }
}

/// Column-major storage for one player's sequence; concatenated into shards.
#[derive(Debug, Default, Clone)]
pub struct FrameColumns {
    pub tick: Vec<i32>,
    pub time_s: Vec<f32>,
    pub phase: Vec<u8>,
    pub overtime: Vec<u8>,
    pub elixir: Vec<f32>,
    pub opp_elixir: Vec<f32>,
    pub hand_ids: Vec<u16>,  // [T,4]
    pub hand_form: Vec<u8>,  // [T,4]
    pub hand_kind: Vec<u8>,  // [T,4]
    pub hand_cost: Vec<f32>, // [T,4]
    pub hand_prob: Vec<f32>, // [T,4] P(card really in hand) over consistent cycle states
    pub next_card: Vec<u16>,
    pub recent_side: Vec<u8>,  // [T,K] 0 own 1 opp 255 none
    pub recent_card: Vec<u16>, // [T,K]
    pub recent_x: Vec<i8>,     // [T,K]
    pub recent_y: Vec<i8>,     // [T,K]
    pub recent_age: Vec<i32>,  // [T,K] ticks since the play; -1 none
    pub waited: Vec<u8>,
    pub card_slot: Vec<i8>, // -1 when waited
    pub place_x: Vec<i8>,
    pub place_y: Vec<i8>,
    pub ability: Vec<u8>,
    pub ability_card: Vec<u16>,
}

impl FrameColumns {
    pub fn len(&self) -> usize {
        self.tick.len()
    }

    pub fn append(&mut self, o: &FrameColumns) {
        macro_rules! ext {
            ($($f:ident),*) => { $( self.$f.extend_from_slice(&o.$f); )* };
        }
        ext!(tick, time_s, phase, overtime, elixir, opp_elixir, hand_ids, hand_form, hand_kind, hand_cost, hand_prob, next_card,
             recent_side, recent_card, recent_x, recent_y, recent_age, waited, card_slot, place_x, place_y, ability, ability_card);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleMeta {
    pub tag: String,
    pub side: &'static str,
    pub start: usize,
    pub length: usize,
    pub result: i8,
    pub own_crowns: u8,
    pub opp_crowns: u8,
    pub battle_type: String,
    pub game_mode: String,
    pub deck_ids: Vec<u16>,
    pub deck_forms: Vec<u8>,
    pub deck_levels: Vec<u8>,
    pub tower_key: String,
    pub opp_deck_ids: Vec<u16>,
    pub duration_s: f64,
    pub n_plays: usize,
    pub n_abilities: usize,
    pub hand_consistent_states: usize,
    pub skipped_plays: usize,
    pub elixir_violations: u32,
    pub opp_elixir_violations: u32,
    pub elixir_leaked: f32,
}

pub struct Sample {
    pub meta: SampleMeta,
    pub cols: FrameColumns,
}

fn to_own_frame(side: SideId, gx: i8, gy: i8) -> (i8, i8) {
    match side {
        SideId::Team => (gx, gy),
        SideId::Opp => (GRID_W - 1 - gx, GRID_H - 1 - gy),
    }
}

/// Build the sequence for `me`.
pub fn build_side(r: &Replay, me: SideId, cat: &Catalog, tl: &Timeline, cfg: &FrameConfig) -> Option<Sample> {
    let (own, other) = match me {
        SideId::Team => (&r.team, &r.opp),
        SideId::Opp => (&r.opp, &r.team),
    };
    if own.deck.len() != 8 {
        return None;
    }
    let deck_ids: Vec<u16> = own.deck.iter().map(|c| c.id).collect();
    let form_of = |id: u16| own.deck.iter().find(|c| c.id == id).map(|c| c.form).unwrap_or(0);
    let own_plays: Vec<u16> = r
        .events
        .iter()
        .filter(|e| e.side == me)
        .filter_map(|e| match e.kind {
            EventKind::Play { card, .. } => Some(card),
            _ => None,
        })
        .collect();
    let inf = infer_initial_hand(&deck_ids, &own_plays)?;
    let tps = tl.ticks_per_second();
    let total_ticks = (r.duration_s * tps).ceil() as u32;
    let n_frames = (total_ticks / cfg.step_ticks) as usize + 1;

    let mut hand: HandState = inf.initial.clone();
    let mut my_elixir = ElixirSim::new(tl);
    let mut opp_elixir = ElixirSim::new(tl);
    let mut cols = FrameColumns::default();
    let mut ev_i = 0usize; // events consumed (tick < frame tick)
    let mut recent: Vec<&Event> = Vec::new();
    let mut n_plays = 0usize;
    let mut n_in_deck_plays = 0usize;
    let mut n_abilities = 0usize;
    let evs = &r.events;

    for f in 0..n_frames {
        let t = (f as u32) * cfg.step_ticks;
        let t_s = t as f64 / tps;
        // apply every event strictly before this frame
        while ev_i < evs.len() && evs[ev_i].tick < t {
            let e = &evs[ev_i];
            let e_s = e.tick as f64 / tps;
            match e.kind {
                EventKind::Play { card, .. } => {
                    let cost = cat.cost(card) as f64;
                    if e.side == me {
                        my_elixir.advance(e_s);
                        my_elixir.spend(cost);
                        if hand.play(card).is_some() {
                            n_in_deck_plays += 1;
                        }
                        n_plays += 1;
                    } else {
                        opp_elixir.advance(e_s);
                        opp_elixir.spend(cost);
                    }
                    recent.push(e);
                }
                EventKind::Ability { card } => {
                    // champion abilities cost elixir too (catalog knows the champion's ability cost only
                    // through the card; use 1 elixir as the common value when unknown)
                    let cost = 1.0f64;
                    if e.side == me {
                        my_elixir.advance(e_s);
                        my_elixir.spend(cost);
                        n_abilities += 1;
                    } else {
                        opp_elixir.advance(e_s);
                        opp_elixir.spend(cost);
                    }
                    let _ = card;
                    recent.push(e);
                }
            }
            ev_i += 1;
        }
        my_elixir.advance(t_s);
        opp_elixir.advance(t_s);

        cols.tick.push(t as i32);
        cols.time_s.push(t_s as f32);
        cols.phase.push(tl.phase_at(t_s));
        cols.overtime.push(tl.is_overtime(t_s) as u8);
        cols.elixir.push(my_elixir.elixir as f32);
        cols.opp_elixir.push(opp_elixir.elixir as f32);
        for &c in &hand.hand {
            cols.hand_ids.push(c);
            cols.hand_form.push(form_of(c));
            cols.hand_kind.push(cat.kind(c) as u8);
            cols.hand_cost.push(cat.cost(c));
            cols.hand_prob.push(inf.prob(n_in_deck_plays, c));
        }
        cols.next_card.push(hand.next_card());
        // recent plays, newest first
        let k = cfg.recent;
        let start = recent.len().saturating_sub(k);
        let window = &recent[start..];
        for j in 0..k {
            if let Some(e) = window.iter().rev().nth(j) {
                let (card, gx, gy) = match e.kind {
                    EventKind::Play { card, gx, gy, .. } => {
                        let (x, y) = to_own_frame(me, gx, gy);
                        (card, x, y)
                    }
                    EventKind::Ability { card } => (card, -1, -1),
                };
                cols.recent_side.push(if e.side == me { 0 } else { 1 });
                cols.recent_card.push(card);
                cols.recent_x.push(gx);
                cols.recent_y.push(gy);
                cols.recent_age.push((t - e.tick) as i32);
            } else {
                cols.recent_side.push(255);
                cols.recent_card.push(0);
                cols.recent_x.push(-1);
                cols.recent_y.push(-1);
                cols.recent_age.push(-1);
            }
        }
        // label: first own action in [t, t + step)
        let mut label: Option<&Event> = None;
        for e in &evs[ev_i..] {
            if e.tick >= t + cfg.step_ticks {
                break;
            }
            if e.side == me {
                label = Some(e);
                break;
            }
        }
        match label.map(|e| e.kind) {
            Some(EventKind::Play { card, gx, gy, .. }) => {
                let slot = hand.hand.iter().position(|&c| c == card).map(|s| s as i8).unwrap_or(-1);
                let (x, y) = to_own_frame(me, gx, gy);
                cols.waited.push((slot < 0) as u8);
                cols.card_slot.push(slot);
                cols.place_x.push(x);
                cols.place_y.push(y);
                cols.ability.push(0);
                cols.ability_card.push(0);
            }
            Some(EventKind::Ability { card }) => {
                cols.waited.push(1);
                cols.card_slot.push(-1);
                cols.place_x.push(-1);
                cols.place_y.push(-1);
                cols.ability.push(1);
                cols.ability_card.push(card);
            }
            None => {
                cols.waited.push(1);
                cols.card_slot.push(-1);
                cols.place_x.push(-1);
                cols.place_y.push(-1);
                cols.ability.push(0);
                cols.ability_card.push(0);
            }
        }
    }
    let result = match me {
        SideId::Team => r.result,
        SideId::Opp => -r.result,
    };
    let meta = SampleMeta {
        tag: r.tag.clone(),
        side: if me == SideId::Team { "team" } else { "opponent" },
        start: 0,
        length: cols.len(),
        result,
        own_crowns: own.crowns,
        opp_crowns: other.crowns,
        battle_type: r.battle_type.clone(),
        game_mode: r.game_mode.clone(),
        deck_ids,
        deck_forms: own.deck.iter().map(|c| c.form).collect(),
        deck_levels: own.deck.iter().map(|c| c.level).collect(),
        tower_key: own.tower_key.clone(),
        opp_deck_ids: other.deck.iter().map(|c| c.id).collect(),
        duration_s: r.duration_s,
        n_plays,
        n_abilities,
        hand_consistent_states: inf.n_consistent,
        skipped_plays: inf.skipped_plays,
        elixir_violations: my_elixir.violations,
        opp_elixir_violations: opp_elixir.violations,
        elixir_leaked: my_elixir.leaked as f32,
    };
    Some(Sample { meta, cols })
}

pub fn build_replay(r: &Replay, cat: &Catalog, tl: &Timeline, cfg: &FrameConfig) -> Vec<Sample> {
    [SideId::Team, SideId::Opp].into_iter().filter_map(|s| build_side(r, s, cat, tl, cfg)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{fixtures, parse_payload};
    use std::path::PathBuf;

    fn load() -> (Catalog, Timeline) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../catalog");
        (Catalog::load(&root.join("cards.json")).unwrap(), Timeline::load(&root.join("timeline.json")).unwrap())
    }

    #[test]
    fn frames_labels_and_perspective() {
        let (cat, tl) = load();
        let team = ["hog-rider", "ice-spirit", "ice-golem", "skeletons", "cannon", "musketeer", "fireball", "the-log"];
        let opp = ["giant", "witch", "knight", "archers", "zap", "minions", "fireball", "tornado"];
        // team plays hog at tick 40 (2.0 s) on its side; opponent plays giant at tick 65 on theirs
        let js = fixtures::payload(&team, &opp, &[("team", "hog-rider", 40, 9, 14), ("opponent", "giant", 65, 8, 20)], 10.0);
        let r = parse_payload("t", &js, &cat).unwrap();
        let cfg = FrameConfig { step_ticks: 10, recent: 4 };
        let samples = build_replay(&r, &cat, &tl, &cfg);
        assert_eq!(samples.len(), 2);
        let me = &samples[0];
        assert_eq!(me.meta.side, "team");
        assert_eq!(me.cols.len(), 21); // 200 ticks / 10 + 1
        // frame 4 covers ticks 40..50 → label = hog play, in hand
        let f = 4;
        assert_eq!(me.cols.waited[f], 0);
        let slot = me.cols.card_slot[f];
        assert!(slot >= 0);
        assert_eq!(me.cols.hand_ids[f * HAND + slot as usize], cat.resolve("hog-rider").0);
        assert_eq!((me.cols.place_x[f], me.cols.place_y[f]), (9, 14));
        assert!((me.cols.hand_prob[f * HAND + slot as usize] - 1.0).abs() < 1e-6);
        let row: f32 = me.cols.hand_prob[f * HAND..(f + 1) * HAND].iter().sum();
        assert!(row <= 4.0 + 1e-4 && row > 1.0);
        // elixir at 2.0 s = start + 2/2.8, before the play (label window starts at the frame)
        let expect = tl.starting_elixir + 2.0 / 2.8;
        assert!((me.cols.elixir[f] as f64 - expect).abs() < 1e-4);
        // after the play the hand no longer holds hog and elixir dropped by 4
        assert!(!me.cols.hand_ids[5 * HAND..6 * HAND].contains(&cat.resolve("hog-rider").0));
        assert!((me.cols.elixir[5] as f64 - (expect - 4.0 + 0.5 / 2.8)).abs() < 1e-4);
        // opponent's giant (5 elixir at tick 65) appears as a recent opp play from frame 7 on, mirrored
        let f7 = 7;
        assert_eq!(me.cols.recent_side[f7 * 4], 1);
        assert_eq!((me.cols.recent_x[f7 * 4], me.cols.recent_y[f7 * 4]), (8, 20));
        assert_eq!(me.cols.recent_age[f7 * 4], 5);
        // opponent perspective: their giant play is at mirrored coordinates and labelled at frame 6
        let them = &samples[1];
        assert_eq!(them.meta.result, -1);
        assert_eq!(them.cols.waited[6], 0);
        assert_eq!((them.cols.place_x[6], them.cols.place_y[6]), (17 - 8, 31 - 20));
        assert_eq!(them.meta.n_plays, 1);
    }
}


/// One player-sequence as a gm-shard sample: compact per-frame fields under `raw.*`
/// (expanded into the adapter's observation by the loader) and the IL targets.
pub fn to_gm(s: Sample, cat: &Catalog) -> gmshard::Sample {
    let c = &s.cols;
    let t = c.len();
    let k = if t == 0 { 0 } else { c.recent_side.len() / t };
    let next_cost: Vec<f32> = c.next_card.iter().map(|&id| cat.cost(id)).collect();
    let recent_cost: Vec<f32> = c.recent_card.iter().map(|&id| cat.cost(id)).collect();
    let place_idx: Vec<i64> = (0..t)
        .map(|i| if c.waited[i] != 0 { -1 } else { c.place_y[i] as i64 * GRID_W as i64 + c.place_x[i] as i64 })
        .collect();
    let value = vec![s.meta.result as f32; t];
    let meta = serde_json::to_value(&s.meta).unwrap_or(serde_json::Value::Null);
    gmshard::Sample::new(meta)
        .i32("raw.tick", &[], c.tick.clone())
        .f32("raw.time_s", &[], c.time_s.clone())
        .u8("raw.phase", &[], c.phase.clone())
        .u8("raw.overtime", &[], c.overtime.clone())
        .f32("raw.elixir", &[], c.elixir.clone())
        .f32("raw.opp_elixir", &[], c.opp_elixir.clone())
        .u16("raw.hand_ids", &[HAND], c.hand_ids.clone())
        .u8("raw.hand_form", &[HAND], c.hand_form.clone())
        .u8("raw.hand_kind", &[HAND], c.hand_kind.clone())
        .f32("raw.hand_cost", &[HAND], c.hand_cost.clone())
        .f32("raw.hand_prob", &[HAND], c.hand_prob.clone())
        .f32("raw.next_cost", &[], next_cost)
        .u8("raw.recent_side", &[k], c.recent_side.clone())
        .u16("raw.recent_card", &[k], c.recent_card.clone())
        .f32("raw.recent_cost", &[k], recent_cost)
        .i8("raw.recent_x", &[k], c.recent_x.clone())
        .i8("raw.recent_y", &[k], c.recent_y.clone())
        .i32("raw.recent_age", &[k], c.recent_age.clone())
        .u8("raw.ability", &[], c.ability.clone())
        .u16("raw.ability_card", &[], c.ability_card.clone())
        .bool("tgt.waited", &[], c.waited.iter().map(|&w| w != 0).collect())
        .i64("tgt.card_slot", &[], c.card_slot.iter().map(|&x| x as i64).collect())
        .i64("tgt.place_idx", &[], place_idx)
        .f32("tgt.value", &[], value)
}
