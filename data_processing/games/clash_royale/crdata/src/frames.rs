//! Turn one replay into two decision-frame sequences (one per player), in
//! each player's own frame of reference (own half = rows 0..16, own back line
//! = row 0, x mirrored for the opponent so left/right stay egocentric).
//!
//! A frame every `step_ticks` (default 10 = 0.5 s) carries only what the player
//! knows at that moment, without seeing units: exact own elixir, the opponent's
//! elixir (a function of their visible plays), the 8 own deck cards — the deck
//! tokens, ordered by card id — each with the causal hand belief P(in hand) and
//! P(next) given the own plays BEFORE the frame ([`CycleBelief`]), the last
//! `recent` plays of both sides with ages (ability events attributed from the
//! public record, `replay::resolve_ability_sources`), the match phase, and the
//! label: the first own action inside the frame window (wait / deck index +
//! cell / champion ability). Labels may use hindsight; inputs never do
//! (`leak_audit.rs` truncates replays and rebuilds to check it).

use crate::catalog::{Catalog, Timeline};
use crate::elixir::ElixirSim;
use crate::hand::{CycleBelief, Observed};
use crate::replay::{Event, EventKind, Replay, SideId};
use serde::Serialize;

pub use crate::hand::DECK;

/// Cards in hand (the belief's rows sum to this).
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
    pub deck_ids: Vec<u16>,     // [T,8] own deck, token order (by card id); constant over the sequence
    pub deck_form: Vec<u8>,     // [T,8] owned form: 0 base, 1.. evolution, 200 hero
    pub deck_kind: Vec<u8>,     // [T,8] catalog kind
    pub deck_cost: Vec<f32>,    // [T,8] catalog elixir cost
    pub deck_in_hand: Vec<f32>, // [T,8] P(card in hand | own plays before the frame)
    pub deck_next: Vec<f32>,    // [T,8] P(card is the next card | own plays before the frame)
    pub recent_side: Vec<u8>,   // [T,K] 0 own 1 opp 255 none
    pub recent_card: Vec<u16>,  // [T,K]
    pub recent_x: Vec<i8>,      // [T,K]
    pub recent_y: Vec<i8>,      // [T,K]
    pub recent_age: Vec<i32>,   // [T,K] ticks since the play; -1 none
    pub waited: Vec<u8>,
    pub card_slot: Vec<i8>, // deck index (token) of the played card; -1 when waited
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
        ext!(tick, time_s, phase, overtime, elixir, opp_elixir, deck_ids, deck_form, deck_kind, deck_cost, deck_in_hand,
             deck_next, recent_side, recent_card, recent_x, recent_y, recent_age, waited, card_slot, place_x, place_y, ability,
             ability_card);
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
    /// initial cycle states consistent with ALL own plays (hindsight statistic; not an input)
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

/// Deck token order: deck indices sorted by (card id, form). The payload's deck order
/// carries no meaning (it is not the dealt cycle), and a live bot rebuilds this order
/// from its deck alone (`belief.py::deck_token_order`).
pub fn deck_token_order(ids: &[u16], forms: &[u8]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| (ids[i], forms.get(i).copied().unwrap_or(0), i));
    order
}

/// One player's frames as the match goes — the converter's (a replay's events) and the
/// simulator's (a live battle's): events are observed in tick order, and a frame read at a
/// tick holds the events strictly before it.
pub struct SideFrames<'a> {
    pub me: SideId,
    cat: &'a Catalog,
    tl: &'a Timeline,
    pub cfg: FrameConfig,
    pub tok_ids: Vec<u16>,
    pub tok_form: Vec<u8>,
    tok_kind: Vec<u8>,
    tok_cost: Vec<f32>,
    pub belief: CycleBelief,
    pub my_elixir: ElixirSim<'a>,
    pub opp_elixir: ElixirSim<'a>,
    recent: Vec<Event>,
    pub n_plays: usize,
    pub n_abilities: usize,
}

impl<'a> SideFrames<'a> {
    /// `deck`: the own 8 (card id, form) in the record's order (tokens are re-ordered by
    /// [`deck_token_order`]); None unless there are 8.
    pub fn new(deck: &[(u16, u8)], me: SideId, cat: &'a Catalog, tl: &'a Timeline, cfg: FrameConfig) -> Option<Self> {
        if deck.len() != DECK {
            return None;
        }
        let ids: Vec<u16> = deck.iter().map(|c| c.0).collect();
        let forms: Vec<u8> = deck.iter().map(|c| c.1).collect();
        let order = deck_token_order(&ids, &forms);
        let tok_ids: Vec<u16> = order.iter().map(|&i| ids[i]).collect();
        let tok_form: Vec<u8> = order.iter().map(|&i| forms[i]).collect();
        let tok_kind: Vec<u8> = tok_ids.iter().map(|&c| cat.kind(c) as u8).collect();
        let tok_cost: Vec<f32> = tok_ids.iter().map(|&c| cat.cost(c)).collect();
        let belief = CycleBelief::new(&tok_ids)?;
        Some(SideFrames { me, cat, tl, cfg, tok_ids, tok_form, tok_kind, tok_cost, belief, my_elixir: ElixirSim::new(tl),
                          opp_elixir: ElixirSim::new(tl), recent: Vec::new(), n_plays: 0, n_abilities: 0 })
    }

    /// Apply one event (in tick order); false when no hand cycle explains the own plays.
    pub fn observe(&mut self, e: &Event) -> bool {
        let e_s = e.tick as f64 / self.tl.ticks_per_second();
        match e.kind {
            EventKind::Play { card, .. } => {
                let cost = self.cat.cost(card) as f64;
                if e.side == self.me {
                    self.my_elixir.advance(e_s);
                    self.my_elixir.spend(cost);
                    if self.belief.observe(card) == Observed::Inconsistent {
                        return false;
                    }
                    self.n_plays += 1;
                } else {
                    self.opp_elixir.advance(e_s);
                    self.opp_elixir.spend(cost);
                }
            }
            EventKind::Ability { .. } => {
                // champion abilities cost elixir too (catalog knows the champion's ability cost only
                // through the card; use 1 elixir as the common value when unknown)
                let cost = 1.0f64;
                if e.side == self.me {
                    self.my_elixir.advance(e_s);
                    self.my_elixir.spend(cost);
                    self.n_abilities += 1;
                } else {
                    self.opp_elixir.advance(e_s);
                    self.opp_elixir.spend(cost);
                }
            }
        }
        self.recent.push(*e);
        true
    }

    /// Append the input columns of the frame at tick `t` (every observed event is before it;
    /// the label columns are the caller's).
    pub fn push_frame(&mut self, t: u32, cols: &mut FrameColumns) {
        let t_s = t as f64 / self.tl.ticks_per_second();
        self.my_elixir.advance(t_s);
        self.opp_elixir.advance(t_s);
        cols.tick.push(t as i32);
        cols.time_s.push(t_s as f32);
        cols.phase.push(self.tl.phase_at(t_s));
        cols.overtime.push(self.tl.is_overtime(t_s) as u8);
        cols.elixir.push(self.my_elixir.elixir as f32);
        cols.opp_elixir.push(self.opp_elixir.elixir as f32);
        cols.deck_ids.extend_from_slice(&self.tok_ids);
        cols.deck_form.extend_from_slice(&self.tok_form);
        cols.deck_kind.extend_from_slice(&self.tok_kind);
        cols.deck_cost.extend_from_slice(&self.tok_cost);
        cols.deck_in_hand.extend_from_slice(self.belief.in_hand());
        cols.deck_next.extend_from_slice(self.belief.next());
        // recent plays, newest first
        let k = self.cfg.recent;
        let start = self.recent.len().saturating_sub(k);
        let window = &self.recent[start..];
        for j in 0..k {
            if let Some(e) = window.iter().rev().nth(j) {
                let (card, gx, gy) = match e.kind {
                    EventKind::Play { card, gx, gy, .. } => {
                        let (x, y) = to_own_frame(self.me, gx, gy);
                        (card, x, y)
                    }
                    EventKind::Ability { card } => (card, -1, -1),
                };
                cols.recent_side.push(if e.side == self.me { 0 } else { 1 });
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
    }
}

/// A grid cell (team frame) in `side`'s own frame (own half rows 0..16, x mirrored for the
/// opponent), and back.
pub fn own_frame(side: SideId, gx: i8, gy: i8) -> (i8, i8) {
    to_own_frame(side, gx, gy)
}

/// Build the sequence for `me`; None when the deck is not 8 cards or no hand cycle
/// explains the own plays (a corrupt record).
pub fn build_side(r: &Replay, me: SideId, cat: &Catalog, tl: &Timeline, cfg: &FrameConfig) -> Option<Sample> {
    let (own, other) = match me {
        SideId::Team => (&r.team, &r.opp),
        SideId::Opp => (&r.opp, &r.team),
    };
    let deck: Vec<(u16, u8)> = own.deck.iter().map(|c| (c.id, c.form)).collect();
    let mut sf = SideFrames::new(&deck, me, cat, tl, *cfg)?;
    let deck_ids: Vec<u16> = own.deck.iter().map(|c| c.id).collect();
    let deck_forms: Vec<u8> = own.deck.iter().map(|c| c.form).collect();
    let tps = tl.ticks_per_second();
    let total_ticks = (r.duration_s * tps).ceil() as u32;
    let n_frames = (total_ticks / cfg.step_ticks) as usize + 1;
    let mut cols = FrameColumns::default();
    let mut ev_i = 0usize; // events consumed (tick < frame tick)
    let evs = &r.events;

    for f in 0..n_frames {
        let t = (f as u32) * cfg.step_ticks;
        // apply every event strictly before this frame
        while ev_i < evs.len() && evs[ev_i].tick < t {
            if !sf.observe(&evs[ev_i]) {
                return None;
            }
            ev_i += 1;
        }
        sf.push_frame(t, &mut cols);
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
                // the deck token of the played card (the belief holds the plays before
                // it: with repeated ids — unknown cards — the copy it can hold)
                let slot = sf.belief.attribute(card).map(|s| s as i8).unwrap_or(-1);
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
    // Own plays at or after the last frame's tick reach no input, but a record that no
    // hand cycle explains is corrupt as a whole (the same samples as converter_version 2
    // drop; the selection is on data integrity, never on a frame's content).
    for e in &evs[ev_i..] {
        if let (true, EventKind::Play { card, .. }) = (e.side == me, e.kind) {
            if sf.belief.observe(card) == Observed::Inconsistent {
                return None;
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
        deck_forms,
        deck_levels: own.deck.iter().map(|c| c.level).collect(),
        tower_key: own.tower_key.clone(),
        opp_deck_ids: other.deck.iter().map(|c| c.id).collect(),
        duration_s: r.duration_s,
        n_plays: sf.n_plays,
        n_abilities: sf.n_abilities,
        hand_consistent_states: sf.belief.n_consistent() as usize,
        skipped_plays: sf.belief.skipped,
        elixir_violations: sf.my_elixir.violations,
        opp_elixir_violations: sf.opp_elixir.violations,
        elixir_leaked: sf.my_elixir.leaked as f32,
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
        // deck tokens: the 8 deck cards sorted by id, repeated every frame
        let mut want: Vec<u16> = team.iter().map(|k| cat.resolve(k).0).collect();
        want.sort_unstable();
        assert_eq!(&me.cols.deck_ids[..DECK], &want[..]);
        assert_eq!(&me.cols.deck_ids[20 * DECK..], &want[..]);
        // frame 4 covers ticks 40..50 → label = hog play, as its deck token
        let f = 4;
        assert_eq!(me.cols.waited[f], 0);
        let slot = me.cols.card_slot[f];
        assert!(slot >= 0);
        let hog = cat.resolve("hog-rider").0;
        assert_eq!(me.cols.deck_ids[f * DECK + slot as usize], hog);
        assert_eq!((me.cols.place_x[f], me.cols.place_y[f]), (9, 14));
        // nothing was played before the decision: the belief is the deal prior (no hindsight)
        assert!(me.cols.deck_in_hand[f * DECK..(f + 1) * DECK].iter().all(|&p| p == 0.5));
        assert!(me.cols.deck_next[f * DECK..(f + 1) * DECK].iter().all(|&p| p == 0.125));
        // after the play: hog left the hand, the other 7 are each in hand with 4/7
        let row = &me.cols.deck_in_hand[5 * DECK..6 * DECK];
        assert_eq!(row[slot as usize], 0.0);
        assert!(row.iter().enumerate().all(|(i, &p)| i == slot as usize || p == (4.0f64 / 7.0) as f32));
        // elixir at 2.0 s = start + 2/2.8, before the play (label window starts at the frame)
        let expect = tl.starting_elixir + 2.0 / 2.8;
        assert!((me.cols.elixir[f] as f64 - expect).abs() < 1e-4);
        // after the play elixir dropped by 4
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

    #[test]
    fn a_record_no_cycle_explains_is_dropped() {
        let (cat, tl) = load();
        let team = ["hog-rider", "ice-spirit", "ice-golem", "skeletons", "cannon", "musketeer", "fireball", "the-log"];
        let opp = ["giant", "witch", "knight", "archers", "zap", "minions", "fireball", "tornado"];
        // hog twice in a row: it cannot be back in hand after one play
        let plays = [("team", "hog-rider", 40, 9, 14), ("team", "hog-rider", 80, 9, 14)];
        let r = parse_payload("t", &fixtures::payload(&team, &opp, &plays, 10.0), &cat).unwrap();
        let cfg = FrameConfig::default();
        assert!(build_side(&r, SideId::Team, &cat, &tl, &cfg).is_none());
        assert!(build_side(&r, SideId::Opp, &cat, &tl, &cfg).is_some());
    }

    #[test]
    fn repeated_ids_label_a_token_the_belief_allows() {
        // two cards the catalog does not know share its unknown id (deck tokens 6 and 7)
        let (cat, tl) = load();
        let team = ["hog-rider", "ice-spirit", "ice-golem", "skeletons", "zz-new-card-a", "zz-new-card-b", "fireball", "the-log"];
        let opp = ["giant", "witch", "knight", "archers", "zap", "minions", "fireball", "tornado"];
        // after the 4 known plays the hand is {fireball, the-log, new-a, new-b} exactly;
        // then both new cards are played
        let plays = [
            ("team", "hog-rider", 20, 9, 14),
            ("team", "ice-spirit", 60, 9, 14),
            ("team", "ice-golem", 100, 9, 14),
            ("team", "skeletons", 140, 9, 14),
            ("team", "zz-new-card-a", 180, 9, 14),
            ("team", "zz-new-card-b", 220, 9, 14),
        ];
        let r = parse_payload("t", &fixtures::payload(&team, &opp, &plays, 15.0), &cat).unwrap();
        let s = build_side(&r, SideId::Team, &cat, &tl, &FrameConfig::default()).unwrap();
        assert_eq!(&s.cols.deck_ids[6..DECK], &[cat.unknown_id, cat.unknown_id]);
        let frames: Vec<usize> = (0..s.cols.len()).filter(|&f| s.cols.card_slot[f] >= 0).collect();
        assert_eq!(frames, vec![2, 6, 10, 14, 18, 22]);
        for &f in &frames {
            let slot = s.cols.card_slot[f] as usize;
            assert!(s.cols.deck_in_hand[f * DECK + slot] > 0.0, "frame {f}: label token {slot} is out of the hand");
        }
        // the second new card is the copy still in hand (token 7), not the first token
        // with the id, which the belief put in the queue with the first play
        assert_eq!((s.cols.card_slot[18], s.cols.card_slot[22]), (6, 7));
        assert_eq!(&s.cols.deck_in_hand[22 * DECK + 6..23 * DECK], &[0.0, 1.0]);
    }
}


/// One player-sequence as a gm-shard sample: compact per-frame fields under `raw.*`
/// (expanded into the adapter's observation by the loader) and the IL targets.
pub fn to_gm(s: Sample, cat: &Catalog) -> gmshard::Sample {
    let c = &s.cols;
    let t = c.len();
    let k = if t == 0 { 0 } else { c.recent_side.len() / t };
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
        .u16("raw.deck_ids", &[DECK], c.deck_ids.clone())
        .u8("raw.deck_form", &[DECK], c.deck_form.clone())
        .u8("raw.deck_kind", &[DECK], c.deck_kind.clone())
        .f32("raw.deck_cost", &[DECK], c.deck_cost.clone())
        .f32("raw.deck_in_hand", &[DECK], c.deck_in_hand.clone())
        .f32("raw.deck_next", &[DECK], c.deck_next.clone())
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
