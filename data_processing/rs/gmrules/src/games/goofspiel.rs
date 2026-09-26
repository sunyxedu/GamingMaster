//! Goofspiel — the framework's SIMULTANEOUS-move test game, exactly OpenSpiel's
//! `goofspiel(num_cards=K, players=N, points_order=..., imp_info=False,
//! returns_type=win_loss)` (pinned by tests/test_candidate_reference.py).
//!
//! Rules. Every player holds the cards 1..K; a prize deck 1..K. Each turn one prize
//! card is revealed (chance draws it uniformly from the remaining ones for
//! `points_order = "random"`; fixed order for "ascending" / "descending"), then all
//! players bid one of their cards simultaneously; the unique highest bid wins the prize
//! (a tie for the highest bid discards it). The last turn is played automatically
//! (one card left each), as in OpenSpiel, so there are K−1 decision turns.
//! win_loss returns: all tied → 0; else winners +1/#winners, losers −1/#losers.
//! Action id = card value − 1, chance outcome id = prize value − 1.
//!
//! Frames (per-seat streams: all seats act every turn; bids are public once played,
//! so a frame shows everything except the simultaneous choices of the turn):
//!
//! ```text
//! entities [6, 17]  slots 0..3 seats relative to the viewer (type 0 me / 1 opponent):
//!                     is_me, k/3, points/total, cards left/K, hand (13, multi-hot)
//!                   slot 4 the prize card (type 2): value/K, one-hot value (13)
//!                   slot 5 prizes still to come (type 3): count/K, multi-hot (13)
//! globals  [5]      turn/K, N==2, N==3, N==4, K/13
//! actions  [13, 15] one per card in hand: value/K, one-hot value (13), value == prize
//! ```

use crate::frame::ObsView;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use serde::{Deserialize, Serialize};

pub const MAX_CARDS: usize = 13;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GoofspielConfig {
    /// 3..=13
    pub num_cards: usize,
    /// 2..=4
    pub players: usize,
    /// "random" | "ascending" | "descending"
    pub points_order: String,
}

impl Default for GoofspielConfig {
    fn default() -> Self {
        GoofspielConfig { num_cards: 5, players: 2, points_order: "random".into() }
    }
}

#[derive(Clone, Debug)]
pub struct Goofspiel {
    k: usize,
    n: usize,
    order: u8, // 0 random, 1 ascending, 2 descending
    /// hands[p][c] = player p still holds card c (id)
    hands: Vec<Vec<bool>>,
    prizes_left: Vec<bool>,
    prize: Option<u32>,
    turn_no: usize,
    points: Vec<u32>,
    /// bids of the current simultaneous turn
    pending: Vec<Option<u32>>,
    /// every resolved turn: (prize, bids)
    log: Vec<(u32, Vec<u32>)>,
    over: bool,
}

impl Goofspiel {
    fn total_points(&self) -> u32 {
        (self.k * (self.k + 1) / 2) as u32
    }

    fn next_prize_fixed(&self) -> Option<u32> {
        let left = (0..self.k as u32).filter(|&c| self.prizes_left[c as usize]);
        match self.order {
            1 => left.min(),
            2 => left.max(),
            _ => None,
        }
    }

    /// Reveal the next prize when it is not a chance event; play the forced last turn.
    fn advance(&mut self) {
        loop {
            if self.over {
                return;
            }
            let cards_left = self.hands[0].iter().filter(|&&h| h).count();
            if self.prize.is_none() {
                if cards_left == 0 {
                    self.over = true;
                    return;
                }
                if self.order == 0 && cards_left > 1 {
                    return; // chance node
                }
                // fixed order, or the last prize (the only one left)
                let p = self.next_prize_fixed().or_else(|| (0..self.k as u32).find(|&c| self.prizes_left[c as usize]));
                self.prize = p;
                self.prizes_left[p.unwrap() as usize] = false;
            }
            if cards_left == 1 {
                // the automatic last turn
                let bids: Vec<u32> = (0..self.n).map(|p| self.hands[p].iter().position(|&h| h).unwrap() as u32).collect();
                self.resolve(bids);
                continue;
            }
            return;
        }
    }

    fn resolve(&mut self, bids: Vec<u32>) {
        let prize = self.prize.take().expect("prize revealed");
        let hi = *bids.iter().max().unwrap();
        let at_hi: Vec<usize> = (0..self.n).filter(|&p| bids[p] == hi).collect();
        if at_hi.len() == 1 {
            self.points[at_hi[0]] += prize + 1;
        }
        for (p, &b) in bids.iter().enumerate() {
            self.hands[p][b as usize] = false;
        }
        self.log.push((prize, bids));
        self.pending = vec![None; self.n];
        self.turn_no += 1;
    }
}

impl Game for Goofspiel {
    type Config = GoofspielConfig;
    const NAME: &'static str = "goofspiel";
    const VERSION: u32 = 1;

    fn check_config(cfg: &GoofspielConfig) -> Result<(), RulesError> {
        if !(3..=MAX_CARDS).contains(&cfg.num_cards) || !(2..=4).contains(&cfg.players) {
            rules_bail!("num_cards must be 3..=13 and players 2..=4, got {} / {}", cfg.num_cards, cfg.players);
        }
        if !["random", "ascending", "descending"].contains(&cfg.points_order.as_str()) {
            rules_bail!("points_order must be random / ascending / descending, got {:?}", cfg.points_order);
        }
        Ok(())
    }

    fn spec(cfg: &GoofspielConfig) -> FrameSpec {
        assert!((3..=MAX_CARDS).contains(&cfg.num_cards), "goofspiel: num_cards must be 3..=13");
        assert!((2..=4).contains(&cfg.players), "goofspiel: players must be 2..=4");
        FrameSpec {
            n_seats: cfg.players,
            max_entities: 6,
            entity_features: 4 + MAX_CARDS,
            entity_types: 4,
            global_features: 5,
            max_actions: MAX_CARDS,
            action_features: 2 + MAX_CARDS,
            hidden_info: false,
            simultaneous: true,
            bits_entities: false,
            bits_globals: false,
            bits_actions: false,
            max_decisions: cfg.num_cards - 1,
        }
    }

    fn new_game(cfg: &GoofspielConfig) -> Self {
        let _ = Self::spec(cfg);
        let order = match cfg.points_order.as_str() {
            "random" => 0,
            "ascending" => 1,
            "descending" => 2,
            o => panic!("goofspiel: points_order must be random / ascending / descending, got {o:?}"),
        };
        let (k, n) = (cfg.num_cards, cfg.players);
        let mut g = Goofspiel {
            k,
            n,
            order,
            hands: vec![vec![true; k]; n],
            prizes_left: vec![true; k],
            prize: None,
            turn_no: 0,
            points: vec![0; n],
            pending: vec![None; n],
            log: Vec::new(),
            over: false,
        };
        g.advance();
        g
    }

    fn turn(&self) -> Turn {
        if self.over {
            Turn::Terminal
        } else if self.prize.is_none() {
            Turn::Chance
        } else {
            let mut s = SeatSet::EMPTY;
            for p in 0..self.n {
                if self.pending[p].is_none() {
                    s = s.with(p);
                }
            }
            Turn::Act(s)
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        let left: Vec<u32> = (0..self.k as u32).filter(|&c| self.prizes_left[c as usize]).collect();
        let p = 1.0 / left.len() as f64;
        out.extend(left.into_iter().map(|c| (c, p)));
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Chance || outcome as usize >= self.k || !self.prizes_left[outcome as usize] {
            rules_bail!("goofspiel: prize {outcome} cannot be drawn now");
        }
        self.prizes_left[outcome as usize] = false;
        self.prize = Some(outcome);
        self.advance();
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if matches!(self.turn(), Turn::Act(s) if s.contains(seat)) {
            out.extend((0..self.k as u32).filter(|&c| self.hands[seat][c as usize]));
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if !matches!(self.turn(), Turn::Act(s) if s.contains(seat)) {
            rules_bail!("goofspiel: seat {seat} is not to act");
        }
        if action as usize >= self.k || !self.hands[seat][action as usize] {
            rules_bail!("goofspiel: seat {seat} does not hold card {}", action + 1);
        }
        self.pending[seat] = Some(action);
        if self.pending.iter().all(|b| b.is_some()) {
            let bids = self.pending.iter().map(|b| b.unwrap()).collect();
            self.resolve(bids);
            self.advance();
        }
        Ok(())
    }

    fn returns(&self) -> Vec<f64> {
        if !self.over {
            return vec![0.0; self.n];
        }
        let hi = *self.points.iter().max().unwrap();
        let w = self.points.iter().filter(|&&p| p == hi).count();
        if w == self.n {
            return vec![0.0; self.n];
        }
        self.points.iter().map(|&p| if p == hi { 1.0 / w as f64 } else { -1.0 / (self.n - w) as f64 }).collect()
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns().iter().map(|&r| r as f32).collect()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        let (k, n) = (self.k, self.n);
        let total = self.total_points() as f32;
        for r in 0..n {
            let p = (seat + r) % n;
            let f = obs.entity(r, (r != 0) as i32);
            f[0] = (r == 0) as u8 as f32;
            f[1] = r as f32 / 3.0;
            f[2] = self.points[p] as f32 / total;
            f[3] = self.hands[p].iter().filter(|&&h| h).count() as f32 / k as f32;
            for c in 0..k {
                f[4 + c] = self.hands[p][c] as u8 as f32;
            }
        }
        if let Some(pz) = self.prize {
            let f = obs.entity(4, 2);
            f[0] = (pz + 1) as f32 / k as f32;
            f[1 + pz as usize] = 1.0;
        }
        let f = obs.entity(5, 3);
        f[0] = self.prizes_left.iter().filter(|&&x| x).count() as f32 / k as f32;
        for c in 0..k {
            f[1 + c] = self.prizes_left[c] as u8 as f32;
        }
        let g = obs.globals();
        g[0] = self.turn_no as f32 / k as f32;
        g[1] = (n == 2) as u8 as f32;
        g[2] = (n == 3) as u8 as f32;
        g[3] = (n == 4) as u8 as f32;
        g[4] = k as f32 / MAX_CARDS as f32;
    }

    fn encode_action(&self, _seat: Seat, action: u32, feat: &mut [f32]) {
        feat[0] = (action + 1) as f32 / self.k as f32;
        feat[1 + action as usize] = 1.0;
        feat[1 + MAX_CARDS] = (self.prize == Some(action)) as u8 as f32;
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        format!("Bid:{}", action + 1)
    }

    fn chance_string(&self, outcome: u32) -> String {
        format!("Prize:{}", outcome + 1)
    }

    fn state_string(&self) -> String {
        format!("turn={} prize={:?} points={:?} log={:?}", self.turn_no, self.prize.map(|p| p + 1), self.points, self.log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariance::check_game;
    use crate::trajectory::{replay, Event};

    #[test]
    fn rules_match_openspiel_examples() {
        // goofspiel(num_cards=3,players=3,points_order=descending): history [2,2,1,0,1,2]
        let cfg = GoofspielConfig { num_cards: 3, players: 3, points_order: "descending".into() };
        let ev: Vec<Event> = [2, 2, 1, 0, 1, 2].iter().enumerate().map(|(i, &a)| Event::Act(i % 3, a)).collect();
        let r = replay::<Goofspiel>(&cfg, &ev).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.points, vec![1, 0, 2]);
        assert_eq!(r.state.returns(), vec![-0.5, -0.5, 1.0]);
        assert_eq!(r.turns, 2);
        // simultaneous: both seats' frames of a turn come from the same state
        assert_eq!(r.decisions[0].turn, r.decisions[2].turn);
        // 2 players ascending, 4 cards
        let cfg = GoofspielConfig { num_cards: 4, players: 2, points_order: "ascending".into() };
        let ev: Vec<Event> = [3, 0, 0, 3, 1, 1].iter().enumerate().map(|(i, &a)| Event::Act(i % 2, a)).collect();
        let r = replay::<Goofspiel>(&cfg, &ev).unwrap();
        assert_eq!(r.state.returns(), vec![-1.0, 1.0]);
        // wrong seat order inside a simultaneous turn is rejected
        assert!(replay::<Goofspiel>(&cfg, &[Event::Act(1, 0), Event::Act(0, 0)]).is_err());
    }

    #[test]
    fn conformance_battery() {
        for (k, n, o) in [(3, 2, "random"), (5, 3, "random"), (6, 4, "descending"), (13, 2, "random")] {
            let cfg = GoofspielConfig { num_cards: k, players: n, points_order: o.into() };
            let rep = check_game::<Goofspiel>(&cfg, 60, 5).unwrap();
            assert_eq!(rep.terminal, 60);
            assert_eq!(rep.max_turns, k - 1);
        }
    }
}
