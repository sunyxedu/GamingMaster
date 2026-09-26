//! Kuhn poker for 2-4 players — the toy reference game of the framework, exactly
//! OpenSpiel's `kuhn_poker(players=N)` (ids, turn order, payoffs; pinned by
//! tests/test_candidate_reference.py against OpenSpiel itself).
//!
//! Rules. N players, N+1 cards (ranks 0..=N). Everyone antes 1; chance deals one
//! private card to each player in seat order (chance outcome = card rank). Players
//! act in rotation from seat 0 with Pass (0) or Bet (1). Nobody bets: after N passes
//! the highest card wins the pot. Once someone bets, every other player acts exactly
//! once more (Pass = fold, Bet = call, 1 chip); the highest card among the bettor and
//! the callers wins. Returns: winner pot − own stake, others −own stake (stake 1, or 2
//! after betting / calling).
//!
//! Frames (one per decision, the acting seat's view; hidden information → per-seat
//! streams). Dimensions are fixed for 2-4 players, so one network plays every
//! variant.
//!
//! ```text
//! entities [5, 8]  slots 0..3 = seats relative to the viewer (slot k = seat
//!                  viewer+k mod N; absent for k >= N), type 0 me / 1 opponent:
//!                    is_me, k/3, has_bet, passes/2, is_first_bettor, has_acted,
//!                    stake/2, folded
//!                  slot 4 = the viewer's own card, type 2:
//!                    one-hot rank (5), rank/N
//! globals  [6]     N==2, N==3, N==4, pot/(2N), bet pending, actions/(2N-1)
//! actions  [2, 4]  Pass, Bet: is_pass, is_bet, facing a bet, costs a chip   (bits)
//! ```
//! Outcome (IL value target, RL reward): return / (2(N-1)), in [-1/(N-1), 1].

use crate::frame::ObsView;
use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rng::Rng;
use crate::rules_bail;
use crate::trajectory::Event;
use serde::{Deserialize, Serialize};

pub const PASS: u32 = 0;
pub const BET: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KuhnConfig {
    /// 2..=4
    pub players: usize,
}

impl Default for KuhnConfig {
    fn default() -> Self {
        KuhnConfig { players: 2 }
    }
}

#[derive(Clone, Debug)]
pub struct KuhnPoker {
    n: usize,
    /// card of each seat, dealt in seat order
    cards: Vec<u32>,
    /// betting actions in order (the acting seat is `index % n`)
    history: Vec<u32>,
    first_bettor: Option<Seat>,
    bet: Vec<bool>,
    passes: Vec<u32>,
    pot: u32,
    winner: Option<Seat>,
}

impl KuhnPoker {
    fn current(&self) -> Seat {
        self.history.len() % self.n
    }

    fn holder(&self, card: u32) -> Option<Seat> {
        self.cards.iter().position(|&c| c == card)
    }

    fn stake(&self, seat: Seat) -> u32 {
        1 + self.bet[seat] as u32
    }
}

impl Game for KuhnPoker {
    type Config = KuhnConfig;
    const NAME: &'static str = "kuhn_poker";
    const VERSION: u32 = 1;

    fn check_config(cfg: &KuhnConfig) -> Result<(), RulesError> {
        if !(2..=4).contains(&cfg.players) {
            rules_bail!("players must be 2..=4, got {}", cfg.players);
        }
        Ok(())
    }

    fn spec(cfg: &KuhnConfig) -> FrameSpec {
        assert!((2..=4).contains(&cfg.players), "kuhn_poker: players must be 2..=4, got {}", cfg.players);
        FrameSpec::candidate(CandidateSpec {
            n_seats: cfg.players,
            max_entities: 5,
            entity_features: 8,
            entity_types: 3,
            global_features: 6,
            max_actions: 2,
            action_features: 4,
            hidden_info: true,
            simultaneous: false,
            bits_entities: false,
            bits_globals: false,
            bits_actions: true,
            max_decisions: 2 * cfg.players - 1,
        })
    }

    fn new_game(cfg: &KuhnConfig) -> Self {
        let n = cfg.players;
        assert!((2..=4).contains(&n), "kuhn_poker: players must be 2..=4, got {n}");
        KuhnPoker { n, cards: Vec::new(), history: Vec::new(), first_bettor: None, bet: vec![false; n], passes: vec![0; n], pot: n as u32, winner: None }
    }

    fn turn(&self) -> Turn {
        if self.cards.len() < self.n {
            Turn::Chance
        } else if self.winner.is_some() {
            Turn::Terminal
        } else {
            Turn::Act(SeatSet::single(self.current()))
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        let left = (self.n + 1 - self.cards.len()) as f64;
        out.extend((0..=self.n as u32).filter(|c| !self.cards.contains(c)).map(|c| (c, 1.0 / left)));
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Chance {
            rules_bail!("kuhn_poker: no chance node");
        }
        if outcome as usize > self.n || self.cards.contains(&outcome) {
            rules_bail!("kuhn_poker: card {outcome} cannot be dealt (dealt: {:?})", self.cards);
        }
        self.cards.push(outcome);
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.turn() == Turn::Act(SeatSet::single(seat)) {
            out.extend([PASS, BET]);
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            rules_bail!("kuhn_poker: seat {seat} is not to act ({:?})", self.turn());
        }
        match action {
            BET => {
                if self.first_bettor.is_none() {
                    self.first_bettor = Some(seat);
                }
                self.bet[seat] = true;
                self.pot += 1;
            }
            PASS => self.passes[seat] += 1,
            _ => rules_bail!("kuhn_poker: action {action} is not Pass (0) / Bet (1)"),
        }
        self.history.push(action);
        let n = self.n;
        let k = self.history.len();
        match self.first_bettor {
            None if k == n => {
                // everybody passed: the highest card dealt wins
                self.winner = self.holder(n as u32).or_else(|| self.holder(n as u32 - 1));
            }
            Some(b) if k == n + b => {
                self.winner = (0..=n as u32).rev().filter_map(|c| self.holder(c)).find(|&p| self.bet[p]);
            }
            _ => {}
        }
        Ok(())
    }

    fn returns(&self) -> Vec<f64> {
        let Some(w) = self.winner else { return vec![0.0; self.n] };
        (0..self.n)
            .map(|p| {
                let stake = self.stake(p) as f64;
                if p == w {
                    self.pot as f64 - stake
                } else {
                    -stake
                }
            })
            .collect()
    }

    fn outcome(&self) -> Vec<f32> {
        let scale = 2.0 * (self.n as f64 - 1.0);
        self.returns().iter().map(|&r| (r / scale) as f32).collect()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        let n = self.n;
        let acted = |p: Seat| self.history.len() > p;
        for k in 0..n {
            let p = (seat + k) % n;
            let f = obs.entity(k, (k != 0) as i32);
            f[0] = (k == 0) as u8 as f32;
            f[1] = k as f32 / 3.0;
            f[2] = self.bet[p] as u8 as f32;
            f[3] = self.passes[p] as f32 / 2.0;
            f[4] = (self.first_bettor == Some(p)) as u8 as f32;
            f[5] = acted(p) as u8 as f32;
            f[6] = self.stake(p) as f32 / 2.0;
            // passed while a bet was pending: the pass after the first bet in rotation
            f[7] = match self.first_bettor {
                Some(b) => {
                    let first_round_after_bet = p > b && self.history.len() > p && self.history[p] == PASS;
                    let second_round = p < b && self.history.len() > n + p && self.history[n + p] == PASS;
                    (first_round_after_bet || second_round) as u8 as f32
                }
                None => 0.0,
            };
        }
        let card = self.cards[seat];
        let f = obs.entity(4, 2);
        f[card as usize] = 1.0;
        f[5] = card as f32 / n as f32;
        let g = obs.globals();
        g[0] = (n == 2) as u8 as f32;
        g[1] = (n == 3) as u8 as f32;
        g[2] = (n == 4) as u8 as f32;
        g[3] = self.pot as f32 / (2 * n) as f32;
        g[4] = self.first_bettor.is_some() as u8 as f32;
        g[5] = self.history.len() as f32 / (2 * n - 1) as f32;
    }

    fn encode_action(&self, _seat: Seat, action: u32, feat: &mut [f32]) {
        feat[0] = (action == PASS) as u8 as f32;
        feat[1] = (action == BET) as u8 as f32;
        feat[2] = self.first_bettor.is_some() as u8 as f32;
        feat[3] = (action == BET) as u8 as f32;
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        match action {
            PASS => "Pass".into(),
            BET => "Bet".into(),
            a => format!("?{a}"),
        }
    }

    fn chance_string(&self, outcome: u32) -> String {
        format!("Deal:{outcome}")
    }

    fn state_string(&self) -> String {
        let h: String = self.history.iter().map(|&a| if a == PASS { 'p' } else { 'b' }).collect();
        format!("cards={:?} history={h} pot={} winner={:?}", self.cards, self.pot, self.winner)
    }

    /// Re-deal the other seats' cards (from every card but `seat`'s own); betting
    /// actions are always legal in Kuhn poker, so the trajectory stays valid.
    fn perturb_hidden(cfg: &KuhnConfig, events: &[Event], seat: Seat, rng: &mut Rng) -> Option<Vec<Event>> {
        let n = cfg.players;
        if events.len() < n || seat >= n {
            return None;
        }
        let dealt: Vec<u32> = events[..n].iter().map(|e| if let Event::Chance(c) = e { Some(*c) } else { None }).collect::<Option<_>>()?;
        let own = dealt[seat];
        let mut pool: Vec<u32> = (0..=n as u32).filter(|&c| c != own).collect();
        for _ in 0..16 {
            rng.shuffle(&mut pool);
            let mut it = pool.iter();
            let new: Vec<u32> = (0..n).map(|p| if p == seat { own } else { *it.next().unwrap() }).collect();
            if new != dealt {
                let mut out = events.to_vec();
                for (p, &c) in new.iter().enumerate() {
                    out[p] = Event::Chance(c);
                }
                return Some(out);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariance::{check_game, check_privacy};
    use crate::trajectory::replay;

    fn ev(deal: &[u32], acts: &[u32]) -> Vec<Event> {
        let n = deal.len();
        let mut e: Vec<Event> = deal.iter().map(|&c| Event::Chance(c)).collect();
        e.extend(acts.iter().enumerate().map(|(i, &a)| Event::Act(i % n, a)));
        e
    }

    #[test]
    fn payoffs_match_openspiel() {
        let cfg = KuhnConfig { players: 2 };
        // pp: showdown, card 2 wins 1
        let r = replay::<KuhnPoker>(&cfg, &ev(&[2, 1], &[PASS, PASS])).unwrap();
        assert_eq!(r.state.returns(), vec![1.0, -1.0]);
        // bp: player 1 folds
        let r = replay::<KuhnPoker>(&cfg, &ev(&[0, 1], &[BET, PASS])).unwrap();
        assert_eq!(r.state.returns(), vec![1.0, -1.0]);
        // pbb: player 0 calls, player 1 (card 2) wins 2
        let r = replay::<KuhnPoker>(&cfg, &ev(&[1, 2], &[PASS, BET, BET])).unwrap();
        assert_eq!(r.state.returns(), vec![-2.0, 2.0]);
        assert_eq!(r.state.outcome(), vec![-1.0, 1.0]);
        // 3 players: p b p b → seat 1 bets, seat 2 folds, seat 0 calls; highest bettor card wins
        let cfg3 = KuhnConfig { players: 3 };
        let r = replay::<KuhnPoker>(&cfg3, &ev(&[3, 1, 2], &[PASS, BET, PASS, BET])).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.returns(), vec![3.0, -2.0, -1.0]);
        // illegal / out-of-turn events are rejected
        assert!(replay::<KuhnPoker>(&cfg, &ev(&[1, 1], &[])).is_err());
        assert!(replay::<KuhnPoker>(&cfg, &[Event::Chance(0), Event::Chance(1), Event::Act(1, PASS)]).is_err());
        assert!(replay::<KuhnPoker>(&cfg, &ev(&[0, 1], &[PASS, PASS, PASS])).is_err());
    }

    #[test]
    fn conformance_battery() {
        for players in 2..=4 {
            let cfg = KuhnConfig { players };
            let rep = check_game::<KuhnPoker>(&cfg, 300, 11 + players as u64).unwrap();
            assert_eq!(rep.terminal, 300);
            assert!(rep.privacy_checks > 300, "{rep:?}");
            assert_eq!(rep.max_legal, 2);
        }
    }

    /// The privacy check must catch an encoder that leaks another seat's card.
    #[test]
    fn privacy_check_catches_a_leak() {
        #[derive(Clone, Debug)]
        struct Leaky(KuhnPoker);
        impl Game for Leaky {
            type Config = KuhnConfig;
            const NAME: &'static str = "leaky_kuhn";
            const VERSION: u32 = 0;
            fn spec(c: &KuhnConfig) -> FrameSpec { KuhnPoker::spec(c) }
            fn new_game(c: &KuhnConfig) -> Self { Leaky(KuhnPoker::new_game(c)) }
            fn turn(&self) -> Turn { self.0.turn() }
            fn chance_outcomes(&self, o: &mut Vec<(u32, f64)>) { self.0.chance_outcomes(o) }
            fn apply_chance(&mut self, o: u32) -> Result<(), RulesError> { self.0.apply_chance(o) }
            fn legal_actions(&self, s: Seat, o: &mut Vec<u32>) { self.0.legal_actions(s, o) }
            fn apply_action(&mut self, s: Seat, a: u32) -> Result<(), RulesError> { self.0.apply_action(s, a) }
            fn outcome(&self) -> Vec<f32> { self.0.outcome() }
            fn returns(&self) -> Vec<f64> { self.0.returns() }
            fn encode_obs(&self, s: Seat, obs: &mut ObsView<'_>) {
                self.0.encode_obs(s, obs);
                obs.globals()[5] = self.0.cards[(s + 1) % self.0.n] as f32; // the next seat's card
            }
            fn encode_action(&self, s: Seat, a: u32, f: &mut [f32]) { self.0.encode_action(s, a, f) }
            fn state_string(&self) -> String { self.0.state_string() }
            fn perturb_hidden(c: &KuhnConfig, e: &[Event], s: Seat, r: &mut Rng) -> Option<Vec<Event>> {
                KuhnPoker::perturb_hidden(c, e, s, r)
            }
        }
        let cfg = KuhnConfig { players: 2 };
        let a = ev(&[0, 1], &[PASS, BET, BET]);
        let b = ev(&[0, 2], &[PASS, BET, BET]);
        assert!(check_privacy::<KuhnPoker>(&cfg, &a, &b, 0).is_ok());
        let err = check_privacy::<Leaky>(&cfg, &a, &b, 0).unwrap_err();
        assert!(err.contains("hidden from it"), "{err}");
        assert!(check_game::<Leaky>(&cfg, 50, 3).is_err());
    }
}
