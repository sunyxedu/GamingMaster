//! Heads-up no-limit Texas hold'em — exactly the Annual Computer Poker Competition
//! (ACPC) rules the 2009-2017 heads-up no-limit logs were played under (the ACPC
//! dealer's `game.c`: `raiseIsValid`, `isValidAction`, `doAction`, `valueOfState`),
//! which is also OpenSpiel's `universal_poker` (it embeds that code). Pinned by
//! randomized differential tests against PokerKit (independent implementation,
//! tests/test_hunl_reference.py) and by replaying real ACPC records with their recorded
//! payoffs (gmdata `hunl convert`).
//!
//! Rules (config: `small_blind`, `big_blind`, `stack`; the ACPC game 50 / 100 /
//! 20000 = 200 big blinds, stacks reset every hand).
//! - Seat 0 posts the big blind, seat 1 (the button) the small blind. Seat 1 acts first
//!   preflop, seat 0 first on the flop, turn and river (ACPC `firstPlayer = 2 1 1 1`;
//!   PHH / PokerKit heads-up: p1 = seat 0).
//! - Chance deals seat 0's two hole cards, then seat 1's, then the flop (3 cards), turn
//!   and river as their betting rounds begin; when the betting ends with both players
//!   in (an all-in), the rest of the board is dealt before the showdown.
//! - Actions (canonical ids = OpenSpiel `universal_poker` full-game ids): [`FOLD`] = 0
//!   (only when facing a bet), [`CALL`] = 1 (check or call; a call larger than the
//!   stack is all-in), and `r >= 2` = RAISE TO `r` chips, the player's TOTAL
//!   commitment in the hand (ACPC semantics, not the street amount). Legal raises are
//!   `min_raise_to ..= stack`: preflop `min_raise_to` = 2 × big blind; each raise to
//!   `r` over the current bet `b` sets it to `max(min_raise_to, 2r − b)`; a new round
//!   resets it to `big_blind + bet`. When `min_raise_to > stack` the only raise is
//!   all-in (none if calling already puts the player all-in). No raise once the other
//!   player is all-in, or once a round holds 63 actions (ACPC `MAX_NUM_ACTIONS` = 64).
//! - A round ends when every player who can still act has called the last raise (or
//!   checked); the hand ends on a fold, after the river, or when fewer than two players
//!   can act. Returns (chips): the folder loses what it put in; at a showdown the best
//!   five-card hand ([`eval::eval`]) wins the matched pot, ties split.
//!
//! Frames: a CUSTOM schema with the CANDIDATE layout (`candidate_abstract.v1`, see
//! [`encode`]) — the exact bet amounts stay in the rules and the records; the policy
//! picks one of [`encode::N_CANDIDATES`] abstract candidates (fold, check/call,
//! all-in, min-raise, pot-fraction raises [`encode::RAISE_FRACTIONS`]); the IL target of
//! a real raise is its nearest candidate, with the exact amount kept as `raise_to` /
//! `raise_frac` targets. Hidden information: a seat sees its own hole cards and the
//! board dealt so far — never the opponent's cards (the ACPC logs record both players'
//! cards; they reach no frame) and never the future board (causality / privacy
//! batteries). Outcome (IL value target, RL reward): returns / stack, in [-1, 1].

pub mod encode;
pub mod eval;

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rng::Rng;
use crate::rules_bail;
use crate::schema::TensorsMut;
use crate::trajectory::Event;
use serde::{Deserialize, Serialize};

/// Fold (legal only when facing a bet).
pub const FOLD: u32 = 0;
/// Check or call.
pub const CALL: u32 = 1;
/// ACPC `MAX_NUM_ACTIONS`: a raise needs room for the raise and the other player's
/// answer in the round's 64 action slots.
pub const MAX_ROUND_ACTIONS: usize = 64;
/// Seat that acts first in each round (ACPC `firstPlayer = 2 1 1 1`, 0-based).
const FIRST_PLAYER: [Seat; 4] = [1, 0, 0, 0];
/// Board cards on the table while round `r` is played.
pub const BOARD_AT: [u8; 4] = [0, 3, 4, 5];
const NO_CARD: u8 = 255;
/// Largest supported stack (the legal set of a decision holds up to `stack` raise ids).
pub const MAX_STACK: u32 = 1_000_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HunlConfig {
    pub small_blind: u32,
    pub big_blind: u32,
    /// Both players' stack at the start of every hand (chips).
    pub stack: u32,
}

impl Default for HunlConfig {
    fn default() -> Self {
        HunlConfig { small_blind: 50, big_blind: 100, stack: 20000 }
    }
}

/// Kind of a betting action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Fold,
    Call,
    Raise,
}

/// One betting action of the hand (the public history the encoders read).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Act {
    pub round: u8,
    pub seat: u8,
    pub kind: Kind,
    /// The actor's total commitment before / after the action.
    pub before: u32,
    pub after: u32,
    /// The bet to match (max commitment) before the action.
    pub faced: u32,
    /// Pot (both commitments) before the action.
    pub pot: u32,
}

#[derive(Clone, Debug)]
pub struct Hunl {
    sb: u32,
    bb: u32,
    stack: u32,
    hole: [[u8; 2]; 2],
    n_hole: u8,
    board: [u8; 5],
    n_board: u8,
    /// Bit c = card c has been dealt.
    dealt: u64,
    round: u8,
    spent: [u32; 2],
    max_spent: u32,
    min_raise_to: u32,
    folded: [bool; 2],
    /// The betting is over (ACPC `finished`): a fold, or the showdown (after the rest
    /// of the board is dealt).
    finished: bool,
    acts: Vec<Act>,
    round_len: [u8; 4],
    /// Commitment of both players when the current round began (street amounts are
    /// commitments minus this).
    round_base: u32,
}

impl Hunl {
    pub fn config(&self) -> HunlConfig {
        HunlConfig { small_blind: self.sb, big_blind: self.bb, stack: self.stack }
    }

    pub fn round(&self) -> usize {
        self.round as usize
    }

    pub fn spent(&self) -> [u32; 2] {
        self.spent
    }

    pub fn max_spent(&self) -> u32 {
        self.max_spent
    }

    pub fn round_base(&self) -> u32 {
        self.round_base
    }

    pub fn acts(&self) -> &[Act] {
        &self.acts
    }

    pub fn board(&self) -> &[u8] {
        &self.board[..self.n_board as usize]
    }

    /// `seat`'s hole cards (empty until dealt).
    pub fn hole(&self, seat: Seat) -> &[u8] {
        let n = (self.n_hole as usize).saturating_sub(2 * seat).min(2);
        &self.hole[seat][..n]
    }

    pub fn folded(&self) -> [bool; 2] {
        self.folded
    }

    fn all_in(&self, p: Seat) -> bool {
        self.spent[p] >= self.stack
    }

    /// Players still able to act (not folded, not all-in).
    fn n_acting(&self) -> usize {
        (0..2).filter(|&p| !self.folded[p] && !self.all_in(p)).count()
    }

    /// ACPC `numCalled`: acting players who have called the current bet this round.
    fn n_called(&self) -> usize {
        let r = self.round;
        let mut n = 0;
        for a in self.acts.iter().rev().take_while(|a| a.round == r) {
            match a.kind {
                Kind::Raise => {
                    if !self.all_in(a.seat as usize) {
                        n += 1;
                    }
                    return n;
                }
                Kind::Call => {
                    if !self.all_in(a.seat as usize) {
                        n += 1;
                    }
                }
                Kind::Fold => {}
            }
        }
        n
    }

    /// ACPC `currentPlayer` (only meaningful while the betting is on).
    pub fn current_player(&self) -> Seat {
        let r = self.round as usize;
        let prev = if self.round_len[r] > 0 {
            self.acts.last().expect("round has actions").seat as usize
        } else {
            (FIRST_PLAYER[r] + 1) % 2
        };
        let mut n = prev;
        for _ in 0..2 {
            n = (n + 1) % 2;
            if !self.folded[n] && !self.all_in(n) {
                return n;
            }
        }
        prev
    }

    /// Fold is legal for the player to act (ACPC: not when it has matched the bet or
    /// is all-in).
    pub fn fold_legal(&self, p: Seat) -> bool {
        self.spent[p] != self.max_spent && self.spent[p] != self.stack
    }

    /// Legal raise-to range `(min, max)` of the player to act (ACPC `raiseIsValid`),
    /// or None.
    pub fn raise_range(&self, p: Seat) -> Option<(u32, u32)> {
        let r = self.round as usize;
        if self.round_len[r] as usize + 2 > MAX_ROUND_ACTIONS || self.n_acting() <= 1 {
            return None;
        }
        let (lo, hi) = (self.min_raise_to, self.stack);
        if lo > hi {
            if self.max_spent >= self.stack {
                return None;
            }
            return Some((hi, hi));
        }
        debug_assert!(self.spent[p] < lo);
        Some((lo, hi))
    }

    /// Chips `p` must add to call.
    pub fn to_call(&self, p: Seat) -> u32 {
        self.max_spent.min(self.stack) - self.spent[p].min(self.max_spent)
    }

    pub fn pot(&self) -> u32 {
        self.spent[0] + self.spent[1]
    }

    /// Chips added to the pot by `seat` taking the legal `action` now.
    pub fn chips_added(&self, seat: Seat, action: u32) -> u32 {
        match action {
            FOLD => 0,
            CALL => self.to_call(seat),
            r => r - self.spent[seat],
        }
    }

    fn is_acting(&self, seat: Seat) -> bool {
        self.turn() == Turn::Act(SeatSet::single(seat))
    }

    fn check_legal(&self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if !self.is_acting(seat) {
            rules_bail!("hunl: seat {seat} is not to act ({:?})", self.turn());
        }
        let ok = match action {
            FOLD => self.fold_legal(seat),
            CALL => true,
            r => self.raise_range(seat).is_some_and(|(lo, hi)| (lo..=hi).contains(&r)),
        };
        if !ok {
            rules_bail!("hunl: action {} is illegal for seat {seat} ({})", self.action_string(seat, action), self.state_string());
        }
        Ok(())
    }
}

impl Game for Hunl {
    type Config = HunlConfig;
    const NAME: &'static str = "hunl";
    const VERSION: u32 = 1;

    fn check_config(cfg: &HunlConfig) -> Result<(), RulesError> {
        let HunlConfig { small_blind: sb, big_blind: bb, stack } = *cfg;
        if sb == 0 || sb > bb {
            rules_bail!("hunl: need 0 < small_blind <= big_blind, got {sb} / {bb}");
        }
        if stack < 2 * bb || stack > MAX_STACK {
            rules_bail!("hunl: stack must be in 2 x big_blind ..= {MAX_STACK}, got {stack} (big blind {bb})");
        }
        Ok(())
    }

    fn spec(cfg: &HunlConfig) -> FrameSpec {
        encode::spec(cfg)
    }

    fn new_game(cfg: &HunlConfig) -> Self {
        Self::check_config(cfg).unwrap_or_else(|e| panic!("{e}"));
        Hunl {
            sb: cfg.small_blind,
            bb: cfg.big_blind,
            stack: cfg.stack,
            hole: [[NO_CARD; 2]; 2],
            n_hole: 0,
            board: [NO_CARD; 5],
            n_board: 0,
            dealt: 0,
            round: 0,
            spent: [cfg.big_blind, cfg.small_blind],
            max_spent: cfg.big_blind,
            min_raise_to: 2 * cfg.big_blind,
            folded: [false; 2],
            finished: false,
            acts: Vec::new(),
            round_len: [0; 4],
            round_base: 0,
        }
    }

    fn turn(&self) -> Turn {
        if self.n_hole < 4 {
            return Turn::Chance;
        }
        if self.finished {
            if self.folded[0] || self.folded[1] || self.n_board >= 5 {
                return Turn::Terminal;
            }
            return Turn::Chance;
        }
        if self.n_board < BOARD_AT[self.round as usize] {
            return Turn::Chance;
        }
        Turn::Act(SeatSet::single(self.current_player()))
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        let left = 52 - self.dealt.count_ones();
        let p = 1.0 / left as f64;
        out.extend((0..52u32).filter(|&c| self.dealt & (1 << c) == 0).map(|c| (c, p)));
    }

    fn apply_chance(&mut self, card: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Chance {
            rules_bail!("hunl: no chance node ({:?})", self.turn());
        }
        if card >= 52 || self.dealt & (1 << card) != 0 {
            rules_bail!("hunl: card {card} cannot be dealt (already dealt or not a card)");
        }
        self.dealt |= 1 << card;
        if self.n_hole < 4 {
            self.hole[(self.n_hole / 2) as usize][(self.n_hole % 2) as usize] = card as u8;
            self.n_hole += 1;
        } else {
            self.board[self.n_board as usize] = card as u8;
            self.n_board += 1;
        }
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if !self.is_acting(seat) {
            return;
        }
        if self.fold_legal(seat) {
            out.push(FOLD);
        }
        out.push(CALL);
        if let Some((lo, hi)) = self.raise_range(seat) {
            out.extend(lo..=hi);
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        self.check_legal(seat, action)?;
        let r = self.round as usize;
        let before = self.spent[seat];
        let (faced, pot) = (self.max_spent, self.pot());
        let kind = match action {
            FOLD => {
                self.folded[seat] = true;
                Kind::Fold
            }
            CALL => {
                self.spent[seat] = self.max_spent.min(self.stack);
                Kind::Call
            }
            to => {
                // next raise must call this bet, and raise by at least this much
                self.min_raise_to = self.min_raise_to.max(2 * to - self.max_spent);
                self.max_spent = to;
                self.spent[seat] = to;
                Kind::Raise
            }
        };
        self.acts.push(Act { round: r as u8, seat: seat as u8, kind, before, after: self.spent[seat], faced, pot });
        self.round_len[r] += 1;
        if self.folded.iter().filter(|&&f| f).count() + 1 >= 2 {
            self.finished = true;
        } else if self.n_called() >= self.n_acting() {
            if self.n_acting() > 1 {
                if r + 1 < 4 {
                    self.round += 1;
                    self.min_raise_to = self.bb.max(1) + self.max_spent;
                    self.round_base = self.max_spent;
                } else {
                    self.finished = true;
                }
            } else {
                // nobody left to bet against: straight to the showdown
                self.finished = true;
                self.round = 3;
                self.round_base = self.max_spent;
            }
        }
        Ok(())
    }

    /// ACPC `valueOfState` for two players: the folder loses its commitment; at the
    /// showdown the better hand wins the matched amount (the larger commitment's excess
    /// comes back), ties split.
    fn returns(&self) -> Vec<f64> {
        if self.turn() != Turn::Terminal {
            return vec![0.0, 0.0];
        }
        if let Some(f) = (0..2).find(|&p| self.folded[p]) {
            let w = 1 - f;
            let mut r = vec![0.0; 2];
            r[f] = -(self.spent[f] as f64);
            r[w] = self.spent[f] as f64;
            return r;
        }
        let size = self.spent[0].min(self.spent[1]) as f64;
        let v = |p: Seat| {
            let mut c = self.board.to_vec();
            c.extend_from_slice(&self.hole[p]);
            eval::eval(&c)
        };
        let (a, b) = (v(0), v(1));
        match a.cmp(&b) {
            std::cmp::Ordering::Greater => vec![size, -size],
            std::cmp::Ordering::Less => vec![-size, size],
            std::cmp::Ordering::Equal => vec![0.0, 0.0],
        }
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns().iter().map(|&r| (r / self.stack as f64) as f32).collect()
    }

    fn encode_custom(&self, seat: Seat, frame: &mut FrameMut<'_>) {
        encode::encode_frame(self, seat, frame)
    }

    fn target_custom(&self, seat: Seat, _legal: &[u32], action: u32, target: &mut TensorsMut<'_>) {
        encode::encode_target(self, seat, action, target)
    }

    fn decode_custom(&self, seat: Seat, _legal: &[u32], keys: &[i64]) -> Option<u32> {
        encode::Candidates::of(self, seat).decode(keys[0])
    }

    fn action_string(&self, seat: Seat, action: u32) -> String {
        match action {
            FOLD => "fold".into(),
            CALL if self.to_call(seat) == 0 => "check".into(),
            CALL => format!("call {}", self.to_call(seat)),
            r => format!("raise to {r}"),
        }
    }

    fn chance_string(&self, card: u32) -> String {
        eval::card_str(card as u8)
    }

    fn oracle_dim(_cfg: &HunlConfig) -> usize {
        2 * 52
    }

    /// Both seats' hole cards (one-hot over the 52 card ids each), the observer's first.
    fn encode_oracle(&self, observer: Seat, out: &mut [f32]) {
        for k in 0..2 {
            for &c in self.hole((observer + k) % 2) {
                out[k * 52 + c as usize] = 1.0;
            }
        }
    }

    fn state_string(&self) -> String {
        let cards = |c: &[u8]| c.iter().map(|&x| eval::card_str(x)).collect::<String>();
        let hist: Vec<String> = self
            .acts
            .iter()
            .map(|a| match a.kind {
                Kind::Fold => format!("{}:f", a.seat),
                Kind::Call => format!("{}:c", a.seat),
                Kind::Raise => format!("{}:r{}", a.seat, a.after),
            })
            .collect();
        format!(
            "hole=[{}|{}] board={} round={} spent={:?} bet={} min_raise_to={} folded={:?} finished={} actions=[{}]",
            cards(self.hole(0)),
            cards(self.hole(1)),
            cards(self.board()),
            self.round,
            self.spent,
            self.max_spent,
            self.min_raise_to,
            self.folded,
            self.finished,
            hist.join(" ")
        )
    }

    /// Re-deal the OTHER seat's hole cards (from the cards neither `seat`'s hole cards
    /// nor any board card of the trajectory use): the betting stays legal (it never
    /// depends on the cards), only the showdown can change.
    fn perturb_hidden(_cfg: &HunlConfig, events: &[Event], seat: Seat, rng: &mut Rng) -> Option<Vec<Event>> {
        if seat > 1 {
            return None;
        }
        let chance: Vec<(usize, u32)> =
            events.iter().enumerate().filter_map(|(i, e)| if let Event::Chance(c) = e { Some((i, *c)) } else { None }).collect();
        if chance.len() < 4 || chance[3].0 != 3 {
            return None; // the opponent's cards are not dealt (or not in dealing order)
        }
        let other = 1 - seat;
        let mut used = 0u64;
        for (k, &(_, c)) in chance.iter().enumerate() {
            if k / 2 != other || k >= 4 {
                used |= 1 << c;
            }
        }
        let mut pool: Vec<u32> = (0..52).filter(|&c| used & (1 << c) == 0).collect();
        let old = [chance[2 * other].1, chance[2 * other + 1].1];
        for _ in 0..16 {
            rng.shuffle(&mut pool);
            let new = [pool[0], pool[1]];
            if new != old {
                let mut out = events.to_vec();
                out[chance[2 * other].0] = Event::Chance(new[0]);
                out[chance[2 * other + 1].0] = Event::Chance(new[1]);
                return Some(out);
            }
        }
        None
    }
}

/// A structured random playout (differential / invariance tests, synthetic data):
/// chance uniform; each decision folds (when legal) with probability `p_fold`, else
/// raises with probability `p_raise` (when legal), else checks / calls; a raise is the
/// minimum, all-in, a pot fraction or uniform in the legal range. Much closer to real
/// play than uniform ids (which almost always raise).
pub fn random_hand(cfg: &HunlConfig, rng: &mut Rng, p_fold: f64, p_raise: f64) -> Vec<Event> {
    let mut g = Hunl::new_game(cfg);
    let mut ev = Vec::new();
    let mut outcomes = Vec::new();
    loop {
        match g.turn() {
            Turn::Terminal => return ev,
            Turn::Chance => {
                outcomes.clear();
                g.chance_outcomes(&mut outcomes);
                let c = outcomes[rng.below(outcomes.len() as u64) as usize].0;
                g.apply_chance(c).expect("legal card");
                ev.push(Event::Chance(c));
            }
            Turn::Act(s) => {
                let seat = s.iter().next().unwrap();
                let a = random_action(&g, seat, rng, p_fold, p_raise);
                g.apply_action(seat, a).expect("legal action");
                ev.push(Event::Act(seat, a));
            }
        }
    }
}

/// One decision of [`random_hand`].
pub fn random_action(g: &Hunl, seat: Seat, rng: &mut Rng, p_fold: f64, p_raise: f64) -> u32 {
    if g.fold_legal(seat) && rng.next_f64() < p_fold {
        return FOLD;
    }
    if let Some((lo, hi)) = g.raise_range(seat) {
        if rng.next_f64() < p_raise {
            let pot_after_call = (g.pot() + g.to_call(seat)) as f64;
            let m = g.max_spent() as f64;
            return match rng.below(5) {
                0 => lo,
                1 => hi,
                2 | 3 => {
                    let f = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0][rng.below(6) as usize];
                    ((m + f * pot_after_call).round() as u32).clamp(lo, hi)
                }
                _ => lo + rng.below((hi - lo + 1) as u64) as u32,
            };
        }
    }
    CALL
}

#[cfg(test)]
mod tests;
