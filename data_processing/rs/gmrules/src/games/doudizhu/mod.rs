//! Dou Dizhu (斗地主), three players: landlord vs two peasants — OpenSpiel's
//! `dou_dizhu` (deal, auction, play, scoring, chance and action ids; pinned by
//! tests/test_doudizhu_reference.py) with the standard four-with-two combinations.
//!
//! Deal (chance). The face-up position `pos` (outcome `0..51`, uniform), then 51 cards
//! one at a time (outcome `51 + card`, uniform over the rest) to seats `k % 3` for
//! `k = 0..51`; the three undealt cards are the kitty. The seat dealt card `pos` (the
//! face-up card, whose rank everybody sees) bids first.
//!
//! Auction. In turn from the first bidder: pass (0) or a bid higher than the current
//! one (1..=3). A bid of 3 ends it, as do two consecutive passes after a bid; three
//! passes without a bid end the GAME (all returns 0). The last bidder becomes the
//! landlord, takes the kitty (shown to everybody) and leads the first trick.
//!
//! Play. The leader of a trick plays any combination ([`cards`]); each following
//! player passes or beats the last play (same kind and length with a higher main, or
//! a bomb / the rocket); after two consecutive passes the last player to play leads a
//! new trick. Combinations: solo, pair, trio, trio+solo, trio+pair, solo chain (>= 5,
//! up to A), pair chain (>= 3), airplane (>= 2 trios) with nothing / one solo per trio
//! / one pair per trio, four+2 solos, four+2 pairs (not bombs), bomb, rocket. Kickers
//! are never of a main rank; an airplane's solos are at most 3 of a rank and never
//! both jokers, pairs are distinct and never jokers (OpenSpiel's kicker rules, applied
//! to four+2 as well). The first player to empty their hand ends the game.
//!
//! Score (OpenSpiel's): `paying = bid * 2^(bombs + spring)` (bombs: bombs and rockets
//! played; spring: the landlord played exactly once, or no peasant played at all);
//! the landlord wins or loses `2 * paying`, each peasant the opposite `paying`.
//! Outcome (IL value target, RL reward), config `scoring`: "score" (default) =
//! returns / `score_scale` (default 24: a bid-3 landlord win with two doublings = 1)
//! clipped to [-1, 1]; "wp" = ±1 for the winning / losing side (0 without landlord).
//!
//! Rules variants (config `rules`): "standard" (default: the above) and "openspiel"
//! (exactly OpenSpiel's three differences: no four+2; no solo / pair chain or
//! kicker-free airplane starting at a rank the player holds four of — its
//! `SearchChainOnlyActions` skips such heads; and any bomb may follow the rocket — its
//! `SearchForLegalActions` only restricts what follows a bomb). The frame schema is
//! the same for both.
//!
//! Decisions. A combination with kickers is chosen in several consecutive decisions
//! of the same seat: its main part first, then one kicker unit per decision in
//! non-decreasing rank order (only completable prefixes are offered; see
//! [`cards::kicker_options`]) — every legal full play has exactly one path, and the
//! candidate count per frame stays bounded ([`encode::MAX_ACTIONS`], exact: see the tests).
//! Frames: the candidate schema, one per decision, the acting seat's view only (own
//! hand, public plays, kitty after it is shown, card counts, bids) — per-seat streams.
//! Layout of the entity / global / candidate features: [`encode`].
//!
//! Records (gmdata `doudizhu` converter) name plays by their OpenSpiel strings
//! ([`cards::play_str`], e.g. `33344`, `333444-56`, `7777-4455`), which
//! [`play_events`] turns into the decision path.

pub mod cards;
pub mod encode;

use crate::frame::ObsView;
use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rng::Rng;
use crate::rules_bail;
use crate::trajectory::Event;
use cards::*;
use serde::{Deserialize, Serialize};

pub const N_SEATS: usize = 3;
/// Cards dealt before the auction (17 each).
pub const N_DEALT: u8 = 51;
/// Hard bound on decisions per game: at most 7 auction decisions, and per card at
/// most one play, one kicker decision and two passes (54 x 4 + 7 = 223).
pub const MAX_DECISIONS: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DouDizhuConfig {
    /// "standard" | "openspiel"
    pub rules: String,
    /// "score" | "wp"
    pub scoring: String,
    /// "score" scoring: outcome = returns / score_scale, clipped to [-1, 1]
    pub score_scale: f64,
}

impl Default for DouDizhuConfig {
    fn default() -> Self {
        DouDizhuConfig { rules: "standard".into(), scoring: "score".into(), score_scale: 24.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Deal,
    Auction,
    Play,
    Over,
}

/// A seat's last action of the play phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastAct {
    Pass,
    Play(Play),
}

/// A combination with kickers being chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pending {
    pub main: Main,
    pub kickers: [u8; 5],
    pub n: u8,
}

impl Pending {
    pub fn chosen(&self) -> &[u8] {
        &self.kickers[..self.n as usize]
    }
}

#[derive(Clone, Debug)]
pub struct DouDizhu {
    variant: Variant,
    wp: bool,
    score_scale: f64,
    phase: Phase,
    face_up_pos: Option<u8>,
    n_dealt: u8,
    /// bit c: card c not dealt yet
    deck: u64,
    hands: [Counts; N_SEATS],
    face_up_rank: Option<u8>,
    first_bidder: u8,
    kitty: Counts,
    kitty_shown: bool,
    /// (seat, 0 = pass / bid)
    auction: Vec<(u8, u8)>,
    winning_bid: u8,
    landlord: Option<u8>,
    /// consecutive passes (auction, then within a trick)
    passes: u8,
    current: u8,
    /// the play to beat and who played it (None: the current player leads)
    trick: Option<(u8, Play)>,
    pending: Option<Pending>,
    played: [Counts; N_SEATS],
    n_plays: [u8; N_SEATS],
    bombs_by: [u8; N_SEATS],
    bombs: u8,
    last_act: [Option<LastAct>; N_SEATS],
    winner: Option<u8>,
}

fn next(s: u8) -> u8 {
    (s + 1) % N_SEATS as u8
}

impl DouDizhu {
    fn variant_of(cfg: &DouDizhuConfig) -> Variant {
        match cfg.rules.as_str() {
            "openspiel" => Variant::OpenSpiel,
            _ => Variant::Standard,
        }
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    pub fn hand(&self, seat: Seat) -> &Counts {
        &self.hands[seat]
    }

    pub fn landlord(&self) -> Option<Seat> {
        self.landlord.filter(|_| self.kitty_shown).map(|s| s as Seat)
    }

    /// The play to beat and its owner (None: the current player leads).
    pub fn trick(&self) -> Option<(Seat, Play)> {
        self.trick.map(|(s, p)| (s as Seat, p))
    }

    pub fn pending(&self) -> Option<&Pending> {
        self.pending.as_ref()
    }

    pub fn bombs(&self) -> u8 {
        self.bombs
    }

    pub fn winning_bid(&self) -> u8 {
        self.winning_bid
    }

    pub fn winner(&self) -> Option<Seat> {
        self.winner.map(|s| s as Seat)
    }

    /// Is the auction over with a landlord (the play phase started)?
    pub fn in_play(&self) -> bool {
        self.phase == Phase::Play || (self.phase == Phase::Over && self.kitty_shown)
    }

    fn legal_into(&self, seat: Seat, out: &mut Vec<u32>) {
        if !matches!(self.phase, Phase::Auction | Phase::Play) || seat != self.current as Seat {
            return;
        }
        if self.phase == Phase::Auction {
            out.push(PASS);
            out.extend((self.winning_bid + 1..=MAX_BID).map(|b| b as u32));
            return;
        }
        let hand = &self.hands[seat];
        if let Some(pd) = &self.pending {
            let mut ks = Vec::new();
            kicker_options(hand, &pd.main, pd.chosen(), &mut ks);
            let pair = pd.main.unit() == Unit::Pair;
            out.extend(ks.into_iter().map(|r| if pair { Action::KickPair(r) } else { Action::KickSolo(r) }.id()));
            return;
        }
        let start = out.len();
        if self.trick.is_some() {
            out.push(PASS);
        }
        main_options(hand, self.trick.as_ref().map(|t| &t.1), self.variant, out);
        out[start..].sort_unstable();
    }

    /// Play a full combination (cards leave the hand, the trick, bombs, the end).
    fn execute(&mut self, seat: u8, p: Play) {
        let s = seat as usize;
        for (r, &n) in p.counts().iter().enumerate() {
            self.hands[s][r] -= n;
            self.played[s][r] += n;
        }
        if p.kind.doubles() {
            self.bombs += 1;
            self.bombs_by[s] += 1;
        }
        self.n_plays[s] += 1;
        self.trick = Some((seat, p));
        self.passes = 0;
        self.last_act[s] = Some(LastAct::Play(p));
        if self.hands[s].iter().all(|&n| n == 0) {
            self.winner = Some(seat);
            self.phase = Phase::Over;
        } else {
            self.current = next(seat);
        }
    }

    fn apply_auction(&mut self, seat: u8, action: u32) {
        self.auction.push((seat, action as u8));
        let done = if action == PASS {
            self.passes += 1;
            if self.passes == N_SEATS as u8 {
                self.phase = Phase::Over; // nobody bid: the game ends, every return 0
                return;
            }
            self.passes == N_SEATS as u8 - 1 && self.winning_bid > 0
        } else {
            self.passes = 0;
            self.landlord = Some(seat);
            self.winning_bid = action as u8;
            self.winning_bid == MAX_BID
        };
        if done {
            let l = self.landlord.expect("a bid was made") as usize;
            for r in 0..N_RANKS {
                self.hands[l][r] += self.kitty[r];
            }
            self.kitty_shown = true;
            self.phase = Phase::Play;
            self.current = l as u8;
            self.trick = None;
            self.passes = 0;
        } else {
            self.current = next(seat);
        }
    }

    fn apply_play(&mut self, seat: u8, action: u32) -> Result<(), RulesError> {
        match Action::from_id(action) {
            Some(Action::Pass) => {
                self.passes += 1;
                self.last_act[seat as usize] = Some(LastAct::Pass);
                if self.passes == N_SEATS as u8 - 1 {
                    let (owner, _) = self.trick.take().expect("a pass follows a play");
                    self.passes = 0;
                    self.current = owner;
                } else {
                    self.current = next(seat);
                }
            }
            Some(Action::Play(p)) => self.execute(seat, p),
            Some(Action::Main(m)) => self.pending = Some(Pending { main: m, kickers: [0; 5], n: 0 }),
            Some(Action::KickSolo(r)) | Some(Action::KickPair(r)) => {
                let mut pd = self.pending.take().expect("a kicker follows a main");
                pd.kickers[pd.n as usize] = r;
                pd.n += 1;
                if pd.n == pd.main.n_kickers() {
                    self.execute(seat, Play::with_kickers(pd.main, pd.chosen()));
                } else {
                    self.pending = Some(pd);
                }
            }
            _ => rules_bail!("doudizhu: no action {action}"),
        }
        Ok(())
    }
}

/// The decision events of `seat` playing `play` (its main + kicker units, or the
/// single id of a kicker-free play) — the converters' mapping of a recorded play.
pub fn play_events(seat: Seat, play: &Play) -> Vec<Event> {
    play_path(play).into_iter().map(|a| Event::Act(seat, a)).collect()
}

impl Game for DouDizhu {
    type Config = DouDizhuConfig;
    const NAME: &'static str = "doudizhu";
    const VERSION: u32 = 1;

    fn check_config(cfg: &DouDizhuConfig) -> Result<(), RulesError> {
        if !["standard", "openspiel"].contains(&cfg.rules.as_str()) {
            rules_bail!("rules must be \"standard\" or \"openspiel\", got {:?}", cfg.rules);
        }
        if !["score", "wp"].contains(&cfg.scoring.as_str()) {
            rules_bail!("scoring must be \"score\" or \"wp\", got {:?}", cfg.scoring);
        }
        if !(cfg.score_scale.is_finite() && cfg.score_scale > 0.0) {
            rules_bail!("score_scale must be > 0, got {}", cfg.score_scale);
        }
        Ok(())
    }

    fn spec(_cfg: &DouDizhuConfig) -> FrameSpec {
        FrameSpec::candidate(CandidateSpec {
            n_seats: N_SEATS,
            max_entities: encode::N_ENTITIES,
            entity_features: encode::ENTITY_FEATURES,
            entity_types: 0,
            global_features: encode::GLOBAL_FEATURES,
            max_actions: encode::MAX_ACTIONS,
            action_features: encode::ACTION_FEATURES,
            hidden_info: true,
            simultaneous: false,
            bits_entities: true,
            bits_globals: true,
            bits_actions: true,
            max_decisions: MAX_DECISIONS,
        })
    }

    fn new_game(cfg: &DouDizhuConfig) -> Self {
        DouDizhu {
            variant: Self::variant_of(cfg),
            wp: cfg.scoring == "wp",
            score_scale: cfg.score_scale,
            phase: Phase::Deal,
            face_up_pos: None,
            n_dealt: 0,
            deck: (1u64 << N_CARDS) - 1,
            hands: [[0; N_RANKS]; N_SEATS],
            face_up_rank: None,
            first_bidder: 0,
            kitty: [0; N_RANKS],
            kitty_shown: false,
            auction: Vec::new(),
            winning_bid: 0,
            landlord: None,
            passes: 0,
            current: 0,
            trick: None,
            pending: None,
            played: [[0; N_RANKS]; N_SEATS],
            n_plays: [0; N_SEATS],
            bombs_by: [0; N_SEATS],
            bombs: 0,
            last_act: [None; N_SEATS],
            winner: None,
        }
    }

    fn turn(&self) -> Turn {
        match self.phase {
            Phase::Deal => Turn::Chance,
            Phase::Auction | Phase::Play => Turn::Act(SeatSet::single(self.current as Seat)),
            Phase::Over => Turn::Terminal,
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        if self.phase != Phase::Deal {
            return;
        }
        if self.face_up_pos.is_none() {
            out.extend((0..N_DEALT as u32).map(|p| (p, 1.0 / N_DEALT as f64)));
        } else {
            let left = self.deck.count_ones() as f64;
            out.extend((0..N_CARDS as u32).filter(|&c| self.deck >> c & 1 == 1).map(|c| (DEAL_BASE + c, 1.0 / left)));
        }
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.phase != Phase::Deal {
            rules_bail!("doudizhu: no chance node");
        }
        if self.face_up_pos.is_none() {
            if outcome >= N_DEALT as u32 {
                rules_bail!("doudizhu: face-up position {outcome} >= 51");
            }
            self.face_up_pos = Some(outcome as u8);
            return Ok(());
        }
        let card = outcome.wrapping_sub(DEAL_BASE);
        if card >= N_CARDS as u32 || self.deck >> card & 1 == 0 {
            rules_bail!("doudizhu: card {} cannot be dealt", outcome as i64 - DEAL_BASE as i64);
        }
        let card = card as u8;
        let k = self.n_dealt;
        let seat = (k % N_SEATS as u8) as usize;
        if Some(k) == self.face_up_pos {
            self.first_bidder = seat as u8;
            self.face_up_rank = Some(card_rank(card));
        }
        self.hands[seat][card_rank(card) as usize] += 1;
        self.deck &= !(1u64 << card);
        self.n_dealt += 1;
        if self.n_dealt == N_DEALT {
            for c in 0..N_CARDS as u8 {
                if self.deck >> c & 1 == 1 {
                    self.kitty[card_rank(c) as usize] += 1;
                }
            }
            self.phase = Phase::Auction;
            self.current = self.first_bidder;
        }
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        self.legal_into(seat, out)
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            rules_bail!("doudizhu: seat {seat} is not to act ({:?})", self.turn());
        }
        let mut legal = Vec::new();
        self.legal_into(seat, &mut legal);
        if legal.binary_search(&action).is_err() {
            rules_bail!("doudizhu: action {action} ({}) is illegal for seat {seat}", self.action_string(seat, action));
        }
        if self.phase == Phase::Auction {
            self.apply_auction(seat as u8, action);
            Ok(())
        } else {
            self.apply_play(seat as u8, action)
        }
    }

    fn returns(&self) -> Vec<f64> {
        let mut r = vec![0.0; N_SEATS];
        let (Some(w), Some(l), true) = (self.winner, self.landlord, self.kitty_shown) else { return r };
        let l = l as usize;
        let spring = self.n_plays[l] == 1 || (self.n_plays[(l + 1) % 3] == 0 && self.n_plays[(l + 2) % 3] == 0);
        let paying = self.winning_bid as f64 * 2f64.powi(self.bombs as i32 + spring as i32);
        let sign = if w as usize == l { 1.0 } else { -1.0 };
        r[l] = sign * 2.0 * paying;
        r[(l + 1) % 3] = -sign * paying;
        r[(l + 2) % 3] = -sign * paying;
        r
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns()
            .iter()
            .map(|&x| if self.wp { x.signum() as f32 } else { (x / self.score_scale).clamp(-1.0, 1.0) as f32 })
            .collect()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        encode::obs(self, seat, obs)
    }

    fn encode_action(&self, seat: Seat, action: u32, feat: &mut [f32]) {
        encode::action(self, seat, action, feat)
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        if self.phase == Phase::Auction {
            return if action == PASS { "Pass".into() } else { format!("Bid {action}") };
        }
        match Action::from_id(action) {
            Some(Action::Pass) => "Pass".into(),
            Some(Action::Bid(b)) => format!("Bid {b}"),
            Some(Action::Play(p)) => play_str(&p),
            Some(Action::Main(m)) => partial_str(&m, &[]),
            Some(Action::KickSolo(r)) | Some(Action::KickPair(r)) => match &self.pending {
                Some(pd) => {
                    let mut ks = pd.chosen().to_vec();
                    ks.push(r);
                    if ks.len() == pd.main.n_kickers() as usize {
                        play_str(&Play::with_kickers(pd.main, &ks))
                    } else {
                        partial_str(&pd.main, &ks)
                    }
                }
                None => format!("+{}", rank_str(r)),
            },
            None => format!("?{action}"),
        }
    }

    fn chance_string(&self, outcome: u32) -> String {
        if self.face_up_pos.is_none() {
            format!("Decide first card up position {outcome}")
        } else {
            format!("Deal {}", card_str(outcome.wrapping_sub(DEAL_BASE) as u8))
        }
    }

    fn state_string(&self) -> String {
        let hands: Vec<String> = self.hands.iter().map(counts_str).collect();
        format!(
            "phase={:?} hands={hands:?} kitty={} shown={} face_up={:?} first={} auction={:?} bid={} landlord={:?} \
             current={} trick={:?} pending={:?} bombs={} plays={:?} winner={:?}",
            self.phase,
            counts_str(&self.kitty),
            self.kitty_shown,
            self.face_up_rank.map(rank_str),
            self.first_bidder,
            self.auction,
            self.winning_bid,
            self.landlord,
            self.current,
            self.trick.map(|(s, p)| (s, play_str(&p))),
            self.pending.map(|pd| partial_str(&pd.main, pd.chosen())),
            self.bombs,
            self.n_plays,
            self.winner
        )
    }

    /// Re-deal what `seat` has not seen: the other seats' cards (but the face-up card
    /// and, per seat, cards covering the ranks it has played) and the kitty while it
    /// is not shown — shuffled among those deal positions (and the kitty). Every
    /// recorded play stays in its player's hand, so the trajectory stays legal.
    fn perturb_hidden(cfg: &DouDizhuConfig, events: &[Event], seat: Seat, rng: &mut Rng) -> Option<Vec<Event>> {
        if seat >= N_SEATS {
            return None;
        }
        let mut g = Self::new_game(cfg);
        for e in events {
            match *e {
                Event::Chance(o) => g.apply_chance(o).ok()?,
                Event::Act(s, a) => g.apply_action(s, a).ok()?,
            }
        }
        if g.n_dealt < N_DEALT {
            return None; // no decision yet: nothing to compare
        }
        let pos = g.face_up_pos? as usize;
        let cards: Vec<u8> = events[1..=N_DEALT as usize]
            .iter()
            .map(|e| match e {
                Event::Chance(o) => (o - DEAL_BASE) as u8,
                Event::Act(..) => unreachable!("51 deals"),
            })
            .collect();
        // each other seat keeps cards covering what it played (the shown kitty counts
        // for the landlord) — including a combination whose kickers are still being
        // chosen: its main part and chosen kickers are committed but not yet executed
        // (`played` counts them only once the play completes), so a record cut inside
        // a factored play must keep those cards with their seat
        let mut need = g.played;
        if let Some(pd) = g.pending {
            let cur = g.current as usize;
            for (r, &n) in pd.main.counts().iter().enumerate() {
                need[cur][r] += n;
            }
            let per = if pd.main.unit() == Unit::Pair { 2 } else { 1 };
            for &r in pd.chosen() {
                need[cur][r as usize] += per;
            }
        }
        if let Some(l) = g.landlord() {
            for (n, &k) in need[l].iter_mut().zip(&g.kitty) {
                *n = n.saturating_sub(k);
            }
        }
        let mut free_pos = Vec::new();
        for (i, &c) in cards.iter().enumerate() {
            let o = i % N_SEATS;
            if o == seat || i == pos {
                continue;
            }
            let r = card_rank(c) as usize;
            if need[o][r] > 0 {
                need[o][r] -= 1;
            } else {
                free_pos.push(i);
            }
        }
        let mut pool: Vec<u8> = free_pos.iter().map(|&i| cards[i]).collect();
        if !g.kitty_shown {
            pool.extend((0..N_CARDS as u8).filter(|&c| g.deck >> c & 1 == 1));
        }
        if pool.len() < 2 {
            return None;
        }
        let ranks_of = |cs: &[u8], seat_filter: usize| -> Counts {
            let mut h = [0u8; N_RANKS];
            for (i, &c) in cs.iter().enumerate() {
                if i % N_SEATS == seat_filter {
                    h[card_rank(c) as usize] += 1;
                }
            }
            h
        };
        let mut fallback = None;
        for _ in 0..16 {
            let mut p = pool.clone();
            rng.shuffle(&mut p);
            let mut new = cards.clone();
            for (j, &i) in free_pos.iter().enumerate() {
                new[i] = p[j];
            }
            if new == cards && g.kitty_shown {
                continue;
            }
            let ranks_changed = (0..N_SEATS).any(|o| o != seat && ranks_of(&new, o) != ranks_of(&cards, o));
            let mut out = events.to_vec();
            for (i, &c) in new.iter().enumerate() {
                out[1 + i] = Event::Chance(DEAL_BASE + c as u32);
            }
            if out == events {
                continue;
            }
            // the recorded actions constrain the hidden hands beyond the cards played
            // (e.g. an airplane main part is legal only for a hand without four of its
            // starting rank): keep only worlds in which every recorded action stays legal
            // — a perturbation must be consistent with what the observing seat saw
            let mut h = Self::new_game(cfg);
            let consistent = out.iter().all(|e| match *e {
                Event::Chance(o) => h.apply_chance(o).is_ok(),
                Event::Act(s, a) => h.apply_action(s, a).is_ok(),
            });
            if !consistent {
                continue;
            }
            if ranks_changed {
                return Some(out);
            }
            fallback.get_or_insert(out);
        }
        fallback
    }
}

#[cfg(test)]
mod tests;
