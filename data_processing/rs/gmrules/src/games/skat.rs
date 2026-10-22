//! Skat (3 players) — the rules of the International Skat Server (ISS,
//! skatgame.net/iss), whose ~9.1M game records are the IL corpus
//! (`gmdata skat convert`, data_processing/rs/gmdata/src/skat.rs). Implemented from the
//! ISS rule engine's documented behaviour (the GPL client sources `SimpleState` /
//! `GameDeclaration` were READ, not copied) and verified by replaying ISS records —
//! every move checked, every recorded result string (`R[d:.. v:.. m:.. p:.. t:.. ..]`)
//! reproduced by [`Skat::iss_result`] — and by randomized differential tests against
//! the ISS rule engine itself (its client package's classes, driven black-box: legal
//! sets at every decision, transitions, results; tests/test_skat_reference.py).
//!
//! Seats: 0 = forehand, 1 = middlehand, 2 = rearhand (ISS `P0` / `P1` / `P2`; the
//! dealer is not a player). Cards: id = suit·8 + rank, suits D, H, S, C (ISS order,
//! by value), ranks 7, 8, 9, T, J, Q, K, A; points A 11, T 10, K 4, Q 3, J 2.
//!
//! Rules (ISS):
//! - Deal (chance, 32 events): cards 1-10 to forehand, 11-20 middlehand, 21-30
//!   rearhand, 31-32 the skat (the ISS deal string order).
//! - Bidding: middlehand bids to forehand, who answers yes / pass; the survivor faces
//!   rearhand's bids; if nobody bid, forehand may bid (then plays) or pass (the game
//!   is passed in: no Ramsch on ISS, every seat scores 0). A bid is any value of the
//!   bid list (18, 20, 22, 23, 24, 27, ... 264) above the current one (jump bids are
//!   legal; ISS clients almost always bid the next value).
//! - The highest bidder (declarer) picks up the skat or declares a hand game: suit
//!   (D 9, H 10, S 11, C 12), grand (24) or null; hand trump games may announce
//!   schneider or schwarz, or be ouvert (open: implies schneider and schwarz
//!   announced); null hand (bid ≤ 35) / null ouvert hand (bid ≤ 59). After pickup:
//!   a suit game, grand, null (bid ≤ 23) or null ouvert (bid ≤ 46), then two cards
//!   are discarded (they count for the declarer).
//! - Card play: forehand leads; follow the led suit (jacks are trumps and belong to
//!   the trump suit in trump games; in null, the natural suits and ranks
//!   7 8 9 T J Q K A); highest trump / highest card of the led suit wins; the winner
//!   leads. Early end: null — the declarer takes a trick; schwarz announced — the
//!   defenders take a trick; schneider announced — the defenders exceed 30 points.
//! - Scoring (ISS `GameDeclaration.value`): null 23 / 35 hand / 46 ouvert / 59
//!   ouvert hand, lost (the declarer took a trick) −2×. Trump games: (matadors + 1 +
//!   hand + schneider ann. + schwarz ann. + ouvert [+ schneider (≥ 90) + schwarz
//!   (10 tricks) when won]) × base value; won = ≥ 61 points incl. skat (≥ 90 with
//!   schneider announced, all tricks with schwarz announced); a lost game counts
//!   −2×, with the announced schneider / schwarz levels, or else the declarer's own
//!   schneider (≤ 30 points) and schwarz (no trick) once he has played two cards;
//!   overbid (value < bid): −2 × the smallest multiple of the base value ≥ the bid.
//!   (These lost-game details are the ISS SERVER's, pinned by its records — every
//!   one of the archive's lost games agrees; the GPL client code counts only the
//!   announcement bonuses and penalises the own schneider / schwarz from the first
//!   complete trick.)
//!   Resignations / timeouts / leaving (ISS-only endings, not moves of this game) are
//!   scored by [`Skat::iss_result`] for the converter.
//!
//! Decisions are FACTORED so that no frame has more than 12 candidates (each
//! sub-decision is taken by the same seat on a state no other seat observes in
//! between, so the game is unchanged): a bid = pass / one of the next 10 bid values /
//! MORE (the next 10 values, repeatable: every bid stays reachable); a hand game =
//! its type (D H S C G N) then its modifier (plain / schneider / schwarz / ouvert;
//! null: plain / ouvert); after pickup: the game (D H S C G N NO) then the two
//! discards one card at a time (any order; the converter keeps the recorded order).
//!
//! Canonical action ids (ascending = candidate order):
//!
//! ```text
//! 0..32     play card c           32..64    discard card c (id + 32)
//! 64        pass (bid / answer)   65        yes (hold the bid)
//! 66..132   bid BIDS[i] (66 + i)  132       MORE (next 10 bid values)
//! 133       pick up the skat      134..140  hand game D H S C G N
//! 140..144  hand modifier plain / schneider / schwarz / ouvert
//! 144..151  game after pickup D H S C G N NO
//! ```
//!
//! Frames (candidate schema, hidden information → per-seat streams; every tensor is
//! 0/1, stored bit-packed). The seat sees its hand, the bids, the skat only after its
//! own pickup (and its own discards), the declarer's hand in ouvert games, and every
//! played card:
//!
//! ```text
//! entities [35, 49]   slots 0..32: card id (suit, rank; trump / effective suit once
//!                     the game is known; in my hand / from the skat / discarded by me
//!                     / in the declarer's open hand / played by me-next-previous in
//!                     trick k at position 0-2 / in the current trick / trick won by
//!                     me-next-previous / location unknown)
//!                     slots 32..35: seats me, next (plays after me), previous
//!                     (position, declarer / defender, passed, bidder / asked, its
//!                     highest bid, shown voids, to move)
//! globals  [149]      phase, my position, game type + hand / ouvert / schneider /
//!                     schwarz / pickup, declarer, trick number, trick size, leader,
//!                     trick points of the declarer and of the defenders
//!                     (thermometers + the 31 / 61 / 90 thresholds), the bid, the known
//!                     skat's points, the declarer's tricks
//! actions  [12, 68]   kind (play, discard, pass, yes, bid, more, pickup, hand game,
//!                     modifier, game), card (suit, rank, trump, beats the trick so
//!                     far, follows suit), bid value, game type, ouvert, modifier
//! ```
//!
//! Outcome (IL value target, RL reward): ISS's zero-sum tournament ("money") score
//! of the game — declarer won: v + 50, lost: v − 50 and 40 per defender (Seeger-
//! Fabian), then `3·score − Σ` — divided by 1260 (the largest possible magnitude:
//! an overbid diamonds game at bid 264); `returns()` = the money scores.

use crate::frame::ObsView;
use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rng::Rng;
use crate::rules_bail;
use crate::trajectory::Event;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// cards
// ---------------------------------------------------------------------------

pub const N_CARDS: usize = 32;
/// ISS suit letters, by value (D = 0 ... C = 3).
pub const SUIT_CHARS: [char; 4] = ['D', 'H', 'S', 'C'];
/// ISS rank letters (T = ten).
pub const RANK_CHARS: [char; 8] = ['7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
pub const JACK: u32 = 4;
pub const ACE: u32 = 7;
/// Card points by rank index.
pub const POINTS: [u32; 8] = [0, 0, 0, 10, 2, 3, 4, 11];
/// Order of the non-jack ranks in suit / grand play (7 8 9 Q K T A).
const SUIT_ORDER: [u32; 8] = [0, 1, 2, 5, 0, 3, 4, 6];

pub fn suit(c: u32) -> u32 {
    c >> 3
}

pub fn rank(c: u32) -> u32 {
    c & 7
}

pub fn card(suit: u32, rank: u32) -> u32 {
    suit * 8 + rank
}

pub fn card_points(c: u32) -> u32 {
    POINTS[rank(c) as usize]
}

/// ISS notation, e.g. "CJ", "HT".
pub fn card_str(c: u32) -> String {
    format!("{}{}", SUIT_CHARS[suit(c) as usize], RANK_CHARS[rank(c) as usize])
}

pub fn parse_card(s: &str) -> Option<u32> {
    let mut it = s.chars();
    let (a, b) = (it.next()?, it.next()?);
    if it.next().is_some() {
        return None;
    }
    let su = SUIT_CHARS.iter().position(|&x| x == a)? as u32;
    let r = RANK_CHARS.iter().position(|&x| x == b)? as u32;
    Some(card(su, r))
}

fn cards_of(mask: u32) -> impl Iterator<Item = u32> {
    (0..32u32).filter(move |&c| mask & (1 << c) != 0)
}

fn bit(c: u32) -> u32 {
    1 << c
}

fn mask_points(mask: u32) -> u32 {
    cards_of(mask).map(card_points).sum()
}

// ---------------------------------------------------------------------------
// games, bids, actions
// ---------------------------------------------------------------------------

pub const DIAMONDS: u8 = 0;
pub const HEARTS: u8 = 1;
pub const SPADES: u8 = 2;
pub const CLUBS: u8 = 3;
pub const GRAND: u8 = 4;
pub const NULL: u8 = 5;
/// Game type letters (ISS), D H S C G N.
pub const GAME_CHARS: [char; 6] = ['D', 'H', 'S', 'C', 'G', 'N'];
/// Base values of the trump games D H S C G.
pub const BASE_VALUE: [u32; 5] = [9, 10, 11, 12, 24];

/// The ISS bid list.
pub const BIDS: [u32; 66] = [
    18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 45, 46, 48, 50, 54, 55, 59, 60, 63, 66, 70, 72, 77, 80, 81, 84, 88,
    90, 96, 99, 100, 108, 110, 117, 120, 121, 126, 130, 132, 135, 140, 143, 144, 150, 153, 154, 156, 160, 162, 165, 168,
    170, 171, 176, 180, 187, 189, 190, 192, 198, 204, 216, 240, 264,
];
/// Bid values offered per bid decision (the rest behind MORE).
pub const BID_WINDOW: usize = 10;

pub const PLAY0: u32 = 0;
pub const DISCARD0: u32 = 32;
pub const PASS: u32 = 64;
pub const YES: u32 = 65;
pub const BID0: u32 = 66;
pub const MORE: u32 = 132;
pub const PICKUP: u32 = 133;
pub const HAND0: u32 = 134;
pub const MOD0: u32 = 140;
pub const DECLARE0: u32 = 144;
pub const N_ACTIONS: u32 = 151;

pub const MOD_PLAIN: u32 = 0;
pub const MOD_SCHNEIDER: u32 = 1;
pub const MOD_SCHWARZ: u32 = 2;
pub const MOD_OUVERT: u32 = 3;
/// `DECLARE0 + DECL_NULL_OUVERT` = null ouvert after pickup.
pub const DECL_NULL_OUVERT: u32 = 6;

pub const MAX_NULL_BID: u32 = 23;
pub const MAX_NULL_HAND_BID: u32 = 35;
pub const MAX_NULL_OUVERT_BID: u32 = 46;
pub const MAX_NULL_OUVERT_HAND_BID: u32 = 59;

/// Normaliser of the money scores (the largest possible |score|).
pub const MONEY_SCALE: f64 = 1260.0;

/// Index of the first bid above `max_bid` (66 = none).
pub fn next_bid_index(max_bid: u32) -> usize {
    BIDS.iter().position(|&b| b > max_bid).unwrap_or(BIDS.len())
}

pub fn bid_index(value: u32) -> Option<usize> {
    BIDS.iter().position(|&b| b == value)
}

/// The declared contract (ISS `GameDeclaration`; `schneider` / `schwarz` = announced).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contract {
    pub ty: u8,
    pub hand: bool,
    pub ouvert: bool,
    pub schneider: bool,
    pub schwarz: bool,
}

impl Contract {
    /// ISS notation (`GameDeclaration.toString`), e.g. "G", "CHS", "NOH", "GO".
    pub fn iss_string(&self) -> String {
        let mut s = GAME_CHARS[self.ty as usize].to_string();
        if self.ouvert {
            s.push('O');
            if self.ty != NULL {
                return s;
            }
        }
        if self.hand {
            s.push('H');
        }
        if self.ty == NULL {
            return s;
        }
        if self.schwarz {
            s.push('Z');
        } else if self.schneider {
            s.push('S');
        }
        s
    }

    /// Parse ISS notation (modifiers in any order, as the ISS engine does): ouvert
    /// trump games imply hand + schneider + schwarz announced, schwarz implies
    /// schneider; null with schneider / schwarz, or schneider without hand, is invalid.
    pub fn parse_iss(s: &str) -> Option<Contract> {
        let mut it = s.chars();
        let first = it.next()?;
        let ty = GAME_CHARS.iter().position(|&c| c == first)? as u8;
        let mut c = Contract { ty, hand: false, ouvert: false, schneider: false, schwarz: false };
        for ch in it {
            match ch {
                'O' => c.ouvert = true,
                'H' => c.hand = true,
                'S' => c.schneider = true,
                'Z' => c.schwarz = true,
                _ => return None,
            }
        }
        if c.ouvert && ty != NULL {
            c.hand = true;
            c.schneider = true;
            c.schwarz = true;
        }
        if c.schwarz {
            c.schneider = true;
        }
        if (ty == NULL && c.schneider) || (!c.hand && c.schneider) {
            return None;
        }
        Some(c)
    }

    /// The modifier action of a hand contract (plain / schneider / schwarz / ouvert).
    pub fn hand_modifier(&self) -> u32 {
        if self.ouvert {
            MOD_OUVERT
        } else if self.schwarz {
            MOD_SCHWARZ
        } else if self.schneider {
            MOD_SCHNEIDER
        } else {
            MOD_PLAIN
        }
    }

    pub fn is_trump(&self, c: u32) -> bool {
        self.ty != NULL && (rank(c) == JACK || (self.ty < GRAND && suit(c) == self.ty as u32))
    }

    /// The suit a card belongs to for following: 0..4 natural suits, 4 = trump.
    pub fn eff_suit(&self, c: u32) -> u32 {
        if self.is_trump(c) {
            4
        } else {
            suit(c)
        }
    }

    /// Strength within its effective suit (higher wins).
    fn strength(&self, c: u32) -> u32 {
        if self.ty == NULL {
            rank(c)
        } else if rank(c) == JACK {
            20 + suit(c)
        } else if self.is_trump(c) {
            10 + SUIT_ORDER[rank(c) as usize]
        } else {
            SUIT_ORDER[rank(c) as usize]
        }
    }

    /// Does `c` beat the trick's current winning card `win` (led effective suit `led`)?
    pub fn beats(&self, c: u32, win: u32, led: u32) -> bool {
        let (ec, ew) = (self.eff_suit(c), self.eff_suit(win));
        if ec == 4 && ew != 4 {
            return true;
        }
        ec == ew && (ec == 4 || ec == led) && self.strength(c) > self.strength(win)
    }
}

/// ISS `trumpMultiplier`: matadors + 1 ("with / without n, game n+1") of a 12-card
/// set (declarer's hand + skat) for trump game `ty`.
pub fn trump_multiplier(h: u32, ty: u8) -> u32 {
    let has = |c: u32| h & bit(c) != 0;
    let with = has(card(3, JACK));
    let mut m = 2;
    for s in [2u32, 1, 0] {
        if has(card(s, JACK)) != with {
            return m;
        }
        m += 1;
    }
    if ty < GRAND {
        for r in [ACE, 3, 6, 5, 2, 1, 0] {
            if has(card(ty as u32, r)) != with {
                return m;
            }
            m += 1;
        }
    }
    m
}

/// Result of a game in ISS's terms (`GameResult`): [`IssResult::iss_string`] is the
/// record's `R[...]` field.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IssResult {
    pub passed: bool,
    pub penalty: bool,
    pub declarer: i32,
    /// The declarer's score (`v`): > 0 won, < 0 lost (−2× the game value).
    pub value: i32,
    /// ± matadors ("with" > 0, "without" < 0).
    pub matadors: i32,
    pub overbid: bool,
    /// Declarer's card points incl. the skat (`p`) and tricks (`t`).
    pub points: i32,
    pub tricks: i32,
    pub schneider: bool,
    pub schwarz: bool,
    pub penalties: [i32; 3],
    pub left: i32,
    pub timeout: i32,
    pub resigned: bool,
}

impl IssResult {
    /// The ISS record's result string (`GameResult.toString`).
    pub fn iss_string(&self) -> String {
        if self.passed {
            return "passed".into();
        }
        let kind = if self.penalty { "penalty" } else if self.value > 0 { "win" } else { "loss" };
        format!(
            "d:{} {kind} v:{} m:{} {} p:{} t:{} s:{} z:{} p0:{} p1:{} p2:{} l:{} to:{} r:{}",
            self.declarer,
            self.value,
            self.matadors,
            if self.overbid { "overbid" } else { "bidok" },
            self.points,
            self.tricks,
            self.schneider as u8,
            self.schwarz as u8,
            self.penalties[0],
            self.penalties[1],
            self.penalties[2],
            self.left,
            self.timeout,
            self.resigned as u8
        )
    }

    /// ISS's zero-sum tournament ("money") scores of the three seats: declarer won
    /// v + 50, lost v − 50 and 40 to each defender, then `3·score − Σ` (0 for passed /
    /// penalty games).
    pub fn money(&self) -> [f64; 3] {
        if self.passed || self.penalty || self.declarer < 0 {
            return [0.0; 3];
        }
        let d = self.declarer as usize;
        let v = self.value as f64;
        let mut s = if v > 0.0 { [0.0; 3] } else { [40.0; 3] };
        s[d] = if v > 0.0 { v + 50.0 } else { v - 50.0 };
        let sum: f64 = s.iter().sum();
        [3.0 * s[0] - sum, 3.0 * s[1] - sum, 3.0 * s[2] - sum]
    }
}

/// How an ISS game ended besides normal play: resignations (declarer, or both
/// defenders, end the game; one defender's does not), a timeout or a player leaving.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IssEnding {
    pub resigned: [bool; 3],
    pub timeout: Option<Seat>,
    pub left: Option<Seat>,
}

impl IssEnding {
    /// The resignations end the game (in card play: the declarer, or both defenders).
    pub fn resignation_ends(&self, declarer: Seat) -> bool {
        self.resigned[declarer] || (0..3).filter(|&s| s != declarer && self.resigned[s]).count() >= 2
    }
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Deal,
    Bid,
    /// The bidder chose MORE: bids `window` levels and beyond above the next one.
    BidMore,
    Answer,
    SkatOrHand,
    HandModifier,
    Declare,
    Discard1,
    Discard2,
    Play,
    Done,
}

impl Phase {
    fn index(self) -> usize {
        match self {
            Phase::Bid => 0,
            Phase::BidMore => 1,
            Phase::Answer => 2,
            Phase::SkatOrHand => 3,
            Phase::HandModifier => 4,
            Phase::Declare => 5,
            Phase::Discard1 => 6,
            Phase::Discard2 => 7,
            Phase::Play => 8,
            Phase::Deal | Phase::Done => 9,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkatConfig {}

#[derive(Clone, Debug)]
pub struct Skat {
    /// cards in deal order (≤ 32)
    deal: Vec<u8>,
    hands: [u32; 3],
    /// the dealt skat
    orig_skat: u32,
    /// the skat as it counts for the declarer: the dealt skat, or the discards
    skat: u32,
    phase: Phase,
    to_move: Seat,
    // bidding (ISS SimpleState)
    bidder: Option<Seat>,
    asked: Option<Seat>,
    max_bid: u32,
    /// per seat: 0 no bid yet, 1 passed without bidding, else its bid / held value
    player_bid: [u32; 3],
    passed: [bool; 3],
    window: usize,
    declarer: Option<Seat>,
    /// hand game type chosen, modifier pending
    hand_ty: Option<u8>,
    contract: Option<Contract>,
    pickup: bool,
    discards: Vec<u8>,
    /// declarer's hand when card play starts (ISS declarerHandBeforeCardplay)
    decl_start: u32,
    // card play
    trick: Vec<(Seat, u32)>,
    tricks_done: usize,
    tricks_won: [u32; 3],
    points: [u32; 3],
    played_by: [i8; 32],
    played_trick: [i8; 32],
    played_pos: [i8; 32],
    trick_winner: [i8; 10],
    /// shown voids: bits 0..4 natural suits, bit 4 trump
    voids: [u8; 3],
}

fn rel(seat: Seat, viewer: Seat) -> usize {
    (seat + 3 - viewer) % 3
}

impl Skat {
    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn to_move(&self) -> Seat {
        self.to_move
    }

    pub fn max_bid(&self) -> u32 {
        self.max_bid
    }

    pub fn declarer(&self) -> Option<Seat> {
        self.declarer
    }

    pub fn contract(&self) -> Option<Contract> {
        self.contract
    }

    pub fn hand(&self, seat: Seat) -> u32 {
        self.hands[seat]
    }

    pub fn dealt_skat(&self) -> u32 {
        self.orig_skat
    }

    /// The skat as it counts for the declarer (dealt skat, or the discards).
    pub fn final_skat(&self) -> u32 {
        self.skat
    }

    pub fn picked_up(&self) -> bool {
        self.pickup
    }

    pub fn tricks_done(&self) -> usize {
        self.tricks_done
    }

    /// Seat that dealt position `k` (0..30) goes to; 3 = the skat.
    pub fn deal_owner(k: usize) -> usize {
        (k / 10).min(3)
    }

    /// The bid window of a bid decision: first bid index offered, end (exclusive), and
    /// whether MORE is offered.
    pub fn bid_window(&self) -> (usize, usize, bool) {
        let n = next_bid_index(self.max_bid);
        let lo = (n + if self.phase == Phase::BidMore { self.window } else { 0 }).min(BIDS.len());
        let hi = (lo + BID_WINDOW).min(BIDS.len());
        (lo, hi, hi < BIDS.len())
    }

    fn game_known_to(&self, seat: Seat) -> Option<u8> {
        match (self.contract, self.hand_ty) {
            (Some(c), _) if self.phase == Phase::Play || self.declarer == Some(seat) => Some(c.ty),
            (None, Some(t)) if self.declarer == Some(seat) => Some(t),
            _ => None,
        }
    }

    fn current_contract(&self) -> Contract {
        self.contract.expect("card play without a contract")
    }

    fn legal_plays(&self, seat: Seat) -> u32 {
        let h = self.hands[seat];
        let Some(&(_, lead)) = self.trick.first() else { return h };
        let c = self.current_contract();
        let led = c.eff_suit(lead);
        let follow = cards_of(h).filter(|&x| c.eff_suit(x) == led).fold(0, |m, x| m | bit(x));
        if follow != 0 {
            follow
        } else {
            h
        }
    }

    /// The trick's winning card so far and its seat.
    fn trick_winner_so_far(&self) -> Option<(Seat, u32)> {
        let (&(s0, c0), rest) = self.trick.split_first()?;
        let c = self.current_contract();
        let led = c.eff_suit(c0);
        let mut best = (s0, c0);
        for &(s, x) in rest {
            if c.beats(x, best.1, led) {
                best = (s, x);
            }
        }
        Some(best)
    }

    fn defenders_points(&self) -> u32 {
        let d = self.declarer.expect("declarer");
        (0..3).filter(|&s| s != d).map(|s| self.points[s]).sum()
    }

    fn start_play(&mut self) {
        let d = self.declarer.expect("declarer");
        self.decl_start = self.hands[d];
        self.phase = Phase::Play;
        self.to_move = 0;
    }

    fn play_card(&mut self, seat: Seat, c: u32) {
        let con = self.current_contract();
        if let Some(&(_, lead)) = self.trick.first() {
            let led = con.eff_suit(lead);
            if con.eff_suit(c) != led {
                self.voids[seat] |= 1 << led;
            }
        }
        self.hands[seat] &= !bit(c);
        self.played_by[c as usize] = seat as i8;
        self.played_trick[c as usize] = self.tricks_done as i8;
        self.played_pos[c as usize] = self.trick.len() as i8;
        self.trick.push((seat, c));
        if self.trick.len() < 3 {
            self.to_move = (seat + 1) % 3;
            return;
        }
        let (w, _) = self.trick_winner_so_far().expect("full trick");
        let pts: u32 = self.trick.iter().map(|&(_, x)| card_points(x)).sum();
        self.tricks_won[w] += 1;
        self.points[w] += pts;
        self.trick_winner[self.tricks_done] = w as i8;
        self.tricks_done += 1;
        self.trick.clear();
        self.to_move = w;
        let d = self.declarer.expect("declarer");
        let over = self.tricks_done == 10
            || if con.ty == NULL {
                w == d
            } else {
                w != d && (con.schwarz || (con.schneider && self.defenders_points() > 30))
            };
        if over {
            self.phase = Phase::Done;
        }
    }

    /// The game's result in ISS's terms at the current state, ended normally (at a
    /// terminal state) or by `end` (resignations / timeout / leaving at this state):
    /// ISS `SimpleState.gameResult` + `GameDeclaration.value`.
    pub fn iss_result(&self, end: &IssEnding) -> IssResult {
        let mut gr = IssResult {
            declarer: -1,
            left: end.left.map_or(-1, |s| s as i32),
            timeout: end.timeout.map_or(-1, |s| s as i32),
            ..Default::default()
        };
        // a game declared after the pickup counts as declared before its discards (ISS:
        // the declaration may come as a half-move; a declarer leaving / timing out then
        // is scored with the dealt skat as his only known cards)
        let declared = self.contract.is_some()
            && matches!(self.phase, Phase::Play | Phase::Done | Phase::Discard1 | Phase::Discard2);
        if !declared {
            if end.left.is_some() || end.timeout.is_some() {
                gr.penalty = true;
                match self.declarer {
                    None => {
                        for s in [end.left, end.timeout].into_iter().flatten() {
                            gr.penalties[s] = 1;
                        }
                    }
                    Some(d) => {
                        gr.declarer = d as i32;
                        if end.left == Some(d) || end.timeout == Some(d) {
                            gr.penalties[d] = 1;
                        } else {
                            gr.penalties = [1, 1, 1];
                            gr.penalties[d] = 0;
                        }
                    }
                }
            } else {
                gr.passed = true;
            }
            return gr;
        }
        let d = self.declarer.expect("declarer");
        let con = self.current_contract();
        let mut points = self.points[d] + mask_points(self.skat);
        let mut tricks = self.tricks_won[d];
        let defs_resigned = (0..3).filter(|&s| s != d && end.resigned[s]).count();
        let decl_loss = end.resigned[d] || end.timeout == Some(d) || end.left == Some(d);
        let defs_loss = !decl_loss
            && (defs_resigned >= 2 || end.timeout.is_some_and(|s| s != d) || end.left.is_some_and(|s| s != d));
        gr.resigned = end.resigned[d] || defs_resigned >= 2;
        if con.ty != NULL && defs_loss {
            let def_pts: u32 = (0..3).filter(|&s| s != d).map(|s| self.points[s]).sum();
            let def_tricks: u32 = (0..3).filter(|&s| s != d).map(|s| self.tricks_won[s]).sum();
            points = 120 - def_pts;
            tricks = 10 - def_tricks;
        }
        gr.declarer = d as i32;
        gr.points = points as i32;
        gr.tricks = tricks as i32;
        if con.ty == NULL {
            let v = [23, 35, 46, 59][(con.ouvert as usize) * 2 + con.hand as usize];
            gr.value = if tricks > 0 || decl_loss { -2 * v } else { v };
            return gr;
        }
        let h = self.decl_start | self.skat;
        let mut m = trump_multiplier(h, con.ty);
        gr.matadors = if h & bit(card(3, JACK)) != 0 { m as i32 - 1 } else { 1 - m as i32 };
        let won = if con.schwarz {
            tricks == 10
        } else if con.schneider {
            points >= 90
        } else {
            points >= 61
        };
        m += con.hand as u32 + con.schneider as u32 + con.schwarz as u32 + con.ouvert as u32;
        let bv = BASE_VALUE[con.ty as usize];
        let decl_cards = self.played_by.iter().filter(|&&s| s == d as i8).count();
        if won {
            if points >= 90 {
                m += 1;
                gr.schneider = true;
            }
            if tricks == 10 {
                m += 1;
                gr.schneider = true;
                gr.schwarz = true;
            }
            if m * bv >= self.max_bid {
                gr.value = (m * bv) as i32;
                return gr;
            }
        } else {
            // (the ISS server, pinned by its records — its GPL client code differs): a
            // lost game counts the announced schneider / schwarz levels too (without
            // flagging them); otherwise the declarer's own schneider (<= 30 points) and
            // schwarz (no trick) count only once the declarer has played two cards (no
            // penalty for resigning / timing out / leaving before that; all 12k lost
            // games of the archive that ended early agree)
            if con.schneider {
                m += 1 + con.schwarz as u32;
            } else if decl_cards >= 2 && points <= 30 {
                m += 1;
                gr.schneider = true;
                if tricks == 0 {
                    m += 1;
                    gr.schwarz = true;
                }
            }
            if m * bv >= self.max_bid {
                gr.value = -2 * (m * bv) as i32;
                return gr;
            }
        }
        gr.overbid = true;
        gr.value = -2 * (self.max_bid.div_ceil(bv) * bv) as i32;
        gr
    }

    /// The normal-end result at a terminal state.
    pub fn final_result(&self) -> IssResult {
        self.iss_result(&IssEnding::default())
    }

    // ---- encoders --------------------------------------------------------------

    fn bid_bits(f: &mut [f32], value: u32) {
        // one-hot of the bid index (0..23 = 18..72, 24 = higher) + its 7-bit code
        let Some(i) = bid_index(value) else { return };
        f[i.min(24)] = 1.0;
        for b in 0..7 {
            if (i >> b) & 1 == 1 {
                f[25 + b] = 1.0;
            }
        }
    }

    fn points_bits(f: &mut [f32], pts: u32) {
        for k in 0..24 {
            if pts >= 5 * (k as u32 + 1) {
                f[k] = 1.0;
            }
        }
        f[24] = (pts >= 31) as u8 as f32;
        f[25] = (pts >= 61) as u8 as f32;
        f[26] = (pts >= 90) as u8 as f32;
    }
}

// entity / global / action feature layout (see the module docs)
pub const N_ENTITIES: usize = 35;
pub const ENTITY_FEATURES: usize = 49;
pub const GLOBAL_FEATURES: usize = 149;
pub const MAX_ACTIONS: usize = 12;
pub const ACTION_FEATURES: usize = 68;
pub const MAX_DECISIONS: usize = 192;

// card entities
const C_SUIT: usize = 0;
const C_RANK: usize = 4;
const C_TRUMP: usize = 12;
const C_EFF: usize = 13;
const C_MINE: usize = 18;
const C_FROM_SKAT: usize = 19;
const C_DISCARDED: usize = 20;
const C_OPEN: usize = 21;
const C_PLAYED: usize = 22;
const C_BY: usize = 23;
const C_TRICKNO: usize = 26;
const C_POS: usize = 36;
const C_CURRENT: usize = 39;
const C_WON_BY: usize = 40;
const C_UNKNOWN: usize = 43;
// seat entities
const S_REL: usize = 0;
const S_POS: usize = 3;
const S_DECL: usize = 6;
const S_DEF: usize = 7;
const S_PASSED: usize = 8;
const S_BIDDER: usize = 9;
const S_ASKED: usize = 10;
const S_BID: usize = 11;
const S_VOID: usize = 43;
const S_TOMOVE: usize = 48;
// globals
const G_PHASE: usize = 0;
const G_POS: usize = 9;
const G_GAME: usize = 12;
const G_HAND: usize = 18;
const G_OUVERT: usize = 19;
const G_SCHNEIDER: usize = 20;
const G_SCHWARZ: usize = 21;
const G_PICKUP: usize = 22;
const G_DECL: usize = 23;
const G_TRICK: usize = 26;
const G_TRICKN: usize = 36;
const G_LEADER: usize = 39;
const G_DPTS: usize = 42;
const G_FPTS: usize = 69;
const G_BID: usize = 96;
const G_SKATPTS: usize = 128;
const G_DTRICKS: usize = 139;
// actions
const A_KIND: usize = 0;
const A_SUIT: usize = 10;
const A_RANK: usize = 14;
const A_TRUMP: usize = 22;
const A_WINS: usize = 23;
const A_FOLLOWS: usize = 24;
const A_BID: usize = 25;
const A_GAME: usize = 57;
const A_OUVERT: usize = 63;
const A_MOD: usize = 64;

const K_PLAY: usize = 0;
const K_DISCARD: usize = 1;
const K_PASS: usize = 2;
const K_YES: usize = 3;
const K_BID: usize = 4;
const K_MORE: usize = 5;
const K_PICKUP: usize = 6;
const K_HAND: usize = 7;
const K_MOD: usize = 8;
const K_DECLARE: usize = 9;

impl Game for Skat {
    type Config = SkatConfig;
    const NAME: &'static str = "skat";
    const VERSION: u32 = 1;

    fn spec(_cfg: &SkatConfig) -> FrameSpec {
        FrameSpec::candidate(CandidateSpec {
            n_seats: 3,
            max_entities: N_ENTITIES,
            entity_features: ENTITY_FEATURES,
            entity_types: 0,
            global_features: GLOBAL_FEATURES,
            max_actions: MAX_ACTIONS,
            action_features: ACTION_FEATURES,
            hidden_info: true,
            simultaneous: false,
            bits_entities: true,
            bits_globals: true,
            bits_actions: true,
            max_decisions: MAX_DECISIONS,
        })
    }

    fn new_game(_cfg: &SkatConfig) -> Self {
        Skat {
            deal: Vec::with_capacity(32),
            hands: [0; 3],
            orig_skat: 0,
            skat: 0,
            phase: Phase::Deal,
            to_move: 1,
            bidder: Some(1),
            asked: Some(0),
            max_bid: 0,
            player_bid: [0; 3],
            passed: [false; 3],
            window: 0,
            declarer: None,
            hand_ty: None,
            contract: None,
            pickup: false,
            discards: Vec::new(),
            decl_start: 0,
            trick: Vec::with_capacity(3),
            tricks_done: 0,
            tricks_won: [0; 3],
            points: [0; 3],
            played_by: [-1; 32],
            played_trick: [-1; 32],
            played_pos: [-1; 32],
            trick_winner: [-1; 10],
            voids: [0; 3],
        }
    }

    fn turn(&self) -> Turn {
        match self.phase {
            Phase::Deal => Turn::Chance,
            Phase::Done => Turn::Terminal,
            _ => Turn::Act(SeatSet::single(self.to_move)),
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        let dealt: u32 = self.deal.iter().fold(0, |m, &c| m | bit(c as u32));
        let p = 1.0 / (32 - self.deal.len()) as f64;
        out.extend((0..32u32).filter(|&c| dealt & bit(c) == 0).map(|c| (c, p)));
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.phase != Phase::Deal {
            rules_bail!("skat: no chance node");
        }
        if outcome >= 32 || self.deal.iter().any(|&c| c as u32 == outcome) {
            rules_bail!("skat: card {outcome} cannot be dealt");
        }
        let k = self.deal.len();
        self.deal.push(outcome as u8);
        match Self::deal_owner(k) {
            3 => self.orig_skat |= bit(outcome),
            s => self.hands[s] |= bit(outcome),
        }
        if self.deal.len() == 32 {
            self.skat = self.orig_skat;
            self.phase = Phase::Bid;
            self.to_move = 1;
        }
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if seat != self.to_move {
            return;
        }
        match self.phase {
            Phase::Deal | Phase::Done => {}
            Phase::Bid | Phase::BidMore => {
                if self.phase == Phase::Bid {
                    out.push(PASS);
                }
                let (lo, hi, more) = self.bid_window();
                out.extend((lo..hi).map(|i| BID0 + i as u32));
                if more {
                    out.push(MORE);
                }
            }
            Phase::Answer => out.extend([PASS, YES]),
            Phase::SkatOrHand => {
                out.push(PICKUP);
                out.extend((0..5).map(|t| HAND0 + t));
                if self.max_bid <= MAX_NULL_OUVERT_HAND_BID {
                    out.push(HAND0 + NULL as u32);
                }
            }
            Phase::HandModifier => {
                if self.hand_ty == Some(NULL) {
                    if self.max_bid <= MAX_NULL_HAND_BID {
                        out.push(MOD0 + MOD_PLAIN);
                    }
                    if self.max_bid <= MAX_NULL_OUVERT_HAND_BID {
                        out.push(MOD0 + MOD_OUVERT);
                    }
                } else {
                    out.extend((0..4).map(|m| MOD0 + m));
                }
            }
            Phase::Declare => {
                out.extend((0..5).map(|t| DECLARE0 + t));
                if self.max_bid <= MAX_NULL_BID {
                    out.push(DECLARE0 + NULL as u32);
                }
                if self.max_bid <= MAX_NULL_OUVERT_BID {
                    out.push(DECLARE0 + DECL_NULL_OUVERT);
                }
            }
            Phase::Discard1 | Phase::Discard2 => out.extend(cards_of(self.hands[seat]).map(|c| DISCARD0 + c)),
            Phase::Play => out.extend(cards_of(self.legal_plays(seat)).map(|c| PLAY0 + c)),
        }
    }

    fn apply_action(&mut self, seat: Seat, a: u32) -> Result<(), RulesError> {
        if matches!(self.phase, Phase::Deal | Phase::Done) || seat != self.to_move {
            rules_bail!("skat: seat {seat} is not to act ({:?}, to move {})", self.phase, self.to_move);
        }
        let mut legal = Vec::new();
        self.legal_actions(seat, &mut legal);
        if legal.binary_search(&a).is_err() {
            rules_bail!("skat: action {} ({a}) is illegal for seat {seat} in {:?}", self.action_string(seat, a), self.phase);
        }
        match self.phase {
            Phase::Bid | Phase::BidMore => {
                if a == MORE {
                    self.window = if self.phase == Phase::Bid { BID_WINDOW } else { self.window + BID_WINDOW };
                    self.phase = Phase::BidMore;
                    return Ok(());
                }
                self.window = 0;
                self.phase = Phase::Bid;
                let bidder = self.bidder.expect("bidder");
                if a == PASS {
                    if self.player_bid[seat] == 0 {
                        self.player_bid[seat] = 1;
                    }
                    self.passed[seat] = true;
                    match bidder {
                        0 => {
                            // nobody bid: passed in
                            self.phase = Phase::Done;
                            self.bidder = None;
                        }
                        1 => {
                            self.bidder = Some(2);
                            self.asked = Some(0);
                            self.to_move = 2;
                        }
                        _ => {
                            if self.max_bid < 18 {
                                self.bidder = Some(0);
                                self.asked = None;
                                self.to_move = 0;
                            } else {
                                let d = self.asked.expect("asked");
                                self.declarer = Some(d);
                                self.bidder = None;
                                self.asked = None;
                                self.phase = Phase::SkatOrHand;
                                self.to_move = d;
                            }
                        }
                    }
                } else {
                    let v = BIDS[(a - BID0) as usize];
                    self.player_bid[seat] = v;
                    self.max_bid = v;
                    if seat == 0 {
                        // forehand bids after both others passed: it plays
                        self.declarer = Some(0);
                        self.bidder = None;
                        self.asked = None;
                        self.phase = Phase::SkatOrHand;
                        self.to_move = 0;
                    } else {
                        self.phase = Phase::Answer;
                        self.to_move = self.asked.expect("asked");
                    }
                }
            }
            Phase::Answer => {
                let bidder = self.bidder.expect("bidder");
                if a == YES {
                    self.player_bid[seat] = self.max_bid;
                    self.phase = Phase::Bid;
                    self.to_move = bidder;
                } else {
                    if self.player_bid[seat] == 0 {
                        self.player_bid[seat] = 1;
                    }
                    self.passed[seat] = true;
                    if bidder == 1 {
                        self.phase = Phase::Bid;
                        self.bidder = Some(2);
                        self.asked = Some(1);
                        self.to_move = 2;
                    } else {
                        self.declarer = Some(bidder);
                        self.bidder = None;
                        self.asked = None;
                        self.phase = Phase::SkatOrHand;
                        self.to_move = bidder;
                    }
                }
            }
            Phase::SkatOrHand => {
                if a == PICKUP {
                    self.pickup = true;
                    self.hands[seat] |= self.orig_skat;
                    self.phase = Phase::Declare;
                } else {
                    self.hand_ty = Some((a - HAND0) as u8);
                    self.phase = Phase::HandModifier;
                }
            }
            Phase::HandModifier => {
                let ty = self.hand_ty.take().expect("hand type");
                let m = a - MOD0;
                let ouvert = m == MOD_OUVERT;
                self.contract = Some(if ty == NULL {
                    Contract { ty, hand: true, ouvert, schneider: false, schwarz: false }
                } else {
                    Contract { ty, hand: true, ouvert, schneider: m >= MOD_SCHNEIDER, schwarz: m >= MOD_SCHWARZ }
                });
                self.start_play();
            }
            Phase::Declare => {
                let d = a - DECLARE0;
                self.contract = Some(if d == DECL_NULL_OUVERT {
                    Contract { ty: NULL, hand: false, ouvert: true, schneider: false, schwarz: false }
                } else {
                    Contract { ty: d as u8, hand: false, ouvert: false, schneider: false, schwarz: false }
                });
                self.phase = Phase::Discard1;
            }
            Phase::Discard1 | Phase::Discard2 => {
                let c = a - DISCARD0;
                self.hands[seat] &= !bit(c);
                self.discards.push(c as u8);
                if self.phase == Phase::Discard1 {
                    self.phase = Phase::Discard2;
                } else {
                    self.skat = self.discards.iter().fold(0, |m, &x| m | bit(x as u32));
                    self.start_play();
                }
            }
            Phase::Play => self.play_card(seat, a - PLAY0),
            Phase::Deal | Phase::Done => unreachable!(),
        }
        Ok(())
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns().iter().map(|&r| (r / MONEY_SCALE) as f32).collect()
    }

    fn returns(&self) -> Vec<f64> {
        if self.phase != Phase::Done {
            return vec![0.0; 3];
        }
        self.final_result().money().to_vec()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        let game = self.game_known_to(seat);
        let is_decl = self.declarer == Some(seat);
        let open = self.phase == Phase::Play
            && !is_decl
            && self.contract.is_some_and(|c| c.ouvert);
        let decl = self.declarer;
        for c in 0..32u32 {
            let f = obs.entity(c as usize, 0);
            f[C_SUIT + suit(c) as usize] = 1.0;
            f[C_RANK + rank(c) as usize] = 1.0;
            if let Some(ty) = game {
                // the type is known before the full contract (hand modifier pending)
                let k = Contract { ty, hand: false, ouvert: false, schneider: false, schwarz: false };
                f[C_TRUMP] = k.is_trump(c) as u8 as f32;
                f[C_EFF + k.eff_suit(c) as usize] = 1.0;
            }
            let mine = self.hands[seat] & bit(c) != 0;
            f[C_MINE] = mine as u8 as f32;
            let knows_skat = is_decl && self.pickup;
            let from_skat = knows_skat && self.orig_skat & bit(c) != 0;
            f[C_FROM_SKAT] = from_skat as u8 as f32;
            let discarded = knows_skat && self.discards.contains(&(c as u8));
            f[C_DISCARDED] = discarded as u8 as f32;
            let in_open = open && decl.is_some_and(|d| self.hands[d] & bit(c) != 0);
            f[C_OPEN] = in_open as u8 as f32;
            let by = self.played_by[c as usize];
            if by >= 0 {
                let t = self.played_trick[c as usize] as usize;
                f[C_PLAYED] = 1.0;
                f[C_BY + rel(by as usize, seat)] = 1.0;
                f[C_TRICKNO + t] = 1.0;
                f[C_POS + self.played_pos[c as usize] as usize] = 1.0;
                if t == self.tricks_done {
                    f[C_CURRENT] = 1.0;
                } else {
                    f[C_WON_BY + rel(self.trick_winner[t] as usize, seat)] = 1.0;
                }
            }
            if !mine && by < 0 && !from_skat && !discarded && !in_open {
                f[C_UNKNOWN] = 1.0;
            }
        }
        for k in 0..3 {
            let s = (seat + k) % 3;
            let f = obs.entity(32 + k, 0);
            f[S_REL + k] = 1.0;
            f[S_POS + s] = 1.0;
            if let Some(d) = decl {
                // the bidding's winner is public as soon as the bidding ends
                f[S_DECL] = (d == s) as u8 as f32;
                f[S_DEF] = (d != s) as u8 as f32;
            }
            f[S_PASSED] = self.passed[s] as u8 as f32;
            f[S_BIDDER] = (self.bidder == Some(s) && decl.is_none() && self.phase != Phase::Done) as u8 as f32;
            f[S_ASKED] = (self.asked == Some(s) && decl.is_none() && self.phase != Phase::Done) as u8 as f32;
            if self.player_bid[s] >= 18 {
                Self::bid_bits(&mut f[S_BID..S_BID + 32], self.player_bid[s]);
            }
            for b in 0..5 {
                f[S_VOID + b] = ((self.voids[s] >> b) & 1) as f32;
            }
            f[S_TOMOVE] = (self.to_move == s) as u8 as f32;
        }
        let g = obs.globals();
        // (decision frames never exist in the deal / after the end)
        if self.phase.index() < 9 {
            g[G_PHASE + self.phase.index()] = 1.0;
        }
        g[G_POS + seat] = 1.0;
        if let Some(ty) = game {
            g[G_GAME + ty as usize] = 1.0;
        }
        if let (Some(c), Some(_)) = (self.contract, game) {
            g[G_HAND] = c.hand as u8 as f32;
            g[G_OUVERT] = c.ouvert as u8 as f32;
            g[G_SCHNEIDER] = c.schneider as u8 as f32;
            g[G_SCHWARZ] = c.schwarz as u8 as f32;
        } else if is_decl && self.hand_ty.is_some() {
            g[G_HAND] = 1.0;
        }
        g[G_PICKUP] = self.pickup as u8 as f32;
        if let Some(d) = decl {
            g[G_DECL + rel(d, seat)] = 1.0;
        }
        if self.phase == Phase::Play {
            g[G_TRICK + self.tricks_done.min(9)] = 1.0;
            g[G_TRICKN + self.trick.len().min(2)] = 1.0;
            let leader = self.trick.first().map_or(self.to_move, |&(s, _)| s);
            g[G_LEADER + rel(leader, seat)] = 1.0;
        }
        if let Some(d) = decl {
            Self::points_bits(&mut g[G_DPTS..G_DPTS + 27], self.points[d]);
            let fp: u32 = (0..3).filter(|&s| s != d).map(|s| self.points[s]).sum();
            Self::points_bits(&mut g[G_FPTS..G_FPTS + 27], fp);
            for k in 0..10 {
                g[G_DTRICKS + k] = (self.tricks_won[d] > k as u32) as u8 as f32;
            }
        }
        if self.max_bid >= 18 {
            Self::bid_bits(&mut g[G_BID..G_BID + 32], self.max_bid);
        }
        if is_decl && self.pickup {
            let known = if self.discards.is_empty() {
                self.orig_skat
            } else {
                self.discards.iter().fold(0, |m, &x| m | bit(x as u32))
            };
            let pts = mask_points(known);
            for k in 0..11 {
                g[G_SKATPTS + k] = (pts >= 2 * (k as u32 + 1)) as u8 as f32;
            }
        }
    }

    fn encode_action(&self, seat: Seat, a: u32, f: &mut [f32]) {
        let card_feat = |f: &mut [f32], c: u32| {
            f[A_SUIT + suit(c) as usize] = 1.0;
            f[A_RANK + rank(c) as usize] = 1.0;
        };
        match a {
            _ if a < DISCARD0 => {
                let c = a - PLAY0;
                f[A_KIND + K_PLAY] = 1.0;
                card_feat(f, c);
                let con = self.current_contract();
                f[A_TRUMP] = con.is_trump(c) as u8 as f32;
                if let (Some(&(_, lead)), Some((_, win))) = (self.trick.first(), self.trick_winner_so_far()) {
                    let led = con.eff_suit(lead);
                    f[A_WINS] = con.beats(c, win, led) as u8 as f32;
                    f[A_FOLLOWS] = (con.eff_suit(c) == led) as u8 as f32;
                }
            }
            _ if a < PASS => {
                let c = a - DISCARD0;
                f[A_KIND + K_DISCARD] = 1.0;
                card_feat(f, c);
                if let Some(con) = self.contract {
                    f[A_TRUMP] = con.is_trump(c) as u8 as f32;
                }
            }
            PASS => f[A_KIND + K_PASS] = 1.0,
            YES => f[A_KIND + K_YES] = 1.0,
            _ if a < MORE => {
                f[A_KIND + K_BID] = 1.0;
                Self::bid_bits(&mut f[A_BID..A_BID + 32], BIDS[(a - BID0) as usize]);
            }
            MORE => {
                f[A_KIND + K_MORE] = 1.0;
                let (_, hi, _) = self.bid_window();
                if hi < BIDS.len() {
                    Self::bid_bits(&mut f[A_BID..A_BID + 32], BIDS[hi]);
                }
            }
            PICKUP => f[A_KIND + K_PICKUP] = 1.0,
            _ if a < MOD0 => {
                f[A_KIND + K_HAND] = 1.0;
                f[A_GAME + (a - HAND0) as usize] = 1.0;
            }
            _ if a < DECLARE0 => {
                f[A_KIND + K_MOD] = 1.0;
                f[A_MOD + (a - MOD0) as usize] = 1.0;
                if let Some(t) = self.hand_ty {
                    f[A_GAME + t as usize] = 1.0;
                }
            }
            _ => {
                let d = a - DECLARE0;
                f[A_KIND + K_DECLARE] = 1.0;
                if d == DECL_NULL_OUVERT {
                    f[A_GAME + NULL as usize] = 1.0;
                    f[A_OUVERT] = 1.0;
                } else {
                    f[A_GAME + d as usize] = 1.0;
                }
            }
        }
        let _ = seat;
    }

    fn action_string(&self, _seat: Seat, a: u32) -> String {
        match a {
            _ if a < DISCARD0 => card_str(a),
            _ if a < PASS => format!("discard {}", card_str(a - DISCARD0)),
            PASS => "p".into(),
            YES => "y".into(),
            _ if a < MORE => BIDS[(a - BID0) as usize].to_string(),
            MORE => "more".into(),
            PICKUP => "s".into(),
            _ if a < MOD0 => format!("{}H", GAME_CHARS[(a - HAND0) as usize]),
            _ if a < DECLARE0 => ["plain", "schneider", "schwarz", "ouvert"][(a - MOD0) as usize].into(),
            _ if a < N_ACTIONS => {
                let d = a - DECLARE0;
                if d == DECL_NULL_OUVERT { "NO".into() } else { GAME_CHARS[d as usize].to_string() }
            }
            _ => format!("?{a}"),
        }
    }

    fn chance_string(&self, outcome: u32) -> String {
        if outcome < 32 {
            card_str(outcome)
        } else {
            format!("?{outcome}")
        }
    }

    fn oracle_dim(_cfg: &SkatConfig) -> usize {
        4 * 32
    }

    /// Every seat's hand (one-hot over the 32 cards; observer first, then the following
    /// seats) and the skat cards no hand holds (the dealt skat, or the declarer's discards;
    /// while the declarer holds the picked-up skat, those cards are in its hand only).
    fn encode_oracle(&self, observer: Seat, out: &mut [f32]) {
        for k in 0..3 {
            let hand = self.hands[(observer + k) % 3];
            for c in 0..32 {
                if hand >> c & 1 == 1 {
                    out[k * 32 + c] = 1.0;
                }
            }
        }
        let skat = self.skat & !(self.hands[0] | self.hands[1] | self.hands[2]);
        for c in 0..32 {
            if skat >> c & 1 == 1 {
                out[3 * 32 + c] = 1.0;
            }
        }
    }

    fn state_string(&self) -> String {
        let h = |m: u32| cards_of(m).map(card_str).collect::<Vec<_>>().join(".");
        format!(
            "phase={:?} to_move={} hands=[{}|{}|{}] skat={} bid={} declarer={:?} contract={} trick={:?} tricks={} points={:?}",
            self.phase,
            self.to_move,
            h(self.hands[0]),
            h(self.hands[1]),
            h(self.hands[2]),
            h(self.skat),
            self.max_bid,
            self.declarer,
            self.contract.map_or("-".into(), |c| c.iss_string()),
            self.trick.iter().map(|&(s, c)| format!("{s}:{}", card_str(c))).collect::<Vec<_>>(),
            self.tricks_done,
            self.points
        )
    }

    /// Swap two dealt cards whose whereabouts `seat` never observes up to the end of
    /// `events`, keeping every recorded action legal: cards neither played, nor in
    /// `seat`'s hand, nor seen by it (its own pickup / discards, the declarer's open
    /// hand) moved between two other hands / the unseen skat — the declarer's
    /// discards follow a card that leaves its hand — or, for a defender after the
    /// declarer's pickup, a card moved between the declarer's dealt hand and the skat
    /// (the declarer's 12 cards stay the same).
    fn perturb_hidden(cfg: &SkatConfig, events: &[Event], seat: Seat, rng: &mut Rng) -> Option<Vec<Event>> {
        if seat >= 3 || events.len() < 32 {
            return None;
        }
        let deal: Vec<u32> = events[..32].iter().map(|e| if let Event::Chance(c) = e { Some(*c) } else { None }).collect::<Option<_>>()?;
        // the final state: who played what, the pickup, the open hand
        let mut g = Skat::new_game(cfg);
        for e in events {
            match *e {
                Event::Chance(c) => g.apply_chance(c).ok()?,
                Event::Act(s, a) => g.apply_action(s, a).ok()?,
            }
        }
        let decl = g.declarer;
        let is_decl = decl == Some(seat);
        let played = |c: u32| g.played_by[c as usize] >= 0;
        // ouvert: a defender sees the declarer's whole hand from the declaration on
        let seen_open: u32 = match g.contract {
            Some(c) if c.ouvert && !is_decl && matches!(g.phase, Phase::Play | Phase::Done) => g.decl_start,
            _ => 0,
        };
        let knows_skat = is_decl && g.pickup;
        let hidden = |k: usize| -> bool {
            let c = deal[k];
            let owner = Self::deal_owner(k);
            owner != seat
                && !played(c)
                && seen_open & bit(c) == 0
                && !(knows_skat && owner == 3)
        };
        // owner groups from seat's view: the declarer's dealt hand and the skat form one
        // pool after the declarer's pickup (seat a defender)
        let pool = !is_decl && g.pickup;
        let group = |k: usize| -> usize {
            let o = Self::deal_owner(k);
            if pool && (o == 3 || Some(o) == decl) {
                4
            } else {
                o
            }
        };
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..32 {
            for j in i + 1..32 {
                let (oi, oj) = (Self::deal_owner(i), Self::deal_owner(j));
                if oi == seat || oj == seat || oi == oj {
                    continue;
                }
                let within_pool = pool && group(i) == 4 && group(j) == 4;
                let open_i = seen_open & bit(deal[i]) != 0;
                let open_j = seen_open & bit(deal[j]) != 0;
                if within_pool && !open_i && !open_j {
                    pairs.push((i, j));
                } else if group(i) != group(j) && hidden(i) && hidden(j) {
                    pairs.push((i, j));
                }
            }
        }
        rng.shuffle(&mut pairs);
        for &(i, j) in pairs.iter().take(48) {
            let (a, b) = (deal[i], deal[j]);
            let mut out = events.to_vec();
            out[i] = Event::Chance(b);
            out[j] = Event::Chance(a);
            // a discarded card that leaves the declarer's 12 cards is replaced by the
            // card that enters them
            if let Some(d) = decl {
                let pool_k = |k: usize| Self::deal_owner(k) == d || Self::deal_owner(k) == 3;
                if g.pickup && pool_k(i) != pool_k(j) {
                    let (gone, came) = if pool_k(i) { (a, b) } else { (b, a) };
                    for e in out.iter_mut().skip(32) {
                        if *e == Event::Act(d, DISCARD0 + gone) {
                            *e = Event::Act(d, DISCARD0 + came);
                        }
                    }
                }
            }
            if out == events {
                continue;
            }
            let mut h = Skat::new_game(cfg);
            let ok = out.iter().all(|e| match *e {
                Event::Chance(c) => h.apply_chance(c).is_ok(),
                Event::Act(s, x) => h.apply_action(s, x).is_ok(),
            });
            if ok {
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

    fn c(s: &str) -> u32 {
        parse_card(s).unwrap()
    }

    fn mask(cards: &str) -> u32 {
        cards.split('.').map(c).fold(0, |m, x| m | bit(x))
    }

    /// Deal events from an ISS deal string (32 cards, `.`-separated).
    fn deal_events(deal: &str) -> Vec<Event> {
        let v: Vec<Event> = deal.split('.').map(|x| Event::Chance(c(x))).collect();
        assert_eq!(v.len(), 32);
        v
    }

    fn bid(v: u32) -> u32 {
        BID0 + bid_index(v).unwrap() as u32
    }

    const DEAL: &str = "S7.ST.CA.DT.CQ.S8.C8.D9.HT.SQ.HQ.DK.H9.DJ.HK.HJ.C9.DQ.HA.H8.SA.CJ.CK.DA.H7.C7.S9.CT.SK.D7.SJ.D8";

    #[test]
    fn cards_and_contracts() {
        assert_eq!((c("D7"), c("DJ"), c("HT"), c("CA")), (0, 4, 11, 31));
        assert_eq!(card_str(31), "CA");
        assert_eq!(parse_card("XX"), None);
        assert_eq!((0..32).map(card_points).sum::<u32>(), 120);
        for s in ["G", "CHS", "NOH", "GO", "HH", "N", "NO", "GHZ"] {
            assert_eq!(Contract::parse_iss(s).unwrap().iss_string(), s);
        }
        assert_eq!(Contract::parse_iss("NHO").unwrap().iss_string(), "NOH");
        let go = Contract::parse_iss("GO").unwrap();
        assert!(go.hand && go.schneider && go.schwarz && go.ouvert);
        assert_eq!(Contract::parse_iss("GHZ").unwrap().schneider, true);
        assert!(Contract::parse_iss("GS").is_none() && Contract::parse_iss("NHS").is_none() && Contract::parse_iss("X").is_none());
        // trump order: CJ > SJ > HJ > DJ > trump A > T > K > Q > 9 > 8 > 7; null: natural
        let hearts = Contract { ty: HEARTS, hand: false, ouvert: false, schneider: false, schwarz: false };
        assert!(hearts.beats(c("DJ"), c("HA"), 4) && hearts.beats(c("CJ"), c("SJ"), 4));
        assert!(hearts.beats(c("HT"), c("HK"), 4) && !hearts.beats(c("HK"), c("HT"), 4));
        assert!(hearts.beats(c("H7"), c("SA"), 2) && !hearts.beats(c("CA"), c("S7"), 2));
        let null = Contract { ty: NULL, hand: false, ouvert: false, schneider: false, schwarz: false };
        assert!(null.beats(c("SJ"), c("ST"), 2) && !null.beats(c("SJ"), c("SQ"), 2) && !null.beats(c("CA"), c("S7"), 2));
        // matadors: with 2 (CJ SJ, no HJ); without 3 (no CJ SJ HJ, DJ); clubs with 4 + CA CT
        assert_eq!(trump_multiplier(mask("CJ.SJ.D7"), GRAND), 3);
        assert_eq!(trump_multiplier(mask("DJ.CA"), CLUBS), 4);
        assert_eq!(trump_multiplier(mask("CJ.SJ.HJ.DJ.CA.CT.C9"), CLUBS), 7);
        assert_eq!(trump_multiplier(0, CLUBS), 12);
        assert_eq!(trump_multiplier(0, GRAND), 5);
    }

    #[test]
    fn iss_bidding() {
        let base = deal_events(DEAL);
        let play = |acts: &[(Seat, u32)]| {
            let mut e = base.clone();
            e.extend(acts.iter().map(|&(s, a)| Event::Act(s, a)));
            replay::<Skat>(&SkatConfig {}, &e)
        };
        // ISS game 9: 1 18, 0 p, 2 20, 1 p -> rearhand declares at 20
        let r = play(&[(1, bid(18)), (0, PASS), (2, bid(20)), (1, PASS)]).unwrap();
        assert_eq!((r.state.declarer, r.state.max_bid, r.state.phase), (Some(2), 20, Phase::SkatOrHand));
        // all pass: passed in, everybody scores 0
        let r = play(&[(1, PASS), (2, PASS), (0, PASS)]).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.returns(), vec![0.0; 3]);
        assert_eq!(r.state.final_result().iss_string(), "passed");
        // middlehand and rearhand pass, forehand bids and plays
        let r = play(&[(1, PASS), (2, PASS), (0, bid(18))]).unwrap();
        assert_eq!(r.state.declarer, Some(0));
        // forehand holds 18, middlehand passes, rearhand passes: forehand plays 18
        let r = play(&[(1, bid(18)), (0, YES), (1, PASS), (2, PASS)]).unwrap();
        assert_eq!((r.state.declarer, r.state.max_bid), (Some(0), 18));
        // jump bid 18 -> 30 (within the window) and via MORE (18 -> 77: index 24)
        assert!(play(&[(1, bid(30))]).is_ok());
        assert!(play(&[(1, bid(77))]).is_err());
        let r = play(&[(1, MORE), (1, MORE), (1, bid(77)), (0, YES), (1, PASS), (2, PASS)]).unwrap();
        assert_eq!((r.state.declarer, r.state.max_bid), (Some(0), 77));
        // wrong seat / illegal bids are rejected
        assert!(play(&[(0, bid(18))]).is_err());
        assert!(play(&[(1, bid(18)), (0, YES), (1, bid(18))]).is_err());
        assert!(play(&[(1, bid(18)), (0, bid(20))]).is_err());
    }

    /// ISS record 9 (2007-10-29): rearhand plays clubs hand at 20, loses with 56 points
    /// (clubs with 2, hand: 4 x 12 = 48, lost -96).
    #[test]
    fn iss_game_9_clubs_hand_lost() {
        let mut e = deal_events(DEAL);
        let acts: Vec<(Seat, u32)> = vec![(1, bid(18)), (0, PASS), (2, bid(20)), (1, PASS), (2, HAND0 + CLUBS as u32), (2, MOD0 + MOD_PLAIN)];
        e.extend(acts.iter().map(|&(s, a)| Event::Act(s, a)));
        let plays = "0 SQ 1 H9 2 SA 2 SK 0 ST 1 HK 0 D9 1 DQ 2 DA 2 D7 0 DT 1 DK 0 HT 1 H8 2 H7 0 S8 1 HQ 2 S9 2 C7 0 CA 1 DJ 1 C9 2 CT 0 C8 2 CK 0 CQ 1 HJ 1 HA 2 CJ 0 S7";
        let t: Vec<&str> = plays.split(' ').collect();
        for p in t.chunks(2) {
            e.push(Event::Act(p[0].parse().unwrap(), PLAY0 + c(p[1])));
        }
        let r = replay::<Skat>(&SkatConfig {}, &e).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.final_result().iss_string(),
                   "d:2 loss v:-96 m:2 bidok p:56 t:5 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0");
        let money = r.state.returns();
        assert_eq!(money, vec![186.0, 186.0, -372.0]);
        assert!((r.state.outcome()[2] - (-372.0 / 1260.0) as f32).abs() < 1e-7);
    }

    #[test]
    fn scoring_rules() {
        // a synthetic state: declarer 0 with the given 12 cards, the result from points;
        // `played`: cards the declarer has played (only their number matters here)
        let st = |ty: u8, hand: bool, sa: bool, za: bool, ouvert: bool, h12: u32, pts: u32, tricks: u32, bid: u32, played: usize| {
            let mut g = Skat::new_game(&SkatConfig {});
            g.phase = Phase::Done;
            g.declarer = Some(0);
            g.contract = Some(Contract { ty, hand, ouvert, schneider: sa, schwarz: za });
            g.skat = 0;
            g.decl_start = h12;
            g.points = [pts, 120 - pts, 0];
            g.tricks_won = [tricks, 10 - tricks, 0];
            for c in 0..played {
                g.played_by[c] = 0;
            }
            g.max_bid = bid;
            g.final_result()
        };
        let with2 = mask("CJ.SJ");
        // grand with 2 won: 3 x 24; schneider 4 x 24; schwarz 5 x 24
        assert_eq!(st(GRAND, false, false, false, false, with2, 61, 5, 18, 10).value, 72);
        assert_eq!(st(GRAND, false, false, false, false, with2, 95, 8, 18, 10).value, 96);
        assert_eq!(st(GRAND, false, false, false, false, with2, 120, 10, 18, 10).value, 120);
        // lost with 60: -2 x 72; schneidered (<= 30): -2 x 96; schwarz: -2 x 120
        assert_eq!(st(GRAND, false, false, false, false, with2, 60, 4, 18, 10).value, -144);
        assert_eq!(st(GRAND, false, false, false, false, with2, 30, 2, 18, 10).value, -192);
        assert_eq!(st(GRAND, false, false, false, false, with2, 0, 0, 18, 10).value, -240);
        // overbid: clubs with 1 (2 x 12 = 24) at bid 36 -> -2 x 36, flagged overbid
        let r = st(CLUBS, false, false, false, false, mask("CJ"), 70, 6, 36, 10);
        assert_eq!((r.value, r.overbid, r.matadors), (-72, true, 1));
        // won diamonds hand schneider announced (with 1): (2 + hand + S ann + S) x 9
        assert_eq!(st(DIAMONDS, true, true, false, false, mask("CJ"), 92, 7, 18, 10).value, 45);
        // schneider announced but only 89: lost, the announced schneider level counts
        // (ISS server), not flagged
        let r = st(DIAMONDS, true, true, false, false, mask("CJ"), 89, 7, 18, 10);
        assert_eq!((r.value, r.schneider), (-2 * 45, false));
        // ... and no extra schwarz level for a declarer without a trick
        assert_eq!(st(DIAMONDS, true, true, false, false, mask("CJ"), 20, 0, 18, 2).value, -2 * 45);
        // grand ouvert with 4 lost at once: 5 + hand + S ann + Z ann + ouvert + S + Z
        let r = st(GRAND, true, true, true, true, mask("CJ.SJ.HJ.DJ"), 0, 0, 18, 0);
        assert_eq!((r.value, r.schneider, r.schwarz), (-528, false, false));
        // grand ouvert with 4 won: 11 x 24 = 264
        assert_eq!(st(GRAND, true, true, true, true, mask("CJ.SJ.HJ.DJ"), 120, 10, 264, 10).value, 264);
        // no schneider / schwarz penalty before the declarer has played two cards
        // (resigned / timed out early: ISS records 11328, 939), then both (ISS 90094,
        // 62909: one trick complete, the declarer's second card played)
        assert_eq!(st(GRAND, false, false, false, false, with2, 0, 0, 18, 0).value, -144);
        assert_eq!(st(GRAND, false, false, false, false, with2, 0, 0, 18, 1).value, -144);
        let r = st(GRAND, false, false, false, false, with2, 0, 0, 18, 2);
        assert_eq!((r.value, r.schneider, r.schwarz), (-240, true, true));
        let r = st(GRAND, false, false, false, false, with2, 25, 1, 18, 2);
        assert_eq!((r.value, r.schneider, r.schwarz), (-192, true, false));
        // null values
        for (hand, ouvert, v) in [(false, false, 23), (true, false, 35), (false, true, 46), (true, true, 59)] {
            assert_eq!(st(NULL, hand, false, false, ouvert, 0, 0, 0, 18, 10).value, v);
            assert_eq!(st(NULL, hand, false, false, ouvert, 0, 0, 1, 18, 10).value, -2 * v);
        }
        // money: won v=72 -> declarer 2*(v+50), defenders -(v+50)
        let r = IssResult { declarer: 1, value: 72, ..Default::default() };
        assert_eq!(r.money(), [-122.0, 244.0, -122.0]);
        let r = IssResult { declarer: 0, value: -540, ..Default::default() };
        assert_eq!(r.money(), [-1260.0, 630.0, 630.0]);
    }

    #[test]
    fn null_early_end_and_follow_suit() {
        // forehand plays null hand with CA: middlehand wins trick 1 and leads C7, rearhand
        // follows CT, the declarer must follow with CA and takes the trick: game over
        let deal = "D7.D8.D9.H7.H8.H9.S7.S8.S9.CA.DA.DK.DQ.HA.HK.HQ.SA.SK.SQ.C7.DT.DJ.HT.HJ.ST.SJ.CT.CJ.CQ.CK.C8.C9";
        let mut e = deal_events(deal);
        e.extend([(1, PASS), (2, PASS), (0, bid(18)), (0, HAND0 + NULL as u32), (0, MOD0 + MOD_PLAIN)].map(|(s, a)| Event::Act(s, a)));
        let mut g = replay::<Skat>(&SkatConfig {}, &e).unwrap().state;
        assert_eq!((g.phase, g.to_move), (Phase::Play, 0));
        assert_eq!(g.contract.unwrap().iss_string(), "NH");
        let legal = |g: &Skat, s: Seat| {
            let mut l = Vec::new();
            g.legal_actions(s, &mut l);
            l.into_iter().map(card_str).collect::<Vec<_>>().join(".")
        };
        g.apply_action(0, PLAY0 + c("D7")).unwrap();
        // in null the jack is a plain diamond; middlehand holds DQ DK DA
        assert_eq!(legal(&g, 1), "DQ.DK.DA");
        g.apply_action(1, PLAY0 + c("DA")).unwrap();
        assert_eq!(legal(&g, 2), "DT.DJ");
        assert!(g.apply_action(2, PLAY0 + c("CT")).is_err());
        g.apply_action(2, PLAY0 + c("DT")).unwrap();
        assert_eq!((g.to_move, g.tricks_done, g.phase), (1, 1, Phase::Play));
        g.apply_action(1, PLAY0 + c("C7")).unwrap();
        g.apply_action(2, PLAY0 + c("CT")).unwrap();
        assert_eq!(legal(&g, 0), "CA");
        g.apply_action(0, PLAY0 + c("CA")).unwrap();
        assert_eq!(g.phase, Phase::Done);
        assert_eq!(g.final_result().iss_string(), "d:0 loss v:-70 m:0 bidok p:21 t:1 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0");
        assert_eq!(g.voids, [0, 0, 0]);
    }

    #[test]
    fn trump_following_and_resignation_scoring() {
        // hearts hand, declarer rearhand: a jack lead must be followed by a jack or a heart
        let deal = "CJ.D8.D9.H7.S8.S9.C7.C8.C9.SA.HJ.DK.DQ.HA.HK.SQ.SK.CA.CT.DA.SJ.DJ.HT.HQ.H9.H8.DT.D7.CK.CQ.S7.ST";
        let mut e = deal_events(deal);
        e.extend([(1, PASS), (2, bid(18)), (0, PASS), (2, HAND0 + HEARTS as u32), (2, MOD0 + MOD_PLAIN)].map(|(s, a)| Event::Act(s, a)));
        let mut g = replay::<Skat>(&SkatConfig {}, &e).unwrap().state;
        assert_eq!((g.declarer, g.contract.unwrap().iss_string()), (Some(2), "HH".into()));
        g.apply_action(0, PLAY0 + c("CJ")).unwrap();
        let mut l = Vec::new();
        g.legal_actions(1, &mut l);
        // middlehand: HJ (trump), HA HK (trump hearts); not CA / CT (clubs)
        assert_eq!(l.iter().map(|&x| card_str(x)).collect::<Vec<_>>(), ["HJ", "HK", "HA"]);
        g.apply_action(1, PLAY0 + c("HA")).unwrap();
        g.apply_action(2, PLAY0 + c("SJ")).unwrap();
        // CJ wins (highest trump) for forehand, a defender
        assert_eq!((g.to_move, g.points), (0, [15, 0, 0]));
        g.apply_action(0, PLAY0 + c("SA")).unwrap();
        g.apply_action(1, PLAY0 + c("SK")).unwrap();
        // rearhand has no plain spade (its SJ is a trump): any of its 9 cards
        l.clear();
        g.legal_actions(2, &mut l);
        assert_eq!(l.len(), 9);
        g.apply_action(2, PLAY0 + c("H8")).unwrap();
        assert_eq!((g.to_move, g.points[2], g.voids[2]), (2, 15, 1 << 2));
        // both defenders resign now: the declarer gets every remaining point and trick
        let end = IssEnding { resigned: [true, true, false], ..Default::default() };
        assert!(end.resignation_ends(2));
        let r = g.iss_result(&end);
        // hearts hand without 1 (no CJ; SJ in hand): 2 + hand = 3 x 10; 105 points -> schneider
        assert_eq!(r.iss_string(), "d:2 win v:40 m:-1 bidok p:105 t:9 s:1 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:1");
        // the declarer resigns instead: lost with the points so far (15 + skat ST 10)
        let r = g.iss_result(&IssEnding { resigned: [false, false, true], ..Default::default() });
        assert_eq!(r.iss_string(), "d:2 loss v:-80 m:-1 bidok p:25 t:1 s:1 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:1");
        // a defender times out: like both defenders resigning, but not a resignation
        let r = g.iss_result(&IssEnding { timeout: Some(1), ..Default::default() });
        assert_eq!(r.iss_string(), "d:2 win v:40 m:-1 bidok p:105 t:9 s:1 z:0 p0:0 p1:0 p2:0 l:-1 to:1 r:0");
        // bidding-phase timeout: a penalty for that seat
        let mut b = Skat::new_game(&SkatConfig {});
        for x in deal_events(deal) {
            if let Event::Chance(o) = x {
                b.apply_chance(o).unwrap();
            }
        }
        let r = b.iss_result(&IssEnding { timeout: Some(2), ..Default::default() });
        assert_eq!(r.iss_string(), "d:-1 penalty v:0 m:0 bidok p:0 t:0 s:0 z:0 p0:0 p1:0 p2:1 l:-1 to:2 r:0");
    }

    /// ISS record 119716: the declarer picks up, declares diamonds as a half-move and
    /// leaves before discarding — ISS scores the declared game with the dealt skat as the
    /// declarer's only known cards (without 2, 3 x 9, lost -54, the skat's 2 points).
    #[test]
    fn declared_before_discards_scoring() {
        let mut e = deal_events("DT.SQ.CT.HQ.H9.S9.DA.CJ.ST.C7.SK.D8.DJ.SA.D7.DQ.HA.D9.C8.CK.S7.DK.CQ.CA.HT.SJ.H8.HK.C9.H7.HJ.S8");
        let mut acts = vec![(1, PASS)];
        for v in [18, 20, 22, 23] {
            acts.extend([(2, bid(v)), (0, YES)]);
        }
        acts.extend([(2, bid(24)), (0, PASS), (2, PICKUP), (2, DECLARE0 + DIAMONDS as u32)]);
        e.extend(acts.iter().map(|&(s, a)| Event::Act(s, a)));
        let g = replay::<Skat>(&SkatConfig {}, &e).unwrap().state;
        assert_eq!(g.phase, Phase::Discard1);
        let r = g.iss_result(&IssEnding { left: Some(2), ..Default::default() });
        assert_eq!(r.iss_string(), "d:2 loss v:-54 m:-2 bidok p:2 t:0 s:0 z:0 p0:0 p1:0 p2:0 l:2 to:-1 r:0");
    }

    #[test]
    fn conformance_battery() {
        let rep = check_game::<Skat>(&SkatConfig {}, 300, 7).unwrap();
        assert_eq!(rep.terminal, 300);
        assert!(rep.privacy_checks > 100, "{rep:?}");
        assert!(rep.max_legal <= MAX_ACTIONS);
    }

    /// The privacy check catches an encoder that leaks the skat to the defenders.
    #[test]
    fn privacy_check_catches_a_leak() {
        #[derive(Clone, Debug)]
        struct Leaky(Skat);
        impl Game for Leaky {
            type Config = SkatConfig;
            const NAME: &'static str = "leaky_skat";
            const VERSION: u32 = 0;
            fn spec(c: &SkatConfig) -> FrameSpec { Skat::spec(c) }
            fn new_game(c: &SkatConfig) -> Self { Leaky(Skat::new_game(c)) }
            fn turn(&self) -> Turn { self.0.turn() }
            fn chance_outcomes(&self, o: &mut Vec<(u32, f64)>) { self.0.chance_outcomes(o) }
            fn apply_chance(&mut self, o: u32) -> Result<(), RulesError> { self.0.apply_chance(o) }
            fn legal_actions(&self, s: Seat, o: &mut Vec<u32>) { self.0.legal_actions(s, o) }
            fn apply_action(&mut self, s: Seat, a: u32) -> Result<(), RulesError> { self.0.apply_action(s, a) }
            fn outcome(&self) -> Vec<f32> { self.0.outcome() }
            fn returns(&self) -> Vec<f64> { self.0.returns() }
            fn encode_obs(&self, s: Seat, obs: &mut ObsView<'_>) {
                self.0.encode_obs(s, obs);
                // leak: mark the (unseen) skat cards as "from the skat" for everybody
                for c in cards_of(self.0.orig_skat) {
                    obs.entity(c as usize, 0)[C_FROM_SKAT] = 1.0;
                }
            }
            fn encode_action(&self, s: Seat, a: u32, f: &mut [f32]) { self.0.encode_action(s, a, f) }
            fn state_string(&self) -> String { self.0.state_string() }
            fn perturb_hidden(c: &SkatConfig, e: &[Event], s: Seat, r: &mut Rng) -> Option<Vec<Event>> {
                Skat::perturb_hidden(c, e, s, r)
            }
        }
        let err = check_game::<Leaky>(&SkatConfig {}, 40, 3).unwrap_err();
        assert!(err.contains("hidden from it"), "{err}");
        let _ = check_privacy::<Skat>;
    }
}
