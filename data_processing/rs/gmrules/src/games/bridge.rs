//! Contract bridge (duplicate scoring): the auction and the card play, exactly
//! OpenSpiel's `bridge(use_double_dummy_result=false, dealer=D, dealer_vul=., non_dealer_vul=.)`
//! — action / chance ids, turn order, legality, scoring (pinned by
//! tests/test_bridge_reference.py against OpenSpiel itself, both directions, and by
//! replaying the real WBridge5 records through both).
//!
//! Rules. Seats N=0, E=1, S=2, W=3; partnerships N/S (0) and E/W (1). A deal is a
//! duplicate BOARD: its number (1..=16) fixes the dealer and the vulnerability by the
//! standard rotation (board 1: dealer N, nobody vulnerable — OpenSpiel's default). The
//! config fixes the board, or (`board: 0`, the default) chance draws it uniformly
//! (outcome id = board − 1) before the cards. Then chance deals the 52 cards one by
//! one, card k of the deal to seat k mod 4 (outcome id = card id, uniform over the
//! undealt cards; a config `deal` scripts them). The auction starts with the dealer:
//! Pass always; Double of the opponents' last bid when undoubled; Redouble of the own
//! side's doubled bid; any higher bid. Four passes pass the deal out (score 0); three
//! passes after a bid end the auction: declarer = the first player of the last bidder's
//! partnership to have bid the final denomination. Declarer's left-hand opponent
//! leads; dummy (declarer's partner) is exposed after the opening lead and its cards
//! are played by declarer — the turn is declarer's on dummy's plays (as in OpenSpiel:
//! the declarer's own row / stream carries both hands). Follow suit if possible; the
//! highest trump, else the highest card of the led suit wins the trick; the winner
//! leads next. Scoring: Law 77 (contract points, over/undertricks, doubled /
//! redoubled, part-score / game / slam bonuses, the insult), by vulnerability.
//!
//! Ids (OpenSpiel's): card = rank · 4 + suit (suits ♣ ♦ ♥ ♠ = 0..3, ranks 2..A = 0..12);
//! calls 52 + (Pass 0, Dbl 1, RDbl 2, bid (level − 1) · 5 + denomination + 3) with
//! denominations ♣ ♦ ♥ ♠ NT = 0..4 — so 1♣ = 55, 1NT = 59, 7NT = 89.
//!
//! Returns: the duplicate score of each seat's side (declaring side ±). Outcome (IL
//! value target, RL reward): `reward = "imp"` (default) — the IMPs of that score
//! against 0, / 24; `"score"` — score / 7600 (OpenSpiel's max). Teams:
//! [`Game::teams`] = N/S vs E/W.
//!
//! Frames (candidate schema; hidden information → per-seat streams; the frame of a
//! dummy play is DECLARER's: own hand + dummy). Everything is relative to the viewer
//! (rel seat 0 = me, 1 = left-hand opponent, 2 = partner, 3 = right-hand opponent);
//! every tensor is 0/1 (stored as bits):
//!
//! ```text
//! entities [91, 40]  slots 0..51 the cards (slot = card id; always present):
//!                      type card; suit (4), rank (13); trump suit; location: in my
//!                      hand / in dummy's exposed hand / unseen / played; played by
//!                      rel seat (4); on the table (current trick); winning the
//!                      current trick; led its trick; in the last completed trick;
//!                      its trick won by my side; highest unplayed card of its suit;
//!                      trick number (1..13, 4 bits binary); in the hand to play
//!                    slots 52..86 the bids (slot = bid 1♣..7NT; present once made):
//!                      type bid; level (7), denomination (5); bidder, doubler,
//!                      redoubler rel seats (4 each); the current bid; bid by my side;
//!                      first bid of its denomination by the bidder's side
//!                    slots 87..90 the seats (slot 87 + rel seat; always present):
//!                      type seat; rel seat (4); dealer; vulnerable; declarer (the
//!                      would-be declarer during the auction); dummy; to act; leads
//!                      the current trick; showed out of a suit (4); passed before the
//!                      opening bid; its last call pass / dbl / rdbl / bid (4); won the
//!                      last trick
//! globals  [66]      auction; play; opening lead; playing dummy's card; dealer rel
//!                    (4); we / they vulnerable; contract level (7), denomination (5),
//!                    doubled, redoubled, declarer rel (4); consecutive passes 0..3
//!                    (4); no bid yet; my side declares; cards on the table 0..3 (4);
//!                    tricks won by my side / the other side (13 + 13, thermometer);
//!                    contract made; contract defeated
//! actions  [38, 48]  the legal calls or cards (ascending ids): call / card; pass,
//!                    double, redouble; bid level (7), denomination (5); jump; a
//!                    denomination my side / the other side has bid; game or higher;
//!                    slam — card suit (4), rank (13); trump; follows suit; ruff;
//!                    discard; lead; beats the current winner; highest unplayed card of
//!                    its suit; lowest of its suit in the hand; from dummy
//! ```
//!
//! Hidden information: the viewer's own hand; dummy's after the opening lead (to
//! everyone); played cards, the auction, the board are public. `perturb_hidden`
//! re-deals every unplayed card the seat has not seen among the hidden hands (void
//! suits revealed by the play respected), so every recorded action stays legal.

use crate::frame::ObsView;
use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rng::Rng;
use crate::rules_bail;
use crate::trajectory::Event;
use serde::{Deserialize, Serialize};

pub const N_CARDS: usize = 52;
pub const N_BIDS: usize = 35;
/// Call action ids (OpenSpiel's): Pass 52, Double 53, Redouble 54, bids 55..=89.
pub const PASS: u32 = 52;
pub const DOUBLE: u32 = 53;
pub const REDOUBLE: u32 = 54;
pub const FIRST_BID: u32 = 55;
pub const N_ACTIONS: u32 = 90;
/// Longest auction: 4 opening passes... every bid followed by the longest
/// pass/double/redouble sequence (OpenSpiel's kMaxAuctionLength = 35 · 9 + 4).
pub const MAX_AUCTION: usize = N_BIDS * 9 + 4;

pub const MAX_ENTITIES: usize = 91;
pub const ENTITY_FEATURES: usize = 40;
pub const GLOBAL_FEATURES: usize = 66;
pub const MAX_ACTIONS: usize = 38;
pub const ACTION_FEATURES: usize = 48;

const NONE: u8 = u8::MAX;
const SLOT_BID: usize = 52;
const SLOT_SEAT: usize = 87;

// entity features
const E_CARD: usize = 0;
const E_BID: usize = 1;
const E_SEAT: usize = 2;
const C_SUIT: usize = 3;
const C_RANK: usize = 7;
const C_TRUMP: usize = 20;
const C_MINE: usize = 21;
const C_DUMMY: usize = 22;
const C_UNSEEN: usize = 23;
const C_PLAYED: usize = 24;
const C_BY: usize = 25;
const C_TABLE: usize = 29;
const C_WINNING: usize = 30;
const C_LED: usize = 31;
const C_LAST: usize = 32;
const C_OURS: usize = 33;
const C_TOP: usize = 34;
const C_TRICK: usize = 35;
const C_ACTING: usize = 39;
const B_LEVEL: usize = 3;
const B_DENOM: usize = 10;
const B_BY: usize = 15;
const B_DBL: usize = 19;
const B_RDBL: usize = 23;
const B_CURRENT: usize = 27;
const B_OURS: usize = 28;
const B_NEW: usize = 29;
const S_REL: usize = 3;
const S_DEALER: usize = 7;
const S_VUL: usize = 8;
const S_DECLARER: usize = 9;
const S_DUMMY: usize = 10;
const S_TO_ACT: usize = 11;
const S_LEADER: usize = 12;
const S_VOID: usize = 13;
const S_INIT_PASS: usize = 17;
const S_LAST_CALL: usize = 18;
const S_WON_LAST: usize = 22;
// globals
const G_AUCTION: usize = 0;
const G_PLAY: usize = 1;
const G_OPENING_LEAD: usize = 2;
const G_FOR_DUMMY: usize = 3;
const G_DEALER: usize = 4;
const G_WE_VUL: usize = 8;
const G_THEY_VUL: usize = 9;
const G_LEVEL: usize = 10;
const G_DENOM: usize = 17;
const G_DBL: usize = 22;
const G_RDBL: usize = 23;
const G_DECLARER: usize = 24;
const G_PASSES: usize = 28;
const G_NO_BID: usize = 32;
const G_WE_DECLARE: usize = 33;
const G_TRICK_POS: usize = 34;
const G_OUR_TRICKS: usize = 38;
const G_THEIR_TRICKS: usize = 51;
const G_MADE: usize = 64;
const G_SET: usize = 65;
// action features
const A_CALL: usize = 0;
const A_CARD: usize = 1;
const A_PASS: usize = 2;
const A_DBL: usize = 3;
const A_RDBL: usize = 4;
const A_LEVEL: usize = 5;
const A_DENOM: usize = 12;
const A_JUMP: usize = 17;
const A_OUR_STRAIN: usize = 18;
const A_THEIR_STRAIN: usize = 19;
const A_GAME: usize = 20;
const A_SLAM: usize = 21;
const A_SUIT: usize = 22;
const A_RANK: usize = 26;
const A_TRUMP: usize = 39;
const A_FOLLOW: usize = 40;
const A_RUFF: usize = 41;
const A_DISCARD: usize = 42;
const A_LEAD: usize = 43;
const A_WINS: usize = 44;
const A_TOP: usize = 45;
const A_LOWEST: usize = 46;
const A_DUMMY: usize = 47;

pub const SEAT_CHARS: [char; 4] = ['N', 'E', 'S', 'W'];
pub const SUIT_CHARS: [char; 4] = ['C', 'D', 'H', 'S'];
pub const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
/// Denomination names (4 = no trump).
pub const DENOM_NAMES: [&str; 5] = ["C", "D", "H", "S", "NT"];

#[inline]
pub fn suit(card: u8) -> u8 {
    card % 4
}

#[inline]
pub fn rank(card: u8) -> u8 {
    card / 4
}

#[inline]
pub fn card_id(suit: u8, rank: u8) -> u8 {
    rank * 4 + suit
}

#[inline]
pub fn partnership(seat: u8) -> u8 {
    seat & 1
}

#[inline]
pub fn partner(seat: u8) -> u8 {
    seat ^ 2
}

/// The bid action id of `level` (1..=7) in `denom` (0..=4, 4 = NT).
pub fn bid_action(level: u8, denom: u8) -> u32 {
    FIRST_BID + (level as u32 - 1) * 5 + denom as u32
}

/// `(level, denom)` of a bid action id.
pub fn bid_of(action: u32) -> Option<(u8, u8)> {
    (FIRST_BID..N_ACTIONS).contains(&action).then(|| {
        let b = (action - FIRST_BID) as u8;
        (b / 5 + 1, b % 5)
    })
}

/// "SA", "HT", "C2": suit letter + rank character.
pub fn card_string(card: u8) -> String {
    format!("{}{}", SUIT_CHARS[suit(card) as usize], RANK_CHARS[rank(card) as usize])
}

/// A rank character (2..9, T or 10, J, Q, K, A; case-insensitive).
pub fn parse_rank(s: &str) -> Option<u8> {
    let s = s.trim().to_ascii_uppercase();
    if s == "10" {
        return Some(8);
    }
    let mut it = s.chars();
    let c = it.next()?;
    if it.next().is_some() {
        return None;
    }
    RANK_CHARS.iter().position(|&r| r == c).map(|r| r as u8)
}

/// A suit letter (C D H S, case-insensitive; also ♣ ♦ ♥ ♠).
pub fn parse_suit(c: char) -> Option<u8> {
    match c.to_ascii_uppercase() {
        'C' | '♣' => Some(0),
        'D' | '♦' => Some(1),
        'H' | '♥' => Some(2),
        'S' | '♠' => Some(3),
        _ => None,
    }
}

/// A card as suit + rank ("SA", "h10", "D2"; case-insensitive).
pub fn parse_card(s: &str) -> Option<u8> {
    let s = s.trim();
    let mut it = s.chars();
    let su = parse_suit(it.next()?)?;
    Some(card_id(su, parse_rank(it.as_str())?))
}

/// A seat letter (N E S W, case-insensitive).
pub fn parse_seat(s: &str) -> Option<u8> {
    let s = s.trim().to_ascii_uppercase();
    match s.as_str() {
        "N" | "NORTH" => Some(0),
        "E" | "EAST" => Some(1),
        "S" | "SOUTH" => Some(2),
        "W" | "WEST" => Some(3),
        _ => None,
    }
}

/// A call: "Pass" / "P" / "PP", "X" / "Dbl" / "D" / "DD", "XX" / "RDbl" / "R" / "RR",
/// or a bid "1C" .. "7N" / "7NT" (case-insensitive; trailing alert marks `!` ignored).
pub fn parse_call(s: &str) -> Option<u32> {
    let t = s.trim().trim_end_matches(['!', '*']).to_ascii_uppercase();
    match t.as_str() {
        "P" | "PASS" | "PP" | "AP" => return Some(PASS),
        "X" | "D" | "DBL" | "DOUBLE" | "DD" => return Some(DOUBLE),
        "XX" | "R" | "RDBL" | "REDOUBLE" | "RR" => return Some(REDOUBLE),
        _ => {}
    }
    let mut it = t.chars();
    let level = it.next()?.to_digit(10)? as u8;
    if !(1..=7).contains(&level) {
        return None;
    }
    let denom = match it.as_str() {
        "C" | "♣" => 0,
        "D" | "♦" => 1,
        "H" | "♥" => 2,
        "S" | "♠" => 3,
        "N" | "NT" | "SA" => 4,
        _ => return None,
    };
    Some(bid_action(level, denom))
}

/// "Pass", "X", "XX", "1C" .. "7NT".
pub fn call_string(action: u32) -> String {
    match action {
        PASS => "Pass".into(),
        DOUBLE => "X".into(),
        REDOUBLE => "XX".into(),
        a => match bid_of(a) {
            Some((l, d)) => format!("{l}{}", DENOM_NAMES[d as usize]),
            None => format!("?{a}"),
        },
    }
}

/// The dealer and the vulnerability (`[N/S, E/W]`) of duplicate board `board`
/// (1..=16, the standard rotation; boards repeat every 16).
pub fn board_dealer_vul(board: u8) -> (u8, [bool; 2]) {
    // (N/S, E/W) vulnerable for boards 1..16
    const VUL: [(bool, bool); 16] = [
        (false, false), (true, false), (false, true), (true, true),
        (true, false), (false, true), (true, true), (false, false),
        (false, true), (true, true), (false, false), (true, false),
        (true, true), (false, false), (true, false), (false, true),
    ];
    let b = (board.max(1) - 1) % 16;
    let (ns, ew) = VUL[b as usize];
    (b % 4, [ns, ew])
}

/// The board (1..=16) with this dealer and vulnerability `[N/S, E/W]` (every pair
/// occurs exactly once in the rotation).
pub fn board_of(dealer: u8, vul: [bool; 2]) -> u8 {
    (1..=16u8).find(|&b| board_dealer_vul(b) == (dealer, vul)).expect("every (dealer, vulnerability) pair has a board")
}

/// The duplicate score (Law 77) of the declaring side: contract `level` (0 = passed
/// out) in `denom`, `dbl` 0 / 1 / 2 = undoubled / doubled / redoubled, `tricks` taken
/// by declarer, declaring side vulnerable or not (OpenSpiel's bridge_scoring.cc).
pub fn score(level: u8, denom: u8, dbl: u8, tricks: u8, vul: bool) -> i32 {
    if level == 0 {
        return 0;
    }
    let ds = [1, 2, 4][dbl as usize];
    let result = tricks as i32 - (6 + level as i32);
    if result < 0 {
        let under = -result;
        if dbl == 0 {
            return (if vul { -100 } else { -50 }) * under;
        }
        let s = if vul {
            -200 - 300 * (under - 1)
        } else {
            match under {
                1 => -100,
                2 => -300,
                _ => -500 - 300 * (under - 3),
            }
        };
        s * (ds / 2)
    } else {
        let per_trick = [20, 20, 30, 30, 30][denom as usize];
        let contract = (level as i32 * per_trick + if denom == 4 { 10 } else { 0 }) * ds;
        let bonus = match level {
            7 => if vul { 2000 } else { 1300 },
            6 => if vul { 1250 } else { 800 },
            _ if contract >= 100 => if vul { 500 } else { 300 },
            _ => 50,
        };
        let insult = 50 * (ds / 2);
        let over = if dbl == 0 { result * per_trick } else { (if vul { 100 } else { 50 }) * result * ds };
        contract + bonus + insult + over
    }
}

/// The WBF IMP scale: IMPs of a score difference (signed, 0..=24).
pub fn imps(diff: i32) -> i32 {
    const UP: [i32; 24] = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750,
                           2000, 2250, 2500, 3000, 3500, 4000];
    let n = UP.iter().take_while(|&&t| diff.abs() >= t).count() as i32;
    n * diff.signum()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BridgeConfig {
    /// Duplicate board 1..=16 (fixes dealer and vulnerability; board 1 = dealer N,
    /// nobody vulnerable), or 0 (default): drawn uniformly per deal, a chance event
    /// before the cards.
    pub board: u8,
    /// A scripted deal: the 52 card ids in deal order (card k to seat k mod 4) — a
    /// given deal played again (duplicate boards, evaluation, golden tests); empty
    /// (default) = shuffled by chance.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deal: Vec<u8>,
    /// Outcome scale: "imp" (default: IMPs of the side's score vs 0, / 24) or
    /// "score" (score / 7600).
    pub reward: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig { board: 0, deal: Vec::new(), reward: "imp".into() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Deal,
    Auction,
    Play,
    Over,
}

/// A final (or, during the auction, current) contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contract {
    /// 1..=7
    pub level: u8,
    /// 0..=4 (4 = NT)
    pub denom: u8,
    /// 0 undoubled, 1 doubled, 2 redoubled
    pub dbl: u8,
    pub declarer: u8,
}

impl Contract {
    /// "4HX by S"
    pub fn to_string_short(&self) -> String {
        format!("{}{}{} by {}", self.level, DENOM_NAMES[self.denom as usize], ["", "X", "XX"][self.dbl as usize],
                SEAT_CHARS[self.declarer as usize])
    }
}

#[derive(Clone, Debug)]
pub struct Bridge {
    imp: bool,
    script: Option<[u8; 52]>,
    /// 1..=16, 0 = not drawn yet
    board: u8,
    dealer: u8,
    /// by partnership
    vul: [bool; 2],
    dealt: u8,
    /// current holder of each card (NONE: undealt or played)
    holder: [u8; 52],
    /// holder after the deal (NONE while undealt)
    orig: [u8; 52],
    phase: Phase,
    // --- auction
    /// call index of every call (0 pass, 1 dbl, 2 rdbl, 3 + bid), in order
    calls: Vec<u8>,
    passes: u8,
    level: u8,
    denom: u8,
    dbl: u8,
    declarer: u8,
    first_bidder: [[u8; 5]; 2],
    bidder: [u8; 35],
    doubler: [u8; 35],
    redoubler: [u8; 35],
    new_strain: [bool; 35],
    initial_pass: [bool; 4],
    /// last call index of each seat (NONE: none yet)
    last_call: [u8; 4],
    /// the seat to call / the hand to play
    current: u8,
    // --- play
    n_played: u8,
    /// the cards in play order
    plays: [u8; 52],
    played_by: [u8; 52],
    /// index of the card in the play order
    play_idx: [u8; 52],
    trick_leader: u8,
    win_card: u8,
    trick_winner: [u8; 13],
    /// tricks won by partnership
    tricks: [u8; 2],
    /// void[seat][suit]: showed out of the suit
    void: [[bool; 4]; 4],
}

impl Bridge {
    pub fn board(&self) -> u8 {
        self.board
    }

    pub fn dealer(&self) -> u8 {
        self.dealer
    }

    /// Vulnerability by partnership `[N/S, E/W]`.
    pub fn vulnerability(&self) -> [bool; 2] {
        self.vul
    }

    /// The contract once the auction produced one (during the auction: the current
    /// highest bid with its would-be declarer); None while no bid has been made.
    pub fn contract(&self) -> Option<Contract> {
        (self.level > 0).then_some(Contract { level: self.level, denom: self.denom, dbl: self.dbl, declarer: self.declarer })
    }

    pub fn in_auction(&self) -> bool {
        self.phase == Phase::Auction
    }

    pub fn in_play(&self) -> bool {
        self.phase == Phase::Play
    }

    /// Tricks won by declarer's side so far.
    pub fn declarer_tricks(&self) -> u8 {
        if self.level == 0 {
            0
        } else {
            self.tricks[partnership(self.declarer) as usize]
        }
    }

    pub fn cards_played(&self) -> u8 {
        self.n_played
    }

    /// The seat holding `card` at the end of the deal (None before it is dealt).
    pub fn dealt_to(&self, card: u8) -> Option<u8> {
        (self.orig[card as usize] != NONE).then_some(self.orig[card as usize])
    }

    /// The calls so far (action ids).
    pub fn auction(&self) -> Vec<u32> {
        self.calls.iter().map(|&c| PASS + c as u32).collect()
    }

    /// The hand to play now (the seat itself, or dummy when declarer plays for it).
    pub fn hand_to_play(&self) -> Option<u8> {
        (self.phase == Phase::Play).then_some(self.current)
    }

    fn cur_bid(&self) -> Option<usize> {
        (self.level > 0).then(|| (self.level as usize - 1) * 5 + self.denom as usize)
    }

    /// The seat choosing now (declarer for dummy's plays).
    fn actor(&self) -> u8 {
        if self.phase == Phase::Play && partnership(self.current) == partnership(self.declarer) {
            self.declarer
        } else {
            self.current
        }
    }

    fn dummy_exposed(&self) -> bool {
        matches!(self.phase, Phase::Play | Phase::Over) && self.n_played > 0
    }

    /// The suit led to the trick in progress (None: no trick in progress).
    fn led_suit(&self) -> Option<u8> {
        (self.phase == Phase::Play && !self.n_played.is_multiple_of(4))
            .then(|| suit(self.plays[(self.n_played - self.n_played % 4) as usize]))
    }

    fn has_suit(&self, seat: u8, s: u8) -> bool {
        (0..13).any(|r| self.holder[card_id(s, r) as usize] == seat)
    }

    fn call_legal(&self, call: u32) -> bool {
        match call {
            PASS => true,
            DOUBLE => self.level > 0 && partnership(self.declarer) != partnership(self.current) && self.dbl == 0,
            REDOUBLE => self.level > 0 && partnership(self.declarer) == partnership(self.current) && self.dbl == 1,
            a => match bid_of(a) {
                Some(_) => self.cur_bid().is_none_or(|b| (a - FIRST_BID) as usize > b),
                None => false,
            },
        }
    }

    fn card_legal(&self, card: u32) -> bool {
        if card >= 52 || self.holder[card as usize] != self.current {
            return false;
        }
        match self.led_suit() {
            Some(led) => suit(card as u8) == led || !self.has_suit(self.current, led),
            None => true,
        }
    }

    /// Would `card` beat the current winner of the trick in progress?
    fn beats(&self, card: u8) -> bool {
        if self.n_played.is_multiple_of(4) {
            return false;
        }
        let w = self.win_card;
        if suit(card) == suit(w) {
            rank(card) > rank(w)
        } else {
            self.denom < 4 && suit(card) == self.denom
        }
    }

    /// The highest unplayed rank of each suit (NONE when all are played).
    fn top_ranks(&self) -> [u8; 4] {
        let mut top = [NONE; 4];
        for (s, t) in top.iter_mut().enumerate() {
            for r in (0..13u8).rev() {
                if self.played_by[card_id(s as u8, r) as usize] == NONE {
                    *t = r;
                    break;
                }
            }
        }
        top
    }

    fn apply_call(&mut self, call: u8) {
        let seat = self.current;
        if call == 0 {
            self.passes += 1;
        } else {
            self.passes = 0;
        }
        self.calls.push(call);
        self.last_call[seat as usize] = call;
        match call {
            0 => {
                if self.level == 0 {
                    self.initial_pass[seat as usize] = true;
                }
                if self.passes == 4 {
                    self.phase = Phase::Over; // passed out
                    return;
                }
                if self.passes == 3 && self.level > 0 {
                    self.phase = Phase::Play;
                    self.current = (self.declarer + 1) % 4;
                    return;
                }
            }
            1 => {
                self.dbl = 1;
                self.doubler[self.cur_bid().unwrap()] = seat;
            }
            2 => {
                self.dbl = 2;
                self.redoubler[self.cur_bid().unwrap()] = seat;
            }
            c => {
                let b = (c - 3) as usize;
                self.level = (b / 5 + 1) as u8;
                self.denom = (b % 5) as u8;
                self.dbl = 0;
                let side = partnership(seat) as usize;
                if self.first_bidder[side][self.denom as usize] == NONE {
                    self.first_bidder[side][self.denom as usize] = seat;
                    self.new_strain[b] = true;
                }
                self.declarer = self.first_bidder[side][self.denom as usize];
                self.bidder[b] = seat;
            }
        }
        self.current = (seat + 1) % 4;
    }

    fn apply_card(&mut self, card: u8) {
        let seat = self.current;
        let pos = self.n_played % 4;
        self.holder[card as usize] = NONE;
        self.plays[self.n_played as usize] = card;
        self.played_by[card as usize] = seat;
        self.play_idx[card as usize] = self.n_played;
        if pos == 0 {
            self.trick_leader = seat;
            self.win_card = card;
        } else {
            let led = suit(self.plays[(self.n_played - pos) as usize]);
            if suit(card) != led {
                self.void[seat as usize][led as usize] = true;
            }
            if self.beats(card) {
                self.win_card = card;
            }
        }
        self.n_played += 1;
        if self.n_played.is_multiple_of(4) {
            let winner = self.played_by[self.win_card as usize];
            self.trick_winner[(self.n_played / 4 - 1) as usize] = winner;
            self.tricks[partnership(winner) as usize] += 1;
            self.current = winner;
        } else {
            self.current = (seat + 1) % 4;
        }
        if self.n_played == 52 {
            self.phase = Phase::Over;
        }
    }

    fn score_declarer(&self) -> i32 {
        if self.level == 0 {
            return 0;
        }
        score(self.level, self.denom, self.dbl, self.declarer_tricks(), self.vul[partnership(self.declarer) as usize])
    }
}

impl Game for Bridge {
    type Config = BridgeConfig;
    const NAME: &'static str = "bridge";
    const VERSION: u32 = 1;

    fn check_config(cfg: &BridgeConfig) -> Result<(), RulesError> {
        if cfg.board > 16 {
            rules_bail!("board must be 0 (drawn) or 1..=16, got {}", cfg.board);
        }
        if !cfg.deal.is_empty() {
            let mut seen = [false; 52];
            if cfg.deal.len() != 52 || cfg.deal.iter().any(|&c| c >= 52 || std::mem::replace(&mut seen[c as usize], true)) {
                rules_bail!("deal must list each card id 0..52 once (52 entries), got {:?}", cfg.deal);
            }
        }
        if cfg.reward != "imp" && cfg.reward != "score" {
            rules_bail!("reward must be \"imp\" or \"score\", got {:?}", cfg.reward);
        }
        Ok(())
    }

    fn spec(_cfg: &BridgeConfig) -> FrameSpec {
        FrameSpec::candidate(CandidateSpec {
            n_seats: 4,
            max_entities: MAX_ENTITIES,
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
            max_decisions: MAX_AUCTION + N_CARDS,
        })
    }

    fn teams(_cfg: &BridgeConfig) -> Option<Vec<usize>> {
        Some(vec![0, 1, 0, 1])
    }

    fn new_game(cfg: &BridgeConfig) -> Self {
        let script = (cfg.deal.len() == 52).then(|| {
            let mut d = [0u8; 52];
            d.copy_from_slice(&cfg.deal);
            d
        });
        let (dealer, vul) = if cfg.board > 0 { board_dealer_vul(cfg.board) } else { (0, [false; 2]) };
        Bridge {
            imp: cfg.reward != "score",
            script,
            board: cfg.board,
            dealer,
            vul,
            dealt: 0,
            holder: [NONE; 52],
            orig: [NONE; 52],
            phase: Phase::Deal,
            calls: Vec::with_capacity(24),
            passes: 0,
            level: 0,
            denom: 0,
            dbl: 0,
            declarer: NONE,
            first_bidder: [[NONE; 5]; 2],
            bidder: [NONE; 35],
            doubler: [NONE; 35],
            redoubler: [NONE; 35],
            new_strain: [false; 35],
            initial_pass: [false; 4],
            last_call: [NONE; 4],
            current: dealer,
            n_played: 0,
            plays: [NONE; 52],
            played_by: [NONE; 52],
            play_idx: [NONE; 52],
            trick_leader: NONE,
            win_card: NONE,
            trick_winner: [NONE; 13],
            tricks: [0; 2],
            void: [[false; 4]; 4],
        }
    }

    fn turn(&self) -> Turn {
        match self.phase {
            Phase::Deal => Turn::Chance,
            Phase::Auction | Phase::Play => Turn::Act(SeatSet::single(self.actor() as usize)),
            Phase::Over => Turn::Terminal,
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        if self.phase != Phase::Deal {
            return;
        }
        if self.board == 0 {
            out.extend((0..16u32).map(|b| (b, 1.0 / 16.0)));
        } else if let Some(d) = &self.script {
            out.push((d[self.dealt as usize] as u32, 1.0));
        } else {
            let p = 1.0 / (52 - self.dealt as usize) as f64;
            out.extend((0..52u32).filter(|&c| self.orig[c as usize] == NONE).map(|c| (c, p)));
        }
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.phase != Phase::Deal {
            rules_bail!("bridge: no chance node (the deal is complete)");
        }
        if self.board == 0 {
            if outcome >= 16 {
                rules_bail!("bridge: board outcome {outcome} (board number - 1, 0..16)");
            }
            self.board = outcome as u8 + 1;
            (self.dealer, self.vul) = board_dealer_vul(self.board);
            self.current = self.dealer;
            return Ok(());
        }
        if outcome >= 52 || self.orig[outcome as usize] != NONE {
            rules_bail!("bridge: card {outcome} cannot be dealt (dealt: {})", self.dealt);
        }
        if let Some(d) = &self.script {
            if d[self.dealt as usize] as u32 != outcome {
                rules_bail!("bridge: the scripted deal has card {} at position {}, got {outcome}", d[self.dealt as usize], self.dealt);
            }
        }
        let seat = self.dealt % 4;
        self.holder[outcome as usize] = seat;
        self.orig[outcome as usize] = seat;
        self.dealt += 1;
        if self.dealt == 52 {
            self.phase = Phase::Auction;
            self.current = self.dealer;
        }
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            return;
        }
        match self.phase {
            Phase::Auction => {
                out.push(PASS);
                if self.call_legal(DOUBLE) {
                    out.push(DOUBLE);
                }
                if self.call_legal(REDOUBLE) {
                    out.push(REDOUBLE);
                }
                let from = self.cur_bid().map_or(0, |b| b + 1) as u32;
                out.extend((FIRST_BID + from)..N_ACTIONS);
            }
            Phase::Play => {
                let hand = self.current;
                if let Some(led) = self.led_suit() {
                    out.extend((0..13).map(|r| card_id(led, r)).filter(|&c| self.holder[c as usize] == hand).map(u32::from));
                }
                if out.is_empty() {
                    out.extend((0..52u32).filter(|&c| self.holder[c as usize] == hand));
                }
            }
            _ => {}
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            rules_bail!("bridge: seat {seat} is not to act ({:?})", self.turn());
        }
        match self.phase {
            Phase::Auction => {
                if !self.call_legal(action) {
                    rules_bail!("bridge: call {} ({action}) is illegal for {} now", call_string(action), SEAT_CHARS[seat]);
                }
                self.apply_call((action - PASS) as u8);
            }
            Phase::Play => {
                if !self.card_legal(action) {
                    rules_bail!("bridge: card {action} is illegal for hand {} now", SEAT_CHARS[self.current as usize]);
                }
                self.apply_card(action as u8);
            }
            _ => rules_bail!("bridge: no decision now"),
        }
        Ok(())
    }

    fn returns(&self) -> Vec<f64> {
        if self.phase != Phase::Over || self.level == 0 {
            return vec![0.0; 4];
        }
        let s = self.score_declarer() as f64;
        let side = partnership(self.declarer);
        (0..4u8).map(|p| if partnership(p) == side { s } else { -s }).collect()
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns()
            .iter()
            .map(|&r| if self.imp { imps(r as i32) as f32 / 24.0 } else { (r / 7600.0) as f32 })
            .collect()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        let me = seat as u8;
        let rel = |p: u8| ((p + 4 - me) % 4) as usize;
        let side = partnership(me);
        let play = self.phase == Phase::Play;
        let exposed = self.dummy_exposed();
        let dummy = if self.level > 0 && play { partner(self.declarer) } else { NONE };
        let hand = if play { self.current } else { me };
        let completed = self.n_played / 4;
        let in_trick = self.n_played % 4;
        let top = self.top_ranks();
        // cards
        for c in 0..52u8 {
            let f = obs.entity(c as usize, 0);
            f[E_CARD] = 1.0;
            f[C_SUIT + suit(c) as usize] = 1.0;
            f[C_RANK + rank(c) as usize] = 1.0;
            if play && self.denom < 4 && suit(c) == self.denom {
                f[C_TRUMP] = 1.0;
            }
            let by = self.played_by[c as usize];
            if by != NONE {
                f[C_PLAYED] = 1.0;
                f[C_BY + rel(by)] = 1.0;
                let idx = self.play_idx[c as usize];
                let t = idx / 4;
                if t == completed && in_trick > 0 {
                    f[C_TABLE] = 1.0;
                    if c == self.win_card {
                        f[C_WINNING] = 1.0;
                    }
                }
                if idx % 4 == 0 {
                    f[C_LED] = 1.0;
                }
                if t < completed {
                    if t + 1 == completed {
                        f[C_LAST] = 1.0;
                    }
                    if partnership(self.trick_winner[t as usize]) == side {
                        f[C_OURS] = 1.0;
                    }
                }
                let n = t + 1;
                for k in 0..4 {
                    f[C_TRICK + k] = ((n >> k) & 1) as f32;
                }
            } else {
                let h = self.holder[c as usize];
                if h == me {
                    f[C_MINE] = 1.0;
                } else if exposed && h == dummy {
                    f[C_DUMMY] = 1.0;
                } else {
                    f[C_UNSEEN] = 1.0;
                }
                if top[suit(c) as usize] == rank(c) {
                    f[C_TOP] = 1.0;
                }
                // the hand to play is the viewer's own or (declarer) dummy's: visible
                if play && h == hand && (h == me || (exposed && h == dummy)) {
                    f[C_ACTING] = 1.0;
                }
            }
        }
        // bids
        let cur = self.cur_bid();
        for b in 0..N_BIDS {
            let by = self.bidder[b];
            if by == NONE {
                continue;
            }
            let f = obs.entity(SLOT_BID + b, 0);
            f[E_BID] = 1.0;
            f[B_LEVEL + b / 5] = 1.0;
            f[B_DENOM + b % 5] = 1.0;
            f[B_BY + rel(by)] = 1.0;
            if self.doubler[b] != NONE {
                f[B_DBL + rel(self.doubler[b])] = 1.0;
            }
            if self.redoubler[b] != NONE {
                f[B_RDBL + rel(self.redoubler[b])] = 1.0;
            }
            if cur == Some(b) {
                f[B_CURRENT] = 1.0;
            }
            if partnership(by) == side {
                f[B_OURS] = 1.0;
            }
            if self.new_strain[b] {
                f[B_NEW] = 1.0;
            }
        }
        // seats
        let leader = if play { if in_trick == 0 { self.current } else { self.trick_leader } } else { NONE };
        for r in 0..4u8 {
            let p = (me + r) % 4;
            let f = obs.entity(SLOT_SEAT + r as usize, 0);
            f[E_SEAT] = 1.0;
            f[S_REL + r as usize] = 1.0;
            if p == self.dealer {
                f[S_DEALER] = 1.0;
            }
            if self.vul[partnership(p) as usize] {
                f[S_VUL] = 1.0;
            }
            if self.level > 0 && p == self.declarer {
                f[S_DECLARER] = 1.0;
            }
            if p == dummy {
                f[S_DUMMY] = 1.0;
            }
            if p == self.current {
                f[S_TO_ACT] = 1.0;
            }
            if p == leader {
                f[S_LEADER] = 1.0;
            }
            for s in 0..4 {
                if self.void[p as usize][s] {
                    f[S_VOID + s] = 1.0;
                }
            }
            if self.initial_pass[p as usize] {
                f[S_INIT_PASS] = 1.0;
            }
            let lc = self.last_call[p as usize];
            if lc != NONE {
                f[S_LAST_CALL + (lc as usize).min(3)] = 1.0;
            }
            if completed > 0 && self.trick_winner[completed as usize - 1] == p {
                f[S_WON_LAST] = 1.0;
            }
        }
        // globals
        let g = obs.globals();
        g[if play { G_PLAY } else { G_AUCTION }] = 1.0;
        if play && self.n_played == 0 {
            g[G_OPENING_LEAD] = 1.0;
        }
        if play && hand != me {
            g[G_FOR_DUMMY] = 1.0;
        }
        g[G_DEALER + rel(self.dealer)] = 1.0;
        if self.vul[side as usize] {
            g[G_WE_VUL] = 1.0;
        }
        if self.vul[1 - side as usize] {
            g[G_THEY_VUL] = 1.0;
        }
        if self.level > 0 {
            g[G_LEVEL + self.level as usize - 1] = 1.0;
            g[G_DENOM + self.denom as usize] = 1.0;
            if self.dbl == 1 {
                g[G_DBL] = 1.0;
            }
            if self.dbl == 2 {
                g[G_RDBL] = 1.0;
            }
            g[G_DECLARER + rel(self.declarer)] = 1.0;
            if partnership(self.declarer) == side {
                g[G_WE_DECLARE] = 1.0;
            }
        } else {
            g[G_NO_BID] = 1.0;
        }
        if !play {
            g[G_PASSES + (self.passes as usize).min(3)] = 1.0;
        } else {
            g[G_TRICK_POS + in_trick as usize] = 1.0;
            let ours = self.tricks[side as usize];
            let theirs = self.tricks[1 - side as usize];
            for k in 0..13u8 {
                if ours > k {
                    g[G_OUR_TRICKS + k as usize] = 1.0;
                }
                if theirs > k {
                    g[G_THEIR_TRICKS + k as usize] = 1.0;
                }
            }
            let decl = self.declarer_tricks();
            let def = completed - decl;
            if decl >= 6 + self.level {
                g[G_MADE] = 1.0;
            }
            if def >= 8 - self.level {
                g[G_SET] = 1.0;
            }
        }
    }

    fn encode_action(&self, seat: Seat, action: u32, feat: &mut [f32]) {
        let me = seat as u8;
        let side = partnership(me);
        if action >= PASS {
            feat[A_CALL] = 1.0;
            match action {
                PASS => feat[A_PASS] = 1.0,
                DOUBLE => feat[A_DBL] = 1.0,
                REDOUBLE => feat[A_RDBL] = 1.0,
                a => {
                    let (level, denom) = bid_of(a).expect("bid");
                    feat[A_LEVEL + level as usize - 1] = 1.0;
                    feat[A_DENOM + denom as usize] = 1.0;
                    let min_level = if self.level == 0 {
                        1
                    } else if denom > self.denom {
                        self.level
                    } else {
                        self.level + 1
                    };
                    if level > min_level {
                        feat[A_JUMP] = 1.0;
                    }
                    if self.first_bidder[side as usize][denom as usize] != NONE {
                        feat[A_OUR_STRAIN] = 1.0;
                    }
                    if self.first_bidder[1 - side as usize][denom as usize] != NONE {
                        feat[A_THEIR_STRAIN] = 1.0;
                    }
                    let points = level as i32 * [20, 20, 30, 30, 30][denom as usize] + if denom == 4 { 10 } else { 0 };
                    if points >= 100 {
                        feat[A_GAME] = 1.0;
                    }
                    if level >= 6 {
                        feat[A_SLAM] = 1.0;
                    }
                }
            }
            return;
        }
        let c = action as u8;
        let (s, r) = (suit(c), rank(c));
        feat[A_CARD] = 1.0;
        feat[A_SUIT + s as usize] = 1.0;
        feat[A_RANK + r as usize] = 1.0;
        let trump = self.denom < 4 && s == self.denom;
        if trump {
            feat[A_TRUMP] = 1.0;
        }
        match self.led_suit() {
            Some(led) => {
                if s == led {
                    feat[A_FOLLOW] = 1.0;
                } else if trump {
                    feat[A_RUFF] = 1.0;
                } else {
                    feat[A_DISCARD] = 1.0;
                }
                if self.beats(c) {
                    feat[A_WINS] = 1.0;
                }
            }
            None => feat[A_LEAD] = 1.0,
        }
        if (r + 1..13).all(|h| self.played_by[card_id(s, h) as usize] != NONE) {
            feat[A_TOP] = 1.0;
        }
        let hand = self.current;
        if (0..r).all(|l| self.holder[card_id(s, l) as usize] != hand) {
            feat[A_LOWEST] = 1.0;
        }
        if hand != me {
            feat[A_DUMMY] = 1.0;
        }
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        if action < 52 {
            card_string(action as u8)
        } else {
            call_string(action)
        }
    }

    fn chance_string(&self, outcome: u32) -> String {
        if self.board == 0 {
            format!("Board {}", outcome + 1)
        } else if outcome < 52 {
            format!("Deal {}->{}", card_string(outcome as u8), SEAT_CHARS[(self.dealt % 4) as usize])
        } else {
            format!("c{outcome}")
        }
    }

    fn state_string(&self) -> String {
        let hand = |p: u8| -> String {
            (0..4u8)
                .rev()
                .map(|s| (0..13u8).rev().filter(|&r| self.orig[card_id(s, r) as usize] == p).map(|r| RANK_CHARS[r as usize]).collect::<String>())
                .collect::<Vec<_>>()
                .join(".")
        };
        let deal = (0..4u8).map(hand).collect::<Vec<_>>().join(" ");
        let auction = self.calls.iter().map(|&c| call_string(PASS + c as u32)).collect::<Vec<_>>().join(" ");
        let contract = self.contract().map_or("-".into(), |c| c.to_string_short());
        format!("board {} dealer {} vul {:?} deal N:{deal} | auction: {auction} | contract {contract} | played {} tricks {:?} | {:?}",
                self.board, SEAT_CHARS[self.dealer as usize], self.vul, self.n_played, self.tricks, self.phase)
    }

    /// Re-deal the unplayed cards `seat` has not seen (the other hands but dummy's
    /// once exposed; cards already played stay with their players) among their
    /// holders, respecting the void suits the play revealed — every recorded call and
    /// card stays legal. None: a scripted deal, an incomplete deal or nothing hidden.
    fn perturb_hidden(cfg: &BridgeConfig, events: &[Event], seat: Seat, rng: &mut Rng) -> Option<Vec<Event>> {
        if !cfg.deal.is_empty() || seat >= 4 {
            return None;
        }
        let mut g = Bridge::new_game(cfg);
        let mut deal_at = Vec::with_capacity(52);
        for (i, e) in events.iter().enumerate() {
            match *e {
                Event::Chance(o) => {
                    if g.phase != Phase::Deal {
                        return None;
                    }
                    if g.board != 0 {
                        deal_at.push(i);
                    }
                    g.apply_chance(o).ok()?;
                }
                Event::Act(s, a) => g.apply_action(s, a).ok()?,
            }
        }
        if g.dealt < 52 {
            return None;
        }
        let dummy = if g.dummy_exposed() { partner(g.declarer) } else { NONE };
        let hidden: Vec<u8> = (0..4u8).filter(|&p| p != seat as u8 && p != dummy).collect();
        let mut pool: Vec<u8> = (0..52u8).filter(|&c| hidden.contains(&g.orig[c as usize]) && g.played_by[c as usize] == NONE).collect();
        if hidden.len() < 2 || pool.len() < 2 {
            return None;
        }
        let mut need = [0usize; 4];
        for &c in &pool {
            need[g.orig[c as usize] as usize] += 1;
        }
        let allowed = |c: u8, p: u8| !g.void[p as usize][suit(c) as usize];
        for _ in 0..32 {
            rng.shuffle(&mut pool);
            // most constrained cards first
            pool.sort_by_key(|&c| hidden.iter().filter(|&&p| allowed(c, p)).count());
            let mut cap = need;
            let mut new = g.orig;
            let mut ok = true;
            for &c in &pool {
                let total: usize = hidden.iter().filter(|&&p| allowed(c, p)).map(|&p| cap[p as usize]).sum();
                if total == 0 {
                    ok = false;
                    break;
                }
                let mut x = rng.below(total as u64) as usize;
                for &p in &hidden {
                    if !allowed(c, p) {
                        continue;
                    }
                    if x < cap[p as usize] {
                        new[c as usize] = p;
                        cap[p as usize] -= 1;
                        break;
                    }
                    x -= cap[p as usize];
                }
            }
            if !ok || new == g.orig {
                continue;
            }
            let mut out = events.to_vec();
            let mut j = [0usize; 4];
            for c in 0..52u8 {
                let p = new[c as usize] as usize;
                out[deal_at[4 * j[p] + p]] = Event::Chance(c as u32);
                j[p] += 1;
            }
            return Some(out);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariance::{check_game, check_privacy, random_trajectory};
    use crate::trajectory::replay;

    /// Events of a board-1 deal (fixed config) from four PBN-style hands N E S W
    /// ("S.H.D.C"), then `calls` / `cards` strings.
    fn deal_events(hands: [&str; 4]) -> Vec<Event> {
        let mut per: [Vec<u8>; 4] = Default::default();
        for (p, h) in hands.iter().enumerate() {
            for (k, part) in h.split('.').enumerate() {
                let s = 3 - k as u8;
                for ch in part.chars() {
                    per[p].push(card_id(s, parse_rank(&ch.to_string()).unwrap()));
                }
            }
            per[p].sort();
            assert_eq!(per[p].len(), 13, "{h}");
        }
        (0..52).map(|k| Event::Chance(per[k % 4][k / 4] as u32)).collect()
    }

    fn play_out(cfg: &BridgeConfig, mut ev: Vec<Event>, actions: &str) -> Vec<Event> {
        let mut g = Bridge::new_game(cfg);
        for e in &ev {
            match *e {
                Event::Chance(o) => g.apply_chance(o).unwrap(),
                Event::Act(s, a) => g.apply_action(s, a).unwrap(),
            }
        }
        for tok in actions.split_whitespace() {
            let a = if tok.len() == 2 && parse_card(tok).is_some() && !tok.starts_with(|c: char| c.is_ascii_digit()) {
                parse_card(tok).unwrap() as u32
            } else {
                parse_call(tok).unwrap_or_else(|| panic!("token {tok}"))
            };
            let Turn::Act(s) = g.turn() else { panic!("no decision for {tok}") };
            let seat = s.iter().next().unwrap();
            g.apply_action(seat, a).unwrap_or_else(|e| panic!("{tok}: {e}"));
            ev.push(Event::Act(seat, a));
        }
        ev
    }

    #[test]
    fn scoring_law_77() {
        // making contracts
        assert_eq!(score(4, 3, 0, 10, false), 420);
        assert_eq!(score(4, 3, 0, 11, true), 650);
        assert_eq!(score(3, 4, 0, 9, false), 400);
        assert_eq!(score(3, 4, 0, 10, true), 630);
        assert_eq!(score(1, 4, 0, 7, false), 90);
        assert_eq!(score(2, 0, 0, 8, false), 90);
        assert_eq!(score(5, 1, 0, 11, true), 600);
        assert_eq!(score(6, 3, 0, 12, false), 980);
        assert_eq!(score(6, 4, 0, 13, true), 1470);
        assert_eq!(score(7, 4, 0, 13, true), 2220);
        assert_eq!(score(7, 0, 0, 13, false), 1440);
        // doubled / redoubled makes: game by doubling, the insult, overtricks
        assert_eq!(score(2, 2, 1, 8, false), 470);
        assert_eq!(score(2, 2, 1, 9, true), 870);
        assert_eq!(score(1, 4, 1, 7, false), 180);
        assert_eq!(score(1, 4, 2, 7, false), 560);
        assert_eq!(score(1, 0, 2, 8, true), 630);
        assert_eq!(score(7, 4, 2, 13, true), 2980);
        // undertricks
        assert_eq!(score(4, 3, 0, 9, false), -50);
        assert_eq!(score(4, 3, 0, 7, true), -300);
        assert_eq!(score(3, 4, 1, 8, false), -100);
        assert_eq!(score(3, 4, 1, 7, false), -300);
        assert_eq!(score(3, 4, 1, 6, false), -500);
        assert_eq!(score(3, 4, 1, 5, false), -800);
        assert_eq!(score(3, 4, 1, 8, true), -200);
        assert_eq!(score(3, 4, 1, 6, true), -800);
        assert_eq!(score(3, 4, 2, 7, false), -600);
        assert_eq!(score(7, 4, 2, 0, true), -7600);
        assert_eq!(score(0, 0, 0, 0, false), 0);
        // IMPs
        assert_eq!((imps(0), imps(10), imps(20), imps(-50), imps(420), imps(620), imps(-1430), imps(7600)), (0, 0, 1, -2, 9, 12, -16, 24));
        assert_eq!((imps(4000), imps(3990), imps(2240)), (24, 23, 19));
    }

    #[test]
    fn boards_and_notation() {
        assert_eq!(board_dealer_vul(1), (0, [false, false]));
        assert_eq!(board_dealer_vul(2), (1, [true, false]));
        assert_eq!(board_dealer_vul(7), (2, [true, true]));
        assert_eq!(board_dealer_vul(16), (3, [false, true]));
        assert_eq!(board_dealer_vul(17), board_dealer_vul(1));
        for b in 1..=16 {
            let (d, v) = board_dealer_vul(b);
            assert_eq!(board_of(d, v), b);
        }
        assert_eq!(parse_card("SA"), Some(51));
        assert_eq!(parse_card("c2"), Some(0));
        assert_eq!(parse_card("H10"), Some(card_id(2, 8)));
        assert_eq!(card_string(card_id(1, 9)), "DJ");
        assert_eq!((parse_call("Pass"), parse_call("x"), parse_call("XX"), parse_call("1c"), parse_call("7NT"), parse_call("3N!")),
                   (Some(52), Some(53), Some(54), Some(55), Some(89), Some(bid_action(3, 4))));
        assert_eq!(parse_call("8C"), None);
        assert_eq!(call_string(59), "1NT");
        assert_eq!(bid_of(89), Some((7, 4)));
    }

    #[test]
    fn a_hand_through_the_rules() {
        // board 1: N deals, nobody vulnerable
        let cfg = BridgeConfig { board: 1, ..Default::default() };
        let hands = ["AKQJ.AKQ.AKQ.AKQ", "T98.JT9.JT9.JT98", "765.876.876.7654", "432.5432.5432.32"];
        let ev = deal_events(hands);
        let mut g = Bridge::new_game(&cfg);
        for e in &ev {
            if let Event::Chance(o) = *e {
                g.apply_chance(o).unwrap();
            }
        }
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)));
        let mut legal = Vec::new();
        g.legal_actions(0, &mut legal);
        assert_eq!(legal.len(), 36);
        assert_eq!(legal[0], PASS);
        // N opens 7NT, all pass: E leads; N declares, S is dummy
        let ev = play_out(&cfg, ev, "7NT Pass Pass Pass");
        let r = replay::<Bridge>(&cfg, &ev).unwrap();
        assert!(!r.terminal());
        assert_eq!(r.state.contract(), Some(Contract { level: 7, denom: 4, dbl: 0, declarer: 0 }));
        assert_eq!(r.state.turn(), Turn::Act(SeatSet::single(1)));
        // E leads the CJ... N wins everything: 13 tricks, dummy's plays are N's turns
        let mut g = r.state.clone();
        let mut ev = ev;
        let mut dummy_turns = 0;
        while g.turn() != Turn::Terminal {
            let Turn::Act(s) = g.turn() else { panic!() };
            let seat = s.iter().next().unwrap();
            if g.current == 2 {
                assert_eq!(seat, 0, "declarer plays dummy's cards");
                dummy_turns += 1;
            }
            let mut l = Vec::new();
            g.legal_actions(seat, &mut l);
            let a = *l.last().unwrap();
            g.apply_action(seat, a).unwrap();
            ev.push(Event::Act(seat, a));
        }
        assert_eq!(dummy_turns, 13);
        assert_eq!(g.declarer_tricks(), 13);
        assert_eq!(g.returns(), vec![1520.0, -1520.0, 1520.0, -1520.0]);
        assert_eq!(g.outcome(), vec![17.0 / 24.0, -17.0 / 24.0, 17.0 / 24.0, -17.0 / 24.0]);
        let r = replay::<Bridge>(&cfg, &ev).unwrap();
        assert!(r.terminal());
        // streams: N (auction + 26 plays), E and W 13 plays each + 1 call, S one call
        let spec = Bridge::spec(&cfg);
        let lens: Vec<usize> = r.streams(&spec).iter().map(|s| s.decisions.len()).collect();
        assert_eq!(lens, vec![27, 14, 1, 14]);
        // passed out
        let ev = play_out(&cfg, deal_events(hands), "Pass Pass Pass Pass");
        let r = replay::<Bridge>(&cfg, &ev).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.returns(), vec![0.0; 4]);
    }

    #[test]
    fn auction_rules() {
        let cfg = BridgeConfig { board: 1, ..Default::default() };
        let hands = ["AKQJ.AKQ.AKQ.AKQ", "T98.JT9.JT9.JT98", "765.876.876.7654", "432.5432.5432.32"];
        let ev = play_out(&cfg, deal_events(hands), "Pass 1H");
        let mut g = replay::<Bridge>(&cfg, &ev).unwrap().state;
        // S (partner of the passer, opponent of E's 1H) may double, not redouble
        let mut l = Vec::new();
        g.legal_actions(2, &mut l);
        assert_eq!(&l[..3], &[PASS, DOUBLE, bid_action(1, 3)]);
        g.apply_action(2, DOUBLE).unwrap();
        l.clear();
        g.legal_actions(3, &mut l);
        assert_eq!(&l[..3], &[PASS, REDOUBLE, bid_action(1, 3)], "W redoubles partner's doubled bid");
        assert!(g.apply_action(3, DOUBLE).is_err());
        g.apply_action(3, PASS).unwrap();
        l.clear();
        g.legal_actions(0, &mut l);
        assert_eq!(&l[..2], &[PASS, bid_action(1, 3)], "N cannot double a doubled bid");
        // declarer = the first of the side to bid the final denomination
        let ev = play_out(&cfg, deal_events(hands), "1S 2H Pass 4H 4S Pass Pass 5H Pass Pass Pass");
        let r = replay::<Bridge>(&cfg, &ev).unwrap();
        assert_eq!(r.state.contract(), Some(Contract { level: 5, denom: 2, dbl: 0, declarer: 1 }));
        assert_eq!(r.state.turn(), Turn::Act(SeatSet::single(2)), "S (left of E) leads");
        // illegal calls are rejected
        let mut bad = play_out(&cfg, deal_events(hands), "1S");
        bad.push(Event::Act(1, bid_action(1, 2)));
        assert!(replay::<Bridge>(&cfg, &bad).is_err());
    }

    #[test]
    fn follow_suit_trumps_and_trick_winner() {
        let cfg = BridgeConfig { board: 1, ..Default::default() };
        // N: all spades + C2; E: hearts; S: diamonds; W: clubs
        let hands = ["AKQJT98765432...", ".AKQJT98765432..", "..AKQJT98765432.", "...AKQJT98765432"];
        let ev = deal_events(hands);
        // E declares 1H (E opens after N passes), S leads a diamond, E trumps... W is dummy
        let ev = play_out(&cfg, ev, "Pass 1H Pass Pass Pass");
        let r = replay::<Bridge>(&cfg, &ev).unwrap();
        assert_eq!(r.state.contract().unwrap().declarer, 1);
        let mut g = r.state;
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(2)));
        g.apply_action(2, card_id(1, 12) as u32).unwrap(); // S: DA
        // W (dummy) plays: declarer E acts; W has only clubs → any club
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(1)));
        let mut l = Vec::new();
        g.legal_actions(1, &mut l);
        assert_eq!(l.len(), 13);
        assert!(l.iter().all(|&c| suit(c as u8) == 0));
        g.apply_action(1, card_id(0, 0) as u32).unwrap(); // C2 from dummy
        // N: no diamond → discards a spade
        g.apply_action(0, card_id(3, 0) as u32).unwrap();
        // E: no diamond → ruffs with H2 and wins
        g.apply_action(1, card_id(2, 0) as u32).unwrap();
        assert_eq!(g.tricks, [0, 1]);
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(1)));
        assert!(g.void[3][1] && g.void[0][1] && g.void[1][1]);
    }

    #[test]
    fn conformance_battery() {
        for cfg in [BridgeConfig::default(), BridgeConfig { board: 7, reward: "score".into(), ..Default::default() }] {
            let rep = check_game::<Bridge>(&cfg, 40, 5).unwrap();
            assert_eq!(rep.terminal, 40);
            assert!(rep.privacy_checks > 100, "{rep:?}");
            assert!(rep.max_legal <= MAX_ACTIONS);
        }
    }

    #[test]
    fn perturbation_respects_voids_and_visibility() {
        let cfg = BridgeConfig::default();
        let mut rng = Rng::new(3);
        let mut checked = 0;
        for _ in 0..40 {
            // cut the record somewhere in the auction or the play: by the end every card
            // has been played, so a whole record has nothing hidden left
            let whole = random_trajectory::<Bridge>(&cfg, &mut rng, 10_000);
            assert!(Bridge::perturb_hidden(&cfg, &whole, 0, &mut rng).is_none());
            let cut = 53 + rng.below(whole.len() as u64 - 53) as usize;
            let ev = whole[..cut].to_vec();
            let full = replay::<Bridge>(&cfg, &ev).unwrap();
            let decl = full.state.contract().map(|c| c.declarer);
            for seat in 0..4 {
                let Some(p) = Bridge::perturb_hidden(&cfg, &ev, seat, &mut rng) else { continue };
                let q = replay::<Bridge>(&cfg, &p).expect("perturbed records stay legal");
                // own hand and (once exposed) dummy's hand unchanged; played cards unchanged
                for c in 0..52u8 {
                    let (a, b) = (full.state.orig[c as usize], q.state.orig[c as usize]);
                    if a == seat as u8 || b == seat as u8 || full.state.played_by[c as usize] != NONE {
                        assert_eq!(a, b);
                    }
                    if let Some(d) = decl {
                        if full.state.n_played > 0 && (a == partner(d) || b == partner(d)) && seat as u8 != partner(d) {
                            assert_eq!(a, b, "dummy's exposed hand is not hidden");
                        }
                    }
                }
                check_privacy::<Bridge>(&cfg, &ev, &p, seat).unwrap();
                checked += 1;
            }
        }
        assert!(checked > 100);
        // scripted deals are never perturbed
        let script = BridgeConfig { deal: (0..52).collect(), ..Default::default() };
        let ev = random_trajectory::<Bridge>(&script, &mut rng, 10_000);
        assert!(Bridge::perturb_hidden(&script, &ev, 0, &mut rng).is_none());
    }

    #[test]
    fn frames_hide_what_the_viewer_cannot_see() {
        let cfg = BridgeConfig { board: 1, ..Default::default() };
        let hands = ["AKQJ.AKQ.AKQ.AKQ", "T98.JT9.JT9.JT98", "765.876.876.7654", "432.5432.5432.32"];
        let ev = play_out(&cfg, deal_events(hands), "1C Pass Pass Pass");
        let spec = Bridge::spec(&cfg);
        // E's opening lead: dummy (S) not visible yet
        let r = replay::<Bridge>(&cfg, &ev).unwrap();
        let mut g = r.state.clone();
        let mut f = crate::frame::Frame::new(&spec);
        crate::frame::encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
        let ents = f.obs[0].as_f32().unwrap();
        let at = |c: u8, k: usize| ents[c as usize * ENTITY_FEATURES + k];
        let s7 = card_id(3, 5); // S's spade 7
        assert_eq!((at(s7, C_UNSEEN), at(s7, C_DUMMY)), (1.0, 0.0));
        // after the lead: the declarer (N) acts for dummy and sees S's hand
        g.apply_action(1, card_id(0, 9) as u32).unwrap(); // E leads CJ
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)));
        crate::frame::encode_frame(&g, 0, &spec, &mut f.as_mut(&spec)).unwrap();
        let ents = f.obs[0].as_f32().unwrap();
        let glob = f.obs[2].as_f32().unwrap();
        let at = |c: u8, k: usize| ents[c as usize * ENTITY_FEATURES + k];
        assert_eq!((at(s7, C_DUMMY), at(s7, C_ACTING)), (1.0, 1.0));
        assert_eq!(glob[G_FOR_DUMMY], 1.0);
        // candidates: dummy's clubs (S follows suit), flagged "from dummy"
        assert!(f.legal_ids.iter().all(|&c| suit(c as u8) == 0 && g.holder[c as usize] == 2));
        let acts = f.obs[3].as_f32().unwrap();
        assert_eq!(acts[A_DUMMY], 1.0);
        // W (a defender) sees dummy now too — never N's or E's cards
        g.apply_action(0, f.legal_ids[0]).unwrap(); // dummy's club
        assert_eq!(g.turn(), Turn::Act(SeatSet::single(3)));
        crate::frame::encode_frame(&g, 3, &spec, &mut f.as_mut(&spec)).unwrap();
        let ents = f.obs[0].as_f32().unwrap();
        let at = |c: u8, k: usize| ents[c as usize * ENTITY_FEATURES + k];
        assert_eq!((at(s7, C_DUMMY), at(s7, C_ACTING)), (1.0, 0.0));
        let (n_sa, e_st) = (card_id(3, 12), card_id(3, 8));
        assert_eq!((at(n_sa, C_UNSEEN), at(e_st, C_UNSEEN)), (1.0, 1.0));
        // E's lead (W's partner, rel seat 2) is on the table; dummy's club did not beat it
        let cj = card_id(0, 9);
        assert_eq!((at(cj, C_PLAYED), at(cj, C_TABLE), at(cj, C_LED), at(cj, C_BY + 2), at(cj, C_WINNING)),
                   (1.0, 1.0, 1.0, 1.0, 1.0));
    }
}
