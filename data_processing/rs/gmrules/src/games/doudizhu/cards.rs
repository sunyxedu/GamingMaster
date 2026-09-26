//! Dou Dizhu cards and combinations: ranks, play kinds, the beating rule, kicker
//! constraints, canonical ids and strings (OpenSpiel's `dou_dizhu` conventions).
//!
//! Ranks `0..15`: `3 4 5 6 7 8 9 T J Q K A 2` (0..=12), black/white joker `(BWJ)` (13),
//! coloured joker `(CJ)` (14). Card ids `0..54` as OpenSpiel: `card < 52` → rank
//! `card % 13`, suit `card / 13` (`C D H S`); 52 / 53 are the jokers. Suits never
//! matter for play: hands are rank counts ([`Counts`]).
//!
//! A full play ([`Play`]) is a kind, a main part (a rank, or a chain `head..head+len`)
//! and kickers (solo cards or pairs, ranks non-decreasing). The policy plays kicker-free
//! combinations in one decision, and a combination with kickers in several: the MAIN
//! (kind + main part, [`Main`]) first, then one kicker unit per decision in
//! non-decreasing rank order — each full play has exactly one such path, and every
//! prefix offered is completable ([`kicker_options`]), so the decisions reach exactly
//! the legal full plays. Canonical action ids ([`Action`]):
//!
//! ```text
//! 0                 pass (auction and play)           1..=3  bids
//! 4..=26056         kicker-free plays with OpenSpiel's ids (solo, solo chain, pair,
//!                   pair chain, trio, airplane, bomb, rocket); OpenSpiel's kicker-play
//!                   ranges (133..=470, 516..=26042) are not used
//! 26057..=26176     mains of the kicker kinds: trio+solo (13), trio+pair (13),
//!                   airplane+solos (38: length 2-5 x head), airplane+pairs (30: 2-4),
//!                   four+2 solos (13), four+2 pairs (13)
//! 26177..=26191     solo kicker of rank r      26192..=26204  pair kicker of rank r
//! ```

use std::fmt::Write as _;

pub const N_RANKS: usize = 15;
pub const N_CARDS: usize = 54;
pub const ACE: u8 = 11;
pub const TWO: u8 = 12;
pub const BJ: u8 = 13;
pub const RJ: u8 = 14;
/// Most cards a hand can hold (the landlord's 17 + 3).
pub const MAX_HAND: u8 = 20;

const RANK_CHARS: &[u8; 13] = b"3456789TJQKA2";
const SUIT_CHARS: &[u8; 4] = b"CDHS";

/// Cards per rank.
pub type Counts = [u8; N_RANKS];

pub fn card_rank(card: u8) -> u8 {
    if card >= 52 { card - 52 + BJ } else { card % 13 }
}

pub fn rank_str(r: u8) -> &'static str {
    const S: [&str; N_RANKS] = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "(BWJ)", "(CJ)"];
    S[r as usize]
}

/// OpenSpiel's `CardString`: suit + rank (`C3`, `S2`), jokers `(BWJ)` / `(CJ)`.
pub fn card_str(card: u8) -> String {
    let r = card_rank(card);
    if r >= BJ {
        rank_str(r).to_string()
    } else {
        format!("{}{}", SUIT_CHARS[(card / 13) as usize] as char, RANK_CHARS[r as usize] as char)
    }
}

pub fn n_cards(c: &Counts) -> u8 {
    c.iter().sum()
}

/// OpenSpiel's `FormatSingleHand`: every card, ascending rank.
pub fn counts_str(c: &Counts) -> String {
    let mut s = String::new();
    for r in 0..N_RANKS as u8 {
        for _ in 0..c[r as usize] {
            s.push_str(rank_str(r));
        }
    }
    s
}

// ---------------------------------------------------------------------------
// kinds
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Solo,
    Pair,
    Trio,
    Bomb,
    Rocket,
    SoloChain,
    PairChain,
    /// a chain of trios without kickers
    Airplane,
    TrioSolo,
    TrioPair,
    AirplaneSolo,
    AirplanePair,
    /// four of a kind + 2 solo cards (not a bomb)
    FourSolo,
    /// four of a kind + 2 pairs (not a bomb)
    FourPair,
}

pub const KINDS: [Kind; 14] = [
    Kind::Solo,
    Kind::Pair,
    Kind::Trio,
    Kind::Bomb,
    Kind::Rocket,
    Kind::SoloChain,
    Kind::PairChain,
    Kind::Airplane,
    Kind::TrioSolo,
    Kind::TrioPair,
    Kind::AirplaneSolo,
    Kind::AirplanePair,
    Kind::FourSolo,
    Kind::FourPair,
];

/// A kicker unit: one card, or a pair of one rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Solo,
    Pair,
}

impl Unit {
    pub fn cards(self) -> u8 {
        match self {
            Unit::Solo => 1,
            Unit::Pair => 2,
        }
    }
}

impl Kind {
    pub fn index(self) -> usize {
        self as usize
    }

    /// Cards per rank of the main part.
    pub fn width(self) -> u8 {
        match self {
            Kind::Solo | Kind::SoloChain | Kind::Rocket => 1,
            Kind::Pair | Kind::PairChain => 2,
            Kind::Trio | Kind::Airplane | Kind::TrioSolo | Kind::TrioPair | Kind::AirplaneSolo | Kind::AirplanePair => 3,
            Kind::Bomb | Kind::FourSolo | Kind::FourPair => 4,
        }
    }

    pub fn kicker(self) -> Option<Unit> {
        match self {
            Kind::TrioSolo | Kind::AirplaneSolo | Kind::FourSolo => Some(Unit::Solo),
            Kind::TrioPair | Kind::AirplanePair | Kind::FourPair => Some(Unit::Pair),
            _ => None,
        }
    }

    pub fn n_kickers(self, len: u8) -> u8 {
        match self {
            Kind::TrioSolo | Kind::TrioPair => 1,
            Kind::AirplaneSolo | Kind::AirplanePair => len,
            Kind::FourSolo | Kind::FourPair => 2,
            _ => 0,
        }
    }

    /// Solo kickers of one rank at most (OpenSpiel: an airplane never carries a bomb;
    /// the two cards of four+2 may be a pair).
    fn solo_cap(self) -> u8 {
        match self {
            Kind::AirplaneSolo => 3,
            Kind::FourSolo => 2,
            _ => 1,
        }
    }

    /// Allowed main lengths (1 = a single-rank main). Chains stay within 3..A; the
    /// maxima are the longest that fit a 20-card hand.
    pub fn len_range(self) -> (u8, u8) {
        match self {
            Kind::SoloChain => (5, 12),
            Kind::PairChain => (3, 10),
            Kind::Airplane => (2, 6),
            Kind::AirplaneSolo => (2, 5),
            Kind::AirplanePair => (2, 4),
            _ => (1, 1),
        }
    }

    pub fn is_chain(self) -> bool {
        self.len_range().1 > 1
    }

    /// Doubles the stake (bombs and the rocket).
    pub fn doubles(self) -> bool {
        matches!(self, Kind::Bomb | Kind::Rocket)
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Solo => "solo",
            Kind::Pair => "pair",
            Kind::Trio => "trio",
            Kind::Bomb => "bomb",
            Kind::Rocket => "rocket",
            Kind::SoloChain => "solo_chain",
            Kind::PairChain => "pair_chain",
            Kind::Airplane => "airplane",
            Kind::TrioSolo => "trio_solo",
            Kind::TrioPair => "trio_pair",
            Kind::AirplaneSolo => "airplane_solo",
            Kind::AirplanePair => "airplane_pair",
            Kind::FourSolo => "four_solo",
            Kind::FourPair => "four_pair",
        }
    }
}

/// The rules variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The standard game: OpenSpiel's rules + four-with-two (four + 2 solos, four + 2
    /// pairs), and a chain may start at a rank the player holds four of.
    Standard,
    /// Exactly OpenSpiel's `dou_dizhu` (no four-with-two; no solo / pair chain or
    /// kicker-free airplane starts at a rank of which the player holds four; any bomb
    /// may follow the rocket) — for the bit-exact differential tests.
    OpenSpiel,
}

impl Variant {
    pub fn four_with_two(self) -> bool {
        self == Variant::Standard
    }

    /// OpenSpiel's `SearchChainOnlyActions` skips chain heads the hand holds four of.
    fn chain_head_quirk(self) -> bool {
        self == Variant::OpenSpiel
    }

    /// OpenSpiel's `SearchForLegalActions` offers every bomb after the rocket (only
    /// after a bomb does it require a higher one); in the real game nothing beats it.
    fn bomb_beats_rocket(self) -> bool {
        self == Variant::OpenSpiel
    }
}

// ---------------------------------------------------------------------------
// plays
// ---------------------------------------------------------------------------

/// The main part of a combination with kickers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Main {
    pub kind: Kind,
    pub head: u8,
    pub len: u8,
}

impl Main {
    pub fn n_kickers(&self) -> u8 {
        self.kind.n_kickers(self.len)
    }

    pub fn unit(&self) -> Unit {
        self.kind.kicker().expect("a main has kickers")
    }

    /// Cards of the main part (no kickers).
    pub fn counts(&self) -> Counts {
        let mut c = [0u8; N_RANKS];
        for r in self.head..self.head + self.len {
            c[r as usize] = self.kind.width();
        }
        c
    }

    fn covers(&self, r: u8) -> bool {
        (self.head..self.head + self.len).contains(&r)
    }
}

/// A full play: kind, main part (`head`, `len`; the rocket has head BJ) and the
/// kicker units (ranks, non-decreasing; pairs strictly increasing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Play {
    pub kind: Kind,
    pub head: u8,
    pub len: u8,
    pub n_kick: u8,
    pub kickers: [u8; 5],
}

impl Play {
    pub fn new(kind: Kind, head: u8, len: u8) -> Play {
        Play { kind, head, len, n_kick: 0, kickers: [0; 5] }
    }

    pub fn rocket() -> Play {
        Play::new(Kind::Rocket, BJ, 1)
    }

    pub fn with_kickers(main: Main, kickers: &[u8]) -> Play {
        let mut p = Play::new(main.kind, main.head, main.len);
        p.n_kick = kickers.len() as u8;
        p.kickers[..kickers.len()].copy_from_slice(kickers);
        p
    }

    pub fn kickers(&self) -> &[u8] {
        &self.kickers[..self.n_kick as usize]
    }

    pub fn main(&self) -> Main {
        Main { kind: self.kind, head: self.head, len: self.len }
    }

    /// Cards of the main part.
    pub fn main_counts(&self) -> Counts {
        let mut c = [0u8; N_RANKS];
        if self.kind == Kind::Rocket {
            c[BJ as usize] = 1;
            c[RJ as usize] = 1;
        } else {
            for r in self.head..self.head + self.len {
                c[r as usize] = self.kind.width();
            }
        }
        c
    }

    /// Every card of the play.
    pub fn counts(&self) -> Counts {
        let mut c = self.main_counts();
        if let Some(u) = self.kind.kicker() {
            for &k in self.kickers() {
                c[k as usize] += u.cards();
            }
        }
        c
    }

    pub fn n_cards(&self) -> u8 {
        n_cards(&self.counts())
    }
}

/// Does a play of `kind` / `head` / `len` beat `prev`? Rocket beats everything, a
/// bomb everything but the rocket and higher bombs; otherwise the same kind and
/// length with a higher main (kickers never matter). (OpenSpiel variant: any bomb
/// also "beats" the rocket.)
pub fn beats(variant: Variant, kind: Kind, head: u8, len: u8, prev: &Play) -> bool {
    match (kind, prev.kind) {
        (Kind::Bomb, Kind::Rocket) => variant.bomb_beats_rocket(),
        (_, Kind::Rocket) => false,
        (Kind::Rocket, _) => true,
        (Kind::Bomb, Kind::Bomb) => head > prev.head,
        (Kind::Bomb, _) => true,
        (k, p) => k == p && len == prev.len && head > prev.head,
    }
}

// ---------------------------------------------------------------------------
// kickers
// ---------------------------------------------------------------------------

/// Kicker cards available for `main`: the hand without the main's ranks (a kicker is
/// never of a main rank).
fn kicker_avail(hand: &Counts, main: &Main) -> Counts {
    let mut a = *hand;
    for r in main.head..main.head + main.len {
        a[r as usize] = 0;
    }
    a
}

/// Can one more unit of rank `r` be taken (given `used` units per rank)?
fn can_take(main: &Main, avail: &Counts, used: &Counts, r: u8) -> bool {
    let (ri, u) = (r as usize, main.unit());
    if main.covers(r) {
        return false;
    }
    match u {
        Unit::Solo => {
            if r >= BJ {
                avail[ri] >= 1 && used[BJ as usize] + used[RJ as usize] == 0 // one joker at most: never the rocket
            } else {
                used[ri] < main.kind.solo_cap() && avail[ri] > used[ri]
            }
        }
        Unit::Pair => r < BJ && used[ri] == 0 && avail[ri] >= 2,
    }
}

/// Can `m` more units be chosen from ranks `>= lo` (solos: non-decreasing, so rank
/// `lo` itself counts again; pairs: `lo` = the next free rank)?
fn can_fill(main: &Main, avail: &Counts, used: &Counts, lo: u8, m: u8) -> bool {
    if m == 0 {
        return true;
    }
    let mut cap = 0u8;
    match main.unit() {
        Unit::Solo => {
            let sc = main.kind.solo_cap();
            for r in lo..BJ {
                if !main.covers(r) {
                    let ri = r as usize;
                    cap += avail[ri].saturating_sub(used[ri]).min(sc.saturating_sub(used[ri]));
                }
            }
            let joker_used = used[BJ as usize] + used[RJ as usize] > 0;
            if !joker_used && ((lo <= BJ && avail[BJ as usize] > 0) || (lo <= RJ && avail[RJ as usize] > 0)) {
                cap += 1;
            }
        }
        Unit::Pair => {
            for r in lo..BJ {
                if can_take(main, avail, used, r) {
                    cap += 1;
                }
            }
        }
    }
    cap >= m
}

/// The kicker ranks `seat` may choose next for `main` after `chosen` (ascending):
/// those keeping the order (solos non-decreasing, pairs increasing) with at least
/// one completion of the remaining units.
pub fn kicker_options(hand: &Counts, main: &Main, chosen: &[u8], out: &mut Vec<u8>) {
    let total = main.n_kickers();
    if chosen.len() >= total as usize {
        return;
    }
    let m = total - chosen.len() as u8;
    let unit = main.unit();
    let avail = kicker_avail(hand, main);
    let mut used = [0u8; N_RANKS];
    for &k in chosen {
        used[k as usize] += 1;
    }
    let lo = match (unit, chosen.last()) {
        (_, None) => 0,
        (Unit::Solo, Some(&k)) => k,
        (Unit::Pair, Some(&k)) => k + 1,
    };
    for r in lo..N_RANKS as u8 {
        if !can_take(main, &avail, &used, r) {
            continue;
        }
        used[r as usize] += 1;
        let next = if unit == Unit::Solo { r } else { r + 1 };
        if can_fill(main, &avail, &used, next, m - 1) {
            out.push(r);
        }
        used[r as usize] -= 1;
    }
}

/// Does `main` have at least one kicker completion from `hand`?
pub fn completable(hand: &Counts, main: &Main) -> bool {
    can_fill(main, &kicker_avail(hand, main), &[0; N_RANKS], 0, main.n_kickers())
}

// ---------------------------------------------------------------------------
// legal options
// ---------------------------------------------------------------------------

/// Kicker-free plays and mains `hand` may choose (canonical ids, unsorted): every one
/// when leading (`prev` None), else those beating `prev` (the caller adds pass).
pub fn main_options(hand: &Counts, prev: Option<&Play>, variant: Variant, out: &mut Vec<u32>) {
    let ok = |kind: Kind, head: u8, len: u8| prev.is_none_or(|p| beats(variant, kind, head, len, p));
    // single-rank plays and the rocket
    for r in 0..N_RANKS as u8 {
        let c = hand[r as usize];
        let ranks: &[(Kind, u8)] = &[(Kind::Solo, 1), (Kind::Pair, 2), (Kind::Trio, 3), (Kind::Bomb, 4)];
        for &(kind, need) in ranks {
            if c >= need && (r < BJ || kind == Kind::Solo) && ok(kind, r, 1) {
                out.push(Action::Play(Play::new(kind, r, 1)).id());
            }
        }
    }
    if hand[BJ as usize] > 0 && hand[RJ as usize] > 0 && ok(Kind::Rocket, BJ, 1) {
        out.push(ROCKET);
    }
    // a follower only plays its kind (or bombs / the rocket, above)
    let wants = |kind: Kind| prev.is_none_or(|p| p.kind == kind);
    for kind in [Kind::SoloChain, Kind::PairChain, Kind::Airplane] {
        if !wants(kind) {
            continue;
        }
        let (lmin, lmax) = kind.len_range();
        let w = kind.width();
        for head in 0..=ACE {
            if variant.chain_head_quirk() && hand[head as usize] == 4 {
                continue;
            }
            for len in 1..=lmax {
                let tail = head + len - 1;
                if tail > ACE || hand[tail as usize] < w {
                    break;
                }
                if len >= lmin && ok(kind, head, len) {
                    out.push(Action::Play(Play::new(kind, head, len)).id());
                }
            }
        }
    }
    // mains of the kicker kinds (with at least one completion)
    let mut push_main = |m: Main| {
        if ok(m.kind, m.head, m.len) && completable(hand, &m) {
            out.push(Action::Main(m).id());
        }
    };
    for kind in [Kind::TrioSolo, Kind::TrioPair] {
        if wants(kind) {
            for t in 0..BJ {
                if hand[t as usize] >= 3 {
                    push_main(Main { kind, head: t, len: 1 });
                }
            }
        }
    }
    for kind in [Kind::AirplaneSolo, Kind::AirplanePair] {
        if !wants(kind) {
            continue;
        }
        let (lmin, lmax) = kind.len_range();
        for head in 0..=ACE {
            for len in 1..=lmax {
                let tail = head + len - 1;
                if tail > ACE || hand[tail as usize] < 3 {
                    break;
                }
                if len >= lmin {
                    push_main(Main { kind, head, len });
                }
            }
        }
    }
    if variant.four_with_two() {
        for kind in [Kind::FourSolo, Kind::FourPair] {
            if wants(kind) {
                for f in 0..BJ {
                    if hand[f as usize] == 4 {
                        push_main(Main { kind, head: f, len: 1 });
                    }
                }
            }
        }
    }
}

/// Every legal full play (kickers included) of `hand` after `prev`, in the order of
/// their decision paths — the set the factored decisions reach (tests, tools).
pub fn legal_plays(hand: &Counts, prev: Option<&Play>, variant: Variant) -> Vec<Play> {
    let mut ids = Vec::new();
    main_options(hand, prev, variant, &mut ids);
    ids.sort_unstable();
    let mut out = Vec::new();
    for id in ids {
        match Action::from_id(id) {
            Some(Action::Play(p)) => out.push(p),
            Some(Action::Main(m)) => {
                let mut chosen = Vec::new();
                complete(hand, &m, &mut chosen, &mut out);
            }
            _ => unreachable!("main options are plays or mains"),
        }
    }
    out
}

fn complete(hand: &Counts, m: &Main, chosen: &mut Vec<u8>, out: &mut Vec<Play>) {
    if chosen.len() == m.n_kickers() as usize {
        out.push(Play::with_kickers(*m, chosen));
        return;
    }
    let mut opts = Vec::new();
    kicker_options(hand, m, chosen, &mut opts);
    for r in opts {
        chosen.push(r);
        complete(hand, m, chosen, out);
        chosen.pop();
    }
}

/// Is `play` legal from `hand` after `prev` (checked from the definitions: the hand
/// holds the cards, the kinds' static constraints, the beating rule)? Independent of
/// [`main_options`] / [`kicker_options`] (the tests compare the two).
pub fn is_legal_play(hand: &Counts, prev: Option<&Play>, variant: Variant, play: &Play) -> bool {
    if !valid_play(play, variant) {
        return false;
    }
    if variant.chain_head_quirk()
        && matches!(play.kind, Kind::SoloChain | Kind::PairChain | Kind::Airplane)
        && hand[play.head as usize] == 4
    {
        return false;
    }
    let c = play.counts();
    if (0..N_RANKS).any(|r| c[r] > hand[r]) {
        return false;
    }
    prev.is_none_or(|p| beats(variant, play.kind, play.head, play.len, p))
}

/// The static constraints of a play (no hand, no previous play).
pub fn valid_play(p: &Play, variant: Variant) -> bool {
    let (lmin, lmax) = p.kind.len_range();
    if p.len < lmin || p.len > lmax {
        return false;
    }
    if p.kind == Kind::Rocket {
        return p.head == BJ && p.len == 1 && p.n_kick == 0;
    }
    let top = p.head as u32 + p.len as u32 - 1;
    let max_rank = match p.kind {
        Kind::Solo => RJ,
        _ if p.kind.is_chain() => ACE,
        _ => TWO,
    };
    if top > max_rank as u32 {
        return false;
    }
    if matches!(p.kind, Kind::FourSolo | Kind::FourPair) && !variant.four_with_two() {
        return false;
    }
    let Some(unit) = p.kind.kicker() else {
        return p.n_kick == 0;
    };
    let ks = p.kickers();
    if ks.len() != p.kind.n_kickers(p.len) as usize {
        return false;
    }
    let main = p.main();
    let mut used = [0u8; N_RANKS];
    for (i, &k) in ks.iter().enumerate() {
        if k as usize >= N_RANKS || main.covers(k) {
            return false;
        }
        if i > 0 && (k < ks[i - 1] || (unit == Unit::Pair && k == ks[i - 1])) {
            return false;
        }
        used[k as usize] += 1;
    }
    match unit {
        Unit::Solo => {
            let joker = used[BJ as usize] + used[RJ as usize];
            joker <= 1 && (0..BJ as usize).all(|r| used[r] <= p.kind.solo_cap())
        }
        Unit::Pair => used[BJ as usize] + used[RJ as usize] == 0,
    }
}

/// Every non-decreasing sequence of `n` ranks `>= lo`.
fn nondecreasing(n: usize, lo: u8, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    if cur.len() == n {
        out.push(cur.clone());
        return;
    }
    for r in lo..N_RANKS as u8 {
        cur.push(r);
        nondecreasing(n, r, cur, out);
        cur.pop();
    }
}

/// Every play of the game (the universe, built from the static constraints alone).
pub fn all_plays(variant: Variant) -> Vec<Play> {
    let mut out = Vec::new();
    for r in 0..N_RANKS as u8 {
        out.push(Play::new(Kind::Solo, r, 1));
    }
    for kind in [Kind::Pair, Kind::Trio, Kind::Bomb] {
        for r in 0..BJ {
            out.push(Play::new(kind, r, 1));
        }
    }
    out.push(Play::rocket());
    for kind in [Kind::SoloChain, Kind::PairChain, Kind::Airplane] {
        let (lmin, lmax) = kind.len_range();
        for len in lmin..=lmax {
            for head in 0..=(ACE + 1 - len) {
                out.push(Play::new(kind, head, len));
            }
        }
    }
    let mut kinds = vec![Kind::TrioSolo, Kind::TrioPair, Kind::AirplaneSolo, Kind::AirplanePair];
    if variant.four_with_two() {
        kinds.extend([Kind::FourSolo, Kind::FourPair]);
    }
    for kind in kinds {
        let (lmin, lmax) = kind.len_range();
        for len in lmin..=lmax {
            let heads = if kind.is_chain() { 0..=(ACE + 1 - len) } else { 0..=TWO };
            for head in heads {
                let main = Main { kind, head, len };
                let mut seqs = Vec::new();
                nondecreasing(main.n_kickers() as usize, 0, &mut Vec::new(), &mut seqs);
                for ks in seqs {
                    let p = Play::with_kickers(main, &ks);
                    if valid_play(&p, variant) {
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// canonical ids
// ---------------------------------------------------------------------------

pub const PASS: u32 = 0;
pub const MAX_BID: u8 = 3;
pub const SOLO_BASE: u32 = 4;
pub const SOLO_CHAIN_BASE: u32 = 19;
pub const PAIR_BASE: u32 = 55;
pub const PAIR_CHAIN_BASE: u32 = 68;
pub const TRIO_BASE: u32 = 120;
pub const AIRPLANE_BASE: u32 = 471;
pub const BOMB_BASE: u32 = 26043;
pub const ROCKET: u32 = 26056;
/// First id after OpenSpiel's action space: the mains of the kicker kinds.
pub const MAIN_BASE: u32 = 26057;
const M_TRIO_SOLO: u32 = MAIN_BASE;
const M_TRIO_PAIR: u32 = MAIN_BASE + 13;
const M_AIR_SOLO: u32 = MAIN_BASE + 26;
const M_AIR_PAIR: u32 = MAIN_BASE + 64;
const M_FOUR_SOLO: u32 = MAIN_BASE + 94;
const M_FOUR_PAIR: u32 = MAIN_BASE + 107;
pub const KICK_SOLO_BASE: u32 = MAIN_BASE + 120;
pub const KICK_PAIR_BASE: u32 = KICK_SOLO_BASE + 15;
/// Size of the canonical action id space.
pub const N_ACTION_IDS: u32 = KICK_PAIR_BASE + 13;
/// Chance outcomes: face-up position `0..51`, then the dealt card `51 + card`.
pub const DEAL_BASE: u32 = 51;

/// Index of a chain (`len`, `head`) among the chains of lengths `lmin..` (OpenSpiel's
/// `ChainOnlyHandToActionId`: by length, then head).
fn chain_index(lmin: u8, len: u8, head: u8) -> u32 {
    (lmin..len).map(|l| 13 - l as u32).sum::<u32>() + head as u32
}

fn chain_of_index(lmin: u8, lmax: u8, mut i: u32) -> Option<(u8, u8)> {
    for len in lmin..=lmax {
        let n = 13 - len as u32;
        if i < n {
            return Some((len, i as u8));
        }
        i -= n;
    }
    None
}

/// A canonical action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Pass,
    Bid(u8),
    /// a kicker-free play
    Play(Play),
    Main(Main),
    KickSolo(u8),
    KickPair(u8),
}

impl Action {
    pub fn id(&self) -> u32 {
        match *self {
            Action::Pass => PASS,
            Action::Bid(b) => b as u32,
            Action::Play(p) => match p.kind {
                Kind::Solo => SOLO_BASE + p.head as u32,
                Kind::Pair => PAIR_BASE + p.head as u32,
                Kind::Trio => TRIO_BASE + p.head as u32,
                Kind::Bomb => BOMB_BASE + p.head as u32,
                Kind::Rocket => ROCKET,
                Kind::SoloChain => SOLO_CHAIN_BASE + chain_index(5, p.len, p.head),
                Kind::PairChain => PAIR_CHAIN_BASE + chain_index(3, p.len, p.head),
                Kind::Airplane => AIRPLANE_BASE + chain_index(2, p.len, p.head),
                k => panic!("{k:?} has kickers: its main is the action"),
            },
            Action::Main(m) => match m.kind {
                Kind::TrioSolo => M_TRIO_SOLO + m.head as u32,
                Kind::TrioPair => M_TRIO_PAIR + m.head as u32,
                Kind::AirplaneSolo => M_AIR_SOLO + chain_index(2, m.len, m.head),
                Kind::AirplanePair => M_AIR_PAIR + chain_index(2, m.len, m.head),
                Kind::FourSolo => M_FOUR_SOLO + m.head as u32,
                Kind::FourPair => M_FOUR_PAIR + m.head as u32,
                k => panic!("{k:?} has no kickers"),
            },
            Action::KickSolo(r) => KICK_SOLO_BASE + r as u32,
            Action::KickPair(r) => KICK_PAIR_BASE + r as u32,
        }
    }

    pub fn from_id(id: u32) -> Option<Action> {
        let range = |lo: u32, n: u32| (lo..lo + n).contains(&id).then(|| id - lo);
        let single = |kind: Kind, r: u32| Action::Play(Play::new(kind, r as u8, 1));
        let chain = |kind: Kind, i: u32| {
            let (lmin, lmax) = kind.len_range();
            chain_of_index(lmin, lmax, i).map(|(len, head)| Action::Play(Play::new(kind, head, len)))
        };
        let main_chain = |kind: Kind, i: u32| {
            let (lmin, lmax) = kind.len_range();
            chain_of_index(lmin, lmax, i).map(|(len, head)| Action::Main(Main { kind, head, len }))
        };
        let main1 = |kind: Kind, r: u32| Action::Main(Main { kind, head: r as u8, len: 1 });
        if id == PASS {
            Some(Action::Pass)
        } else if (1..=MAX_BID as u32).contains(&id) {
            Some(Action::Bid(id as u8))
        } else if let Some(r) = range(SOLO_BASE, 15) {
            Some(single(Kind::Solo, r))
        } else if let Some(i) = range(SOLO_CHAIN_BASE, 36) {
            chain(Kind::SoloChain, i)
        } else if let Some(r) = range(PAIR_BASE, 13) {
            Some(single(Kind::Pair, r))
        } else if let Some(i) = range(PAIR_CHAIN_BASE, 52) {
            chain(Kind::PairChain, i)
        } else if let Some(r) = range(TRIO_BASE, 13) {
            Some(single(Kind::Trio, r))
        } else if let Some(i) = range(AIRPLANE_BASE, 45) {
            chain(Kind::Airplane, i)
        } else if let Some(r) = range(BOMB_BASE, 13) {
            Some(single(Kind::Bomb, r))
        } else if id == ROCKET {
            Some(Action::Play(Play::rocket()))
        } else if let Some(r) = range(M_TRIO_SOLO, 13) {
            Some(main1(Kind::TrioSolo, r))
        } else if let Some(r) = range(M_TRIO_PAIR, 13) {
            Some(main1(Kind::TrioPair, r))
        } else if let Some(i) = range(M_AIR_SOLO, 38) {
            main_chain(Kind::AirplaneSolo, i)
        } else if let Some(i) = range(M_AIR_PAIR, 30) {
            main_chain(Kind::AirplanePair, i)
        } else if let Some(r) = range(M_FOUR_SOLO, 13) {
            Some(main1(Kind::FourSolo, r))
        } else if let Some(r) = range(M_FOUR_PAIR, 13) {
            Some(main1(Kind::FourPair, r))
        } else if let Some(r) = range(KICK_SOLO_BASE, 15) {
            Some(Action::KickSolo(r as u8))
        } else {
            range(KICK_PAIR_BASE, 13).map(|r| Action::KickPair(r as u8))
        }
    }
}

/// The decision path of a full play: its id (kicker-free), or its main followed by
/// one kicker id per unit.
pub fn play_path(p: &Play) -> Vec<u32> {
    match p.kind.kicker() {
        None => vec![Action::Play(*p).id()],
        Some(u) => {
            let mut v = vec![Action::Main(p.main()).id()];
            v.extend(p.kickers().iter().map(|&k| match u {
                Unit::Solo => Action::KickSolo(k).id(),
                Unit::Pair => Action::KickPair(k).id(),
            }));
            v
        }
    }
}

// ---------------------------------------------------------------------------
// strings
// ---------------------------------------------------------------------------

/// A play as OpenSpiel prints it: every card by ascending rank (`34567`, `3444`,
/// `(BWJ)(CJ)`), airplanes with kickers as chain `-` kickers (`333444-56`); four+2
/// (not in OpenSpiel) likewise (`7777-45`, `7777-4455`).
pub fn play_str(p: &Play) -> String {
    match p.kind {
        Kind::AirplaneSolo | Kind::AirplanePair | Kind::FourSolo | Kind::FourPair => {
            let mut s = counts_str(&p.main_counts());
            s.push('-');
            let mut k = p.counts();
            for r in p.head..p.head + p.len {
                k[r as usize] = 0;
            }
            s.push_str(&counts_str(&k));
            s
        }
        _ => counts_str(&p.counts()),
    }
}

/// A main with `chosen` kickers so far: main `-` chosen kicker cards, `*` per card
/// still to choose (`444-*`, `333444-5*`, `7777-44**`).
pub fn partial_str(m: &Main, chosen: &[u8]) -> String {
    let mut s = counts_str(&m.counts());
    s.push('-');
    let u = m.unit().cards();
    let mut k = [0u8; N_RANKS];
    for &r in chosen {
        k[r as usize] += u;
    }
    s.push_str(&counts_str(&k));
    for _ in 0..(m.n_kickers() as usize - chosen.len()) * u as usize {
        s.push('*');
    }
    s
}

/// Rank counts of a card string (`3344(BWJ)`; at most 4 of a rank, 1 of a joker).
pub fn parse_counts(s: &str) -> Result<Counts, String> {
    let mut c = [0u8; N_RANKS];
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'(' {
            let end = s[i..].find(')').ok_or_else(|| format!("unclosed joker in {s:?}"))? + i;
            match &s[i..=end] {
                "(BWJ)" => c[BJ as usize] += 1,
                "(CJ)" => c[RJ as usize] += 1,
                j => return Err(format!("unknown joker {j:?} in {s:?}")),
            }
            i = end + 1;
        } else {
            let r = RANK_CHARS.iter().position(|&x| x == b[i]).ok_or_else(|| format!("rank {:?} in {s:?}", b[i] as char))?;
            c[r] += 1;
            i += 1;
        }
    }
    if c.iter().enumerate().any(|(r, &n)| n > if r >= BJ as usize { 1 } else { 4 }) {
        return Err(format!("too many cards of one rank in {s:?}"));
    }
    Ok(c)
}

/// The consecutive run of ranks `c` covers with `w` cards each, as (head, len).
fn run_of(c: &Counts, w: u8) -> Option<(u8, u8)> {
    let ranks: Vec<u8> = (0..N_RANKS as u8).filter(|&r| c[r as usize] > 0).collect();
    let (&head, &tail) = (ranks.first()?, ranks.last()?);
    let consecutive = tail as usize - head as usize + 1 == ranks.len();
    (consecutive && ranks.iter().all(|&r| c[r as usize] == w)).then_some((head, ranks.len() as u8))
}

/// Parse [`play_str`]'s format back into a play (validated by [`valid_play`] for the
/// standard variant).
pub fn parse_play(s: &str) -> Result<Play, String> {
    let s = s.trim();
    let p = if let Some((main_s, kick_s)) = s.split_once('-') {
        let mc = parse_counts(main_s)?;
        let kc = parse_counts(kick_s)?;
        let nk = n_cards(&kc);
        let (kind, head, len) = if let Some((head, len)) = run_of(&mc, 3).filter(|&(_, l)| l >= 2) {
            let kind = if nk == len {
                Kind::AirplaneSolo
            } else if nk == 2 * len {
                Kind::AirplanePair
            } else {
                return Err(format!("{s:?}: an airplane of {len} carries {len} solos or {len} pairs, not {nk} cards"));
            };
            (kind, head, len)
        } else if let Some((head, 1)) = run_of(&mc, 4) {
            let kind = match nk {
                2 => Kind::FourSolo,
                4 => Kind::FourPair,
                _ => return Err(format!("{s:?}: four of a kind carries 2 solos or 2 pairs")),
            };
            (kind, head, 1)
        } else {
            return Err(format!("{s:?}: the part before '-' must be an airplane or four of a kind"));
        };
        let unit = kind.kicker().unwrap();
        let mut ks = Vec::new();
        for r in 0..N_RANKS as u8 {
            let n = kc[r as usize];
            match unit {
                Unit::Solo => ks.extend(std::iter::repeat_n(r, n as usize)),
                Unit::Pair if n == 2 => ks.push(r),
                Unit::Pair if n != 0 => return Err(format!("{s:?}: pair kickers must be distinct pairs")),
                Unit::Pair => {}
            }
        }
        Play::with_kickers(Main { kind, head, len }, &ks)
    } else {
        let c = parse_counts(s)?;
        let ranks: Vec<u8> = (0..N_RANKS as u8).filter(|&r| c[r as usize] > 0).collect();
        match ranks.as_slice() {
            [] => return Err("empty play".into()),
            [r] => {
                let kind = [Kind::Solo, Kind::Pair, Kind::Trio, Kind::Bomb][c[*r as usize] as usize - 1];
                Play::new(kind, *r, 1)
            }
            [BJ, RJ] => Play::rocket(),
            _ => {
                let trio: Vec<u8> = ranks.iter().copied().filter(|&r| c[r as usize] == 3).collect();
                if let Some((head, len)) = (1..=3).find_map(|w| run_of(&c, w)) {
                    let kind = [Kind::SoloChain, Kind::PairChain, Kind::Airplane][c[head as usize] as usize - 1];
                    Play::new(kind, head, len)
                } else if trio.len() == 1 && ranks.len() == 2 {
                    let k = ranks.iter().copied().find(|&r| r != trio[0]).unwrap();
                    let kind = if c[k as usize] == 1 { Kind::TrioSolo } else { Kind::TrioPair };
                    Play::with_kickers(Main { kind, head: trio[0], len: 1 }, &[k])
                } else {
                    return Err(format!("{s:?} is no combination"));
                }
            }
        }
    };
    if !valid_play(&p, Variant::Standard) {
        return Err(format!("{s:?} is no valid combination ({p:?})"));
    }
    Ok(p)
}

/// Debug string of rank counts with explicit zeros skipped (`3x2 5x1 ...`).
pub fn counts_debug(c: &Counts) -> String {
    let mut s = String::new();
    for r in 0..N_RANKS as u8 {
        if c[r as usize] > 0 {
            let _ = write!(s, "{}x{} ", rank_str(r), c[r as usize]);
        }
    }
    s.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use std::collections::HashSet;

    fn hand(s: &str) -> Counts {
        parse_counts(s).unwrap()
    }

    #[test]
    fn universe_matches_openspiel_counts() {
        let os = all_plays(Variant::OpenSpiel);
        // OpenSpiel's play ids 4..=26056: 26053 plays
        assert_eq!(os.len(), 26053);
        let count = |k: Kind, len: Option<u8>| os.iter().filter(|p| p.kind == k && len.map_or(true, |l| p.len == l)).count();
        assert_eq!(count(Kind::TrioSolo, None), 182);
        assert_eq!(count(Kind::TrioPair, None), 156);
        // kicker combinations per airplane head (kNumKickersAirplane*Comb*)
        for (len, n) in [(2u8, 88usize), (3, 330), (4, 816), (5, 1372)] {
            assert_eq!(count(Kind::AirplaneSolo, Some(len)), n * (13 - len as usize), "airplane+solo {len}");
        }
        for (len, n) in [(2u8, 55usize), (3, 120), (4, 126)] {
            assert_eq!(count(Kind::AirplanePair, Some(len)), n * (13 - len as usize), "airplane+pair {len}");
        }
        let std = all_plays(Variant::Standard);
        assert_eq!(std.len(), 26053 + 13 * 102 + 13 * 66);
        let strs: HashSet<String> = std.iter().map(play_str).collect();
        assert_eq!(strs.len(), std.len(), "play strings are unique");
        for p in &std {
            assert_eq!(parse_play(&play_str(p)).unwrap(), *p, "{}", play_str(p));
        }
    }

    #[test]
    fn ids_roundtrip_and_match_openspiel() {
        // OpenSpiel ids of kicker-free plays
        assert_eq!(Action::Play(Play::new(Kind::Solo, 0, 1)).id(), 4);
        assert_eq!(Action::Play(Play::new(Kind::Solo, RJ, 1)).id(), 18);
        assert_eq!(Action::Play(Play::new(Kind::SoloChain, 0, 5)).id(), 19);
        assert_eq!(Action::Play(Play::new(Kind::SoloChain, 0, 12)).id(), 54);
        assert_eq!(Action::Play(Play::new(Kind::PairChain, 0, 10)).id(), 117);
        assert_eq!(Action::Play(Play::new(Kind::PairChain, 2, 10)).id(), 119);
        assert_eq!(Action::Play(Play::new(Kind::Airplane, 6, 6)).id(), 515);
        assert_eq!(Action::Play(Play::new(Kind::Bomb, TWO, 1)).id(), 26055);
        let mut seen = HashSet::new();
        for id in 0..N_ACTION_IDS + 5 {
            match Action::from_id(id) {
                Some(a) => {
                    assert_eq!(a.id(), id);
                    assert!(seen.insert(id));
                }
                None => assert!((133..=470).contains(&id) || (516..=26042).contains(&id) || id >= N_ACTION_IDS, "{id}"),
            }
        }
        assert_eq!(Action::from_id(KICK_PAIR_BASE + 12), Some(Action::KickPair(TWO)));
        assert_eq!(Action::from_id(M_FOUR_PAIR + 12), Some(Action::Main(Main { kind: Kind::FourPair, head: TWO, len: 1 })));
    }

    #[test]
    fn strings_and_parsing() {
        let p = parse_play("333444-5(BWJ)").unwrap();
        assert_eq!((p.kind, p.head, p.len, p.kickers()), (Kind::AirplaneSolo, 0, 2, &[2, BJ][..]));
        assert_eq!(play_str(&parse_play("3444").unwrap()), "3444");
        assert_eq!(parse_play("3444").unwrap().kind, Kind::TrioSolo);
        assert_eq!(parse_play("(BWJ)(CJ)").unwrap(), Play::rocket());
        assert_eq!(parse_play("333444").unwrap().kind, Kind::Airplane);
        assert_eq!(parse_play("444555666-333").unwrap().head, 1);
        assert_eq!(parse_play("7777-4455").unwrap().kind, Kind::FourPair);
        assert_eq!(parse_play("7777-44").unwrap().kind, Kind::FourSolo);
        for bad in ["", "34", "3456", "3344", "JQKA2", "333444-5", "7777-4", "(BWJ)(BWJ)", "33333", "7777-(BWJ)(CJ)",
                    "333444-(BWJ)(CJ)", "333444-5555", "333-4", "3334(BWJ)"] {
            assert!(parse_play(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse_play("22").unwrap().kind, Kind::Pair);
        let m = Main { kind: Kind::AirplanePair, head: 0, len: 2 };
        assert_eq!(partial_str(&m, &[]), "333444-****");
        assert_eq!(partial_str(&m, &[5]), "333444-88**");
        assert_eq!(card_str(0), "C3");
        assert_eq!(card_str(51), "S2");
        assert_eq!(card_str(53), "(CJ)");
    }

    #[test]
    fn kicker_rules() {
        // airplane solos never carry the rocket nor four of a kind
        let h = hand("333444(BWJ)(CJ)");
        let ps = legal_plays(&h, None, Variant::Standard);
        assert!(!ps.iter().any(|p| p.kind == Kind::AirplaneSolo));
        let h = hand("3334445(BWJ)(CJ)");
        let air: Vec<String> = legal_plays(&h, None, Variant::Standard).iter().filter(|p| p.kind == Kind::AirplaneSolo)
            .map(play_str).collect();
        assert_eq!(air, ["333444-5(BWJ)", "333444-5(CJ)"]);
        // 3+3+3 kickers may be a trio (ambiguous 333444555666 = three plays)
        let h = hand("333444555666");
        let ps: Vec<String> = legal_plays(&h, None, Variant::Standard).iter().filter(|p| p.len >= 3 && p.kind.width() == 3)
            .map(play_str).collect();
        for s in ["333444555666", "333444555-666", "444555666-333"] {
            assert!(ps.contains(&s.to_string()), "{s} in {ps:?}");
        }
        // four+2: a pair may be the two solos, the rocket may not; pairs distinct
        let h = hand("77774455(BWJ)(CJ)");
        let four: Vec<String> = legal_plays(&h, None, Variant::Standard).iter()
            .filter(|p| matches!(p.kind, Kind::FourSolo | Kind::FourPair)).map(play_str).collect();
        assert!(four.contains(&"7777-44".to_string()) && four.contains(&"7777-4455".to_string()));
        assert!(four.contains(&"7777-4(CJ)".to_string()) && !four.contains(&"7777-(BWJ)(CJ)".to_string()));
        assert!(legal_plays(&h, None, Variant::OpenSpiel).iter().all(|p| !matches!(p.kind, Kind::FourSolo | Kind::FourPair)));
        // the OpenSpiel chain-head quirk
        let h = hand("33334567");
        let chains = |v| legal_plays(&h, None, v).iter().filter(|p| p.kind == Kind::SoloChain).map(play_str).collect::<Vec<_>>();
        assert_eq!(chains(Variant::Standard), ["34567"]);
        assert!(chains(Variant::OpenSpiel).is_empty());
        // following: same kind and length, higher main; bombs and the rocket always
        let prev = parse_play("34567").unwrap();
        let ps: Vec<String> = legal_plays(&hand("456789TT(BWJ)(CJ)"), Some(&prev), Variant::Standard).iter().map(play_str).collect();
        assert_eq!(ps, ["45678", "56789", "6789T", "(BWJ)(CJ)"]);
        let bomb = parse_play("5555").unwrap();
        let ps: Vec<String> = legal_plays(&hand("44447777(BWJ)(CJ)"), Some(&bomb), Variant::Standard).iter().map(play_str).collect();
        assert_eq!(ps, ["7777", "(BWJ)(CJ)"]);
        assert!(legal_plays(&hand("2222"), Some(&Play::rocket()), Variant::Standard).is_empty());
        let ps: Vec<String> = legal_plays(&hand("22223"), Some(&Play::rocket()), Variant::OpenSpiel).iter().map(play_str).collect();
        assert_eq!(ps, ["2222"], "OpenSpiel: any bomb after the rocket");
    }

    /// The factored decisions (mains, then kicker options) reach exactly the legal full
    /// plays of the definition (universe filtered by `is_legal_play`), on random hands
    /// and previous plays.
    #[test]
    fn factored_decisions_reach_exactly_the_legal_plays() {
        let mut rng = Rng::new(7);
        for variant in [Variant::Standard, Variant::OpenSpiel] {
            let universe = all_plays(variant);
            for trial in 0..400 {
                let mut deck: Vec<u8> = (0..N_CARDS as u8).collect();
                rng.shuffle(&mut deck);
                let n = [20usize, 17, 12, 6, 3][trial % 5];
                let mut h = [0u8; N_RANKS];
                for &c in &deck[..n] {
                    h[card_rank(c) as usize] += 1;
                }
                let prev = if trial % 3 == 0 { None } else { Some(universe[rng.below(universe.len() as u64) as usize]) };
                let want: HashSet<Play> = universe.iter().copied().filter(|p| is_legal_play(&h, prev.as_ref(), variant, p)).collect();
                let got_v = legal_plays(&h, prev.as_ref(), variant);
                let got: HashSet<Play> = got_v.iter().copied().collect();
                assert_eq!(got.len(), got_v.len(), "duplicate paths");
                assert_eq!(got, want, "{variant:?} hand {} prev {:?}", counts_debug(&h), prev.map(|p| play_str(&p)));
            }
        }
    }
}
