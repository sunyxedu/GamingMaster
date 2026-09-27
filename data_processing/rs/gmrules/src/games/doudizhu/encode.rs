//! Dou Dizhu decision frames (the candidate schema; every feature is 0/1, stored
//! bit-packed). Only what the acting seat `me` knows: its own hand, the public plays
//! and passes, card counts, the auction, the face-up card's rank, the kitty once it
//! is shown. "Unseen" cards = the deck minus my hand minus every played card (the
//! other two hands, plus the kitty while it is face down). Seats are relative: k = 0
//! me, 1 next (acts after me), 2 previous. Thermometers: bit i = count > i.
//!
//! ```text
//! entities [20, 43]  slots 0..15 ranks 3..2, BJ, RJ (slot = rank):
//!                      0..4 my count, 4..8 unseen count, 8..20 played by me / next /
//!                      prev, 20..23 kitty count (shown), 23..26 kitty cards the
//!                      landlord still holds, 26..30 in the play to beat, 30..38 the
//!                      last play-phase action of next / prev, 38..42 my combination
//!                      under construction (main + kickers so far), 42 face-up rank
//!                    slots 15..18 seats (me, next, prev):
//!                      0..3 relative one-hot, 3..6 landlord / peasant / undecided,
//!                      6 first bidder, 7..28 cards left one-hot (0..20), 28 passed in
//!                      the auction, 29..32 its highest bid 1/2/3, 32 current high
//!                      bidder (auction), 33 owns the play to beat, 34 its last play
//!                      action was a pass, 35..37 plays made > 0 / > 1, 37..40 bombs
//!                      (and rockets) it played > 0 / > 1 / > 2
//!                    slot 18 the play to beat (when following): 0..14 kind, 14..29
//!                      main rank (head), 29..41 length - 1, 41..43 owner next / prev
//!                    slot 19 my combination under construction (kicker decisions):
//!                      0..14 kind, 14..29 head, 29..41 length - 1, 41 a kicker
//!                      chosen, 42 one kicker left
//! globals  [24]      0..3 phase auction / play / kicker, 3..7 my role undecided /
//!                    landlord / peasant after the landlord / peasant before it, 7..11
//!                    winning bid 0..3, 11 free lead, 12..16 bombs played > 0..3,
//!                    16 one pass since the play to beat, 17..19 auction passes > 0 /
//!                    > 1, 19 no peasant has played (spring), 20 the landlord played at
//!                    most once, 21 no bid yet, 22 an unseen bomb or rocket is
//!                    possible, 23 kitty shown
//! actions  [MAX_ACTIONS, 48]  per candidate: 0..5 pass / bid / play / main (kickers
//!                    follow) / kicker, 5..8 bid 1/2/3, 8..22 kind (a kicker: its
//!                    main's), 22..37 ranks covered (a main: its main ranks; a kicker:
//!                    its rank), 37..39 kicker unit solo / pair, 39 bomb or rocket,
//!                    40 breaks a bigger group of a rank, 41..46 my cards left after
//!                    the full play 0 / 1 / 2 / 3-5 / 6+, 46 the kicker completing
//!                    the play, 47 no same-kind play of the unseen cards beats it
//! ```

use super::cards::*;
use super::{DouDizhu, LastAct, Phase, N_SEATS};
use crate::frame::ObsView;
use crate::game::Seat;

pub const N_ENTITIES: usize = 20;
pub const ENTITY_FEATURES: usize = 43;
pub const GLOBAL_FEATURES: usize = 24;
pub const ACTION_FEATURES: usize = 48;
/// Candidates per frame: the most main-level options any hand has when leading
/// (91, for 33344455566677788899 — exhaustive over all 153,009,740 twenty-card hands,
/// `max_main_options_exhaustive` in the tests; options only grow with the hand and a
/// follower's are a subset of a leader's), + 1 for pass — exact, not a cap: no frame
/// is ever truncated. Kicker decisions have at most 15 candidates, the auction 4.
/// (Unfactored, a decision could have up to 521 full plays, for 333444555666789TJQKA.)
pub const MAX_ACTIONS: usize = 92;

const SLOT_SEAT: usize = 15;
const SLOT_TRICK: usize = 18;
const SLOT_PENDING: usize = 19;

// rank entities
const R_MINE: usize = 0;
const R_UNSEEN: usize = 4;
const R_PLAYED: usize = 8;
const R_KITTY: usize = 20;
const R_LLKNOWN: usize = 23;
const R_TRICK: usize = 26;
const R_LAST: usize = 30;
const R_PENDING: usize = 38;
const R_FACEUP: usize = 42;
// seat entities
const S_REL: usize = 0;
const S_ROLE: usize = 3;
const S_FIRST: usize = 6;
const S_LEFT: usize = 7;
const S_APASS: usize = 28;
const S_ABID: usize = 29;
const S_ALEAD: usize = 32;
const S_OWNER: usize = 33;
const S_LASTPASS: usize = 34;
const S_PLAYS: usize = 35;
const S_BOMBS: usize = 37;
// trick / pending entities
const C_KIND: usize = 0;
const C_HEAD: usize = 14;
const C_LEN: usize = 29;
const T_OWNER: usize = 41;
const P_CHOSEN: usize = 41;
const P_LAST: usize = 42;
// globals
const G_PHASE: usize = 0;
const G_ROLE: usize = 3;
const G_BID: usize = 7;
const G_LEAD: usize = 11;
const G_BOMBS: usize = 12;
const G_PASS1: usize = 16;
const G_APASS: usize = 17;
const G_SPRING: usize = 19;
const G_ASPRING: usize = 20;
const G_NOBID: usize = 21;
const G_THREAT: usize = 22;
const G_KITTY: usize = 23;
// candidates
const A_TYPE: usize = 0;
const A_BID: usize = 5;
const A_KIND: usize = 8;
const A_RANKS: usize = 22;
const A_UNIT: usize = 37;
const A_DOUBLE: usize = 39;
const A_SPLIT: usize = 40;
const A_LEFT: usize = 41;
const A_LASTK: usize = 46;
const A_SAFE: usize = 47;

fn thermo(f: &mut [f32], at: usize, n: u8, width: usize) {
    for x in f[at..at + (n as usize).min(width)].iter_mut() {
        *x = 1.0;
    }
}

fn total(r: usize) -> u8 {
    if r >= BJ as usize { 1 } else { 4 }
}

/// Cards neither in `me`'s hand nor played (the others' hands + a face-down kitty).
fn unseen(g: &DouDizhu, me: Seat) -> Counts {
    let mut u = [0u8; N_RANKS];
    for (r, x) in u.iter_mut().enumerate() {
        *x = total(r) - g.hands[me][r] - g.played.iter().map(|p| p[r]).sum::<u8>();
    }
    u
}

/// Can a same-kind play of the `unseen` cards beat (kind, head, len)? (Bombs and the
/// rocket aside; kickers ignored.)
fn beatable(unseen: &Counts, kind: Kind, head: u8, len: u8) -> bool {
    let w = kind.width();
    match kind {
        Kind::Rocket => false,
        Kind::Bomb => (head + 1..BJ).any(|r| unseen[r as usize] == 4) || (unseen[BJ as usize] > 0 && unseen[RJ as usize] > 0),
        Kind::Solo => (head + 1..N_RANKS as u8).any(|r| unseen[r as usize] >= 1),
        k if !k.is_chain() => (head + 1..BJ).any(|r| unseen[r as usize] >= w),
        _ => (head + 1..=(ACE + 1).saturating_sub(len)).any(|h| (h..h + len).all(|r| unseen[r as usize] >= w)),
    }
}

fn combo(f: &mut [f32], kind: Kind, head: u8, len: u8) {
    f[C_KIND + kind.index()] = 1.0;
    f[C_HEAD + head as usize] = 1.0;
    f[C_LEN + len as usize - 1] = 1.0;
}

pub(super) fn obs(g: &DouDizhu, me: Seat, obs: &mut ObsView<'_>) {
    let rel = |k: usize| (me + k) % N_SEATS;
    let un = unseen(g, me);
    let landlord = g.landlord();
    let trick = g.trick.map(|(s, p)| (s as usize, p, p.counts()));
    let last: [Option<Counts>; 2] = [1, 2].map(|k| match g.last_act[rel(k)] {
        Some(LastAct::Play(p)) => Some(p.counts()),
        _ => None,
    });
    let pending = g.pending.map(|pd| {
        let mut c = pd.main.counts();
        let u = pd.main.unit().cards();
        for &k in pd.chosen() {
            c[k as usize] += u;
        }
        (pd, c)
    });
    for r in 0..N_RANKS {
        let f = obs.entity(r, 0);
        thermo(f, R_MINE, g.hands[me][r], 4);
        thermo(f, R_UNSEEN, un[r], 4);
        for k in 0..N_SEATS {
            thermo(f, R_PLAYED + 4 * k, g.played[rel(k)][r], 4);
        }
        if g.kitty_shown {
            thermo(f, R_KITTY, g.kitty[r], 3);
            if let Some(l) = landlord {
                thermo(f, R_LLKNOWN, g.kitty[r].saturating_sub(g.played[l][r]), 3);
            }
        }
        if let Some((_, _, c)) = &trick {
            thermo(f, R_TRICK, c[r], 4);
        }
        for (j, c) in last.iter().enumerate() {
            if let Some(c) = c {
                thermo(f, R_LAST + 4 * j, c[r], 4);
            }
        }
        if let Some((_, c)) = &pending {
            thermo(f, R_PENDING, c[r], 4);
        }
        if g.face_up_rank == Some(r as u8) {
            f[R_FACEUP] = 1.0;
        }
    }
    let auction_phase = g.phase == Phase::Auction;
    for k in 0..N_SEATS {
        let s = rel(k);
        let f = obs.entity(SLOT_SEAT + k, 0);
        f[S_REL + k] = 1.0;
        f[S_ROLE + match landlord {
            Some(l) if l == s => 0,
            Some(_) => 1,
            None => 2,
        }] = 1.0;
        if g.first_bidder as usize == s {
            f[S_FIRST] = 1.0;
        }
        f[S_LEFT + n_cards(&g.hands[s]) as usize] = 1.0;
        let mine = g.auction.iter().filter(|&&(a, _)| a as usize == s);
        let (mut passed, mut best) = (false, 0u8);
        for &(_, b) in mine {
            passed |= b == 0;
            best = best.max(b);
        }
        if passed {
            f[S_APASS] = 1.0;
        }
        if best > 0 {
            f[S_ABID + best as usize - 1] = 1.0;
        }
        if auction_phase && g.winning_bid > 0 && g.landlord == Some(s as u8) {
            f[S_ALEAD] = 1.0;
        }
        if matches!(trick, Some((o, _, _)) if o == s) {
            f[S_OWNER] = 1.0;
        }
        if g.last_act[s] == Some(LastAct::Pass) {
            f[S_LASTPASS] = 1.0;
        }
        thermo(f, S_PLAYS, g.n_plays[s], 2);
        thermo(f, S_BOMBS, g.bombs_by[s], 3);
    }
    if let Some((owner, p, _)) = &trick {
        let f = obs.entity(SLOT_TRICK, 0);
        combo(f, p.kind, p.head, p.len);
        let k = (owner + N_SEATS - me) % N_SEATS;
        if k > 0 {
            f[T_OWNER + k - 1] = 1.0;
        }
    }
    if let Some((pd, _)) = &pending {
        let f = obs.entity(SLOT_PENDING, 0);
        combo(f, pd.main.kind, pd.main.head, pd.main.len);
        f[P_CHOSEN] = (pd.n > 0) as u8 as f32;
        f[P_LAST] = (pd.main.n_kickers() - pd.n == 1) as u8 as f32;
    }
    let gl = obs.globals();
    gl[G_PHASE + if auction_phase { 0 } else if g.pending.is_some() { 2 } else { 1 }] = 1.0;
    gl[G_ROLE + match landlord {
        None => 0,
        Some(l) if l == me => 1,
        Some(l) if (l + 1) % N_SEATS == me => 2,
        Some(_) => 3,
    }] = 1.0;
    gl[G_BID + g.winning_bid as usize] = 1.0;
    let play_phase = g.phase == Phase::Play;
    if play_phase && g.pending.is_none() && g.trick.is_none() {
        gl[G_LEAD] = 1.0;
    }
    thermo(gl, G_BOMBS, g.bombs, 4);
    if play_phase && g.trick.is_some() && g.passes == 1 {
        gl[G_PASS1] = 1.0;
    }
    if auction_phase {
        thermo(gl, G_APASS, g.passes, 2);
        if g.winning_bid == 0 {
            gl[G_NOBID] = 1.0;
        }
    }
    if let (true, Some(l)) = (play_phase, landlord) {
        if g.n_plays[(l + 1) % N_SEATS] == 0 && g.n_plays[(l + 2) % N_SEATS] == 0 {
            gl[G_SPRING] = 1.0;
        }
        if g.n_plays[l] <= 1 {
            gl[G_ASPRING] = 1.0;
        }
    }
    if (0..BJ as usize).any(|r| un[r] == 4) || (un[BJ as usize] > 0 && un[RJ as usize] > 0) {
        gl[G_THREAT] = 1.0;
    }
    if g.kitty_shown {
        gl[G_KITTY] = 1.0;
    }
}

fn left_bucket(f: &mut [f32], left: i32) {
    let b = match left {
        i32::MIN..=0 => 0,
        1 => 1,
        2 => 2,
        3..=5 => 3,
        _ => 4,
    };
    f[A_LEFT + b] = 1.0;
}

pub(super) fn action(g: &DouDizhu, me: Seat, action: u32, f: &mut [f32]) {
    let hand = &g.hands[me];
    let held = n_cards(hand) as i32;
    if g.phase == Phase::Auction {
        if action == PASS {
            f[A_TYPE] = 1.0;
        } else {
            f[A_TYPE + 1] = 1.0;
            f[A_BID + action as usize - 1] = 1.0;
        }
        return;
    }
    let un = unseen(g, me);
    match Action::from_id(action) {
        Some(Action::Pass) => f[A_TYPE] = 1.0,
        Some(Action::Play(p)) => {
            f[A_TYPE + 2] = 1.0;
            f[A_KIND + p.kind.index()] = 1.0;
            let c = p.counts();
            for r in 0..N_RANKS {
                if c[r] > 0 {
                    f[A_RANKS + r] = 1.0;
                    if hand[r] > c[r] {
                        f[A_SPLIT] = 1.0;
                    }
                }
            }
            if p.kind.doubles() {
                f[A_DOUBLE] = 1.0;
            }
            left_bucket(f, held - p.n_cards() as i32);
            if !beatable(&un, p.kind, p.head, p.len) {
                f[A_SAFE] = 1.0;
            }
        }
        Some(Action::Main(m)) => {
            f[A_TYPE + 3] = 1.0;
            f[A_KIND + m.kind.index()] = 1.0;
            for r in m.head..m.head + m.len {
                f[A_RANKS + r as usize] = 1.0;
                if hand[r as usize] > m.kind.width() {
                    f[A_SPLIT] = 1.0;
                }
            }
            let u = m.unit();
            f[A_UNIT + (u == Unit::Pair) as usize] = 1.0;
            let cards = n_cards(&m.counts()) as i32 + m.n_kickers() as i32 * u.cards() as i32;
            left_bucket(f, held - cards);
            if !beatable(&un, m.kind, m.head, m.len) {
                f[A_SAFE] = 1.0;
            }
        }
        Some(Action::KickSolo(r)) | Some(Action::KickPair(r)) => {
            f[A_TYPE + 4] = 1.0;
            let Some(pd) = g.pending else { return };
            let m = pd.main;
            f[A_KIND + m.kind.index()] = 1.0;
            f[A_RANKS + r as usize] = 1.0;
            let u = m.unit();
            f[A_UNIT + (u == Unit::Pair) as usize] = 1.0;
            let taken = pd.chosen().iter().filter(|&&k| k == r).count() as u8 * u.cards();
            if hand[r as usize] > taken + u.cards() {
                f[A_SPLIT] = 1.0;
            }
            let cards = n_cards(&m.counts()) as i32 + m.n_kickers() as i32 * u.cards() as i32;
            left_bucket(f, held - cards);
            if pd.n + 1 == m.n_kickers() {
                f[A_LASTK] = 1.0;
            }
            if !beatable(&un, m.kind, m.head, m.len) {
                f[A_SAFE] = 1.0;
            }
        }
        _ => {}
    }
}
