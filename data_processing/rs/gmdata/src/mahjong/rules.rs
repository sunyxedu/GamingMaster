//! Tiles, hand shapes and the hora (win) rules of Tenhou's four-player riichi mahjong.
//!
//! Shapes and waits are computed here; whether a winning shape has a yaku (the one-han
//! minimum, yakuman included) is decided by the hand evaluator of `riichienv-core`
//! (RiichiEnv, Apache-2.0), the reference implementation this converter's legality is
//! also differential-tested against (`reference.rs`).

use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::types::{Conditions, Meld as RMeld, MeldType, Wind};

pub(super) const NK: usize = 34;

/// A malformed or unsupported record: the game is skipped (counted, never half-converted).
pub(super) struct Skip;
pub(super) type R<T> = std::result::Result<T, Skip>;

/// mjai tile name ("1m".."9s", "5mr" / "0m" red fives, "E" "S" "W" "N" "P" "F" "C") → kind 0..33.
pub(super) fn tile_kind(p: &str) -> R<usize> {
    let b = p.as_bytes();
    let honors = ["E", "S", "W", "N", "P", "F", "C"];
    if let Some(i) = honors.iter().position(|&h| h == p) {
        return Ok(27 + i);
    }
    let suit = |c: u8| match c {
        b'm' => Some(0),
        b'p' => Some(1),
        b's' => Some(2),
        _ => None,
    };
    match b.len() {
        2 => {
            let s = suit(b[1]).ok_or(Skip)?;
            match b[0] {
                b'1'..=b'9' => Ok(s * 9 + (b[0] - b'1') as usize),
                b'0' => Ok(s * 9 + 4),
                _ => Err(Skip),
            }
        }
        3 if b[0] == b'5' && b[2] == b'r' => Ok(suit(b[1]).ok_or(Skip)? * 9 + 4),
        _ => Err(Skip),
    }
}

pub(super) fn is_red(p: &str) -> bool {
    p.ends_with('r') || p.starts_with('0')
}

pub(super) fn dora_from_marker(k: usize) -> usize {
    if k < 27 {
        let base = (k / 9) * 9;
        base + (k - base + 1) % 9
    } else if k < 31 {
        27 + (k - 27 + 1) % 4
    } else {
        31 + (k - 31 + 1) % 3
    }
}

pub(super) fn is_terminal_or_honor(k: usize) -> bool {
    k >= 27 || k.is_multiple_of(9) || k % 9 == 8
}

/// Remove sets (triplets, runs inside one suit) from `c` until empty: the lowest
/// remaining kind must start a triplet or a run, so the search only branches there.
fn only_sets(c: &mut [i64; NK]) -> bool {
    let Some(i) = c.iter().position(|&x| x > 0) else { return true };
    if c[i] >= 3 {
        c[i] -= 3;
        let ok = only_sets(c);
        c[i] += 3;
        if ok {
            return true;
        }
    }
    if i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0 {
        for d in 0..3 {
            c[i + d] -= 1;
        }
        let ok = only_sets(c);
        for d in 0..3 {
            c[i + d] += 1;
        }
        if ok {
            return true;
        }
    }
    false
}

/// `c` = `sets` sets + one pair.
fn regular_agari(c: &[i64; NK], sets: i64) -> bool {
    if sets < 0 || c.iter().sum::<i64>() != 3 * sets + 2 {
        return false;
    }
    let mut w = *c;
    for p in 0..NK {
        if w[p] >= 2 {
            w[p] -= 2;
            if only_sets(&mut w) {
                return true;
            }
            w[p] += 2;
        }
    }
    false
}

/// Seven distinct pairs (a quad is not two pairs).
fn chiitoitsu(c: &[i64; NK]) -> bool {
    c.iter().all(|&x| x == 0 || x == 2) && c.iter().filter(|&&x| x == 2).count() == 7
}

/// All 13 terminal/honor kinds, one of them paired, nothing else.
fn kokushi(c: &[i64; NK]) -> bool {
    (0..NK).all(|k| if is_terminal_or_honor(k) { c[k] >= 1 } else { c[k] == 0 }) && c.iter().sum::<i64>() == 14
}

/// Winning shape of the concealed tiles `c` next to `meld_n` fixed melds (yaku not checked).
pub(super) fn is_agari(c: &[i64; NK], meld_n: i64) -> bool {
    if c.iter().any(|&x| x < 0) || c.iter().sum::<i64>() != 14 - 3 * meld_n {
        return false;
    }
    regular_agari(c, 4 - meld_n) || (meld_n == 0 && (chiitoitsu(c) || kokushi(c)))
}

/// Tenhou's waits of the 13-tile-equivalent concealed hand `c`: the kinds that complete a
/// winning shape, except kinds the concealed hand already holds four of (形式聴牌: a wait
/// on a fifth copy counts only while the concealed tiles do not use all four; the same set
/// RiichiEnv's `get_waits` returns).  Furiten and riichi tenpai are judged on this set.
pub(super) fn waits(c: &[i64; NK], meld_n: i64) -> [bool; NK] {
    let mut w = *c;
    std::array::from_fn(|k| {
        if w[k] >= 4 {
            return false;
        }
        w[k] += 1;
        let ok = is_agari(&w, meld_n);
        w[k] -= 1;
        ok
    })
}

pub(super) fn is_tenpai(c: &[i64; NK], meld_n: i64) -> bool {
    waits(c, meld_n).iter().any(|&x| x)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MeldKind {
    Chi,
    Pon,
    Daiminkan,
    Ankan,
    Kakan,
}

/// A fixed meld: its kind and tile kinds (3, or 4 for a kan).
#[derive(Clone, Debug)]
pub(super) struct Meld {
    pub kind: MeldKind,
    pub tiles: Vec<usize>,
}

/// Situation of a win declaration (everything the one-han minimum can depend on besides
/// the tiles).  Dora never count as yaku, so no dora information is needed.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WinCtx {
    pub tsumo: bool,
    pub riichi: bool,
    /// tsumo on the last live-wall draw (not a rinshan draw)
    pub haitei: bool,
    /// ron on the last discard
    pub houtei: bool,
    /// tsumo on a rinshan draw
    pub rinshan: bool,
    /// ron on a kakan tile
    pub chankan: bool,
    /// tsumo on the seat's first draw with no call (ankan included) made yet: tenhou / chiihou
    pub first_draw: bool,
    /// 0..3 = East..North
    pub seat_wind: u8,
    pub round_wind: u8,
}

/// Whether the winning shape `concealed` (13-tile equivalent, the winning tile excluded) +
/// `win` next to `melds` has at least one yaku under `ctx`.  Tile identities (which copy,
/// red or not) cannot change the answer: red fives are dora, not yaku.
pub(super) fn has_yaku(concealed: &[i64; NK], melds: &[Meld], win: usize, ctx: &WinCtx) -> bool {
    // a distinct 136-format id per physical copy; ids k*4 are the red fives, used last
    let mut used = [0usize; NK];
    let mut id = |k: usize| -> Option<u8> {
        let j = *used.get(k)?;
        if j >= 4 {
            return None;
        }
        used[k] += 1;
        Some((k * 4 + [1, 2, 3, 0][j]) as u8)
    };
    let mut tiles = Vec::with_capacity(14);
    for (k, &n) in concealed.iter().enumerate() {
        for _ in 0..n.max(0) {
            match id(k) {
                Some(t) => tiles.push(t),
                None => return false,
            }
        }
    }
    let mut ms = Vec::with_capacity(melds.len());
    for m in melds {
        let mut ts = Vec::with_capacity(m.tiles.len());
        for &k in &m.tiles {
            match id(k) {
                Some(t) => ts.push(t),
                None => return false,
            }
        }
        let (mt, opened) = match m.kind {
            MeldKind::Chi => (MeldType::Chi, true),
            MeldKind::Pon => (MeldType::Pon, true),
            MeldKind::Daiminkan => (MeldType::Daiminkan, true),
            MeldKind::Ankan => (MeldType::Ankan, false),
            MeldKind::Kakan => (MeldType::Kakan, true),
        };
        ms.push(RMeld::new(mt, ts, opened, -1, None));
    }
    let Some(w) = id(win) else { return false };
    let cond = Conditions {
        tsumo: ctx.tsumo,
        riichi: ctx.riichi,
        haitei: ctx.haitei,
        houtei: ctx.houtei,
        rinshan: ctx.rinshan,
        chankan: ctx.chankan,
        tsumo_first_turn: ctx.first_draw,
        player_wind: Wind::from(ctx.seat_wind),
        round_wind: Wind::from(ctx.round_wind),
        ..Default::default()
    };
    HandEvaluator::new(tiles, ms).calc(w, vec![], vec![], Some(cond)).is_win
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// "123m456p789s1122z" → kind counts (z: 1..7 = E S W N P F C).
    pub(crate) fn hand(s: &str) -> [i64; NK] {
        let mut c = [0i64; NK];
        let mut digits: Vec<usize> = Vec::new();
        for ch in s.chars() {
            match ch {
                '0'..='9' => digits.push(ch.to_digit(10).unwrap() as usize),
                'm' | 'p' | 's' | 'z' => {
                    let base = match ch {
                        'm' => 0,
                        'p' => 9,
                        's' => 18,
                        _ => 27,
                    };
                    for d in digits.drain(..) {
                        c[base + if d == 0 { 4 } else { d - 1 }] += 1;
                    }
                }
                _ => panic!("bad hand {s}"),
            }
        }
        c
    }

    fn wait_list(c: &[i64; NK], meld_n: i64) -> Vec<usize> {
        let w = waits(c, meld_n);
        (0..NK).filter(|&k| w[k]).collect()
    }

    #[test]
    fn agari_shapes() {
        assert!(is_agari(&hand("123m456p789s11122z"), 0));
        assert!(is_agari(&hand("11122233344455m"), 0)); // needs the triplet/run backtrack
        assert!(is_agari(&hand("11123456789999m"), 0)); // chuuren shape
        assert!(!is_agari(&hand("123m456p789s11123z"), 0));
        // chiitoitsu: seven distinct pairs; a quad is not two pairs
        assert!(is_agari(&hand("1199m2288p3377s11z"), 0));
        assert!(!is_agari(&hand("1111m2288p3377s11z"), 0));
        // kokushi (closed only)
        assert!(is_agari(&hand("19m19p19s12345677z"), 0));
        assert!(!is_agari(&hand("19m19p19s12345566z"), 0)); // no chun
        // open melds: 2 melds → 8 concealed tiles = 2 sets + pair; no chiitoitsu with melds
        assert!(is_agari(&hand("123m456p11z"), 2));
        assert!(!is_agari(&hand("123m456p11z"), 0));
        assert!(is_agari(&hand("55z"), 4));
        assert!(!is_agari(&hand("99m2288p3377s11z"), 1));
        // honors never form runs
        assert!(!is_agari(&hand("123z456p789s11122z"), 0));
        // wrong tile count for the meld count
        assert!(!is_agari(&hand("123m456p789s11122z"), 1));
    }

    #[test]
    fn tenpai_waits() {
        // ryanmen 23m → 1m / 4m
        assert_eq!(wait_list(&hand("23m456p789s11122z"), 0), vec![0, 3]);
        // shanpon EE SS → E / S; EEE + S → S tanki only
        assert_eq!(wait_list(&hand("123m456p789s1122z"), 0), vec![27, 28]);
        assert_eq!(wait_list(&hand("123m456p789s1112z"), 0), vec![28]);
        // nobetan 1234m → 1m / 4m
        assert_eq!(wait_list(&hand("1234m456p789s111z"), 0), vec![0, 3]);
        // chuuren: all nine kinds of the suit
        assert_eq!(wait_list(&hand("1112345678999m"), 0), (0..9).collect::<Vec<_>>());
        // chiitoitsu tanki
        assert_eq!(wait_list(&hand("1199m2288p3377s1z"), 0), vec![27]);
        // kokushi 13-sided
        let all13: Vec<usize> = (0..NK).filter(|&k| is_terminal_or_honor(k)).collect();
        assert_eq!(wait_list(&hand("19m19p19s1234567z"), 0), all13);
        // 4 melds: a tanki on the last concealed tile
        assert_eq!(wait_list(&hand("5z"), 4), vec![31]);
        assert!(!is_tenpai(&hand("13579m2468p1357s"), 0));
        // a fifth copy is no wait while the concealed hand holds all four (純カラ):
        // 1111m + three sets waits on 1m only as a fifth tile → not tenpai
        assert_eq!(wait_list(&hand("1111m456p789s111z"), 0), Vec::<usize>::new());
        assert!(!is_tenpai(&hand("1111m456p789s111z"), 0));
        // ... but with one of the four in a meld it is (5枚目を待つ聴牌あり)
        assert_eq!(wait_list(&hand("1m"), 4), vec![0]);
    }

    fn ctx() -> WinCtx {
        WinCtx { seat_wind: 1, round_wind: 0, ..Default::default() }
    }

    #[test]
    fn yaku_decides_open_and_closed_wins() {
        let chi = |a: usize| Meld { kind: MeldKind::Chi, tiles: vec![a, a + 1, a + 2] };
        let pon = |a: usize| Meld { kind: MeldKind::Pon, tiles: vec![a, a, a] };
        // closed 13m 456p 678s 111p 99p waiting on 2m (kanchan, a triplet, terminals and
        // non-terminal runs): no pinfu, tanyao, yakuhai or chanta
        let h = hand("13m456p678s11199p");
        assert!(!has_yaku(&h, &[], 1, &ctx())); // ron 2m: kanchan, no pinfu / tanyao / yakuhai
        // the same hand by tsumo: menzen tsumo
        assert!(has_yaku(&h, &[], 1, &WinCtx { tsumo: true, ..ctx() }));
        // in riichi: riichi
        assert!(has_yaku(&h, &[], 1, &WinCtx { riichi: true, ..ctx() }));
        // houtei makes the yakuless ron legal, haitei the tsumo of an open hand
        assert!(has_yaku(&h, &[], 1, &WinCtx { houtei: true, ..ctx() }));
        // open hand: chi 456p, 13m 678s 111s 99p waiting 2m: no yaku by ron
        let h2 = hand("13m678s11199p");
        assert!(!has_yaku(&h2, &[chi(12)], 1, &ctx()));
        assert!(!has_yaku(&h2, &[chi(12)], 1, &WinCtx { tsumo: true, ..ctx() })); // open: no menzen tsumo
        assert!(has_yaku(&h2, &[chi(12)], 1, &WinCtx { tsumo: true, haitei: true, ..ctx() }));
        assert!(has_yaku(&h2, &[chi(12)], 1, &WinCtx { tsumo: true, rinshan: true, ..ctx() }));
        assert!(has_yaku(&h2, &[chi(12)], 1, &WinCtx { chankan: true, ..ctx() }));
        // kuitan (open tanyao) is allowed on Tenhou: chi 456s, 234m 678s 22p 88p + 8p
        let h3 = hand("234m678s2288p");
        assert!(has_yaku(&h3, &[chi(21)], 16, &ctx()));
        // yakuhai: a pon of haku (P = 31), the seat wind (S for seat 1) or the round wind (E)
        let h4 = hand("123m456p1188s");
        assert!(has_yaku(&h4, &[pon(31)], 25, &ctx()));
        assert!(has_yaku(&h4, &[pon(28)], 25, &ctx())); // seat wind South
        assert!(has_yaku(&h4, &[pon(27)], 25, &ctx())); // round wind East
        assert!(!has_yaku(&h4, &[pon(29)], 25, &ctx())); // West: guest wind, no yaku
        // chiitoitsu and kokushi are yaku on their own
        assert!(has_yaku(&hand("1199m2288p3377s1z"), &[], 27, &ctx()));
        assert!(has_yaku(&hand("19m19p19s1234567z"), &[], 0, &ctx()));
        // an ankan keeps the hand closed (menzen tsumo), a kakan does not
        let ankan = Meld { kind: MeldKind::Ankan, tiles: vec![29; 4] };
        let kakan = Meld { kind: MeldKind::Kakan, tiles: vec![29; 4] };
        let h5 = hand("13m678s11199p");
        assert!(has_yaku(&h5, &[ankan], 1, &WinCtx { tsumo: true, ..ctx() }));
        assert!(!has_yaku(&h5, &[kakan], 1, &WinCtx { tsumo: true, ..ctx() }));
    }
}
