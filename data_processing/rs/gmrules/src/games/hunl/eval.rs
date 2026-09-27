//! Poker hand evaluation: the value of the best five-card hand among 5-7 cards
//! (Texas hold'em showdown: two hole cards + five board cards), and small card helpers.
//!
//! Cards are ids `0..52` in the ACPC / OpenSpiel `universal_poker` encoding:
//! `id = rank * 4 + suit`, ranks `0..13` = `2 3 4 5 6 7 8 9 T J Q K A`, suits `0..4` =
//! `c d h s` (so `2c` = 0, `As` = 51).
//!
//! [`eval`] returns a `u32` that orders hands exactly as poker does (higher = better,
//! equal = a split pot): `category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5`
//! with the category ([`CATEGORY_NAMES`]) and the ranks that break ties within it, most
//! significant first (e.g. a full house: trips rank, pair rank; two pair: high pair, low
//! pair, kicker; a straight: its top card, the wheel A-2-3-4-5 counting as 5-high).

pub const RANK_CHARS: &[u8; 13] = b"23456789TJQKA";
pub const SUIT_CHARS: &[u8; 4] = b"cdhs";

pub const HIGH_CARD: u32 = 0;
pub const PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;

/// Names of the hand categories (index = the category of [`eval`]).
pub const CATEGORY_NAMES: [&str; 9] =
    ["high card", "pair", "two pair", "three of a kind", "straight", "flush", "full house", "four of a kind", "straight flush"];

#[inline]
pub fn rank_of(card: u8) -> u8 {
    card / 4
}

#[inline]
pub fn suit_of(card: u8) -> u8 {
    card % 4
}

/// `"Ac"` style name of a card id.
pub fn card_str(card: u8) -> String {
    if card >= 52 {
        return "??".into();
    }
    format!("{}{}", RANK_CHARS[rank_of(card) as usize] as char, SUIT_CHARS[suit_of(card) as usize] as char)
}

/// Card id of a two-character name (`"Ac"`, `"td"`; rank case-insensitive, suit lower
/// or upper case).
pub fn parse_card(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let r = RANK_CHARS.iter().position(|&c| c == b[0].to_ascii_uppercase())?;
    let su = SUIT_CHARS.iter().position(|&c| c == b[1].to_ascii_lowercase())?;
    Some((r * 4 + su) as u8)
}

/// Parse concatenated card names (`"AcKd"` → `[51-3, ...]`); None on a malformed string.
pub fn parse_cards(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !s.is_ascii() {
        return None;
    }
    (0..s.len() / 2).map(|i| parse_card(&s[2 * i..2 * i + 2])).collect()
}

/// The category of an [`eval`] value.
#[inline]
pub fn category(value: u32) -> u32 {
    value >> 20
}

fn pack(cat: u32, ranks: &[u8]) -> u32 {
    let mut v = cat << 20;
    for (i, &r) in ranks.iter().enumerate() {
        v |= (r as u32) << (16 - 4 * i);
    }
    v
}

/// Top card rank of the highest straight in a 13-bit rank mask (the wheel = 3, i.e.
/// 5-high), or None.
pub fn straight_top(mask: u16) -> Option<u8> {
    let m = mask as u32;
    // bit i + 1 = rank i; bit 0 = the ace playing low
    let ext = (m << 1) | ((m >> 12) & 1);
    (4..=13u32).rev().find(|&top| ext & (0b11111 << (top - 4)) == 0b11111 << (top - 4)).map(|top| (top - 1) as u8)
}

/// The `n` highest ranks set in a rank mask, descending.
fn top_ranks(mask: u16, n: usize) -> Vec<u8> {
    (0..13u8).rev().filter(|&r| mask & (1 << r) != 0).take(n).collect()
}

/// Value of the best five-card hand among `cards` (5..=7 distinct card ids; see the
/// module docs for the ordering).
pub fn eval(cards: &[u8]) -> u32 {
    debug_assert!((5..=7).contains(&cards.len()), "eval takes 5-7 cards, got {}", cards.len());
    let mut counts = [0u8; 13];
    let mut suits = [0u16; 4];
    for &c in cards {
        counts[rank_of(c) as usize] += 1;
        suits[suit_of(c) as usize] |= 1 << rank_of(c);
    }
    let ranks = suits.iter().fold(0u16, |a, &m| a | m);
    // with at most 7 cards a flush excludes quads and full houses (they would need 8)
    if let Some(&sm) = suits.iter().find(|m| m.count_ones() >= 5) {
        return match straight_top(sm) {
            Some(top) => pack(STRAIGHT_FLUSH, &[top]),
            None => pack(FLUSH, &top_ranks(sm, 5)),
        };
    }
    let (mut quad, mut trips, mut pairs) = (None, Vec::new(), Vec::new());
    for r in (0..13u8).rev() {
        match counts[r as usize] {
            4 => quad = Some(r),
            3 => trips.push(r),
            2 => pairs.push(r),
            _ => {}
        }
    }
    let kickers = |exclude: &[u8], n: usize| top_ranks(exclude.iter().fold(ranks, |m, &r| m & !(1 << r)), n);
    if let Some(q) = quad {
        return pack(QUADS, &[q, kickers(&[q], 1)[0]]);
    }
    if let Some(&t) = trips.first() {
        // the pair of a full house: a second trips or the best pair
        let p = trips.get(1).copied().into_iter().chain(pairs.first().copied()).max();
        if let Some(p) = p {
            return pack(FULL_HOUSE, &[t, p]);
        }
    }
    if let Some(top) = straight_top(ranks) {
        return pack(STRAIGHT, &[top]);
    }
    if let Some(&t) = trips.first() {
        let k = kickers(&[t], 2);
        return pack(TRIPS, &[t, k[0], k[1]]);
    }
    if pairs.len() >= 2 {
        let (p1, p2) = (pairs[0], pairs[1]);
        return pack(TWO_PAIR, &[p1, p2, kickers(&[p1, p2], 1)[0]]);
    }
    if let Some(&p) = pairs.first() {
        let k = kickers(&[p], 3);
        return pack(PAIR, &[p, k[0], k[1], k[2]]);
    }
    pack(HIGH_CARD, &top_ranks(ranks, 5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn v(s: &str) -> u32 {
        eval(&parse_cards(s).unwrap())
    }

    /// An independent five-card classifier (sorting by (count, rank)), packed like
    /// `eval` — the best of the 21 five-card subsets must equal `eval` of the seven.
    fn naive5(c: &[u8]) -> u32 {
        let mut rc: Vec<(u8, u8)> = (0..13u8).map(|r| (c.iter().filter(|&&x| rank_of(x) == r).count() as u8, r)).filter(|p| p.0 > 0).collect();
        rc.sort_by(|a, b| b.cmp(a));
        let flush = c.iter().all(|&x| suit_of(x) == suit_of(c[0]));
        let mut rs: Vec<u8> = c.iter().map(|&x| rank_of(x)).collect();
        rs.sort_unstable_by(|a, b| b.cmp(a));
        let distinct = rc.len() == 5;
        let straight = if distinct && rs[0] - rs[4] == 4 {
            Some(rs[0])
        } else if distinct && rs == [12, 3, 2, 1, 0] {
            Some(3)
        } else {
            None
        };
        let order: Vec<u8> = rc.iter().map(|p| p.1).collect();
        let shape: Vec<u8> = rc.iter().map(|p| p.0).collect();
        let (cat, key) = match (straight, flush, shape.as_slice()) {
            (Some(t), true, _) => (8, vec![t]),
            (_, _, [4, 1]) => (7, order),
            (_, _, [3, 2]) => (6, order),
            (None, true, _) => (5, rs),
            (Some(t), false, _) => (4, vec![t]),
            (_, _, [3, 1, 1]) => (3, order),
            (_, _, [2, 2, 1]) => (2, order),
            (_, _, [2, 1, 1, 1]) => (1, order),
            _ => (0, rs),
        };
        pack(cat, &key)
    }

    #[test]
    fn known_hands_order() {
        let hands = [
            "2c3d4h5s7c9d8h", // high card 9
            "AcAd4h5s7c9dTh", // pair
            "AcAdKhKs7c9d2h", // two pair
            "7c7d7hKs2c9d3h", // trips
            "Ac2d3h4s5c9dKh", // wheel
            "6c2d3h4s5c9dKh", // 6-high straight
            "Ac2c7c4c5c9dKh", // flush
            "7c7d7hKsKc9d2h", // full house
            "7c7d7h7sKc9d2h", // quads
            "Ac2c3c4c5c9dKh", // steel wheel
            "TsJsQsKsAs2d3h", // royal flush
        ];
        let vals: Vec<u32> = hands.iter().map(|h| v(h)).collect();
        assert!(vals.windows(2).all(|w| w[0] < w[1]), "{vals:?}");
        assert_eq!(vals.iter().map(|&x| category(x)).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8]);
        // kickers and splits
        assert!(v("AcAd4h5s7c9dKh") > v("AcAd4h5s7c9dQh"));
        assert_eq!(v("AcAd4h5s7c9dKh"), v("AhAs4d5c7d9sKc"));
        assert_eq!(v("2c2d3h3s4c4dAh"), v("4c4d3h3sAd2s2h"), "three pairs: the best two + kicker");
        assert!(v("2c2d3h3s4c4dAh") > v("2c2d3h3s4c4dKh"));
        assert_eq!(v("7c7d7hKsKcKd2h") >> 8, (FULL_HOUSE << 12) | (11 << 8) | (5 << 4), "two trips: K full of 7");
        assert!(v("7c7d7h7sKcAd2h") > v("7c7d7h7sKcQd2h"), "quads kicker");
        assert_eq!(card_str(0), "2c");
        assert_eq!(card_str(51), "As");
        assert_eq!(parse_card("tD"), Some(8 * 4 + 1));
        assert!(parse_cards("AcK").is_none() && parse_cards("Ax").is_none());
    }

    #[test]
    fn seven_cards_equal_the_best_five_card_subset() {
        let mut rng = Rng::new(7);
        let mut deck: Vec<u8> = (0..52).collect();
        let mut cats = [0usize; 9];
        for n in [5usize, 6, 7] {
            for _ in 0..if n == 7 { 20000 } else { 3000 } {
                rng.shuffle(&mut deck);
                let c = &deck[..n];
                let mut best = 0;
                for mask in 0u32..(1 << n) {
                    if mask.count_ones() == 5 {
                        let sub: Vec<u8> = (0..n).filter(|&i| mask & (1 << i) != 0).map(|i| c[i]).collect();
                        best = best.max(naive5(&sub));
                    }
                }
                let e = eval(c);
                assert_eq!(e, best, "{:?}", c.iter().map(|&x| card_str(x)).collect::<Vec<_>>());
                cats[category(e) as usize] += 1;
            }
        }
        assert!(cats.iter().take(8).all(|&k| k > 0), "{cats:?}");
    }
}
