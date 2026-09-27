//! Hand-cycle belief.
//!
//! Clash Royale deals 4 of the 8 deck cards into the hand and keeps the other
//! 4 in a queue; a played card's slot is refilled from the queue front and the
//! played card goes to the queue back. The player sees the hand and the next
//! card, the replay records only the plays. What the record says about the hand
//! at a decision is therefore a belief: of the 70 × 24 initial (hand set, queue
//! order) states — equally likely after the shuffle — keep those under which
//! every own play SO FAR was in hand. [`CycleBelief`] maintains it online:
//! [`CycleBelief::observe`] filters and advances the surviving states with one
//! play, [`CycleBelief::in_hand`] / [`CycleBelief::next`] are the fractions of
//! surviving initial states in which each deck card is in hand / at the queue
//! front. A later play never reaches it, so the belief at a frame is a function
//! of the frame's past. (converter_version <= 2 filtered the states with the
//! WHOLE match and picked a canonical hand from them, which marked the card about
//! to be played "certainly in hand" — see `leak_audit.rs`.)
//!
//! Closed form (a check on the enumeration, `tests::matches_the_closed_form`):
//! the constraints are symmetric in the cards not played yet, and a played card
//! needs 4 more plays to come back, so after m ≤ 3 in-deck plays the played cards
//! are out of the hand (P = 0) and every other card has P(in hand) = 4/(8 − m) and
//! P(next) = 1/(8 − m); from the 4th play on the state is exact (queue = the last
//! 4 plays in order, hand = the other 4 cards), so both are 0/1. The enumeration
//! also covers what the closed form does not: decks with repeated ids (unknown
//! cards) and records no cycle explains.
//!
//! `data_processing/games/clash_royale/belief.py` is the same computation in
//! Python for the live bot (bit-identical outputs, pinned by
//! `tests/test_clash_royale_belief.py`).

use std::collections::VecDeque;

/// Cards in a deck (= deck tokens of the frame).
pub const DECK: usize = 8;

/// A concrete hand cycle over card ids: 4 hand slots (a played slot is refilled
/// in place) and the queue, front first. The synthetic matches of the leak audit
/// play from one.
#[derive(Debug, Clone)]
pub struct HandState {
    /// slot -> card id (4 slots; the played slot is refilled in place)
    pub hand: [u16; 4],
    pub queue: VecDeque<u16>,
}

impl HandState {
    pub fn new(hand: [u16; 4], queue: [u16; 4]) -> Self {
        Self { hand, queue: queue.into_iter().collect() }
    }

    pub fn next_card(&self) -> u16 {
        self.queue.front().copied().unwrap_or(0)
    }

    /// Play `card`; returns the slot it occupied, or None if it was not in hand.
    pub fn play(&mut self, card: u16) -> Option<usize> {
        let slot = self.hand.iter().position(|&c| c == card)?;
        let incoming = self.queue.pop_front().unwrap_or(card);
        self.hand[slot] = incoming;
        self.queue.push_back(card);
        Some(slot)
    }
}

/// One cycle state over deck indices: hand membership bitmask + queue (front first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Cycle {
    hand: u8,
    queue: [u8; 4],
}

/// What [`CycleBelief::observe`] did with a play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observed {
    /// an in-deck card: the states were filtered by it and advanced
    Cycled,
    /// a card that is not in the deck (unknown key, mode card): the cycle is unchanged
    NotInDeck,
    /// no surviving state had the card in hand (a record no cycle explains): the
    /// belief is left as it was
    Inconsistent,
}

/// The causal belief over one player's hand cycle (see the module docs).
#[derive(Debug, Clone)]
pub struct CycleBelief {
    /// card id per deck index, in the caller's token order
    deck: Vec<u16>,
    /// distinct current states, weighted by the number of initial states leading to them
    states: Vec<(Cycle, u32)>,
    in_hand: [f32; DECK],
    next: [f32; DECK],
    /// in-deck plays observed
    pub plays: usize,
    /// plays of cards that are not in the deck (skipped)
    pub skipped: usize,
}

impl CycleBelief {
    /// The deal prior over `deck` (8 card ids in the caller's token order: every
    /// probability is reported in that order). None unless there are exactly 8.
    pub fn new(deck: &[u16]) -> Option<Self> {
        if deck.len() != DECK {
            return None;
        }
        let mut states = Vec::with_capacity(1680);
        for hand in 0u16..256 {
            let hand = hand as u8;
            if hand.count_ones() != 4 {
                continue;
            }
            let rest: Vec<u8> = (0..DECK as u8).filter(|&i| hand & (1u8 << i) == 0).collect();
            for queue in permutations4(&rest) {
                states.push((Cycle { hand, queue }, 1u32));
            }
        }
        let mut b = Self { deck: deck.to_vec(), states, in_hand: [0.0; DECK], next: [0.0; DECK], plays: 0, skipped: 0 };
        b.refresh();
        Some(b)
    }

    /// Condition on one own play of `card` (in play order). With repeated ids in the
    /// deck the lowest in-hand index holding the id is the one played.
    pub fn observe(&mut self, card: u16) -> Observed {
        let idx: Vec<u8> = (0..DECK as u8).filter(|&i| self.deck[i as usize] == card).collect();
        if idx.is_empty() {
            self.skipped += 1;
            return Observed::NotInDeck;
        }
        let mut advanced: Vec<(Cycle, u32)> = Vec::with_capacity(self.states.len());
        for &(c, w) in &self.states {
            let Some(&i) = idx.iter().find(|&&i| c.hand & (1u8 << i) != 0) else { continue };
            let q = c.queue;
            let hand = (c.hand & !(1u8 << i)) | (1u8 << q[0]);
            advanced.push((Cycle { hand, queue: [q[1], q[2], q[3], i] }, w));
        }
        if advanced.is_empty() {
            return Observed::Inconsistent;
        }
        advanced.sort_unstable_by_key(|&(c, _)| c);
        let mut merged: Vec<(Cycle, u32)> = Vec::with_capacity(advanced.len());
        for (c, w) in advanced {
            match merged.last_mut() {
                Some((last, lw)) if *last == c => *lw += w,
                _ => merged.push((c, w)),
            }
        }
        self.states = merged;
        self.plays += 1;
        self.refresh();
        Observed::Cycled
    }

    /// The deck index a play of `card` takes, judged from the plays before it: the
    /// index holding the id or — with repeated ids (unknown cards) — the one most
    /// probably played under [`Self::observe`]'s convention (the lowest in-hand
    /// index holding the id), ties to the lower index. For a consistent play it is
    /// never an index the belief rules out of the hand (the first index holding the
    /// id can be: after both copies were in hand, a play moved it to the queue).
    /// None when `card` is not in the deck.
    pub fn attribute(&self, card: u16) -> Option<usize> {
        let idx: Vec<usize> = (0..DECK).filter(|&i| self.deck[i] == card).collect();
        if idx.len() <= 1 {
            return idx.first().copied();
        }
        let mut w = [0u64; DECK];
        for &(c, n) in &self.states {
            if let Some(&i) = idx.iter().find(|&&i| c.hand & (1u8 << i) != 0) {
                w[i] += n as u64;
            }
        }
        // the first maximum (idx ascends); an inconsistent play (all zero) keeps idx[0]
        let mut best = idx[0];
        for &i in &idx[1..] {
            if w[i] > w[best] {
                best = i;
            }
        }
        Some(best)
    }

    fn refresh(&mut self) {
        let total: u64 = self.states.iter().map(|&(_, w)| w as u64).sum();
        let (mut hand, mut next) = ([0u64; DECK], [0u64; DECK]);
        for &(c, w) in &self.states {
            for (i, h) in hand.iter_mut().enumerate() {
                if c.hand & (1u8 << i) != 0 {
                    *h += w as u64;
                }
            }
            next[c.queue[0] as usize] += w as u64;
        }
        // one f64 division, then f32 (belief.py does the same: bit-identical)
        for i in 0..DECK {
            self.in_hand[i] = (hand[i] as f64 / total as f64) as f32;
            self.next[i] = (next[i] as f64 / total as f64) as f32;
        }
    }

    /// P(deck card i is in hand | the plays observed so far), token order; sums to 4.
    pub fn in_hand(&self) -> &[f32; DECK] {
        &self.in_hand
    }

    /// P(deck card i is the next card | the plays observed so far); sums to 1.
    pub fn next(&self) -> &[f32; DECK] {
        &self.next
    }

    /// Initial (hand set, queue order) states consistent with every play observed so far.
    pub fn n_consistent(&self) -> u32 {
        self.states.iter().map(|&(_, w)| w).sum()
    }

    /// Whether the current hand and queue are known exactly.
    pub fn determined(&self) -> bool {
        self.states.len() == 1
    }
}

fn permutations4<T: Copy>(v: &[T]) -> Vec<[T; 4]> {
    let mut out = Vec::with_capacity(24);
    for i in 0..4 {
        for j in 0..4 {
            if j == i {
                continue;
            }
            for k in 0..4 {
                if k == i || k == j {
                    continue;
                }
                let l = 6 - i - j - k;
                out.push([v[i], v[j], v[k], v[l]]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rnd(u64);

    impl Rnd {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// A random deck of ids 1..=8 (shuffled), a random true cycle and a random legal
    /// play sequence from it.
    fn random_game(rnd: &mut Rnd, n_plays: usize) -> (Vec<u16>, HandState, Vec<u16>) {
        let mut deck: Vec<u16> = (1..=8).collect();
        for i in (1..8).rev() {
            deck.swap(i, rnd.below(i + 1));
        }
        let mut order = deck.clone();
        for i in (1..8).rev() {
            order.swap(i, rnd.below(i + 1));
        }
        let truth = HandState::new([order[0], order[1], order[2], order[3]], [order[4], order[5], order[6], order[7]]);
        let mut st = truth.clone();
        let mut plays = Vec::new();
        for _ in 0..n_plays {
            let c = st.hand[rnd.below(4)];
            st.play(c);
            plays.push(c);
        }
        (deck, truth, plays)
    }

    /// Brute force: the fraction of the 1680 initial states that replay `plays` with
    /// card i in hand / next afterwards.
    fn brute_force(deck: &[u16], plays: &[u16]) -> Option<([f64; DECK], [f64; DECK])> {
        let (mut hand, mut next, mut n) = ([0f64; DECK], [0f64; DECK], 0f64);
        for mask in 0u16..256 {
            if mask.count_ones() != 4 {
                continue;
            }
            let h: Vec<u16> = (0..8).filter(|i| mask & (1 << i) != 0).map(|i| deck[i]).collect();
            let rest: Vec<u16> = (0..8).filter(|i| mask & (1 << i) == 0).map(|i| deck[i]).collect();
            for q in permutations4(&rest) {
                let mut st = HandState::new([h[0], h[1], h[2], h[3]], q);
                if plays.iter().all(|&p| st.play(p).is_some()) {
                    n += 1.0;
                    for (i, &c) in deck.iter().enumerate() {
                        hand[i] += st.hand.contains(&c) as u8 as f64;
                        next[i] += (st.next_card() == c) as u8 as f64;
                    }
                }
            }
        }
        (n > 0.0).then(|| (hand.map(|x| x / n), next.map(|x| x / n)))
    }

    #[test]
    fn cycle_mechanics() {
        let mut h = HandState::new([1, 2, 3, 4], [5, 6, 7, 8]);
        assert_eq!(h.play(3), Some(2));
        assert_eq!(h.hand, [1, 2, 5, 4]);
        assert_eq!(h.next_card(), 6);
        assert_eq!(h.play(9), None);
    }

    #[test]
    fn deal_prior_then_the_first_play() {
        let deck: Vec<u16> = vec![11, 12, 13, 14, 15, 16, 17, 18];
        let mut b = CycleBelief::new(&deck).unwrap();
        assert_eq!(b.n_consistent(), 1680);
        assert!(b.in_hand().iter().all(|&p| p == 0.5), "{:?}", b.in_hand());
        assert!(b.next().iter().all(|&p| p == 0.125), "{:?}", b.next());
        assert_eq!(b.observe(13), Observed::Cycled);
        // 840 initial hands held 13; it went to the queue back
        assert_eq!(b.n_consistent(), 840);
        assert_eq!((b.in_hand()[2], b.next()[2]), (0.0, 0.0));
        for i in [0, 1, 3, 4, 5, 6, 7] {
            assert_eq!(b.in_hand()[i], (4.0f64 / 7.0) as f32);
            assert_eq!(b.next()[i], (1.0f64 / 7.0) as f32);
        }
        assert_eq!(b.observe(99), Observed::NotInDeck);
        assert_eq!((b.plays, b.skipped, b.n_consistent()), (1, 1, 840));
        // 13 needs 4 more plays to come back: playing it again is impossible
        let before = b.clone();
        assert_eq!(b.observe(13), Observed::Inconsistent);
        assert_eq!(b.in_hand(), before.in_hand());
        assert_eq!(b.n_consistent(), 840);
        assert!(CycleBelief::new(&deck[..7]).is_none());
    }

    #[test]
    fn matches_brute_force_and_never_excludes_the_truth() {
        let mut rnd = Rnd(0x9E37_79B9_7F4A_7C15);
        for _ in 0..40 {
            let n = 1 + rnd.below(14);
            let (deck, truth, plays) = random_game(&mut rnd, n);
            let mut b = CycleBelief::new(&deck).unwrap();
            let mut st = truth.clone();
            for k in 0..=plays.len() {
                let (bh, bn) = brute_force(&deck, &plays[..k]).expect("the truth is consistent");
                for i in 0..DECK {
                    assert_eq!(b.in_hand()[i], bh[i] as f32, "in_hand {i} after {k} plays of {plays:?}");
                    assert_eq!(b.next()[i], bn[i] as f32, "next {i} after {k} plays of {plays:?}");
                    // the true hand / next card always keep positive probability
                    if st.hand.contains(&deck[i]) {
                        assert!(b.in_hand()[i] > 0.0);
                    }
                    if st.next_card() == deck[i] {
                        assert!(b.next()[i] > 0.0);
                    }
                }
                assert!((b.in_hand().iter().sum::<f32>() - 4.0).abs() < 1e-5);
                assert!((b.next().iter().sum::<f32>() - 1.0).abs() < 1e-6);
                if k < plays.len() {
                    assert_eq!(b.observe(plays[k]), Observed::Cycled);
                    st.play(plays[k]);
                }
            }
        }
    }

    #[test]
    fn matches_the_closed_form() {
        let mut rnd = Rnd(0xC0FF_EE00_1234_5678);
        for _ in 0..60 {
            let n = rnd.below(30);
            let (deck, truth, plays) = random_game(&mut rnd, n);
            let mut b = CycleBelief::new(&deck).unwrap();
            let mut st = truth.clone();
            for k in 0..=plays.len() {
                let played: Vec<u16> = plays[..k].to_vec();
                let m = played.len();
                for (i, &c) in deck.iter().enumerate() {
                    let (h, nx) = if m >= 4 {
                        // exact: the true state
                        ((st.hand.contains(&c) as u8) as f64, ((st.next_card() == c) as u8) as f64)
                    } else if played.contains(&c) {
                        (0.0, 0.0)
                    } else {
                        (4.0 / (8 - m) as f64, 1.0 / (8 - m) as f64)
                    };
                    assert_eq!((b.in_hand()[i], b.next()[i]), (h as f32, nx as f32), "card {c} after {played:?}");
                }
                assert_eq!(b.determined(), m >= 4);
                if k < plays.len() {
                    assert_eq!(b.observe(plays[k]), Observed::Cycled);
                    st.play(plays[k]);
                }
            }
        }
    }

    #[test]
    fn repeated_ids_keep_one_token_per_deck_card() {
        // two unknown cards (same id 99): the enumeration still runs over deck positions
        let deck: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 99, 99];
        let mut b = CycleBelief::new(&deck).unwrap();
        assert_eq!(b.observe(99), Observed::Cycled);
        assert!((b.in_hand().iter().sum::<f32>() - 4.0).abs() < 1e-5);
        assert!((b.next().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(b.observe(99), Observed::Cycled, "the second copy may still be in hand");
        // both copies are now in the queue
        assert_eq!((b.in_hand()[6], b.in_hand()[7]), (0.0, 0.0));
        assert_eq!(b.observe(99), Observed::Inconsistent);
    }

    #[test]
    fn attribute_keeps_repeated_ids_inside_the_support() {
        let deck: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 99, 99];
        let mut b = CycleBelief::new(&deck).unwrap();
        assert_eq!((b.attribute(5), b.attribute(42)), (Some(4), None));
        for c in [1, 2, 3, 4] {
            b.observe(c);
        }
        // hand = {5, 6, 99, 99} exactly: a 99 play takes index 6 by convention ...
        assert_eq!(b.attribute(99), Some(6));
        b.observe(99);
        // ... so index 6 is in the queue and the other copy is the one left to play
        assert_eq!((b.in_hand()[6], b.in_hand()[7]), (0.0, 1.0));
        assert_eq!(b.attribute(99), Some(7), "the first index holding 99 is out of the hand");
        // property: every consistent play is attributed to an index the belief allows
        let mut rnd = Rnd(0xA77E_1B07_E5EE_D5EE);
        let mut n_dup = 0;
        for _ in 0..300 {
            let mut order: Vec<u16> = (0..8).collect();
            for i in (1..8).rev() {
                order.swap(i, rnd.below(i + 1));
            }
            // the true cycle over deck positions; the belief sees card ids only
            let mut st = HandState::new([order[0], order[1], order[2], order[3]], [order[4], order[5], order[6], order[7]]);
            let mut b = CycleBelief::new(&deck).unwrap();
            for _ in 0..24 {
                let pos = st.hand[rnd.below(4)];
                st.play(pos);
                let card = deck[pos as usize];
                let i = b.attribute(card).unwrap();
                assert!(deck[i] == card && b.in_hand()[i] > 0.0, "{card} attributed to index {i}: {:?}", b.in_hand());
                n_dup += (card == 99) as usize;
                assert_eq!(b.observe(card), Observed::Cycled);
            }
        }
        assert!(n_dup > 100);
    }
}
