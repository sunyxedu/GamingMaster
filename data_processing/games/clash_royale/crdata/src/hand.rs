//! Hand-cycle reconstruction.
//!
//! Clash Royale deals 4 of the 8 deck cards into the hand and keeps the other
//! 4 in a queue; a played card's slot is refilled from the queue front and the
//! played card goes to the queue back. The replay never shows the hand, but
//! the play sequence constrains it: enumerate the 70 × 24 possible initial
//! (hand set, queue order) states and keep the ones under which every recorded
//! play was in hand. Several may be consistent (usually because a card was
//! never played); we pick a canonical one and record how many there were.

use std::collections::VecDeque;

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

#[derive(Debug, Clone)]
pub struct Inference {
    /// canonical initial state: the consistent state whose hands agree most with the ensemble
    pub initial: HandState,
    /// number of (hand set, queue order) states consistent with the plays
    pub n_consistent: usize,
    /// plays that were not in the deck at all (mirror, unknown cards) and were skipped
    pub skipped_plays: usize,
    /// deck ids in the order used by `in_hand_prob`
    pub deck: Vec<u16>,
    /// `[plays_so_far][deck index]` = fraction of consistent states in which that card is in hand
    /// after `plays_so_far` in-deck plays (rows 0..=n_in_deck_plays)
    pub in_hand_prob: Vec<[f32; 8]>,
}

impl Inference {
    /// probability that `card` is in hand after `plays_so_far` in-deck plays
    pub fn prob(&self, plays_so_far: usize, card: u16) -> f32 {
        let row = self.in_hand_prob.get(plays_so_far.min(self.in_hand_prob.len().saturating_sub(1)));
        match (row, self.deck.iter().position(|&c| c == card)) {
            (Some(r), Some(i)) => r[i],
            _ => 0.0,
        }
    }
}

/// `deck`: 8 card ids (order irrelevant). `plays`: card ids in play order.
pub fn infer_initial_hand(deck: &[u16], plays: &[u16]) -> Option<Inference> {
    if deck.len() != 8 {
        return None;
    }
    let mut deck: Vec<u16> = deck.to_vec();
    deck.sort_unstable();
    let in_deck: Vec<u16> = plays.iter().copied().filter(|c| deck.contains(c)).collect();
    let skipped = plays.len() - in_deck.len();
    let idx_of = |c: u16| deck.iter().position(|&d| d == c).unwrap();

    // every consistent initial state, with its hand membership after each play
    let mut states: Vec<(HandState, Vec<[bool; 8]>)> = Vec::new();
    let mut counts: Vec<[u32; 8]> = vec![[0; 8]; in_deck.len() + 1];
    for a in 0..8 {
        for b in (a + 1)..8 {
            for c in (b + 1)..8 {
                for d in (c + 1)..8 {
                    let hand = [deck[a], deck[b], deck[c], deck[d]];
                    let rest: Vec<u16> = (0..8).filter(|i| ![a, b, c, d].contains(i)).map(|i| deck[i]).collect();
                    for perm in permutations4(&rest) {
                        let mut st = HandState::new(hand, perm);
                        let mut trail: Vec<[bool; 8]> = Vec::with_capacity(in_deck.len() + 1);
                        let mut ok = true;
                        let membership = |st: &HandState| {
                            let mut m = [false; 8];
                            for &h in &st.hand {
                                m[idx_of(h)] = true;
                            }
                            m
                        };
                        trail.push(membership(&st));
                        for &card in &in_deck {
                            if st.play(card).is_none() {
                                ok = false;
                                break;
                            }
                            trail.push(membership(&st));
                        }
                        if ok {
                            for (k, m) in trail.iter().enumerate() {
                                for i in 0..8 {
                                    counts[k][i] += m[i] as u32;
                                }
                            }
                            states.push((HandState::new(hand, perm), trail));
                        }
                    }
                }
            }
        }
    }
    if states.is_empty() {
        return None;
    }
    let n = states.len();
    let in_hand_prob: Vec<[f32; 8]> =
        counts.iter().map(|row| std::array::from_fn(|i| row[i] as f32 / n as f32)).collect();
    // canonical = the state whose hand trail has the highest ensemble agreement
    let best = states
        .iter()
        .map(|(st, trail)| {
            let score: u64 = trail
                .iter()
                .enumerate()
                .map(|(k, m)| (0..8).filter(|&i| m[i]).map(|i| counts[k][i] as u64).sum::<u64>())
                .sum();
            (score, st)
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, st)| st.clone())
        .unwrap();
    Some(Inference { initial: best, n_consistent: n, skipped_plays: skipped, deck, in_hand_prob })
}

fn permutations4(v: &[u16]) -> Vec<[u16; 4]> {
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

    #[test]
    fn cycle_mechanics() {
        let mut h = HandState::new([1, 2, 3, 4], [5, 6, 7, 8]);
        assert_eq!(h.play(3), Some(2));
        assert_eq!(h.hand, [1, 2, 5, 4]);
        assert_eq!(h.next_card(), 6);
        assert_eq!(h.play(9), None);
    }

    #[test]
    fn canonical_state_replays_every_legal_sequence() {
        // property: for random initial states and random legal play sequences, the inferred
        // canonical state replays the whole sequence, and the truth is counted as consistent
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..50 {
            let mut deck: Vec<u16> = (1..=8).collect();
            for i in (1..8).rev() {
                let j = (rnd() % (i as u64 + 1)) as usize;
                deck.swap(i, j);
            }
            let mut truth = HandState::new([deck[0], deck[1], deck[2], deck[3]], [deck[4], deck[5], deck[6], deck[7]]);
            let n_plays = 1 + (rnd() % 40) as usize;
            let mut plays = Vec::new();
            for _ in 0..n_plays {
                let c = truth.hand[(rnd() % 4) as usize];
                truth.play(c);
                plays.push(c);
            }
            let inf = infer_initial_hand(&deck, &plays).expect("truth is consistent, so some state is");
            assert!(inf.n_consistent >= 1);
            // the first play was certainly in hand; probabilities are a distribution over 8 cards summing to 4
            assert!((inf.prob(0, plays[0]) - 1.0).abs() < 1e-6);
            for row in &inf.in_hand_prob {
                assert!((row.iter().sum::<f32>() - 4.0).abs() < 1e-4);
            }
            let mut st = inf.initial.clone();
            for &p in &plays {
                assert!(st.play(p).is_some(), "canonical state cannot replay {plays:?}");
            }
        }
    }

    #[test]
    fn long_alternating_sequence_pins_the_queue() {
        // play a hand card, then immediately the card that replaced it: every queue position is observed
        let mut truth = HandState::new([1, 2, 3, 4], [5, 6, 7, 8]);
        let mut plays = Vec::new();
        for &c in [1u16, 5, 2, 6, 3, 7, 4, 8].iter().cycle().take(32) {
            assert!(truth.play(c).is_some());
            plays.push(c);
        }
        let inf = infer_initial_hand(&[8, 7, 6, 5, 4, 3, 2, 1], &plays).unwrap();
        assert!(inf.n_consistent < 1680, "plays must exclude most of the {} states", 1680);
        assert!(inf.initial.hand.contains(&1));
    }

    #[test]
    fn few_plays_leave_ambiguity_but_always_a_solution() {
        let inf = infer_initial_hand(&[1, 2, 3, 4, 5, 6, 7, 8], &[2]).unwrap();
        assert!(inf.n_consistent > 1);
        assert!(inf.initial.hand.contains(&2));
        let inf2 = infer_initial_hand(&[1, 2, 3, 4, 5, 6, 7, 8], &[42, 2]).unwrap();
        assert_eq!(inf2.skipped_plays, 1);
    }
}
