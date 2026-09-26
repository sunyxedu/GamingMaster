//! Deterministic, platform-independent RNG for chance events, random playouts and
//! test perturbations: xoshiro256** seeded through SplitMix64 (no dependency, the
//! same stream on every OS / architecture / Rust version, so a seed reproduces an
//! episode anywhere).

/// xoshiro256** (Blackman & Vigna) with SplitMix64 seeding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        Rng { s: [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n) without modulo bias (Lemire). Panics on n = 0.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below(0)");
        let mut m = (self.next_u64() as u128) * (n as u128);
        if (m as u64) < n {
            let t = n.wrapping_neg() % n;
            while (m as u64) < t {
                m = (self.next_u64() as u128) * (n as u128);
            }
        }
        (m >> 64) as u64
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    /// Index drawn with probability ∝ `weights[i]` (non-negative, positive sum): one
    /// uniform draw against the running sum in list order, so the result is a pure
    /// function of the RNG state and the list. Panics on an empty / zero-sum list.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0 && total.is_finite(), "Rng::weighted: bad weights {weights:?}");
        let x = self.next_f64() * total;
        let mut acc = 0.0;
        let mut last = 0;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.0 {
                acc += w;
                last = i;
                if x < acc {
                    return i;
                }
            }
        }
        last // x rounded up to the total: the last positive entry
    }

    /// An independent child stream (e.g. one per env).
    pub fn split(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_unbiased_enough() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let xa: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let xb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(xa, xb);
        // pinned: the stream must never change (episode seeds reproduce anywhere)
        assert_eq!(Rng::new(0).next_u64(), 11091344671253066420);
        let mut r = Rng::new(7);
        let mut counts = [0usize; 3];
        for _ in 0..30000 {
            counts[r.below(3) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (9500..10500).contains(&c)), "{counts:?}");
        let mut w = [0usize; 3];
        for _ in 0..30000 {
            w[r.weighted(&[1.0, 0.0, 2.0])] += 1;
        }
        assert_eq!(w[1], 0);
        assert!((9500..10500).contains(&w[0]), "{w:?}");
        let mut v: Vec<u32> = (0..10).collect();
        r.shuffle(&mut v);
        let mut s = v.clone();
        s.sort();
        assert_eq!(s, (0..10).collect::<Vec<_>>());
    }
}
