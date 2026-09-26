//! Teacher transforms: engine outputs → expected scores → policy targets → packed rows.
//!
//! Expected score Q ∈ [0, 1] is always from the side to move (the "mover"):
//! W + D/2 of its win / draw / loss probabilities. The engine value target is
//! `2Q − 1` ∈ [−1, 1] (the `tgt.engine_value` contract).
//!
//! * [`q_from_wdl`]: WDL (permille or any unit) → Q — preferred when the engine
//!   reports WDL (Stockfish with `UCI_ShowWDL`).
//! * [`q_from_cp`]: `1 / (1 + exp(−cp / scale))` — the logistic fallback for
//!   centipawn-like scores (the scale is per engine; document it in the teacher meta).
//! * [`q_from_mate`]: a forced mate for the mover → 1, against it → 0.
//!
//! Policy (default, `--policy softmax`): `π(a) ∝ exp(Q(a) / τ + w·u(a))` over the
//! scored moves only (τ = `--tau`, default 0.05; w = `--tie-weight`); moves the engine
//! did not score get 0. `--policy visits` uses normalised search visits instead
//! (KataGo; the AlphaZero target).
//!
//! The secondary tie-break `w·u(a)`: Q saturates in decided positions (a mate, a won
//! endgame or a big advantage: every scored move has Q = 1, or 0; exact othello solves
//! give Q ∈ {0, ½, 1}), and `exp(Q/τ)` alone is then uniform over the scored moves — a
//! move that allows mate weighs as much as the best defence, and a won position is
//! never converted (the failure of DeepMind's searchless chess agent). u(a) ∈ [−1, 1] is
//! the game's secondary utility of the move ([`super::game::Labeller::secondary`]: mate
//! distance and centipawns, discs), a strictly increasing function of the engine's
//! native score `tgt.teacher_raw`. Within a position Q is a non-decreasing function of
//! the same score (WDL permille, logistic cp, exact W/D/L), so the two never disagree
//! on a pair of moves: the effective score `Q/τ + w·u` orders the moves by Q and, among
//! equal Q, by u (lexicographic by construction — the term never overturns a Q
//! difference, it only splits ties; were an engine to break the monotonicity, e.g.
//! othello rows of one position scored partly exactly and partly by the midgame search,
//! a Q gap below 2wτ could be overturned), and within a class of equal Q the secondary
//! preference ratio is at most `e^{2w}` (u spans [−1, 1]). Near equality u only adds a
//! small slope (a few % of Q/τ's; see each game). w = 0 (and games without a secondary
//! utility, e.g. go) give `π ∝ exp(Q/τ)` bit for bit. τ ≤ 0 is the limit τ → 0: all
//! mass on the arg-max-Q moves, weighted `exp(w·u)` (uniform when w = 0). The driver
//! sorts, truncates to `--max-moves` and computes π from the same scores
//! ([`tie_scores`], [`softmax_scores`]); [`softmax_q_tie`] re-derives π from the
//! stored columns.
//!
//! Factorised decoders get rows derived from the full-action policy with
//! [`marginal`] (e.g. chess `from_probs`) and [`conditional`] (e.g. chess `to_probs`
//! = p(to | from = tgt.from_idx)); a conditional whose event has zero teacher mass is
//! an empty row (= no teacher for that head on that frame).
//!
//! Packing: [`SparseRows`] (`<head>_probs_idx` i64 −1-padded + `<head>_probs_val`
//! f32 0-padded, top-k by probability), [`DenseRows`] (`<head>_probs` f32 [n]) and
//! [`Padded`] (fixed-width rows with a pad value: `teacher_idx` −1, `teacher_q` NaN).

use anyhow::{bail, Result};
use std::collections::BTreeMap;

/// `--tie-weight` default (w of the secondary tie-break, see the module docs): within a
/// class of equal Q the secondary preference ratio is at most e^(2w) ≈ 55.
pub const DEFAULT_TIE_WEIGHT: f64 = 2.0;

/// How the teacher policy is derived from an [`super::game::Analysis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PolicyKind {
    /// π ∝ exp(Q / τ + w·u) over the scored moves (w·u: the secondary tie-break)
    Softmax,
    /// normalised search visits (MCTS engines)
    Visits,
}

impl PolicyKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyKind::Softmax => "softmax",
            PolicyKind::Visits => "visits",
        }
    }
}

/// Q from win / draw / loss weights (permille or any common unit).
pub fn q_from_wdl(w: f32, d: f32, l: f32) -> f32 {
    let s = w + d + l;
    if s <= 0.0 {
        return 0.5;
    }
    ((w + 0.5 * d) / s).clamp(0.0, 1.0)
}

/// Logistic Q of a centipawn-like score: `1 / (1 + exp(−cp / scale))`.
pub fn q_from_cp(cp: f32, scale: f32) -> f32 {
    1.0 / (1.0 + (-(cp as f64) / scale as f64).exp()) as f32
}

/// Q of a mate score: mover mates (n > 0) → 1, mover is mated (n ≤ 0) → 0.
pub fn q_from_mate(n: i32) -> f32 {
    if n > 0 { 1.0 } else { 0.0 }
}

/// The engine value target `2Q − 1`, clamped to [−1, 1].
pub fn value_from_q(q: f32) -> f32 {
    (2.0 * q - 1.0).clamp(-1.0, 1.0)
}

/// `π ∝ exp(q / τ)` (numerically stable); τ ≤ 0 → uniform over the arg-max set.
pub fn softmax_q(q: &[f32], tau: f32) -> Vec<f32> {
    if q.is_empty() {
        return Vec::new();
    }
    let m = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if tau <= 0.0 {
        let n = q.iter().filter(|&&x| x == m).count() as f32;
        return q.iter().map(|&x| if x == m { 1.0 / n } else { 0.0 }).collect();
    }
    let e: Vec<f64> = q.iter().map(|&x| ((x - m) as f64 / tau as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|&x| (x / s) as f32).collect()
}

/// Exponents of the tie-broken softmax teacher `π ∝ exp(Q/τ + w·u)` (module docs),
/// shifted by the best Q: `(Q(a) − Q*)/τ + w·u(a)` in f64 (Q* = max Q, computed as in
/// [`softmax_q`], so `w = 0` reproduces its exponents exactly). τ ≤ 0 is the limit
/// τ → 0: `w·u(a)` for the arg-max-Q moves, −∞ for the others. `u` has one entry per Q.
pub fn tie_scores(q: &[f32], u: &[f32], tau: f32, w: f32) -> Vec<f64> {
    assert_eq!(q.len(), u.len(), "one secondary utility per move");
    let m = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let tie = |y: f32| if w == 0.0 { 0.0 } else { w as f64 * y as f64 };
    if tau <= 0.0 {
        return q.iter().zip(u).map(|(&x, &y)| if x == m { tie(y) } else { f64::NEG_INFINITY }).collect();
    }
    q.iter().zip(u).map(|(&x, &y)| (x - m) as f64 / tau as f64 + tie(y)).collect()
}

/// `π ∝ exp(s)` (numerically stable) over scores from [`tie_scores`] (at least one finite).
pub fn softmax_scores(s: &[f64]) -> Vec<f32> {
    if s.is_empty() {
        return Vec::new();
    }
    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|&x| (x / sum) as f32).collect()
}

/// The tie-broken softmax teacher `π ∝ exp(Q/τ + w·u)` of the module docs — e.g. to
/// re-derive π from `tgt.teacher_q` and the game's secondary utility of
/// `tgt.teacher_raw`. `w = 0` is [`softmax_q`] (u ignored).
pub fn softmax_q_tie(q: &[f32], u: &[f32], tau: f32, w: f32) -> Vec<f32> {
    if w == 0.0 {
        return softmax_q(q, tau);
    }
    softmax_scores(&tie_scores(q, u, tau, w))
}

/// Secondary utility of a forced mate in `n` moves of the mating side (n ≥ 1, smaller
/// = clamped to 1): `0.5 + 0.5·5/(n + 4)` ∈ (0.5, 1] — mate in 1 = 1, mate in 6 = 0.75,
/// long mates → 0.5⁺, so every mate outranks every non-mate utility (< 0.5). Negate it
/// for the mated side (being mated slower is preferred). Strictly decreasing in n (in
/// f32 up to n ≈ 2000 in half-move steps; engines report mates within a few hundred).
pub fn u_mate(n: f32) -> f32 {
    let n = (n as f64).max(1.0);
    (0.5 + 2.5 / (n + 4.0)) as f32
}

/// Secondary utility of a non-mate centipawn score: `0.5·tanh(cp / scale)` ∈ (−0.5, 0.5).
/// In f32 it is strictly increasing on integer cp up to |cp| ≈ 5·scale, non-decreasing
/// beyond, and saturates at ±(0.5 − 2⁻²⁵) — still below every [`u_mate`] — beyond ≈ 9·scale.
pub fn u_cp(cp: f32, scale: f32) -> f32 {
    const MAX: f32 = 0.5 - f32::EPSILON / 4.0; // the largest f32 below 0.5
    ((0.5 * (cp as f64 / scale as f64).tanh()) as f32).clamp(-MAX, MAX)
}

/// Non-negative weights → probabilities; `None` when nothing is positive.
pub fn normalise(w: &[f32]) -> Option<Vec<f32>> {
    let s: f64 = w.iter().filter(|x| x.is_finite() && **x > 0.0).map(|&x| x as f64).sum();
    if s <= 0.0 {
        return None;
    }
    Some(w.iter().map(|&x| if x.is_finite() && x > 0.0 { (x as f64 / s) as f32 } else { 0.0 }).collect())
}

/// Σ p over actions with the same `key` (sorted by key; zero-mass keys dropped).
pub fn marginal<A>(dist: &[(A, f32)], key: impl Fn(&A) -> i64) -> Vec<(i64, f32)> {
    let mut m: BTreeMap<i64, f64> = BTreeMap::new();
    for (a, p) in dist {
        *m.entry(key(a)).or_default() += *p as f64;
    }
    m.into_iter().filter(|(_, p)| *p > 0.0).map(|(k, p)| (k, p as f32)).collect()
}

/// p(key | cond) = marginal over the actions satisfying `cond`, renormalised;
/// empty when the event has zero mass (no teacher).
pub fn conditional<A>(dist: &[(A, f32)], cond: impl Fn(&A) -> bool, key: impl Fn(&A) -> i64) -> Vec<(i64, f32)> {
    let mut m: BTreeMap<i64, f64> = BTreeMap::new();
    let mut total = 0.0f64;
    for (a, p) in dist {
        if cond(a) {
            *m.entry(key(a)).or_default() += *p as f64;
            total += *p as f64;
        }
    }
    if total <= 0.0 {
        return Vec::new();
    }
    m.into_iter().filter(|(_, p)| *p > 0.0).map(|(k, p)| (k, (p / total) as f32)).collect()
}

/// Validate a categorical row over `n` classes: ids in range and unique, probabilities
/// finite and ≥ 0; returns it normalised to sum 1 (empty if it has no mass).
pub fn clean_row(row: &[(i64, f32)], n: usize) -> Result<Vec<(i64, f32)>> {
    let mut seen = std::collections::HashSet::new();
    for &(i, p) in row {
        if i < 0 || i as usize >= n {
            bail!("teacher class {i} outside [0, {n})");
        }
        if !seen.insert(i) {
            bail!("teacher class {i} repeated in one row");
        }
        if !p.is_finite() || p < 0.0 {
            bail!("teacher probability {p} for class {i} is not finite and ≥ 0");
        }
    }
    let s: f64 = row.iter().map(|&(_, p)| p as f64).sum();
    if s <= 0.0 {
        return Ok(Vec::new());
    }
    Ok(row.iter().filter(|(_, p)| *p > 0.0).map(|&(i, p)| (i, (p as f64 / s) as f32)).collect())
}

/// Sparse categorical rows: per frame the top-`k` classes by probability.
#[derive(Debug, Clone)]
pub struct SparseRows {
    pub k: usize,
    pub idx: Vec<i64>,
    pub val: Vec<f32>,
    /// total probability mass dropped by the top-k truncation
    pub truncated_mass: f64,
}

impl SparseRows {
    pub fn new(k: usize) -> Self {
        SparseRows { k: k.max(1), idx: Vec::new(), val: Vec::new(), truncated_mass: 0.0 }
    }

    /// Append one row (sorted by probability desc, ties by id; beyond `k` dropped; the
    /// kept values are not renormalised — the trainer renormalises rows).
    pub fn push(&mut self, row: &[(i64, f32)]) {
        let mut r: Vec<(i64, f32)> = row.iter().copied().filter(|(_, p)| *p > 0.0).collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for &(_, p) in r.iter().skip(self.k) {
            self.truncated_mass += p as f64;
        }
        r.truncate(self.k);
        for j in 0..self.k {
            let (i, p) = r.get(j).copied().unwrap_or((-1, 0.0));
            self.idx.push(i);
            self.val.push(p);
        }
    }

    /// Append a no-teacher row.
    pub fn push_empty(&mut self) {
        self.push(&[]);
    }

    pub fn rows(&self) -> usize {
        self.idx.len() / self.k
    }
}

/// Dense categorical rows `[n]` (rows of zeros = no teacher).
#[derive(Debug, Clone)]
pub struct DenseRows {
    pub n: usize,
    pub data: Vec<f32>,
}

impl DenseRows {
    pub fn new(n: usize) -> Self {
        DenseRows { n, data: Vec::new() }
    }

    /// Append one row (ids must be in [0, n) — use [`clean_row`] first).
    pub fn push(&mut self, row: &[(i64, f32)]) {
        let base = self.data.len();
        self.data.resize(base + self.n, 0.0);
        for &(i, p) in row {
            self.data[base + i as usize] += p;
        }
    }

    pub fn push_empty(&mut self) {
        self.data.resize(self.data.len() + self.n, 0.0);
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.n.max(1)
    }
}

/// Fixed-width rows with a pad value (`teacher_idx`: −1, `teacher_q`: NaN).
#[derive(Debug, Clone)]
pub struct Padded<T: Copy> {
    pub k: usize,
    pub pad: T,
    pub data: Vec<T>,
}

impl<T: Copy> Padded<T> {
    pub fn new(k: usize, pad: T) -> Self {
        Padded { k: k.max(1), pad, data: Vec::new() }
    }

    /// Append one row (longer rows are truncated to `k`).
    pub fn push(&mut self, row: impl IntoIterator<Item = T>) {
        let base = self.data.len();
        self.data.extend(row.into_iter().take(self.k));
        self.data.resize(base + self.k, self.pad);
    }

    pub fn push_empty(&mut self) {
        self.data.resize(self.data.len() + self.k, self.pad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn expected_scores() {
        assert!(close(q_from_wdl(500.0, 500.0, 0.0), 0.75));
        assert!(close(q_from_wdl(0.0, 1000.0, 0.0), 0.5));
        assert!(close(q_from_wdl(0.0, 0.0, 0.0), 0.5));
        assert!(close(q_from_cp(0.0, 400.0), 0.5));
        assert!(close(q_from_cp(400.0, 400.0), 1.0 / (1.0 + (-1.0f32).exp())));
        assert!(close(q_from_cp(-400.0, 400.0) + q_from_cp(400.0, 400.0), 1.0));
        assert_eq!((q_from_mate(3), q_from_mate(-2), q_from_mate(0)), (1.0, 0.0, 0.0));
        assert_eq!((value_from_q(1.0), value_from_q(0.5), value_from_q(0.0)), (1.0, 0.0, -1.0));
    }

    #[test]
    fn softmax_and_normalise() {
        let p = softmax_q(&[0.6, 0.5, 0.5], 0.05);
        let e = (0.1f64 / 0.05).exp();
        assert!(close(p[0], (e / (e + 2.0)) as f32) && close(p[1], p[2]));
        assert!(close(p.iter().sum::<f32>(), 1.0));
        assert_eq!(softmax_q(&[0.2, 0.9, 0.9], 0.0), vec![0.0, 0.5, 0.5]);
        assert!(softmax_q(&[], 0.1).is_empty());
        // extreme temperatures stay finite
        assert!(softmax_q(&[1.0, 0.0], 1e-6).iter().all(|x| x.is_finite()));
        assert_eq!(normalise(&[2.0, 0.0, 6.0]).unwrap(), vec![0.25, 0.0, 0.75]);
        assert!(normalise(&[0.0, -1.0]).is_none());
    }

    /// Deterministic pseudo-random f32 in [0, 1).
    fn rnd(state: &mut u64) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }

    #[test]
    fn tie_break_with_zero_weight_is_the_plain_softmax_bit_for_bit() {
        let mut r = 7u64;
        for case in 0..400 {
            let n = 1 + case % 12;
            // Q with ties and saturation (a third of the cases all 1.0), u anywhere (incl. NaN)
            let q: Vec<f32> = (0..n)
                .map(|_| match case % 3 { 0 => 1.0, 1 => (rnd(&mut r) * 4.0).floor() / 4.0, _ => rnd(&mut r) })
                .collect();
            let u: Vec<f32> = (0..n).map(|i| if i == 1 && case % 5 == 0 { f32::NAN } else { 2.0 * rnd(&mut r) - 1.0 }).collect();
            for tau in [0.05f32, 0.025, 0.3, 1e-4, 0.0, -1.0] {
                let old: Vec<u32> = softmax_q(&q, tau).iter().map(|x| x.to_bits()).collect();
                let tie: Vec<u32> = softmax_q_tie(&q, &u, tau, 0.0).iter().map(|x| x.to_bits()).collect();
                assert_eq!(old, tie, "q {q:?} tau {tau}");
                // the scores the driver sorts and softmaxes reproduce it too
                let via: Vec<u32> = softmax_scores(&tie_scores(&q, &u, tau, 0.0)).iter().map(|x| x.to_bits()).collect();
                assert_eq!(old, via, "q {q:?} tau {tau}");
            }
        }
    }

    #[test]
    fn saturated_q_is_ordered_by_the_secondary_utility() {
        // every move wins (Q = 1): the plain teacher is uniform, the tie-break follows u
        let (q, u) = ([1.0f32; 4], [0.2f32, 1.0, -0.5, 0.7]);
        assert_eq!(softmax_q(&q, 0.05), vec![0.25; 4]);
        let p = softmax_q_tie(&q, &u, 0.05, 2.0);
        let z: f64 = u.iter().map(|&x| (2.0 * x as f64).exp()).sum();
        for (pi, ui) in p.iter().zip(u) {
            assert!(close(*pi, ((2.0 * ui as f64).exp() / z) as f32), "π ∝ exp(w·u): {p:?}");
        }
        assert!(p[1] > p[3] && p[3] > p[0] && p[0] > p[2]);
        assert!(close(p.iter().sum::<f32>(), 1.0));
        // equal Q, equal u → equal probabilities
        let p = softmax_q_tie(&[0.0, 0.0, 0.0], &[-0.3, -0.3, -0.9], 0.05, 2.0);
        assert!(p[0] == p[1] && p[1] > p[2]);
        // Q classes stay ordered (u increases with Q, as the games' utilities do): the
        // tie-break only splits the Q = 1 class
        let p = softmax_q_tie(&[1.0, 1.0, 0.9, 0.5], &[0.45, 0.3, 0.2, 0.0], 0.05, 2.0);
        assert!(p[0] > p[1] && p[1] > p[2] && p[2] > p[3], "{p:?}");
        let s = tie_scores(&[1.0, 1.0, 0.9, 0.5], &[0.45, 0.3, 0.2, 0.0], 0.05, 2.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]) && (s[0] - 0.9).abs() < 1e-6, "{s:?}");
        // τ ≤ 0 (the limit τ → 0): mass on the arg-max-Q moves only, ∝ exp(w·u)
        let p = softmax_q_tie(&[1.0, 1.0, 0.9], &[0.2, -0.2, 1.0], 0.0, 2.0);
        assert_eq!(p[2], 0.0);
        assert!(close(p[0] / p[1], 0.8f32.exp()));
        assert_eq!(tie_scores(&[1.0, 0.9], &[0.0, 1.0], -1.0, 2.0), vec![0.0, f64::NEG_INFINITY]);
        assert!(softmax_q_tie(&[], &[], 0.05, 2.0).is_empty());
    }

    #[test]
    fn secondary_preference_ratio_is_bounded() {
        let mut r = 11u64;
        for w in [0.5f32, 2.0, 5.0] {
            // within one Q class the ratio is at most e^(2w), reached at u = ±1
            let p = softmax_q_tie(&[0.7, 0.7, 0.7], &[1.0, 0.0, -1.0], 0.05, w);
            assert!(((p[0] / p[2]) as f64 / (2.0 * w as f64).exp() - 1.0).abs() < 1e-5, "{p:?}");
            for _ in 0..100 {
                let u: Vec<f32> = (0..6).map(|_| 2.0 * rnd(&mut r) - 1.0).collect();
                let p = softmax_q_tie(&[1.0; 6], &u, 0.05, w);
                let (hi, lo) = (p.iter().copied().fold(0.0, f32::max), p.iter().copied().fold(1.0, f32::min));
                assert!((hi / lo) as f64 <= (2.0 * w as f64).exp() * (1.0 + 1e-5));
            }
        }
        // extreme weights and temperatures stay finite and normalised
        for (tau, w) in [(1e-6f32, 2.0f32), (0.05, 1e6), (100.0, 1e-9)] {
            let p = softmax_q_tie(&[1.0, 0.0, 0.5], &[-1.0, 1.0, 0.0], tau, w);
            assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0) && close(p.iter().sum::<f32>(), 1.0), "{tau} {w}: {p:?}");
        }
    }

    #[test]
    fn mate_and_centipawn_utilities() {
        assert_eq!((u_mate(1.0), u_mate(6.0), u_mate(0.0)), (1.0, 0.75, 1.0));
        assert!(close(u_mate(2.0), 0.5 + 2.5 / 6.0));
        // half-move steps (shogi counts plies) up to 2000 moves are strictly ordered in f32
        let mates: Vec<f32> = (2..=4000).map(|h| u_mate(h as f32 / 2.0)).collect();
        assert!(mates.windows(2).all(|w| w[0] > w[1]), "a faster mate ranks higher");
        assert!(u_mate(9_999.0) > 0.5 && u_mate(50_000.0) > 0.5, "every mate above every non-mate");
        assert_eq!(u_cp(0.0, 1000.0), 0.0);
        assert!(close(u_cp(1000.0, 1000.0), 0.5 * 1f32.tanh()) && close(u_cp(-1000.0, 1000.0), -0.5 * 1f32.tanh()));
        for cp in [1e4f32, 2e4, 35281.0, 1e9] {
            assert!(u_cp(cp, 1000.0) < 0.5 && u_cp(-cp, 1000.0) > -0.5 && u_cp(cp, 1000.0) < u_mate(50_000.0), "{cp}");
        }
        let cps: Vec<f32> = (-5000..=5000).map(|c| u_cp(c as f32, 1000.0)).collect();
        assert!(cps.windows(2).all(|w| w[0] < w[1]), "strictly increasing in cp");
        let far: Vec<f32> = (5000..=60_000).map(|c| u_cp(c as f32, 1000.0)).collect();
        assert!(far.windows(2).all(|w| w[0] <= w[1]), "non-decreasing beyond (f32 resolution)");
    }

    #[test]
    fn marginal_and_conditional_of_a_from_to_policy() {
        // actions (from, to): p
        let dist = [((1, 5), 0.4f32), ((1, 6), 0.1), ((2, 5), 0.5)];
        assert_eq!(marginal(&dist, |a| a.0), vec![(1, 0.5), (2, 0.5)]);
        let to_given_1 = conditional(&dist, |a| a.0 == 1, |a| a.1);
        assert!(close(to_given_1[0].1, 0.8) && close(to_given_1[1].1, 0.2));
        assert!(conditional(&dist, |a| a.0 == 3, |a| a.1).is_empty(), "zero-mass event = no teacher");
    }

    #[test]
    fn row_packing() {
        let mut s = SparseRows::new(2);
        s.push(&[(4, 0.2), (7, 0.5), (1, 0.3)]);
        s.push_empty();
        s.push(&[(3, 0.5), (2, 0.5)]);
        assert_eq!(s.idx, vec![7, 1, -1, -1, 2, 3]);
        assert_eq!(s.val, vec![0.5, 0.3, 0.0, 0.0, 0.5, 0.5]);
        assert!((s.truncated_mass - 0.2).abs() < 1e-6);
        assert_eq!(s.rows(), 3);
        let mut d = DenseRows::new(3);
        d.push(&[(2, 0.75), (0, 0.25)]);
        d.push_empty();
        assert_eq!(d.data, vec![0.25, 0.0, 0.75, 0.0, 0.0, 0.0]);
        let mut q = Padded::new(3, f32::NAN);
        q.push([0.5, 0.25]);
        q.push([1.0, 2.0, 3.0, 4.0]);
        assert!(q.data[2].is_nan() && q.data[5] == 3.0 && q.data.len() == 6);
        let mut i = Padded::new(2, -1i64);
        i.push_empty();
        assert_eq!(i.data, vec![-1, -1]);
        assert_eq!(clean_row(&[(1, 2.0), (0, 2.0)], 2).unwrap(), vec![(1, 0.5), (0, 0.5)]);
        assert!(clean_row(&[(2, 1.0)], 2).is_err() && clean_row(&[(0, f32::NAN)], 2).is_err());
        assert!(clean_row(&[(0, 0.5), (0, 0.5)], 2).is_err());
        assert!(clean_row(&[(0, 0.0)], 2).unwrap().is_empty());
    }
}
