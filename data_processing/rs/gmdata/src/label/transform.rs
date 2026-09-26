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
//! Policy (default, `--policy softmax`): `π(a) ∝ exp(Q(a) / τ)` over the scored
//! moves only (τ = `--tau`, default 0.05; τ ≤ 0 → uniform over the best moves);
//! moves the engine did not score get 0. `--policy visits` uses normalised search
//! visits instead (KataGo; the AlphaZero target).
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

/// How the teacher policy is derived from an [`super::game::Analysis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PolicyKind {
    /// π ∝ exp(Q / τ) over the scored moves
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
