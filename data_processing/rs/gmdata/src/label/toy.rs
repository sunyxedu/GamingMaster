//! A toy game for the label pipeline's self-test (`gmdata label selftest`) and unit
//! tests — and a compact worked example of a [`Labeller`].
//!
//! Chess-shaped on purpose, so its labelled shards feed the real chess network in the
//! Python consumer test (tests/test_label_overlay.py): the same tensors as
//! `gmdata chess convert` (obs.board_seq [T,14,8,8] bits, obs.globals_seq [T,10],
//! tgt.from_idx / to_idx / promotion / value, tgt.legal_from / legal_to_of /
//! legal_promotion_of bits) and the FromToDecoder heads (from, to, promotion).
//!
//! The "rules" are pseudo-random but deterministic: a position is a u64 state; its
//! legal moves, observation planes and globals are hashes of the state; playing move
//! `a` moves to `hash(state, a)`. Sample meta `{"seed": s}` is the first state, so a
//! sample is replayed from its seed and its targets, like a real game.
//!
//! Flat action id = `(from·64 + to)·5 + promotion` (the suggested chess encoding).
//! Engine protocol: UCI, `position toy <state as 16 hex digits>` + `go nodes N`; moves
//! are written `m<action id>`. [`fake_engine`] is the matching engine.

use super::engine::EngineCommand;
use super::fake::{FakeMove, UciFake};
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove};
use super::transform::{conditional, marginal, q_from_cp, q_from_wdl};
use super::uci::{Flavor, Score, UciSession};
use anyhow::{bail, Context, Result};
use gmshard::{Sample, ShardReader, ShardWriter};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;

pub const GAME: &str = "toy";
pub const CONVERTER_VERSION: u32 = 1;
const PLANES: usize = 14;
const GLOBALS: usize = 10;
const N_PROMO: i64 = 5;
/// centipawn scale of the logistic fallback (when the engine sends no WDL)
pub const CP_SCALE: f32 = 400.0;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn action_id(from: i64, to: i64, promo: i64) -> i64 {
    (from * 64 + to) * N_PROMO + promo
}

/// (from, to, promotion) of a flat action id.
pub fn decode_action(id: i64) -> (i64, i64, i64) {
    (id / N_PROMO / 64, id / N_PROMO % 64, id % N_PROMO)
}

/// Legal moves (from, to, promotion) of `state`: 1–8 (from, to) pairs on six
/// from-squares; a quarter of the pairs are promotions (4 moves each).
pub fn legal_moves(state: u64) -> Vec<(i64, i64, i64)> {
    let pairs = 1 + (splitmix(state) % 8) as usize;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut j = 0u64;
    while seen.len() < pairs {
        j += 1;
        let x = splitmix(state ^ j.wrapping_mul(0xA24B_AED4_963E_E407));
        let from = [0i64, 9, 18, 27, 36, 45][(x % 6) as usize];
        let to = ((x >> 8) % 64) as i64;
        if to == from || !seen.insert((from, to)) {
            continue;
        }
        if (x >> 20) % 4 == 0 {
            out.extend((1..N_PROMO).map(|p| (from, to, p)));
        } else {
            out.push((from, to, 0));
        }
    }
    out
}

pub fn next_state(state: u64, action: i64) -> u64 {
    splitmix(state ^ splitmix(action as u64 ^ 0x1234_5678))
}

/// obs.board_seq frame [14·64] ({0,1}) of `state`.
pub fn planes(state: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(PLANES * 64);
    for w in 0..PLANES as u64 {
        let word = splitmix(state ^ (w + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93));
        v.extend((0..64).map(|i| ((word >> i) & 1) as f32));
    }
    v
}

/// obs.globals_seq frame [10] of `state`.
pub fn globals(state: u64) -> Vec<f32> {
    (0..GLOBALS as u64).map(|i| ((splitmix(state ^ 0xABC ^ (i << 32)) >> 40) % 256) as f32 / 255.0).collect()
}

/// Legality masks (legal_from, legal_to_of, legal_promotion_of) given the GT move.
fn masks(legal: &[(i64, i64, i64)], gt: (i64, i64, i64)) -> ([f32; 64], [f32; 64], [f32; 5]) {
    let (mut f, mut t, mut p) = ([0f32; 64], [0f32; 64], [0f32; 5]);
    for &(a, b, c) in legal {
        f[a as usize] = 1.0;
        if a == gt.0 {
            t[b as usize] = 1.0;
            if b == gt.1 {
                p[c as usize] = 1.0;
            }
        }
    }
    (f, t, p)
}

/// The converter's move choice at `state`.
fn chosen(state: u64, legal: &[(i64, i64, i64)]) -> (i64, i64, i64) {
    legal[(splitmix(state ^ 0xFEED) % legal.len() as u64) as usize]
}

/// A toy "game record" converted like a real converter would.
pub fn sample(seed: u64) -> Sample {
    let t = 4 + (splitmix(seed ^ 77) % 9) as usize;
    let outcome = [-1.0f32, 0.0, 1.0][(splitmix(seed ^ 5) % 3) as usize];
    let (mut board, mut glob, mut from, mut to, mut promo, mut value) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut lf, mut lt, mut lp) = (vec![], vec![], vec![]);
    let mut state = seed;
    for k in 0..t {
        let legal = legal_moves(state);
        let gt = chosen(state, &legal);
        board.extend(planes(state));
        glob.extend(globals(state));
        let (f, tt, p) = masks(&legal, gt);
        lf.extend_from_slice(&f);
        lt.extend_from_slice(&tt);
        lp.extend_from_slice(&p);
        from.push(gt.0);
        to.push(gt.1);
        promo.push(gt.2);
        value.push(if k % 2 == 0 { outcome } else { -outcome });
        state = next_state(state, action_id(gt.0, gt.1, gt.2));
    }
    Sample::new(json!({"seed": seed}))
        .bits("obs.board_seq", &[PLANES, 8, 8], &board)
        .f32("obs.globals_seq", &[GLOBALS], glob)
        .i64("tgt.from_idx", &[], from)
        .i64("tgt.to_idx", &[], to)
        .i64("tgt.promotion", &[], promo)
        .f32("tgt.value", &[], value)
        .bits("tgt.legal_from", &[64], &lf)
        .bits("tgt.legal_to_of", &[64], &lt)
        .bits("tgt.legal_promotion_of", &[5], &lp)
}

/// Write `shards × per_shard` toy samples as gm-shards under `out`.
pub fn write_shards(out: &Path, shards: usize, per_shard: usize, seed: u64) -> Result<()> {
    let mut w = ShardWriter::new(out, GAME, CONVERTER_VERSION, json!({"global_features": GLOBALS}), per_shard)?;
    for i in 0..shards * per_shard {
        w.push(sample(splitmix(seed ^ i as u64)))?;
    }
    w.flush()
}

/// The toy engine: every legal move scored `cp ∈ [−300, 300]` (a hash) with a WDL
/// derived from it; unknown positions crash it.
pub fn fake_engine() -> UciFake<impl FnMut(&str) -> Option<Vec<FakeMove>> + Send> {
    UciFake::new(|pos: &str| {
        let hex = pos.strip_prefix("position toy ")?;
        let state = u64::from_str_radix(hex.trim(), 16).ok()?;
        Some(
            legal_moves(state)
                .into_iter()
                .map(|(f, t, p)| {
                    let id = action_id(f, t, p);
                    let cp = (splitmix(state ^ id as u64) % 601) as i32 - 300;
                    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
                    let w = (1000.0 * sig((cp as f32 - 50.0) / 120.0)).round() as u32;
                    let l = (1000.0 * sig((-cp as f32 - 50.0) / 120.0)).round() as u32;
                    FakeMove::cp(&format!("m{id}"), cp).wdl([w, 1000 - w - l, l])
                })
                .collect(),
        )
    })
}

/// [`fake_engine`] as an in-process engine command.
pub fn fake_engine_command() -> EngineCommand {
    EngineCommand::in_process("toy-uci", fake_engine)
}

/// Replayed toy position.
pub struct ToyPos {
    pub state: u64,
    pub legal: Vec<(i64, i64, i64)>,
}

/// The toy labeller. `visits` / `bernoulli` switch on test-only extras: visit counts
/// derived from Q (exercises `--policy visits` and `tgt.teacher_visits`) and a
/// bernoulli head `promo_any` = P(the move promotes).
#[derive(Default, Clone)]
pub struct ToyLabeller {
    pub visits: bool,
    pub bernoulli: bool,
}

impl Labeller for ToyLabeller {
    type Position = ToyPos;
    type Key = u64;
    type Session = UciSession;

    fn game(&self) -> &'static str {
        GAME
    }

    fn version(&self) -> u32 {
        1
    }

    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        if shard.game != GAME || shard.converter_version != CONVERTER_VERSION {
            bail!("{}: not a toy v{CONVERTER_VERSION} shard", shard.dir.display());
        }
        for t in ["obs.board_seq", "obs.globals_seq", "tgt.from_idx", "tgt.to_idx", "tgt.promotion"] {
            shard.tensor_info(t)?;
        }
        Ok(())
    }

    fn heads(&self) -> Vec<HeadSpec> {
        let mut h = vec![
            HeadSpec { name: "from", kind: HeadKind::Categorical { n: 64, k: 8 } },
            HeadSpec { name: "to", kind: HeadKind::Categorical { n: 64, k: 16 } },
            HeadSpec { name: "promotion", kind: HeadKind::Categorical { n: 5, k: 5 } },
        ];
        if self.bernoulli {
            h.push(HeadSpec { name: "promo_any", kind: HeadKind::Bernoulli });
        }
        h
    }

    fn raw_unit(&self) -> Option<&'static str> {
        Some("cp")
    }

    fn has_visits(&self) -> bool {
        self.visits
    }

    fn default_max_moves(&self) -> usize {
        16
    }

    fn resolve_limits(&self, given: &Limits) -> Result<Limits> {
        given.with_defaults(&[("nodes", "1000")], &["nodes"])
    }

    fn open_session(&self, cfg: &EngineConfig) -> Result<UciSession> {
        let mut opts = cfg.options.clone();
        opts.push(("MultiPV".into(), cfg.max_moves.to_string()));
        opts.push(("UCI_ShowWDL".into(), "true".into()));
        UciSession::open(cfg.command.clone(), Flavor::Uci, &opts, cfg.timeout)
    }

    fn engine_id(&self, s: &UciSession) -> String {
        s.id_name.clone()
    }

    fn teacher_meta(&self, _cfg: &EngineConfig) -> Value {
        json!({"cp_scale": CP_SCALE, "q": "wdl, else logistic(cp / cp_scale)"})
    }

    fn replay(&self, s: &SampleView) -> Result<Vec<ToyPos>> {
        let mut state = s.meta()["seed"].as_u64().context("sample meta has no seed")?;
        let (from, to, promo) = (s.i64s("tgt.from_idx")?, s.i64s("tgt.to_idx")?, s.i64s("tgt.promotion")?);
        let mut out = Vec::with_capacity(s.len);
        for t in 0..s.len {
            let legal = legal_moves(state);
            let gt = (from[t], to[t], promo[t]);
            if !legal.contains(&gt) {
                bail!("frame {t}: recorded move {gt:?} is illegal");
            }
            let next = next_state(state, action_id(gt.0, gt.1, gt.2));
            out.push(ToyPos { state, legal });
            state = next;
        }
        Ok(out)
    }

    fn verify(&self, s: &SampleView, positions: &[ToyPos]) -> Result<()> {
        let (mut board, mut glob, mut lf, mut lt, mut lp) = (vec![], vec![], vec![], vec![], vec![]);
        let (from, to, promo) = (s.i64s("tgt.from_idx")?, s.i64s("tgt.to_idx")?, s.i64s("tgt.promotion")?);
        for (t, p) in positions.iter().enumerate() {
            board.extend(planes(p.state));
            glob.extend(globals(p.state));
            let (f, tt, pp) = masks(&p.legal, (from[t], to[t], promo[t]));
            lf.extend_from_slice(&f);
            lt.extend_from_slice(&tt);
            lp.extend_from_slice(&pp);
        }
        s.check_f32("obs.board_seq", &board)?;
        s.check_f32("obs.globals_seq", &glob)?;
        s.check_f32("tgt.legal_from", &lf)?;
        s.check_f32("tgt.legal_to_of", &lt)?;
        s.check_f32("tgt.legal_promotion_of", &lp)
    }

    fn skip_reason(&self, _s: &SampleView, _t: usize, pos: &ToyPos) -> Option<&'static str> {
        (pos.legal.len() == 1).then_some("forced")
    }

    fn key(&self, pos: &ToyPos) -> u64 {
        pos.state
    }

    fn query(&self, session: &mut UciSession, pos: &ToyPos, cfg: &EngineConfig) -> Result<Analysis> {
        session.new_game()?;
        let nodes = cfg.limits.u64_or("nodes", 1000)?;
        let r = session.search(&format!("position toy {:016x}", pos.state), &format!("go nodes {nodes}"), cfg.timeout)?;
        let mut moves = Vec::new();
        for line in &r.lines {
            let mv = &line.pv[0];
            let id: i64 = mv.strip_prefix('m').and_then(|x| x.parse().ok()).with_context(|| format!("bad move {mv:?}"))?;
            if !pos.legal.contains(&decode_action(id)) {
                bail!("engine move {mv} is illegal");
            }
            let (q, raw) = match (line.wdl, line.score) {
                (Some([w, d, l]), Some(Score::Cp(cp))) => (q_from_wdl(w as f32, d as f32, l as f32), cp as f32),
                (None, Some(Score::Cp(cp))) => (q_from_cp(cp as f32, CP_SCALE), cp as f32),
                (_, s) => bail!("unexpected score {s:?}"),
            };
            let mut m = ScoredMove::new(id, q).raw(raw);
            if self.visits {
                m = m.visits((1000.0 * q * q).round());
            }
            moves.push(m);
        }
        let value = moves.first().map(|m| m.q);
        Ok(Analysis { moves, value })
    }

    fn head_rows(&self, s: &SampleView, t: usize, _pos: &ToyPos, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>> {
        let (gf, gt) = (s.i64s("tgt.from_idx")?[t], s.i64s("tgt.to_idx")?[t]);
        let dist: Vec<((i64, i64, i64), f32)> = policy.iter().map(|&(a, p)| (decode_action(a), p)).collect();
        let mut rows = vec![
            HeadRow::Categorical(marginal(&dist, |a| a.0)),
            HeadRow::Categorical(conditional(&dist, |a| a.0 == gf, |a| a.1)),
            HeadRow::Categorical(conditional(&dist, |a| a.0 == gf && a.1 == gt, |a| a.2)),
        ];
        if self.bernoulli {
            rows.push(HeadRow::Bernoulli(dist.iter().filter(|(a, _)| a.2 > 0).map(|(_, p)| p).sum::<f32>().min(1.0)));
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::driver::{run, RunOpts};
    use crate::label::engine::EngineCommand;
    use crate::label::testkit::{overlay_tensor, tmpdir};
    use gmshard::{Column, DType};
    use std::time::Duration;

    fn opts(root: &Path, cmd: EngineCommand) -> RunOpts {
        let mut o = RunOpts::new(&root.join("base"), &root.join("out"), cmd);
        o.quiet = true;
        o.workers = 2;
        o
    }

    fn setup(tag: &str) -> std::path::PathBuf {
        let root = tmpdir(tag);
        write_shards(&root.join("base"), 2, 6, 42).unwrap();
        root
    }

    #[test]
    fn toy_rules_are_consistent() {
        for seed in 0..50u64 {
            let legal = legal_moves(seed);
            assert!(!legal.is_empty());
            let uniq: HashSet<_> = legal.iter().collect();
            assert_eq!(uniq.len(), legal.len());
            for &(f, t, p) in &legal {
                assert_eq!(decode_action(action_id(f, t, p)), (f, t, p));
            }
        }
        assert!((0..400u64).any(|s| legal_moves(s).len() == 1), "some forced positions");
        assert!((0..400u64).any(|s| legal_moves(s).iter().any(|m| m.2 > 0)), "some promotions");
    }

    #[test]
    fn end_to_end_labels() {
        let root = setup("toy-e2e");
        let s = run(ToyLabeller::default(), &opts(&root, fake_engine_command())).unwrap();
        assert_eq!((s.shards_total, s.shards_written, s.stats.samples), (2, 2, 12));
        assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis), (0, 0, 0));
        let forced = s.stats.skipped.get("forced").copied().unwrap_or(0);
        assert_eq!(s.stats.labelled + forced, s.stats.frames);
        assert_eq!(s.stats.queries + s.stats.cache_hits, s.stats.labelled);
        assert_eq!(s.teacher["engine"], "FakeUci 1.0");
        assert_eq!(s.teacher["heads"]["from"], json!({"kind": "categorical", "n": 64, "k": 8, "storage": "sparse"}));
        assert_eq!(s.teacher["heads"]["promotion"]["storage"], "dense");
        for shard in ["toy_00000", "toy_00001"] {
            let base = ShardReader::open(&root.join("base").join(shard)).unwrap();
            let out = ShardReader::open(&root.join("out").join(shard)).unwrap();
            assert_eq!(out.meta()["teacher"], s.teacher);
            assert_eq!(out.meta()["global_features"], 10);
            assert_eq!(out.index["teacher_stats"]["samples"], 6);
            for (name, info) in &base.tensors {
                assert_eq!(out.tensors[name], *info);
            }
            let k = 16;
            let idx = overlay_tensor(&out, "tgt.teacher_idx", DType::I64, &[k]);
            let q = overlay_tensor(&out, "tgt.teacher_q", DType::F32, &[k]);
            let raw = overlay_tensor(&out, "tgt.teacher_raw", DType::F32, &[k]);
            let ev = overlay_tensor(&out, "tgt.engine_value", DType::F32, &[]);
            let evm = overlay_tensor(&out, "tgt.engine_value_mask", DType::Bool, &[]);
            let fi = overlay_tensor(&out, "tgt.from_probs_idx", DType::I64, &[8]);
            let fv = overlay_tensor(&out, "tgt.from_probs_val", DType::F32, &[8]);
            let ti = overlay_tensor(&out, "tgt.to_probs_idx", DType::I64, &[16]);
            let tv = overlay_tensor(&out, "tgt.to_probs_val", DType::F32, &[16]);
            let pd = overlay_tensor(&out, "tgt.promotion_probs", DType::F32, &[5]);
            assert!(!out.has("tgt.teacher_visits"));
            let (Column::I64(idx), Column::F32(q), Column::F32(raw), Column::F32(ev), Column::Bool(evm)) = (&idx.col, &q.col, &raw.col, &ev.col, &evm.col)
            else { panic!() };
            let (Column::I64(fi), Column::F32(fv), Column::I64(ti), Column::F32(tv), Column::F32(pd)) = (&fi.col, &fv.col, &ti.col, &tv.col, &pd.col)
            else { panic!() };
            let (gf, gt, gp) = (base.read_i64("tgt.from_idx").unwrap(), base.read_i64("tgt.to_idx").unwrap(), base.read_i64("tgt.promotion").unwrap());
            for (si, sm) in base.samples.iter().enumerate() {
                let mut state = sm.meta["seed"].as_u64().unwrap();
                for f in sm.range() {
                    let legal = legal_moves(state);
                    let row = &idx[f * k..(f + 1) * k];
                    let qs = &q[f * k..(f + 1) * k];
                    if legal.len() == 1 {
                        // forced: all pads, no engine value
                        assert!(row.iter().all(|&x| x == -1) && qs.iter().all(|x| x.is_nan()) && evm[f] == 0 && ev[f] == 0.0);
                        assert!(fi[f * 8..(f + 1) * 8].iter().all(|&x| x == -1));
                        assert!(pd[f * 5..(f + 1) * 5].iter().all(|&x| x == 0.0));
                    } else {
                        assert_eq!(evm[f], 1, "shard {shard} sample {si} frame {f}");
                        let n = legal.len().min(k);
                        assert!(row[..n].iter().all(|&a| legal.contains(&decode_action(a))));
                        assert!(row[n..].iter().all(|&a| a == -1) && qs[n..].iter().all(|x| x.is_nan()));
                        assert!(qs[..n].windows(2).all(|w| w[0] >= w[1]), "sorted by Q");
                        assert!((ev[f] - (2.0 * qs[0] - 1.0)).abs() < 1e-6);
                        assert!(raw[f * k..f * k + n].iter().all(|x| (-300.0..=300.0).contains(x)));
                        // re-derive the policy from the raw columns and check every head row
                        let pi = crate::label::transform::softmax_q(&qs[..n], 0.05);
                        let dist: Vec<((i64, i64, i64), f32)> = row[..n].iter().zip(&pi).map(|(&a, &p)| (decode_action(a), p)).collect();
                        let mut from_m = marginal(&dist, |a| a.0);
                        from_m.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                        for (j, (c, p)) in from_m.iter().enumerate().take(8) {
                            assert_eq!(fi[f * 8 + j], *c);
                            assert!((fv[f * 8 + j] - p).abs() < 1e-5);
                        }
                        let to_c = conditional(&dist, |a| a.0 == gf[f], |a| a.1);
                        let to_sum: f32 = tv[f * 16..(f + 1) * 16].iter().sum();
                        assert!(if to_c.is_empty() { to_sum == 0.0 } else { (to_sum - 1.0).abs() < 1e-5 });
                        for (c, p) in &to_c {
                            let j = ti[f * 16..(f + 1) * 16].iter().position(|x| x == c).unwrap();
                            assert!((tv[f * 16 + j] - p).abs() < 1e-5);
                        }
                        let pr = conditional(&dist, |a| a.0 == gf[f] && a.1 == gt[f], |a| a.2);
                        let dense = &pd[f * 5..(f + 1) * 5];
                        for c in 0..5i64 {
                            let want = pr.iter().find(|x| x.0 == c).map(|x| x.1).unwrap_or(0.0);
                            assert!((dense[c as usize] - want).abs() < 1e-5);
                        }
                    }
                    state = next_state(state, action_id(gf[f], gt[f], gp[f]));
                }
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn deterministic_across_worker_counts_and_resumable() {
        let root = setup("toy-det");
        let mut o = opts(&root, fake_engine_command());
        o.workers = 1;
        run(ToyLabeller::default(), &o).unwrap();
        let mut o3 = o.clone();
        o3.workers = 3;
        o3.out = root.join("out3");
        o3.cache_size = 0;
        let s3 = run(ToyLabeller::default(), &o3).unwrap();
        assert_eq!(s3.stats.cache_hits, 0);
        for shard in ["toy_00000", "toy_00001"] {
            for f in std::fs::read_dir(root.join("out").join(shard)).unwrap() {
                let name = f.unwrap().file_name();
                if name == "index.json" {
                    continue;
                }
                let a = std::fs::read(root.join("out").join(shard).join(&name)).unwrap();
                let b = std::fs::read(root.join("out3").join(shard).join(&name)).unwrap();
                assert_eq!(a, b, "{shard}/{name:?} differs between 1 and 3 workers");
            }
        }
        // resume: everything exists → nothing written
        let again = run(ToyLabeller::default(), &o).unwrap();
        assert_eq!((again.shards_written, again.shards_existing), (0, 2));
        // a crashed run leaves temp debris, which is cleaned when that shard is redone
        std::fs::remove_dir_all(root.join("out").join("toy_00001")).unwrap();
        std::fs::create_dir_all(root.join("out").join(".toy_00001.tmp-1-2")).unwrap();
        let redo = run(ToyLabeller::default(), &o).unwrap();
        assert_eq!((redo.shards_written, redo.shards_existing), (1, 1));
        assert!(!root.join("out").join(".toy_00001.tmp-1-2").exists());
        // different settings into the same output are refused
        let mut other = o.clone();
        other.tau = 0.1;
        assert!(run(ToyLabeller::default(), &other).unwrap_err().to_string().contains("different settings"));
        // --part and --max-shards
        let mut p = o.clone();
        p.out = root.join("part");
        p.part = Some((1, 2));
        let sp = run(ToyLabeller::default(), &p).unwrap();
        assert_eq!(sp.shards_written, 1);
        assert!(root.join("part").join("toy_00001").exists() && !root.join("part").join("toy_00000").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn frame_sampling() {
        let root = setup("toy-sampling");
        let mut o = opts(&root, fake_engine_command());
        o.max_samples = Some(2);
        o.every_nth_frame = 2;
        let s = run(ToyLabeller::default(), &o).unwrap();
        assert!(s.stats.sampled_out > s.stats.labelled);
        assert_eq!(s.teacher["every_nth_frame"], 2);
        assert_eq!(s.teacher["max_samples"], 2);
        let out = ShardReader::open(&root.join("out").join("toy_00000")).unwrap();
        let evm = out.read_bool("tgt.engine_value_mask").unwrap();
        for (i, sm) in out.samples.iter().enumerate() {
            let m: Vec<bool> = evm[sm.range()].to_vec();
            if i >= 2 {
                assert!(m.iter().all(|x| !x), "sample {i} beyond --max-samples");
            } else {
                let phase = crate::label::driver::frame_phase("toy_00000", i, 2);
                for (t, &x) in m.iter().enumerate() {
                    if (t + phase) % 2 != 0 {
                        assert!(!x);
                    }
                }
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn engine_crashes_and_timeouts_are_retried() {
        let root = setup("toy-crash");
        // every engine process crashes on its 3rd search and hangs on its 2nd
        let cmd = EngineCommand::in_process("flaky", || fake_engine().crash_on(3).hang_on(2, Duration::from_millis(400)));
        let mut o = opts(&root, cmd);
        o.timeout = Duration::from_millis(150);
        let s = run(ToyLabeller::default(), &o).unwrap();
        assert!(s.stats.retries > 0 && s.stats.timeouts > 0, "{:?}", s.stats);
        assert_eq!(s.stats.engine_errors, 0, "a retry on a fresh process always succeeds here");
        let forced = s.stats.skipped.get("forced").copied().unwrap_or(0);
        assert_eq!(s.stats.labelled + forced, s.stats.frames);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn dead_engine_fails_fast() {
        let root = setup("toy-dead");
        // a working handshake, but every search crashes
        let cmd = EngineCommand::in_process("dead", || UciFake::new(|_: &str| None));
        let err = run(ToyLabeller::default(), &opts(&root, cmd)).unwrap_err().to_string();
        assert!(err.contains("systematically"), "{err}");
        assert!(!root.join("out").join("toy_00000").exists());
        // a missing binary fails before any shard is touched
        let missing = EngineCommand::process("/nonexistent/stockfish", &[]);
        let err = run(ToyLabeller::default(), &opts(&root, missing)).unwrap_err();
        assert!(format!("{err:#}").contains("start the engine"), "{err:#}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn verification_failures_are_counted_and_fail_the_run() {
        let root = tmpdir("toy-verify");
        let base = root.join("base");
        let mut w = ShardWriter::new(&base, GAME, CONVERTER_VERSION, json!({"global_features": GLOBALS}), 8).unwrap();
        for i in 0..8u64 {
            let mut s = sample(1000 + i);
            if i == 3 {
                // corrupt one observation value of one frame
                let g = s.tensors.iter_mut().find(|t| t.name == "obs.globals_seq").unwrap();
                let Column::F32(v) = &mut g.col else { panic!() };
                v[GLOBALS + 2] += 0.5;
            }
            w.push(s).unwrap();
        }
        w.flush().unwrap();
        let err = run(ToyLabeller::default(), &opts(&root, fake_engine_command())).unwrap_err().to_string();
        assert!(err.contains("1 of 8 samples failed"), "{err}");
        assert!(err.contains("obs.globals_seq: frame 1 element 2"), "{err}");
        // the shard is still a valid overlay with the bad sample unlabelled
        let out = ShardReader::open(&root.join("out").join("toy_00000")).unwrap();
        assert_eq!(out.index["teacher_stats"]["verify_failed"], 1);
        let evm = out.read_bool("tgt.engine_value_mask").unwrap();
        assert!(evm[out.sample_range(3)].iter().all(|x| !x));
        assert!(evm[out.sample_range(2)].iter().any(|x| *x));
        // a tolerant threshold lets the run pass
        let mut o = opts(&root, fake_engine_command());
        o.out = root.join("out2");
        o.max_verify_fail_frac = 0.2;
        assert_eq!(run(ToyLabeller::default(), &o).unwrap().stats.verify_failed, 1);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn visits_policy_and_bernoulli_heads() {
        let root = setup("toy-visits");
        let mut o = opts(&root, fake_engine_command());
        o.policy = Some(crate::label::transform::PolicyKind::Visits);
        o.max_moves = Some(4);
        let lab = ToyLabeller { visits: true, bernoulli: true };
        let s = run(lab, &o).unwrap();
        assert_eq!(s.teacher["policy"], json!({"kind": "visits"}));
        let out = ShardReader::open(&root.join("out").join("toy_00000")).unwrap();
        let vis = out.read_f32("tgt.teacher_visits").unwrap();
        let q = out.read_f32("tgt.teacher_q").unwrap();
        let pb = out.read_f32("tgt.promo_any_probs").unwrap();
        let pm = out.read_bool("tgt.promo_any_probs_mask").unwrap();
        let evm = out.read_bool("tgt.engine_value_mask").unwrap();
        assert_eq!(out.tensor_info("tgt.teacher_idx").unwrap().frame_shape, vec![4]);
        for f in 0..out.n_frames {
            assert_eq!(pm[f], evm[f]);
            assert!((0.0..=1.0).contains(&pb[f]));
            if evm[f] {
                let v = &vis[f * 4..f * 4 + 4];
                let n = v.iter().filter(|x| !x.is_nan()).count();
                assert!(v[..n].windows(2).all(|w| w[0] >= w[1]), "sorted by visits");
                for j in 0..n {
                    assert_eq!(v[j], (1000.0 * q[f * 4 + j] * q[f * 4 + j]).round());
                }
            } else {
                assert_eq!(pb[f], 0.0);
            }
        }
        // visits requested from a labeller without visits
        let mut bad = o.clone();
        bad.out = root.join("bad");
        assert!(run(ToyLabeller::default(), &bad).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
