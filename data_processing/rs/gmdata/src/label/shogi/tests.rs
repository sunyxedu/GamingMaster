//! Tests of the shogi labeller. The always-on ones use real floodgate games
//! (tests/fixtures/shogi, converted here with the real converter) and the in-process fake
//! engine; `real_yaneuraou_*` run the installed YaneuraOu (`GM_YANEURAOU=/path` or
//! `${GM_ENGINES:-~/gm-engines}/bin/yaneuraou`) and skip without it; `real_cluster_shards`
//! replays and verifies every shard under `$GM_SHOGI_SHARDS` (skipped when unset).

use super::*;
use crate::label::driver::{find_shards, run, RunOpts};
use crate::label::game::ShardData;
use crate::label::testkit::{engine_path, overlay_tensor, tmpdir};
use crate::label::fake::FakeMove;
use crate::label::transform::{softmax_q, softmax_q_tie, tie_scores, DEFAULT_TIE_WEIGHT};
use gmshard::{Column, DType, ShardWriter, Tensor};
use std::collections::HashMap;
use std::time::Duration;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/shogi")
}

/// The fixture games converted like `gmdata shogi convert` (16 games per shard).
fn fixture_shards(root: &Path) -> PathBuf {
    let base = root.join("base");
    let stats = rules::convert(&fixture_dir(), &base, 16, None, 30, false, &Default::default()).unwrap();
    assert!(stats["converted"].as_u64().unwrap() >= 30, "{stats}");
    base
}

fn load_all(base: &Path) -> Vec<ShardData> {
    find_shards(base, Path::new("/nonexistent")).unwrap().into_iter().map(|(d, _)| ShardData::load(ShardReader::open(&d).unwrap()).unwrap()).collect()
}

/// The board with owners cleared on empty squares (`Board::push` leaves them stale).
fn normalized(b: &Board) -> Board {
    let mut n = b.clone();
    for sq in 0..81 {
        if n.piece[sq] == NONE {
            n.color[sq] = 0;
        }
    }
    n
}

fn opts(root: &Path, cmd: EngineCommand) -> RunOpts {
    let mut o = RunOpts::new(&root.join("base"), &root.join("out"), cmd);
    o.quiet = true;
    o.workers = 3;
    o
}

#[test]
fn action_ids_round_trip() {
    assert_eq!((N_BOARD_ACTIONS, N_ACTIONS), (13122, 13689));
    for id in 0..N_ACTIONS {
        assert_eq!(Action::decode(id).unwrap().id(), id);
    }
    assert!(Action::decode(-1).is_none() && Action::decode(N_ACTIONS).is_none());
    assert_eq!(Action::Board { from: 60, to: 51, promote: false }.id(), (60 * 81 + 51) * 2);
    assert_eq!(Action::Drop { slot: 6, to: 80 }.id(), N_ACTIONS - 1);
    // white's moves are in white's POV: 3c3d (white) ≡ 7g7f (black)
    let black = parse_usi_move("7g7f").unwrap();
    let white = parse_usi_move("3c3d").unwrap();
    assert_eq!(Action::of(&black, true), Action::of(&white, false));
    assert_eq!(Action::of(&parse_usi_move("P*5e").unwrap(), true), Action::Drop { slot: 0, to: 40 });
    // white's R*1a: 1a = square 8, white POV 72
    assert_eq!(Action::of(&parse_usi_move("R*1a").unwrap(), false), Action::Drop { slot: 6, to: 72 });
}

#[test]
fn usi_and_sfen_notation() {
    let h = hirate_board();
    assert_eq!(sfen(&h), "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    assert_eq!(parse_position("position startpos").unwrap(), h);
    let m = parse_usi_move("8h2b+").unwrap();
    assert_eq!((m.from, m.to, m.promotion), (rules::csa_square("88"), rules::csa_square("22").unwrap(), true));
    assert_eq!(usi_move(&m), "8h2b+");
    assert_eq!(usi_move(&parse_usi_move("G*5b").unwrap()), "G*5b");
    for bad in ["", "7g7", "7g7f=", "0a1a", "7j7f", "K*5e", "p*5e", "P*5", "7g7f++"] {
        assert!(parse_usi_move(bad).is_none(), "{bad}");
    }
    // a sfen with hands, promoted pieces and white to move round-trips
    let s = "ln1g4l/1r1s1kg2/p2pppn+Bp/2p3p2/1p5P1/2P6/PPSPPPP1P/7R1/LN1GKGSNL w B2Pp 42";
    let b = parse_sfen(&s.split(' ').collect::<Vec<_>>()).unwrap();
    assert_eq!(sfen(&b), s);
    assert_eq!((b.hand[BLACK][rules::BISHOP as usize], b.hand[BLACK][PAWN as usize], b.hand[WHITE][PAWN as usize]), (1, 2, 1));
    assert!(parse_position("position startpos moves 7g7f 7g7f").is_err(), "illegal move");
    assert!(parse_sfen(&["9/9/9", "b", "-"]).is_err());
    // the converter's frame decoding inverts its encoding
    let mut board = vec![0f32; OBS_BOARD];
    let (mut hand, mut glb) = (vec![], vec![]);
    rules::encode_obs(&b, 0, &mut board, &mut hand, &mut glb);
    assert_eq!(rules::board_from_obs(&board, &hand, &glb).unwrap(), b);
}

#[test]
fn effective_options_merge() {
    let user = vec![("usi_hash".to_string(), "64".to_string()), ("MultiPV".into(), "3".into()), ("EvalDir".into(), "/x/eval".into())];
    let o = effective_options(&user, 8);
    assert_eq!(o.iter().find(|(k, _)| k == "USI_Hash").unwrap().1, "64", "overridden case-insensitively");
    assert_eq!(o.iter().filter(|(k, _)| k.eq_ignore_ascii_case("MultiPV")).collect::<Vec<_>>(), vec![&("MultiPV".to_string(), "8".to_string())]);
    assert_eq!(o.last().unwrap().0, "MultiPV");
    assert!(o.iter().any(|(k, v)| k == "EvalDir" && v == "/x/eval"));
    assert_eq!(o.len(), DEFAULT_OPTIONS.len() + 2);
    assert_eq!(meta_value("/x/eval"), "eval");
    // scores
    assert_eq!((raw_score(Score::Cp(-35)), raw_score(Score::Mate(5)), raw_score(Score::Mate(-4))), (-35.0, 99995.0, -99996.0));
    assert_eq!(raw_score(Score::Mate(crate::label::uci::MATE_UNKNOWN)), 90001.0);
    assert_eq!(raw_score(Score::Mate(-crate::label::uci::MATE_UNKNOWN)), -90001.0);
    assert_eq!((q_score(Score::Mate(3)), q_score(Score::Mate(-1)), q_score(Score::Cp(0))), (1.0, 0.0, 0.5));
    assert!((q_score(Score::Cp(600)) - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
}

/// Real floodgate games: every sample replays from its first frame + targets and
/// re-encodes bit-exactly (planes, hands, globals, masks, targets, values).
#[test]
fn fixture_games_replay_and_verify() {
    let root = tmpdir("shogi-replay");
    let base = fixture_shards(&root);
    let lab = ShogiLabeller::default();
    let (mut frames, mut drops, mut repeated, mut forced) = (0, 0, 0, 0);
    let mut line_of_key: HashMap<u128, String> = HashMap::new();
    for data in load_all(&base) {
        lab.check_shard(&data.reader).unwrap();
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            let pos = lab.replay(&v).unwrap();
            assert_eq!(pos.len(), v.len);
            lab.verify(&v, &pos).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
            assert_eq!(pos[0].game.start, "startpos", "floodgate games start from hirate");
            for p in &pos {
                frames += 1;
                drops += p.mv.from.is_none() as usize;
                forced += (lab.skip_reason(&v, p.ply, p) == Some("forced")) as usize;
                // the position line reproduces the board; the key identifies the line
                let line = p.position_line();
                let b = parse_position(&line).unwrap();
                assert_eq!(b.piece, p.board.piece);
                assert_eq!((b.hand, b.turn), (p.board.hand, p.board.turn));
                if let Some(prev) = line_of_key.insert(p.key, line.clone()) {
                    assert_eq!(prev, line, "key collision");
                    repeated += 1;
                }
                // every legal move round-trips through USI and the action ids
                for m in &p.legal {
                    assert_eq!(parse_usi_move(&usi_move(m)).as_ref(), Some(m));
                }
                assert!(p.is_legal_action(Action::of(&p.mv, p.mover_black()).id()));
                assert_eq!(p.legal_ids.len(), p.legal.len(), "distinct action ids");
                assert_eq!(normalized(&parse_sfen(&sfen(&p.board).split(' ').collect::<Vec<_>>()).unwrap()), normalized(&p.board));
            }
        }
    }
    assert!(frames > 3000 && drops > 100, "{frames} frames, {drops} drops");
    assert!(repeated > 0, "shared opening prefixes hit the cache");
    eprintln!("{frames} frames, {drops} drops, {repeated} repeated position lines, {forced} forced");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Corrupted shards fail verification, naming the tensor; illegal targets fail replay.
#[test]
fn corruption_is_detected() {
    let text = std::fs::read_to_string(fixture_dir().join("wdoor+floodgate-300-10F+910+AobaZero_w4513_n_p800+20250601070005.csa")).unwrap();
    let lab = ShogiLabeller::default();
    let cases: Vec<(&str, usize)> = vec![
        ("obs.board_seq", 3 * OBS_BOARD + 100),
        ("obs.hand_seq", 20 * 56 + 4),
        ("obs.globals_seq", 5 * 7 + 4),
        ("tgt.legal_from", 7 * 81 + 40),
        ("tgt.legal_to_of", 9 * 81 + 3),
        ("tgt.value", 2),
        ("tgt.to_idx", 11),
    ];
    for (name, at) in cases {
        let root = tmpdir("shogi-corrupt");
        let mut s = rules::convert_game(&text, 0, false).unwrap();
        let t = s.tensors.iter_mut().find(|t| t.name == name).unwrap();
        match &mut t.col {
            Column::F32(v) => v[at] = if v[at] == 0.0 { 1.0 } else { 0.0 },
            Column::Bits(v, per) => {
                let (frame, i) = (at / *per, at % *per);
                v[frame * per.div_ceil(8) + i / 8] ^= 0x80 >> (i % 8);
            }
            Column::I64(v) => v[at] = (v[at] + 1) % 81,
            c => panic!("{name}: {:?}", c.dtype()),
        }
        let mut w = ShardWriter::new(&root, "shogi", rules::CONVERTER_VERSION, json!({"global_features": 7}), 4).unwrap();
        w.push(s).unwrap();
        w.flush().unwrap();
        let data = ShardData::load(ShardReader::open(&root.join("shogi_00000")).unwrap()).unwrap();
        let v = data.view(0);
        let err = match lab.replay(&v) {
            Ok(pos) => lab.verify(&v, &pos).unwrap_err(),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        match name {
            "obs.board_seq" if at < OBS_BOARD => assert!(msg.contains("frame 0"), "{msg}"),
            // a changed destination is (almost surely) illegal → replay fails; else verify names it
            // another destination: illegal → replay fails; legal → the game diverges and a
            // later frame's planes (or this frame's masks) no longer match
            "tgt.to_idx" => assert!(msg.contains("not legal") || msg.contains("obs.") || msg.contains("tgt."), "{msg}"),
            _ => assert!(msg.contains(name), "{name}: {msg}"),
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// Re-derive the head rows of a frame from the raw columns (the documented contract:
/// π ∝ exp(Q/τ + w·u), u = [`secondary_of_raw`] of teacher_raw).
fn expected_rows(idx: &[i64], q: &[f32], raw: &[f32], gt: [i64; 6], tau: f32, w: f32) -> Vec<Vec<(i64, f32)>> {
    let n = idx.iter().take_while(|&&a| a >= 0).count();
    let u: Vec<f32> = raw[..n].iter().map(|&r| secondary_of_raw(r).unwrap()).collect();
    let pi = softmax_q_tie(&q[..n], &u, tau, w);
    expected_rows_of(idx, &pi, gt)
}

/// The head rows of a frame from its policy π over the first `π.len()` teacher moves.
fn expected_rows_of(idx: &[i64], pi: &[f32], gt: [i64; 6]) -> Vec<Vec<(i64, f32)>> {
    let n = pi.len();
    let dist: Vec<(Action, f32)> = idx[..n].iter().zip(pi).map(|(&a, &p)| (Action::decode(a).unwrap(), p)).collect();
    let [dropped, gf, gtt, _, gp, _] = gt;
    let mut rows = vec![Vec::new(); 6];
    let (mut tot_b, mut tot_d, mut tot_to, mut tot_pr, mut tot_dt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut acc: Vec<HashMap<i64, f64>> = vec![HashMap::new(); 6];
    for (a, p) in &dist {
        let p = *p as f64;
        match *a {
            Action::Board { from, to, promote } => {
                *acc[0].entry(0).or_default() += p;
                *acc[1].entry(from).or_default() += p;
                tot_b += p;
                if dropped == 0 && from == gf {
                    *acc[2].entry(to).or_default() += p;
                    tot_to += p;
                    if to == gtt {
                        *acc[3].entry(promote as i64).or_default() += p;
                        tot_pr += p;
                    }
                }
            }
            Action::Drop { slot, to } => {
                *acc[0].entry(1).or_default() += p;
                *acc[4].entry(slot).or_default() += p;
                tot_d += p;
                if dropped == 1 && slot == gp {
                    *acc[5].entry(to).or_default() += p;
                    tot_dt += p;
                }
            }
        }
    }
    for (h, tot) in [(0, 1.0), (1, tot_b), (2, tot_to), (3, tot_pr), (4, tot_d), (5, tot_dt)] {
        if tot > 0.0 {
            rows[h] = acc[h].iter().filter(|(_, p)| **p > 0.0).map(|(&c, &p)| (c, (p / tot) as f32)).collect();
            rows[h].sort_by_key(|r| r.0);
        }
    }
    rows
}

/// Dense or sparse stored row → sorted (class, p) pairs with p > 0.
fn stored_row(t: &Tensor, f: usize, sparse_idx: Option<&Tensor>) -> Vec<(i64, f32)> {
    let per = t.frame_len();
    let Column::F32(v) = &t.col else { panic!() };
    let vals = &v[f * per..(f + 1) * per];
    let mut row: Vec<(i64, f32)> = match sparse_idx {
        None => vals.iter().enumerate().filter(|(_, p)| **p > 0.0).map(|(c, p)| (c as i64, *p)).collect(),
        Some(it) => {
            let Column::I64(ids) = &it.col else { panic!() };
            ids[f * per..(f + 1) * per].iter().zip(vals).filter(|(c, _)| **c >= 0).map(|(c, p)| (*c, *p)).collect()
        }
    };
    row.sort_by_key(|r| r.0);
    row
}

#[test]
fn end_to_end_with_fake_engine() {
    let root = tmpdir("shogi-e2e");
    fixture_shards(&root);
    let summary = run(ShogiLabeller::default(), &opts(&root, fake_engine_command())).unwrap();
    let st = &summary.stats;
    assert_eq!((st.failed_samples(), st.engine_errors, st.bad_analysis), (0, 0, 0), "{:?}", st.errors);
    let forced = st.skipped.get("forced").copied().unwrap_or(0);
    assert_eq!(st.labelled + forced, st.frames);
    assert!(st.cache_hits > 0, "shared opening prefixes");
    assert_eq!(summary.teacher["engine"], "FakeUsi 1.0");
    assert_eq!(summary.teacher["limits"], json!({"nodes": "1000000"}));
    assert_eq!(summary.teacher["max_moves"], 8);
    assert_eq!(summary.teacher["raw_unit"], "cp");
    assert_eq!(summary.teacher["game"]["usi_options"]["FV_SCALE"], "20");
    assert_eq!(summary.teacher["game"]["usi_options"]["MultiPV"], "8");
    assert_eq!(summary.teacher["heads"]["from"], json!({"kind": "categorical", "n": 81, "k": 8, "storage": "sparse"}));
    assert_eq!(summary.teacher["heads"]["piece"]["storage"], "dense");
    assert_eq!((summary.teacher["labeller_version"].as_u64(), summary.teacher["core_version"].as_u64()), (Some(3), Some(2)));
    assert_eq!(summary.teacher["policy"]["tie_weight"], 2.0);
    assert_eq!(summary.teacher["policy"]["secondary"], ShogiLabeller::default().secondary_meta().unwrap());
    let lab = ShogiLabeller::default();
    let k = 8;
    let (mut mates, mut drop_rows, mut promo_rows, mut both_modes) = (0, 0, 0, 0);
    for (dir, rel) in find_shards(&root.join("base"), &root.join("out")).unwrap() {
        let out = ShardReader::open(&root.join("out").join(&rel)).unwrap();
        let data = ShardData::load(ShardReader::open(&dir).unwrap()).unwrap();
        let idx = overlay_tensor(&out, "tgt.teacher_idx", DType::I64, &[k]);
        let q = overlay_tensor(&out, "tgt.teacher_q", DType::F32, &[k]);
        let raw = overlay_tensor(&out, "tgt.teacher_raw", DType::F32, &[k]);
        let ev = overlay_tensor(&out, "tgt.engine_value", DType::F32, &[]);
        let evm = overlay_tensor(&out, "tgt.engine_value_mask", DType::Bool, &[]);
        let heads = [
            (overlay_tensor(&out, "tgt.mode_probs", DType::F32, &[2]), None),
            (overlay_tensor(&out, "tgt.from_probs_val", DType::F32, &[k]), Some(overlay_tensor(&out, "tgt.from_probs_idx", DType::I64, &[k]))),
            (overlay_tensor(&out, "tgt.to_probs_val", DType::F32, &[k]), Some(overlay_tensor(&out, "tgt.to_probs_idx", DType::I64, &[k]))),
            (overlay_tensor(&out, "tgt.promo_probs", DType::F32, &[2]), None),
            (overlay_tensor(&out, "tgt.piece_probs", DType::F32, &[14]), None),
            (overlay_tensor(&out, "tgt.drop_to_probs_val", DType::F32, &[k]), Some(overlay_tensor(&out, "tgt.drop_to_probs_idx", DType::I64, &[k]))),
        ];
        let (Column::I64(idx), Column::F32(q), Column::F32(raw), Column::F32(ev), Column::Bool(evm)) = (&idx.col, &q.col, &raw.col, &ev.col, &evm.col)
        else { panic!() };
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            let pos = lab.replay(&v).unwrap();
            let tg = targets(&v).unwrap();
            for (t, p) in pos.iter().enumerate() {
                let f = v.start + t;
                let (row, qs, rs) = (&idx[f * k..(f + 1) * k], &q[f * k..(f + 1) * k], &raw[f * k..(f + 1) * k]);
                if p.legal.len() == 1 {
                    assert!(evm[f] == 0 && row.iter().all(|&a| a == -1) && qs.iter().all(|x| x.is_nan()));
                    continue;
                }
                assert_eq!(evm[f], 1, "sample {i} frame {t}");
                let n = p.legal.len().min(k);
                assert!(row[..n].iter().all(|&a| p.is_legal_action(a)), "teacher moves are legal");
                assert!(row[n..].iter().all(|&a| a == -1) && qs[n..].iter().all(|x| x.is_nan()) && rs[n..].iter().all(|x| x.is_nan()));
                assert!(qs[..n].windows(2).all(|w| w[0] >= w[1]), "sorted by Q");
                let u: Vec<f32> = rs[..n].iter().map(|&r| secondary_of_raw(r).unwrap()).collect();
                assert!(tie_scores(&qs[..n], &u, 0.05, 2.0).windows(2).all(|w| w[0] >= w[1]), "sorted by Q/τ + w·u");
                assert!((ev[f] - (2.0 * qs[0] - 1.0)).abs() < 1e-6, "value = first MultiPV line = best here");
                for j in 0..n {
                    if rs[j].abs() > 90000.0 {
                        mates += 1;
                        assert_eq!(qs[j], if rs[j] > 0.0 { 1.0 } else { 0.0 });
                    } else {
                        assert!((qs[j] - q_from_cp(rs[j], CP_SCALE)).abs() < 1e-6);
                    }
                }
                let want = expected_rows(row, qs, rs, tg[t], 0.05, DEFAULT_TIE_WEIGHT as f32);
                for (h, (vals, ids)) in heads.iter().enumerate() {
                    let got = stored_row(vals, f, ids.as_ref());
                    assert_eq!(got.len(), want[h].len(), "head {h} sample {i} frame {t}: {got:?} vs {:?}", want[h]);
                    for ((gc, gp), (wc, wp)) in got.iter().zip(&want[h]) {
                        assert_eq!(gc, wc, "head {h}");
                        assert!((gp - wp).abs() < 1e-5, "head {h}: {gp} vs {wp}");
                    }
                }
                drop_rows += !want[5].is_empty() as usize;
                promo_rows += want[3].len();
                both_modes += (want[0].len() == 2) as usize;
            }
        }
    }
    assert!(mates > 0 && drop_rows > 0 && promo_rows > 0 && both_modes > 0, "{mates} {drop_rows} {promo_rows} {both_modes}");
    // resume: nothing left to do; other settings are refused
    let again = run(ShogiLabeller::default(), &opts(&root, fake_engine_command())).unwrap();
    assert_eq!(again.shards_written, 0);
    let mut other = opts(&root, fake_engine_command());
    other.engine_options = vec![("USI_Hash".into(), "64".into())];
    assert!(run(ShogiLabeller::default(), &other).unwrap_err().to_string().contains("different settings"));
    std::fs::remove_dir_all(&root).unwrap();
}

/// Secondary utility of a raw score, written out independently of the implementation:
/// USI mate plies → UCI-style moves, the chess mate / cp shape with cp scale 3000.
fn u_ref(raw: f32) -> f64 {
    let r = raw as f64;
    if r.abs() > 90_000.0 {
        let p = 100_000.0 - r.abs();
        let n = if r > 0.0 { (p + 1.0) / 2.0 } else { p / 2.0 };
        r.signum() * (0.5 + 2.5 / (n.max(1.0) + 4.0))
    } else {
        0.5 * (r / 3000.0).tanh()
    }
}

#[test]
fn secondary_utility_of_raw_scores() {
    let u = |s: Score| secondary_of_raw(raw_score(s)).unwrap();
    // mate in 1 ply = +1, mated in 2 plies (the fastest defeat after a move) = −1; the
    // chess utility of the same move count: mate in 3 plies = 2 moves, mated in 4 = 2 moves
    assert_eq!((u(Score::Mate(1)), u(Score::Mate(-2))), (1.0, -1.0));
    assert!((u(Score::Mate(3)) - (0.5 + 2.5 / 6.0)).abs() < 1e-7 && u(Score::Mate(-4)) == -u(Score::Mate(3)));
    assert_eq!((u(Score::Mate(0)), u(Score::Mate(-1))), (-1.0, -1.0), "impossible distances clamp to 1 move");
    assert_eq!(u(Score::Cp(0)), 0.0);
    assert!((u(Score::Cp(3000)) - 0.5 * 1f32.tanh()).abs() < 1e-7);
    // strictly increasing in teacher_raw over the distances a move can have (the mover
    // mates in odd plies, is mated in even ones) and the cp range; mates above all cp
    let mut scores: Vec<Score> = (1..=150).map(|n| Score::Mate(-2 * n)).collect();
    scores.extend((-15_000..=15_000).step_by(3).map(Score::Cp));
    scores.push(Score::Cp(35281)); // VALUE_MAX_EVAL
    scores.extend((0..150).rev().map(|n| Score::Mate(2 * n + 1)));
    let raws: Vec<f32> = scores.iter().map(|s| raw_score(*s)).collect();
    assert!(raws.windows(2).all(|w| w[0] < w[1]));
    let us: Vec<f32> = scores.iter().map(|s| u(*s)).collect();
    assert!(us.windows(2).all(|w| w[0] < w[1]), "strictly increasing");
    assert!(us.iter().all(|x| (-1.0..=1.0).contains(x)));
    assert!(u(Score::Cp(35281)) < 0.5 && u(Score::Mate(crate::label::uci::MATE_UNKNOWN)) > 0.5, "mate + (unknown distance) above every cp");
    for x in [-100_000.0, -99_997.0, -35_281.0, -1234.0, 0.0, 2500.0, 99_999.0, 90_001.0] {
        assert!((secondary_of_raw(x).unwrap() as f64 - u_ref(x)).abs() < 1e-7, "{x}");
    }
    assert!(secondary_of_raw(f32::NAN).is_none());
}

/// A fake USI engine whose moves are all mates: in a "won" position (hash parity) every
/// legal move mates in 1 … 29 plies (odd), in a "lost" one every move is mated in 2 … 30
/// plies (even) — all Q equal (1 or 0), only the mate distance tells them apart.
fn mating_engine() -> EngineCommand {
    EngineCommand::in_process("fake-usi-mates", || {
        FakeYaneuraOu(crate::label::fake::UciFake::new(|pos: &str| {
            let b = parse_position(pos).ok()?;
            let h0 = fnv(pos);
            let won = h0.is_multiple_of(2);
            Some(
                b.legal_moves()
                    .iter()
                    .map(|m| {
                        let u = usi_move(m);
                        let h = ((h0 ^ fnv(&u)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) % 15;
                        FakeMove::mate(&u, if won { 1 + 2 * h as i32 } else { -2 - 2 * h as i32 })
                    })
                    .collect(),
            )
        }))
    })
}

/// Saturated Q: every scored move mates (Q = 1) or is mated (Q = 0); π ∝ exp(w·u), best
/// first by u — faster mates and slower defeats first.
#[test]
fn saturated_q_follows_the_secondary_utility_end_to_end() {
    let root = tmpdir("shogi-tie");
    fixture_shards(&root);
    let mut o = opts(&root, mating_engine());
    o.max_shards = Some(1);
    o.max_samples = Some(4);
    let s = run(ShogiLabeller::default(), &o).unwrap();
    assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis), (0, 0, 0), "{:?}", s.stats.errors);
    let (dir, rel) = find_shards(&root.join("base"), &root.join("out")).unwrap().remove(0);
    let out = ShardReader::open(&root.join("out").join(&rel)).unwrap();
    let data = ShardData::load(ShardReader::open(&dir).unwrap()).unwrap();
    let (idx, q, raw, evm) =
        (out.read_i64("tgt.teacher_idx").unwrap(), out.read_f32("tgt.teacher_q").unwrap(), out.read_f32("tgt.teacher_raw").unwrap(), out.read_bool("tgt.engine_value_mask").unwrap());
    let heads = [
        (overlay_tensor(&out, "tgt.mode_probs", DType::F32, &[2]), None),
        (overlay_tensor(&out, "tgt.from_probs_val", DType::F32, &[8]), Some(overlay_tensor(&out, "tgt.from_probs_idx", DType::I64, &[8]))),
        (overlay_tensor(&out, "tgt.to_probs_val", DType::F32, &[8]), Some(overlay_tensor(&out, "tgt.to_probs_idx", DType::I64, &[8]))),
        (overlay_tensor(&out, "tgt.promo_probs", DType::F32, &[2]), None),
        (overlay_tensor(&out, "tgt.piece_probs", DType::F32, &[14]), None),
        (overlay_tensor(&out, "tgt.drop_to_probs_val", DType::F32, &[8]), Some(overlay_tensor(&out, "tgt.drop_to_probs_idx", DType::I64, &[8]))),
    ];
    let (mut won, mut lost, mut spread) = (0, 0, 0);
    for i in 0..4 {
        let v = data.view(i);
        for (t, gt) in targets(&v).unwrap().into_iter().enumerate() {
            let f = v.start + t;
            if !evm[f] {
                continue;
            }
            let n = idx[f * 8..f * 8 + 8].iter().filter(|&&a| a >= 0).count();
            let (qs, rs) = (&q[f * 8..f * 8 + n], &raw[f * 8..f * 8 + n]);
            assert!(qs.iter().all(|&x| x == qs[0]) && (qs[0] == 1.0 || qs[0] == 0.0), "all Q equal: {qs:?}");
            assert!(rs.windows(2).all(|w| w[0] >= w[1]), "best first by the raw score (= by u): {rs:?}");
            won += (qs[0] == 1.0) as usize;
            lost += (qs[0] == 0.0) as usize;
            // π ∝ exp(w·u), independently of the implementation
            let e: Vec<f64> = rs.iter().map(|&r| (2.0 * u_ref(r)).exp()).collect();
            let z: f64 = e.iter().sum();
            let pi: Vec<f32> = e.iter().map(|x| (x / z) as f32).collect();
            spread += (pi[0] > pi[n - 1] * 1.05) as usize;
            let want = expected_rows_of(&idx[f * 8..f * 8 + 8], &pi, gt);
            for (h, (vals, ids)) in heads.iter().enumerate() {
                let got = stored_row(vals, f, ids.as_ref());
                assert_eq!(got.len(), want[h].len(), "head {h} frame {f}");
                for ((gc, gp), (wc, wp)) in got.iter().zip(&want[h]) {
                    assert!(gc == wc && (gp - wp).abs() < 1e-5, "head {h} frame {f}: {got:?} vs {:?}", want[h]);
                }
            }
        }
    }
    assert!(won > 20 && lost > 20 && spread > 20, "{won} won, {lost} lost, {spread} frames with a spread teacher");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn engine_crashes_and_timeouts_are_retried() {
    let root = tmpdir("shogi-crash");
    fixture_shards(&root);
    // odd engine processes crash on their 12th search, even ones hang on it (→ timeout)
    let spawns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cmd = EngineCommand::in_process("flaky-usi", move || {
        let n = spawns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n % 2 == 1 { fake_engine().crash_on(12) } else { fake_engine().hang_on(12, Duration::from_millis(400)) }
    });
    let mut o = opts(&root, cmd);
    o.timeout = Duration::from_millis(150);
    o.max_shards = Some(1);
    o.max_samples = Some(4);
    let s = run(ShogiLabeller::new(Some(4)), &o).unwrap();
    assert!(s.stats.retries > 0 && s.stats.timeouts > 0, "{:?}", s.stats);
    assert_eq!(s.stats.engine_errors, 0);
    // same labels as an engine that never fails (answers do not depend on history)
    let mut clean = opts(&root, fake_engine_command());
    clean.out = root.join("clean");
    clean.max_shards = Some(1);
    clean.max_samples = Some(4);
    run(ShogiLabeller::new(Some(4)), &clean).unwrap();
    let name = std::fs::read_dir(root.join("clean")).unwrap().flatten().find(|e| !e.file_name().to_string_lossy().starts_with('.')).unwrap().file_name();
    for f in ["tgt.teacher_idx.npy", "tgt.teacher_q.npy", "tgt.from_probs_val.npy", "tgt.engine_value.npy"] {
        assert_eq!(std::fs::read(root.join("out").join(&name).join(f)).unwrap(), std::fs::read(root.join("clean").join(&name).join(f)).unwrap(), "{f}");
    }
    std::fs::remove_dir_all(&root).unwrap();
}

/// A handicap game (white without rook and bishop, white to move first) replays from its
/// first frame and is sent to the engine as `position sfen … moves …`.
#[test]
fn handicap_start_uses_sfen() {
    let text = "PI82HI22KA\n-\n-3334FU\n+7776FU\n-4344FU\n+2726FU\n-5142OU\n+2625FU\n%TORYO\n";
    let root = tmpdir("shogi-handicap");
    let mut w = ShardWriter::new(&root.join("base"), "shogi", rules::CONVERTER_VERSION, json!({"global_features": 7}), 4).unwrap();
    w.push(rules::convert_game(text, 0, false).unwrap()).unwrap();
    w.flush().unwrap();
    let lab = ShogiLabeller::default();
    let data = load_all(&root.join("base")).remove(0);
    let v = data.view(0);
    let pos = lab.replay(&v).unwrap();
    lab.verify(&v, &pos).unwrap();
    assert_eq!(pos[0].game.start, "sfen lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1");
    assert_eq!(pos[2].position_line(), "position sfen lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1 moves 3c3d 7g7f");
    let s = run(ShogiLabeller::default(), &opts(&root, fake_engine_command())).unwrap();
    assert_eq!((s.stats.labelled, s.stats.engine_errors, s.stats.bad_analysis), (6, 0, 0), "{:?}", s.stats.errors);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn declared_win_and_illegal_engine_moves() {
    // an engine that declares on every position, and one that only names an illegal move
    let root = tmpdir("shogi-odd");
    fixture_shards(&root);
    let lab = ShogiLabeller::new(Some(4));
    let data = load_all(&root.join("base")).remove(0);
    let v = data.view(0);
    let pos = lab.replay(&v).unwrap();
    let cfg = EngineConfig {
        command: EngineCommand::in_process("declare", || {
            crate::label::fake::Canned::new()
                .rule("usi", &["id name Declarer", "usiok"])
                .rule("isready", &["readyok"])
                .rule("go *", &["info depth 1 score mate + pv win", "bestmove win"])
        }),
        options: vec![],
        limits: lab.resolve_limits(&Limits::default()).unwrap(),
        timeout: Duration::from_secs(5),
        max_moves: 4,
    };
    let mut s = lab.open_session(&cfg).unwrap();
    assert_eq!(s.eval_file, None);
    assert!(format!("{:#}", lab.query(&mut s, &pos[0], &cfg).unwrap_err()).contains("declares a win"));
    let cfg2 = EngineConfig {
        command: EngineCommand::in_process("illegal", || {
            crate::label::fake::Canned::new()
                .rule("usi", &["id name Illegal", "usiok"])
                .rule("isready", &["readyok"])
                .rule("go *", &["info depth 1 multipv 1 score cp 50 pv 5e5d", "info depth 1 multipv 2 score cp 10 pv 7g7f", "bestmove 5e5d"])
        }),
        ..cfg.clone()
    };
    let mut s2 = lab.open_session(&cfg2).unwrap();
    let a = lab.query(&mut s2, &pos[0], &cfg2).unwrap();
    assert_eq!(a.moves.len(), 1, "5e5d dropped, 7g7f kept");
    assert_eq!(a.moves[0].action, Action::of(&parse_usi_move("7g7f").unwrap(), true).id());
    assert_eq!(a.value, None, "the first line was dropped → the driver takes the best kept Q");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Options the engine does not announce fail the session (YaneuraOu would only print
/// "No such option" and search with its default, while the meta records the setting).
#[test]
fn unannounced_options_are_refused() {
    let lab = ShogiLabeller::new(Some(4));
    // a YaneuraOu-like engine announcing every default option except FV_SCALE
    let engine = |name: &'static str| {
        EngineCommand::in_process(name, || {
            let mut hello: Vec<String> = vec!["id name Partial 1.0".into()];
            hello.extend(DEFAULT_OPTIONS.iter().filter(|(k, _)| *k != "FV_SCALE").map(|(k, _)| format!("option name {k} type string default x")));
            hello.push("option name MultiPV type spin default 1 min 1 max 600".into());
            hello.push("option name EvalDir type string default eval".into());
            hello.push("usiok".into());
            let hello: Vec<&str> = hello.iter().map(String::as_str).collect();
            crate::label::fake::Canned::new().rule("usi", &hello).rule("isready", &["readyok"])
        })
    };
    let cfg = |options: Vec<(String, String)>| EngineConfig {
        command: engine("partial"),
        options,
        limits: lab.resolve_limits(&Limits::default()).unwrap(),
        timeout: Duration::from_secs(5),
        max_moves: 4,
    };
    let err = format!("{:#}", lab.open_session(&cfg(vec![])).err().unwrap());
    assert!(err.contains("FV_SCALE") && err.contains("Partial 1.0"), "{err}");
    // FV_SCALE is a default: overriding it does not help; a typo is refused too
    assert!(lab.open_session(&cfg(vec![("fv_scale".into(), "16".into())])).is_err());
    let hello: Vec<String> = ["id name X", "option name Threads type spin default 4 min 1 max 1024", "option name Clear Hash type button", "usiok"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let opts = |ks: &[&str]| ks.iter().map(|k| (k.to_string(), "1".to_string())).collect::<Vec<_>>();
    assert_eq!(unknown_options(&hello, &opts(&["threads", "Clear Hash", "Hash"])), vec!["Hash"]);
    assert!(unknown_options(&hello[..1], &opts(&["Anything"])).is_empty(), "no options announced: not checked");
}

/// Shards under `$GM_SHOGI_SHARDS` (e.g. converter output scp'd from the cluster) replay
/// and verify completely.
#[test]
fn real_cluster_shards() {
    let Some(dir) = std::env::var_os("GM_SHOGI_SHARDS").map(PathBuf::from) else {
        eprintln!("skipping: GM_SHOGI_SHARDS not set");
        return;
    };
    let lab = ShogiLabeller::default();
    let (mut samples, mut frames) = (0, 0);
    for data in load_all(&dir) {
        lab.check_shard(&data.reader).unwrap();
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            let pos = lab.replay(&v).unwrap_or_else(|e| panic!("{} sample {i}: {e:#}", data.reader.dir.display()));
            lab.verify(&v, &pos).unwrap_or_else(|e| panic!("{} sample {i}: {e:#}", data.reader.dir.display()));
            samples += 1;
            frames += v.len;
        }
    }
    assert!(samples > 0, "no shogi shards under {}", dir.display());
    eprintln!("{samples} samples / {frames} frames replayed and verified bit-exactly");
}

fn yaneuraou() -> Option<PathBuf> {
    engine_path("GM_YANEURAOU", "yaneuraou")
}

/// Real YaneuraOu (skipped when not installed): labels two games end to end at a small
/// budget, deterministic across worker counts, every scored move legal, the teacher
/// ordered by (Q, raw) with heads = the tie-broken softmax of the stored columns.
#[test]
fn real_yaneuraou_end_to_end() {
    let Some(exe) = yaneuraou() else {
        eprintln!("skipping: yaneuraou not found (scripts/engines/fetch_engines.sh yaneuraou)");
        return;
    };
    let root = tmpdir("shogi-yo");
    fixture_shards(&root);
    let mut o = opts(&root, EngineCommand::process(&exe, &[]));
    o.limits = Limits::parse(&["nodes=20000".into()]).unwrap();
    o.max_shards = Some(1);
    o.max_samples = Some(2);
    o.workers = 2;
    let s = run(ShogiLabeller::default(), &o).unwrap();
    assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis), (0, 0, 0), "{:?}", s.stats.errors);
    let engine = s.teacher["engine"].as_str().unwrap();
    assert!(engine.starts_with("YaneuraOu") && engine.contains("+ eval ("), "{engine}");
    let mut o1 = o.clone();
    o1.workers = 1;
    o1.out = root.join("out1");
    run(ShogiLabeller::default(), &o1).unwrap();
    let (dir, rel) = find_shards(&root.join("base"), &root.join("out")).unwrap().remove(0);
    for f in ["tgt.teacher_idx.npy", "tgt.teacher_q.npy", "tgt.teacher_raw.npy", "tgt.engine_value.npy", "tgt.from_probs_val.npy"] {
        assert_eq!(std::fs::read(root.join("out").join(&rel).join(f)).unwrap(), std::fs::read(root.join("out1").join(&rel).join(f)).unwrap(), "{f}: 1 vs 2 workers");
    }
    let out = ShardReader::open(&root.join("out").join(&rel)).unwrap();
    let (idx, q, evm) = (out.read_i64("tgt.teacher_idx").unwrap(), out.read_f32("tgt.teacher_q").unwrap(), out.read_bool("tgt.engine_value_mask").unwrap());
    let ev = out.read_f32("tgt.engine_value").unwrap();
    let data = ShardData::load(ShardReader::open(&dir).unwrap()).unwrap();
    let lab = ShogiLabeller::default();
    let raw = out.read_f32("tgt.teacher_raw").unwrap();
    let heads = [
        (overlay_tensor(&out, "tgt.mode_probs", DType::F32, &[2]), None),
        (overlay_tensor(&out, "tgt.from_probs_val", DType::F32, &[8]), Some(overlay_tensor(&out, "tgt.from_probs_idx", DType::I64, &[8]))),
        (overlay_tensor(&out, "tgt.to_probs_val", DType::F32, &[8]), Some(overlay_tensor(&out, "tgt.to_probs_idx", DType::I64, &[8]))),
        (overlay_tensor(&out, "tgt.promo_probs", DType::F32, &[2]), None),
        (overlay_tensor(&out, "tgt.piece_probs", DType::F32, &[14]), None),
        (overlay_tensor(&out, "tgt.drop_to_probs_val", DType::F32, &[8]), Some(overlay_tensor(&out, "tgt.drop_to_probs_idx", DType::I64, &[8]))),
    ];
    let (mut scored, mut labelled, mut above_first, mut tied) = (0, 0, 0, 0);
    for i in 0..2 {
        let v = data.view(i);
        let tg = targets(&v).unwrap();
        for (t, p) in lab.replay(&v).unwrap().iter().enumerate() {
            let f = v.start + t;
            if !evm[f] {
                continue;
            }
            let row = &idx[f * 8..(f + 1) * 8];
            let n = row.iter().filter(|&&a| a >= 0).count();
            assert!(n >= 1 && n <= p.legal.len().min(8));
            assert!(row[..n].iter().all(|&a| p.is_legal_action(a)));
            let (qs, rs) = (&q[f * 8..f * 8 + n], &raw[f * 8..f * 8 + n]);
            assert!(qs.iter().all(|x| (0.0..=1.0).contains(x)));
            scored += n;
            labelled += 1;
            // a move scored above the first MultiPV line (the value) = a stale line left uncapped
            above_first += (ev[f] < 2.0 * q[f * 8] - 1.0 - 1e-6) as usize;
            // the tie-broken teacher on real scores: best first by (Q, raw), heads = π ∝
            // exp(Q/τ + w·u) re-derived with the independent formula
            for j in 1..n {
                assert!(qs[j - 1] > qs[j] || (qs[j - 1] == qs[j] && rs[j - 1] >= rs[j]), "frame {f}: {qs:?} {rs:?}");
            }
            tied += (1..n).any(|j| qs[j] == qs[j - 1] && rs[j] != rs[j - 1]) as usize;
            let e: Vec<f64> = qs.iter().zip(rs).map(|(&x, &r)| ((x as f64 - qs[0] as f64) / 0.05 + 2.0 * u_ref(r)).exp()).collect();
            let z: f64 = e.iter().sum();
            let pi: Vec<f32> = e.iter().map(|x| (x / z) as f32).collect();
            let want = expected_rows_of(row, &pi, tg[t]);
            for (h, (vals, ids)) in heads.iter().enumerate() {
                let got = stored_row(vals, f, ids.as_ref());
                assert_eq!(got.len(), want[h].len(), "head {h} frame {f}");
                for ((gc, gp), (wc, wp)) in got.iter().zip(&want[h]) {
                    assert!(gc == wc && (gp - wp).abs() < 1e-5, "head {h} frame {f}: {got:?} vs {:?}", want[h]);
                }
            }
        }
    }
    assert!(scored > 100);
    eprintln!("{labelled} frames, {above_first} with a move above the first line, {tied} with a Q tie split by the raw score");
    assert!(tied > 0, "the games' mating attacks give Q ties (mates) that the raw score splits");
    assert!(above_first * 100 <= labelled, "{above_first} of {labelled} frames rank a move above the engine's first line");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Frames of the fixture games (position lines), for the throughput / budget probes.
fn fixture_positions(n: usize) -> Vec<(String, Vec<i64>, bool)> {
    let root = tmpdir("shogi-probe");
    let base = fixture_shards(&root);
    let lab = ShogiLabeller::default();
    let mut out = Vec::new();
    'outer: for data in load_all(&base) {
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            for (t, p) in lab.replay(&v).unwrap().into_iter().enumerate() {
                // every 7th frame of every game: openings, middlegames and endgames
                if t % 7 == 3 && p.legal.len() > 1 {
                    out.push((p.position_line(), p.legal_ids.clone(), p.mover_black()));
                    if out.len() == n {
                        break 'outer;
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
    out
}

fn yo_session(exe: &Path, multipv: usize) -> (ShogiSession, EngineConfig) {
    let lab = ShogiLabeller::new(Some(multipv));
    let cfg = EngineConfig {
        command: EngineCommand::process(exe, &[]),
        options: vec![],
        limits: lab.resolve_limits(&Limits::default()).unwrap(),
        timeout: Duration::from_secs(600),
        max_moves: multipv,
    };
    (lab.open_session(&cfg).unwrap(), cfg)
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Positions/s of one single-threaded YaneuraOu at labelling settings. Opt-in:
/// `cargo test --release -p gmdata real_yaneuraou_throughput -- --ignored --nocapture`
/// (`YO_NODES`, `YO_MULTIPV`, `YO_POSITIONS` override 1000000 / 8 / 40; `YO_DUMP=FILE` also
/// writes the position lines). Wall-clock numbers: run it on an otherwise idle machine.
#[test]
#[ignore]
fn real_yaneuraou_throughput() {
    let Some(exe) = yaneuraou() else {
        eprintln!("skipping: yaneuraou not found");
        return;
    };
    let (nodes, multipv, n): (u64, usize, usize) = (env_or("YO_NODES", 1_000_000), env_or("YO_MULTIPV", 8), env_or("YO_POSITIONS", 40));
    let positions = fixture_positions(n);
    if let Some(dump) = std::env::var_os("YO_DUMP") {
        // the position lines, for benchmarking the engine outside the test
        let text: String = positions.iter().map(|(l, _, _)| format!("{l}\n")).collect();
        std::fs::write(dump, text).unwrap();
    }
    let (mut s, cfg) = yo_session(&exe, multipv);
    let t0 = std::time::Instant::now();
    let (mut lines, mut depth, mut searched) = (0usize, 0u64, 0u64);
    for (line, _, _) in &positions {
        s.uci.new_game().unwrap();
        let r = s.uci.search(line, &format!("go nodes {nodes}"), cfg.timeout).unwrap();
        lines += r.lines.len();
        depth += r.lines.first().and_then(|l| l.depth).unwrap_or(0) as u64;
        searched += r.lines.iter().filter_map(|l| l.nodes).max().unwrap_or(0);
    }
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "{}: {} positions, nodes {nodes}, MultiPV {multipv}: {:.2} positions/s per process, {:.2} M nodes/s, depth {:.1}, {:.1} lines/position",
        s.uci.id_name,
        positions.len(),
        positions.len() as f64 / secs,
        searched as f64 / secs / 1e6,
        depth as f64 / positions.len() as f64,
        lines as f64 / positions.len() as f64
    );
}

/// Teacher quality of smaller budgets against a large one, on the labelled transform
/// (MultiPV 8, softmax(Q/0.05)): best-move agreement, total-variation distance between the
/// two teacher policies, and the reference's Q loss of the smaller budget's best move (a move
/// outside the reference's 8 lines counts as the reference's lowest line). Opt-in:
/// `cargo test --release -p gmdata real_yaneuraou_budget_agreement -- --ignored --nocapture`
/// (`YO_POSITIONS` default 50, `YO_BUDGETS` default "100000,250000,1000000", `YO_REF_NODES`
/// default 4000000).
#[test]
#[ignore]
fn real_yaneuraou_budget_agreement() {
    let Some(exe) = yaneuraou() else {
        eprintln!("skipping: yaneuraou not found");
        return;
    };
    let n: usize = env_or("YO_POSITIONS", 50);
    let reference: u64 = env_or("YO_REF_NODES", 4_000_000);
    let mut budgets: Vec<u64> = std::env::var("YO_BUDGETS")
        .unwrap_or_else(|_| "100000,250000,1000000".into())
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();
    budgets.push(reference);
    let positions = fixture_positions(n);
    let (mut s, cfg) = yo_session(&exe, DEFAULT_MAX_MOVES);
    // per budget and position: the teacher policy {move: p}, the moves' Q, the best move
    type Run = (HashMap<String, f64>, HashMap<String, f32>, String);
    let mut runs: Vec<Vec<Run>> = vec![Vec::new(); budgets.len()];
    let mut secs = vec![0.0; budgets.len()];
    for (line, _, _) in &positions {
        for (b, &nodes) in budgets.iter().enumerate() {
            s.uci.new_game().unwrap();
            let t0 = std::time::Instant::now();
            let r = s.uci.search(line, &format!("go nodes {nodes}"), cfg.timeout).unwrap();
            secs[b] += t0.elapsed().as_secs_f64();
            let q: Vec<f32> = r.lines.iter().map(|l| q_score(l.score.unwrap())).collect();
            let pi = softmax_q(&q, 0.05);
            let mv: Vec<String> = r.lines.iter().map(|l| l.pv[0].clone()).collect();
            runs[b].push((
                mv.iter().cloned().zip(pi.iter().map(|&p| p as f64)).collect(),
                mv.iter().cloned().zip(q.iter().copied()).collect(),
                mv.first().cloned().unwrap_or_default(),
            ));
        }
    }
    let r = budgets.len() - 1;
    for (b, &nodes) in budgets.iter().enumerate().take(r) {
        let (mut agree, mut tv, mut regret) = (0usize, 0.0f64, 0.0f64);
        for (x, y) in runs[b].iter().zip(&runs[r]) {
            agree += (x.2 == y.2) as usize;
            let keys: std::collections::BTreeSet<&String> = x.0.keys().chain(y.0.keys()).collect();
            tv += 0.5 * keys.iter().map(|k| (x.0.get(*k).unwrap_or(&0.0) - y.0.get(*k).unwrap_or(&0.0)).abs()).sum::<f64>();
            let q_best = y.1[&y.2];
            let q_floor = y.1.values().copied().fold(f32::INFINITY, f32::min);
            regret += (q_best - y.1.get(&x.2).copied().unwrap_or(q_floor)) as f64;
        }
        let m = positions.len() as f64;
        eprintln!(
            "nodes {nodes:>8} vs {reference}: best move agrees {:.0}%, policy TV distance {:.3}, Q loss of its best move {:.4}, {:.2} positions/s",
            100.0 * agree as f64 / m,
            tv / m,
            regret / m,
            m / secs[b]
        );
    }
}

/// A frame of a `position startpos moves …` line (recorded move: a placeholder).
fn pos_from_line(line: &str) -> ShogiPos {
    let toks: Vec<&str> = line.split_whitespace().collect();
    assert_eq!((toks[0], toks[1], toks[2]), ("position", "startpos", "moves"));
    let usi: Vec<String> = toks[3..].iter().map(|s| s.to_string()).collect();
    let board = parse_position(line).unwrap();
    let legal = board.legal_moves();
    let mb = board.turn == BLACK;
    let mut legal_ids: Vec<i64> = legal.iter().map(|m| Action::of(m, mb).id()).collect();
    legal_ids.sort_unstable();
    let ply = usi.len();
    ShogiPos { game: Arc::new(GameLine { start: "startpos".into(), usi }), ply, board, mv: legal[0], legal, legal_ids, key: 0 }
}

/// A real floodgate frame (white to move) where YaneuraOu 9.40 + Háo stopped at 1M nodes
/// inside iteration 10: its first line failed low there, the other seven still carry their
/// iteration-9 scores. An 8M-node search scores the stale leader 3h3d+ at ≤ −1235 cp.
const STALE_FRAME: &str = "position startpos moves 7g7f 8c8d 2g2f 8d8e 8h7g 3c3d 7i8h 2b7g+ 8h7g 3a2b 6i7h 4a3b \
    9g9f 9c9d 1g1f 1c1d 3g3f 2b3c 3i3h 6c6d 4g4f 5a4b 2i3g 7a7b 3h4g 7c7d 4i4h 8a7c 5i6h 7d7e 7f7e 7c6e 7g8h 8e8f 8g8f \
    8b8f P*8g 8f7f 2h2i 3d3e 3g4e 3e3f 8i7g 3f3g+ 4h3g B*4h 7g6e 4h3g+ 6e5c+ 4b3a 4e3c+ 2a3c N*3d 3g4g S*2b 3a4a 7h7g \
    4g2i 7g7f R*3h 6h7g";
/// Its final block (YaneuraOu prints it twice; the second, with the completed depth, wins).
const STALE_OUTPUT: &[&str] = &[
    "info depth 10 seldepth 26 multipv 1 score cp -1270 upperbound nodes 1000734 nps 1235474 hashfull 409 time 810 pv G*7h 7g8f",
    "info depth 9 seldepth 14 multipv 2 score cp 346 nodes 1000734 nps 1235474 hashfull 409 time 810 pv 3h3d+ R*2a P*3a",
    "info depth 9 seldepth 26 multipv 1 score cp -1270 upperbound nodes 1000734 nps 1235474 hashfull 409 time 810 pv G*7h 7g8f",
    "info depth 8 seldepth 14 multipv 2 score cp 346 nodes 1000734 nps 1235474 hashfull 409 time 810 pv 3h3d+ R*2a P*3a 2a1a+",
    "info depth 8 seldepth 18 multipv 3 score cp 168 nodes 1000734 nps 1235474 hashfull 409 time 810 pv S*6h 7g8f 3h3d+",
    "info depth 8 seldepth 24 multipv 4 score cp -335 nodes 1000734 nps 1235474 hashfull 409 time 810 pv N*6e 7g8f 3h3d+",
    "info depth 8 seldepth 15 multipv 5 score cp -367 nodes 1000734 nps 1235474 hashfull 409 time 810 pv N*8e 7f8e G*7h",
    "info depth 8 seldepth 16 multipv 6 score cp -400 nodes 1000734 nps 1235474 hashfull 409 time 810 pv 3h3d R*2a P*3a",
    "info depth 8 seldepth 12 multipv 7 score cp -794 nodes 1000734 nps 1235474 hashfull 409 time 810 pv G*5b 2b3c 3h3d+",
    "info depth 8 seldepth 12 multipv 8 score cp -1366 nodes 1000734 nps 1235474 hashfull 409 time 810 pv 6a5b 2b3c+ N*6e",
    "bestmove G*7h ponder 7g8f",
];

/// Lines the node limit left stale (shallower than the deepest) are capped at the lowest
/// deepest line: the engine's best move keeps the top of the teacher instead of the refuted
/// stale leader getting nearly all of the mass.
#[test]
fn stale_lines_are_capped() {
    let pos = pos_from_line(STALE_FRAME);
    assert!(!pos.mover_black());
    let lines: Vec<String> = STALE_OUTPUT.iter().map(|s| s.to_string()).collect();
    let r = crate::label::uci::collect(&lines).unwrap();
    let a = analysis(&pos, &r.lines).unwrap();
    let id = |u: &str| Action::of(&parse_usi_move(u).unwrap(), false).id();
    let q = |cp| q_score(Score::Cp(cp));
    let got: HashMap<i64, (f32, f32)> = a.moves.iter().map(|m| (m.action, (m.q, m.raw))).collect();
    assert_eq!(got.len(), 8, "all eight lines are legal moves");
    assert_eq!(a.value, Some(q(-1270)), "value = the first line");
    assert_eq!(got[&id("G*7h")], (q(-1270), -1270.0));
    for u in ["3h3d+", "S*6h", "N*6e", "N*8e", "3h3d", "G*5b"] {
        assert_eq!(got[&id(u)], (q(-1270), -1270.0), "{u}: stale score capped");
    }
    assert_eq!(got[&id("6a5b")], (q(-1366), -1366.0), "already below the cap");
    // the policy: without the cap the stale 3h3d+ / S*6h would take > 95% of the mass and the
    // engine's choice G*7h almost none
    let raw_q: Vec<f32> = [-1270, 346, 168, -335, -367, -400, -794, -1366].iter().map(|&c| q(c)).collect();
    let uncapped = softmax_q(&raw_q, 0.05);
    assert!(uncapped[1] + uncapped[2] > 0.95 && uncapped[0] < 1e-4, "{uncapped:?}");
    let capped: Vec<f32> = a.moves.iter().map(|m| m.q).collect();
    let p = softmax_q(&capped, 0.05);
    let best = a.moves.iter().position(|m| m.action == id("G*7h")).unwrap();
    assert!(p.iter().all(|&x| x <= p[best] + 1e-7), "no line above the engine's choice: {p:?}");
    // lines of one depth are left alone
    let same: Vec<String> = ["info depth 5 multipv 1 score cp 20 pv 3h3d+", "info depth 5 multipv 2 score cp 90 pv S*6h", "bestmove 3h3d+"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let b = analysis(&pos, &crate::label::uci::collect(&same).unwrap().lines).unwrap();
    assert_eq!(b.moves.iter().map(|m| m.raw).collect::<Vec<_>>(), vec![20.0, 90.0]);
    assert_eq!(b.value, Some(q(20)));
}

/// teacher_raw: mates lie beyond every cp YaneuraOu prints (its evaluations saturate at
/// VALUE_MAX_EVAL = 31753 → 35281 cp, seen on real games), so |raw| > 90000 ⇔ mate.
#[test]
fn raw_mates_do_not_overlap_cp() {
    let max_cp = 100 * (32000 - 246 - 1) / 90;
    assert_eq!(max_cp, 35281);
    for n in [1, 17, 245, crate::label::uci::MATE_UNKNOWN] {
        assert!(raw_score(Score::Mate(n)) > 90000.0 && raw_score(Score::Mate(-n)) < -90000.0, "{n}");
        assert!(raw_score(Score::Mate(n)) >= raw_score(Score::Mate(n + 1)), "faster mates rank higher");
        assert!(n >= MATE_PLY_CAP || raw_score(Score::Mate(n)) > raw_score(Score::Mate(n + 1)));
    }
    assert_eq!(raw_score(Score::Mate(0)), -MATE_RAW, "mate -0 (mated now)");
    assert_eq!((raw_score(Score::Cp(max_cp)), raw_score(Score::Cp(-max_cp))), (35281.0, -35281.0));
}
