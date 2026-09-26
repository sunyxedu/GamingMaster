//! Othello labeller tests: rules / notation / protocol parsing, replay + verification
//! of converter shards (synthetic WTHOR games and — when present — a real WTHOR shard),
//! end-to-end labelling with the in-process fake Egaroucid, and real-engine integration
//! tests that skip when Egaroucid is not installed.

use super::*;
use crate::label::driver::{run, RunOpts};
use crate::label::testkit::{engine_path, overlay_tensor, quiet_opts, tmpdir};
use crate::label::transform::softmax_q;
use gmshard::{Column, DType};
use std::path::Path;

// ------------------------------------------------------------------ helpers

/// A random legal game (WTHOR move list: passes are implicit).
fn random_game(seed: u64) -> Vec<(i32, i32)> {
    let mut pos = OthPos::start();
    let mut moves = Vec::new();
    let mut r = seed;
    loop {
        let legal: Vec<i64> = (0..64).filter(|&s| pos.is_legal(s)).collect();
        r = splitmix(r);
        let sq = legal[(r % legal.len() as u64) as usize];
        moves.push(((sq / 8) as i32, (sq % 8) as i32));
        match pos.after(sq) {
            Some(n) => pos = n,
            None => return moves,
        }
    }
}

/// Whether a WTHOR move list contains an (implicit) pass.
fn has_pass(moves: &[(i32, i32)]) -> bool {
    let mut pos = OthPos::start();
    for &(r, c) in moves {
        let sq = (r * 8 + c) as i64;
        let next = pos.after(sq);
        if let Some(n) = &next {
            if n.mover == pos.mover {
                return true;
            }
        }
        match next {
            Some(n) => pos = n,
            None => break,
        }
    }
    false
}

/// A `.wtb` database of `games` (the layout `crate::othello::parse_wtb` reads).
fn wtb(games: &[Vec<(i32, i32)>]) -> Vec<u8> {
    let mut d = vec![0u8; 16];
    d[4..8].copy_from_slice(&(games.len() as u32).to_le_bytes());
    d[10..12].copy_from_slice(&2001u16.to_le_bytes());
    d[12] = 8;
    for g in games {
        let mut rec = vec![0u8; 68];
        for (i, &(r, c)) in g.iter().enumerate() {
            rec[8 + i] = (10 * (r + 1) + (c + 1)) as u8;
        }
        d.extend(rec);
    }
    d
}

/// `n` random games (at least 3 with a pass) converted by `gmdata othello convert`
/// into shards of `per_shard` games under `root/base`.
fn converter_shards(root: &Path, n: usize, per_shard: usize, seed: u64) {
    let mut games = Vec::new();
    let mut with_pass = 0;
    let mut s = seed;
    while games.len() < n || with_pass < 3 {
        s = splitmix(s);
        let g = random_game(s);
        let p = has_pass(&g);
        if games.len() < n || p {
            with_pass += p as usize;
            games.push(g);
        }
    }
    let wthor = root.join("wthor");
    std::fs::create_dir_all(&wthor).unwrap();
    std::fs::write(wthor.join("WTH_2001.wtb"), wtb(&games)).unwrap();
    let r = crate::othello::convert(&wthor, &root.join("base"), per_shard, None, None, None).unwrap();
    assert_eq!(r["illegal"], 0, "{r}");
    assert_eq!(r["converted"].as_u64().unwrap() as usize, games.len());
}

/// Every sample of a shard as (sample meta, replayed positions), after verification.
fn replay_shard(dir: &Path) -> Vec<(Value, Vec<OthPos>)> {
    let lab = OthelloLabeller::default();
    let reader = ShardReader::open(dir).unwrap();
    lab.check_shard(&reader).unwrap();
    let data = ShardData::load(reader).unwrap();
    (0..data.reader.n_samples())
        .map(|i| {
            let v = data.view(i);
            let ps = lab.replay(&v).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
            lab.verify(&v, &ps).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
            (v.meta().clone(), ps)
        })
        .collect()
}

/// Exact negamax: best final disc difference for the mover.
fn solve(pos: &OthPos) -> i32 {
    if pos.legal == 0 {
        let opp = OthPos::new(pos.board, crate::othello::opponent(pos.mover));
        return if opp.legal == 0 { pos.final_diff() } else { -solve(&opp) };
    }
    (0..64).filter(|&s| pos.is_legal(s)).map(|s| solve_move(pos, s)).max().unwrap()
}

/// Exact value of `sq` for the mover of `pos`.
fn solve_move(pos: &OthPos, sq: i64) -> i32 {
    let next = OthPos::new(play(&pos.board, pos.mover, sq as usize), crate::othello::opponent(pos.mover));
    -solve(&next)
}

fn opts(root: &Path, cmd: EngineCommand) -> RunOpts {
    quiet_opts(root, cmd)
}

// ------------------------------------------------------------------ rules, notation, protocol

#[test]
fn notation_and_board_strings() {
    for sq in 0..64 {
        assert_eq!(parse_square(&square_name(sq)), Some(sq));
    }
    assert_eq!((square_name(37), parse_square("F5"), parse_square("a1"), parse_square("h8")), ("f5".into(), Some(37), Some(0), Some(63)));
    assert_eq!((parse_square("i1"), parse_square("a9"), parse_square("ps")), (None, None, None));
    let start = OthPos::start();
    assert_eq!(start.to_egaroucid(), "---------------------------OX------XO---------------------------X");
    assert_eq!(OthPos::from_egaroucid(&start.to_egaroucid()), Some(start.clone()));
    // black's four opening moves d3 c4 f5 e6
    let legal: Vec<String> = (0..64).filter(|&s| start.is_legal(s)).map(square_name).collect();
    assert_eq!(legal, ["d3", "c4", "f5", "e6"]);
    let after = start.after(37).unwrap();
    assert_eq!((after.mover, after.empties(), after.n_legal()), (WHITE, 59, 3));
    assert_eq!(after.to_egaroucid().chars().last(), Some('O'));
    assert!(OthPos::from_egaroucid("X").is_none() && OthPos::from_egaroucid(&"-".repeat(65)).is_none());
    // final disc difference: empties go to the winner
    let mut b = [0u8; 64];
    b[0] = BLACK;
    b[1] = BLACK;
    b[2] = WHITE;
    let p = OthPos::new(b, BLACK);
    assert_eq!(p.final_diff(), 2 - 1 + 61);
    assert_eq!(OthPos::new(b, WHITE).final_diff(), -62);
}

#[test]
fn hint_rows_parse_real_egaroucid_output() {
    // verbatim Egaroucid for Console 7.8 lines
    let head = "|          Level|          Depth|           Move|          Score|           Time|          Nodes|            NPS|";
    let mid = "|             10|        10@100%|             e6|             +0|  000:00:00.002|          21876|       10938000|";
    let end = "|         custom|        20@100%|             a8|            -12|  000:00:00.131|        1180463|        9011167|";
    let sel = "|         custom|         15@74%|             h1|             +3|  000:00:00.031|         180463|        5011167|";
    let book = "|           Book|              -|             f5|             +0|  000:00:00.000|              0|              0|";
    assert_eq!(parse_hint_row(head).unwrap(), None);
    let r = parse_hint_row(mid).unwrap().unwrap();
    assert_eq!(r, HintRow { square: 44, discs: 0, depth: Some((10, 100.0)) });
    assert!(!r.exact(59), "a depth-10 midgame search is not exact");
    let e = parse_hint_row(end).unwrap().unwrap();
    assert_eq!((e.square, e.discs, e.exact(20), e.exact(21)), (56, -12, true, false));
    let s = parse_hint_row(sel).unwrap().unwrap();
    assert_eq!((s.square, s.discs, s.depth, s.exact(15)), (7, 3, Some((15, 74.0)), false));
    let b = parse_hint_row(book).unwrap().unwrap();
    assert_eq!((b.depth, b.exact(55)), (None, false));
    assert!(parse_hint_row("|x|y|").is_err() && parse_hint_row("|1|2@3%|z9|+1|").is_err() && parse_hint_row("|1|2@3%|a1|x|").is_err());
}

#[test]
fn engine_flags_and_options() {
    let lab = OthelloLabeller::default();
    let l = lab.resolve_limits(&Limits::default()).unwrap();
    let f = engine_flags(&l, &EgaOptions::parse(&[]).unwrap()).unwrap();
    assert_eq!(
        f.join(" "),
        "-quiet -thread 1 -hash 20 -level 15 -nobook -dpr 1 40 15 74 -dpr 41 60 60 100",
        "defaults: level-15 midgame above 20 empties, exact at or below"
    );
    let l8 = lab.resolve_limits(&Limits::parse(&["level=8".into(), "exact_empties=0".into()]).unwrap()).unwrap();
    assert_eq!(engine_flags(&l8, &EgaOptions::parse(&[]).unwrap()).unwrap()[7..].join(" "), "-nobook -dpr 1 60 8 100");
    let l60 = lab.resolve_limits(&Limits::parse(&["exact_empties=60".into()]).unwrap()).unwrap();
    assert_eq!(engine_flags(&l60, &EgaOptions::parse(&[]).unwrap()).unwrap()[7..].join(" "), "-nobook -dpr 1 60 60 100");
    let o = EgaOptions::parse(&[("Hash".into(), "16".into()), ("book".into(), "on".into()), ("eval".into(), "/x/y.egev2".into())]).unwrap();
    assert_eq!(o, EgaOptions { hash: 16, threads: 1, book: true, eval: Some("/x/y.egev2".into()) });
    assert_eq!(engine_flags(&l, &o).unwrap()[..9].join(" "), "-quiet -thread 1 -hash 16 -level 15 -eval /x/y.egev2");
    assert!(EgaOptions::parse(&[("MultiPV".into(), "3".into())]).is_err());
    assert!(EgaOptions::parse(&[("hash".into(), "40".into())]).is_err() && EgaOptions::parse(&[("book".into(), "maybe".into())]).is_err());
    assert!(lab.resolve_limits(&Limits::parse(&["nodes=5".into()]).unwrap()).is_err());
    assert!(lab.resolve_limits(&Limits::parse(&["level=0".into()]).unwrap()).is_err());
    assert!(lab.resolve_limits(&Limits::parse(&["exact_empties=61".into()]).unwrap()).is_err());
    // a raw Egaroucid binary next to resources/ runs in that directory; the wrapper does not
    let root = tmpdir("oth-cwd");
    std::fs::create_dir_all(root.join("bin/resources")).unwrap();
    let cfg = |p: &Path| EngineConfig {
        command: EngineCommand::process(p, &["-x".into()]),
        options: vec![],
        limits: l.clone(),
        timeout: Duration::from_secs(1),
        max_moves: 32,
    };
    let EngineCommand::Process { args, cwd, .. } = lab.command(&cfg(&root.join("bin/Egaroucid_for_Console.out"))).unwrap() else { panic!() };
    assert_eq!((args[0].as_str(), args[1].as_str(), cwd), ("-x", "-quiet", Some(root.join("bin"))));
    let EngineCommand::Process { cwd, .. } = lab.command(&cfg(&root.join("egaroucid"))).unwrap() else { panic!() };
    assert_eq!(cwd, None);
    let EngineCommand::Process { cwd, .. } = lab.command(&cfg(Path::new("egaroucid"))).unwrap() else { panic!() };
    assert_eq!(cwd, None, "a bare program name keeps the working directory");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn expected_scores() {
    assert_eq!((q_from_discs(2, true, 8.8), q_from_discs(0, true, 8.8), q_from_discs(-64, true, 8.8)), (1.0, 0.5, 0.0));
    assert_eq!(q_from_discs(0, false, 8.8), 0.5);
    let (a, b) = (q_from_discs(6, false, 8.8), q_from_discs(-6, false, 8.8));
    assert!((a + b - 1.0).abs() < 1e-6 && (a - 1.0 / (1.0 + (-6.0f32 / 8.8).exp())).abs() < 1e-6);
    assert!(OthelloLabeller::new(0.0, None).is_err() && OthelloLabeller::new(5.0, Some(-1.0)).is_err());
}

#[test]
fn fake_engine_speaks_the_console_subset() {
    let mut f = FakeEgaroucid::new();
    assert_eq!(f.handle("version").unwrap()[0], FakeEgaroucid::VERSION);
    assert_eq!(f.handle("gm-sync-1").unwrap(), ["[ERROR] command `gm-sync-1` not found"]);
    assert!(f.handle("clearcache").unwrap().is_empty());
    let start = OthPos::start();
    f.handle(&format!("setboard {}", start.to_egaroucid())).unwrap();
    let out = f.handle("hint 3").unwrap();
    assert_eq!(out.len(), 4);
    assert_eq!(parse_hint_row(&out[0]).unwrap(), None);
    let rows: Vec<HintRow> = out[1..].iter().map(|l| parse_hint_row(l).unwrap().unwrap()).collect();
    assert_eq!(rows, FakeEgaroucid::rows(&start)[..3]);
    assert!(rows.windows(2).all(|w| w[0].discs >= w[1].discs));
    assert!(f.handle("quit").is_none());
    let mut c = FakeEgaroucid::new().crash_on(1);
    assert!(c.handle("hint 1").is_none());
}

#[test]
fn session_hint_through_the_engine_wrapper() {
    let lab = OthelloLabeller::default();
    let cfg = EngineConfig {
        command: fake_engine_command(),
        options: vec![],
        limits: lab.resolve_limits(&Limits::default()).unwrap(),
        timeout: Duration::from_secs(10),
        max_moves: 32,
    };
    let mut s = lab.open_session(&cfg).unwrap();
    assert_eq!(s.id, "Egaroucid for Console 7.8.0");
    let pos = OthPos::start().after(37).unwrap().after(45).unwrap();
    let rows = s.hint(&pos, 32, cfg.timeout).unwrap();
    assert_eq!(rows, FakeEgaroucid::rows(&pos));
    let a = lab.query(&mut s, &pos, &cfg).unwrap();
    assert_eq!(a.moves.len(), pos.n_legal() as usize);
    for (m, r) in a.moves.iter().zip(&rows) {
        assert_eq!((m.action, m.raw), (r.square, r.discs as f32));
        assert_eq!(m.q, q_from_discs(r.discs, false, DISC_SCALE), "59 empties: not exact");
    }
    assert_eq!(s.hint(&pos, 2, cfg.timeout).unwrap().len(), 2);
    assert!(s.hint(&OthPos::new(pos.board, pos.mover), 0, cfg.timeout).is_err());
    // an engine answering for another position (e.g. a rejected setboard) is caught
    let wrong = EngineCommand::in_process("stuck", || {
        let mut f = FakeEgaroucid::new();
        move |l: &str| if l.starts_with("setboard") { Some(vec![]) } else { f.handle(l) }
    });
    let mut stuck = EgaSession::open(wrong, cfg.timeout).unwrap();
    assert!(format!("{:#}", stuck.hint(&pos, 32, cfg.timeout).unwrap_err()).contains("illegal"));
}

// ------------------------------------------------------------------ replay + verify

#[test]
fn replays_and_verifies_converter_shards() {
    let root = tmpdir("oth-replay");
    converter_shards(&root, 30, 16, 7);
    let (mut frames, mut passes, mut forced) = (0, 0, 0);
    for shard in ["othello_00000", "othello_00001"] {
        for (meta, ps) in replay_shard(&root.join("base").join(shard)) {
            assert_eq!(ps[0], OthPos::start());
            frames += ps.len();
            passes += ps.iter().filter(|p| p.legal == 0).count();
            forced += ps.iter().filter(|p| p.n_legal() == 1).count();
            assert_eq!(meta["n_pass_frames"].as_u64().unwrap() as usize, ps.iter().filter(|p| p.legal == 0).count());
        }
    }
    assert!(frames > 1000 && passes >= 3 && forced > 0, "{frames} frames, {passes} passes, {forced} forced");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn verification_catches_corrupt_frames_and_values() {
    let root = tmpdir("oth-corrupt");
    converter_shards(&root, 4, 8, 11);
    let lab = OthelloLabeller::default();
    let dir = root.join("base").join("othello_00000");
    let fresh = || ShardData::load(ShardReader::open(&dir).unwrap()).unwrap();
    // one flipped bit of obs.board_seq (sample 1, frame 5, plane 2 = empty)
    let mut data = fresh();
    let start = data.reader.samples[1].start;
    let t = data.tensors.get_mut("obs.board_seq").unwrap();
    let Column::Bits(bytes, per) = &mut t.col else { panic!("obs.board_seq is packed bits") };
    let per_bytes = per.div_ceil(8);
    bytes[(start + 5) * per_bytes + 16] ^= 0x80;
    let v = data.view(1);
    let ps = lab.replay(&v).unwrap();
    let err = format!("{:#}", lab.verify(&v, &ps).unwrap_err());
    assert!(err.contains("obs.board_seq: frame 5 element 128"), "{err}");
    assert!(lab.verify(&data.view(0), &lab.replay(&data.view(0)).unwrap()).is_ok());
    // a wrong outcome
    let mut data = fresh();
    let Column::F32(val) = &mut data.tensors.get_mut("tgt.value").unwrap().col else { panic!() };
    let s2 = data.reader.samples[2].start;
    val[s2 + 3] = -val[s2 + 3] + 0.5;
    let v = data.view(2);
    assert!(format!("{:#}", lab.verify(&v, &lab.replay(&v).unwrap()).unwrap_err()).contains("tgt.value: frame 3"));
    // an illegal recorded move breaks the replay
    let mut data = fresh();
    let Column::I64(place) = &mut data.tensors.get_mut("tgt.place_idx").unwrap().col else { panic!() };
    let s3 = data.reader.samples[3].start;
    place[s3] = 0; // a1 is never legal at the start
    assert!(format!("{:#}", lab.replay(&data.view(3)).unwrap_err()).contains("frame 0: recorded move a1 is illegal"));
    std::fs::remove_dir_all(&root).unwrap();
}

/// Golden test on a real WTHOR shard of `gmdata othello convert` (skips when absent):
/// every game replays and re-encodes bit-exactly, and the replayed result matches the
/// converter's recorded disc counts.
#[test]
fn golden_real_wthor_shard() {
    let dir = std::env::var_os("GM_OTHELLO_SHARD").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("gm-data/shards/othello/othello_00000")
    });
    if !dir.join("index.json").exists() {
        eprintln!("skip: no real othello shard at {} (set GM_OTHELLO_SHARD)", dir.display());
        return;
    }
    let lab = OthelloLabeller::default();
    let reader = ShardReader::open(&dir).unwrap();
    lab.check_shard(&reader).unwrap();
    let data = ShardData::load(reader).unwrap();
    let (mut frames, mut passes, mut forced) = (0usize, 0usize, 0usize);
    for i in 0..data.reader.n_samples() {
        let v = data.view(i);
        let ps = lab.replay(&v).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
        lab.verify(&v, &ps).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
        let (place, passed) = (v.i64s("tgt.place_idx").unwrap(), v.bools("tgt.passed").unwrap());
        assert_eq!(ps[0], OthPos::start());
        frames += ps.len();
        passes += passed.iter().filter(|&&p| p).count();
        forced += ps.iter().filter(|p| p.n_legal() == 1).count();
        // the position after the last frame holds the converter's final disc counts
        let last = ps.last().unwrap();
        let fin = if passed[v.len - 1] { last.board } else { play(&last.board, last.mover, place[v.len - 1] as usize) };
        let end = OthPos::new(fin, BLACK);
        let meta = v.meta();
        assert_eq!(
            (end.mine().count_ones() as u64, end.theirs().count_ones() as u64),
            (meta["replayed_black"].as_u64().unwrap(), meta["replayed_white"].as_u64().unwrap()),
            "sample {i}"
        );
        assert_eq!(meta["n_pass_frames"].as_u64().unwrap() as usize, ps.iter().filter(|p| p.legal == 0).count());
    }
    eprintln!("golden {}: {} games, {frames} frames, {passes} pass frames, {forced} forced", dir.display(), data.reader.n_samples());
    assert!(data.reader.n_samples() > 0 && frames > 50 * data.reader.n_samples() && passes > 0);
}

// ------------------------------------------------------------------ end to end (fake engine)

/// Check every teacher column of an overlay against the fake engine's answers;
/// returns (labelled frames, exact frames, pass frames).
fn check_overlay(root: &Path, out: &str, shard: &str, disc_tau: Option<f64>) -> (usize, usize, usize) {
    let base = ShardReader::open(&root.join("base").join(shard)).unwrap();
    let o = ShardReader::open(&root.join(out).join(shard)).unwrap();
    for (name, info) in &base.tensors {
        assert_eq!(o.tensors[name], *info, "base column {name} carried over");
    }
    let k = DEFAULT_MAX_MOVES;
    let Column::I64(idx) = overlay_tensor(&o, "tgt.teacher_idx", DType::I64, &[k]).col else { panic!() };
    let Column::F32(q) = overlay_tensor(&o, "tgt.teacher_q", DType::F32, &[k]).col else { panic!() };
    let Column::F32(raw) = overlay_tensor(&o, "tgt.teacher_raw", DType::F32, &[k]).col else { panic!() };
    let Column::F32(ev) = overlay_tensor(&o, "tgt.engine_value", DType::F32, &[]).col else { panic!() };
    let Column::Bool(evm) = overlay_tensor(&o, "tgt.engine_value_mask", DType::Bool, &[]).col else { panic!() };
    let Column::F32(pp) = overlay_tensor(&o, "tgt.place_probs", DType::F32, &[64]).col else { panic!() };
    let Column::F32(pb) = overlay_tensor(&o, "tgt.pass_probs", DType::F32, &[]).col else { panic!() };
    let Column::Bool(pm) = overlay_tensor(&o, "tgt.pass_probs_mask", DType::Bool, &[]).col else { panic!() };
    assert!(!o.has("tgt.teacher_visits"));
    let (mut labelled, mut exact_frames, mut passes) = (0, 0, 0);
    let data = ShardData::load(base).unwrap();
    let lab = OthelloLabeller::default();
    for i in 0..data.reader.n_samples() {
        let v = data.view(i);
        for (t, pos) in lab.replay(&v).unwrap().iter().enumerate() {
            let f = v.start + t;
            let (row, qs, rs) = (&idx[f * k..(f + 1) * k], &q[f * k..(f + 1) * k], &raw[f * k..(f + 1) * k]);
            let probs = &pp[f * 64..(f + 1) * 64];
            let exact = pos.empties() <= FakeEgaroucid::EXACT_EMPTIES;
            if pos.n_legal() == 1 {
                assert!(row.iter().all(|&a| a == -1) && qs.iter().all(|x| x.is_nan()) && rs.iter().all(|x| x.is_nan()));
                assert!(evm[f] == 0 && ev[f] == 0.0 && pm[f] == 0 && pb[f] == 0.0 && probs.iter().all(|&p| p == 0.0), "frame {f}");
                continue;
            }
            labelled += 1;
            exact_frames += exact as usize;
            if pos.legal == 0 {
                // pass frame: the pass scored by the opponent's best reply; pass teacher only
                passes += 1;
                let reply = FakeEgaroucid::rows(&OthPos::new(pos.board, crate::othello::opponent(pos.mover)))[0].clone();
                let want = 1.0 - q_from_discs(reply.discs, exact, DISC_SCALE);
                assert_eq!((row[0], qs[0], rs[0]), (PASS_ID, want, -reply.discs as f32), "frame {f}");
                assert!(row[1..].iter().all(|&a| a == -1) && qs[1..].iter().all(|x| x.is_nan()));
                assert!(evm[f] == 1 && (ev[f] - (2.0 * want - 1.0)).abs() < 1e-6);
                assert!(probs.iter().all(|&p| p == 0.0), "no place teacher on a pass frame");
                assert_eq!((pb[f], pm[f]), (1.0, 1), "P(pass) = 1 on a pass frame");
                continue;
            }
            let mut want: Vec<(i64, i32, f32)> =
                FakeEgaroucid::rows(pos).iter().map(|r| (r.square, r.discs, q_from_discs(r.discs, exact, DISC_SCALE))).collect();
            want.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
            let n = want.len();
            assert_eq!(row[..n], want.iter().map(|w| w.0).collect::<Vec<_>>()[..], "frame {f}: every legal move, sorted by Q then square");
            assert!(row[n..].iter().all(|&a| a == -1) && qs[n..].iter().all(|x| x.is_nan()));
            for (j, w) in want.iter().enumerate() {
                assert_eq!((qs[j], rs[j]), (w.2, w.1 as f32));
            }
            if exact {
                assert!(qs[..n].iter().all(|&x| x == 0.0 || x == 0.5 || x == 1.0));
            }
            assert_eq!(evm[f], 1);
            assert!((ev[f] - (2.0 * want[0].2 - 1.0)).abs() < 1e-6);
            let pi = match disc_tau {
                None => softmax_q(&qs[..n], 0.05),
                Some(t) => {
                    let e: Vec<f64> = want.iter().map(|w| ((w.1 - want.iter().map(|x| x.1).max().unwrap()) as f64 / t).exp()).collect();
                    let s: f64 = e.iter().sum();
                    e.iter().map(|x| (x / s) as f32).collect()
                }
            };
            for sq in 0..64i64 {
                let p = row[..n].iter().position(|&a| a == sq).map(|j| pi[j]).unwrap_or(0.0);
                assert!((probs[sq as usize] - p).abs() < 1e-5, "frame {f} square {sq}: {} vs {p}", probs[sq as usize]);
            }
            assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert_eq!((pb[f], pm[f]), (0.0, 1), "P(pass) = 0 while a move exists");
        }
    }
    (labelled, exact_frames, passes)
}

#[test]
fn end_to_end_with_the_fake_engine() {
    let root = tmpdir("oth-e2e");
    converter_shards(&root, 24, 12, 3);
    let s = run(OthelloLabeller::default(), &opts(&root, fake_engine_command())).unwrap();
    assert_eq!((s.shards_total, s.shards_written), (2, 2));
    assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis), (0, 0, 0));
    let skipped = s.stats.skipped.values().sum::<u64>();
    assert!(s.stats.skipped.len() == 1 && s.stats.skipped["forced"] >= 1, "only forced frames are skipped: {:?}", s.stats.skipped);
    assert_eq!(s.stats.labelled + skipped, s.stats.frames);
    assert!(s.stats.cache_hits > 0, "the opening positions repeat across games");
    assert_eq!(s.stats.queries + s.stats.cache_hits, s.stats.labelled);
    assert_eq!(s.teacher["engine"], "Egaroucid for Console 7.8.0");
    assert_eq!(s.teacher["raw_unit"], "discs");
    assert_eq!(s.teacher["policy"], json!({"kind": "softmax", "tau": 0.05}));
    assert_eq!(s.teacher["limits"], json!({"level": "15", "exact_empties": "20"}));
    assert_eq!(s.teacher["heads"]["place"], json!({"kind": "categorical", "n": 64, "k": 64, "storage": "dense"}));
    assert_eq!(s.teacher["heads"]["pass"], json!({"kind": "bernoulli"}));
    assert_eq!(s.teacher["game"]["disc_scale"], DISC_SCALE);
    assert_eq!((s.teacher["game"]["hash"].as_u64(), s.teacher["game"]["book"].as_bool()), (Some(20), Some(false)));
    let (mut labelled, mut exact, mut passes) = (0, 0, 0);
    for shard in ["othello_00000", "othello_00001"] {
        let (l, e, p) = check_overlay(&root, "out", shard, None);
        labelled += l;
        exact += e;
        passes += p;
        let o = ShardReader::open(&root.join("out").join(shard)).unwrap();
        assert_eq!(o.meta()["teacher"], s.teacher);
    }
    assert_eq!(labelled as u64, s.stats.labelled);
    assert!(exact > 0 && exact < labelled && passes >= 3, "{labelled} labelled, {exact} exact, {passes} pass frames");
    // --disc-tau: softmax over the raw disc scores, Q / engine value unchanged
    let mut o = opts(&root, fake_engine_command());
    o.out = root.join("disc");
    let s2 = run(OthelloLabeller::new(DISC_SCALE, Some(2.0)).unwrap(), &o).unwrap();
    assert_eq!(s2.teacher["policy"], json!({"kind": "softmax_discs", "disc_tau": 2.0}));
    for shard in ["othello_00000", "othello_00001"] {
        check_overlay(&root, "disc", shard, Some(2.0));
        for col in ["tgt.teacher_q.npy", "tgt.teacher_idx.npy", "tgt.engine_value.npy"] {
            assert_eq!(std::fs::read(root.join("out").join(shard).join(col)).unwrap(), std::fs::read(root.join("disc").join(shard).join(col)).unwrap());
        }
    }
    // resuming the first run with --disc-tau is refused (different teacher settings)
    let mut clash = opts(&root, fake_engine_command());
    clash.out = root.join("out");
    assert!(run(OthelloLabeller::new(DISC_SCALE, Some(2.0)).unwrap(), &clash).unwrap_err().to_string().contains("different settings"));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn deterministic_across_workers_and_sampling() {
    let root = tmpdir("oth-det");
    converter_shards(&root, 12, 12, 5);
    let mut o1 = opts(&root, fake_engine_command());
    o1.workers = 1;
    run(OthelloLabeller::default(), &o1).unwrap();
    let mut o3 = opts(&root, fake_engine_command());
    o3.workers = 3;
    o3.cache_size = 0;
    o3.out = root.join("out3");
    run(OthelloLabeller::default(), &o3).unwrap();
    for f in std::fs::read_dir(root.join("out/othello_00000")).unwrap() {
        let name = f.unwrap().file_name();
        if name != "index.json" {
            assert_eq!(std::fs::read(root.join("out/othello_00000").join(&name)).unwrap(), std::fs::read(root.join("out3/othello_00000").join(&name)).unwrap(), "{name:?}");
        }
    }
    let mut every = opts(&root, fake_engine_command());
    every.out = root.join("every2");
    every.every_nth_frame = 2;
    let s = run(OthelloLabeller::default(), &every).unwrap();
    assert!(s.stats.sampled_out > 0 && s.stats.labelled > 0);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn engine_crashes_and_hangs_are_retried() {
    let root = tmpdir("oth-crash");
    converter_shards(&root, 6, 6, 9);
    let cmd = EngineCommand::in_process("flaky-egaroucid", || FakeEgaroucid::new().crash_on(7).hang_on(3, Duration::from_millis(400)));
    let mut o = opts(&root, cmd);
    o.timeout = Duration::from_millis(150);
    let s = run(OthelloLabeller::default(), &o).unwrap();
    assert!(s.stats.retries > 0 && s.stats.timeouts > 0, "{:?}", s.stats);
    assert_eq!(s.stats.engine_errors, 0);
    check_overlay(&root, "out", "othello_00000", None);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fit_beta_recovers_a_known_scale() {
    // outcomes drawn from sigma(v / 7) on a grid (expected values, no noise)
    let pts: Vec<(f64, f64)> = (-30..=30).map(|v| (v as f64, 1.0 / (1.0 + (-(v as f64) / 7.0).exp()))).collect();
    let (beta, _) = fit_beta(&pts);
    assert!((1.0 / beta - 7.0).abs() < 1e-3, "{}", 1.0 / beta);
}

#[test]
fn fit_disc_scale_runs_with_the_fake_engine() {
    let root = tmpdir("oth-fit");
    converter_shards(&root, 6, 6, 13);
    let mut o = opts(&root, fake_engine_command());
    o.out = root.join("fit");
    let r = fit_disc_scale(&OthelloLabeller::default(), &o, 40).unwrap();
    assert_eq!(r["n"], 40);
    assert!(r["disc_scale"].as_f64().unwrap() > 0.0);
    assert!(root.join("fit/disc_scale_fit.json").exists());
    std::fs::remove_dir_all(&root).unwrap();
}

// ------------------------------------------------------------------ real Egaroucid (skipped when absent)

fn real_engine() -> Option<PathBuf> {
    let p = engine_path("GM_EGAROUCID", "egaroucid");
    if p.is_none() {
        eprintln!("skip: Egaroucid not installed (scripts/engines/fetch_engines.sh egaroucid, or set GM_EGAROUCID)");
    }
    p
}

fn real_cfg(engine: &Path, limits: &[&str]) -> EngineConfig {
    let lab = OthelloLabeller::default();
    EngineConfig {
        command: EngineCommand::process(engine, &[]),
        options: vec![],
        limits: lab.resolve_limits(&Limits::parse(&limits.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()).unwrap(),
        timeout: Duration::from_secs(120),
        max_moves: 32,
    }
}

/// Protocol, orientation and sign conventions against an exact solver: at ≤ 7 empties
/// Egaroucid's per-move scores must equal our own negamax values.
#[test]
fn real_egaroucid_matches_an_exact_solver() {
    let Some(engine) = real_engine() else { return };
    let lab = OthelloLabeller::default();
    let cfg = real_cfg(&engine, &["level=5", "exact_empties=10"]);
    let mut s = lab.open_session(&cfg).unwrap();
    assert!(s.id.starts_with("Egaroucid for Console "), "{}", s.id);
    // the start position: black's four symmetric moves, all equal
    let rows = s.hint(&OthPos::start(), 32, cfg.timeout).unwrap();
    let mut names: Vec<String> = rows.iter().map(|r| square_name(r.square)).collect();
    names.sort();
    assert_eq!(names, ["c4", "d3", "e6", "f5"]);
    assert!(rows.iter().all(|r| r.discs == rows[0].discs && !r.exact(60)));
    // late positions of random games: exact per-move values (both colours to move)
    let (mut checked, mut white) = (0, 0);
    for seed in 0..40u64 {
        let g = random_game(splitmix(seed ^ 0x5eed));
        let mut pos = OthPos::start();
        for &(r, c) in &g {
            if pos.empties() <= 7 && pos.n_legal() >= 2 {
                let rows = s.hint(&pos, 32, cfg.timeout).unwrap();
                assert_eq!(rows.len(), pos.n_legal() as usize);
                for row in &rows {
                    assert!(row.exact(pos.empties()), "{row:?} at {} empties", pos.empties());
                    assert_eq!(row.discs, solve_move(&pos, row.square), "{} {}", pos.to_egaroucid(), square_name(row.square));
                }
                assert_eq!(rows[0].discs, solve(&pos));
                checked += 1;
                white += (pos.mover == WHITE) as usize;
                break;
            }
            match pos.after((r * 8 + c) as i64) {
                Some(n) => pos = n,
                None => break,
            }
        }
    }
    assert!(checked >= 20 && white > 0, "{checked} positions checked, {white} with white to move");
}

/// The full pipeline with the real engine on converter shards (fast search settings):
/// labels are well formed, exact frames have W/D/L Q, and two worker layouts agree.
#[test]
fn real_egaroucid_end_to_end() {
    let Some(engine) = real_engine() else { return };
    let root = tmpdir("oth-real");
    converter_shards(&root, 4, 4, 21);
    let mut o = opts(&root, EngineCommand::process(&engine, &[]));
    o.limits = Limits::parse(&["level=6".into(), "exact_empties=12".into()]).unwrap();
    o.workers = 2;
    let s = run(OthelloLabeller::default(), &o).unwrap();
    assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis), (0, 0, 0), "{:?}", s.stats);
    assert!(s.teacher["engine"].as_str().unwrap().starts_with("Egaroucid for Console "));
    let out = ShardReader::open(&root.join("out/othello_00000")).unwrap();
    let (idx, q, raw, evm, pp) = (
        out.read_i64("tgt.teacher_idx").unwrap(),
        out.read_f32("tgt.teacher_q").unwrap(),
        out.read_f32("tgt.teacher_raw").unwrap(),
        out.read_bool("tgt.engine_value_mask").unwrap(),
        out.read_f32("tgt.place_probs").unwrap(),
    );
    let data = ShardData::load(ShardReader::open(&root.join("base/othello_00000")).unwrap()).unwrap();
    let lab = OthelloLabeller::default();
    let mut exact_frames = 0;
    for i in 0..data.reader.n_samples() {
        let v = data.view(i);
        for (t, pos) in lab.replay(&v).unwrap().iter().enumerate() {
            let f = v.start + t;
            assert_eq!(evm[f], pos.n_legal() != 1);
            if pos.n_legal() == 1 {
                continue;
            }
            if pos.legal == 0 {
                let (qq, rr) = (q[f * 32], raw[f * 32]);
                assert_eq!(idx[f * 32..f * 32 + 2], [PASS_ID, -1]);
                assert!((qq - q_from_discs(rr as i32, pos.empties() <= 12, DISC_SCALE)).abs() < 1e-6);
                assert!(pp[f * 64..(f + 1) * 64].iter().all(|&p| p == 0.0));
                continue;
            }
            let n = pos.n_legal() as usize;
            let row = &idx[f * 32..f * 32 + n];
            let mut got: Vec<i64> = row.to_vec();
            got.sort();
            assert_eq!(got, (0..64).filter(|&s| pos.is_legal(s)).collect::<Vec<_>>(), "all legal moves scored");
            for j in 0..n {
                let (qq, rr) = (q[f * 32 + j], raw[f * 32 + j]);
                if pos.empties() <= 12 {
                    assert_eq!(qq, q_from_discs(rr as i32, true, DISC_SCALE));
                } else {
                    assert_eq!(qq, q_from_discs(rr as i32, false, DISC_SCALE));
                }
            }
            exact_frames += (pos.empties() <= 12) as usize;
            assert!((pp[f * 64..(f + 1) * 64].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }
    assert!(exact_frames > 0);
    let mut o1 = opts(&root, EngineCommand::process(&engine, &[]));
    o1.limits = o.limits.clone();
    o1.workers = 1;
    o1.out = root.join("out1");
    run(OthelloLabeller::default(), &o1).unwrap();
    for col in ["tgt.teacher_idx.npy", "tgt.teacher_raw.npy", "tgt.place_probs.npy"] {
        assert_eq!(std::fs::read(root.join("out/othello_00000").join(col)).unwrap(), std::fs::read(root.join("out1/othello_00000").join(col)).unwrap(), "{col}");
    }
    std::fs::remove_dir_all(&root).unwrap();
}
