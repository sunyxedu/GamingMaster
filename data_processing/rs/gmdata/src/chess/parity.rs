//! Parity of the chess converter, its pre-gmrules code and the simulator, on real records
//! (the Lichess fixture), the purpose-built edge cases (tests/fixtures/chess/edge_cases.pgn:
//! castling both ways for both colours, e.p., every promotion piece, captures of promoted
//! pieces, checks, repetitions, 50-move clocks, stalemate, dead material) and random games:
//!
//! - `converter_frames_are_the_v3_reference_frames`: every column of every frame the
//!   converter writes (obs.board_seq, obs.globals_seq, tgt.from_idx / to_idx / promotion
//!   / value / legal_from / legal_to_of / legal_promotion_of) is bit-identical to the
//!   pre-gmrules converter code ([`super::v3_reference`], verbatim);
//! - `simulator_steps_reproduce_the_converter_frames`: a batched `VecSim` of the gmrules
//!   game stepped with the converter's ACTION KEYS (as the RL env is) emits, at every
//!   step, obs bit-identical to the converter's frame, legality tables whose (from) row /
//!   (from, to) cell ARE the converter's IL masks, applied keys = the converter's targets,
//!   the record's own trajectory — and, per rule config, ends exactly where the rules say:
//!   checkmate / stalemate at the record's end, the automatic threefold draw on the first
//!   frame whose converter flag "seen >= 2 times before" (globals 9) is set, the 50-move
//!   draw on the first frame whose clock feature (globals 6) saturates, dead material by
//!   shakmaty, the ply cap — with the rules' outcome as reward.

use super::v3_reference as v3;
use super::{encode_moves, events_of, result_white, Collector, Frames, Game};
use gmrules::games::chess::{ChessConfig, End, StandardChess, GLOBAL_FEATURES, N_PROMOTIONS, PLANE_LEN};
use gmrules::{Game as _, Rng, SimOptions, VecSim};
use pgn_reader::Reader;
use shakmaty::{Chess, Color, Move, Position};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/chess").join(name)
}

/// Every readable game of a PGN file (no filters).
fn read_games(name: &str) -> Vec<Game> {
    let mut r = Reader::new(std::fs::File::open(fixture(name)).unwrap());
    let mut out = Vec::new();
    while let Some(g) = r.read_game(&mut Collector).unwrap() {
        if !g.bad {
            out.push(g);
        }
    }
    out
}

/// (label, moves, result for white) of the fixture games and `n_random` random games
/// (uniform legal moves up to 300 plies; records continue past every automatic draw).
fn corpus(n_random: usize) -> Vec<(String, Vec<Move>, f32)> {
    let mut out = Vec::new();
    for file in ["lichess_2013-01_head.pgn", "edge_cases.pgn"] {
        for (i, g) in read_games(file).into_iter().enumerate() {
            let wv = result_white(&g.headers["Result"]).expect("fixture results are known");
            out.push((format!("{file}#{i} {}", g.headers.get("Site").map_or("", |s| s.as_str())), g.moves, wv));
        }
    }
    let mut rng = Rng::new(20260927);
    let cfg = ChessConfig { max_plies: 300, ..ChessConfig::records() };
    for i in 0..n_random {
        let mut g = StandardChess::new_game(&cfg);
        let mut moves = Vec::new();
        while g.end().is_none() {
            let legal = g.legal_moves();
            let (id, m) = legal[rng.below(legal.len() as u64) as usize];
            let seat = moves.len() % 2;
            moves.push(m);
            g.apply_action(seat, id).unwrap();
        }
        let wv = [1.0, 0.0, -1.0][i % 3];
        out.push((format!("random#{i}"), moves, wv));
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_same(what: &str, new: &Frames, old: &v3::Frames) {
    assert_eq!(bits(&new.planes), bits(&old.planes), "{what}: obs.board_seq");
    assert_eq!(bits(&new.globals), bits(&old.globals), "{what}: obs.globals_seq");
    assert_eq!(new.from, old.from, "{what}: tgt.from_idx");
    assert_eq!(new.to, old.to, "{what}: tgt.to_idx");
    assert_eq!(new.promo, old.promo, "{what}: tgt.promotion");
    assert_eq!(bits(&new.value), bits(&old.value), "{what}: tgt.value");
    assert_eq!(bits(&new.legal_from), bits(&old.legal_from), "{what}: tgt.legal_from");
    assert_eq!(bits(&new.legal_to_of), bits(&old.legal_to_of), "{what}: tgt.legal_to_of");
    assert_eq!(bits(&new.legal_promotion_of), bits(&old.legal_promotion_of), "{what}: tgt.legal_promotion_of");
}

#[test]
fn converter_frames_are_the_v3_reference_frames() {
    let games = corpus(40);
    let (mut frames, mut seen2, mut clock_full, mut castles, mut eps) = (0, 0, 0, 0, 0);
    let mut promos = [0usize; N_PROMOTIONS];
    for (what, moves, wv) in &games {
        let old = v3::encode_moves(moves, *wv).unwrap_or_else(|e| panic!("{what}: reference {e:?}"));
        let new = encode_moves(moves, *wv).unwrap_or_else(|e| panic!("{what}: converter {e:?}"));
        assert_same(what, &new, &old);
        frames += new.from.len();
        for g in new.globals.chunks(GLOBAL_FEATURES) {
            seen2 += (g[9] == 1.0) as usize;
            clock_full += (g[6] == 1.0) as usize;
        }
        for &p in &new.promo {
            promos[p as usize] += 1;
        }
        castles += moves.iter().filter(|m| m.is_castle()).count();
        eps += moves.iter().filter(|m| m.is_en_passant()).count();
    }
    // the corpus reaches every branch of the frame code (clock >= 100: the edge cases'
    // knight shuffles, 5 frames each)
    assert!(frames > 30_000 && seen2 > 100 && clock_full >= 10 && castles > 200 && eps > 10, "{frames} {seen2} {clock_full} {castles} {eps}");
    assert!(promos.iter().all(|&n| n > 0), "every promotion class: {promos:?}");
    // an illegal record is rejected by both
    let (_, mut moves, wv) = games[0].clone();
    moves.swap(0, 1);
    assert!(v3::encode_moves(&moves, wv).is_err() && encode_moves(&moves, wv).is_err());
}

/// Where the rules end a record under `cfg` — (position index t in 1..=T, why, outcome) —
/// derived from the v3 reference frames (their "seen >= 2 times before" flag and their
/// clock feature) and shakmaty (mate, stalemate, dead material), independently of the
/// gmrules game; checks on the way that the reference's features mean exactly the
/// repetition count / clock.
fn expected_end(cfg: &ChessConfig, moves: &[Move], r: &v3::Frames) -> Option<(usize, End, [f32; 2])> {
    let n = moves.len();
    let mut pos = Chess::default();
    let mut reps = v3::Repetitions::default();
    for t in 0..=n {
        let prior = reps.see(&pos);
        if t > 0 {
            let (clock_full, third) = if t < n {
                let g = &r.globals[t * GLOBAL_FEATURES..(t + 1) * GLOBAL_FEATURES];
                assert_eq!(g[6] == 1.0, pos.halfmoves() >= 100);
                assert_eq!((g[8] == 1.0, g[9] == 1.0), (prior >= 1, prior >= 2));
                (g[6] == 1.0, g[9] == 1.0)
            } else {
                (pos.halfmoves() >= 100, prior >= 2)
            };
            let end = if pos.legal_moves().is_empty() {
                Some(if pos.is_check() { End::Checkmate } else { End::Stalemate })
            } else if cfg.insufficient_material_draw && pos.is_insufficient_material() {
                Some(End::InsufficientMaterial)
            } else if cfg.fifty_move_draw && clock_full {
                Some(End::FiftyMoves)
            } else if cfg.threefold_draw && third {
                Some(End::Threefold)
            } else if t >= cfg.max_plies {
                Some(End::PlyCap)
            } else {
                None
            };
            if let Some(e) = end {
                let outcome = match (e, pos.turn()) {
                    (End::Checkmate, Color::Black) => [1.0, -1.0],
                    (End::Checkmate, Color::White) => [-1.0, 1.0],
                    _ => [0.0, 0.0],
                };
                return Some((t, e, outcome));
            }
        }
        if t < n {
            pos.play_unchecked(moves[t]);
        }
    }
    None
}

/// Step one batched simulator through every record of `games` with the converter's
/// action keys (see the module docs); returns the ends seen.
fn step_through(cfg: &ChessConfig, games: &[(String, Vec<Move>, f32)]) -> Vec<Option<End>> {
    let n = games.len();
    let refs: Vec<v3::Frames> = games.iter().map(|(_, m, wv)| v3::encode_moves(m, *wv).unwrap()).collect();
    let ends: Vec<Option<(usize, End, [f32; 2])>> = games.iter().zip(&refs).map(|((_, m, _), r)| expected_end(cfg, m, r)).collect();
    // frames the env emits for a record: up to its end (or all of them)
    let span: Vec<usize> = games.iter().zip(&ends).map(|((_, m, _), e)| e.map_or(m.len(), |e| e.0)).collect();
    let mut sim = VecSim::<StandardChess>::new(cfg.clone(), n, SimOptions { threads: 2, ..SimOptions::default() }).unwrap();
    sim.reset(&(0..n).collect::<Vec<_>>(), &(0..n as u64).collect::<Vec<_>>()).unwrap();
    assert_eq!(sim.rows_per_env(), 1);
    let mut t = 0;
    while (0..n).any(|e| !sim.done(e) && t < games[e].1.len()) {
        let mut keys = sim.random_actions(t as u64).unwrap();
        {
            let b = sim.buffers();
            for e in 0..n {
                if sim.done(e) || t >= span[e] {
                    continue;
                }
                let (what, r) = (&games[e].0, &refs[e]);
                assert_eq!((b.active[e], b.step_idx[e], b.to_play[e]), (1, t as i64, (t % 2) as i64), "{what} ply {t}");
                let planes = &b.obs[0].as_f32().unwrap()[e * PLANE_LEN..(e + 1) * PLANE_LEN];
                let globals = &b.obs[1].as_f32().unwrap()[e * GLOBAL_FEATURES..(e + 1) * GLOBAL_FEATURES];
                assert_eq!(bits(planes), bits(&r.planes[t * PLANE_LEN..(t + 1) * PLANE_LEN]), "{what} ply {t}: board_seq");
                assert_eq!(bits(globals), bits(&r.globals[t * GLOBAL_FEATURES..(t + 1) * GLOBAL_FEATURES]), "{what} ply {t}: globals_seq");
                let (fr, to, p) = (r.from[t], r.to[t], r.promo[t]);
                let from = &b.legal[0].as_bool().unwrap()[e * 64..(e + 1) * 64];
                let to_cond = &b.legal[1].as_bool().unwrap()[(e * 64 + fr as usize) * 64..(e * 64 + fr as usize + 1) * 64];
                let cell = ((e * 64 + fr as usize) * 64 + to as usize) * N_PROMOTIONS;
                let promo = &b.legal[2].as_bool().unwrap()[cell..cell + N_PROMOTIONS];
                let as_f = |m: &[u8]| m.iter().map(|&x| x as f32).collect::<Vec<f32>>();
                assert_eq!(as_f(from), r.legal_from[t * 64..(t + 1) * 64].to_vec(), "{what} ply {t}: legal from = tgt.legal_from");
                assert_eq!(as_f(to_cond), r.legal_to_of[t * 64..(t + 1) * 64].to_vec(), "{what} ply {t}: to_cond[from] = tgt.legal_to_of");
                assert_eq!(as_f(promo), r.legal_promotion_of[t * N_PROMOTIONS..(t + 1) * N_PROMOTIONS].to_vec(),
                           "{what} ply {t}: promotion_cond[from, to] = tgt.legal_promotion_of");
                let row = &b.legal[2].as_bool().unwrap()[e * 64 * 64 * N_PROMOTIONS..(e + 1) * 64 * 64 * N_PROMOTIONS];
                assert_eq!(b.n_legal[e] as usize, row.iter().filter(|&&x| x == 1).count(), "{what} ply {t}: one cell per legal move");
                keys[e * 3..e * 3 + 3].copy_from_slice(&[fr, to, p]);
            }
        }
        let info = sim.step(&keys).unwrap();
        for e in 0..n {
            if t >= span[e] {
                continue;
            }
            let what = &games[e].0;
            let r = &refs[e];
            assert_eq!((info.acted[e], &info.applied[e * 3..e * 3 + 3]), (1, &[r.from[t], r.to[t], r.promo[t]][..]), "{what} ply {t}");
            match ends[e] {
                Some((te, why, outcome)) if te == t + 1 => {
                    assert_eq!((info.done[e], info.terminated[e], info.truncated[e]), (1, 1, 0), "{what}: ends at ply {te} ({why:?})");
                    assert_eq!(&info.rewards[e * 2..e * 2 + 2], &outcome[..], "{what}: outcome");
                    assert_eq!(info.episode_len[e], te as i64);
                    assert_eq!(sim.state(e).end().map(|x| x.1), Some(why), "{what}");
                }
                _ => assert_eq!(info.done[e], 0, "{what}: the env ended at ply {} but the rules do not", t + 1),
            }
        }
        t += 1;
    }
    eprintln!("chess sim parity {cfg:?}: {} records, {} frames compared, ends {:?}", n, span.iter().sum::<usize>(),
              [End::Checkmate, End::Stalemate, End::InsufficientMaterial, End::FiftyMoves, End::Threefold, End::PlyCap]
                  .map(|w| (w, ends.iter().filter(|e| e.map(|x| x.1) == Some(w)).count())));
    for (e, (what, moves, _)) in games.iter().enumerate() {
        // the env's trajectory is the record (up to where the rules end it)
        assert_eq!(&sim.trajectory(e)[..span[e]], &events_of(moves)[..span[e]], "{what}");
        if ends[e].is_some() {
            assert!(sim.done(e) && sim.trajectory(e).len() == span[e], "{what}");
        }
    }
    ends.iter().map(|e| e.map(|x| x.1)).collect()
}

fn count(ends: &[Option<End>], why: End) -> usize {
    ends.iter().filter(|e| **e == Some(why)).count()
}

#[test]
fn simulator_steps_reproduce_the_converter_frames() {
    let games = corpus(12);
    // records' config (the converter's): only mate / stalemate end a record — the env
    // reproduces every frame of every record
    let ends = step_through(&ChessConfig::records(), &games);
    assert!(count(&ends, End::Checkmate) > 50 && count(&ends, End::Stalemate) >= 1, "{ends:?}");
    assert!(ends.iter().all(|e| matches!(e, None | Some(End::Checkmate | End::Stalemate))));
    // the RL env's defaults: the automatic draws end records where the converter's own
    // features say so
    let ends = step_through(&ChessConfig::default(), &games);
    for why in [End::Checkmate, End::Stalemate, End::InsufficientMaterial, End::Threefold] {
        assert!(count(&ends, why) >= 1, "{why:?}: {ends:?}");
    }
    // without the threefold draw, the 214-ply knight shuffle ends by the fifty-move rule
    let ends = step_through(&ChessConfig { threefold_draw: false, ..ChessConfig::default() }, &games);
    assert!(count(&ends, End::FiftyMoves) >= 1, "{ends:?}");
    // the ply cap
    let ends = step_through(&ChessConfig { max_plies: 40, ..ChessConfig::default() }, &games);
    assert!(count(&ends, End::PlyCap) > 100, "{ends:?}");
}
