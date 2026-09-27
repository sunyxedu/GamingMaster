//! Unit tests of the `chess` game: perft through the game's own action enumeration, the
//! action ids / keys, the frames and masks, the automatic ends, the repetition memory and
//! the conformance battery.

use super::*;
use crate::frame::{action_keys_of, decode_action, encode_frame, encode_target, new_target, Frame};
use crate::game::Layout;
use crate::invariance::check_game;
use crate::rng::Rng;
use crate::trajectory::{replay, Event};
use shakmaty::san::San;
use shakmaty::uci::UciMove;

fn records() -> ChessConfig {
    ChessConfig::records()
}

fn game(fen: &str, cfg: &ChessConfig) -> StandardChess {
    StandardChess::from_fen(fen, cfg).unwrap()
}

/// The legal action of `seat` whose SAN is `san`.
fn id_of_san(g: &StandardChess, san: &str) -> u32 {
    let m = San::from_ascii(san.as_bytes()).unwrap().to_move(g.position()).unwrap_or_else(|e| panic!("{san} in {}: {e}", g.fen()));
    action_id(&m)
}

fn play(g: &mut StandardChess, sans: &[&str]) {
    for s in sans {
        let seat = g.mover_seat();
        let id = id_of_san(g, s);
        g.apply_action(seat, id).unwrap_or_else(|e| panic!("{s}: {e}"));
    }
}

fn ucis(g: &StandardChess) -> Vec<String> {
    g.legal_moves().iter().map(|&(id, _)| uci(id)).collect()
}

fn frame(g: &StandardChess) -> Frame {
    let spec = StandardChess::spec(g.config());
    let mut f = Frame::new(&spec);
    encode_frame(g, g.mover_seat(), &spec, &mut f.as_mut(&spec)).unwrap();
    f
}

fn target(g: &StandardChess, id: u32) -> Vec<crate::schema::Buf> {
    let spec = StandardChess::spec(g.config());
    let mut legal = Vec::new();
    g.legal_actions(g.mover_seat(), &mut legal);
    let mut t = new_target(&spec);
    encode_target(g, g.mover_seat(), &spec, &legal, id, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    t
}

// ---- perft -------------------------------------------------------------------------

/// Leaf count of the move tree through the GAME's action enumeration (legal_actions,
/// apply_action on the canonical ids).
fn perft(g: &StandardChess, depth: usize) -> u64 {
    let seat = g.mover_seat();
    let mut ids = Vec::new();
    g.legal_actions(seat, &mut ids);
    if depth <= 1 {
        return ids.len() as u64;
    }
    ids.iter()
        .map(|&a| {
            let mut c = g.clone();
            c.apply_action(seat, a).unwrap();
            perft(&c, depth - 1)
        })
        .sum()
}

/// [`perft`] through the policy's view of every node: the frame (its legal tables hold
/// exactly one (from, to, promotion) cell per legal action), and every action's keys
/// (encode_target) decode (decode_action) back to it before it is played.
fn perft_keys(g: &StandardChess, depth: usize) -> u64 {
    let spec = StandardChess::spec(g.config());
    let seat = g.mover_seat();
    let f = frame(g);
    let cells = f.legal[LEGAL_PROMOTION_COND].as_bool().unwrap().iter().filter(|&&x| x == 1).count();
    assert_eq!(cells, f.legal_ids.len(), "one promotion_cond cell per legal move in {}", g.fen());
    if depth == 0 {
        return 1;
    }
    let mut n = 0;
    for &a in &f.legal_ids {
        let t = target(g, a);
        let keys = action_keys_of(&spec, &t);
        assert_eq!(decode_action(g, seat, &spec, &f.legal_ids, &keys).unwrap(), a, "{} in {}", uci(a), g.fen());
        let mut c = g.clone();
        c.apply_action(seat, a).unwrap();
        n += if depth == 1 { 1 } else { perft_keys(&c, depth - 1) };
    }
    n
}

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
/// chessprogramming.org "Perft Results" positions 3-6 (e.p. pins, promotions, castling)
const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const POSITION_4_MIRRORED: &str = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";
const POSITION_5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const POSITION_6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

#[test]
fn perft_through_the_action_enumeration() {
    let cases: [(&str, &[u64]); 7] = [
        (START, &[20, 400, 8902, 197_281]),
        (KIWIPETE, &[48, 2039, 97_862]),
        (POSITION_3, &[14, 191, 2812, 43_238, 674_624]),
        (POSITION_4, &[6, 264, 9467, 422_333]),
        (POSITION_4_MIRRORED, &[6, 264, 9467, 422_333]),
        (POSITION_5, &[44, 1486, 62_379]),
        (POSITION_6, &[46, 2079, 89_890]),
    ];
    for (fen, want) in cases {
        let g = game(fen, &records());
        for (d, &n) in want.iter().enumerate() {
            assert_eq!(perft(&g, d + 1), n, "perft {} of {fen}", d + 1);
        }
    }
    // the start position through new_game (the simulator's initial state)
    assert_eq!(perft(&StandardChess::new_game(&records()), 3), 8902);
}

#[test]
fn perft_through_the_action_keys() {
    for (fen, depth, n) in [(START, 3, 8902), (KIWIPETE, 2, 2039), (POSITION_3, 3, 2812), (POSITION_4, 3, 9467),
                            (POSITION_5, 2, 1486)] {
        assert_eq!(perft_keys(&game(fen, &records()), depth), n, "{fen}");
    }
}

/// Deep perft (release builds: `cargo test --release -p gmrules deep_perft -- --ignored`).
#[test]
#[ignore]
fn deep_perft() {
    assert_eq!(perft(&game(START, &records()), 5), 4_865_609);
    assert_eq!(perft(&game(KIWIPETE, &records()), 4), 4_085_603);
    assert_eq!(perft(&game(POSITION_5, &records()), 4), 2_103_487);
    assert_eq!(perft(&game(POSITION_6, &records()), 4), 3_894_594);
}

// ---- ids, keys, frames ---------------------------------------------------------------

#[test]
fn action_ids_are_standard_uci_and_keys_invert() {
    let mut rng = Rng::new(3);
    let (mut castles, mut promos, mut eps) = (0, 0, 0);
    for _ in 0..40 {
        let mut g = StandardChess::new_game(&ChessConfig { max_plies: 400, ..ChessConfig::default() });
        while g.end().is_none() {
            let flip = g.flip();
            let ids: Vec<u32> = g.legal_moves().iter().map(|&(id, _)| id).collect();
            assert!(ids.windows(2).all(|w| w[0] < w[1]), "ids ascending, unique");
            for &(id, m) in g.legal_moves() {
                assert_eq!(uci(id), UciMove::from_standard(m).to_string(), "{m:?}");
                let keys = move_target(&m, flip).unwrap();
                assert_eq!(action_keys(id, flip), Some(keys));
                assert_eq!(action_of_keys(keys, flip), Some(id));
                castles += matches!(m, Move::Castle { .. }) as usize;
                promos += m.is_promotion() as usize;
                eps += m.is_en_passant() as usize;
            }
            let a = ids[rng.below(ids.len() as u64) as usize];
            let seat = g.mover_seat();
            g.apply_action(seat, a).unwrap();
        }
    }
    assert!(castles > 0 && promos > 0 && eps > 0, "{castles} {promos} {eps}");
    for (u, id) in [("e2e4", (12 * 64 + 28) as u32), ("e7e8q", (4 * 64 + 52) * 64 + 60), ("a2a1n", (64 + 8) * 64)] {
        assert_eq!(uci(id), u);
    }
    assert_eq!(action_parts(N_ACTIONS), None);
    assert_eq!(action_of_keys((0, 64, 0), false), None);
    assert_eq!(action_of_keys((0, 0, 5), false), None);
    for p in 0..64 {
        for flip in [false, true] {
            assert_eq!(pov(unpov(p, flip), flip), p);
        }
    }
}

#[test]
fn castling_is_labelled_king_to_g_or_c() {
    let mut g = game("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", &records());
    let u = ucis(&g);
    assert!(u.contains(&"e1g1".to_string()) && u.contains(&"e1c1".to_string()));
    assert!(!u.contains(&"e1h1".to_string()) && !u.contains(&"e1a1".to_string()));
    let f = frame(&g);
    let e1 = pov(Square::E1, false) as usize;
    let to = &f.legal[LEGAL_TO_COND].as_bool().unwrap()[e1 * 64..(e1 + 1) * 64];
    for (sq, want) in [(Square::G1, 1), (Square::C1, 1), (Square::H1, 0), (Square::A1, 0), (Square::F1, 1), (Square::D1, 1)] {
        assert_eq!(to[pov(sq, false) as usize], want, "{sq}");
    }
    let t = target(&g, id_of_san(&g, "O-O"));
    assert_eq!((t[0].as_i64().unwrap()[0], t[1].as_i64().unwrap()[0], t[2].as_i64().unwrap()[0]),
               (pov(Square::E1, false), pov(Square::G1, false), 0));
    play(&mut g, &["O-O"]);
    let b = g.position().board();
    assert_eq!((b.piece_at(Square::G1).map(|p| p.role), b.piece_at(Square::F1).map(|p| p.role)), (Some(Role::King), Some(Role::Rook)));
    // black's O-O-O in its own (rank-mirrored) POV: e8 → c8 is row 0 → row 0
    let t = target(&g, id_of_san(&g, "O-O-O"));
    assert_eq!((t[0].as_i64().unwrap()[0], t[1].as_i64().unwrap()[0]), (pov(Square::E8, true), pov(Square::C8, true)));
    assert_eq!((pov(Square::E8, true), pov(Square::C8, true)), (60, 58));
    // no castling through an attacked square (f1), or out of check
    let g = game("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1", &records());
    assert!(!ucis(&g).contains(&"e1g1".to_string()) && ucis(&g).contains(&"e1c1".to_string()));
    let g = game("r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1", &records());
    assert!(!ucis(&g).iter().any(|u| u == "e1g1" || u == "e1c1"));
}

#[test]
fn en_passant_plane_and_moves() {
    // the plane marks the e.p. square after EVERY double push (EnPassantMode::Always) ...
    let mut g = StandardChess::new_game(&records());
    play(&mut g, &["e4"]);
    let f = frame(&g);
    let planes = f.obs[0].as_f32().unwrap();
    assert_eq!(planes[12 * 64 + pov(Square::E3, true) as usize], 1.0, "e3 in black's POV");
    assert_eq!(planes[12 * 64..13 * 64].iter().sum::<f32>(), 1.0);
    assert!(g.position().legal_ep_square().is_none(), "... capturable or not");
    // ... and an e.p. capture is a legal move (keys: pawn → the e.p. square)
    let mut g = StandardChess::new_game(&records());
    play(&mut g, &["e4", "a6", "e5", "d5"]);
    assert!(ucis(&g).contains(&"e5d6".to_string()));
    let t = target(&g, id_of_san(&g, "exd6"));
    assert_eq!((t[0].as_i64().unwrap()[0], t[1].as_i64().unwrap()[0]), (pov(Square::E5, false), pov(Square::D6, false)));
    play(&mut g, &["exd6"]);
    assert_eq!(g.position().board().piece_at(Square::D5), None, "the captured pawn leaves d5");
    // an e.p. capture that would expose the king (position 3's rank pin) is not legal
    let g = game("8/8/8/KPp4r/8/8/8/7k w - c6 0 1", &records());
    assert!(!ucis(&g).contains(&"b5c6".to_string()));
    let planes = frame(&g).obs[0].as_f32().unwrap().to_vec();
    assert_eq!(planes[12 * 64 + pov(Square::C6, false) as usize], 1.0, "the plane is still set");
}

#[test]
fn promotions_have_four_keys() {
    for (fen, flip, from, to) in [("8/P6k/8/8/8/8/8/K7 w - - 0 1", false, Square::A7, Square::A8),
                                  ("k7/8/8/8/8/8/1p5K/R7 b - - 0 1", true, Square::B2, Square::A1)] {
        let g = game(fen, &records());
        let f = frame(&g);
        let (fr, to_p) = (pov(from, flip) as usize, pov(to, flip) as usize);
        let cell = &f.legal[LEGAL_PROMOTION_COND].as_bool().unwrap()[(fr * 64 + to_p) * 5..(fr * 64 + to_p + 1) * 5];
        assert_eq!(cell, &[0, 1, 1, 1, 1], "{fen}");
        for (p, piece) in [(1, 'n'), (2, 'b'), (3, 'r'), (4, 'q')] {
            let id = action_of_keys((fr as i64, to_p as i64, p), flip).unwrap();
            assert_eq!(uci(id), format!("{from}{to}{piece}"));
            let t = target(&g, id);
            assert_eq!(t[2].as_i64().unwrap()[0], p);
            assert_eq!(t[5].as_f32().unwrap(), &[0.0, 1.0, 1.0, 1.0, 1.0], "legal_promotion_of");
        }
        // a promotion move needs its promotion key
        let spec = StandardChess::spec(&records());
        assert!(decode_action(&g, g.mover_seat(), &spec, &f.legal_ids, &[fr as i64, to_p as i64, 0]).is_err());
    }
    // a promoted piece is an ordinary piece of its kind (planes, captures)
    let mut g = game("8/P6k/8/8/8/8/8/K7 w - - 0 1", &records());
    let seat = g.mover_seat();
    let id = action_of_keys((pov(Square::A7, false), pov(Square::A8, false), 1), false).unwrap();
    g.apply_action(seat, id).unwrap();
    let planes = frame(&g).obs[0].as_f32().unwrap().to_vec();
    assert_eq!(planes[(6 + 1) * 64 + pov(Square::A8, true) as usize], 1.0, "opponent knight on a8");
}

#[test]
fn frames_agree_with_their_targets_and_the_history() {
    let spec = StandardChess::spec(&records());
    let mut rng = Rng::new(9);
    for gi in 0..30 {
        let cfg = ChessConfig { max_plies: 300, ..if gi % 2 == 0 { ChessConfig::default() } else { records() } };
        let mut g = StandardChess::new_game(&cfg);
        let mut all = Repetitions::default();
        while g.end().is_none() {
            let f = frame(&g);
            let prior = all.see(g.position());
            assert_eq!(g.prior_occurrences(), prior, "repetition memory = whole-game counts");
            assert!(g.remembered_positions() <= g.position().halfmoves() as usize + 1);
            // obs = the converter's per-frame code on the position and its history
            let (mut planes, mut globals) = (vec![0f32; PLANE_LEN], Vec::new());
            encode_position(g.position(), prior, &mut planes, &mut globals);
            assert_eq!(f.obs[0].as_f32().unwrap(), &planes[..]);
            assert_eq!(f.obs[1].as_f32().unwrap(), &globals[..]);
            // the IL masks of every legal action = the env's legality tables at its keys
            let (from, to_cond, promo) = (f.legal[0].as_bool().unwrap(), f.legal[1].as_bool().unwrap(), f.legal[2].as_bool().unwrap());
            for &a in &f.legal_ids {
                let t = target(&g, a);
                let (fr, to, p) = (t[0].as_i64().unwrap()[0] as usize, t[1].as_i64().unwrap()[0] as usize, t[2].as_i64().unwrap()[0] as usize);
                assert_eq!(promo[(fr * 64 + to) * 5 + p], 1);
                let as_f = |m: &[u8]| m.iter().map(|&x| x as f32).collect::<Vec<f32>>();
                assert_eq!(t[3].as_f32().unwrap(), &as_f(from)[..]);
                assert_eq!(t[4].as_f32().unwrap(), &as_f(&to_cond[fr * 64..(fr + 1) * 64])[..]);
                assert_eq!(t[5].as_f32().unwrap(), &as_f(&promo[(fr * 64 + to) * 5..(fr * 64 + to + 1) * 5])[..]);
                assert_eq!(legal_masks(g.position(), g.flip(), (fr as i64, to as i64, p as i64)).unwrap().0.to_vec(), as_f(from));
            }
            let a = f.legal_ids[rng.below(f.legal_ids.len() as u64) as usize];
            let seat = g.mover_seat();
            g.apply_action(seat, a).unwrap();
        }
        let _ = &spec;
    }
}

// ---- game ends -------------------------------------------------------------------

#[test]
fn checkmate_stalemate_and_outcomes() {
    let mut g = StandardChess::new_game(&ChessConfig::default());
    play(&mut g, &["f3", "e5", "g4", "Qh4#"]);
    assert_eq!(g.end(), Some((KnownOutcome::Decisive { winner: Color::Black }, End::Checkmate)));
    assert_eq!((g.turn(), g.outcome(), g.returns()), (Turn::Terminal, vec![-1.0, 1.0], vec![-1.0, 1.0]));
    let mut g = StandardChess::new_game(&ChessConfig::default());
    play(&mut g, &["e3", "a5", "Qh5", "Ra6", "Qxa5", "h5", "h4", "Rah6", "Qxc7", "f6", "Qxd7+", "Kf7", "Qxb7", "Qd3", "Qxb8", "Qh7",
                   "Qxc8", "Kg6", "Qe6"]);
    assert_eq!(g.end(), Some((KnownOutcome::Draw, End::Stalemate)));
    assert_eq!(g.outcome(), vec![0.0, 0.0]);
    let mut out = Vec::new();
    g.legal_actions(1, &mut out);
    assert!(out.is_empty());
    assert!(g.clone().apply_action(1, 0).is_err());
}

#[test]
fn threefold_repetition_is_an_automatic_draw() {
    let shuffle = ["Nf3", "Nf6", "Ng1", "Ng8"];
    let mut g = StandardChess::new_game(&ChessConfig::default());
    play(&mut g, &shuffle);
    assert_eq!((g.prior_occurrences(), g.end()), (1, None));
    let f = frame(&g);
    assert_eq!(&f.obs[1].as_f32().unwrap()[8..], &[1.0, 0.0], "seen once before");
    play(&mut g, &shuffle[..3]);
    assert_eq!(g.end(), None);
    play(&mut g, &shuffle[3..]);
    assert_eq!((g.prior_occurrences(), g.end()), (2, Some((KnownOutcome::Draw, End::Threefold))));
    assert_eq!(g.plies(), 8);
    // off: the record goes on, the frame says "seen twice before"
    let mut g = StandardChess::new_game(&ChessConfig { threefold_draw: false, ..ChessConfig::default() });
    for _ in 0..4 {
        play(&mut g, &shuffle);
    }
    assert_eq!((g.prior_occurrences(), g.end()), (4, None));
    assert_eq!(&frame(&g).obs[1].as_f32().unwrap()[8..], &[1.0, 1.0]);
    // castling rights and e.p. rights are part of a position's identity
    let mut g = StandardChess::new_game(&ChessConfig::default());
    play(&mut g, &["e4", "e5", "Ke2", "Ke7", "Ke1", "Ke8"]);
    assert_eq!(g.prior_occurrences(), 0, "the same board without castling rights is a new position");
    play(&mut g, &["Ke2", "Ke7", "Ke1", "Ke8"]);
    assert_eq!(g.prior_occurrences(), 1);
    // the kings on e2 / e7 (no rights either side) occurred after plies 4 and 8: the
    // third time ends the game
    play(&mut g, &["Ke2"]);
    assert_eq!((g.prior_occurrences(), g.end()), (1, None));
    play(&mut g, &["Ke7"]);
    assert_eq!((g.plies(), g.end()), (12, Some((KnownOutcome::Draw, End::Threefold))));
    let mut g = StandardChess::new_game(&ChessConfig::default());
    play(&mut g, &["e4", "Nf6", "e5", "d5"]);           // exd6 e.p. possible
    play(&mut g, &["Nf3", "Nc6", "Ng1", "Nb8"]);        // same board, the e.p. chance gone
    assert_eq!(g.prior_occurrences(), 0);
    play(&mut g, &["Nf3", "Nc6", "Ng1", "Nb8"]);
    assert_eq!(g.prior_occurrences(), 1);
}

#[test]
fn fifty_move_rule_is_an_automatic_draw_but_mate_wins() {
    let mut g = game("4k3/8/8/8/8/8/8/R3K3 w - - 99 80", &ChessConfig::default());
    assert_eq!(frame(&g).obs[1].as_f32().unwrap()[6], 0.99);
    play(&mut g, &["Ra2"]);
    assert_eq!(g.end(), Some((KnownOutcome::Draw, End::FiftyMoves)));
    // checkmate on the 100th halfmove takes precedence
    let mut g = game("4k3/R7/8/8/8/8/8/1R2K3 w - - 99 80", &ChessConfig::default());
    play(&mut g, &["Rb8#"]);
    assert_eq!(g.end(), Some((KnownOutcome::Decisive { winner: Color::White }, End::Checkmate)));
    // a capture on the 100th halfmove resets the clock
    let mut g = game("4k3/8/8/8/8/8/r7/R3K3 w - - 99 80", &ChessConfig::default());
    play(&mut g, &["Rxa2"]);
    assert_eq!((g.end(), g.position().halfmoves()), (None, 0));
    // off: the clock runs on, the feature is clipped at 1
    let mut g = game("4k3/8/8/8/8/8/8/R3K3 w - - 99 80", &ChessConfig { fifty_move_draw: false, ..ChessConfig::default() });
    play(&mut g, &["Ra2", "Kd8", "Ra3"]);
    assert_eq!(g.end(), None);
    let gl = frame(&g).obs[1].as_f32().unwrap().to_vec();
    assert_eq!((gl[5], gl[6]), (0.81, 1.0), "fullmove 81, halfmove 102");
}

#[test]
fn insufficient_material_is_an_automatic_draw() {
    // K+B vs K+N is not dead; Bxb2 leaves K+B vs K
    let mut g = game("4k3/8/8/8/8/8/1n6/B3K3 w - - 0 1", &ChessConfig::default());
    assert_eq!(g.end(), None);
    play(&mut g, &["Bxb2"]);
    assert_eq!(g.end(), Some((KnownOutcome::Draw, End::InsufficientMaterial)));
    let mut g = game("4k3/8/8/8/8/8/1n6/B3K3 w - - 0 1", &ChessConfig { insufficient_material_draw: false, ..ChessConfig::default() });
    play(&mut g, &["Bxb2"]);
    assert_eq!(g.end(), None);
    // (python-chess' is_insufficient_material agrees on every one; b8 / c1 dark, c8 light)
    for (fen, dead) in [("4k3/8/8/8/8/8/8/4K3 w - - 0 1", true), ("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1", true),
                        ("4k3/8/8/8/8/8/8/2B1K3 b - - 0 1", true), ("1b2k3/8/8/8/8/8/8/2B1K3 w - - 0 1", true),
                        ("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1", false), ("1n2k3/8/8/8/8/8/8/1N2K3 w - - 0 1", false),
                        ("4k3/8/8/8/8/8/8/NN2K3 w - - 0 1", false), ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false),
                        ("1n2k3/8/8/8/8/8/8/2B1K3 w - - 0 1", false)] {
        let g = game(fen, &ChessConfig::default());
        assert_eq!(g.end() == Some((KnownOutcome::Draw, End::InsufficientMaterial)), dead, "{fen}");
    }
}

#[test]
fn the_ply_cap_is_an_adjudicated_draw() {
    let cfg = ChessConfig { max_plies: 4, ..ChessConfig::default() };
    let mut g = StandardChess::new_game(&cfg);
    play(&mut g, &["e4", "e5", "Nf3"]);
    assert_eq!(g.end(), None);
    play(&mut g, &["Nc6"]);
    assert_eq!((g.end(), g.outcome()), (Some((KnownOutcome::Draw, End::PlyCap)), vec![0.0, 0.0]));
    assert_eq!(StandardChess::spec(&cfg).max_decisions, 4);
}

// ---- spec, config, replay, battery -----------------------------------------------------

#[test]
fn spec_config_and_replay() {
    let spec = StandardChess::spec(&ChessConfig::default());
    spec.validate().unwrap();
    assert_eq!((spec.layout(), spec.schema.as_str(), spec.max_decisions), (Layout::Interleaved, SCHEMA, DEFAULT_MAX_PLIES));
    assert_eq!(spec.meta, json!({"global_features": 10}));
    assert_eq!(spec.action_keys, vec!["from_idx", "to_idx", "promotion"]);
    let names = |ts: &[TensorSpec]| ts.iter().map(|t| (t.name.clone(), t.shape.clone(), t.dtype, t.bits)).collect::<Vec<_>>();
    use crate::schema::DType::{Bool, F32, I64};
    assert_eq!(names(&spec.obs), vec![("board_seq".into(), vec![14, 8, 8], F32, true), ("globals_seq".into(), vec![10], F32, false)]);
    assert_eq!(names(&spec.legal), vec![("from".into(), vec![64], Bool, false), ("to_cond".into(), vec![64, 64], Bool, false),
                                        ("promotion_cond".into(), vec![64, 64, 5], Bool, false)]);
    assert_eq!(names(&spec.target), vec![("from_idx".into(), vec![], I64, false), ("to_idx".into(), vec![], I64, false),
                                         ("promotion".into(), vec![], I64, false), ("legal_from".into(), vec![64], F32, true),
                                         ("legal_to_of".into(), vec![64], F32, true), ("legal_promotion_of".into(), vec![5], F32, true)]);
    // configs: JSON defaults, bounds
    let c: ChessConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(c, ChessConfig::default());
    assert!(serde_json::from_str::<ChessConfig>(r#"{"auto_threefold": true}"#).is_err());
    assert!(StandardChess::check_config(&ChessConfig { max_plies: 0, ..c.clone() }).is_err());
    assert!(StandardChess::check_config(&ChessConfig { max_plies: RECORD_MAX_PLIES + 1, ..c.clone() }).is_err());
    StandardChess::check_config(&ChessConfig::records()).unwrap();
    // a replayed trajectory: seats alternate from White, no chance events
    let g = StandardChess::new_game(&c);
    let e4 = id_of_san(&g, "e4");
    let mut g2 = g.clone();
    g2.apply_action(0, e4).unwrap();
    let e5 = id_of_san(&g2, "e5");
    let r = replay::<StandardChess>(&c, &[Event::Act(0, e4), Event::Act(1, e5)]).unwrap();
    assert_eq!((r.decisions.len(), r.decisions[1].seat, r.terminal()), (2, 1, false));
    assert!(replay::<StandardChess>(&c, &[Event::Act(1, e4)]).is_err(), "black does not start");
    assert!(replay::<StandardChess>(&c, &[Event::Chance(0)]).is_err());
    let mut out = Vec::new();
    g.chance_outcomes(&mut out);
    assert!(out.is_empty() && g.clone().apply_chance(0).is_err());
    assert!(g.clone().apply_action(1, e4).is_err(), "not black's turn");
    assert!(g.clone().apply_action(0, e5).is_err(), "e7e5 is no move of White");
    assert_eq!(g.action_string(0, e4), "e2e4");
}

#[test]
fn conformance_battery() {
    // few long random games each (causality replays are quadratic in the game length)
    let rep = check_game::<StandardChess>(&ChessConfig { max_plies: 160, ..ChessConfig::default() }, 6, 1).unwrap();
    assert_eq!(rep.terminal, 6);
    let rep = check_game::<StandardChess>(&ChessConfig { max_plies: 120, ..ChessConfig::records() }, 3, 2).unwrap();
    assert_eq!(rep.terminal, 3);
}
