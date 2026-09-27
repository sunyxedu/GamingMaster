//! Shogi rules / frames / simulator tests (the converter parity tests live next to the
//! converter, `gmdata/src/shogi.rs`, which owns the CSA parser).

use super::*;
use crate::frame::{decode_action, encode_frame, encode_target, new_target, Frame};
use crate::game::Layout;
use crate::invariance::check_game;
use crate::sim::{SimOptions, VecSim};
use crate::trajectory::{replay, Event};

fn cfg(sfen: &str, max_plies: usize) -> ShogiConfig {
    ShogiConfig { sfen: sfen.into(), max_plies, ..ShogiConfig::default() }
}

fn usi(g: &Shogi, s: &str) -> u32 {
    let a = action_of(&parse_usi_move(s).unwrap_or_else(|| panic!("bad USI {s}")));
    assert!(g.legal_actions_ids().binary_search(&a).is_ok(), "{s} is not legal in {}", g.state_string());
    a
}

fn play(g: &mut Shogi, moves: &[&str]) {
    for m in moves {
        let a = usi(g, m);
        let seat = g.side_to_move();
        g.apply_action(seat, a).unwrap();
    }
}

fn frame(g: &Shogi) -> Frame {
    let spec = Shogi::spec(&ShogiConfig::default());
    let mut f = Frame::new(&spec);
    encode_frame(g, g.side_to_move(), &spec, &mut f.as_mut(&spec)).unwrap();
    f
}

// ---------------------------------------------------------------- perft

#[test]
fn perft_from_the_start_position() {
    // the well-known values (e.g. yasai, github.com/sugyan/yasai, src/position.rs `perft`)
    let g = Shogi::new_game(&ShogiConfig::default());
    assert_eq!(perft(&g, 1), 30);
    assert_eq!(perft(&g, 2), 900);
    assert_eq!(perft(&g, 3), 25_470);
    assert_eq!(perft(&g, 4), 719_731);
}

/// The position with the maximum number of legal moves in shogi (593, drops of every
/// piece type), perft values from yasai (github.com/sugyan/yasai, src/position.rs, test
/// `perft`: "from maximum moves", depth 1 = 593, depth 2 = 105677).
const MAX_MOVES_SFEN: &str = "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b RBGSNLP3g3n17p 1";

#[test]
fn perft_with_drops_maximum_moves_position() {
    let g = Shogi::new_game(&cfg(MAX_MOVES_SFEN, 256));
    assert_eq!(perft(&g, 1), 593);
    let drops = g.legal_moves().iter().filter(|m| m.drop != NONE).count();
    let promos = g.legal_moves().iter().filter(|m| m.promotion).count();
    assert!(drops > 400 && promos > 0, "{drops} drops, {promos} promotions");
    assert_eq!(perft(&g, 2), 105_677);
}

// ---------------------------------------------------------------- frames

#[test]
fn spec_and_start_frame() {
    let c = ShogiConfig::default();
    let spec = Shogi::spec(&c);
    spec.validate().unwrap();
    assert_eq!(spec.layout(), Layout::Interleaved);
    assert_eq!(spec.max_decisions, DEFAULT_MAX_PLIES);
    let names = |ts: &[TensorSpec]| ts.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&spec.obs), ["board_seq", "hand_seq", "globals_seq"]);
    assert_eq!(names(&spec.legal), ["mode", "from", "to_cond", "promo_cond", "drop_piece", "drop_to_cond"]);
    assert_eq!(names(&spec.target), ["dropped", "from_idx", "to_idx", "promoted", "piece_idx", "drop_to_idx", "legal_mode",
                                      "legal_from", "legal_to_of", "legal_promo_of", "legal_drop_piece", "legal_drop_to_of"]);
    assert_eq!(spec.action_keys, ["dropped", "from_idx", "to_idx", "promoted", "piece_idx", "drop_to_idx"]);
    assert_eq!(spec.schema_meta()["global_features"], 7);
    let g = Shogi::new_game(&c);
    let f = frame(&g);
    let board = f.obs[0].as_f32().unwrap();
    assert_eq!(board.iter().sum::<f32>(), 40.0);
    // own king (plane 7) on 5i = square 76, the opponent's (plane 21) on 5a = square 4
    assert_eq!((board[7 * 81 + 76], board[21 * 81 + 4]), (1.0, 1.0));
    let hand = f.obs[1].as_f32().unwrap();
    assert!((0..14).all(|i| hand[i * 4..i * 4 + 4] == [0.0, 0.0, 0.0, 1.0]));
    assert_eq!(f.obs[2].as_f32().unwrap(), &[1.0 / 512.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    let legal = |i: usize| f.legal[i].as_bool().unwrap();
    assert_eq!(legal(0), &[1, 0]);
    assert_eq!(legal(1).iter().filter(|&&x| x == 1).count(), 17);
    // 7g (56) → 7f (47) only, without promotion
    let to: Vec<usize> = (0..81).filter(|&t| legal(2)[56 * 81 + t] == 1).collect();
    assert_eq!(to, vec![47]);
    assert_eq!(&legal(3)[(56 * 81 + 47) * 2..(56 * 81 + 47) * 2 + 2], &[1, 0]);
    assert!(legal(4).iter().all(|&x| x == 0) && legal(5).iter().all(|&x| x == 0));
    assert_eq!(f.legal_ids.len(), 30);
}

/// The start position is symmetric under the 180° rotation: black's frame of the start and
/// white's frame of the start with white to move are the same but for globals[1].
#[test]
fn white_frames_are_the_rotated_board() {
    let b = Shogi::new_game(&ShogiConfig::default());
    let w = Shogi::new_game(&cfg("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1", 256));
    let (fb, fw) = (frame(&b), frame(&w));
    assert_eq!(fb.obs[0], fw.obs[0]);
    assert_eq!(fb.obs[1], fw.obs[1]);
    assert_eq!(fb.legal, fw.legal);
    let (gb, gw) = (fb.obs[2].as_f32().unwrap(), fw.obs[2].as_f32().unwrap());
    assert_eq!((gb[1], gw[1]), (1.0, 0.0));
    assert_eq!((&gb[..1], &gb[2..]), (&gw[..1], &gw[2..]));
    // the rotated move has the same POV keys: 7g7f for black = 3c3d for white
    let spec = Shogi::spec(&ShogiConfig::default());
    let keys = |g: &Shogi, s: &str| {
        let mut t = new_target(&spec);
        let legal = g.legal_actions_ids().to_vec();
        encode_target(g, g.side_to_move(), &spec, &legal, usi(g, s), &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
        (crate::frame::action_keys_of(&spec, &t), t)
    };
    let (kb, tb) = keys(&b, "7g7f");
    let (kw, tw) = keys(&w, "3c3d");
    assert_eq!(kb, kw);
    assert_eq!(kb, vec![0, 56, 47, 0, 0, 0]);
    assert_eq!(tb, tw);
    assert_eq!(decode_action(&w, WHITE, &spec, w.legal_actions_ids(), &kw).unwrap(), usi(&w, "3c3d"));
}

/// Drops, captured promoted pieces going back unpromoted, promotions (optional and
/// forced), hands in the frames.
#[test]
fn drops_promotions_and_hands_in_frames() {
    let spec = Shogi::spec(&ShogiConfig::default());
    let mut g = Shogi::new_game(&ShogiConfig::default());
    play(&mut g, &["7g7f", "3c3d", "8h2b+"]); // bishop takes bishop, promotes
    assert_eq!(g.board().hand[BLACK][BISHOP as usize], 1);
    let f = frame(&g); // white to move: its hand is empty, black's holds a bishop
    let hand = f.obs[1].as_f32().unwrap();
    let opp_bishop = (7 + (BISHOP as usize - 1)) * 4;
    assert_eq!(&hand[opp_bishop..opp_bishop + 4], &[1.0, 1.0, 0.25, 1.0]);
    // the horse on 2b is an opponent PROM_BISHOP (plane 14 + 12) at POV 80 − 16 = 64
    assert_eq!(f.obs[0].as_f32().unwrap()[(14 + 12) * 81 + 64], 1.0);
    assert!(!g.in_check());
    play(&mut g, &["3a2b"]); // silver takes the horse: a BISHOP in white's hand
    assert_eq!(g.board().hand[WHITE][BISHOP as usize], 1);
    assert_eq!(g.board().hand[WHITE].iter().sum::<u32>(), 1);
    let f = frame(&g); // black to move: both hold a bishop, both may drop
    assert_eq!(f.legal[0].as_bool().unwrap(), &[1, 1]);
    let drop_piece = f.legal[4].as_bool().unwrap();
    assert_eq!(drop_piece[BISHOP as usize - 1], 1);
    assert!(drop_piece[7..].iter().all(|&x| x == 0), "never the opponent's slots");
    // B*5e: the drop's targets and decode
    let a = usi(&g, "B*5e");
    let mut t = new_target(&spec);
    encode_target(&g, BLACK, &spec, g.legal_actions_ids(), a, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    let keys = crate::frame::action_keys_of(&spec, &t);
    assert_eq!(keys, vec![1, 0, 0, 0, 5, 40]);
    assert_eq!(decode_action(&g, BLACK, &spec, g.legal_actions_ids(), &keys).unwrap(), a);
    // the other branch's keys are ignored on a drop, and vice versa
    assert_eq!(decode_action(&g, BLACK, &spec, g.legal_actions_ids(), &[1, 33, 7, 1, 5, 40]).unwrap(), a);
    assert!(decode_action(&g, BLACK, &spec, g.legal_actions_ids(), &[1, 0, 0, 0, 12, 40]).is_err(), "opponent slot");
    assert!(decode_action(&g, BLACK, &spec, g.legal_actions_ids(), &[2, 0, 0, 0, 5, 40]).is_err());
    let drop_to_of = t[11].as_f32().unwrap();
    let row = &f.legal[5].as_bool().unwrap()[5 * 81..6 * 81];
    assert!(drop_to_of.iter().zip(row).all(|(&x, &y)| x == y as f32), "legal_drop_to_of = drop_to_cond[piece]");
    // a forced promotion: a pawn reaching the last rank (SFEN: black pawn on 1b)
    let g = Shogi::new_game(&cfg("4k4/8P/9/9/9/9/9/9/4K4 b - 1", 256));
    let a_promo = usi(&g, "1b1a+");
    assert!(parse_usi_move("1b1a").map(|m| g.legal_actions_ids().binary_search(&action_of(&m)).is_err()).unwrap());
    let mut t = new_target(&spec);
    encode_target(&g, BLACK, &spec, g.legal_actions_ids(), a_promo, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    assert_eq!(t[3].as_bool().unwrap(), &[1]);
    assert_eq!(t[9].as_f32().unwrap(), &[0.0, 1.0], "legal_promo_of: promotion only");
    let f = frame(&g);
    let (fi, ti) = (pov(parse_usi_move("1b1a").unwrap().from.unwrap(), true) as usize, pov(parse_usi_move("1b1a").unwrap().to, true) as usize);
    assert_eq!(&f.legal[3].as_bool().unwrap()[(fi * 81 + ti) * 2..(fi * 81 + ti) * 2 + 2], &[0, 1]);
}

// ---------------------------------------------------------------- game ends

const SHUFFLE: [&str; 4] = ["5i5h", "5a5b", "5h5i", "5b5a"];

#[test]
fn sennichite_is_a_draw_at_the_fourth_occurrence() {
    let mut g = Shogi::new_game(&ShogiConfig::default());
    for i in 0..12 {
        assert!(g.end().is_none(), "ply {i}");
        let rep = g.repetitions();
        assert_eq!(rep, (i / 4) as u32, "ply {i}");
        let glb = frame(&g).obs[2].as_f32().unwrap().to_vec();
        assert_eq!(&glb[4..], &[(rep >= 1) as u8 as f32, (rep >= 2) as u8 as f32, 0.0]);
        play(&mut g, &[SHUFFLE[i % 4]]);
    }
    assert_eq!(g.end(), Some((0, EndReason::Sennichite)));
    assert_eq!((g.turn(), g.outcome()), (Turn::Terminal, vec![0.0, 0.0]));
    // replaying a record: no game end, the count goes on (rep ≥ 3 on the 4th occurrence)
    let mut r = Shogi::replaying(Board::hirate());
    for i in 0..16 {
        let a = usi(&r, SHUFFLE[i % 4]);
        let seat = r.side_to_move();
        r.apply_action(seat, a).unwrap();
    }
    assert_eq!((r.end(), r.repetitions()), (None, 4));
    assert_eq!(&frame(&r).obs[2].as_f32().unwrap()[4..], &[1.0, 1.0, 1.0]);
}

/// Black's rook checks the white king on every move of the cycle: at the 4th occurrence
/// black loses (and white would, with the colours swapped).
#[test]
fn perpetual_check_loses() {
    let cycle = ["1c2c", "2a1a", "2c1c", "1a2a"];
    let mut g = Shogi::new_game(&cfg("7k1/9/8R/9/9/9/9/9/K8 b - 1", 256));
    for i in 0..12 {
        assert!(g.end().is_none(), "ply {i}");
        play(&mut g, &[cycle[i % 4]]);
    }
    assert_eq!(g.end(), Some((-1, EndReason::PerpetualCheck)));
    assert_eq!(g.outcome(), vec![-1.0, 1.0]);
    // the mirrored position: white checks, white loses
    let mut g = Shogi::new_game(&cfg("k8/9/9/9/9/9/r8/9/1K7 w - 1", 256));
    for i in 0..12 {
        play(&mut g, &[["9g8g", "8i9i", "8g9g", "9i8i"][i % 4]]);
    }
    assert_eq!(g.end(), Some((1, EndReason::PerpetualCheck)));
    // checks by one side on only some of the moves: a plain draw
    let mut g = Shogi::new_game(&cfg("7k1/9/8R/9/9/9/9/9/K8 b - 1", 256));
    let mixed = ["1c2c", "2a1a", "2c1c", "1a2a", "1c1d", "2a2b", "1d1c", "2b2a"];
    for i in 0..24 {
        if g.end().is_some() {
            break;
        }
        play(&mut g, &[mixed[i % 8]]);
    }
    assert_eq!(g.end(), Some((0, EndReason::Sennichite)));
}

#[test]
fn checkmate_max_plies_and_declaration() {
    // G*1b mates the cornered king (the 1c gold protects it)
    let mut g = Shogi::new_game(&cfg("8k/9/8G/9/9/9/9/9/K8 b G 1", 256));
    play(&mut g, &["G*1b"]);
    assert_eq!(g.end(), Some((1, EndReason::Checkmate)));
    assert_eq!((g.outcome(), g.returns()), (vec![1.0, -1.0], vec![1.0, -1.0]));
    // the ply cap: a draw after max_plies plies
    let mut g = Shogi::new_game(&cfg("", 6));
    play(&mut g, &["7g7f", "3c3d", "2g2f", "8c8d", "2f2e"]);
    assert!(g.end().is_none());
    play(&mut g, &["8d8e"]);
    assert_eq!(g.end(), Some((0, EndReason::MaxPlies)));
    // entering king (CSA rule 27): black's king steps into the camp, white moves, black
    // declares (2 dragons + 2 horses + 8 tokins = 28 points, 12 pieces in the camp)
    let decl = "1+R+R+B+B4/+P+P+P+P+P+P+P+P1/9/4K4/9/9/9/9/4k4 b - 1";
    let mut g = Shogi::new_game(&cfg(decl, 256));
    assert!(g.end().is_none());
    play(&mut g, &["5d5c", "5i5h"]);
    assert_eq!(g.end(), Some((1, EndReason::Declaration)));
    // with entering_king = none the game goes on
    let mut g = Shogi::new_game(&ShogiConfig { entering_king: EnteringKing::None, ..cfg(decl, 256) });
    play(&mut g, &["5d5c", "5i5h"]);
    assert!(g.end().is_none() && g.board().can_declare());
    // a start position that is already over is terminal at once
    assert_eq!(Shogi::new_game(&cfg("1+R+R+B+B4/+P+P+P+P+P+P+P+P1/4K4/9/9/9/9/9/4k4 b - 1", 256)).turn(), Turn::Terminal);
}

#[test]
fn config_parsing_and_validation() {
    let c: ShogiConfig = serde_json::from_str(r#"{"max_plies": 512, "entering_king": "none"}"#).unwrap();
    assert_eq!((c.max_plies, c.entering_king), (512, EnteringKing::None));
    assert!(serde_json::from_str::<ShogiConfig>(r#"{"entering_king": "csa24"}"#).is_err());
    assert!(serde_json::from_str::<ShogiConfig>(r#"{"bogus": 1}"#).is_err());
    assert!(Shogi::check_config(&cfg("", 0)).is_err());
    assert!(Shogi::check_config(&cfg("9/9/9", 256)).is_err());
    assert!(Shogi::check_config(&cfg("4k4/9/9/9/9/9/9/9/4K4 b 19p 1", 256)).is_err());
    Shogi::check_config(&cfg(MAX_MOVES_SFEN, 256)).unwrap();
    Shogi::check_config(&cfg(&format!("sfen {MAX_MOVES_SFEN}"), 256)).unwrap();
    // the SFEN's move number starts the frames' move count
    let g = Shogi::new_game(&cfg("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 100", 256));
    assert_eq!(frame(&g).obs[2].as_f32().unwrap()[0], 100.0 / 512.0);
    assert_eq!(g.state_string(), "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 100");
    assert_eq!(g.action_string(BLACK, usi(&g, "7g7f")), "7g7f");
}

// ---------------------------------------------------------------- simulator

/// The replaying state (the converter's) emits the adjudicating state's frames and
/// targets on every position both reach, and the simulator's rows are the replay frames.
#[test]
fn replaying_and_simulated_frames_agree() {
    let c = cfg("", 60);
    let spec = Shogi::spec(&c);
    let mut rng = crate::rng::Rng::new(7);
    for _ in 0..3 {
        let ev = crate::invariance::random_trajectory::<Shogi>(&c, &mut rng, 1000);
        let r = replay::<Shogi>(&c, &ev).unwrap();
        let mut rec = Shogi::replaying(Board::hirate());
        for (d, e) in r.decisions.iter().zip(&ev) {
            let Event::Act(seat, a) = *e else { unreachable!() };
            let mut f = Frame::new(&spec);
            encode_frame(&rec, seat, &spec, &mut f.as_mut(&spec)).unwrap();
            f.check_equal(&d.frame, &spec).unwrap();
            let mut t = new_target(&spec);
            encode_target(&rec, seat, &spec, &f.legal_ids, a, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
            crate::frame::check_targets_equal(&t, &d.target, &spec).unwrap();
            rec.apply_action(seat, a).unwrap();
        }
    }
    // VecSim rows from recorded prefixes = the replay frames
    let ev = crate::invariance::random_trajectory::<Shogi>(&c, &mut rng, 1000);
    let r = replay::<Shogi>(&c, &ev).unwrap();
    let n = r.decisions.len().min(12);
    let mut sim = VecSim::<Shogi>::new(c.clone(), n, SimOptions { threads: 2, ..SimOptions::default() }).unwrap();
    let prefixes: Vec<Vec<Event>> = (0..n).map(|d| ev[..d].to_vec()).collect();
    sim.reset_from(&(0..n).collect::<Vec<_>>(), &vec![0; n], &prefixes).unwrap();
    for d in 0..n {
        let b = sim.buffers();
        let mut f = Frame::new(&spec);
        for (i, t) in spec.obs.iter().enumerate() {
            f.obs[i].view_mut().copy_from(b.obs[i].range(d * t.numel()..(d + 1) * t.numel()));
        }
        for (i, t) in spec.legal.iter().enumerate() {
            f.legal[i].view_mut().copy_from(b.legal[i].range(d * t.numel()..(d + 1) * t.numel()));
        }
        f.legal_ids = sim.legal(d).to_vec();
        r.decisions[d].frame.check_equal(&f, &spec).unwrap();
    }
}

#[test]
fn conformance_battery() {
    // full-board openings (short cap: the O(n²) causality check), an endgame with every
    // piece type in hand (drops, mates), and a bare-kings-plus-rooks ending (repetitions)
    let rep = check_game::<Shogi>(&cfg("", 24), 4, 1).unwrap();
    assert_eq!(rep.terminal, 4);
    assert!(rep.max_legal >= 30);
    let rep = check_game::<Shogi>(&cfg("4k4/9/9/9/9/9/9/9/4K4 b RBGSNLPrbgsnlp 1", 40), 3, 2).unwrap();
    assert_eq!(rep.terminal, 3);
    assert!(rep.max_legal > 100, "drops: {}", rep.max_legal);
    let rep = check_game::<Shogi>(&cfg("8k/9/9/9/9/9/9/9/K8 b R 1", 60), 3, 3).unwrap();
    assert_eq!(rep.terminal, 3);
}
