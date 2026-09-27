//! Go rules / frames / scoring tests: hand-checked positions (captures, ko, snapback,
//! suicide, Tromp-Taylor counts), the end rules, the frame / target / decoding contract,
//! the one-pass legal mask against the converter's `play` (and against a frozen copy of
//! the converter's original code), and the conformance battery on several boards.

use super::*;
use crate::frame::{decode_action, encode_frame, encode_target, new_target, Frame};
use crate::game::Layout;
use crate::invariance::check_game;
use crate::rng::Rng;
use crate::trajectory::{replay, Event};

/// The converter's rules exactly as `gmdata go convert` v2 shipped them (before they
/// moved into gmrules), frozen: the reference the moved code and the one-pass legal
/// mask are held to.
mod reference {
    use std::collections::HashSet;
    type Point = (i32, i32);
    const EMPTY: u8 = 0;
    const BLACK: u8 = 1;
    const WHITE: u8 = 2;
    fn opponent(c: u8) -> u8 {
        if c == BLACK {
            WHITE
        } else {
            BLACK
        }
    }
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Board {
        pub n: i32,
        pub g: Vec<u8>,
        pub ko: Option<Point>,
    }
    impl Board {
        pub fn new(n: i32) -> Self {
            Board { n, g: vec![EMPTY; (n * n) as usize], ko: None }
        }
        pub fn at(&self, p: Point) -> u8 {
            self.g[(p.0 * self.n + p.1) as usize]
        }
        pub fn set(&mut self, p: Point, v: u8) {
            let n = self.n;
            self.g[(p.0 * n + p.1) as usize] = v;
        }
        fn neighbors(&self, (r, c): Point) -> Vec<Point> {
            let mut v = Vec::with_capacity(4);
            if r > 0 {
                v.push((r - 1, c));
            }
            if r < self.n - 1 {
                v.push((r + 1, c));
            }
            if c > 0 {
                v.push((r, c - 1));
            }
            if c < self.n - 1 {
                v.push((r, c + 1));
            }
            v
        }
        fn group(&self, p: Point) -> (HashSet<Point>, HashSet<Point>) {
            let color = self.at(p);
            let (mut stones, mut libs) = (HashSet::new(), HashSet::new());
            let mut stack = vec![p];
            while let Some(q) = stack.pop() {
                if !stones.insert(q) {
                    continue;
                }
                for nb in self.neighbors(q) {
                    let v = self.at(nb);
                    if v == EMPTY {
                        libs.insert(nb);
                    } else if v == color && !stones.contains(&nb) {
                        stack.push(nb);
                    }
                }
            }
            (stones, libs)
        }
        pub fn play(&mut self, color: u8, p: Option<Point>) -> bool {
            let Some((r, c)) = p else {
                self.ko = None;
                return true;
            };
            if !(0..self.n).contains(&r) || !(0..self.n).contains(&c) || self.at((r, c)) != EMPTY || self.ko == Some((r, c)) {
                return false;
            }
            self.set((r, c), color);
            let opp = opponent(color);
            let mut captured: Vec<Point> = Vec::new();
            let mut seen: HashSet<Point> = HashSet::new();
            for q in self.neighbors((r, c)) {
                if self.at(q) == opp && !seen.contains(&q) {
                    let (stones, libs) = self.group(q);
                    seen.extend(stones.iter().copied());
                    if libs.is_empty() {
                        for s in &stones {
                            self.set(*s, EMPTY);
                        }
                        captured.extend(stones);
                    }
                }
            }
            let (own, own_libs) = self.group((r, c));
            if own_libs.is_empty() {
                self.set((r, c), EMPTY);
                return false;
            }
            self.ko = if captured.len() == 1 && own.len() == 1 && own_libs.len() == 1 && own_libs.contains(&captured[0]) {
                Some(captured[0])
            } else {
                None
            };
            true
        }
    }
}

/// A board from rows of `X` (black), `O` (white), `.` (empty).
fn board(rows: &[&str]) -> Board {
    let n = rows.len() as i32;
    let mut b = Board::new(n);
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), rows.len(), "square boards only");
        for (c, ch) in row.chars().enumerate() {
            let v = match ch {
                'X' => BLACK,
                'O' => WHITE,
                _ => EMPTY,
            };
            b.set((r as i32, c as i32), v);
        }
    }
    b
}

fn rows(b: &Board) -> Vec<String> {
    let n = b.n as usize;
    (0..n).map(|r| b.g[r * n..(r + 1) * n].iter().map(|&v| [b'.', b'X', b'O'][v as usize] as char).collect()).collect()
}

/// `play` legality of every point (the converter's definition of a legal move).
fn legal_by_play(b: &Board, color: u8) -> Vec<u8> {
    let n = b.n;
    (0..n * n).map(|i| b.clone().play(color, Some((i / n, i % n))) as u8).collect()
}

fn mask(b: &Board, color: u8) -> Vec<u8> {
    let mut m = vec![0u8; b.g.len()];
    b.legal_points(color, &mut m);
    m
}

#[test]
fn captures_single_multi_edge_corner() {
    // one stone in the centre, taken on its last liberty; the capturing stone keeps
    // four liberties: no ko
    let mut b = board(&[".....", "..X..", ".XO..", "..X..", "....."]);
    assert!(b.play(BLACK, Some((2, 3))));
    assert_eq!((b.at((2, 2)), b.ko), (EMPTY, None));
    // four stones (a bent group) at once
    let mut b = board(&[".XXX.", "XOOOX", ".XO..", "..X..", "....."]);
    assert!(b.play(BLACK, Some((2, 3))));
    assert_eq!(rows(&b), [".XXX.", "X...X", ".X.X.", "..X..", "....."]);
    // two stones on the top edge
    let mut b = board(&["XOO..", ".XX..", ".....", ".....", "....."]);
    assert!(b.play(BLACK, Some((0, 3))));
    assert_eq!(rows(&b)[0], "X..X.");
    // the corner stone
    let mut b = board(&["OX...", ".....", ".....", ".....", "....."]);
    assert!(b.play(BLACK, Some((1, 0))));
    assert_eq!((b.at((0, 0)), b.ko), (EMPTY, None));
    // three separate stones in one move: no ko
    let mut b = board(&["O.OX.", "XOX..", ".X...", ".....", "....."]);
    assert!(b.play(BLACK, Some((0, 1))));
    assert_eq!(rows(&b)[..2], [".X.X.", "X.X.."]);
    assert_eq!(b.ko, None);
    // a stone without an empty neighbour is legal when it captures: W at the corner takes
    // both black stones (in atari); two captured stones: no ko
    let mut b = board(&[".XO..", "XO...", "O....", ".....", "....."]);
    assert_eq!(mask(&b, WHITE)[0], 1);
    assert!(b.play(WHITE, Some((0, 0))));
    assert_eq!(rows(&b)[..3], ["O.O..", ".O...", "O...."]);
    assert_eq!(b.ko, None);
}

#[test]
fn ko_ban_timing_and_clearing() {
    // B takes the white stone at (1,1) from (1,2), which is left with that single
    // liberty: ko at (1,1)
    let ko = board(&[".XO..", "XO.O.", ".XO..", ".....", "....."]);
    let mut b = ko.clone();
    assert!(b.play(BLACK, Some((1, 2))));
    assert_eq!((b.at((1, 1)), b.ko), (EMPTY, Some((1, 1))));
    // the immediate retake is illegal — for whoever moves next (the converter's SGF
    // colours need not alternate)
    assert!(!b.clone().play(WHITE, Some((1, 1))));
    assert!(!b.clone().play(BLACK, Some((1, 1))));
    assert_eq!((mask(&b, WHITE)[6], mask(&b, BLACK)[6]), (0, 0));
    assert_eq!(legal_by_play(&b, WHITE), mask(&b, WHITE));
    // a threat exchange clears the ban: the retake is legal and sets the ko the other way
    let mut c = b.clone();
    assert!(c.play(WHITE, Some((4, 4))) && c.ko.is_none());
    assert!(c.play(BLACK, Some((4, 0))) && c.ko.is_none());
    assert!(c.play(WHITE, Some((1, 1))));
    assert_eq!((c.at((1, 2)), c.ko), (EMPTY, Some((1, 2))));
    // so does a pass
    let mut d = b.clone();
    assert!(d.play(WHITE, None) && d.ko.is_none());
    assert!(d.clone().play(WHITE, Some((1, 1))));
    assert_eq!(mask(&d, WHITE)[6], 1);
    // taking one stone with a stone that keeps two liberties is no ko
    let mut e = board(&[".....", ".XO..", "XO...", ".XO..", "....."]);
    assert!(e.play(BLACK, Some((2, 2))));
    assert_eq!((e.at((2, 1)), e.ko), (EMPTY, None));
}

#[test]
fn snapback_is_no_ko() {
    // W {(1,0), (1,1), (1,2), (0,2)} has the liberties (0,0), (0,1); B throws in at
    // (0,1), W takes it from (0,0) (one stone, by a five-stone group: no ko), B plays
    // (0,1) again — no empty neighbour, but it captures the five stones
    let mut b = board(&["..OX.", "OOOX.", "XXX..", ".....", "....."]);
    assert!(b.play(BLACK, Some((0, 1))));
    assert!(b.play(WHITE, Some((0, 0))));
    assert_eq!((b.at((0, 1)), b.ko), (EMPTY, None));
    assert_eq!(mask(&b, BLACK)[1], 1);
    assert!(b.play(BLACK, Some((0, 1))));
    assert_eq!(rows(&b)[..3], [".X.X.", "...X.", "XXX.."]);
    assert_eq!(b.ko, None);
}

#[test]
fn suicide_is_illegal_single_and_multi() {
    let b = board(&[".X...", "X....", ".....", ".....", "....."]);
    assert!(!b.clone().play(WHITE, Some((0, 0))), "single-stone suicide");
    assert!(b.clone().play(BLACK, Some((0, 0))), "filling an own eye is legal");
    assert_eq!((mask(&b, WHITE)[0], mask(&b, BLACK)[0]), (0, 1));
    let b = board(&["O.X..", "XX...", ".....", ".....", "....."]);
    assert!(!b.clone().play(WHITE, Some((0, 1))), "two-stone suicide");
    assert_eq!((mask(&b, WHITE)[1], mask(&b, BLACK)[1]), (0, 1));
    let mut after = b.clone();
    assert!(!after.play(WHITE, Some((0, 1))));
    assert_eq!(after, b, "an illegal move changes nothing");
}

#[test]
fn legal_mask_equals_play_on_random_positions() {
    // random games with random colours (the converter's records need not alternate),
    // passes, on every board size up to 9 plus 13 and 19: the one-pass mask of both
    // colours = play() = the frozen original code, and the moved play() keeps agreeing
    // with the original (stones and ko point) move by move
    let mut rng = Rng::new(7);
    let mut checked = 0;
    for (n, games, len) in (2..=9).map(|n| (n, 12, 3 * n * n)).chain([(13, 2, 3 * 169), (19, 1, 361)]) {
        for _ in 0..games {
            let mut b = Board::new(n);
            let mut r = reference::Board::new(n);
            for _ in 0..len {
                for color in [BLACK, WHITE] {
                    let m = mask(&b, color);
                    assert_eq!(m, legal_by_play(&b, color), "n={n}\n{}", rows(&b).join("\n"));
                    let rm: Vec<u8> = (0..n * n).map(|i| r.clone().play(color, Some((i / n, i % n))) as u8).collect();
                    assert_eq!(m, rm);
                    checked += 1;
                }
                let color = if rng.below(2) == 0 { BLACK } else { WHITE };
                let p = if rng.below(12) == 0 {
                    None
                } else {
                    let i = rng.below((n * n) as u64) as i32;
                    Some((i / n, i % n))
                };
                assert_eq!(b.play(color, p), r.play(color, p));
                assert_eq!((&b.g, b.ko), (&r.g, r.ko));
            }
        }
    }
    assert!(checked > 20_000, "{checked}");
}

#[test]
fn tromp_taylor_area_on_final_positions() {
    // the empty board is nobody's
    assert_eq!(Board::new(5).area(), (0, 0));
    // one stone owns every empty point
    assert_eq!(board(&[".....", ".....", "..X..", ".....", "....."]).area(), (25, 0));
    // walls: columns 0 / 4 are black's / white's area, column 2 reaches both (dame)
    assert_eq!(board(&[".X.O.", ".X.O.", ".X.O.", ".X.O.", ".X.O."]).area(), (10, 10));
    // every stone counts as alive: the white stone inside black's wall makes its region
    // (5 points) reach both colours; black: 6 stones + the 13 points right and below
    assert_eq!(board(&["..X..", ".OX..", "..X..", "XXX..", "....."]).area(), (19, 1));
    // a finished 5x5 game: black 7 stones + 3 eyes, white 7 stones + 8 points
    let fin = [".XO..", "XXO..", ".XOOO", "XXO..", ".XO.."];
    assert_eq!(board(&fin).area(), (10, 15));
    // ... scored at the double pass with komi 0.5: W+5.5
    let idx = |ch: char| fin.concat().chars().enumerate().filter(|&(_, c)| c == ch).map(|(i, _)| i).collect::<Vec<_>>();
    let cfg = GoConfig { board_size: 5, komi: Some(0.5), setup_black: idx('X'), setup_white: idx('O'), ..GoConfig::default() };
    Go::check_config(&cfg).unwrap();
    let (g, _) = play_game(&cfg, &[None, None]);
    assert_eq!((g.end(), g.returns(), g.outcome()), (Some(End::DoublePass), vec![-5.5, 5.5], vec![-1.0, 1.0]));
}

/// A game from moves `(row, col)` (None = pass) of whoever is to move.
fn play_game(cfg: &GoConfig, moves: &[Option<Point>]) -> (Go, Vec<Event>) {
    let mut g = Go::new_game(cfg);
    let n = cfg.board_size as i32;
    let mut ev = Vec::new();
    for &m in moves {
        let Turn::Act(s) = g.turn() else { panic!("game over before {m:?}:\n{}", g.state_string()) };
        let seat = s.iter().next().unwrap();
        let a = m.map_or((n * n) as u32, |(r, c)| (r * n + c) as u32);
        g.apply_action(seat, a).unwrap_or_else(|e| panic!("{e}\n{}", g.state_string()));
        ev.push(Event::Act(seat, a));
    }
    (g, ev)
}

#[test]
fn end_rules_and_scores() {
    let cfg = GoConfig { board_size: 5, komi: Some(0.5), ..GoConfig::default() };
    // two passes end the game, one does not
    let (g, _) = play_game(&cfg, &[Some((2, 2)), None]);
    assert!(matches!(g.turn(), Turn::Act(_)));
    let (g, ev) = play_game(&cfg, &[Some((2, 2)), None, None]);
    assert_eq!((g.turn(), g.end()), (Turn::Terminal, Some(End::DoublePass)));
    // one black stone: area 25 - 0 - 0.5
    assert_eq!((g.returns(), g.outcome()), (vec![24.5, -24.5], vec![1.0, -1.0]));
    assert!(replay::<Go>(&cfg, &ev).unwrap().terminal());
    // the move cap ends the game and counts it: 2 - 1 - 0.5 (the rest reaches both)
    let capped = GoConfig { max_moves: Some(3), ..cfg.clone() };
    let (g, _) = play_game(&capped, &[Some((0, 0)), Some((4, 4)), Some((0, 4))]);
    assert_eq!((g.end(), g.returns()), (Some(End::MoveCap), vec![0.5, -0.5]));
    assert_eq!(Go::spec(&capped).max_decisions, 3);
    assert_eq!(GoConfig::default().move_cap(), 722);
    // an integral komi can tie (+0.0 for both seats)
    let (g, _) = play_game(&GoConfig { komi: Some(0.0), ..cfg.clone() }, &[None, None]);
    assert_eq!((g.outcome(), g.returns()), (vec![0.0, 0.0], vec![0.0, 0.0]));
    assert!(g.outcome().iter().chain([&0.0]).all(|x| x.to_bits() == 0));
    // white wins the empty board by komi; an unknown komi counts as 0
    assert_eq!(play_game(&cfg, &[None, None]).0.outcome(), vec![-1.0, 1.0]);
    assert_eq!(play_game(&GoConfig { komi: None, ..cfg.clone() }, &[None, None]).0.outcome(), vec![0.0, 0.0]);
    // no result before the end
    let (g, _) = play_game(&cfg, &[Some((1, 1))]);
    assert_eq!((g.outcome(), g.returns()), (vec![0.0, 0.0], vec![0.0, 0.0]));
}

/// The moves of tests/fixtures/go_sim/double_ko_cycle.sgf (SGF coordinates, "" = pass).
fn double_ko() -> Vec<Option<Point>> {
    ["dd", "ed", "ce", "fe", "df", "ef", "cg", "de", "dh", "bg", "ci", "ah", "bh", "bi", "ee", "ch", "", "de", "bh", "gg",
     "ee", "ch", "gc"]
        .iter()
        .map(|s| {
            let b = s.as_bytes();
            (!s.is_empty()).then(|| ((b[1] - b'a') as i32, (b[0] - b'a') as i32))
        })
        .collect()
}

#[test]
fn positional_superko_ends_cycles() {
    let cfg = GoConfig { board_size: 9, ..GoConfig::default() };
    let moves = double_ko();
    // move 18 (B retakes ko 2) recreates the position after move 13: the game ends there
    let (g, _) = play_game(&cfg, &moves[..18]);
    assert_eq!(g.end(), None, "no repetition before");
    let (g, _) = play_game(&cfg, &moves[..19]);
    assert_eq!((g.end(), g.turn()), (Some(End::Superko), Turn::Terminal));
    assert_eq!(g.outcome()[0].abs(), 1.0);
    // every move of the cycle is legal under simple ko; without superko the game goes on
    let (g, _) = play_game(&GoConfig { superko: false, ..cfg }, &moves);
    assert_eq!((g.end(), g.moves()), (None, moves.len()));
    // passes never repeat a position
    let (g, _) = play_game(&GoConfig { board_size: 5, ..GoConfig::default() }, &[Some((0, 0)), None, Some((4, 4)), None]);
    assert_eq!(g.end(), None);
}

#[test]
fn frames_targets_and_decoding() {
    let cfg = GoConfig { board_size: 5, komi: Some(7.5), ..GoConfig::default() };
    let spec = Go::spec(&cfg);
    spec.validate().unwrap();
    assert_eq!(spec.layout(), Layout::Interleaved);
    assert_eq!(spec.meta, json!({"global_features": 6, "board_size": 5}));
    assert_eq!(spec.action_keys, ["place_idx", "passed"]);
    // B (2,2), W (1,2), B passes: white to move
    let (g, _) = play_game(&cfg, &[Some((2, 2)), Some((1, 2)), None]);
    let mut f = Frame::new(&spec);
    encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
    let planes = f.obs[0].as_f32().unwrap();
    assert_eq!((planes[7], planes[12], planes[25 + 12], planes[50 + 12], planes[50]), (1.0, 0.0, 1.0, 0.0, 1.0));
    assert_eq!(planes[75..].iter().sum::<f32>(), 0.0, "no ko");
    assert_eq!(f.obs[1].as_f32().unwrap(), &[0.0, (7.5f64 / 20.0) as f32, 1.0, 1.0, 0.0, 1.0][..]);
    assert_eq!((f.legal_ids.len(), f.legal_ids.last()), (23 + 1, Some(&25)));
    let place = f.legal[0].as_bool().unwrap();
    assert_eq!((place[7], place[12], place[0], place.iter().map(|&x| x as usize).sum::<usize>()), (0, 0, 1, 23));
    assert_eq!(f.legal[1].as_bool().unwrap(), &[1]);
    // targets of a point and of the pass; the keys decode back
    let mut t = new_target(&spec);
    encode_target(&g, 1, &spec, &f.legal_ids, 3, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    assert_eq!((t[0].as_i64().unwrap(), t[1].as_bool().unwrap()), (&[3i64][..], &[0u8][..]));
    encode_target(&g, 1, &spec, &f.legal_ids, 25, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    assert_eq!((t[0].as_i64().unwrap(), t[1].as_bool().unwrap()), (&[0i64][..], &[1u8][..]));
    assert_eq!(crate::frame::action_keys_of(&spec, &t), vec![0, 1]);
    assert_eq!(decode_action(&g, 1, &spec, &f.legal_ids, &[0, 1]).unwrap(), 25);
    assert_eq!(decode_action(&g, 1, &spec, &f.legal_ids, &[7, 1]).unwrap(), 25, "passed wins over place_idx");
    assert_eq!(decode_action(&g, 1, &spec, &f.legal_ids, &[3, 0]).unwrap(), 3);
    assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[7, 0]).is_err(), "occupied");
    assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[25, 0]).is_err(), "off the board");
    assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[3, 2]).is_err(), "passed is 0 / 1");
    // black's komi feature at komi 0 is -0.0, white's +0.0 (the converter's bits)
    let g0 = Go::new_game(&GoConfig { komi: Some(0.0), ..cfg.clone() });
    let (mut b, mut gl) = (vec![0f32; 100], vec![0f32; 6]);
    g0.encode_obs_as(BLACK, &mut b, &mut gl);
    assert_eq!(gl[1].to_bits(), (-0.0f32).to_bits());
    g0.encode_obs_as(WHITE, &mut b, &mut gl);
    assert_eq!(gl[1].to_bits(), 0.0f32.to_bits());
    // the ko plane marks the banned point on the next mover's frame: W (1,1) is taken
    // by B (1,2), which is left with that one liberty
    let (g, _) = play_game(&cfg, &[Some((0, 1)), Some((0, 2)), Some((1, 0)), Some((1, 3)), Some((2, 1)), Some((2, 2)),
                                   Some((4, 4)), Some((1, 1)), Some((1, 2))]);
    assert_eq!(g.board().ko, Some((1, 1)));
    encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
    let planes = f.obs[0].as_f32().unwrap();
    assert_eq!(planes[75..].iter().enumerate().filter(|&(_, &v)| v == 1.0).map(|(i, _)| i).collect::<Vec<_>>(), vec![6]);
    assert!(!f.legal_ids.contains(&6) && f.legal[0].as_bool().unwrap()[6] == 0);
    assert_eq!((g.action_string(1, 6), g.action_string(1, 25)), ("B4".to_string(), "pass".to_string()));
}

#[test]
fn setup_handicap_and_first_player() {
    // fixed handicap: the stones are placed and white moves first
    let cfg = GoConfig { board_size: 9, komi: Some(0.5), handicap: 2, setup_black: vec![20, 60], first_player: Colour::White,
                         ..GoConfig::default() };
    Go::check_config(&cfg).unwrap();
    let g = Go::new_game(&cfg);
    assert_eq!((g.turn(), g.board().g[20], g.board().g[60]), (Turn::Act(SeatSet::single(1)), BLACK, BLACK));
    let spec = Go::spec(&cfg);
    let mut f = Frame::new(&spec);
    encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
    assert_eq!(f.obs[1].as_f32().unwrap(), &[0.0, (0.5f64 / 20.0) as f32, 1.0, 1.0, 2.0 / 9.0, 0.0][..]);
    assert_eq!(f.obs[0].as_f32().unwrap()[81 + 20], 1.0, "black's stones on white's opponent plane");
    // free placement: black's first three moves, then white
    let free = GoConfig { board_size: 9, handicap: 3, free_handicap: 3, ..GoConfig::default() };
    assert_eq!(play_game(&free, &[Some((2, 2)), Some((6, 6))]).0.turn(), Turn::Act(SeatSet::single(0)));
    assert_eq!(play_game(&free, &[Some((2, 2)), Some((6, 6)), None]).0.turn(), Turn::Act(SeatSet::single(1)));
    // a setup stone can be captured
    let cap = GoConfig { board_size: 5, setup_black: vec![1], setup_white: vec![0, 2], first_player: Colour::White,
                         ..GoConfig::default() };
    assert_eq!(play_game(&cap, &[Some((1, 1))]).0.board().g[1], EMPTY);
    // configs the rules refuse
    for bad in [
        GoConfig { board_size: 1, ..GoConfig::default() },
        GoConfig { board_size: 26, ..GoConfig::default() },
        GoConfig { komi: Some(f64::NAN), ..GoConfig::default() },
        GoConfig { setup_black: vec![361], ..GoConfig::default() },
        GoConfig { setup_black: vec![3], setup_white: vec![3], ..GoConfig::default() },
        GoConfig { setup_black: vec![3, 3], ..GoConfig::default() },
        GoConfig { free_handicap: 2, first_player: Colour::White, ..GoConfig::default() },
        GoConfig { max_moves: Some(0), ..GoConfig::default() },
    ] {
        assert!(Go::check_config(&bad).is_err(), "{bad:?}");
    }
    assert!(Go::setup(&GoConfig { setup_black: vec![3, 3], ..GoConfig::default() }).is_none());
    // JSON configs (the Python env kwargs): defaults, null komi, colour names
    let c: GoConfig = serde_json::from_str(r#"{"board_size": 9, "komi": null, "first_player": "white"}"#).unwrap();
    assert_eq!((c.board_size, c.komi, c.first_player, c.area_scoring, c.superko), (9, None, Colour::White, true, true));
    assert!(serde_json::from_str::<GoConfig>(r#"{"size": 9}"#).is_err());
}

#[test]
fn conformance_battery() {
    let cases = [
        (GoConfig { board_size: 2, ..GoConfig::default() }, 40),
        (GoConfig { board_size: 3, komi: Some(0.0), ..GoConfig::default() }, 40),
        (GoConfig { board_size: 5, ..GoConfig::default() }, 30),
        (GoConfig { board_size: 5, superko: false, max_moves: Some(60), ..GoConfig::default() }, 20),
        (GoConfig { board_size: 7, komi: None, area_scoring: false, handicap: 2, setup_black: vec![16, 32],
                    first_player: Colour::White, ..GoConfig::default() }, 10),
        (GoConfig { board_size: 9, handicap: 2, free_handicap: 2, ..GoConfig::default() }, 6),
        (GoConfig { board_size: 13, max_moves: Some(200), ..GoConfig::default() }, 2),
        (GoConfig { max_moves: Some(300), ..GoConfig::default() }, 1),
    ];
    for (i, (cfg, n)) in cases.iter().enumerate() {
        Go::check_config(cfg).unwrap();
        let rep = check_game::<Go>(cfg, *n, i as u64 + 1).unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
        assert_eq!(rep.terminal, *n, "{cfg:?}");
    }
}
