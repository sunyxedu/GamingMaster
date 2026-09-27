//! Breakthrough — the framework's CUSTOM-SCHEMA reference game: a perfect-information
//! board game played through the board adapter's from-to decoder
//! (`model/games/board.py`: BoardEncoder + FromToDecoder), the pattern for chess-like
//! games (chess variants, xiangqi, ...). Exactly OpenSpiel's
//! `breakthrough(rows=R,columns=C)` (ids, turn order, returns; pinned by
//! tests/test_gmrules_reference.py).
//!
//! Rules. Black (seat 0) starts on rows 0 (and 1 when R >= 6), White (seat 1) on the
//! last one (two) rows; Black moves first. A piece moves one row forward (Black: +1,
//! White: −1), straight or diagonally, to an empty cell; it captures diagonally
//! forward only. Reaching the far row, or capturing the last enemy piece, wins
//! (returns +1 / −1). Action id = ((r·C + c)·6 + dir)·2 + capture (OpenSpiel's mixed
//! radix: dir 0..2 = Black's (−1, 0, +1 column), 3..5 = White's).
//!
//! Frames (interleaved: one stream per game, every frame from the mover's view —
//! rows mirrored for White so the mover always moves "up"; POV square = row·C + col):
//!
//! ```text
//! obs     board_seq    [3, R, C] bits  mover's pieces, opponent's pieces, ones
//!         globals_seq  [2] f32         mover is Black, moves / max_decisions
//! legal   from         [R·C] bool      squares with a legal move
//!         to_cond      [R·C, R·C] bool destinations per from-square (exact table)
//! target  from_idx     [] i64          action key: the move's POV from-square
//!         to_idx       [] i64          action key: the move's POV to-square
//!         legal_from   [R·C] bits      IL legality (= legal.from)
//!         legal_to_of  [R·C] bits      destinations of the taken move's from-square
//! ```
//!
//! Shard meta (the model's META_ARCH_KEYS): board_h = R, board_w = C,
//! piece_channels = 3, global_features = 2.

use crate::frame::FrameMut;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use crate::schema::{TensorSpec, TensorsMut};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DR: [i32; 6] = [1, 1, 1, -1, -1, -1];
const DC: [i32; 6] = [-1, 0, 1, -1, 0, 1];

/// Obs / legal / target tensor indices (see [`Breakthrough::spec`]).
const OBS_BOARD: usize = 0;
const OBS_GLOBALS: usize = 1;
const LEGAL_FROM: usize = 0;
const LEGAL_TO_COND: usize = 1;
const TGT_FROM: usize = 0;
const TGT_TO: usize = 1;
const TGT_LEGAL_FROM: usize = 2;
const TGT_LEGAL_TO_OF: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BreakthroughConfig {
    /// 3..=16
    pub rows: usize,
    /// 2..=16
    pub columns: usize,
}

impl Default for BreakthroughConfig {
    fn default() -> Self {
        BreakthroughConfig { rows: 8, columns: 8 }
    }
}

#[derive(Clone, Debug)]
pub struct Breakthrough {
    rows: usize,
    cols: usize,
    /// 0 empty, 1 Black (seat 0), 2 White (seat 1)
    board: Vec<u8>,
    pieces: [u32; 2],
    cur: Seat,
    winner: Option<Seat>,
    moves: usize,
    max_moves: usize,
}

/// A decoded action: from (r, c), direction, capture flag.
fn unrank(action: u32, cols: usize) -> (usize, usize, usize, bool) {
    let a = action as usize;
    let capture = a % 2 == 1;
    let dir = (a / 2) % 6;
    let cell = a / 12;
    (cell / cols, cell % cols, dir, capture)
}

fn rank(r: usize, c: usize, dir: usize, capture: bool, cols: usize) -> u32 {
    (((r * cols + c) * 6 + dir) * 2 + capture as usize) as u32
}

impl Breakthrough {
    fn max_decisions(cfg: &BreakthroughConfig) -> usize {
        // every move takes one piece one row forward
        let per_side = cfg.columns * if cfg.rows >= 6 { 2 } else { 1 };
        2 * per_side * (cfg.rows - 1)
    }

    fn at(&self, r: usize, c: usize) -> u8 {
        self.board[r * self.cols + c]
    }

    /// The destination of a move.
    fn dest(&self, r: usize, c: usize, dir: usize) -> Option<(usize, usize)> {
        let (r2, c2) = (r as i32 + DR[dir], c as i32 + DC[dir]);
        (r2 >= 0 && c2 >= 0 && (r2 as usize) < self.rows && (c2 as usize) < self.cols).then_some((r2 as usize, c2 as usize))
    }

    /// POV row of board row `r` for `seat` (White sees the board mirrored).
    fn pov_row(&self, seat: Seat, r: usize) -> usize {
        if seat == 0 { r } else { self.rows - 1 - r }
    }

    fn pov_square(&self, seat: Seat, r: usize, c: usize) -> usize {
        self.pov_row(seat, r) * self.cols + c
    }

    /// (POV from-square, POV to-square) of a legal action of `seat`.
    fn squares(&self, seat: Seat, action: u32) -> (usize, usize) {
        let (r, c, dir, _) = unrank(action, self.cols);
        let (r2, c2) = self.dest(r, c, dir).expect("legal action stays on the board");
        (self.pov_square(seat, r, c), self.pov_square(seat, r2, c2))
    }
}

impl Game for Breakthrough {
    type Config = BreakthroughConfig;
    const NAME: &'static str = "breakthrough";
    const VERSION: u32 = 1;

    fn check_config(cfg: &BreakthroughConfig) -> Result<(), RulesError> {
        if !(3..=16).contains(&cfg.rows) || !(2..=16).contains(&cfg.columns) {
            rules_bail!("rows must be 3..=16 and columns 2..=16, got {} x {}", cfg.rows, cfg.columns);
        }
        Ok(())
    }

    fn spec(cfg: &BreakthroughConfig) -> FrameSpec {
        let (h, w) = (cfg.rows, cfg.columns);
        let n = h * w;
        FrameSpec {
            n_seats: 2,
            hidden_info: false,
            simultaneous: false,
            max_decisions: Self::max_decisions(cfg),
            schema: "fromto.v1".into(),
            obs: vec![TensorSpec::bits("board_seq", &[3, h, w]), TensorSpec::f32("globals_seq", &[2])],
            legal: vec![TensorSpec::bool("from", &[n]), TensorSpec::bool("to_cond", &[n, n])],
            target: vec![
                TensorSpec::i64("from_idx", &[]),
                TensorSpec::i64("to_idx", &[]),
                TensorSpec::bits("legal_from", &[n]),
                TensorSpec::bits("legal_to_of", &[n]),
            ],
            action_keys: vec!["from_idx".into(), "to_idx".into()],
            meta: json!({"board_h": h, "board_w": w, "piece_channels": 3, "global_features": 2}),
            candidate: None,
        }
    }

    fn new_game(cfg: &BreakthroughConfig) -> Self {
        let (rows, cols) = (cfg.rows, cfg.columns);
        let mut board = vec![0u8; rows * cols];
        let mut pieces = [0u32; 2];
        for r in 0..rows {
            for c in 0..cols {
                if r == 0 || (rows >= 6 && r == 1) {
                    board[r * cols + c] = 1;
                    pieces[0] += 1;
                } else if r == rows - 1 || (rows >= 6 && r == rows - 2) {
                    board[r * cols + c] = 2;
                    pieces[1] += 1;
                }
            }
        }
        Breakthrough { rows, cols, board, pieces, cur: 0, winner: None, moves: 0, max_moves: Self::max_decisions(cfg) }
    }

    fn turn(&self) -> Turn {
        if self.winner.is_some() || self.pieces[0] == 0 || self.pieces[1] == 0 {
            Turn::Terminal
        } else {
            Turn::Act(SeatSet::single(self.cur))
        }
    }

    fn chance_outcomes(&self, _out: &mut Vec<(u32, f64)>) {}

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        rules_bail!("breakthrough has no chance events (got {outcome})")
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            return;
        }
        let me = seat as u8 + 1;
        let opp = 3 - me;
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.at(r, c) != me {
                    continue;
                }
                for o in 0..3 {
                    let dir = seat * 3 + o;
                    if let Some((r2, c2)) = self.dest(r, c, dir) {
                        let t = self.at(r2, c2);
                        if t == 0 {
                            out.push(rank(r, c, dir, false, self.cols));
                        } else if o != 1 && t == opp {
                            out.push(rank(r, c, dir, true, self.cols));
                        }
                    }
                }
            }
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            rules_bail!("breakthrough: seat {seat} is not to act");
        }
        let (r, c, dir, capture) = unrank(action, self.cols);
        let me = seat as u8 + 1;
        let ok = r < self.rows && dir / 3 == seat && self.at(r, c) == me;
        let Some((r2, c2)) = self.dest(r, c, dir).filter(|_| ok) else {
            rules_bail!("breakthrough: action {action} is illegal for seat {seat}");
        };
        let t = self.at(r2, c2);
        let legal = if capture { dir % 3 != 1 && t == 3 - me } else { t == 0 };
        if !legal {
            rules_bail!("breakthrough: action {action} is illegal for seat {seat}");
        }
        if capture {
            self.pieces[1 - seat] -= 1;
        }
        self.board[r2 * self.cols + c2] = me;
        self.board[r * self.cols + c] = 0;
        if (seat == 0 && r2 == self.rows - 1) || (seat == 1 && r2 == 0) {
            self.winner = Some(seat);
        }
        self.cur = 1 - seat;
        self.moves += 1;
        Ok(())
    }

    fn returns(&self) -> Vec<f64> {
        if self.winner == Some(0) || self.pieces[1] == 0 {
            vec![1.0, -1.0]
        } else if self.winner == Some(1) || self.pieces[0] == 0 {
            vec![-1.0, 1.0]
        } else {
            vec![0.0, 0.0]
        }
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns().iter().map(|&r| r as f32).collect()
    }

    fn encode_custom(&self, seat: Seat, f: &mut FrameMut<'_>) {
        let (h, w) = (self.rows, self.cols);
        let n = h * w;
        let me = seat as u8 + 1;
        {
            let planes = f.obs.f32(OBS_BOARD);
            for r in 0..h {
                let pr = self.pov_row(seat, r);
                for c in 0..w {
                    let sq = pr * w + c;
                    match self.at(r, c) {
                        0 => {}
                        p if p == me => planes[sq] = 1.0,
                        _ => planes[n + sq] = 1.0,
                    }
                    planes[2 * n + sq] = 1.0;
                }
            }
        }
        let g = f.obs.f32(OBS_GLOBALS);
        g[0] = (seat == 0) as u8 as f32;
        g[1] = self.moves as f32 / self.max_moves as f32;
        for &a in f.legal_ids.iter() {
            let (from, to) = self.squares(seat, a);
            f.legal.bool(LEGAL_FROM)[from] = 1;
            f.legal.bool(LEGAL_TO_COND)[from * n + to] = 1;
        }
    }

    fn target_custom(&self, seat: Seat, legal: &[u32], action: u32, t: &mut TensorsMut<'_>) {
        let (from, to) = self.squares(seat, action);
        t.i64(TGT_FROM)[0] = from as i64;
        t.i64(TGT_TO)[0] = to as i64;
        for &a in legal {
            let (f2, t2) = self.squares(seat, a);
            t.f32(TGT_LEGAL_FROM)[f2] = 1.0;
            if f2 == from {
                t.f32(TGT_LEGAL_TO_OF)[t2] = 1.0;
            }
        }
    }

    fn decode_custom(&self, seat: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
        let n = (self.rows * self.cols) as i64;
        let (from, to) = (keys[0], keys[1]);
        if !(0..n).contains(&from) || !(0..n).contains(&to) {
            return None;
        }
        legal.iter().copied().find(|&a| self.squares(seat, a) == (from as usize, to as usize))
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        let (r, c, dir, capture) = unrank(action, self.cols);
        let (r2, c2) = (r as i32 + DR[dir], c as i32 + DC[dir]);
        let col = |c: i32| (b'a' + c as u8) as char;
        let row = |r: i32| self.rows as i32 - r;
        format!("{}{}{}{}{}", col(c as i32), row(r as i32), col(c2), row(r2), if capture { "*" } else { "" })
    }

    fn state_string(&self) -> String {
        let mut s = String::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                s.push(match self.at(r, c) {
                    1 => 'b',
                    2 => 'w',
                    _ => '.',
                });
            }
            s.push('\n');
        }
        format!("{s}to move: {} moves: {} winner: {:?}", self.cur, self.moves, self.winner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{decode_action, encode_frame, encode_target, new_target, Frame};
    use crate::game::Layout;
    use crate::invariance::check_game;
    use crate::trajectory::{replay, Event};

    #[test]
    fn rules_match_openspiel_conventions() {
        let cfg = BreakthroughConfig { rows: 6, columns: 5 };
        let g = Breakthrough::new_game(&cfg);
        assert_eq!(g.pieces, [10, 10]);
        let mut legal = Vec::new();
        g.legal_actions(0, &mut legal);
        // row-1 pieces only (row 0 is blocked by its own pieces): 5 pieces x 3 dirs minus the edges
        assert_eq!(legal.len(), 13);
        assert!(legal.windows(2).all(|w| w[0] < w[1]));
        // a1 → straight: ((1*5+0)*6+1)*2+0
        assert!(legal.contains(&rank(1, 0, 1, false, 5)));
        assert_eq!(Breakthrough::spec(&cfg).layout(), Layout::Interleaved);
        // a race to the far row, won by a diagonal capture on it
        let mut g = Breakthrough::new_game(&BreakthroughConfig { rows: 4, columns: 3 });
        let mut ev = Vec::new();
        for (seat, a) in [(0, rank(0, 0, 2, false, 3)), (1, rank(3, 2, 4, false, 3)), (0, rank(1, 1, 1, false, 3)),
                          (1, rank(2, 2, 3, false, 3)), (0, rank(2, 1, 0, true, 3))] {
            let mut l = Vec::new();
            g.legal_actions(seat, &mut l);
            assert!(l.contains(&a), "{seat} {a} {}", g.state_string());
            g.apply_action(seat, a).unwrap();
            ev.push(Event::Act(seat, a));
        }
        assert_eq!(g.turn(), Turn::Terminal);
        assert_eq!(g.returns(), vec![1.0, -1.0]);
        assert!(replay::<Breakthrough>(&BreakthroughConfig { rows: 4, columns: 3 }, &ev).unwrap().terminal());
    }

    #[test]
    fn custom_schema_frames_targets_and_decoding() {
        let cfg = BreakthroughConfig { rows: 6, columns: 5 };
        let spec = Breakthrough::spec(&cfg);
        spec.validate().unwrap();
        let mut g = Breakthrough::new_game(&cfg);
        g.apply_action(0, rank(1, 2, 1, false, 5)).unwrap(); // Black c2 → c3
        // White to move: its frame is mirrored (its pieces on POV rows 0-1)
        let mut f = Frame::new(&spec);
        encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
        let planes = f.obs[0].as_f32().unwrap();
        let n = 30;
        assert!((0..10).all(|sq| planes[sq] == 1.0), "own pieces at the bottom of the POV board");
        assert!((0..30).all(|sq| planes[2 * n + sq] == 1.0), "the ones plane");
        // Black's moved piece (board row 2, col 2) = White POV row 3
        assert_eq!(planes[n + 3 * 5 + 2], 1.0);
        let from = f.legal[0].as_bool().unwrap();
        assert_eq!(from.iter().filter(|&&x| x == 1).count(), 5);
        let a = f.legal_ids[0];
        let mut t = new_target(&spec);
        encode_target(&g, 1, &spec, &f.legal_ids, a, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
        let keys = [t[0].as_i64().unwrap()[0], t[1].as_i64().unwrap()[0]];
        assert_eq!(decode_action(&g, 1, &spec, &f.legal_ids, &keys).unwrap(), a);
        assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[0, 29]).is_err());
        assert_eq!(t[3].as_f32().unwrap().iter().sum::<f32>(), f.legal[1].as_bool().unwrap()[keys[0] as usize * n..(keys[0] as usize + 1) * n]
            .iter().map(|&x| x as f32).sum::<f32>());
    }

    #[test]
    fn conformance_battery() {
        for (rows, columns) in [(3, 2), (5, 4), (6, 5), (8, 8)] {
            let cfg = BreakthroughConfig { rows, columns };
            let rep = check_game::<Breakthrough>(&cfg, 30, rows as u64).unwrap();
            assert_eq!(rep.terminal, 30);
        }
    }
}
