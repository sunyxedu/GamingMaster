//! Shogi teacher labels: `gmdata label shogi` scores every frame of converter_version 2
//! shogi shards (`gmdata shogi convert`, see [`crate::shogi`]) with a USI engine —
//! YaneuraOu, NNUE edition, with the Háo eval — and maps the scores onto the
//! `ShogiDecoder` heads (model/games/shogi.py).
//!
//! ```text
//! scripts/engines/fetch_engines.sh yaneuraou      # → ~/gm-engines/bin/yaneuraou (+ Háo nn.bin)
//! gmdata label shogi --input data/shogi/train --out data/shogi_yo/train \
//!     --engine ~/gm-engines/bin/yaneuraou --workers 32 --tau 0.025   # one worker per core
//! #  defaults: --limit nodes=1000000 --max-moves 8 --policy softmax (--tau: see below)
//! gmdata label shogi --input data/shogi/val --out data/shogi_yo/val --engine … (same flags!)
//! python -m training.train_il_game --game shogi --data_dir data/shogi_yo/train \
//!     --val_dir data/shogi_yo/val --soft_target_weight 1.0 --engine_value_weight 0.5 ...
//! ```
//!
//! Engine conversation (one single-threaded YaneuraOu per worker, see [`DEFAULT_OPTIONS`]):
//!
//! ```text
//! usi → setoption … (defaults, then --engine-option overrides, MultiPV = --max-moves) → isready
//! per position:  usinewgame; isready             (isready clears the hash: every answer is
//!                                                 independent of what the worker searched before)
//!                position startpos moves 7g7f 3c3d …     (the whole game so far — `position
//!                                                 sfen <start> moves …` for other starts)
//!                go nodes N  →  info … multipv k score cp X | mate ±n … pv <move> …  → bestmove
//! ```
//!
//! The full move history is sent (not the SFEN of the frame) so the engine knows the game's
//! repetitions: a 4th occurrence is sennichite (a draw) and a perpetual check loses for the
//! checking side, which YaneuraOu scores in search (it looks up to 16 plies back, chained
//! through earlier repetition counts). The position cache key is therefore the exact
//! position line (start + moves, as a 128-bit hash): a hit is an identical game prefix —
//! mostly shared opening lines —, never a transposition.
//!
//! Default USI options ([`DEFAULT_OPTIONS`]; any can be overridden by `--engine-option
//! NAME=VALUE`, and all effective values are recorded in `meta.teacher.game.usi_options`;
//! an option the engine does not announce in its `usi` answer is an error — YaneuraOu itself
//! would only print `No such option` and search with its default):
//! `Threads=1` and `USI_Hash=16` (one process per core; a 1M-node MultiPV-8 search fills
//! about a third of a 16 MB table, hashfull ≈ 370/1000), no opening book
//! (`USI_OwnBook=false`, `BookFile=no_book`), `FV_SCALE=20` (the value the Háo authors
//! specify; other evals need their own), draws scored 0
//! (`DrawValueBlack/White=0`, YaneuraOu's default −2 is a playing contempt),
//! `GenerateAllLegalMoves=true` (by default YaneuraOu never considers pawn / bishop / rook
//! non-promotions, which are legal and in the decoder's action space), the floodgate
//! entering-king rule `EnteringKingRule=CSARule27`, and no move-count draw
//! (`MaxMovesToDraw=0`).
//!
//! Search budget and policy, and why (measured on an Apple M4 Pro core with YaneuraOu 9.40 +
//! Háo, MultiPV 8, floodgate positions; the `real_yaneuraou_*` tests in `shogi/tests.rs`):
//!
//! * `--limit nodes=1000000` (the only accepted limit — node counts keep single-threaded
//!   searches deterministic): 1.26 positions/s per process (1.26 M nodes/s, average depth
//!   12.9, plus ~12 ms of isready per position). Against a 4M-node reference (80 positions,
//!   policies at τ = 0.05), 1M nodes pick the same best move on 59% of the positions (250k:
//!   49%, 100k: 51%). Shogi positions often have several near-equal moves: by the
//!   reference's own scores the 1M choice loses only 0.003 Q (≈ 8 cp) on average. The two
//!   teacher policies differ by a total-variation distance of 0.17 (250k: 0.25, 100k: 0.28).
//!   On 12 real floodgate games (1455 labelled frames) the 1M-node teacher's best move is the
//!   move played 54% of the time (53% before stale lines were capped), the played move is
//!   among its 8 scored moves 96% of the time, and the sign of its value matches the game
//!   result on 87% of the frames. (The budget comparison above predates the cap.) 1M labels
//!   the whole floodgate split in a few node-hours (below); `--limit nodes=4000000` buys a
//!   stronger, 4× slower teacher.
//! * `--max-moves 8` = MultiPV 8: at the recommended τ a move 100 cp behind the best (near
//!   equality) gets 0.19× the best move's weight, so moves beyond the top 8 carry negligible
//!   mass; each extra PV line costs search.
//! * `--tau`: the pipeline default 0.05 (shared by all games) makes a flat shogi teacher
//!   (mean top-move probability 0.37, entropy 1.6 nats). The likelihood of the moves actually
//!   played on floodgate peaks for τ between 0.02 and 0.03 (mean log-probability −1.54, vs
//!   −1.61 at 0.05), where the teacher is also calibrated (mean top probability 0.48 ≈ its 53%
//!   top-1 agreement with the played move): use `--tau 0.025` for shogi. Re-measured with
//!   capped stale lines on the 1455 real frames (log-probability floored at 10⁻⁶ for played
//!   moves outside the 8): −1.80 at τ = 0.025, −1.81 at 0.02 / 0.03, −1.87 at 0.05 (uncapped:
//!   −1.83 / −1.89) — the same optimum, a slightly better teacher. teacher_q /
//!   teacher_raw keep the raw scores, so the policy can be re-derived with another τ.
//! * Q = logistic(cp / 600) — the usual shogi-engine win-rate model (the "Ponanza constant"
//!   600 of YaneuraOu's learner); mate for the mover → 1, mated → 0 ([`CP_SCALE`]). Note that
//!   YaneuraOu 9.x prints cp = 100·v/90 of its internal value v (PawnValue 90), so this is
//!   logistic(v / 540) in internal units; the scale is recorded in
//!   `meta.teacher.game.cp_scale` and `teacher_raw` keeps the cp, so Q (and the policy) can be
//!   re-derived with another scale without relabelling. `tgt.engine_value` = 2·Q − 1 of the
//!   engine's first MultiPV line (its root score).
//! * Stale lines ([`analysis`]): the node limit stops the search inside an iteration, and
//!   YaneuraOu (like Stockfish) then prints the lines it has not re-searched in that
//!   iteration with their score from the previous, shallower one (depth − 1 in the final
//!   block). Such a score can be far too high: each MultiPV pass of the unfinished iteration
//!   picks the best of the remaining moves, so it had already ranked them below its own,
//!   deeper lines (at their depth, not at the old one). On 12 real floodgate
//!   games at 1M nodes 9% of the labelled frames had a stale line above the first line
//!   (0 with the cap, which lowers some stale line on 39% of the frames),
//!   e.g. `multipv 1 depth 10 cp −1270 upperbound G*7h` with `multipv 2 depth 9 cp +346
//!   3h3d+` (an 8M-node search scores 3h3d+ ≤ −1235), which would give the refuted move
//!   80% of the policy mass at τ = 0.05 (94% at 0.025) and the engine's choice G*7h none.
//!   So the Q and raw score of every line shallower than the
//!   deepest one are capped at the lowest Q / raw score among the deepest lines (the chess
//!   labeller does the same): the teacher keeps the engine's own ranking.
//!
//! Expected throughput on the labelling nodes (AMD EPYC 9355, Zen 5 with AVX-512 VNNI;
//! fetch_engines.sh builds TARGET_CPU=AVX512VNNI with g++) — an estimate, not measured
//! there: YaneuraOu's halfkp_256x2 NNUE search runs at a similar speed per core on Zen 5
//! and Apple M4 cores, so ~1.0–1.3 positions/s per single-threaded process at 1M nodes,
//! ~32–40 positions/s per 32-core socket with one worker per physical core (SMT siblings
//! add little to NNUE search). The floodgate train split (52 shards, ~1.4 M frames, ~1.3 M
//! searches after forced frames and opening cache hits) is then ~10 socket-hours; spread it
//! over jobs with `--part i/n`. To measure on a node: label one shard with
//! `--max-samples 64 --workers 32` and read `positions_per_s` in the JSON summary.
//!
//! Replay and verification: a sample's start position is decoded from its first frame
//! (obs.board_seq / hand_seq / globals_seq — hirate for floodgate, but handicap starts
//! work too), the moves from tgt.dropped / from_idx / to_idx / promoted / piece_idx /
//! drop_to_idx (POV squares → absolute), each checked against the converter's move
//! generator. Every frame is then re-encoded with the converter's own functions
//! ([`crate::shogi::encode_obs`], [`crate::shogi::frame_masks`], repetition counts) and
//! compared bit-exactly with obs.board_seq, obs.hand_seq, obs.globals_seq, the six
//! tgt.legal_* masks, the target indices and tgt.value.
//!
//! Flat action ids (`tgt.teacher_idx`), squares in the mover's POV (the tgt.from_idx
//! convention: square = row·9 + col with 0 = 9a, rotated 180° for white):
//!
//! ```text
//! board move  (from·81 + to)·2 + promote        ∈ [0, 13122)
//! drop        13122 + slot·81 + to               ∈ [13122, 13689), slot 0..6 = P L N S G B R
//! ```
//!
//! Heads (`ShogiDecoder.il_head_terms`), from the policy π over full actions:
//!
//! ```text
//! tgt.mode_probs      [T,2]   dense   (P(board move), P(drop))
//! tgt.from_probs_*    [T,k]   sparse  p(from | board move)
//! tgt.to_probs_*      [T,k]   sparse  p(to | board move, from = tgt.from_idx)          move frames
//! tgt.promo_probs     [T,2]   dense   p(promote | from = tgt.from_idx, to = tgt.to_idx)  move frames
//! tgt.piece_probs     [T,14]  dense   p(hand slot | drop) (slots 7..13 never)
//! tgt.drop_to_probs_* [T,k]   sparse  p(square | drop, slot = tgt.piece_idx)            drop frames
//! ```
//!
//! k = min(--max-moves, 81) (a conditional has at most one class per scored move). Rows
//! conditioned on a ground-truth index are written only on frames of that index's mode
//! (the converter stores dummy zeros for the other mode, and the decoder gives those heads
//! no weight there); a conditioning event without teacher mass is an all-zero row (no
//! teacher). Raw columns: `teacher_q` (Q per scored move) and `teacher_raw` in the
//! engine's centipawns, a mate in n plies as ±(100000 − min(n, 9999)) ([`MATE_RAW`]; outside
//! the cp range: YaneuraOu's evaluations reach ±35281 cp = its VALUE_MAX_EVAL, e.g. for
//! inferior-position detections, so |raw| > 90000 ⇔ mate).
//! Engine moves outside the converter's move generator (e.g. a pawn-drop the reference
//! rules call mate) are dropped from the analysis. Forced frames (one legal move) get no
//! teacher (skip reason `forced`).
//!
//! Tests without an engine binary use [`fake_engine`] — in-process, or as a child process
//! via the hidden `gmdata label shogi fake-usi` — whose moves come from the converter's
//! move generator with hashed scores.

use super::engine::{Engine, EngineCommand, LineHandler};
use super::fake::{FakeMove, UciFake};
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove};
use super::transform::{conditional, marginal, q_from_cp, q_from_mate};
use super::uci::{Flavor, Info, Score, UciSession};
use super::LabelArgs;
use crate::shogi::{self as rules, Board, Mv, BLACK, NONE, OBS_BOARD, PAWN, ROOK, WHITE};
use anyhow::{bail, ensure, Context, Result};
use gmshard::ShardReader;
use serde_json::{json, Value};
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Labeller version (bump when a label's meaning changes; recorded in the teacher meta).
/// 2: stale MultiPV lines capped ([`analysis`]); mates in teacher_raw at ±(100000 − n).
pub const LABELLER_VERSION: u32 = 2;
/// Q = 1 / (1 + exp(−cp / CP_SCALE)).
pub const CP_SCALE: f32 = 600.0;
/// `--limit nodes` default.
pub const DEFAULT_NODES: u64 = 1_000_000;
/// `--max-moves` default (MultiPV).
pub const DEFAULT_MAX_MOVES: usize = 8;
/// Board-move action ids: (from·81 + to)·2 + promote.
pub const N_BOARD_ACTIONS: i64 = 81 * 81 * 2;
/// All action ids (board moves, then drops).
pub const N_ACTIONS: i64 = N_BOARD_ACTIONS + 7 * 81;
/// `teacher_raw` of a mate in n plies: ±(MATE_RAW − min(n, 9999)) — beyond every cp score
/// YaneuraOu prints (at most 100·VALUE_MAX_EVAL/90 = 35281).
pub const MATE_RAW: f32 = 100_000.0;
const MATE_PLY_CAP: i32 = 9999;

/// USI options sent before `--engine-option` (which overrides them by name); see the
/// module docs. MultiPV is always `--max-moves`.
pub const DEFAULT_OPTIONS: &[(&str, &str)] = &[
    ("Threads", "1"),
    ("USI_Hash", "16"),
    ("USI_OwnBook", "false"),
    ("BookFile", "no_book"),
    ("FV_SCALE", "20"),
    ("DrawValueBlack", "0"),
    ("DrawValueWhite", "0"),
    ("GenerateAllLegalMoves", "true"),
    ("EnteringKingRule", "CSARule27"),
    ("MaxMovesToDraw", "0"),
];

/// Tensors the labeller replays and verifies.
const REQUIRED: [&str; 16] = [
    "obs.board_seq",
    "obs.hand_seq",
    "obs.globals_seq",
    "tgt.dropped",
    "tgt.from_idx",
    "tgt.to_idx",
    "tgt.promoted",
    "tgt.piece_idx",
    "tgt.drop_to_idx",
    "tgt.value",
    "tgt.legal_mode",
    "tgt.legal_from",
    "tgt.legal_to_of",
    "tgt.legal_promo_of",
    "tgt.legal_drop_piece",
    "tgt.legal_drop_to_of",
];

#[derive(clap::Args, Debug, Clone)]
#[command(args_conflicts_with_subcommands = true)]
pub struct Args {
    #[command(flatten)]
    pub common: Option<LabelArgs>,
    #[command(subcommand)]
    pub tool: Option<Tool>,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum Tool {
    /// Serve the deterministic fake USI engine on stdin/stdout (tests; no engine binary)
    #[command(hide = true)]
    FakeUsi,
}

pub fn run(args: Args) -> Result<()> {
    match (args.tool, args.common) {
        (Some(Tool::FakeUsi), _) => super::fake::run_stdio(&mut fake_engine()),
        (None, Some(c)) => super::run_labeller(ShogiLabeller::new(c.max_moves), &c),
        (None, None) => bail!("gmdata label shogi needs --input, --out and --engine"),
    }
}

// ------------------------------------------------------------------ actions, USI, SFEN

/// A decoded flat action id (POV squares).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Board { from: i64, to: i64, promote: bool },
    Drop { slot: i64, to: i64 },
}

impl Action {
    pub fn id(&self) -> i64 {
        match *self {
            Action::Board { from, to, promote } => (from * 81 + to) * 2 + promote as i64,
            Action::Drop { slot, to } => N_BOARD_ACTIONS + slot * 81 + to,
        }
    }

    /// `None` for ids outside [0, N_ACTIONS).
    pub fn decode(id: i64) -> Option<Action> {
        if (0..N_BOARD_ACTIONS).contains(&id) {
            Some(Action::Board { from: id / 2 / 81, to: id / 2 % 81, promote: id % 2 == 1 })
        } else if (N_BOARD_ACTIONS..N_ACTIONS).contains(&id) {
            let r = id - N_BOARD_ACTIONS;
            Some(Action::Drop { slot: r / 81, to: r % 81 })
        } else {
            None
        }
    }

    /// The action of move `m` (absolute squares) for the side `mover_black`.
    pub(crate) fn of(m: &Mv, mover_black: bool) -> Action {
        let to = rules::pov(m.to, mover_black);
        match m.from {
            Some(f) => Action::Board { from: rules::pov(f, mover_black), to, promote: m.promotion },
            None => Action::Drop { slot: m.drop as i64 - 1, to },
        }
    }
}

const USI_PIECES: [char; 8] = ['?', 'P', 'L', 'N', 'S', 'G', 'B', 'R'];

fn usi_square(sq: usize) -> String {
    format!("{}{}", 9 - sq % 9, (b'a' + (sq / 9) as u8) as char)
}

fn parse_usi_square(file: u8, rank: u8) -> Option<usize> {
    let (f, r) = (file.checked_sub(b'0')?, rank.checked_sub(b'a')?);
    ((1..=9).contains(&f) && r < 9).then(|| r as usize * 9 + (9 - f as usize))
}

/// USI notation of a move (absolute squares): `7g7f`, `8h2b+`, `P*5e`.
pub(crate) fn usi_move(m: &Mv) -> String {
    match m.from {
        Some(f) => format!("{}{}{}", usi_square(f), usi_square(m.to), if m.promotion { "+" } else { "" }),
        None => format!("{}*{}", USI_PIECES[m.drop as usize], usi_square(m.to)),
    }
}

/// Parse a USI move (no legality check).
pub(crate) fn parse_usi_move(s: &str) -> Option<Mv> {
    let b = s.as_bytes();
    if b.len() == 4 && b[1] == b'*' {
        let drop = USI_PIECES.iter().position(|&c| c as u8 == b[0]).filter(|&p| p >= 1)? as u8;
        return Some(Mv { from: None, to: parse_usi_square(b[2], b[3])?, promotion: false, drop });
    }
    if b.len() == 4 || (b.len() == 5 && b[4] == b'+') {
        let from = parse_usi_square(b[0], b[1])?;
        let to = parse_usi_square(b[2], b[3])?;
        return Some(Mv { from: Some(from), to, promotion: b.len() == 5, drop: NONE });
    }
    None
}

const SFEN_PIECES: [&str; 15] = ["", "P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R"];

/// The hirate start position (black to move, move 1).
pub(crate) fn hirate_board() -> Board {
    let mut b = Board { piece: [NONE; 81], color: [0; 81], hand: [[0; 9]; 2], turn: BLACK, king: [None, None], move_number: 1 };
    for (sq, p) in rules::hirate().iter().enumerate() {
        if let Some((pt, c)) = *p {
            b.piece[sq] = pt;
            b.color[sq] = c as u8;
            if pt == rules::KING {
                b.king[c] = Some(sq);
            }
        }
    }
    b
}

/// SFEN of a position (board, side to move, hands in R B G S N L P order, move number).
pub(crate) fn sfen(b: &Board) -> String {
    let mut s = String::new();
    for r in 0..9 {
        let mut empty = 0;
        for c in 0..9 {
            let sq = r * 9 + c;
            if b.piece[sq] == NONE {
                empty += 1;
                continue;
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
                empty = 0;
            }
            let p = SFEN_PIECES[b.piece[sq] as usize];
            s.push_str(&if b.color[sq] as usize == BLACK { p.to_string() } else { p.to_lowercase() });
        }
        if empty > 0 {
            s.push_str(&empty.to_string());
        }
        if r < 8 {
            s.push('/');
        }
    }
    s.push_str(if b.turn == BLACK { " b " } else { " w " });
    let mut hands = String::new();
    for c in [BLACK, WHITE] {
        for pt in (PAWN..=ROOK).rev().map(|p| p as usize) {
            let n = b.hand[c][pt];
            if n == 0 {
                continue;
            }
            if n > 1 {
                hands.push_str(&n.to_string());
            }
            let l = SFEN_PIECES[pt];
            hands.push_str(&if c == BLACK { l.to_string() } else { l.to_lowercase() });
        }
    }
    s.push_str(if hands.is_empty() { "-" } else { &hands });
    s.push_str(&format!(" {}", b.move_number));
    s
}

/// Parse SFEN fields (board, side, hands, [move number]).
pub(crate) fn parse_sfen(fields: &[&str]) -> Result<Board> {
    ensure!(fields.len() >= 3, "SFEN needs board, side and hands");
    let mut b = Board { piece: [NONE; 81], color: [0; 81], hand: [[0; 9]; 2], turn: BLACK, king: [None, None], move_number: 1 };
    let rows: Vec<&str> = fields[0].split('/').collect();
    ensure!(rows.len() == 9, "SFEN board {:?} has {} rows", fields[0], rows.len());
    for (r, row) in rows.iter().enumerate() {
        let (mut c, mut promoted) = (0usize, false);
        for ch in row.chars() {
            if let Some(d) = ch.to_digit(10) {
                c += d as usize;
                continue;
            }
            if ch == '+' {
                promoted = true;
                continue;
            }
            let name = format!("{}{}", if promoted { "+" } else { "" }, ch.to_ascii_uppercase());
            let pt = SFEN_PIECES.iter().position(|&p| p == name).filter(|&p| p > 0).with_context(|| format!("SFEN piece {name:?}"))?;
            ensure!(c < 9, "SFEN row {row:?} is longer than 9 squares");
            let sq = r * 9 + c;
            let owner = if ch.is_ascii_uppercase() { BLACK } else { WHITE };
            b.piece[sq] = pt as u8;
            b.color[sq] = owner as u8;
            if pt as u8 == rules::KING {
                b.king[owner] = Some(sq);
            }
            c += 1;
            promoted = false;
        }
        ensure!(c == 9, "SFEN row {row:?} covers {c} squares");
    }
    b.turn = match fields[1] {
        "b" => BLACK,
        "w" => WHITE,
        s => bail!("SFEN side {s:?}"),
    };
    if fields[2] != "-" {
        let mut n = 0u32;
        for ch in fields[2].chars() {
            if let Some(d) = ch.to_digit(10) {
                n = n * 10 + d;
                continue;
            }
            let pt = SFEN_PIECES.iter().position(|&p| p == ch.to_ascii_uppercase().to_string()).filter(|&p| (1..=7).contains(&p));
            let pt = pt.with_context(|| format!("SFEN hand piece {ch:?}"))?;
            b.hand[if ch.is_ascii_uppercase() { BLACK } else { WHITE }][pt] += n.max(1);
            n = 0;
        }
    }
    if let Some(m) = fields.get(3) {
        b.move_number = m.parse().with_context(|| format!("SFEN move number {m:?}"))?;
    }
    Ok(b)
}

/// The position of a USI `position startpos|sfen … [moves …]` line (moves checked for
/// legality with the converter's rules).
pub(crate) fn parse_position(line: &str) -> Result<Board> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    ensure!(toks.first() == Some(&"position"), "not a position line: {line:?}");
    let mv_at = toks.iter().position(|t| *t == "moves").unwrap_or(toks.len());
    let mut b = match toks.get(1) {
        Some(&"startpos") => hirate_board(),
        Some(&"sfen") => parse_sfen(&toks[2..mv_at])?,
        _ => bail!("position line without startpos / sfen: {line:?}"),
    };
    for u in toks.iter().skip(mv_at + 1) {
        let m = parse_usi_move(u).with_context(|| format!("bad USI move {u:?}"))?;
        ensure!(b.is_legal(&m), "illegal move {u} in {line:?}");
        b.push(&m);
    }
    Ok(b)
}

// ------------------------------------------------------------------ replay

/// A replayed game: the USI start and the moves of every frame.
#[derive(Debug)]
pub struct GameLine {
    /// `startpos` or `sfen <sfen>`
    start: String,
    /// USI moves, one per frame
    usi: Vec<String>,
}

impl GameLine {
    /// The USI `position` line of the position before move `ply`.
    pub fn position(&self, ply: usize) -> String {
        let mut s = format!("position {}", self.start);
        if ply > 0 {
            s.push_str(" moves");
            for m in &self.usi[..ply] {
                s.push(' ');
                s.push_str(m);
            }
        }
        s
    }
}

/// A replayed frame.
pub struct ShogiPos {
    game: Arc<GameLine>,
    ply: usize,
    board: Board,
    /// the recorded move
    mv: Mv,
    /// the converter's legal moves (its generation order)
    legal: Vec<Mv>,
    /// their action ids, sorted
    legal_ids: Vec<i64>,
    key: u128,
}

impl ShogiPos {
    pub fn position_line(&self) -> String {
        self.game.position(self.ply)
    }

    pub fn mover_black(&self) -> bool {
        self.board.turn == BLACK
    }

    pub fn is_legal_action(&self, id: i64) -> bool {
        self.legal_ids.binary_search(&id).is_ok()
    }
}

/// Two fixed-key 64-bit hashers → a 128-bit key of everything written so far.
#[derive(Clone)]
struct Key128(std::collections::hash_map::DefaultHasher, std::collections::hash_map::DefaultHasher);

impl Key128 {
    fn new() -> Self {
        let (mut a, mut b) = (std::collections::hash_map::DefaultHasher::new(), std::collections::hash_map::DefaultHasher::new());
        a.write_u8(0xA5);
        b.write_u8(0x5A);
        Key128(a, b)
    }

    fn write(&mut self, s: &str) {
        self.0.write(s.as_bytes());
        self.0.write_u8(0xFF);
        self.1.write(s.as_bytes());
        self.1.write_u8(0xFE);
    }

    fn get(&self) -> u128 {
        ((self.0.finish() as u128) << 64) | self.1.finish() as u128
    }
}

/// Target columns of one frame → the move (absolute squares).
fn target_move(t: [i64; 6], mover_black: bool) -> Result<Mv> {
    let [dropped, from, to, promoted, piece, drop_to] = t;
    let abs = |p: i64| -> Result<usize> {
        ensure!((0..81).contains(&p), "square index {p} outside 0..81");
        Ok(if mover_black { p as usize } else { 80 - p as usize })
    };
    match dropped {
        1 => {
            ensure!((0..7).contains(&piece), "drop piece_idx {piece} outside 0..7");
            Ok(Mv { from: None, to: abs(drop_to)?, promotion: false, drop: (piece + 1) as u8 })
        }
        0 => Ok(Mv { from: Some(abs(from)?), to: abs(to)?, promotion: promoted != 0, drop: NONE }),
        d => bail!("tgt.dropped = {d}"),
    }
}

/// The frame targets of a sample: [dropped, from, to, promoted, piece, drop_to] per frame.
fn targets(s: &SampleView) -> Result<Vec<[i64; 6]>> {
    let cols: Vec<_> = ["tgt.dropped", "tgt.from_idx", "tgt.to_idx", "tgt.promoted", "tgt.piece_idx", "tgt.drop_to_idx"]
        .iter()
        .map(|n| s.i64s(n))
        .collect::<Result<_>>()?;
    Ok((0..s.len).map(|t| std::array::from_fn(|j| cols[j][t])).collect())
}

// ------------------------------------------------------------------ engine

/// A YaneuraOu session: the USI conversation plus the eval file it loaded.
pub struct ShogiSession {
    pub uci: UciSession,
    /// the NNUE file named by `info string loading eval file : …` (absent for other engines)
    pub eval_file: Option<PathBuf>,
}

/// Default options overridden by the user's (by case-insensitive name), user-only options
/// appended in order, MultiPV last.
pub fn effective_options(user: &[(String, String)], max_moves: usize) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = DEFAULT_OPTIONS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    for (k, v) in user {
        if k.eq_ignore_ascii_case("MultiPV") {
            continue; // always --max-moves
        }
        match out.iter_mut().find(|(d, _)| d.eq_ignore_ascii_case(k)) {
            Some(slot) => slot.1 = v.clone(),
            None => out.push((k.clone(), v.clone())),
        }
    }
    out.push(("MultiPV".into(), max_moves.to_string()));
    out
}

/// The option name of a `usi` handshake line `option name <NAME> type …`.
fn usi_option_name(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("option name ")?;
    Some(rest.split(" type ").next().unwrap_or(rest).trim())
}

/// Options of `opts` the engine did not announce in its `usi` answer (case-insensitive).
/// An engine that announces no options at all (the fake engines) is not checked.
pub fn unknown_options<'a>(hello: &[String], opts: &'a [(String, String)]) -> Vec<&'a str> {
    let known: Vec<&str> = hello.iter().filter_map(|l| usi_option_name(l)).collect();
    if known.is_empty() {
        return Vec::new();
    }
    opts.iter().map(|(k, _)| k.as_str()).filter(|k| !known.iter().any(|n| n.eq_ignore_ascii_case(k))).collect()
}

/// A content fingerprint of an eval file (size + a 64-bit hash), machine independent.
fn file_fingerprint(p: &Path) -> Result<String> {
    let data = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in data.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    Ok(format!("{} bytes, h64 {h:016x}", data.len()))
}

/// Path-like values reduced to their file name (the teacher meta must match across machines).
fn meta_value(v: &str) -> String {
    if v.contains('/') || v.contains('\\') {
        Path::new(v).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| v.to_string())
    } else {
        v.to_string()
    }
}

/// `teacher_raw` of a score (cp as is; mates ±(MATE_RAW − min(n, 9999))).
pub fn raw_score(s: Score) -> f32 {
    match s {
        Score::Cp(cp) => cp as f32,
        Score::Mate(n) if n > 0 => MATE_RAW - n.min(MATE_PLY_CAP) as f32,
        Score::Mate(n) => -(MATE_RAW - n.saturating_neg().min(MATE_PLY_CAP) as f32),
    }
}

/// Mover-POV expected score of a score.
pub fn q_score(s: Score) -> f32 {
    match s {
        Score::Cp(cp) => q_from_cp(cp as f32, CP_SCALE),
        Score::Mate(n) => q_from_mate(n),
    }
}

/// The final MultiPV lines of a search → scored moves (flat ids of the converter's legal
/// moves; others dropped) and the value (Q of line 1, when its move is kept).
///
/// Stale lines: a search stopped by the node limit ends inside an iteration, and the lines
/// not yet re-searched in it are printed with their score from the previous, shallower
/// iteration (a lower `depth` than the lines of the unfinished one). Such a score can be far
/// too high — each MultiPV pass of the unfinished iteration picks the best remaining move, so
/// it had ranked them below its own lines — and would dominate the softmax, so the Q and raw
/// score of every line shallower than the deepest one are capped at the lowest Q / raw score
/// among the deepest lines (dropped lines included: the bound comes from the search, not from
/// our move list).
pub fn analysis(pos: &ShogiPos, lines: &[Info]) -> Result<Analysis> {
    let mb = pos.mover_black();
    // (multipv, depth, action id if the converter's rules allow the move, Q, raw)
    let mut scored = Vec::with_capacity(lines.len());
    for line in lines {
        let (Some(score), Some(first)) = (line.score, line.pv.first()) else { continue };
        let Some(m) = parse_usi_move(first) else {
            if first == "win" || first == "resign" {
                continue;
            }
            bail!("engine move {first:?} is not USI");
        };
        let id = Action::of(&m, mb).id();
        // outside the converter's move generator → dropped (see the module docs)
        scored.push((line.multipv, line.depth.unwrap_or(0), pos.is_legal_action(id).then_some(id), q_score(score), raw_score(score)));
    }
    let deepest = scored.iter().map(|s| s.1).max().unwrap_or(0);
    let (q_cap, raw_cap) =
        scored.iter().filter(|s| s.1 == deepest).fold((f32::INFINITY, f32::INFINITY), |(q, r), s| (q.min(s.3), r.min(s.4)));
    let mut moves = Vec::with_capacity(scored.len());
    let mut value = None;
    for (multipv, depth, id, mut q, mut raw) in scored {
        if depth < deepest {
            q = q.min(q_cap);
            raw = raw.min(raw_cap);
        }
        let Some(id) = id else { continue };
        if multipv == 1 {
            value = Some(q);
        }
        moves.push(ScoredMove::new(id, q).raw(raw));
    }
    Ok(Analysis { moves, value })
}

// ------------------------------------------------------------------ labeller

/// The shogi labeller (see the module docs).
#[derive(Debug, Clone)]
pub struct ShogiLabeller {
    /// width of the sparse head rows (from / to / drop_to)
    pub k: usize,
}

impl ShogiLabeller {
    /// `max_moves` = `--max-moves` (the sparse heads keep one class per scored move).
    pub fn new(max_moves: Option<usize>) -> Self {
        ShogiLabeller { k: max_moves.unwrap_or(DEFAULT_MAX_MOVES).clamp(1, 81) }
    }
}

impl Default for ShogiLabeller {
    fn default() -> Self {
        ShogiLabeller::new(None)
    }
}

impl Labeller for ShogiLabeller {
    type Position = ShogiPos;
    type Key = u128;
    type Session = ShogiSession;

    fn game(&self) -> &'static str {
        "shogi"
    }

    fn version(&self) -> u32 {
        LABELLER_VERSION
    }

    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        if shard.game != "shogi" || shard.converter_version != rules::CONVERTER_VERSION {
            bail!(
                "{}: {} converter_version {} — gmdata label shogi needs shogi converter_version {} shards (re-convert with gmdata shogi convert)",
                shard.dir.display(),
                shard.game,
                shard.converter_version,
                rules::CONVERTER_VERSION
            );
        }
        for t in REQUIRED {
            shard.tensor_info(t)?;
        }
        Ok(())
    }

    fn heads(&self) -> Vec<HeadSpec> {
        let k = self.k;
        vec![
            HeadSpec { name: "mode", kind: HeadKind::Categorical { n: 2, k: 2 } },
            HeadSpec { name: "from", kind: HeadKind::Categorical { n: 81, k } },
            HeadSpec { name: "to", kind: HeadKind::Categorical { n: 81, k } },
            HeadSpec { name: "promo", kind: HeadKind::Categorical { n: 2, k: 2 } },
            HeadSpec { name: "piece", kind: HeadKind::Categorical { n: 14, k: 14 } },
            HeadSpec { name: "drop_to", kind: HeadKind::Categorical { n: 81, k } },
        ]
    }

    fn raw_unit(&self) -> Option<&'static str> {
        Some("cp")
    }

    fn default_max_moves(&self) -> usize {
        DEFAULT_MAX_MOVES
    }

    fn resolve_limits(&self, given: &Limits) -> Result<Limits> {
        let l = given.with_defaults(&[("nodes", &DEFAULT_NODES.to_string())], &["nodes"])?;
        ensure!(l.u64_or("nodes", DEFAULT_NODES)? > 0, "--limit nodes must be positive");
        Ok(l)
    }

    fn open_session(&self, cfg: &EngineConfig) -> Result<ShogiSession> {
        // UciSession::open, keeping the isready output (it names the eval file)
        let engine = Engine::spawn(cfg.command.clone())?;
        let mut uci = UciSession { engine, flavor: Flavor::Usi, id_name: String::new(), timeout: cfg.timeout };
        let hello = uci.engine.request("usi", cfg.timeout, |l| l.trim() == "usiok")?;
        uci.id_name = hello.iter().find_map(|l| l.strip_prefix("id name ")).unwrap_or("unknown engine").trim().to_string();
        let opts = effective_options(&cfg.options, cfg.max_moves);
        // YaneuraOu only prints "No such option: X" and carries on — a typo in --engine-option
        // or a build without e.g. FV_SCALE would label with other settings than the meta records
        let unknown = unknown_options(&hello, &opts);
        ensure!(
            unknown.is_empty(),
            "{} has no USI option(s) {unknown:?} (misspelt --engine-option, or another YaneuraOu build than scripts/engines/fetch_engines.sh installs?)",
            uci.id_name
        );
        for (k, v) in &opts {
            uci.setoption(k, v)?;
        }
        let ready = uci.engine.request("isready", cfg.timeout, |l| l.trim() == "readyok")?;
        let eval_file = ready
            .iter()
            .find_map(|l| l.split_once("loading eval file :").map(|(_, p)| PathBuf::from(p.trim())))
            .filter(|p| p.is_file());
        Ok(ShogiSession { uci, eval_file })
    }

    fn engine_id(&self, s: &ShogiSession) -> String {
        match &s.eval_file {
            Some(p) => match file_fingerprint(p) {
                Ok(f) => format!("{} + eval ({f})", s.uci.id_name),
                Err(e) => format!("{} + eval (unreadable: {e})", s.uci.id_name),
            },
            None => s.uci.id_name.clone(),
        }
    }

    fn teacher_meta(&self, cfg: &EngineConfig) -> Value {
        let opts: serde_json::Map<String, Value> =
            effective_options(&cfg.options, cfg.max_moves).into_iter().map(|(k, v)| (k, Value::String(meta_value(&v)))).collect();
        json!({
            "action_id": "board move (from*81 + to)*2 + promote; drop 13122 + slot*81 + to; mover-POV squares (tgt.from_idx convention), slot P L N S G B R",
            "q": "logistic(cp / cp_scale); mate for the mover 1, mated 0",
            "cp_scale": CP_SCALE,
            "raw": "engine cp; mate in n plies +-(100000 - min(n, 9999))",
            "value": "Q of the first MultiPV line",
            "stale_lines": "lines shallower than the deepest capped at the lowest Q / raw of the deepest lines",
            "position": "position startpos|sfen <start> moves <game so far>",
            "usi_options": opts,
        })
    }

    fn replay(&self, s: &SampleView) -> Result<Vec<ShogiPos>> {
        ensure!(s.len > 0, "empty sample");
        let board = s.f32s("obs.board_seq")?;
        let hand = s.f32s("obs.hand_seq")?;
        let glb = s.f32s("obs.globals_seq")?;
        let g = glb.len() / s.len;
        let start = rules::board_from_obs(&board[..OBS_BOARD], &hand[..56], &glb[..g]).map_err(|e| anyhow::anyhow!("frame 0: {e}"))?;
        let tg = targets(s)?;
        // pass 1: the moves (checked against the converter's generator)
        let mut b = start.clone();
        let mut frames: Vec<(Board, Mv, Vec<Mv>)> = Vec::with_capacity(s.len);
        for (t, row) in tg.iter().enumerate() {
            let m = target_move(*row, b.turn == BLACK).with_context(|| format!("frame {t}"))?;
            let legal = b.legal_moves();
            if !legal.contains(&m) {
                bail!("frame {t}: recorded move {} is not legal in {}", usi_move(&m), sfen(&b));
            }
            let next = {
                let mut n = b.clone();
                n.push(&m);
                n
            };
            frames.push((b, m, legal));
            b = next;
        }
        let game = Arc::new(GameLine {
            start: if start == hirate_board() { "startpos".to_string() } else { format!("sfen {}", sfen(&start)) },
            usi: frames.iter().map(|(_, m, _)| usi_move(m)).collect(),
        });
        // pass 2: positions with their cache keys (the exact position line)
        let mut key = Key128::new();
        key.write(&game.start);
        let mut out = Vec::with_capacity(s.len);
        for (ply, (board, mv, legal)) in frames.into_iter().enumerate() {
            if ply > 0 {
                key.write(&game.usi[ply - 1]);
            }
            let mb = board.turn == BLACK;
            let mut legal_ids: Vec<i64> = legal.iter().map(|m| Action::of(m, mb).id()).collect();
            legal_ids.sort_unstable();
            out.push(ShogiPos { game: game.clone(), ply, board, mv, legal, legal_ids, key: key.get() });
        }
        Ok(out)
    }

    fn verify(&self, s: &SampleView, positions: &[ShogiPos]) -> Result<()> {
        let n = positions.len();
        let result = s.meta()["result"].as_str().context("sample meta has no \"result\"")?;
        let moves: Vec<Mv> = positions.iter().map(|p| p.mv).collect();
        let reps = rules::repetition_counts(&positions[0].board, &moves);
        let mut board = vec![0f32; n * OBS_BOARD];
        let (mut hand, mut glb, mut value) = (Vec::with_capacity(n * 56), Vec::with_capacity(n * 7), Vec::with_capacity(n));
        let mut idx: [Vec<i64>; 6] = Default::default();
        let (mut lm, mut lf, mut lto, mut lpr, mut ldp, mut ldt) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        let bit = |v: &[bool], out: &mut Vec<f32>| out.extend(v.iter().map(|&x| x as u8 as f32));
        for (t, p) in positions.iter().enumerate() {
            let mb = p.mover_black();
            rules::encode_obs(&p.board, reps[t], &mut board[t * OBS_BOARD..(t + 1) * OBS_BOARD], &mut hand, &mut glb);
            let (d, fi, ti, pr, pi, dt) = rules::move_indices(&p.mv, mb);
            for (col, v) in idx.iter_mut().zip([d, fi, ti, pr as i64, pi, dt]) {
                col.push(v);
            }
            let f = rules::frame_masks(&p.legal, mb, fi, ti, pi);
            bit(&f.mode, &mut lm);
            bit(&f.from, &mut lf);
            bit(&f.to_of, &mut lto);
            bit(&f.promo_of, &mut lpr);
            bit(&f.drop_piece, &mut ldp);
            bit(&f.drop_to_of, &mut ldt);
            value.push(rules::result_value(result, mb));
        }
        s.check_f32("obs.board_seq", &board)?;
        s.check_f32("obs.hand_seq", &hand)?;
        s.check_f32("obs.globals_seq", &glb)?;
        for (name, col) in ["tgt.dropped", "tgt.from_idx", "tgt.to_idx", "tgt.promoted", "tgt.piece_idx", "tgt.drop_to_idx"].iter().zip(&idx) {
            s.check_i64(name, col)?;
        }
        s.check_f32("tgt.legal_mode", &lm)?;
        s.check_f32("tgt.legal_from", &lf)?;
        s.check_f32("tgt.legal_to_of", &lto)?;
        s.check_f32("tgt.legal_promo_of", &lpr)?;
        s.check_f32("tgt.legal_drop_piece", &ldp)?;
        s.check_f32("tgt.legal_drop_to_of", &ldt)?;
        s.check_f32("tgt.value", &value)
    }

    fn skip_reason(&self, _s: &SampleView, _t: usize, pos: &ShogiPos) -> Option<&'static str> {
        (pos.legal.len() == 1).then_some("forced")
    }

    fn key(&self, pos: &ShogiPos) -> u128 {
        pos.key
    }

    fn query(&self, s: &mut ShogiSession, pos: &ShogiPos, cfg: &EngineConfig) -> Result<Analysis> {
        s.uci.new_game()?;
        let nodes = cfg.limits.u64_or("nodes", DEFAULT_NODES)?;
        let r = s.uci.search(&pos.position_line(), &format!("go nodes {nodes}"), cfg.timeout)?;
        let a = analysis(pos, &r.lines)?;
        if a.moves.is_empty() && r.bestmove == "win" {
            bail!("the engine declares a win (entering-king rule) instead of scoring moves");
        }
        Ok(a)
    }

    fn head_rows(&self, s: &SampleView, t: usize, _pos: &ShogiPos, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>> {
        let (dropped, gf, gt, gp) = (s.i64s("tgt.dropped")?[t], s.i64s("tgt.from_idx")?[t], s.i64s("tgt.to_idx")?[t], s.i64s("tgt.piece_idx")?[t]);
        let dist: Vec<(Action, f32)> =
            policy.iter().map(|&(a, p)| Action::decode(a).map(|x| (x, p)).with_context(|| format!("action id {a} out of range"))).collect::<Result<_>>()?;
        let is_board = |a: &Action| matches!(a, Action::Board { .. });
        let is_drop = |a: &Action| matches!(a, Action::Drop { .. });
        let from_of = |a: &Action| if let Action::Board { from, .. } = *a { from } else { -1 };
        let to_of = |a: &Action| match *a {
            Action::Board { to, .. } | Action::Drop { to, .. } => to,
        };
        let slot_of = |a: &Action| if let Action::Drop { slot, .. } = *a { slot } else { -1 };
        let cat = HeadRow::Categorical;
        Ok(vec![
            cat(marginal(&dist, |a| is_drop(a) as i64)),
            cat(conditional(&dist, is_board, from_of)),
            if dropped == 0 { cat(conditional(&dist, |a| is_board(a) && from_of(a) == gf, to_of)) } else { HeadRow::Missing },
            if dropped == 0 {
                cat(conditional(
                    &dist,
                    |a| is_board(a) && from_of(a) == gf && to_of(a) == gt,
                    |a| matches!(a, Action::Board { promote: true, .. }) as i64,
                ))
            } else {
                HeadRow::Missing
            },
            cat(conditional(&dist, is_drop, slot_of)),
            if dropped == 1 { cat(conditional(&dist, |a| is_drop(a) && slot_of(a) == gp, to_of)) } else { HeadRow::Missing },
        ])
    }
}

// ------------------------------------------------------------------ fake engine

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// A [`UciFake`] that announces YaneuraOu's option names ([`DEFAULT_OPTIONS`]) in its `usi`
/// answer, like the real engine — so sessions pass the option check of `open_session`.
pub struct FakeYaneuraOu<S>(pub UciFake<S>);

impl<S: FnMut(&str) -> Option<Vec<FakeMove>> + Send> FakeYaneuraOu<S> {
    /// Crash (exit) on the `n`-th search ([`UciFake::crash_on`]).
    pub fn crash_on(self, n: usize) -> Self {
        FakeYaneuraOu(self.0.crash_on(n))
    }

    /// Sleep `d` before answering the `n`-th search ([`UciFake::hang_on`]).
    pub fn hang_on(self, n: usize, d: std::time::Duration) -> Self {
        FakeYaneuraOu(self.0.hang_on(n, d))
    }
}

impl<S: FnMut(&str) -> Option<Vec<FakeMove>> + Send> LineHandler for FakeYaneuraOu<S> {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        let mut out = self.0.handle(line)?;
        if line.trim() == "usi" {
            let at = out.len().saturating_sub(1); // before usiok
            for (k, _) in DEFAULT_OPTIONS.iter().rev() {
                out.insert(at, format!("option name {k} type string default <empty>"));
            }
        }
        Some(out)
    }
}

/// The fake USI engine: every legal move of the position line (the converter's move
/// generator) scored by a hash — `cp ∈ [−600, 600]`, ~2% of the moves a mate for the
/// mover and ~2% a mate against it; an unparsable or illegal position line crashes it.
pub fn fake_engine() -> FakeYaneuraOu<impl FnMut(&str) -> Option<Vec<FakeMove>> + Send> {
    FakeYaneuraOu(UciFake::new(|pos: &str| {
        let b = parse_position(pos).ok()?;
        let h0 = fnv(pos);
        Some(
            b.legal_moves()
                .iter()
                .map(|m| {
                    let u = usi_move(m);
                    let h = (h0 ^ fnv(&u)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let h = h ^ (h >> 31);
                    match h % 50 {
                        0 => FakeMove::mate(&u, 1 + (h >> 8) as i32 % 15),
                        1 => FakeMove::mate(&u, -1 - (h >> 8) as i32 % 15),
                        _ => FakeMove::cp(&u, ((h >> 16) % 1201) as i32 - 600),
                    }
                })
                .collect(),
        )
    }))
}

/// [`fake_engine`] as an in-process engine command.
pub fn fake_engine_command() -> EngineCommand {
    EngineCommand::in_process("fake-usi", fake_engine)
}

#[cfg(test)]
mod tests;
