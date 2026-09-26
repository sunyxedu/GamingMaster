//! `gmdata label chess` — Stockfish (UCI) teacher labels for chess gm-shards
//! (converter v3, [`crate::chess`]).
//!
//! ```text
//! scripts/engines/fetch_engines.sh stockfish          # → ${GM_ENGINES:-$HOME/gm-engines}/bin/stockfish
//! gmdata label chess --input data/chess/train --out data/chess_sf/train \
//!     --engine ~/gm-engines/bin/stockfish --workers 64
//! gmdata label chess --input data/chess/val   --out data/chess_sf/val \
//!     --engine ~/gm-engines/bin/stockfish --workers 64          # same settings as train!
//! python -m training.train_il_game --game chess --data_dir data/chess_sf/train \
//!     --val_dir data/chess_sf/val --soft_target_weight 1.0 --engine_value_weight 0.5 ...
//! ```
//!
//! Defaults (every flag of [`super::LabelArgs`] applies; these are the chess ones):
//!
//! * `--limit nodes=120000` (`go nodes N`; `--limit depth=D` adds a depth cap, `nodes=0`
//!   drops the node cap). Chosen for ≈ 0.1 s per position: Stockfish 19, one thread,
//!   MultiPV 8, `ucinewgame` before every search, measured on an Apple M4 Pro
//!   performance core: 50k nodes 27.7 pos/s, 100k 13.9, 150k 8.0, 200k 6.2 (≈ 1.4 M
//!   nodes/s; ≈ 2.5 ms fixed cost per position for `ucinewgame` + `isready`). A node
//!   budget (not movetime) keeps labels reproducible and machine independent.
//! * `--max-moves 8` = UCI `MultiPV` = the width K of the teacher columns (the task's
//!   default; every extra PV line costs search depth at a fixed node budget). With
//!   τ = 0.05 a move 0.15 below the best in Q keeps e⁻³ ≈ 5% of the best move's
//!   weight — but WDL-based Q saturates: on real 2200+ games (default settings) 40% of
//!   the labelled frames have |engine value| > 0.9, where many moves are equally won
//!   (or lost), and the 8th move keeps > 20% of the best move's weight in 55% of the
//!   frames (median 0.3). There π is flat over the 8 scored moves and moves beyond
//!   them get 0. Raise `--max-moves`, lower `--tau`, or re-derive π from
//!   `tgt.teacher_raw` (cp) for decided positions if that matters.
//!   Measured on the same data: the human move is among the 8 scored moves in 90% of
//!   the frames and is the engine's first choice in ≈ 40%.
//! * `--tau 0.05`, `--policy softmax`: π(a) ∝ exp(Q(a)/τ) over the scored moves.
//! * Engine options: `Threads=1`, `Hash=16` (MB; plenty for ~10⁵ nodes, and cheap to
//!   clear) unless overridden with `--engine-option NAME=VALUE`; `MultiPV = K` and
//!   `UCI_ShowWDL = true` are always set last (the labels need them).
//! * `--workers` = all hardware threads. One single-threaded Stockfish per worker.
//!
//! Query per labelled frame: `ucinewgame`, `isready` (clears hash and history tables,
//! so an answer does not depend on what the worker searched before — labels are
//! deterministic whatever the worker count or order), `position startpos moves <the
//! game's moves before this frame>` (the full history, so repetitions and the 50-move
//! counter are seen as in the game), `go nodes N`. Of each MultiPV line the final
//! `info` line is used ([`super::uci::collect`]). The node limit stops the search inside
//! an iteration: a line can carry a bound (its score is used as is), and moves that
//! failed low in the unfinished iteration are printed with their older, shallower score
//! (depth below the final one), which can be far too high — on real 2200+ games at
//! 2000 nodes 9% of the frames had such a line above the principal one. Those lines are
//! capped at the lowest final-depth line's Q and cp ([`analysis`]), so the teacher
//! keeps Stockfish's own ranking:
//!
//! ```text
//! Q(move)            = (W + D/2) / 1000 of the line's `wdl` (mover POV; mate lines are
//!                      1000 0 0 / 0 0 1000); a line without `wdl` is an engine error
//! tgt.engine_value   = 2·Q(multipv 1) − 1   (the engine's root evaluation)
//! tgt.teacher_idx    [T,K] i64  promotion·4096 + from·64 + to, best (highest Q) first,
//!                               −1 pad; from / to = POV squares row·8 + file, row = 7 −
//!                               rank for white to move and rank for black (ranks / files
//!                               0-based from a1: the FromToDecoder labels), castling =
//!                               king → g/c file, promotion 0 none 1 N 2 B 3 R 4 Q
//! tgt.teacher_q      [T,K] f32  Q per move, NaN pad
//! tgt.teacher_raw    [T,K] f32  centipawns (meta.teacher.raw_unit "cp"); mate in n
//!                               (moves, UCI sign) = ±(100000 − |n|), NaN pad
//! tgt.from_probs_idx/_val [T,K] marginal of π over from-squares (sparse)
//! tgt.to_probs_idx/_val   [T,K] p(to | from = tgt.from_idx) (sparse; all-pad when the
//!                               teacher puts no mass on the ground-truth from-square)
//! tgt.promotion_probs     [T,5] p(promotion | tgt.from_idx, tgt.to_idx) (dense; a
//!                               non-promotion move is class 0; zeros = no teacher)
//! tgt.engine_value_mask   [T]   frame labelled
//! ```
//!
//! These are exactly the soft targets of `FromToDecoder.il_head_terms`
//! (model/games/board.py). Head widths are K, so the sparse rows are lossless.
//! Frames with a single legal move are skipped (`"forced"`: no teacher, no engine value).
//!
//! Replay and verification: each sample is replayed from the standard start position,
//! decoding tgt.from_idx / to_idx / promotion (POV) to the unique legal move with that
//! label ([`crate::chess::move_target`]); the positions are then re-encoded with the
//! converter's own per-frame code ([`crate::chess::encode_position`],
//! [`crate::chess::legal_masks`], repetition counts) and obs.board_seq,
//! obs.globals_seq, tgt.legal_from / legal_to_of / legal_promotion_of and tgt.value
//! (sample meta `result`) must match bit for bit, else the sample is skipped and
//! counted (the run fails above `--max-verify-fail-frac`, 0.1%).
//!
//! Position cache key (exact): Stockfish's answer to `position startpos moves …` +
//! `go nodes N` after `ucinewgame` is a function of the current position, its halfmove
//! clock and the positions of the last `halfmove clock` plies (repetition detection
//! looks back at most `rule50` plies; the evaluation depends on the rule-50 counter; the
//! game ply and move number are unused without time controls). That reversible tail is
//! fully determined by the position where it starts (the frame after the last capture
//! or pawn move, halfmove clock 0) and the moves since, so the key is
//! `EPD(tail start) + " " + UCI moves since`. Repeated opening lines hit the cache;
//! a repeated position inside a game has a longer tail and is (correctly) searched again.
//! Measured on two real 2200+ Lichess v3 shards (512 games, 40.5k frames): 7.3% of the
//! labelled frames hit with one worker (one cache), 4.8% with 8 workers.
//!
//! Throughput: positions/s per process ≈ 1 / (nodes ÷ nodes-per-second + ≈ 3 ms fixed
//! cost), ÷ (1 − cache hit rate). Measured end to end (`gmdata label chess`, default
//! settings, real 2200+ shards, an Apple M4 Pro shared with other jobs): one worker
//! 11.0 pos/s (≈ 0.09 s per position); 8 workers 68–76 pos/s (8.5–9.6 per worker), a
//! whole 20k-frame shard in 298 s with no engine error or verification failure. One
//! engine process needs ≈ 0.2–0.4 GB RSS (NNUE net + hash).
//! Expected on the cluster labelling nodes (AMD EPYC 9355, Zen 5, AVX-512 VNNI; the
//! universal sf_19 binary selects its avx512 VNNI build at runtime): ≈ 1.0–1.3 M nodes/s
//! per single-threaded process with every core busy, i.e. ≈ 8–11 pos/s per process,
//! ≈ 250–350 pos/s per 32-core socket with one worker per physical core (SMT siblings,
//! `--workers` = hardware threads, add ≈ 20–30%). A 20k-frame shard then takes
//! ≈ 30–40 s on a dual-socket node (64 workers); 1 M positions ≈ 0.4–0.6 node-hours.
//! Measure on the node with `cargo test --release -p gmdata real_stockfish_throughput
//! -- --ignored --nocapture` (`SF_NODES=120000 SF_MULTIPV=8`) or read `positions_per_s`
//! from a run's JSON summary (the progress log on stderr has pos/s and cache hit rate).
//!
//! Tests: fake-engine unit tests (no binary needed; [`fake_engine`], also served by
//! `gmdata label fake-engine chess`), replay/verify of crafted games (castling, en
//! passant, under-promotion, threefold repetition, forced moves) and of the PGN fixture,
//! real-shard golden replay (`GM_CHESS_SHARDS=<dir of chess shards>`, else
//! `${GM_DATA_SHARDS:-$HOME/gm-data/shards}/chess`; skipped when absent) and real
//! Stockfish tests (skipped when `scripts/engines/fetch_engines.sh stockfish` has not
//! been run and `STOCKFISH` is unset).

use super::fake::{FakeMove, UciFake};
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove};
use super::transform::{conditional, marginal, q_from_wdl};
use super::uci::{Flavor, Score, UciSession};
use super::LabelArgs;
use crate::chess::{encode_position, legal_masks, move_target, result_white, Repetitions, CONVERTER_VERSION, GLOBAL_FEATURES, N_PROMOTIONS, PLANE_LEN};
use anyhow::{bail, Context, Result};
use gmshard::ShardReader;
use serde_json::{json, Value};
use shakmaty::fen::Epd;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};
use std::sync::Arc;

/// Bump when the meaning of a chess label changes (recorded in the teacher meta).
pub const LABELLER_VERSION: u32 = 1;
/// `--limit nodes` default (≈ 0.1 s per position, see the module docs).
pub const DEFAULT_NODES: u64 = 120_000;
/// `--max-moves` default (= UCI MultiPV).
pub const DEFAULT_MAX_MOVES: usize = 8;
/// Engine options applied before `--engine-option` (which may override them).
pub const DEFAULT_OPTIONS: [(&str, &str); 2] = [("Threads", "1"), ("Hash", "16")];
/// `tgt.teacher_raw` of a mate in n: ±(MATE_RAW − |n|).
pub const MATE_RAW: f32 = 100_000.0;

/// POV (from, to, promotion) of a move — the FromToDecoder labels.
pub type Target = (i64, i64, i64);

/// Flat action id of `tgt.teacher_idx`: `promotion·4096 + from·64 + to`.
pub fn action_id((from, to, promo): Target) -> i64 {
    promo * 4096 + from * 64 + to
}

/// Inverse of [`action_id`].
pub fn decode_action(id: i64) -> Target {
    (id % 4096 / 64, id % 64, id / 4096)
}

/// `tgt.teacher_raw` of an engine score (centipawns; mates see [`MATE_RAW`]).
pub fn raw_score(s: Score) -> f32 {
    match s {
        Score::Cp(cp) => cp as f32,
        Score::Mate(n) if n > 0 => MATE_RAW - n as f32,
        Score::Mate(n) => -MATE_RAW - n as f32,
    }
}

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
}

pub fn run(args: Args) -> Result<()> {
    super::run_labeller(ChessLabeller::new(args.common.max_moves.unwrap_or(DEFAULT_MAX_MOVES)), &args.common)
}

/// The moves of one game as `position` commands: `text` = `"position startpos moves"`
/// followed by `" <uci>"` per move; `ends[t]` = length of the prefix holding the first
/// `t` moves.
#[derive(Debug)]
pub struct GameLine {
    text: String,
    ends: Vec<usize>,
}

impl GameLine {
    const HEAD: &'static str = "position startpos moves";

    fn new() -> Self {
        GameLine { text: Self::HEAD.to_string(), ends: vec![Self::HEAD.len()] }
    }

    fn push(&mut self, m: Move) {
        self.text.push(' ');
        m.to_uci(CastlingMode::Standard).append_to_string(&mut self.text);
        self.ends.push(self.text.len());
    }

    /// The `position` command of the position before move `t`.
    pub fn command(&self, t: usize) -> &str {
        if t == 0 {
            "position startpos"
        } else {
            &self.text[..self.ends[t]]
        }
    }

    /// `" m_{a+1} … m_b"`: the moves played from frame `a` to frame `b`.
    fn moves(&self, a: usize, b: usize) -> &str {
        &self.text[self.ends[a]..self.ends[b]]
    }
}

/// A replayed frame: the position before move `t` of a game.
#[derive(Debug)]
pub struct ChessPos {
    pub pos: Chess,
    /// legal moves with their POV targets
    pub legal: Vec<(Move, Target)>,
    /// index of the ground-truth move in `legal`
    pub gt: usize,
    /// earlier occurrences of the position in the game (repetition globals)
    pub prior: u32,
    /// frame index in the sample
    pub t: usize,
    pub line: Arc<GameLine>,
    /// the cache key (see the module docs)
    pub key: String,
}

impl ChessPos {
    pub fn gt_target(&self) -> Target {
        self.legal[self.gt].1
    }

    pub fn flip(&self) -> bool {
        self.pos.turn() == Color::Black
    }

    /// The UCI `position` command of this frame.
    pub fn command(&self) -> &str {
        self.line.command(self.t)
    }
}

/// Replay a game from the start position given its POV move labels (see the module docs).
pub fn replay_moves(from: &[i64], to: &[i64], promo: &[i64]) -> Result<Vec<ChessPos>> {
    let n = from.len();
    if to.len() != n || promo.len() != n {
        bail!("label columns of different lengths");
    }
    let mut pos = Chess::default();
    let mut reps = Repetitions::default();
    let mut line = GameLine::new();
    let (mut tail_start, mut tail_epd) = (0usize, String::new());
    let mut frames = Vec::with_capacity(n);
    for t in 0..n {
        let flip = pos.turn() == Color::Black;
        let prior = reps.see(&pos);
        let h = pos.halfmoves() as usize;
        if h == 0 {
            tail_start = t;
            tail_epd = Epd::from_position(&pos, EnPassantMode::Legal).to_string();
        } else if t < h || t - h != tail_start {
            bail!("frame {t}: halfmove clock {h} inconsistent with the replay (last reset at frame {tail_start})");
        }
        let legal: Vec<(Move, Target)> = pos.legal_moves().iter().filter_map(|m| move_target(m, flip).map(|x| (*m, x))).collect();
        let want = (from[t], to[t], promo[t]);
        let mut hits = legal.iter().enumerate().filter(|(_, l)| l.1 == want).map(|(i, _)| i);
        let gt = match (hits.next(), hits.next()) {
            (Some(i), None) => i,
            (None, _) => bail!("frame {t}: recorded move {want:?} (POV from, to, promotion) is not legal in {}", fen(&pos)),
            (Some(_), Some(_)) => bail!("frame {t}: recorded move {want:?} matches several legal moves in {}", fen(&pos)),
        };
        let key = format!("{tail_epd}{}", line.moves(tail_start, t));
        let m = legal[gt].0;
        frames.push((pos.clone(), legal, gt, prior, key));
        line.push(m);
        pos.play_unchecked(m);
    }
    let line = Arc::new(line);
    Ok(frames
        .into_iter()
        .enumerate()
        .map(|(t, (pos, legal, gt, prior, key))| ChessPos { pos, legal, gt, prior, t, line: line.clone(), key })
        .collect())
}

fn fen(pos: &Chess) -> String {
    shakmaty::fen::Fen::from_position(pos, EnPassantMode::Legal).to_string()
}

/// The converter's re-encoding of replayed frames: (obs.board_seq, obs.globals_seq,
/// legal_from, legal_to_of, legal_promotion_of), frame-major.
#[allow(clippy::type_complexity)]
pub fn reencode(frames: &[ChessPos]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let n = frames.len();
    let mut planes = vec![0f32; n * PLANE_LEN];
    let mut globals = Vec::with_capacity(n * GLOBAL_FEATURES);
    let (mut lf, mut lt, mut lp) = (Vec::with_capacity(n * 64), Vec::with_capacity(n * 64), Vec::with_capacity(n * N_PROMOTIONS));
    for (t, f) in frames.iter().enumerate() {
        encode_position(&f.pos, f.prior, &mut planes[t * PLANE_LEN..(t + 1) * PLANE_LEN], &mut globals);
        let (a, b, c) = legal_masks(&f.pos, f.flip(), f.gt_target()).with_context(|| format!("frame {t}: ground truth outside its legality masks"))?;
        lf.extend_from_slice(&a);
        lt.extend_from_slice(&b);
        lp.extend_from_slice(&c);
    }
    Ok((planes, globals, lf, lt, lp))
}

/// The `go` command of resolved limits.
pub fn go_command(limits: &Limits) -> Result<String> {
    let nodes = limits.u64_or("nodes", DEFAULT_NODES)?;
    let depth = match limits.get("depth") {
        Some(_) => Some(limits.u64_or("depth", 0)?),
        None => None,
    };
    let mut go = String::from("go");
    if nodes > 0 {
        go.push_str(&format!(" nodes {nodes}"));
    }
    match depth {
        Some(0) => bail!("--limit depth must be ≥ 1"),
        Some(d) => go.push_str(&format!(" depth {d}")),
        None if nodes == 0 => bail!("--limit nodes=0 needs a --limit depth (the search would never stop)"),
        None => {}
    }
    Ok(go)
}

/// The chess labeller (see the module docs). `max_moves` must equal the run's
/// `--max-moves` (it sizes the sparse heads; the CLI takes both from the same flag).
#[derive(Clone, Debug)]
pub struct ChessLabeller {
    pub max_moves: usize,
}

impl Default for ChessLabeller {
    fn default() -> Self {
        ChessLabeller::new(DEFAULT_MAX_MOVES)
    }
}

impl ChessLabeller {
    pub fn new(max_moves: usize) -> Self {
        ChessLabeller { max_moves: max_moves.max(1) }
    }
}

impl Labeller for ChessLabeller {
    type Position = ChessPos;
    type Key = String;
    type Session = UciSession;

    fn game(&self) -> &'static str {
        "chess"
    }

    fn version(&self) -> u32 {
        LABELLER_VERSION
    }

    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        let d = shard.dir.display();
        if shard.game != "chess" || shard.converter_version != CONVERTER_VERSION {
            bail!("{d}: game {:?} converter v{}, the chess labeller replays chess v{CONVERTER_VERSION} shards", shard.game, shard.converter_version);
        }
        if shard.meta()["global_features"].as_u64() != Some(GLOBAL_FEATURES as u64) {
            bail!("{d}: meta.global_features {} (expected {GLOBAL_FEATURES})", shard.meta()["global_features"]);
        }
        let want: [(&str, &[usize]); 9] = [
            ("obs.board_seq", &[14, 8, 8]),
            ("obs.globals_seq", &[GLOBAL_FEATURES]),
            ("tgt.from_idx", &[]),
            ("tgt.to_idx", &[]),
            ("tgt.promotion", &[]),
            ("tgt.value", &[]),
            ("tgt.legal_from", &[64]),
            ("tgt.legal_to_of", &[64]),
            ("tgt.legal_promotion_of", &[N_PROMOTIONS]),
        ];
        for (name, shape) in want {
            let info = shard.tensor_info(name)?;
            if info.frame_shape != shape {
                bail!("{d}: {name} has frame shape {:?}, expected {shape:?}", info.frame_shape);
            }
        }
        Ok(())
    }

    fn heads(&self) -> Vec<HeadSpec> {
        // at most one class per scored move → width K is lossless for from / to
        let k = self.max_moves;
        vec![
            HeadSpec { name: "from", kind: HeadKind::Categorical { n: 64, k: k.min(64) } },
            HeadSpec { name: "to", kind: HeadKind::Categorical { n: 64, k: k.min(64) } },
            HeadSpec { name: "promotion", kind: HeadKind::Categorical { n: N_PROMOTIONS, k: N_PROMOTIONS } },
        ]
    }

    fn raw_unit(&self) -> Option<&'static str> {
        Some("cp")
    }

    fn default_max_moves(&self) -> usize {
        self.max_moves
    }

    fn resolve_limits(&self, given: &Limits) -> Result<Limits> {
        let l = given.with_defaults(&[("nodes", &DEFAULT_NODES.to_string())], &["nodes", "depth"])?;
        go_command(&l)?;
        Ok(l)
    }

    fn open_session(&self, cfg: &EngineConfig) -> Result<UciSession> {
        let mut opts: Vec<(String, String)> = DEFAULT_OPTIONS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        opts.extend(cfg.options.iter().cloned());
        opts.push(("MultiPV".into(), cfg.max_moves.to_string()));
        opts.push(("UCI_ShowWDL".into(), "true".into()));
        UciSession::open(cfg.command.clone(), Flavor::Uci, &opts, cfg.timeout)
    }

    fn engine_id(&self, s: &UciSession) -> String {
        s.id_name.clone()
    }

    fn teacher_meta(&self, cfg: &EngineConfig) -> Value {
        json!({
            "action_id": "promotion*4096 + from*64 + to (POV squares row*8+file, row = 7-rank for white to move, rank for black; castling = king to g/c file; promotion 0 none 1 N 2 B 3 R 4 Q)",
            "q": "(W + D/2) / 1000 of each MultiPV line's final info (UCI_ShowWDL)",
            "value": "Q of the multipv 1 line",
            "stale_lines": "lines shallower than the deepest line are capped at the deepest lines' lowest Q and raw score",
            "raw": "cp; mate in n = +-(100000 - |n|)",
            "query": "ucinewgame; position startpos moves <game history>; go",
            "go": go_command(&cfg.limits).unwrap_or_default(),
            "default_engine_options": DEFAULT_OPTIONS.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect::<serde_json::Map<_, _>>(),
            "skip": "forced (one legal move)",
            "converter_version": CONVERTER_VERSION,
        })
    }

    fn replay(&self, s: &SampleView) -> Result<Vec<ChessPos>> {
        replay_moves(&s.i64s("tgt.from_idx")?, &s.i64s("tgt.to_idx")?, &s.i64s("tgt.promotion")?)
    }

    fn verify(&self, s: &SampleView, frames: &[ChessPos]) -> Result<()> {
        let (planes, globals, lf, lt, lp) = reencode(frames)?;
        s.check_f32("obs.board_seq", &planes)?;
        s.check_f32("obs.globals_seq", &globals)?;
        s.check_f32("tgt.legal_from", &lf)?;
        s.check_f32("tgt.legal_to_of", &lt)?;
        s.check_f32("tgt.legal_promotion_of", &lp)?;
        let result = s.meta()["result"].as_str().context("sample meta has no result")?;
        let wv = result_white(result).with_context(|| format!("sample meta result {result:?}"))?;
        let value: Vec<f32> = frames.iter().map(|f| if f.flip() { -wv } else { wv }).collect();
        s.check_f32("tgt.value", &value)
    }

    fn skip_reason(&self, _s: &SampleView, _t: usize, pos: &ChessPos) -> Option<&'static str> {
        (pos.legal.len() == 1).then_some("forced")
    }

    fn key(&self, pos: &ChessPos) -> String {
        pos.key.clone()
    }

    fn query(&self, session: &mut UciSession, pos: &ChessPos, cfg: &EngineConfig) -> Result<Analysis> {
        session.new_game()?;
        let r = session.search(pos.command(), &go_command(&cfg.limits)?, cfg.timeout)?;
        analysis(pos, &r.lines)
    }

    fn head_rows(&self, _s: &SampleView, _t: usize, pos: &ChessPos, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>> {
        let (gf, gt, _) = pos.gt_target();
        let dist: Vec<(Target, f32)> = policy.iter().map(|&(a, p)| (decode_action(a), p)).collect();
        Ok(vec![
            HeadRow::Categorical(marginal(&dist, |a| a.0)),
            HeadRow::Categorical(conditional(&dist, |a| a.0 == gf, |a| a.1)),
            HeadRow::Categorical(conditional(&dist, |a| a.0 == gf && a.1 == gt, |a| a.2)),
        ])
    }
}

/// Engine lines → scored moves (flat ids of the frame's legal moves, Q from WDL, raw cp).
///
/// Stale lines: a search stopped by the node limit ends inside an iteration, and
/// Stockfish then prints the moves that failed low in that iteration with their score
/// from the previous, shallower iteration (depth − 1), ranked below the lines of the
/// final depth. Such a score can be far too high (a move refuted at the final depth
/// still shows its old, winning score) and would dominate the softmax, so the Q and raw
/// score of every line shallower than the deepest one are capped at the lowest Q / raw
/// score among the deepest lines: the teacher keeps Stockfish's own ranking.
pub fn analysis(pos: &ChessPos, lines: &[super::uci::Info]) -> Result<Analysis> {
    if lines.is_empty() {
        bail!("the engine reported no scored line for {}", fen(&pos.pos));
    }
    let mut scored = Vec::with_capacity(lines.len());
    for l in lines {
        let mv = &l.pv[0];
        let u: UciMove = mv.parse().with_context(|| format!("engine move {mv:?} is not UCI"))?;
        let target = pos
            .legal
            .iter()
            .find(|(m, _)| UciMove::from_standard(*m) == u)
            .map(|(_, x)| *x)
            .with_context(|| format!("engine move {mv} is not legal in {}", fen(&pos.pos)))?;
        let [w, d, lo] = l.wdl.with_context(|| format!("line {} ({mv}) has no wdl: the engine must support UCI_ShowWDL", l.multipv))?;
        let score = l.score.context("line without score")?;
        scored.push((l.multipv, l.depth.unwrap_or(0), action_id(target), q_from_wdl(w as f32, d as f32, lo as f32), raw_score(score)));
    }
    let deepest = scored.iter().map(|s| s.1).max().unwrap_or(0);
    let final_lines = scored.iter().filter(|s| s.1 == deepest);
    let (q_cap, raw_cap) = final_lines.fold((f32::INFINITY, f32::INFINITY), |(q, r), s| (q.min(s.3), r.min(s.4)));
    let mut moves = Vec::with_capacity(scored.len());
    let mut value = None;
    for (multipv, depth, id, mut q, mut raw) in scored {
        if depth < deepest {
            q = q.min(q_cap);
            raw = raw.min(raw_cap);
        }
        if multipv == 1 {
            value = Some(q);
        }
        moves.push(ScoredMove::new(id, q).raw(raw));
    }
    Ok(Analysis { moves, value })
}

/// cp added to promotions by [`fake_engine`].
pub const FAKE_PROMOTION_BONUS: i32 = 700;

/// A chess UCI fake engine for tests: parses `position startpos [moves …]` with
/// shakmaty and scores every legal move — mate in 1 for a checkmating move (wdl
/// 1000 0 0), otherwise a deterministic pseudo-random cp ∈ [−300, 300] of (position,
/// move), + [`FAKE_PROMOTION_BONUS`] for promotions, with a logistic WDL. Unknown /
/// illegal positions crash it.
pub fn fake_engine() -> UciFake<impl FnMut(&str) -> Option<Vec<FakeMove>> + Send> {
    UciFake::new(|line: &str| {
        let pos = parse_position(line)?;
        let epd = Epd::from_position(&pos, EnPassantMode::Legal).to_string();
        Some(
            pos.legal_moves()
                .iter()
                .map(|m| {
                    let uci = m.to_uci(CastlingMode::Standard).to_string();
                    let mut after = pos.clone();
                    after.play_unchecked(*m);
                    if after.is_checkmate() {
                        return FakeMove::mate(&uci, 1).wdl([1000, 0, 0]);
                    }
                    let bonus = if m.is_promotion() { FAKE_PROMOTION_BONUS } else { 0 };
                    let cp = (fnv1a(&format!("{epd} {uci}")) % 601) as i32 - 300 + bonus;
                    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
                    let w = (1000.0 * sig((cp as f32 - 60.0) / 110.0)).round() as u32;
                    let l = (1000.0 * sig((-cp as f32 - 60.0) / 110.0)).round() as u32;
                    FakeMove::cp(&uci, cp).wdl([w, 1000 - w - l, l])
                })
                .collect(),
        )
    })
}

/// [`fake_engine`] as an in-process engine command.
pub fn fake_engine_command() -> super::engine::EngineCommand {
    super::engine::EngineCommand::in_process("chess-uci", fake_engine)
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The position of a `position startpos [moves …]` command (`None` if not one or illegal).
pub fn parse_position(line: &str) -> Option<Chess> {
    let mut toks = line.split_whitespace();
    if toks.next()? != "position" || toks.next()? != "startpos" {
        return None;
    }
    let mut pos = Chess::default();
    match toks.next() {
        None => return Some(pos),
        Some("moves") => {}
        Some(_) => return None,
    }
    for t in toks {
        let m = t.parse::<UciMove>().ok()?.to_move(&pos).ok()?;
        pos.play_unchecked(m);
    }
    Some(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::driver::{run, RunOpts};
    use crate::label::engine::EngineCommand;
    use crate::label::fake::Canned;
    use crate::label::testkit::{engine_path, overlay_tensor, quiet_opts, tmpdir};
    use crate::label::transform::softmax_q;
    use gmshard::{Column, DType};
    use shakmaty::san::San;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    /// Crafted games (validated with python-chess): A = white en passant, white
    /// under-promotion with capture, both O-O; B = both O-O-O, a knight shuffle giving a
    /// threefold repetition (frames 10, 14, 18), black en passant, black promotion with
    /// capture; C = two forced replies (frames 3, 5).
    const GAMES: [(&str, &str); 3] = [
        ("1-0", "e4 a6 e5 d5 exd6 Nf6 dxc7 Qd6 cxb8=N Rxb8 Nf3 Bg4 Be2 e6 O-O Be7 d4 O-O"),
        ("1/2-1/2", "d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O O-O-O Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 g5 Ng1 g4 h4 gxh3 Nf3 hxg2 Ne5 gxh1=Q"),
        ("0-1", "e4 f5 Qh5+ g6 Qxg6+ hxg6"),
    ];

    fn pgn(games: &[(&str, &str)]) -> String {
        let mut s = String::new();
        for (i, (res, moves)) in games.iter().enumerate() {
            s.push_str(&format!("[Event \"Rated test {i}\"]\n[White \"w{i}\"]\n[Black \"b{i}\"]\n[Result \"{res}\"]\n\n{moves} {res}\n\n"));
        }
        s
    }

    /// Convert `games` with the real converter into `root/base` (shards of `per_shard`).
    fn convert_games(root: &Path, games: &[(&str, &str)], per_shard: usize) {
        let src = root.join("games.pgn");
        std::fs::write(&src, pgn(games)).unwrap();
        let s = crate::chess::convert(&src, &root.join("base"), per_shard, 0, None, 1, false, Default::default()).unwrap();
        assert_eq!(s["kept"], games.len(), "{s}");
    }

    fn fixture_pgn() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/chess/lichess_2013-01_head.pgn")
    }

    fn shard(root: &Path, name: &str) -> ShardReader {
        ShardReader::open(&root.join(name)).unwrap()
    }

    fn san_moves(sans: &str) -> Vec<Move> {
        let mut pos = Chess::default();
        sans.split_whitespace()
            .map(|s| {
                let m = San::from_ascii(s.trim_end_matches(['+', '#']).as_bytes()).unwrap().to_move(&pos).unwrap();
                pos.play_unchecked(m);
                m
            })
            .collect()
    }

    /// Labels of a SAN game as the converter writes them.
    fn labels(sans: &str) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let mut pos = Chess::default();
        let (mut f, mut t, mut p) = (vec![], vec![], vec![]);
        for m in san_moves(sans) {
            let x = move_target(&m, pos.turn() == Color::Black).unwrap();
            f.push(x.0);
            t.push(x.1);
            p.push(x.2);
            pos.play_unchecked(m);
        }
        (f, t, p)
    }

    #[test]
    fn action_ids_round_trip() {
        for f in 0..64 {
            for t in 0..64 {
                for p in 0..5 {
                    let id = action_id((f, t, p));
                    assert!((0..5 * 4096).contains(&id));
                    assert_eq!(decode_action(id), (f, t, p));
                }
            }
        }
        assert_eq!(raw_score(Score::Cp(-35)), -35.0);
        assert_eq!((raw_score(Score::Mate(3)), raw_score(Score::Mate(-2)), raw_score(Score::Mate(0))), (99_997.0, -99_998.0, -100_000.0));
    }

    #[test]
    fn replay_crafted_games() {
        for (_, sans) in GAMES {
            let (f, t, p) = labels(sans);
            let frames = replay_moves(&f, &t, &p).unwrap();
            let moves = san_moves(sans);
            assert_eq!(frames.len(), moves.len());
            let mut pos = Chess::default();
            for (k, fr) in frames.iter().enumerate() {
                assert_eq!(fr.pos, pos, "frame {k}");
                assert_eq!(fr.legal[fr.gt].0, moves[k], "frame {k}");
                assert_eq!(fr.pos.halfmoves(), pos.halfmoves());
                // every legal move has a distinct POV target
                let uniq: std::collections::HashSet<Target> = fr.legal.iter().map(|l| l.1).collect();
                assert_eq!(uniq.len(), fr.legal.len());
                pos.play_unchecked(moves[k]);
            }
            // the command of the last frame replays to its position
            let last = frames.last().unwrap();
            assert_eq!(parse_position(last.command()).unwrap(), last.pos);
            assert_eq!(frames[0].command(), "position startpos");
        }
        // game B: castling long in UCI = king to c-file; en passant; promotion suffix
        let (f, t, p) = labels(GAMES[1].1);
        let b = replay_moves(&f, &t, &p).unwrap();
        let cmd = b.last().unwrap().command();
        assert!(cmd.starts_with("position startpos moves d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8 "), "{cmd}");
        assert!(cmd.ends_with(" h2h4 g4h3 g1f3 h3g2 f3e5"), "{cmd}");
        assert_eq!(b[27].legal[b[27].gt].0.to_uci(CastlingMode::Standard).to_string(), "g2h1q");
        assert_eq!(b[27].gt_target().2, 4);
        assert_eq!((b[10].prior, b[14].prior, b[18].prior), (0, 1, 2), "threefold repetition");
        let (f, t, p) = labels(GAMES[0].1);
        let a = replay_moves(&f, &t, &p).unwrap();
        assert_eq!(a[8].gt_target().2, 1, "cxb8=N is an under-promotion");
        assert!(matches!(a[4].legal[a[4].gt].0, Move::EnPassant { .. }));
        assert!(matches!(a[14].legal[a[14].gt].0, Move::Castle { .. }));
        let (f, t, p) = labels(GAMES[2].1);
        let c = replay_moves(&f, &t, &p).unwrap();
        let forced: Vec<usize> = c.iter().filter(|x| x.legal.len() == 1).map(|x| x.t).collect();
        assert_eq!(forced, vec![3, 5]);
        // an illegal label fails the replay
        let (mut f, t, p) = labels(GAMES[0].1);
        f[3] = 0;
        assert!(replay_moves(&f, &t, &p).unwrap_err().to_string().contains("frame 3"));
    }

    fn replay_san(sans: &str) -> Vec<ChessPos> {
        let (f, t, p) = labels(sans);
        replay_moves(&f, &t, &p).unwrap()
    }

    #[test]
    fn cache_keys_follow_the_reversible_tail() {
        // B's knight shuffle: the repeated positions have longer tails → distinct keys
        let b = replay_san(GAMES[1].1);
        assert_eq!(b[10].pos, b[14].pos);
        assert_ne!(b[10].key, b[14].key);
        assert_ne!(b[14].key, b[18].key);
        let tail_start = Epd::from_position(&b[2].pos, EnPassantMode::Legal).to_string();
        assert_eq!(b[14].key, format!("{tail_start} b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8 g1f3 g8f6 f3g1 f6g8"));
        // after a pawn move the key is the position alone
        assert_eq!(b[1].key, Epd::from_position(&b[1].pos, EnPassantMode::Legal).to_string());
        // the same tail start (move number aside) + the same moves → the same key
        let x = replay_san("e4 e5 Nf3 Nc6");
        let y = replay_san("e3 e6 e4 e5 Nf3 Nc6");
        assert_ne!(x[2].pos.fullmoves(), y[4].pos.fullmoves());
        assert_eq!((x[2].key.as_str(), x[3].key.as_str()), (y[4].key.as_str(), y[5].key.as_str()));
        assert_eq!(x[3].key, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - g1f3");
        // the same position reached through a different reversible tail → a different key
        // (the tails' positions differ, and so may Stockfish's repetition-aware answer)
        let n1 = replay_san("Nf3 Nf6 Ng1 Ng8 e4");
        let n2 = replay_san("Nc3 Nc6 Nb1 Nb8 e4");
        assert_eq!(n1[4].pos, n2[4].pos);
        assert_ne!(n1[4].key, n2[4].key);
        assert_eq!(n1[4].key, format!("{} g1f3 g8f6 f3g1 f6g8", Epd::from_position(&Chess::default(), EnPassantMode::Legal)));
        // label columns of different lengths are rejected
        let (f, t, p) = labels("e4 e5");
        assert!(replay_moves(&f, &t[..1], &p).is_err());
    }

    #[test]
    fn crafted_games_verify_and_corruption_is_caught() {
        let root = tmpdir("chess-verify");
        convert_games(&root, &GAMES, 8);
        let r = ShardReader::open(&root.join("base").join("chess_00000")).unwrap();
        let lab = ChessLabeller::default();
        lab.check_shard(&r).unwrap();
        let data = super::super::game::ShardData::load(r).unwrap();
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            let frames = lab.replay(&v).unwrap();
            lab.verify(&v, &frames).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
        }
        // corrupt the repetition flag of game B frame 14 → verification names it
        let mut tensors = data.tensors.clone();
        let g = tensors.get_mut("obs.globals_seq").unwrap();
        let start = data.reader.samples[1].start;
        let Column::F32(v) = &mut g.col else { panic!() };
        assert_eq!(v[(start + 14) * GLOBAL_FEATURES + 8], 1.0);
        v[(start + 14) * GLOBAL_FEATURES + 8] = 0.0;
        let bad = super::super::game::ShardData { reader: ShardReader::open(&root.join("base").join("chess_00000")).unwrap(), tensors };
        let v = bad.view(1);
        let err = lab.verify(&v, &lab.replay(&v).unwrap()).unwrap_err().to_string();
        assert!(err.contains("obs.globals_seq: frame 14 element 8"), "{err}");
        // a wrong result in the meta is caught through tgt.value
        let mut idx: Value = serde_json::from_slice(&std::fs::read(root.join("base/chess_00000/index.json")).unwrap()).unwrap();
        idx["samples"][0]["meta"]["result"] = json!("0-1");
        std::fs::write(root.join("base/chess_00000/index.json"), serde_json::to_vec(&idx).unwrap()).unwrap();
        let data = super::super::game::ShardData::load(ShardReader::open(&root.join("base").join("chess_00000")).unwrap()).unwrap();
        let v = data.view(0);
        assert!(lab.verify(&v, &lab.replay(&v).unwrap()).unwrap_err().to_string().contains("tgt.value"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn fixture_games_replay_and_verify() {
        let root = tmpdir("chess-fixture");
        let s = crate::chess::convert(&fixture_pgn(), &root.join("base"), 64, 0, None, 1, false, Default::default()).unwrap();
        assert!(s["kept"].as_u64().unwrap() > 100, "{s}");
        let lab = ChessLabeller::default();
        let (mut samples, mut frames_n, mut promos, mut eps, mut castles) = (0, 0, 0, 0, 0);
        for e in std::fs::read_dir(root.join("base")).unwrap() {
            let r = ShardReader::open(&e.unwrap().path()).unwrap();
            lab.check_shard(&r).unwrap();
            let data = super::super::game::ShardData::load(r).unwrap();
            for i in 0..data.reader.n_samples() {
                let v = data.view(i);
                let fr = lab.replay(&v).unwrap();
                lab.verify(&v, &fr).unwrap_or_else(|e| panic!("sample {i}: {e:#}"));
                samples += 1;
                frames_n += fr.len();
                for f in &fr {
                    match f.legal[f.gt].0 {
                        Move::EnPassant { .. } => eps += 1,
                        Move::Castle { .. } => castles += 1,
                        m if m.promotion().is_some() => promos += 1,
                        _ => {}
                    }
                }
            }
        }
        eprintln!("fixture: {samples} games, {frames_n} frames, {castles} castlings, {eps} en passant, {promos} promotions verified");
        assert!(castles > 50 && promos > 0 && eps > 0);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Real converter shards (read-only): every sample of every shard under
    /// `GM_CHESS_SHARDS` (else `${GM_DATA_SHARDS:-$HOME/gm-data/shards}/chess`) replays
    /// and verifies bit-exactly. Skipped when the directory does not exist.
    #[test]
    fn real_shards_replay_and_verify() {
        let root = std::env::var_os("GM_CHESS_SHARDS").map(PathBuf::from).or_else(|| {
            std::env::var_os("GM_DATA_SHARDS")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("gm-data").join("shards")))
                .map(|d| d.join("chess"))
        });
        let Some(root) = root.filter(|r| r.is_dir()) else {
            eprintln!("skipping: no real chess shards (set GM_CHESS_SHARDS)");
            return;
        };
        let shards = crate::label::driver::find_shards(&root, &root.join(".none")).unwrap();
        let lab = ChessLabeller::default();
        let (mut samples, mut frames_n, mut checked) = (0usize, 0usize, 0usize);
        for (dir, _) in shards.iter().take(4) {
            let Ok(r) = ShardReader::open(dir) else { continue };
            if lab.check_shard(&r).is_err() {
                eprintln!("skipping {} (not a chess v{CONVERTER_VERSION} shard)", dir.display());
                continue;
            }
            let data = super::super::game::ShardData::load(r).unwrap();
            for i in 0..data.reader.n_samples() {
                let v = data.view(i);
                let fr = lab.replay(&v).unwrap_or_else(|e| panic!("{} sample {i}: {e:#}", dir.display()));
                lab.verify(&v, &fr).unwrap_or_else(|e| panic!("{} sample {i}: {e:#}", dir.display()));
                samples += 1;
                frames_n += fr.len();
            }
            checked += 1;
        }
        eprintln!("real chess shards: {checked} shard(s), {samples} games, {frames_n} frames replayed and verified");
    }

    fn fake_opts(root: &Path, cmd: EngineCommand) -> RunOpts {
        let mut o = quiet_opts(root, cmd);
        o.limits = Limits::parse(&["nodes=1000".into()]).unwrap();
        o
    }

    #[test]
    fn fake_engine_speaks_uci() {
        let mut f = fake_engine();
        use crate::label::engine::LineHandler;
        f.handle("uci").unwrap();
        f.handle("setoption name MultiPV value 3").unwrap();
        f.handle("position startpos moves e2e4 e7e5 f1c4 b8c6 d1h5 g8f6").unwrap();
        let out = f.handle("go nodes 10").unwrap();
        assert_eq!(out.len(), 4);
        assert!(out[0].contains("score mate 1 wdl 1000 0 0") && out[0].ends_with("pv h5f7"), "{out:?}");
        assert_eq!(out[3], "bestmove h5f7");
        f.handle("position startpos moves e2e5").unwrap();
        assert!(f.handle("go").is_none(), "illegal move → crash");
        assert!(parse_position("position fen 8/8/8/8/8/8/8/8 w - - 0 1").is_none());
    }

    /// Check every teacher column of a labelled shard against a replay of the base.
    fn check_labelled(base: &ShardReader, out: &ShardReader, k: usize, tau: f32) -> (usize, usize) {
        let idx = overlay_tensor(out, "tgt.teacher_idx", DType::I64, &[k]);
        let q = overlay_tensor(out, "tgt.teacher_q", DType::F32, &[k]);
        let raw = overlay_tensor(out, "tgt.teacher_raw", DType::F32, &[k]);
        let ev = overlay_tensor(out, "tgt.engine_value", DType::F32, &[]);
        let evm = overlay_tensor(out, "tgt.engine_value_mask", DType::Bool, &[]);
        let fi = overlay_tensor(out, "tgt.from_probs_idx", DType::I64, &[k]);
        let fv = overlay_tensor(out, "tgt.from_probs_val", DType::F32, &[k]);
        let ti = overlay_tensor(out, "tgt.to_probs_idx", DType::I64, &[k]);
        let tv = overlay_tensor(out, "tgt.to_probs_val", DType::F32, &[k]);
        let pd = overlay_tensor(out, "tgt.promotion_probs", DType::F32, &[5]);
        assert!(!out.has("tgt.teacher_visits"));
        let (Column::I64(idx), Column::F32(q), Column::F32(raw), Column::F32(ev), Column::Bool(evm)) = (&idx.col, &q.col, &raw.col, &ev.col, &evm.col)
        else { panic!() };
        let (Column::I64(fi), Column::F32(fv), Column::I64(ti), Column::F32(tv), Column::F32(pd)) = (&fi.col, &fv.col, &ti.col, &tv.col, &pd.col)
        else { panic!() };
        let (gf, gt, gp) = (base.read_i64("tgt.from_idx").unwrap(), base.read_i64("tgt.to_idx").unwrap(), base.read_i64("tgt.promotion").unwrap());
        let (mut labelled, mut forced) = (0, 0);
        for sm in &base.samples {
            let r = sm.range();
            let frames = replay_moves(&gf[r.clone()], &gt[r.clone()], &gp[r.clone()]).unwrap();
            for (fr, f) in frames.iter().zip(r) {
                let row = &idx[f * k..(f + 1) * k];
                let qs = &q[f * k..(f + 1) * k];
                if fr.legal.len() == 1 {
                    forced += 1;
                    assert!(row.iter().all(|&x| x == -1) && qs.iter().all(|x| x.is_nan()) && evm[f] == 0 && ev[f] == 0.0);
                    assert!(fi[f * k..(f + 1) * k].iter().all(|&x| x == -1) && pd[f * 5..(f + 1) * 5].iter().all(|&x| x == 0.0));
                    continue;
                }
                assert_eq!(evm[f], 1, "frame {f}");
                labelled += 1;
                let n = fr.legal.len().min(k);
                let targets: Vec<Target> = fr.legal.iter().map(|l| l.1).collect();
                assert!(row[..n].iter().all(|&a| targets.contains(&decode_action(a))), "teacher moves are legal");
                assert!(row[n..].iter().all(|&a| a == -1) && qs[n..].iter().all(|x| x.is_nan()));
                assert!(qs[..n].windows(2).all(|w| w[0] >= w[1]), "sorted by Q");
                // the fake's multipv 1 line is its best move → engine value = 2·Q[0] − 1
                assert!((ev[f] - (2.0 * qs[0] - 1.0)).abs() < 1e-6);
                for j in 0..n {
                    let r = raw[f * k + j];
                    assert!((-300.0..=300.0 + FAKE_PROMOTION_BONUS as f32).contains(&r) || r == MATE_RAW - 1.0, "{r}");
                    if r == MATE_RAW - 1.0 {
                        assert_eq!(qs[j], 1.0);
                    }
                }
                let pi = softmax_q(&qs[..n], tau);
                let dist: Vec<(Target, f32)> = row[..n].iter().zip(&pi).map(|(&a, &p)| (decode_action(a), p)).collect();
                let got_from: Vec<(i64, f32)> = (0..k).filter(|&j| fi[f * k + j] >= 0).map(|j| (fi[f * k + j], fv[f * k + j])).collect();
                let want_from = marginal(&dist, |a| a.0);
                assert_eq!(got_from.len(), want_from.len());
                for (c, p) in want_from {
                    let g = got_from.iter().find(|x| x.0 == c).unwrap();
                    assert!((g.1 - p).abs() < 1e-5);
                }
                let (g0, g1, _) = fr.gt_target();
                let to_c = conditional(&dist, |a| a.0 == g0, |a| a.1);
                let to_sum: f32 = tv[f * k..(f + 1) * k].iter().sum();
                assert!(if to_c.is_empty() { to_sum == 0.0 } else { (to_sum - 1.0).abs() < 1e-5 });
                for (c, p) in &to_c {
                    let j = ti[f * k..(f + 1) * k].iter().position(|x| x == c).unwrap();
                    assert!((tv[f * k + j] - p).abs() < 1e-5);
                }
                let pr = conditional(&dist, |a| a.0 == g0 && a.1 == g1, |a| a.2);
                for c in 0..5i64 {
                    let want = pr.iter().find(|x| x.0 == c).map(|x| x.1).unwrap_or(0.0);
                    assert!((pd[f * 5 + c as usize] - want).abs() < 1e-5);
                }
            }
        }
        (labelled, forced)
    }

    #[test]
    fn end_to_end_with_fake_engine() {
        let root = tmpdir("chess-e2e");
        let mut games: Vec<(&str, &str)> = GAMES.to_vec();
        games.extend(GAMES); // repeated games → cache hits across samples
        convert_games(&root, &games, 3);
        let s = run(ChessLabeller::default(), &fake_opts(&root, fake_engine_command())).unwrap();
        assert_eq!((s.shards_total, s.shards_written, s.stats.samples), (2, 2, 6));
        assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis, s.stats.truncated_mass), (0, 0, 0, 0.0));
        assert_eq!(s.stats.skipped.get("forced").copied(), Some(4));
        assert_eq!(s.stats.labelled + 4, s.stats.frames);
        assert_eq!(s.stats.queries + s.stats.cache_hits, s.stats.labelled);
        assert!(s.stats.cache_hits > 0);
        assert_eq!(s.teacher["engine"], "FakeUci 1.0");
        assert_eq!(s.teacher["limits"], json!({"nodes": "1000"}));
        assert_eq!(s.teacher["max_moves"], 8);
        assert_eq!(s.teacher["raw_unit"], "cp");
        assert_eq!(s.teacher["heads"]["from"], json!({"kind": "categorical", "n": 64, "k": 8, "storage": "sparse"}));
        assert_eq!(s.teacher["heads"]["promotion"]["storage"], "dense");
        assert_eq!(s.teacher["game"]["go"], "go nodes 1000");
        let (mut labelled, mut forced) = (0, 0);
        for name in ["chess_00000", "chess_00001"] {
            let base = shard(&root.join("base"), name);
            let out = shard(&root.join("out"), name);
            assert_eq!(out.meta()["teacher"], s.teacher);
            for (t, info) in &base.tensors {
                assert_eq!(out.tensors[t], *info);
            }
            let (l, f) = check_labelled(&base, &out, 8, 0.05);
            labelled += l;
            forced += f;
        }
        assert_eq!((labelled as u64, forced), (s.stats.labelled, 4));
        // a non-promotion GT move that the teacher scored → promotion row [1, 0, 0, 0, 0]
        let out = shard(&root.join("out"), "chess_00000");
        let pd = out.read_f32("tgt.promotion_probs").unwrap();
        let idx = out.read_i64("tgt.teacher_idx").unwrap();
        let (gf, gt) = (out.read_i64("tgt.from_idx").unwrap(), out.read_i64("tgt.to_idx").unwrap());
        let some = (0..out.n_frames).find(|&f| idx[f * 8..f * 8 + 8].contains(&action_id((gf[f], gt[f], 0)))).unwrap();
        assert_eq!(&pd[some * 5..some * 5 + 5], &[1.0, 0.0, 0.0, 0.0, 0.0]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn promotion_head_is_conditional_on_the_ground_truth_squares() {
        // game B ends with g2xh1=Q; black also has g2g1 and g2xf1 promotions (12 in all,
        // which the fake ranks above every other move) — all inside MultiPV 16
        let root = tmpdir("chess-promo");
        convert_games(&root, &GAMES[1..2], 4);
        let mut o = fake_opts(&root, fake_engine_command());
        o.max_moves = Some(16);
        let s = run(ChessLabeller::new(16), &o).unwrap();
        assert_eq!(s.teacher["heads"]["to"]["k"], 16);
        let base = shard(&root.join("base"), "chess_00000");
        let out = shard(&root.join("out"), "chess_00000");
        check_labelled(&base, &out, 16, 0.05);
        let f = 27;
        let idx = out.read_i64("tgt.teacher_idx").unwrap();
        let promos = idx[f * 16..f * 16 + 16].iter().filter(|&&a| a >= 4096).count();
        assert_eq!(promos, 12);
        let pd = out.read_f32("tgt.promotion_probs").unwrap();
        let row = &pd[f * 5..f * 5 + 5];
        assert!(row[0] == 0.0 && row[1..].iter().all(|&x| x > 0.0) && (row.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{row:?}");
        // the to-head conditions on the GT from-square g2: destinations g1, h1, f1 only
        let (ti, tv) = (out.read_i64("tgt.to_probs_idx").unwrap(), out.read_f32("tgt.to_probs_val").unwrap());
        let mut to: Vec<i64> = (0..16).filter(|&j| tv[f * 16 + j] > 0.0).map(|j| ti[f * 16 + j]).collect();
        to.sort();
        let bsq = |file: i64, rank: i64| (rank - 1) * 8 + file; // black POV, 1-based rank
        assert_eq!(to, vec![bsq(5, 1), bsq(6, 1), bsq(7, 1)]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn deterministic_across_workers_and_cache() {
        let root = tmpdir("chess-det");
        convert_games(&root, &GAMES, 2);
        let mut o = fake_opts(&root, fake_engine_command());
        o.workers = 1;
        run(ChessLabeller::default(), &o).unwrap();
        let mut o3 = o.clone();
        o3.workers = 3;
        o3.cache_size = 0;
        o3.out = root.join("out3");
        let s3 = run(ChessLabeller::default(), &o3).unwrap();
        assert_eq!(s3.stats.cache_hits, 0);
        for name in ["chess_00000", "chess_00001"] {
            for f in std::fs::read_dir(root.join("out").join(name)).unwrap() {
                let n = f.unwrap().file_name();
                if n != "index.json" {
                    assert_eq!(std::fs::read(root.join("out").join(name).join(&n)).unwrap(), std::fs::read(root.join("out3").join(name).join(&n)).unwrap(), "{n:?}");
                }
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn narrow_max_moves_and_depth_limit() {
        let root = tmpdir("chess-k2");
        convert_games(&root, &GAMES[..1], 4);
        let mut o = fake_opts(&root, fake_engine_command());
        o.max_moves = Some(2);
        o.limits = Limits::parse(&["depth=3".into()]).unwrap();
        let s = run(ChessLabeller::new(2), &o).unwrap();
        assert_eq!(s.teacher["game"]["go"], "go nodes 120000 depth 3");
        let base = shard(&root.join("base"), "chess_00000");
        let out = shard(&root.join("out"), "chess_00000");
        check_labelled(&base, &out, 2, 0.05);
        assert!(ChessLabeller::default().resolve_limits(&Limits::parse(&["nodes=0".into()]).unwrap()).is_err());
        assert!(ChessLabeller::default().resolve_limits(&Limits::parse(&["movetime=10".into()]).unwrap()).is_err());
        assert_eq!(go_command(&Limits::parse(&["nodes=0".into(), "depth=9".into()]).unwrap()).unwrap(), "go depth 9");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn wrong_shards_are_rejected() {
        let root = tmpdir("chess-wrong");
        crate::label::toy::write_shards(&root.join("base"), 1, 2, 3).unwrap();
        let err = run(ChessLabeller::default(), &fake_opts(&root, fake_engine_command())).unwrap_err().to_string();
        assert!(err.contains("chess v3"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A session with an engine that answers every `go` with `go_reply`.
    fn canned_session(go_reply: &[&str]) -> UciSession {
        let reply: Vec<String> = go_reply.iter().map(|s| s.to_string()).collect();
        let cmd = EngineCommand::in_process("canned-sf", move || {
            let r: Vec<&str> = reply.iter().map(String::as_str).collect();
            Canned::new().rule("uci", &["id name Canned 1.0", "uciok"]).rule("isready", &["readyok"]).rule("go *", &r)
        });
        UciSession::open(cmd, Flavor::Uci, &[], Duration::from_secs(5)).unwrap()
    }

    fn cfg() -> EngineConfig {
        EngineConfig {
            command: fake_engine_command(),
            options: vec![],
            limits: Limits::parse(&["nodes=10".into()]).unwrap(),
            timeout: Duration::from_secs(5),
            max_moves: 8,
        }
    }

    #[test]
    fn engine_lines_to_analysis() {
        // white to move after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 (the GT move O-O is frame 6)
        let fr = replay_san("e4 e5 Nf3 Nc6 Bc4 Nf6 O-O");
        let white = &fr[6];
        assert_eq!(white.command(), "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 g8f6");
        let replies = [
            "info depth 5 seldepth 7 multipv 1 score mate 3 wdl 1000 0 0 nodes 10 pv f3g5 d7d5",
            "info depth 5 multipv 2 score cp 35 lowerbound wdl 100 850 50 nodes 10 pv e1g1 e8g8",
            "info depth 4 multipv 3 score mate -2 wdl 0 0 1000 nodes 10 pv d2d4",
            "bestmove f3g5",
        ];
        let lab = ChessLabeller::default();
        let mut s = canned_session(&replies);
        let a = lab.query(&mut s, white, &cfg()).unwrap();
        assert_eq!(a.value, Some(1.0), "the multipv 1 line");
        let sq = |file: i64, rank: i64| (8 - rank) * 8 + file; // white POV
        let m0 = a.moves[0];
        assert_eq!((m0.action, m0.q, m0.raw), (action_id((sq(5, 3), sq(6, 5), 0)), 1.0, 99_997.0));
        assert!(m0.visits.is_nan());
        assert_eq!(a.moves[1].action, action_id((sq(4, 1), sq(6, 1), 0)), "castling = king to g1");
        assert_eq!(a.moves[1].action, action_id(white.gt_target()));
        assert!((a.moves[1].q - 0.525).abs() < 1e-6 && a.moves[1].raw == 35.0);
        assert_eq!((a.moves[2].q, a.moves[2].raw), (0.0, -99_998.0));
        // a stale line (depth below the final one) with an old, too-high score is capped at
        // the lowest final-depth line (seen with Stockfish 19 at small node limits)
        let mut s = canned_session(&[
            "info depth 9 multipv 1 score cp 30 wdl 60 900 40 nodes 10 pv e1g1",
            "info depth 9 multipv 2 score cp 10 upperbound wdl 30 930 40 nodes 10 pv d2d3",
            "info depth 8 multipv 3 score cp 400 wdl 900 100 0 nodes 10 pv f3g5",
            "info depth 8 multipv 4 score cp -50 wdl 10 900 90 nodes 10 pv h2h3",
            "bestmove e1g1",
        ]);
        let a = lab.query(&mut s, white, &cfg()).unwrap();
        let got: Vec<(f32, f32)> = a.moves.iter().map(|m| (m.q, m.raw)).collect();
        assert_eq!(got, vec![(0.51, 30.0), (0.495, 10.0), (0.495, 10.0), (0.46, -50.0)]);
        assert_eq!(a.value, Some(0.51));
        // an illegal engine move, a missing wdl and an empty answer are errors
        for bad in [
            vec!["info depth 1 multipv 1 score cp 5 wdl 1 998 1 pv e2e5", "bestmove e2e5"],
            vec!["info depth 1 multipv 1 score cp 5 pv e1g1", "bestmove e1g1"],
            vec!["bestmove (none)"],
        ] {
            let mut s = canned_session(&bad);
            assert!(lab.query(&mut s, white, &cfg()).is_err(), "{bad:?}");
        }
        // black to move: rank mirror, e7e5 = row 6 → row 4 like white's e2e4
        let fr3 = replay_san("e4 e5");
        let mut s = canned_session(&["info depth 1 multipv 1 score cp -20 wdl 20 900 80 pv e7e5", "bestmove e7e5"]);
        let a = lab.query(&mut s, &fr3[1], &cfg()).unwrap();
        assert_eq!(a.moves[0].action, action_id((6 * 8 + 4, 4 * 8 + 4, 0)));
        assert_eq!(a.moves[0].action, action_id(fr3[1].gt_target()));
        assert_eq!(action_id(fr3[0].gt_target()), a.moves[0].action, "white's e2e4 has the same POV id");
        assert!((a.moves[0].q - 0.47).abs() < 1e-6 && a.value == Some(a.moves[0].q));
    }

    fn triples(a: &Analysis) -> Vec<(i64, f32, f32)> {
        a.moves.iter().map(|m| (m.action, m.q, m.raw)).collect()
    }

    /// The cache key's premise on the real engine (skipped when not installed): the same
    /// reversible tail reached through a different history and move number gets the
    /// identical answer, and so does a repeated query on a used session.
    #[test]
    fn real_stockfish_answer_depends_only_on_the_key() {
        let Some(sf) = engine_path("STOCKFISH", "stockfish") else {
            eprintln!("skipping: stockfish not found");
            return;
        };
        let lab = ChessLabeller::default();
        let mut c = cfg();
        c.command = EngineCommand::process(sf, &[]);
        c.limits = lab.resolve_limits(&Limits::parse(&["nodes=30000".into()]).unwrap()).unwrap();
        let mut s = lab.open_session(&c).unwrap();
        let x = replay_san("e4 e5 Nf3 Nc6 Bb5 a6");
        let y = replay_san("e3 e6 e4 e5 Nf3 Nc6 Bb5 a6");
        assert_eq!(x[5].key, y[7].key);
        assert_ne!((x[5].command(), x[5].pos.fullmoves()), (y[7].command(), y[7].pos.fullmoves()));
        let a = lab.query(&mut s, &x[5], &c).unwrap();
        let other = lab.query(&mut s, &replay_san("d4 d5 c4")[2], &c).unwrap();
        let b = lab.query(&mut s, &y[7], &c).unwrap();
        let a2 = lab.query(&mut s, &x[5], &c).unwrap();
        assert_eq!(a.moves.len(), 8);
        assert_eq!((triples(&a), a.value), (triples(&b), b.value));
        assert_eq!((triples(&a), a.value), (triples(&a2), a2.value));
        assert_ne!(triples(&a), triples(&other));
    }

    /// Real Stockfish on converted fixture games (skipped when not installed).
    #[test]
    fn real_stockfish_labels_fixture_games() {
        let Some(sf) = engine_path("STOCKFISH", "stockfish") else {
            eprintln!("skipping: stockfish not found");
            return;
        };
        let root = tmpdir("chess-sf");
        let mut games = GAMES.to_vec();
        games.push(("1-0", "e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7#"));
        convert_games(&root, &games, 4);
        let mut o = fake_opts(&root, EngineCommand::process(sf, &[]));
        o.limits = Limits::parse(&["nodes=20000".into()]).unwrap();
        let s = run(ChessLabeller::default(), &o).unwrap();
        assert_eq!((s.stats.failed_samples(), s.stats.engine_errors, s.stats.bad_analysis), (0, 0, 0), "{:?}", s.stats);
        assert!(s.teacher["engine"].as_str().unwrap().starts_with("Stockfish"));
        let base = shard(&root.join("base"), "chess_00000");
        let out = shard(&root.join("out"), "chess_00000");
        let idx = out.read_i64("tgt.teacher_idx").unwrap();
        let q = out.read_f32("tgt.teacher_q").unwrap();
        let raw = out.read_f32("tgt.teacher_raw").unwrap();
        let ev = out.read_f32("tgt.engine_value").unwrap();
        let evm = out.read_bool("tgt.engine_value_mask").unwrap();
        let (gf, gt, gp) = (base.read_i64("tgt.from_idx").unwrap(), base.read_i64("tgt.to_idx").unwrap(), base.read_i64("tgt.promotion").unwrap());
        for sm in &base.samples {
            let r = sm.range();
            let frames = replay_moves(&gf[r.clone()], &gt[r.clone()], &gp[r.clone()]).unwrap();
            for (fr, f) in frames.iter().zip(r) {
                if !evm[f] {
                    assert_eq!(fr.legal.len(), 1);
                    continue;
                }
                let n = fr.legal.len().min(8);
                assert!(idx[f * 8..f * 8 + n].iter().all(|&a| a >= 0), "MultiPV 8 fills min(8, legal) lines");
                for j in 0..n {
                    let (qq, rr) = (q[f * 8 + j], raw[f * 8 + j]);
                    assert!((0.0..=1.0).contains(&qq));
                    // WDL and cp agree in sign away from zero
                    if rr > 150.0 {
                        assert!(qq > 0.5, "cp {rr} q {qq}");
                    }
                    if rr < -150.0 {
                        assert!(qq < 0.5, "cp {rr} q {qq}");
                    }
                }
                assert!((-1.0..=1.0).contains(&ev[f]));
            }
        }
        // game C: after 1.e4 f5 white is better; after 2.Qh5+ g6 the queen must retreat —
        // 3.Qxg6+?? (h5g6) loses it
        let c = out.samples[2].start;
        assert!(ev[c + 2] > 0.1, "white to move after 1.e4 f5: {}", ev[c + 2]);
        let sq = |file: i64, rank: i64| (8 - rank) * 8 + file; // white POV
        let blunder = action_id((sq(7, 5), sq(6, 6), 0));
        assert_ne!(idx[(c + 4) * 8], blunder);
        if let Some(j) = idx[(c + 4) * 8..(c + 5) * 8].iter().position(|&a| a == blunder) {
            assert!(q[(c + 4) * 8 + j] < 0.2, "Qxg6 Q {}", q[(c + 4) * 8 + j]);
        }
        // game D's last frame: white to move with Qxf7# → mate in 1
        let d = &out.samples[3];
        let last = d.start + d.length - 1;
        assert_eq!(idx[last * 8], action_id((sq(7, 5), sq(5, 7), 0)));
        assert_eq!((q[last * 8], raw[last * 8], ev[last]), (1.0, 99_999.0, 1.0));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
