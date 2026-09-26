//! Othello teacher labels from Egaroucid for Console (GPL-3.0, run as a separate
//! process; installed by `scripts/engines/fetch_engines.sh egaroucid`).
//!
//! ```text
//! scripts/engines/fetch_engines.sh egaroucid
//! gmdata label othello --input data/othello/train --out data/othello_ega/train \
//!     --engine ~/gm-engines/bin/egaroucid --workers 64
//! #   defaults: --limit level=15 --limit exact_empties=20 --tau 0.05 --disc-scale 2.3
//! #             --max-moves 32 --engine-option hash=20 (threads=1, book=off)
//! python -m training.train_il_game --game othello --data_dir data/othello_ega/train \
//!     --soft_target_weight 1.0 --engine_value_weight 0.5 ...
//! ```
//!
//! Input: shards of `gmdata othello convert` (converter v1: obs.board_seq [T,3,8,8]
//! mover / opponent / empty bits, tgt.place_idx, tgt.passed, tgt.value; WTHOR games with
//! the forced pass frames the converter inserts). Each sample is replayed from the
//! standard start position with tgt.place_idx / tgt.passed; the planes of every frame are
//! re-encoded with the converter's own [`crate::othello::planes`] and compared bit-exactly
//! with obs.board_seq, tgt.value is recomputed from the final position, and a pass frame
//! must be a real pass (no legal move, the game not over). Failing samples are skipped
//! and counted (`--max-verify-fail-frac`, default 0.1%).
//!
//! # Engine conversation
//!
//! One Egaroucid process per worker, started as
//! `<engine> [--engine-arg …] -quiet -thread 1 -hash 20 -level L -nobook
//! -dpr 1 (60−E) L P -dpr (61−E) 60 60 100` (`-dpr m M depth prob`: search depth and
//! MPC selectivity for plies m..M, ply = discs − 3, so empties ≤ E are solved exactly and
//! every earlier position gets Egaroucid's level-L midgame search: L plies, P = 74%
//! Multi-ProbCut for L ≥ 11, full width for L ≤ 10). Per position:
//!
//! ```text
//! clearcache                     fresh transposition table: answers do not depend on history
//! setboard <64 × X/O/-><X|O>     a1 b1 … h8 (row-major, black X, white O), side to move
//! hint K                         K = min(--max-moves, legal moves): Egaroucid searches the
//!                                best move, removes it, searches the best of the rest, …
//!                                → one row per move: |level|depth@prob%|move|±discs|…|
//! gm-sync-<n>                    unknown command → "[ERROR] command `gm-sync-<n>` not
//!                                found" on stdout: the end-of-answer marker
//! ```
//!
//! so every move is scored by its own full-window search (a value, not a bound: exact
//! results at ≤ E empties; above, Egaroucid's midgame estimate, which it reports as the
//! rounded blend (0.9·v(d−1) + 1.1·v(d)) / 2 of its last two iterative-deepening depths).
//! Scores are final disc differences from
//! the side to move (`tgt.teacher_raw`, unit "discs": exact results, or the evaluation's
//! estimate of the final difference). The engine identity recorded in the teacher meta
//! is "Egaroucid for Console <version>" without the build flavour (OS / SIMD / compiler),
//! so shards labelled on different machines can be mixed (the search is the same). The
//! release tag fetch_engines.sh pins (v7.8.3, a GUI release) reports its console engine as
//! "7.8.0": build every labelling node from the same `EGAROUCID_REF`.
//!
//! Conventions checked against real data: at ≤ 7 empties every per-move score equals an
//! independent negamax (both colours to move), and the exact solve of the WTHOR position
//! with 24 empties reproduces the database's recorded perfect-play ("theoretical") black
//! score (tests/fixtures/othello/WTH_2021.wtb; empties count for the winner).
//!
//! # Labels
//!
//! * `tgt.teacher_idx` [T,K] = square `row·8 + col` (the tgt.place_idx encoding: a1 = 0,
//!   h1 = 7, a8 = 56; the letter is the column), best first (by Q, ties by the disc score,
//!   then by square — under `--disc-tau` by Q, ties by square); 64 = pass, the single
//!   entry of a pass frame.
//! * `tgt.teacher_q` [T,K] = mover-POV expected score: an exact result (the row reports
//!   depth = empties at 100%) gives win 1 / draw 0.5 / loss 0; any other score
//!   `Q = 1 / (1 + exp(−discs / s))` with s = `--disc-scale`.
//! * `tgt.teacher_raw` [T,K] = the disc score (refit Q with another s without relabelling).
//! * `tgt.engine_value` = 2·max Q − 1, `tgt.engine_value_mask`.
//! * PlacementDecoder heads (model/games/board.py `il_head_terms`): `tgt.place_probs`
//!   [T,64] dense = p(place | not pass) = the teacher policy; `tgt.pass_probs` [T] = P(pass)
//!   with `tgt.pass_probs_mask`: 0 on move frames (passing is illegal while a move exists),
//!   1 on pass frames. The decoder supervises its pass head on every valid frame, so pass
//!   frames carry this pass teacher (and no place teacher: the place head is unsupervised
//!   there); their Q / engine value is the passing side's, 1 − Q(the opponent's best
//!   reply), scored with `hint 1` on the position after the pass.
//! * Not labelled (pad rows, masks false): frames with a single legal move ("forced", the
//!   global rule for forced moves), engine failures, failed samples, `--every-nth-frame`
//!   / `--max-samples` sampling.
//!
//! Cache key: (mover discs, opponent discs) bitboards — Egaroucid is colour-blind (it
//! searches player / opponent boards), so this is the full identity of the answer.
//!
//! # Defaults and why
//!
//! * `--policy softmax --tau 0.05 --tie-weight 2`: π ∝ exp(Q/τ + w·u), u = tanh(discs/16)
//!   of `tgt.teacher_raw` ([`secondary_utility`]; exact solves and midgame estimates
//!   alike), as for the other games. Q is the unit shared across games and the value
//!   target, and it plays for the win: a move that keeps a win beats any that does not.
//!   But Q saturates in decided positions — exact solves give Q ∈ {0, ½, 1}, midgame
//!   estimates Q > 0.97 from |discs| ≳ 8 — where `exp(Q/τ)` alone is uniform over the
//!   moves: on two real 2000+ WTHOR shards labelled at the defaults (5.0k move frames)
//!   11–12% of the move frames are exactly lost (5–8% exactly won), their moves 15–17
//!   discs apart on average (up to 48), and the plain teacher put the same mass on the
//!   smallest and the largest loss (and on every winning move alike). The tie-break
//!   orders those by margin: within one Q class the weight ratio is e^(w·Δu), at most
//!   e^(2w) ≈ 55 (a 4-disc loss vs a 20-disc loss: u = −0.24 vs −0.85, 3.3×); on those
//!   shards the smallest loss's mean probability in exactly lost frames rises from
//!   0.26–0.29 to 0.33–0.35. The scale 16 keeps the margin informative across the
//!   decided range (u = 0.24 / 0.46 / 0.73 / 0.93 at 4 / 8 / 15 / 26 discs — the 10 / 25 /
//!   50 / 75th percentiles of |best score| over real frames with Q ≥ 0.97 — and 0.995 at
//!   48). Near Q = 0.5 the disc
//!   temperature is 4·s·τ ≈ 0.46 discs (a move one disc worse gets e^−2.2 ≈ 0.11 of the
//!   weight; the tie-break adds 6% to that slope, w/16 = 0.125 vs 2.2 per disc); mean
//!   teacher entropy on WTHOR frames is 1.2 nats. `--tie-weight 0` gives π ∝ exp(Q/τ)
//!   exactly (the labeller-v1 teacher). Softer: a larger `--tau`, or `--disc-tau T`:
//!   π ∝ exp(discs / T) in every position — margin-maximising, like the WTHOR
//!   "theoretical score" — recorded as policy `{"kind": "softmax_discs", "disc_tau": T}`
//!   (it replaces the softmax, so it takes no `--tau` / `--tie-weight`); Q and
//!   engine_value are the same either way.
//! * `--disc-scale 2.3`: fitted with `--fit-disc-scale 1000` at the defaults (level 15,
//!   exact ≤ 20 empties) on 1000 distinct positions with 21–60 empties from two
//!   2000+ WTHOR shards: the maximum-likelihood s of P(the side to move wins the
//!   Egaroucid self-play game from here) = σ(discs / s) is 2.29 (mean log-likelihood
//!   −0.281, Brier 0.072, 7% draws; per phase 1.9 at 21–30 empties, 2.1 at 31–40, 3.0 at
//!   41–50). Calibration, discs → observed / model win rate: −4…−3 0.12 / 0.18,
//!   −2…−1 0.25 / 0.34, 0 0.47 / 0.50, 1…2 0.61 / 0.65, 3…4 0.79 / 0.81, 5…6 0.92 / 0.91,
//!   7…10 0.97 / 0.97. Egaroucid exposes no win-rate model of its own; this is how
//!   Stockfish's WDL model is built (self-play outcomes vs eval). Human players convert
//!   advantages less reliably: the same fit against the WTHOR outcomes of 64 games gives
//!   s ≈ 5.7 (`--disc-scale 5.7` for Q = the expected score of a human game). Refit when
//!   you change `level` (teacher_raw lets you refit Q later without relabelling).
//! * `--limit level=15`: measured on 180 WTHOR positions stratified by empties (Apple
//!   M4 Pro, Egaroucid 7.8 generic ARM build, 1 thread, per CPU-second incl. process start;
//!   the machine was shared, so read ratios, not absolutes):
//!
//!   ```text
//!   level / exact  pos per CPU-s   best move agrees with Egaroucid level 21   mean loss (discs)
//!   11 / 20        9.0             0.87                                        0.29
//!   13 / 20        7.1             0.88                                        0.21
//!   15 / 20        4.3             0.90                                        0.22
//!   17 / 20        2.6             0.92                                        0.17
//!   19 / 20        1.2             0.92                                        0.14
//!   21 / 18        0.6             0.96                                        0.10
//!   (level 21 as Egaroucid plays it: exact ≤ 24, selective endgame ≤ 30 empties: ~0.3 pos/s)
//!   ```
//!
//!   (126 positions above 18 empties, so the agreement columns are ±0.03.) Level 15 is the
//!   compromise: about twice the cost of level 11 and slightly closer to level 21, a
//!   quarter of the cost of level 19, and it keeps a full WTHOR relabel within a few
//!   socket-hours. The WTHOR players being imitated play a maximum-probability teacher
//!   move on 71% of the labelled frames (64 games from 2000 on, default settings).
//! * `--limit exact_empties=20`: solving all moves exactly costs ~0.1 s per position at
//!   16–18 empties and ~0.45 s at 19–20 (Mac, generic build), about as much as a level-15
//!   midgame position; `exact_empties=22` costs ~15% more overall (~1.3 s per position at
//!   21–22 empties). Exact endgame labels are where human games have most errors.
//! * `--max-moves 32`: every legal move is scored (positions with more than 32 legal moves
//!   are unknown in practice); `hint 8` would save only ~12%.
//! * `--engine-option hash=20` (32 MB transposition table; 16–23 measured within 5%),
//!   `threads=1`, `book=off` (the bundled opening book would replace searched scores by
//!   book values): one single-threaded process per worker. Other options: `eval=FILE`
//!   (→ `-eval`). A process built by fetch_engines.sh is ~0.8 GB resident on macOS
//!   (~0.55 GB live evaluation tables, the rest freed load buffers; + the hash) — the
//!   script moves the transposition table from a fixed 2^25-entry static array to the
//!   heap (1.5 GB per process otherwise), so 64 workers need ~35–50 GB.
//! * `clearcache` per position costs ~5% throughput and makes every answer independent of
//!   the worker's history. Reproducible across processes needs the fetch_engines.sh build:
//!   a stock Egaroucid draws its transposition-table hash codes from an RNG seeded with
//!   the start-up time whenever no code file ships for the `-hash` level (only 25–27 do),
//!   so each process searched a little differently (two runs of 8 WTHOR games at the
//!   defaults: 7 of 472 positions got a move scored 1–3 discs apart). The script seeds
//!   that RNG with a constant, and its smoke test and
//!   `real_egaroucid_is_deterministic_across_processes` compare two processes' searches
//!   node for node.
//!
//! # Expected cluster throughput
//!
//! Measured on the Mac above: ~4.3 positions per CPU-second per worker at the defaults
//! (one full hint per position). Of the 2.10M frames of the 68 local 2000+ WTHOR shards
//! 3.9% are skipped as forced and 1.6% are pass frames (one cheaper `hint 1` each); ~7%
//! of the positions hit the per-worker cache (repeated openings). Not measured on the
//! cluster; estimate for an AMD EPYC 9355 node with the
//! AVX-512 build: Egaroucid's docs put its SIMD build at ~1.5× the generic one, a Zen 5
//! core at server clocks is somewhat slower per thread than an M4 P-core, and two SMT
//! workers per core share it, so ~5–6.5 positions/s per physical core, i.e. ~160–210
//! positions/s per 32-core socket with 64 workers: the 2.1M frames of the 2000+ WTHOR
//! set (~1.9M engine queries) in ~2.5–3.3 socket-hours. Measure first with
//! `--max-shards 1` (the progress line reports positions/s and searches/s). Real runs on
//! the Mac at the defaults, the machine shared with other jobs: 64 games with 8 workers
//! (oversubscribed ~8×) 13 positions/s; 24 games of a WTH_2021 shard with 4 workers (load
//! ~20 on 12 cores) 14.5 positions/s, 1409 frames labelled, 3.8 engine queries per engine
//! CPU-second, 0 verification failures, 0 engine errors. The x86 builds (AVX-512 + AMD,
//! AVX2, generic) compile with clang; the g++ 13 AVX-512 build itself is unverified here.
//!
//! # `--fit-disc-scale N` (calibration; writes no shards)
//!
//! Samples N distinct positions with more than `exact_empties` empties and ≥ 2 legal moves
//! from the input shards (deterministic hash order), scores each with `hint 1`, lets
//! Egaroucid play both sides from there with the same search (one `hint 1` per move) and
//! fits s by maximum likelihood of the outcome (win 1 / draw 0.5 / loss 0 for the
//! position's mover) under σ(discs / s). Prints and writes `<out>/disc_scale_fit.json`.
//!
//! The fake engine [`FakeEgaroucid`] speaks the same console subset (tests; the hidden
//! `gmdata label fake-engine othello` serves it on stdin/stdout).

use super::driver::{find_shards, RunOpts};
use super::engine::{Engine, EngineCommand, LineHandler};
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove, ShardData};
use super::LabelArgs;
use crate::othello::{self as rules, BLACK, WHITE};
use anyhow::{bail, Context, Result};
use gmshard::ShardReader;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const GAME: &str = "othello";
/// The `gmdata othello convert` layout this labeller replays.
pub const CONVERTER_VERSION: u32 = 1;
/// Bump when the meaning of a label changes.
/// 2: the softmax teacher's secondary tie-break ([`secondary_utility`]).
pub const LABELLER_VERSION: u32 = 2;
pub const DEFAULT_LEVEL: u64 = 15;
pub const DEFAULT_EXACT_EMPTIES: u64 = 20;
pub const DEFAULT_HASH: u32 = 20;
pub const DEFAULT_MAX_MOVES: usize = 32;
/// Logistic scale (discs) of Q for non-exact scores (see the module docs).
pub const DISC_SCALE: f64 = 2.3;
/// The pass action id (`tgt.teacher_idx` of pass frames).
pub const PASS_ID: i64 = 64;
/// Disc scale of the secondary utility `tanh(discs / TIE_DISC_SCALE)` (see the module docs).
pub const TIE_DISC_SCALE: f32 = 16.0;

/// Secondary utility u = tanh(discs / 16) ∈ (−1, 1) of a `teacher_raw` disc score (the
/// softmax tie-break, see the module docs); `None` for a non-finite raw.
pub fn secondary_utility(discs: f32) -> Option<f32> {
    discs.is_finite().then(|| ((discs as f64) / TIE_DISC_SCALE as f64).tanh() as f32)
}

/// `gmdata label othello` flags.
#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
    /// logistic scale s (discs) of Q = 1/(1+exp(−discs/s)) for scores that are not exact
    #[arg(long, default_value_t = DISC_SCALE)]
    pub disc_scale: f64,
    /// teacher policy ∝ exp(discs / T) over the raw disc scores instead of
    /// exp(Q / --tau + --tie-weight · u)
    #[arg(long, conflicts_with_all = ["tau", "tie_weight"])]
    pub disc_tau: Option<f64>,
    /// calibration mode: fit --disc-scale from N self-play playouts of input positions
    /// (writes <out>/disc_scale_fit.json, no shards)
    #[arg(long)]
    pub fit_disc_scale: Option<usize>,
}

pub fn run(args: Args) -> Result<()> {
    let lab = OthelloLabeller::new(args.disc_scale, args.disc_tau)?;
    if args.common.policy == Some(super::transform::PolicyKind::Visits) {
        bail!("--policy visits: Egaroucid reports no visits (use the default softmax or --disc-tau)");
    }
    if let Some(n) = args.fit_disc_scale {
        let report = fit_disc_scale(&lab, &args.common.to_opts()?, n)?;
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }
    super::run_labeller(lab, &args.common)
}

// ------------------------------------------------------------------ positions

/// A replayed frame: the board before the frame's action, the side to move, its legal
/// placements (bit `row·8 + col`; 0 = pass frame).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OthPos {
    /// 0 empty, [`BLACK`], [`WHITE`] (the converter's encoding), index `row·8 + col`
    pub board: [u8; 64],
    pub mover: u8,
    pub legal: u64,
}

/// Legal placements of `player` as a bitmask (the converter's flip rule).
pub fn legal_mask(b: &[u8; 64], player: u8) -> u64 {
    (0..64u32).filter(|&i| !rules::flips(b, player, (i / 8) as i32, (i % 8) as i32).is_empty()).fold(0, |m, i| m | 1 << i)
}

/// `b` after `player` places on `sq` (must be legal).
pub fn play(b: &[u8; 64], player: u8, sq: usize) -> [u8; 64] {
    let mut n = *b;
    let fl = rules::flips(b, player, (sq / 8) as i32, (sq % 8) as i32);
    n[sq] = player;
    for i in fl {
        n[i] = player;
    }
    n
}

fn bits(b: &[u8; 64], who: u8) -> u64 {
    b.iter().enumerate().filter(|(_, &v)| v == who).fold(0, |m, (i, _)| m | 1 << i)
}

/// Square name ("f5" = row 4, column 5 = id 37).
pub fn square_name(sq: i64) -> String {
    if !(0..64).contains(&sq) {
        return format!("#{sq}");
    }
    format!("{}{}", (b'a' + (sq % 8) as u8) as char, sq / 8 + 1)
}

/// Square id of a name like "f5" / "F5".
pub fn parse_square(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let col = b[0].to_ascii_lowercase().checked_sub(b'a')? as i64;
    let row = b[1].checked_sub(b'1')? as i64;
    ((0..8).contains(&col) && (0..8).contains(&row)).then_some(row * 8 + col)
}

impl OthPos {
    pub fn new(board: [u8; 64], mover: u8) -> Self {
        let legal = legal_mask(&board, mover);
        OthPos { board, mover, legal }
    }

    pub fn start() -> Self {
        OthPos::new(rules::initial_board(), BLACK)
    }

    pub fn mine(&self) -> u64 {
        bits(&self.board, self.mover)
    }

    pub fn theirs(&self) -> u64 {
        bits(&self.board, rules::opponent(self.mover))
    }

    pub fn empties(&self) -> u32 {
        self.board.iter().filter(|&&v| v == 0).count() as u32
    }

    pub fn n_legal(&self) -> u32 {
        self.legal.count_ones()
    }

    pub fn is_legal(&self, sq: i64) -> bool {
        (0..64).contains(&sq) && self.legal >> sq & 1 == 1
    }

    /// The position after `sq`, with the next side to move (the opponent, or the
    /// mover again when the opponent must pass); `None` when the game is over.
    pub fn after(&self, sq: i64) -> Option<OthPos> {
        let b = play(&self.board, self.mover, sq as usize);
        let opp = OthPos::new(b, rules::opponent(self.mover));
        if opp.legal != 0 {
            return Some(opp);
        }
        let me = OthPos::new(b, self.mover);
        (me.legal != 0).then_some(me)
    }

    /// Egaroucid `setboard` argument: 64 cells a1 … h8 (X black, O white, - empty) + side.
    pub fn to_egaroucid(&self) -> String {
        let mut s: String = self.board.iter().map(|&v| match v { BLACK => 'X', WHITE => 'O', _ => '-' }).collect();
        s.push(if self.mover == BLACK { 'X' } else { 'O' });
        s
    }

    /// Inverse of [`OthPos::to_egaroucid`] (also accepts B/W/*/0/1 and `.`, like Egaroucid).
    pub fn from_egaroucid(s: &str) -> Option<OthPos> {
        let c: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
        if c.len() != 65 {
            return None;
        }
        let colour = |ch: char| match ch {
            'X' | 'x' | 'B' | 'b' | '*' | '0' => Some(BLACK),
            'O' | 'o' | 'W' | 'w' | '1' => Some(WHITE),
            '-' | '.' => Some(0),
            _ => None,
        };
        let mut board = [0u8; 64];
        for (i, &ch) in c[..64].iter().enumerate() {
            board[i] = colour(ch)?;
        }
        let mover = colour(c[64]).filter(|&m| m != 0)?;
        Some(OthPos::new(board, mover))
    }

    /// Final disc difference for the mover of a finished game (empties to the winner,
    /// the WTHOR / Egaroucid convention).
    pub fn final_diff(&self) -> i32 {
        let (me, opp) = (self.mine().count_ones() as i32, self.theirs().count_ones() as i32);
        let e = 64 - me - opp;
        match me.cmp(&opp) {
            std::cmp::Ordering::Greater => me - opp + e,
            std::cmp::Ordering::Less => me - opp - e,
            std::cmp::Ordering::Equal => 0,
        }
    }
}

/// Expected score of a disc result for the mover: exact → 1 / 0.5 / 0, else σ(discs / s).
pub fn q_from_discs(discs: i32, exact: bool, scale: f64) -> f32 {
    if exact {
        return match discs.signum() {
            1 => 1.0,
            0 => 0.5,
            _ => 0.0,
        };
    }
    (1.0 / (1.0 + (-(discs as f64) / scale).exp())) as f32
}

// ------------------------------------------------------------------ Egaroucid session

/// Effective `--engine-option`s.
#[derive(Debug, Clone, PartialEq)]
pub struct EgaOptions {
    pub hash: u32,
    pub threads: u32,
    pub book: bool,
    pub eval: Option<String>,
}

impl EgaOptions {
    pub fn parse(opts: &[(String, String)]) -> Result<Self> {
        let mut o = EgaOptions { hash: DEFAULT_HASH, threads: 1, book: false, eval: None };
        for (k, v) in opts {
            let num = || v.parse::<u32>().with_context(|| format!("--engine-option {k}={v}: not an integer"));
            match k.to_ascii_lowercase().as_str() {
                "hash" => o.hash = num()?,
                "threads" | "thread" => o.threads = num()?.max(1),
                "book" => {
                    o.book = match v.to_ascii_lowercase().as_str() {
                        "on" | "true" | "1" | "yes" => true,
                        "off" | "false" | "0" | "no" => false,
                        _ => bail!("--engine-option book={v}: use on / off"),
                    }
                }
                "eval" => o.eval = Some(v.clone()),
                _ => bail!("unknown --engine-option {k:?} for Egaroucid (hash, threads, book, eval)"),
            }
        }
        if o.hash > 30 {
            bail!("--engine-option hash={}: Egaroucid hash levels are 0–30 (2^level entries)", o.hash);
        }
        Ok(o)
    }
}

/// Egaroucid command-line flags for the resolved limits and options (see the module docs).
pub fn engine_flags(limits: &Limits, opts: &EgaOptions) -> Result<Vec<String>> {
    let level = limits.u64_or("level", DEFAULT_LEVEL)?;
    let exact = limits.u64_or("exact_empties", DEFAULT_EXACT_EMPTIES)?;
    let mut f: Vec<String> = ["-quiet", "-thread", &opts.threads.to_string(), "-hash", &opts.hash.to_string(), "-level", &level.to_string()]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if !opts.book {
        f.push("-nobook".into());
    }
    if let Some(e) = &opts.eval {
        f.extend(["-eval".to_string(), e.clone()]);
    }
    let prob = if level <= 10 { "100" } else { "74" };
    // ply m = 61 − empties: exact for empties ≤ E ⇔ plies 61−E … 60
    let first_exact = 61 - exact.min(60);
    if first_exact > 1 {
        f.extend(["-dpr", "1", &(first_exact - 1).to_string(), &level.to_string(), prob].iter().map(|s| s.to_string()));
    }
    if exact >= 1 {
        f.extend(["-dpr", &first_exact.to_string(), "60", "60", "100"].iter().map(|s| s.to_string()));
    }
    Ok(f)
}

/// One `hint` row.
#[derive(Debug, Clone, PartialEq)]
pub struct HintRow {
    pub square: i64,
    /// final disc difference for the side to move (exact, or estimated)
    pub discs: i32,
    /// search depth and MPC probability (%); `None` = a book value
    pub depth: Option<(u32, f32)>,
}

impl HintRow {
    /// An exact result: an endgame search to the last empty at 100%.
    pub fn exact(&self, empties: u32) -> bool {
        matches!(self.depth, Some((d, p)) if p >= 100.0 && d >= empties)
    }
}

/// Parse `|level|depth@prob%|move|±score|time|nodes|nps|` (`None` for the header row).
pub fn parse_hint_row(line: &str) -> Result<Option<HintRow>> {
    let f: Vec<&str> = line.trim().trim_matches('|').split('|').map(str::trim).collect();
    if f.len() < 4 {
        bail!("unexpected hint line {line:?}");
    }
    if f[0] == "Level" {
        return Ok(None);
    }
    let square = parse_square(f[2]).with_context(|| format!("bad move {:?} in hint line {line:?}", f[2]))?;
    let discs: i32 = f[3].trim_start_matches('+').parse().with_context(|| format!("bad score {:?} in hint line {line:?}", f[3]))?;
    let depth = if f[1] == "-" {
        None
    } else {
        let (d, p) = f[1].trim_end_matches('%').split_once('@').with_context(|| format!("bad depth {:?} in hint line {line:?}", f[1]))?;
        Some((d.parse().with_context(|| format!("bad depth {d:?}"))?, p.parse().with_context(|| format!("bad probability {p:?}"))?))
    };
    Ok(Some(HintRow { square, discs, depth }))
}

/// A live Egaroucid console process.
pub struct EgaSession {
    engine: Engine,
    /// "Egaroucid for Console <version>" (teacher meta)
    pub id: String,
    /// the full version line incl. the build flavour (logged)
    pub build: String,
    sync: u64,
}

impl EgaSession {
    pub fn open(cmd: EngineCommand, timeout: Duration) -> Result<Self> {
        let engine = Engine::spawn(cmd)?;
        let mut s = EgaSession { engine, id: String::new(), build: String::new(), sync: 0 };
        let lines = s.exchange(&["version".to_string()], timeout).context("Egaroucid handshake (version)")?;
        let v = lines.iter().find(|l| l.starts_with("Egaroucid")).with_context(|| format!("no version line from the engine: {lines:?}"))?;
        s.build = v.trim().to_string();
        s.id = v.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
        Ok(s)
    }

    /// Send `cmds` and a sync marker; the output lines before the marker's echo.
    pub fn exchange(&mut self, cmds: &[String], timeout: Duration) -> Result<Vec<String>> {
        self.sync += 1;
        let marker = format!("gm-sync-{}", self.sync);
        for c in cmds {
            self.engine.send(c)?;
        }
        self.engine.send(&marker)?;
        let echo = format!("[ERROR] command `{marker}` not found");
        let mut lines = self.engine.recv_until(timeout, |l| l.trim() == echo)?;
        lines.pop();
        Ok(lines)
    }

    /// The engine's top `k` moves of `pos` (each scored by its own search), best first.
    pub fn hint(&mut self, pos: &OthPos, k: usize, timeout: Duration) -> Result<Vec<HintRow>> {
        let k = k.min(pos.n_legal() as usize);
        if k == 0 {
            bail!("no legal move to score");
        }
        let cmds = ["clearcache".to_string(), format!("setboard {}", pos.to_egaroucid()), format!("hint {k}")];
        let lines = self.exchange(&cmds, timeout)?;
        let mut rows = Vec::with_capacity(k);
        for l in &lines {
            if l.trim().is_empty() {
                continue;
            }
            if !l.trim_start().starts_with('|') {
                bail!("unexpected engine output {l:?}");
            }
            if let Some(r) = parse_hint_row(l)? {
                if !pos.is_legal(r.square) {
                    bail!("engine scored {} which is illegal in {}", square_name(r.square), pos.to_egaroucid());
                }
                if rows.iter().any(|x: &HintRow| x.square == r.square) {
                    bail!("engine scored {} twice", square_name(r.square));
                }
                rows.push(r);
            }
        }
        if rows.len() != k {
            bail!("engine scored {} moves, asked for {k}: {lines:?}", rows.len());
        }
        Ok(rows)
    }
}

// ------------------------------------------------------------------ labeller

/// The othello labeller (see the module docs).
#[derive(Debug, Clone)]
pub struct OthelloLabeller {
    pub disc_scale: f64,
    pub disc_tau: Option<f64>,
}

impl Default for OthelloLabeller {
    fn default() -> Self {
        OthelloLabeller { disc_scale: DISC_SCALE, disc_tau: None }
    }
}

impl OthelloLabeller {
    pub fn new(disc_scale: f64, disc_tau: Option<f64>) -> Result<Self> {
        if !(disc_scale.is_finite() && disc_scale > 0.0) {
            bail!("--disc-scale must be > 0");
        }
        if let Some(t) = disc_tau {
            if !(t.is_finite() && t > 0.0) {
                bail!("--disc-tau must be > 0");
            }
        }
        Ok(OthelloLabeller { disc_scale, disc_tau })
    }

    /// The engine command with the Egaroucid flags appended (processes; an in-process
    /// fake takes it as is). A raw Egaroucid binary next to its `resources/` directory
    /// is started in that directory (the macOS build looks its files up relative to the
    /// working directory; the fetch_engines.sh wrapper does the same).
    pub fn command(&self, cfg: &EngineConfig) -> Result<EngineCommand> {
        let flags = engine_flags(&cfg.limits, &EgaOptions::parse(&cfg.options)?)?;
        Ok(match cfg.command.clone() {
            EngineCommand::Process { program, mut args, cwd, env } => {
                args.extend(flags);
                let cwd = cwd.or_else(|| program.parent().filter(|d| d.join("resources").is_dir()).map(PathBuf::from));
                EngineCommand::Process { program, args, cwd, env }
            }
            c => c,
        })
    }

    fn analysis(&self, pos: &OthPos, rows: &[HintRow]) -> Analysis {
        let e = pos.empties();
        let moves = rows
            .iter()
            .map(|r| ScoredMove::new(r.square, q_from_discs(r.discs, r.exact(e), self.disc_scale)).raw(r.discs as f32))
            .collect();
        Analysis { moves, value: None }
    }
}

impl Labeller for OthelloLabeller {
    type Position = OthPos;
    type Key = (u64, u64);
    type Session = EgaSession;

    fn game(&self) -> &'static str {
        GAME
    }

    fn version(&self) -> u32 {
        LABELLER_VERSION
    }

    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        if shard.game != GAME || shard.converter_version != CONVERTER_VERSION {
            bail!("{}: game {:?} v{}, expected an othello v{CONVERTER_VERSION} shard (gmdata othello convert)", shard.dir.display(),
                  shard.game, shard.converter_version);
        }
        let board = shard.tensor_info("obs.board_seq")?;
        if board.frame_shape != [3, 8, 8] {
            bail!("{}: obs.board_seq frame shape {:?}, expected [3, 8, 8]", shard.dir.display(), board.frame_shape);
        }
        for t in ["tgt.place_idx", "tgt.passed", "tgt.value"] {
            shard.tensor_info(t)?;
        }
        Ok(())
    }

    fn heads(&self) -> Vec<HeadSpec> {
        vec![
            HeadSpec { name: "place", kind: HeadKind::Categorical { n: 64, k: 64 } },
            HeadSpec { name: "pass", kind: HeadKind::Bernoulli },
        ]
    }

    fn raw_unit(&self) -> Option<&'static str> {
        Some("discs")
    }

    fn secondary(&self, m: &ScoredMove) -> Option<f32> {
        secondary_utility(m.raw)
    }

    fn secondary_meta(&self) -> Option<Value> {
        Some(json!({"of": "teacher_raw", "discs": "u = tanh(raw / 16)"}))
    }

    fn default_max_moves(&self) -> usize {
        DEFAULT_MAX_MOVES
    }

    fn resolve_limits(&self, given: &Limits) -> Result<Limits> {
        let l = given.with_defaults(
            &[("level", &DEFAULT_LEVEL.to_string()), ("exact_empties", &DEFAULT_EXACT_EMPTIES.to_string())],
            &["level", "exact_empties"],
        )?;
        let level = l.u64_or("level", DEFAULT_LEVEL)?;
        if !(1..=60).contains(&level) {
            bail!("--limit level={level}: Egaroucid levels are 1–60");
        }
        if l.u64_or("exact_empties", DEFAULT_EXACT_EMPTIES)? > 60 {
            bail!("--limit exact_empties must be ≤ 60");
        }
        Ok(l)
    }

    fn custom_policy(&self, moves: &[ScoredMove]) -> Option<Vec<f32>> {
        let t = self.disc_tau?;
        let best = moves.iter().map(|m| m.raw).fold(f32::NEG_INFINITY, f32::max);
        Some(moves.iter().map(|m| (((m.raw - best) as f64) / t).exp() as f32).collect())
    }

    fn custom_policy_meta(&self) -> Option<Value> {
        self.disc_tau.map(|t| json!({"kind": "softmax_discs", "disc_tau": t}))
    }

    fn open_session(&self, cfg: &EngineConfig) -> Result<EgaSession> {
        EgaSession::open(self.command(cfg)?, cfg.timeout)
    }

    fn engine_id(&self, s: &EgaSession) -> String {
        eprintln!("[label othello] engine build: {}", s.build);
        s.id.clone()
    }

    fn teacher_meta(&self, cfg: &EngineConfig) -> Value {
        let o = EgaOptions::parse(&cfg.options).ok();
        json!({
            "action_id": "row*8+col (a1=0, h1=7, a8=56; tgt.place_idx encoding); 64 = pass (pass frames)",
            "q": "exact (depth = empties @100%): win 1 / draw 0.5 / loss 0; else 1/(1+exp(-discs/disc_scale))",
            "disc_scale": self.disc_scale,
            "search": "egaroucid hint per position after clearcache; -dpr: level-L midgame above exact_empties, exact solve at or below",
            "hash": o.as_ref().map(|o| o.hash),
            "threads": o.as_ref().map(|o| o.threads),
            "book": o.as_ref().map(|o| o.book),
            "eval": o.as_ref().and_then(|o| o.eval.as_ref()).map(|e| std::path::Path::new(e).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()),
        })
    }

    fn replay(&self, s: &SampleView) -> Result<Vec<OthPos>> {
        let (place, passed) = (s.i64s("tgt.place_idx")?, s.bools("tgt.passed")?);
        let mut pos = OthPos::start();
        let mut out = Vec::with_capacity(s.len);
        for t in 0..s.len {
            let next_board = if passed[t] {
                if pos.legal != 0 {
                    bail!("frame {t}: pass recorded with {} legal moves", pos.n_legal());
                }
                if legal_mask(&pos.board, rules::opponent(pos.mover)) == 0 {
                    bail!("frame {t}: pass recorded after the game ended");
                }
                pos.board
            } else {
                if !pos.is_legal(place[t]) {
                    bail!("frame {t}: recorded move {} is illegal", square_name(place[t]));
                }
                play(&pos.board, pos.mover, place[t] as usize)
            };
            let next = OthPos::new(next_board, rules::opponent(pos.mover));
            out.push(std::mem::replace(&mut pos, next));
        }
        if pos.legal != 0 || legal_mask(&pos.board, rules::opponent(pos.mover)) != 0 {
            bail!("the game is not over after its last frame");
        }
        Ok(out)
    }

    fn verify(&self, s: &SampleView, positions: &[OthPos]) -> Result<()> {
        let mut planes = Vec::with_capacity(positions.len() * 192);
        for p in positions {
            rules::planes(&p.board, p.mover, &mut planes);
        }
        s.check_f32("obs.board_seq", &planes)?;
        let (place, passed) = (s.i64s("tgt.place_idx")?, s.bools("tgt.passed")?);
        if let Some(t) = (0..s.len).find(|&t| passed[t] && place[t] != 0) {
            bail!("tgt.place_idx: pass frame {t} holds {} (the converter writes 0)", place[t]);
        }
        let last = positions.last().context("empty sample")?;
        let fin = if passed[s.len - 1] { last.board } else { play(&last.board, last.mover, place[s.len - 1] as usize) };
        let (b, w) = (bits(&fin, BLACK).count_ones(), bits(&fin, WHITE).count_ones());
        let sign = (b > w) as i32 as f32 - (b < w) as i32 as f32;
        let value: Vec<f32> = positions.iter().map(|p| if p.mover == BLACK { sign } else { -sign }).collect();
        s.check_f32("tgt.value", &value)
    }

    fn skip_reason(&self, _s: &SampleView, _t: usize, pos: &OthPos) -> Option<&'static str> {
        (pos.n_legal() == 1).then_some("forced")
    }

    fn key(&self, pos: &OthPos) -> (u64, u64) {
        (pos.mine(), pos.theirs())
    }

    fn query(&self, session: &mut EgaSession, pos: &OthPos, cfg: &EngineConfig) -> Result<Analysis> {
        if pos.legal == 0 {
            // a pass frame: the pass is scored by the opponent's best reply after it
            let after = OthPos::new(pos.board, rules::opponent(pos.mover));
            let best = session.hint(&after, 1, cfg.timeout)?.remove(0);
            let q = 1.0 - q_from_discs(best.discs, best.exact(after.empties()), self.disc_scale);
            return Ok(Analysis { moves: vec![ScoredMove::new(PASS_ID, q).raw(-best.discs as f32)], value: None });
        }
        let rows = session.hint(pos, cfg.max_moves, cfg.timeout)?;
        Ok(self.analysis(pos, &rows))
    }

    fn head_rows(&self, _s: &SampleView, _t: usize, pos: &OthPos, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>> {
        Ok(if pos.legal == 0 {
            // p(place | not pass) has no support; P(pass) = 1 (the pass head is supervised here)
            vec![HeadRow::Missing, HeadRow::Bernoulli(1.0)]
        } else {
            vec![HeadRow::Categorical(policy.to_vec()), HeadRow::Bernoulli(0.0)]
        })
    }
}

// ------------------------------------------------------------------ fake engine

/// An in-process stand-in for Egaroucid for Console (tests; `gmdata label fake-engine
/// othello`): answers `version`, `setboard`, `hint N`, `clearcache` / `level` / `init`
/// (silent) and unknown commands (`[ERROR] command `…` not found`), exits on `quit`.
/// Scores are [`FakeEgaroucid::score`] (deterministic in position and move; another
/// scorer with [`FakeEgaroucid::with_scores`]); rows report `E@100%` (exact) at ≤
/// [`FakeEgaroucid::EXACT_EMPTIES`] empties, else `15@74%`.
pub struct FakeEgaroucid {
    pos: Option<OthPos>,
    hints: usize,
    crash_on: Option<usize>,
    hang_on: Option<(usize, Duration)>,
    score: fn(&OthPos, i64) -> i32,
}

impl Default for FakeEgaroucid {
    fn default() -> Self {
        Self::new()
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl FakeEgaroucid {
    pub const EXACT_EMPTIES: u32 = 20;
    pub const VERSION: &'static str = "Egaroucid for Console 7.8.0 fake (in-process)";

    pub fn new() -> Self {
        FakeEgaroucid { pos: None, hints: 0, crash_on: None, hang_on: None, score: Self::score }
    }

    /// Score moves with `f(position, square)` instead of [`FakeEgaroucid::score`].
    pub fn with_scores(mut self, f: fn(&OthPos, i64) -> i32) -> Self {
        self.score = f;
        self
    }

    /// Exit on the `n`-th `hint` (1-based, per process).
    pub fn crash_on(mut self, n: usize) -> Self {
        self.crash_on = Some(n);
        self
    }

    /// Sleep `d` before answering the `n`-th `hint`.
    pub fn hang_on(mut self, n: usize, d: Duration) -> Self {
        self.hang_on = Some((n, d));
        self
    }

    /// The fake's disc score of `sq` in `pos` (−20 … 20).
    pub fn score(pos: &OthPos, sq: i64) -> i32 {
        (splitmix(pos.mine() ^ splitmix(pos.theirs()) ^ (sq as u64).wrapping_mul(0xA24B_AED4_963E_E407)) % 41) as i32 - 20
    }

    /// The fake's rows for `pos`: every legal move, best first (ties by square).
    pub fn rows(pos: &OthPos) -> Vec<HintRow> {
        Self::rows_with(pos, Self::score)
    }

    /// [`FakeEgaroucid::rows`] with the scorer `f`.
    pub fn rows_with(pos: &OthPos, f: fn(&OthPos, i64) -> i32) -> Vec<HintRow> {
        let e = pos.empties();
        let depth = if e <= Self::EXACT_EMPTIES { (e, 100.0) } else { (15, 74.0) };
        let mut rows: Vec<HintRow> =
            (0..64).filter(|&sq| pos.is_legal(sq)).map(|sq| HintRow { square: sq, discs: f(pos, sq), depth: Some(depth) }).collect();
        rows.sort_by(|a, b| b.discs.cmp(&a.discs).then(a.square.cmp(&b.square)));
        rows
    }
}

impl LineHandler for FakeEgaroucid {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        let line = line.trim();
        let (cmd, arg) = line.split_once(' ').unwrap_or((line, ""));
        match cmd {
            "quit" | "exit" => None,
            "version" | "ver" => Some(vec![Self::VERSION.to_string(), "  @date     2021-2026".into(), String::new()]),
            "setboard" => {
                self.pos = OthPos::from_egaroucid(arg);
                Some(vec![])
            }
            "clearcache" | "level" | "init" | "reset" => Some(vec![]),
            "hint" => {
                self.hints += 1;
                if self.crash_on == Some(self.hints) {
                    return None;
                }
                if let Some((n, d)) = self.hang_on {
                    if n == self.hints {
                        std::thread::sleep(d);
                    }
                }
                let pos = self.pos.clone().unwrap_or_else(OthPos::start);
                let n = arg.trim().parse::<usize>().unwrap_or(1).max(1);
                let w = 15;
                let mut out = vec![format!("|{:>w$}|{:>w$}|{:>w$}|{:>w$}|{:>w$}|{:>w$}|{:>w$}|", "Level", "Depth", "Move", "Score", "Time", "Nodes", "NPS")];
                for r in Self::rows_with(&pos, self.score).into_iter().take(n) {
                    let (d, p) = r.depth.unwrap();
                    out.push(format!(
                        "|{:>w$}|{:>w$}|{:>w$}|{:>w$}|{:>w$}|{:>w$}|{:>w$}|",
                        "custom",
                        format!("{d}@{p}%"),
                        square_name(r.square),
                        format!("{:+}", r.discs),
                        "000:00:00.000",
                        1,
                        1
                    ));
                }
                Some(out)
            }
            _ => Some(vec![format!("[ERROR] command `{cmd}` not found")]),
        }
    }
}

/// [`FakeEgaroucid`] as an in-process engine command.
pub fn fake_engine_command() -> EngineCommand {
    EngineCommand::in_process("fake-egaroucid", FakeEgaroucid::new)
}

// ------------------------------------------------------------------ --fit-disc-scale

/// A calibration point: engine score, empties, outcome for the mover.
#[derive(Debug, Clone, Copy)]
struct FitPoint {
    discs: i32,
    empties: u32,
    outcome: f64,
}

/// Maximum-likelihood β of P(win) = σ(β·v) (Newton on the concave log-likelihood);
/// returns (β, mean log-likelihood).
fn fit_beta(pts: &[(f64, f64)]) -> (f64, f64) {
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut beta = 0.1f64;
    for _ in 0..100 {
        let (mut g, mut h) = (0.0, 0.0);
        for &(v, y) in pts {
            let p = sig(beta * v);
            g += (y - p) * v;
            h += p * (1.0 - p) * v * v;
        }
        if h <= 1e-12 {
            break;
        }
        let step = g / h;
        beta = (beta + step).clamp(1e-4, 10.0);
        if step.abs() < 1e-10 {
            break;
        }
    }
    let ll = pts
        .iter()
        .map(|&(v, y)| {
            let p = sig(beta * v).clamp(1e-12, 1.0 - 1e-12);
            y * p.ln() + (1.0 - y) * (1.0 - p).ln()
        })
        .sum::<f64>()
        / pts.len().max(1) as f64;
    (beta, ll)
}

/// Egaroucid plays both sides from `start` (one `hint 1` per move); (first score, outcome).
fn playout(session: &mut EgaSession, start: &OthPos, timeout: Duration) -> Result<(i32, f64)> {
    let mut pos = start.clone();
    let mut first = None;
    loop {
        let row = session.hint(&pos, 1, timeout)?.remove(0);
        first.get_or_insert(row.discs);
        match pos.after(row.square) {
            Some(n) => pos = n,
            None => {
                let fin = OthPos::new(play(&pos.board, pos.mover, row.square as usize), start.mover);
                let d = fin.final_diff();
                return Ok((first.unwrap(), if d > 0 { 1.0 } else if d == 0 { 0.5 } else { 0.0 }));
            }
        }
    }
}

/// `--fit-disc-scale N` (see the module docs).
pub fn fit_disc_scale(lab: &OthelloLabeller, opts: &RunOpts, n: usize) -> Result<Value> {
    let limits = lab.resolve_limits(&opts.limits)?;
    let exact = limits.u64_or("exact_empties", DEFAULT_EXACT_EMPTIES)? as u32;
    let cfg = EngineConfig { command: opts.command.clone(), options: opts.engine_options.clone(), limits: limits.clone(), timeout: opts.timeout, max_moves: 1 };
    // candidate positions: distinct (mover, opponent) boards outside the exact zone
    let mut cands: Vec<(u64, OthPos)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let shards = find_shards(&opts.input, &opts.out)?;
    for (dir, _) in shards.iter().take(opts.max_shards.unwrap_or(usize::MAX)) {
        let reader = ShardReader::open(dir)?;
        lab.check_shard(&reader)?;
        let data = ShardData::load(reader)?;
        for i in 0..data.reader.n_samples() {
            let v = data.view(i);
            let Ok(ps) = lab.replay(&v) else { continue };
            if lab.verify(&v, &ps).is_err() {
                continue;
            }
            for p in ps {
                if p.empties() > exact && p.n_legal() >= 2 && seen.insert(lab.key(&p)) {
                    let (a, b) = lab.key(&p);
                    cands.push((splitmix(a ^ splitmix(b)), p));
                }
            }
        }
        if cands.len() >= 50 * n {
            break;
        }
    }
    if cands.is_empty() {
        bail!("no positions with more than {exact} empties under {}", opts.input.display());
    }
    cands.sort_by_key(|c| c.0);
    cands.truncate(n);
    let total = cands.len();
    let t0 = Instant::now();
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let points: Mutex<Vec<FitPoint>> = Mutex::new(Vec::with_capacity(total));
    let workers = opts.workers.max(1).min(total);
    std::thread::scope(|sc| -> Result<()> {
        let mut handles = Vec::new();
        for w in 0..workers {
            let (cands, cfg, points, done, failed) = (&cands, &cfg, &points, &done, &failed);
            handles.push(sc.spawn(move || -> Result<()> {
                let mut session = lab.open_session(cfg)?;
                for (_, p) in cands.iter().skip(w).step_by(workers) {
                    let mut r = playout(&mut session, p, cfg.timeout);
                    if r.is_err() {
                        session = lab.open_session(cfg)?;
                        r = playout(&mut session, p, cfg.timeout);
                    }
                    match r {
                        Ok((discs, outcome)) => points.lock().unwrap().push(FitPoint { discs, empties: p.empties(), outcome }),
                        Err(e) => {
                            if failed.fetch_add(1, Ordering::Relaxed) < 5 {
                                eprintln!("[fit-disc-scale] playout failed: {e:#}");
                            }
                        }
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if !opts.quiet && (d.is_multiple_of(50) || d == total) {
                        eprintln!("[fit-disc-scale] {d}/{total} playouts, {:.0}s", t0.elapsed().as_secs_f64());
                    }
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().map_err(|_| anyhow::anyhow!("a fit worker panicked"))??;
        }
        Ok(())
    })?;
    let pts = points.into_inner().unwrap();
    if pts.len() < 10 {
        bail!("only {} playouts succeeded", pts.len());
    }
    let xy: Vec<(f64, f64)> = pts.iter().map(|p| (p.discs as f64, p.outcome)).collect();
    let (beta, ll) = fit_beta(&xy);
    let s = 1.0 / beta;
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let brier = xy.iter().map(|&(v, y)| (sig(v / s) - y).powi(2)).sum::<f64>() / xy.len() as f64;
    let edges = [-64, -16, -10, -6, -4, -2, 0, 1, 3, 5, 7, 11, 17, 65];
    let bins: Vec<Value> = edges
        .windows(2)
        .filter_map(|w| {
            let sel: Vec<&(f64, f64)> = xy.iter().filter(|(v, _)| *v >= w[0] as f64 && *v < w[1] as f64).collect();
            (!sel.is_empty()).then(|| {
                let m = sel.len() as f64;
                json!({"discs": [w[0], w[1] - 1], "n": sel.len(),
                       "outcome_mean": sel.iter().map(|x| x.1).sum::<f64>() / m,
                       "model_mean": sel.iter().map(|x| sig(x.0 / s)).sum::<f64>() / m})
            })
        })
        .collect();
    let phases: Vec<Value> = [(exact + 1, 30), (31, 40), (41, 50), (51, 60)]
        .iter()
        .filter_map(|&(lo, hi)| {
            let sub: Vec<(f64, f64)> = pts.iter().filter(|p| (lo..=hi).contains(&p.empties)).map(|p| (p.discs as f64, p.outcome)).collect();
            (sub.len() >= 10).then(|| json!({"empties": [lo, hi], "n": sub.len(), "disc_scale": 1.0 / fit_beta(&sub).0}))
        })
        .collect();
    let report = json!({
        "disc_scale": (s * 100.0).round() / 100.0,
        "n": pts.len(),
        "failed": failed.load(Ordering::Relaxed),
        "mean_log_likelihood": ll,
        "brier": brier,
        "draw_rate": pts.iter().filter(|p| p.outcome == 0.5).count() as f64 / pts.len() as f64,
        "calibration": bins,
        "by_phase": phases,
        "limits": limits.to_json(),
        "engine_options": opts.engine_options.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect::<serde_json::Map<_, _>>(),
        "elapsed_s": t0.elapsed().as_secs_f64().round(),
        "method": "outcome of Egaroucid self-play (hint 1 per move, same search) from each position vs its hint-1 score; ML fit of sigma(discs/s)",
    });
    std::fs::create_dir_all(&opts.out)?;
    std::fs::write(opts.out.join("disc_scale_fit.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests;
