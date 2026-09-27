//! `gmdata label xiangqi` — Pikafish (UCI) teacher labels for xiangqi gm-shards
//! (`gmdata xiangqi convert`, [`crate::xiangqi`]; rules and frames
//! `gmrules::games::xiangqi`, the FromToDecoder of model/games/board.py).
//!
//! ```text
//! scripts/engines/fetch_engines.sh pikafish          # → ${GM_ENGINES:-$HOME/gm-engines}/bin/pikafish
//! gmdata label xiangqi --input data/xiangqi/train --out data/xiangqi_pf/train \
//!     --engine ~/gm-engines/bin/pikafish --workers 32
//! gmdata label xiangqi --input data/xiangqi/val --out data/xiangqi_pf/val --engine … (same flags!)
//! ```
//!
//! Engine conversation (one single-threaded Pikafish per worker, [`DEFAULT_OPTIONS`] then
//! `--engine-option` overrides, `MultiPV` = `--max-moves` and `UCI_ShowWDL = true` last):
//!
//! ```text
//! per position:  ucinewgame; isready     (clears hash and history: every answer is
//!                                          independent of what the worker searched before)
//!                position startpos moves h2e2 h9g7 …    (the whole game so far, so the
//!                                          engine sees its repetitions and move counter)
//!                go nodes N  →  info … multipv k score cp X wdl W D L … pv <move> …
//! ```
//!
//! Pikafish's coordinates are files a-i and ranks 0-9 from Red's side ("h2e2"); gmrules'
//! (Fairy-Stockfish's) ranks are 1-10 ("h3e3"): [`engine_move`] / [`parse_engine_move`].
//! Rules: the frames follow gmrules' AXF rules (Fairy-Stockfish's); Pikafish differs in
//! its move limit (60 moves without a capture instead of 50) and in its own chase
//! detection — it only scores positions, the rules of the frames stay gmrules'.
//!
//! Defaults: `--limit nodes=100000` (`go nodes N`, deterministic single-threaded; ≈ 0.07 s
//! per position on an Apple M4 Pro core with MultiPV 8 — 1.5 M nodes/s — plus ~3 ms for
//! `ucinewgame` + `isready`), `--max-moves 8` (= MultiPV), `--tau 0.05`, `--tie-weight 2`
//! (π ∝ exp(Q/τ + w·u), u the secondary utility of `tgt.teacher_raw`, [`secondary_of_raw`]:
//! mates above every cp score, faster mates first; cp through 0.5·tanh(cp / 1000)).
//! Pikafish's WDL is very drawish in quiet positions (the start position: W 95 D 897 L 8
//! permille at 100k nodes), so Q ranges over little there and the tie-break's cp term
//! carries much of the ordering.
//!
//! ```text
//! Q(move)            = (W + D/2) / 1000 of the line's `wdl` (mover POV); mate lines 1 / 0
//! tgt.engine_value   = 2·Q(multipv 1) − 1
//! tgt.teacher_idx    [T,K] i64  from·90 + to, POV squares (tgt.from_idx / to_idx: row·9 +
//!                               col, the mover's side at the bottom), best first; −1 pad
//! tgt.teacher_q      [T,K] f32  Q per move, NaN pad
//! tgt.teacher_raw    [T,K] f32  centipawns; mate in n (moves, UCI sign) = ±(100000 − |n|)
//! tgt.from_probs_idx/_val [T,K] marginal of π over from-squares (sparse)
//! tgt.to_probs_idx/_val   [T,K] p(to | from = tgt.from_idx) (sparse; all-pad when the
//!                               teacher puts no mass on the ground-truth from-square)
//! tgt.engine_value_mask   [T]   frame labelled
//! ```
//!
//! Stale lines (a node-limited search stops inside an iteration and prints the moves that
//! failed low with their older, shallower score) are capped at the lowest Q / raw of the
//! deepest lines, as for chess ([`super::chess::analysis`]).
//!
//! Replay and verification: a sample is one game (the interleaved stream, every frame
//! from its mover's view) from the standard start position; it is replayed with gmrules'
//! rules from the POV labels tgt.from_idx / to_idx, and gmrules::replay re-encodes every
//! frame with the simulator's (= the converter's) encoder: every obs / target tensor of
//! the schema and tgt.value (the sample meta's per-seat `outcome`) must match bit for bit.
//!
//! Position cache key (exact): the answer to `position startpos moves …` after
//! `ucinewgame` depends on the position, the plies since the last capture (the move-limit
//! counter) and the positions since then (repetitions — nothing before a capture can
//! recur), all fixed by the position at the last capture (or the start) and the moves
//! since: key = its FEN + those moves.
//!
//! Tests: fake-engine unit tests ([`fake_engine`], also served by
//! `gmdata label fake-engine xiangqi`) on games of the CCPD fixture converted by the real
//! converter, and real Pikafish tests (skipped unless `scripts/engines/fetch_engines.sh
//! pikafish` has run or `PIKAFISH` points to a binary).

use super::fake::{FakeMove, UciFake};
use super::game::{Analysis, EngineConfig, HeadKind, HeadRow, HeadSpec, Labeller, Limits, SampleView, ScoredMove};
use super::transform::{conditional, marginal, q_from_wdl, u_cp, u_mate};
use super::uci::{Flavor, Score, UciSession};
use super::LabelArgs;
use anyhow::{bail, ensure, Context, Result};
use gmrules::games::xiangqi::{self as xq, Xiangqi, XiangqiConfig, FILES, RANKS, SQUARES};
use gmrules::{replay, Buf, Event, Game, Seat};
use gmshard::ShardReader;
use serde_json::{json, Value};
use std::sync::Arc;

/// Bump when the meaning of a xiangqi label changes (recorded in the teacher meta).
pub const LABELLER_VERSION: u32 = 1;
/// `--limit nodes` default (see the module docs).
pub const DEFAULT_NODES: u64 = 100_000;
/// `--max-moves` default (= UCI MultiPV).
pub const DEFAULT_MAX_MOVES: usize = 8;
/// Engine options applied before `--engine-option` (which may override them).
pub const DEFAULT_OPTIONS: [(&str, &str); 2] = [("Threads", "1"), ("Hash", "16")];
/// `tgt.teacher_raw` of a mate in n: ±(MATE_RAW − |n|).
pub const MATE_RAW: f32 = 100_000.0;
/// |teacher_raw| above this is a mate.
pub const MATE_THRESHOLD: f32 = 50_000.0;
/// cp scale of the secondary utility `0.5·tanh(cp / TIE_CP_SCALE)`.
pub const TIE_CP_SCALE: f32 = 1000.0;
/// The shard tensors the labeller replays and verifies.
const REQUIRED: [(&str, &[usize]); 7] = [
    ("obs.board_seq", &[xq::PIECE_CHANNELS, RANKS, FILES]),
    ("obs.globals_seq", &[xq::GLOBAL_FEATURES]),
    ("tgt.from_idx", &[]),
    ("tgt.to_idx", &[]),
    ("tgt.legal_from", &[SQUARES]),
    ("tgt.legal_to_of", &[SQUARES]),
    ("tgt.value", &[]),
];

/// POV (from, to) of a move — the FromToDecoder labels.
pub type Target = (i64, i64);

/// Flat action id of `tgt.teacher_idx`: `from·90 + to` (POV squares).
pub fn action_id((from, to): Target) -> i64 {
    from * SQUARES as i64 + to
}

/// Inverse of [`action_id`].
pub fn decode_action(id: i64) -> Target {
    (id / SQUARES as i64, id % SQUARES as i64)
}

/// Pikafish's name of a square: file a-i, rank 0-9 from Red's side.
fn engine_square(sq: usize) -> String {
    format!("{}{}", (b'a' + (sq % FILES) as u8) as char, sq / FILES)
}

/// Pikafish's coordinate move of a gmrules action ("h2e2").
pub fn engine_move(action: u32) -> String {
    let (f, t) = xq::move_of(action);
    format!("{}{}", engine_square(f), engine_square(t))
}

/// The gmrules action of a Pikafish coordinate move (None when malformed).
pub fn parse_engine_move(s: &str) -> Option<u32> {
    let b = s.trim().as_bytes();
    if b.len() != 4 {
        return None;
    }
    let sq = |f: u8, r: u8| -> Option<usize> {
        let (f, r) = (f.checked_sub(b'a')? as usize, r.checked_sub(b'0')? as usize);
        (f < FILES && r < RANKS).then_some(r * FILES + f)
    };
    Some(xq::action_of(sq(b[0], b[1])?, sq(b[2], b[3])?))
}

/// `tgt.teacher_raw` of an engine score (centipawns; mates see [`MATE_RAW`]).
pub fn raw_score(s: Score) -> f32 {
    match s {
        Score::Cp(cp) => cp as f32,
        Score::Mate(n) if n > 0 => MATE_RAW - n as f32,
        Score::Mate(n) => -MATE_RAW - n as f32,
    }
}

/// Secondary utility u ∈ [−1, 1] of a `tgt.teacher_raw`: a mate in n moves →
/// ±(0.5 + 0.5·5/(n + 4)), else 0.5·tanh(cp / 1000); `None` for a non-finite raw.
pub fn secondary_of_raw(raw: f32) -> Option<f32> {
    if !raw.is_finite() {
        return None;
    }
    if raw.abs() > MATE_THRESHOLD {
        return Some(raw.signum() * u_mate(MATE_RAW - raw.abs()));
    }
    Some(u_cp(raw, TIE_CP_SCALE))
}

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub common: LabelArgs,
}

pub fn run(args: Args) -> Result<()> {
    super::run_labeller(XiangqiLabeller::new(args.common.max_moves.unwrap_or(DEFAULT_MAX_MOVES)), &args.common)
}

/// The moves of one game as `position` commands (see [`super::chess::GameLine`]).
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

    fn push(&mut self, action: u32) {
        self.text.push(' ');
        self.text.push_str(&engine_move(action));
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

    fn moves(&self, a: usize, b: usize) -> &str {
        &self.text[self.ends[a]..self.ends[b]]
    }
}

/// A replayed frame: the position before move `t` of a game.
#[derive(Debug)]
pub struct XiangqiPos {
    /// the mover
    pub seat: Seat,
    /// legal moves: (gmrules action, POV target)
    pub legal: Vec<(u32, Target)>,
    /// index of the ground-truth move in `legal`
    pub gt: usize,
    pub t: usize,
    pub line: Arc<GameLine>,
    /// the cache key (see the module docs)
    pub key: String,
    /// FEN (Fairy-Stockfish notation; error messages)
    pub fen: String,
}

impl XiangqiPos {
    pub fn gt_target(&self) -> Target {
        self.legal[self.gt].1
    }

    pub fn command(&self) -> &str {
        self.line.command(self.t)
    }
}

/// The FEN of `g` without its move counters (the part of it a search depends on).
fn position_fen(g: &Xiangqi) -> String {
    let f = g.fen();
    f.rsplitn(3, ' ').nth(2).unwrap_or(&f).to_string()
}

/// Replay a game from the start position given its POV move labels.
pub fn replay_moves(from: &[i64], to: &[i64]) -> Result<Vec<XiangqiPos>> {
    ensure!(from.len() == to.len(), "label columns of different lengths");
    let mut g = Xiangqi::new_game(&XiangqiConfig::default());
    let mut line = GameLine::new();
    let (mut tail_start, mut tail_fen) = (0usize, String::new());
    let mut frames = Vec::with_capacity(from.len());
    for t in 0..from.len() {
        let seat = g.side_to_move();
        if t == 0 || g.rule50() == 0 {
            tail_start = t;
            tail_fen = position_fen(&g);
        }
        let legal: Vec<(u32, Target)> = g
            .legal_moves()
            .iter()
            .map(|&a| {
                let (f, to) = xq::move_of(a);
                (a, (xq::pov(seat, f) as i64, xq::pov(seat, to) as i64))
            })
            .collect();
        let want = (from[t], to[t]);
        let gt = legal
            .iter()
            .position(|l| l.1 == want)
            .with_context(|| format!("frame {t}: recorded move {want:?} (POV from, to) is not legal in {}", g.fen()))?;
        let action = legal[gt].0;
        frames.push((seat, legal, gt, format!("{tail_fen}{}", line.moves(tail_start, t)), g.fen()));
        line.push(action);
        g.play(action).map_err(|e| anyhow::anyhow!("frame {t}: {e}"))?;
    }
    let line = Arc::new(line);
    Ok(frames
        .into_iter()
        .enumerate()
        .map(|(t, (seat, legal, gt, key, fen))| XiangqiPos { seat, legal, gt, t, line: line.clone(), key, fen })
        .collect())
}

/// The frames of `positions` re-encoded by gmrules (the simulator's and converter's
/// encoder), compared bit for bit with every obs / target tensor of the schema.
fn verify_frames(s: &SampleView, positions: &[XiangqiPos]) -> Result<()> {
    let cfg = XiangqiConfig::default();
    let events: Vec<Event> = positions.iter().map(|p| Event::Act(p.seat, p.legal[p.gt].0)).collect();
    let r = replay::<Xiangqi>(&cfg, &events).map_err(|e| anyhow::anyhow!("gmrules replay: {e}"))?;
    ensure!(r.decisions.len() == positions.len(), "gmrules replay has {} decisions, the sample {} frames", r.decisions.len(), positions.len());
    let spec = Xiangqi::spec(&cfg);
    let groups = [("obs", &spec.obs, false), ("tgt", &spec.target, true)];
    for (prefix, specs, is_target) in groups {
        for (j, ts) in specs.iter().enumerate() {
            let name = format!("{prefix}.{}", ts.name);
            let (mut f, mut i) = (Vec::new(), Vec::new());
            for d in &r.decisions {
                let buf = if is_target { &d.target[j] } else { &d.frame.obs[j] };
                match buf {
                    Buf::F32(v) => f.extend_from_slice(v),
                    Buf::Bool(v) => i.extend(v.iter().map(|&b| b as i64)),
                    Buf::I32(v) => i.extend(v.iter().map(|&x| x as i64)),
                    Buf::I64(v) => i.extend_from_slice(v),
                }
            }
            if i.is_empty() {
                s.check_f32(&name, &f)?;
            } else {
                s.check_i64(&name, &i)?;
            }
        }
    }
    let outcome: Vec<f32> = s.meta()["outcome"]
        .as_array()
        .context("sample meta has no per-seat \"outcome\"")?
        .iter()
        .map(|v| v.as_f64().map(|x| x as f32).context("non-numeric outcome"))
        .collect::<Result<_>>()?;
    ensure!(outcome.len() == 2, "sample meta outcome {outcome:?}: two seats expected");
    let value: Vec<f32> = positions.iter().map(|p| outcome[p.seat]).collect();
    s.check_f32("tgt.value", &value)
}

/// Reject `--engine-option`s the labeller overrides (`MultiPV` = `--max-moves`,
/// `UCI_ShowWDL` = true; UCI option names are case insensitive).
pub fn check_engine_options(cfg: &EngineConfig) -> Result<()> {
    for (k, v) in &cfg.options {
        if k.eq_ignore_ascii_case("MultiPV") && v.trim() != cfg.max_moves.to_string() {
            bail!("--engine-option {k}={v}: MultiPV is set from --max-moves ({}); use --max-moves {v}", cfg.max_moves);
        }
        if k.eq_ignore_ascii_case("UCI_ShowWDL") && !v.trim().eq_ignore_ascii_case("true") {
            bail!("--engine-option {k}={v}: the xiangqi labeller needs UCI_ShowWDL=true (Q comes from the WDL)");
        }
    }
    Ok(())
}

/// The `go` command of resolved limits.
pub fn go_command(limits: &Limits) -> Result<String> {
    let nodes = limits.u64_or("nodes", DEFAULT_NODES)?;
    ensure!(nodes > 0, "--limit nodes must be positive");
    Ok(format!("go nodes {nodes}"))
}

/// The xiangqi labeller (see the module docs). `max_moves` must equal the run's
/// `--max-moves` (it sizes the sparse heads).
#[derive(Clone, Debug)]
pub struct XiangqiLabeller {
    pub max_moves: usize,
}

impl Default for XiangqiLabeller {
    fn default() -> Self {
        XiangqiLabeller::new(DEFAULT_MAX_MOVES)
    }
}

impl XiangqiLabeller {
    pub fn new(max_moves: usize) -> Self {
        XiangqiLabeller { max_moves: max_moves.clamp(1, SQUARES) }
    }
}

impl Labeller for XiangqiLabeller {
    type Position = XiangqiPos;
    type Key = String;
    type Session = UciSession;

    fn game(&self) -> &'static str {
        "xiangqi"
    }

    fn version(&self) -> u32 {
        LABELLER_VERSION
    }

    fn check_shard(&self, shard: &ShardReader) -> Result<()> {
        let d = shard.dir.display();
        if shard.game != "xiangqi" || shard.converter_version != Xiangqi::VERSION {
            bail!("{d}: game {:?} converter v{}, the xiangqi labeller replays xiangqi v{} shards", shard.game, shard.converter_version, Xiangqi::VERSION);
        }
        let fen = shard.meta()["config"]["fen"].as_str().unwrap_or("");
        if !fen.is_empty() {
            bail!("{d}: games from {fen:?}; the labeller replays games from the standard start position");
        }
        for (name, shape) in REQUIRED {
            let info = shard.tensor_info(name)?;
            if info.frame_shape != shape {
                bail!("{d}: {name} has frame shape {:?}, expected {shape:?}", info.frame_shape);
            }
        }
        Ok(())
    }

    fn heads(&self) -> Vec<HeadSpec> {
        let k = self.max_moves;
        vec![
            HeadSpec { name: "from", kind: HeadKind::Categorical { n: SQUARES, k } },
            HeadSpec { name: "to", kind: HeadKind::Categorical { n: SQUARES, k } },
        ]
    }

    fn raw_unit(&self) -> Option<&'static str> {
        Some("cp")
    }

    fn secondary(&self, m: &ScoredMove) -> Option<f32> {
        secondary_of_raw(m.raw)
    }

    fn secondary_meta(&self) -> Option<Value> {
        Some(json!({
            "of": "teacher_raw",
            "mate": "|raw| > 50000: mate in n = max(100000 - |raw|, 1) moves, u = sign(raw) * (0.5 + 0.5*5/(n + 4))",
            "cp": "u = 0.5 * tanh(raw / 1000)",
        }))
    }

    fn default_max_moves(&self) -> usize {
        self.max_moves
    }

    fn resolve_limits(&self, given: &Limits) -> Result<Limits> {
        let l = given.with_defaults(&[("nodes", &DEFAULT_NODES.to_string())], &["nodes"])?;
        go_command(&l)?;
        Ok(l)
    }

    fn open_session(&self, cfg: &EngineConfig) -> Result<UciSession> {
        check_engine_options(cfg)?;
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
            "action_id": "from*90 + to (POV squares row*9 + col: row = 9 - rank, col = file for Red to move; row = rank, col = 8 - file for Black)",
            "q": "(W + D/2) / 1000 of each MultiPV line's final info (UCI_ShowWDL)",
            "value": "Q of the multipv 1 line",
            "stale_lines": "lines shallower than the deepest line are capped at the deepest lines' lowest Q and raw score",
            "raw": "cp; mate in n = +-(100000 - |n|)",
            "query": "ucinewgame; position startpos moves <game history, files a-i ranks 0-9 from Red>; go",
            "go": go_command(&cfg.limits).unwrap_or_default(),
            "default_engine_options": DEFAULT_OPTIONS.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect::<serde_json::Map<_, _>>(),
            "skip": "forced (one legal move)",
            "converter_version": Xiangqi::VERSION,
        })
    }

    fn replay(&self, s: &SampleView) -> Result<Vec<XiangqiPos>> {
        replay_moves(&s.i64s("tgt.from_idx")?, &s.i64s("tgt.to_idx")?)
    }

    fn verify(&self, s: &SampleView, positions: &[XiangqiPos]) -> Result<()> {
        verify_frames(s, positions)
    }

    fn skip_reason(&self, _s: &SampleView, _t: usize, pos: &XiangqiPos) -> Option<&'static str> {
        (pos.legal.len() == 1).then_some("forced")
    }

    fn key(&self, pos: &XiangqiPos) -> String {
        pos.key.clone()
    }

    fn query(&self, session: &mut UciSession, pos: &XiangqiPos, cfg: &EngineConfig) -> Result<Analysis> {
        session.new_game()?;
        let r = session.search(pos.command(), &go_command(&cfg.limits)?, cfg.timeout)?;
        analysis(pos, &r.lines)
    }

    fn head_rows(&self, _s: &SampleView, _t: usize, pos: &XiangqiPos, policy: &[(i64, f32)]) -> Result<Vec<HeadRow>> {
        let (gf, _) = pos.gt_target();
        let dist: Vec<(Target, f32)> = policy.iter().map(|&(a, p)| (decode_action(a), p)).collect();
        Ok(vec![
            HeadRow::Categorical(marginal(&dist, |a| a.0)),
            HeadRow::Categorical(conditional(&dist, |a| a.0 == gf, |a| a.1)),
        ])
    }
}

/// Engine lines → scored moves (flat POV ids of the frame's legal moves, Q from WDL, raw
/// cp; stale lines capped, see the module docs).
pub fn analysis(pos: &XiangqiPos, lines: &[super::uci::Info]) -> Result<Analysis> {
    if lines.is_empty() {
        bail!("the engine reported no scored line for {}", pos.fen);
    }
    let mut scored = Vec::with_capacity(lines.len());
    for l in lines {
        let mv = &l.pv[0];
        let a = parse_engine_move(mv).with_context(|| format!("engine move {mv:?} is not a coordinate move"))?;
        let target = pos
            .legal
            .iter()
            .find(|(x, _)| *x == a)
            .map(|(_, t)| *t)
            .with_context(|| format!("engine move {mv} is not legal in {}", pos.fen))?;
        let [w, d, lo] = l.wdl.with_context(|| format!("line {} ({mv}) has no wdl: the engine must support UCI_ShowWDL", l.multipv))?;
        let score = l.score.context("line without score")?;
        scored.push((l.multipv, l.depth.unwrap_or(0), action_id(target), q_from_wdl(w as f32, d as f32, lo as f32), raw_score(score)));
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
        if multipv == 1 {
            value = Some(q);
        }
        moves.push(ScoredMove::new(id, q).raw(raw));
    }
    Ok(Analysis { moves, value })
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The game of a `position startpos [moves …]` command in Pikafish notation (`None` if
/// not one or illegal).
pub fn parse_position(line: &str) -> Option<Xiangqi> {
    let mut toks = line.split_whitespace();
    if toks.next()? != "position" || toks.next()? != "startpos" {
        return None;
    }
    let mut g = Xiangqi::new_game(&XiangqiConfig::default());
    match toks.next() {
        None => return Some(g),
        Some("moves") => {}
        Some(_) => return None,
    }
    for t in toks {
        g.play(parse_engine_move(t)?).ok()?;
    }
    Some(g)
}

/// A xiangqi UCI fake engine for tests: parses `position startpos [moves …]` (Pikafish
/// notation) with gmrules and scores every legal move — mate in 1 for a move that wins
/// on the spot (wdl 1000 0 0), otherwise a deterministic pseudo-random cp ∈ [−300, 300]
/// of (position, move) with a logistic WDL. Unknown / illegal positions crash it.
pub fn fake_engine() -> UciFake<impl FnMut(&str) -> Option<Vec<FakeMove>> + Send> {
    UciFake::new(|line: &str| {
        let g = parse_position(line)?;
        let fen = g.fen();
        let mover = g.side_to_move();
        Some(
            g.legal_moves()
                .iter()
                .map(|&a| {
                    let uci = engine_move(a);
                    let mut after = g.clone();
                    after.play(a).expect("a legal move");
                    if matches!(after.end(), Some((r, _)) if (r > 0) == (mover == xq::RED) && r != 0) {
                        return FakeMove::mate(&uci, 1).wdl([1000, 0, 0]);
                    }
                    let cp = (fnv1a(&format!("{fen} {uci}")) % 601) as i32 - 300;
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
    super::engine::EngineCommand::in_process("xiangqi-uci", fake_engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::driver::run;
    use crate::label::testkit::{engine_path, overlay_tensor, quiet_opts, tmpdir};
    use crate::traj::{self, ConvertOptions};
    use crate::xiangqi::{records, Filter};
    use gmshard::DType;
    use std::path::{Path, PathBuf};

    fn fixture_zip() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/xiangqi/ccpd_sample.zip")
    }

    /// The fixture's first `n` games, converted by the real converter into `root/base`.
    fn convert_fixture(root: &Path, n: usize, per_shard: usize) -> usize {
        let filter = Filter::default();
        let recs = records(&fixture_zip(), &filter).unwrap().take(n);
        let opts = ConvertOptions { shard_size: per_shard, ..Default::default() };
        let st = traj::convert::<Xiangqi>(&XiangqiConfig::default(), recs, &root.join("base"), &opts).unwrap();
        assert!(st.converted > 0, "{st:?}");
        st.converted
    }

    #[test]
    fn engine_notation_round_trips() {
        // Red's central cannon: gmrules h3e3 = Pikafish h2e2; Black's horse h10g8 = h9g7
        let g = Xiangqi::new_game(&XiangqiConfig::default());
        let a = xq::parse_uci("h3e3").unwrap();
        assert!(g.legal_moves().contains(&a));
        assert_eq!(engine_move(a), "h2e2");
        assert_eq!(parse_engine_move("h2e2"), Some(a));
        assert_eq!(engine_move(xq::parse_uci("h10g8").unwrap()), "h9g7");
        for &m in g.legal_moves() {
            assert_eq!(parse_engine_move(&engine_move(m)), Some(m));
        }
        assert_eq!(parse_engine_move("j0a0"), None);
        assert_eq!(parse_engine_move("a10a9"), None);
        assert!(parse_position("position startpos moves h2e2 h9g7").is_some());
        assert!(parse_position("position startpos moves h2e2 h2e2").is_none());
    }

    #[test]
    fn fake_labels_verify_and_map_onto_the_heads() {
        let root = tmpdir("xq-label");
        let games = convert_fixture(&root, 6, 4);
        let summary = serde_json::to_value(run(XiangqiLabeller::default(), &quiet_opts(&root, fake_engine_command())).unwrap()).unwrap();
        let st = &summary["stats"];
        assert_eq!(st["samples"], games as u64, "{summary}");
        assert_eq!(st["verify_failed"], 0, "{summary}");
        assert_eq!(st["replay_failed"], 0, "{summary}");
        assert_eq!(st["engine_errors"], 0, "{summary}");
        assert!(st["labelled"].as_u64().unwrap() > 0);
        assert_eq!(summary["teacher"]["heads"]["from"], json!({"kind": "categorical", "n": 90, "k": 8, "storage": "sparse"}));
        let mut checked = 0;
        for e in std::fs::read_dir(root.join("out")).unwrap() {
            let dir = e.unwrap().path();
            if dir.file_name().unwrap().to_string_lossy().starts_with('.') {
                continue;
            }
            let out = ShardReader::open(&dir).unwrap();
            let n = out.n_frames;
            let idx = overlay_tensor(&out, "tgt.teacher_idx", DType::I64, &[8]).rows_i64(0, n).unwrap().into_owned();
            let fi = overlay_tensor(&out, "tgt.from_probs_idx", DType::I64, &[8]).rows_i64(0, n).unwrap().into_owned();
            let fv = overlay_tensor(&out, "tgt.from_probs_val", DType::F32, &[8]).rows_f32(0, n).unwrap().into_owned();
            let ti = overlay_tensor(&out, "tgt.to_probs_idx", DType::I64, &[8]).rows_i64(0, n).unwrap().into_owned();
            let tv = overlay_tensor(&out, "tgt.to_probs_val", DType::F32, &[8]).rows_f32(0, n).unwrap().into_owned();
            let mask = out.read("tgt.engine_value_mask").unwrap().rows_i64(0, n).unwrap().into_owned();
            let from = out.read("tgt.from_idx").unwrap().rows_i64(0, n).unwrap().into_owned();
            let lf = out.read("tgt.legal_from").unwrap().rows_f32(0, n).unwrap().into_owned();
            for t in 0..n {
                if mask[t] == 0 {
                    continue;
                }
                checked += 1;
                let row = &idx[t * 8..(t + 1) * 8];
                // every scored move's from-square is a legal from-square of the frame
                for &a in row.iter().filter(|&&a| a >= 0) {
                    assert_eq!(lf[t * SQUARES + decode_action(a).0 as usize], 1.0, "frame {t}");
                }
                let fsum: f32 = fv[t * 8..(t + 1) * 8].iter().sum();
                assert!((fsum - 1.0).abs() < 1e-5, "from marginal sums to {fsum}");
                assert!(fi[t * 8..(t + 1) * 8].iter().all(|&x| (-1..90).contains(&x)));
                // the to-row conditions on the ground-truth from-square
                let tsum: f32 = tv[t * 8..(t + 1) * 8].iter().sum();
                let gt_from_mass: f32 =
                    fi[t * 8..(t + 1) * 8].iter().zip(&fv[t * 8..(t + 1) * 8]).filter(|(&i, _)| i == from[t]).map(|(_, &v)| v).sum();
                if gt_from_mass > 0.0 {
                    assert!((tsum - 1.0).abs() < 1e-5, "frame {t}: to-row sums to {tsum}");
                    for (&i, &v) in ti[t * 8..(t + 1) * 8].iter().zip(&tv[t * 8..(t + 1) * 8]) {
                        if v > 0.0 {
                            assert!(row.contains(&action_id((from[t], i))), "frame {t}: to {i} not a scored move from {}", from[t]);
                        }
                    }
                } else {
                    assert_eq!(tsum, 0.0);
                }
            }
        }
        assert!(checked > 20, "{checked} labelled frames checked");
    }

    #[test]
    fn replay_keys_and_illegal_labels() {
        let root = tmpdir("xq-replay");
        convert_fixture(&root, 3, 8);
        let dir = std::fs::read_dir(root.join("base")).unwrap().map(|e| e.unwrap().path()).find(|p| p.is_dir()).unwrap();
        let r = ShardReader::open(&dir).unwrap();
        let n = r.sample_range(0).len();
        let from = r.read("tgt.from_idx").unwrap().rows_i64(0, n).unwrap().into_owned();
        let to = r.read("tgt.to_idx").unwrap().rows_i64(0, n).unwrap().into_owned();
        let pos = replay_moves(&from, &to).unwrap();
        assert_eq!(pos.len(), n);
        assert_eq!(pos[0].command(), "position startpos");
        assert!(pos[1].command().starts_with("position startpos moves ") && pos[1].command().split(' ').count() == 4);
        // the key: the FEN at the last capture (no move counters) + the moves since
        assert_eq!(pos[0].key, "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - -");
        assert!(pos[2].key.starts_with(&pos[0].key) && pos[2].key.split(' ').count() == 6, "{}", pos[2].key);
        for p in &pos {
            assert_eq!(p.seat, p.t % 2, "Red moves first, then the sides alternate");
            assert!(p.legal.iter().filter(|l| l.1 == p.gt_target()).count() == 1);
        }
        // a label that is no legal move of its frame is a replay error, not a panic
        let mut bad = from.clone();
        bad[4] = (0..90).find(|&f| !pos[4].legal.iter().any(|l| l.1 .0 == f)).unwrap();
        assert!(replay_moves(&bad, &to).is_err());
        assert!(replay_moves(&from[..3], &to).is_err());
    }

    #[test]
    #[ignore = "needs a Pikafish binary (scripts/engines/fetch_engines.sh pikafish or PIKAFISH)"]
    fn real_pikafish_labels_the_fixture() {
        let Some(exe) = engine_path("PIKAFISH", "pikafish") else { return };
        let root = tmpdir("xq-pikafish");
        convert_fixture(&root, 2, 8);
        let mut o = quiet_opts(&root, crate::label::engine::EngineCommand::process(&exe, &[]));
        o.limits = Limits::parse(&["nodes=20000".into()]).unwrap();
        let summary = serde_json::to_value(run(XiangqiLabeller::default(), &o).unwrap()).unwrap();
        let st = &summary["stats"];
        assert_eq!((st["verify_failed"].as_u64(), st["engine_errors"].as_u64()), (Some(0), Some(0)), "{summary}");
        assert!(summary["teacher"]["engine"].as_str().unwrap().starts_with("Pikafish"), "{summary}");
    }
}
