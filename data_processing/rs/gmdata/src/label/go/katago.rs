//! KataGo's JSON analysis engine (`katago analysis -config C -model M`): coordinates,
//! query lines, response parsing, the session (handshake + one query at a time) and
//! the generated analysis config.
//!
//! Protocol (one JSON object per line, <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>):
//!
//! * query → one response line per `analyzeTurns` entry, in any order, each with the
//!   query's `id`, `turnNumber`, `moveInfos` (per searched move: `move`, `visits`,
//!   `edgeVisits`, `winrate`, `scoreLead`, `prior`, …) and `rootInfo` (`winrate`,
//!   `scoreLead`, `currentPlayer`, …). With `reportAnalysisWinratesAs = SIDETOMOVE`
//!   winrates and scores are from the side to move at that turn. That is a config
//!   setting only — KataGo ignores a query field of that name (with an "unused field"
//!   warning), and its own analysis_example.cfg says `BLACK` — so every process gets
//!   [`FORCED_CONFIG`] through `-override-config` ([`override_args`]) and every new
//!   session checks the perspective on a probe position ([`KataSession::check_side_to_move`]).
//! * `{"id":…,"error":…}` = the query was rejected (e.g. an illegal move),
//!   `{"id":…,"warning":…}` = accepted with a warning (ignored here),
//!   `"noResults": true` = the turn was not analysed (terminated).
//! * `{"id":…,"action":"query_version"}` → `{"version":…}` (the handshake: KataGo
//!   prints nothing on stdout at startup and reads stdin only once the model is
//!   loaded); `query_models` → the network's `internalName`.
//! * The side to move at turn t is `initialPlayer` for t = 0 and the opponent of the
//!   colour of move t−1 otherwise (move colours need not alternate; KataGo accepts any).
//!
//! Coordinates: GTP — column letters `A…T` skipping `I` from the left, row numbers
//! `1…N` from the BOTTOM. Shards (SGF) count rows from the TOP: point `row·N + col`
//! with row 0 = SGF row `a` = GTP row N, so SGF `pd` = (3, 15) = GTP `Q16`.

use super::super::engine::{Engine, EngineCommand, EngineError};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

/// GTP column letters (no `I`); boards up to 25×25 have GTP names.
pub const GTP_LETTERS: &[u8; 25] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// GTP name (`Q16`) of point `idx = row·n + col` (row 0 = top).
pub fn gtp_of(idx: usize, n: usize) -> String {
    let (row, col) = (idx / n, idx % n);
    format!("{}{}", GTP_LETTERS[col] as char, n - row)
}

/// Point of a GTP vertex: `Some(None)` = pass, `Some(Some(idx))` = `row·n + col`,
/// `None` = not a vertex of an n×n board. Case-insensitive.
pub fn parse_gtp(s: &str, n: usize) -> Option<Option<usize>> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("pass") {
        return Some(None);
    }
    let mut chars = s.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    let col = GTP_LETTERS.iter().position(|&c| c as char == letter)?;
    let number: usize = chars.as_str().parse().ok()?;
    if col >= n || number == 0 || number > n {
        return None;
    }
    Some(Some((n - number) * n + col))
}

/// `"B"` / `"W"` of a crate::go colour.
pub fn colour_str(c: u8) -> &'static str {
    if c == crate::go::BLACK {
        "B"
    } else {
        "W"
    }
}

/// Everything one query needs from a game (absolute colours, shard point indices).
#[derive(Debug, Clone, PartialEq)]
pub struct QueryGame {
    pub n: usize,
    /// stones on the board at frame 0 (handicap / setup stones), (colour, point)
    pub initial: Vec<(u8, usize)>,
    /// side to move at frame 0
    pub initial_player: u8,
    /// every recorded move: (colour, point; None = pass)
    pub moves: Vec<(u8, Option<usize>)>,
    /// KataGo `rules` (a shorthand string like "chinese", or a rules object)
    pub rules: Value,
    /// white's komi as sent (a multiple of 0.5)
    pub komi: f64,
}

/// The query line for `turns` of `g` (moves up to the last analysed turn).
pub fn query_line(id: &str, g: &QueryGame, turns: &[usize], max_visits: u64) -> String {
    let last = turns.iter().copied().max().unwrap_or(0).min(g.moves.len());
    let vertex = |p: Option<usize>| p.map(|i| gtp_of(i, g.n)).unwrap_or_else(|| "pass".to_string());
    let q = json!({
        "id": id,
        "boardXSize": g.n,
        "boardYSize": g.n,
        "rules": g.rules,
        "komi": g.komi,
        "initialStones": g.initial.iter().map(|&(c, i)| json!([colour_str(c), gtp_of(i, g.n)])).collect::<Vec<_>>(),
        "initialPlayer": colour_str(g.initial_player),
        "moves": g.moves[..last].iter().map(|&(c, p)| json!([colour_str(c), vertex(p)])).collect::<Vec<_>>(),
        "analyzeTurns": turns,
        "maxVisits": max_visits,
    });
    q.to_string()
}

/// Config values the response parsing relies on, forced on every KataGo process through
/// `-override-config` whatever config file it runs with (see the module docs): winrates
/// and score leads from the side to move.
pub const FORCED_CONFIG: &[(&str, &str)] = &[("reportAnalysisWinratesAs", "SIDETOMOVE")];

/// `args` + `-override-config K=V,…` carrying `options` (`--engine-option`, in order) and
/// [`FORCED_CONFIG`]; an `-override-config` already among `args` is extended instead of
/// repeated. A forced key may only be given its forced value.
pub fn override_args(args: &[String], options: &[(String, String)]) -> Result<Vec<String>> {
    let mut out = args.to_vec();
    let existing = out.iter().position(|a| a == "-override-config" || a == "--override-config");
    let mut kv: Vec<(String, String)> = Vec::new();
    if let Some(i) = existing {
        let v = out.get(i + 1).with_context(|| format!("{} without a value", out[i]))?;
        for item in v.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let (k, v) = item.split_once('=').with_context(|| format!("-override-config item {item:?} is not KEY=VALUE"))?;
            kv.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    for (k, v) in options {
        if k.is_empty() || k.contains([',', '=']) || v.contains([',', '=']) {
            bail!("--engine-option {k}={v}: KataGo config keys / values cannot contain ',' or '='");
        }
        kv.push((k.clone(), v.clone()));
    }
    for (fk, fv) in FORCED_CONFIG {
        if let Some((_, v)) = kv.iter().find(|(k, v)| k == fk && !v.eq_ignore_ascii_case(fv)) {
            bail!("{fk}={v}: the go labeller parses KataGo's answers as {fk}={fv} and sets it itself");
        }
        if !kv.iter().any(|(k, _)| k == fk) {
            kv.push((fk.to_string(), fv.to_string()));
        }
    }
    let joined = kv.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",");
    match existing {
        Some(i) => out[i + 1] = joined,
        None => {
            out.push("-override-config".into());
            out.push(joined);
        }
    }
    Ok(out)
}

/// One searched move of a response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveInfo {
    /// shard point (row·N + col) or N² for pass
    pub action: i64,
    /// side-to-move winrate (draws count ½)
    pub winrate: f32,
    /// side-to-move score lead (points)
    pub score_lead: f32,
    /// root visits invested in the move (`edgeVisits`; `visits` for engines without it)
    pub visits: f32,
    /// raw network policy prior
    pub prior: f32,
}

/// One analysed turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnResult {
    pub moves: Vec<MoveInfo>,
    /// rootInfo.winrate (side to move)
    pub root_winrate: f32,
    /// rootInfo.currentPlayer
    pub current_player: u8,
}

fn num(v: &Value, key: &str) -> Result<f64> {
    v.get(key).and_then(Value::as_f64).with_context(|| format!("missing number {key:?}"))
}

/// Parse one response line's analysis (`n` = board size).
pub fn parse_turn(v: &Value, n: usize) -> Result<TurnResult> {
    let root = v.get("rootInfo").context("response without rootInfo")?;
    let current_player = match root.get("currentPlayer").and_then(Value::as_str) {
        Some("B") | Some("b") => crate::go::BLACK,
        Some("W") | Some("w") => crate::go::WHITE,
        other => bail!("rootInfo.currentPlayer {other:?}"),
    };
    let root_winrate = num(root, "winrate")? as f32;
    let infos = v.get("moveInfos").and_then(Value::as_array).context("response without moveInfos")?;
    let mut moves = Vec::with_capacity(infos.len());
    let mut seen = HashSet::new();
    for m in infos {
        let mv = m.get("move").and_then(Value::as_str).context("moveInfo without move")?;
        let action = match parse_gtp(mv, n) {
            Some(None) => (n * n) as i64,
            Some(Some(i)) => i as i64,
            None => bail!("moveInfo move {mv:?} is not a vertex of a {n}×{n} board"),
        };
        if !seen.insert(action) {
            bail!("move {mv} reported twice");
        }
        let visits = m.get("edgeVisits").and_then(Value::as_f64).map(Ok).unwrap_or_else(|| num(m, "visits"))?;
        moves.push(MoveInfo {
            action,
            winrate: num(m, "winrate")? as f32,
            score_lead: m.get("scoreLead").and_then(Value::as_f64).unwrap_or(f64::NAN) as f32,
            visits: visits as f32,
            prior: m.get("prior").and_then(Value::as_f64).unwrap_or(f64::NAN) as f32,
        });
    }
    Ok(TurnResult { moves, root_winrate, current_player })
}

/// A running `katago analysis` process (or a fake), one query at a time.
pub struct KataSession {
    engine: Engine,
    /// `query_version` → version
    pub version: String,
    /// `query_models` → the network's internal name (None if unsupported)
    pub model: Option<String>,
    next: u64,
}

/// Wait for the response line with `id` (other lines are ignored), within `timeout`.
fn wait_for(engine: &mut Engine, id: &str, timeout: Duration) -> Result<Value> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let line = engine.recv(left)?;
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        if v.get("id").and_then(Value::as_str) == Some(id) {
            return Ok(v);
        }
    }
}

impl KataSession {
    /// Spawn and handshake (`query_version`, waiting up to `startup` for the model to
    /// load; then `query_models`).
    pub fn open(cmd: EngineCommand, startup: Duration, timeout: Duration) -> Result<Self> {
        let mut engine = Engine::spawn(cmd)?;
        engine.send(r#"{"id":"gm-version","action":"query_version"}"#)?;
        let v = wait_for(&mut engine, "gm-version", startup).context("KataGo handshake (query_version)")?;
        if let Some(e) = v.get("error") {
            bail!("KataGo query_version failed: {e}");
        }
        let version = v.get("version").and_then(Value::as_str).context("query_version without version")?.to_string();
        engine.send(r#"{"id":"gm-models","action":"query_models"}"#)?;
        let m = wait_for(&mut engine, "gm-models", timeout).context("KataGo query_models")?;
        let model = m
            .get("models")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(|x| x.get("internalName").and_then(Value::as_str))
            .map(str::to_string);
        Ok(KataSession { engine, version, model, next: 0 })
    }

    /// Analyse `turns` of `g` with `max_visits` visits each; one [`TurnResult`] per turn
    /// (in the order of `turns`), all within `deadline`.
    pub fn analyze(&mut self, g: &QueryGame, turns: &[usize], max_visits: u64, deadline: Duration) -> Result<Vec<TurnResult>> {
        if turns.is_empty() {
            return Ok(Vec::new());
        }
        self.next += 1;
        let id = format!("gm-{}", self.next);
        self.engine.send(&query_line(&id, g, turns, max_visits))?;
        let want: HashSet<usize> = turns.iter().copied().collect();
        let mut got: BTreeMap<usize, TurnResult> = BTreeMap::new();
        let end = Instant::now() + deadline;
        while got.len() < want.len() {
            let left = end.saturating_duration_since(Instant::now());
            let line = self.engine.recv(left).map_err(|e| match e.downcast::<EngineError>() {
                Ok(EngineError::Timeout { .. }) => EngineError::Timeout {
                    waited: deadline,
                    context: format!("KataGo query {id}: {} of {} turns answered", got.len(), want.len()),
                }
                .into(),
                Ok(other) => other.into(),
                Err(e) => e,
            })?;
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            if v.get("id").and_then(Value::as_str) != Some(id.as_str()) {
                continue; // a stale answer or an unrelated message
            }
            if let Some(e) = v.get("error") {
                bail!("KataGo rejected query {id}: {e} (field {})", v.get("field").unwrap_or(&Value::Null));
            }
            if v.get("warning").is_some() && v.get("turnNumber").is_none() {
                continue;
            }
            let turn = v.get("turnNumber").and_then(Value::as_u64).context("response without turnNumber")? as usize;
            if v.get("noResults").and_then(Value::as_bool) == Some(true) {
                bail!("KataGo returned no results for turn {turn} of query {id}");
            }
            if !want.contains(&turn) {
                bail!("KataGo answered turn {turn}, which query {id} did not ask for");
            }
            let r = parse_turn(&v, g.n).with_context(|| format!("query {id} turn {turn}"))?;
            got.insert(turn, r);
        }
        Ok(turns.iter().map(|t| got[t].clone()).collect())
    }

    /// "KataGo <version>" + the network's internal name when known.
    pub fn engine_id(&self, model_fallback: Option<&str>) -> String {
        match self.model.as_deref().or(model_fallback) {
            Some(m) => format!("KataGo {} {m}", self.version),
            None => format!("KataGo {}", self.version),
        }
    }

    /// Config keys KataGo reported as unused at startup (its stderr: "Unused key 'K' in …";
    /// typos, or keys this KataGo version / backend does not know).
    pub fn unused_config_keys(&self) -> Vec<String> {
        unused_keys(&self.engine.stderr_tail())
    }

    /// Fail unless winrates come from the side to move: an empty `n`×`n` board with komi
    /// 150 (white wins by any measure), black to move at turn 0 and — after black passes —
    /// white at turn 1. The side to move reads ≈0 then ≈1; black's view gives ≈0 twice and
    /// white's ≈1 twice (a config's `reportAnalysisWinratesAs = BLACK / WHITE`).
    pub fn check_side_to_move(&mut self, n: usize, timeout: Duration) -> Result<()> {
        use crate::go::{BLACK, WHITE};
        let g = QueryGame { n, initial: Vec::new(), initial_player: BLACK, moves: vec![(BLACK, None)], rules: json!("chinese"), komi: 150.0 };
        let r = self.analyze(&g, &[0, 1], 2, timeout).context("KataGo perspective check")?;
        let (b, w) = (&r[0], &r[1]);
        if b.current_player != BLACK || w.current_player != WHITE {
            bail!("KataGo perspective check: sides to move {} / {}, expected B / W", colour_str(b.current_player), colour_str(w.current_player));
        }
        if !(b.root_winrate < 0.5 && w.root_winrate > 0.5) {
            bail!(
                "KataGo does not report winrates from the side to move (reportAnalysisWinratesAs): with komi 150 it gives \
                 black to move {:.3} and white to move {:.3}, expected about 0 and 1",
                b.root_winrate,
                w.root_winrate
            );
        }
        Ok(())
    }
}

/// Keys of KataGo's "Unused key 'K' in …" startup warnings in `stderr` (deduplicated, in
/// order).
pub fn unused_keys(stderr: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in stderr.lines() {
        if let Some(rest) = line.split("Unused key '").nth(1) {
            if let Some((k, _)) = rest.split_once('\'') {
                if !out.iter().any(|x| x == k) {
                    out.push(k.to_string());
                }
            }
        }
    }
    out
}

/// Settings of the generated analysis config (`--engine-option K=V` overrides them via
/// `-override-config`). Semantic keys change the teacher and go into the teacher meta;
/// performance keys (threads, batch, cache) do not.
pub fn config_settings(analysis_threads: usize, search_threads: usize, nn_batch: usize, nn_cache_pow2: u32) -> Vec<(String, String, bool)> {
    let s = |k: &str, v: String, semantic: bool| (k.to_string(), v, semantic);
    vec![
        // semantic
        s("reportAnalysisWinratesAs", "SIDETOMOVE".into(), true),
        s("wideRootNoise", "0.0".into(), true),
        s("ignorePreRootHistory", "false".into(), true),
        s("nnRandomize", "true".into(), true),
        s("conservativePass", "true".into(), true),
        s("preventCleanupPhase", "true".into(), true),
        // output size / performance
        s("analysisPVLen", "1".into(), false),
        s("numAnalysisThreads", analysis_threads.max(1).to_string(), false),
        s("numSearchThreadsPerAnalysisThread", search_threads.max(1).to_string(), false),
        s("nnMaxBatchSize", nn_batch.max(1).to_string(), false),
        s("nnCacheSizePowerOfTwo", nn_cache_pow2.to_string(), false),
        s("nnMutexPoolSizePowerOfTwo", nn_cache_pow2.saturating_sub(4).clamp(10, 17).to_string(), false),
        s("maxVisits", "100".into(), false), // every query sets maxVisits
    ]
}

/// KataGo's recommended Metal layout (analysis_example.cfg "mux mode"): two GPU
/// (MPSGraph, FP32) and two Neural Engine (CoreML, FP16) server threads — ~2.7× the
/// GPU-only throughput of b18c384nbt on an M4 Pro.
pub const METAL_MUX: &[(&str, &str)] = &[
    ("numNNServerThreadsPerModel", "4"),
    ("metalDeviceToUseThread0", "0"),
    ("metalDeviceToUseThread1", "0"),
    ("metalDeviceToUseThread2", "100"),
    ("metalDeviceToUseThread3", "100"),
];

/// Whether KataGo config key `k` only affects speed, device placement or logging — the
/// keys `--perf-option` accepts (kept out of the teacher meta). Search parameters
/// (`cpuctExploration`, `rootSymmetryPruning`, `useEvalCache`, …) change the teacher and
/// must go through `--engine-option`.
pub fn is_perf_key(k: &str) -> bool {
    const EXACT: &[&str] = &[
        "numNNServerThreadsPerModel",
        "numAnalysisThreads",
        "numSearchThreadsPerAnalysisThread",
        "numEigenThreadsPerModel",
        "nnMaxBatchSize",
        "nnCacheSizePowerOfTwo",
        "nnMutexPoolSizePowerOfTwo",
        "analysisPVLen",
        "homeDataDir",
        "warnUnusedFields",
    ];
    // backend device / precision / tuning keys (cudaDeviceToUseThread0, openclUseFP16,
    // metalDeviceToUseThread2, trtUseFP16, …) and logging (logDir, logToStderr, …)
    const PREFIX: &[&str] = &["cuda", "trt", "opencl", "metal", "coreml", "rocm", "hip", "onnx", "openvino", "eigen", "log"];
    EXACT.contains(&k) || PREFIX.iter().any(|p| k.starts_with(p))
}

/// Add / replace speed-only settings (never recorded in the teacher meta).
pub fn apply_perf(settings: &mut Vec<(String, String, bool)>, perf: &[(String, String)]) {
    for (k, v) in perf {
        match settings.iter_mut().find(|s| s.0 == *k) {
            Some(s) => {
                s.1 = v.clone();
                s.2 = false;
            }
            None => settings.push((k.clone(), v.clone(), false)),
        }
    }
}

/// The config file text of [`config_settings`].
pub fn config_text(settings: &[(String, String, bool)]) -> String {
    let mut t = String::from("# generated by `gmdata label go` (data_processing/rs/gmdata/src/label/go.rs); do not edit\n");
    for (k, v, _) in settings {
        t.push_str(&format!("{k} = {v}\n"));
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gtp_coordinates_pinned() {
        // SGF "pd" (col p = 15, row d = 3) is the upper-right star point Q16
        assert_eq!(gtp_of(3 * 19 + 15, 19), "Q16");
        assert_eq!(parse_gtp("Q16", 19), Some(Some(3 * 19 + 15)));
        // corners: SGF "aa" = top-left = A19, SGF "as" = bottom-left = A1, SGF "ss" = T1
        assert_eq!(gtp_of(0, 19), "A19");
        assert_eq!(gtp_of(18 * 19, 19), "A1");
        assert_eq!(gtp_of(18 * 19 + 18, 19), "T1");
        // I is skipped: col 8 = J; SGF "ij" = (9, 8) = J10 (tengen of 19x19 is K10 = (9, 9))
        assert_eq!(gtp_of(9 * 19 + 8, 19), "J10");
        assert_eq!(gtp_of(9 * 19 + 9, 19), "K10");
        assert_eq!(parse_gtp("k10", 19), Some(Some(9 * 19 + 9)));
        assert_eq!(parse_gtp("pass", 19), Some(None));
        assert_eq!(parse_gtp("PASS", 9), Some(None));
        for bad in ["I5", "U1", "A0", "A20", "", "Q", "Q16x", "resign"] {
            assert_eq!(parse_gtp(bad, 19), None, "{bad}");
        }
        assert_eq!(parse_gtp("J9", 9), Some(Some(8)), "9x9: J = col 8, row 9 = top");
        assert_eq!(parse_gtp("K1", 9), None);
        for n in [5usize, 9, 13, 19, 25] {
            for i in 0..n * n {
                assert_eq!(parse_gtp(&gtp_of(i, n), n), Some(Some(i)), "n={n} i={i}");
            }
        }
    }

    #[test]
    fn query_line_layout() {
        let g = QueryGame {
            n: 9,
            initial: vec![(crate::go::BLACK, 2 * 9 + 2)],
            initial_player: crate::go::WHITE,
            moves: vec![(crate::go::WHITE, Some(4 * 9 + 4)), (crate::go::BLACK, None), (crate::go::WHITE, Some(0))],
            rules: json!("japanese"),
            komi: 6.5,
        };
        let v: Value = serde_json::from_str(&query_line("x", &g, &[0, 2], 50)).unwrap();
        assert_eq!(v["initialStones"], json!([["B", "C7"]]));
        assert_eq!(v["initialPlayer"], "W");
        assert_eq!(v["moves"], json!([["W", "E5"], ["B", "pass"]]), "moves up to the last analysed turn");
        assert_eq!((v["analyzeTurns"].clone(), v["maxVisits"].clone(), v["komi"].clone()), (json!([0, 2]), json!(50), json!(6.5)));
        assert_eq!((v["rules"].clone(), v["boardXSize"].clone()), (json!("japanese"), json!(9)));
        // KataGo has no such query field (it warns "unused field" and ignores it): the
        // perspective is config-only, forced through -override-config
        assert!(v.get("reportAnalysisWinratesAs").is_none());
    }

    #[test]
    fn parse_real_response_line() {
        // trimmed from KataGo 1.18.2 output (19x19)
        let line = r#"{"id":"q2","isDuringSearch":false,"moveInfos":[{"edgeVisits":10,"edgeWeight":34.2,"lcb":0.96,"move":"D4","order":0,"playSelectionValue":37.9,"prior":0.158677042,"pv":["D4","Q4"],"scoreLead":12.991228,"scoreMean":12.99,"scoreSelfplay":16.59,"scoreStdev":15.98,"utility":1.027,"utilityLcb":0.955,"visits":10,"weight":34.2,"winrate":0.98952233},{"edgeVisits":9,"isSymmetryOf":"Q4","move":"D16","order":2,"prior":0.12722832,"scoreLead":13.06,"visits":9,"winrate":0.989250795},{"edgeVisits":0,"move":"pass","order":3,"prior":0.001,"scoreLead":-3.0,"visits":1,"winrate":0.2}],"rootInfo":{"currentPlayer":"B","scoreLead":12.9814,"visits":20,"winrate":0.989282585},"turnNumber":2}"#;
        let r = parse_turn(&serde_json::from_str(line).unwrap(), 19).unwrap();
        assert_eq!(r.current_player, crate::go::BLACK);
        assert!((r.root_winrate - 0.989_282_6).abs() < 1e-6);
        assert_eq!(r.moves.len(), 3);
        assert_eq!(r.moves[0].action, (15 * 19 + 3) as i64, "D4 = row 15 from the top, col 3");
        assert_eq!(r.moves[1].action, (3 * 19 + 3) as i64, "D16");
        assert_eq!(r.moves[2].action, 361, "pass = N²");
        assert_eq!((r.moves[0].visits, r.moves[2].visits), (10.0, 0.0), "edgeVisits, not visits");
        assert!((r.moves[0].prior - 0.158_677).abs() < 1e-6 && (r.moves[0].score_lead - 12.991_228).abs() < 1e-5);
        let dup = line.replace(r#""move":"D16""#, r#""move":"D4""#);
        assert!(parse_turn(&serde_json::from_str(&dup).unwrap(), 19).is_err());
        let off = line.replace(r#""move":"D16""#, r#""move":"Z30""#);
        assert!(parse_turn(&serde_json::from_str(&off).unwrap(), 19).is_err());
    }

    #[test]
    fn side_to_move_is_forced_on_every_command_line() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        let kv = |k: &str, v: &str| (k.to_string(), v.to_string());
        let base = s(&["analysis", "-config", "a.cfg", "-model", "m.bin.gz"]);
        assert_eq!(override_args(&base, &[]).unwrap()[5..], s(&["-override-config", "reportAnalysisWinratesAs=SIDETOMOVE"]));
        let with = override_args(&base, &[kv("wideRootNoise", "0.04")]).unwrap();
        assert_eq!(with[5..], s(&["-override-config", "wideRootNoise=0.04,reportAnalysisWinratesAs=SIDETOMOVE"]));
        // an -override-config among --engine-arg is extended, not repeated
        let own = s(&["analysis", "-override-config", "maxTime=5, rootSymmetryPruning=false", "-model", "m"]);
        let got = override_args(&own, &[kv("wideRootNoise", "0")]).unwrap();
        assert_eq!(got, s(&["analysis", "-override-config", "maxTime=5,rootSymmetryPruning=false,wideRootNoise=0,reportAnalysisWinratesAs=SIDETOMOVE", "-model", "m"]));
        assert_eq!(got.iter().filter(|a| *a == "-override-config").count(), 1);
        // the forced value itself is fine (any case); another perspective is refused
        assert!(override_args(&base, &[kv("reportAnalysisWinratesAs", "sidetomove")]).unwrap().last().unwrap().ends_with("=sidetomove"));
        for bad in ["BLACK", "WHITE"] {
            assert!(override_args(&base, &[kv("reportAnalysisWinratesAs", bad)]).is_err());
            let own = s(&["analysis", "-override-config", &format!("reportAnalysisWinratesAs={bad}")]);
            assert!(override_args(&own, &[]).is_err());
        }
        assert!(override_args(&base, &[kv("a,b", "1")]).is_err() && override_args(&base, &[kv("a", "1=2")]).is_err());
        assert!(override_args(&s(&["analysis", "-override-config"]), &[]).is_err());
    }

    #[test]
    fn unused_config_keys_from_katago_stderr() {
        // KataGo 1.18.2 startup (trimmed): each warning is logged twice
        let err = "2026-09-26 12:29:50+0100: Running with following config:\n\
                   wideRootNoize = 0.04\n\
                   2026-09-26 12:29:51+0100: WARNING: Config had unused keys! You may have a typo, an option you specified is being unused from a.cfg and/or command-line and query overrides\n\
                   2026-09-26 12:29:51+0100: WARNING: Unused key 'wideRootNoize' in a.cfg and/or command-line and query overrides\n\
                   2026-09-26 12:29:51+0100: WARNING: Unused key 'cudaUseFP16' in a.cfg and/or command-line and query overrides\n\
                   WARNING: Unused key 'wideRootNoize' in a.cfg and/or command-line and query overrides\n\
                   2026-09-26 12:29:51+0100: Started, ready to begin handling requests\n";
        assert_eq!(unused_keys(err), vec!["wideRootNoize".to_string(), "cudaUseFP16".to_string()]);
        assert!(unused_keys("Started, ready to begin handling requests").is_empty());
    }

    #[test]
    fn generated_config() {
        let s = config_settings(8, 1, 8, 18);
        let t = config_text(&s);
        assert!(t.contains("numAnalysisThreads = 8\n") && t.contains("reportAnalysisWinratesAs = SIDETOMOVE\n"));
        assert!(t.contains("wideRootNoise = 0.0\n") && t.contains("ignorePreRootHistory = false\n"));
        let keys: HashSet<&str> = s.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(keys.len(), s.len(), "unique keys (KataGo rejects duplicates)");
    }
}
