//! A fake KataGo analysis engine for tests: speaks the JSON protocol of
//! [`super::katago`] with computed answers, so the go labeller is tested end to end
//! without a KataGo binary or network.
//!
//! It replays every query on its own board (GTP coordinates parsed here by a separate,
//! arithmetic implementation — `A…T` without `I`, rows from the bottom — and the
//! converter's rules), rejecting illegal moves like KataGo
//! (`{"id":…,"error":"Illegal move …","field":"moves"}`), and answers each analysed
//! turn with every legal move of the side to move scored by a transparent rule:
//!
//! * winrate = 0.45 + 0.1·h + 0.3·min(captured, 3)/3 (h ∈ [0,1) a hash of point and
//!   turn), so a capturing move always beats every quiet move and bigger captures win;
//!   pass scores 0.3 (0.5 when it is the only legal move);
//! * the top 12 moves (by winrate) get `edgeVisits` ∝ exp((wr − best)/0.03) out of
//!   `maxVisits` (at least one each), `prior` ∝ exp(2h), `scoreLead` = 40·(wr − ½);
//! * `rootInfo`: `currentPlayer` (initialPlayer, then the opponent of the previous
//!   move's colour), `winrate` = the best move's.
//!
//! Turns are answered in reverse order (the real engine answers in completion order),
//! preceded by a warning line. `query_version` answers `1.18.2-fake`, `query_models`
//! the network `fake-b1c1`. Failure injection: crash (exit) / hang / error reply on
//! the n-th analysis query of a process.
//!
//! As a child process: `gmdata label fake-engine katago [--crash-on N]`.

use super::super::engine::LineHandler;
use crate::go::{opponent, Board, BLACK, EMPTY, WHITE};
use serde_json::{json, Value};
use std::time::Duration;

/// Reported version.
pub const FAKE_VERSION: &str = "1.18.2-fake";
/// Reported network.
pub const FAKE_MODEL: &str = "fake-b1c1";
/// Moves reported per turn.
pub const FAKE_TOP: usize = 12;

/// The fake engine (see the module docs).
#[derive(Default)]
pub struct KataFake {
    queries: usize,
    crash_on: Option<usize>,
    error_on: Option<usize>,
    hang_on: Option<(usize, Duration)>,
}

/// Point of a GTP vertex, computed arithmetically (independent of
/// [`super::katago::parse_gtp`]): `Some(None)` = pass.
fn vertex(s: &str, n: usize) -> Option<Option<usize>> {
    if s.eq_ignore_ascii_case("pass") {
        return Some(None);
    }
    let b = s.as_bytes();
    let letter = b.first()?.to_ascii_uppercase();
    if !letter.is_ascii_uppercase() || letter == b'I' {
        return None;
    }
    let col = (letter - b'A') as usize - usize::from(letter > b'I');
    let number: usize = std::str::from_utf8(&b[1..]).ok()?.parse().ok()?;
    (col < n && (1..=n).contains(&number)).then_some(Some((n - number) * n + col))
}

fn name(idx: usize, n: usize) -> String {
    let col = idx % n;
    let letter = b'A' + col as u8 + u8::from(col >= 8);
    format!("{}{}", letter as char, n - idx / n)
}

fn colour(v: &Value) -> Option<u8> {
    match v.as_str()? {
        "B" | "b" => Some(BLACK),
        "W" | "w" => Some(WHITE),
        _ => None,
    }
}

fn hash01(a: u64, b: u64) -> f64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_add(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

fn stones(b: &Board) -> usize {
    b.g.iter().filter(|&&v| v != EMPTY).count()
}

impl KataFake {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exit on the `n`-th analysis query (1-based, per process).
    pub fn crash_on(mut self, n: usize) -> Self {
        self.crash_on = Some(n);
        self
    }

    /// Reply with an error line to the `n`-th analysis query.
    pub fn error_on(mut self, n: usize) -> Self {
        self.error_on = Some(n);
        self
    }

    /// Sleep `d` before answering the `n`-th analysis query.
    pub fn hang_on(mut self, n: usize, d: Duration) -> Self {
        self.hang_on = Some((n, d));
        self
    }

    fn analyse(&self, q: &Value) -> Result<Vec<String>, (String, &'static str)> {
        let id = q["id"].clone();
        let n = q["boardXSize"].as_u64().ok_or(("boardXSize missing".to_string(), "boardXSize"))? as usize;
        if q["boardYSize"].as_u64() != Some(n as u64) || !(2..=25).contains(&n) {
            return Err(("non-square board".into(), "boardYSize"));
        }
        if q["reportAnalysisWinratesAs"] != "SIDETOMOVE" {
            return Err(("the fake only reports SIDETOMOVE".into(), "reportAnalysisWinratesAs"));
        }
        match q["rules"].as_str() {
            Some("chinese") | Some("japanese") | Some("korean") | Some("aga") | Some("tromp-taylor") | Some("new-zealand") => {}
            _ if q["rules"].is_object() => {}
            _ => return Err(("unknown rules".into(), "rules")),
        }
        let komi = q["komi"].as_f64().ok_or(("komi missing".to_string(), "komi"))?;
        if (komi * 2.0).fract() != 0.0 || komi.abs() > 150.0 {
            return Err(("komi must be an integer or half-integer".into(), "komi"));
        }
        let visits = q["maxVisits"].as_u64().unwrap_or(100).max(1) as f64;
        let mut board = Board::new(n as i32);
        for s in q["initialStones"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            let (Some(c), Some(Some(p))) = (colour(&s[0]), s[1].as_str().and_then(|x| vertex(x, n))) else {
                return Err((format!("bad initial stone {s}"), "initialStones"));
            };
            if !board.setup(c, &[((p / n) as i32, (p % n) as i32)]) {
                return Err((format!("initial stone on an occupied point {s}"), "initialStones"));
            }
        }
        let moves = q["moves"].as_array().cloned().unwrap_or_default();
        let initial_player = match q.get("initialPlayer") {
            Some(v) => colour(v).ok_or(("bad initialPlayer".to_string(), "initialPlayer"))?,
            None => moves.first().and_then(|m| colour(&m[0])).unwrap_or(BLACK),
        };
        let turns: Vec<usize> = q["analyzeTurns"]
            .as_array()
            .ok_or(("analyzeTurns missing".to_string(), "analyzeTurns"))?
            .iter()
            .map(|t| t.as_u64().map(|x| x as usize))
            .collect::<Option<_>>()
            .ok_or(("bad analyzeTurns".to_string(), "analyzeTurns"))?;
        if turns.iter().any(|&t| t > moves.len()) {
            return Err(("analyzeTurns beyond the moves".into(), "analyzeTurns"));
        }
        // replay, snapshotting the analysed turns
        let mut snaps: Vec<(usize, Board, u8)> = Vec::new();
        let mut to_move = initial_player;
        for t in 0..=moves.len() {
            if turns.contains(&t) {
                snaps.push((t, board.clone(), to_move));
            }
            if t == moves.len() {
                break;
            }
            let m = &moves[t];
            let (Some(c), Some(p)) = (colour(&m[0]), m[1].as_str().and_then(|x| vertex(x, n))) else {
                return Err((format!("bad move {t}: {m}"), "moves"));
            };
            if !board.play(c, p.map(|i| ((i / n) as i32, (i % n) as i32))) {
                return Err((format!("Illegal move {t}: {}", m[1].as_str().unwrap_or("?")), "moves"));
            }
            to_move = opponent(c);
        }
        let mut out = vec![json!({"id": id, "warning": "fake engine: no real search", "field": "maxVisits"}).to_string()];
        snaps.reverse();
        for (t, b, pla) in snaps {
            let before = stones(&b);
            let mut scored: Vec<(Option<usize>, f64, f64)> = Vec::new(); // (point, winrate, h)
            for i in 0..n * n {
                let mut after = b.clone();
                if b.g[i] != EMPTY || !after.play(pla, Some(((i / n) as i32, (i % n) as i32))) {
                    continue;
                }
                let captured = (before + 1 - stones(&after)) as f64;
                let h = hash01(i as u64, t as u64);
                scored.push((Some(i), 0.45 + 0.1 * h + 0.3 * captured.min(3.0) / 3.0, h));
            }
            let only_pass = scored.is_empty();
            scored.push((None, if only_pass { 0.5 } else { 0.3 }, 0.0));
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            scored.truncate(FAKE_TOP);
            let best = scored[0].1;
            let w: Vec<f64> = scored.iter().map(|s| ((s.1 - best) / 0.03).exp()).collect();
            let (ws, ps): (f64, f64) = (w.iter().sum(), scored.iter().map(|s| (2.0 * s.2).exp()).sum());
            let infos: Vec<Value> = scored
                .iter()
                .zip(&w)
                .enumerate()
                .map(|(order, (s, wi))| {
                    let v = (visits * wi / ws).floor().max(1.0);
                    json!({
                        "move": s.0.map(|i| name(i, n)).unwrap_or_else(|| "pass".into()),
                        "order": order, "visits": v, "edgeVisits": v, "winrate": s.1,
                        "scoreLead": 40.0 * (s.1 - 0.5), "prior": (2.0 * s.2).exp() / ps,
                        "pv": [s.0.map(|i| name(i, n)).unwrap_or_else(|| "pass".into())],
                    })
                })
                .collect();
            out.push(
                json!({
                    "id": id, "isDuringSearch": false, "turnNumber": t, "moveInfos": infos,
                    "rootInfo": {"currentPlayer": if pla == BLACK { "B" } else { "W" }, "winrate": best,
                                 "scoreLead": 40.0 * (best - 0.5), "visits": visits},
                })
                .to_string(),
            );
        }
        Ok(out)
    }
}

impl LineHandler for KataFake {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        let Ok(q) = serde_json::from_str::<Value>(line) else {
            return Some(vec![json!({"error": "could not parse json", "field": ""}).to_string()]);
        };
        let id = q["id"].clone();
        match q["action"].as_str() {
            Some("query_version") => {
                return Some(vec![json!({"id": id, "action": "query_version", "version": FAKE_VERSION, "git_hash": "fake"}).to_string()])
            }
            Some("query_models") => {
                return Some(vec![json!({"id": id, "action": "query_models", "models": [{"name": "/nets/fake.bin.gz", "internalName": FAKE_MODEL, "version": 15}]}).to_string()])
            }
            Some(other) => return Some(vec![json!({"id": id, "error": format!("unknown action {other}")}).to_string()]),
            None => {}
        }
        self.queries += 1;
        if self.crash_on == Some(self.queries) {
            return None;
        }
        if let Some((k, d)) = self.hang_on {
            if k == self.queries {
                std::thread::sleep(d);
            }
        }
        if self.error_on == Some(self.queries) {
            return Some(vec![json!({"id": id, "error": "injected error", "field": "moves"}).to_string()]);
        }
        Some(match self.analyse(&q) {
            Ok(lines) => lines,
            Err((msg, field)) => vec![json!({"id": id, "error": msg, "field": field}).to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::katago::{gtp_of, parse_gtp};
    use super::*;

    #[test]
    fn independent_coordinates_agree() {
        for n in [5usize, 9, 19, 25] {
            for i in 0..n * n {
                assert_eq!(name(i, n), gtp_of(i, n), "n={n} i={i}");
                assert_eq!(vertex(&name(i, n), n), parse_gtp(&name(i, n), n));
            }
        }
        assert_eq!(vertex("I3", 19), None);
        assert_eq!(vertex("Q16", 19), Some(Some(3 * 19 + 15)));
    }

    fn ask(f: &mut KataFake, q: Value) -> Vec<Value> {
        f.handle(&q.to_string()).unwrap().iter().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    #[test]
    fn answers_captures_and_rejects_illegal_moves() {
        let mut f = KataFake::new();
        assert_eq!(ask(&mut f, json!({"id": "v", "action": "query_version"}))[0]["version"], FAKE_VERSION);
        // 5x5: white C3 surrounded on three sides; black to capture at C2
        let q = json!({"id": "a", "boardXSize": 5, "boardYSize": 5, "rules": "chinese", "komi": 7.5,
                       "initialStones": [["W", "C3"], ["B", "B3"], ["B", "D3"], ["B", "C4"]], "initialPlayer": "B",
                       "moves": [], "analyzeTurns": [0], "maxVisits": 50, "reportAnalysisWinratesAs": "SIDETOMOVE"});
        let r = ask(&mut f, q.clone());
        assert!(r[0].get("warning").is_some());
        assert_eq!(r[1]["moveInfos"][0]["move"], "C2");
        assert_eq!(r[1]["rootInfo"]["currentPlayer"], "B");
        let mut bad = q.clone();
        bad["moves"] = json!([["B", "C3"]]);
        let r = ask(&mut f, bad);
        assert_eq!(r[0]["field"], "moves", "{r:?}");
        let mut km = q;
        km["komi"] = json!(7.25);
        assert_eq!(ask(&mut f, km)[0]["field"], "komi");
    }
}
