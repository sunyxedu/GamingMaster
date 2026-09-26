//! UCI (chess: Stockfish) and USI (shogi: YaneuraOu) sessions — the two protocols
//! differ only in their handshake words, so one implementation serves both.
//!
//! ```text
//! open:    uci|usi → id name … → uciok|usiok;  setoption name K value V …;  isready → readyok
//! search:  [ucinewgame|usinewgame; isready → readyok]   (UciSession::new_game)
//!          position …;  go nodes N  →  info … multipv k score cp X [wdl W D L] … pv m1 m2 …
//!                                   →  bestmove m1
//! ```
//!
//! [`parse_info`] reads `info` lines in any token order: depth, seldepth, multipv
//! (default 1), score cp/mate (USI `mate +` / `mate -` without a distance = mate of
//! unknown length), lowerbound/upperbound, wdl (permille, side to move), nodes, pv.
//! [`collect`] keeps, per multipv index, the last exact-bound line with a score and a
//! pv (bound lines only when a line never got an exact score) — for a node-limited
//! search that is the final iteration.
//!
//! Scores are from the side to move. Stockfish reports `wdl` only with
//! `setoption name UCI_ShowWDL value true`; convert with
//! [`super::transform::q_from_wdl`], or cp with [`super::transform::q_from_cp`].

use super::engine::{Engine, EngineCommand, EngineError};
use anyhow::{bail, Result};
use std::time::Duration;

/// Which dialect a [`UciSession`] speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Uci,
    Usi,
}

impl Flavor {
    fn hello(&self) -> (&'static str, &'static str, &'static str) {
        match self {
            Flavor::Uci => ("uci", "uciok", "ucinewgame"),
            Flavor::Usi => ("usi", "usiok", "usinewgame"),
        }
    }
}

/// Mate distance reported for USI `mate +` / `mate -` (distance unknown).
pub const MATE_UNKNOWN: i32 = 9999;

/// An engine score from the side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
    /// centipawns (shogi engines: their eval units)
    Cp(i32),
    /// mate in n (>0: the side to move mates; ≤0: it is mated); UCI counts moves, USI plies
    Mate(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bound {
    #[default]
    Exact,
    Lower,
    Upper,
}

/// One parsed `info` line.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Info {
    pub depth: Option<u32>,
    pub seldepth: Option<u32>,
    pub multipv: usize,
    pub score: Option<Score>,
    pub bound: Bound,
    /// win / draw / loss permille for the side to move
    pub wdl: Option<[u32; 3]>,
    pub nodes: Option<u64>,
    pub pv: Vec<String>,
}

/// Parse an `info` line; `None` for other lines and `info string …`.
pub fn parse_info(line: &str) -> Option<Info> {
    let mut it = line.split_whitespace().peekable();
    if it.next()? != "info" {
        return None;
    }
    let mut info = Info { multipv: 1, ..Info::default() };
    let num = |s: Option<&str>| s.and_then(|x| x.parse::<i64>().ok());
    while let Some(tok) = it.next() {
        match tok {
            "string" => return None,
            "depth" => info.depth = num(it.next()).map(|x| x as u32),
            "seldepth" => info.seldepth = num(it.next()).map(|x| x as u32),
            "multipv" => info.multipv = num(it.next()).unwrap_or(1).max(1) as usize,
            "nodes" => info.nodes = num(it.next()).map(|x| x as u64),
            "score" => match it.next() {
                Some("cp") => info.score = num(it.next()).map(|x| Score::Cp(x as i32)),
                Some("mate") => {
                    let v = it.next().unwrap_or("");
                    info.score = match v {
                        "+" => Some(Score::Mate(MATE_UNKNOWN)),
                        "-" => Some(Score::Mate(-MATE_UNKNOWN)),
                        _ => v.parse::<i32>().ok().map(Score::Mate),
                    };
                }
                _ => {}
            },
            "lowerbound" => info.bound = Bound::Lower,
            "upperbound" => info.bound = Bound::Upper,
            "wdl" => {
                let w: Vec<u32> = (0..3).filter_map(|_| it.next().and_then(|x| x.parse().ok())).collect();
                if w.len() == 3 {
                    info.wdl = Some([w[0], w[1], w[2]]);
                }
            }
            "pv" => {
                info.pv = it.by_ref().map(str::to_string).collect();
            }
            _ => {}
        }
    }
    Some(info)
}

/// The final lines of one search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// one line per multipv index (with a score and a pv), sorted by index
    pub lines: Vec<Info>,
    /// the `bestmove` argument (may be `resign` / `win` / `(none)`)
    pub bestmove: String,
}

/// Fold a search's output into its final multipv lines (see the module docs).
pub fn collect(lines: &[String]) -> Result<SearchResult> {
    let mut exact: Vec<Option<Info>> = Vec::new();
    let mut any: Vec<Option<Info>> = Vec::new();
    let mut bestmove = None;
    for l in lines {
        if let Some(rest) = l.strip_prefix("bestmove") {
            bestmove = Some(rest.split_whitespace().next().unwrap_or("(none)").to_string());
            continue;
        }
        let Some(info) = parse_info(l) else { continue };
        if info.score.is_none() || info.pv.is_empty() {
            continue;
        }
        let k = info.multipv - 1;
        if exact.len() <= k {
            exact.resize(k + 1, None);
            any.resize(k + 1, None);
        }
        if info.bound == Bound::Exact {
            exact[k] = Some(info.clone());
        }
        any[k] = Some(info);
    }
    let Some(bestmove) = bestmove else { bail!(EngineError::Protocol("search output has no bestmove".into())) };
    let lines = exact.into_iter().zip(any).filter_map(|(e, a)| e.or(a)).collect();
    Ok(SearchResult { lines, bestmove })
}

/// A UCI / USI engine after the handshake.
pub struct UciSession {
    pub engine: Engine,
    pub flavor: Flavor,
    /// `id name` (e.g. "Stockfish 19")
    pub id_name: String,
    /// default timeout of handshake / isready exchanges
    pub timeout: Duration,
}

impl UciSession {
    /// Spawn, handshake, apply `options` (`setoption name K value V`), wait until ready.
    pub fn open(cmd: EngineCommand, flavor: Flavor, options: &[(String, String)], timeout: Duration) -> Result<Self> {
        let engine = Engine::spawn(cmd)?;
        let mut s = UciSession { engine, flavor, id_name: String::new(), timeout };
        let (hello, ok, _) = flavor.hello();
        let lines = s.engine.request(hello, timeout, |l| l.trim() == ok)?;
        s.id_name = lines.iter().find_map(|l| l.strip_prefix("id name ")).unwrap_or("unknown engine").trim().to_string();
        for (k, v) in options {
            s.setoption(k, v)?;
        }
        s.is_ready()?;
        Ok(s)
    }

    pub fn setoption(&mut self, name: &str, value: &str) -> Result<()> {
        self.engine.send(&format!("setoption name {name} value {value}"))
    }

    /// `isready` → wait for `readyok` (earlier unread lines are discarded).
    pub fn is_ready(&mut self) -> Result<()> {
        let t = self.timeout;
        self.engine.request("isready", t, |l| l.trim() == "readyok")?;
        Ok(())
    }

    /// `ucinewgame` / `usinewgame` + `isready` — clears the engine's hash, so a
    /// position's result does not depend on what the worker searched before.
    pub fn new_game(&mut self) -> Result<()> {
        let (_, _, newgame) = self.flavor.hello();
        self.engine.send(newgame)?;
        self.is_ready()
    }

    /// `position` line + `go` line → final multipv lines and bestmove, within `timeout`.
    pub fn search(&mut self, position: &str, go: &str, timeout: Duration) -> Result<SearchResult> {
        self.engine.send(position)?;
        let lines = self.engine.request(go, timeout, |l| l.starts_with("bestmove"))?;
        collect(&lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::fake::{FakeMove, UciFake};

    #[test]
    fn parses_stockfish_and_yaneuraou_lines() {
        let sf = "info depth 24 seldepth 31 multipv 2 score cp -35 wdl 40 800 160 nodes 1000512 nps 900000 hashfull 12 tbhits 0 time 1100 pv e7e5 g1f3 b8c6";
        let i = parse_info(sf).unwrap();
        assert_eq!((i.depth, i.seldepth, i.multipv, i.score, i.bound), (Some(24), Some(31), 2, Some(Score::Cp(-35)), Bound::Exact));
        assert_eq!((i.wdl, i.nodes), (Some([40, 800, 160]), Some(1000512)));
        assert_eq!(i.pv, vec!["e7e5", "g1f3", "b8c6"]);
        let lb = parse_info("info depth 10 multipv 1 score cp 20 lowerbound nodes 5 pv d2d4").unwrap();
        assert_eq!(lb.bound, Bound::Lower);
        let yo = parse_info("info depth 12 seldepth 18 score cp 132 multipv 3 nodes 40000 nps 1 time 40 pv 7g7f 3c3d 2g2f").unwrap();
        assert_eq!((yo.multipv, yo.score, yo.pv.len()), (3, Some(Score::Cp(132)), 3));
        assert_eq!(parse_info("info score mate -3 pv a1a2").unwrap().score, Some(Score::Mate(-3)));
        assert_eq!(parse_info("info depth 3 score mate + pv 5e5d").unwrap().score, Some(Score::Mate(MATE_UNKNOWN)));
        assert_eq!(parse_info("info depth 3 score mate - pv 5e5d").unwrap().score, Some(Score::Mate(-MATE_UNKNOWN)));
        assert!(parse_info("info string NNUE evaluation using nn.nnue").is_none());
        assert!(parse_info("bestmove e2e4").is_none());
        assert_eq!(parse_info("info depth 1 nodes 0 pv").unwrap().pv, Vec::<String>::new());
    }

    #[test]
    fn collect_keeps_last_exact_line_per_multipv() {
        let lines: Vec<String> = [
            "info depth 1 multipv 1 score cp 10 pv a",
            "info depth 1 multipv 2 score cp 5 pv b",
            "info depth 2 multipv 1 score cp 30 lowerbound pv a",
            "info depth 2 multipv 1 score cp 25 pv c",
            "info depth 2 multipv 2 score cp 99 upperbound pv b",
            "info depth 3 multipv 3 score cp -7 upperbound pv d",
            "info string hello",
            "bestmove c ponder a",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let r = collect(&lines).unwrap();
        assert_eq!(r.bestmove, "c");
        let got: Vec<(String, Score)> = r.lines.iter().map(|i| (i.pv[0].clone(), i.score.unwrap())).collect();
        assert_eq!(got, vec![("c".into(), Score::Cp(25)), ("b".into(), Score::Cp(5)), ("d".into(), Score::Cp(-7))]);
        assert!(collect(&lines[..3]).is_err(), "no bestmove");
    }

    #[test]
    fn session_against_fake_engine() {
        let cmd = crate::label::engine::EngineCommand::in_process("fake-uci", || {
            UciFake::new(|pos: &str| {
                if !pos.starts_with("position startpos") {
                    return None;
                }
                Some(vec![FakeMove::cp("e2e4", 30), FakeMove::cp("d2d4", 25).wdl([60, 900, 40]), FakeMove::cp("a2a3", -40)])
            })
        });
        let opts = vec![("MultiPV".to_string(), "2".to_string())];
        let mut s = UciSession::open(cmd, Flavor::Uci, &opts, Duration::from_secs(5)).unwrap();
        assert_eq!(s.id_name, "FakeUci 1.0");
        s.new_game().unwrap();
        let r = s.search("position startpos", "go nodes 100", Duration::from_secs(5)).unwrap();
        assert_eq!(r.bestmove, "e2e4");
        assert_eq!(r.lines.len(), 2, "MultiPV option honoured");
        assert_eq!(r.lines[1].wdl, Some([60, 900, 40]));
        // unknown position → the fake crashes → typed error, session needs a restart
        assert!(s.search("position fen x", "go nodes 1", Duration::from_secs(5)).is_err());
    }
}
