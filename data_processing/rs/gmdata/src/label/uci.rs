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
//! [`collect`] keeps, per multipv index, the last line with a score and a pv, whatever
//! its bound: engines print one consistent block of all multipv lines when the search
//! stops, whereas picking "the last exact line" per index mixes iterations and can
//! list one move twice (seen with Stockfish 19 under a node limit). A line's `bound`
//! and `depth` are kept for the labeller; a repeated first move is dropped (first,
//! i.e. best-ranked, occurrence wins).
//!
//! Scores are from the side to move. Stockfish reports `wdl` only with
//! `setoption name UCI_ShowWDL value true`; convert with
//! [`super::transform::q_from_wdl`], or cp with [`super::transform::q_from_cp`].

use super::engine::{Engine, EngineCommand, EngineError};
use anyhow::{bail, Context, Result};
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
    let mut last: Vec<Option<Info>> = Vec::new();
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
        if last.len() <= k {
            last.resize(k + 1, None);
        }
        last[k] = Some(info);
    }
    let Some(bestmove) = bestmove else { bail!(EngineError::Protocol("search output has no bestmove".into())) };
    let mut seen = std::collections::HashSet::new();
    let lines = last.into_iter().flatten().filter(|i| seen.insert(i.pv[0].clone())).collect();
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

/// An option announced in the `uci` / `usi` handshake:
/// `option name <NAME> type <TYPE> [default D] [min A] [max B] [var …]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionSpec {
    pub name: String,
    pub kind: String,
    pub min: Option<i64>,
    pub max: Option<i64>,
}

/// Parse a handshake `option name …` line (`None` for other lines).
pub fn parse_option(line: &str) -> Option<OptionSpec> {
    let rest = line.trim().strip_prefix("option name ")?;
    let (name, spec) = rest.split_once(" type ")?;
    let toks: Vec<&str> = spec.split_whitespace().collect();
    let num = |key: &str| toks.iter().position(|t| *t == key).and_then(|i| toks.get(i + 1)).and_then(|v| v.parse::<i64>().ok());
    Some(OptionSpec { name: name.trim().to_string(), kind: toks.first()?.to_string(), min: num("min"), max: num("max") })
}

/// Reject option values the engine would silently ignore. Stockfish (and other UCI
/// engines) drop a `setoption` whose `spin` value lies outside the announced
/// `[min, max]` (or is not a number) and a `check` value other than `true` / `false`,
/// and carry on with the previous value — e.g. `MultiPV 300` (Stockfish's maximum is
/// 256) would leave MultiPV at 1 and every position would be labelled with a single
/// move while the teacher meta claims 300. Options the engine does not announce are
/// not checked (names are case-insensitive).
pub fn check_option_values(hello: &[String], options: &[(String, String)]) -> Result<()> {
    let specs: Vec<OptionSpec> = hello.iter().filter_map(|l| parse_option(l)).collect();
    for (k, v) in options {
        let Some(spec) = specs.iter().find(|s| s.name.eq_ignore_ascii_case(k.trim())) else { continue };
        let v = v.trim();
        match spec.kind.as_str() {
            "spin" => {
                let (lo, hi) = (spec.min.unwrap_or(i64::MIN), spec.max.unwrap_or(i64::MAX));
                match v.parse::<i64>() {
                    Ok(x) if (lo..=hi).contains(&x) => {}
                    _ => bail!("option {k}={v}: the engine accepts integers in [{lo}, {hi}] for {} and would silently ignore this value", spec.name),
                }
            }
            "check" if v != "true" && v != "false" => {
                bail!("option {k}={v}: {} is a check option (true / false); the engine would silently ignore this value", spec.name)
            }
            _ => {}
        }
    }
    Ok(())
}

impl UciSession {
    /// Spawn, handshake, apply `options` (`setoption name K value V`; values the engine
    /// would silently ignore are an error, [`check_option_values`]), wait until ready.
    pub fn open(cmd: EngineCommand, flavor: Flavor, options: &[(String, String)], timeout: Duration) -> Result<Self> {
        let engine = Engine::spawn(cmd)?;
        let mut s = UciSession { engine, flavor, id_name: String::new(), timeout };
        let (hello, ok, _) = flavor.hello();
        let lines = s.engine.request(hello, timeout, |l| l.trim() == ok)?;
        s.id_name = lines.iter().find_map(|l| l.strip_prefix("id name ")).unwrap_or("unknown engine").trim().to_string();
        check_option_values(&lines, options).with_context(|| s.id_name.clone())?;
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
    fn collect_keeps_the_final_block() {
        let lines: Vec<String> = [
            "info depth 1 multipv 1 score cp 10 pv a",
            "info depth 1 multipv 2 score cp 5 pv b",
            "info depth 1 multipv 3 score cp 1 pv c",
            "info depth 2 multipv 1 score cp 30 lowerbound pv a",
            "info depth 2 multipv 1 score cp 25 pv c",
            "info string hello",
            // final block at the node limit: line 2 failed low in the last iteration
            "info depth 2 multipv 1 score cp 25 pv c a",
            "info depth 2 multipv 2 score cp 12 upperbound pv a",
            "info depth 1 multipv 3 score cp 5 pv b",
            "info depth 1 multipv 4 score cp 3 pv a",
            "bestmove c ponder a",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let r = collect(&lines).unwrap();
        assert_eq!(r.bestmove, "c");
        let got: Vec<(String, Score, Bound)> = r.lines.iter().map(|i| (i.pv[0].clone(), i.score.unwrap(), i.bound)).collect();
        assert_eq!(got, vec![
            ("c".into(), Score::Cp(25), Bound::Exact),
            ("a".into(), Score::Cp(12), Bound::Upper),
            ("b".into(), Score::Cp(5), Bound::Exact),
        ], "last line per index; the repeated first move of line 4 dropped");
        assert!(collect(&lines[..3]).is_err(), "no bestmove");
    }

    /// Moves of a real game (Ruy Lopez, Morphy defence) as UCI strings.
    const GAME: &str = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 \
                        b3c2 c7c5 d2d4 d8c7 b1d2 c5d4 c3d4 a5c6 d2b3 a6a5 c1e3 a5a4 b3d2 c8d7 a1c1 c7b7 d2f1 f8c8";

    fn stockfish(multipv: usize) -> Option<UciSession> {
        let path = crate::label::testkit::engine_path("STOCKFISH", "stockfish")?;
        let opts: Vec<(String, String)> = [("Threads", "1"), ("Hash", "16"), ("UCI_ShowWDL", "true")]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .chain(std::iter::once(("MultiPV".to_string(), multipv.to_string())))
            .collect();
        let cmd = crate::label::engine::EngineCommand::process(path, &[]);
        Some(UciSession::open(cmd, Flavor::Uci, &opts, Duration::from_secs(60)).expect("stockfish handshake"))
    }

    /// Real Stockfish (skipped when not installed: `scripts/engines/fetch_engines.sh stockfish`
    /// or `STOCKFISH=/path/to/stockfish`).
    #[test]
    fn real_stockfish_multipv_wdl() {
        let Some(mut s) = stockfish(8) else {
            eprintln!("skipping: stockfish not found");
            return;
        };
        assert!(s.id_name.starts_with("Stockfish"), "{}", s.id_name);
        s.new_game().unwrap();
        let r = s.search("position startpos", "go nodes 20000", Duration::from_secs(60)).unwrap();
        assert_eq!(r.lines.len(), 8, "{r:?}");
        assert_eq!(r.bestmove, r.lines[0].pv[0]);
        let mut seen = std::collections::HashSet::new();
        for (k, l) in r.lines.iter().enumerate() {
            assert_eq!(l.multipv, k + 1);
            assert!(l.depth.unwrap_or(0) > 0);
            let [w, d, lo] = l.wdl.expect("UCI_ShowWDL");
            assert_eq!(w + d + lo, 1000);
            assert!(matches!(l.score, Some(Score::Cp(cp)) if cp.abs() < 200));
            assert!(seen.insert(l.pv[0].clone()), "multipv first moves are distinct");
        }
        // mate scores are parsed too (scholar's mate in one for white)
        s.new_game().unwrap();
        let r = s.search("position startpos moves e2e4 e7e5 f1c4 b8c6 d1h5 g8f6", "go nodes 20000", Duration::from_secs(60)).unwrap();
        assert_eq!(r.lines[0].score, Some(Score::Mate(1)));
        assert_eq!(r.bestmove, "h5f7");
    }

    /// Positions/s of one single-threaded Stockfish at labelling settings. Opt-in:
    /// `cargo test --release -p gmdata real_stockfish_throughput -- --ignored --nocapture`
    /// (`SF_NODES`, `SF_MULTIPV` override the defaults 1000000 / 32).
    #[test]
    #[ignore]
    fn real_stockfish_throughput() {
        let nodes: u64 = std::env::var("SF_NODES").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
        let multipv: usize = std::env::var("SF_MULTIPV").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let Some(mut s) = stockfish(multipv) else {
            eprintln!("skipping: stockfish not found");
            return;
        };
        let moves: Vec<&str> = GAME.split_whitespace().collect();
        let t0 = std::time::Instant::now();
        let mut lines = 0;
        for ply in 0..moves.len() {
            s.new_game().unwrap();
            let pos = format!("position startpos moves {}", moves[..ply].join(" "));
            lines += s.search(pos.trim_end(), &format!("go nodes {nodes}"), Duration::from_secs(600)).unwrap().lines.len();
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "{}: {} positions, nodes {nodes}, multipv {multipv}: {:.2} positions/s per process ({:.1} scored moves/position)",
            s.id_name,
            moves.len(),
            moves.len() as f64 / secs,
            lines as f64 / moves.len() as f64
        );
    }

    /// Stockfish 19 ignores `setoption name MultiPV value 300` (its maximum is 256) and
    /// searches with MultiPV 1: such values must fail before any labelling.
    #[test]
    fn option_values_the_engine_would_ignore_are_rejected() {
        let hello: Vec<String> = [
            "id name Stockfish 19",
            "option name Debug Log File type string default <empty>",
            "option name Threads type spin default 1 min 1 max 1024",
            "option name Clear Hash type button",
            "option name MultiPV type spin default 1 min 1 max 256",
            "option name UCI_ShowWDL type check default false",
            "uciok",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            parse_option(&hello[4]),
            Some(OptionSpec { name: "MultiPV".into(), kind: "spin".into(), min: Some(1), max: Some(256) })
        );
        assert_eq!(parse_option(&hello[1]).unwrap().name, "Debug Log File");
        assert_eq!(parse_option(&hello[3]).unwrap().kind, "button");
        assert!(parse_option("id name x").is_none());
        let opts = |kv: &[(&str, &str)]| kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<Vec<_>>();
        let ok = opts(&[("Threads", "1"), ("multipv", "256"), ("UCI_ShowWDL", "true"), ("Debug Log File", "x"), ("Unknown", "7")]);
        check_option_values(&hello, &ok).unwrap();
        for bad in [("MultiPV", "300"), ("MultiPV", "0"), ("Threads", "two"), ("UCI_ShowWDL", "1"), ("uci_showwdl", "True")] {
            let e = check_option_values(&hello, &opts(&[bad])).unwrap_err().to_string();
            assert!(e.contains(&format!("{}={}", bad.0, bad.1)) && e.contains("silently ignore"), "{e}");
        }
        // through the session: the fake announces MultiPV max 500
        let cmd = crate::label::engine::EngineCommand::in_process("fake-uci", || UciFake::new(|_: &str| Some(vec![FakeMove::cp("e2e4", 1)])));
        let e = UciSession::open(cmd.clone(), Flavor::Uci, &opts(&[("MultiPV", "501")]), Duration::from_secs(5)).err().unwrap();
        assert!(format!("{e:#}").contains("[1, 500]"), "{e:#}");
        assert!(UciSession::open(cmd, Flavor::Uci, &opts(&[("MultiPV", "500")]), Duration::from_secs(5)).is_ok());
    }

    /// The premise of [`check_option_values`] on the real engine (skipped when absent).
    #[test]
    fn real_stockfish_ignores_out_of_range_multipv() {
        let Some(path) = crate::label::testkit::engine_path("STOCKFISH", "stockfish") else {
            eprintln!("skipping: stockfish not found");
            return;
        };
        let cmd = crate::label::engine::EngineCommand::process(path, &[]);
        let opts = vec![("MultiPV".to_string(), "300".to_string())];
        let e = UciSession::open(cmd.clone(), Flavor::Uci, &opts, Duration::from_secs(60)).err().expect("MultiPV 300 must be rejected");
        assert!(format!("{e:#}").contains("[1, 256]"), "{e:#}");
        // what the check prevents: the engine keeps MultiPV 1
        let mut s = UciSession::open(cmd, Flavor::Uci, &[], Duration::from_secs(60)).unwrap();
        s.setoption("MultiPV", "300").unwrap();
        s.new_game().unwrap();
        let r = s.search("position startpos", "go nodes 20000", Duration::from_secs(60)).unwrap();
        assert_eq!(r.lines.len(), 1, "{r:?}");
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
