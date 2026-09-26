//! Fake engines — test doubles that speak engine protocols with canned or computed
//! answers, so no engine binary is needed to test a labeller.
//!
//! * [`UciFake`]: a UCI/USI engine whose search results come from a closure over the
//!   last `position …` line (return the scored moves; `None` = crash). It honours
//!   `setoption name MultiPV`, answers `isready`, and can inject a crash or a hang on
//!   the n-th search. Chess / shogi labellers use it with a scorer that derives legal
//!   moves from the position line.
//! * [`Canned`]: a table of `input → reply lines` rules for any line protocol (GTP,
//!   Egaroucid console, KataGo JSON, …).
//!
//! Both are [`LineHandler`]s: run them in-process with
//! `EngineCommand::in_process(name, || UciFake::new(...))` (unit tests), or as a real
//! process through the hidden CLI (`gmdata label fake-engine toy`,
//! `gmdata label fake-engine canned --script FILE`; [`run_stdio`]) for end-to-end tests.
//!
//! Canned script format — one rule per line, fields separated by TAB characters
//! (written `⇥` here):
//!
//! ```text
//! # comment
//! uci⇥id name Canned 1.0⇥uciok                   exact input line → two reply lines
//! go *⇥info depth 1 multipv 1 score cp 12 pv e2e4⇥bestmove e2e4
//! quit⇥<exit>                                    <exit> = terminate (a crash if unexpected)
//! ```
//!
//! A pattern ending in `*` matches by prefix (the text before `*`); the first matching
//! rule wins; an unmatched line gets no reply.

use super::engine::LineHandler;
use super::uci::Score;
use anyhow::{bail, Result};
use std::io::{BufRead, Write};
use std::time::Duration;

/// One scored move a [`UciFake`] reports.
#[derive(Debug, Clone, PartialEq)]
pub struct FakeMove {
    pub mv: String,
    pub score: Score,
    pub wdl: Option<[u32; 3]>,
}

impl FakeMove {
    pub fn cp(mv: &str, cp: i32) -> Self {
        FakeMove { mv: mv.to_string(), score: Score::Cp(cp), wdl: None }
    }

    pub fn mate(mv: &str, n: i32) -> Self {
        FakeMove { mv: mv.to_string(), score: Score::Mate(n), wdl: None }
    }

    pub fn wdl(mut self, wdl: [u32; 3]) -> Self {
        self.wdl = Some(wdl);
        self
    }

    fn order_key(&self) -> i64 {
        match self.score {
            Score::Cp(x) => x as i64,
            Score::Mate(n) if n > 0 => 1_000_000 - n as i64,
            Score::Mate(n) => -1_000_000 - n as i64,
        }
    }
}

/// A scripted UCI / USI engine (see the module docs).
pub struct UciFake<S> {
    scorer: S,
    usi: bool,
    multipv: usize,
    position: String,
    searches: usize,
    crash_on: Option<usize>,
    hang_on: Option<(usize, Duration)>,
}

impl<S: FnMut(&str) -> Option<Vec<FakeMove>> + Send> UciFake<S> {
    pub fn new(scorer: S) -> Self {
        UciFake { scorer, usi: false, multipv: 1, position: String::new(), searches: 0, crash_on: None, hang_on: None }
    }

    /// Crash (exit) on the `n`-th search (1-based, counted per process lifetime).
    pub fn crash_on(mut self, n: usize) -> Self {
        self.crash_on = Some(n);
        self
    }

    /// Sleep `d` before answering the `n`-th search (timeout tests).
    pub fn hang_on(mut self, n: usize, d: Duration) -> Self {
        self.hang_on = Some((n, d));
        self
    }

    fn search(&mut self) -> Option<Vec<String>> {
        self.searches += 1;
        if self.crash_on == Some(self.searches) {
            return None;
        }
        if let Some((n, d)) = self.hang_on {
            if n == self.searches {
                std::thread::sleep(d);
            }
        }
        let mut moves = (self.scorer)(&self.position)?;
        moves.sort_by_key(|m| std::cmp::Reverse(m.order_key()));
        let mut out = Vec::new();
        for (k, m) in moves.iter().take(self.multipv).enumerate() {
            let score = match m.score {
                Score::Cp(x) => format!("cp {x}"),
                Score::Mate(n) => format!("mate {n}"),
            };
            let wdl = m.wdl.map(|[w, d, l]| format!(" wdl {w} {d} {l}")).unwrap_or_default();
            out.push(format!("info depth 1 seldepth 1 multipv {} score {score}{wdl} nodes 1 pv {}", k + 1, m.mv));
        }
        let none = if self.usi { "resign" } else { "(none)" };
        out.push(format!("bestmove {}", moves.first().map(|m| m.mv.as_str()).unwrap_or(none)));
        Some(out)
    }
}

impl<S: FnMut(&str) -> Option<Vec<FakeMove>> + Send> LineHandler for UciFake<S> {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        let line = line.trim();
        let mut toks = line.split_whitespace();
        match toks.next().unwrap_or("") {
            "uci" | "usi" => {
                self.usi = line == "usi";
                let (name, ok) = if self.usi { ("FakeUsi", "usiok") } else { ("FakeUci", "uciok") };
                Some(vec![
                    format!("id name {name} 1.0"),
                    "id author GamingMaster".into(),
                    "option name MultiPV type spin default 1 min 1 max 500".into(),
                    ok.into(),
                ])
            }
            "isready" => Some(vec!["readyok".into()]),
            "setoption" => {
                // setoption name <name> value <value>
                let rest: Vec<&str> = toks.collect();
                if let Some(v) = rest.iter().position(|t| *t == "value") {
                    if rest[..v].iter().skip(1).copied().collect::<Vec<_>>().join(" ").eq_ignore_ascii_case("multipv") {
                        self.multipv = rest.get(v + 1).and_then(|x| x.parse().ok()).unwrap_or(1).max(1);
                    }
                }
                Some(vec![])
            }
            "position" => {
                self.position = line.to_string();
                Some(vec![])
            }
            "go" => self.search(),
            "quit" => None,
            _ => Some(vec![]),
        }
    }
}

enum Pattern {
    Exact(String),
    Prefix(String),
}

/// Rule-table fake engine for any line protocol (see the module docs).
#[derive(Default)]
pub struct Canned {
    rules: Vec<(Pattern, Vec<String>)>,
}

impl Canned {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule (`pattern` ending in `*` = prefix match; reply `<exit>` = terminate).
    pub fn rule(mut self, pattern: &str, replies: &[&str]) -> Self {
        let p = match pattern.strip_suffix('*') {
            Some(prefix) => Pattern::Prefix(prefix.to_string()),
            None => Pattern::Exact(pattern.to_string()),
        };
        self.rules.push((p, replies.iter().map(|s| s.to_string()).collect()));
        self
    }

    /// Parse a script (TAB-separated rules, `#` comments).
    pub fn parse(text: &str) -> Result<Self> {
        let mut c = Canned::new();
        for (n, l) in text.lines().enumerate() {
            if l.trim().is_empty() || l.trim_start().starts_with('#') {
                continue;
            }
            let mut f = l.split('\t');
            let pat = f.next().unwrap_or("");
            if pat.is_empty() {
                bail!("fake-engine script line {}: empty pattern", n + 1);
            }
            let replies: Vec<&str> = f.collect();
            c = c.rule(pat, &replies);
        }
        Ok(c)
    }
}

impl LineHandler for Canned {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        let line = line.trim_end();
        for (p, replies) in &self.rules {
            let hit = match p {
                Pattern::Exact(s) => s == line,
                Pattern::Prefix(s) => line.starts_with(s.as_str()),
            };
            if hit {
                if replies.iter().any(|r| r == "<exit>") {
                    return None;
                }
                return Some(replies.clone());
            }
        }
        Some(vec![])
    }
}

/// Serve `handler` over stdin/stdout (the hidden `gmdata label fake-engine` command).
pub fn run_stdio(handler: &mut dyn LineHandler) -> Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        match handler.handle(&line) {
            Some(replies) => {
                for r in replies {
                    writeln!(out, "{r}")?;
                }
                out.flush()?;
            }
            None => return Ok(()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uci_fake_protocol() {
        let mut f = UciFake::new(|p: &str| (p == "position startpos").then(|| vec![FakeMove::cp("b", 5), FakeMove::mate("a", 3), FakeMove::cp("c", -2)]))
            .crash_on(3);
        assert_eq!(f.handle("uci").unwrap().last().unwrap(), "uciok");
        assert_eq!(f.handle("isready").unwrap(), vec!["readyok"]);
        f.handle("setoption name MultiPV value 2").unwrap();
        f.handle("position startpos").unwrap();
        let out = f.handle("go nodes 10").unwrap();
        assert_eq!(out, vec![
            "info depth 1 seldepth 1 multipv 1 score mate 3 nodes 1 pv a",
            "info depth 1 seldepth 1 multipv 2 score cp 5 nodes 1 pv b",
            "bestmove a",
        ]);
        f.handle("position other").unwrap();
        assert!(f.handle("go").is_none(), "scorer None → crash");
        f.handle("position startpos").unwrap();
        assert!(f.handle("go").is_none(), "injected crash on the 3rd search");
        let mut u = UciFake::new(|_: &str| Some(vec![]));
        assert_eq!(u.handle("usi").unwrap()[0], "id name FakeUsi 1.0");
        assert_eq!(u.handle("go").unwrap(), vec!["bestmove resign"]);
        assert!(u.handle("quit").is_none());
    }

    #[test]
    fn canned_rules() {
        let mut c = Canned::parse("# gtp\nname\t= fake\t\ngenmove *\t= D4\t\nboom\t<exit>\n").unwrap();
        assert_eq!(c.handle("name").unwrap(), vec!["= fake", ""]);
        assert_eq!(c.handle("genmove b").unwrap(), vec!["= D4", ""]);
        assert_eq!(c.handle("unknown").unwrap(), Vec::<String>::new());
        assert!(c.handle("boom").is_none());
        assert!(Canned::parse("\tx").is_err());
    }
}
