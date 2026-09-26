//! Kuhn poker (the toy reference game): hand logs → candidate-action gm-shards, and a
//! synthetic log generator standing in for a download. The worked example of a
//! gmrules converter: parse the source format into trajectories, hand them to
//! [`cand::convert`] (replay + verification + frames), done. Rules and frames:
//! `gmrules::games::kuhn_poker`.
//!
//! Log format (ACPC-style, one hand per line; `#` comments and blank lines skipped):
//!
//! ```text
//! STATE:<hand id>:<betting>:<card seat0>|<card seat1>|...:<return0>|<return1>|...:<name0>|<name1>|...
//! STATE:17:pbb:1|2:-2|2:expert|random
//! ```
//!
//! betting = one character per action in turn order, `p` (Pass / fold) or `b` (Bet /
//! call); cards = ranks 0..=N (N players); returns = chips won (checked against the
//! replayed payoffs; an empty field = unknown). Unfinished hands are rejected unless
//! `--allow-truncated`. High-quality recipe (`--hq` = `--player expert`): learn only from
//! the seats a strong player played (per-seat samples of the other seats are dropped).
//!
//! ```sh
//! gmdata kuhn-poker generate --out data/kuhn_poker/raw/hands.log --hands 100000 --players 2
//! gmdata kuhn-poker convert --input data/kuhn_poker/raw --out data/kuhn_poker/train --hq
//! ```

use crate::cand::{self, ConvertOptions, Record};
use crate::common;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use gmrules::games::kuhn_poker::{KuhnConfig, KuhnPoker, BET, PASS};
use gmrules::{Event, Game, Rng, Seat, Turn};
use serde_json::json;
use std::cell::Cell;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// Parse one log line into a record (None for comments / blank lines).
pub fn parse_line(line: &str, id: &str) -> Result<Option<Record>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let parts: Vec<&str> = line.split(':').collect();
    if parts.len() != 6 || parts[0] != "STATE" {
        bail!("expected STATE:<id>:<betting>:<cards>:<returns>:<names>, got {line:?}");
    }
    let cards: Vec<u32> = parts[3].split('|').map(|c| c.parse().with_context(|| format!("card {c:?}"))).collect::<Result<_>>()?;
    // an empty returns field = unknown (an unfinished hand, a hand edited in a test)
    let returns: Option<Vec<f64>> = if parts[4].is_empty() {
        None
    } else {
        Some(parts[4].split('|').map(|r| r.parse().with_context(|| format!("return {r:?}"))).collect::<Result<_>>()?)
    };
    let names: Vec<String> = parts[5].split('|').map(String::from).collect();
    let n = cards.len();
    if returns.as_ref().is_some_and(|r| r.len() != n) || names.len() != n {
        bail!("{n} cards but {:?} returns / {} names", returns.as_ref().map(Vec::len), names.len());
    }
    let mut events: Vec<Event> = cards.iter().map(|&c| Event::Chance(c)).collect();
    for (i, ch) in parts[2].chars().enumerate() {
        let a = match ch {
            'p' => PASS,
            'b' => BET,
            c => bail!("betting character {c:?} (p / b)"),
        };
        events.push(Event::Act(i % n, a));
    }
    Ok(Some(Record {
        id: format!("{id}#{}", parts[1]),
        events,
        meta: json!({"hand": parts[1], "players": names}),
        seats: None,
        expected_returns: returns,
        ..Default::default()
    }))
}

/// The lines of a log file, streamed (`.gz` decompressed on the fly).
fn open_lines(path: &Path) -> Result<Box<dyn Iterator<Item = std::io::Result<String>>>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let gz = path.to_string_lossy().to_lowercase().ends_with(".gz");
    let r: Box<dyn Read> = if gz { Box::new(flate2::read::MultiGzDecoder::new(f)) } else { Box::new(f) };
    Ok(Box::new(BufReader::new(r).lines()))
}

/// Which hands / seats to keep.
pub struct Filter {
    /// hands with another player count are skipped (counted in `other_variant`)
    pub players: usize,
    /// non-empty: emit only the seats these names played (hands without them dropped)
    pub keep_players: Vec<String>,
    pub other_variant: Cell<usize>,
}

impl Filter {
    fn apply(&self, mut rec: Record) -> Option<Record> {
        let names: Vec<&str> = rec.meta["players"].as_array()?.iter().filter_map(|v| v.as_str()).collect();
        if names.len() != self.players {
            self.other_variant.set(self.other_variant.get() + 1);
            return None;
        }
        if !self.keep_players.is_empty() {
            let seats: Vec<Seat> = (0..self.players).filter(|&s| self.keep_players.iter().any(|k| k == names[s])).collect();
            if seats.is_empty() {
                return None;
            }
            rec.seats = Some(seats);
        }
        Some(rec)
    }
}

/// The records of every `.log` / `.log.gz` file under `input` (sorted), streamed line
/// by line and filtered by `filter`; unreadable lines come out as `Err` (counted by
/// the converter as unparseable).
pub fn records<'a>(input: &Path, filter: &'a Filter) -> Result<impl Iterator<Item = Result<Record>> + 'a> {
    let files = common::find_files(input, &[".log", ".log.gz"])?;
    anyhow::ensure!(!files.is_empty(), "no .log files under {}", input.display());
    Ok(files
        .into_iter()
        .flat_map(|path| -> Box<dyn Iterator<Item = Result<Record>>> {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            match open_lines(&path) {
                Ok(lines) => Box::new(lines.enumerate().filter_map(move |(ln, line)| {
                    let id = format!("{name}:{}", ln + 1);
                    match line {
                        Ok(l) => parse_line(&l, &id).with_context(|| id.clone()).transpose(),
                        Err(e) => Some(Err(anyhow::Error::from(e).context(id))),
                    }
                })),
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        })
        .filter_map(move |r| match r {
            Ok(rec) => filter.apply(rec).map(Ok),
            Err(e) => Some(Err(e)),
        }))
}

/// A scripted near-equilibrium player ("expert"): bet the best card, bluff the worst
/// one time in three; facing a bet call with the best card, with the second best one
/// time in three.
fn expert_action(facing: bool, card: u32, n: u32, rng: &mut Rng) -> u32 {
    let p = rng.next_f64();
    if !facing {
        if card == n || (card == 0 && p < 1.0 / 3.0) { BET } else { PASS }
    } else if card == n || (card + 1 == n && p < 1.0 / 3.0) {
        BET
    } else {
        PASS
    }
}

/// `hands` synthetic hands: every seat is played by "expert" (probability
/// `expert_frac`) or "random" (uniform actions). One log line per hand.
pub fn generate(out: &Path, hands: usize, players: usize, seed: u64, expert_frac: f64) -> Result<usize> {
    let cfg = KuhnConfig { players };
    KuhnPoker::check_config(&cfg).map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    writeln!(f, "# kuhn_poker synthetic hands: players={players} seed={seed} expert_frac={expert_frac}")?;
    let mut rng = Rng::new(seed);
    let mut outcomes = Vec::new();
    for h in 0..hands {
        let names: Vec<&str> = (0..players).map(|_| if rng.next_f64() < expert_frac { "expert" } else { "random" }).collect();
        let mut g = KuhnPoker::new_game(&cfg);
        let mut cards = Vec::new();
        let mut betting = String::new();
        loop {
            match g.turn() {
                Turn::Chance => {
                    outcomes.clear();
                    g.chance_outcomes(&mut outcomes);
                    let c = outcomes[rng.below(outcomes.len() as u64) as usize].0;
                    g.apply_chance(c).map_err(|e| anyhow::anyhow!("{e}"))?;
                    cards.push(c);
                }
                Turn::Act(s) => {
                    let seat = s.iter().next().unwrap();
                    let a = if names[seat] == "expert" {
                        expert_action(betting.contains('b'), cards[seat], players as u32, &mut rng)
                    } else {
                        rng.below(2) as u32
                    };
                    g.apply_action(seat, a).map_err(|e| anyhow::anyhow!("{e}"))?;
                    betting.push(if a == PASS { 'p' } else { 'b' });
                }
                Turn::Terminal => break,
            }
        }
        let join = |v: Vec<String>| v.join("|");
        writeln!(
            f,
            "STATE:{h}:{betting}:{}:{}:{}",
            join(cards.iter().map(|c| c.to_string()).collect()),
            join(g.returns().iter().map(|r| format!("{r}")).collect()),
            names.join("|")
        )?;
    }
    f.flush()?;
    Ok(hands)
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Write synthetic hand logs (expert / random players) — the toy's "download"
    Generate {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10000)]
        hands: usize,
        #[arg(long, default_value_t = 2)]
        players: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// probability that a seat is played by the scripted expert
        #[arg(long, default_value_t = 0.5)]
        expert_frac: f64,
    },
    /// Convert hand logs (.log / .log.gz files) to candidate-action gm-shards
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// hands with another player count are skipped
        #[arg(long, default_value_t = 2)]
        players: usize,
        /// emit only the seats these players played (repeatable)
        #[arg(long)]
        player: Vec<String>,
        /// high-quality preset: --player expert
        #[arg(long)]
        hq: bool,
        #[arg(long, default_value_t = 1024)]
        shard_size: usize,
        #[arg(long)]
        max_records: Option<usize>,
        /// store each sample's trajectory in its meta (golden tests)
        #[arg(long)]
        store_events: bool,
        /// keep unfinished hands (tgt.value_mask false on their frames; the causality
        /// invariance test converts truncated records)
        #[arg(long)]
        allow_truncated: bool,
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Generate { out, hands, players, seed, expert_frac } => {
            let n = generate(&out, hands, players, seed, expert_frac)?;
            println!("{}", json!({"hands": n, "out": out}));
            Ok(())
        }
        Cmd::Convert { input, out, players, mut player, hq, shard_size, max_records, store_events, allow_truncated } => {
            if hq && !player.iter().any(|p| p == "expert") {
                player.push("expert".into());
            }
            let filter = Filter { players, keep_players: player, other_variant: Cell::new(0) };
            let recs = records(&input, &filter)?.take(max_records.unwrap_or(usize::MAX));
            let opts = ConvertOptions { shard_size, store_events, allow_truncated, ..Default::default() };
            let st = cand::convert::<KuhnPoker>(&KuhnConfig { players }, recs, &out, &opts)?;
            let mut v = serde_json::to_value(&st)?;
            v["other_variant"] = json!(filter.other_variant.get());
            println!("{v}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_generate_roundtrip() {
        let r = parse_line("STATE:17:pbb:1|2:-2|2:expert|random", "f").unwrap().unwrap();
        assert_eq!(r.events, vec![Event::Chance(1), Event::Chance(2), Event::Act(0, 0), Event::Act(1, 1), Event::Act(0, 1)]);
        assert_eq!(r.expected_returns, Some(vec![-2.0, 2.0]));
        assert!(parse_line("# comment", "f").unwrap().is_none());
        assert!(parse_line("STATE:1:px:1|2:0|0:a|b", "f").is_err());
        let dir = std::env::temp_dir().join(format!("gmdata-kuhn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        generate(&dir.join("raw/h.log"), 200, 3, 5, 0.5).unwrap();
        let all = Filter { players: 3, keep_players: vec![], other_variant: Cell::new(0) };
        let st = cand::convert::<KuhnPoker>(&KuhnConfig { players: 3 }, records(&dir.join("raw"), &all).unwrap(), &dir.join("out"), &ConvertOptions::default()).unwrap();
        assert_eq!((st.records, st.converted, st.illegal, st.returns_mismatch, st.unparseable), (200, 200, 0, 0, 0));
        assert_eq!((st.samples, all.other_variant.get()), (600, 0));
        // --hq keeps only the expert seats; other variants are counted, broken lines skipped
        std::fs::write(dir.join("raw/x.log"), "STATE:1:pp:0|1:1|-1:expert|random\nSTATE:2:zz\n").unwrap();
        let hq = Filter { players: 3, keep_players: vec!["expert".into()], other_variant: Cell::new(0) };
        let st = cand::convert::<KuhnPoker>(&KuhnConfig { players: 3 }, records(&dir.join("raw"), &hq).unwrap(), &dir.join("hq"), &ConvertOptions::default()).unwrap();
        assert!(st.samples < 600 && st.samples > 150, "{st:?}");
        assert_eq!((st.unparseable, hq.other_variant.get()), (1, 1));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
