//! The simulator's frames are the converter's: every fixture game replayed through
//! `gmmahjong::sim` — each kyoku dealt from the wall the log implies (`reference::build_wall`),
//! each decision answered with the log's choice and its physical tiles — shows every seat the
//! frames `convert_game` writes for that game (tile planes, player and table rows, legality),
//! the engine starts every kyoku with the log's dealer / counters / scores, and ends with the
//! log's final scores.

use super::game::{ty, walk_game, Decision};
use super::reference::build_wall;
use super::state::*;
use super::tests::fixture_games;
use gmmahjong::sim::{Hint, MjGame};
use serde_json::Value;
use std::collections::VecDeque;

/// One expected frame: its encoding, legality and the log's answer.
struct Expected {
    tiles_q: Vec<u8>,
    players: Vec<f32>,
    table: Vec<f32>,
    legal: Legal,
    label: (i64, i64),
    hint: Hint,
    event: usize,
}

/// The seat's response event to the decision at `at`: its next dahai / call (skipping dora,
/// reach, reach_accepted), for the physical tiles.
fn hint_of(evs: &[Value], at: usize, seat: usize, label: (i64, i64)) -> Hint {
    let mut h = Hint::default();
    if label.0 == PASS || label.0 == HORA || label.0 == RYUKYOKU {
        return h;
    }
    for ev in &evs[at + 1..] {
        let t = ty(ev);
        if matches!(t, "dora" | "reach" | "reach_accepted") {
            continue;
        }
        if ev.get("actor").and_then(|v| v.as_i64()) != Some(seat as i64) {
            if matches!(t, "tsumo" | "dahai") {
                break;
            }
            continue;
        }
        if t == "dahai" {
            h.pai = ev["pai"].as_str().map(String::from);
            h.tsumogiri = ev.get("tsumogiri").and_then(|v| v.as_bool());
        } else if matches!(t, "chi" | "pon" | "daiminkan" | "ankan" | "kakan") {
            h.consumed = ev.get("consumed").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect());
            if t == "kakan" {
                h.consumed = ev["pai"].as_str().map(|p| vec![p.to_string()]);
            }
        }
        break;
    }
    h
}

fn expected(evs: &[Value]) -> [VecDeque<Expected>; 4] {
    let mut out: [VecDeque<Expected>; 4] = Default::default();
    walk_game(evs, &mut |d: &Decision| {
        let Some(label) = d.label else { return };
        if !(d.own || d.legal.has_claim() || label.0 != PASS) {
            return;
        }
        let mut b = ObsBuf::default();
        d.st.encode(d.seat, d.focus, d.own, d.focus_actor, &mut b);
        // a triple ron (三家和) is logged as a bare ryukyoku after the discard, its claims
        // weighted 0 with the label pass: the seats that could ron did (the abortive draw)
        let triple_ron = !d.own && d.weight == 0.0 && d.legal.types[HORA as usize]
            && evs[d.event + 1..].iter().find(|e| !matches!(ty(e), "dora" | "reach_accepted")).map(|e| ty(e) == "ryukyoku").unwrap_or(false);
        let label = if triple_ron { (HORA, 0) } else { label };
        out[d.seat].push_back(Expected { tiles_q: b.tiles_q, players: b.players, table: b.table, legal: d.legal.clone(), label,
                                         hint: hint_of(evs, d.event, d.seat, label), event: d.event });
    }).ok().expect("the fixture converts");
    out
}

fn kyoku_ranges(evs: &[Value]) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = (0..evs.len()).filter(|&i| ty(&evs[i]) == "start_kyoku").collect();
    starts.iter().enumerate().map(|(n, &s)| (s, starts.get(n + 1).copied().unwrap_or(evs.len()))).collect()
}

#[test]
fn fixture_games_replay_through_the_simulator_frame_for_frame() {
    let mut games = 0;
    let mut frames = 0;
    for (name, evs) in fixture_games() {
        let ranges = kyoku_ranges(&evs);
        let walls: Vec<Vec<u8>> = ranges.iter().map(|&(s, e)| build_wall(&evs[s..e]).unwrap_or_else(|e| panic!("{name}: {e}"))).collect();
        let mut want = expected(&evs);
        let mut g = MjGame::with_walls(walls).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut kyoku_seen = 0;
        while !g.done() {
            // the engine's kyoku is the log's
            let k = g.kyoku().unwrap();
            let (s, _) = ranges[kyoku_seen.min(ranges.len() - 1)];
            if ty(&evs[s]) == "start_kyoku" && (k.oya as i64, k.honba) != (evs[s]["oya"].as_i64().unwrap(), evs[s]["honba"].as_i64().unwrap()) {
                kyoku_seen += 1;
            }
            let mut choices = Vec::new();
            for p in g.pending() {
                let e = want[p.seat].pop_front().unwrap_or_else(|| panic!("{name}: an extra decision of seat {}", p.seat));
                let mut b = ObsBuf::default();
                g.encode(p, &mut b);
                assert_eq!(b.tiles_q, e.tiles_q, "{name}: seat {} tile planes at event {}", p.seat, e.event);
                assert_eq!(b.players, e.players, "{name}: seat {} player rows at event {}", p.seat, e.event);
                assert_eq!(b.table, e.table, "{name}: seat {} table row at event {}", p.seat, e.event);
                assert_eq!(p.legal, e.legal, "{name}: seat {} legality at event {}", p.seat, e.event);
                choices.push((e.label.0, e.label.1, Some(e.hint)));
                frames += 1;
            }
            g.act(&choices).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        for (seat, q) in want.iter().enumerate() {
            assert!(q.is_empty(), "{name}: seat {seat} has {} frames the simulator never asked", q.len());
        }
        assert_eq!(g.stats.claim_mismatch, 0, "{name}");
        // the log's final placement (the engine also pays the leftover riichi sticks)
        let (_, log_scores) = walk_game(&evs, &mut |_: &Decision| {}).ok().expect("converts");
        let rank = |sc: [i64; 4]| {
            let mut o: Vec<usize> = (0..4).collect();
            o.sort_by_key(|&s| (-sc[s], s));
            o
        };
        assert_eq!(rank(g.scores()), rank(log_scores), "{name}: final scores {:?} vs the log's {log_scores:?}", g.scores());
        games += 1;
    }
    eprintln!("sim parity: {games} games, {frames} frames identical");
    assert!(games >= 3 && frames > 1000, "{games} games, {frames} frames");
}
