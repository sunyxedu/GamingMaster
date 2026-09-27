//! Differential test of the converter's legality against the game engine of RiichiEnv
//! (crate riichienv-core, Apache-2.0), an independent Tenhou-rules implementation.
//!
//! Every kyoku of a log is replayed through `GameState::step`: the wall is rebuilt from the
//! log (the deal, every draw and rinshan draw, the dora / ura indicators; tiles the log never
//! shows fill the rest), the recorded actions are stepped, and at every decision point —
//! each own draw, each chi / pon, and for every discard / kakan the three other seats — the
//! engine's legal actions (types and tile kinds; a riichi's tiles by stepping the riichi on a
//! copy) must equal the converter's masks (`walk_game`).  The engine tracks furiten, the
//! one-han minimum, kuikae, kan limits, abortive draws, haitei / houtei / rinshan / chankan
//! on its own, so the comparison checks the converter's whole state machine and rules.
//!
//! Runs on the committed fixture games; `GM_MAHJONG_REF_RAW=<dir of .mjson>` repeats it on
//! more (`GM_MAHJONG_REF_MAX_GAMES`, default 2000).

use super::game::{ty, walk_game, Decision};
use super::rules::*;
use super::state::*;
use super::tests::{fixture_games, read_events};
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::state::GameState;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A legal set: per action type, the tile kinds (empty for tile-less types).
type LegalSet = BTreeMap<i64, BTreeSet<usize>>;
/// (prompting event index, seat) → legal set
type Decisions = BTreeMap<(usize, usize), LegalSet>;

fn from_legal(l: &Legal) -> LegalSet {
    (0..N_TYPES)
        .filter(|&t| l.types[t])
        .map(|t| (t as i64, if USES_TILE[t] { (0..NK).filter(|&k| l.tiles[t][k]).collect() } else { BTreeSet::new() }))
        .collect()
}

/// The converter's decisions of a game.
fn converter_decisions(evs: &[Value]) -> Decisions {
    let mut out = Decisions::new();
    walk_game(evs, &mut |d: &Decision| {
        out.insert((d.event, d.seat), from_legal(d.legal));
    })
    .ok()
    .expect("the converter skipped the game");
    out
}

fn is_red_id(t: u8) -> bool {
    matches!(t, 16 | 52 | 88)
}

/// Free 136-format ids per (kind, red).
struct Pool {
    normal: Vec<Vec<u8>>,
    red: Vec<Option<u8>>,
}

impl Pool {
    fn new() -> Self {
        let mut normal = vec![Vec::new(); NK];
        let mut red = vec![None; NK];
        for k in 0..NK {
            for j in 0..4u8 {
                let id = (k * 4) as u8 + j;
                if is_red_id(id) {
                    red[k] = Some(id);
                } else {
                    normal[k].push(id);
                }
            }
        }
        Pool { normal, red }
    }
    fn take(&mut self, p: &str) -> Result<u8, String> {
        let k = tile_kind(p).map_err(|_| format!("bad tile {p}"))?;
        if is_red(p) {
            return self.red[k].take().ok_or(format!("second red {p}"));
        }
        if !self.normal[k].is_empty() {
            return Ok(self.normal[k].remove(0));
        }
        // a game without red fives (aka_flag false) uses the fourth copy as a plain five
        self.red[k].take().ok_or(format!("fifth {p}"))
    }
    fn rest(self) -> Vec<u8> {
        let mut v: Vec<u8> = self.normal.into_iter().flatten().chain(self.red.into_iter().flatten()).collect();
        v.sort_unstable();
        v
    }
}

fn i64_of(ev: &Value, k: &str) -> i64 {
    ev.get(k).and_then(|v| v.as_i64()).unwrap_or(0)
}
fn str_of<'a>(ev: &'a Value, k: &str) -> &'a str {
    ev.get(k).and_then(|v| v.as_str()).unwrap_or("")
}
fn actor(ev: &Value) -> usize {
    i64_of(ev, "actor") as usize
}

/// The wall RiichiEnv deals the kyoku `ky` (events from start_kyoku) from, in the order
/// `GameState::_initialize_round` takes it (it reverses the vector: the deal and the live
/// draws pop from the end, rinshan draws come from the front, dora / ura indicators sit at
/// 4, 6, .. / 5, 7, ..).
pub(super) fn build_wall(ky: &[Value]) -> Result<Vec<u8>, String> {
    let sk = &ky[0];
    let oya = i64_of(sk, "oya") as usize;
    let mut pool = Pool::new();
    let mut t: Vec<Option<u8>> = vec![None; 136];
    let tehais = sk["tehais"].as_array().ok_or("no tehais")?;
    for r in 0..3 {
        for idx in 0..4 {
            let p = (idx + oya) % 4;
            for j in 0..4 {
                t[135 - 16 * r - 4 * idx - j] = Some(pool.take(tehais[p][4 * r + j].as_str().ok_or("tile")?)?);
            }
        }
    }
    for idx in 0..4 {
        let p = (idx + oya) % 4;
        t[87 - idx] = Some(pool.take(tehais[p][12].as_str().ok_or("tile")?)?);
    }
    let mut live = 83usize;
    let mut rinshan = 0usize;
    let mut after_kan = false;
    let mut doras = vec![str_of(sk, "dora_marker").to_string()];
    let mut uras: Vec<String> = Vec::new();
    for ev in &ky[1..] {
        match ty(ev) {
            "tsumo" => {
                let id = pool.take(str_of(ev, "pai"))?;
                if after_kan {
                    if rinshan >= 4 {
                        return Err("fifth rinshan draw".into());
                    }
                    t[rinshan] = Some(id);
                    rinshan += 1;
                } else {
                    if live < 14 {
                        return Err("live wall exhausted".into());
                    }
                    t[live] = Some(id);
                    live -= 1;
                }
                after_kan = false;
            }
            "daiminkan" | "ankan" | "kakan" => after_kan = true,
            "dora" => doras.push(str_of(ev, "dora_marker").to_string()),
            "hora" if uras.is_empty() => {
                if let Some(u) = ev.get("ura_markers").and_then(|v| v.as_array()) {
                    uras = u.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                }
            }
            _ => {}
        }
    }
    for (i, d) in doras.iter().enumerate() {
        t[4 + 2 * i] = Some(pool.take(d)?);
    }
    for (i, u) in uras.iter().enumerate().take(5) {
        t[5 + 2 * i] = Some(pool.take(u)?);
    }
    let mut rest = pool.rest().into_iter();
    let mut wall: Vec<u8> = t.into_iter().map(|x| x.or_else(|| rest.next()).unwrap()).collect();
    wall.reverse();
    Ok(wall)
}

/// An id of kind / redness `p` in `hand`.
fn pick(hand: &[u8], p: &str) -> Result<u8, String> {
    let k = tile_kind(p).map_err(|_| format!("bad tile {p}"))?;
    let red = is_red(p);
    hand.iter()
        .copied()
        .find(|&t| t as usize / 4 == k && is_red_id(t) == red)
        .or_else(|| hand.iter().copied().find(|&t| t as usize / 4 == k))
        .ok_or(format!("{p} not in hand {hand:?}"))
}

fn kind_set(tiles: impl Iterator<Item = u8>) -> BTreeSet<usize> {
    tiles.map(|t| t as usize / 4).collect()
}

/// RiichiEnv's legal actions of `pid` as a legal set; the riichi tiles by stepping the
/// declaration on a copy of the state.
fn engine_legal(gs: &GameState, pid: u8) -> LegalSet {
    let acts = gs._get_legal_actions_internal(pid);
    let mut out = LegalSet::new();
    for a in &acts {
        let (t, tile): (i64, Option<usize>) = match a.action_type {
            ActionType::Discard => (DAHAI, a.tile.map(|x| x as usize / 4)),
            ActionType::Riichi => (RIICHI, None),
            ActionType::Chi => {
                let base = a.consume_tiles.iter().copied().chain(a.tile).map(|x| x as usize / 4).min();
                (CHI, base)
            }
            ActionType::Pon => (PON, a.tile.map(|x| x as usize / 4)),
            ActionType::Daiminkan | ActionType::Ankan | ActionType::Kakan => {
                (KAN, a.tile.or(a.consume_tiles.first().copied()).map(|x| x as usize / 4))
            }
            ActionType::Tsumo | ActionType::Ron => (HORA, None),
            ActionType::KyushuKyuhai => (RYUKYOKU, None),
            ActionType::Pass => (PASS, None),
            ActionType::Kita => continue,
        };
        let e = out.entry(t).or_default();
        if let Some(k) = tile {
            e.insert(k);
        }
    }
    if out.contains_key(&RIICHI) {
        let mut g = gs.clone();
        g.step(&HashMap::from([(pid, Action::new(ActionType::Riichi, None, vec![], Some(pid)))]));
        let tiles = kind_set(
            g._get_legal_actions_internal(pid).iter().filter(|a| a.action_type == ActionType::Discard).filter_map(|a| a.tile),
        );
        out.insert(RIICHI, tiles);
    }
    out
}

fn wind(s: &str) -> u8 {
    match s {
        "S" => 1,
        "W" => 2,
        "N" => 3,
        _ => 0,
    }
}

/// Wins whose payments RiichiEnv computed like Tenhou did (the log's hora deltas), and the
/// others: a check of the shared hand evaluator (yaku, han, fu, dora) on real hands.
#[derive(Default)]
struct Scoring {
    agree: usize,
    differ: Vec<String>,
}

/// The log's summed hora deltas for the wins at `evs[j..]` (consecutive hora events).
fn hora_deltas(evs: &[Value], mut j: usize, end: usize) -> [i64; 4] {
    let mut d = [0i64; 4];
    while j < end && ty(&evs[j]) == "hora" {
        if let Some(v) = evs[j].get("deltas").and_then(|v| v.as_array()) {
            for (i, x) in v.iter().take(4).enumerate() {
                d[i] += x.as_i64().unwrap_or(0);
            }
        }
        j += 1;
    }
    d
}

fn check_payments(gs: &GameState, before: [i32; 4], want: [i64; 4], at: usize, sc: &mut Scoring) {
    let got: [i64; 4] = std::array::from_fn(|i| (gs.players[i].score - before[i]) as i64);
    if got == want {
        sc.agree += 1;
    } else {
        sc.differ.push(format!("event {at}: RiichiEnv pays {got:?}, the log {want:?}"));
    }
}

/// Replay the kyoku `evs[start..end]` (start = its start_kyoku) through RiichiEnv, recording
/// its legal sets at every decision point (keys: absolute event index, seat).
fn engine_kyoku(evs: &[Value], start: usize, end: usize, out: &mut Decisions, sc: &mut Scoring) -> Result<(), String> {
    let ky = &evs[start..end];
    let sk = &ky[0];
    let wall = build_wall(ky)?;
    let rw = wind(str_of(sk, "bakaze"));
    let mut gs = GameState::new(0, true, Some(0), rw, GameRule::default_tenhou());
    let scores: Vec<i32> = sk["scores"].as_array().ok_or("scores")?.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect();
    gs._initialize_round(i64_of(sk, "oya") as u8, rw, i64_of(sk, "honba") as u8, i64_of(sk, "kyotaku") as u32, Some(wall),
                         Some(scores));
    let step = |gs: &mut GameState, acts: HashMap<u8, Action>, at: usize| -> Result<(), String> {
        gs.step(&acts);
        match &gs.last_error {
            Some(e) => Err(format!("event {at}: RiichiEnv refused {acts:?}: {e}")),
            None => Ok(()),
        }
    };
    // claims of the three seats other than `a` on the discard / kakan at `i`
    let record_claims = |gs: &GameState, a: usize, i: usize, out: &mut Decisions| {
        for rel in 1..4 {
            let sidx = (a + rel) % 4;
            let set = if gs.phase == Phase::WaitResponse && !gs.is_done {
                engine_legal(gs, sidx as u8)
            } else {
                LegalSet::from([(PASS, BTreeSet::new())])
            };
            out.insert((i, sidx), set);
        }
    };
    let mut i = start + 1;
    while i < end {
        let ev = &evs[i];
        match ty(ev) {
            "tsumo" => {
                let a = actor(ev);
                if gs.phase != Phase::WaitAct || gs.current_player as usize != a {
                    return Err(format!("event {i}: the log draws for seat {a}, RiichiEnv waits on {:?} / seat {}", gs.phase,
                                       gs.current_player));
                }
                let drawn = gs.drawn_tile.ok_or(format!("event {i}: RiichiEnv drew nothing"))?;
                if drawn as usize / 4 != tile_kind(str_of(ev, "pai")).map_err(|_| "tile")? {
                    return Err(format!("event {i}: RiichiEnv drew {drawn}, the log {}", str_of(ev, "pai")));
                }
                out.insert((i, a), engine_legal(&gs, a as u8));
            }
            "chi" | "pon" => {
                // RiichiEnv applied the call when the response was stepped; the caller discards
                let a = actor(ev);
                out.insert((i, a), engine_legal(&gs, a as u8));
            }
            "reach" => {
                let a = actor(ev) as u8;
                step(&mut gs, HashMap::from([(a, Action::new(ActionType::Riichi, None, vec![], Some(a)))]), i)?;
            }
            "dahai" => {
                let a = actor(ev);
                // the drawn copy itself for a tsumogiri (the only discard a riichi hand has),
                // another copy of the kind for a discard from the hand
                let p = str_of(ev, "pai");
                let tsumogiri = ev.get("tsumogiri").and_then(|v| v.as_bool()).unwrap_or(false);
                let id = match gs.drawn_tile {
                    Some(d) if tsumogiri && pick(&[d], p).is_ok() => d,
                    d => {
                        let others: Vec<u8> = gs.players[a].hand.iter().copied().filter(|&t| Some(t) != d).collect();
                        pick(&others, p).or_else(|_| pick(&gs.players[a].hand, p))?
                    }
                };
                step(&mut gs, HashMap::from([(a as u8, Action::new(ActionType::Discard, Some(id), vec![], Some(a as u8)))]), i)?;
                record_claims(&gs, a, i, out);
                // the response: every ron, else one call (left in the log for its own turn)
                let mut j = i + 1;
                while j < end && matches!(ty(&evs[j]), "reach_accepted" | "dora") {
                    j += 1;
                }
                let mut acts = HashMap::new();
                while j < end && ty(&evs[j]) == "hora" && i64_of(&evs[j], "target") as usize == a {
                    let r = actor(&evs[j]) as u8;
                    acts.insert(r, Action::new(ActionType::Ron, Some(id), vec![], Some(r)));
                    j += 1;
                }
                if acts.is_empty() && j < end && ty(&evs[j]) == "ryukyoku" && gs.phase == Phase::WaitResponse {
                    // a triple ron (三家和) ends the kyoku in a draw
                    let rons: Vec<u8> = (1..4).map(|r| ((a + r) % 4) as u8).filter(|&s| {
                        gs.current_claims.get(&s).map(|v| v.iter().any(|x| x.action_type == ActionType::Ron)).unwrap_or(false)
                    }).collect();
                    if rons.len() == 3 {
                        for r in rons {
                            acts.insert(r, Action::new(ActionType::Ron, Some(id), vec![], Some(r)));
                        }
                    }
                }
                if acts.is_empty() && j < end && matches!(ty(&evs[j]), "chi" | "pon" | "daiminkan") {
                    let c = &evs[j];
                    let ca = actor(c) as u8;
                    let hand = gs.players[ca as usize].hand.clone();
                    let mut cons = Vec::new();
                    let mut left = hand.clone();
                    for p in c["consumed"].as_array().ok_or("consumed")? {
                        let x = pick(&left, p.as_str().unwrap_or(""))?;
                        left.retain(|&y| y != x);
                        cons.push(x);
                    }
                    let at = match ty(c) {
                        "chi" => ActionType::Chi,
                        "pon" => ActionType::Pon,
                        _ => ActionType::Daiminkan,
                    };
                    acts.insert(ca, Action::new(at, Some(id), cons, Some(ca)));
                }
                let rons = acts.values().any(|x: &Action| x.action_type == ActionType::Ron) && ty(&evs[j - 1]) == "hora";
                let before: [i32; 4] = std::array::from_fn(|p| gs.players[p].score);
                if gs.phase == Phase::WaitResponse {
                    step(&mut gs, acts, i)?;
                } else if !acts.is_empty() {
                    return Err(format!("event {i}: the log answers the discard with {acts:?}, RiichiEnv allows no claim"));
                }
                if rons {
                    let mut h = j;
                    while h > i + 1 && ty(&evs[h - 1]) == "hora" {
                        h -= 1;
                    }
                    check_payments(&gs, before, hora_deltas(evs, h, end), i, sc);
                }
            }
            "ankan" => {
                let a = actor(ev) as u8;
                let k = tile_kind(ev["consumed"][0].as_str().unwrap_or("")).map_err(|_| "tile")?;
                let ids: Vec<u8> = gs.players[a as usize].hand.iter().copied().filter(|&t| t as usize / 4 == k).collect();
                let lowest = (k * 4) as u8;
                step(&mut gs, HashMap::from([(a, Action::new(ActionType::Ankan, Some(lowest), ids, Some(a)))]), i)?;
                if gs.phase == Phase::WaitResponse {
                    // a robbable ankan (the converter has none: Tenhou forbids it)
                    record_claims(&gs, a as usize, i, out);
                    step(&mut gs, HashMap::new(), i)?;
                }
            }
            "kakan" => {
                let a = actor(ev) as u8;
                let id = pick(&gs.players[a as usize].hand, str_of(ev, "pai"))?;
                step(&mut gs, HashMap::from([(a, Action::new(ActionType::Kakan, Some(id), vec![], Some(a)))]), i)?;
                record_claims(&gs, a as usize, i, out);
                let mut j = i + 1;
                let mut acts = HashMap::new();
                while j < end && ty(&evs[j]) == "hora" && i64_of(&evs[j], "target") as u8 == a {
                    let r = actor(&evs[j]) as u8;
                    acts.insert(r, Action::new(ActionType::Ron, Some(id), vec![], Some(r)));
                    j += 1;
                }
                let rons = !acts.is_empty();
                let before: [i32; 4] = std::array::from_fn(|p| gs.players[p].score);
                if gs.phase == Phase::WaitResponse {
                    step(&mut gs, acts, i)?;
                } else if rons {
                    return Err(format!("event {i}: chankan in the log, RiichiEnv allows none"));
                }
                if rons {
                    check_payments(&gs, before, hora_deltas(evs, i + 1, end), i, sc);
                }
            }
            "hora" if i64_of(ev, "actor") == i64_of(ev, "target") => {
                let a = actor(ev) as u8;
                let t = gs.drawn_tile;
                let before: [i32; 4] = std::array::from_fn(|p| gs.players[p].score);
                step(&mut gs, HashMap::from([(a, Action::new(ActionType::Tsumo, t, vec![], Some(a)))]), i)?;
                check_payments(&gs, before, hora_deltas(evs, i, end), i, sc);
                break;
            }
            "hora" => break, // rons were stepped with their discard / kakan
            "ryukyoku" => {
                if gs.phase == Phase::WaitAct && !gs.is_done && gs.drawn_tile.is_some() && gs.players[gs.current_player as usize].discards.is_empty() {
                    let a = gs.current_player;
                    let _ = step(&mut gs, HashMap::from([(a, Action::new(ActionType::KyushuKyuhai, None, vec![], Some(a)))]), i);
                }
                break;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(())
}

/// Compare the converter with RiichiEnv on one game: the list of disagreements.
fn compare_game(name: &str, evs: &[Value], sc: &mut Scoring) -> Result<Vec<String>, String> {
    let mine = converter_decisions(evs);
    let mut theirs = Decisions::new();
    let starts: Vec<usize> = (0..evs.len()).filter(|&i| ty(&evs[i]) == "start_kyoku").collect();
    let mut kyoku_sc = Scoring::default();
    for (n, &s) in starts.iter().enumerate() {
        let e = starts.get(n + 1).copied().unwrap_or(evs.len());
        engine_kyoku(evs, s, e, &mut theirs, &mut kyoku_sc).map_err(|m| format!("{name}: kyoku at event {s}: {m}"))?;
    }
    sc.agree += kyoku_sc.agree;
    sc.differ.extend(kyoku_sc.differ.into_iter().map(|d| format!("{name} {d}")));
    let mut diffs = Vec::new();
    for (key, want) in &theirs {
        match mine.get(key) {
            Some(got) if got == want => {}
            got => diffs.push(format!("{name} event {} seat {}: converter {:?}, RiichiEnv {:?}", key.0, key.1, got, want)),
        }
    }
    for key in mine.keys() {
        if !theirs.contains_key(key) {
            // decisions after the engine ended the kyoku (none expected)
            diffs.push(format!("{name} event {} seat {}: no RiichiEnv decision", key.0, key.1));
        }
    }
    Ok(diffs)
}

#[derive(Default)]
struct Report {
    games: usize,
    decisions: usize,
    diffs: Vec<String>,
    errors: Vec<String>,
    /// logs the converter finds breaking the rules (mask mismatches): RiichiEnv must refuse
    /// them too (`corrupt_accepted` lists those it replays)
    corrupt: usize,
    corrupt_accepted: Vec<String>,
    scoring: Scoring,
}

fn report(games: Vec<(String, Vec<Value>)>) -> Report {
    let mut r = Report::default();
    for (name, evs) in games {
        let (_, mismatch) = super::game::convert_game(&evs, &name, 1).ok().expect("converter skipped the game");
        let result = compare_game(&name, &evs, &mut r.scoring);
        if mismatch > 0 {
            r.corrupt += 1;
            if result.is_ok() {
                r.corrupt_accepted.push(name);
            }
            continue;
        }
        r.decisions += converter_decisions(&evs).len();
        match result {
            Ok(d) => r.diffs.extend(d),
            Err(e) => r.errors.push(e),
        }
        r.games += 1;
    }
    r
}

#[test]
fn legality_matches_riichienv_on_the_fixture_games() {
    let r = report(fixture_games());
    assert!(r.errors.is_empty(), "{} replay errors:\n{}", r.errors.len(), r.errors.join("\n"));
    assert!(r.diffs.is_empty(), "{} of {} decisions differ:\n{}", r.diffs.len(), r.decisions, r.diffs[..r.diffs.len().min(30)].join("\n"));
    assert!(r.games >= 16 && r.decisions > 20000 && r.corrupt == 0, "{} games, {} decisions", r.games, r.decisions);
    // the hand evaluator scores the fixtures' wins as Tenhou did
    assert!(r.scoring.differ.is_empty() && r.scoring.agree > 100, "{} wins agree, {:?}", r.scoring.agree, r.scoring.differ);
}

/// The wide sweep: GM_MAHJONG_REF_RAW=<dir of .mjson> (skipped when unset).
#[test]
fn legality_matches_riichienv_on_more_games() {
    let Ok(root) = std::env::var("GM_MAHJONG_REF_RAW") else { return };
    let max: usize = std::env::var("GM_MAHJONG_REF_MAX_GAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let paths = crate::common::find_files(std::path::Path::new(&root), &[".mjson"]).unwrap();
    let games: Vec<(String, Vec<Value>)> = paths
        .iter()
        .take(max)
        .map(|p| (p.file_name().unwrap().to_string_lossy().to_string(), read_events(p)))
        .filter(|(_, evs)| walk_game(evs, &mut |_| {}).is_ok())
        .collect();
    let r = report(games);
    eprintln!("{} games, {} decisions, {} differ, {} replay errors; {} corrupt logs ({} replayed by RiichiEnv); {} wins \
               paid as in the log, {} not", r.games, r.decisions, r.diffs.len(), r.errors.len(), r.corrupt,
              r.corrupt_accepted.len(), r.scoring.agree, r.scoring.differ.len());
    for d in r.diffs.iter().take(40).chain(r.errors.iter().take(20)).chain(r.scoring.differ.iter().take(20)) {
        eprintln!("  {d}");
    }
    assert!(r.diffs.is_empty() && r.errors.is_empty() && r.corrupt_accepted.is_empty());
}
