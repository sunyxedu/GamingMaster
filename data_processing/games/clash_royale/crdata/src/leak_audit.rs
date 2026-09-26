//! Information-leak audit of the frame builder (tests only).
//!
//! Every per-frame column the loader turns into a network input (`raw.*` minus the
//! label columns, expanded by `expand.py`) must be information the acting player
//! really has at that frame:
//!
//! * FUTURE (causality): frames `0..=f` are a function of the record up to frame
//!   `f`'s tick. Test: truncate the replay at `t_f` (events with tick < t_f, match
//!   length = t_f, result / crowns / final tower HP blanked), rebuild, and compare
//!   frames `0..=f` bit for bit with the full build.
//! * PRIVACY: a player's frames are invariant to what only the opponent knows —
//!   deck cards not yet played, deck order (hence their hand / cycle), card levels,
//!   evolution / hero ownership, tower troop, end-of-match facts. Test: perturb all
//!   of those at the payload level (so the parser is covered too), rebuild, compare.
//! * Opponent elixir is not displayed by the client; it is legitimate only as a
//!   deterministic function of public history (timeline + the opponent's observed
//!   plays and their catalog costs). Test: an independent integrator that sees only
//!   `(tick, cost)` of the opponent's visible events reproduces `opp_elixir`.
//!
//! * SEMANTIC (synthetic matches with a known true cycle and a uniformly random
//!   policy): no input may predict the coming play better than the TRUE hand does
//!   (chance); the hindsight hand does, in the opening.
//!
//! Known leaks are pinned in [`KNOWN_FUTURE_LEAKS`]: the causality tests fail both
//! when a new column starts leaking and when a listed one is fixed (then shrink the
//! list); likewise the `known_leak_*` / `known_latent_leak_*` tests. Real-data checks read `tests/fixtures/clash_royale/replays.jsonl` (skipped
//! when absent); `CRDATA_AUDIT_PARQUET=<dir or part>` (+ `CRDATA_AUDIT_LIMIT`) runs
//! the same checks on downloaded replays and prints leak statistics
//! (`cargo test --release -p crdata leak_audit -- --nocapture`).

use crate::catalog::{CardKind, Catalog, Timeline};
use crate::frames::{build_side, FrameColumns, FrameConfig, Sample, HAND};
use crate::hand::HandState;
use crate::replay::{parse_payload, read_rows, EventKind, Replay, SideId};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Per-frame columns that reach the network (via `expand.py`) at decision time.
const INPUT_FIELDS: [&str; 17] = [
    "tick", "time_s", "phase", "overtime", "elixir", "opp_elixir", "hand_ids", "hand_form", "hand_kind", "hand_cost",
    "hand_prob", "next_card", "recent_side", "recent_card", "recent_x", "recent_y", "recent_age",
];

/// Input columns that currently depend on the future of the record: the hand-cycle
/// inference (`hand::infer_initial_hand`) filters initial (hand, queue) states by
/// EVERY play of the match, so the canonical hand, the next card and
/// "P(card really in hand)" at frame t are chosen with hindsight. Confined to the
/// opening (after 4 in-deck plays the queue is exactly the last 4 played cards), but
/// there it is blatant: on 4973 real replays the hindsight probability is 1.0 for the
/// card about to be played and < 1 for every other hand card in 99.7% of the
/// decisions before the 4th own play (10.9% of all play decisions).
const KNOWN_FUTURE_LEAKS: [&str; 6] = ["hand_cost", "hand_form", "hand_ids", "hand_kind", "hand_prob", "next_card"];

fn load() -> (Catalog, Timeline) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../catalog");
    (Catalog::load(&root.join("cards.json")).unwrap(), Timeline::load(&root.join("timeline.json")).unwrap())
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../tests/fixtures/clash_royale/replays.jsonl")
}

// ---- column comparison ---------------------------------------------------------

/// First differing frame (among the first `n`) per column; floats compare bitwise.
fn column_diffs(a: &FrameColumns, b: &FrameColumns, n: usize, with_labels: bool) -> BTreeMap<&'static str, usize> {
    assert!(a.len() >= n && b.len() >= n, "compare {n} frames of {} / {}", a.len(), b.len());
    let k = if a.len() == 0 { 0 } else { a.recent_side.len() / a.len() };
    let mut out = BTreeMap::new();
    macro_rules! cmp {
        ($w:expr, $key:expr; $($f:ident),*) => {$(
            let w = $w;
            if let Some(i) = (0..n * w).find(|&i| $key(a.$f[i]) != $key(b.$f[i])) {
                out.insert(stringify!($f), i / w);
            }
        )*};
    }
    let bits = |x: f32| x.to_bits();
    cmp!(1, |x| x; tick, phase, overtime, next_card);
    cmp!(1, bits; time_s, elixir, opp_elixir);
    cmp!(HAND, |x| x; hand_ids, hand_form, hand_kind);
    cmp!(HAND, bits; hand_cost, hand_prob);
    cmp!(k, |x| x; recent_side, recent_card, recent_x, recent_y, recent_age);
    if with_labels {
        cmp!(1, |x| x; waited, card_slot, place_x, place_y, ability, ability_card);
    }
    out
}

// ---- truncation (causality) ------------------------------------------------------

/// The record as it stood at tick `cut`: earlier events only, match length `cut`,
/// no end-of-match facts.
fn truncate(r: &Replay, cut: u32, tps: f64) -> Replay {
    let mut t = r.clone();
    t.events.retain(|e| e.tick < cut);
    t.duration_s = cut as f64 / tps;
    t.result = 0;
    for s in [&mut t.team, &mut t.opp] {
        s.crowns = 0;
        s.final_tower_hp = None;
    }
    t
}

#[derive(Default)]
struct Causality {
    /// column -> number of cut points at which some frame 0..=f differed
    leaking: BTreeMap<&'static str, usize>,
    cuts: usize,
    /// own-play decisions: the decision frame f itself (truncated at t_f) differs in
    /// the hand columns
    play_decisions: usize,
    decision_hand_ids_differ: usize,
    decision_hand_prob_differ: usize,
    decision_next_card_differ: usize,
    /// the card actually played is absent from the causal (prefix-only) hand
    causal_hand_misses_played_card: usize,
    /// hindsight P(in hand) of the played card == 1.0 while another hand card is < 1.0
    hindsight_prob_marks_played: usize,
    /// played card is the unique argmax of the hindsight / causal hand_prob row
    hindsight_unique_argmax: usize,
    causal_unique_argmax: usize,
    /// sum over decisions of 1{played in argmax set} / |argmax set| ("pick the most
    /// probable card" oracle accuracy; 0.25 = uninformative)
    hindsight_oracle: f64,
    causal_oracle: f64,
    /// the same for the opening: decisions before the 4th own play (after 4 in-deck
    /// plays the queue is exactly the last 4 played cards, so the hand is known causally)
    opening_decisions: usize,
    opening_hand_ids_differ: usize,
    opening_unique_argmax: usize,
}

fn argmax_credit(probs: &[f32], slot: usize) -> (f64, bool) {
    let m = probs.iter().cloned().fold(f32::MIN, f32::max);
    let top: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] == m).collect();
    if top.contains(&slot) {
        (1.0 / top.len() as f64, top.len() == 1)
    } else {
        (0.0, false)
    }
}

impl Causality {
    fn add(&mut self, r: &Replay, me: SideId, cat: &Catalog, tl: &Timeline, cfg: &FrameConfig, extra_every: usize) {
        let Some(full) = build_side(r, me, cat, tl, cfg) else { return };
        let tps = tl.ticks_per_second();
        let n = full.cols.len();
        let plays: BTreeSet<usize> = (0..n).filter(|&f| full.cols.card_slot[f] >= 0).collect();
        let mut cuts: BTreeSet<usize> = plays.clone();
        cuts.extend((0..n).step_by(extra_every.max(1)));
        for f in cuts {
            let tr = truncate(r, f as u32 * cfg.step_ticks, tps);
            let pre = build_side(&tr, me, cat, tl, cfg).expect("a prefix of a consistent play sequence is consistent");
            self.cuts += 1;
            for (col, _) in column_diffs(&full.cols, &pre.cols, f + 1, false) {
                *self.leaking.entry(col).or_default() += 1;
            }
            if !plays.contains(&f) {
                continue;
            }
            self.play_decisions += 1;
            let opening = plays.range(..f).count() < 4;
            self.opening_decisions += opening as usize;
            let h = f * HAND..(f + 1) * HAND;
            let slot = full.cols.card_slot[f] as usize;
            let played = full.cols.hand_ids[f * HAND + slot];
            let ids_differ = full.cols.hand_ids[h.clone()] != pre.cols.hand_ids[h.clone()];
            self.decision_hand_ids_differ += ids_differ as usize;
            self.opening_hand_ids_differ += (opening && ids_differ) as usize;
            let bits = |c: &FrameColumns| c.hand_prob[h.clone()].iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            self.decision_hand_prob_differ += (bits(&full.cols) != bits(&pre.cols)) as usize;
            self.decision_next_card_differ += (full.cols.next_card[f] != pre.cols.next_card[f]) as usize;
            let hp = &full.cols.hand_prob[h.clone()];
            self.hindsight_prob_marks_played += (hp[slot] == 1.0 && hp.iter().any(|&p| p < 1.0)) as usize;
            let (credit, unique) = argmax_credit(hp, slot);
            self.hindsight_oracle += credit;
            self.hindsight_unique_argmax += unique as usize;
            self.opening_unique_argmax += (opening && unique) as usize;
            match pre.cols.hand_ids[h.clone()].iter().position(|&c| c == played) {
                None => self.causal_hand_misses_played_card += 1,
                Some(ps) => {
                    let (credit, unique) = argmax_credit(&pre.cols.hand_prob[h.clone()], ps);
                    self.causal_oracle += credit;
                    self.causal_unique_argmax += unique as usize;
                }
            }
        }
    }

    fn leaking_set(&self) -> BTreeSet<&'static str> {
        self.leaking.keys().copied().collect()
    }

    fn report(&self, what: &str) {
        let d = self.play_decisions.max(1) as f64;
        eprintln!(
            "[leak_audit] {what}: {} cut points, leaking columns {:?}\n  {} own-play decisions: decision frame differs from the \
             prefix-only build in hand_ids {:.1}% / hand_prob {:.1}% / next_card {:.1}%; played card missing from the causal hand \
             {:.1}%\n  hindsight P(in hand)=1.0 singles out the played card {:.1}% (unique argmax {:.1}%, argmax-oracle accuracy \
             {:.1}%) vs causal posterior unique argmax {:.1}% (oracle {:.1}%; 25% = uninformative)\n  opening ({} decisions \
             before the 4th own play): hand_ids differ {:.1}%, played card singled out by P=1.0 {:.1}%",
            self.cuts,
            self.leaking,
            self.play_decisions,
            100.0 * self.decision_hand_ids_differ as f64 / d,
            100.0 * self.decision_hand_prob_differ as f64 / d,
            100.0 * self.decision_next_card_differ as f64 / d,
            100.0 * self.causal_hand_misses_played_card as f64 / d,
            100.0 * self.hindsight_prob_marks_played as f64 / d,
            100.0 * self.hindsight_unique_argmax as f64 / d,
            100.0 * self.hindsight_oracle / d,
            100.0 * self.causal_unique_argmax as f64 / d,
            100.0 * self.causal_oracle / d,
            self.opening_decisions,
            100.0 * self.opening_hand_ids_differ as f64 / self.opening_decisions.max(1) as f64,
            100.0 * self.opening_unique_argmax as f64 / self.opening_decisions.max(1) as f64,
        );
    }
}

// ---- synthetic payloads (raw schema, so the parser is covered) --------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[derive(Clone)]
struct SideSpec {
    /// (dataset card key incl. form suffix, level)
    deck: Vec<(String, u8)>,
    tower: String,
    crowns: u8,
    final_hp: [f32; 3],
}

#[derive(Clone)]
struct Ev {
    side: &'static str,
    /// event card key (base card)
    key: String,
    /// the deck key the parser takes the form from (`deck_card_key_candidates`)
    deck_key: String,
    tick: u32,
    x: i32,
    y: i32,
    ability: bool,
}

#[derive(Clone)]
struct Spec {
    team: SideSpec,
    opp: SideSpec,
    events: Vec<Ev>,
    duration: f64,
    result: &'static str,
}

fn payload(s: &Spec) -> String {
    let side = |p: &SideSpec| {
        json!({"crowns": p.crowns, "players": [{
            "deck": p.deck.iter().map(|(k, l)| json!({"card_key": k, "level": l, "name": k})).collect::<Vec<_>>(),
            "tower_card": {"card_key": p.tower, "level": 15, "name": p.tower},
            "final_tower_hitpoints": {"king": p.final_hp[0], "princess_left": p.final_hp[1], "princess_right": p.final_hp[2]},
        }]})
    };
    let events: Vec<_> = s
        .events
        .iter()
        .map(|e| {
            if e.ability {
                json!({"kind": "activate_ability", "side": e.side, "card_key": e.key, "replay_tick_20hz": e.tick,
                       "deck_card_key_candidates": [], "ability_source_candidates": [e.deck_key]})
            } else {
                json!({"kind": "play_card", "side": e.side, "card_key": e.key, "replay_tick_20hz": e.tick,
                       "form_at_play": "unknown", "deck_card_key_candidates": [e.deck_key], "ability_source_candidates": [],
                       "coordinates": {"grid_cell_floor": {"x": e.x, "y": e.y}, "inside_18x32_arena": true}})
            }
        })
        .collect();
    json!({
        "battle": {"battle_type": "pathOfLegend", "game_mode": "Ranked", "result": s.result,
                   "team": side(&s.team), "opponent": side(&s.opp)},
        "events": events,
        "replay": {"duration": {"timeline_seconds": s.duration, "display_seconds": s.duration.round()}},
        "schema_version": "royaleapi-battle-replay.v1",
    })
    .to_string()
}

fn card_pool(cat: &Catalog) -> Vec<String> {
    cat.cards.iter().filter(|c| c.cost > 0.0 && c.kind != CardKind::Other).map(|c| c.key.clone()).collect()
}

fn random_deck(pool: &[String], rng: &mut Rng) -> Vec<String> {
    let mut d: Vec<String> = Vec::new();
    while d.len() < 8 {
        let k = &pool[rng.below(pool.len())];
        if !d.contains(k) {
            d.push(k.clone());
        }
    }
    d
}

/// A match both players play from their real (shuffled) hand cycle, with random
/// timing, placements and the occasional champion ability; up to 5 minutes so
/// double / triple elixir and overtime are covered.
fn random_spec(cat: &Catalog, rng: &mut Rng) -> Spec {
    random_spec_with_truth(cat, rng).0
}

/// [`random_spec`] plus each side's TRUE initial cycle (team, opponent), as deck
/// indices into that side's `SideSpec::deck`.
fn random_spec_with_truth(cat: &Catalog, rng: &mut Rng) -> (Spec, [HandState; 2]) {
    let pool = card_pool(cat);
    let decks = [random_deck(&pool, rng), random_deck(&pool, rng)];
    // deck keys as the dataset writes them: two evolution-owning cards per deck
    let deck_keys: Vec<Vec<String>> = decks
        .iter()
        .map(|d| {
            let (e1, e2) = (rng.below(8), rng.below(8));
            d.iter().enumerate().map(|(i, k)| if i == e1 || i == e2 { format!("{k}-ev1") } else { k.clone() }).collect()
        })
        .collect();
    let mut cycles: Vec<HandState> = (0..2)
        .map(|_| {
            let mut order: Vec<u16> = (0..8).collect();
            rng.shuffle(&mut order);
            HandState::new([order[0], order[1], order[2], order[3]], [order[4], order[5], order[6], order[7]])
        })
        .collect();
    let initial = [cycles[0].clone(), cycles[1].clone()];
    let duration = 60.0 + rng.below(240) as f64 + 0.35;
    let mut events = Vec::new();
    let mut tick = 0u32;
    loop {
        tick += 5 + rng.below(90) as u32;
        if tick as f64 / 20.0 >= duration {
            break;
        }
        let s = rng.below(2);
        let side = if s == 0 { "team" } else { "opponent" };
        if rng.below(15) == 0 {
            let i = rng.below(8);
            events.push(Ev { side, key: decks[s][i].clone(), deck_key: deck_keys[s][i].clone(), tick, x: 0, y: 0, ability: true });
            continue;
        }
        let idx = cycles[s].hand[rng.below(HAND)];
        cycles[s].play(idx);
        let (key, deck_key) = (decks[s][idx as usize].clone(), deck_keys[s][idx as usize].clone());
        let (x, y) = (rng.below(18) as i32, if s == 0 { rng.below(16) } else { 16 + rng.below(16) } as i32);
        events.push(Ev { side, key, deck_key, tick, x, y, ability: false });
    }
    let side = |d: &[String]| SideSpec {
        deck: d.iter().map(|k| (k.clone(), 16)).collect(),
        tower: "tower-princess".into(),
        crowns: 1,
        final_hp: [4000.0, 0.0, 2000.0],
    };
    (Spec { team: side(&deck_keys[0]), opp: side(&deck_keys[1]), events, duration, result: "victory" }, initial)
}

/// Change everything `who` knows and the other side does not: deck cards `who`
/// never plays, deck order, levels, evolution ownership (deck keys and the form the
/// parser reads from the play events), tower troop, crowns, final tower HP, result.
fn perturb_private(s: &Spec, who: &'static str, cat: &Catalog, rng: &mut Rng) -> Spec {
    let pool = card_pool(cat);
    let mut out = s.clone();
    let played: BTreeSet<String> =
        s.events.iter().filter(|e| e.side == who && !e.ability).map(|e| e.key.clone()).collect();
    let mut used: BTreeSet<String> = s.team.deck.iter().chain(s.opp.deck.iter()).map(|(k, _)| k.clone()).collect();
    let evo = |k: &str| if k.ends_with("-ev1") { k.trim_end_matches("-ev1").to_string() } else { format!("{k}-ev1") };
    let side = if who == "team" { &mut out.team } else { &mut out.opp };
    for (k, lvl) in side.deck.iter_mut() {
        if !played.contains(k.trim_end_matches("-ev1")) {
            loop {
                let c = &pool[rng.below(pool.len())];
                if used.insert(c.clone()) {
                    *k = c.clone();
                    break;
                }
            }
        }
        *k = evo(k);
        *lvl = 9 + rng.below(8) as u8;
    }
    rng.shuffle(&mut side.deck);
    side.tower = ["cannoneer", "dagger-duchess", "royal-chef"][rng.below(3)].into();
    side.crowns = 3 - side.crowns.min(3);
    side.final_hp = [1.0, 2.0, 3.0];
    for e in out.events.iter_mut().filter(|e| e.side == who && !e.ability) {
        e.deck_key = evo(&e.deck_key);
    }
    out.result = if s.result == "victory" { "defeat" } else { "victory" };
    out
}

fn build(js: &str, me: SideId, cat: &Catalog, tl: &Timeline, cfg: &FrameConfig) -> Sample {
    let r = parse_payload("audit", js, cat).unwrap();
    build_side(&r, me, cat, tl, cfg).expect("hand inference")
}

// ---- opponent elixir: public-history reference ------------------------------------

/// Elixir after regenerating from `a` to `b` seconds under the timeline, capped.
fn regen(tl: &Timeline, mut e: f64, a: f64, b: f64) -> f64 {
    let mut s0 = 0.0;
    let n = tl.rates.len();
    for (i, &(len, spe)) in tl.rates.iter().enumerate() {
        let s1 = if i + 1 == n { f64::INFINITY } else { s0 + len };
        let (lo, hi) = (a.max(s0), b.min(s1));
        if hi > lo {
            e = (e + (hi - lo) / spe).min(tl.max_elixir);
        }
        s0 += len;
    }
    e
}

/// Elixir at each frame of a player whose only known expenses are the visible
/// `(tick, cost)` events: starting elixir, the regeneration schedule, the cap,
/// minus each observed cost (clamped at zero). Nothing else goes in.
fn reference_elixir(tl: &Timeline, public: &[(u32, f64)], frame_ticks: &[i32]) -> Vec<f64> {
    let tps = tl.ticks_per_second();
    frame_ticks
        .iter()
        .map(|&t| {
            let (mut e, mut clock) = (tl.starting_elixir, 0.0);
            for &(tick, cost) in public.iter().filter(|(tick, _)| (*tick as i32) < t) {
                let ts = tick as f64 / tps;
                e = (regen(tl, e, clock, ts) - cost).max(0.0);
                clock = ts;
            }
            regen(tl, e, clock, t as f64 / tps)
        })
        .collect()
}

/// `(tick, cost)` of every event of `side` the other player sees (plays: the
/// visible card's catalog cost; champion abilities: 1).
fn visible_spend(r: &Replay, side: SideId, cat: &Catalog) -> Vec<(u32, f64)> {
    r.events
        .iter()
        .filter(|e| e.side == side)
        .map(|e| match e.kind {
            EventKind::Play { card, .. } => (e.tick, cat.cost(card) as f64),
            EventKind::Ability { .. } => (e.tick, 1.0),
        })
        .collect()
}

fn other(s: SideId) -> SideId {
    match s {
        SideId::Team => SideId::Opp,
        SideId::Opp => SideId::Team,
    }
}

/// max |converter - reference| over frames, for `opp_elixir` (and `elixir`).
fn elixir_errors(r: &Replay, me: SideId, cat: &Catalog, tl: &Timeline, cfg: &FrameConfig) -> Option<(f64, f64)> {
    let s = build_side(r, me, cat, tl, cfg)?;
    let err = |col: &[f32], side: SideId| {
        let reference = reference_elixir(tl, &visible_spend(r, side, cat), &s.cols.tick);
        col.iter().zip(&reference).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max)
    };
    Some((err(&s.cols.opp_elixir, other(me)), err(&s.cols.elixir, me)))
}

// ---- real replays -------------------------------------------------------------------

/// `tests/fixtures/clash_royale/replays.jsonl` (`{"tag", "category", "payload"}` per
/// line): 13 real anonymised RoyaleAPI replays from `crdata fetch`'s default dataset
/// (VanguardX101/IL_Replay, part-000000 — regulation / overtime / short matches,
/// champion abilities, a Mirror play, a non-Ranked mode), stripped to the payload
/// fields `replay::parse_payload` reads.
fn fixture_replays(cat: &Catalog) -> Option<Vec<Replay>> {
    let text = std::fs::read_to_string(fixture_path()).ok()?;
    let out: Vec<Replay> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            parse_payload(v["tag"].as_str().unwrap(), &v["payload"].to_string(), cat).unwrap()
        })
        .collect();
    Some(out)
}

fn env_replays(cat: &Catalog) -> Option<Vec<Replay>> {
    let root = PathBuf::from(std::env::var_os("CRDATA_AUDIT_PARQUET")?);
    let limit: usize = std::env::var("CRDATA_AUDIT_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    let mut parts = Vec::new();
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if p.is_file() {
            out.push(p.to_path_buf());
        } else if let Ok(rd) = std::fs::read_dir(p) {
            let mut v: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            v.sort();
            for q in v {
                if q.is_dir() && q.file_name().map(|n| n != "actions").unwrap_or(true) {
                    walk(&q, out);
                } else if q.extension().map(|e| e == "parquet").unwrap_or(false) {
                    out.push(q);
                }
            }
        }
    }
    walk(&root, &mut parts);
    let mut out = Vec::new();
    for part in parts {
        for (tag, js) in read_rows(&part).ok()? {
            if out.len() >= limit {
                return Some(out);
            }
            if let Ok(r) = parse_payload(&tag, &js, cat) {
                out.push(r);
            }
        }
    }
    Some(out)
}

// ---- tests ----------------------------------------------------------------------------

#[test]
fn causality_synthetic_matches_pin_the_known_hand_leaks() {
    let (cat, tl) = load();
    let cfg = FrameConfig { step_ticks: 10, recent: 8 };
    let mut rng = Rng(0x5EED_CAFE_F00D_1234);
    let mut c = Causality::default();
    for _ in 0..24 {
        let js = payload(&random_spec(&cat, &mut rng));
        let r = parse_payload("synthetic", &js, &cat).unwrap();
        for me in [SideId::Team, SideId::Opp] {
            c.add(&r, me, &cat, &tl, &cfg, 13);
        }
    }
    c.report("synthetic");
    let public: BTreeSet<&str> = INPUT_FIELDS.iter().copied().filter(|f| !KNOWN_FUTURE_LEAKS.contains(f)).collect();
    let leaking = c.leaking_set();
    assert!(leaking.is_disjoint(&public), "FUTURE LEAK in public columns: {:?}", c.leaking);
    assert_eq!(
        leaking,
        KNOWN_FUTURE_LEAKS.iter().copied().collect::<BTreeSet<_>>(),
        "the set of future-dependent input columns changed — a new leak, or a fixed one (then update KNOWN_FUTURE_LEAKS)"
    );
}

#[test]
fn known_leak_hand_prob_at_kickoff_reveals_the_first_play() {
    // One play in the whole match: hog rider at 2.0 s. Before it (frame 0) nothing has
    // been played, so the player's belief from the record is the deal prior: each of
    // the 8 deck cards is in hand with probability 1/2 (70 of the 70 x 24 states each).
    let (cat, tl) = load();
    let cfg = FrameConfig { step_ticks: 10, recent: 4 };
    let deck: Vec<String> = ["hog-rider", "ice-spirit", "ice-golem", "skeletons", "cannon", "musketeer", "fireball", "the-log"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let side = |d: &[String]| SideSpec {
        deck: d.iter().map(|k| (k.clone(), 16)).collect(),
        tower: "tower-princess".into(),
        crowns: 0,
        final_hp: [0.0; 3],
    };
    let hog = Ev { side: "team", key: "hog-rider".into(), deck_key: "hog-rider".into(), tick: 40, x: 9, y: 14, ability: false };
    let spec = Spec { team: side(&deck), opp: side(&deck), events: vec![hog], duration: 10.0, result: "draw" };
    let full = build(&payload(&spec), SideId::Team, &cat, &tl, &cfg);
    let prefix = Spec { events: vec![], duration: 0.0, ..spec.clone() };
    let pre = build(&payload(&prefix), SideId::Team, &cat, &tl, &cfg);
    let hog_id = cat.resolve("hog-rider").0;
    // causal build: the deal prior
    assert!(pre.cols.hand_prob[..HAND].iter().all(|&p| p == 0.5), "{:?}", &pre.cols.hand_prob[..HAND]);
    // full build (what the shards hold): at kickoff the hand already contains the card
    // that will be played 2 s later, marked "certainly in hand", the rest 3/7
    let slot = full.cols.hand_ids[..HAND].iter().position(|&c| c == hog_id).expect("hindsight puts the future play in hand");
    assert_eq!(full.cols.hand_prob[slot], 1.0);
    assert!(full.cols.hand_prob[..HAND].iter().enumerate().all(|(i, &p)| i == slot || p < 0.5));
    assert_eq!(full.cols.card_slot[4], slot as i8, "and the label 2 s later picks exactly that slot");
}

#[test]
fn privacy_synthetic_frames_ignore_opponent_private_information() {
    let (cat, tl) = load();
    let cfg = FrameConfig { step_ticks: 10, recent: 8 };
    let mut rng = Rng(0xD00D_FEED_0BAD_5EED);
    for g in 0..24 {
        let spec = random_spec(&cat, &mut rng);
        for (me, who) in [(SideId::Team, "opponent"), (SideId::Opp, "team")] {
            let a = build(&payload(&spec), me, &cat, &tl, &cfg);
            let pert = perturb_private(&spec, who, &cat, &mut rng);
            let b = build(&payload(&pert), me, &cat, &tl, &cfg);
            assert_eq!(a.cols.len(), b.cols.len());
            let d = column_diffs(&a.cols, &b.cols, a.cols.len(), true);
            assert!(d.is_empty(), "PRIVACY LEAK (game {g}, {me:?} frames change with {who}'s private info): {d:?}");
        }
    }
}

#[test]
fn opp_elixir_is_a_function_of_public_opponent_plays_synthetic() {
    let (cat, tl) = load();
    let cfg = FrameConfig { step_ticks: 10, recent: 8 };
    let mut rng = Rng(0xE11C_1B00_0000_0001);
    for g in 0..24 {
        let r = parse_payload("synthetic", &payload(&random_spec(&cat, &mut rng)), &cat).unwrap();
        for me in [SideId::Team, SideId::Opp] {
            let (opp_err, own_err) = elixir_errors(&r, me, &cat, &tl, &cfg).unwrap();
            assert!(opp_err < 1e-4, "game {g} {me:?}: opp_elixir is not the public-history integral (max err {opp_err})");
            assert!(own_err < 1e-4, "game {g} {me:?}: elixir is not the own-history integral (max err {own_err})");
        }
    }
}

#[test]
fn real_fixture_causality_privacy_and_opp_elixir() {
    let (cat, tl) = load();
    let Some(replays) = fixture_replays(&cat) else {
        eprintln!("[leak_audit] SKIP: {} not found", fixture_path().display());
        return;
    };
    assert!(!replays.is_empty());
    let cfg = FrameConfig { step_ticks: 10, recent: 8 };
    let mut c = Causality::default();
    let mut rng = Rng(0xF1C5_7A2E_0000_0002);
    for r in &replays {
        for me in [SideId::Team, SideId::Opp] {
            c.add(r, me, &cat, &tl, &cfg, 17);
            // opponent elixir from public history only
            if let Some((opp_err, own_err)) = elixir_errors(r, me, &cat, &tl, &cfg) {
                assert!(opp_err < 1e-4 && own_err < 1e-4, "{}: elixir reference mismatch {opp_err} / {own_err}", r.tag);
            }
            // privacy at the struct level: the opponent's deck (unplayed cards, order,
            // levels, forms), tower, crowns, final HP, the result and the form their
            // plays carry must not move this side's frames
            let Some(a) = build_side(r, me, &cat, &tl, &cfg) else { continue };
            let mut p = r.clone();
            p.result = -p.result + if p.result == 0 { 1 } else { 0 };
            let opp_played: BTreeSet<u16> = r
                .events
                .iter()
                .filter(|e| e.side == other(me))
                .filter_map(|e| match e.kind {
                    EventKind::Play { card, .. } => Some(card),
                    _ => None,
                })
                .collect();
            let side = if me == SideId::Team { &mut p.opp } else { &mut p.team };
            let pool: Vec<u16> = (1..=cat.len() as u16).filter(|id| !side.deck.iter().any(|c| c.id == *id)).collect();
            for (i, card) in side.deck.iter_mut().enumerate() {
                if !opp_played.contains(&card.id) {
                    card.id = pool[(rng.below(pool.len()) + i) % pool.len()];
                }
                card.level = 9 + rng.below(8) as u8;
                card.form = if card.form == 0 { 1 } else { 0 };
            }
            rng.shuffle(&mut side.deck);
            side.tower_key = "cannoneer".into();
            side.crowns = 3 - side.crowns.min(3);
            side.final_tower_hp = Some((1.0, 2.0, 3.0));
            for e in p.events.iter_mut().filter(|e| e.side == other(me)) {
                if let EventKind::Play { form, .. } = &mut e.kind {
                    *form = if *form == 0 { 1 } else { 0 };
                }
            }
            let b = build_side(&p, me, &cat, &tl, &cfg).unwrap();
            let d = column_diffs(&a.cols, &b.cols, a.cols.len(), true);
            assert!(d.is_empty(), "PRIVACY LEAK in {} ({me:?}): {d:?}", r.tag);
        }
    }
    c.report(&format!("fixture ({} replays)", replays.len()));
    let public: BTreeSet<&str> = INPUT_FIELDS.iter().copied().filter(|f| !KNOWN_FUTURE_LEAKS.contains(f)).collect();
    assert!(c.leaking_set().is_disjoint(&public), "FUTURE LEAK in public columns on real replays: {:?}", c.leaking);
    // the pinned hand leak is visible on real data too (fails once the inference is causal)
    assert!(c.leaking.contains_key("hand_ids") && c.leaking.contains_key("hand_prob"), "{:?}", c.leaking);
    assert!(c.hindsight_prob_marks_played > 0);
    // ... and so is its semantic form: the kickoff hand is the set of the first 4 plays
    let (n, hits) = first4_stat(&replays, &cat, &tl, &cfg);
    eprintln!("[leak_audit] fixture: frame-0 hand == set of the first 4 own plays in {hits}/{n} sides");
    assert!(n > 0 && hits * 10 >= n * 9, "kickoff hand no longer the first 4 plays ({hits}/{n}) — leak fixed? update this pin");
}

/// Frame 0's hand equals the set of the side's first 4 in-deck plays: (sides with >= 4
/// such plays, sides where it holds). About 6/64 for a random policy's true hand (see
/// `known_leak_hindsight_hand_predicts_opening_plays_beyond_the_true_hand`).
fn first4_stat(replays: &[Replay], cat: &Catalog, tl: &Timeline, cfg: &FrameConfig) -> (usize, usize) {
    let (mut n, mut hits) = (0usize, 0usize);
    for r in replays {
        for me in [SideId::Team, SideId::Opp] {
            let Some(s) = build_side(r, me, cat, tl, cfg) else { continue };
            let deck: Vec<u16> = if me == SideId::Team { &r.team } else { &r.opp }.deck.iter().map(|c| c.id).collect();
            let first4: BTreeSet<u16> = r
                .events
                .iter()
                .filter(|e| e.side == me)
                .filter_map(|e| match e.kind {
                    EventKind::Play { card, .. } if deck.contains(&card) => Some(card),
                    _ => None,
                })
                .take(4)
                .collect();
            if first4.len() < 4 {
                continue;
            }
            n += 1;
            hits += (s.cols.hand_ids[..HAND].iter().copied().collect::<BTreeSet<_>>() == first4) as usize;
        }
    }
    (n, hits)
}

/// Large-sample statistics on downloaded replays; no-op unless CRDATA_AUDIT_PARQUET is set.
#[test]
fn real_parquet_leak_statistics() {
    let (cat, tl) = load();
    let Some(replays) = env_replays(&cat) else {
        eprintln!("[leak_audit] SKIP: set CRDATA_AUDIT_PARQUET=<parquet dir or part> for the large-sample audit");
        return;
    };
    let cfg = FrameConfig { step_ticks: 10, recent: 8 };
    let mut c = Causality::default();
    let (mut worst_opp, mut worst_own, mut n_consistent_1, mut n_samples) = (0.0f64, 0.0f64, 0usize, 0usize);
    for r in &replays {
        for me in [SideId::Team, SideId::Opp] {
            c.add(r, me, &cat, &tl, &cfg, 1_000_000);
            if let Some((o, w)) = elixir_errors(r, me, &cat, &tl, &cfg) {
                worst_opp = worst_opp.max(o);
                worst_own = worst_own.max(w);
            }
            if let Some(s) = build_side(r, me, &cat, &tl, &cfg) {
                n_samples += 1;
                n_consistent_1 += (s.meta.hand_consistent_states == 1) as usize;
            }
        }
    }
    c.report(&format!("parquet ({} replays)", replays.len()));
    eprintln!(
        "[leak_audit] {n_samples} samples: hand cycle pinned to a single initial state by the WHOLE match in {:.1}%; \
         max |opp_elixir - public reference| = {worst_opp:.2e}, own = {worst_own:.2e}",
        100.0 * n_consistent_1 as f64 / n_samples.max(1) as f64
    );
    let (n4, hits4) = first4_stat(&replays, &cat, &tl, &cfg);
    eprintln!(
        "[leak_audit] frame-0 hand == set of the first 4 own plays in {:.1}% of {n4} sides (random-policy truth: 9.4%)",
        100.0 * hits4 as f64 / n4.max(1) as f64
    );
    let public: BTreeSet<&str> = INPUT_FIELDS.iter().copied().filter(|f| !KNOWN_FUTURE_LEAKS.contains(f)).collect();
    assert!(c.leaking_set().is_disjoint(&public), "FUTURE LEAK in public columns: {:?}", c.leaking);
    assert!(worst_opp < 1e-4 && worst_own < 1e-4);
}

// ---- semantic checks against the TRUE hand (independent verification) -------------------

/// "Choose uniformly among `set`; credit if `slot` is in it" (all 4 slots when `set` is empty).
fn set_credit(set: &[usize], slot: usize) -> f64 {
    let all: Vec<usize> = (0..HAND).collect();
    let s = if set.is_empty() { &all[..] } else { set };
    if s.contains(&slot) {
        1.0 / s.len() as f64
    } else {
        0.0
    }
}

/// The truncation test shows the hand columns depend on the future, but the player
/// really does see their exact hand, so the semantic question is whether the
/// converter's hand says MORE about the coming play than the true hand does. Synthetic
/// matches know the true cycle, and under their uniformly random policy the card about
/// to be played is uniform over the true hand: no function of what the player has at
/// the decision (true hand, next card, the past) predicts it better than 1/4. On the
/// converter's inputs, in the opening (before the 4th own play):
///  * argmax of `hand_prob` picks the played card almost always;
///  * "a card held since kickoff" (hand_ids history only) scores about 1/4, 1/3, 1/2, 1
///    at the 1st..4th play: the canonical initial hand is, almost always, exactly the
///    set of the first 4 plays (only the first 4 plays constrain the cycle's hand
///    membership; after them the hand is deck minus the last 4 played, whatever the
///    start), so a card that came from the queue is never played in the opening;
///  * frame 0's hand equals the set of the first 4 plays in ~100% of sides vs 6/64 for
///    the true hand (the 2nd, 3rd and 4th plays must avoid the refills: 3/4 * 2/4 * 1/4).
/// The same predictors on the true hand score chance (checked, as a control). Pinned as
/// known leaks: once the hand representation is causal, flip the converter assertions
/// to `<= 0.29` / `< 0.2`.
#[test]
fn known_leak_hindsight_hand_predicts_opening_plays_beyond_the_true_hand() {
    let (cat, tl) = load();
    let cfg = FrameConfig { step_ticks: 10, recent: 8 };
    let mut rng = Rng(0x0DD5_1DE0_F7E5_7ACE);
    let (mut n_dec, mut conv_prob, mut conv_held, mut true_held) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    let (mut n_sides, mut conv_first4, mut true_first4) = (0usize, 0usize, 0usize);
    for _ in 0..300 {
        let (spec, truth) = random_spec_with_truth(&cat, &mut rng);
        let r = parse_payload("synthetic", &payload(&spec), &cat).unwrap();
        for (s, me) in [SideId::Team, SideId::Opp].into_iter().enumerate() {
            let side = if s == 0 { &spec.team } else { &spec.opp };
            let id_of = |i: u16| cat.resolve(&side.deck[i as usize].0).0;
            let full = build_side(&r, me, &cat, &tl, &cfg).expect("hand inference");
            let own: Vec<(u32, u16)> = r
                .events
                .iter()
                .filter(|e| e.side == me)
                .filter_map(|e| match e.kind {
                    EventKind::Play { card, .. } => Some((e.tick, card)),
                    _ => None,
                })
                .collect();
            let true_hand_after = |n: usize| {
                let mut st = truth[s].clone();
                for &(_, card) in &own[..n] {
                    let idx = st.hand.iter().position(|&i| id_of(i) == card).expect("synthetic plays come from the true hand");
                    st.play(st.hand[idx]);
                }
                st.hand.map(id_of)
            };
            if own.len() >= 4 {
                n_sides += 1;
                let first4: BTreeSet<u16> = own[..4].iter().map(|&(_, c)| c).collect();
                conv_first4 += (full.cols.hand_ids[..HAND].iter().copied().collect::<BTreeSet<_>>() == first4) as usize;
                true_first4 += (true_hand_after(0).iter().copied().collect::<BTreeSet<_>>() == first4) as usize;
            }
            let conv0 = &full.cols.hand_ids[..HAND];
            let true0 = true_hand_after(0);
            for f in 0..full.cols.len() {
                if full.cols.card_slot[f] < 0 {
                    continue;
                }
                let t = f as u32 * cfg.step_ticks;
                let n = own.iter().filter(|&&(tick, _)| tick < t).count();
                if n >= 4 {
                    continue;
                }
                n_dec += 1;
                let h = &full.cols.hand_ids[f * HAND..(f + 1) * HAND];
                let slot = full.cols.card_slot[f] as usize;
                let played = h[slot];
                conv_prob += argmax_credit(&full.cols.hand_prob[f * HAND..(f + 1) * HAND], slot).0;
                // before the 4th play a card that left the hand cannot be back, so "in the
                // frame-0 hand and in the current hand" = held continuously since kickoff
                let held: Vec<usize> = (0..HAND).filter(|&j| conv0.contains(&h[j])).collect();
                conv_held += set_credit(&held, slot);
                let th = true_hand_after(n);
                let tslot = th.iter().position(|&c| c == played).expect("the played card is in the true hand");
                let theld: Vec<usize> = (0..HAND).filter(|&j| true0.contains(&th[j])).collect();
                true_held += set_credit(&theld, tslot);
            }
        }
    }
    let d = n_dec as f64;
    let (cp, ch, th) = (conv_prob / d, conv_held / d, true_held / d);
    let (cf, tf) = (conv_first4 as f64 / n_sides as f64, true_first4 as f64 / n_sides as f64);
    eprintln!(
        "[leak_audit] random policy, {n_dec} opening decisions: P(played card) by argmax hand_prob {cp:.3}, by 'held since \
         kickoff' on the converter hand {ch:.3} vs on the TRUE hand {th:.3} (chance 0.25); frame-0 hand == first 4 plays in \
         {cf:.3} (converter) vs {tf:.3} (truth, 6/64 = 0.094) of {n_sides} sides"
    );
    // controls: the harness is right (the true hand carries no information beyond chance)
    assert!((th - 0.25).abs() < 0.04, "control: the true hand predicts a random policy's play at {th}");
    assert!((tf - 6.0 / 64.0).abs() < 0.04, "control: true frame-0 hand == first 4 plays at {tf}");
    // pinned known leaks (hand_prob, hand_ids): fail once the representation is causal
    assert!(cp > 0.9, "hand_prob no longer marks the coming opening play ({cp}) — leak fixed? update this pin");
    assert!(ch > 0.4, "hand_ids history no longer predicts the opening plays ({ch}) — leak fixed? update this pin");
    assert!(cf > 0.9, "frame-0 hand is no longer the set of the first 4 plays ({cf}) — leak fixed? update this pin");
}

/// Champion / hero ability events: RoyaleAPI's `ability_source_candidates` is the acting
/// side's ability-capable DECK cards, sorted by key, and constant over the match (checked
/// on 18,913 real ability events: always in that side's deck, always sorted, one list per
/// side). `parse_payload` takes the first one as the ability's card, so with two such
/// cards in the deck the opponent's ability event can name one they have not played yet
/// (171 of 18,913 real events). That id lands in `raw.recent_card` of the other side's
/// frames: perturbing only an UNPLAYED opponent deck card changes them. Latent today:
/// `expand.py` never reads `recent_card`, and ability events have no cell so they are
/// skipped in the arena planes (pinned by the Python test
/// `test_expand_ignores_recent_card_ids`), but the column is stored as a per-frame input.
/// Pinned: fails once the source is resolved to an on-field card (or unknown).
#[test]
fn known_latent_leak_ability_card_names_an_unplayed_opponent_card() {
    let (cat, tl) = load();
    let cfg = FrameConfig { step_ticks: 10, recent: 4 };
    let team: Vec<String> = ["hog-rider", "ice-spirit", "ice-golem", "skeletons", "cannon", "musketeer", "fireball", "the-log"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let opp_deck = |unplayed_hero: &str| -> Vec<String> {
        ["giant", unplayed_hero, "musketeer-hero", "archers", "zap", "minions", "fireball", "tornado"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    let side = |d: &[String]| SideSpec {
        deck: d.iter().map(|k| (k.clone(), 16)).collect(),
        tower: "tower-princess".into(),
        crowns: 0,
        final_hp: [0.0; 3],
    };
    // the opponent plays its musketeer hero at 5 s and activates its ability at 7 s
    let spec_with = |unplayed_hero: &str| Spec {
        team: side(&team),
        opp: side(&opp_deck(unplayed_hero)),
        events: vec![
            Ev { side: "opponent", key: "musketeer".into(), deck_key: "musketeer-hero".into(), tick: 100, x: 9, y: 20, ability: false },
            Ev { side: "opponent", key: "musketeer".into(), deck_key: "musketeer-hero".into(), tick: 140, x: 0, y: 0, ability: true },
        ],
        duration: 10.0,
        result: "draw",
    };
    // RoyaleAPI's candidate list: the side's ability-capable deck cards, sorted
    let build_with = |unplayed_hero: &str| {
        let mut v: serde_json::Value = serde_json::from_str(&payload(&spec_with(unplayed_hero))).unwrap();
        let mut cands = vec![unplayed_hero.to_string(), "musketeer-hero".to_string()];
        cands.sort();
        for e in v["events"].as_array_mut().unwrap() {
            if e["kind"] == "activate_ability" {
                e["ability_source_candidates"] = json!(cands);
            }
        }
        build(&v.to_string(), SideId::Team, &cat, &tl, &cfg)
    };
    let a = build_with("ice-golem-hero"); // candidates [ice-golem-hero, musketeer-hero]
    let b = build_with("wizard-hero"); // candidates [musketeer-hero, wizard-hero]
    let f = 16; // tick 160: the newest recent entry is the ability event
    assert_eq!(a.cols.recent_side[f * cfg.recent], 1);
    assert_eq!(a.cols.recent_x[f * cfg.recent], -1, "ability events carry no cell");
    let never_played = cat.resolve("ice-golem-hero").0;
    assert_eq!(a.cols.recent_card[f * cfg.recent], never_played, "the ability is attributed to the unplayed ice golem");
    let d = column_diffs(&a.cols, &b.cols, a.cols.len(), true);
    assert_eq!(
        d.keys().copied().collect::<Vec<_>>(),
        vec!["recent_card"],
        "only recent_card may move with the opponent's unplayed deck card (known latent leak): {d:?}"
    );
}
