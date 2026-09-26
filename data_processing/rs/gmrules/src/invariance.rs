//! Invariance and conformance checks every gmrules game ships (call
//! [`check_game`] from the game's unit tests; the Python side has the same checks
//! for converters in `tests/invariance.py`).
//!
//! - causality: truncating a trajectory right after decision t leaves the frames and
//!   targets of decisions 0..=t bit-identical (no encoder may look ahead);
//! - privacy (hidden-information games): perturbing information hidden from seat s
//!   ([`Game::perturb_hidden`]) leaves every frame and target of s bit-identical;
//! - conformance: the spec validates; finite f32 values, 0/1 in bit-stored and bool
//!   tensors; candidate schema: legal ids within `max_actions`, the legality mask =
//!   the first n slots, entity type ids below `entity_types`, absent entities zero;
//!   every schema: the action keys of every legal action decode back to it (the
//!   simulator's step inverts the IL targets); outcomes in [-1, 1] (equal for
//!   teammates in team games, [`Game::teams`]); the simulator's own trajectories
//!   replay to exactly the frames it emitted.

use crate::frame::{action_keys_of, check_targets_equal, decode_action, encode_target, new_target, Frame};
use crate::game::{FrameSpec, Game, Seat, Turn};
use crate::rng::Rng;
use crate::schema::{Buf, TensorSpec, TensorsMut};
use crate::sim::{SimOptions, VecSim};
use crate::trajectory::{replay, Decision, Event};

/// A uniformly random playout: chance by its probabilities, actions uniform among the
/// legal ones, until the end or `max_events` events.
pub fn random_trajectory<G: Game>(cfg: &G::Config, rng: &mut Rng, max_events: usize) -> Vec<Event> {
    let mut g = G::new_game(cfg);
    let mut events = Vec::new();
    let mut buf = Vec::new();
    let mut out = Vec::new();
    while events.len() < max_events {
        match g.turn() {
            Turn::Terminal => break,
            Turn::Chance => {
                out.clear();
                g.chance_outcomes(&mut out);
                let w: Vec<f64> = out.iter().map(|&(_, p)| p).collect();
                let o = out[rng.weighted(&w)].0;
                g.apply_chance(o).expect("chance outcome from chance_outcomes");
                events.push(Event::Chance(o));
            }
            Turn::Act(seats) => {
                // simultaneous seats choose on the same state
                let mut acts = Vec::new();
                for s in seats.iter() {
                    buf.clear();
                    g.legal_actions(s, &mut buf);
                    if buf.is_empty() {
                        return events; // reported by replay / check_game
                    }
                    acts.push((s, buf[rng.below(buf.len() as u64) as usize]));
                }
                for (s, a) in acts {
                    g.apply_action(s, a).expect("action from legal_actions");
                    events.push(Event::Act(s, a));
                }
            }
        }
    }
    events
}

fn same_decision(spec: &FrameSpec, a: &Decision, b: &Decision) -> Result<(), String> {
    if (a.seat, a.action, a.action_idx) != (b.seat, b.action, b.action_idx) {
        return Err(format!("seat / action ({}, {}) vs ({}, {})", a.seat, a.action, b.seat, b.action));
    }
    a.frame.check_equal(&b.frame, spec)?;
    check_targets_equal(&a.target, &b.target, spec)
}

/// Causality: for every decision t, replaying the trajectory truncated right after
/// t's action reproduces decisions 0..=t exactly (seat, action, frame, targets).
pub fn check_causality<G: Game>(cfg: &G::Config, events: &[Event]) -> Result<(), String> {
    let spec = G::spec(cfg);
    let full = replay::<G>(cfg, events).map_err(|e| format!("replay: {e}"))?;
    for (t, d) in full.decisions.iter().enumerate() {
        let part = replay::<G>(cfg, &events[..=d.event]).map_err(|e| format!("truncated replay (t={t}): {e}"))?;
        if part.decisions.len() != t + 1 {
            return Err(format!("truncated after decision {t}: {} decisions, expected {}", part.decisions.len(), t + 1));
        }
        for (k, (a, b)) in full.decisions.iter().zip(&part.decisions).enumerate() {
            same_decision(&spec, a, b).map_err(|e| {
                format!("causality: decision {k} changes when the record is truncated after decision {t}: {e}")
            })?;
        }
    }
    Ok(())
}

/// Privacy: every frame and target of `seat` is bit-identical in `events` and
/// `perturbed` (which must differ, and must give `seat` the same number of decisions).
pub fn check_privacy<G: Game>(cfg: &G::Config, events: &[Event], perturbed: &[Event], seat: Seat) -> Result<(), String> {
    if events == perturbed {
        return Err("privacy: the perturbed trajectory equals the original".into());
    }
    let spec = G::spec(cfg);
    let a = replay::<G>(cfg, events).map_err(|e| format!("replay: {e}"))?;
    let b = replay::<G>(cfg, perturbed).map_err(|e| format!("perturbed replay: {e}"))?;
    let fa: Vec<_> = a.decisions.iter().filter(|d| d.seat == seat).collect();
    let fb: Vec<_> = b.decisions.iter().filter(|d| d.seat == seat).collect();
    if fa.len() != fb.len() {
        return Err(format!("privacy: seat {seat} has {} decisions, {} after the perturbation", fa.len(), fb.len()));
    }
    for (k, (x, y)) in fa.iter().zip(&fb).enumerate() {
        same_decision(&spec, x, y)
            .map_err(|e| format!("privacy: seat {seat}'s decision {k} depends on information hidden from it: {e}"))?;
    }
    Ok(())
}

/// [`check_privacy`] for `tries` perturbations ([`Game::perturb_hidden`]) of every
/// seat; returns the number of perturbations checked.
pub fn check_privacy_random<G: Game>(cfg: &G::Config, events: &[Event], rng: &mut Rng, tries: usize) -> Result<usize, String> {
    let n_seats = G::spec(cfg).n_seats;
    let mut n = 0;
    for seat in 0..n_seats {
        for _ in 0..tries {
            if let Some(p) = G::perturb_hidden(cfg, events, seat, rng) {
                check_privacy::<G>(cfg, events, &p, seat)?;
                n += 1;
            }
        }
    }
    Ok(n)
}

/// Value checks of a group of tensors: finite f32, 0/1 in bit-stored f32 and in bool.
fn check_values(group: &str, specs: &[TensorSpec], bufs: &[Buf], what: &str) -> Result<(), String> {
    for (t, b) in specs.iter().zip(bufs) {
        if b.dtype() != t.dtype || b.len() != t.numel() {
            return Err(format!("{what}: {group}.{} holds {}[{}], the spec says {}{:?}", t.name, b.dtype().name(), b.len(),
                               t.dtype.name(), t.shape));
        }
        match b {
            Buf::F32(v) => {
                if let Some(i) = v.iter().position(|x| !x.is_finite()) {
                    return Err(format!("{what}: {group}.{}[{i}] = {} is not finite", t.name, v[i]));
                }
                if t.bits {
                    if let Some(i) = v.iter().position(|&x| x != 0.0 && x != 1.0) {
                        return Err(format!("{what}: {group}.{}[{i}] = {} but the tensor is stored as bits", t.name, v[i]));
                    }
                }
            }
            Buf::Bool(v) => {
                if let Some(i) = v.iter().position(|&x| x > 1) {
                    return Err(format!("{what}: {group}.{}[{i}] = {} is not a bool", t.name, v[i]));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Frame conformance of one decision (see the module docs).
fn check_decision<G: Game>(spec: &FrameSpec, g: &G, d: &Decision, what: &str) -> Result<(), String> {
    let f = &d.frame;
    check_values("obs", &spec.obs, &f.obs, what)?;
    check_values("legal", &spec.legal, &f.legal, what)?;
    check_values("target", &spec.target, &d.target, what)?;
    if let Some(c) = &spec.candidate {
        let get = |name: &str| f.obs_named(spec, name).expect("candidate tensor");
        let mask = get("entity_mask").as_bool().unwrap();
        let types = f.obs_named(spec, "entity_type").and_then(Buf::as_i32);
        for i in 0..c.max_entities {
            let t = types.map_or(0, |t| t[i]);
            if mask[i] == 0 && t != 0 {
                return Err(format!("{what}: absent entity {i} has type {t}"));
            }
            if t < 0 || t as usize >= c.entity_types.max(1) {
                return Err(format!("{what}: entity {i} type {t} outside 0..{}", c.entity_types.max(1)));
            }
        }
        let fe = c.entity_features;
        let ents = get("entities").as_f32().unwrap();
        for i in 0..c.max_entities {
            if mask[i] == 0 && ents[i * fe..(i + 1) * fe].iter().any(|&x| x != 0.0) {
                return Err(format!("{what}: absent entity {i} has non-zero features"));
            }
        }
        let n = f.legal_ids.len();
        let amask = get("action_mask").as_bool().unwrap();
        if amask.iter().enumerate().any(|(i, &m)| (m == 1) != (i < n)) {
            return Err(format!("{what}: action_mask is not the first {n} slots"));
        }
        if f.legal[0].as_bool().unwrap() != amask {
            return Err(format!("{what}: legal.action differs from obs.action_mask"));
        }
        let fa = c.action_features;
        let acts = get("actions").as_f32().unwrap();
        if acts[n * fa..].iter().any(|&x| x != 0.0) {
            return Err(format!("{what}: padded candidate slots have non-zero features"));
        }
    }
    // action keys ↔ actions: the keys of every legal action decode to a legal action
    // with the same keys (the simulator plays what the IL target names); one-to-one
    // for the candidate schema, many-to-one allowed for an abstracted custom action
    // space (e.g. exact bet amounts labelled with the nearest discretised size)
    let mut t = new_target(spec);
    let mut seen = std::collections::HashSet::new();
    for &a in &f.legal_ids {
        encode_target(g, d.seat, spec, &f.legal_ids, a, &mut TensorsMut::of(&spec.target, &mut t)).map_err(|e| format!("{what}: {e}"))?;
        let keys = action_keys_of(spec, &t);
        let back = decode_action(g, d.seat, spec, &f.legal_ids, &keys).map_err(|e| format!("{what}: action {a}: {e}"))?;
        if back != a {
            encode_target(g, d.seat, spec, &f.legal_ids, back, &mut TensorsMut::of(&spec.target, &mut t))
                .map_err(|e| format!("{what}: {e}"))?;
            let again = action_keys_of(spec, &t);
            if again != keys || spec.is_candidate() {
                return Err(format!("{what}: action {a} has action keys {keys:?}, which decode to action {back} \
                                    (action keys {again:?})"));
            }
        }
        if spec.is_candidate() && !seen.insert(keys.clone()) {
            return Err(format!("{what}: two legal actions share the action keys {keys:?}"));
        }
    }
    Ok(())
}

/// Summary of [`check_game`].
#[derive(Clone, Debug, Default)]
pub struct CheckReport {
    pub games: usize,
    pub terminal: usize,
    pub decisions: usize,
    pub privacy_checks: usize,
    pub max_legal: usize,
    pub max_turns: usize,
}

/// The conformance battery of a game: `n_games` random playouts (seeded), each
/// replayed and checked for frame conformance, causality and (hidden-information
/// games) privacy; plus a batched [`VecSim`] run whose logged trajectories must
/// replay to exactly the frames the simulator emitted.
pub fn check_game<G: Game>(cfg: &G::Config, n_games: usize, seed: u64) -> Result<CheckReport, String> {
    G::check_config(cfg).map_err(|e| format!("config: {e}"))?;
    let spec = G::spec(cfg);
    spec.validate()?;
    let teams = G::teams(cfg);
    if let Some(t) = &teams {
        crate::game::validate_teams(t, spec.n_seats)?;
    }
    let mut rng = Rng::new(seed);
    let mut rep = CheckReport::default();
    let max_events = 64 * spec.max_decisions.max(1) + 4096;
    for gi in 0..n_games {
        let events = random_trajectory::<G>(cfg, &mut rng, max_events);
        let r = replay::<G>(cfg, &events).map_err(|e| format!("game {gi}: {e}"))?;
        rep.games += 1;
        rep.decisions += r.decisions.len();
        rep.max_turns = rep.max_turns.max(r.turns);
        if r.terminal() {
            rep.terminal += 1;
            let out = r.state.outcome();
            let ret = r.state.returns();
            if out.len() != spec.n_seats || ret.len() != spec.n_seats {
                return Err(format!("game {gi}: outcome / returns must have n_seats = {} entries", spec.n_seats));
            }
            if out.iter().any(|&v| !(-1.0..=1.0).contains(&v)) {
                return Err(format!("game {gi}: outcome {out:?} outside [-1, 1]"));
            }
            if let Some(t) = &teams {
                if (0..out.len()).any(|s| (0..out.len()).any(|q| t[s] == t[q] && out[s] != out[q])) {
                    return Err(format!("game {gi}: teammates (teams {t:?}) get different outcomes {out:?}"));
                }
            }
        } else {
            return Err(format!("game {gi}: random playout did not end within {max_events} events"));
        }
        // re-walk the record to check each decision on the state it was taken in
        let mut g = G::new_game(cfg);
        let mut next = 0usize;
        for (i, ev) in events.iter().enumerate() {
            while next < r.decisions.len() && r.decisions[next].event == i {
                let d = &r.decisions[next];
                rep.max_legal = rep.max_legal.max(d.frame.legal_ids.len());
                // simultaneous turns: every decision of the turn is on the pre-turn state
                check_decision(&spec, &g, d, &format!("game {gi} decision {next}"))?;
                next += 1;
                // later seats of a simultaneous turn: skip ahead without applying
                while next < r.decisions.len() && r.decisions[next].turn == d.turn {
                    let d2 = &r.decisions[next];
                    rep.max_legal = rep.max_legal.max(d2.frame.legal_ids.len());
                    check_decision(&spec, &g, d2, &format!("game {gi} decision {next}"))?;
                    next += 1;
                }
            }
            match *ev {
                Event::Chance(o) => g.apply_chance(o).map_err(|e| e.0)?,
                Event::Act(s, a) => g.apply_action(s, a).map_err(|e| e.0)?,
            }
        }
        check_causality::<G>(cfg, &events).map_err(|e| format!("game {gi}: {e}"))?;
        if spec.hidden_info {
            rep.privacy_checks += check_privacy_random::<G>(cfg, &events, &mut rng, 2).map_err(|e| format!("game {gi}: {e}"))?;
        }
    }
    if spec.hidden_info && n_games >= 20 && rep.privacy_checks == 0 {
        return Err(format!("{}: hidden-information game, but perturb_hidden never produced a perturbation", G::NAME));
    }
    check_sim_replays::<G>(cfg, 8.min(n_games.max(1)), seed ^ 0x5EED, 1)?;
    if n_games >= 2 {
        check_sim_replays::<G>(cfg, 8.min(n_games), seed ^ 0x7EED, 3)?;
    }
    Ok(rep)
}

/// Run a [`VecSim`] (`threads` stepping threads) with random legal actions and check
/// that every env's logged trajectory replays to exactly the frames the simulator
/// emitted, in stream order, and that the applied action keys are the replay's targets.
pub fn check_sim_replays<G: Game>(cfg: &G::Config, n_envs: usize, seed: u64, threads: usize) -> Result<(), String> {
    let spec = G::spec(cfg);
    let mut sim = VecSim::<G>::new(cfg.clone(), n_envs, SimOptions { threads, ..SimOptions::default() })?;
    let mut rng = Rng::new(seed);
    let seeds: Vec<u64> = (0..n_envs).map(|_| rng.next_u64()).collect();
    sim.reset(&(0..n_envs).collect::<Vec<_>>(), &seeds)?;
    let rpe = sim.rows_per_env();
    let k = spec.action_keys.len();
    // per row: (frame, applied action keys)
    let mut seen: Vec<Vec<(Frame, Vec<i64>)>> = vec![Vec::new(); sim.n_rows()];
    let mut steps = 0;
    while (0..n_envs).any(|e| !sim.done(e)) {
        let acts = sim.random_actions(rng.next_u64())?;
        let mut pending = Vec::new();
        {
            let b = sim.buffers();
            for (r, row_seen) in seen.iter().enumerate() {
                if b.active[r] == 1 {
                    if b.step_idx[r] as usize != row_seen.len() {
                        return Err(format!("sim row {r}: step_idx {} after {} frames", b.step_idx[r], row_seen.len()));
                    }
                    let mut f = Frame::new(&spec);
                    for (i, t) in spec.obs.iter().enumerate() {
                        let n = t.numel();
                        f.obs[i].view_mut().copy_from(b.obs[i].range(r * n..(r + 1) * n));
                    }
                    for (i, t) in spec.legal.iter().enumerate() {
                        let n = t.numel();
                        f.legal[i].view_mut().copy_from(b.legal[i].range(r * n..(r + 1) * n));
                    }
                    f.legal_ids = sim.legal(r).to_vec();
                    pending.push((r, f));
                }
            }
        }
        let info = sim.step(&acts)?;
        for (r, f) in pending {
            if info.acted[r] != 1 {
                return Err(format!("sim row {r} was active but did not act"));
            }
            seen[r].push((f, info.applied[r * k..(r + 1) * k].to_vec()));
        }
        steps += 1;
        if steps > 64 * spec.max_decisions + 64 {
            return Err("sim did not finish".into());
        }
    }
    for e in 0..n_envs {
        let r = replay::<G>(cfg, sim.trajectory(e)).map_err(|err| format!("sim env {e} trajectory: {err}"))?;
        for st in r.streams(&spec) {
            let row = e * rpe + st.seat.unwrap_or(0);
            if st.decisions.len() != seen[row].len() {
                return Err(format!("sim env {e} row {row}: {} frames, the replay has {}", seen[row].len(), st.decisions.len()));
            }
            for (j, &d) in st.decisions.iter().enumerate() {
                let dec = &r.decisions[d];
                dec.frame
                    .check_equal(&seen[row][j].0, &spec)
                    .map_err(|err| format!("sim env {e} row {row} frame {j} differs from its replay: {err}"))?;
                if action_keys_of(&spec, &dec.target) != seen[row][j].1 {
                    return Err(format!("sim env {e} row {row} frame {j}: applied action keys {:?}, the replay's targets say {:?}",
                                       seen[row][j].1, action_keys_of(&spec, &dec.target)));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameMut;
    use crate::game::{RulesError, SeatSet};
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    struct BidCfg {
        broken: bool,
    }

    /// Two seats bid exact amounts 1..=9 once each (the rules are exact); the policy
    /// only names a size bucket (1-3, 4-6, 7-9) — an abstracted action space.
    #[derive(Clone, Debug)]
    struct Bid {
        bids: Vec<u32>,
        broken: bool,
    }

    fn bucket(a: u32) -> i64 {
        (a as i64 - 1) / 3
    }

    impl Game for Bid {
        type Config = BidCfg;
        const NAME: &'static str = "bid_test";
        const VERSION: u32 = 0;
        fn spec(_: &BidCfg) -> FrameSpec {
            FrameSpec {
                n_seats: 2,
                hidden_info: false,
                simultaneous: false,
                max_decisions: 2,
                schema: "bucket.v1".into(),
                obs: vec![TensorSpec::f32("bids", &[2])],
                legal: vec![TensorSpec::bool("bucket", &[3])],
                target: vec![TensorSpec::i64("bucket", &[]), TensorSpec::i64("amount", &[])],
                action_keys: vec!["bucket".into()],
                meta: serde_json::json!({}),
                candidate: None,
            }
        }
        fn new_game(c: &BidCfg) -> Self {
            Bid { bids: Vec::new(), broken: c.broken }
        }
        fn turn(&self) -> Turn {
            if self.bids.len() == 2 { Turn::Terminal } else { Turn::Act(SeatSet::single(self.bids.len())) }
        }
        fn chance_outcomes(&self, _: &mut Vec<(u32, f64)>) {}
        fn apply_chance(&mut self, _: u32) -> Result<(), RulesError> {
            Err(RulesError("no chance".into()))
        }
        fn legal_actions(&self, s: Seat, out: &mut Vec<u32>) {
            if self.turn() == Turn::Act(SeatSet::single(s)) {
                out.extend(1..=9);
            }
        }
        fn apply_action(&mut self, _: Seat, a: u32) -> Result<(), RulesError> {
            self.bids.push(a);
            Ok(())
        }
        fn outcome(&self) -> Vec<f32> {
            let d = (self.bids[0] as f32 - self.bids[1] as f32).signum();
            vec![d, -d]
        }
        fn returns(&self) -> Vec<f64> {
            self.outcome().iter().map(|&x| x as f64).collect()
        }
        fn encode_custom(&self, _: Seat, f: &mut FrameMut<'_>) {
            for (i, &b) in self.bids.iter().enumerate() {
                f.obs.f32(0)[i] = b as f32 / 9.0;
            }
            for &a in f.legal_ids.iter() {
                f.legal.bool(0)[bucket(a) as usize] = 1;
            }
        }
        fn target_custom(&self, _: Seat, _: &[u32], a: u32, t: &mut TensorsMut<'_>) {
            t.i64(0)[0] = bucket(a);
            t.i64(1)[0] = a as i64;
        }
        fn decode_custom(&self, _: Seat, legal: &[u32], keys: &[i64]) -> Option<u32> {
            // the bucket's middle amount; a broken decoder lands in the next bucket
            let a = 3 * keys[0] as u32 + 2 + if self.broken { 3 } else { 0 };
            legal.contains(&a).then_some(a)
        }
        fn state_string(&self) -> String {
            format!("{:?}", self.bids)
        }
    }

    #[test]
    fn abstracted_action_keys_pass_inconsistent_decoders_fail() {
        let rep = check_game::<Bid>(&BidCfg { broken: false }, 20, 1).unwrap();
        assert_eq!(rep.terminal, 20);
        let err = check_game::<Bid>(&BidCfg { broken: true }, 20, 1).unwrap_err();
        assert!(err.contains("decode to action") || err.contains("no legal action"), "{err}");
    }
}
