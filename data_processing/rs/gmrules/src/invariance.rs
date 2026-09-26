//! Invariance and conformance checks every gmrules game ships (call
//! [`check_game`] from the game's unit tests; the Python side has the same checks
//! for converters in `tests/invariance.py`).
//!
//! - causality: truncating a trajectory right after decision t leaves the frames of
//!   decisions 0..=t bit-identical (no encoder may look ahead);
//! - privacy (hidden-information games): perturbing information hidden from seat s
//!   ([`Game::perturb_hidden`]) leaves every frame of s bit-identical;
//! - conformance: legal ids ascending and within `max_actions`, entity type ids below
//!   `entity_types`, finite features, 0/1 values in bit-stored tensors, outcomes in
//!   [-1, 1], replay of the simulator's own trajectories.

use crate::frame::Frame;
use crate::game::{Game, Seat, Turn};
use crate::rng::Rng;
use crate::sim::{SimOptions, VecSim};
use crate::trajectory::{replay, Event};

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
                for s in seats.iter() {
                    buf.clear();
                    g.legal_actions(s, &mut buf);
                    if buf.is_empty() {
                        return events; // reported by replay / check_game
                    }
                    let a = buf[rng.below(buf.len() as u64) as usize];
                    g.apply_action(s, a).expect("action from legal_actions");
                    events.push(Event::Act(s, a));
                }
            }
        }
    }
    events
}

/// Causality: for every decision t, replaying the trajectory truncated right after
/// t's action reproduces decisions 0..=t exactly (seat, action, candidate index, frame).
pub fn check_causality<G: Game>(cfg: &G::Config, events: &[Event]) -> Result<(), String> {
    let full = replay::<G>(cfg, events).map_err(|e| format!("replay: {e}"))?;
    for (t, d) in full.decisions.iter().enumerate() {
        let part = replay::<G>(cfg, &events[..=d.event]).map_err(|e| format!("truncated replay (t={t}): {e}"))?;
        if part.decisions.len() != t + 1 {
            return Err(format!("truncated after decision {t}: {} decisions, expected {}", part.decisions.len(), t + 1));
        }
        for (k, (a, b)) in full.decisions.iter().zip(&part.decisions).enumerate() {
            if (a.seat, a.action, a.action_idx) != (b.seat, b.action, b.action_idx) {
                return Err(format!("truncated after decision {t}: decision {k} changed"));
            }
            a.frame
                .check_equal(&b.frame)
                .map_err(|e| format!("causality: frame of decision {k} changes when the record is truncated after decision {t}: {e}"))?;
        }
    }
    Ok(())
}

/// Privacy: every frame of `seat` is bit-identical in `events` and `perturbed`
/// (which must differ, and must give `seat` the same number of decisions).
pub fn check_privacy<G: Game>(cfg: &G::Config, events: &[Event], perturbed: &[Event], seat: Seat) -> Result<(), String> {
    if events == perturbed {
        return Err("privacy: the perturbed trajectory equals the original".into());
    }
    let a = replay::<G>(cfg, events).map_err(|e| format!("replay: {e}"))?;
    let b = replay::<G>(cfg, perturbed).map_err(|e| format!("perturbed replay: {e}"))?;
    let fa: Vec<_> = a.decisions.iter().filter(|d| d.seat == seat).collect();
    let fb: Vec<_> = b.decisions.iter().filter(|d| d.seat == seat).collect();
    if fa.len() != fb.len() {
        return Err(format!("privacy: seat {seat} has {} decisions, {} after the perturbation", fa.len(), fb.len()));
    }
    for (k, (x, y)) in fa.iter().zip(&fb).enumerate() {
        x.frame
            .check_equal(&y.frame)
            .map_err(|e| format!("privacy: seat {seat}'s frame {k} depends on information hidden from it: {e}"))?;
        if (x.action, x.action_idx) != (y.action, y.action_idx) {
            return Err(format!("privacy: seat {seat}'s action {k} changed under the perturbation"));
        }
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

fn check_frame<G: Game>(cfg: &G::Config, f: &Frame, what: &str) -> Result<(), String> {
    let s = G::spec(cfg);
    let finite = |name: &str, v: &[f32]| match v.iter().position(|x| !x.is_finite()) {
        Some(i) => Err(format!("{what}: {name}[{i}] = {} is not finite", v[i])),
        None => Ok(()),
    };
    let binary = |name: &str, v: &[f32]| match v.iter().position(|&x| x != 0.0 && x != 1.0) {
        Some(i) => Err(format!("{what}: {name}[{i}] = {} but the spec stores {name} as bits", v[i])),
        None => Ok(()),
    };
    finite("entities", &f.entities)?;
    finite("globals", &f.globals)?;
    finite("actions", &f.actions)?;
    if s.bits_entities {
        binary("entities", &f.entities)?;
    }
    if s.bits_globals {
        binary("globals", &f.globals)?;
    }
    if s.bits_actions {
        binary("actions", &f.actions)?;
    }
    for (i, (&m, &t)) in f.entity_mask.iter().zip(&f.entity_type).enumerate() {
        if m == 0 && t != 0 {
            return Err(format!("{what}: absent entity {i} has type {t}"));
        }
        if t < 0 || (t as usize >= s.entity_types.max(1)) {
            return Err(format!("{what}: entity {i} type {t} outside 0..{}", s.entity_types.max(1)));
        }
    }
    let fe = s.entity_features;
    for i in 0..s.max_entities {
        if f.entity_mask[i] == 0 && f.entities[i * fe..(i + 1) * fe].iter().any(|&x| x != 0.0) {
            return Err(format!("{what}: absent entity {i} has non-zero features"));
        }
    }
    let n = f.legal.len();
    if f.action_mask.iter().enumerate().any(|(i, &m)| (m == 1) != (i < n)) {
        return Err(format!("{what}: action_mask is not the first {n} slots"));
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
    let spec = G::spec(cfg);
    // every tensor has at least one element per frame (gm-shard frame shapes)
    if spec.n_seats == 0 || spec.n_seats > 32 || spec.max_actions == 0 || spec.max_entities == 0
        || spec.entity_features == 0 || spec.global_features == 0 || spec.action_features == 0
        || spec.max_decisions == 0
    {
        return Err(format!("bad spec (every size must be >= 1, n_seats <= 32): {spec:?}"));
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
        } else {
            return Err(format!("game {gi}: random playout did not end within {max_events} events"));
        }
        for (k, d) in r.decisions.iter().enumerate() {
            rep.max_legal = rep.max_legal.max(d.frame.legal.len());
            check_frame::<G>(cfg, &d.frame, &format!("game {gi} decision {k}"))?;
        }
        check_causality::<G>(cfg, &events).map_err(|e| format!("game {gi}: {e}"))?;
        if spec.hidden_info {
            rep.privacy_checks += check_privacy_random::<G>(cfg, &events, &mut rng, 2).map_err(|e| format!("game {gi}: {e}"))?;
        }
    }
    if spec.hidden_info && n_games >= 20 && rep.privacy_checks == 0 {
        return Err(format!("{}: hidden-information game, but perturb_hidden never produced a perturbation", G::NAME));
    }
    check_sim_replays::<G>(cfg, 8.min(n_games.max(1)), seed ^ 0x5EED)?;
    Ok(rep)
}

/// Run a [`VecSim`] with random legal actions and check that every env's logged
/// trajectory replays to exactly the frames the simulator emitted, in stream order.
pub fn check_sim_replays<G: Game>(cfg: &G::Config, n_envs: usize, seed: u64) -> Result<(), String> {
    let spec = G::spec(cfg);
    let mut sim = VecSim::<G>::new(cfg.clone(), n_envs, SimOptions::default())?;
    let mut rng = Rng::new(seed);
    let seeds: Vec<u64> = (0..n_envs).map(|_| rng.next_u64()).collect();
    sim.reset(&(0..n_envs).collect::<Vec<_>>(), &seeds)?;
    let rpe = sim.rows_per_env();
    // frames seen per row: (step_idx, frame)
    let mut seen: Vec<Vec<Frame>> = vec![Vec::new(); sim.n_rows()];
    let mut steps = 0;
    while (0..n_envs).any(|e| !sim.done(e)) {
        let b = sim.buffers();
        let mut acts = vec![0i64; sim.n_rows()];
        for r in 0..sim.n_rows() {
            if b.active[r] == 1 {
                if b.step_idx[r] as usize != seen[r].len() {
                    return Err(format!("sim row {r}: step_idx {} after {} frames", b.step_idx[r], seen[r].len()));
                }
                let mut f = Frame::new(&spec);
                let (el, al) = (spec.entities_len(), spec.actions_len());
                f.entities.copy_from_slice(&b.entities[r * el..(r + 1) * el]);
                f.entity_mask.copy_from_slice(&b.entity_mask[r * spec.max_entities..(r + 1) * spec.max_entities]);
                f.entity_type.copy_from_slice(&b.entity_type[r * spec.max_entities..(r + 1) * spec.max_entities]);
                f.globals.copy_from_slice(&b.globals[r * spec.global_features..(r + 1) * spec.global_features]);
                f.actions.copy_from_slice(&b.actions[r * al..(r + 1) * al]);
                f.action_mask.copy_from_slice(&b.action_mask[r * spec.max_actions..(r + 1) * spec.max_actions]);
                f.legal = sim.legal(r).to_vec();
                seen[r].push(f);
                acts[r] = rng.below(b.n_legal[r] as u64) as i64;
            }
        }
        sim.step(&acts)?;
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
            for (k, &d) in st.decisions.iter().enumerate() {
                r.decisions[d]
                    .frame
                    .check_equal(&seen[row][k])
                    .map_err(|err| format!("sim env {e} row {row} frame {k} differs from its replay: {err}"))?;
            }
        }
    }
    Ok(())
}
