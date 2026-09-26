//! Trajectories — the universal record format of a gmrules game — and [`replay`],
//! which re-plays one with full verification and yields its decision frames.
//!
//! A trajectory is the sequence of [`Event`]s from the initial state: chance
//! outcomes and seat actions, all as canonical ids (like an OpenSpiel history).
//! Converters parse their source format into a trajectory; the simulator logs one
//! per episode; the invariance tests truncate / perturb them.

use crate::frame::{encode_frame, Frame};
use crate::game::{FrameSpec, Game, Layout, Seat, Turn};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// A chance outcome id.
    Chance(u32),
    /// `(seat, action id)`.
    Act(Seat, u32),
}

impl Event {
    /// `(seat, id)` with seat −1 for chance: the flat form Python / JSON use.
    pub fn to_pair(self) -> (i64, i64) {
        match self {
            Event::Chance(o) => (-1, o as i64),
            Event::Act(s, a) => (s as i64, a as i64),
        }
    }

    pub fn from_pair(seat: i64, id: i64) -> Result<Event, String> {
        let id = u32::try_from(id).map_err(|_| format!("event id {id} out of range"))?;
        match seat {
            -1 => Ok(Event::Chance(id)),
            s if (0..32).contains(&s) => Ok(Event::Act(s as usize, id)),
            s => Err(format!("event seat {s} out of range (-1 = chance, 0..32)")),
        }
    }
}

/// One decision of one seat: its frame (encoded before the turn's actions) and the
/// action taken.
#[derive(Clone, Debug)]
pub struct Decision {
    pub seat: Seat,
    /// Index of the action's event in the trajectory.
    pub event: usize,
    /// Index of the decision turn (simultaneous seats share one).
    pub turn: usize,
    pub frame: Frame,
    pub action: u32,
    /// Candidate index of `action` in `frame` (the IL target).
    pub action_idx: usize,
}

/// A verified replay.
pub struct Replay<G: Game> {
    /// State after the last event.
    pub state: G,
    pub decisions: Vec<Decision>,
    /// Decision turns played.
    pub turns: usize,
}

impl<G: Game> Replay<G> {
    pub fn terminal(&self) -> bool {
        self.state.turn() == Turn::Terminal
    }

    /// Per-seat outcome when the replay reached the end of the game.
    pub fn outcome(&self) -> Option<Vec<f32>> {
        self.terminal().then(|| self.state.outcome())
    }

    /// The temporal streams of the replay (= the IL samples of the record, = the RL
    /// rows of an episode): per-seat layout — one stream per seat that decided at
    /// least once, in seat order; interleaved — one stream with every decision.
    pub fn streams(&self, spec: &FrameSpec) -> Vec<Stream> {
        match spec.layout() {
            Layout::Interleaved => {
                if self.decisions.is_empty() {
                    Vec::new()
                } else {
                    vec![Stream { seat: None, decisions: (0..self.decisions.len()).collect() }]
                }
            }
            Layout::PerSeat => (0..spec.n_seats)
                .filter_map(|s| {
                    let d: Vec<usize> = (0..self.decisions.len()).filter(|&i| self.decisions[i].seat == s).collect();
                    (!d.is_empty()).then_some(Stream { seat: Some(s), decisions: d })
                })
                .collect(),
        }
    }
}

/// Decisions forming one temporal stream (indices into [`Replay::decisions`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    /// The stream's seat (None = interleaved: every seat, each frame its mover's view).
    pub seat: Option<Seat>,
    pub decisions: Vec<usize>,
}

/// Why a trajectory does not replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayError {
    /// Index of the offending event (= len for errors at the end).
    pub event: usize,
    pub msg: String,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "event {}: {}", self.event, self.msg)
    }
}

impl std::error::Error for ReplayError {}

/// Replay `events` from the initial state, verifying every event (chance outcome with
/// positive probability, the right seat, a legal action, at most `max_decisions`
/// turns), and encode every decision frame. The trajectory may stop anywhere (a
/// truncated record); a decision turn is only kept up to its last recorded action.
pub fn replay<G: Game>(cfg: &G::Config, events: &[Event]) -> Result<Replay<G>, ReplayError> {
    let spec = G::spec(cfg);
    let mut g = G::new_game(cfg);
    let mut decisions = Vec::new();
    let mut outcomes = Vec::new();
    let mut turns = 0usize;
    let mut i = 0usize;
    let err = |event: usize, msg: String| ReplayError { event, msg };
    while i < events.len() {
        match g.turn() {
            Turn::Terminal => return Err(err(i, format!("game over, but {} events remain", events.len() - i))),
            Turn::Chance => {
                let Event::Chance(o) = events[i] else {
                    return Err(err(i, format!("chance node, got {:?}", events[i])));
                };
                outcomes.clear();
                g.chance_outcomes(&mut outcomes);
                if !outcomes.iter().any(|&(id, p)| id == o && p > 0.0) {
                    return Err(err(i, format!("chance outcome {o} impossible here")));
                }
                g.apply_chance(o).map_err(|e| err(i, e.0))?;
                i += 1;
            }
            Turn::Act(seats) => {
                if seats.is_empty() {
                    return Err(err(i, "Act turn with no seat".into()));
                }
                if turns >= spec.max_decisions {
                    return Err(err(i, format!("more than max_decisions = {} decision turns", spec.max_decisions)));
                }
                // every acting seat decides on the same state
                let mut frames = Vec::with_capacity(seats.len());
                for s in seats.iter() {
                    let mut f = Frame::new(&spec);
                    encode_frame(&g, s, &spec, &mut f.as_mut()).map_err(|e| err(i, e.0))?;
                    frames.push((s, f));
                }
                for (s, frame) in frames {
                    if i >= events.len() {
                        break; // truncated inside a simultaneous turn
                    }
                    let Event::Act(es, a) = events[i] else {
                        return Err(err(i, format!("seat {s} to act, got {:?}", events[i])));
                    };
                    if es != s {
                        return Err(err(i, format!("seat {s} to act, got an action of seat {es}")));
                    }
                    let Some(idx) = frame.index_of(a) else {
                        return Err(err(i, format!("action {a} ({}) is illegal for seat {s}", g.action_string(s, a))));
                    };
                    g.apply_action(s, a).map_err(|e| err(i, e.0))?;
                    decisions.push(Decision { seat: s, event: i, turn: turns, frame, action: a, action_idx: idx });
                    i += 1;
                }
                turns += 1;
            }
        }
    }
    Ok(Replay { state: g, decisions, turns })
}
