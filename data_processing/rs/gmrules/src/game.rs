//! The [`Game`] trait every rule engine implements, and its vocabulary: seats, turns,
//! canonical action ids, and the [`FrameSpec`] — the tensor schema of the game's
//! decision frames (see [`crate::schema`]).
//!
//! Two kinds of schema:
//!
//! - the CANDIDATE-ACTION schema ([`FrameSpec::candidate`], `model/games/candidate.py`):
//!   entity slots + globals + one token per legal action; the game writes only
//!   [`Game::encode_obs`] and [`Game::encode_action`], the framework builds the tensors,
//!   the legality mask and the target (the taken action's candidate index). Card
//!   games, bidding, poker with discretised sizes, backgammon full moves, ...
//! - a CUSTOM schema (a [`FrameSpec`] with `candidate: None`): the game declares its own
//!   obs / legal / target tensors and writes them ([`Game::encode_custom`],
//!   [`Game::target_custom`], [`Game::decode_custom`]) — e.g. chess-like games feeding
//!   the board adapter's from-to decoder (`games/breakthrough.rs` is the example).

use crate::frame::{FrameMut, ObsView};
use crate::rng::Rng;
use crate::schema::{index_of, DType, TensorSpec, TensorsMut};
use crate::trajectory::Event;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::fmt;

/// A player seat, `0..n_seats`.
pub type Seat = usize;

/// A set of seats (bit `s` = seat `s`; at most 32 seats).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SeatSet(pub u32);

impl SeatSet {
    pub const EMPTY: SeatSet = SeatSet(0);

    pub fn single(seat: Seat) -> Self {
        assert!(seat < 32, "seat {seat} >= 32");
        SeatSet(1 << seat)
    }

    pub fn from_seats(seats: &[Seat]) -> Self {
        seats.iter().fold(SeatSet::EMPTY, |acc, &s| acc.with(s))
    }

    pub fn with(self, seat: Seat) -> Self {
        SeatSet(self.0 | SeatSet::single(seat).0)
    }

    pub fn without(self, seat: Seat) -> Self {
        SeatSet(self.0 & !SeatSet::single(seat).0)
    }

    pub fn contains(self, seat: Seat) -> bool {
        seat < 32 && self.0 & (1 << seat) != 0
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Seats in ascending order.
    pub fn iter(self) -> impl Iterator<Item = Seat> {
        (0..32).filter(move |&s| self.0 & (1 << s) != 0)
    }
}

/// Who moves next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    /// A chance event (deal, dice, ...): [`Game::chance_outcomes`] lists the outcomes.
    Chance,
    /// A decision of these seats. One seat in alternating games; several in a
    /// simultaneous turn — every one of them decides on the SAME state (their frames
    /// and targets are encoded before any of the turn's actions is applied) and the
    /// actions are applied in ascending seat order.
    Act(SeatSet),
    /// The game is over: [`Game::outcome`] / [`Game::returns`] are final.
    Terminal,
}

/// A rules violation (illegal action, wrong turn, malformed chance outcome).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RulesError(pub String);

impl fmt::Display for RulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RulesError {}

/// `Err(RulesError(format!(...)))`.
#[macro_export]
macro_rules! rules_bail {
    ($($t:tt)*) => { return Err($crate::game::RulesError(format!($($t)*))) };
}

/// How a game's decision frames form temporal streams (IL samples = RL rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// One stream per seat holding only that seat's own decision frames (its own
    /// view). Hidden-information and simultaneous games: a stream must never carry
    /// another seat's private information in its temporal state.
    PerSeat,
    /// One stream per game holding every decision frame in order, each from the
    /// acting seat's perspective (perfect-information alternating games, like the
    /// board games' IL samples).
    Interleaved,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Layout::PerSeat => "per_seat",
            Layout::Interleaved => "interleaved",
        }
    }
}

/// Schema name of the candidate-action schema (shard meta `schema`).
pub const CANDIDATE_SCHEMA: &str = "candidate.v1";

/// Dimensions of a candidate-action game ([`FrameSpec::candidate`] turns them into the
/// schema). Per decision frame:
///
/// ```text
/// obs     entities     [max_entities, entity_features] f32 (bits: bits_entities)
///         entity_mask  [max_entities] bool
///         entity_type  [max_entities] i32   (only when entity_types > 0; ids < entity_types)
///         globals      [global_features] f32 (bits: bits_globals)
///         actions      [max_actions, action_features] f32 (bits: bits_actions)
///         action_mask  [max_actions] bool
/// legal   action       [max_actions] bool   (= action_mask)
/// target  action_idx   [] i64               (action key)
/// ```
///
/// The candidates of a frame are the acting seat's LEGAL actions in ascending
/// canonical-id order, padded to `max_actions`; `action_idx` is the index of the taken
/// action among them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CandidateSpec {
    pub n_seats: usize,
    pub max_entities: usize,
    pub entity_features: usize,
    /// Number of entity type ids (0 = no `entity_type` tensor).
    pub entity_types: usize,
    pub global_features: usize,
    pub max_actions: usize,
    pub action_features: usize,
    /// See [`FrameSpec::hidden_info`].
    pub hidden_info: bool,
    /// See [`FrameSpec::simultaneous`].
    pub simultaneous: bool,
    /// Storage hints: the tensor holds only 0.0 / 1.0 (stored bit-packed).
    pub bits_entities: bool,
    pub bits_globals: bool,
    pub bits_actions: bool,
    /// See [`FrameSpec::max_decisions`].
    pub max_decisions: usize,
}

/// A game's decision-frame schema: fixed tensor shapes (static model shapes, CUDA-graph
/// friendly), plus the facts the converters, simulators and trainers need.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameSpec {
    pub n_seats: usize,
    /// The observation of a seat hides information other seats have (cards, dice
    /// known to one player, ...): per-seat streams, and the privacy invariance test is
    /// mandatory ([`Game::perturb_hidden`] must be implemented).
    pub hidden_info: bool,
    /// Some turns have several acting seats (per-seat streams).
    pub simultaneous: bool,
    /// Hard bound on decision turns per episode (records / simulations longer than
    /// this are rejected / truncated).
    pub max_decisions: usize,
    /// Schema name (shard meta `schema`): [`CANDIDATE_SCHEMA`], or the custom schema's
    /// own versioned name (e.g. `"fromto.v1"`).
    pub schema: String,
    /// Observation tensors (gm-shard `obs.<name>`, the env's obs).
    pub obs: Vec<TensorSpec>,
    /// Legality tensors of the RL decoder (the env's `legal`; not stored in shards).
    pub legal: Vec<TensorSpec>,
    /// IL targets of the taken action (gm-shard `tgt.<name>`; `value`, `value_mask`
    /// and `action_probs_*` are reserved for the converter).
    pub target: Vec<TensorSpec>,
    /// The i64 (or bool, stepped as 0 / 1) scalar targets that make up the policy's
    /// action: the simulator's step input and the decoder's replay keys (candidate
    /// schema: `["action_idx"]`).
    pub action_keys: Vec<String>,
    /// Schema facts the model config must match (written into the shard meta; the
    /// trainers' META_ARCH_KEYS), e.g. the candidate frame dimensions.
    pub meta: Value,
    /// `Some` for candidate-schema games (the framework encodes their frames).
    pub candidate: Option<CandidateSpec>,
}

impl FrameSpec {
    /// The candidate-action schema of `c` (see [`CandidateSpec`]).
    pub fn candidate(c: CandidateSpec) -> FrameSpec {
        let (e, a) = (c.max_entities, c.max_actions);
        let f32_or_bits = |name: &str, shape: &[usize], bits: bool| {
            if bits { TensorSpec::bits(name, shape) } else { TensorSpec::f32(name, shape) }
        };
        let mut obs = vec![f32_or_bits("entities", &[e, c.entity_features], c.bits_entities), TensorSpec::bool("entity_mask", &[e])];
        if c.entity_types > 0 {
            obs.push(TensorSpec::i32("entity_type", &[e]));
        }
        obs.push(f32_or_bits("globals", &[c.global_features], c.bits_globals));
        obs.push(f32_or_bits("actions", &[a, c.action_features], c.bits_actions));
        obs.push(TensorSpec::bool("action_mask", &[a]));
        FrameSpec {
            n_seats: c.n_seats,
            hidden_info: c.hidden_info,
            simultaneous: c.simultaneous,
            max_decisions: c.max_decisions,
            schema: CANDIDATE_SCHEMA.into(),
            obs,
            legal: vec![TensorSpec::bool("action", &[a])],
            target: vec![TensorSpec::i64("action_idx", &[])],
            action_keys: vec!["action_idx".into()],
            meta: json!({
                "max_entities": e,
                "entity_features": c.entity_features,
                "entity_types": c.entity_types,
                "global_features": c.global_features,
                "max_actions": a,
                "action_features": c.action_features,
            }),
            candidate: Some(c),
        }
    }

    pub fn layout(&self) -> Layout {
        if self.hidden_info || self.simultaneous {
            Layout::PerSeat
        } else {
            Layout::Interleaved
        }
    }

    pub fn is_candidate(&self) -> bool {
        self.candidate.is_some()
    }

    /// Indices of the action keys in `target`.
    pub fn action_key_indices(&self) -> Vec<usize> {
        self.action_keys
            .iter()
            .map(|k| index_of(&self.target, k).unwrap_or_else(|| panic!("action key {k:?} is not a target tensor")))
            .collect()
    }

    /// The gm-shard / model schema meta: `meta` + `schema`, `layout`, `n_seats`.
    pub fn schema_meta(&self) -> Value {
        let mut m = match &self.meta {
            Value::Object(o) => Value::Object(o.clone()),
            _ => json!({}),
        };
        m["schema"] = json!(self.schema);
        m["layout"] = json!(self.layout().name());
        m["n_seats"] = json!(self.n_seats);
        m
    }

    /// Structural checks (called by the registry, the simulators and the converters).
    pub fn validate(&self) -> Result<(), String> {
        let bad = |m: String| Err(format!("frame spec ({}): {m}", self.schema));
        if self.n_seats == 0 || self.n_seats > 32 {
            return bad(format!("n_seats {} outside 1..=32", self.n_seats));
        }
        if self.max_decisions == 0 {
            return bad("max_decisions must be >= 1".into());
        }
        if self.schema.is_empty() {
            return bad("empty schema name".into());
        }
        for (group, ts) in [("obs", &self.obs), ("legal", &self.legal), ("target", &self.target)] {
            let mut seen = std::collections::HashSet::new();
            for t in ts.iter() {
                let valid = !t.name.is_empty() && t.name.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
                if !valid {
                    return bad(format!("{group} tensor name {:?} (use [A-Za-z0-9_])", t.name));
                }
                if !seen.insert(&t.name) {
                    return bad(format!("{group} tensor {:?} declared twice", t.name));
                }
                if t.numel() == 0 {
                    return bad(format!("{group} tensor {:?} has no elements ({:?})", t.name, t.shape));
                }
                if t.bits && t.dtype != DType::F32 {
                    return bad(format!("{group} tensor {:?}: bits storage needs f32", t.name));
                }
            }
        }
        if self.obs.is_empty() {
            return bad("no obs tensor".into());
        }
        for t in &self.target {
            if t.name == "value" || t.name == "value_mask" || t.name.starts_with("action_probs") {
                return bad(format!("target name {:?} is reserved for the converter", t.name));
            }
        }
        if self.action_keys.is_empty() {
            return bad("no action key".into());
        }
        for k in &self.action_keys {
            match index_of(&self.target, k).map(|i| &self.target[i]) {
                Some(t) if matches!(t.dtype, DType::I64 | DType::Bool) && t.shape.is_empty() => {}
                _ => return bad(format!("action key {k:?} must be an i64 or bool scalar target tensor")),
            }
        }
        if let Some(c) = &self.candidate {
            if c.max_entities == 0 || c.entity_features == 0 || c.global_features == 0 || c.max_actions == 0 || c.action_features == 0 {
                return bad(format!("every candidate dimension must be >= 1: {c:?}"));
            }
        }
        if !self.meta.is_object() && !self.meta.is_null() {
            return bad("meta must be a JSON object".into());
        }
        Ok(())
    }
}

/// Checks a [`Game::teams`] table: one entry per seat, team ids `0..n_teams` all
/// used, at least two teams.
pub fn validate_teams(teams: &[usize], n_seats: usize) -> Result<(), String> {
    if teams.len() != n_seats {
        return Err(format!("teams {teams:?}: {} entries for {n_seats} seats", teams.len()));
    }
    let n_teams = teams.iter().max().map_or(0, |&t| t + 1);
    if n_teams < 2 || (0..n_teams).any(|t| !teams.contains(&t)) {
        return Err(format!("teams {teams:?}: ids must be 0..n_teams (every id used, at least 2 teams)"));
    }
    Ok(())
}

/// A game's rules and its frame encoders.
///
/// One implementation serves BOTH the converter (gmdata: replay + verification of
/// every record, IL frames) and the simulator (gmsim: the RL env), so IL and RL
/// observations are identical by construction. No I/O here.
///
/// Conventions:
/// - Actions are canonical `u32` ids of a game-wide id space (use the reference
///   implementation's ids, e.g. OpenSpiel's, where there is one: differential tests
///   then compare ids directly). Chance outcomes are `u32` ids too.
/// - [`legal_actions`](Game::legal_actions) lists the ids in STRICTLY ASCENDING order;
///   this order is the candidate order of the frame (never reorder per frame).
/// - Encoders may only read what `seat` knows at this point: public state, its own
///   private state, its own past actions — never other seats' private information
///   and never the future (the causality / privacy tests in [`crate::invariance`]
///   check it).
/// - [`outcome`](Game::outcome) is the per-seat final result in [-1, 1]: the IL value
///   target of every frame of that seat AND the RL reward paid at the end.
///
/// Encoders by schema (see the module docs): candidate games implement
/// [`encode_obs`](Game::encode_obs) + [`encode_action`](Game::encode_action); custom
/// schema games implement [`encode_custom`](Game::encode_custom) +
/// [`target_custom`](Game::target_custom) + [`decode_custom`](Game::decode_custom).
/// The other kind's methods are never called (their defaults panic).
pub trait Game: Clone + Send + Sync + 'static {
    /// Variant / parameters (player count, deck, scoring). JSON (serde) from the
    /// converter CLI, the Python env kwargs and the shard meta; use
    /// `#[serde(default, deny_unknown_fields)]`.
    type Config: Clone + fmt::Debug + Default + Send + Sync + Serialize + DeserializeOwned + 'static;

    /// Registry name: gm-shard `game`, gmsim / Python registry key.
    const NAME: &'static str;
    /// Bump on any semantic change of rules or encoders (the converters write it as
    /// their `converter_version`, so shards of different generations never mix).
    const VERSION: u32;

    /// Reject an invalid config with a message (called before [`spec`](Game::spec) /
    /// [`new_game`](Game::new_game) by the registry, the simulators and the converters,
    /// which may then assume a valid config).
    fn check_config(_cfg: &Self::Config) -> Result<(), RulesError> {
        Ok(())
    }

    /// The frame schema: [`FrameSpec::candidate`] or a custom [`FrameSpec`].
    fn spec(cfg: &Self::Config) -> FrameSpec;

    /// Fixed teams — the team id of every seat (`n_seats` entries, ids `0..n_teams`)
    /// when teammates share one outcome (bridge partnerships: `[0, 1, 0, 1]`) — or
    /// None (every seat plays for itself; the default). Teammates must get equal
    /// outcomes (checked by [`crate::invariance::check_game`]); the simulator reports
    /// the winning TEAM as `winner` (its lowest seat) and the Python league gives a
    /// policy whole teams (env/league.py).
    fn teams(_cfg: &Self::Config) -> Option<Vec<usize>> {
        None
    }

    /// The initial state (before any chance event).
    fn new_game(cfg: &Self::Config) -> Self;

    fn turn(&self) -> Turn;

    /// At a chance node: every outcome with positive probability, `(id, p)`, p
    /// summing to 1 (appended to `out`, which the caller cleared).
    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>);

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError>;

    /// The legal actions of an acting `seat`, strictly ascending ids (appended to
    /// `out`, which the caller cleared). Must not depend on other seats' pending
    /// actions of the same simultaneous turn.
    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>);

    /// Apply `seat`'s action (it must be one of `legal_actions(seat)`).
    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError>;

    /// Per-seat final result in [-1, 1] (IL value target, RL terminal reward).
    fn outcome(&self) -> Vec<f32>;

    /// Per-seat raw payoff (chips, points) at terminal — what reference
    /// implementations report (differential tests, logs).
    fn returns(&self) -> Vec<f64>;

    // ---- candidate schema ------------------------------------------------------

    /// Candidate schema: write `seat`'s observation (entity slots / globals; the
    /// buffers arrive zeroed).
    fn encode_obs(&self, _seat: Seat, _obs: &mut ObsView<'_>) {
        panic!("{}: a candidate-schema game must implement Game::encode_obs", Self::NAME)
    }

    /// Candidate schema: features of the legal candidate `action` for `seat` (`feat`
    /// arrives zeroed, `action_features` long).
    fn encode_action(&self, _seat: Seat, _action: u32, _feat: &mut [f32]) {
        panic!("{}: a candidate-schema game must implement Game::encode_action", Self::NAME)
    }

    // ---- custom schema -----------------------------------------------------------

    /// Custom schema: write `seat`'s frame — `frame.obs` and `frame.legal` tensors
    /// (zeroed on entry) — given `frame.legal_ids` (the legal actions, already filled,
    /// ascending).
    fn encode_custom(&self, _seat: Seat, _frame: &mut FrameMut<'_>) {
        panic!("{}: a custom-schema game must implement Game::encode_custom", Self::NAME)
    }

    /// Custom schema: the target tensors (zeroed on entry) of `seat` taking the legal
    /// `action` in this state (`legal` = its legal actions, ascending) — the action
    /// keys (e.g. from / to square indices) plus any IL-only labels (e.g. legality
    /// masks conditioned on the taken action's first component).
    fn target_custom(&self, _seat: Seat, _legal: &[u32], _action: u32, _target: &mut TensorsMut<'_>) {
        panic!("{}: a custom-schema game must implement Game::target_custom", Self::NAME)
    }

    /// Custom schema: the legal action the simulator plays for the action keys `keys`
    /// (in `FrameSpec::action_keys` order), or None when they name no legal action.
    /// For every legal `a`, `decode(target(a))` must be a legal action with the same
    /// keys (the conformance battery checks it): the inverse of
    /// [`target_custom`](Game::target_custom) for a one-to-one action space; with an
    /// abstracted one (exact bet amounts in the rules, discretised sizes as keys)
    /// several legal actions share keys and decode to their representative.
    fn decode_custom(&self, _seat: Seat, _legal: &[u32], _keys: &[i64]) -> Option<u32> {
        panic!("{}: a custom-schema game must implement Game::decode_custom", Self::NAME)
    }

    // ---- strings, privacy ----------------------------------------------------------

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        format!("a{action}")
    }

    fn chance_string(&self, outcome: u32) -> String {
        format!("c{outcome}")
    }

    /// Full state, for debugging.
    fn state_string(&self) -> String;

    /// Privacy-test support (mandatory when `spec().hidden_info`): a trajectory that
    /// differs from `events` ONLY in information `seat` never observes up to the end
    /// of `events` (typically: other seats' private chance outcomes re-dealt among
    /// themselves), whose recorded actions all stay legal — or None when this record
    /// has nothing to perturb. [`crate::invariance`] then requires every frame of
    /// `seat` to be bit-identical under both.
    fn perturb_hidden(_cfg: &Self::Config, _events: &[Event], _seat: Seat, _rng: &mut Rng) -> Option<Vec<Event>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand() -> CandidateSpec {
        CandidateSpec {
            n_seats: 2,
            max_entities: 3,
            entity_features: 4,
            entity_types: 0,
            global_features: 2,
            max_actions: 5,
            action_features: 3,
            hidden_info: true,
            simultaneous: false,
            bits_entities: false,
            bits_globals: false,
            bits_actions: true,
            max_decisions: 10,
        }
    }

    #[test]
    fn candidate_schema_and_validation() {
        let s = FrameSpec::candidate(cand());
        s.validate().unwrap();
        let names: Vec<&str> = s.obs.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["entities", "entity_mask", "globals", "actions", "action_mask"]);
        assert!(s.obs[3].bits && !s.obs[0].bits);
        assert_eq!(s.action_key_indices(), vec![0]);
        assert_eq!(s.layout(), Layout::PerSeat);
        let m = s.schema_meta();
        assert_eq!((m["schema"].as_str(), m["max_actions"].as_u64(), m["layout"].as_str()), (Some(CANDIDATE_SCHEMA), Some(5), Some("per_seat")));
        let s = FrameSpec::candidate(CandidateSpec { entity_types: 2, ..cand() });
        assert_eq!(s.obs[2].name, "entity_type");
        // broken custom schemas are rejected
        let mut bad = s.clone();
        bad.action_keys = vec!["nope".into()];
        assert!(bad.validate().is_err());
        let mut bad = s.clone();
        bad.target.push(TensorSpec::f32("value", &[]));
        assert!(bad.validate().is_err());
        let mut bad = s.clone();
        bad.obs.push(TensorSpec::f32("entities", &[1]));
        assert!(bad.validate().unwrap_err().contains("twice"));
        let mut bad = s;
        bad.legal.push(TensorSpec::f32("x", &[0]));
        assert!(bad.validate().is_err());
    }
}
