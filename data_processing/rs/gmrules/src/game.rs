//! The [`Game`] trait every rule engine implements, and its vocabulary: seats,
//! turns, canonical action ids, the [`FrameSpec`] of the candidate-action
//! observation schema.

use crate::frame::ObsView;
use crate::rng::Rng;
use crate::trajectory::Event;
use serde::de::DeserializeOwned;
use serde::Serialize;
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
    /// are encoded before any of the turn's actions is applied) and the actions are
    /// applied in ascending seat order.
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

/// Fixed shapes of a game's candidate-action frames (gm-shard obs schema; the model's
/// `CandidateConfig`): per decision frame
///
/// ```text
/// entities     [max_entities, entity_features] f32   entity_mask [max_entities] bool
/// entity_type  [max_entities] i32 (only when entity_types > 0; ids < entity_types)
/// globals      [global_features] f32
/// actions      [max_actions, action_features] f32    action_mask [max_actions] bool
/// ```
///
/// The candidates of a frame are the acting seat's LEGAL actions in ascending
/// canonical-id order, padded to `max_actions`; `tgt.action_idx` is the index of the
/// taken action among them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameSpec {
    pub n_seats: usize,
    pub max_entities: usize,
    pub entity_features: usize,
    /// Number of entity type ids (0 = no `entity_type` tensor).
    pub entity_types: usize,
    pub global_features: usize,
    pub max_actions: usize,
    pub action_features: usize,
    /// The observation of a seat hides information other seats have (cards, dice
    /// known to one player, ...): per-seat streams, and the privacy invariance test
    /// is mandatory ([`Game::perturb_hidden`] must be implemented).
    pub hidden_info: bool,
    /// Some turns have several acting seats (per-seat streams).
    pub simultaneous: bool,
    /// Storage hints for the converters: the tensor holds only 0.0 / 1.0 and is
    /// stored bit-packed (32x smaller; loaded back as float32).
    pub bits_entities: bool,
    pub bits_globals: bool,
    pub bits_actions: bool,
    /// Hard bound on decision turns per episode (records / simulations longer than
    /// this are rejected / truncated).
    pub max_decisions: usize,
}

impl FrameSpec {
    pub fn layout(&self) -> Layout {
        if self.hidden_info || self.simultaneous {
            Layout::PerSeat
        } else {
            Layout::Interleaved
        }
    }

    /// Per-frame lengths of the flat buffers.
    pub fn entities_len(&self) -> usize {
        self.max_entities * self.entity_features
    }

    pub fn actions_len(&self) -> usize {
        self.max_actions * self.action_features
    }

    /// The gm-shard / model schema meta (`index.json` "meta" keys the trainers check
    /// against the model config, train_il_game META_ARCH_KEYS).
    pub fn schema_meta(&self) -> serde_json::Value {
        serde_json::json!({
            "max_entities": self.max_entities,
            "entity_features": self.entity_features,
            "entity_types": self.entity_types,
            "global_features": self.global_features,
            "max_actions": self.max_actions,
            "action_features": self.action_features,
            "n_seats": self.n_seats,
            "layout": self.layout().name(),
        })
    }
}

/// A game's rules and its observation / candidate-action encoders.
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
/// - [`encode_obs`](Game::encode_obs) may only read what `seat` knows at this point:
///   public state, its own private state, its own past actions — never other seats'
///   private information and never the future (the causality / privacy tests in
///   [`crate::invariance`] check it).
/// - [`outcome`](Game::outcome) is the per-seat final result in [-1, 1]: the IL value
///   target of every frame of that seat AND the RL reward paid at the end.
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

    fn spec(cfg: &Self::Config) -> FrameSpec;

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

    /// Write `seat`'s observation (entities / globals; the buffers arrive zeroed).
    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>);

    /// Features of the legal candidate `action` for `seat` (`feat` arrives zeroed,
    /// `action_features` long).
    fn encode_action(&self, seat: Seat, action: u32, feat: &mut [f32]);

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
