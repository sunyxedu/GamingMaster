//! Decision frames: the buffers a [`Game`] encodes into, and the ONE encoding path the
//! converters and the simulators share — [`encode_frame`] (obs + legal tensors),
//! [`encode_target`] (IL targets / action keys of the taken action) and
//! [`decode_action`] (action keys → canonical action, the simulator's step).
//!
//! Candidate-schema games are encoded here from `Game::encode_obs` /
//! `Game::encode_action` ([`ObsView`]); custom-schema games write their own tensors.

use crate::game::{CandidateSpec, FrameSpec, Game, RulesError, Seat};
use crate::schema::{Buf, BufMut, TensorsMut};

/// One decision frame of one seat (owned): obs and legal tensors in schema order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    pub obs: Vec<Buf>,
    pub legal: Vec<Buf>,
    /// Canonical ids of the legal actions, ascending (candidate schema: candidate `i`
    /// is `legal_ids[i]`).
    pub legal_ids: Vec<u32>,
}

impl Frame {
    pub fn new(spec: &FrameSpec) -> Self {
        Frame { obs: Buf::frame(&spec.obs), legal: Buf::frame(&spec.legal), legal_ids: Vec::new() }
    }

    pub fn as_mut<'a>(&'a mut self, spec: &'a FrameSpec) -> FrameMut<'a> {
        FrameMut {
            obs: TensorsMut::of(&spec.obs, &mut self.obs),
            legal: TensorsMut::of(&spec.legal, &mut self.legal),
            legal_ids: &mut self.legal_ids,
        }
    }

    /// Index of canonical action `action` among the legal actions (None = illegal).
    pub fn index_of(&self, action: u32) -> Option<usize> {
        self.legal_ids.binary_search(&action).ok()
    }

    /// Obs tensor `name`.
    pub fn obs_named<'a>(&'a self, spec: &FrameSpec, name: &str) -> Option<&'a Buf> {
        crate::schema::index_of(&spec.obs, name).map(|i| &self.obs[i])
    }

    /// Bit-exact comparison (f32 by `to_bits`, so -0.0 != 0.0); the error names the
    /// first differing tensor element.
    pub fn check_equal(&self, other: &Frame, spec: &FrameSpec) -> Result<(), String> {
        for (group, specs, a, b) in [("obs", &spec.obs, &self.obs, &other.obs), ("legal", &spec.legal, &self.legal, &other.legal)] {
            for ((t, x), y) in specs.iter().zip(a).zip(b) {
                x.check_equal(y).map_err(|e| format!("{group}.{}{e}", t.name))?;
            }
        }
        if self.legal_ids != other.legal_ids {
            return Err(format!("legal ids {:?} vs {:?}", self.legal_ids, other.legal_ids));
        }
        Ok(())
    }
}

/// Bit-exact comparison of two decisions' target tensors.
pub fn check_targets_equal(a: &[Buf], b: &[Buf], spec: &FrameSpec) -> Result<(), String> {
    for ((t, x), y) in spec.target.iter().zip(a).zip(b) {
        x.check_equal(y).map_err(|e| format!("target.{}{e}", t.name))?;
    }
    Ok(())
}

/// Mutable views of one frame (an owned [`Frame`], or one row of a simulator's batch
/// buffers).
pub struct FrameMut<'a> {
    pub obs: TensorsMut<'a>,
    pub legal: TensorsMut<'a>,
    /// The legal actions (filled, ascending, before `Game::encode_custom` is called).
    pub legal_ids: &'a mut Vec<u32>,
}

impl FrameMut<'_> {
    /// Zero every tensor and forget the legal actions (an empty / inactive row).
    pub fn clear(&mut self) {
        self.obs.clear();
        self.legal.clear();
        self.legal_ids.clear();
    }
}

/// Candidate schema: what [`Game::encode_obs`] writes into — entity slots and the
/// global features. Buffers arrive zeroed; untouched slots stay absent (mask 0,
/// features 0, type 0).
pub struct ObsView<'a> {
    entity_features: usize,
    entities: &'a mut [f32],
    entity_mask: &'a mut [u8],
    entity_type: Option<&'a mut [i32]>,
    globals: &'a mut [f32],
}

impl<'a> ObsView<'a> {
    /// Mark entity `slot` present with type id `entity_type` (0 when the game has no
    /// types) and return its feature slice. Keep slots stable across frames (slot =
    /// identity: a card, a seat, a zone): the core mixes each token slot over time.
    pub fn entity(&mut self, slot: usize, entity_type: i32) -> &mut [f32] {
        let n = self.entity_mask.len();
        assert!(slot < n, "entity slot {slot} >= max_entities {n}");
        self.entity_mask[slot] = 1;
        match &mut self.entity_type {
            Some(t) => t[slot] = entity_type,
            None => assert!(entity_type == 0, "entity type {entity_type}, but the game declares entity_types = 0"),
        }
        let f = self.entity_features;
        &mut self.entities[slot * f..(slot + 1) * f]
    }

    pub fn globals(&mut self) -> &mut [f32] {
        self.globals
    }
}

/// The candidate encoding of `seat`'s frame (legal ids already filled).
fn encode_candidate<G: Game>(g: &G, seat: Seat, c: &CandidateSpec, f: &mut FrameMut<'_>) {
    let (fe, fa) = (c.entity_features, c.action_features);
    let n = f.legal_ids.len();
    {
        let (ents, mask, types, globals, actions, amask) = match f.obs.bufs_mut() {
            [BufMut::F32(e), BufMut::Bool(m), BufMut::I32(t), BufMut::F32(g), BufMut::F32(a), BufMut::Bool(am)] => {
                (e, m, Some(t), g, a, am)
            }
            [BufMut::F32(e), BufMut::Bool(m), BufMut::F32(g), BufMut::F32(a), BufMut::Bool(am)] => (e, m, None, g, a, am),
            _ => unreachable!("candidate obs layout"),
        };
        let mut view = ObsView {
            entity_features: fe,
            entities: ents,
            entity_mask: mask,
            entity_type: types.map(|t| &mut **t),
            globals,
        };
        g.encode_obs(seat, &mut view);
        for (i, &a) in f.legal_ids.iter().enumerate() {
            g.encode_action(seat, a, &mut actions[i * fa..(i + 1) * fa]);
            amask[i] = 1;
        }
    }
    let legal = f.legal.bool(0);
    legal[..n].fill(1);
}

/// Encode `seat`'s decision frame of state `g` (obs + legal tensors + legal ids). The
/// single encoding path of converters and simulators. Errors: no legal action, legal
/// ids not strictly ascending, more legal actions than `max_actions` (candidate
/// schema).
pub fn encode_frame<G: Game>(g: &G, seat: Seat, spec: &FrameSpec, f: &mut FrameMut<'_>) -> Result<(), RulesError> {
    f.clear();
    g.legal_actions(seat, f.legal_ids);
    let n = f.legal_ids.len();
    if n == 0 {
        return Err(RulesError(format!("{}: seat {seat} is to act but has no legal action", G::NAME)));
    }
    if f.legal_ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err(RulesError(format!("{}: legal actions not strictly ascending: {:?}", G::NAME, f.legal_ids)));
    }
    match &spec.candidate {
        Some(c) => {
            if n > c.max_actions {
                return Err(RulesError(format!("{}: {n} legal actions exceed max_actions {} (seat {seat})", G::NAME, c.max_actions)));
            }
            encode_candidate(g, seat, c, f);
        }
        None => g.encode_custom(seat, f),
    }
    Ok(())
}

/// The target tensors of `seat` taking the legal `action` (`legal` = its legal ids,
/// ascending) in state `g` — before the action is applied. Candidate schema: the
/// candidate index.
pub fn encode_target<G: Game>(g: &G, seat: Seat, spec: &FrameSpec, legal: &[u32], action: u32,
                              t: &mut TensorsMut<'_>) -> Result<(), RulesError> {
    t.clear();
    if spec.is_candidate() {
        let Ok(i) = legal.binary_search(&action) else {
            return Err(RulesError(format!("{}: action {action} is illegal for seat {seat}", G::NAME)));
        };
        t.i64(0)[0] = i as i64;
    } else {
        if legal.binary_search(&action).is_err() {
            return Err(RulesError(format!("{}: action {action} is illegal for seat {seat}", G::NAME)));
        }
        g.target_custom(seat, legal, action, t);
    }
    Ok(())
}

/// Owned target buffers of one decision.
pub fn new_target(spec: &FrameSpec) -> Vec<Buf> {
    Buf::frame(&spec.target)
}

/// The action keys (in `FrameSpec::action_keys` order) of target buffers.
pub fn action_keys_of(spec: &FrameSpec, target: &[Buf]) -> Vec<i64> {
    spec.action_key_indices()
        .into_iter()
        .map(|i| target[i].as_i64().expect("action keys are i64")[0])
        .collect()
}

/// The canonical action with action keys `keys` for `seat` (`legal` = its legal ids),
/// or an error naming why it is not a legal action.
pub fn decode_action<G: Game>(g: &G, seat: Seat, spec: &FrameSpec, legal: &[u32], keys: &[i64]) -> Result<u32, String> {
    if keys.len() != spec.action_keys.len() {
        return Err(format!("{} action keys {:?} expected, got {keys:?}", spec.action_keys.len(), spec.action_keys));
    }
    if spec.is_candidate() {
        let i = keys[0];
        if i < 0 || i as usize >= legal.len() {
            return Err(format!("action_idx {i} is not a legal candidate (n_legal {})", legal.len()));
        }
        Ok(legal[i as usize])
    } else {
        match g.decode_custom(seat, legal, keys) {
            Some(a) if legal.binary_search(&a).is_ok() => Ok(a),
            Some(a) => Err(format!("action keys {:?} = {keys:?} decode to action {a}, which is illegal", spec.action_keys)),
            None => Err(format!("action keys {:?} = {keys:?} are no legal action ({} legal)", spec.action_keys, legal.len())),
        }
    }
}
