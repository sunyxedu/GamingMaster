//! Candidate-action decision frames: the buffers a [`Game`] encodes into, and
//! [`encode_frame`], the ONE encoding routine the converters and the simulators call.

use crate::game::{FrameSpec, Game, RulesError, Seat};

/// One decision frame of one seat (owned). Masks are 0 / 1 bytes.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub entities: Vec<f32>,
    pub entity_mask: Vec<u8>,
    pub entity_type: Vec<i32>,
    pub globals: Vec<f32>,
    pub actions: Vec<f32>,
    pub action_mask: Vec<u8>,
    /// Canonical ids of the candidates (the legal actions, ascending); candidate `i`
    /// is `legal[i]`.
    pub legal: Vec<u32>,
}

impl Frame {
    pub fn new(spec: &FrameSpec) -> Self {
        Frame {
            entities: vec![0.0; spec.entities_len()],
            entity_mask: vec![0; spec.max_entities],
            entity_type: vec![0; spec.max_entities],
            globals: vec![0.0; spec.global_features],
            actions: vec![0.0; spec.actions_len()],
            action_mask: vec![0; spec.max_actions],
            legal: Vec::new(),
        }
    }

    pub fn as_mut(&mut self) -> FrameMut<'_> {
        FrameMut {
            entities: &mut self.entities,
            entity_mask: &mut self.entity_mask,
            entity_type: &mut self.entity_type,
            globals: &mut self.globals,
            actions: &mut self.actions,
            action_mask: &mut self.action_mask,
            legal: &mut self.legal,
        }
    }

    /// Candidate index of canonical action `action` (None = not legal here).
    pub fn index_of(&self, action: u32) -> Option<usize> {
        self.legal.binary_search(&action).ok()
    }

    /// Bit-exact comparison (f32 by `to_bits`, so -0.0 != 0.0); the error names the
    /// first differing element.
    pub fn check_equal(&self, other: &Frame) -> Result<(), String> {
        fn f32s(name: &str, a: &[f32], b: &[f32]) -> Result<(), String> {
            if a.len() != b.len() {
                return Err(format!("{name}: length {} vs {}", a.len(), b.len()));
            }
            match a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits()) {
                Some(i) => Err(format!("{name}[{i}]: {} vs {}", a[i], b[i])),
                None => Ok(()),
            }
        }
        fn eq<T: PartialEq + std::fmt::Debug>(name: &str, a: &[T], b: &[T]) -> Result<(), String> {
            if a.len() != b.len() {
                return Err(format!("{name}: length {} vs {}", a.len(), b.len()));
            }
            match a.iter().zip(b).position(|(x, y)| x != y) {
                Some(i) => Err(format!("{name}[{i}]: {:?} vs {:?}", a[i], b[i])),
                None => Ok(()),
            }
        }
        f32s("entities", &self.entities, &other.entities)?;
        eq("entity_mask", &self.entity_mask, &other.entity_mask)?;
        eq("entity_type", &self.entity_type, &other.entity_type)?;
        f32s("globals", &self.globals, &other.globals)?;
        f32s("actions", &self.actions, &other.actions)?;
        eq("action_mask", &self.action_mask, &other.action_mask)?;
        eq("legal", &self.legal, &other.legal)
    }
}

/// Mutable views of one frame's buffers (an owned [`Frame`], or one row of a
/// simulator's flat batch buffers).
pub struct FrameMut<'a> {
    pub entities: &'a mut [f32],
    pub entity_mask: &'a mut [u8],
    pub entity_type: &'a mut [i32],
    pub globals: &'a mut [f32],
    pub actions: &'a mut [f32],
    pub action_mask: &'a mut [u8],
    pub legal: &'a mut Vec<u32>,
}

impl FrameMut<'_> {
    /// Zero every buffer (an empty / inactive row).
    pub fn clear(&mut self) {
        self.entities.fill(0.0);
        self.entity_mask.fill(0);
        self.entity_type.fill(0);
        self.globals.fill(0.0);
        self.actions.fill(0.0);
        self.action_mask.fill(0);
        self.legal.clear();
    }
}

/// What [`Game::encode_obs`] writes into: entity slots and the global features.
/// Buffers arrive zeroed; untouched slots stay absent (mask 0, features 0, type 0).
pub struct ObsView<'a> {
    entity_features: usize,
    entities: &'a mut [f32],
    entity_mask: &'a mut [u8],
    entity_type: &'a mut [i32],
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
        self.entity_type[slot] = entity_type;
        let f = self.entity_features;
        &mut self.entities[slot * f..(slot + 1) * f]
    }

    pub fn globals(&mut self) -> &mut [f32] {
        self.globals
    }
}

/// Encode `seat`'s decision frame of state `g`: observation, then the legal
/// candidates (ids, features, mask). The single encoding path of converters and
/// simulators. Errors: no legal action, more legal actions than `max_actions`, ids not
/// strictly ascending.
pub fn encode_frame<G: Game>(g: &G, seat: Seat, spec: &FrameSpec, f: &mut FrameMut<'_>) -> Result<(), RulesError> {
    f.clear();
    let mut view = ObsView {
        entity_features: spec.entity_features,
        entities: f.entities,
        entity_mask: f.entity_mask,
        entity_type: f.entity_type,
        globals: f.globals,
    };
    g.encode_obs(seat, &mut view);
    g.legal_actions(seat, f.legal);
    let n = f.legal.len();
    if n == 0 {
        return Err(RulesError(format!("{}: seat {seat} is to act but has no legal action", G::NAME)));
    }
    if n > spec.max_actions {
        return Err(RulesError(format!(
            "{}: {n} legal actions exceed max_actions {} (seat {seat})",
            G::NAME,
            spec.max_actions
        )));
    }
    if f.legal.windows(2).any(|w| w[0] >= w[1]) {
        return Err(RulesError(format!("{}: legal actions not strictly ascending: {:?}", G::NAME, f.legal)));
    }
    let fa = spec.action_features;
    for (i, &a) in f.legal.iter().enumerate() {
        g.encode_action(seat, a, &mut f.actions[i * fa..(i + 1) * fa]);
        f.action_mask[i] = 1;
    }
    Ok(())
}
