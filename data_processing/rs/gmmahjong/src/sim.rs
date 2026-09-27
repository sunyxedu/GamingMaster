//! Whole hanchan games (Tenhou rules) on riichienv-core's engine, observed through the
//! converter's per-kyoku state ([`Kyoku`]): the engine deals, applies the rules and scores,
//! and every mjai event it logs is fed to a `Kyoku` exactly as gmdata's `mahjong convert`
//! feeds a Tenhou log's events — so a decision frame here is the converter's frame of the same
//! position ([`Kyoku::encode`]) with the same legality ([`Kyoku::legal`]).
//!
//! Decisions (the converter's frames): every own turn — after a draw (`Ctx::Draw`, focus =
//! the drawn kind) and after a chi / pon (`Ctx::AfterCall`) — and, after another seat's
//! discard or kakan (chankan), every seat that may do something besides pass
//! (`Ctx::Claim`); a seat with nothing but pass passes without a decision.  The seats of
//! one claim window decide on the same state, then the engine resolves their calls by
//! priority (ron > pon / kan > chi; a double ron stands).
//!
//! An action is (type, tile kind) — the converter's labels: dahai / riichi k (riichi = the
//! declaration and its discard, one decision, as in the converter), chi b (the run's lowest
//! kind), pon k, kan k (daiminkan on a claim, ankan / kakan on the own turn), hora (tsumo /
//! ron), ryukyoku (kyuushu kyuuhai), pass.  The physical tile is canonical unless a [`Hint`]
//! names it (tests replaying a log): the drawn tile when it is of the kind (tsumogiri) and not
//! a red five with a plain copy in hand, else a plain copy, else the red one; a call consumes
//! the engine's first matching option.
//!
//! The riichi stick, furiten and the claim-window bookkeeping follow the converter's order:
//! the claim frames of a discard show its dora flip but not its `reach_accepted`, and the
//! seats that did not ron note the pass ([`Kyoku::note_pass`]) once the window closes.

use crate::rules::*;
use crate::state::*;
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::state::GameState;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

/// Final placement 1st .. 4th → value (the converter's IL value target, the RL reward).
pub const PLACEMENT: [f32; 4] = [1.0, (1.0f64 / 3.0) as f32, (-1.0f64 / 3.0) as f32, -1.0];
/// riichienv-core game mode of a four-player hanchan (East + South, West extension).
const HANCHAN: u8 = 2;
/// Tile planes of a frame (u8 quarters) and the other two obs tensors.
pub const TILES_Q: usize = NK * TILE_FEATURES;
pub const PLAYERS: usize = 4 * 12;
pub const TABLE: usize = 12;

/// A decision a seat owes.
#[derive(Clone, Debug)]
pub struct Pending {
    pub seat: usize,
    pub ctx: Ctx,
    pub legal: Legal,
    pub focus: Option<usize>,
    pub own: bool,
    pub focus_actor: Option<usize>,
}

/// The physical tiles of an action when the kind is not enough (a log's replay): the
/// discarded tile (mjai name: red or not) and whether it was the drawn one, a call's consumed
/// tiles.  None = the canonical choice (module docs).
#[derive(Clone, Debug, Default)]
pub struct Hint {
    pub pai: Option<String>,
    pub tsumogiri: Option<bool>,
    pub consumed: Option<Vec<String>>,
}

/// Disagreements between the converter's legality and the engine's (0 expected).
#[derive(Clone, Debug, Default)]
pub struct SimStats {
    /// a claim window where the engine and the converter differ on who may claim
    pub claim_mismatch: usize,
    pub decisions: usize,
}

fn is_red_id(t: u8) -> bool {
    matches!(t, 16 | 52 | 88)
}

fn kind_of(t: u8) -> usize {
    t as usize / 4
}

pub struct MjGame {
    pub gs: GameState,
    pub k: Option<Kyoku>,
    cursor: usize,
    pending: Vec<Pending>,
    /// the discard / kakan whose claim window is open: (discarder, kind, chankan)
    open_claim: Option<(usize, usize, bool)>,
    /// the closed claim window whose passes are not noted yet: (discarder, kind, ronned)
    passes: Option<(usize, usize, [bool; 4])>,
    drawn: Option<usize>,
    /// the call just made: (seat, called kind, consumed kinds, is_chi)
    after_call: Option<(usize, usize, Vec<usize>, bool)>,
    /// test mode: each kyoku's wall (riichienv `_initialize_round` order), in kyoku order
    walls: Option<VecDeque<Vec<u8>>>,
    /// test mode: events of a kyoku start being replaced by the forced wall are skipped
    skip_to_start: bool,
    forced_start: bool,
    pub stats: SimStats,
}

type SimResult<T> = Result<T, String>;

impl MjGame {
    /// A new hanchan dealt from the engine's walls for `seed`.
    pub fn new(seed: u64) -> SimResult<Self> {
        Self::build(seed, None)
    }

    /// A hanchan dealt from the given walls, one per kyoku (tests replaying a log).
    pub fn with_walls(walls: Vec<Vec<u8>>) -> SimResult<Self> {
        Self::build(0, Some(walls.into()))
    }

    fn build(seed: u64, walls: Option<VecDeque<Vec<u8>>>) -> SimResult<Self> {
        let gs = GameState::new(HANCHAN, false, Some(seed), 0, GameRule::default_tenhou());
        let mut g = MjGame { gs, k: None, cursor: 0, pending: vec![], open_claim: None, passes: None, drawn: None,
                             after_call: None, walls, skip_to_start: false, forced_start: false, stats: SimStats::default() };
        g.advance()?;
        Ok(g)
    }

    pub fn done(&self) -> bool {
        self.gs.is_done
    }

    /// The decisions owed now (empty when the game is over).
    pub fn pending(&self) -> &[Pending] {
        &self.pending
    }

    pub fn kyoku(&self) -> Option<&Kyoku> {
        self.k.as_ref()
    }

    /// Final scores (the engine's: the leftover riichi sticks go to the top seat).
    pub fn scores(&self) -> [i64; 4] {
        std::array::from_fn(|i| self.gs.players[i].score as i64)
    }

    /// Per-seat final placement value (ties: the lower seat ranks first, as the converter).
    pub fn outcome(&self) -> [f32; 4] {
        let sc = self.scores();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by_key(|&s| (-sc[s], s));
        let mut v = [0f32; 4];
        for (r, &s) in order.iter().enumerate() {
            v[s] = PLACEMENT[r];
        }
        v
    }

    /// The frame of a pending decision: `ObsBuf` gets one frame appended.
    pub fn encode(&self, p: &Pending, out: &mut ObsBuf) {
        self.k.as_ref().expect("a kyoku is running").encode(p.seat, p.focus, p.own, p.focus_actor, out);
    }

    fn kyoku_mut(&mut self) -> SimResult<&mut Kyoku> {
        self.k.as_mut().ok_or_else(|| "an event before start_kyoku".to_string())
    }

    /// Note the passes of a closed claim window (the converter's order: after its frames).
    fn flush_passes(&mut self) -> SimResult<()> {
        if let Some((d, kind, ronned)) = self.passes.take() {
            let k = self.kyoku_mut()?;
            for rel in 1..4 {
                let s = (d + rel) % 4;
                if !ronned[s] {
                    k.note_pass(s, kind);
                }
            }
        }
        Ok(())
    }

    /// Feed the engine's new mjai events to the kyoku state.
    fn sync(&mut self) -> SimResult<()> {
        while self.cursor < self.gs.mjai_log.len() {
            let ev: Value = serde_json::from_str(&self.gs.mjai_log[self.cursor]).map_err(|e| e.to_string())?;
            self.cursor += 1;
            let t = ev["type"].as_str().unwrap_or("").to_string();
            if self.skip_to_start && t != "start_kyoku" {
                continue;
            }
            let bad = |what: &str| format!("the engine's {what} event does not fit the kyoku state: {ev}");
            // a claim window closes with the next event other than its dora flip, the riichi
            // stick or a ron on it
            if let Some((d, _, ronned)) = self.passes.as_mut() {
                let ron_on_it = t == "hora" && ev["target"].as_u64() == Some(*d as u64) && ev["actor"].as_u64() != Some(*d as u64);
                if ron_on_it {
                    ronned[ev["actor"].as_u64().unwrap_or(0) as usize] = true;
                } else if t != "dora" && t != "reach_accepted" {
                    self.flush_passes()?;
                }
            }
            match t.as_str() {
                "start_kyoku" => {
                    if let (Some(walls), false) = (self.walls.as_mut(), self.forced_start) {
                        // test mode: deal this kyoku from the given wall instead
                        let wall = walls.pop_front().ok_or("more kyoku than walls")?;
                        let scores: Vec<i32> = self.gs.players.iter().map(|p| p.score).collect();
                        let (oya, rw, honba, sticks) = (self.gs.oya, self.gs.round_wind, self.gs.honba, self.gs.riichi_sticks);
                        self.skip_to_start = true;
                        self.forced_start = true;
                        self.gs._initialize_round(oya, rw, honba, sticks, Some(wall), Some(scores));
                        continue;
                    }
                    self.skip_to_start = false;
                    self.forced_start = false;
                    self.k = Some(Kyoku::new(&ev).map_err(|_| bad("start_kyoku"))?);
                    self.drawn = None;
                    self.after_call = None;
                    self.open_claim = None;
                    self.passes = None;
                }
                "tsumo" => {
                    let a = seat(&ev, "actor").map_err(|_| bad("tsumo"))?;
                    let p = s(&ev, "pai").map_err(|_| bad("tsumo"))?.to_string();
                    let kind = self.kyoku_mut()?.draw(a, &p).map_err(|_| bad("tsumo"))?;
                    self.drawn = Some(kind);
                    self.after_call = None;
                }
                "dahai" => {
                    let a = seat(&ev, "actor").map_err(|_| bad("dahai"))?;
                    let p = s(&ev, "pai").map_err(|_| bad("dahai"))?.to_string();
                    let tg = ev.get("tsumogiri").and_then(|v| v.as_bool()).unwrap_or(false);
                    let kind = self.kyoku_mut()?.discard(a, &p, tg).map_err(|_| bad("dahai"))?;
                    self.drawn = None;
                    self.after_call = None;
                    self.open_claim = Some((a, kind, false));
                    self.passes = Some((a, kind, [false; 4]));
                }
                "chi" | "pon" => {
                    self.kyoku_mut()?.call(&ev).map_err(|_| bad(&t))?;
                    let a = seat(&ev, "actor").map_err(|_| bad(&t))?;
                    let called = tile_kind(s(&ev, "pai").map_err(|_| bad(&t))?).map_err(|_| bad(&t))?;
                    let consumed = Kyoku::consumed(&ev).map_err(|_| bad(&t))?.iter()
                        .map(|p| tile_kind(p)).collect::<R<Vec<_>>>().map_err(|_| bad(&t))?;
                    self.after_call = Some((a, called, consumed, t == "chi"));
                    self.open_claim = None;
                }
                "daiminkan" => {
                    self.kyoku_mut()?.call(&ev).map_err(|_| bad(&t))?;
                    self.open_claim = None;
                }
                "ankan" => {
                    self.kyoku_mut()?.ankan(&ev).map_err(|_| bad(&t))?;
                }
                "kakan" => {
                    let a = seat(&ev, "actor").map_err(|_| bad(&t))?;
                    let kind = self.kyoku_mut()?.kakan(&ev).map_err(|_| bad(&t))?;
                    self.open_claim = Some((a, kind, true));
                    self.passes = Some((a, kind, [false; 4]));
                }
                "reach" => {
                    let a = seat(&ev, "actor").map_err(|_| bad(&t))?;
                    self.kyoku_mut()?.pending[a] = true;
                }
                "reach_accepted" => {
                    let a = seat(&ev, "actor").map_err(|_| bad(&t))?;
                    self.kyoku_mut()?.accept_riichi(a);
                }
                "dora" => self.kyoku_mut()?.add_dora(&ev).map_err(|_| bad(&t))?,
                "hora" => {
                    let k = self.kyoku_mut()?;
                    k.add_deltas(&ev).map_err(|_| bad("hora"))?;
                    k.kyotaku = 0;
                }
                "ryukyoku" => self.kyoku_mut()?.add_deltas(&ev).map_err(|_| bad(&t))?,
                _ => {}
            }
        }
        Ok(())
    }

    /// The decisions owed at the engine's state (after [`sync`](Self::sync)).
    fn compute_pending(&mut self) -> SimResult<()> {
        self.pending.clear();
        if self.gs.is_done {
            return Ok(());
        }
        let k = self.k.as_ref().ok_or("no kyoku")?;
        match self.gs.phase {
            Phase::WaitAct => {
                let seat = self.gs.current_player as usize;
                let (ctx, focus) = match &self.after_call {
                    Some((s, called, consumed, is_chi)) if *s == seat => {
                        (Ctx::AfterCall { called: *called, consumed: consumed.clone(), is_chi: *is_chi }, None)
                    }
                    _ => {
                        let drawn = self.drawn.ok_or("an own turn without a draw or a call")?;
                        (Ctx::Draw { drawn }, Some(drawn))
                    }
                };
                let legal = k.legal(seat, &ctx);
                self.pending.push(Pending { seat, ctx, legal, focus, own: true, focus_actor: None });
            }
            Phase::WaitResponse => {
                let (d, kind, chankan) = self.open_claim.ok_or("a claim window without a discard")?;
                let engine: Vec<usize> = self.gs.active_players.iter().map(|&p| p as usize).collect();
                for rel in 1..4 {
                    let s = (d + rel) % 4;
                    let ctx = Ctx::Claim { kind, discarder: d, chankan };
                    let legal = k.legal(s, &ctx);
                    if legal.has_claim() {
                        if !engine.contains(&s) {
                            self.stats.claim_mismatch += 1;
                            continue;
                        }
                        self.pending.push(Pending { seat: s, ctx, legal, focus: Some(kind), own: false, focus_actor: Some(d) });
                    } else if engine.contains(&s) {
                        self.stats.claim_mismatch += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Step the engine with these actions; an engine refusal is an error.
    fn engine_step(&mut self, acts: HashMap<u8, Action>) -> SimResult<()> {
        self.gs.step(&acts);
        if let Some(e) = self.gs.last_error.take() {
            return Err(format!("the engine refused {acts:?}: {e}"));
        }
        Ok(())
    }

    /// Sync, and pass the claim windows nobody can do anything in, until a decision is owed
    /// or the game ends.
    fn advance(&mut self) -> SimResult<()> {
        loop {
            self.sync()?;
            self.compute_pending()?;
            if !self.pending.is_empty() || self.gs.is_done {
                return Ok(());
            }
            if self.gs.phase != Phase::WaitResponse {
                return Err(format!("no decision and no claim window (phase {:?})", self.gs.phase));
            }
            let acts: HashMap<u8, Action> = self.gs.active_players.iter()
                .map(|&p| (p, Action::new(ActionType::Pass, None, vec![], Some(p)))).collect();
            self.engine_step(acts)?;
        }
    }

    /// The engine action for `seat`'s choice (type, kind) under `ctx`.
    fn resolve(&self, seat: usize, atype: i64, tile: i64, own: bool, hint: Option<&Hint>) -> SimResult<Action> {
        let acts = self.gs._get_legal_actions_internal(seat as u8);
        let pid = Some(seat as u8);
        let find = |f: &dyn Fn(&Action) -> bool| acts.iter().find(|a| f(a)).cloned();
        let none = || format!("seat {seat}: no engine action for ({}, {tile}) among {acts:?}", ACTION_NAMES[atype.clamp(0, 7) as usize]);
        let hint_red = |p: &str| is_red(p);
        match atype {
            DAHAI => {
                let k = tile as usize;
                let hand = &self.gs.players[seat].hand;
                let drawn = self.gs.drawn_tile.filter(|&d| kind_of(d) == k);
                let plain = hand.iter().copied().find(|&t| kind_of(t) == k && !is_red_id(t) && Some(t) != self.gs.drawn_tile);
                let red = hand.iter().copied().find(|&t| kind_of(t) == k && is_red_id(t) && Some(t) != self.gs.drawn_tile);
                let any = hand.iter().copied().find(|&t| kind_of(t) == k && Some(t) != self.gs.drawn_tile);
                let id = match hint {
                    Some(Hint { pai: Some(p), tsumogiri, .. }) => {
                        let want_red = hint_red(p);
                        match (tsumogiri, drawn) {
                            (Some(true), Some(d)) => Some(d),
                            _ => {
                                let from_hand = if want_red { red } else { plain };
                                from_hand.or(any).or(drawn.filter(|&d| is_red_id(d) == want_red)).or(drawn)
                            }
                        }
                    }
                    _ => match drawn {
                        Some(d) if !(is_red_id(d) && plain.is_some()) => Some(d),
                        _ => plain.or(red).or(any).or(drawn),
                    },
                };
                let id = id.ok_or_else(none)?;
                find(&|a: &Action| a.action_type == ActionType::Discard && a.tile == Some(id)).ok_or_else(none)
            }
            CHI => {
                let base = tile as usize;
                let want: Option<Vec<(usize, bool)>> = hint.and_then(|h| h.consumed.as_ref())
                    .map(|c| c.iter().map(|p| (tile_kind(p).unwrap_or(99), is_red(p))).collect());
                find(&|a: &Action| {
                    a.action_type == ActionType::Chi
                        && a.consume_tiles.iter().copied().chain(a.tile).map(kind_of).min() == Some(base)
                        && want.as_ref().map(|w| {
                            let mut got: Vec<(usize, bool)> = a.consume_tiles.iter().map(|&t| (kind_of(t), is_red_id(t))).collect();
                            let mut w = w.clone();
                            got.sort();
                            w.sort();
                            got == w
                        }).unwrap_or(true)
                }).ok_or_else(none)
            }
            PON | KAN => {
                let k = tile as usize;
                // the hint's red fives: a pon / daiminkan / ankan consumes hand tiles, a kakan
                // adds one tile (`tile`) to its pon (whose tiles are `consume_tiles`)
                let want_reds: Option<usize> = hint.and_then(|h| h.consumed.as_ref()).map(|c| c.iter().filter(|p| is_red(p)).count());
                let types: &[ActionType] = if atype == PON {
                    &[ActionType::Pon]
                } else if own {
                    &[ActionType::Ankan, ActionType::Kakan]
                } else {
                    &[ActionType::Daiminkan]
                };
                find(&|a: &Action| {
                    let reds = if a.action_type == ActionType::Kakan {
                        a.tile.map(is_red_id).unwrap_or(false) as usize
                    } else {
                        a.consume_tiles.iter().filter(|&&t| is_red_id(t)).count()
                    };
                    types.contains(&a.action_type)
                        && a.tile.or(a.consume_tiles.first().copied()).map(kind_of) == Some(k)
                        && want_reds.map(|r| reds == r).unwrap_or(true)
                }).ok_or_else(none)
            }
            HORA => find(&|a: &Action| a.action_type == if own { ActionType::Tsumo } else { ActionType::Ron }).ok_or_else(none),
            RYUKYOKU => find(&|a: &Action| a.action_type == ActionType::KyushuKyuhai).ok_or_else(none),
            PASS => Ok(Action::new(ActionType::Pass, None, vec![], pid)),
            RIICHI => Ok(Action::new(ActionType::Riichi, None, vec![], pid)),
            _ => Err(none()),
        }
    }

    /// Answer every pending decision: `choices[i]` = (type, tile kind, hint) of
    /// `pending()[i]`.  A choice outside the decision's legality is an error.
    pub fn act(&mut self, choices: &[(i64, i64, Option<Hint>)]) -> SimResult<()> {
        if choices.len() != self.pending.len() {
            return Err(format!("{} choices for {} decisions", choices.len(), self.pending.len()));
        }
        for (p, (t, tile, _)) in self.pending.iter().zip(choices) {
            let ok = (0..N_TYPES as i64).contains(t) && p.legal.types[*t as usize]
                && (!USES_TILE[*t as usize] || ((0..NK as i64).contains(tile) && p.legal.tiles[*t as usize][*tile as usize]));
            if !ok {
                return Err(format!("seat {}: ({}, {tile}) is not legal", p.seat, ACTION_NAMES[(*t).clamp(0, 7) as usize]));
            }
        }
        self.stats.decisions += choices.len();
        let pending = std::mem::take(&mut self.pending);
        if self.gs.phase == Phase::WaitAct {
            let (p, (t, tile, hint)) = (&pending[0], &choices[0]);
            let a = self.resolve(p.seat, *t, *tile, true, hint.as_ref())?;
            self.engine_step(HashMap::from([(p.seat as u8, a)]))?;
            if *t == RIICHI {
                // the declaration's discard: part of the same decision
                self.sync()?;
                if self.gs.phase != Phase::WaitAct || self.gs.current_player as usize != p.seat {
                    return Err("riichi: the engine does not wait for the declaration discard".into());
                }
                let d = self.resolve(p.seat, DAHAI, *tile, true, hint.as_ref())?;
                self.engine_step(HashMap::from([(p.seat as u8, d)]))?;
            }
        } else {
            let mut acts: HashMap<u8, Action> = self.gs.active_players.iter()
                .map(|&s| (s, Action::new(ActionType::Pass, None, vec![], Some(s)))).collect();
            for (p, (t, tile, hint)) in pending.iter().zip(choices) {
                acts.insert(p.seat as u8, self.resolve(p.seat, *t, *tile, false, hint.as_ref())?);
            }
            self.engine_step(acts)?;
        }
        self.advance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random legal play: whole hanchan games end, every decision's frame encodes, and
    /// nothing disagrees with the engine.
    #[test]
    fn random_games_play_to_the_end() {
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut next = move |n: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % n as u64) as usize
        };
        let mut decisions = 0;
        for seed in 0..6u64 {
            let mut g = MjGame::new(seed).unwrap();
            let mut steps = 0;
            while !g.done() {
                let mut choices = Vec::new();
                for p in g.pending() {
                    let mut buf = ObsBuf::default();
                    g.encode(p, &mut buf);
                    assert_eq!((buf.tiles_q.len(), buf.players.len(), buf.table.len()), (TILES_Q, PLAYERS, TABLE));
                    let opts: Vec<(i64, i64)> = (0..N_TYPES)
                        .filter(|&t| p.legal.types[t])
                        .flat_map(|t| {
                            let tiles: Vec<i64> = if USES_TILE[t] { (0..NK).filter(|&k| p.legal.tiles[t][k]).map(|k| k as i64).collect() } else { vec![0] };
                            tiles.into_iter().map(move |k| (t as i64, k))
                        })
                        .collect();
                    assert!(!opts.is_empty(), "a decision with no legal action");
                    // prefer winning, else uniform
                    let (t, k) = opts.iter().copied().find(|&(t, _)| t == HORA).unwrap_or(opts[next(opts.len())]);
                    choices.push((t, k, None));
                }
                g.act(&choices).unwrap_or_else(|e| panic!("seed {seed} step {steps}: {e}"));
                steps += 1;
                assert!(steps < 20_000, "a game that does not end");
            }
            decisions += g.stats.decisions;
            assert_eq!(g.stats.claim_mismatch, 0, "seed {seed}");
            let v = g.outcome();
            let mut sorted = v;
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert_eq!(sorted, PLACEMENT);
            let total: i64 = g.scores().iter().sum();
            assert_eq!(total, 100_000, "points are conserved (leftover sticks paid to the top)");
        }
        assert!(decisions > 1000, "{decisions}");
    }
}
