//! Pig — the framework's PERFECT-INFORMATION test game with chance (dice), played
//! through the INTERLEAVED layout (one stream per game holding every decision from
//! the mover's view, like backgammon would be). Exactly OpenSpiel's
//! `pig(players=N, winscore=W, diceoutcomes=D, horizon=H)` (pinned by
//! tests/test_candidate_reference.py).
//!
//! Rules. Players take turns from seat 0. On its turn a player repeatedly chooses Roll
//! (0) or Stop (1). Roll: chance throws a D-sided die (outcome id = face − 1); a 1
//! loses the turn total and ends the turn, any other face adds to the turn total.
//! Stop: the turn total is banked into the player's score and the turn passes. When
//! score + turn total reaches W, Stop is the only legal action; banking W or more
//! wins. After H decisions the game ends undecided. Returns: winner +1, the others
//! −1/(N−1); undecided 0.
//!
//! Frames (interleaved: the mover's view; nothing is hidden):
//!
//! ```text
//! entities [4, 4]  seats relative to the mover (type 0 me / 1 opponent):
//!                    is_me, k/3, score/W, (score + turn total)/W for the mover
//! globals  [7]     turn total/W, can only stop, decisions/H, N==2, N==3, N==4, D/10
//! actions  [2, 3]  Roll, Stop: is_roll, is_stop, banking wins
//! ```

use crate::frame::ObsView;
use crate::game::{FrameSpec, Game, RulesError, Seat, SeatSet, Turn};
use crate::rules_bail;
use serde::{Deserialize, Serialize};

pub const ROLL: u32 = 0;
pub const STOP: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PigConfig {
    /// 2..=4
    pub players: usize,
    /// score to win, >= 1
    pub winscore: u32,
    /// die faces, 2..=10
    pub diceoutcomes: u32,
    /// decisions before the game ends undecided, >= 1
    pub horizon: usize,
}

impl Default for PigConfig {
    fn default() -> Self {
        PigConfig { players: 2, winscore: 20, diceoutcomes: 6, horizon: 200 }
    }
}

#[derive(Clone, Debug)]
pub struct Pig {
    cfg: PigConfig,
    scores: Vec<u32>,
    turn_total: u32,
    player: Seat,
    rolling: bool,
    decisions: usize,
    winner: Option<Seat>,
}

impl Pig {
    fn over(&self) -> bool {
        self.winner.is_some() || self.decisions >= self.cfg.horizon
    }

    fn only_stop(&self) -> bool {
        self.scores[self.player] + self.turn_total >= self.cfg.winscore
    }

    fn next_player(&mut self) {
        self.turn_total = 0;
        self.player = (self.player + 1) % self.cfg.players;
    }
}

impl Game for Pig {
    type Config = PigConfig;
    const NAME: &'static str = "pig";
    const VERSION: u32 = 1;

    fn check_config(cfg: &PigConfig) -> Result<(), RulesError> {
        if !(2..=4).contains(&cfg.players) || cfg.winscore < 1 || !(2..=10).contains(&cfg.diceoutcomes) || cfg.horizon < 1 {
            rules_bail!("players 2..=4, winscore >= 1, diceoutcomes 2..=10, horizon >= 1; got {cfg:?}");
        }
        Ok(())
    }

    fn spec(cfg: &PigConfig) -> FrameSpec {
        FrameSpec {
            n_seats: cfg.players,
            max_entities: 4,
            entity_features: 4,
            entity_types: 2,
            global_features: 7,
            max_actions: 2,
            action_features: 3,
            hidden_info: false,
            simultaneous: false,
            bits_entities: false,
            bits_globals: false,
            bits_actions: true,
            max_decisions: cfg.horizon,
        }
    }

    fn new_game(cfg: &PigConfig) -> Self {
        Pig { cfg: cfg.clone(), scores: vec![0; cfg.players], turn_total: 0, player: 0, rolling: false, decisions: 0, winner: None }
    }

    fn turn(&self) -> Turn {
        if self.over() {
            Turn::Terminal
        } else if self.rolling {
            Turn::Chance
        } else {
            Turn::Act(SeatSet::single(self.player))
        }
    }

    fn chance_outcomes(&self, out: &mut Vec<(u32, f64)>) {
        let d = self.cfg.diceoutcomes;
        out.extend((0..d).map(|f| (f, 1.0 / d as f64)));
    }

    fn apply_chance(&mut self, outcome: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Chance || outcome >= self.cfg.diceoutcomes {
            rules_bail!("pig: die outcome {outcome} not possible now");
        }
        self.rolling = false;
        if outcome == 0 {
            self.next_player(); // a 1: the turn total is lost
        } else {
            self.turn_total += outcome + 1;
        }
        Ok(())
    }

    fn legal_actions(&self, seat: Seat, out: &mut Vec<u32>) {
        if self.turn() == Turn::Act(SeatSet::single(seat)) {
            if self.only_stop() {
                out.push(STOP);
            } else {
                out.extend([ROLL, STOP]);
            }
        }
    }

    fn apply_action(&mut self, seat: Seat, action: u32) -> Result<(), RulesError> {
        if self.turn() != Turn::Act(SeatSet::single(seat)) {
            rules_bail!("pig: seat {seat} is not to act");
        }
        match action {
            ROLL if !self.only_stop() => self.rolling = true,
            STOP => {
                self.scores[seat] += self.turn_total;
                if self.scores[seat] >= self.cfg.winscore {
                    self.winner = Some(seat);
                }
                self.next_player();
            }
            _ => rules_bail!("pig: action {action} is illegal"),
        }
        self.decisions += 1;
        if self.decisions >= self.cfg.horizon {
            self.rolling = false; // the game ends before the die is thrown
        }
        Ok(())
    }

    fn returns(&self) -> Vec<f64> {
        let n = self.cfg.players;
        match self.winner {
            Some(w) => (0..n).map(|p| if p == w { 1.0 } else { -1.0 / (n - 1) as f64 }).collect(),
            None => vec![0.0; n],
        }
    }

    fn outcome(&self) -> Vec<f32> {
        self.returns().iter().map(|&r| r as f32).collect()
    }

    fn encode_obs(&self, seat: Seat, obs: &mut ObsView<'_>) {
        let n = self.cfg.players;
        let w = self.cfg.winscore as f32;
        for k in 0..n {
            let p = (seat + k) % n;
            let f = obs.entity(k, (k != 0) as i32);
            f[0] = (k == 0) as u8 as f32;
            f[1] = k as f32 / 3.0;
            f[2] = self.scores[p] as f32 / w;
            f[3] = if k == 0 { (self.scores[p] + self.turn_total) as f32 / w } else { f[2] };
        }
        let g = obs.globals();
        g[0] = self.turn_total as f32 / w;
        g[1] = self.only_stop() as u8 as f32;
        g[2] = self.decisions as f32 / self.cfg.horizon as f32;
        g[3] = (n == 2) as u8 as f32;
        g[4] = (n == 3) as u8 as f32;
        g[5] = (n == 4) as u8 as f32;
        g[6] = self.cfg.diceoutcomes as f32 / 10.0;
    }

    fn encode_action(&self, _seat: Seat, action: u32, feat: &mut [f32]) {
        feat[0] = (action == ROLL) as u8 as f32;
        feat[1] = (action == STOP) as u8 as f32;
        feat[2] = (action == STOP && self.only_stop()) as u8 as f32;
    }

    fn action_string(&self, _seat: Seat, action: u32) -> String {
        if action == ROLL { "roll".into() } else { "stop".into() }
    }

    fn chance_string(&self, outcome: u32) -> String {
        format!("Face:{}", outcome + 1)
    }

    fn state_string(&self) -> String {
        format!("scores={:?} turn_total={} player={} rolling={} decisions={} winner={:?}",
                self.scores, self.turn_total, self.player, self.rolling, self.decisions, self.winner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Layout;
    use crate::invariance::check_game;
    use crate::trajectory::{replay, Event};

    #[test]
    fn rules_match_openspiel_examples() {
        // pig(players=2,winscore=10): the start of OpenSpiel's episode 0 in the probe
        let cfg = PigConfig { players: 2, winscore: 10, diceoutcomes: 6, horizon: 40 };
        use Event::{Act, Chance};
        let ev = vec![Act(0, ROLL), Chance(4), Act(0, ROLL), Chance(2), Act(0, STOP), Act(1, ROLL), Chance(4),
                      Act(1, ROLL), Chance(3), Act(1, STOP), Act(0, ROLL), Chance(1), Act(0, STOP)];
        let r = replay::<Pig>(&cfg, &ev).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.scores, vec![10, 9]);
        assert_eq!(r.state.returns(), vec![1.0, -1.0]);
        // score + turn total >= winscore: stop is the only candidate
        assert_eq!(r.decisions.last().unwrap().frame.legal, vec![STOP]);
        assert_eq!(Pig::spec(&cfg).layout(), Layout::Interleaved);
        // a 1 loses the turn total and passes the turn
        let r = replay::<Pig>(&cfg, &[Act(0, ROLL), Chance(3), Act(0, ROLL), Chance(0)]).unwrap();
        assert_eq!((r.state.player, r.state.turn_total, r.state.scores[0]), (1, 0, 0));
        // horizon: the game ends undecided after H decisions (no die thrown)
        let cfg = PigConfig { players: 2, winscore: 100, diceoutcomes: 6, horizon: 3 };
        let r = replay::<Pig>(&cfg, &[Act(0, ROLL), Chance(1), Act(0, ROLL), Chance(1), Act(0, ROLL)]).unwrap();
        assert!(r.terminal());
        assert_eq!(r.state.returns(), vec![0.0, 0.0]);
    }

    #[test]
    fn conformance_battery() {
        for cfg in [PigConfig::default(), PigConfig { players: 3, winscore: 8, diceoutcomes: 3, horizon: 30 },
                    PigConfig { players: 4, winscore: 15, diceoutcomes: 6, horizon: 60 }] {
            let rep = check_game::<Pig>(&cfg, 60, 9).unwrap();
            assert_eq!(rep.terminal, 60);
        }
    }
}
