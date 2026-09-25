//! Deterministic elixir reconstruction. The replay only records plays, but
//! elixir is a closed-form function of time and of what was spent, so both
//! players' elixir is recoverable exactly (up to the moment a card is played,
//! which the replay gives at 20 Hz).

use crate::catalog::Timeline;

#[derive(Debug, Clone)]
pub struct ElixirSim<'a> {
    tl: &'a Timeline,
    pub elixir: f64,
    pub t_s: f64,
    /// plays that cost more than the reconstructed elixir (rule / replay mismatch)
    pub violations: u32,
    /// total elixir lost to the cap (overflow / "leak")
    pub leaked: f64,
}

impl<'a> ElixirSim<'a> {
    pub fn new(tl: &'a Timeline) -> Self {
        Self { tl, elixir: tl.starting_elixir, t_s: 0.0, violations: 0, leaked: 0.0 }
    }

    /// Advance the clock to `to_s`, integrating the piecewise-constant regen rate.
    pub fn advance(&mut self, to_s: f64) {
        if to_s <= self.t_s {
            return;
        }
        let mut t = self.t_s;
        while t < to_s {
            // segment boundary after t
            let mut seg_start = 0.0;
            let mut seg_end = f64::INFINITY;
            for &(len, _) in &self.tl.rates {
                if t < seg_start + len {
                    seg_end = seg_start + len;
                    break;
                }
                seg_start += len;
            }
            let step_end = seg_end.min(to_s);
            let gained = (step_end - t) * self.tl.rate_at(t);
            let before = self.elixir;
            self.elixir = (self.elixir + gained).min(self.tl.max_elixir);
            self.leaked += (before + gained) - self.elixir;
            t = step_end;
        }
        self.t_s = to_s;
    }

    /// Spend `cost` at the current time. Returns false (and counts a violation)
    /// if the reconstruction says the player could not afford it; elixir is
    /// clamped at zero so one mismatch does not poison the rest of the match.
    pub fn spend(&mut self, cost: f64) -> bool {
        let ok = self.elixir + 1e-6 >= cost;
        if !ok {
            self.violations += 1;
        }
        self.elixir = (self.elixir - cost).max(0.0);
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl() -> Timeline {
        Timeline {
            tick_ms: 50,
            starting_elixir: 5.0,
            max_elixir: 10.0,
            rates: vec![(120.0, 2.8), (120.0, 1.4), (60.0, 0.93)],
            normal_s: 180.0,
            overtime_s: 120.0,
        }
    }

    #[test]
    fn regen_and_cap() {
        let tl = tl();
        let mut s = ElixirSim::new(&tl);
        s.advance(2.8);
        assert!((s.elixir - 6.0).abs() < 1e-9);
        s.advance(100.0);
        assert!((s.elixir - 10.0).abs() < 1e-9);
        assert!(s.leaked > 0.0);
    }

    #[test]
    fn crosses_double_elixir_boundary() {
        let tl = tl();
        let mut s = ElixirSim::new(&tl);
        s.elixir = 0.0;
        s.t_s = 119.0;
        s.advance(121.0); // 1 s at 1/2.8 + 1 s at 1/1.4
        assert!((s.elixir - (1.0 / 2.8 + 1.0 / 1.4)).abs() < 1e-9);
    }

    #[test]
    fn spend_counts_violations() {
        let tl = tl();
        let mut s = ElixirSim::new(&tl);
        assert!(s.spend(4.0));
        assert!(!s.spend(4.0));
        assert_eq!(s.violations, 1);
        assert_eq!(s.elixir, 0.0);
    }
}
