//! Replay-metadata filter (the API's `replaysForUsername` entries).

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Filter {
    pub min_turns: i64,
    pub game_types: Vec<String>,
    pub max_players: usize,
    /// ISO `YYYY-MM-DDTHH:MM:SSZ`, compared against `started` (unix ms or ISO string)
    pub start_date: Option<String>,
    pub player_name: Option<String>,
    pub min_opponent_stars: Option<i64>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            min_turns: 80,
            game_types: vec!["1v1".into(), "custom".into()],
            max_players: 2,
            start_date: None,
            player_name: None,
            min_opponent_stars: None,
        }
    }
}

/// Unix seconds for an ISO `YYYY-MM-DDTHH:MM:SSZ` timestamp (UTC, no leap seconds).
pub fn iso_to_unix(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |a: usize, e: usize| s.get(a..e)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (hh, mm, ss) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    // days from civil (Howard Hinnant)
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
    let yoe = y2 - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86400 + hh * 3600 + mm * 60 + ss)
}

pub fn passes(entry: &Value, f: &Filter) -> bool {
    let gtype = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let ranking = entry.get("ranking").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let n_players = ranking.len();
    let valid_type = f.game_types.iter().any(|t| t == gtype) && (gtype == "1v1" || (2..=f.max_players).contains(&n_players));
    let has_turns = entry.get("turns").and_then(|v| v.as_i64()).unwrap_or(0) >= f.min_turns;
    let passes_date = match &f.start_date {
        None => true,
        Some(sd) => {
            let filter_ts = iso_to_unix(sd);
            let started = entry.get("started");
            let entry_ts = match started {
                Some(Value::Number(n)) => n.as_f64().map(|ms| (ms / 1000.0) as i64),
                Some(Value::String(s)) => iso_to_unix(s),
                _ => None,
            };
            matches!((entry_ts, filter_ts), (Some(e), Some(fl)) if e >= fl)
        }
    };
    let passes_stars = match (&f.min_opponent_stars, &f.player_name) {
        (Some(min), Some(me)) => {
            let opp: Vec<i64> = ranking
                .iter()
                .filter(|p| p.get("name").and_then(|n| n.as_str()) != Some(me.as_str()))
                .map(|p| p.get("stars").and_then(|s| s.as_i64()).unwrap_or(0))
                .collect();
            match opp.iter().max() {
                Some(m) => m >= min,
                None => false,
            }
        }
        _ => true,
    };
    valid_type && has_turns && passes_date && passes_stars
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn iso_conversion() {
        assert_eq!(iso_to_unix("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(iso_to_unix("2025-04-01T00:00:00Z"), Some(1743465600));
    }

    #[test]
    fn filter_rules() {
        let f = Filter { start_date: Some("2025-04-01T00:00:00Z".into()), ..Default::default() };
        let ok = json!({"type": "1v1", "turns": 100, "started": 1787583202827i64, "ranking": [{"name": "a", "stars": 5}, {"name": "b"}]});
        assert!(passes(&ok, &f));
        let short = json!({"type": "1v1", "turns": 10, "started": 1787583202827i64, "ranking": []});
        assert!(!passes(&short, &f));
        let ffa = json!({"type": "ffa", "turns": 100, "started": 1787583202827i64, "ranking": [{}, {}]});
        assert!(!passes(&ffa, &f));
        let solo_custom = json!({"type": "custom", "turns": 100, "started": 1787583202827i64, "ranking": [{"name": "a"}]});
        assert!(!passes(&solo_custom, &f));
        let old = json!({"type": "1v1", "turns": 100, "started": 1600000000000i64, "ranking": []});
        assert!(!passes(&old, &f));
        let stars = Filter {
            player_name: Some("me".into()),
            min_opponent_stars: Some(30),
            ..Default::default()
        };
        let weak = json!({"type": "1v1", "turns": 100, "ranking": [{"name": "me", "stars": 50}, {"name": "x", "stars": 10}]});
        assert!(!passes(&weak, &stars));
        let strong = json!({"type": "1v1", "turns": 100, "ranking": [{"name": "me", "stars": 50}, {"name": "x", "stars": 40}]});
        assert!(passes(&strong, &stars));
    }
}
