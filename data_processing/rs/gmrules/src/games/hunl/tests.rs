use super::encode::{Candidates, N_CANDIDATES, SLOT_ALL_IN, SLOT_CALL, SLOT_FOLD, SLOT_FRAC0, SLOT_MIN_RAISE};
use super::eval::{card_str, parse_cards};
use super::*;
use crate::frame::{action_keys_of, decode_action, encode_frame, encode_target, new_target, Frame};
use crate::invariance::{check_causality, check_game, check_privacy, check_privacy_random};
use crate::trajectory::replay;

const CFG: HunlConfig = HunlConfig { small_blind: 50, big_blind: 100, stack: 20000 };

fn deal(cards: &str) -> Vec<Event> {
    parse_cards(cards).unwrap().into_iter().map(|c| Event::Chance(c as u32)).collect()
}

fn legal(g: &Hunl, seat: Seat) -> Vec<u32> {
    let mut v = Vec::new();
    g.legal_actions(seat, &mut v);
    v
}

fn play(cfg: &HunlConfig, events: &[Event]) -> Hunl {
    let mut g = Hunl::new_game(cfg);
    for e in events {
        match *e {
            Event::Chance(c) => g.apply_chance(c).unwrap(),
            Event::Act(s, a) => g.apply_action(s, a).unwrap(),
        }
    }
    g
}

#[test]
fn acpc_betting_rules() {
    let mut ev = deal("AcKdQhJs");
    let g = play(&CFG, &ev);
    // seat 1 (small blind, button) acts first preflop: fold, call, raise to 200..=20000
    assert_eq!(g.turn(), Turn::Act(SeatSet::single(1)));
    let l = legal(&g, 1);
    assert_eq!((l[0], l[1], l[2], *l.last().unwrap(), l.len()), (FOLD, CALL, 200, 20000, 2 + 19801));
    assert_eq!(g.hole(0).len(), 2);
    // the small blind folds: the big blind wins 50
    let mut f = ev.clone();
    f.push(Event::Act(1, FOLD));
    let r = replay::<Hunl>(&CFG, &f).unwrap();
    assert!(r.terminal());
    assert_eq!(r.state.returns(), vec![50.0, -50.0]);
    assert_eq!(r.state.outcome(), vec![0.0025, -0.0025]);
    // limp, check: the flop; seat 0 acts first and cannot fold; min bet = one big blind
    ev.extend([Event::Act(1, CALL), Event::Act(0, CALL)]);
    let g = play(&CFG, &ev);
    assert_eq!(g.turn(), Turn::Chance);
    ev.extend(deal("2c7d9h"));
    let g = play(&CFG, &ev);
    assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)));
    let l = legal(&g, 0);
    assert_eq!((l[0], l[1], l.len()), (CALL, 200, 1 + 19801));
    // bet 300 (to 400 total), raise to 1000 (min raise-to was 700), min re-raise 1600
    ev.push(Event::Act(0, 400));
    let g = play(&CFG, &ev);
    assert_eq!(g.raise_range(1), Some((700, 20000)));
    assert!(g.fold_legal(1));
    ev.push(Event::Act(1, 1000));
    let g = play(&CFG, &ev);
    assert_eq!(g.raise_range(0), Some((1600, 20000)));
    // all-in, called: the rest of the board is dealt, showdown
    ev.extend([Event::Act(0, 20000), Event::Act(1, CALL)]);
    let g = play(&CFG, &ev);
    assert_eq!(g.turn(), Turn::Chance);
    ev.extend(deal("2d3d"));
    let g = play(&CFG, &ev);
    assert_eq!(g.turn(), Turn::Terminal);
    // seat 0 AcKd vs seat 1 QhJs on 2c7d9h2d3d: ace high wins
    assert_eq!(g.returns(), vec![20000.0, -20000.0]);
    assert_eq!(g.outcome(), vec![1.0, -1.0]);
    // facing an all-in: fold / call only; a re-raise is illegal, a raise below the minimum too
    let mut ev2 = deal("AcKdQhJs");
    ev2.push(Event::Act(1, 20000));
    let g = play(&CFG, &ev2);
    assert_eq!(legal(&g, 0), vec![FOLD, CALL]);
    let mut bad = deal("AcKdQhJs");
    bad.push(Event::Act(1, 199));
    assert!(replay::<Hunl>(&CFG, &bad).is_err());
    bad[4] = Event::Act(0, CALL); // out of turn
    assert!(replay::<Hunl>(&CFG, &bad).is_err());
    let mut dup = deal("AcKdQhJs");
    dup[1] = Event::Chance(parse_cards("Ac").unwrap()[0] as u32);
    assert!(replay::<Hunl>(&CFG, &dup).is_err());
}

#[test]
fn split_pot_and_short_stack_all_in() {
    // both play the board (a broadway straight): split
    let mut ev = deal("2c3d2h3s");
    ev.extend([Event::Act(1, CALL), Event::Act(0, CALL)]);
    ev.extend(deal("TcJdQh"));
    ev.extend([Event::Act(0, CALL), Event::Act(1, CALL)]);
    ev.extend(deal("Ks"));
    ev.extend([Event::Act(0, CALL), Event::Act(1, CALL)]);
    ev.extend(deal("Ad"));
    ev.extend([Event::Act(0, CALL), Event::Act(1, CALL)]);
    let r = replay::<Hunl>(&CFG, &ev).unwrap();
    assert!(r.terminal());
    assert_eq!(r.state.returns(), vec![0.0, 0.0]);
    // min raise above the stack: the only raise is all-in (1/2 blinds, 7-chip stacks)
    let cfg = HunlConfig { small_blind: 1, big_blind: 2, stack: 7 };
    let mut ev = deal("AcKdQhJs");
    ev.push(Event::Act(1, 5)); // raise to 5: min re-raise 8 > 7
    let g = play(&cfg, &ev);
    assert_eq!(g.raise_range(0), Some((7, 7)));
    assert_eq!(legal(&g, 0), vec![FOLD, CALL, 7]);
    let c = Candidates::of(&g, 0);
    assert!(c.avail[SLOT_ALL_IN] && !c.avail[SLOT_MIN_RAISE] && !c.avail[SLOT_FRAC0..].iter().any(|&x| x));
}

#[test]
fn round_action_cap_forces_call_or_fold() {
    // a min-raise war: raising stays legal for the first 63 actions of a round
    let mut g = play(&CFG, &deal("AcKdQhJs"));
    let mut n = 0;
    while let Some((lo, _)) = g.raise_range(g.current_player()) {
        let s = g.current_player();
        g.apply_action(s, lo).unwrap();
        n += 1;
    }
    assert_eq!(n, MAX_ROUND_ACTIONS - 1);
    let s = g.current_player();
    assert_eq!(legal(&g, s), vec![FOLD, CALL]);
    g.apply_action(s, CALL).unwrap();
    assert_eq!(g.round(), 1);
}

#[test]
fn candidates_abstraction() {
    let g = play(&CFG, &deal("AcKdQhJs"));
    let c = Candidates::of(&g, 1);
    // fold, call, all-in, min-raise 200; fractions of the 200 pot after calling
    // (100 + f * 200) strictly between 200 and 20000
    assert_eq!(c.action[SLOT_ALL_IN], 20000);
    assert_eq!(c.action[SLOT_MIN_RAISE], 200);
    let fr: Vec<Option<u32>> = (SLOT_FRAC0..N_CANDIDATES).map(|k| c.avail[k].then(|| c.action[k])).collect();
    assert_eq!(fr, vec![None, None, None, Some(233), Some(250), Some(300), Some(350), Some(400), Some(500), Some(700), Some(1100)]);
    // nearest in log(increment): 275 → increment 175, between 150 (to 250) and 200 (to 300)
    assert_eq!(c.action[c.nearest(275)], 300);
    assert_eq!(c.action[c.nearest(260)], 250);
    assert_eq!(c.nearest(200), SLOT_MIN_RAISE);
    assert_eq!(c.nearest(20000), SLOT_ALL_IN);
    // (log scale: an increment of 2900 is nearer 1000 than 19900, one of 4900 is not)
    assert_eq!(c.action[c.nearest(3000)], 1100);
    assert_eq!(c.action[c.nearest(5000)], 20000);
    assert_eq!((c.nearest(FOLD), c.nearest(CALL)), (SLOT_FOLD, SLOT_CALL));
    assert_eq!(c.decode(3), Some(200));
    assert_eq!(c.decode(4), None);
    assert_eq!(c.decode(99), None);
    assert_eq!(c.decode(-1), None);
    // every candidate's amount maps back to its own slot; every legal amount maps to an
    // available slot
    for k in 0..N_CANDIDATES {
        if let Some(a) = c.decode(k as i64) {
            assert_eq!(c.nearest(a), k);
        }
    }
    for a in legal(&g, 1) {
        assert!(c.avail[c.nearest(a)]);
    }
}

#[test]
fn frames_and_targets() {
    let spec = Hunl::spec(&CFG);
    spec.validate().unwrap();
    let mut ev = deal("AcKdQhJs");
    ev.push(Event::Act(1, 275));
    let g = play(&CFG, &ev[..4]);
    let mut f = Frame::new(&spec);
    encode_frame(&g, 1, &spec, &mut f.as_mut(&spec)).unwrap();
    let mask = f.obs[5].as_bool().unwrap();
    assert_eq!(mask.iter().filter(|&&m| m == 1).count(), 2 + 1 + 1 + 8);
    assert_eq!(f.legal[0].as_bool().unwrap(), mask);
    // own hole cards (sorted: Qh then Js), no board, the two players
    let em = f.obs[1].as_bool().unwrap();
    assert_eq!(em, &[1, 1, 0, 0, 0, 0, 0, 1, 1]);
    let ents = f.obs[0].as_f32().unwrap();
    assert_eq!(ents[10], 1.0, "Q rank one-hot");
    assert_eq!(ents[18 + 9], 1.0, "J rank one-hot");
    assert_eq!(f.obs[2].as_i32().unwrap(), &[1, 1, 0, 0, 0, 0, 0, 3, 4]);
    // target of the real raise to 275: candidate "to 300", exact amount kept
    let mut t = new_target(&spec);
    encode_target(&g, 1, &spec, &f.legal_ids, 275, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    let idx = t[0].as_i64().unwrap()[0];
    assert_eq!(Candidates::of(&g, 1).action[idx as usize], 300);
    assert_eq!(t[1].as_i64().unwrap()[0], 275);
    assert!((t[2].as_f32().unwrap()[0] - 0.875).abs() < 1e-6);
    assert_eq!(decode_action(&g, 1, &spec, &f.legal_ids, &[idx]).unwrap(), 300);
    assert!(decode_action(&g, 1, &spec, &f.legal_ids, &[4]).is_err());
    // calls carry no amount
    encode_target(&g, 1, &spec, &f.legal_ids, CALL, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
    assert_eq!(action_keys_of(&spec, &t), vec![SLOT_CALL as i64]);
    assert_eq!((t[1].as_i64().unwrap()[0], t[2].as_f32().unwrap()[0]), (0, 0.0));
    // the replayed record: seat 1's frame, target = the nearest candidate
    let r = replay::<Hunl>(&CFG, &ev).unwrap();
    assert_eq!(r.decisions.len(), 1);
    assert_eq!(r.decisions[0].target[1].as_i64().unwrap()[0], 275);
    assert!(r.decisions[0].frame.check_equal(&f, &spec).is_ok());
}

#[test]
fn conformance_battery() {
    // uniform-random ids (mostly raise wars) through the standard battery
    for (cfg, n) in [(CFG, 60), (HunlConfig { small_blind: 1, big_blind: 2, stack: 400 }, 120), (HunlConfig { small_blind: 1, big_blind: 2, stack: 12 }, 200)] {
        let rep = check_game::<Hunl>(&cfg, n, 5).unwrap();
        assert_eq!(rep.terminal, n);
        assert!(rep.privacy_checks >= n, "{rep:?}");
    }
}

/// Causality, privacy and the action-key round trip on realistic random hands
/// (folds, checks, calls, every raise size; every street).
#[test]
fn invariance_on_structured_random_hands() {
    let spec = Hunl::spec(&CFG);
    let mut rng = Rng::new(11);
    let (mut decisions, mut showdowns, mut privacy, mut streets) = (0, 0, 0, [0usize; 4]);
    for i in 0..400 {
        let cfg = if i % 4 == 3 { HunlConfig { small_blind: 1, big_blind: 2, stack: 400 } } else { CFG };
        let ev = random_hand(&cfg, &mut rng, 0.15, 0.3);
        let r = replay::<Hunl>(&cfg, &ev).unwrap();
        assert!(r.terminal());
        let out = r.state.outcome();
        assert!(out.iter().all(|x| (-1.0..=1.0).contains(x)) && (out[0] + out[1]).abs() < 1e-6);
        showdowns += !r.state.folded().iter().any(|&f| f) as usize;
        check_causality::<Hunl>(&cfg, &ev).unwrap();
        privacy += check_privacy_random::<Hunl>(&cfg, &ev, &mut rng, 2).unwrap();
        // every legal action's candidate decodes to a legal action with the same candidate
        let mut g = Hunl::new_game(&cfg);
        let mut next = 0;
        for (j, e) in ev.iter().enumerate() {
            if next < r.decisions.len() && r.decisions[next].event == j {
                let d = &r.decisions[next];
                streets[g.round()] += 1;
                let mut t = new_target(&spec);
                // fold / call, the raise bounds and a random sample of raise amounts
                let ids = &d.frame.legal_ids;
                let sample: Vec<u32> = if ids.len() <= 400 {
                    ids.clone()
                } else {
                    let mut s: Vec<u32> = ids[..100].iter().chain(&ids[ids.len() - 100..]).copied().collect();
                    s.extend((0..200).map(|_| ids[rng.below(ids.len() as u64) as usize]));
                    s
                };
                for &a in &sample {
                    encode_target(&g, d.seat, &spec, &d.frame.legal_ids, a, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
                    let keys = action_keys_of(&spec, &t);
                    let back = decode_action(&g, d.seat, &spec, &d.frame.legal_ids, &keys).unwrap();
                    encode_target(&g, d.seat, &spec, &d.frame.legal_ids, back, &mut TensorsMut::of(&spec.target, &mut t)).unwrap();
                    assert_eq!(action_keys_of(&spec, &t), keys);
                }
                next += 1;
                decisions += 1;
            }
            match *e {
                Event::Chance(c) => g.apply_chance(c).unwrap(),
                Event::Act(s, a) => g.apply_action(s, a).unwrap(),
            }
        }
    }
    assert!(decisions > 1500 && showdowns > 40 && privacy > 700, "{decisions} {showdowns} {privacy}");
    assert!(streets.iter().all(|&n| n > 100), "{streets:?}");
}

/// The privacy battery catches an encoder that reads the opponent's cards.
#[test]
fn privacy_check_catches_a_leak() {
    #[derive(Clone, Debug)]
    struct Leaky(Hunl);
    impl Game for Leaky {
        type Config = HunlConfig;
        const NAME: &'static str = "leaky_hunl";
        const VERSION: u32 = 0;
        fn spec(c: &HunlConfig) -> FrameSpec { Hunl::spec(c) }
        fn new_game(c: &HunlConfig) -> Self { Leaky(Hunl::new_game(c)) }
        fn turn(&self) -> Turn { self.0.turn() }
        fn chance_outcomes(&self, o: &mut Vec<(u32, f64)>) { self.0.chance_outcomes(o) }
        fn apply_chance(&mut self, o: u32) -> Result<(), RulesError> { self.0.apply_chance(o) }
        fn legal_actions(&self, s: Seat, o: &mut Vec<u32>) { self.0.legal_actions(s, o) }
        fn apply_action(&mut self, s: Seat, a: u32) -> Result<(), RulesError> { self.0.apply_action(s, a) }
        fn outcome(&self) -> Vec<f32> { self.0.outcome() }
        fn returns(&self) -> Vec<f64> { self.0.returns() }
        fn encode_custom(&self, s: Seat, f: &mut FrameMut<'_>) {
            self.0.encode_custom(s, f);
            f.obs.f32(3)[47] = self.0.hole(1 - s)[0] as f32; // the opponent's first card
        }
        fn target_custom(&self, s: Seat, l: &[u32], a: u32, t: &mut TensorsMut<'_>) { self.0.target_custom(s, l, a, t) }
        fn decode_custom(&self, s: Seat, l: &[u32], k: &[i64]) -> Option<u32> { self.0.decode_custom(s, l, k) }
        fn state_string(&self) -> String { self.0.state_string() }
        fn perturb_hidden(c: &HunlConfig, e: &[Event], s: Seat, r: &mut Rng) -> Option<Vec<Event>> {
            Hunl::perturb_hidden(c, e, s, r)
        }
    }
    let mut a = deal("AcKdQhJs");
    a.extend([Event::Act(1, CALL), Event::Act(0, CALL)]);
    let mut b = a.clone();
    b[2] = Event::Chance(parse_cards("2s").unwrap()[0] as u32);
    assert!(check_privacy::<Hunl>(&CFG, &a, &b, 0).is_ok());
    let err = check_privacy::<Leaky>(&CFG, &a, &b, 0).unwrap_err();
    assert!(err.contains("hidden from it"), "{err}");
    // the perturbation re-deals only the other seat's cards and avoids every board card
    let mut rng = Rng::new(3);
    let mut full = a.clone();
    full.extend(deal("2c7d9h"));
    for _ in 0..50 {
        let p = Hunl::perturb_hidden(&CFG, &full, 0, &mut rng).unwrap();
        assert_eq!(p[..2], full[..2]);
        assert_eq!(p[4..], full[4..]);
        assert_ne!(p[2..4], full[2..4]);
        let cards: Vec<u32> = p.iter().filter_map(|e| if let Event::Chance(c) = e { Some(*c) } else { None }).collect();
        let mut s = cards.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), cards.len(), "{}", cards.iter().map(|&c| card_str(c as u8)).collect::<String>());
    }
    assert!(Hunl::perturb_hidden(&CFG, &full[..3], 0, &mut rng).is_none());
}
