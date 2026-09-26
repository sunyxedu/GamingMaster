use super::*;
use crate::frame::{encode_frame, Frame};
use crate::invariance::{check_game, check_privacy, random_trajectory};
use crate::trajectory::replay;
use rayon::prelude::*;

fn counts(s: &str) -> Counts {
    cards::parse_counts(s).unwrap()
}

fn ranks(s: &str) -> Vec<u8> {
    let c = counts(s);
    (0..N_RANKS as u8).flat_map(|r| std::iter::repeat_n(r, c[r as usize] as usize)).collect()
}

/// Deal events giving each seat its hand (rank strings, 17 cards each; the kitty is
/// what remains) with the face-up card at position `face_up`.
fn deal(face_up: u32, hands: [&str; 3]) -> Vec<Event> {
    let mut suit = [0u8; N_RANKS];
    let mut card = |r: u8| {
        if r >= BJ {
            52 + r - BJ
        } else {
            let s = suit[r as usize];
            suit[r as usize] += 1;
            s * 13 + r
        }
    };
    let per_seat: Vec<Vec<u8>> = hands.iter().map(|h| ranks(h).into_iter().map(&mut card).collect()).collect();
    assert!(per_seat.iter().all(|h| h.len() == 17), "17 cards each");
    let mut ev = vec![Event::Chance(face_up)];
    for k in 0..51 {
        ev.push(Event::Chance(DEAL_BASE + per_seat[k % 3][k / 3] as u32));
    }
    ev
}

fn cfg() -> DouDizhuConfig {
    DouDizhuConfig::default()
}

fn play(g: &mut DouDizhu, seat: Seat, s: &str) {
    let p = if s == "Pass" { None } else { Some(parse_play(s).unwrap()) };
    let path = match p {
        None => vec![PASS],
        Some(p) => play_path(&p),
    };
    for a in path {
        let mut legal = Vec::new();
        g.legal_actions(seat, &mut legal);
        assert!(legal.contains(&a), "{s}: {a} not legal for {seat}: {}", g.state_string());
        g.apply_action(seat, a).unwrap();
    }
}

#[test]
fn spring_by_the_landlord_with_one_play() {
    // with the kitty, the landlord holds a 10-pair chain: one play wins, no peasant plays
    let hands = ["3344556677889TTJJ", "3456789TJQKKAA22(BWJ)", "3456789TJQKKAA22(CJ)"];
    let mut ev = deal(0, hands);
    let mut g = DouDizhu::new_game(&cfg());
    for e in &ev {
        let Event::Chance(o) = *e else { unreachable!() };
        g.apply_chance(o).unwrap();
    }
    assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)), "face-up position 0: seat 0 bids first");
    assert_eq!(g.kitty, counts("9QQ"));
    assert_eq!(g.face_up_rank, Some(0));
    g.apply_action(0, 3).unwrap(); // a bid of 3 ends the auction at once
    assert_eq!(g.landlord(), Some(0));
    assert_eq!(g.hand(0), &counts("33445566778899TTJJQQ"));
    let chain = parse_play("33445566778899TTJJQQ").unwrap();
    assert_eq!(Action::Play(chain).id(), 117);
    let mut legal = Vec::new();
    g.legal_actions(0, &mut legal);
    assert!(legal.contains(&117) && !legal.contains(&PASS), "the leader cannot pass");
    g.apply_action(0, 117).unwrap();
    ev.extend([Event::Act(0, 3), Event::Act(0, 117)]);
    assert_eq!(g.turn(), Turn::Terminal);
    // spring: no peasant played (and the landlord played once); paying = 3 * 2^1
    assert_eq!(g.returns(), vec![12.0, -6.0, -6.0]);
    assert_eq!(g.outcome(), vec![0.5, -0.25, -0.25]);
    let wp = DouDizhuConfig { scoring: "wp".into(), ..cfg() };
    let r = replay::<DouDizhu>(&wp, &ev).unwrap();
    assert_eq!(r.state.outcome(), vec![1.0, -1.0, -1.0]);
    assert_eq!(r.decisions.len(), 2);
    assert_eq!(r.streams(&DouDizhu::spec(&wp)).len(), 1, "only the landlord decided");
}

#[test]
fn bomb_and_anti_spring() {
    let hands = ["45678TTJJQQKKAA22", "33334455667788999", "456789TTJJQQKKAA2"];
    let mut g = DouDizhu::new_game(&cfg());
    for e in deal(0, hands) {
        let Event::Chance(o) = e else { unreachable!() };
        g.apply_chance(o).unwrap();
    }
    assert_eq!(g.kitty, counts("2(BWJ)(CJ)"));
    // bid 1, pass, pass: two passes after a bid end the auction
    g.apply_action(0, 1).unwrap();
    g.apply_action(1, PASS).unwrap();
    assert_eq!(g.landlord(), None, "not yet decided");
    g.apply_action(2, PASS).unwrap();
    assert_eq!((g.landlord(), g.winning_bid()), (Some(0), 1));
    play(&mut g, 0, "4");
    play(&mut g, 1, "3333");
    play(&mut g, 2, "Pass");
    play(&mut g, 0, "Pass"); // could rocket; passing is always legal when following
    assert_eq!(g.turn(), Turn::Act(SeatSet::single(1)), "two passes: the bomber leads");
    assert!(g.trick().is_none());
    play(&mut g, 1, "4455667788");
    play(&mut g, 2, "Pass");
    play(&mut g, 0, "Pass");
    play(&mut g, 1, "999");
    assert_eq!(g.turn(), Turn::Terminal);
    assert_eq!(g.bombs(), 1);
    // the landlord played once: spring; paying = 1 * 2^(1 bomb + 1 spring)
    assert_eq!(g.returns(), vec![-8.0, 4.0, 4.0]);
}

#[test]
fn auction_variants() {
    let hands = ["45678TTJJQQKKAA22", "33334455667788999", "456789TTJJQQKKAA2"];
    let run = |acts: &[(Seat, u32)]| {
        let mut g = DouDizhu::new_game(&cfg());
        for e in deal(4, hands) {
            let Event::Chance(o) = e else { unreachable!() };
            g.apply_chance(o).unwrap();
        }
        for &(s, a) in acts {
            g.apply_action(s, a).unwrap();
        }
        g
    };
    // face-up position 4 → seat 1 bids first; three passes end the game
    let g = run(&[(1, PASS), (2, PASS), (0, PASS)]);
    assert_eq!(g.turn(), Turn::Terminal);
    assert_eq!((g.returns(), g.outcome()), (vec![0.0; 3], vec![0.0; 3]));
    // a seat that passed may bid later; the bid must rise
    let g = run(&[(1, PASS), (2, 1), (0, PASS), (1, 2)]);
    let mut legal = Vec::new();
    g.legal_actions(2, &mut legal);
    assert_eq!(legal, vec![PASS, 3]);
    let g = run(&[(1, PASS), (2, 1), (0, PASS), (1, 2), (2, PASS), (0, PASS)]);
    assert_eq!((g.landlord(), g.winning_bid()), (Some(1), 2));
    assert_eq!(n_cards(g.hand(1)), 20);
    // out of turn / illegal bids are rejected
    let mut g = run(&[(1, 2)]);
    assert!(g.apply_action(2, 1).is_err() && g.apply_action(1, PASS).is_err());
    assert_eq!(g.action_string(2, 3), "Bid 3");
}

#[test]
fn kicker_decisions_paths_and_strings() {
    let hands = ["3444567899TJQKAAA", "33456789TJQKA2222", "3556677889TTJJQQK"];
    let mut g = DouDizhu::new_game(&cfg());
    for e in deal(0, hands) {
        let Event::Chance(o) = e else { unreachable!() };
        g.apply_chance(o).unwrap();
    }
    g.apply_action(0, 3).unwrap();
    // trio 4s + a solo: main, then the kicker (never a 4)
    let main = Action::Main(Main { kind: Kind::TrioSolo, head: 1, len: 1 }).id();
    let mut legal = Vec::new();
    g.legal_actions(0, &mut legal);
    assert!(legal.contains(&main) && legal.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(g.action_string(0, main), "444-*");
    g.apply_action(0, main).unwrap();
    assert_eq!(g.turn(), Turn::Act(SeatSet::single(0)), "the same seat chooses the kicker");
    legal.clear();
    g.legal_actions(0, &mut legal);
    let ks: Vec<u8> = legal.iter().map(|&a| match Action::from_id(a) {
        Some(Action::KickSolo(r)) => r,
        a => panic!("{a:?}"),
    }).collect();
    assert!(!ks.contains(&1) && ks.contains(&0));
    assert_eq!(g.action_string(0, Action::KickSolo(0).id()), "3444");
    g.apply_action(0, Action::KickSolo(0).id()).unwrap();
    assert_eq!(g.trick().map(|(s, p)| (s, play_str(&p))), Some((0, "3444".to_string())));
    assert_eq!(play_events(0, &parse_play("3444").unwrap()), vec![Event::Act(0, main), Event::Act(0, Action::KickSolo(0).id())]);
    // the next seat may beat it with 2222 + a kicker? trio 2s + solo, or the bomb
    legal.clear();
    g.legal_actions(1, &mut legal);
    assert!(legal.contains(&PASS) && legal.contains(&(BOMB_BASE + TWO as u32)));
    assert!(legal.contains(&Action::Main(Main { kind: Kind::TrioSolo, head: TWO, len: 1 }).id()));
    assert!(!legal.contains(&Action::Main(Main { kind: Kind::TrioPair, head: TWO, len: 1 }).id()), "same kind only");
}

#[test]
fn openspiel_variant_differs_exactly_by_four_with_two_and_the_chain_head_rule() {
    let os = DouDizhuConfig { rules: "openspiel".into(), ..cfg() };
    // kitty A 2 (BWJ): no 8, so the only chain from the 3333 is 34567
    let hands = ["33334567TTJJQQKK2", "445678899TJQKAA22", "45566778899TJQKA(CJ)"];
    let mut a = DouDizhu::new_game(&cfg());
    let mut b = DouDizhu::new_game(&os);
    for e in deal(0, hands) {
        let Event::Chance(o) = e else { unreachable!() };
        a.apply_chance(o).unwrap();
        b.apply_chance(o).unwrap();
    }
    a.apply_action(0, 3).unwrap();
    b.apply_action(0, 3).unwrap();
    let (mut la, mut lb) = (Vec::new(), Vec::new());
    a.legal_actions(0, &mut la);
    b.legal_actions(0, &mut lb);
    let extra: Vec<String> = la.iter().filter(|x| !lb.contains(x)).map(|&x| a.action_string(0, x)).collect();
    assert!(lb.iter().all(|x| la.contains(x)));
    assert_eq!(extra, ["34567", "3333-**", "3333-****"]);
}

/// Every legal play of random positions is reachable through the decisions, and a
/// recorded play string maps to its decision path (the converter's direction).
#[test]
fn recorded_play_strings_replay() {
    let c = cfg();
    let mut rng = Rng::new(3);
    for _ in 0..40 {
        let ev = random_trajectory::<DouDizhu>(&c, &mut rng, 100_000);
        // rebuild the trajectory from full-play strings
        let mut g = DouDizhu::new_game(&c);
        let mut rebuilt = Vec::new();
        let mut i = 0;
        while i < ev.len() {
            match ev[i] {
                Event::Chance(o) => {
                    g.apply_chance(o).unwrap();
                    rebuilt.push(ev[i]);
                    i += 1;
                }
                Event::Act(s, a) => {
                    if g.phase == Phase::Auction || a == PASS {
                        g.apply_action(s, a).unwrap();
                        rebuilt.push(ev[i]);
                        i += 1;
                        continue;
                    }
                    // one full play: follow the path, read the completed string
                    let mut path = Vec::new();
                    let mut s_str;
                    loop {
                        let Event::Act(_, a) = ev[i] else { unreachable!() };
                        s_str = g.action_string(s, a);
                        g.apply_action(s, a).unwrap();
                        path.push(a);
                        i += 1;
                        if !s_str.contains('*') {
                            break;
                        }
                    }
                    let p = parse_play(&s_str).unwrap();
                    assert_eq!(play_path(&p), path, "{s_str}");
                    rebuilt.extend(play_events(s, &p));
                }
            }
        }
        assert_eq!(rebuilt, ev);
    }
}

#[test]
fn conformance_battery() {
    for c in [
        cfg(),
        DouDizhuConfig { rules: "openspiel".into(), ..cfg() },
        DouDizhuConfig { scoring: "wp".into(), ..cfg() },
    ] {
        let rep = check_game::<DouDizhu>(&c, 120, 17).unwrap();
        assert_eq!(rep.terminal, 120);
        assert!(rep.privacy_checks > 200, "{rep:?}");
        assert!(rep.max_legal <= encode::MAX_ACTIONS, "{rep:?}");
    }
    assert!(DouDizhu::check_config(&DouDizhuConfig { rules: "x".into(), ..cfg() }).is_err());
    assert!(DouDizhu::check_config(&DouDizhuConfig { score_scale: 0.0, ..cfg() }).is_err());
}

/// The privacy battery catches an encoder that shows another seat's hand, and the
/// perturbation keeps what the seat saw (own cards, face-up card, a shown kitty).
#[test]
fn privacy_perturbation_and_a_leak() {
    let c = cfg();
    let mut rng = Rng::new(5);
    let ev = random_trajectory::<DouDizhu>(&c, &mut rng, 100_000);
    let r = replay::<DouDizhu>(&c, &ev).unwrap();
    let spec = DouDizhu::spec(&c);
    for seat in 0..3 {
        let p = DouDizhu::perturb_hidden(&c, &ev, seat, &mut rng).expect("a perturbation");
        check_privacy::<DouDizhu>(&c, &ev, &p, seat).unwrap();
        let rp = replay::<DouDizhu>(&c, &p).unwrap();
        assert_eq!(rp.state.hand(seat), r.state.hand(seat));
        assert_eq!(rp.state.kitty_shown, r.state.kitty_shown);
        let _ = &spec;
    }
    // a leak: seat 0's frame includes the next seat's hand → the battery fails
    #[derive(Clone, Debug)]
    struct Leaky(DouDizhu);
    impl Game for Leaky {
        type Config = DouDizhuConfig;
        const NAME: &'static str = "leaky_doudizhu";
        const VERSION: u32 = 0;
        fn spec(c: &DouDizhuConfig) -> FrameSpec { DouDizhu::spec(c) }
        fn new_game(c: &DouDizhuConfig) -> Self { Leaky(DouDizhu::new_game(c)) }
        fn turn(&self) -> Turn { self.0.turn() }
        fn chance_outcomes(&self, o: &mut Vec<(u32, f64)>) { self.0.chance_outcomes(o) }
        fn apply_chance(&mut self, o: u32) -> Result<(), RulesError> { self.0.apply_chance(o) }
        fn legal_actions(&self, s: Seat, o: &mut Vec<u32>) { self.0.legal_actions(s, o) }
        fn apply_action(&mut self, s: Seat, a: u32) -> Result<(), RulesError> { self.0.apply_action(s, a) }
        fn outcome(&self) -> Vec<f32> { self.0.outcome() }
        fn returns(&self) -> Vec<f64> { self.0.returns() }
        fn encode_obs(&self, s: Seat, obs: &mut ObsView<'_>) {
            self.0.encode_obs(s, obs);
            let h = self.0.hands[(s + 1) % 3];
            obs.globals()[23] = (h[0] >= 2) as u8 as f32; // the next seat holds a pair of 3s
        }
        fn encode_action(&self, s: Seat, a: u32, f: &mut [f32]) { self.0.encode_action(s, a, f) }
        fn state_string(&self) -> String { self.0.state_string() }
        fn perturb_hidden(c: &DouDizhuConfig, e: &[Event], s: Seat, r: &mut Rng) -> Option<Vec<Event>> {
            DouDizhu::perturb_hidden(c, e, s, r)
        }
    }
    assert!(check_game::<Leaky>(&c, 40, 9).unwrap_err().contains("hidden from it"));
}

#[test]
fn frames_have_one_candidate_type_and_one_phase() {
    let c = cfg();
    let spec = DouDizhu::spec(&c);
    spec.validate().unwrap();
    let mut rng = Rng::new(11);
    let mut kinds = [0usize; 5];
    let mut f = Frame::new(&spec);
    for _ in 0..20 {
        let ev = random_trajectory::<DouDizhu>(&c, &mut rng, 100_000);
        let mut g = DouDizhu::new_game(&c);
        for e in &ev {
            if let Event::Act(s, _) = *e {
                encode_frame(&g, s, &spec, &mut f.as_mut(&spec)).unwrap();
                let acts = f.obs[3].as_f32().unwrap();
                for i in 0..f.legal_ids.len() {
                    let t = &acts[i * encode::ACTION_FEATURES..i * encode::ACTION_FEATURES + 5];
                    assert_eq!(t.iter().sum::<f32>(), 1.0, "one candidate type");
                    kinds[t.iter().position(|&x| x == 1.0).unwrap()] += 1;
                }
                let gl = f.obs[2].as_f32().unwrap();
                assert_eq!(gl[..3].iter().sum::<f32>(), 1.0, "one phase");
            }
            match *e {
                Event::Chance(o) => g.apply_chance(o).unwrap(),
                Event::Act(s, a) => g.apply_action(s, a).unwrap(),
            }
        }
    }
    assert!(kinds.iter().all(|&k| k > 0), "{kinds:?}");
}

/// Number of main-level options of a hand when leading (the candidate count bound).
fn n_main_options(h: &Counts, buf: &mut Vec<u32>) -> usize {
    buf.clear();
    main_options(h, None, Variant::Standard, buf);
    buf.len()
}

/// The hand with the most main-level options found by the exhaustive search below.
const ARGMAX_HAND: &str = "33344455566677788899";

#[test]
fn candidate_bound_is_attained() {
    let h = counts(ARGMAX_HAND);
    let mut buf = Vec::new();
    let n = n_main_options(&h, &mut buf);
    assert_eq!(n + 1, encode::MAX_ACTIONS, "{n}");
    // random 20-card hands stay below
    let mut rng = Rng::new(1);
    for _ in 0..20000 {
        let mut deck: Vec<u8> = (0..54).collect();
        rng.shuffle(&mut deck);
        let mut h = [0u8; N_RANKS];
        for &c in &deck[..20] {
            h[card_rank(c) as usize] += 1;
        }
        assert!(n_main_options(&h, &mut buf) < encode::MAX_ACTIONS);
    }
}

/// Kicker completions of `main` from `hand` (multisets: solos with per-rank caps and
/// at most one joker, distinct pairs).
fn n_completions(hand: &Counts, main: &Main) -> u64 {
    let m = main.n_kickers() as usize;
    let covers = |r: u8| (main.head..main.head + main.len).contains(&r);
    match main.unit() {
        Unit::Solo => {
            let cap = match main.kind {
                Kind::AirplaneSolo => 3,
                Kind::FourSolo => 2,
                _ => 1,
            };
            let mut poly = vec![0u64; m + 1];
            poly[0] = 1;
            for r in 0..BJ {
                let c = if covers(r) { 0 } else { hand[r as usize].min(cap) as usize };
                let mut next = vec![0u64; m + 1];
                for (i, &p) in poly.iter().enumerate() {
                    for k in 0..=c.min(m - i) {
                        next[i + k] += p;
                    }
                }
                poly = next;
            }
            let jokers = (hand[BJ as usize] + hand[RJ as usize]) as u64;
            poly[m] + if m >= 1 { jokers * poly[m - 1] } else { 0 }
        }
        Unit::Pair => {
            let e = (0..BJ).filter(|&r| !covers(r) && hand[r as usize] >= 2).count() as u64;
            (0..m as u64).fold(1u64, |acc, i| acc * (e - i.min(e)) / (i + 1))
        }
    }
}

/// Number of full plays (kickers expanded) of a hand when leading.
fn n_full_plays(h: &Counts, buf: &mut Vec<u32>) -> u64 {
    buf.clear();
    main_options(h, None, Variant::Standard, buf);
    buf.iter()
        .map(|&id| match Action::from_id(id) {
            Some(Action::Main(m)) => n_completions(h, &m),
            _ => 1,
        })
        .sum()
}

#[test]
fn completion_counts_match_enumeration() {
    let mut rng = Rng::new(2);
    let mut buf = Vec::new();
    for _ in 0..300 {
        let mut deck: Vec<u8> = (0..54).collect();
        rng.shuffle(&mut deck);
        let mut h = [0u8; N_RANKS];
        for &c in &deck[..20] {
            h[card_rank(c) as usize] += 1;
        }
        assert_eq!(n_full_plays(&h, &mut buf), legal_plays(&h, None, Variant::Standard).len() as u64);
    }
}

/// Exhaustive: the most main-level options (and full plays) over EVERY 20-card hand
/// (153,009,740 rank vectors; options only grow with the hand, and a follower's
/// options are a subset of a leader's + pass, so this bounds every frame).
/// `cargo test --release -p gmrules max_main_options_exhaustive -- --ignored --nocapture`
#[test]
#[ignore]
fn max_main_options_exhaustive() {
    #[derive(Clone, Copy)]
    struct Best {
        mains: usize,
        mains_hand: Counts,
        plays: u64,
        plays_hand: Counts,
        hands: u64,
    }
    fn rec(h: &mut Counts, r: usize, left: u8, best: &mut Best, buf: &mut Vec<u32>) {
        if r == N_RANKS {
            if left == 0 {
                best.hands += 1;
                let n = n_main_options(h, buf);
                if n > best.mains {
                    (best.mains, best.mains_hand) = (n, *h);
                }
                let p = n_full_plays(h, buf);
                if p > best.plays {
                    (best.plays, best.plays_hand) = (p, *h);
                }
            }
            return;
        }
        let cap = if r >= BJ as usize { 1 } else { 4 };
        let rest_cap: u8 = (r + 1..N_RANKS).map(|x| if x >= BJ as usize { 1 } else { 4 }).sum();
        for c in 0..=cap.min(left) {
            if left - c > rest_cap {
                continue;
            }
            h[r] = c;
            rec(h, r + 1, left - c, best, buf);
        }
        h[r] = 0;
    }
    let prefixes: Vec<[u8; 3]> = (0..125).map(|i| [(i / 25) as u8, (i / 5 % 5) as u8, (i % 5) as u8]).collect();
    let results: Vec<Best> = prefixes
        .par_iter()
        .map(|p| {
            let mut h = [0u8; N_RANKS];
            h[..3].copy_from_slice(p);
            let used: u8 = p.iter().sum();
            let mut best = Best { mains: 0, mains_hand: h, plays: 0, plays_hand: h, hands: 0 };
            let mut buf = Vec::new();
            rec(&mut h, 3, MAX_HAND - used, &mut best, &mut buf);
            best
        })
        .collect();
    let hands: u64 = results.iter().map(|r| r.hands).sum();
    let m = results.iter().max_by_key(|r| r.mains).unwrap();
    let p = results.iter().max_by_key(|r| r.plays).unwrap();
    eprintln!("{hands} hands; max main options {} for {}; max full plays {} for {}", m.mains,
              counts_str(&m.mains_hand), p.plays, counts_str(&p.plays_hand));
    assert_eq!(hands, 153_009_740);
    assert_eq!(m.mains + 1, encode::MAX_ACTIONS);
}
