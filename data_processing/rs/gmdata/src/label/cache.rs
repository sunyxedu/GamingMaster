//! Per-worker position cache: repeated positions (openings above all) are searched
//! once per worker. Two generations of at most `cap / 2` entries each: when the hot
//! map fills it becomes the cold one and the old cold map is dropped; a cold hit is
//! promoted. Recently used positions survive, memory stays bounded, no bookkeeping
//! per access. The key must be the full identity the engine's answer depends on
//! (board, side to move, castling / ko / hands, rule set, and whatever history the
//! query sends); the search limits are fixed for a run and need not be part of it.

use std::collections::HashMap;
use std::hash::Hash;

pub struct PositionCache<K, V> {
    half: usize,
    hot: HashMap<K, V>,
    cold: HashMap<K, V>,
    pub hits: u64,
    pub misses: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> PositionCache<K, V> {
    /// `cap` = maximum entries kept (0 disables the cache).
    pub fn new(cap: usize) -> Self {
        PositionCache { half: cap / 2, hot: HashMap::new(), cold: HashMap::new(), hits: 0, misses: 0 }
    }

    pub fn get(&mut self, k: &K) -> Option<V> {
        if self.half == 0 {
            self.misses += 1;
            return None;
        }
        if let Some(v) = self.hot.get(k) {
            self.hits += 1;
            return Some(v.clone());
        }
        if let Some(v) = self.cold.remove(k) {
            self.hits += 1;
            self.insert(k.clone(), v.clone());
            return Some(v);
        }
        self.misses += 1;
        None
    }

    pub fn insert(&mut self, k: K, v: V) {
        if self.half == 0 {
            return;
        }
        if self.hot.len() >= self.half && !self.hot.contains_key(&k) {
            self.cold = std::mem::take(&mut self.hot);
        }
        self.hot.insert(k, v);
    }

    pub fn len(&self) -> usize {
        self.hot.len() + self.cold.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_with_promotion() {
        let mut c = PositionCache::new(4);
        for k in 0..2 {
            c.insert(k, k * 10);
        }
        assert_eq!(c.get(&1), Some(10));
        c.insert(2, 20); // hot full → {0,1} become cold
        assert_eq!(c.get(&0), Some(0)); // cold hit, promoted into hot {2, 0}
        c.insert(3, 30); // hot {2,0} → cold; hot {3}; 1 is gone
        assert_eq!(c.get(&1), None);
        assert_eq!(c.get(&0), Some(0));
        assert!(c.len() <= 4);
        assert_eq!((c.hits, c.misses), (3, 1));
        let mut off: PositionCache<i32, i32> = PositionCache::new(0);
        off.insert(1, 1);
        assert_eq!((off.get(&1), off.is_empty()), (None, true));
    }
}
