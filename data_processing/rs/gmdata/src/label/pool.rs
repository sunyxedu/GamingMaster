//! A fixed pool of long-lived worker threads, each owning its state (an engine
//! session + position cache in the labeller), fed one job at a time from a shared
//! queue. [`Pool::map`] returns results in job order, so output does not depend on the
//! number of workers or on scheduling. A panicking job does not wedge the pool: the
//! panic is re-raised by `map` after every job of the batch has finished.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

type Outcome<R> = (usize, std::thread::Result<R>);

pub struct Pool<J: Send + 'static, R: Send + 'static> {
    jobs: Option<Sender<(usize, J)>>,
    results: Receiver<Outcome<R>>,
    threads: Vec<JoinHandle<()>>,
}

impl<J: Send + 'static, R: Send + 'static> Pool<J, R> {
    /// One thread per worker; a worker is any `FnMut(J) -> R` owning its state.
    pub fn new<W: FnMut(J) -> R + Send + 'static>(workers: Vec<W>) -> Self {
        assert!(!workers.is_empty(), "a pool needs at least one worker");
        let (job_tx, job_rx) = mpsc::channel::<(usize, J)>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (res_tx, res_rx) = mpsc::channel::<Outcome<R>>();
        let threads = workers
            .into_iter()
            .enumerate()
            .map(|(i, mut w)| {
                let rx = job_rx.clone();
                let tx = res_tx.clone();
                std::thread::Builder::new()
                    .name(format!("label-worker-{i}"))
                    .spawn(move || loop {
                        let next = rx.lock().unwrap().recv();
                        let Ok((idx, job)) = next else { return };
                        let r = catch_unwind(AssertUnwindSafe(|| w(job)));
                        if tx.send((idx, r)).is_err() {
                            return;
                        }
                    })
                    .expect("spawn worker thread")
            })
            .collect();
        Pool { jobs: Some(job_tx), results: res_rx, threads }
    }

    pub fn workers(&self) -> usize {
        self.threads.len()
    }

    /// Run every job; results in job order. `on_done(finished_so_far, &result)` is
    /// called on the calling thread as results arrive (progress, health checks).
    pub fn map(&self, jobs: Vec<J>, mut on_done: impl FnMut(usize, &R)) -> Vec<R> {
        let n = jobs.len();
        let tx = self.jobs.as_ref().expect("pool is shut down");
        for (i, j) in jobs.into_iter().enumerate() {
            tx.send((i, j)).expect("worker threads are gone");
        }
        let mut out: Vec<Option<R>> = (0..n).map(|_| None).collect();
        let mut panic = None;
        for done in 1..=n {
            let (i, r) = self.results.recv().expect("worker threads are gone");
            match r {
                Ok(r) => {
                    on_done(done, &r);
                    out[i] = Some(r);
                }
                Err(p) => panic = panic.or(Some(p)),
            }
        }
        if let Some(p) = panic {
            std::panic::resume_unwind(p);
        }
        out.into_iter().map(|r| r.expect("every job returns")).collect()
    }
}

impl<J: Send + 'static, R: Send + 'static> Drop for Pool<J, R> {
    fn drop(&mut self) {
        self.jobs = None; // workers see the closed queue and exit
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn ordered_results_and_persistent_state() {
        // each worker counts the jobs it ran; results are in job order regardless
        let workers: Vec<_> = (0..4)
            .map(|w| {
                let mut ran = 0usize;
                move |x: u64| {
                    ran += 1;
                    std::thread::sleep(Duration::from_millis((x * 7919) % 5));
                    (x * x, w, ran)
                }
            })
            .collect();
        let pool = Pool::new(workers);
        assert_eq!(pool.workers(), 4);
        let mut seen = 0;
        let out = pool.map((0..50).collect(), |done, _| seen = done);
        assert_eq!(seen, 50);
        assert_eq!(out.iter().map(|r| r.0).collect::<Vec<_>>(), (0..50u64).map(|x| x * x).collect::<Vec<_>>());
        // state survives across map calls (engines stay alive between shards)
        let again = pool.map((0..8).collect(), |_, _| {});
        let total: usize = (0..4).map(|w| again.iter().filter(|r| r.1 == w).map(|r| r.2).max().unwrap_or(0)).sum();
        assert!(total >= 50, "per-worker counters were reset");
    }

    #[test]
    fn panics_propagate_after_the_batch() {
        let pool = Pool::new(vec![|x: i32| if x == 3 { panic!("boom") } else { x }; 2]);
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| pool.map((0..6).collect(), |_, _| {})));
        assert!(r.is_err());
        // the pool still works afterwards
        assert_eq!(pool.map(vec![10, 11], |_, _| {}), vec![10, 11]);
    }
}
