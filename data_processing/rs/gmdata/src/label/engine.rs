//! External engine processes speaking a line protocol (UCI, USI, GTP, KataGo analysis
//! JSON, Egaroucid console, …): spawn with arguments, line I/O with a per-command
//! timeout, crash detection with the engine's stderr tail, restart.
//!
//! Engines are never linked: an [`Engine`] runs an [`EngineCommand`] — normally a
//! process (`Process`), or for tests an in-process [`LineHandler`] (`InProcess`) that
//! goes through the same channels, timeouts and crash handling, so game labellers can
//! be unit-tested without an engine binary (see [`super::fake`]).
//!
//! Reads happen on a dedicated thread per engine, so [`Engine::recv`] can time out. A
//! timeout leaves the protocol state unknown: the engine is marked poisoned and every
//! later call fails until [`Engine::restart`]. The label driver restarts a session
//! after any error and retries the position once.

use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// An in-process stand-in for an engine process: one input line → reply lines.
/// `None` simulates a crash (the "process" exits, its output closes).
pub trait LineHandler: Send {
    fn handle(&mut self, line: &str) -> Option<Vec<String>>;
}

impl<F: FnMut(&str) -> Option<Vec<String>> + Send> LineHandler for F {
    fn handle(&mut self, line: &str) -> Option<Vec<String>> {
        self(line)
    }
}

/// Factory of fresh handlers (one per spawn / restart).
pub type HandlerFactory = Arc<dyn Fn() -> Box<dyn LineHandler> + Send + Sync>;

/// What an [`Engine`] runs.
#[derive(Clone)]
pub enum EngineCommand {
    /// A program with arguments; stdin/stdout piped, stderr captured (last lines kept).
    Process { program: PathBuf, args: Vec<String>, cwd: Option<PathBuf>, env: Vec<(String, String)> },
    /// An in-process handler (tests / fakes).
    InProcess { name: String, factory: HandlerFactory },
}

impl EngineCommand {
    pub fn process(program: impl Into<PathBuf>, args: &[String]) -> Self {
        EngineCommand::Process { program: program.into(), args: args.to_vec(), cwd: None, env: Vec::new() }
    }

    pub fn in_process<H: LineHandler + 'static>(name: &str, factory: impl Fn() -> H + Send + Sync + 'static) -> Self {
        EngineCommand::InProcess { name: name.to_string(), factory: Arc::new(move || Box::new(factory()) as Box<dyn LineHandler>) }
    }

    /// Human-readable command line.
    pub fn describe(&self) -> String {
        match self {
            EngineCommand::Process { program, args, .. } => {
                std::iter::once(program.display().to_string()).chain(args.iter().cloned()).collect::<Vec<_>>().join(" ")
            }
            EngineCommand::InProcess { name, .. } => format!("<in-process {name}>"),
        }
    }
}

impl fmt::Debug for EngineCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Typed engine failures (wrapped in `anyhow::Error`; downcast to count them).
#[derive(Debug)]
pub enum EngineError {
    /// no reply line within the timeout (the engine is poisoned until restarted)
    Timeout { waited: Duration, context: String },
    /// the engine's output closed (process exit / crash)
    Exited { status: String, stderr: String },
    /// the engine said something the protocol does not allow
    Protocol(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Timeout { waited, context } => write!(f, "engine timeout after {:.1}s ({context})", waited.as_secs_f64()),
            EngineError::Exited { status, stderr } => {
                write!(f, "engine exited ({status})")?;
                if !stderr.is_empty() {
                    write!(f, "; stderr tail:\n{stderr}")?;
                }
                Ok(())
            }
            EngineError::Protocol(m) => write!(f, "engine protocol error: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Whether `e` is (or wraps) an [`EngineError::Timeout`].
pub fn is_timeout(e: &anyhow::Error) -> bool {
    matches!(e.downcast_ref::<EngineError>(), Some(EngineError::Timeout { .. }))
}

const STDERR_TAIL: usize = 40;

enum Input {
    Process(ChildStdin),
    Thread(Sender<String>),
}

struct Running {
    input: Option<Input>,
    lines: Receiver<String>,
    child: Option<Child>,
    threads: Vec<JoinHandle<()>>,
    stderr: Arc<Mutex<VecDeque<String>>>,
}

/// One long-lived engine (see the module docs).
pub struct Engine {
    cmd: EngineCommand,
    run: Option<Running>,
    poisoned: bool,
    /// times the engine was started (1 + restarts)
    pub spawns: usize,
}

impl Engine {
    pub fn spawn(cmd: EngineCommand) -> Result<Self> {
        let mut e = Engine { cmd, run: None, poisoned: false, spawns: 0 };
        e.start()?;
        Ok(e)
    }

    pub fn command(&self) -> &EngineCommand {
        &self.cmd
    }

    fn start(&mut self) -> Result<()> {
        self.kill();
        let (tx, rx) = mpsc::channel::<String>();
        let stderr = Arc::new(Mutex::new(VecDeque::new()));
        let run = match &self.cmd {
            EngineCommand::Process { program, args, cwd, env } => {
                let mut c = Command::new(program);
                c.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
                if let Some(d) = cwd {
                    c.current_dir(d);
                }
                c.envs(env.iter().map(|(k, v)| (k, v)));
                let mut child = c.spawn().with_context(|| format!("spawn engine {}", self.cmd.describe()))?;
                let stdin = child.stdin.take().context("engine stdin")?;
                let stdout = child.stdout.take().context("engine stdout")?;
                let err = child.stderr.take().context("engine stderr")?;
                let out_thread = std::thread::spawn(move || {
                    let mut r = BufReader::new(stdout);
                    let mut buf = Vec::new();
                    loop {
                        buf.clear();
                        match r.read_until(b'\n', &mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                let line = String::from_utf8_lossy(&buf).trim_end_matches(['\n', '\r']).to_string();
                                if tx.send(line).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                });
                let tail = stderr.clone();
                let err_thread = std::thread::spawn(move || {
                    let mut r = BufReader::new(err);
                    let mut buf = Vec::new();
                    loop {
                        buf.clear();
                        match r.read_until(b'\n', &mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                let mut t = tail.lock().unwrap();
                                t.push_back(String::from_utf8_lossy(&buf).trim_end().to_string());
                                if t.len() > STDERR_TAIL {
                                    t.pop_front();
                                }
                            }
                        }
                    }
                });
                Running { input: Some(Input::Process(stdin)), lines: rx, child: Some(child), threads: vec![out_thread, err_thread], stderr }
            }
            EngineCommand::InProcess { factory, .. } => {
                let (in_tx, in_rx) = mpsc::channel::<String>();
                let mut handler = factory();
                let t = std::thread::spawn(move || {
                    while let Ok(line) = in_rx.recv() {
                        match handler.handle(&line) {
                            Some(out) => {
                                for l in out {
                                    if tx.send(l).is_err() {
                                        return;
                                    }
                                }
                            }
                            None => return,
                        }
                    }
                });
                Running { input: Some(Input::Thread(in_tx)), lines: rx, child: None, threads: vec![t], stderr }
            }
        };
        self.run = Some(run);
        self.poisoned = false;
        self.spawns += 1;
        Ok(())
    }

    /// Kill the current process (if any) and start a fresh one.
    pub fn restart(&mut self) -> Result<()> {
        self.start()
    }

    /// Stop the engine (kill + reap). Idempotent. The reader threads are detached, not
    /// joined: they end when the pipes close, which a grandchild holding them open (or a
    /// hung in-process handler) may delay.
    pub fn kill(&mut self) {
        if let Some(mut run) = self.run.take() {
            run.input = None; // closes stdin / the handler's input
            if let Some(mut c) = run.child.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
            run.threads.clear();
        }
    }

    /// The last lines the engine wrote to stderr.
    pub fn stderr_tail(&self) -> String {
        self.run.as_ref().map(|r| r.stderr.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n")).unwrap_or_default()
    }

    fn exited(&mut self) -> anyhow::Error {
        // let the stderr reader drain before reporting
        std::thread::sleep(Duration::from_millis(20));
        let stderr = self.stderr_tail();
        let status = match self.run.as_mut().and_then(|r| r.child.as_mut()) {
            Some(c) => match c.try_wait() {
                Ok(Some(s)) => s.to_string(),
                Ok(None) => "output closed".to_string(),
                Err(e) => e.to_string(),
            },
            None => "in-process handler stopped".to_string(),
        };
        self.poisoned = true;
        EngineError::Exited { status, stderr }.into()
    }

    fn ready(&self) -> Result<()> {
        if self.run.is_none() {
            bail!("engine {} is not running", self.cmd.describe());
        }
        if self.poisoned {
            bail!("engine {} failed earlier (timeout / exit); restart it first", self.cmd.describe());
        }
        Ok(())
    }

    /// Write one line (a newline is appended).
    pub fn send(&mut self, line: &str) -> Result<()> {
        self.ready()?;
        let run = self.run.as_mut().unwrap();
        let ok = match run.input.as_mut() {
            Some(Input::Process(stdin)) => stdin.write_all(line.as_bytes()).and_then(|_| stdin.write_all(b"\n")).and_then(|_| stdin.flush()).is_ok(),
            Some(Input::Thread(tx)) => tx.send(line.to_string()).is_ok(),
            None => false,
        };
        if !ok {
            return Err(self.exited());
        }
        Ok(())
    }

    /// Next output line, waiting at most `timeout`.
    pub fn recv(&mut self, timeout: Duration) -> Result<String> {
        self.ready()?;
        match self.run.as_ref().unwrap().lines.recv_timeout(timeout) {
            Ok(l) => Ok(l),
            Err(RecvTimeoutError::Timeout) => {
                self.poisoned = true;
                Err(EngineError::Timeout { waited: timeout, context: "waiting for a line".into() }.into())
            }
            Err(RecvTimeoutError::Disconnected) => Err(self.exited()),
        }
    }

    /// Lines up to and including the first one for which `done` is true, all within
    /// `timeout` (one overall deadline).
    pub fn recv_until(&mut self, timeout: Duration, mut done: impl FnMut(&str) -> bool) -> Result<Vec<String>> {
        let deadline = Instant::now() + timeout;
        let mut out = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let line = self.recv(left).map_err(|e| match e.downcast::<EngineError>() {
                Ok(EngineError::Timeout { .. }) => EngineError::Timeout {
                    waited: timeout,
                    context: format!("{} lines received, last: {:?}", out.len(), out.last().map(String::as_str).unwrap_or("")),
                }
                .into(),
                Ok(other) => other.into(),
                Err(e) => e,
            })?;
            let stop = done(&line);
            out.push(line);
            if stop {
                return Ok(out);
            }
        }
    }

    /// `send(line)` then [`Engine::recv_until`].
    pub fn request(&mut self, line: &str, timeout: Duration, done: impl FnMut(&str) -> bool) -> Result<Vec<String>> {
        self.send(line)?;
        self.recv_until(timeout, done)
    }

    /// Output lines already received (non-blocking).
    pub fn drain(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(run) = self.run.as_ref() {
            while let Ok(l) = run.lines.try_recv() {
                out.push(l);
            }
        }
        out
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> EngineCommand {
        EngineCommand::process("/bin/sh", &["-c".to_string(), script.to_string()])
    }

    const T: Duration = Duration::from_secs(10);

    #[test]
    fn process_line_io_and_restart() {
        // echo engine: replies "ok <line>" to every line, "bye" exits with status 3
        let mut e = Engine::spawn(sh(r#"while read -r l; do [ "$l" = bye ] && { echo gone >&2; exit 3; }; echo "ok $l"; done"#)).unwrap();
        assert_eq!(e.request("hello", T, |_| true).unwrap(), vec!["ok hello"]);
        e.send("a").unwrap();
        e.send("b").unwrap();
        assert_eq!(e.recv_until(T, |l| l == "ok b").unwrap(), vec!["ok a", "ok b"]);
        e.send("bye").unwrap();
        let err = e.recv(T).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err.downcast_ref::<EngineError>(), Some(EngineError::Exited { .. })), "{msg}");
        assert!(msg.contains("gone"), "stderr tail missing: {msg}");
        assert!(e.send("again").is_err(), "a dead engine must be restarted first");
        e.restart().unwrap();
        assert_eq!(e.spawns, 2);
        assert_eq!(e.request("again", T, |_| true).unwrap(), vec!["ok again"]);
    }

    #[test]
    fn timeout_poisons_until_restart() {
        let mut e = Engine::spawn(sh(r#"while read -r l; do [ "$l" = slow ] && sleep 5; echo "ok $l"; done"#)).unwrap();
        e.send("slow").unwrap();
        let err = e.recv_until(Duration::from_millis(200), |_| true).unwrap_err();
        assert!(is_timeout(&err), "{err}");
        assert!(e.send("x").is_err());
        let t0 = Instant::now();
        e.restart().unwrap(); // kills the sleeping shell
        assert!(t0.elapsed() < Duration::from_secs(4));
        assert_eq!(e.request("fast", T, |_| true).unwrap(), vec!["ok fast"]);
    }

    #[test]
    fn missing_program_fails_to_spawn() {
        let err = Engine::spawn(EngineCommand::process("/nonexistent/engine-binary", &[])).err().unwrap();
        assert!(err.to_string().contains("spawn engine"), "{err}");
    }

    #[test]
    fn in_process_handler_crash_and_restart() {
        let cmd = EngineCommand::in_process("counter", || {
            let mut n = 0;
            move |l: &str| {
                n += 1;
                if l == "crash" {
                    return None;
                }
                Some(vec![format!("{n}:{l}")])
            }
        });
        let mut e = Engine::spawn(cmd).unwrap();
        assert_eq!(e.request("a", T, |_| true).unwrap(), vec!["1:a"]);
        assert_eq!(e.request("b", T, |_| true).unwrap(), vec!["2:b"]);
        e.send("crash").unwrap();
        assert!(e.recv(T).is_err());
        e.restart().unwrap();
        assert_eq!(e.request("c", T, |_| true).unwrap(), vec!["1:c"], "restart must create a fresh handler");
        assert!(e.drain().is_empty());
    }
}
