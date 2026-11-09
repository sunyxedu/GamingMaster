"""Real 5v5 Honor of Kings simulator interface over Tencent's team-game SDK.

The released SDK exposes this native processor through ``hok.hok3v3`` and must
run under its own Python 3.6-3.9 interpreter.  :class:`Hok5v5Env` keeps that
binary boundary in a worker process and presents the same two-phase contract as
the official 3v3 environment:

``step_feature(agent_id)`` -> five hero observations and a prediction token;
``step_action(agent_id, probabilities, frame)`` -> five sampled action results.

No proprietary archive is stored in this repository.  On construction the
installed 3v3 ``config.dat`` is copied and patched into the worker directory by
:mod:`env.hok5v5_config`.
"""

from __future__ import annotations

import fcntl
import glob
import os
import pickle
import re
import shutil
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Also allows PYTHONPATH=.../env for an SDK-only Python smoke test.
    from .hok5v5_config import (
        ACTION_HEAD_SIZES,
        HEROES_PER_CAMP,
        OBSERVATION_SIZE,
        POLICY_SIZE,
        BuildReport,
        build_5v5_config,
    )
except ImportError:  # pragma: no cover - exercised by the real-cluster smoke command
    from hok5v5_config import (  # type: ignore[no-redef]
        ACTION_HEAD_SIZES,
        HEROES_PER_CAMP,
        OBSERVATION_SIZE,
        POLICY_SIZE,
        BuildReport,
        build_5v5_config,
    )


_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "hok5v5_worker.py"
_LAUNCHER = _HERE.parent / "scripts" / "hok" / "sgame_simulator.sh"
DEFAULT_CAMP = (
    (190, 173, 117, 169, 112),
    (141, 111, 107, 128, 163),
)


def resolve_root(root: Optional[str] = None) -> Path:
    return Path(os.path.expanduser(root or os.environ.get("HOK_ROOT", "~/hok"))).resolve()


def _wait_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"gamecore server did not open {host}:{port} in {timeout:.0f}s")


def _gamecore_dir(root: Path) -> Path:
    found = sorted(Path(path) for path in glob.glob(str(root / "gamecore_*" / "gamecore")))
    if not found:
        raise FileNotFoundError(f"no gamecore_<version>/gamecore under {root}")
    gamecore = found[-1]
    if not (gamecore / "core_assets" / "license.dat").is_file():
        raise FileNotFoundError(f"{gamecore}/core_assets/license.dat is missing")
    return gamecore


class Hok5v5GamecoreServer:
    """A local gamecore server using the repository's portable-Wine launcher."""

    def __init__(
        self,
        root: Optional[str] = None,
        port: int = 23432,
        *,
        local: Optional[str] = None,
        log: Optional[str] = None,
    ) -> None:
        self.root = resolve_root(root)
        self.port = int(port)
        source = _gamecore_dir(self.root)
        local_root = Path(local or f"/tmp/{os.environ.get('USER', 'user')}-hok5v5").resolve()
        local_root.mkdir(parents=True, exist_ok=True)
        self.gamecore = local_root / source.parent.name
        with (local_root / ".copy.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (self.gamecore / ".complete").is_file():
                if self.gamecore.exists():
                    shutil.rmtree(self.gamecore)
                shutil.copytree(source, self.gamecore, symlinks=True)
                os.chmod(self.gamecore / "gamecore-server-linux-amd64", 0o755)
                (self.gamecore / ".complete").touch()

        bindir = local_root / "bin"
        bindir.mkdir(exist_ok=True)
        for name in ("sgame_simulator_remote_zmq", "sgame_simulator_repeated_zmq"):
            destination = bindir / name
            if not destination.exists():
                shutil.copy(_LAUNCHER, destination)
                os.chmod(destination, 0o755)

        child_env = dict(
            os.environ,
            GAMECORE_PATH=str(self.gamecore),
            HOK=str(self.root),
            WINE_PREFIX_ROOT=str(local_root / "wine"),
        )
        log_path = Path(log) if log else local_root / f"gamecore-server-{self.port}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("ab")
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                [
                    str(self.gamecore / "gamecore-server-linux-amd64"),
                    "server",
                    f"--server-address=:{self.port}",
                    "--simulator-remote-bin",
                    str(bindir / "sgame_simulator_remote_zmq"),
                    "--simulator-repeat-bin",
                    str(bindir / "sgame_simulator_repeated_zmq"),
                ],
                cwd=self.gamecore,
                env=child_env,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
            _wait_port("127.0.0.1", self.port, 60.0)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"gamecore server exited with {self.proc.returncode}; see {log_path}"
                )
        except BaseException:
            self.close()
            raise

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != size:
        raise EOFError(f"worker returned {len(data)} of {size} bytes")
    return data


class _WorkerBridge:
    def __init__(self, command: Sequence[str], init: Mapping[str, Any], log: Any) -> None:
        self.proc = subprocess.Popen(
            list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log
        )
        try:
            self.send(dict(init))
            self.metadata = self.recv()
        except BaseException:
            self.close(kill=True)
            raise

    def send(self, obj: Any) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("worker stdin is closed")
        data = pickle.dumps(obj, protocol=4)
        self.proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self.proc.stdin.flush()

    def recv(self) -> Any:
        if self.proc.stdout is None:
            raise RuntimeError("worker stdout is closed")
        head = _read_exact(self.proc.stdout, 8)
        size = struct.unpack("<Q", head)[0]
        status, payload = pickle.loads(_read_exact(self.proc.stdout, size))
        if status != "ok":
            raise RuntimeError(f"HoK 5v5 worker {self.proc.pid}:\n{payload}")
        return payload

    def request(self, command: str, argument: Any) -> Any:
        if self.proc.poll() is not None:
            raise RuntimeError(f"HoK 5v5 worker exited with {self.proc.returncode}")
        self.send((command, argument))
        return self.recv()

    def close(self, *, kill: bool = False) -> None:
        if self.proc.poll() is None and not kill:
            try:
                self.request("close", None)
            except Exception:
                kill = True
        if self.proc.poll() is None:
            if kill:
                self.proc.kill()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


@dataclass(frozen=True)
class HeroFeature:
    hero_runtime_id: int
    camp_id: int
    features: np.ndarray
    reward: Any
    hero_rid_reward: Mapping[Any, Any]


@dataclass(frozen=True)
class FeatureFrame:
    agent_id: int
    continue_process: bool
    token: Optional[int]
    heroes: Tuple[HeroFeature, ...]
    frame_no: int
    gameover: bool
    state: Mapping[str, Any]

    @property
    def observations(self) -> np.ndarray:
        if not self.heroes:
            return np.empty((0, OBSERVATION_SIZE), dtype=np.float32)
        return np.stack([hero.features for hero in self.heroes])


@dataclass(frozen=True)
class ActionResult:
    actions: Tuple[int, ...]
    legal_action: Tuple[np.ndarray, ...]
    sub_actions: np.ndarray
    final_prob_list: Tuple[np.ndarray, ...]
    is_train: bool


def _sdk_python(root: Path, value: Optional[str]) -> Path:
    configured = value or os.environ.get("HOK_SDK_PYTHON")
    if configured:
        path = Path(os.path.abspath(os.path.expanduser(configured)))
    else:
        venv = os.environ.get("HOK_SDK_VENV", "venv38")
        candidate = Path(os.path.expanduser(venv))
        if not candidate.is_absolute():
            candidate = root / candidate
        # Do not resolve this symlink: executing the venv launcher path is what
        # makes CPython discover that venv's site-packages.
        path = Path(os.path.abspath(candidate / "bin" / "python"))
    if not path.is_file():
        raise FileNotFoundError(f"hok_env Python interpreter not found: {path}")
    return path


def _sdk_config(root: Path, python: Path, value: Optional[str]) -> Path:
    if value:
        path = Path(os.path.expanduser(value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"hok_env config.dat not found: {path}")
        return path
    candidates = (
        root / "hok_env" / "hok_env" / "hok" / "hok3v3" / "config.dat",
        root / "hok_env" / "hok" / "hok3v3" / "config.dat",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    code = (
        "from hok.hok3v3.hero_config import interface_default_config; "
        "print(interface_default_config)"
    )
    try:
        output = subprocess.check_output([str(python), "-c", code], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError("could not locate hok.hok3v3 config.dat") from exc
    path = Path(output).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"hok_env reported a missing config.dat: {path}")
    return path


def _normalise_camp(camp: Any) -> Dict[str, Any]:
    if camp is None:
        camps: Any = DEFAULT_CAMP
    elif isinstance(camp, Mapping):
        if camp.get("mode") != "5v5":
            raise ValueError("5v5 camp mapping must have mode='5v5'")
        camps = camp.get("heroes")
    else:
        camps = camp
    if not isinstance(camps, Sequence) or len(camps) != 2:
        raise ValueError("camp must contain exactly two camps")
    output: List[List[Dict[str, Any]]] = []
    for camp_index, heroes in enumerate(camps):
        if not isinstance(heroes, Sequence) or len(heroes) != HEROES_PER_CAMP:
            raise ValueError(f"camp {camp_index} must contain exactly five heroes")
        row = []
        seen = set()
        for hero in heroes:
            item = dict(hero) if isinstance(hero, Mapping) else {"hero_id": hero}
            if "hero_id" not in item:
                raise ValueError(f"camp {camp_index} hero is missing hero_id")
            item["hero_id"] = int(item["hero_id"])
            if item["hero_id"] <= 0:
                raise ValueError(f"invalid hero_id {item['hero_id']}")
            if item["hero_id"] in seen:
                raise ValueError(f"camp {camp_index} repeats hero_id {item['hero_id']}")
            seen.add(item["hero_id"])
            row.append(item)
        output.append(row)
    return {"mode": "5v5", "heroes": output}


class Hok5v5Env:
    """One real 5v5 match using the official team-game request protocol."""

    def __init__(
        self,
        *,
        gamecore: Optional[str] = None,
        hok_root: Optional[str] = None,
        sdk_python: Optional[str] = None,
        sdk_config: Optional[str] = None,
        worker_cmd: Optional[Sequence[str]] = None,
        ports: Sequence[int] = (35350, 35351),
        runtime_id: str = "gm-hok5v5",
        aiserver_ip: str = "127.0.0.1",
        gamecore_port: int = 23432,
        gamecore_local: Optional[str] = None,
        workdir: Optional[str] = None,
        max_frame_num: int = 20000,
        req_timeout_ms: int = 30000,
        wait_game_max_timeout: int = 30,
        reward_config: Optional[Mapping[str, Any]] = None,
        log_level: Optional[int] = None,
        log: Optional[str] = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9-]+", runtime_id):
            raise ValueError("runtime_id may contain only letters, digits, and hyphens")
        if len(ports) != 2 or len({int(port) for port in ports}) != 2:
            raise ValueError("ports must contain two distinct AI-server ports")
        self.root = resolve_root(hok_root)
        self.runtime_id = runtime_id
        self._server: Optional[Hok5v5GamecoreServer] = None
        self._log_handle: Any = subprocess.DEVNULL
        self._closed = False
        self._worker: Optional[_WorkerBridge] = None
        try:
            python = _sdk_python(self.root, sdk_python) if worker_cmd is None else None
            if sdk_config is None and worker_cmd is not None:
                raise ValueError("sdk_config is required with a custom worker_cmd")
            source_config = _sdk_config(
                self.root,
                python if python is not None else Path(os.sys.executable),
                sdk_config,
            )
            self.workdir = Path(
                workdir or f"/tmp/{os.environ.get('USER', 'user')}-hok5v5/work/{runtime_id}"
            ).resolve()
            self.workdir.mkdir(parents=True, exist_ok=True)
            patched_config = self.workdir / "config.5v5.dat"
            self.config_report: BuildReport = build_5v5_config(
                source_config, patched_config, overwrite=True
            )

            if gamecore is None and worker_cmd is None:
                self._server = Hok5v5GamecoreServer(
                    str(self.root), gamecore_port, local=gamecore_local
                )
                gamecore = self._server.address
            if gamecore is None:
                gamecore = "127.0.0.1:0"  # Fake workers do not contact it.

            command = list(worker_cmd) if worker_cmd is not None else [str(python), str(_WORKER)]
            if log:
                log_path = Path(log).resolve()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = log_path.open("ab")
            init = {
                "gamecore": gamecore,
                "config": str(patched_config),
                "ports": [int(port) for port in ports],
                "runtime_id": runtime_id,
                "aiserver_ip": aiserver_ip,
                "max_frame_num": int(max_frame_num),
                "req_timeout_ms": int(req_timeout_ms),
                "wait_game_max_timeout": int(wait_game_max_timeout),
                "reward_config": dict(reward_config or {}),
                "log_level": log_level,
            }
            self._worker = _WorkerBridge(command, init, self._log_handle)
        except BaseException:
            self.close()
            raise
        metadata = self._worker.metadata
        if tuple(metadata.get("action_head_sizes", ())) != ACTION_HEAD_SIZES:
            self.close()
            raise RuntimeError(f"worker action ABI mismatch: {metadata}")
        if metadata.get("observation_size") != OBSERVATION_SIZE:
            self.close()
            raise RuntimeError(f"worker observation ABI mismatch: {metadata}")
        if metadata.get("heroes_per_camp") != HEROES_PER_CAMP:
            self.close()
            raise RuntimeError(f"worker hero-count ABI mismatch: {metadata}")
        if metadata.get("config_sha256") != self.config_report.target_sha256:
            self.close()
            raise RuntimeError(f"worker loaded the wrong config archive: {metadata}")
        self.metadata = metadata
        self._pending: Dict[int, FeatureFrame] = {}
        self._controlled: Tuple[int, ...] = ()

    def __enter__(self) -> "Hok5v5Env":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def controlled_agents(self) -> Tuple[int, ...]:
        return self._controlled

    def reset(
        self,
        camp: Any = None,
        *,
        use_common_ai: Sequence[bool] = (True, False),
        eval_mode: bool = False,
        extra_abs_key_info: Any = None,
    ) -> None:
        if len(use_common_ai) != 2:
            raise ValueError("use_common_ai must contain one flag per camp")
        common = tuple(bool(value) for value in use_common_ai)
        controlled = tuple(index for index, value in enumerate(common) if not value)
        if not controlled:
            raise ValueError("at least one camp must be controlled by this interface")
        payload = {
            "camp": _normalise_camp(camp),
            "use_common_ai": common,
            "eval_mode": bool(eval_mode),
            "extra_abs_key_info": extra_abs_key_info,
        }
        self._worker.request("reset", payload)
        self._pending.clear()
        self._controlled = controlled

    @staticmethod
    def _parse_frame(agent_id: int, payload: Mapping[str, Any]) -> FeatureFrame:
        heroes = []
        for row in payload["features"]:
            values = np.asarray(row["features"], dtype=np.float32).reshape(-1)
            if values.size != OBSERVATION_SIZE:
                raise RuntimeError(
                    f"worker returned {values.size} observation values, expected {OBSERVATION_SIZE}"
                )
            heroes.append(
                HeroFeature(
                    hero_runtime_id=int(row["hero_runtime_id"]),
                    camp_id=int(row["camp_id"]),
                    features=values.copy(),
                    reward=row.get("reward", 0.0),
                    hero_rid_reward=dict(row.get("hero_rid_reward", {})),
                )
            )
        state = dict(payload["state"])
        continuing = bool(payload["continue_process"])
        if continuing and len(heroes) != HEROES_PER_CAMP:
            raise RuntimeError(
                f"prediction frame has {len(heroes)} hero rows, expected {HEROES_PER_CAMP}"
            )
        return FeatureFrame(
            agent_id=agent_id,
            continue_process=continuing,
            token=None if payload.get("token") is None else int(payload["token"]),
            heroes=tuple(heroes),
            frame_no=int(state["frame_no"]),
            gameover=bool(state["gameover"]),
            state=state,
        )

    def step_feature(self, agent_id: int) -> FeatureFrame:
        agent_id = int(agent_id)
        if agent_id not in self._controlled:
            raise ValueError(f"agent {agent_id} is not controlled; controlled={self._controlled}")
        if agent_id in self._pending:
            raise RuntimeError(f"agent {agent_id} must answer its pending prediction frame first")
        frame = self._parse_frame(agent_id, self._worker.request("feature", agent_id))
        if frame.continue_process:
            self._pending[agent_id] = frame
        return frame

    def next_decision(self, agent_id: int, *, max_receives: int = 10_000) -> FeatureFrame:
        """Consume non-prediction frames until an action or game-over frame."""
        if len(self._controlled) != 1:
            raise RuntimeError(
                "next_decision is only safe with one controlled camp; "
                "use step_round for two-camp self-play"
            )
        for _ in range(max_receives):
            frame = self.step_feature(agent_id)
            if frame.continue_process or frame.gameover:
                return frame
        raise RuntimeError(f"no decision for agent {agent_id} in {max_receives} receives")

    @staticmethod
    def _probabilities(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape == (HEROES_PER_CAMP, 1, POLICY_SIZE):
            array = array[:, 0, :]
        if array.shape != (HEROES_PER_CAMP, POLICY_SIZE):
            raise ValueError(
                f"probabilities have shape {array.shape}, expected "
                f"({HEROES_PER_CAMP}, {POLICY_SIZE})"
            )
        if not np.isfinite(array).all():
            raise ValueError("probabilities contain NaN or infinity")
        offset = 0
        for width in ACTION_HEAD_SIZES[:-1]:
            head = array[:, offset : offset + width]
            if (head < 0).any() or (head.sum(axis=1) <= 0).any():
                raise ValueError("each categorical action head must have nonnegative, nonzero mass")
            offset += width
        return np.ascontiguousarray(array)

    @staticmethod
    def normalise_probabilities(values: Any) -> np.ndarray:
        """Normalize the first five heads while preserving the value scalar."""
        array = Hok5v5Env._probabilities(values).copy()
        offset = 0
        for width in ACTION_HEAD_SIZES[:-1]:
            head = array[:, offset : offset + width]
            head /= head.sum(axis=1, keepdims=True)
            offset += width
        return array

    @staticmethod
    def random_probabilities(seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        output = np.zeros((HEROES_PER_CAMP, POLICY_SIZE), dtype=np.float32)
        offset = 0
        for width in ACTION_HEAD_SIZES[:-1]:
            head = rng.random((HEROES_PER_CAMP, width), dtype=np.float32)
            head /= head.sum(axis=1, keepdims=True)
            output[:, offset : offset + width] = head
            offset += width
        return output

    def step_action(
        self,
        agent_id: int,
        probabilities: Any,
        frame: Optional[FeatureFrame] = None,
    ) -> Tuple[ActionResult, ...]:
        agent_id = int(agent_id)
        expected = self._pending.get(agent_id)
        if expected is None:
            raise RuntimeError(f"agent {agent_id} has no pending prediction frame")
        selected = expected if frame is None else frame
        if selected.agent_id != agent_id or selected.token != expected.token:
            raise ValueError("frame is stale or belongs to another agent")
        array = self._probabilities(probabilities)
        payload = self._worker.request(
            "action",
            {"agent_id": agent_id, "token": expected.token, "probabilities": array},
        )
        self._pending.pop(agent_id, None)
        results = []
        for row in payload:
            actions = tuple(int(value) for value in row["actions"])
            legal = tuple(np.asarray(head, dtype=np.float32) for head in row["legal_action"])
            sub_actions = np.asarray(row["sub_actions"], dtype=np.float32).reshape(-1)
            final = tuple(np.asarray(head, dtype=np.float32) for head in row["final_prob_list"])
            if len(actions) != len(ACTION_HEAD_SIZES) - 1:
                raise RuntimeError(f"worker returned an invalid action layout: {actions}")
            if tuple(head.size for head in legal) != ACTION_HEAD_SIZES[:-1]:
                raise RuntimeError(f"worker returned an invalid legal-action layout: {legal}")
            if sub_actions.size != len(ACTION_HEAD_SIZES) - 1:
                raise RuntimeError(f"worker returned an invalid sub-action layout: {sub_actions}")
            if tuple(head.size for head in final) != ACTION_HEAD_SIZES:
                raise RuntimeError(f"worker returned an invalid final-probability layout: {final}")
            results.append(
                ActionResult(
                    actions=actions,
                    legal_action=legal,
                    sub_actions=sub_actions,
                    final_prob_list=final,
                    is_train=bool(row["is_train"]),
                )
            )
        if len(results) != HEROES_PER_CAMP:
            raise RuntimeError(f"worker returned {len(results)} actions, expected five")
        return tuple(results)

    def step_round(
        self, policies: Mapping[int, Callable[[FeatureFrame], Any]]
    ) -> Dict[int, Tuple[FeatureFrame, Optional[Tuple[ActionResult, ...]]]]:
        """Service one incoming simulator frame for every controlled camp.

        The simulator may wait for camp 0's prediction response before sending
        camp 1's frame.  Calling each policy immediately is therefore required
        for two-camp self-play; collecting both observations first can deadlock.
        """
        missing = set(self._controlled) - set(policies)
        if missing:
            raise KeyError(f"missing policies for controlled camps {sorted(missing)}")
        output = {}
        for agent_id in self._controlled:
            frame = self.step_feature(agent_id)
            results = None
            if frame.continue_process:
                results = self.step_action(agent_id, policies[agent_id](frame), frame)
            output[agent_id] = (frame, results)
        return output

    def close_game(self, *, force: bool = True) -> None:
        self._worker.request("end", {"force": bool(force)})
        self._pending.clear()
        self._controlled = ()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.close()
            self._worker = None
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._log_handle is not subprocess.DEVNULL:
            self._log_handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "ACTION_HEAD_SIZES",
    "DEFAULT_CAMP",
    "HEROES_PER_CAMP",
    "OBSERVATION_SIZE",
    "POLICY_SIZE",
    "ActionResult",
    "FeatureFrame",
    "HeroFeature",
    "Hok5v5Env",
    "Hok5v5GamecoreServer",
]
