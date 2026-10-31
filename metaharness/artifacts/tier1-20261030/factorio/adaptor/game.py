"""Offline Factorio 2.0.72 remote-manager benchmark; standard-library only."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from typing import Any
from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = True
SEEDS = {"train": [101, 202, 303, 404], "heldout": [1101, 1202, 1303, 1404],
         "dev": [9001, 9002], "validation": [2101, 2202, 2303, 2404]}
EVAL_KWARGS = {}
ROOT = Path(__file__).resolve().parent
SCIENCE = {f"{s}-science-pack": w for s, w in [("automation", 1), ("logistic", 2),
           ("chemical", 6), ("production", 12), ("utility", 12), ("space", 30)]}
GRAMMAR = {
    "wait": "wait: advance one turn without operations",
    "mine_resource": "mine_resource ID COUNT: queue manual mining cycles on an ore tile",
    "mine_tree": "mine_tree ID: queue mining one real tree",
    "handcraft": "handcraft RECIPE COUNT: reserve inputs and queue nonrecursive crafts",
    "place": "place ITEM X Y [DIR]: consume a building item; DIR is 0/4/8/12 (N/E/S/W)",
    "rotate": "rotate ID: rotate an owned building clockwise",
    "set_recipe": "set_recipe ID RECIPE: set recipe on an empty, idle assembler",
    "insert": "insert ID ROLE ITEM COUNT: move stock into a machine inventory",
    "extract": "extract ID ROLE ITEM COUNT: move machine inventory to stock",
    "mine_building": "mine_building ID: recover an empty building; fluids/burning fuel are lost",
    "set_research": "set_research TECHNOLOGY: select research with satisfied prerequisites",
    "inspect": "inspect OFFSET: read-only page of 100 buildings, no turn charged",
    "recipe": "recipe NAME: read-only exact recipe details, including locked recipes",
    "technology": "technology NAME: read-only technology details",
}
ARITIES = {"wait": (0,), "mine_resource": (2,), "mine_tree": (1,), "handcraft": (2,),
           "place": (3, 4), "rotate": (1,), "set_recipe": (2,), "insert": (4,),
           "extract": (4,), "mine_building": (1,), "set_research": (1,),
           "inspect": (1,), "recipe": (1,), "technology": (1,)}
NUMERIC = {"mine_resource": (1, 2), "mine_tree": (1,), "handcraft": (2,),
           "place": (2, 3, 4), "rotate": (1,), "set_recipe": (1,),
           "insert": (1, 4), "extract": (1, 4), "mine_building": (1,), "inspect": (1,)}

class Rcon:
    def __init__(self, port: int, password: str):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.sock.settimeout(30)
        self.serial = 0
        self.send(3, password)
        while True:
            ident, typ, _ = self.receive()
            if ident == -1:
                self.close()
                raise RuntimeError("RCON authentication failed")
            if typ == 2:
                break

    def send(self, typ: int, text: str) -> int:
        self.serial += 1
        b = struct.pack("<ii", self.serial, typ) + text.encode() + b"\0\0"
        self.sock.sendall(struct.pack("<i", len(b)) + b)
        return self.serial

    def read(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = self.sock.recv(n - len(chunks))
            if not chunk:
                raise ConnectionError("Factorio closed RCON")
            chunks.extend(chunk)
        return bytes(chunks)

    def receive(self):
        size, = struct.unpack("<i", self.read(4))
        if not 10 <= size <= 4_000_000:
            raise ConnectionError("Invalid RCON frame length")
        b = self.read(size)
        ident, typ = struct.unpack("<ii", b[:8])
        return ident, typ, b[8:-2].decode("utf-8")

    def command(self, command: str) -> str:
        ident = self.send(2, command)
        end = self.send(2, '/mh {"method":"status"}')
        pieces = []
        while True:
            got, _, body = self.receive()
            if got == end:
                break
            if got == ident:
                pieces.append(body)
        return "".join(pieces)

    def request(self, **request):
        raw = self.command("/mh " + json.dumps(request, separators=(",", ":"), allow_nan=False))
        data = "".join(line[3:] for line in raw.splitlines() if line.startswith("MH:"))
        try:
            response = json.loads(data)
        except ValueError as exc:
            raise RuntimeError(f"Malformed bridge response: {raw[:600]}") from exc
        if not response.get("ok"):
            raise RuntimeError(f"Bridge: {response.get('error')}")
        return response["result"]

    def close(self):
        self.sock.close()


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FactorioGame:
    name = "Factorio"
    action_spec = ActionSpec(names=list(GRAMMAR), descriptions=GRAMMAR, noop="wait",
                            notes="One string, or up to 32 semicolon-separated operations per turn. Read-only queries must be alone.")

    def __init__(self):
        self.process = None
        self.rcon = None
        self.log = None
        self.directory = None
        self.state = {}
        self.invalid_reason = None
        self.closed = False
        self.seed = None
        self.elapsed_start = None
        self._score = 0.0
        self.components = {}

    def reset(self, seed: int):
        self.close()
        self.closed = False
        self.invalid_reason = None
        self.seed = int(seed)
        self.state = {}
        self._score = 0.0
        self.components = {}
        self.elapsed_start = time.monotonic()
        binary = Path(os.environ.get("FACTORIO_BINARY", ROOT / "runtime/factorio/bin/x64/factorio")).resolve()
        if not binary.is_file():
            raise RuntimeError("Factorio is not staged. Run setup.sh before evaluation.")
        runtime = Path(os.environ.get("FACTORIO_WRITE_DIR", ROOT / "runtime")).resolve()
        runtime.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="episode-", dir=runtime))
        d = self.directory
        mods = d / "mods"
        mods.mkdir()
        shutil.copytree(ROOT / "bridge", mods / "mh-bridge_1.0.0")
        (mods / "mod-list.json").write_text(json.dumps({"mods": [{"name": "base", "enabled": True}, {"name": "mh-bridge", "enabled": True}, {"name": "space-age", "enabled": False}, {"name": "quality", "enabled": False}, {"name": "elevated-rails", "enabled": False}]}))
        (d / "config.ini").write_text(f"[path]\nread-data={binary.parents[2] / 'data'}\nwrite-data={d}\n[other]\ncheck-updates=false\n")
        # Hash arbitrary Python integers deterministically into a legal uint32 map seed.
        engine_seed = int.from_bytes(hashlib.sha256(f"mh-factorio-cold-v1:{self.seed}".encode()).digest()[:4], "big") or 1
        (d / "map.json").write_text(json.dumps({"seed": engine_seed, "width": 256, "height": 256,
            "peaceful_mode": True, "water": 0, "autoplace_controls": {"enemy-base": {"size": 0}},
            "autoplace_settings": {"entity": {"treat_missing_as_default": False, "settings": {}},
                                   "decorative": {"treat_missing_as_default": False, "settings": {}}}}))
        (d / "server.json").write_text(json.dumps({"name": "offline-meta-harness", "description": "Local benchmark",
            "visibility": {"public": False, "lan": False}, "require_user_verification": False,
            "max_players": 0, "auto_pause": False, "autosave_interval": 0, "only_admins_can_pause_the_game": True,
            "allow_commands": "true"}))
        self.log = (d / "process.log").open("wb")
        base = [str(binary), "--config", str(d / "config.ini"), "--mod-directory", str(mods)]
        try:
            for attempt in range(4):
                created = subprocess.run(base + ["--create", str(d / "initial.zip"), "--map-gen-settings", str(d / "map.json")],
                                         stdout=self.log, stderr=subprocess.STDOUT, timeout=90)
                if not created.returncode:
                    self.map_attempt = attempt
                    break
                failure = self._log_tail()
                if attempt == 3 or not any(s in failure for s in ("insufficient resources", "no valid pump site")):
                    raise RuntimeError(failure)
                engine_seed = (engine_seed + 0x9E3779B9) % (2**32 - 1) or 1
                settings = json.loads((d / "map.json").read_text())
                settings["seed"] = engine_seed
                (d / "map.json").write_text(json.dumps(settings))
            port = free_port()
            password = os.environ.get("FACTORIO_RCON_PASSWORD") or secrets.token_urlsafe(32)
            self.process = subprocess.Popen(base + ["--start-server", str(d / "initial.zip"),
                "--server-settings", str(d / "server.json"), "--bind", f"127.0.0.1:{free_port(socket.SOCK_DGRAM)}",
                "--rcon-bind", f"127.0.0.1:{port}", "--rcon-password", password],
                stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(self._log_tail())
                try:
                    self.rcon = Rcon(port, password)
                    break
                except (ConnectionError, OSError):
                    time.sleep(.05)
            if self.rcon is None:
                raise TimeoutError("Factorio RCON startup timeout")
            self._refresh()
            if not self.state["paused"] or self.state["charged_ticks"] != 0:
                raise RuntimeError("Initial world did not remain paused at tick zero")
            return self.observe()
        except Exception as exc:
            self.invalid_reason = str(exc)[:2000]
            self.close()
            raise

    def _log_tail(self):
        if self.log:
            self.log.flush()
        return (self.directory / "process.log").read_text(errors="replace")[-4000:]

    def _refresh(self):
        previous = self.state
        self.state = self.rcon.request(method="observe")
        # Lua serializes empty tables as arrays: normalize semantic maps/lists.
        for key in ("stock", "manual_output", "production", "consumption", "baseline_production", "baseline_consumption", "lab_consumption"):
            if not self.state.get(key):
                self.state[key] = {}
        for key in ("queue", "results", "entities", "trees", "resources", "enabled_recipes", "available_research"):
            if not self.state.get(key):
                self.state[key] = []
        p = self.state["production"]
        for k, v in self.state["baseline_production"].items():
            if p.get(k, 0) < v:
                raise RuntimeError("Production counter decreased below reset baseline")
            p[k] = p.get(k, 0) - v
        c = self.state["consumption"]
        for k, v in self.state["baseline_consumption"].items():
            if c.get(k, 0) < v:
                raise RuntimeError("Consumption counter decreased below reset baseline")
            c[k] = c.get(k, 0) - v
        for key in ("production", "consumption"):
            for name, value in previous.get(key, {}).items():
                if self.state[key].get(name, 0) < value:
                    raise RuntimeError(f"Non-monotonic engine {key} counter for {name}")
        self.components = {"bootstrap": sum(w * min(p.get(i, 0), cap) for i, w, cap in [
            ("iron-plate", .01, 1000), ("copper-plate", .01, 1000),
            ("iron-gear-wheel", .02, 200), ("electronic-circuit", .05, 200)])}
        for name, weight in SCIENCE.items():
            lab = self.state["lab_consumption"].get(name, 0)
            if lab != c.get(name, 0):
                raise RuntimeError("Science consumption audit disagrees with engine counters")
            self.components[name] = float(weight * (p.get(name, 0) + 4 * lab))
        self._score = float(sum(self.components.values()))
        self.state["score"] = self._score
        self.state["score_breakdown"] = dict(self.components)
        self.state["evaluation_seed"] = self.seed
        self.state["map_attempt"] = self.map_attempt
        self.state["entity_page_count"] = len(self.state["entities"])
        while len(json.dumps(self.state, allow_nan=False).encode()) >= 190_000 and self.state["entities"]:
            self.state["entities"].pop()
            self.state["entity_page_count"] = len(self.state["entities"])
        if len(json.dumps(self.state, allow_nan=False).encode()) >= 200_000:
            raise RuntimeError("Observation exceeded 200 kB")

    def observe(self):
        if self.process is not None and self.process.poll() is not None:
            self.invalid_reason = "Factorio server exited unexpectedly"
        result = copy.deepcopy(self.state)
        if self.invalid_reason:
            result["infrastructure_error"] = self.invalid_reason
        return result

    def step(self, action: Any):
        if self.done:
            return self.observe()
        try:
            if not isinstance(action, str) or len(action) > 8192:
                raise ValueError("Action must be a string of at most 8192 characters")
            strings = action.split(";")
            if len(strings) > 32:
                raise ValueError("At most 32 operations per turn")
            actions = []
            for string in strings:
                a = shlex.split(string)
                if not a or a[0] not in ARITIES or len(a) - 1 not in ARITIES[a[0]]:
                    raise ValueError("Invalid action grammar; see action_spec")
                for idx in NUMERIC.get(a[0], ()):
                    if idx < len(a):
                        v = float(a[idx])
                        if not math.isfinite(v):
                            raise ValueError("Coordinates/counts must be finite")
                        a[idx] = int(v) if v.is_integer() else v
                actions.append(a)
            queries = {"inspect", "recipe", "technology"}
            if any(a[0] in queries for a in actions) and len(actions) != 1:
                raise ValueError("Read-only queries must be alone")
        except (ValueError, TypeError, OverflowError) as exc:
            self.state["results"] = [{"ok": False, "error": str(exc)}]
            return self.observe()
        try:
            if actions[0][0] in queries:
                a = actions[0]
                try:
                    if a[0] == "inspect":
                        self.rcon.request(method="page", offset=a[1])
                        self._refresh()
                    else:
                        self.state["query_result"] = self.rcon.request(method=a[0], name=a[1])
                except RuntimeError as exc:
                    if str(exc).startswith("Bridge:"):
                        self.state["results"] = [{"ok": False, "error": str(exc)}]
                    else:
                        raise
            else:
                before = self.state["charged_ticks"]
                turn = self.state["turn"] + 1
                self.rcon.request(method="step", turn=turn, actions=actions)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    status = self.rcon.request(method="status")
                    if status["paused"]:
                        break
                    time.sleep(.025)
                else:
                    raise TimeoutError("Factorio failed to pause after fixed turn")
                self._refresh()
                if self.state["charged_ticks"] != before + 1800 or self.state["turn"] != turn:
                    raise RuntimeError("Engine advanced an incorrect tick count")
        except Exception as exc:
            self.invalid_reason = f"{type(exc).__name__}: {exc}"[:2000]
        return self.observe()

    @property
    def done(self):
        return bool(self.invalid_reason or self.state.get("turn", 0) >= 128)

    @property
    def score(self):
        return self._score

    def score_breakdown(self):
        return dict(self.components)

    def info(self):
        self.observe()
        return {"valid_trial": self.invalid_reason is None, "invalid_reason": self.invalid_reason,
                "family": "cold-coast-v1", "seed": self.seed, "engine": "2.0.72",
                "charged_ticks": self.state.get("charged_ticks", 0),
                "runtime_directory": str(self.directory) if self.directory else None,
                "wall_seconds": time.monotonic() - self.elapsed_start if self.elapsed_start else 0}

    def describe(self):
        return '''Factorio 2.0.72 base-only cold-coast-v1: manage a real factory remotely on a seeded 256x256 island/coast area. No avatar, enemies, pollution, expansion or GUI. Initial stock: 1 burner mining drill, 1 stone furnace, 8 iron plates, 1 wood in a 200-slot inventory. All other items require finite mining/crafting or real machine production. 128 turns, exactly 1800 engine ticks (30 seconds) per turn; thinking and observation cost no game time. A turn is one action string or up to 32 semicolon-separated operations, applied in order. Individual failed operations retain earlier successes and still charge the turn. Malformed strings change no world state and cost no turn. Wait also charges one turn. No arbitrary code, item grants or clock controls.
Actions: ''' + "; ".join(GRAMMAR.values()) + '''.
Manual mining, tree mining and handcrafting share one serial worker: at most 32 queued jobs and 3600 remaining ticks, running concurrently with machines. Resource/tree mining uses prototype mining time divided by character mining speed; crafting uses recipe seconds at speed 1. Handcraft inputs must already exist; no recursive subcrafting, no fluid/probabilistic recipes. Craft counts mean recipe executions, including multiple outputs. Inputs reserved immediately; full stock blocks completion. Only categories allowed by the vanilla character prototype. Automated machines follow actual engine recipes, fuels, fluids, electricity and research unlocks. Vanilla craft-item technology triggers are also honored for completed, paid manual crafts, without adding production statistics; steam-power requires 50 engine-smelted iron plates, electronics 10 copper plates, then crafting a lab unlocks red science.
Transfer roles: fuel, source/result (furnace), input/output (assembler), lab, chest. Insert+extract share 100 actual items per turn. Empty buildings only can be mined; empty idle assemblers only can change recipe. Extract cargo first. Fluids, unfinished burning fuel are lost on mining. Normal-quality items only. Placement uses engine collision/terrain checks and consumes stock on success. Use half-tile coordinates for odd-size buildings; DIR 0/4/8/12 means north/east/south/west; drill direction points output, belts move in direction. Belts, pipes, inserters and power poles use ordinary engine connections. Fixed finite ores and wood vary by seed. No oil is supplied in this cold-start family, so chemical and later science are not reachable; focus on red/green science and research.
Observations: paused engine tick, charged_ticks, turn/budget; stock/slots; manual queue/outputs; exact transfer counts; last action results; bounds [-128,128); water_rows with inclusive x intervals (other tiles grass); patches and per-resource rows [id,name,x,y,remaining amount]; trees [id,x,y]; validated pump_site; up to 100 buildings with stable IDs, positions, boxes, status code, inventories, burner, fluid amounts and temperatures, stored joules and electric network ID (stored energy alone does not establish supply); inspect OFFSET selects another building page; entity_page_count is the actual page size (at most 100, reduced for large inventories). enabled_recipes lists exact ingredients/products/time; available_research lists prerequisites and science costs; research current/progress/completions; raw normal-quality engine production and consumption, laboratory science consumption and score. recipe NAME and technology NAME query locked details without time. IDs remain stable until mining/depletion; invalid IDs fail safely.
Score: bootstrap .01*min(iron plates produced,1000)+.01*min(copper plates,1000)+.02*min(gears,200)+.05*min(circuits,200), cap 34. Plus weights automation/logistic/chemical/production/utility/space=1/2/6/12/12/30 times (engine-produced science + 4*science actually consumed in labs). Initial stock, manual outputs, building recovery and transfers earn no production score. Manually crafted science earns consumption credit only when used by a real powered lab. Final score is measured after the last charged turn. This is a benchmark score, not an official vanilla score.'''

    def close(self):
        if self.rcon is not None:
            try:
                self.rcon.close()
            except OSError:
                pass
            self.rcon = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.closed = True


def make_game(**kwargs):
    if kwargs:
        raise TypeError(f"Unsupported game options: {', '.join(kwargs)}")
    return FactorioGame()
