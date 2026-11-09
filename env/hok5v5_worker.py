"""SDK-side worker for :mod:`env.hok5v5`.

This file is intentionally standalone: it runs under hok_env's Python 3.6-3.9
interpreter, which owns the compiled ``hok.hok3v3`` extension.  The host-side
interface sends length-prefixed pickle messages over stdin/stdout.  Native and
Python SDK logs are redirected to stderr so they cannot corrupt that channel.
"""

import hashlib
import os
import pickle
import struct
import sys
import traceback

import numpy as np

HEROES_PER_CAMP = 5
OBSERVATION_SIZE = 5630
ACTION_HEAD_SIZES = (13, 25, 42, 42, 39, 1)
POLICY_SIZE = sum(ACTION_HEAD_SIZES)


def _read_exact(stream, size):
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
        raise EOFError("short worker message: got %d of %d bytes" % (len(data), size))
    return data


def _read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    size = struct.unpack("<Q", head)[0]
    return pickle.loads(_read_exact(stream, size))


def _write(stream, obj):
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _plain(value):
    """Convert extension/numpy containers to pickle-stable Python values."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_plain(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    try:
        return [_plain(item) for item in value]
    except TypeError:
        return repr(value)


def _object_fields(obj, names):
    result = {}
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            result[name] = value
    return result


_HERO_FIELDS = (
    "runtime_id",
    "config_id",
    "camp",
    "hp",
    "max_hp",
    "moneyCnt",
    "killCnt",
    "deadCnt",
    "assistCnt",
    "totalHurt",
    "totalHurtToHero",
    "totalBeHurtByHero",
    "level",
)
_ORGAN_FIELDS = ("runtime_id", "config_id", "camp", "type", "hp", "max_hp")


def _serialize_state(frame_state):
    state = _object_fields(frame_state, ("frame_no", "gameover", "sgame_id"))
    state["frame_no"] = int(frame_state.frame_no)
    state["gameover"] = bool(frame_state.gameover)
    state["heroes"] = [
        _object_fields(hero, _HERO_FIELDS) for hero in getattr(frame_state, "hero_list", [])
    ]
    state["organs"] = [
        _object_fields(organ, _ORGAN_FIELDS) for organ in getattr(frame_state, "organ_list", [])
    ]
    return state


def _serialize_features(features):
    rows = []
    for index, feature in enumerate(features):
        values = np.asarray(feature.feature, dtype=np.float32).reshape(-1)
        if values.size != OBSERVATION_SIZE:
            raise ValueError(
                "hero %d observation has %d values, expected %d"
                % (index, values.size, OBSERVATION_SIZE)
            )
        rows.append(
            {
                "hero_runtime_id": int(feature.hero_runtime_id),
                "camp_id": int(feature.camp_id),
                "features": values,
                "reward": _plain(getattr(feature, "reward", 0.0)),
                "hero_rid_reward": _plain(getattr(feature, "hero_rid_reward", {})),
            }
        )
    return rows


def _serialize_results(results):
    serialized = []
    for result in results:
        serialized.append(
            {
                "actions": tuple(int(value) for value in result.actions),
                "legal_action": tuple(
                    np.asarray(head, dtype=np.float32).reshape(-1) for head in result.legal_action
                ),
                "sub_actions": np.asarray(result.sub_actions, dtype=np.float32),
                "final_prob_list": tuple(
                    np.asarray(head, dtype=np.float32).reshape(-1)
                    for head in result.final_prob_list
                ),
                "is_train": bool(result.is_train),
            }
        )
    return serialized


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():
    stdin = sys.stdin.buffer
    protocol_stdout = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    init = _read(stdin)
    if init is None:
        return

    try:
        from hok.common.gamecore_client import GamecoreClient
        from hok.hok3v3.env import Environment
        from hok.hok3v3.hero_config import get_default_hero_config
        from hok.hok3v3.lib import lib3v3 as interface
        from hok.hok3v3.reward import update_reward_config
        from hok.hok3v3.server import AIServer

        config_path = os.path.abspath(init["config"])
        lib_processor = interface.Interface()
        lib_processor.Init(config_path)
        if init.get("log_level") is not None:
            lib_processor.SetLogLevel(int(init["log_level"]))
        update_reward_config(lib_processor, init.get("reward_config"))

        aiservers = [
            AIServer("tcp://0.0.0.0:%d" % int(port), lib_processor) for port in init["ports"]
        ]
        launcher = GamecoreClient(
            server_addr=init["gamecore"],
            gamecore_req_timeout=int(init.get("req_timeout_ms", 30000)),
            default_hero_config=get_default_hero_config(),
            max_frame_num=int(init.get("max_frame_num", 20000)),
        )
        env = Environment(
            aiservers,
            lib_processor,
            launcher,
            runtime_id=init["runtime_id"],
            wait_game_max_timeout=int(init.get("wait_game_max_timeout", 30)),
            aiserver_ip=init.get("aiserver_ip", "127.0.0.1"),
        )
        pending = {}
        next_token = [1, 1]
        interface_path = os.path.join(os.path.dirname(interface.__file__), "libinterface.so")
        _write(
            protocol_stdout,
            (
                "ok",
                {
                    "pid": os.getpid(),
                    "config_sha256": _sha256(config_path),
                    "interface_sha256": (
                        _sha256(interface_path) if os.path.isfile(interface_path) else None
                    ),
                    "observation_size": OBSERVATION_SIZE,
                    "action_head_sizes": ACTION_HEAD_SIZES,
                    "heroes_per_camp": HEROES_PER_CAMP,
                },
            ),
        )
    except Exception:
        _write(protocol_stdout, ("error", traceback.format_exc()))
        return

    while True:
        message = _read(stdin)
        if message is None:
            break
        command, argument = message
        try:
            if command == "reset":
                pending.clear()
                next_token[:] = [1, 1]
                env.reset(
                    [bool(value) for value in argument["use_common_ai"]],
                    argument["camp"],
                    eval_mode=bool(argument.get("eval_mode", False)),
                    extra_abs_key_info=argument.get("extra_abs_key_info"),
                )
                _write(protocol_stdout, ("ok", {"reset": True}))
            elif command == "feature":
                agent_id = int(argument)
                if agent_id in pending:
                    raise RuntimeError(
                        "agent %d still has an unanswered prediction frame" % agent_id
                    )
                continuing, features, frame_state = env.step_feature(agent_id)
                token = None
                if continuing:
                    token = next_token[agent_id]
                    next_token[agent_id] += 1
                    pending[agent_id] = (token, features, frame_state)
                payload = {
                    "continue_process": bool(continuing),
                    "token": token,
                    "features": _serialize_features(features),
                    "state": _serialize_state(frame_state),
                }
                _write(protocol_stdout, ("ok", payload))
            elif command == "action":
                agent_id = int(argument["agent_id"])
                token = int(argument["token"])
                if agent_id not in pending or pending[agent_id][0] != token:
                    raise RuntimeError("stale prediction token %d for agent %d" % (token, agent_id))
                raw = np.asarray(argument["probabilities"], dtype=np.float32)
                if raw.shape != (HEROES_PER_CAMP, POLICY_SIZE):
                    raise ValueError(
                        "probabilities have shape %r, expected (%d, %d)"
                        % (raw.shape, HEROES_PER_CAMP, POLICY_SIZE)
                    )
                _, features, frame_state = pending[agent_id]
                probabilities = [raw[index].reshape(1, POLICY_SIZE) for index in range(5)]
                ok, results = env.step_action(agent_id, probabilities, features, frame_state)
                if not ok:
                    raise RuntimeError("native ResultProcess failed")
                pending.pop(agent_id)
                _write(protocol_stdout, ("ok", _serialize_results(results)))
            elif command == "end":
                pending.clear()
                env.close_game(force=bool(argument.get("force", True)))
                _write(protocol_stdout, ("ok", {}))
            elif command == "close":
                try:
                    env.close_game(force=True)
                except Exception:
                    pass
                _write(protocol_stdout, ("ok", {}))
                break
            else:
                raise ValueError("unknown command %r" % (command,))
        except Exception:
            _write(protocol_stdout, ("error", traceback.format_exc()))


if __name__ == "__main__":
    main()
