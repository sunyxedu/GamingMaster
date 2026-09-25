"""Deserializer for generals.io replay files (.gior format)."""

import json

try:
    import lzstring
    _HAS_LZSTRING = True
except ImportError:
    _HAS_LZSTRING = False


# Constants from generals.io
class Constants:
    DEFAULT_CITY_DENSITY_OPTION = 1.0
    DEFAULT_MOUNTAIN_DENSITY_OPTION = 1.0
    DEFAULT_SWAMP_DENSITY_OPTION = 0.0
    DEFAULT_CITY_FAIRNESS_OPTION = 0
    DEFAULT_SPAWN_FAIRNESS_OPTION = 0
    DEFAULT_DESERT_DENSITY_OPTION = 0.0
    DEFAULT_LOOKOUT_DENSITY_OPTION = 0.0
    DEFAULT_OBSERVATORY_DENSITY_OPTION = 0.0


def deserialize_move(serialized):
    """Deserialize a move object."""
    return {
        "index": serialized[0],
        "start": serialized[1],
        "end": serialized[2],
        "is50": serialized[3],
        "turn": serialized[4],
    }


def deserialize_afk(serialized):
    """Deserialize an AFK object."""
    return {
        "index": serialized[0],
        "turn": serialized[1],
    }


def deserialize_chat(serialized):
    """Deserialize a chat message."""
    return {
        "text": serialized[0],
        "prefix": serialized[1],
        "playerIndex": serialized[2],
        "turn": serialized[3],
    }


def deserialize_replay_python(serialized_bytes: bytes) -> list:
    """Pure-Python decompression — no Node.js needed (containers, clusters).

    .gior files are LZString.compressToUint8Array output: the UTF-16 code
    units of the raw compressed string packed as big-endian byte pairs.
    Reverse the packing, then apply the raw LZ-string decompress.
    """
    data = serialized_bytes
    if len(data) % 2:  # defensive: compressToUint8Array output is even-length
        data = data + b"\x00"
    compressed = "".join(
        chr((data[i] << 8) | data[i + 1]) for i in range(0, len(data), 2)
    )
    text = lzstring.LZString().decompress(compressed)
    if not text:
        raise ValueError("LZ-string decompression produced no output")
    return json.loads(text)


def deserialize_replay(serialized_bytes: bytes) -> dict:
    """
    Deserialize a generals.io replay from .gior format.
    
    Args:
        serialized_bytes: Raw bytes from .gior file
        
    Returns:
        Dictionary containing deserialized replay data
    """
    if not _HAS_LZSTRING:
        raise ImportError("the lzstring package is required (it is a project dependency)")
    obj = deserialize_replay_python(serialized_bytes)

    replay = {}
    i = 0

    replay["version"] = obj[i]; i += 1
    replay["id"] = obj[i]; i += 1
    replay["mapWidth"] = obj[i]; i += 1
    replay["mapHeight"] = obj[i]; i += 1
    replay["usernames"] = obj[i]; i += 1
    replay["stars"] = obj[i]; i += 1
    replay["cities"] = obj[i]; i += 1
    replay["cityArmies"] = obj[i]; i += 1
    replay["generals"] = obj[i]; i += 1
    replay["mountains"] = obj[i] or []; i += 1
    replay["moves"] = [deserialize_move(m) for m in obj[i]]; i += 1
    replay["afks"] = [deserialize_afk(a) for a in obj[i]]; i += 1
    replay["teams"] = obj[i]; i += 1
    replay["map"] = obj[i]; i += 1
    replay["neutrals"] = obj[i] or []; i += 1
    replay["neutralArmies"] = obj[i] or []; i += 1
    replay["swamps"] = obj[i] or []; i += 1
    replay["chat"] = [deserialize_chat(c) for c in (obj[i] or [])]; i += 1
    replay["playerColors"] = obj[i] or list(range(len(replay["usernames"]))); i += 1
    replay["lights"] = obj[i] or []; i += 1

    # Options
    options = obj[i] or [
        1,
        Constants.DEFAULT_CITY_DENSITY_OPTION,
        Constants.DEFAULT_MOUNTAIN_DENSITY_OPTION,
        Constants.DEFAULT_SWAMP_DENSITY_OPTION,
        Constants.DEFAULT_CITY_FAIRNESS_OPTION,
        Constants.DEFAULT_SPAWN_FAIRNESS_OPTION,
        Constants.DEFAULT_DESERT_DENSITY_OPTION,
        Constants.DEFAULT_LOOKOUT_DENSITY_OPTION,
        Constants.DEFAULT_OBSERVATORY_DENSITY_OPTION
    ]; i += 1

    replay["speed"] = options[0]
    replay["city_density"] = options[1]
    replay["mountain_density"] = options[2]
    replay["swamp_density"] = options[3]
    replay["modifiers"] = obj[i] or []; i += 1

    replay["observatories"] = obj[i] or []; i += 1
    replay["lookouts"] = obj[i] or []; i += 1
    replay["deserts"] = obj[i] or []; i += 1

    if replay["version"] > 12:
        replay["city_fairness"] = options[4]
        replay["spawn_fairness"] = options[5]
        replay["desert_density"] = options[6]
        replay["lookout_density"] = options[7]
        replay["observatory_density"] = options[8]
    else:
        replay["city_fairness"] = -1
        replay["spawn_fairness"] = -1
        replay["desert_density"] = 0.0
        replay["lookout_density"] = 0.0
        replay["observatory_density"] = 0.0

    return replay
