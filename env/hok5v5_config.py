"""Build the missing Honor of Kings 5v5 team-interface configuration.

Tencent's released ``hok.hok3v3`` package contains a generic team-game native
processor and a 5v5 feature tree, but its bundled ``config.dat`` selects three
heroes per camp.  This module patches that archive in memory.  It does not
contain, extract, or redistribute Tencent assets.

The archive is a small table followed by encrypted file payloads.  Payloads are
XORed in complete four-byte words with ``ef cd ab 00``; the native encryptor's
tail path cancels itself, so the final one to three bytes are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

HEROES_PER_CAMP = 5
FEATURE_IMAGE_SIZE = 1734
FEATURE_VECTOR_SIZE = 3896
OBSERVATION_SIZE = FEATURE_IMAGE_SIZE + FEATURE_VECTOR_SIZE
ACTION_HEAD_SIZES = (13, 25, 42, 42, 39, 1)
POLICY_SIZE = sum(ACTION_HEAD_SIZES)

# The public SDK revision on which the native layout was measured end to end.
TESTED_CONFIG_SHA256 = "8416d8fa84ce73c9442fcef4145d995129e15a913bbbe01ed00204c8f30390b0"
TESTED_INTERFACE_SHA256 = "d5d9aa1341fbc5a0a37e9442837ccc6c48ef590c55294b149d39dc9577704518"

_XOR_KEY = bytes.fromhex("efcdab00")

MODEL_CONFIG = "AILab/ai_config/5v5/game_ai_model_config.txt"
COMBINE_CONFIG = "AILab/ai_config/5v5/tactics/feature/vec_combine_config.txt"
COMMON_CONFIG = "AILab/ai_config/5v5/tactics/feature/vec_feature_common.txt"
PRIVATE_CONFIG = "AILab/ai_config/5v5/tactics/feature/vec_feature_private.txt"
TACTICS_CONFIG = (
    "AILab/ai_config/5v5/tactics/"
    "multi_task_tactics_config_file_two_hand_action_minimap_union_model_rl.txt"
)
TEAM_CONFIG = "AILab/ai_config/5v5/tactics/team_model_conf.txt"


class ConfigArchiveError(ValueError):
    """The SDK archive or one of its expected stock fields is invalid."""


@dataclass(frozen=True)
class ConfigEntry:
    """One decrypted file in ``config.dat``."""

    name: str
    data: bytes


@dataclass(frozen=True)
class BuildReport:
    source: str
    target: str
    source_sha256: str
    target_sha256: str
    tested_source: bool
    files: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "tested_source": self.tested_source,
            "files": self.files,
            "heroes_per_camp": HEROES_PER_CAMP,
            "observation_size": OBSERVATION_SIZE,
            "action_head_sizes": list(ACTION_HEAD_SIZES),
            "policy_size": POLICY_SIZE,
        }


def _xor_payload(data: bytes) -> bytes:
    """Apply the native archive transform (symmetric)."""
    out = bytearray(data)
    word_bytes = len(out) - len(out) % 4
    for index in range(word_bytes):
        out[index] ^= _XOR_KEY[index % 4]
    return bytes(out)


def _take(blob: bytes, offset: int, size: int, label: str) -> Tuple[bytes, int]:
    end = offset + size
    if size < 0 or end > len(blob):
        raise ConfigArchiveError(
            f"truncated config.dat while reading {label}: need {size} bytes at {offset}, "
            f"file has {len(blob)}"
        )
    return blob[offset:end], end


def loads_archive(blob: bytes) -> List[ConfigEntry]:
    """Decode an encrypted ``config.dat`` byte string without extracting files."""
    if len(blob) < 4:
        raise ConfigArchiveError("config.dat is shorter than its entry count")
    count = struct.unpack_from("<I", blob, 0)[0]
    if count == 0 or count > 100_000:
        raise ConfigArchiveError(f"implausible config.dat entry count: {count}")
    offset = 4
    table: List[Tuple[str, int]] = []
    names = set()
    for index in range(count):
        raw, offset = _take(blob, offset, 4, f"entry {index} name length")
        name_len = struct.unpack("<I", raw)[0]
        raw_name, offset = _take(blob, offset, name_len, f"entry {index} name")
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigArchiveError(f"entry {index} has a non-UTF-8 name") from exc
        if not name or name in names:
            raise ConfigArchiveError(f"invalid or duplicate archive name: {name!r}")
        names.add(name)
        raw, offset = _take(blob, offset, 4, f"entry {name!r} payload length")
        table.append((name, struct.unpack("<I", raw)[0]))

    entries = []
    for name, size in table:
        encrypted, offset = _take(blob, offset, size, f"entry {name!r} payload")
        entries.append(ConfigEntry(name, _xor_payload(encrypted)))
    if offset != len(blob):
        raise ConfigArchiveError(
            f"config.dat has {len(blob) - offset} trailing bytes after {count} entries"
        )
    return entries


def load_archive(path: os.PathLike[str] | str) -> List[ConfigEntry]:
    return loads_archive(Path(path).read_bytes())


def dumps_archive(entries: Iterable[ConfigEntry]) -> bytes:
    """Encode decrypted entries in their supplied order."""
    materialized = list(entries)
    if not materialized:
        raise ConfigArchiveError("cannot write an empty config.dat")
    names = set()
    encoded = []
    for entry in materialized:
        if entry.name in names:
            raise ConfigArchiveError(f"duplicate archive name: {entry.name!r}")
        names.add(entry.name)
        try:
            name = entry.name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ConfigArchiveError(f"archive name is not UTF-8: {entry.name!r}") from exc
        encoded.append((name, _xor_payload(bytes(entry.data))))

    chunks = [struct.pack("<I", len(encoded))]
    for name, payload in encoded:
        chunks.extend((struct.pack("<I", len(name)), name, struct.pack("<I", len(payload))))
    chunks.extend(payload for _, payload in encoded)
    return b"".join(chunks)


def write_archive(
    entries: Iterable[ConfigEntry],
    path: os.PathLike[str] | str,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write an encrypted archive."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = dumps_archive(entries)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _decode_text(entries: Mapping[str, bytes], name: str) -> str:
    try:
        data = entries[name]
    except KeyError:
        raise ConfigArchiveError(f"SDK config is missing {name}") from None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigArchiveError(f"SDK config file {name} is not UTF-8") from exc


def _set_field(text: str, key: str, value: str, accepted: Sequence[str]) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}=([^\r\n]*)$")
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ConfigArchiveError(f"expected exactly one active {key}= line, found {len(matches)}")
    current = matches[0]
    if current not in set(accepted) | {value}:
        raise ConfigArchiveError(
            f"unexpected {key}={current!r}; expected one of {list(accepted)!r}"
        )
    return pattern.sub(f"{key}={value}", text, count=1)


def _hero_feature_names(count: int) -> str:
    return "".join(f"hero{i}_feature_img;hero{i}_feature_vec;" for i in range(count))


def _hero_feature_sizes(count: int) -> str:
    return ";".join([f"289;{FEATURE_VECTOR_SIZE}"] * count)


def _hero_probability_names(count: int) -> str:
    names = []
    for i in range(count):
        names.extend(f"hero{i}_fc2_label_{head}_result" for head in range(6))
        names.append(f"hero{i}_fc2_value_0_result")
    return ";".join(names)


def patch_5v5(entries: Iterable[ConfigEntry]) -> List[ConfigEntry]:
    """Return an idempotently patched copy of decoded SDK entries."""
    source = list(entries)
    by_name = {entry.name: entry.data for entry in source}
    if len(by_name) != len(source):
        raise ConfigArchiveError("config.dat contains duplicate names")

    text = _decode_text(by_name, MODEL_CONFIG)
    by_name[MODEL_CONFIG] = _set_field(text, "hero_cnt_each_camp", "5", ("3",)).encode()

    text = _decode_text(by_name, COMBINE_CONFIG)
    by_name[COMBINE_CONFIG] = _set_field(
        text, "num_unit", "10,20,6,20,0", ("6,20,6,20,0",)
    ).encode()

    text = _decode_text(by_name, COMMON_CONFIG)
    by_name[COMMON_CONFIG] = _set_field(
        text, "ten_hero_vec_size", "2410", ("1422", "2364")
    ).encode()

    text = _decode_text(by_name, PRIVATE_CONFIG)
    by_name[PRIVATE_CONFIG] = _set_field(text, "ten_hero_vec_size", "140", ("84",)).encode()

    text = _decode_text(by_name, TACTICS_CONFIG)
    text = _set_field(text, "game_type", "5v5", ("3v3",))
    text = _set_field(text, "feature_size_list", "1734;3896", ("1734;2852", "1734;3794"))
    text = _set_field(
        text,
        "feature_size_list_in_one_frame",
        "1734;3896",
        ("1734;2852", "1734;3794"),
    )
    by_name[TACTICS_CONFIG] = text.encode()

    text = _decode_text(by_name, TEAM_CONFIG)
    text = _set_field(
        text,
        "feature_name_list",
        _hero_feature_names(5),
        (_hero_feature_names(3),),
    )
    text = _set_field(
        text,
        "feature_size_list",
        _hero_feature_sizes(5),
        ("289;2852;289;2852;289;2852", "289;3794;289;3794;289;3794"),
    )
    text = _set_field(
        text,
        "prob_name_list",
        _hero_probability_names(5),
        (_hero_probability_names(3),),
    )
    # The native target/action contract stays 39-wide.  The apparently stray
    # width 3 is metadata and is deliberately not expanded to five.
    if not re.search(r"(?m)^output_size_list=13;25;42;42;3;39;1$", text):
        raise ConfigArchiveError("unexpected team output_size_list; native ABI is not supported")
    by_name[TEAM_CONFIG] = text.encode()

    patched = [ConfigEntry(entry.name, by_name[entry.name]) for entry in source]
    validate_5v5(patched)
    return patched


def _field(text: str, key: str) -> str:
    found = re.findall(rf"(?m)^{re.escape(key)}=([^\r\n]*)$", text)
    if len(found) != 1:
        raise ConfigArchiveError(f"expected one {key}= field, found {len(found)}")
    return found[0]


def validate_5v5(entries: Iterable[ConfigEntry]) -> None:
    """Raise if entries do not describe the verified native 5v5 contract."""
    by_name = {entry.name: entry.data for entry in entries}
    expected = {
        (MODEL_CONFIG, "hero_cnt_each_camp"): "5",
        (COMBINE_CONFIG, "num_unit"): "10,20,6,20,0",
        (COMMON_CONFIG, "ten_hero_vec_size"): "2410",
        (PRIVATE_CONFIG, "ten_hero_vec_size"): "140",
        (TACTICS_CONFIG, "game_type"): "5v5",
        (TACTICS_CONFIG, "feature_size_list"): "1734;3896",
        (TACTICS_CONFIG, "feature_size_list_in_one_frame"): "1734;3896",
        (TEAM_CONFIG, "feature_name_list"): _hero_feature_names(5),
        (TEAM_CONFIG, "feature_size_list"): _hero_feature_sizes(5),
        (TEAM_CONFIG, "prob_name_list"): _hero_probability_names(5),
        (TEAM_CONFIG, "output_size_list"): "13;25;42;42;3;39;1",
    }
    cache = {}
    for (name, key), value in expected.items():
        if name not in cache:
            cache[name] = _decode_text(by_name, name)
        actual = _field(cache[name], key)
        if actual != value:
            raise ConfigArchiveError(f"{name}: {key}={actual!r}, expected {value!r}")


def build_5v5_config(
    source: os.PathLike[str] | str,
    target: os.PathLike[str] | str,
    *,
    overwrite: bool = False,
) -> BuildReport:
    """Patch the installed 3v3 archive into a separate 5v5 archive."""
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if source_path == target_path:
        raise ValueError("source and target config.dat must be different files")
    source_blob = source_path.read_bytes()
    source_hash = hashlib.sha256(source_blob).hexdigest()
    patched = patch_5v5(loads_archive(source_blob))
    write_archive(patched, target_path, overwrite=overwrite)
    target_blob = target_path.read_bytes()
    validate_5v5(loads_archive(target_blob))
    return BuildReport(
        source=str(source_path),
        target=str(target_path),
        source_sha256=source_hash,
        target_sha256=hashlib.sha256(target_blob).hexdigest(),
        tested_source=source_hash == TESTED_CONFIG_SHA256,
        files=len(patched),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="installed hok.hok3v3/config.dat")
    parser.add_argument("target", help="new patched config.dat")
    parser.add_argument("--force", action="store_true", help="replace target atomically")
    args = parser.parse_args(argv)
    report = build_5v5_config(args.source, args.target, overwrite=args.force)
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
