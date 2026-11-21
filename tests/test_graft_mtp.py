"""Tests of vllm/graft_mtp.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
from types import ModuleType

import pytest

from agx_ai.hostname import REPO_DIR

_spec = importlib.util.spec_from_file_location("graft_mtp", REPO_DIR / "vllm" / "graft_mtp.py")
assert _spec is not None
assert _spec.loader is not None
graft_mtp: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graft_mtp)


def write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    """A minimal safetensors file of uint8 vectors."""
    header: dict = {}
    offset = 0
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(tensors.values()))


def read_safetensors(path: Path) -> dict[str, bytes]:
    """The tensors of a safetensors file as bytes."""
    header, start = graft_mtp.read_header(path)
    data = path.read_bytes()
    return {k: data[start + v["data_offsets"][0]:start + v["data_offsets"][1]] for k, v in header.items()
            if k != "__metadata__"}


def make_checkpoint(path: Path, shards: dict[str, dict[str, bytes]]) -> None:
    path.mkdir()
    weight_map = {}
    for shard, tensors in shards.items():
        write_safetensors(path / shard, tensors)
        weight_map.update(dict.fromkeys(tensors, shard))
    (path / graft_mtp.INDEX).write_text(json.dumps({"metadata": {"total_size": 10}, "weight_map": weight_map}))
    (path / "config.json").write_text("{}")


def test_graft(tmp_path: Path) -> None:
    make_checkpoint(tmp_path / "target", {"a.safetensors": {"model.w": b"\x01\x02"}})
    make_checkpoint(tmp_path / "base", {
        "a.safetensors": {"model.w": b"\x09", "mtp.fc.weight": b"\x03\x04\x05"},
        "b.safetensors": {"mtp.norm.weight": b"\x06"},
    })
    out = tmp_path / "out"
    assert graft_mtp.graft(tmp_path / "target", tmp_path / "base", out) == 2
    assert (out / "config.json").is_symlink()
    assert (out / "a.safetensors").resolve() == (tmp_path / "target" / "a.safetensors").resolve()
    assert read_safetensors(out / graft_mtp.MTP_SHARD) == {"mtp.fc.weight": b"\x03\x04\x05", "mtp.norm.weight": b"\x06"}
    index = json.loads((out / graft_mtp.INDEX).read_text())
    assert index["weight_map"] == {"model.w": "a.safetensors", "mtp.fc.weight": graft_mtp.MTP_SHARD,
                                   "mtp.norm.weight": graft_mtp.MTP_SHARD}
    assert index["metadata"]["total_size"] == 14
    # The data section of the new shard is aligned to 8 bytes.
    assert graft_mtp.read_header(out / graft_mtp.MTP_SHARD)[1] % 8 == 0


def test_graft_refuses_target_with_mtp(tmp_path: Path) -> None:
    make_checkpoint(tmp_path / "target", {"a.safetensors": {"mtp.fc.weight": b"\x01"}})
    make_checkpoint(tmp_path / "base", {"a.safetensors": {"mtp.fc.weight": b"\x02"}})
    with pytest.raises(ValueError, match="already has"):
        graft_mtp.graft(tmp_path / "target", tmp_path / "base", tmp_path / "out")


def test_hf_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snap = tmp_path / "models--org--name" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert graft_mtp.hf_snapshot("org/name") == snap
    with pytest.raises(FileNotFoundError):
        graft_mtp.hf_snapshot("org/missing")
