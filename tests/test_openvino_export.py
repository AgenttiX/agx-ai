"""Tests of agx_ai.openvino.export, which skips up-to-date conversions and replaces the old ones atomically."""

import json
from pathlib import Path

import pytest

from agx_ai.openvino.export import MARKER, convert_once, is_up_to_date

INFO = {"revision": "abc", "versions": {"openvino": "2026.4.1"}}


def write_model(path: Path) -> None:
    (path / "model.xml").write_text("model", encoding="utf-8")


def test_convert_once_writes_output_and_marker(tmp_path: Path) -> None:
    output = tmp_path / "model"
    assert convert_once(output, INFO, write_model)
    assert (output / "model.xml").read_text(encoding="utf-8") == "model"
    assert json.loads((output / MARKER).read_text(encoding="utf-8")) == INFO
    assert not output.with_name("model.tmp").exists()


def test_convert_once_skips_up_to_date(tmp_path: Path) -> None:
    output = tmp_path / "model"
    convert_once(output, INFO, write_model)
    calls = []
    assert not convert_once(output, INFO, calls.append)
    assert calls == []


def test_convert_once_replaces_outdated(tmp_path: Path) -> None:
    output = tmp_path / "model"
    convert_once(output, INFO, write_model)
    (output / "stale.bin").write_text("old", encoding="utf-8")
    new_info = {**INFO, "revision": "def"}
    assert convert_once(output, new_info, write_model)
    assert not (output / "stale.bin").exists()
    assert is_up_to_date(output, new_info)
    assert not is_up_to_date(output, INFO)


def test_failed_conversion_keeps_old_output(tmp_path: Path) -> None:
    output = tmp_path / "model"
    convert_once(output, INFO, write_model)

    def fail(path: Path) -> None:
        raise RuntimeError("conversion failed")

    with pytest.raises(RuntimeError):
        convert_once(output, {**INFO, "revision": "def"}, fail)
    assert is_up_to_date(output, INFO)


def test_is_up_to_date_with_missing_or_invalid_marker(tmp_path: Path) -> None:
    assert not is_up_to_date(tmp_path / "missing", INFO)
    (tmp_path / MARKER).write_text("{not json", encoding="utf-8")
    assert not is_up_to_date(tmp_path, INFO)
