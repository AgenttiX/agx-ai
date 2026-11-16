"""Tests for the hook that records the claude.ai links of Claude Code sessions."""

import io
import json
from pathlib import Path

import pytest

from agx_ai.claude import session_urls as urls


def write_process_files(directory: Path) -> Path:
    directory.mkdir()
    files = {
        1: {"sessionId": "a", "bridgeSessionId": "session_old", "updatedAt": 1},
        2: {"sessionId": "a", "bridgeSessionId": "session_new", "updatedAt": 2},
        3: {"sessionId": "b", "bridgeSessionId": "bad id/"},
        4: {"sessionId": "c"},
    }
    for pid, data in files.items():
        (directory / f"{pid}.json").write_text(json.dumps(data), encoding="utf-8")
    (directory / "5.json").write_text("not json", encoding="utf-8")
    (directory / "6.json").write_text("[]", encoding="utf-8")
    return directory


def test_running_remote_ids(tmp_path: Path) -> None:
    assert urls.running_remote_ids(write_process_files(tmp_path / "sessions")) == {"a": "session_new"}
    assert urls.running_remote_ids(tmp_path / "missing") == {}


def test_record_and_session_urls(tmp_path: Path) -> None:
    sessions = write_process_files(tmp_path / "sessions")
    path = tmp_path / "session-urls.json"
    assert urls.recorded_remote_ids(path) == {}
    assert not urls.record("c", sessions, path)
    assert not path.exists()
    path.write_text(json.dumps({"old": "session_x", "bad": 1}), encoding="utf-8")
    assert urls.record("a", sessions, path)
    assert urls.recorded_remote_ids(path) == {"a": "session_new", "old": "session_x"}
    mtime = path.stat().st_mtime_ns
    assert not urls.record("a", sessions, path)
    assert path.stat().st_mtime_ns == mtime
    assert urls.session_urls(tmp_path / "none", path) == {
        "a": "https://claude.ai/code/session_new", "old": "https://claude.ai/code/session_x"}
    assert not urls.record("a", sessions, tmp_path / "missing-dir" / "x.json")
    assert list(tmp_path.glob("*.tmp")) == []


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(urls, "SESSIONS_DIR", write_process_files(tmp_path / "sessions"))
    monkeypatch.setattr(urls, "SESSION_URLS_FILE", tmp_path / "session-urls.json")
    for stdin in ("not json", "[]", json.dumps({"session_id": "a", "hook_event_name": "SessionStart"})):
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
        urls.main()
    assert capsys.readouterr().out == ""
    assert urls.recorded_remote_ids(tmp_path / "session-urls.json") == {"a": "session_new"}
