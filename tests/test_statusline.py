"""Tests for the Claude Code status line."""

from datetime import UTC, datetime
import io
import json
from pathlib import Path

import pytest

from agx_ai.claude import statusline


@pytest.mark.parametrize(
    ("percentage", "color"),
    [
        (0, statusline.GREEN),
        (69.9, statusline.GREEN),
        (70, statusline.YELLOW),
        (89.9, statusline.YELLOW),
        (90, statusline.RED),
        (100, statusline.RED),
    ],
)
def test_color_for(percentage: float, color: str) -> None:
    assert statusline.color_for(percentage) == color


def test_format_statusline_all() -> None:
    data = {
        "context_window": {"used_percentage": 42.4},
        "rate_limits": {"five_hour": {"used_percentage": 75}, "seven_day": {"used_percentage": 95.0}},
    }
    assert statusline.format_statusline(data) == (
        f"ctx {statusline.GREEN}42%{statusline.RESET} | "
        f"5h {statusline.YELLOW}75%{statusline.RESET} | "
        f"7d {statusline.RED}95%{statusline.RESET}"
    )


def test_format_statusline_missing() -> None:
    assert statusline.format_statusline({}) == ""
    assert statusline.format_statusline({"context_window": {"used_percentage": None}, "rate_limits": "x"}) == ""
    assert statusline.format_statusline({"context_window": {"used_percentage": True}}) == ""
    assert statusline.format_statusline([]) == ""


def test_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr(statusline, "USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"context_window": {"used_percentage": 10}})))
    statusline.main()
    assert capsys.readouterr().out == f"ctx {statusline.GREEN}10%{statusline.RESET}\n"

    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    statusline.main()
    assert capsys.readouterr().out == "\n"


def test_save_usage(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    now = datetime(2026, 11, 15, 13, 0, tzinfo=UTC)
    rate_limits = {"five_hour": {"used_percentage": 42.4, "resets_at": 1794757800}}
    assert statusline.save_usage({"session_id": "abc", "rate_limits": rate_limits}, path, now)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "obtained_at": "2026-11-15T13:00:00+00:00",
        "session_id": "abc",
        "rate_limits": rate_limits,
    }
    # Inputs without rate limits keep the previous data.
    for data in ({}, {"rate_limits": None}, {"rate_limits": {}}, []):
        assert not statusline.save_usage(data, path, now)
    assert json.loads(path.read_text(encoding="utf-8"))["rate_limits"] == rate_limits
    # Write errors are ignored.
    assert not statusline.save_usage({"rate_limits": rate_limits}, tmp_path / "missing" / "usage.json", now)
    assert list(tmp_path.iterdir()) == [path]
