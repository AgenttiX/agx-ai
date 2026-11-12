"""Tests for the Claude Code status line."""

import io
import json

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


def test_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"context_window": {"used_percentage": 10}})))
    statusline.main()
    assert capsys.readouterr().out == f"ctx {statusline.GREEN}10%{statusline.RESET}\n"

    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    statusline.main()
    assert capsys.readouterr().out == "\n"
