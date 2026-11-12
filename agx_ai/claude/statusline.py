"""Claude Code status line: context, 5-hour limit and weekly limit usage.

Reads the status line JSON from stdin and prints one colored line.
Uses only the Python standard library, so it can be run with plain `python3`.
"""

import json
import sys
import typing as tp

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def color_for(percentage: float) -> str:
    """Return the ANSI color for a usage percentage."""
    if percentage >= 90:
        return RED
    if percentage >= 70:
        return YELLOW
    return GREEN


def _get_percentage(data: tp.Any, *keys: str) -> float | None:
    """Get a nested numeric value from the JSON data, or None if missing."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        return None
    return float(data)


def format_item(label: str, percentage: float | None) -> str | None:
    """Format a single colored item, or None if the value is unavailable."""
    if percentage is None:
        return None
    return f"{label} {color_for(percentage)}{percentage:.0f}%{RESET}"


def format_statusline(data: tp.Any) -> str:
    """Build the status line text from the Claude Code JSON input."""
    items = (
        format_item("ctx", _get_percentage(data, "context_window", "used_percentage")),
        format_item("5h", _get_percentage(data, "rate_limits", "five_hour", "used_percentage")),
        format_item("7d", _get_percentage(data, "rate_limits", "seven_day", "used_percentage")),
    )
    return " | ".join(item for item in items if item is not None)


def main() -> None:
    """Read the status line JSON from stdin and print the status line."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        data = {}
    print(format_statusline(data))


if __name__ == "__main__":
    main()
