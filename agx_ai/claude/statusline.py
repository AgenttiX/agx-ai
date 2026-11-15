"""Claude Code status line: context, 5-hour limit and weekly limit usage.

Reads the status line JSON from stdin and prints one colored line.
Also saves the rate limits to USAGE_FILE so that the check-usage skill can read them.
Uses only the Python standard library, so it can be run with plain `python3`.
"""

import contextlib
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import typing as tp

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

USAGE_FILE = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "usage.json"


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


def save_usage(data: tp.Any, path: Path | None = None, now: datetime | None = None) -> bool:
    """Save the rate limits with a timestamp to a JSON file. Return whether the file was written.

    The file is not written if the input has no rate limits, so that the previous data is kept.
    """
    if not isinstance(data, dict):
        return False
    rate_limits = data.get("rate_limits")
    if not isinstance(rate_limits, dict) or not rate_limits:
        return False
    if path is None:
        path = USAGE_FILE
    if now is None:
        now = datetime.now(UTC)
    usage = {
        "obtained_at": now.isoformat(timespec="seconds"),
        "session_id": data.get("session_id"),
        "rate_limits": rate_limits,
    }
    # Write to a temporary file first so that readers never see a partially written file.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        # The status line has to work even if the file cannot be written.
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        return False
    return True


def main() -> None:
    """Read the status line JSON from stdin, save the rate limits and print the status line."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        data = {}
    save_usage(data)
    print(format_statusline(data))


if __name__ == "__main__":
    main()
