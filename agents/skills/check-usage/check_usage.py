#!/usr/bin/env python3
"""Parse the rate limit status from the output of the get_session tool of Claude Code Remote.

Reads the JSON returned by mcp__claude-code-remote__get_session from a file or stdin
and prints the rate limit information as "key: value" lines.
This uses only the Python standard library so that it can be run without uv.
"""

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sys
import typing as tp


def format_duration(delta: timedelta) -> str:
    """Format a duration as e.g. "2h 51m"."""
    minutes = int(delta.total_seconds() // 60)
    if minutes < 0:
        return "already passed"
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    return f"{hours}h {minutes}m"


def yes_no(value: object) -> str:
    """Format a boolean as "yes" or "no", and other values as is."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def get_metadata(data: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Get the external_metadata of a session, with or without the "ccr" wrapper."""
    session = data.get("ccr", data)
    return session.get("external_metadata") or {}


def parse(data: dict[str, tp.Any], now: datetime | None = None) -> list[tuple[str, str]]:
    """Parse the rate limit information from the output of get_session to key-value pairs."""
    if now is None:
        now = datetime.now(UTC)
    metadata = get_metadata(data)
    info = metadata.get("rate_limit_info")
    if not info:
        return [("error", "external_metadata.rate_limit_info not found in the input")]

    status = info.get("status", "unknown")
    lines = [
        ("status", str(status)),
        ("limited", yes_no(status != "allowed")),
        ("limit_window", str(info.get("rateLimitType", "unknown"))),
        ("using_overage", yes_no(info.get("isUsingOverage", "unknown"))),
    ]
    resets_at = info.get("resetsAt")
    if isinstance(resets_at, (int, float)):
        reset = datetime.fromtimestamp(resets_at, UTC)
        lines += [
            ("resets_at_utc", reset.strftime("%Y-%m-%d %H:%M UTC")),
            ("resets_at_local", reset.astimezone().strftime("%Y-%m-%d %H:%M %Z (UTC%z)")),
            ("resets_in", format_duration(reset - now)),
        ]
    # Report any fields that this script does not know about yet.
    known = {"status", "rateLimitType", "isUsingOverage", "resetsAt"}
    lines += [(f"other.{key}", json.dumps(value)) for key, value in info.items() if key not in known]

    cost = (metadata.get("usage") or {}).get("cost_usd")
    if isinstance(cost, (int, float)):
        lines.append(("session_cost_usd", f"{cost:.2f}"))
    context = metadata.get("context_usage") or {}
    used, maximum = context.get("used_tokens"), context.get("max_tokens")
    if isinstance(used, int) and isinstance(maximum, int) and maximum > 0:
        lines.append(("session_context", f"{used}/{maximum} tokens ({100 * used / maximum:.1f} %)"))
    return lines


def main() -> int:
    """Read the get_session output and print the parsed rate limit information."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="JSON file to read (default: stdin)")
    args = parser.parse_args()
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"error: the input is not valid JSON: {e}")
        return 1
    lines = parse(data)
    for key, value in lines:
        print(f"{key}: {value}")
    return 1 if lines[0][0] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
