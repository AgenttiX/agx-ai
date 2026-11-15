#!/usr/bin/env python3
"""Print the Claude usage limit status as "key: value" lines.

Reads either the usage file saved by the status line (agx_ai/claude/statusline.py, by default),
or the JSON returned by the mcp__claude-code-remote__get_session tool (from a file or stdin).
The format is detected from the content.
This uses only the Python standard library so that it can be run without uv.
"""

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sys
import typing as tp

USAGE_FILE = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "usage.json"


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


def format_reset(prefix: str, resets_at: object, now: datetime) -> list[tuple[str, str]]:
    """Format a reset time given in Unix seconds."""
    if isinstance(resets_at, bool) or not isinstance(resets_at, (int, float)):
        return []
    reset = datetime.fromtimestamp(resets_at, UTC)
    return [
        (f"{prefix}resets_at_utc", reset.strftime("%Y-%m-%d %H:%M UTC")),
        (f"{prefix}resets_at_local", reset.astimezone().strftime("%Y-%m-%d %H:%M %Z (UTC%z)")),
        (f"{prefix}resets_in", format_duration(reset - now)),
    ]


def parse_usage_file(data: dict[str, tp.Any], now: datetime) -> list[tuple[str, str]]:
    """Parse the usage file saved by the status line."""
    lines = [("source", "status line usage file")]
    obtained_at = data.get("obtained_at")
    if isinstance(obtained_at, str):
        obtained = datetime.fromisoformat(obtained_at)
        lines += [
            ("obtained_at_local", obtained.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")),
            ("data_age", format_duration(now - obtained)),
        ]
    for name, limit in data["rate_limits"].items():
        if not isinstance(limit, dict):
            continue
        used = limit.get("used_percentage")
        if isinstance(used, (int, float)) and not isinstance(used, bool):
            lines.append((f"{name}.used", f"{used:.0f} %"))
        resets_at = limit.get("resets_at")
        lines += format_reset(f"{name}.", resets_at, now)
        if isinstance(resets_at, (int, float)) and datetime.fromtimestamp(resets_at, UTC) < now:
            lines.append((f"{name}.note", "the window has reset since the data was obtained"))
        known = {"used_percentage", "resets_at"}
        lines += [(f"{name}.{key}", json.dumps(value)) for key, value in limit.items() if key not in known]
    return lines


def parse_session(data: dict[str, tp.Any], now: datetime) -> list[tuple[str, str]]:
    """Parse the output of the get_session tool."""
    session = data.get("ccr", data)
    metadata = session.get("external_metadata") or {}
    info = metadata.get("rate_limit_info")
    if not info:
        return [("error", "external_metadata.rate_limit_info not found in the input")]

    status = info.get("status", "unknown")
    lines = [
        ("source", "get_session"),
        ("status", str(status)),
        ("limited", yes_no(status != "allowed")),
        ("limit_window", str(info.get("rateLimitType", "unknown"))),
        ("using_overage", yes_no(info.get("isUsingOverage", "unknown"))),
    ]
    lines += format_reset("", info.get("resetsAt"), now)
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


def parse(data: object, now: datetime | None = None) -> list[tuple[str, str]]:
    """Parse the usage file or the get_session output to key-value pairs."""
    if now is None:
        now = datetime.now(UTC)
    if not isinstance(data, dict):
        return [("error", "the input is not a JSON object")]
    if isinstance(data.get("rate_limits"), dict):
        return parse_usage_file(data, now)
    return parse_session(data, now)


def main() -> int:
    """Read the input and print the parsed usage information."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "file", nargs="?", default=str(USAGE_FILE),
        help=f"JSON file to read, or - for stdin (default: {USAGE_FILE})",
    )
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: cannot read the input: {e}")
        return 1
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
