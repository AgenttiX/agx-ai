"""Tests for the parser of the check-usage skill."""

from datetime import UTC, datetime
import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).parent.parent / "agents" / "skills" / "check-usage" / "check_usage.py"
_spec = importlib.util.spec_from_file_location("check_usage", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
check_usage: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_usage)

NOW: datetime = datetime(2026, 11, 15, 13, 5, tzinfo=UTC)
# 2026-11-15 15:50 UTC
RESETS_AT = 1794757800


def test_parse_session() -> None:
    data = {
        "ccr": {
            "external_metadata": {
                "context_usage": {"max_tokens": 1000000, "used_tokens": 88102},
                "rate_limit_info": {
                    "isUsingOverage": False,
                    "rateLimitType": "five_hour",
                    "resetsAt": RESETS_AT,
                    "status": "allowed",
                    "newField": 1,
                },
                "usage": {"cost_usd": 0.37009474},
            }
        }
    }
    lines = dict(check_usage.parse(data, NOW))
    assert lines["source"] == "get_session"
    assert lines["status"] == "allowed"
    assert lines["limited"] == "no"
    assert lines["limit_window"] == "five_hour"
    assert lines["using_overage"] == "no"
    assert lines["resets_at_utc"] == "2026-11-15 15:50 UTC"
    assert lines["resets_in"] == "2h 45m"
    assert lines["other.newField"] == "1"
    assert lines["session_cost_usd"] == "0.37"
    assert lines["session_context"] == "88102/1000000 tokens (8.8 %)"


def test_parse_usage_file() -> None:
    data = {
        "obtained_at": "2026-11-15T13:00:00+00:00",
        "session_id": "abc",
        "rate_limits": {
            "five_hour": {"used_percentage": 42.4, "resets_at": RESETS_AT},
            "seven_day": {"used_percentage": 12, "resets_at": RESETS_AT - 86400},
        },
    }
    lines = dict(check_usage.parse(data, NOW))
    assert lines["source"] == "status line usage file"
    assert lines["data_age"] == "0h 5m"
    assert lines["five_hour.used"] == "42 %"
    assert lines["five_hour.resets_in"] == "2h 45m"
    assert "five_hour.note" not in lines
    assert lines["seven_day.used"] == "12 %"
    assert lines["seven_day.resets_in"] == "already passed"
    assert "seven_day.note" in lines


def test_parse_errors() -> None:
    assert check_usage.parse({"ccr": {}}, NOW)[0][0] == "error"
    assert check_usage.parse([], NOW)[0][0] == "error"
