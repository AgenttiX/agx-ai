"""Record the claude.ai links of Claude Code sessions, for linking to them later (e.g. in claude2pdf).

The transcripts in ``~/.claude/projects/`` do not contain the ID of a session on claude.ai.
While a session connected to claude.ai (Remote Control, the desktop or mobile app) is running,
Claude Code has it as ``bridgeSessionId`` in its process file ``~/.claude/sessions/<pid>.json``,
but removes it when the session ends.
Run as a Claude Code hook (SessionStart and UserPromptSubmit in ``claude/settings.json``),
this script copies the ID of the current session to SESSION_URLS_FILE,
a JSON object of local session ID: claude.ai session ID.
It reads the hook JSON from stdin and prints nothing, as the output of a UserPromptSubmit hook would be added
to the context. Uses only the Python standard library, so it can be run with plain `python3`.
"""

import contextlib
import json
import os
from pathlib import Path
import re
import sys
import typing as tp

CONFIG_DIR: Path = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
SESSIONS_DIR: Path = CONFIG_DIR / "sessions"
SESSION_URLS_FILE: Path = CONFIG_DIR / "session-urls.json"
SESSION_URL: str = "https://claude.ai/code/{}"
REMOTE_ID_RE: re.Pattern[str] = re.compile(r"\w+")


def _read_json(path: Path) -> tp.Any:
    """Read a JSON file, or return None if it cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def running_remote_ids(sessions_dir: Path = SESSIONS_DIR) -> dict[str, str]:
    """Return the claude.ai session IDs of the running sessions from their process files, by local session ID.

    If a session has several process files (e.g. it was resumed), the most recently updated one is used.
    """
    found: dict[str, tuple[float, str]] = {}
    for path in sessions_dir.glob("*.json"):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        local, remote = data.get("sessionId"), data.get("bridgeSessionId")
        if not (isinstance(local, str) and isinstance(remote, str) and REMOTE_ID_RE.fullmatch(remote)):
            continue
        updated = data.get("updatedAt") or data.get("startedAt") or 0
        updated = updated if isinstance(updated, (int, float)) else 0
        if local not in found or updated >= found[local][0]:
            found[local] = (updated, remote)
    return {local: remote for local, (_, remote) in found.items()}


def recorded_remote_ids(path: Path = SESSION_URLS_FILE) -> dict[str, str]:
    """Return the claude.ai session IDs recorded by :func:`record`, by local session ID."""
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and REMOTE_ID_RE.fullmatch(v)}


def session_urls(sessions_dir: Path = SESSIONS_DIR, path: Path = SESSION_URLS_FILE) -> dict[str, str]:
    """Return the claude.ai links of the recorded and the running sessions, by local session ID."""
    ids = recorded_remote_ids(path) | running_remote_ids(sessions_dir)
    return {local: SESSION_URL.format(remote) for local, remote in ids.items()}


def record(session_id: str, sessions_dir: Path = SESSIONS_DIR, path: Path = SESSION_URLS_FILE) -> bool:
    """Save the claude.ai session ID of a running session to the file. Return whether the file was written.

    The file is only written when the ID is new or has changed.
    """
    remote = running_remote_ids(sessions_dir).get(session_id)
    if remote is None:
        return False
    recorded = recorded_remote_ids(path)
    if recorded.get(session_id) == remote:
        return False
    recorded[session_id] = remote
    # Write to a temporary file first so that readers never see a partially written file.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        # The hook must not disturb the session even if the file cannot be written.
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        return False
    return True


def main() -> None:
    """Read the hook JSON from stdin and record the claude.ai session ID of the session."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(data, dict) and isinstance(data.get("session_id"), str):
        record(data["session_id"], SESSIONS_DIR, SESSION_URLS_FILE)


if __name__ == "__main__":
    main()
