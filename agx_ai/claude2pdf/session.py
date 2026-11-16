"""Reading Claude Code session transcripts (``~/.claude/projects/<project>/<session id>.jsonl``).

Each line of a transcript is a JSON record. The records that matter here are

* ``user``: a prompt typed by the user (string or a list of text and image
  blocks), a slash command, tool results, or text injected by Claude Code
  (``isMeta``: skill bodies, subagent reports, task notifications, ...),
* ``assistant``: one content block (text, thinking or tool use) of a model
  response; a response with several blocks is split over several records
  that share ``message.id``,
* ``attachment``: context injected by Claude Code; only ``queued_command``
  attachments are used here, as they hold the prompts typed while Claude was
  working,
* ``system``: notices such as compaction boundaries,
* ``custom-title`` / ``agent-name`` / ``summary``: the name of the session.

The records form a tree through ``parentUuid`` (``logicalParentUuid`` across
compaction boundaries). Rewinding a conversation and resending a prompt makes
a new branch; the active branch is the one that ends at the last message.
Parallel tool calls also branch the tree, so the siblings that belong to the
messages and tool calls of the active branch are included as well.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import datetime as dt
import functools
import json
from pathlib import Path
import re
import sys
from typing import Any

from agx_ai.claude import session_urls as urls
from agx_ai.convert.util import DATETIME_FORMAT

PROJECTS_DIR: Path = Path("~/.claude/projects").expanduser()

COMMAND_NAME_RE: re.Pattern[str] = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.S)
COMMAND_ARGS_RE: re.Pattern[str] = re.compile(r"<command-args>(.*?)</command-args>", re.S)
LOCAL_STDOUT_RE: re.Pattern[str] = re.compile(
    r"<local-command-(?:stdout|stderr)>(.*?)</local-command-(?:stdout|stderr)>", re.S)
XML_TAG_RE: re.Pattern[str] = re.compile(r"<(/?)([\w-]+)>")


@dataclass
class Event:
    """One item of the conversation, in the order in which it happened.

    ``kind`` is one of ``prompt`` (typed by the user, also slash commands and
    prompts queued during a turn), ``text``, ``thinking``, ``tool_use``,
    ``tool_result``, ``interrupt``, ``error`` and ``meta`` (text injected by
    Claude Code, with ``label`` saying what it is).
    """

    kind: str
    timestamp: str | None = None
    text: str = ""
    label: str = ""
    images: list[dict] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    """Metadata of a session for listing and for the title page."""

    path: Path
    session_id: str
    title: str = ""
    first_prompt: str = ""
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    prompts: int = 0
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    models: list[str] = field(default_factory=list)
    url: str = ""

    @property
    def name(self) -> str:
        """The custom title of the session, or else a preview of its first prompt."""
        return self.title or preview(re.sub(r"[`*]", "", self.first_prompt)) or "(no prompts)"


def project_dir(repo: Path) -> Path:
    """Return the directory of ``~/.claude/projects`` that holds the sessions of a repository.

    Claude Code names it after the absolute path of the working directory,
    with every character other than a letter or a digit replaced by ``-``.
    """
    return PROJECTS_DIR / re.sub(r"[^A-Za-z0-9]", "-", str(repo.expanduser().resolve()))


@functools.cache
def session_urls(sessions_dir: Path, urls_file: Path) -> dict[str, str]:
    """Return the claude.ai links of the sessions, by local session ID.

    The transcripts do not contain the ID of a session on claude.ai. It is read
    from the running sessions and from the file of the hook that records it
    (see :mod:`agx_ai.claude.session_urls`).
    """
    return urls.session_urls(sessions_dir, urls_file)


def parse_timestamp(ts: str | None) -> dt.datetime | None:
    """Parse an ISO 8601 timestamp of a record into a local aware datetime."""
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None


def fmt_time(ts: str | dt.datetime | None, fmt: str = DATETIME_FORMAT) -> str:
    """Format a timestamp in local time with the strftime format ``fmt`` (default ``YYYY-MM-DD HH:MM``)."""
    t = parse_timestamp(ts) if isinstance(ts, str) or ts is None else ts
    return t.strftime(fmt) if t and fmt else ""


def preview(text: str, limit: int = 70) -> str:
    """Return the start of a text on one line."""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def read_records(path: Path) -> list[dict]:
    """Read the JSON records of a transcript, skipping lines that are not valid JSON."""
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: {path}:{i}: invalid JSON, skipped", file=sys.stderr)
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _is_message(rec: dict) -> bool:
    return rec.get("type") in ("user", "assistant") and isinstance(rec.get("message"), dict) and bool(rec.get("uuid"))


def _tool_use_ids(rec: dict) -> set[str]:
    content = rec["message"].get("content")
    if not isinstance(content, list):
        return set()
    return {c["id"] for c in content if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("id")}


def _tool_result_ids(rec: dict) -> set[str]:
    content = rec["message"].get("content")
    if not isinstance(content, list):
        return set()
    return {c["tool_use_id"] for c in content
            if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("tool_use_id")}


def active_branch(records: list[dict]) -> list[dict]:
    """Return the records of the active conversation branch in file order.

    Records without a ``uuid`` (titles, bookkeeping) and records of abandoned
    branches (rewound prompts) are left out.
    """
    by_uuid = {r["uuid"]: r for r in records if r.get("uuid")}
    messages = [r for r in records if _is_message(r)]
    if not messages:
        return []
    keep: set[str] = set()
    uuid: str | None = messages[-1]["uuid"]
    while uuid and uuid in by_uuid and uuid not in keep:
        keep.add(uuid)
        rec = by_uuid[uuid]
        uuid = rec.get("parentUuid") or rec.get("logicalParentUuid")
    # Parallel tool calls branch the tree: add the other blocks of the kept
    # responses and the results of their tool calls until nothing changes.
    while True:
        kept = [by_uuid[u] for u in keep if _is_message(by_uuid[u])]
        msg_ids = {r["message"].get("id") for r in kept if r["type"] == "assistant"} - {None}
        tool_ids = set().union(*(_tool_use_ids(r) for r in kept if r["type"] == "assistant"))
        added = {
            r["uuid"] for r in messages
            if r["uuid"] not in keep and (
                (r["type"] == "assistant" and r["message"].get("id") in msg_ids)
                or (r["type"] == "user" and _tool_result_ids(r) & tool_ids))
        }
        if not added:
            break
        keep |= added
    return [r for r in records if r.get("uuid") in keep]


def _strip_tags(text: str) -> str:
    return XML_TAG_RE.sub("", text).strip()


def _user_text(content: Any) -> tuple[str, list[dict]]:
    """Return the text and the image blocks of the content of a user message."""
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "image":
            images.append(block)
    return "\n\n".join(texts), images


def _meta_label(text: str, rec: dict) -> str | None:  # noqa: PLR0911
    """Classify text injected by Claude Code; None for text typed by the user."""
    stripped = text.lstrip()
    origin = (rec.get("origin") or {}).get("kind")
    if rec.get("isCompactSummary"):
        return "Compaction summary"
    if stripped.startswith("<task-notification>") or origin == "task-notification":
        return "Task notification"
    if stripped.startswith("Another Claude session sent a message") or origin == "peer":
        return "Subagent report" if "[Subagent hand-back]" in text else "Message from another session"
    if stripped.startswith("Base directory for this skill"):
        return "Skill instructions"
    if stripped.startswith(("<local-command-stdout>", "<local-command-stderr>")):
        return "Command output"
    if rec.get("isMeta"):
        return "Claude Code"
    return None


def user_events(rec: dict) -> list[Event]:  # noqa: PLR0911
    """Turn a ``user`` record into events."""
    ts = rec.get("timestamp")
    content = rec["message"].get("content")
    if isinstance(content, list) and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content):
        events = []
        for c in content:
            if not isinstance(c, dict) or c.get("type") != "tool_result":
                continue
            result = c.get("content")
            text, images = _user_text(result) if isinstance(result, list) else (str(result or ""), [])
            events.append(Event("tool_result", ts, text, images=images,
                                data={"tool_use_id": c.get("tool_use_id"), "is_error": bool(c.get("is_error"))}))
        return events
    text, images = _user_text(content)
    if text.lstrip().startswith("<local-command-caveat>"):
        return []
    if text.lstrip().startswith("[Request interrupted by user"):
        return [Event("interrupt", ts, text.strip().strip("[]"))]
    name = COMMAND_NAME_RE.search(text)
    if name and text.lstrip().startswith(("<command-name>", "<command-message>")):
        args = COMMAND_ARGS_RE.search(text)
        command = name.group(1) + (" " + args.group(1).strip() if args and args.group(1).strip() else "")
        return [Event("prompt", ts, command, images=images, data={"command": True})]
    label = _meta_label(text, rec)
    if label is not None:
        if label == "Command output":
            text = "\n".join(m.group(1) for m in LOCAL_STDOUT_RE.finditer(text))
        elif label == "Task notification":
            text = _strip_tags(text)
        if not text.strip() and not images:
            return []
        return [Event("meta", ts, text, label=label, images=images)]
    if not text.strip() and not images:
        return []
    return [Event("prompt", ts, text, images=images)]


def assistant_events(rec: dict) -> list[Event]:
    """Turn an ``assistant`` record into events."""
    ts = rec.get("timestamp")
    msg = rec["message"]
    model = msg.get("model") or ""
    content = msg.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    events = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text", "").strip():
            ev_kind = "error" if rec.get("isApiErrorMessage") else "text"
            events.append(Event(ev_kind, ts, block["text"], data={"model": model}))
        elif kind == "thinking" and block.get("thinking", "").strip():
            events.append(Event("thinking", ts, block["thinking"], data={"model": model}))
        elif kind == "tool_use":
            events.append(Event("tool_use", ts, label=block.get("name", ""),
                                data={"id": block.get("id"), "input": block.get("input") or {}, "model": model}))
    return events


def system_events(rec: dict) -> list[Event]:
    """Turn a ``system`` record (compaction boundary, notices) into events."""
    text = str(rec.get("content") or "").strip()
    subtype = rec.get("subtype") or ""
    if subtype == "compact_boundary":
        meta = rec.get("compactMetadata") or {}
        tokens = meta.get("preTokens")
        text = "Conversation compacted" + (f" ({meta.get('trigger', '')}, {tokens:,} tokens before)"
                                           if isinstance(tokens, int) else "")
        return [Event("meta", rec.get("timestamp"), text, label="Compaction", data={"notice": True})]
    if not text:
        return []
    return [Event("meta", rec.get("timestamp"), text, label="Notice", data={"notice": True})]


def queued_prompt_events(rec: dict) -> list[Event]:
    """Turn a ``queued_command`` attachment typed by the user into a prompt event."""
    att = rec.get("attachment") or {}
    if att.get("type") != "queued_command" or (att.get("origin") or {}).get("kind") != "human":
        return []
    prompt = att.get("prompt")
    text, images = _user_text(prompt) if isinstance(prompt, list) else (str(prompt or ""), [])
    if not text.strip() and not images:
        return []
    return [Event("prompt", att.get("timestamp") or rec.get("timestamp"), text, images=images,
                  data={"queued": True})]


def session_events(records: list[dict]) -> list[Event]:
    """Return the events of the active branch of a transcript in order."""
    events: list[Event] = []
    for rec in active_branch(records):
        kind = rec.get("type")
        if kind == "user":
            events += user_events(rec)
        elif kind == "assistant":
            events += assistant_events(rec)
        elif kind == "system":
            events += system_events(rec)
        elif kind == "attachment":
            events += queued_prompt_events(rec)
    return events


def session_info(path: Path, records: list[dict] | None = None) -> SessionInfo:
    """Collect the metadata of a session."""
    if records is None:
        records = read_records(path)
    info = SessionInfo(path=path, session_id=path.stem)
    titles: dict[str, str] = {}
    times = []
    models: dict[str, None] = {}
    for rec in records:
        kind = rec.get("type")
        if kind == "custom-title" and rec.get("customTitle"):
            titles["custom"] = rec["customTitle"]
        elif kind == "agent-name" and rec.get("agentName"):
            titles["agent"] = rec["agentName"]
        elif kind == "summary" and rec.get("summary"):
            titles["summary"] = rec["summary"]
        if not _is_message(rec):
            continue
        if rec.get("timestamp"):
            times.append(rec["timestamp"])
        info.cwd = rec.get("cwd") or info.cwd
        info.git_branch = rec.get("gitBranch") or info.git_branch
        info.version = rec.get("version") or info.version
        if kind == "assistant":
            model = rec["message"].get("model")
            if model and not model.startswith("<"):
                models[model] = None
    for rec in active_branch(records):
        if rec.get("type") != "user":
            continue
        for ev in user_events(rec):
            if ev.kind == "prompt":
                info.prompts += 1
                info.first_prompt = info.first_prompt or ev.text
    title_file = path.with_suffix("") / "custom-title.json"
    if "custom" not in titles and title_file.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError, AttributeError):
            titles["custom"] = json.loads(title_file.read_text(encoding="utf-8")).get("customTitle") or ""
    info.title = titles.get("custom") or titles.get("agent") or titles.get("summary") or ""
    parsed = [t for t in (parse_timestamp(ts) for ts in times) if t]
    if parsed:
        info.start, info.end = min(parsed), max(parsed)
    info.models = list(models)
    info.url = session_urls(urls.SESSIONS_DIR, urls.SESSION_URLS_FILE).get(info.session_id, "")
    return info


def list_sessions(directory: Path) -> list[SessionInfo]:
    """Return the sessions of a project directory that have messages, oldest first."""
    infos = [session_info(p) for p in sorted(directory.glob("*.jsonl"))]
    infos = [i for i in infos if i.start is not None]
    infos.sort(key=lambda i: i.start or dt.datetime.min.replace(tzinfo=dt.UTC))
    return infos
