"""Reading an Open WebUI export and extracting per-message information."""

from __future__ import annotations

import json
from pathlib import Path

from .util import md_escape_inline


def load_chats(path: Path) -> list[dict]:
    """Return the chat objects in an export (single chat or list of chats).

    Raises ValueError when the file is not a readable Open WebUI chat export.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read JSON ({e})") from e
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("unexpected JSON structure (expected a chat object or a list of chats)")
    chats = []
    for item in data:
        if isinstance(item, dict) and ("chat" in item or "history" in item or "messages" in item):
            chats.append(item)
    if not chats:
        raise ValueError("no chats found in the file")
    return chats


def chat_body(chat_item: dict) -> dict:
    """The inner chat object (exports wrap it in {"chat": {...}})."""
    return chat_item.get("chat") if isinstance(chat_item.get("chat"), dict) else chat_item


def chat_title(chat_item: dict) -> str:
    return chat_item.get("title") or chat_body(chat_item).get("title") or "Chat"


def ordered_messages(chat: dict) -> list[dict]:
    """Messages of the selected branch, oldest first.

    Follows history.currentId back to the root. If there is no current id the
    latest child is chosen at every step; if there is no history at all the
    flat ``messages`` list is used.
    """
    history = chat.get("history") or {}
    messages = history.get("messages") or {}
    current = history.get("currentId")
    chain: list[dict] = []
    if messages and current in messages:
        seen = set()
        cur = current
        while cur and cur in messages and cur not in seen:
            seen.add(cur)
            chain.append(messages[cur])
            cur = messages[cur].get("parentId")
        chain.reverse()
    elif messages:
        roots = [m for m in messages.values() if not m.get("parentId")]
        roots.sort(key=lambda m: m.get("timestamp") or 0)
        cur = roots[0] if roots else None
        seen = set()
        while cur and cur.get("id") not in seen:
            seen.add(cur.get("id"))
            chain.append(cur)
            kids = [k for k in (cur.get("childrenIds") or []) if k in messages]
            cur = messages[kids[-1]] if kids else None
    else:
        chain = list(chat.get("messages") or [])
    return chain


def message_text(msg: dict) -> str:
    """Message content as Markdown (handles OpenAI-style content parts)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    parts.append(f"![image]({url})")
            elif isinstance(part, str):
                parts.append(part)
        return "\n\n".join(parts)
    return "" if content is None else str(content)


def attachments_markdown(msg: dict) -> str:
    """Images (inline) and other attachments (as a list) of a message."""
    files = msg.get("files") or []
    images, others = [], []
    for f in files:
        if not isinstance(f, dict):
            continue
        ftype = (f.get("type") or "file").lower()
        url = f.get("url") or ""
        name = (f.get("name") or f.get("title") or (f.get("file") or {}).get("filename")
                or f.get("id") or "attachment")
        if ftype == "image" and url.startswith("data:image/"):
            images.append(f"![{name}]({url})")
        elif ftype == "image":
            others.append(f"{name} (image)")
        else:
            others.append(f"{name} ({ftype})")
    out = []
    if images:
        out.append("\n\n".join(images))
    if others:
        out.append("*Attachments: " + ", ".join(md_escape_inline(o) for o in others) + "*")
    return "\n\n".join(out)


def usage_text(msg: dict) -> str:
    """Token counts and generation speed, e.g. '5870→1157 tokens, 22.3 tok/s'."""
    usage = msg.get("usage") or {}
    if not isinstance(usage, dict):
        return ""
    bits = []
    out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("predicted_n")
    in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_n")
    if in_tok or out_tok:
        bits.append(f"{in_tok or '?'}→{out_tok or '?'} tokens")
    tps = usage.get("predicted_per_second") or usage.get("response_token/s")
    if tps:
        try:
            bits.append(f"{float(tps):.1f} tok/s")
        except (TypeError, ValueError):
            pass
    return ", ".join(bits)


def error_text(msg: dict) -> str:
    err = msg.get("error")
    if not err:
        return ""
    return err.get("content") if isinstance(err, dict) else str(err)
