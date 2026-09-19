"""Normalising Open WebUI message content and assembling the Markdown document.

Open WebUI renders Markdown with marked (GFM) plus KaTeX. Before handing the
text to pandoc we

* turn ``<details>`` blocks (reasoning, tool calls) and raw ``<think>`` tags
  into Markdown,
* rewrite ``\\( … \\)`` / ``\\[ … \\]`` math into ``$ … $`` / ``$$ … $$``,
* map the few inline HTML tags Open WebUI shows onto Markdown or LaTeX,
* extract base64 images to files, and
* close unclosed code fences so a message cannot swallow the following ones.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

from .chat import (attachments_markdown, chat_body, chat_title, error_text, message_text,
                   ordered_messages, sources_markdown, usage_text)
from .util import fmt_timestamp, tex_escape

FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
DETAILS_RE = re.compile(r"<details\b([^>]*)>((?:(?!<details\b).)*?)</details>", re.S | re.I)
SUMMARY_RE = re.compile(r"\s*<summary>(.*?)</summary>\s*", re.S | re.I)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
THINK_RE = re.compile(r"<(think|thinking)>(.*?)</\1>", re.S | re.I)
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.S)
DISPLAY_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.S)
INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.S)
DATA_IMG_RE = re.compile(
    r"!\[([^\]]*)\]\((data:image/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+))\)")
INLINE_HTML = [
    (re.compile(r"<(b|strong)>(.*?)</\1>", re.S | re.I), r"**\2**"),
    (re.compile(r"<(i|em)>(.*?)</\1>", re.S | re.I), r"*\2*"),
    (re.compile(r"<(s|del|strike)>(.*?)</\1>", re.S | re.I), r"~~\2~~"),
    (re.compile(r"<code>(.*?)</code>", re.S | re.I), r"`\1`"),
    (re.compile(r"<(u|mark|small|span|font)(\s[^>]*)?>(.*?)</\1>", re.S | re.I), r"\3"),
]
SUPSUB_RE = re.compile(r"<(sup|sub)>(.*?)</\1>", re.S | re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)


# ---------------------------------------------------------------------------
# Block-level: <details>, <think>
# ---------------------------------------------------------------------------

def blockquote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in text.strip("\n").splitlines())


def convert_details(md: str, include_reasoning: bool) -> str:
    """Replace Open WebUI <details> blocks (reasoning, tool calls, ...)."""

    def repl(m: re.Match) -> str:
        attrs = dict(ATTR_RE.findall(m.group(1)))
        body = m.group(2)
        summary = ""
        sm = SUMMARY_RE.search(body)
        if sm:
            summary = re.sub(r"<[^>]+>", "", sm.group(1)).strip()
            body = body[:sm.start()] + body[sm.end():]
        body = body.strip("\n")
        kind = attrs.get("type", "").lower()
        if kind == "reasoning":
            if not include_reasoning:
                return "\n"
            # Open WebUI already prefixes reasoning lines with "> ".
            inner = body if body.lstrip().startswith(">") else blockquote(body)
            title = summary or "Reasoning"
            return f"\n> *{title}*\n>\n{inner}\n\n"
        if kind == "tool_calls":
            name = attrs.get("name", "")
            args = html.unescape(attrs.get("arguments", ""))
            result = html.unescape(attrs.get("result", ""))
            parts = [f"> *Tool call{': `' + name + '`' if name else ''}*"]
            if summary:
                parts.append(f"> {summary}")
            if args:
                parts.append(">")
                parts.append(blockquote("```json\n" + args + "\n```" if args.strip().startswith("{") else args))
            if result:
                parts.append(">")
                parts.append(blockquote("Result: " + result[:2000]))
            return "\n" + "\n".join(parts) + "\n\n"
        # Generic <details>: summary in bold, then the content.
        head = f"**{summary}**\n\n" if summary else ""
        return f"\n{head}{body}\n\n"

    # Raw <think>/<thinking> tags that Open WebUI did not wrap in <details>.
    md = THINK_RE.sub(lambda m: '<details type="reasoning"><summary>Thinking</summary>\n'
                      + blockquote(m.group(2)) + "\n</details>\n", md)
    prev = None
    while prev != md:
        prev = md
        md = DETAILS_RE.sub(repl, md)
    return md


# ---------------------------------------------------------------------------
# Inline-level: math delimiters, inline HTML
# ---------------------------------------------------------------------------

def convert_inline_html(text: str) -> str:
    """Map the few inline HTML tags Open WebUI renders onto Markdown/LaTeX."""
    if "<" not in text:
        return text
    for rx, rep in INLINE_HTML:
        text = rx.sub(rep, text)
    text = SUPSUB_RE.sub(lambda m: "`\\text%sscript{%s}`{=latex}"
                         % ("super" if m.group(1).lower() == "sup" else "sub",
                            tex_escape(re.sub(r"<[^>]+>", "", m.group(2)))), text)
    lines = []
    for line in text.split("\n"):
        if BR_RE.search(line):
            # Inside a table row a newline would end the row; use a space there.
            line = BR_RE.sub(" " if line.lstrip().startswith("|") else "\\\n", line)
        lines.append(line)
    return "\n".join(lines)


def _convert_math(text: str) -> str:
    text = DISPLAY_MATH_RE.sub(lambda m: "$$" + m.group(1).strip() + "$$", text)
    text = INLINE_MATH_RE.sub(lambda m: "$" + m.group(1).strip() + "$", text)
    return convert_inline_html(text)


def convert_inline_segment(text: str) -> str:
    """Inline conversions on text that contains no fenced code (skips code spans)."""
    pieces = []
    pos = 0
    for m in CODE_SPAN_RE.finditer(text):
        pieces.append(_convert_math(text[pos:m.start()]))
        pieces.append(m.group(0))
        pos = m.end()
    pieces.append(_convert_math(text[pos:]))
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Whole message
# ---------------------------------------------------------------------------

def preprocess_markdown(md: str, include_reasoning: bool) -> str:
    md = md.replace("\r\n", "\n")
    md = convert_details(md, include_reasoning)
    out_lines = []
    text_buf: list[str] = []
    fence: tuple[str, int] | None = None

    def flush_text():
        if text_buf:
            out_lines.append(convert_inline_segment("\n".join(text_buf)))
            text_buf.clear()

    for line in md.split("\n"):
        m = FENCE_RE.match(line)
        if fence is None:
            if m and not (m.group(2)[0] == "`" and "`" in m.group(3)):
                flush_text()
                fence = (m.group(2)[0], len(m.group(2)))
                out_lines.append(line)
            else:
                text_buf.append(line)
        else:
            out_lines.append(line)
            if m and m.group(2)[0] == fence[0] and len(m.group(2)) >= fence[1] and not m.group(3).strip():
                fence = None
    flush_text()
    if fence is not None:  # unclosed fence: close it so it cannot swallow later messages
        out_lines.append(fence[0] * fence[1])
    return "\n".join(out_lines)


def extract_data_images(md: str, media_dir: Path, counter: list[int]) -> str:
    """Write base64 images to media_dir and point the Markdown at the files."""

    def repl(m: re.Match) -> str:
        ext = m.group(3).lower().split("+")[0]
        ext = {"jpeg": "jpg"}.get(ext, ext)
        if ext == "svg":  # LaTeX cannot include SVG directly
            return f"*[image: {m.group(1) or 'svg'}]*"
        counter[0] += 1
        path = media_dir / f"img{counter[0]}.{ext}"
        try:
            path.write_bytes(base64.b64decode(re.sub(r"\s+", "", m.group(4))))
        except Exception:
            return f"*[image: {m.group(1) or 'undecodable'}]*"
        return f"![{m.group(1)}]({path.as_posix()})"

    return DATA_IMG_RE.sub(repl, md)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

ROLE_STYLES = {
    "user": ("User", "owuiUserBg", "owuiUserFg"),
    "system": ("System", "owuiSystemBg", "owuiSystemFg"),
    "assistant": ("Assistant", "owuiAssistantBg", "owuiAssistantFg"),
}


def raw_latex_block(tex: str) -> str:
    return "\n\n```{=latex}\n" + tex + "\n```\n\n"


def bookmark_preview(body: str, limit: int = 60) -> str:
    preview = re.sub(r"\s+", " ", re.sub(r"[#*`>_\[\]\\{}~^$&%]", "", body)).strip()
    return preview[:limit] + ("…" if len(preview) > limit else "")


def build_markdown(chat_item: dict, opts, media_dir: Path) -> tuple[str, dict]:
    """Return (markdown document, pandoc metadata) for one chat.

    ``opts`` needs the attributes ``show_usage``, ``show_sources`` and
    ``include_reasoning`` (the parsed command line arguments work).
    """
    chat = chat_body(chat_item)
    models = chat.get("models") or []
    created = chat_item.get("created_at") or chat.get("timestamp")
    tags = (chat_item.get("meta") or {}).get("tags") or chat.get("tags") or []
    meta = {
        "title": chat_title(chat_item),
        "date": fmt_timestamp(created),
        "subtitle": ", ".join(str(m) for m in models),
        "keywords": [str(t) for t in tags if isinstance(t, (str, int))],
    }

    parts = []
    img_counter = [0]
    for msg in ordered_messages(chat):
        role = (msg.get("role") or "assistant").lower()
        label, bg, fg = ROLE_STYLES.get(role, ROLE_STYLES["assistant"])
        if role == "user":
            user = msg.get("user") if isinstance(msg.get("user"), dict) else {}
            if user.get("name"):
                label = f"User ({user['name']})"

        meta_bits = []
        if role != "user" and msg.get("model"):
            meta_bits.append(str(msg["model"]))
        ts = fmt_timestamp(msg.get("timestamp"))
        if ts:
            meta_bits.append(ts)
        if opts.show_usage:
            u = usage_text(msg)
            if u:
                meta_bits.append(u)
        meta_str = " · ".join(meta_bits)

        body = message_text(msg)
        err = error_text(msg)
        if err:
            body = (body + "\n\n" if body else "") + f"> **Error:** {err}"
        att = attachments_markdown(msg)
        if att:
            body = att + "\n\n" + body
        body = extract_data_images(body, media_dir, img_counter)
        body = preprocess_markdown(body, opts.include_reasoning)
        if opts.show_sources:
            src = sources_markdown(msg)
            if src:
                body = body.rstrip() + "\n\n" + src
        if not body.strip():
            body = "*(empty message)*"

        preview = bookmark_preview(body)
        bookmark = f"{label}: {preview}" if preview else label
        header = "\\owuiheader{%s}{%s}{%s}{%s}{%s}" % (
            bg, fg, tex_escape(label), tex_escape(meta_str), tex_escape(bookmark))
        parts.append(raw_latex_block(header) + body.strip("\n") + "\n")

    return "\n\n".join(parts), meta
