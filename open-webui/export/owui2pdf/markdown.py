"""Normalising Open WebUI message content and assembling the Markdown document.

Open WebUI renders Markdown with marked (GFM) plus KaTeX. Before handing the
text to pandoc we

* turn ``<details>`` blocks (reasoning, tool calls) and raw ``<think>`` tags
  into Markdown,
* rewrite ``\\( … \\)`` / ``\\[ … \\]`` math into ``$ … $`` / ``$$ … $$``,
* map the few inline HTML tags Open WebUI shows onto Markdown or LaTeX,
* turn ``[n]`` source markers into biblatex citations,
* extract base64 images to files, and
* close unclosed code fences so a message cannot swallow the following ones.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

from .chat import (attachments_markdown, chat_body, chat_title, error_text, message_text,
                   ordered_messages, usage_text)
from .notes import resolve_notes
from .sources import SourceRegistry, cite_inline, convert_citation_marks
from .util import fmt_timestamp, tex_escape

FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
DETAILS_RE = re.compile(r"<details\b([^>]*)>((?:(?!<details\b).)*?)</details>", re.S | re.I)
SUMMARY_RE = re.compile(r"\s*<summary>(.*?)</summary>\s*", re.S | re.I)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
THINK_RE = re.compile(r"<(think|thinking)>(.*?)</\1>", re.S | re.I)
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.S)
# Code spans and math spans: no inline conversions inside these.
PROTECTED_RE = re.compile(r"(`+)(?:.+?)\1|\$\$.+?\$\$|\$(?!\s)[^$\n]+?(?<!\s)\$", re.S)
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
RAW_LATEX_INLINE_RE = re.compile(r"`\\[a-zA-Z]+\{[^`]*\}`\{=latex\}")


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
# Inline-level: math delimiters, inline HTML, citation marks
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


def _convert_math_delimiters(text: str) -> str:
    text = DISPLAY_MATH_RE.sub(lambda m: "$$" + m.group(1).strip() + "$$", text)
    return INLINE_MATH_RE.sub(lambda m: "$" + m.group(1).strip() + "$", text)


def _outside(rx: re.Pattern, text: str, func) -> str:
    """Apply func to the parts of text not matched by rx."""
    pieces, pos = [], 0
    for m in rx.finditer(text):
        pieces.append(func(text[pos:m.start()]))
        pieces.append(m.group(0))
        pos = m.end()
    pieces.append(func(text[pos:]))
    return "".join(pieces)


def convert_inline_segment(text: str, cite_map: dict[int, str] | None = None) -> str:
    """Inline conversions on text that contains no fenced code."""
    text = _outside(CODE_SPAN_RE, text, _convert_math_delimiters)
    return _outside(PROTECTED_RE, text,
                    lambda t: convert_citation_marks(convert_inline_html(t), cite_map or {}))


# Lines whose meaning depends on staying adjacent to their neighbours: list
# items, table rows, block quotes, headings (ATX and setext underlines),
# indented code, fences, link/footnote definitions.
STRUCTURAL_LINE_RE = re.compile(
    r"^\s*(?:[-*+]\s|\d+[.)]\s|\||>|#{1,6}\s|```|~~~|\[[^\]]+\]:|(?:=+|-+)\s*$)|^(?: {4}|\t)")


def single_newlines_to_paragraphs(text: str) -> str:
    """Insert a blank line between adjacent plain text lines.

    For prompts typed with a single Enter between paragraphs: Markdown would
    join such lines into one paragraph, this makes each line its own paragraph.
    """
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if (line.strip() and nxt.strip()
                    and not STRUCTURAL_LINE_RE.match(line) and not STRUCTURAL_LINE_RE.match(nxt)):
                out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Whole message
# ---------------------------------------------------------------------------

def preprocess_markdown(md: str, include_reasoning: bool,
                        cite_map: dict[int, str] | None = None,
                        paragraph_breaks: bool = False) -> str:
    md = md.replace("\r\n", "\n")
    md = convert_details(md, include_reasoning)
    out_lines = []
    text_buf: list[str] = []
    fence: tuple[str, int] | None = None

    def flush_text():
        if text_buf:
            text = "\n".join(text_buf)
            if paragraph_breaks:
                text = single_newlines_to_paragraphs(text)
            out_lines.append(convert_inline_segment(text, cite_map))
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
    body = RAW_LATEX_INLINE_RE.sub("", body)
    preview = re.sub(r"\s+", " ", re.sub(r"[#*`>_\[\]\\{}~^$&%]", "", body)).strip()
    return preview[:limit] + ("…" if len(preview) > limit else "")


def message_markdown(msg: dict, opts, media_dir: Path, img_counter: list[int],
                     registry: SourceRegistry | None) -> str:
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

    cite_map = registry.register_message_sources(msg) if registry is not None else {}

    body = message_text(msg)
    err = error_text(msg)
    if err:
        body = (body + "\n\n" if body else "") + f"> **Error:** {err}"
    att = attachments_markdown(msg)
    if att:
        body = att + "\n\n" + body
    body = extract_data_images(body, media_dir, img_counter)
    body = preprocess_markdown(body, opts.include_reasoning, cite_map,
                               paragraph_breaks=opts.user_paragraphs and role == "user")
    if cite_map:
        body = body.rstrip() + "\n\n*Sources:* " + cite_inline(list(cite_map.values()))
    if not body.strip():
        body = "*(empty message)*"

    preview = bookmark_preview(body)
    bookmark = f"{label}: {preview}" if preview else label
    header = "\\owuiheader{%s}{%s}{%s}{%s}{%s}" % (
        bg, fg, tex_escape(label), tex_escape(meta_str), tex_escape(bookmark))
    return raw_latex_block(header) + body.strip("\n") + "\n"


def bibliography_markdown() -> str:
    return raw_latex_block(
        "\\nocite{*}\n"
        "\\par\\Needspace*{6\\baselineskip}\n"
        "\\pdfbookmark[0]{References}{owuirefs}\n"
        "\\section*{References}\n"
        "\\printbibliography[heading=none]")


def appendix_markdown(registry: SourceRegistry) -> str:
    notes = registry.notes_with_content()
    if not notes:
        return ""
    parts = [raw_latex_block(
        "\\clearpage\n"
        "\\pdfbookmark[0]{Appendix: Referenced notes}{owuiappendix}\n"
        "\\section*{Appendix: Referenced notes}")]
    for src in notes:
        meta = src.metadata_text()
        head = "\\section{%s}\\label{note:%s}\n\\owuinote{Reference \\cite{%s}%s}" % (
            tex_escape(src.title), src.key, src.key, tex_escape("; " + meta) if meta else "")
        body = preprocess_markdown(src.text, False).strip("\n")
        if src.content_kind == "truncated":
            body += "\n\n*[… note text truncated in the export; pass --notes or --notes-url for the full note]*"
        elif src.content_kind == "excerpts":
            body += "\n\n*[retrieved excerpts only; the full note is not in the export]*"
        parts.append(raw_latex_block(head) + body + "\n")
    return "\n\n".join(parts)


def build_markdown(chat_item: dict, opts, media_dir: Path) -> tuple[str, dict, str | None]:
    """Return (markdown document, pandoc metadata, bibtex or None) for one chat.

    ``opts`` needs the attributes ``show_usage``, ``show_sources``,
    ``include_reasoning``, ``include_notes`` and ``user_paragraphs`` (the
    parsed command line arguments work).
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

    messages = ordered_messages(chat)
    registry = SourceRegistry() if opts.show_sources else None
    if registry is not None:
        # Full note/file objects (with content) live in chat.files and in the
        # files attached to user messages; sources only carry metadata.
        for f in chat.get("files") or []:
            registry.register_file(f)
        for msg in messages:
            for f in msg.get("files") or []:
                registry.register_file(f)

    parts = []
    img_counter = [0]
    for msg in messages:
        parts.append(message_markdown(msg, opts, media_dir, img_counter, registry))

    bib = None
    if registry is not None and len(registry):
        resolve_notes(registry, opts)
        bib = registry.bibtex(opts.include_notes)
        parts.append(bibliography_markdown())
        if opts.include_notes:
            parts.append(appendix_markdown(registry))

    return "\n\n".join(parts), meta, bib
