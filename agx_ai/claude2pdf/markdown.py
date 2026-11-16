"""Assembling the Markdown document of a Claude Code session.

The verbosity is set by the level (``--level``):

* ``chat``: the prompts of the user and the text replies of Claude,
* ``tools``: also the tool calls (the command, file, edit, ...),
* ``results``: also the results of the tool calls, truncated,
* ``full``: also the thinking, the text injected by Claude Code (skill
  instructions, subagent reports, task notifications, compaction summaries)
  and the complete tool calls and results.

Interruptions, errors and compaction notices are shown at every level.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from pathlib import Path
import re

from agx_ai.convert.util import tex_escape
from agx_ai.owui2pdf.markdown import FENCE_RE, bookmark_preview, preprocess_markdown, raw_latex_block

from .session import Event, SessionInfo, fmt_time, preview

LEVELS: tuple[str, ...] = ("chat", "tools", "results", "full")
DEFAULT_LEVEL: str = "tools"
# Default of --max-lines for the levels below "full".
DEFAULT_MAX_LINES: int = 40
# A line counts as this many characters when limiting the length of a truncated text.
CHARS_PER_LINE: int = 160

ANSI_RE: re.Pattern[str] = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
CONTROL_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
HTML_TAG_START_RE: re.Pattern[str] = re.compile(r"<(?=[A-Za-z/!?])")
CODE_SPAN_RE: re.Pattern[str] = re.compile(r"(`+)(?:.+?)\1", re.S)
IMAGE_TYPES: dict[str, str] = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "application/pdf": "pdf"}
LANGUAGES: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".sh": "bash", ".bash": "bash", ".zsh": "zsh", ".md": "markdown",
    ".tex": "latex", ".bib": "bibtex", ".json": "json", ".jsonl": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".lua": "lua", ".js": "javascript", ".ts": "typescript", ".html": "html", ".css": "css",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".rs": "rust", ".go": "go", ".java": "java",
    ".sql": "sql", ".xml": "xml", ".ini": "ini", ".cfg": "ini", ".dockerfile": "dockerfile", ".r": "r",
}

# Appended to the LaTeX header of agx_ai.convert.
HEADER_EXTRA: str = r"""
% ---- claude2pdf -------------------------------------------------------------
\definecolor{claudeAssistantBg}{RGB}{248,236,226}
\definecolor{claudeAssistantFg}{RGB}{150,70,30}
\definecolor{claudeToolFg}{RGB}{70,80,120}
\definecolor{claudeResultFg}{RGB}{60,100,80}
\definecolor{claudeErrorFg}{RGB}{170,30,30}
% \claudelabel{colour}{label}{detail}: a small label above a tool call, a result or a note.
\newcommand{\claudelabel}[3]{%
  \par\Needspace*{3\baselineskip}\vspace{0.5ex}%
  \noindent{\small\sffamily\textcolor{#1}{\textbf{#2}}%
  \if\relax\detokenize{#3}\relax\else\enspace{\color{owuiMeta}#3}\fi}\par\nobreak\vspace{0.2ex}\noindent\ignorespaces}
"""


def sanitize(text: str) -> str:
    """Remove terminal escape sequences and control characters, and expand tabs."""
    text = ANSI_RE.sub("", text.replace("\r\n", "\n"))
    return CONTROL_RE.sub("", text.replace("\r", "\n")).expandtabs(4)


def truncate(text: str, max_lines: int) -> str:
    """Shorten a text to ``max_lines`` lines (and as many characters as that many full lines); 0 for no limit."""
    if max_lines <= 0:
        return text
    lines = text.split("\n")
    kept = "\n".join(lines[:max_lines])
    max_chars = max_lines * CHARS_PER_LINE
    if len(kept) > max_chars:
        kept = kept[:max_chars]
    if len(kept) >= len(text):
        return text
    rest_lines = text[len(kept):].count("\n")
    more = f"{rest_lines} more lines" if rest_lines else f"{len(text) - len(kept)} more characters"
    return kept.rstrip("\n") + f"\n[… {more}]"


def fence(text: str, lang: str = "") -> str:
    """Return a fenced code block whose fence is longer than any backtick run in the text."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


def language(path: str) -> str:
    """Guess the highlighting language of a file from its name."""
    p = Path(path)
    if p.name.lower() == "dockerfile":
        return "dockerfile"
    return LANGUAGES.get(p.suffix.lower(), "")


def escape_html(md: str) -> str:
    """Escape ``<`` outside code so that text such as ``<key>`` is shown instead of being dropped as HTML.

    Claude Code shows the text of the user and Claude as is, but pandoc would
    treat ``<tag>`` as raw HTML and leave it out of the PDF.
    """
    out, buf = [], []
    fence_char: str | None = None

    def flush() -> None:
        if buf:
            text = "\n".join(buf)
            pieces, pos = [], 0
            for m in CODE_SPAN_RE.finditer(text):
                pieces.append(HTML_TAG_START_RE.sub(r"\\<", text[pos:m.start()]))
                pieces.append(m.group(0))
                pos = m.end()
            pieces.append(HTML_TAG_START_RE.sub(r"\\<", text[pos:]))
            out.append("".join(pieces))
            buf.clear()

    for line in md.split("\n"):
        m = FENCE_RE.match(line)
        if fence_char is None:
            if m:
                flush()
                fence_char = m.group(2)
                out.append(line)
            else:
                buf.append(line)
        else:
            out.append(line)
            if m and m.group(2)[0] == fence_char[0] and len(m.group(2)) >= len(fence_char) and not m.group(3).strip():
                fence_char = None
    flush()
    return "\n".join(out)


def text_markdown(text: str, user: bool = False) -> str:
    """Convert the Markdown text of a message into pandoc-ready Markdown."""
    return preprocess_markdown(escape_html(sanitize(text)), False, paragraph_breaks=user).strip("\n")


def label_block(colour: str, label: str, detail: str = "") -> str:
    r"""Return a raw LaTeX label line (``\claudelabel``)."""
    detail = tex_escape(preview(detail, 300)).replace("/", "/\\allowbreak{}")
    return raw_latex_block("\\claudelabel{%s}{%s}{%s}" % (colour, tex_escape(label), detail))


def header_block(bg: str, fg: str, label: str, meta: str, bookmark: str) -> str:
    r"""Return a raw LaTeX message header (``\owuiheader``)."""
    return raw_latex_block("\\owuiheader{%s}{%s}{%s}{%s}{%s}" % (
        bg, fg, tex_escape(label), tex_escape(meta), tex_escape(bookmark)))


def images_markdown(images: list[dict], media_dir: Path, counter: list[int]) -> str:
    """Write base64 image blocks to media_dir and return Markdown that shows them."""
    parts = []
    for block in images:
        source = block.get("source") or {}
        ext = IMAGE_TYPES.get(str(source.get("media_type", "")).lower())
        if source.get("type") != "base64" or ext is None:
            parts.append(f"*[image: {source.get('media_type') or source.get('type') or 'unknown'}]*")
            continue
        counter[0] += 1
        path = media_dir / f"img{counter[0]}.{ext}"
        try:
            path.write_bytes(base64.b64decode(source.get("data", ""), validate=False))
        except (binascii.Error, ValueError, OSError):
            parts.append("*[image: undecodable]*")
            continue
        parts.append(f"![]({path.as_posix()})")
    return "\n\n".join(parts)


def _edit_diff(old: str, new: str) -> str:
    lines = [f"-{line}" for line in old.split("\n")] if old else []
    return "\n".join(lines + [f"+{line}" for line in new.split("\n")])


def tool_use_markdown(name: str, inp: dict, max_lines: int) -> tuple[str, str]:  # noqa: PLR0911
    """Return (detail, Markdown body) of a tool call."""

    def code(text: object, lang: str = "") -> str:
        return fence(truncate(sanitize(str(text)), max_lines), lang)

    path = str(inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or "")
    if name == "Bash":
        detail = str(inp.get("description") or "")
        if inp.get("run_in_background"):
            detail = (detail + " " if detail else "") + "(in the background)"
        return detail, code(inp.get("command", ""), "bash")
    if name == "Read":
        extra = []
        if inp.get("offset") or inp.get("limit"):
            extra.append(f"from line {inp.get('offset', 1)}" + (f", {inp['limit']} lines" if inp.get("limit") else ""))
        if inp.get("pages"):
            extra.append(f"pages {inp['pages']}")
        return path + (f" ({', '.join(extra)})" if extra else ""), ""
    if name == "Write":
        return path, code(inp.get("content", ""), language(path))
    if name == "Edit":
        detail = path + (" (all occurrences)" if inp.get("replace_all") else "")
        return detail, code(_edit_diff(str(inp.get("old_string", "")), str(inp.get("new_string", ""))), "diff")
    if name == "MultiEdit":
        diffs = [_edit_diff(str(e.get("old_string", "")), str(e.get("new_string", "")))
                 for e in inp.get("edits") or [] if isinstance(e, dict)]
        return path, code("\n@@\n".join(diffs), "diff")
    if name == "NotebookEdit":
        return f"{path} ({inp.get('edit_mode', 'replace')})", code(inp.get("new_source", ""), "python")
    if name in ("Grep", "Glob"):
        where = f" in {path}" if path else ""
        glob = f" ({inp['glob']})" if inp.get("glob") else ""
        return f"{inp.get('pattern', '')}{where}{glob}", ""
    if name in ("Agent", "Task"):
        agent = inp.get("subagent_type") or "agent"
        return f"{agent}: {inp.get('description', '')}", code(inp.get("prompt", ""))
    if name == "WebFetch":
        return str(inp.get("url", "")), code(inp["prompt"]) if inp.get("prompt") else ""
    if name == "WebSearch":
        return str(inp.get("query", "")), ""
    if name == "Skill":
        return f"{inp.get('skill', '')} {inp.get('args', '')}".strip(), ""
    if name == "TodoWrite":
        marks = {"completed": "[x]", "in_progress": "[~]"}
        todos = [f"- {marks.get(str(t.get('status')), '[ ]')} {t.get('content', '')}"
                 for t in inp.get("todos") or [] if isinstance(t, dict)]
        return "", code("\n".join(todos))
    if len(inp) == 1 and isinstance(next(iter(inp.values())), str) and "\n" not in next(iter(inp.values())):
        return next(iter(inp.values())), ""
    return "", code(json.dumps(inp, indent=2, ensure_ascii=False), "json") if inp else ""


def options_for_level(opts: argparse.Namespace) -> argparse.Namespace:
    """Fill in the options that default to the level: ``thinking``, ``meta`` and ``max_lines``."""
    level = LEVELS.index(opts.level)
    if getattr(opts, "thinking", None) is None:
        opts.thinking = level >= LEVELS.index("full")
    if getattr(opts, "meta", None) is None:
        opts.meta = level >= LEVELS.index("full")
    if getattr(opts, "max_lines", None) is None:
        opts.max_lines = 0 if level >= LEVELS.index("full") else DEFAULT_MAX_LINES
    return opts


def build_markdown(info: SessionInfo, events: list[Event], opts: argparse.Namespace,
                   media_dir: Path) -> tuple[str, dict, None]:
    """Return (Markdown document, pandoc metadata, None) for a session.

    ``opts`` needs the attributes ``level``, ``thinking``, ``meta`` and
    ``max_lines`` (see :func:`options_for_level`).
    """
    level = LEVELS.index(opts.level)
    show_tools = level >= LEVELS.index("tools")
    show_results = level >= LEVELS.index("results")
    tool_names: dict[str, str] = {}
    img_counter = [0]
    parts: list[str] = []
    # Whether the last header was Claude's, so that a turn gets only one header.
    in_claude = False

    def claude_header(ev: Event) -> None:
        nonlocal in_claude
        if not in_claude:
            meta = " · ".join(x for x in (ev.data.get("model", ""), fmt_time(ev.timestamp)) if x)
            parts.append(header_block("claudeAssistantBg", "claudeAssistantFg", "Claude", meta,
                                      f"Claude {fmt_time(ev.timestamp)}"))
            in_claude = True

    for ev in events:
        if ev.kind == "prompt":
            label = "User"
            if ev.data.get("command"):
                label = "User (command)"
            elif ev.data.get("queued"):
                label = "User (during the turn)"
            body = fence(ev.text, "") if ev.data.get("command") else text_markdown(ev.text, user=True)
            if ev.images:
                body = (body + "\n\n" if body else "") + images_markdown(ev.images, media_dir, img_counter)
            preview_text = bookmark_preview(ev.text)
            parts.append(header_block("owuiUserBg", "owuiUserFg", label, fmt_time(ev.timestamp),
                                      f"User: {preview_text}" if preview_text else "User"))
            parts.append(body)
            in_claude = False
        elif ev.kind == "text":
            claude_header(ev)
            parts.append(text_markdown(ev.text))
        elif ev.kind == "error":
            claude_header(ev)
            parts.append(label_block("claudeErrorFg", "Error"))
            parts.append(fence(sanitize(ev.text)))
        elif ev.kind == "thinking" and opts.thinking:
            claude_header(ev)
            parts.append(label_block("owuiMeta", "Thinking"))
            parts.append("\n".join("> " + line if line else ">" for line in text_markdown(ev.text).split("\n")))
        elif ev.kind == "tool_use":
            tool_names[ev.data.get("id") or ""] = ev.label
            if not show_tools:
                continue
            claude_header(ev)
            detail, body = tool_use_markdown(ev.label, ev.data.get("input") or {}, opts.max_lines)
            parts.append(label_block("claudeToolFg", ev.label or "Tool", detail))
            if body:
                parts.append(body)
        elif ev.kind == "tool_result":
            if not show_results:
                continue
            claude_header(ev)
            name = tool_names.get(ev.data.get("tool_use_id") or "", "")
            error = ev.data.get("is_error")
            label = ("Error" if error else "Result") + (f" of {name}" if name else "")
            parts.append(label_block("claudeErrorFg" if error else "claudeResultFg", label))
            text = truncate(sanitize(ev.text), opts.max_lines)
            if text.strip():
                parts.append(fence(text))
            if ev.images:
                parts.append(images_markdown(ev.images, media_dir, img_counter))
            if not text.strip() and not ev.images:
                parts.append("*(no output)*")
        elif ev.kind == "interrupt":
            parts.append(label_block("claudeErrorFg", ev.text or "Interrupted"))
        elif ev.kind == "meta":
            if ev.data.get("notice"):
                parts.append(label_block("owuiMeta", ev.label, ev.text))
                continue
            if not opts.meta:
                continue
            parts.append(header_block("owuiSystemBg", "owuiSystemFg", ev.label, fmt_time(ev.timestamp), ev.label))
            if ev.label in ("Skill instructions", "Compaction summary"):
                parts.append(text_markdown(ev.text))
            else:
                parts.append(fence(truncate(sanitize(ev.text), opts.max_lines)))
            if ev.images:
                parts.append(images_markdown(ev.images, media_dir, img_counter))
            in_claude = False

    date = fmt_time(info.start)
    if info.end and fmt_time(info.end) != date:
        end = fmt_time(info.end)
        date += " – " + (end[11:] if end[:10] == date[:10] else end)
    where = info.cwd + (f" ({info.git_branch})" if info.git_branch and info.git_branch != "HEAD" else "")
    meta = {
        "title": info.name,
        "date": date,
        "subtitle": where,
        "keywords": [],
    }
    intro = " · ".join(x for x in (
        f"Session {info.session_id}",
        ", ".join(info.models),
        f"Claude Code {info.version}" if info.version else "",
        f"level: {opts.level}",
    ) if x)
    parts.insert(0, raw_latex_block("{\\small\\color{owuiMeta}\\sffamily %s\\par}" % tex_escape(intro)))
    return "\n\n".join(p for p in parts if p), meta, None
