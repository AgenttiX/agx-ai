"""Referenced notes, files and web pages: citations, bibliography, appendix.

Open WebUI marks retrieved sources in assistant messages as ``[n]``, where n
is the position in the message's (de-duplicated) ``sources`` list. The
registry assigns every distinct source a biblatex key, so the markers become
``\\cite`` commands and a numeric bibliography (biber backend) lists each source
once with its metadata. Personal metadata (user name, e-mail) is never
included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .util import fmt_timestamp, tex_escape

# Open WebUI's notes list API (used by the note picker) truncates the note
# Markdown to this many characters; the chat export then carries only that.
NOTE_TRUNCATION_LIMIT = 1000


def _is_url(v) -> bool:
    return isinstance(v, str) and v.startswith(("http://", "https://"))


def _text_content(obj: dict) -> str | None:
    """Extract note/file text from the various shapes Open WebUI uses."""
    if not isinstance(obj, dict):
        return None
    for holder in (obj, obj.get("file") or {}):
        data = holder.get("data") if isinstance(holder, dict) else None
        if not isinstance(data, dict):
            continue
        content = data.get("content")
        if isinstance(content, dict):
            content = content.get("md") or content.get("markdown") or content.get("text")
        if isinstance(content, str):
            return content
    return None


@dataclass
class Source:
    key: str
    kind: str  # note | file | web | other
    title: str
    ident: str = ""
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    content: str | None = None
    content_kind: str = "full"  # full | truncated | excerpts
    excerpts: list[str] = field(default_factory=list)
    content_type: str = ""
    size: int | None = None

    @property
    def text(self) -> str:
        if self.content is not None:
            return self.content
        return "\n\n".join(self.excerpts)

    def counts(self) -> tuple[int, int]:
        t = self.text
        return len(t), len(t.split())

    @property
    def has_full_content(self) -> bool:
        return self.content is not None and self.content_kind == "full"

    def set_full_content(self, content: str, created_at=None, updated_at=None) -> None:
        self.content = content
        self.content_kind = "full"
        if created_at and not self.created_at:
            self.created_at = fmt_timestamp(created_at)
        if updated_at and not self.updated_at:
            self.updated_at = fmt_timestamp(updated_at)

    def _set_export_content(self, content: str) -> None:
        """Content found in the chat export; may be truncated for notes."""
        self.content = content
        truncated = self.kind == "note" and len(content) >= NOTE_TRUNCATION_LIMIT
        self.content_kind = "truncated" if truncated else "full"

    def counts_text(self) -> str:
        chars, words = self.counts()
        if self.content_kind == "full":
            return f"{chars} characters, {words} words"
        if self.content_kind == "truncated":
            return (f"at least {chars} characters, {words} words "
                    f"(text truncated to {NOTE_TRUNCATION_LIMIT} characters in the export)")
        n = len(self.excerpts)
        return (f"{chars} characters, {words} words in {n} retrieved excerpt{'s' if n != 1 else ''} "
                "(full text not in the export)")

    def label(self) -> str:
        return {"note": "Open WebUI note", "file": "Uploaded file", "web": "Web page"}.get(self.kind, "Source")

    def metadata_text(self) -> str:
        bits = []
        if self.created_at:
            bits.append(f"created {self.created_at}")
        if self.updated_at:
            bits.append(f"updated {self.updated_at}")
        if self.content_type:
            bits.append(self.content_type)
        if self.size:
            bits.append(f"{self.size / 1024:.0f} kB" if self.size >= 1024 else f"{self.size} B")
        if self.kind != "web" and self.text:
            bits.append(self.counts_text())
        return "; ".join(bits)


class SourceRegistry:
    """All sources referenced in a chat, keyed for biblatex."""

    def __init__(self):
        self.sources: dict[str, Source] = {}   # key -> Source
        self._by_ident: dict[str, str] = {}    # id or url -> key

    def __len__(self) -> int:
        return len(self.sources)

    def _get_or_create(self, ident: str, kind: str, title: str) -> Source:
        key = self._by_ident.get(ident)
        if key is None:
            key = f"src{len(self.sources) + 1}"
            self._by_ident[ident] = key
            self.sources[key] = Source(key=key, kind=kind, title=title, ident=ident)
        return self.sources[key]

    def register_file(self, f: dict) -> Source | None:
        """A file/note object from chat.files or a message's files list."""
        if not isinstance(f, dict):
            return None
        ident = f.get("id") or (f.get("file") or {}).get("id")
        if not ident:
            return None
        kind = (f.get("type") or "file").lower()
        if kind not in ("note", "file"):
            kind = "file" if kind in ("doc", "document", "text") else "other"
        title = f.get("name") or f.get("title") or (f.get("file") or {}).get("filename") or ident
        src = self._get_or_create(ident, kind, str(title))
        self._update_meta(src, f)
        content = _text_content(f)
        if content is not None and not src.has_full_content:
            src._set_export_content(content)
        return src

    def register_source(self, entry: dict) -> Source | None:
        """An element of a message's ``sources`` list; returns its Source."""
        if not isinstance(entry, dict):
            return None
        s = entry.get("source") or {}
        metas = entry.get("metadata") or []
        meta0 = metas[0] if metas and isinstance(metas[0], dict) else {}
        docs = [d for d in (entry.get("document") or []) if isinstance(d, str)]

        url = next((c for c in [s.get("url"), meta0.get("source"), s.get("name")] if _is_url(c)), "")
        ident = s.get("id") or meta0.get("file_id") or url or s.get("name") or meta0.get("name")
        if not ident:
            return None
        candidates = [s.get("title"), s.get("name"), meta0.get("title"), meta0.get("name")]
        title = next((c for c in candidates if c and not _is_url(c)), None) or url or str(ident)
        kind = (s.get("type") or "").lower()
        if url and kind not in ("note", "file"):
            kind = "web"
        elif kind not in ("note", "file"):
            kind = "file" if s.get("id") or meta0.get("file_id") else "other"
        src = self._get_or_create(str(ident), kind, str(title))
        if url and not src.url:
            src.url = url
        self._update_meta(src, s)
        content = _text_content(s)
        if content is not None and not src.has_full_content:
            src._set_export_content(content)
        for d in docs:
            if d not in src.excerpts:
                src.excerpts.append(d)
        if src.content is None:
            src.content_kind = "excerpts"
        elif src.content_kind == "full" and src.kind == "note" and any(
                d.strip() and d.strip() not in src.content for d in docs):
            # A retrieved chunk that is not part of the stored text: the
            # stored text cannot be the complete note.
            src.content_kind = "truncated"
        return src

    @staticmethod
    def _update_meta(src: Source, obj: dict) -> None:
        inner = obj.get("file") if isinstance(obj.get("file"), dict) else {}
        meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
        created = obj.get("created_at") or inner.get("created_at")
        updated = obj.get("updated_at") or inner.get("updated_at")
        if created and not src.created_at:
            src.created_at = fmt_timestamp(created)
        if updated and not src.updated_at:
            src.updated_at = fmt_timestamp(updated)
        ctype = meta.get("content_type") or obj.get("content_type")
        if ctype and not src.content_type:
            src.content_type = str(ctype)
        size = obj.get("size") or meta.get("size")
        if isinstance(size, (int, float)) and not src.size:
            src.size = int(size)

    def register_message_sources(self, msg: dict) -> dict[int, str]:
        """Register a message's sources; returns {citation number: key}."""
        cite_map: dict[int, str] = {}
        for entry in msg.get("sources") or msg.get("citations") or []:
            src = self.register_source(entry)
            if src and src.key not in cite_map.values():
                cite_map[len(cite_map) + 1] = src.key
        return cite_map

    # -- output -------------------------------------------------------------

    def notes_with_content(self) -> list[Source]:
        return [s for s in self.sources.values() if s.kind == "note" and s.text.strip()]

    def bibtex(self, include_notes: bool) -> str:
        entries = []
        for src in self.sources.values():
            fields = {"title": tex_escape(src.title)}
            if src.kind == "web":
                etype = "online"
                fields["url"] = src.url.replace("{", "%7B").replace("}", "%7D")
                if src.created_at:
                    fields["note"] = tex_escape(f"Retrieved {src.created_at}")
            else:
                etype = "misc"
                fields["howpublished"] = src.label()
                meta = src.metadata_text()
                if meta:
                    fields["note"] = tex_escape(meta[0].upper() + meta[1:])
                if include_notes and src.kind == "note" and src.text.strip():
                    fields["addendum"] = "Full text in \\hyperref[note:%s]{the appendix}" % src.key
            body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
            entries.append(f"@{etype}{{{src.key},\n{body}\n}}")
        return "\n\n".join(entries) + "\n"


CITE_MARK_RE = re.compile(r"(?<![\[\w\\])\[(\d{1,3})\](?![\(\[:\w])")


def convert_citation_marks(text: str, cite_map: dict[int, str]) -> str:
    """Turn Open WebUI's ``[n]`` source markers into biblatex citations."""
    if not cite_map:
        return text

    def repl(m: re.Match) -> str:
        key = cite_map.get(int(m.group(1)))
        return f"`\\cite{{{key}}}`{{=latex}}" if key else m.group(0)

    return CITE_MARK_RE.sub(repl, text)


def cite_inline(keys: list[str]) -> str:
    return "`\\cite{%s}`{=latex}" % ",".join(keys)
