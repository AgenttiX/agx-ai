"""Tests of agx_ai.owui2pdf.

The conversion runs pandoc for real (it is fast) but replaces latexmk with a stub, as a LuaLaTeX build takes from
ten seconds to a minute. The real PDF build of ``example.json`` is in a test marked ``slow``, which is run only
with ``pytest --run-slow`` (and is skipped if pandoc, latexmk, LuaLaTeX or biber is missing).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading
from typing import Any

import pytest

from agx_ai import owui2pdf
from agx_ai.owui2pdf import build, chat, cli, fonts, markdown, notes, sources, util

EXAMPLE = owui2pdf.PACKAGE_DIR / "example.json"
TOOLS = ("pandoc", "latexmk", "lualatex", "biber")


def options(**kwargs: Any) -> argparse.Namespace:
    """The parsed default options of the command line, with overrides."""
    args = cli.build_parser().parse_args(["x.json"])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------

def test_tex_escape() -> None:
    assert util.tex_escape(r"50% of $x_1 & {y}# ~^\ ") == (
        r"50\% of \$x\_1 \& \{y\}\# \textasciitilde{}\textasciicircum{}\textbackslash{} ")


def test_md_escape_inline() -> None:
    assert util.md_escape_inline("a*b_c [d] `e` <f> #g |h") == r"a\*b\_c \[d\] \`e\` \<f\> \#g \|h"


def test_fmt_timestamp() -> None:
    seconds = 1_700_000_000
    expected = dt.datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M")
    # Seconds, milliseconds and nanoseconds
    assert util.fmt_timestamp(seconds) == expected
    assert util.fmt_timestamp(seconds * 1000) == expected
    assert util.fmt_timestamp(str(seconds * 10**9)) == expected
    assert util.fmt_timestamp(None) == ""
    assert util.fmt_timestamp(0) == ""
    assert util.fmt_timestamp("yesterday") == "yesterday"


def test_slugify() -> None:
    assert util.slugify("Hello, World! Ääkköset_ja -- väli") == "hello-world-ääkköset-ja-väli"
    assert util.slugify("!!!") == "chat"
    assert len(util.slugify("x" * 100)) == 60


def test_which_or_die() -> None:
    assert util.which_or_die("python3")
    with pytest.raises(SystemExit, match="not found"):
        util.which_or_die("no-such-program-agx")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def test_load_chats(tmp_path: Path) -> None:
    assert chat.load_chats(write_json(tmp_path / "a.json", {"title": "A", "chat": {}})) == [{"title": "A", "chat": {}}]
    two = [{"chat": {}}, {"messages": []}, {"other": 1}, "text"]
    assert len(chat.load_chats(write_json(tmp_path / "b.json", two))) == 2
    with pytest.raises(ValueError, match="no chats"):
        chat.load_chats(write_json(tmp_path / "c.json", [{"other": 1}]))
    with pytest.raises(ValueError, match="unexpected JSON structure"):
        chat.load_chats(write_json(tmp_path / "d.json", 42))
    (tmp_path / "e.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read JSON"):
        chat.load_chats(tmp_path / "e.json")
    assert len(chat.load_chats(EXAMPLE)) == 1


def test_chat_title() -> None:
    assert chat.chat_title({"title": "Outer", "chat": {"title": "Inner"}}) == "Outer"
    assert chat.chat_title({"chat": {"title": "Inner"}}) == "Inner"
    assert chat.chat_title({"messages": []}) == "Chat"


def _history(current: str | None) -> dict[str, Any]:
    # root -> a -> (b1, b2); b2 is the newer branch
    messages = {
        "root": {"id": "root", "parentId": None, "childrenIds": ["a"], "timestamp": 1},
        "a": {"id": "a", "parentId": "root", "childrenIds": ["b1", "b2"], "timestamp": 2},
        "b1": {"id": "b1", "parentId": "a", "childrenIds": [], "timestamp": 3},
        "b2": {"id": "b2", "parentId": "a", "childrenIds": [], "timestamp": 4},
    }
    return {"history": {"currentId": current, "messages": messages}}


def test_ordered_messages() -> None:
    ids = [m["id"] for m in chat.ordered_messages(_history("b1"))]
    assert ids == ["root", "a", "b1"]
    # Without a current id, the latest child is followed.
    assert [m["id"] for m in chat.ordered_messages(_history(None))] == ["root", "a", "b2"]
    flat = [{"role": "user", "content": "x"}]
    assert chat.ordered_messages({"messages": flat}) == flat
    assert chat.ordered_messages({}) == []


def test_ordered_messages_cycle() -> None:
    messages = {"a": {"id": "a", "parentId": "b"}, "b": {"id": "b", "parentId": "a"}}
    assert len(chat.ordered_messages({"history": {"currentId": "a", "messages": messages}})) == 2


def test_message_text() -> None:
    assert chat.message_text({"content": "plain"}) == "plain"
    parts = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "u.png"}}, "raw"]
    assert chat.message_text({"content": parts}) == "look\n\n![image](u.png)\n\nraw"
    assert chat.message_text({}) == ""
    assert chat.message_text({"content": 5}) == "5"


def test_attachments_markdown() -> None:
    files = [
        {"type": "image", "url": "data:image/png;base64,AAAA", "name": "pic.png"},
        {"type": "image", "url": "https://x/y.png", "name": "remote.png"},
        {"type": "file", "file": {"filename": "notes_1.pdf"}},
        "ignored",
    ]
    out = chat.attachments_markdown({"files": files})
    assert out.startswith("![pic.png](data:image/png;base64,AAAA)")
    assert r"*Attachments: remote.png (image), notes\_1.pdf (file)*" in out
    assert chat.attachments_markdown({}) == ""


def test_usage_text() -> None:
    assert chat.usage_text({"usage": {"prompt_tokens": 5870, "completion_tokens": 1157,
                                      "response_token/s": "22.34"}}) == "5870→1157 tokens, 22.3 tok/s"
    assert chat.usage_text({"usage": {"predicted_n": 10}}) == "?→10 tokens"
    assert chat.usage_text({"usage": "bad"}) == ""
    assert chat.usage_text({"usage": {"predicted_per_second": "fast"}}) == ""


def test_error_text() -> None:
    assert chat.error_text({"error": {"content": "boom"}}) == "boom"
    assert chat.error_text({"error": "plain"}) == "plain"
    assert chat.error_text({}) == ""


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def test_convert_details_reasoning() -> None:
    md = 'Before\n<details type="reasoning" done="true"><summary>Thought for 3 s</summary>\n> step\n</details>\nAfter'
    assert "step" not in markdown.convert_details(md, include_reasoning=False)
    kept = markdown.convert_details(md, include_reasoning=True)
    assert "> *Thought for 3 s*" in kept
    assert "> step" in kept


def test_convert_details_tool_call_and_generic() -> None:
    md = ('<details type="tool_calls" name="search" arguments="{&quot;q&quot;: 1}" result="&quot;ok&quot;">'
          "<summary>Tool Executed</summary></details>\n"
          "<details><summary>More</summary>\nHidden text\n</details>")
    out = markdown.convert_details(md, include_reasoning=False)
    assert "> *Tool call: `search`*" in out
    assert '> {"q": 1}' in out
    assert '> Result: "ok"' in out
    assert "**More**\n\nHidden text" in out


def test_convert_think_tags() -> None:
    md = "<think>secret plan</think>Answer"
    assert markdown.convert_details(md, include_reasoning=False).strip() == "Answer"
    assert "> secret plan" in markdown.convert_details(md, include_reasoning=True)


def test_convert_inline_html() -> None:
    out = markdown.convert_inline_html("<b>bold</b> <em>it</em> <del>x</del> <code>c</code> <span a='1'>s</span>")
    assert out == "**bold** *it* ~~x~~ `c` s"
    assert markdown.convert_inline_html("E = mc<sup>2</sup>") == r"E = mc`\textsuperscript{2}`{=latex}"
    assert markdown.convert_inline_html("a<br>b") == "a\\\nb"
    assert markdown.convert_inline_html("| a<br/>b |") == "| a b |"
    assert markdown.convert_inline_html("no tags") == "no tags"


def test_convert_inline_segment() -> None:
    text = r"Math \(x^2\) and \[y\] but not `\(code\)`, cite [1], link [2](u)"
    out = markdown.convert_inline_segment(text, {1: "src1", 2: "src2"})
    assert "$x^2$" in out
    assert "$$y$$" in out
    assert r"`\(code\)`" in out
    assert r"`\cite{src1}`{=latex}" in out
    assert "[2](u)" in out
    # No conversions inside math
    assert markdown.convert_inline_segment("$a<b>c</b>$") == "$a<b>c</b>$"


def test_single_newlines_to_paragraphs() -> None:
    text = "line one\nline two\n- item\n- item two\n| a |\n| b |"
    assert markdown.single_newlines_to_paragraphs(text) == (
        "line one\n\nline two\n- item\n- item two\n| a |\n| b |")


def test_preprocess_markdown_fences() -> None:
    md = "Text **x**\n```python\n<b>not converted</b> \\(x\\)\n```\nafter <b>b</b>\n````\nunclosed"
    out = markdown.preprocess_markdown(md, include_reasoning=False)
    assert "<b>not converted</b> \\(x\\)" in out
    assert "after **b**" in out
    # The unclosed fence is closed with the same fence.
    assert out.endswith("unclosed\n````")
    # Inline code with backticks in the info string is not a fence.
    assert markdown.preprocess_markdown("``` a ` b ```", False) == "``` a ` b ```"


def test_preprocess_markdown_paragraph_breaks() -> None:
    assert markdown.preprocess_markdown("a\nb", False, paragraph_breaks=True) == "a\n\nb"
    assert markdown.preprocess_markdown("a\r\nb", False) == "a\nb"


def test_extract_data_images(tmp_path: Path) -> None:
    png = base64.b64encode(b"\x89PNG fake").decode()
    md = (f"![one](data:image/png;base64,{png}) ![two](data:image/jpeg;base64,{png}) "
          f"![v](data:image/svg+xml;base64,{png})")
    counter = [0]
    out = markdown.extract_data_images(md, tmp_path, counter)
    assert counter == [2]
    assert (tmp_path / "img1.png").read_bytes() == b"\x89PNG fake"
    assert (tmp_path / "img2.jpg").exists()
    assert f"![one]({(tmp_path / 'img1.png').as_posix()})" in out
    assert "*[image: v]*" in out


def test_bookmark_preview() -> None:
    assert markdown.bookmark_preview("**Hello** `x` \\cite{a}") == "Hello x citea"
    assert markdown.bookmark_preview(r"`\cite{src1}`{=latex} rest") == "rest"
    assert markdown.bookmark_preview("word " * 30, limit=10) == "word word …"


def test_message_markdown(tmp_path: Path) -> None:
    msg = {"role": "user", "user": {"name": "Mika"}, "content": "Hi_there", "timestamp": 1_700_000_000,
           "error": {"content": "failed"}}
    out = markdown.message_markdown(msg, options(), tmp_path, [0], None)
    assert r"\owuiheader{owuiUserBg}{owuiUserFg}{User (Mika)}" in out
    assert "> **Error:** failed" in out
    empty = markdown.message_markdown({"role": "assistant", "model": "m"}, options(), tmp_path, [0], None)
    assert "*(empty message)*" in empty
    assert "{owuiAssistantBg}" in empty


def test_build_markdown_example(tmp_path: Path) -> None:
    (item,) = chat.load_chats(EXAMPLE)
    md, meta, bib = markdown.build_markdown(item, options(include_notes=True), tmp_path)
    assert meta["title"] == "PDF Conversion Test"
    assert meta["subtitle"] == "google/gemma-4-31b-qat"
    assert md.count(r"\owuiheader{") == len(chat.ordered_messages(chat.chat_body(item)))
    if bib is not None:
        assert r"\printbibliography" in md
        assert bib.startswith("@")
    md2, _, bib2 = markdown.build_markdown(item, options(show_sources=False), tmp_path)
    assert bib2 is None
    assert r"\printbibliography" not in md2


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def test_text_content() -> None:
    assert sources.text_content({"data": {"content": "a"}}) == "a"
    assert sources.text_content({"file": {"data": {"content": {"md": "b"}}}}) == "b"
    assert sources.text_content({"data": {"content": {"other": 1}}}) is None
    assert sources.text_content("x") is None  # type: ignore[arg-type]


def test_registry_web_and_note() -> None:
    reg = sources.SourceRegistry()
    msg = {"sources": [
        {"source": {"name": "https://example.com/a", "type": "web_search"}, "document": ["chunk one"]},
        {"source": {"id": "n1", "type": "note", "name": "My note", "created_at": 1_700_000_000},
         "document": ["part"]},
        {"source": {"name": "https://example.com/a"}, "document": ["chunk two"]},  # the same web page
        {"source": {}},  # no identity
    ]}
    cite_map = reg.register_message_sources(msg)
    assert cite_map == {1: "src1", 2: "src2"}
    web, note = reg.sources["src1"], reg.sources["src2"]
    assert (web.kind, web.url, web.excerpts) == ("web", "https://example.com/a", ["chunk one", "chunk two"])
    assert (note.kind, note.title, note.content_kind) == ("note", "My note", "excerpts")
    assert "retrieved excerpt" in note.counts_text()
    assert note.created_at
    bib = reg.bibtex(include_notes=True)
    assert "@online{src1," in bib
    assert "url = {https://example.com/a}" in bib
    assert "@misc{src2," in bib
    assert "howpublished = {Open WebUI note}" in bib
    assert r"\hyperref[note:src2]" in bib


def test_registry_note_truncation() -> None:
    reg = sources.SourceRegistry()
    long_note = reg.register_file({"id": "n", "type": "note", "title": "Long",
                                   "data": {"content": {"md": "x" * sources.NOTE_TRUNCATION_LIMIT}}})
    assert long_note is not None
    assert long_note.content_kind == "truncated"
    assert "at least" in long_note.counts_text()
    short = reg.register_file({"id": "s", "type": "note", "data": {"content": "short text"}})
    assert short is not None
    assert short.has_full_content
    # A retrieved chunk that is not in the stored text means that the text is incomplete.
    reg.register_source({"source": {"id": "s", "type": "note"}, "document": ["something else"]})
    assert short.content_kind == "truncated"
    short.set_full_content("short text and something else", created_at=1_700_000_000)
    assert short.has_full_content
    assert reg.register_file({"no": "id"}) is None
    assert [s.ident for s in reg.notes_with_content()] == ["n", "s"]


def test_source_metadata_text() -> None:
    src = sources.Source(key="k", kind="file", title="t", content="two words", content_type="text/plain",
                         size=2048, created_at="2026-01-01 10:00")
    assert src.metadata_text() == "created 2026-01-01 10:00; text/plain; 2 kB; 9 characters, 2 words"
    assert src.label() == "Uploaded file"
    assert sources.Source(key="k", kind="web", title="t").label() == "Web page"


def test_convert_citation_marks() -> None:
    cite_map = {1: "src1"}
    assert sources.convert_citation_marks("see [1] and [2]", cite_map) == r"see `\cite{src1}`{=latex} and [2]"
    # Links, footnotes and array indices are not citations.
    for text in ("[1](url)", "[1]: def", "a[1]", "[[1]]"):
        assert sources.convert_citation_marks(text, cite_map) == text
    assert sources.convert_citation_marks("[1]", {}) == "[1]"
    assert sources.cite_inline(["a", "b"]) == r"`\cite{a,b}`{=latex}"


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

def test_load_note_library(tmp_path: Path) -> None:
    lib_dir = tmp_path / "notes"
    lib_dir.mkdir()
    write_json(lib_dir / "one.json", {"id": "n1", "title": "One", "data": {"content": {"md": "full one"}}})
    write_json(lib_dir / "many.json", {"items": [{"id": "n2", "content": "full two"}, {"id": "n3"}, {"x": 1}]})
    (lib_dir / "n4.md").write_text("full four", encoding="utf-8")
    (lib_dir / "bad.json").write_text("{", encoding="utf-8")
    library = notes.load_note_library([lib_dir])
    assert {k: v["content"] for k, v in library.items()} == {"n1": "full one", "n2": "full two", "n4": "full four"}
    with pytest.raises(SystemExit, match="not found"):
        notes.load_note_library([tmp_path / "missing"])


def _note_registry() -> sources.SourceRegistry:
    reg = sources.SourceRegistry()
    reg.register_source({"source": {"id": "n1", "type": "note", "name": "One"}, "document": ["chunk"]})
    reg.register_source({"source": {"id": "other", "type": "note", "name": "By title"}, "document": ["chunk"]})
    return reg


def test_resolve_notes_from_library() -> None:
    reg = _note_registry()
    library = {"n1": {"id": "n1", "title": "One", "content": "full one", "created_at": None, "updated_at": None},
               "x": {"id": "x", "title": "By title", "content": "found by title"}}
    notes.resolve_notes(reg, argparse.Namespace(note_library=library))
    assert [(s.content, s.content_kind) for s in reg.sources.values()] == [
        ("full one", "full"), ("found by title", "full")]


class _NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the signature of the base class)
        pass

    def do_GET(self) -> None:
        self.server.requests.append((self.path, self.headers.get("Authorization")))  # type: ignore[attr-defined]
        if self.path == "/api/v1/notes/n1":
            body = json.dumps({"id": "n1", "title": "One", "data": {"content": {"md": "fetched"}},
                               "created_at": 1_700_000_000}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def test_resolve_notes_from_server() -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _NotesHandler)
    httpd.requests = []  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True).start()
    try:
        reg = _note_registry()
        opts = argparse.Namespace(notes_url=f"http://127.0.0.1:{httpd.server_address[1]}", notes_token="tok")
        notes.resolve_notes(reg, opts)
        notes.resolve_notes(reg, opts)  # the failed fetch is cached
    finally:
        httpd.shutdown()
        httpd.server_close()
    one, other = reg.sources.values()
    assert (one.content, one.content_kind) == ("fetched", "full")
    assert one.created_at
    assert other.content_kind == "excerpts"
    assert httpd.requests == [("/api/v1/notes/n1", "Bearer tok"), ("/api/v1/notes/other", "Bearer tok")]  # type: ignore[attr-defined]


def test_fetch_note_unreachable(capsys: pytest.CaptureFixture[str]) -> None:
    assert notes.fetch_note("http://127.0.0.1:1", "tok", "n1") is None
    assert "could not fetch note n1" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

def _font(path: Path, tables: dict[bytes, bytes]) -> Path:
    """A minimal sfnt file with the given tables (only the table directory and the data)."""
    offset = 12 + 16 * len(tables)
    directory, data = b"", b""
    for tag, payload in tables.items():
        directory += struct.pack(">4sIII", tag, 0, offset + len(data), len(payload))
        data += payload
    sfnt = b"\0\1\0\0" + struct.pack(">HHHH", len(tables), 0, 0, 0)[:8] + directory + data
    path.write_bytes(sfnt)
    return path


def test_font_color_format(tmp_path: Path) -> None:
    assert fonts.font_color_format(str(_font(tmp_path / "a.ttf", {b"CBDT": b"x", b"glyf": b"y"}))) == "CBDT"
    assert fonts.font_color_format(str(_font(tmp_path / "b.ttf", {b"COLR": b"\0\0"}))) == "COLR0"
    assert fonts.font_color_format(str(_font(tmp_path / "c.ttf", {b"COLR": b"\0\1"}))) == "COLR1"
    assert fonts.font_color_format(str(_font(tmp_path / "d.ttf", {b"sbix": b""}))) == "sbix"
    assert fonts.font_color_format(str(_font(tmp_path / "e.ttf", {b"glyf": b""}))) is None
    assert fonts.font_color_format(str(tmp_path / "missing.ttf")) is None
    (tmp_path / "short.ttf").write_bytes(b"abc")
    assert fonts.font_color_format(str(tmp_path / "short.ttf")) is None


def test_font_color_format_collection(tmp_path: Path) -> None:
    path = tmp_path / "c.ttc"
    inner = _font(tmp_path / "inner.ttf", {b"CBDT": b"x"})
    # A collection whose first font starts after the 16-byte collection header
    raw = inner.read_bytes()
    directory_end = 12 + 16
    tag, cs, off, length = struct.unpack(">4sIII", raw[12:28])
    raw = raw[:12] + struct.pack(">4sIII", tag, cs, off + 16, length) + raw[directory_end:]
    path.write_bytes(b"ttcf" + b"\0\1\0\0" + struct.pack(">II", 1, 16) + raw)
    assert fonts.font_color_format(str(path)) == "CBDT"


def test_find_emoji_font(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = _font(tmp_path / "any.ttf", {b"glyf": b""})
    assert fonts.find_emoji_font(str(explicit)) == str(explicit.resolve())
    with pytest.raises(SystemExit, match="emoji font not found"):
        fonts.find_emoji_font(str(tmp_path / "missing.ttf"))
    _font(tmp_path / "TwemojiMozilla.ttf", {b"COLR": b"\0\0"})
    _font(tmp_path / "NotoColorEmoji.ttf", {b"CBDT": b""})
    _font(tmp_path / "Colrv1Emoji.ttf", {b"COLR": b"\0\1"})
    monkeypatch.setattr(fonts.shutil, "which", lambda _: None)
    monkeypatch.setattr(fonts, "EMOJI_FONT_GLOBS", [str(tmp_path / "*.ttf")])
    # CBDT is preferred over COLRv0, and COLRv1 is not supported.
    assert fonts.find_emoji_font(None) == str(tmp_path / "NotoColorEmoji.ttf")
    monkeypatch.setattr(fonts, "EMOJI_FONT_GLOBS", [str(tmp_path / "Colr*.ttf")])
    assert fonts.find_emoji_font(None) is None


def test_build_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fonts, "fc_families", lambda: {"Noto Serif", "DejaVu Sans"})
    tex = fonts.build_header(options(), "/fonts/NotoColorEmoji.ttf", use_bib=True)
    assert "@@" not in tex
    assert "Noto Serif" in tex
    assert "Latin Modern Mono" in tex  # no preferred monospace font installed
    assert r"\newfontfamily\owuiemojifont{NotoColorEmoji.ttf}[Path=/fonts/" in tex
    assert r"\addbibresource{refs.bib}" in tex
    plain = fonts.build_header(options(main_font="My Serif"), None)
    assert "My Serif" in plain
    assert r"\newcommand{\owuiemoji}[1]{#1}" in plain
    assert r"\addbibresource" not in plain


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def test_collect_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "sub").mkdir()
    a = write_json(tmp_path / "a.json", {})
    b = write_json(tmp_path / "sub" / "b.json", {})
    jobs = cli.collect_inputs([tmp_path, a])
    assert jobs == [(a, tmp_path), (b, tmp_path)]  # a.json only once
    (tmp_path / "empty").mkdir()
    assert cli.collect_inputs([tmp_path / "empty"]) == []
    assert "no *.json files" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="input not found"):
        cli.collect_inputs([tmp_path / "missing.json"])


def test_output_paths(tmp_path: Path) -> None:
    json_path = tmp_path / "in" / "sub" / "export.json"
    one = [{"title": "Only"}]
    assert cli.output_paths(json_path, tmp_path / "in", one, options(output=None, outdir=None)) == [
        json_path.with_suffix(".pdf")]
    assert cli.output_paths(json_path, tmp_path / "in", one, options(output=tmp_path / "o.pdf")) == [
        tmp_path / "o.pdf"]
    many = [{"title": "Same"}, {"title": "Same"}, {}]
    paths = cli.output_paths(json_path, tmp_path / "in", many, options(output=None, outdir=tmp_path / "pdfs"))
    assert [p.name for p in paths] == ["export-001-same.pdf", "export-002-same.pdf", "export-003-chat.pdf"]
    assert paths[0].parent == tmp_path / "pdfs" / "sub"


def test_is_up_to_date(tmp_path: Path) -> None:
    json_path = write_json(tmp_path / "a.json", {})
    pdf = tmp_path / "a.pdf"
    assert not cli.is_up_to_date(pdf, json_path)
    pdf.write_bytes(b"%PDF")
    os.utime(json_path, (1, 1))
    assert cli.is_up_to_date(pdf, json_path)
    os.utime(pdf, (0, 0))
    assert not cli.is_up_to_date(pdf, json_path)


def test_main_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "which_or_die", lambda name: name)
    with pytest.raises(SystemExit, match="single input file"):
        cli.main([str(tmp_path), "-o", str(tmp_path / "x.pdf")])
    two = write_json(tmp_path / "two.json", [{"chat": {}}, {"chat": {}}])
    with pytest.raises(SystemExit, match="single-chat export"):
        cli.main([str(two), "-o", str(tmp_path / "x.pdf"), "--no-emoji-font"])
    monkeypatch.delenv("OWUI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="needs an API key"):
        cli.main([str(two), "--notes-url", "http://localhost"])


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def test_latex_errors(tmp_path: Path) -> None:
    log = tmp_path / "doc.log"
    log.write_text("ok\n! Undefined control sequence.\n./doc.tex:12: Missing $ inserted.\n"
                   "! Undefined control sequence.\n", encoding="utf-8")
    assert build.latex_errors(log) == ["! Undefined control sequence.", "./doc.tex:12: Missing $ inserted."]


def _fake_latexmk(calls: list[list[str]], fail: bool = False) -> Any:
    """``util.run`` with the real pandoc and a latexmk that writes an empty PDF and a log."""
    real_run = util.run

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[0] != "latexmk":
            return real_run(cmd, **kw)
        build_dir = Path(kw["cwd"])
        (build_dir / "doc.log").write_text("! A harmless error.\n", encoding="utf-8")
        if not fail:
            (build_dir / "doc.pdf").write_bytes(b"%PDF-1.5 fake")
        return subprocess.CompletedProcess(cmd, 0 if not fail else 1, "latexmk output", "")

    return run


needs_pandoc: pytest.MarkDecorator = pytest.mark.skipif(
    shutil.which("pandoc") is None, reason="pandoc is not installed")


@needs_pandoc
def test_convert_with_pandoc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(build, "run", _fake_latexmk(calls))
    monkeypatch.setattr(build, "which_or_die", lambda name: name)
    monkeypatch.setattr(cli, "which_or_die", lambda name: name)
    out = tmp_path / "out" / "example.pdf"
    assert cli.main([str(EXAMPLE), "-o", str(out), "--no-emoji-font", "--keep-tex", "--include-reasoning"]) == 0
    assert out.read_bytes() == b"%PDF-1.5 fake"
    build_dir = tmp_path / "out" / "example_tex"
    tex = (build_dir / "doc.tex").read_text(encoding="utf-8")
    assert r"\begin{document}" in tex
    assert r"\owuiheader{" in tex
    assert (build_dir / "filter.lua").exists()
    latexmk = next(c for c in calls if c[0] == "latexmk")
    assert "-f" in latexmk
    err = capsys.readouterr().err
    assert "LaTeX reported 1 error(s)" in err


@needs_pandoc
def test_convert_failure_and_skip_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(build, "run", _fake_latexmk(calls, fail=True))
    monkeypatch.setattr(build, "which_or_die", lambda name: name)
    monkeypatch.setattr(cli, "which_or_die", lambda name: name)
    source = shutil.copy(EXAMPLE, tmp_path / "chat.json")
    # The default build directory is a temporary one, which is removed also when LaTeX fails.
    before = set(Path(os.environ.get("TMPDIR", "/tmp")).glob("owui2pdf-*"))
    assert cli.main([str(source), "--no-emoji-font", "--no-sources", "--strict"]) == 1
    assert set(Path(os.environ.get("TMPDIR", "/tmp")).glob("owui2pdf-*")) == before
    assert "-halt-on-error" in next(c for c in calls if c[0] == "latexmk")
    assert "LaTeX failed" in capsys.readouterr().err
    assert not (tmp_path / "chat.pdf").exists()
    # An up-to-date PDF is skipped without running anything.
    (tmp_path / "chat.pdf").write_bytes(b"%PDF")
    calls.clear()
    assert cli.main([str(tmp_path), "--no-emoji-font", "--skip-existing", "-v"]) == 0
    assert calls == []


@pytest.mark.slow
@pytest.mark.skipif(any(shutil.which(t) is None for t in TOOLS), reason="pandoc or TeX Live is not installed")
def test_example_pdf(tmp_path: Path) -> None:
    """The real build of example.json (about 30 s)."""
    out = tmp_path / "example.pdf"
    assert cli.main([str(EXAMPLE), "-o", str(out)]) == 0
    assert out.read_bytes().startswith(b"%PDF")
    assert out.stat().st_size > 10_000
