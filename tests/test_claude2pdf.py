"""Tests of agx_ai.claude2pdf.

The transcript is synthetic. The pandoc test replaces latexmk with a stub, as a LuaLaTeX build is slow.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from agx_ai.claude2pdf import cli, markdown, session
from agx_ai.convert import build

PNG: str = base64.b64encode(b"\x89PNG\r\n\x1a\n fake").decode()


def _rec(kind: str, uuid: str, parent: str | None, body: Any, **extra: Any) -> dict:
    rec = {"type": kind, "uuid": uuid, "parentUuid": parent, "timestamp": f"2026-11-16T12:{len(uuid):02d}:00Z",
           "cwd": "/home/user/repo", "gitBranch": "main", "version": "2.1.0",
           "message": {"role": kind, "content": body}}
    rec.update(extra)
    return rec


def _assistant(uuid: str, parent: str, msg_id: str, block: dict) -> dict:
    rec = _rec("assistant", uuid, parent, [block])
    rec["message"]["id"] = msg_id
    rec["message"]["model"] = "claude-test"
    return rec


def transcript() -> list[dict]:
    """A session with a rewound prompt, parallel tool calls, an image, a skill and a queued prompt."""
    return [
        {"type": "custom-title", "customTitle": "Test session", "sessionId": "s"},
        _rec("user", "u1", None, "Please <fix> the `<b>` bug.\nSecond line"),
        _rec("attachment", "a1", "u1", None, attachment={"type": "date", "date": "2026-11-16"}),
        _assistant("a2", "a1", "m1", {"type": "thinking", "thinking": "Hmm, let me think."}),
        _assistant("a3", "a2", "m1", {"type": "tool_use", "id": "t1", "name": "Bash",
                                      "input": {"command": "ls ```", "description": "List files"}}),
        _assistant("a4", "a3", "m1", {"type": "tool_use", "id": "t2", "name": "Edit",
                                      "input": {"file_path": "/x/y.py", "old_string": "a", "new_string": "b"}}),
        _rec("user", "r1", "a4", [{"type": "tool_result", "tool_use_id": "t1", "content": "\x1b[31mred\x1b[0m"}]),
        _rec("user", "r2", "a4", [{"type": "tool_result", "tool_use_id": "t2", "is_error": True,
                                   "content": [{"type": "text", "text": "failed"},
                                               {"type": "image", "source": {"type": "base64",
                                                                            "media_type": "image/png",
                                                                            "data": PNG}}]}]),
        _rec("attachment", "q1", "r2", None, attachment={
            "type": "queued_command", "prompt": "Also this", "origin": {"kind": "human"}}),
        _assistant("a5", "q1", "m2", {"type": "text", "text": "Done: $x^2$ and <key>."}),
        # An abandoned branch: the prompt was rewound and resent.
        _rec("user", "old", "a5", "Abandoned prompt"),
        _rec("user", "u2", "a5", "<command-message>skill</command-message>\n<command-name>/skill</command-name>\n"
                                 "<command-args>arg</command-args>"),
        _rec("user", "u3", "u2", [{"type": "text", "text": "Base directory for this skill: /s\n\n# Skill"}],
             isMeta=True),
        _rec("system", "s1", "u3", None, subtype="compact_boundary", content="Conversation compacted",
             compactMetadata={"trigger": "manual", "preTokens": 1234}),
        _rec("user", "u4", "s1", "[Request interrupted by user]"),
    ]


@pytest.fixture
def jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "proj" / "abcdef12-0000.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(r) for r in transcript()) + "\nnot json\n", encoding="utf-8")
    return path


def options(level: str = "results", **kw: Any) -> argparse.Namespace:
    return markdown.options_for_level(argparse.Namespace(level=level, thinking=None, meta=None, max_lines=None,
                                                         **kw))


def test_project_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "PROJECTS_DIR", tmp_path)
    assert session.project_dir(Path("/home/a.b/my_repo")) == tmp_path / "-home-a-b-my-repo"


def test_active_branch_and_events(jsonl: Path) -> None:
    events = session.session_events(session.read_records(jsonl))
    kinds = [e.kind for e in events]
    assert kinds == ["prompt", "thinking", "tool_use", "tool_use", "tool_result", "tool_result", "prompt", "text",
                     "prompt", "meta", "meta", "interrupt"]
    assert all(e.text != "Abandoned prompt" for e in events)
    assert events[6].data == {"queued": True}
    assert events[8].text == "/skill arg"
    assert events[8].data == {"command": True}
    assert events[9].label == "Skill instructions"
    assert "1,234 tokens" in events[10].text
    assert events[5].data["is_error"]
    assert len(events[5].images) == 1


def test_session_info(jsonl: Path) -> None:
    info = session.session_info(jsonl)
    assert info.name == "Test session"
    assert info.prompts == 2
    assert info.models == ["claude-test"]
    assert info.start is not None
    assert info.end is not None
    assert info.start < info.end
    info.title = ""
    assert info.name == "Please <fix> the <b> bug. Second line"


def test_list_and_select(jsonl: Path, capsys: pytest.CaptureFixture[str]) -> None:
    infos = session.list_sessions(jsonl.parent)
    assert [i.session_id for i in infos] == ["abcdef12-0000"]
    assert cli.main([str(jsonl.parent)]) == 0
    out = capsys.readouterr().out
    assert "Test session" in out
    assert "abcdef12" in out
    assert cli.select_sessions(infos, ["1"]) == infos
    assert cli.select_sessions(infos, ["abc", "all"]) == infos
    with pytest.raises(SystemExit, match="matches 0 sessions"):
        cli.select_sessions(infos, ["zzz"])


def test_truncate_and_fence() -> None:
    text = "\n".join(str(i) for i in range(10))
    assert markdown.truncate(text, 0) == text
    assert markdown.truncate(text, 3) == "0\n1\n2\n[… 7 more lines]"
    assert markdown.truncate("x" * 1000, 1).endswith("[… 840 more characters]")
    assert markdown.fence("a ```` b", "py").startswith("`````py\n")
    assert markdown.sanitize("\x1b[1mbold\x1b[0m\tx\x07") == "bold    x"


def test_escape_html() -> None:
    md = "a <key> `<b>` b\n```\n<keep>\n```\n<c>"
    assert markdown.escape_html(md) == "a \\<key> `<b>` b\n```\n<keep>\n```\n\\<c>"


@pytest.mark.parametrize(("level", "present", "absent"), [
    ("chat", ["Done:", "Also this", "Request interrupted"], ["List files", "Thinking", "red", "Skill"]),
    ("tools", ["List files", "+b"], ["red", "Thinking"]),
    ("results", ["Result of Bash", "Error of Edit", "red", "img1.png"], ["Thinking", "# Skill"]),
    ("full", ["Thinking", "# Skill", "Skill instructions"], []),
])
def test_build_markdown_levels(jsonl: Path, tmp_path: Path, level: str, present: list[str],
                               absent: list[str]) -> None:
    records = session.read_records(jsonl)
    md, meta, bib = markdown.build_markdown(session.session_info(jsonl, records), session.session_events(records),
                                            options(level), tmp_path)
    assert bib is None
    assert meta["title"] == "Test session"
    assert meta["subtitle"] == "/home/user/repo (main)"
    for text in present:
        assert text in md, text
    for text in absent:
        assert text not in md, text
    assert "Please \\<fix> the `<b>` bug." in md


def test_tool_use_markdown() -> None:
    assert markdown.tool_use_markdown("Read", {"file_path": "/a", "offset": 5, "limit": 2}, 0) == (
        "/a (from line 5, 2 lines)", "")
    detail, body = markdown.tool_use_markdown("Write", {"file_path": "/a.py", "content": "x = 1"}, 0)
    assert detail == "/a.py"
    assert body.startswith("```python")
    assert markdown.tool_use_markdown("Grep", {"pattern": "p", "path": "src"}, 0) == ("p in src", "")
    assert markdown.tool_use_markdown("Other", {"query": "q"}, 0) == ("q", "")
    assert "```json" in markdown.tool_use_markdown("Other", {"a": 1, "b": 2}, 0)[1]


def test_options_for_level() -> None:
    assert (options("tools").thinking, options("tools").max_lines) == (False, markdown.DEFAULT_MAX_LINES)
    full = options("full")
    assert (full.thinking, full.meta, full.max_lines) == (True, True, 0)


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc is not installed")
def test_convert_with_pandoc(jsonl: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = build.run

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        if cmd[0] != "latexmk":
            return real_run(cmd, **kw)
        (Path(kw["cwd"]) / "doc.pdf").write_bytes(b"%PDF-1.5 fake")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(build, "run", run)
    monkeypatch.setattr(cli, "which_or_die", lambda name: name)
    outdir = tmp_path / "dist"
    assert cli.main([str(jsonl), str(jsonl), "--outdir", str(outdir), "--no-emoji-font", "-l", "full", "-j", "2"]) == 0
    pdf = outdir / "2026-11-16-test-session-abcdef12-full.pdf"
    assert pdf.read_bytes() == b"%PDF-1.5 fake"
    md = pdf.with_suffix(".md").read_text(encoding="utf-8")
    assert "![](2026-11-16-test-session-abcdef12-full_media/img1.png)" in md
    assert (outdir / "2026-11-16-test-session-abcdef12-full_media" / "img1.png").is_file()
    assert cli.main([str(jsonl), "-o", str(tmp_path / "x.pdf"), "--no-markdown", "--no-emoji-font"]) == 0
    assert (tmp_path / "x.pdf").exists()
    assert not (tmp_path / "x.md").exists()
