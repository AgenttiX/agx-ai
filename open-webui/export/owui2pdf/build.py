"""Running pandoc and latexmk to turn one chat into a PDF."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from . import RESOURCE_DIR
from .fonts import build_header
from .markdown import build_markdown
from .util import run

# pandoc's commonmark_x is close to the GFM dialect Open WebUI (marked) uses.
# Extensions that marked does not have are switched off for fidelity.
PANDOC_FORMAT = ("commonmark_x-smart-subscript-superscript-attributes-bracketed_spans"
                 "-definition_lists-implicit_header_references-yaml_metadata_block-emoji"
                 "+raw_attribute+autolink_bare_uris")


def pandoc_langs() -> list[str]:
    out = run(["pandoc", "--list-highlight-languages"]).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def latex_errors(log: Path) -> list[str]:
    errors = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("!") or re.match(r"^\./doc\.tex:\d+: ", line):
            errors.append(line)
    return list(dict.fromkeys(errors))


def convert_chat(chat_item: dict, out_pdf: Path, opts, emoji_font: str | None) -> bool:
    """Build ``out_pdf`` from one chat. Returns True when a PDF was written."""
    if opts.keep_tex:
        build_dir = (out_pdf.parent / (out_pdf.stem + "_tex")).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="owui2pdf-")
        build_dir = Path(tmp.name).resolve()

    media_dir = build_dir / "media"
    media_dir.mkdir(exist_ok=True)
    md, meta = build_markdown(chat_item, opts, media_dir)

    (build_dir / "doc.md").write_text(md, encoding="utf-8")
    (build_dir / "header.tex").write_text(build_header(opts, emoji_font), encoding="utf-8")
    shutil.copyfile(RESOURCE_DIR / "filter.lua", build_dir / "filter.lua")
    (build_dir / "langs.txt").write_text("\n".join(pandoc_langs()), encoding="utf-8")

    fmt = PANDOC_FORMAT + ("+hard_line_breaks" if opts.hard_breaks else "")
    cmd = [
        "pandoc", "doc.md", "-f", fmt, "-t", "latex", "-s", "-o", "doc.tex",
        "--wrap=none",
        "-H", "header.tex",
        "--lua-filter", "filter.lua",
        "--highlight-style", opts.highlight_style,
        "-V", f"papersize={opts.paper}",
        "-V", f"fontsize={opts.font_size}",
        "-V", "colorlinks=true", "-V", "urlcolor=NavyBlue", "-V", "linkcolor=NavyBlue",
        "-V", "citecolor=NavyBlue",
        "-M", f"title={meta['title']}",
        "-M", f"date={meta['date']}",
    ]
    if meta["subtitle"]:
        cmd += ["-M", f"subtitle={meta['subtitle']}"]
    for kw in meta["keywords"]:
        cmd += ["-M", f"keywords={kw}"]

    env = dict(os.environ, OWUI_LANGS="langs.txt", max_print_line="10000")
    res = run(cmd, cwd=build_dir, env=env)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        print(f"error: pandoc failed for '{meta['title']}'", file=sys.stderr)
        return False
    if res.stderr.strip() and opts.verbose:
        print(res.stderr, file=sys.stderr)

    lm = ["latexmk", "-lualatex", "-interaction=nonstopmode", "-file-line-error",
          "-halt-on-error" if opts.strict else "-f", "doc.tex"]
    res = run(lm, cwd=build_dir, env=env)
    pdf = build_dir / "doc.pdf"
    log = build_dir / "doc.log"
    errors = latex_errors(log) if log.exists() else []
    if not pdf.exists():
        print(res.stdout[-4000:], file=sys.stderr)
        print("\n".join(errors[:20]), file=sys.stderr)
        print(f"error: LaTeX failed for '{meta['title']}' (build dir: {build_dir})", file=sys.stderr)
        if tmp:
            print("hint: rerun with --keep-tex to inspect the LaTeX sources", file=sys.stderr)
        return False

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, out_pdf)
    if errors:
        print(f"warning: LaTeX reported {len(errors)} error(s) while building '{out_pdf}'; "
              f"the PDF was produced but may have glitches:", file=sys.stderr)
        for e in errors[:15]:
            print("  " + e, file=sys.stderr)
    print(f"wrote {out_pdf}")
    if tmp:
        tmp.cleanup()
    return True
