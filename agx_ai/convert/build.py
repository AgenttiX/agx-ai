"""Running pandoc and latexmk to turn a Markdown document into a PDF."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from . import RESOURCE_DIR
from .fonts import build_header
from .util import run, which_or_die

# pandoc's commonmark_x is close to the GFM dialect that Open WebUI (marked) and Claude Code use.
# Extensions that marked does not have are switched off for fidelity.
PANDOC_FORMAT = ("commonmark_x-smart-subscript-superscript-attributes-bracketed_spans"
                 "-definition_lists-implicit_header_references-yaml_metadata_block-emoji"
                 "+raw_attribute+autolink_bare_uris")


def pandoc_langs() -> list[str]:
    """List the syntax highlighting languages supported by pandoc."""
    out = run(["pandoc", "--list-highlight-languages"]).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def latex_errors(log: Path) -> list[str]:
    """Extract the error lines from a LaTeX log."""
    errors = [
        line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("!") or re.match(r"^\./doc\.tex:\d+: ", line)
    ]
    return list(dict.fromkeys(errors))


def build_pdf(markdown: Callable[[Path], tuple[str, dict, str | None]], out_pdf: Path, opts: argparse.Namespace,
              emoji_font: str | None, prefix: str = "owui2pdf-", header_extra: str = "",
              save_markdown: bool = False) -> bool:
    """Build ``out_pdf`` from a Markdown document. Returns True when a PDF was written.

    ``markdown`` is called with the directory for extracted images and returns
    (markdown document, pandoc metadata, bibtex or None); the metadata has the
    keys ``title``, ``date``, ``subtitle`` and ``keywords``. ``header_extra`` is
    appended to the LaTeX header. With ``save_markdown`` the Markdown document
    is also saved next to the PDF (see :func:`write_markdown`). ``opts`` needs the attributes ``keep_tex``,
    ``hard_breaks``, ``highlight_style``, ``paper``, ``font_size``, ``strict``
    and ``verbose``, and the font overrides of :func:`build_header`.

    All intermediate files (Markdown, LaTeX sources, extracted images, TeX
    auxiliary files, bibliography) are written to one build directory. Without
    ``--keep-tex`` that is a fresh directory under ``$TMPDIR`` (usually /tmp)
    created with mode 0700 and removed again when the conversion finishes,
    also on errors. With ``--keep-tex`` it is ``<pdf name>_tex`` next to the
    PDF, created with the same (umask-based) permissions as the PDF.
    """
    if opts.keep_tex:
        build_dir = (out_pdf.parent / (out_pdf.stem + "_tex")).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        return _build(markdown, out_pdf, opts, emoji_font, build_dir, True, header_extra, save_markdown)
    # mkdtemp creates the directory with mode 0700; the context manager deletes it.
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        return _build(markdown, out_pdf, opts, emoji_font, Path(tmp).resolve(), False, header_extra,
                      save_markdown)


def write_markdown(md: str, media_dir: Path, path: Path) -> None:
    """Save the Markdown document, with the images of ``media_dir`` copied to ``<name>_media/`` next to it.

    The image links are rewritten to point at the copies. The document is
    pandoc Markdown with raw LaTeX blocks, as it is given to pandoc.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in media_dir.iterdir() if p.is_file())
    if images:
        copy_dir = path.with_name(path.stem + "_media")
        if copy_dir.exists():
            shutil.rmtree(copy_dir)
        copy_dir.mkdir()
        for image in images:
            shutil.copyfile(image, copy_dir / image.name)
        md = md.replace(media_dir.as_posix() + "/", copy_dir.name + "/")
    path.write_text(md, encoding="utf-8")


def _build(markdown: Callable[[Path], tuple[str, dict, str | None]], out_pdf: Path, opts: argparse.Namespace,
           emoji_font: str | None, build_dir: Path, keep: bool, header_extra: str, save_markdown: bool) -> bool:
    media_dir = build_dir / "media"
    media_dir.mkdir(exist_ok=True)
    md, meta, bib = markdown(media_dir)
    if save_markdown:
        write_markdown(md, media_dir, out_pdf.with_suffix(".md"))
    if bib is not None:
        which_or_die("biber")
        (build_dir / "refs.bib").write_text(bib, encoding="utf-8")

    (build_dir / "doc.md").write_text(md, encoding="utf-8")
    (build_dir / "header.tex").write_text(build_header(opts, emoji_font, use_bib=bib is not None) + header_extra,
                                          encoding="utf-8")
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
        if not keep:
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
    return True
