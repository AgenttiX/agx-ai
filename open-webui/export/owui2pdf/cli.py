"""Command line interface."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .build import convert_chat
from .chat import chat_title, load_chats
from .fonts import find_emoji_font
from .notes import load_note_library
from .util import slugify, which_or_die


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="owui2pdf",
        description="Convert Open WebUI JSON chat exports to PDF. INPUT may be a JSON file "
                    "(single chat or list of chats) or a directory, which is searched "
                    "recursively for *.json files; each PDF is written next to its JSON "
                    "file with the same name.")
    p.add_argument("input", type=Path, nargs="+", help="JSON export file(s) or directory(ies)")
    p.add_argument("-o", "--output", type=Path,
                   help="output PDF (only for a single input file containing one chat)")
    p.add_argument("--outdir", type=Path,
                   help="directory for output PDFs (default: next to each input file; "
                        "for directory inputs the directory structure is mirrored)")
    p.add_argument("--skip-existing", action="store_true",
                   help="do not rebuild PDFs that already exist and are newer than the JSON")
    p.add_argument("--keep-tex", action="store_true",
                   help="keep the LaTeX build directory (<name>_tex) next to the PDF")
    p.add_argument("--hard-breaks", action="store_true", help="treat single newlines as line breaks")
    p.add_argument("--user-paragraphs", action="store_true",
                   help="in user messages, treat single newlines between plain text lines as "
                        "paragraph breaks (for prompts typed with one Enter between paragraphs)")
    p.add_argument("--include-reasoning", action="store_true",
                   help="include model reasoning/thinking blocks")
    p.add_argument("--no-sources", dest="show_sources", action="store_false",
                   help="omit citations and the bibliography of referenced notes/files/web pages")
    p.add_argument("--include-notes", action="store_true",
                   help="append the full text of referenced notes as an appendix "
                        "(default: only metadata in the bibliography)")
    p.add_argument("--notes", type=Path, action="append", default=[], metavar="PATH",
                   help="JSON export(s) of notes (GET /api/v1/notes/<id>), Markdown files named "
                        "<note id>.md, or directories of these, providing the full note text "
                        "(chat exports contain at most the first 1000 characters); repeatable")
    p.add_argument("--notes-url", metavar="URL",
                   help="Open WebUI base URL, e.g. https://ai.example.com, to fetch the full text "
                        "of referenced notes; the API key is read from --notes-token or the "
                        "OWUI_API_KEY environment variable")
    p.add_argument("--notes-token", metavar="KEY", help="API key for --notes-url")
    p.add_argument("--no-usage", dest="show_usage", action="store_false",
                   help="omit token usage in message headers")
    p.add_argument("--emoji-font", help="path to a colour emoji font (CBDT/COLRv0 TTF)")
    p.add_argument("--no-emoji-font", action="store_true", help="do not use a colour emoji font")
    p.add_argument("--main-font", help="serif body font (default: Noto Serif)")
    p.add_argument("--sans-font", help="sans font (default: Noto Sans)")
    p.add_argument("--mono-font", help="monospace font (default: Noto Sans Mono)")
    p.add_argument("--math-font", help="math font (default: TeX Gyre Pagella Math)")
    p.add_argument("--paper", default="a4", help="paper size: a4, letter, ... (default: a4)")
    p.add_argument("--font-size", default="10pt", help="base font size (default: 10pt)")
    p.add_argument("--highlight-style", default="tango",
                   help="pandoc highlight style (default: tango)")
    p.add_argument("--strict", action="store_true",
                   help="fail on the first LaTeX error instead of continuing")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def collect_inputs(inputs: list[Path]) -> list[tuple[Path, Path]]:
    """Expand files/directories into (json file, base directory) pairs.

    The base directory is the input directory the file was found in (or the
    file's own directory) and is used to mirror the layout under --outdir.
    """
    jobs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for inp in inputs:
        if inp.is_dir():
            files = sorted(p for p in inp.rglob("*.json") if p.is_file())
            if not files:
                print(f"warning: no *.json files found under {inp}", file=sys.stderr)
            base = inp
        elif inp.is_file():
            files, base = [inp], inp.parent
        else:
            sys.exit(f"error: input not found: {inp}")
        for f in files:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                jobs.append((f, base))
    return jobs


def output_paths(json_path: Path, base: Path, chats: list[dict], args) -> list[Path]:
    """PDF paths for the chats of one JSON file: <stem>.pdf, or <stem>-NNN-<title>.pdf."""
    if args.output:
        return [args.output]
    if args.outdir:
        outdir = args.outdir / json_path.parent.resolve().relative_to(base.resolve())
    else:
        outdir = json_path.parent
    if len(chats) == 1:
        return [outdir / (json_path.stem + ".pdf")]
    paths, used = [], set()
    for i, chat_item in enumerate(chats, 1):
        name = f"{json_path.stem}-{i:03d}-{slugify(chat_title(chat_item))}"
        while name in used:
            name += "-x"
        used.add(name)
        paths.append(outdir / f"{name}.pdf")
    return paths


def is_up_to_date(pdf: Path, json_path: Path) -> bool:
    try:
        return pdf.stat().st_mtime >= json_path.stat().st_mtime
    except OSError:
        return False


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    for tool in ("pandoc", "latexmk", "lualatex"):
        which_or_die(tool)
    jobs = collect_inputs(args.input)
    if args.output and (len(jobs) != 1 or args.input[0].is_dir()):
        sys.exit("error: -o/--output requires a single input file; use --outdir instead")

    args.note_library = load_note_library(args.notes) if args.notes else {}
    if args.notes_url:
        args.notes_token = args.notes_token or os.environ.get("OWUI_API_KEY")
        if not args.notes_token:
            sys.exit("error: --notes-url needs an API key (--notes-token or OWUI_API_KEY)")

    emoji_font = None if args.no_emoji_font else find_emoji_font(args.emoji_font)
    if emoji_font is None and not args.no_emoji_font:
        print("warning: no colour emoji font found; emoji will be missing or monochrome "
              "(install Noto Color Emoji or pass --emoji-font)", file=sys.stderr)
    elif args.verbose:
        print(f"emoji font: {emoji_font}", file=sys.stderr)

    converted = skipped = failed = 0
    for json_path, base in jobs:
        try:
            chats = load_chats(json_path)
        except ValueError as e:
            print(f"warning: skipping {json_path}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if args.output and len(chats) > 1:
            sys.exit("error: -o/--output can only be used with a single-chat export; use --outdir")
        for chat_item, out_pdf in zip(chats, output_paths(json_path, base, chats, args)):
            if args.skip_existing and is_up_to_date(out_pdf, json_path):
                if args.verbose:
                    print(f"up to date: {out_pdf}")
                skipped += 1
                continue
            if convert_chat(chat_item, out_pdf, args, emoji_font):
                converted += 1
            else:
                failed += 1
    if len(jobs) > 1 or failed:
        print(f"{converted} PDF(s) written, {skipped} skipped, {failed} failed", file=sys.stderr)
    return 0 if failed == 0 else 1
