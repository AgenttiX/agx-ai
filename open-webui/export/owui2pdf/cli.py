"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import convert_chat
from .chat import chat_title, load_chats
from .fonts import find_emoji_font
from .util import slugify, which_or_die


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="owui2pdf",
                                description="Convert an Open WebUI JSON chat export to PDF.")
    p.add_argument("input", type=Path, help="JSON export (single chat or list of chats)")
    p.add_argument("-o", "--output", type=Path, help="output PDF (single-chat exports only)")
    p.add_argument("--outdir", type=Path, help="directory for output PDFs (default: next to input)")
    p.add_argument("--keep-tex", action="store_true",
                   help="keep the LaTeX build directory (<name>_tex) next to the PDF")
    p.add_argument("--hard-breaks", action="store_true", help="treat single newlines as line breaks")
    p.add_argument("--include-reasoning", action="store_true",
                   help="include model reasoning/thinking blocks")
    p.add_argument("--no-sources", dest="show_sources", action="store_false",
                   help="omit RAG/web sources")
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


def output_paths(chats: list[dict], args) -> list[Path]:
    outdir = args.outdir or args.input.resolve().parent
    if args.output and len(chats) > 1:
        sys.exit("error: -o/--output can only be used with a single-chat export; use --outdir")
    if args.output:
        return [args.output]
    if len(chats) == 1:
        return [outdir / (args.input.stem + ".pdf")]
    paths, used = [], set()
    for i, chat_item in enumerate(chats, 1):
        name = f"{i:03d}-{slugify(chat_title(chat_item))}"
        while name in used:
            name += "-x"
        used.add(name)
        paths.append(outdir / f"{name}.pdf")
    return paths


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    for tool in ("pandoc", "latexmk", "lualatex"):
        which_or_die(tool)
    if not args.input.is_file():
        sys.exit(f"error: input file not found: {args.input}")

    emoji_font = None if args.no_emoji_font else find_emoji_font(args.emoji_font)
    if emoji_font is None and not args.no_emoji_font:
        print("warning: no colour emoji font found; emoji will be missing or monochrome "
              "(install Noto Color Emoji or pass --emoji-font)", file=sys.stderr)
    elif args.verbose:
        print(f"emoji font: {emoji_font}", file=sys.stderr)

    chats = load_chats(args.input)
    ok = True
    for chat_item, out_pdf in zip(chats, output_paths(chats, args)):
        ok &= convert_chat(chat_item, out_pdf, args, emoji_font)
    return 0 if ok else 1
