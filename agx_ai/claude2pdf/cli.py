"""Command line interface."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

from agx_ai.convert.build import build_pdf
from agx_ai.convert.fonts import find_emoji_font
from agx_ai.convert.util import date_time_options, slugify, which_or_die

from .markdown import DEFAULT_LEVEL, DEFAULT_MAX_LINES, HEADER_EXTRA, LEVELS, build_markdown, options_for_level
from .session import (
    PROJECTS_DIR,
    SessionInfo,
    fmt_time,
    list_sessions,
    project_dir,
    read_records,
    session_events,
    session_info,
)

DEFAULT_OUTDIR: Path = Path("dist")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="claude2pdf",
        description="Convert Claude Code sessions to PDF. Given .jsonl session files, convert them "
                    "(in parallel). Given a repository (or a directory of ~/.claude/projects), list its "
                    "sessions; select sessions to convert with -s.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="levels of verbosity (-l):\n"
               "  chat     only the prompts of the user and the text replies of Claude\n"
               "  tools    also the tool calls (default)\n"
               f"  results  also the results of the tool calls, truncated to {DEFAULT_MAX_LINES} lines\n"
               "  full     also the thinking, the text injected by Claude Code (skills, subagent\n"
               "           reports, task notifications, compaction summaries), nothing truncated")
    p.add_argument("input", type=Path, nargs="+",
                   help="session file(s) (.jsonl), or a repository or project directory to list the sessions of")
    p.add_argument("-s", "--session", action="append", default=[], metavar="N|ID|all",
                   help="with a repository: convert the session with this number in the list, "
                        "or whose ID starts with this, or all sessions; repeatable")
    p.add_argument("-l", "--level", choices=LEVELS, default=DEFAULT_LEVEL,
                   help=f"what to include (default: {DEFAULT_LEVEL}; see below)")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="include the thinking of Claude, if it was stored (default: only with -l full)")
    p.add_argument("--meta", action=argparse.BooleanOptionalAction, default=None,
                   help="include the text injected by Claude Code (default: only with -l full)")
    p.add_argument("--max-lines", type=int, default=None, metavar="N",
                   help=f"truncate tool calls and results to N lines, 0 for no limit "
                        f"(default: {DEFAULT_MAX_LINES}, with -l full 0)")
    date_time_options(p, "session")
    p.add_argument("-o", "--output", type=Path, help="output PDF (only for a single session)")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                   help=f"directory for the output PDFs (default: {DEFAULT_OUTDIR}/ in the working directory)")
    p.add_argument("-j", "--jobs", type=int, default=os.process_cpu_count(),
                   help="number of parallel conversions (default: number of CPUs)")
    p.add_argument("--no-markdown", dest="save_markdown", action="store_false",
                   help="do not save the intermediate Markdown document (<name>.md) next to the PDF")
    p.add_argument("--keep-tex", action="store_true",
                   help="keep the LaTeX build directory (<name>_tex) next to the PDF")
    p.add_argument("--hard-breaks", action="store_true", help="treat single newlines as line breaks")
    p.add_argument("--emoji-font", help="path to a colour emoji font (CBDT/COLRv0 TTF)")
    p.add_argument("--no-emoji-font", action="store_true", help="do not use a colour emoji font")
    p.add_argument("--main-font", help="serif body font (default: Noto Serif)")
    p.add_argument("--sans-font", help="sans font (default: Noto Sans)")
    p.add_argument("--mono-font", help="monospace font (default: Noto Sans Mono)")
    p.add_argument("--math-font", help="math font (default: TeX Gyre Pagella Math)")
    p.add_argument("--paper", default="a4", help="paper size: a4, letter, ... (default: a4)")
    p.add_argument("--font-size", default="10pt", help="base font size (default: 10pt)")
    p.add_argument("--highlight-style", default="tango", help="pandoc highlight style (default: tango)")
    p.add_argument("--strict", action="store_true", help="fail on the first LaTeX error instead of continuing")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def sessions_dir(path: Path) -> Path:
    """Return the directory of ``~/.claude/projects`` for a repository, or the path if it is one already."""
    if path.resolve().parent == PROJECTS_DIR.resolve() or any(path.glob("*.jsonl")):
        return path
    return project_dir(path)


def format_size(size: int) -> str:
    """Format a file size in kB or MB."""
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"


def print_sessions(directory: Path, infos: list[SessionInfo]) -> None:
    """Print a numbered table of the sessions."""
    print(f"Sessions in {directory}:")
    print(f"{'#':>3}  {'Started':16}  {'Last message':16}  {'Prompts':>7}  {'Size':>8}  {'ID':8}  Name")
    for i, info in enumerate(infos, 1):
        size = format_size(info.path.stat().st_size)
        print(f"{i:>3}  {fmt_time(info.start):16}  {fmt_time(info.end):16}  {info.prompts:>7}  {size:>8}  "
              f"{info.session_id[:8]:8}  {info.name}")


def select_sessions(infos: list[SessionInfo], selectors: list[str]) -> list[SessionInfo]:
    """Return the sessions chosen with ``-s``: numbers of the list, ID prefixes or ``all``."""
    chosen: dict[Path, SessionInfo] = {}
    for sel in selectors:
        if sel == "all":
            matches = infos
        elif sel.isdigit() and 1 <= int(sel) <= len(infos):
            matches = [infos[int(sel) - 1]]
        else:
            matches = [i for i in infos if i.session_id.startswith(sel)]
        if len(matches) != 1 and sel != "all":
            sys.exit(f"error: -s {sel} matches {len(matches)} sessions")
        chosen.update((m.path, m) for m in matches)
    return list(chosen.values())


def output_path(info: SessionInfo, args: argparse.Namespace) -> Path:
    """Return ``<outdir>/<start date>-<name>-<id>[-<level>].pdf``, or ``-o``."""
    if args.output:
        return args.output
    date = info.start.strftime("%Y-%m-%d") if info.start else "undated"
    level = "" if args.level == DEFAULT_LEVEL else f"-{args.level}"
    return args.outdir / f"{date}-{slugify(info.name)}-{info.session_id[:8]}{level}.pdf"


def convert_session(path: Path, args: argparse.Namespace, emoji_font: str | None) -> bool:
    """Convert one session file to PDF. Returns True when a PDF was written."""
    records = read_records(path)
    info = session_info(path, records)
    events = session_events(records)
    if not events:
        print(f"warning: {path} has no messages, skipped", file=sys.stderr)
        return False
    out_pdf = output_path(info, args)

    def markdown(media_dir: Path) -> tuple[str, dict, None]:
        return build_markdown(info, events, args, media_dir)

    return build_pdf(markdown, out_pdf, args, emoji_font, prefix="claude2pdf-", header_extra=HEADER_EXTRA,
                     save_markdown=args.save_markdown)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return the exit code."""
    args = options_for_level(build_parser().parse_args(argv))

    files: list[Path] = []
    for inp in args.input:
        if inp.is_file():
            files.append(inp)
        elif inp.is_dir():
            directory = sessions_dir(inp)
            if not directory.is_dir():
                sys.exit(f"error: no Claude Code sessions for {inp} (looked in {directory})")
            infos = list_sessions(directory)
            if not args.session:
                print_sessions(directory, infos)
                continue
            files += [i.path for i in select_sessions(infos, args.session)]
        else:
            sys.exit(f"error: input not found: {inp}")
    files = list({f.resolve(): f for f in files}.values())
    if not files:
        return 0
    if args.output and len(files) != 1:
        sys.exit("error: -o/--output requires a single session; use --outdir instead")
    if args.jobs < 1:
        sys.exit("error: --jobs must be at least 1")

    for tool in ("pandoc", "latexmk", "lualatex"):
        which_or_die(tool)
    emoji_font = None if args.no_emoji_font else find_emoji_font(args.emoji_font)
    if emoji_font is None and not args.no_emoji_font:
        print("warning: no colour emoji font found; emoji will be missing or monochrome "
              "(install Noto Color Emoji or pass --emoji-font)", file=sys.stderr)
    elif args.verbose:
        print(f"emoji font: {emoji_font}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(files))) as executor:
        results = list(executor.map(lambda f: convert_session(f, args, emoji_font), files))
    failed = results.count(False)
    if len(files) > 1 or failed:
        print(f"{len(files) - failed} PDF(s) written, {failed} failed", file=sys.stderr)
    return 0 if failed == 0 else 1
