"""Small helpers shared by the other modules."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from typing import Any

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    """Escape text for use in LaTeX macro arguments."""
    return "".join(LATEX_SPECIALS.get(c, c) for c in s)


def md_escape_inline(s: str) -> str:
    """Escape text so that Markdown renders it literally."""
    return re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", s)


DATETIME_FORMAT = "%Y-%m-%d %H:%M"


def datetime_format(dates: bool = True, times: bool = True) -> str:
    """Return the strftime format of ``YYYY-MM-DD HH:MM`` without the left out parts ("" for neither)."""
    return " ".join(fmt for fmt, show in (("%Y-%m-%d", dates), ("%H:%M", times)) if show)


def date_time_options(p: argparse.ArgumentParser, what: str) -> None:
    """Add the options ``--no-dates`` and ``--no-times`` (``show_dates``, ``show_times``) to a parser."""
    p.add_argument("--no-dates", dest="show_dates", action="store_false",
                   help=f"omit the dates (YYYY-MM-DD) of the {what} and its messages; "
                        "with --no-times, omit the timestamps altogether")
    p.add_argument("--no-times", dest="show_times", action="store_false",
                   help=f"omit the times of day (HH:MM) of the {what} and its messages")


def fmt_timestamp(ts: float | str | None, fmt: str = DATETIME_FORMAT) -> str:
    """Format a timestamp with the strftime format ``fmt``.

    Open WebUI stores seconds, milliseconds or nanoseconds; normalise.
    """
    if not ts or not fmt:
        return ""
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return str(ts)
    while ts > 1e11:  # ms or ns -> s
        ts /= 1000.0
    return dt.datetime.fromtimestamp(ts).strftime(fmt)


def slugify(s: str) -> str:
    """Convert a string to a short filename-safe slug."""
    s = re.sub(r"[^\w\s-]", "", s, flags=re.U).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:60].rstrip("-") or "chat"


def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing its text output without raising on failure."""
    return subprocess.run(cmd, text=True, capture_output=True, check=False, **kw)


def which_or_die(name: str) -> str:
    """Return the path of a required program or exit with an error."""
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: required program '{name}' not found in PATH")
    return path
