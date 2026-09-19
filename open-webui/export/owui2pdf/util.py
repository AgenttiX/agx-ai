"""Small helpers shared by the other modules."""

from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
import sys

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


def fmt_timestamp(ts) -> str:
    """Open WebUI stores seconds, milliseconds or nanoseconds; normalise."""
    if not ts:
        return ""
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return str(ts)
    while ts > 1e11:  # ms or ns -> s
        ts /= 1000.0
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.U).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:60].rstrip("-") or "chat"


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def which_or_die(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: required program '{name}' not found in PATH")
    return path
