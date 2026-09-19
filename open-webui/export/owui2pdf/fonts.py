"""Font discovery and generation of the LaTeX header (resources/header.tex)."""

from __future__ import annotations

import glob
import os
import shutil
import struct
import sys

from . import RESOURCE_DIR, PACKAGE_DIR
from .util import run

EMOJI_FONT_NAMES = ["Noto Color Emoji", "Twemoji Mozilla", "Twemoji", "OpenMoji Color",
                    "Apple Color Emoji", "Segoe UI Emoji", "JoyPixels"]
EMOJI_FONT_GLOBS = [
    str(PACKAGE_DIR.parent / "fonts" / "*.ttf"),
    os.path.expanduser("~/.local/share/fonts/**/*.ttf"),
    os.path.expanduser("~/.fonts/**/*.ttf"),
    "/usr/share/fonts/**/*Emoji*.ttf",
    "/usr/local/share/fonts/**/*Emoji*.ttf",
    "/var/lib/flatpak/runtime/*/*/*/active/files/share/fonts/**/NotoColorEmoji.ttf",
    os.path.expanduser("~/.local/share/flatpak/runtime/*/*/*/active/files/share/fonts/**/NotoColorEmoji.ttf"),
    "/var/lib/flatpak/app/*/*/*/active/files/**/TwemojiMozilla.ttf",
    "/var/lib/flatpak/app/*/*/*/active/files/**/Twemoji.Mozilla.ttf",
]
# Symbol/CJK fonts used when a glyph is missing from the body font.
TEXT_FALLBACK_FONTS = ["Noto Sans Symbols2", "Noto Sans Symbols", "DejaVu Sans",
                       "Noto Sans CJK SC", "Noto Sans Math"]
DEFAULT_FONTS = {
    "main": (["Noto Serif", "TeX Gyre Pagella", "DejaVu Serif"], "Latin Modern Roman"),
    "sans": (["Noto Sans", "TeX Gyre Heros", "DejaVu Sans"], "Latin Modern Sans"),
    "mono": (["Noto Sans Mono", "DejaVu Sans Mono", "Fira Mono"], "Latin Modern Mono"),
    "math": (["TeX Gyre Pagella Math", "Latin Modern Math"], "Latin Modern Math"),
}


def fc_families() -> set[str]:
    """Font family names known to fontconfig (empty if fc-list is missing)."""
    if not shutil.which("fc-list"):
        return set()
    out = run(["fc-list", ":", "family"]).stdout
    fams = set()
    for line in out.splitlines():
        for fam in line.split(","):
            fams.add(fam.strip())
    return fams


def font_color_format(path: str) -> str | None:
    """Return 'CBDT', 'sbix', 'COLR0', 'COLR1' or None for a TrueType/OpenType file."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12:
                return None
            if head[:4] == b"ttcf":
                f.seek(12)
                (offset,) = struct.unpack(">I", f.read(4))
                f.seek(offset)
                head = f.read(12)
            num_tables = struct.unpack(">H", head[4:6])[0]
            tables = {}
            for _ in range(num_tables):
                t, _cs, off, length = struct.unpack(">4sIII", f.read(16))
                tables[t.decode("latin1")] = (off, length)
            if "CBDT" in tables:
                return "CBDT"
            if "sbix" in tables:
                return "sbix"
            if "COLR" in tables:
                f.seek(tables["COLR"][0])
                version = struct.unpack(">H", f.read(2))[0]
                return "COLR1" if version >= 1 else "COLR0"
    except OSError:
        pass
    return None


def find_emoji_font(explicit: str | None) -> str | None:
    """Locate a colour emoji font that LuaLaTeX can render (CBDT/COLRv0/sbix)."""
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"error: emoji font not found: {explicit}")
        return os.path.abspath(explicit)
    candidates: list[str] = []
    if shutil.which("fc-list"):
        out = run(["fc-list", ":", "file", "family"]).stdout
        for line in out.splitlines():
            if ":" not in line:
                continue
            path, fams = line.split(":", 1)
            if any(n.lower() in fams.lower() for n in EMOJI_FONT_NAMES):
                candidates.append(path.strip())
    for pat in EMOJI_FONT_GLOBS:
        for p in sorted(glob.glob(pat, recursive=True)):
            if "emoji" in os.path.basename(p).lower():
                candidates.append(p)
    # Prefer bitmap/COLRv0 fonts (supported by luaotfload); COLRv1 is not.
    rank = {"CBDT": 0, "COLR0": 1, "sbix": 2}
    best = None
    for p in candidates:
        fmt = font_color_format(p)
        if fmt in rank:
            score = (rank[fmt], 0 if "noto" in p.lower() else 1)
            if best is None or score < best[0]:
                best = (score, p)
    return best[1] if best else None


def build_header(opts, emoji_font: str | None) -> str:
    """Fill the placeholders of resources/header.tex.

    ``opts`` may carry ``main_font``, ``sans_font``, ``mono_font`` and
    ``math_font`` overrides (None to auto-detect).
    """
    fams = fc_families()

    def pick(kind: str, override: str | None) -> str:
        if override:
            return override
        preferred, fallback = DEFAULT_FONTS[kind]
        return next((n for n in preferred if n in fams), fallback)

    main = pick("main", getattr(opts, "main_font", None))
    sans = pick("sans", getattr(opts, "sans_font", None))
    mono = pick("mono", getattr(opts, "mono_font", None))
    math = pick("math", getattr(opts, "math_font", None))

    symbol_fonts = [f'"{n}:mode=harf;"' for n in TEXT_FALLBACK_FONTS if n in fams]
    fallbacks = []
    if emoji_font:
        fallbacks.append(f'"[{emoji_font}]:mode=harf;"')
    fallbacks += symbol_fonts
    if not fallbacks:
        fallbacks.append('"Latin Modern Roman:mode=harf;"')

    if emoji_font:
        font_dir = os.path.dirname(emoji_font) + "/"
        font_file = os.path.basename(emoji_font)
        text_fb = [f'"{main}:mode=harf;"'] + symbol_fonts
        emoji_family = (
            "\\directlua{luaotfload.add_fallback(\"owuitextfallback\", {" + ", ".join(text_fb) + "})}\n"
            f"\\newfontfamily\\owuiemojifont{{{font_file}}}[Path={font_dir},Renderer=HarfBuzz,"
            "RawFeature={fallback=owuitextfallback}]\n"
            "\\newcommand{\\owuiemoji}[1]{\\texorpdfstring{{\\owuiemojifont #1}}{#1}}"
        )
    else:
        emoji_family = "\\newcommand{\\owuiemoji}[1]{#1}"

    tex = (RESOURCE_DIR / "header.tex").read_text(encoding="utf-8")
    replacements = {
        "@@FALLBACK_FONTS@@": ",\n".join("    " + f for f in fallbacks),
        "@@EMOJI_FAMILY@@": emoji_family,
        "@@MAINFONT@@": main,
        "@@SANSFONT@@": sans,
        "@@MONOFONT@@": mono,
        "@@MATHFONT@@": math,
    }
    for key, value in replacements.items():
        tex = tex.replace(key, value)
    return tex
