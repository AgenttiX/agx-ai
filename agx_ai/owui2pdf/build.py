"""Turning one chat into a PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from agx_ai.convert.build import build_pdf

from .markdown import build_markdown


def convert_chat(chat_item: dict, out_pdf: Path, opts: argparse.Namespace, emoji_font: str | None) -> bool:
    """Build ``out_pdf`` from one chat. Returns True when a PDF was written.

    See :func:`build_pdf` for the build directory.
    """

    def markdown(media_dir: Path) -> tuple[str, dict, str | None]:
        return build_markdown(chat_item, opts, media_dir)

    return build_pdf(markdown, out_pdf, opts, emoji_font)
