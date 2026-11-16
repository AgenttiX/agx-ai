"""Shared parts of the converters of chats to PDF (:mod:`agx_ai.owui2pdf`, :mod:`agx_ai.claude2pdf`).

Modules:
  build  - running pandoc and latexmk on a Markdown document
  fonts  - font discovery and the LaTeX header
  util   - escaping, timestamps and subprocess helpers

The LaTeX preamble and the pandoc Lua filter are in ``resources/``.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
