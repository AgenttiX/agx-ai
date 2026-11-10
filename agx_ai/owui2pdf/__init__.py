"""Convert Open WebUI JSON chat exports to PDF.

Modules:
  chat      - reading the export and walking the conversation
  markdown  - normalising message content and assembling the Markdown document
  fonts     - font discovery and the LaTeX header
  build     - running pandoc and latexmk
  sources   - referenced notes, files and web pages: citations and the bibliography
  notes     - the full text of referenced notes from files or the Open WebUI API
  util      - escaping, timestamps and subprocess helpers
  cli       - command line interface

Run it with ``uv run -m agx_ai.owui2pdf`` (see README.md).
"""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
