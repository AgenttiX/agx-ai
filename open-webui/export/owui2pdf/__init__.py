"""Convert Open WebUI JSON chat exports to PDF.

Modules:
  chat      - reading the export and walking the conversation
  markdown  - normalising message content and assembling the Markdown document
  fonts     - font discovery and the LaTeX header
  build     - running pandoc and latexmk
  cli       - command line interface
"""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
