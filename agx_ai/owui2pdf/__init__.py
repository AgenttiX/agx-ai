"""Convert Open WebUI JSON chat exports to PDF.

Modules:
  chat      - reading the export and walking the conversation
  markdown  - normalising message content and assembling the Markdown document
  build     - turning one chat into a PDF
  sources   - referenced notes, files and web pages: citations and the bibliography
  notes     - the full text of referenced notes from files or the Open WebUI API
  cli       - command line interface

The shared rendering code (pandoc, LaTeX, fonts) is in :mod:`agx_ai.convert`.
Run it with ``uv run -m agx_ai.owui2pdf`` (see README.md).
"""

__version__ = "0.1.0"
