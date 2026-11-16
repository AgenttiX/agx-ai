"""Convert Claude Code sessions (``~/.claude/projects/<project>/<session id>.jsonl``) to PDF.

Modules:
  session   - reading the transcript, the active branch, and listing the sessions of a repository
  markdown  - assembling the Markdown document at the chosen level of verbosity
  cli       - command line interface

The rendering (pandoc, LaTeX, fonts) is shared with :mod:`agx_ai.owui2pdf` in :mod:`agx_ai.convert`.
Run it with ``uv run claude2pdf`` (see README.md).
"""

__version__ = "0.1.0"
