#!/usr/bin/env python3
"""Convert a chat exported from Open WebUI (JSON) into a PDF.

Thin launcher for the ``owui2pdf`` package next to this file; see README.md
and ``./owui2pdf.py --help``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from owui2pdf.cli import main

if __name__ == "__main__":
    sys.exit(main())
