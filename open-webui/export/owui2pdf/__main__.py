"""Allow ``python -m owui2pdf``."""

import sys

from .cli import main

sys.exit(main())
