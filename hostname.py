#!/usr/bin/env python3
"""Print the name of this computer that may be written in this repository.

Prints "big-machine" for the big machine, "censored" if the hostname is gitignored in ``benchmark/results/``
and must therefore not be written in this repository, and the hostname otherwise.
The implementation is in ``agx_ai/hostname.py``.
"""

from agx_ai.hostname import main

if __name__ == "__main__":
    main()
