#!/usr/bin/env python3
"""Print the name of this computer that may be written in this repository.

Prints "big-machine" for the big machine, "censored" if the hostname is gitignored in ``benchmark/results/``
and must therefore not be written in this repository, and the hostname otherwise.
"""

import hashlib
from pathlib import Path
import socket
import subprocess

REPO_DIR = Path(__file__).resolve().parent
# The hostname of the big machine is not written in this repository, only its hash.
BIG_MACHINE_SHA256 = "6840565ab99a668b85233d4271338291da04c90517c43ca37382f9fff09d5c91"


def repo_hostname() -> str | None:
    """The name of this computer for this repository, or None if the hostname is censored."""
    host = socket.gethostname()
    if hashlib.sha256(host.encode()).hexdigest() == BIG_MACHINE_SHA256:
        return "big-machine"
    try:
        # Exit code 0 means ignored and 1 not ignored. Errors are treated as ignored to be on the safe side.
        ignored = subprocess.run(
            ["git", "-C", str(REPO_DIR), "check-ignore", "-q", f"benchmark/results/{host}.jsonl"], check=False
        ).returncode != 1
    except FileNotFoundError:
        ignored = True
    return None if ignored else host


def main() -> None:
    """Print the name of this computer."""
    print(repo_hostname() or "censored")


if __name__ == "__main__":
    main()
