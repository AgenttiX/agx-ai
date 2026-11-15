#!/usr/bin/env python3
"""Configure Claude Code on Linux and Windows by linking its configuration in ~/.claude to this repository.

Links:
- settings.json -> claude/settings.json
- CLAUDE.md -> agents/AGENTS.md, which imports ~/.claude/AGENTS.local.md
  (with the full path, since Claude Code resolves imports relative to the real path of a symlinked file)
- AGENTS.local.md -> agents/local/<computer>.md, if this computer has local instructions
- skills -> agents/skills and agents -> agents/agents (directory junctions on Windows)

Existing files are moved to e.g. settings-old.json.
The existing skills and agents are copied to the repository first, without overwriting the files there.
Finally, the language servers (LSP) are installed.
Uses only the Python standard library, so it can be run with plain `python3`.
"""

import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CLAUDE_DIR = Path.home() / ".claude"

# The keys are hostnames or their SHA256 hashes.
# The hostname of the big machine is not written in this repository, only its hash (see agx_ai/hostname.py).
LOCAL_INSTRUCTIONS = {
    "6840565ab99a668b85233d4271338291da04c90517c43ca37382f9fff09d5c91": "big-machine.md",
    "agx-z2e-kubuntu": "agx-z2e.md",
    "agx-l14-kubuntu": "agx-l14.md",
}
# Linked as directories, which are directory junctions on Windows
LINKED_DIRS = ("skills", "agents")


def local_instructions(hostname: str) -> str | None:
    """The file name of the local instructions in agents/local/ for a computer, or None if there are none."""
    hashed = hashlib.sha256(hostname.encode()).hexdigest()
    for host, file_name in LOCAL_INSTRUCTIONS.items():
        if host in (hostname, hashed):
            return file_name
    return None


def is_link(path: Path) -> bool:
    """Whether a path is a symlink or a directory junction."""
    return path.is_symlink() or path.is_junction()


def backup_path(path: Path) -> Path:
    """A free path for the backup of a file or directory, e.g. settings-old.json or settings-old-2.json."""
    backup = path.with_name(f"{path.stem}-old{path.suffix}")
    i = 2
    while backup.exists() or is_link(backup):
        backup = path.with_name(f"{path.stem}-old-{i}{path.suffix}")
        i += 1
    return backup


def copy_if_missing(src: str, dst: str) -> str:
    """Copy a file without overwriting an existing file. For shutil.copytree."""
    if not os.path.lexists(dst):
        shutil.copy2(src, dst)
    return dst


def create_link(link: Path, target: Path, directory: bool = False) -> None:
    """Link a path in the Claude directory to the repository, moving any existing file out of the way.

    Directories are linked with directory junctions on Windows, since they don't need special privileges.
    """
    if is_link(link):
        if link.resolve() == target.resolve():
            print(f"{link} is already linked to {target}.")
            return
        print(f"{link} is linked to {link.resolve()}. Replacing it.")
        if link.is_junction():
            # A junction is removed like an empty directory, without removing the target's contents.
            link.rmdir()
        else:
            link.unlink()
    elif link.exists():
        if directory and link.is_dir():
            shutil.copytree(link, target, copy_function=copy_if_missing, dirs_exist_ok=True)
        backup = backup_path(link)
        print(f"Moving the existing {link} to {backup}.")
        link.rename(backup)

    if directory and sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=directory)
    print(f"Linked {link} to {target}.")


def configure(claude_dir: Path, repo_dir: Path, hostname: str) -> None:
    """Create the links in the Claude directory."""
    agents_dir = repo_dir / "agents"
    create_link(claude_dir / "settings.json", repo_dir / "claude" / "settings.json")
    create_link(claude_dir / "CLAUDE.md", agents_dir / "AGENTS.md")
    local = local_instructions(hostname)
    if local is None:
        print("This computer has no local instructions.")
    else:
        create_link(claude_dir / "AGENTS.local.md", agents_dir / "local" / local)
    for name in LINKED_DIRS:
        create_link(claude_dir / name, agents_dir / name, directory=True)


def run_optional(command: list[str]) -> None:
    """Run a command if the program is installed, and only warn if it fails."""
    program = shutil.which(command[0])
    if program is None:
        print(f"{command[0]} was not found. Skipping: {' '.join(command)}")
        return
    # Print the previous output before the output of the command.
    sys.stdout.flush()
    try:
        subprocess.run([program, *command[1:]], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: the command failed with exit code {e.returncode}: {' '.join(command)}")


def main() -> int:
    """Configure Claude Code and install the language servers."""
    if not CLAUDE_DIR.is_dir():
        print(f"The Claude directory {CLAUDE_DIR} does not exist. Install Claude first.")
        return 1
    try:
        configure(CLAUDE_DIR, REPO_DIR, socket.gethostname())
    except OSError as e:
        print(f"Error: {e}")
        if sys.platform == "win32":
            print("Creating symlinks on Windows requires Developer Mode or administrator privileges.")
        return 1

    print("Installing language servers (LSP)")
    run_optional(["uv", "tool", "install", "--upgrade", "pyright"])
    run_optional(["npm", "install", "-g", "typescript-language-server", "typescript"])
    print("Claude configured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
