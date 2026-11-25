"""Tests for the Claude Code setup script."""

import hashlib
from pathlib import Path

import pytest

from agx_ai.claude import setup as claude_setup
from tests.conftest import requires_symlinks


def test_local_instructions() -> None:
    assert claude_setup.local_instructions("agx-z2e-kubuntu") == "agx-z2e.md"
    assert claude_setup.local_instructions("agx-l14-kubuntu") == "agx-l14.md"
    assert claude_setup.local_instructions("unknown") is None


def test_local_instructions_hashed(monkeypatch: pytest.MonkeyPatch) -> None:
    hashed = hashlib.sha256(b"secret-host").hexdigest()
    monkeypatch.setattr(claude_setup, "LOCAL_INSTRUCTIONS", {hashed: "secret.md"})
    assert claude_setup.local_instructions("secret-host") == "secret.md"
    assert claude_setup.local_instructions("other-host") is None


def test_local_instruction_files_exist() -> None:
    for file_name in claude_setup.LOCAL_INSTRUCTIONS.values():
        assert (claude_setup.REPO_DIR / "agents" / "local" / file_name).is_file()


def make_repo(repo: Path) -> None:
    (repo / "claude").mkdir(parents=True)
    (repo / "claude" / "settings.json").write_text("{}")
    (repo / "agents" / "local").mkdir(parents=True)
    (repo / "agents" / "AGENTS.md").write_text("@~/.claude/AGENTS.local.md\n")
    (repo / "agents" / "local" / "agx-z2e.md").write_text("local\n")
    for name in claude_setup.LINKED_DIRS:
        (repo / "agents" / name / "repo-item").mkdir(parents=True)
        (repo / "agents" / name / "repo-item" / "file.md").write_text("repo\n")


@requires_symlinks
def test_configure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    claude = tmp_path / "claude"
    make_repo(repo)
    claude.mkdir()
    (claude / "settings.json").write_text("old settings")
    (claude / "settings-old.json").write_text("older settings")
    (claude / "CLAUDE.md").write_text("old instructions")
    (claude / "skills" / "existing").mkdir(parents=True)
    (claude / "skills" / "existing" / "SKILL.md").write_text("existing\n")
    (claude / "skills" / "repo-item").mkdir()
    (claude / "skills" / "repo-item" / "file.md").write_text("conflicting\n")

    claude_setup.configure(claude, repo, "agx-z2e-kubuntu")
    assert (claude / "settings.json").resolve() == repo / "claude" / "settings.json"
    assert (claude / "settings-old.json").read_text() == "older settings"
    assert (claude / "settings-old-2.json").read_text() == "old settings"
    assert (claude / "CLAUDE.md").resolve() == repo / "agents" / "AGENTS.md"
    assert (claude / "CLAUDE-old.md").read_text() == "old instructions"
    assert (claude / "AGENTS.local.md").read_text() == "local\n"
    for name in claude_setup.LINKED_DIRS:
        assert (claude / name).resolve() == repo / "agents" / name
    # The existing skills are copied to the repository, without overwriting the files there.
    assert (repo / "agents" / "skills" / "existing" / "SKILL.md").read_text() == "existing\n"
    assert (repo / "agents" / "skills" / "repo-item" / "file.md").read_text() == "repo\n"
    assert (claude / "skills-old" / "existing" / "SKILL.md").is_file()

    # Running again keeps the links, and a link to elsewhere is replaced.
    (claude / "CLAUDE.md").unlink()
    (claude / "CLAUDE.md").symlink_to(repo / "claude" / "settings.json")
    claude_setup.configure(claude, repo, "agx-z2e-kubuntu")
    assert (claude / "CLAUDE.md").resolve() == repo / "agents" / "AGENTS.md"
    assert not (claude / "settings-old-3.json").exists()


@requires_symlinks
def test_configure_without_local_instructions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    claude = tmp_path / "claude"
    make_repo(repo)
    claude.mkdir()
    claude_setup.configure(claude, repo, "unknown")
    assert not (claude / "AGENTS.local.md").exists()
    assert (claude / "CLAUDE.md").is_symlink()
