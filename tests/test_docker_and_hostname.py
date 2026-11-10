"""Tests of agx_ai.benchmark.docker and agx_ai.hostname."""

from __future__ import annotations

import subprocess
import sys

import pytest

from agx_ai import hostname
from agx_ai.benchmark import RESULTS_DIR, docker


def test_matching_lines() -> None:
    log = "start\nERROR one\nok\nCUDA out of memory\nException two\n"
    assert docker.matching_lines(log, ("error", "exception")) == ["ERROR one", "Exception two"]
    assert docker.matching_lines(log, ("error", "out of memory", "exception"), 2) == [
        "CUDA out of memory", "Exception two"]
    assert docker.matching_lines(log, ("missing",)) == []


def test_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker, "is_running", lambda _: True)
    states = iter([None, None, True])
    assert docker.wait("x", lambda: next(states), timeout=10, interval=0)
    assert not docker.wait("x", lambda: False, timeout=10, interval=0)
    assert not docker.wait("x", lambda: None, timeout=0.01, interval=0)
    # The container stopped
    monkeypatch.setattr(docker, "is_running", lambda _: False)
    assert not docker.wait("x", lambda: None, timeout=10, interval=0)


def test_results_dir() -> None:
    assert RESULTS_DIR == hostname.REPO_DIR / "benchmark" / "results"
    assert RESULTS_DIR.is_dir()


def test_repo_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hostname that is not gitignored in benchmark/results/
    monkeypatch.setattr(hostname.socket, "gethostname", lambda: "agx-unit-test")
    assert hostname.repo_hostname() == "agx-unit-test"
    # A hostname that is gitignored (see .gitignore)
    monkeypatch.setattr(hostname.socket, "gethostname", lambda: "dx2-unit-test")
    assert hostname.repo_hostname() is None


def test_repo_hostname_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(hostname.subprocess, "run", no_git)
    monkeypatch.setattr(hostname.socket, "gethostname", lambda: "agx-unit-test")
    assert hostname.repo_hostname() is None


def test_hostname_module() -> None:
    out = subprocess.run([sys.executable, "-m", "agx_ai.hostname"], capture_output=True, text=True, check=True,
                         cwd=hostname.REPO_DIR).stdout.strip()
    assert out == (hostname.repo_hostname() or "censored")
