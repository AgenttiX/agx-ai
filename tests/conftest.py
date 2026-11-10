"""Fixtures for the unit tests.

The tests marked ``backend`` use the LLM servers running on this computer. The URLs are taken from the environment
variable ``AGX_AI_TEST_URLS`` (comma-separated, default: the ports 9931-9935 of localhost) and the API key from
``LLAMA_API_KEY`` or ``llama-cpp/llama-cpp.env``. Only servers with a loaded chat model are used, so that the tests
do not load models (which would be slow and could evict the production models of a llama.cpp router). If no such
server is found, the tests are skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import socket
from typing import Any
import urllib.parse

import pytest

from agx_ai.benchmark import bench_http
from agx_ai.hostname import REPO_DIR
from tests.fake_server import FakeServer

DEFAULT_URLS: str = ",".join(f"http://localhost:{port}" for port in range(9931, 9936))
ENV_FILE: Path = REPO_DIR / "llama-cpp" / "llama-cpp.env"


@dataclass
class Backend:
    """A running LLM server with a loaded chat model."""

    name: str
    server: bench_http.Server
    model: str
    api_key: str | None

    @property
    def is_vllm(self) -> bool:
        """Whether the server is vLLM."""
        return isinstance(self.server, bench_http.VllmServer)


def api_key() -> str | None:
    """The API key of the local servers."""
    if key := os.environ.get("LLAMA_API_KEY"):
        return key
    return bench_http.read_env_key(ENV_FILE) if ENV_FILE.exists() else None


def is_loaded_chat_model(model: dict[str, Any]) -> bool:
    """Whether the model is a chat model that is loaded (always true for servers without a router)."""
    status = model.get("status")
    is_embedding = "embed" in model["id"].lower() or "--embeddings" in ((status or {}).get("args") or [])
    return not is_embedding and (status is None or status.get("value") == "loaded")


def listens(url: str) -> bool:
    """Whether something accepts connections at the host and port of the URL."""
    parts = urllib.parse.urlsplit(url)
    try:
        socket.create_connection((parts.hostname or "localhost", parts.port or 80), timeout=1).close()
    except OSError:
        return False
    return True


def discover_backends() -> list[Backend]:
    """The local servers that answer and have a loaded chat model."""
    key = api_key()
    backends = []
    for url in (u.strip() for u in os.environ.get("AGX_AI_TEST_URLS", DEFAULT_URLS).split(",")):
        if not url or not listens(url):
            continue
        try:
            server = bench_http.connect(url, key, timeout=120)
            models = server.models()
        except SystemExit:
            continue
        model = next((m["id"] for m in models if is_loaded_chat_model(m)), None)
        if model is None or isinstance(server, bench_http.OvmsServer):
            continue
        name = "vllm" if isinstance(server, bench_http.VllmServer) else "llama.cpp"
        backends.append(Backend(f"{name}@{url}", server, model, key))
    return backends


_backends: list[Backend] | None = None


def backends() -> list[Backend]:
    """The local backends, discovered once per test session."""
    global _backends  # noqa: PLW0603 (a cache for the parametrization)
    if _backends is None:
        _backends = discover_backends()
    return _backends


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers."""
    config.addinivalue_line("markers", "backend: the test needs a running LLM server (skipped if there is none)")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run the tests with a ``backend``, ``llama_backend`` or ``vllm_backend`` argument once per local server."""
    for name, kind in (("backend", None), ("llama_backend", "llama.cpp"), ("vllm_backend", "vllm")):
        if name in metafunc.fixturenames:
            found = [b for b in backends() if kind is None or b.name.startswith(kind)]
            if found:
                metafunc.parametrize(name, found, ids=[b.name for b in found])
            else:
                metafunc.parametrize(name, [pytest.param(None, marks=pytest.mark.skip(
                    reason=f"no local {kind or 'LLM'} server with a loaded chat model"))])


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the tests that use a backend."""
    for item in items:
        if {"backend", "llama_backend", "vllm_backend"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.backend)


@pytest.fixture
def fake_llama() -> Iterator[FakeServer]:
    """A fake llama-server router."""
    with FakeServer("llama") as server:
        yield server


@pytest.fixture
def fake_vllm() -> Iterator[FakeServer]:
    """A fake vLLM server."""
    with FakeServer("vllm") as server:
        yield server


@pytest.fixture
def fake_ovms() -> Iterator[FakeServer]:
    """A fake OpenVINO Model Server."""
    with FakeServer("ovms") as server:
        yield server


@pytest.fixture
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the default results directory of the benchmarks to a temporary directory."""
    from agx_ai import benchmark  # noqa: PLC0415

    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)
    for module in ("bench_http", "quality_http", "embedding_retrieval_eval"):
        monkeypatch.setattr(f"agx_ai.benchmark.{module}.RESULTS_DIR", tmp_path)
    monkeypatch.setenv("LLAMA_BENCH_HOSTNAME", "test-host")
    return tmp_path
