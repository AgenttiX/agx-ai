"""Tests with the LLM servers running on this computer (see conftest.py); skipped if there are none.

The requests are short, so that the tests stay fast and do not disturb the servers much.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agx_ai.benchmark import bench_http, chat_stream, quality_http
from agx_ai.benchmark.vllm import prefix_return
from tests.conftest import Backend


def test_tokenize_round_trip(backend: Backend) -> None:
    text = "A first-order phase transition proceeds through bubble nucleation."
    tokens = backend.server.tokenize(backend.model, text)
    assert 5 < len(tokens) < 40
    assert backend.server.detokenize(backend.model, tokens).strip() == text
    truncated = backend.server.truncate(backend.model, text, 3)
    assert len(backend.server.tokenize(backend.model, truncated)) <= 3


def test_props(backend: Backend) -> None:
    props = backend.server.props(backend.model)
    assert props["build_info"]
    assert int(props["total_slots"]) >= 1


def test_generation(backend: Backend) -> None:
    result = bench_http.test_generation(backend.server, backend.model, n_predict=8, repeat=1)
    assert result["generated_tokens"] == 8
    assert result["generation_tps"] > 0


def test_prompt_processing(backend: Backend) -> None:
    result = bench_http.test_prompt_processing(backend.server, backend.model, prompt_tokens=128, repeat=1)
    assert abs(result["prompt_tokens"] - 128) <= 2
    assert result["prompt_tps"] > 0


def test_quality_chat(backend: Backend) -> None:
    out = quality_http.chat(backend.server, backend.model, "What is 2 + 3? Reply with only the number.", 8,
                            thinking=False, cache=False, logprobs=3)
    assert "5" in out["content"]
    assert out["tokens"]
    assert all(len(t["top"]) <= 3 for t in out["tokens"])


def test_chat_stream(backend: Backend) -> None:
    client = chat_stream.ChatClient(backend.server.url, backend.api_key, backend.model, thinking=False, timeout=120)
    out = client.stream(chat_stream.nonce() + "What is the capital of Finland? Answer with the city name only.", 16)
    assert "Helsinki" in out["text"]
    assert out["tokens_per_s"] > 0


def test_bench_http_main(backend: Backend, results_dir: Path, monkeypatch: pytest.MonkeyPatch,
                         capsys: pytest.CaptureFixture[str]) -> None:
    if backend.api_key:
        monkeypatch.setenv("LLAMA_API_KEY", backend.api_key)
    monkeypatch.setattr("sys.argv", [
        "bench_http", "--url", backend.server.url, "--model", backend.model, "--no-embedding",
        "--tests", "generation", "prompt", "--n-predict", "8", "--prompt-tokens", "128", "--repeat", "1",
        "--label", "unit test",
    ])
    assert bench_http.main() == 0
    result = json.loads((results_dir / "test-host.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["chat_model"] == backend.model
    assert result["generation"]["generated_tokens"] == 8
    assert "| Prompt processing, " in capsys.readouterr().out


def test_vllm_prefix_cache(vllm_backend: Backend) -> None:
    server = vllm_backend.server
    assert isinstance(server, bench_http.VllmServer)
    content = bench_http.build_code_context(server, vllm_backend.model, 1000) + "\n\nWhat is the first module?"
    salt = os.urandom(8).hex()
    _, n, cold = prefix_return.ask(server, vllm_backend.model, content, salt, 8)
    _, _, warm = prefix_return.ask(server, vllm_backend.model, content, salt, 8)
    assert n > 1000
    assert warm == cold


def test_llama_cpp_server_args(llama_backend: Backend) -> None:
    models = llama_backend.server.models()
    args = bench_http.server_args(models, llama_backend.model)
    if args is None:
        pytest.skip("not a llama-server router")
    assert any(a.startswith("--") for a in args)
