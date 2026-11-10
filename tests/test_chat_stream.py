"""Tests of agx_ai.benchmark.chat_stream with fake servers."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from agx_ai.benchmark import chat_stream
from tests.fake_server import CHAT_MODEL, REASONING, REPLY, FakeServer


def test_nonce() -> None:
    a, b = chat_stream.nonce(), chat_stream.nonce()
    assert a != b
    assert a.startswith("[")
    assert a.endswith("] ")


def test_gradient_png() -> None:
    url = chat_stream.gradient_png(8, 4)
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix):]).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("kind", ["llama", "vllm"])
def test_stream(kind: str) -> None:
    with FakeServer(kind, api_key="key", reasoning=True) as fake:  # type: ignore[arg-type]
        client = chat_stream.ChatClient(fake.url, "key", thinking=False)
        assert client.default_model() == CHAT_MODEL
        client.model = CHAT_MODEL
        out = client.stream("one two", 16)
        # The reasoning is timed but not part of the text.
        assert out["text"] == "".join(REPLY)
        assert out["usage"]["completion_tokens"] == len(REASONING) + len(REPLY)
        assert out["ttft_s"] >= 0
        payload = fake.requests[-1][2]
        assert payload["model"] == CHAT_MODEL
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_stream_without_model_list() -> None:
    with FakeServer("vllm", models=[]) as fake:
        client = chat_stream.ChatClient(fake.url)
        assert client.default_model() is None
        client.stream("hello", 8)
        payload = fake.requests[-1][2]
        # The defaults of the NPU server: no model and no chat template arguments
        assert "model" not in payload
        assert "chat_template_kwargs" not in payload


def test_no_tokens() -> None:
    with FakeServer("vllm") as fake, pytest.raises(RuntimeError, match="No tokens"):
        chat_stream.ChatClient(fake.url).stream("hello", 0)


def test_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "results.json"
    with FakeServer("vllm") as fake:
        chat_stream.main(["--url", fake.url, "--repeat", "1", "--json", str(output)])
    results = json.loads(output.read_text(encoding="utf-8"))
    assert [len(results[k]) for k in ("short", "long", "image")] == [1, 1, 1]
    assert len(results["answers"]) == len(chat_stream.QUESTIONS)
    out = capsys.readouterr().out
    assert "short:" in out
    assert "Correct answers: " in out
