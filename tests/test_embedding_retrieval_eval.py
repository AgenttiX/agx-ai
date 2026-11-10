"""Tests of agx_ai.benchmark.embedding_retrieval_eval with a fake server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np
import pytest

from agx_ai.benchmark import embedding_retrieval_eval as ere
from tests.fake_server import CHAT_MODEL, FakeServer

CHUNKS = [
    "The bubble wall speed sets the peak frequency of the sound waves.",
    "Llamas live in the Andes and eat grass.",
    "The GPU has sixteen gigabytes of HBM2 memory.",
    "Sourdough bread needs a starter and a long fermentation.",
]
# Questions that share words with their source chunk, which the fake bag-of-words embedding finds
QUESTIONS = [
    {"chunk": 0, "question": "What sets the peak frequency of the sound waves?"},
    {"chunk": 2, "question": "How much HBM2 memory has the GPU?"},
    {"chunk": 3, "question": "What does sourdough bread need?"},
]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert ere.load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_embed_is_normalized(fake_vllm: FakeServer) -> None:
    vecs = ere.embed(fake_vllm.url, "m", CHUNKS, None, batch=3, concurrency=2)
    assert vecs.shape == (4, 64)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)


def test_eval(fake_vllm: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              capsys: pytest.CaptureFixture[str]) -> None:
    chunks = _write_jsonl(tmp_path / "chunks.jsonl", [{"name": str(i), "text": t} for i, t in enumerate(CHUNKS)])
    questions = _write_jsonl(tmp_path / "questions.jsonl", QUESTIONS)
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr("sys.argv", [
        "ere", "eval", "--chunks", str(chunks), "--questions", str(questions), "--url", fake_vllm.url,
        "--hostname", "test-host", "--output", str(output), "--label", "fake",
    ])
    ere.main()
    (result,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["host"] == "test-host"
    assert result["model"] == CHAT_MODEL  # the first model of the server
    assert (result["chunks"], result["questions"], result["dimension"]) == (4, 3, 64)
    assert result["recall_at_1"] == 1.0
    assert result["mrr"] == 1.0
    assert "| fake | 1.0 |" in capsys.readouterr().out


def test_questions(fake_vllm: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _write_jsonl(tmp_path / "chunks.jsonl", [{"name": str(i), "text": t} for i, t in enumerate(CHUNKS)])
    out = tmp_path / "questions.jsonl"
    monkeypatch.setattr("sys.argv", ["ere", "questions", "--chunks", str(chunks), "--out", str(out), "--n", "2",
                                     "--url", fake_vllm.url])
    ere.main()
    # The fake server's reply is not a question, so the chunks are skipped.
    assert out.read_text(encoding="utf-8") == ""
