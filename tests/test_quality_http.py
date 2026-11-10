"""Tests of agx_ai.benchmark.quality_http."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random

import pytest

from agx_ai.benchmark import bench_http, quality_http
from tests.fake_server import CHAT_MODEL, FakeServer


def test_common_prefix() -> None:
    assert quality_http.common_prefix("abcdef", "abcxyz") == 3
    assert quality_http.common_prefix("abc", "abcdef") == 3
    assert quality_http.common_prefix("", "abc") == 0
    assert quality_http.common_prefix("same", "same") == 4


def test_token_kld() -> None:
    top = {"a": math.log(0.7), "b": math.log(0.2), "c": math.log(0.1)}
    assert quality_http.token_kld(top, top) == pytest.approx(0.0, abs=1e-12)
    shifted = {"a": math.log(0.2), "b": math.log(0.7), "c": math.log(0.1)}
    expected = 0.7 * math.log(0.7 / 0.2) + 0.2 * math.log(0.2 / 0.7)
    assert quality_http.token_kld(top, shifted) == pytest.approx(expected)
    # A token missing from this run's list gets the lowest log-probability of the list.
    missing = {"b": math.log(0.2), "c": math.log(0.1)}
    assert quality_http.token_kld(top, missing) == pytest.approx(0.7 * math.log(0.7 / 0.1))


def test_make_needles_is_deterministic() -> None:
    a = quality_http.make_needles(random.Random(1), 5, 3)
    b = quality_http.make_needles(random.Random(1), 5, 3)
    assert a == b
    lines, questions, pairs = a
    assert len(lines) == 5
    assert len(pairs) == 3
    assert [q["kind"] for q in questions] == ["lookup"] * 5 + ["two-step"] * 3
    for line, question in zip(lines, questions, strict=False):
        assert question["expected"] in line
    for (first, second), question in zip(pairs, questions[5:], strict=True):
        assert question["expected"] in second
        db = first.rsplit(" ", 1)[1].rstrip(".")
        assert db in second


def test_build_haystack(fake_vllm: FakeServer) -> None:
    server = bench_http.connect(fake_vllm.url, None, 5)
    haystack, questions = quality_http.build_haystack(server, CHAT_MODEL, 500, random.Random(2), 4, 2)
    assert len(questions) == 6
    assert haystack.count("# Deployment note:") == 6
    assert haystack.count("# Infrastructure note:") == 2


def _answer(question: str, answer: str, tokens: list[tuple[str, dict[str, float]]]) -> dict[str, object]:
    return {"kind": "lookup", "question": question, "expected": "1", "answer": answer, "correct": answer == "1",
            "tokens": [{"t": t, "top": top} for t, top in tokens]}


def test_compare() -> None:
    same = {"1": math.log(0.9), "2": math.log(0.1)}
    flipped = {"1": math.log(0.4), "2": math.log(0.6)}
    ref = {
        "label": "ref", "date": "2026-01-01",
        "retrieval": [{"depth": 100, "answers": [_answer("q1", "1", [("1", same), ("", same)]),
                                                 _answer("q2", "1", [("1", same)])]}],
        "agreement": [{"id": "a", "reasoning": "", "content": "hello world"}],
    }
    result = {
        "retrieval": [{"depth": 100, "answers": [_answer("q1", "1", [("1", same), ("", same)]),
                                                 _answer("q2", "2", [("2", flipped)])]}],
        "agreement": [{"id": "a", "reasoning": "", "content": "hello there"}],
    }
    cmp = quality_http.compare(result, ref)
    assert cmp["retrieval_same_answer"] == "1/2"
    assert cmp["agreement_identical"] == "0/1"
    assert cmp["agreement"][0]["common_prefix_chars"] == len("\nhello ")
    lp = cmp["retrieval_logprobs"]
    assert lp["positions"] == 2
    assert lp["top_token_flips"] == 1
    assert lp["mean_kld"] > 0


def test_compact() -> None:
    result = {"label": "x", "retrieval": [{"depth": 1, "answers": [_answer("q", "1", [("1", {"1": 0.0})])]}],
              "agreement": [{"id": "a", "reasoning": "r", "content": "c"}]}
    out = quality_http.compact(result)
    assert out["retrieval"][0]["answers"] == [{"kind": "lookup", "expected": "1", "answer": "1", "correct": True}]
    assert out["agreement"] == [{"id": "a", "length": 3}]
    # The original is not modified.
    assert "tokens" in result["retrieval"][0]["answers"][0]


def test_main_with_reference(fake_llama: FakeServer, results_dir: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:
    base = ["quality_http", "--url", fake_llama.url, "--model", CHAT_MODEL, "--tests", "retrieval",
            "--depth", "300", "--lookups", "2", "--pairs", "1", "--logprobs", "0"]
    monkeypatch.setattr("sys.argv", [*base, "--label", "first"])
    assert quality_http.main() == 0
    monkeypatch.setattr("sys.argv", [*base, "--label", "second", "--reference", "first"])
    assert quality_http.main() == 0
    compact = [json.loads(line) for line in (results_dir / "test-host-quality.jsonl").read_text().splitlines()]
    full = [json.loads(line) for line in (results_dir / "full" / "test-host-quality.jsonl").read_text().splitlines()]
    assert [r["label"] for r in compact] == ["first", "second"]
    assert compact[1]["comparison"]["retrieval_same_answer"] == "3/3"
    assert "question" in full[0]["retrieval"][0]["answers"][0]
    assert "question" not in compact[0]["retrieval"][0]["answers"][0]
    assert "Compared with `first`" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", [*base, "--label", "third", "--reference", "missing"])
    with pytest.raises(SystemExit, match="No run labelled"):
        quality_http.main()
