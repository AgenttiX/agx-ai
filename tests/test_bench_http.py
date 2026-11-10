"""Tests of agx_ai.benchmark.bench_http with fake servers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest

from agx_ai.benchmark import bench_http
from tests.fake_server import CHAT_MODEL, EMBEDDING_MODEL, REPLY, FakeServer


def test_read_env_key(tmp_path: Path) -> None:
    env = tmp_path / "test.env"
    env.write_text("# comment\nOTHER=1\nLLAMA_API_KEY='secret-value'\nEMPTY=\n", encoding="utf-8")
    assert bench_http.read_env_key(env) == "secret-value"
    assert bench_http.read_env_key(env, "OTHER") == "1"
    assert bench_http.read_env_key(env, "EMPTY") == ""
    assert bench_http.read_env_key(env, "MISSING") is None


def test_host_name_given() -> None:
    assert bench_http.host_name(name="work-pc") == "work-pc"
    assert bench_http.host_name(hashed=True, name="work-pc") == hashlib.sha256(b"work-pc").hexdigest()


def test_host_name_censored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_http, "repo_hostname", lambda: None)
    with pytest.raises(SystemExit, match="censored"):
        bench_http.host_name()
    # Hashing does not need the name from hostname.py.
    assert len(bench_http.host_name(hashed=True)) == 64


def test_host_name_from_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_http, "repo_hostname", lambda: "agx-test")
    assert bench_http.host_name() == "agx-test"


def test_pick_models_router() -> None:
    models = [
        {"id": "a-chat", "status": {"value": "unloaded"}},
        {"id": "b-embed", "status": {"value": "unloaded"}},
        {"id": "c-chat", "status": {"value": "loaded"}},
        {"id": "d", "status": {"value": "loaded", "args": ["llama-server", "--embeddings"]}},
    ]
    # The loaded models are preferred, and unloaded embedding models are not picked.
    assert bench_http.pick_models(models, None, None) == ("c-chat", "d")
    assert bench_http.pick_models(models, "a-chat", "b-embed") == ("a-chat", "b-embed")
    assert bench_http.pick_models(models[:2], None, None) == ("a-chat", None)


def test_pick_models_without_router() -> None:
    assert bench_http.pick_models([{"id": "Qwen/Qwen3-Embedding-4B"}], None, None) == (
        None, "Qwen/Qwen3-Embedding-4B")
    assert bench_http.pick_models([{"id": "Qwen/Qwen3.8-27B"}], None, None) == ("Qwen/Qwen3.8-27B", None)


def test_server_args() -> None:
    models = [{"id": "m", "status": {"args": ["/app/llama-server", "--ctx-size", "4096"]}}, {"id": "n"}]
    assert bench_http.server_args(models, "m") == ["--ctx-size", "4096"]
    assert bench_http.server_args(models, "n") is None
    assert bench_http.server_args(models, "missing") is None


def test_timings_summary() -> None:
    timings = {"prompt_n": 100, "prompt_per_second": 1234.56, "predicted_n": 10, "predicted_per_second": 45.67,
               "draft_n": 8, "draft_n_accepted": 6}
    assert bench_http.timings_summary(timings) == {
        "prompt_tokens": 100, "prompt_tps": 1234.6, "generated_tokens": 10, "generation_tps": 45.7,
        "draft_acceptance": 75.0,
    }
    assert "draft_acceptance" not in bench_http.timings_summary({"draft_n": 0})


def test_synthetic_png() -> None:
    png = bench_http.synthetic_png(32, 16, seed=1)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert png == bench_http.synthetic_png(32, 16, seed=1)
    assert png != bench_http.synthetic_png(32, 16, seed=2)
    # Walk the chunks and check their CRCs and the image size.
    pos, chunks = 8, {}
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(kind + data)
        chunks[kind] = data
        pos += 12 + length
    assert struct.unpack(">II", chunks[b"IHDR"][:8]) == (32, 16)
    # One filter byte and 3 bytes per pixel on each row
    assert len(zlib.decompress(chunks[b"IDAT"])) == 16 * (1 + 32 * 3)


def test_connect_detects_the_server_type(fake_llama: FakeServer, fake_vllm: FakeServer, fake_ovms: FakeServer) -> None:
    assert type(bench_http.connect(fake_llama.url, None, 5)) is bench_http.Server
    assert type(bench_http.connect(fake_vllm.url, None, 5)) is bench_http.VllmServer
    assert type(bench_http.connect(fake_ovms.url, None, 5)) is bench_http.OvmsServer


def test_api_key() -> None:
    with FakeServer("llama", api_key="secret") as fake:
        assert bench_http.Server(fake.url, "secret", 5).models()
        with pytest.raises(SystemExit, match="HTTP 401"):
            bench_http.Server(fake.url, "wrong", 5).models()


def test_unreachable_server() -> None:
    with FakeServer("llama") as fake:
        url = fake.url
    with pytest.raises(SystemExit, match="Cannot reach"):
        bench_http.Server(url, None, 2).models()


@pytest.mark.parametrize("kind", ["llama", "vllm", "ovms"])
def test_build_prompt(kind: str) -> None:
    with FakeServer(kind) as fake:  # type: ignore[arg-type]
        server = bench_http.connect(fake.url, None, 5)
        prompt, n = bench_http.build_prompt(server, CHAT_MODEL, 50, salt="Salt. ")
        assert prompt.startswith("Salt. ")
        assert n == len(server.tokenize(CHAT_MODEL, prompt))
        # OVMS cuts at word boundaries, which gives exactly n tokens with the fake tokenizer too.
        assert n == 50


def test_build_code_context(fake_vllm: FakeServer) -> None:
    server = bench_http.connect(fake_vllm.url, None, 5)
    text = bench_http.build_code_context(server, CHAT_MODEL, 200, rotate=3)
    assert len(server.tokenize(CHAT_MODEL, text)) == 200
    assert "### " in text
    with pytest.raises(SystemExit, match="Could not build"):
        bench_http.build_code_context(server, CHAT_MODEL, 10**9)


def test_vllm_stream(fake_vllm: FakeServer) -> None:
    server = bench_http.connect(fake_vllm.url, None, 5)
    assert isinstance(server, bench_http.VllmServer)
    out = server.chat(CHAT_MODEL, "one two three", 16)
    assert out["text"] == "".join(REPLY)
    assert out["usage"]["completion_tokens"] == len(REPLY)
    timings = out["timings"]
    assert timings["prompt_n"] == 3
    assert timings["predicted_n"] == len(REPLY)
    assert timings["predicted_per_second"] > 0
    # The fake server counts 4 draft tokens and 2 accepted ones per request.
    assert (timings["draft_n"], timings["draft_n_accepted"]) == (4, 2)
    payload = next(p for method, path, p in fake_vllm.requests if path == "/v1/chat/completions")
    assert len(payload["cache_salt"]) == 16
    assert payload["stream"] is True


def test_vllm_props(fake_vllm: FakeServer) -> None:
    props = bench_http.connect(fake_vllm.url, None, 5).props(CHAT_MODEL)
    assert props["build_info"] == "vLLM 0.0.0-fake"
    assert props["total_slots"] == 1


def test_tests_with_llama_server(fake_llama: FakeServer) -> None:
    server = bench_http.connect(fake_llama.url, None, 5)
    gen = bench_http.test_generation(server, CHAT_MODEL, n_predict=8, repeat=2)
    assert gen["generated_tokens"] == 8
    assert gen["generation_tps_all"] == [100.0, 100.0]
    assert gen["draft_acceptance"] == 75.0
    pp = bench_http.test_prompt_processing(server, CHAT_MODEL, prompt_tokens=100, repeat=1)
    assert pp["prompt_tokens"] == 100
    conc = bench_http.test_concurrency(server, CHAT_MODEL, 3, 50, 8, (EMBEDDING_MODEL, ["a b", "c"], 3))
    assert conc["concurrency"] == 3
    assert len(conc["per_request_generation_tps"]) == 3
    assert conc["concurrent_embedding"]["batches"] >= 0
    emb = bench_http.test_embeddings(server, EMBEDDING_MODEL, n_texts=3, tokens_each=20)
    assert emb["dimension"] == 64
    assert emb["reported_prompt_tokens"] == 60


def test_main_writes_the_result(fake_vllm: FakeServer, results_dir: Path, capsys: pytest.CaptureFixture[str],
                                monkeypatch: pytest.MonkeyPatch) -> None:
    # Without GPU tools, so that the test does not depend on the computer
    monkeypatch.setattr(bench_http.shutil, "which", lambda _: None)
    config = results_dir / "config.yaml"
    config.write_text("max-model-len: 4096\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "bench_http", "--url", fake_vllm.url, "--model", CHAT_MODEL, "--no-embedding", "--tests", "generation",
        "prompt", "chat", "depth", "image", "--n-predict", "4", "--prompt-tokens", "64", "--repeat", "1",
        "--depth", "100", "--image-size", "8x8", "--server-config", str(config), "--label", "unit test",
    ])
    assert bench_http.main() == 0
    lines = (results_dir / "test-host.jsonl").read_text(encoding="utf-8").splitlines()
    result = json.loads(lines[-1])
    assert result["host"] == "test-host"
    assert result["label"] == "unit test"
    assert result["chat_server_config"] == "max-model-len: 4096\n"
    assert {"generation", "prompt_processing", "chat", "depth", "image"} <= result.keys()
    assert result["chat"]["requests"] == len(bench_http.CHAT_PROMPTS)
    out = capsys.readouterr().out
    assert "| Generation, 1 request" in out
    assert "Appended to" in out


def test_main_without_chat_model(fake_llama: FakeServer, results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llama.models = []
    monkeypatch.setattr("sys.argv", ["bench_http", "--url", fake_llama.url])
    with pytest.raises(SystemExit, match="No chat model"):
        bench_http.main()
    assert not list(results_dir.iterdir())
