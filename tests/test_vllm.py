"""Tests of the vLLM-specific benchmarks with a fake vLLM server."""

from __future__ import annotations

from pathlib import Path

import pytest

from agx_ai.benchmark import bench_http
from agx_ai.benchmark.vllm import concurrent_agents, prefix_return, test_server
from tests.fake_server import CHAT_MODEL, REPLY, FakeServer


def test_split_argv() -> None:
    assert test_server.split_argv(["img", "cfg"]) == (["img", "cfg"], [])
    assert test_server.split_argv(["img", "cfg", "--", "--max-model-len", "8", "--", "x"]) == (
        ["img", "cfg"], ["--max-model-len", "8", "--", "x"])


def test_gpu_args(tmp_path: Path) -> None:
    assert test_server.gpu_args("none", has_nvidia=True) == []
    assert test_server.gpu_args("", has_nvidia=True) == ["--gpus", '"device=all"']
    assert test_server.gpu_args("0,1", has_nvidia=True) == ["--gpus", '"device=0,1"']
    assert test_server.gpu_args("", has_nvidia=False, dev=tmp_path) == ["--device", "/dev/dri", "--group-add", "video"]
    (tmp_path / "kfd").touch()
    assert test_server.gpu_args("amd", has_nvidia=True, dev=tmp_path)[-2:] == ["--device", "/dev/kfd"]


def test_docker_run_args(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    env = tmp_path / "test.env"
    cmd = test_server.docker_run_args("img", config, env, 9940, "vllm-x", "vol", ["--gpus", '"device=1"'],
                                      ["-e", "A=1"], ["--max-model-len", "8"])
    assert cmd[cmd.index("-p"):cmd.index("-p") + 2] == ["-p", "9940:8000"]
    assert "vol:/root/.cache/vllm" in cmd
    assert f"{config}:/etc/vllm/config.yaml:ro" in cmd
    i = cmd.index("img")
    assert cmd[i - 4:i] == ["-e", "A=1", "--entrypoint", "/bin/sh"]
    # The API key is expanded by the shell in the container, not given in the arguments.
    assert cmd[i + 1:] == ["-c", 'exec vllm serve --config /etc/vllm/config.yaml --api-key "$LLAMA_API_KEY" "$@"',
                           "vllm", "--max-model-len", "8"]
    no_key = test_server.docker_run_args("img", config, None, 9933, "n", "vol", [], [], [])
    assert no_key[-3:] == ["-c", 'exec vllm serve --config /etc/vllm/config.yaml "$@"', "vllm"]


def test_is_healthy(fake_vllm: FakeServer) -> None:
    port = int(fake_vllm.url.rsplit(":", 1)[1])
    assert test_server.is_healthy(port)
    fake_vllm.__exit__()
    assert not test_server.is_healthy(port)


def test_prefix_return_ask(fake_vllm: FakeServer) -> None:
    server = bench_http.connect(fake_vllm.url, None, 5)
    assert isinstance(server, bench_http.VllmServer)
    ttft, n, text = prefix_return.ask(server, CHAT_MODEL, "a b c d", "salt", 16)
    assert ttft >= 0
    assert n == 4
    assert text == "".join(REPLY)
    payload = next(p for method, path, p in reversed(fake_vllm.requests) if method == "POST")
    assert payload["cache_salt"] == "salt"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_run_agents(fake_vllm: FakeServer) -> None:
    server = bench_http.connect(fake_vllm.url, None, 5)
    assert isinstance(server, bench_http.VllmServer)
    result = concurrent_agents.run_agents(server, CHAT_MODEL, depth=100, n=3, max_tokens=16)
    assert len(result["ttft_s"]) == 3
    assert result["total_tps"] > 0
    salts = {p["cache_salt"] for _, path, p in fake_vllm.requests if path == "/v1/chat/completions"}
    assert len(salts) == 3  # one per agent, shared by its warm-up and its turn


@pytest.mark.parametrize("module", [prefix_return, concurrent_agents])
def test_needs_vllm(module: object, fake_llama: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["x", "--url", fake_llama.url])
    with pytest.raises(SystemExit, match="needs a vLLM server"):
        module.main()  # type: ignore[attr-defined]
