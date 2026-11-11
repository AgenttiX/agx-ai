"""Tests of the llama.cpp-specific benchmarks that do not need llama.cpp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

from agx_ai.benchmark.llama_cpp import bench_container, kld, test_server

PERPLEXITY_OUTPUT = """\
llama_context: n_ctx=2048 n_batch=2048
perplexity: calculating perplexity over 12 chunks, n_ctx=2048
====== Perplexity statistics ======
Mean PPL(Q)                   :   8.123456 ±   0.123456
Mean PPL(base)                :   8.100000 ±   0.120000
Mean ln(PPL(Q)/PPL(base))     :   0.002891 ±   0.000500
====== KL divergence statistics ======
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   1.234567
99.9%   KLD:   0.456789
Median  KLD:   0.004567
====== Token probability statistics ======
RMS Δp    :  1.234 ± 0.012 %
Same top p: 98.765 ± 0.123 %
"""


def test_parse_stats() -> None:
    stats = kld.parse_stats(PERPLEXITY_OUTPUT)
    assert stats["Mean KLD"] == 0.012345
    assert stats["Mean KLD ±"] == 0.000123
    assert stats["Maximum KLD"] == 1.234567
    assert "Maximum KLD ±" not in stats
    assert stats["99.9% KLD"] == 0.456789
    assert stats["Median KLD"] == 0.004567
    assert stats["Mean PPL(Q)"] == 8.123456
    assert stats["Mean ln(PPL(Q)/PPL(base))"] == 0.002891
    assert stats["RMS Δp"] == 1.234
    assert stats["Same top p"] == 98.765
    assert stats["chunks"] == 12
    assert stats["n_ctx"] == 2048
    assert kld.parse_stats("nothing here") == {}


def test_pick_chat_model() -> None:
    models = [
        {"id": "embed-model", "status": {"value": "loaded"}},
        {"id": "unloaded-chat", "status": {"value": "unloaded"}},
        {"id": "sleeping-chat", "status": {"value": "sleeping"}},
    ]
    assert bench_container.pick_chat_model(models) == "sleeping-chat"
    assert bench_container.pick_chat_model([{"id": "plain"}]) == "plain"
    with pytest.raises(SystemExit, match="No chat model"):
        bench_container.pick_chat_model(models[:1])


def _args(**kwargs: object) -> argparse.Namespace:
    defaults = {"exe": None, "image": None, "engine": "docker", "container": "llama-cpp", "hf_cache": "/hf",
                "env": [], "docker_arg": []}
    return argparse.Namespace(**(defaults | kwargs))


def test_docker_prefix() -> None:
    assert bench_container.docker_prefix(_args(exe=Path("/opt/llama"))) == ["/opt/llama"]
    assert bench_container.docker_prefix(_args()) == ["docker", "exec", "llama-cpp"]
    cmd = bench_container.docker_prefix(_args(image="img:tag", env=["A=1"], docker_arg=["--gpus=all"]))
    assert cmd[:2] == ["docker", "run"]
    assert cmd[-3:] == ["--entrypoint", "/app/llama", "img:tag"]
    assert cmd[cmd.index("-e"):cmd.index("-e") + 2] == ["-e", "A=1"]
    assert "--gpus=all" in cmd
    assert "/hf:/root/.cache/huggingface/hub:ro" in cmd
    podman = bench_container.docker_prefix(_args(image="img", engine="podman"))
    assert podman[podman.index("--group-add") + 1] == "keep-groups"


FAKE_LLAMA_BENCH = """
import json, sys
print("| model | test | t/s |")
print(json.dumps({"args": sys.argv[1:], "avg_ts": 12.5}), file=sys.stderr)
print("some log line", file=sys.stderr)
"""


def test_run_bench(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "out" / "bench.jsonl"
    prefix = [sys.executable, "-c", FAKE_LLAMA_BENCH]
    code = bench_container.run_bench(prefix, "/m.gguf", ["-p", "512"], "label", None, output, host_exe=True)
    assert code == 0
    (record,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert record["args"] == ["bench", "-m", "/m.gguf", "-o", "md", "-oe", "jsonl", "-p", "512"]
    assert record["label"] == "label"
    assert record["exe"] == sys.executable
    captured = capsys.readouterr()
    assert "| model |" in captured.out
    assert "some log line" in captured.err


def test_build_text(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    try:
        kld.build_text(path, code_chars=1000, prose_chars=500)
    except SystemExit:
        pytest.skip("the bash man page is not available")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("\n### ")
    assert len(text) == 1000 + 2 + 500


def test_test_server_docker_args(tmp_path: Path) -> None:
    dev = tmp_path / "dev"
    (dev / "dri").mkdir(parents=True)
    preset, config, env = (tmp_path / name for name in ("preset.ini", "config.ini", "test.env"))
    cmd = test_server.docker_run_args("img:tag", preset, config, env, 9940, "llama-x", ["--gpus", "all"], dev)
    assert cmd[:4] == ["docker", "run", "-d", "--name"]
    assert cmd[cmd.index("-p"):cmd.index("-p") + 2] == ["-p", "9940:9931"]
    assert f"{preset}:/etc/llama.cpp/preset.ini:ro" in cmd
    assert f"{config}:/etc/llama.cpp/config.ini:ro" in cmd
    assert ["--env-file", str(env)] == cmd[cmd.index("--env-file"):cmd.index("--env-file") + 2]
    assert "/dev/dri" in cmd
    assert "/dev/kfd" not in cmd
    assert cmd[-5:] == ["--gpus", "all", "--entrypoint", "/app/llama-server", "img:tag"]
    (dev / "kfd").touch()
    cmd = test_server.docker_run_args("img", preset, config, None, 9933, "n", [], dev)
    assert "/dev/kfd" in cmd
    assert "--env-file" not in cmd


def test_test_server_load_state() -> None:
    assert test_server.load_state("starting\nllama_server: model loaded\n") is True
    assert test_server.load_state("[60861] 0.08.890.514 I srv  server_run_i: model loaded\n") is True
    assert test_server.load_state("run: instance name=x exited with status 1") is False
    assert test_server.load_state("loading...") is None
