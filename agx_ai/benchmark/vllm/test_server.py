r"""Start a temporary vLLM server from any image with a given ``vllm serve --config`` YAML file, for A/B tests.

Compares images or configuration variants without touching the production container (the vLLM counterpart of
``llama_cpp.test_server``). Waits until the server answers, then prints the GPU memory use and the KV cache size.
Benchmark it with ``bench_http --url``, which detects vLLM. Extra ``vllm serve`` arguments go after ``--``.

The Hugging Face cache, ``~/.cache/vllm-models`` (as ``/models``) and the compile caches (the volume ``--cache-volume``
or ``$CACHE_VOLUME``, default that of ``vllm-radeon-vii``) are mounted like in the compose files.
Extra docker arguments come from ``--docker-args`` or ``$DOCKER_ARGS`` (e.g. ``"-e VAR=..."``).
On NVIDIA (nvidia-smi exists), the GPUs are given with ``--gpus "device=GPUS"`` (``--gpus`` or ``$GPUS``,
default all), e.g. ``GPUS=1``. ``GPUS=none`` gives no GPUs, for the CPU image of vLLM. ``GPUS=amd`` gives the AMD GPUs
even when nvidia-smi exists (a machine with both, e.g. agx-z2e with the Radeon VII and the RTX 3090).
The container name comes from ``--name`` or ``$NAME`` (default ``vllm-test``). Stop the production container first,
and remove this one afterwards: ``docker rm -f vllm-test``.

Example:
    uv run -m agx_ai.benchmark.vllm.test_server \\
        aiinfos/vllm-gfx906-mobydick:v0.30.0.x-rocm7.14-pytorch2.13.0-260726fedb27 \\
        /tmp/config-variant.yaml llama-cpp/llama-cpp.env -- --max-model-len 32768
    uv run -m agx_ai.benchmark.bench_http --url http://localhost:9933 --env-file llama-cpp/llama-cpp.env \\
        --no-embedding --tests chat depth --repeat 2 --depth 6400 16000 --server-config /tmp/config-variant.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from agx_ai.benchmark import docker

CONTAINER_PORT = 8000
DEFAULT_CACHE_VOLUME = "vllm-radeon-vii_vllm-cache"
FAILURE_PATTERNS = ("error", "out of memory", "exception")
STARTUP_PATTERNS = (
    "Model loading took", "Available KV cache memory", "GPU KV cache size", "Maximum concurrency",
    "Graph capturing finished",
)


def gpu_args(gpus: str, has_nvidia: bool, dev: Path = Path("/dev")) -> list[str]:
    """The docker arguments that give the container its GPUs (see the module documentation for ``gpus``)."""
    if gpus == "none":
        # A CPU image: with the GPUs (and libcuda), Triton would select its CUDA driver instead of the CPU one.
        return []
    if gpus != "amd" and has_nvidia:
        return ["--gpus", f'"device={gpus or "all"}"']
    args = ["--device", "/dev/dri", "--group-add", "video"]
    if (dev / "kfd").exists():
        args += ["--device", "/dev/kfd"]
    return args


def docker_run_args(
    image: str,
    config: Path,
    env_file: Path | None,
    port: int,
    name: str,
    cache_volume: str,
    gpus: list[str],
    docker_args: list[str],
    serve_args: list[str],
) -> list[str]:
    """The ``docker run`` command line of the test server.

    With an env file, its ``LLAMA_API_KEY`` is the API key. It is expanded by the shell in the container, so that it
    is not visible in the arguments of the container.
    """
    home = Path.home()
    args = [
        "docker", "run", "-d", "--name", name, "-p", f"{port}:{CONTAINER_PORT}", "--ipc", "host",
        "-v", f"{home / '.cache' / 'huggingface' / 'hub'}:/root/.cache/huggingface/hub",
        "-v", f"{home / '.cache' / 'vllm-models'}:/models:ro",
        "-v", f"{cache_volume}:/root/.cache/vllm",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRITON_CACHE_DIR=/root/.cache/vllm/triton",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/inductor",
        "-v", f"{config.resolve()}:/etc/vllm/config.yaml:ro",
        *gpus,
    ]
    key_arg = ""
    if env_file is not None:
        args += ["--env-file", str(env_file.resolve())]
        key_arg = '--api-key "$LLAMA_API_KEY" '
    return [
        *args, *docker_args, "--entrypoint", "/bin/sh", image,
        "-c", f'exec vllm serve --config /etc/vllm/config.yaml {key_arg}"$@"', "vllm", *serve_args,
    ]


def is_healthy(port: int) -> bool:
    """Whether the server answers its health check."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split the command line at the first ``--`` into the arguments of this script and those of ``vllm serve``."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    """Start the test server and wait until it answers."""
    own_argv, serve_args = split_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     usage="%(prog)s [options] image config [env_file] [port] [-- vllm serve args]")
    parser.add_argument("image", help="Docker image of vLLM")
    parser.add_argument("config", type=Path, help="YAML file for vllm serve --config")
    parser.add_argument("env_file", type=Path, nargs="?",
                        help="env file with LLAMA_API_KEY, used as the API key (default: none)")
    parser.add_argument("port", type=int, nargs="?", default=9933, help="host port (default %(default)s)")
    parser.add_argument("--name", default=os.environ.get("NAME", "vllm-test"),
                        help="container name (default: $NAME or vllm-test)")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", ""),
                        help="NVIDIA GPUs (e.g. 1), none or amd (default: $GPUS or all)")
    parser.add_argument("--cache-volume", default=os.environ.get("CACHE_VOLUME", DEFAULT_CACHE_VOLUME),
                        help="Docker volume of the compile caches (default: $CACHE_VOLUME or %(default)s)")
    parser.add_argument("--docker-args", default=os.environ.get("DOCKER_ARGS", ""),
                        help="extra docker run arguments as one string (default: $DOCKER_ARGS)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds to wait for the server (default 30 min)")
    args = parser.parse_args(own_argv)

    cmd = docker_run_args(
        args.image, args.config, args.env_file, args.port, args.name, args.cache_volume,
        gpu_args(args.gpus, shutil.which("nvidia-smi") is not None), shlex.split(args.docker_args), serve_args,
    )
    docker.remove(args.name)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    start = time.monotonic()
    if docker.wait(args.name, lambda: True if is_healthy(args.port) else None, args.timeout, 5):
        print(f"ready on port {args.port} after {round(time.monotonic() - start)} s")
        print("\n".join(docker.matching_lines(docker.logs(args.name), STARTUP_PATTERNS)))
        docker.print_gpu_memory("index,memory.used,memory.total")
        return 0
    print("FAILED to start:", file=sys.stderr)
    print("\n".join(docker.matching_lines(docker.logs(args.name), FAILURE_PATTERNS, 8)), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
