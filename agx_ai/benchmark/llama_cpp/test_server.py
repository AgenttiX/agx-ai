r"""Start a temporary llama-server router from any image with a given preset, for A/B tests.

Compares images, backends (ROCm vs Vulkan) or preset variants without touching the production container. Waits until
the load-on-startup model has loaded, then prints the GPU memory use. Benchmark it with ``bench_http --url``.

Extra docker arguments come from ``--docker-args`` or the environment variable ``DOCKER_ARGS``
(e.g. ``"-e GGML_VK_VISIBLE_DEVICES=0"``, or ``"--gpus all"`` for a CUDA image), and the container name from
``--name`` or ``NAME`` (default ``llama-test``), e.g. to run test servers on two GPUs at the same time.
Stop the production container first if the GPU memory is needed, and remove this one afterwards:
``docker rm -f llama-test``.

Example (Vulkan on the Radeon VII with a preset variant):
    uv run -m agx_ai.benchmark.llama_cpp.test_server ghcr.io/ggml-org/llama.cpp:full-vulkan \\
        /tmp/preset-variant.ini llama-cpp/config.ini llama-cpp/llama-cpp.env
    uv run -m agx_ai.benchmark.bench_http --url http://localhost:9933 --env-file llama-cpp/llama-cpp.env \\
        --no-embedding --tests chat depth --repeat 2 --depth 6400 16000 36500 --label "Vulkan"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from agx_ai.benchmark import docker
from agx_ai.hostname import REPO_DIR

DEFAULT_CONFIG = REPO_DIR / "llama-cpp" / "config.ini"
# The port of the server in config.ini.
CONTAINER_PORT = 9931
HF_CACHE = "/root/.cache/huggingface/hub"
FAILURE_PATTERNS = ("error", "fail", "out of memory", "exception")


def docker_run_args(
    image: str,
    preset: Path,
    config: Path,
    env_file: Path | None,
    port: int,
    name: str,
    docker_args: list[str],
    dev: Path = Path("/dev"),
) -> list[str]:
    """The ``docker run`` command line of the test server. ``dev`` is the device directory (for testing)."""
    args = [
        "docker", "run", "-d", "--name", name, "-p", f"{port}:{CONTAINER_PORT}",
        # Newer images set LLAMA_ARG_PORT=8080, which would override the port of config.ini.
        "-e", f"LLAMA_ARG_PORT={CONTAINER_PORT}",
        "-v", f"{Path.home() / '.cache' / 'huggingface' / 'hub'}:{HF_CACHE}",
        "-v", f"{config.resolve()}:/etc/llama.cpp/config.ini:ro",
        "-v", f"{preset.resolve()}:/etc/llama.cpp/preset.ini:ro",
    ]
    if (dev / "dri").exists():  # Vulkan, ROCm
        args += ["--device", "/dev/dri", "--group-add", "video"]
    if (dev / "kfd").exists():  # ROCm
        args += ["--device", "/dev/kfd"]
    if env_file is not None:
        args += ["--env-file", str(env_file.resolve())]
    return [*args, *docker_args, "--entrypoint", "/app/llama-server", image]


def load_state(log: str) -> bool | None:
    """True if the model has loaded, False if the router's model instance failed, None if still loading."""
    if "llama_server: model loaded" in log:
        return True
    if "exited with status" in log:
        return False
    return None


def main(argv: list[str] | None = None) -> int:
    """Start the test server and wait until its model has loaded."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Docker image of llama.cpp")
    parser.add_argument("preset", type=Path, help="preset file of the router (preset.ini)")
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG,
                        help="server config.ini (default llama-cpp/config.ini of this repository)")
    parser.add_argument("env_file", type=Path, nargs="?", help="env file with LLAMA_API_KEY (default: none)")
    parser.add_argument("port", type=int, nargs="?", default=9933, help="host port (default %(default)s)")
    parser.add_argument("--name", default=os.environ.get("NAME", "llama-test"),
                        help="container name (default: $NAME or llama-test)")
    parser.add_argument("--docker-args", default=os.environ.get("DOCKER_ARGS", ""),
                        help="extra docker run arguments as one string (default: $DOCKER_ARGS)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for the model (default 15 min)")
    args = parser.parse_args(argv)

    cmd = docker_run_args(args.image, args.preset, args.config, args.env_file, args.port, args.name,
                          shlex.split(args.docker_args))
    docker.remove(args.name)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    if docker.wait(args.name, lambda: load_state(docker.logs(args.name)), args.timeout, 3):
        print(f"loaded on port {args.port}")
        docker.print_gpu_memory("name,memory.used", "llama")
        return 0
    print("FAILED to load:", file=sys.stderr)
    print("\n".join(docker.matching_lines(docker.logs(args.name), FAILURE_PATTERNS, 5)), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
