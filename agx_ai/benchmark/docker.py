"""Helpers for the temporary test server containers of ``llama_cpp.test_server`` and ``vllm.test_server``."""

from __future__ import annotations

from collections.abc import Callable
import shutil
import subprocess
import time


def logs(name: str) -> str:
    """The combined stdout and stderr of the container."""
    out = subprocess.run(["docker", "logs", name], capture_output=True, text=True, check=False)
    return out.stdout + out.stderr


def is_running(name: str) -> bool:
    """Whether a container with exactly this name is running."""
    out = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{name}$"], capture_output=True, text=True, check=False)
    return bool(out.stdout.strip())


def remove(name: str) -> None:
    """Remove the container if it exists."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def matching_lines(log: str, patterns: tuple[str, ...], n: int | None = None) -> list[str]:
    """The lines of the log that contain any of the patterns (case-insensitive), the last ``n`` if given."""
    lines = [line for line in log.splitlines() if any(p.lower() in line.lower() for p in patterns)]
    return lines[-n:] if n else lines


def wait(name: str, ready: Callable[[], bool | None], timeout: float, interval: float) -> bool:
    """Poll ``ready`` until it returns True (success) or False (failure), the container stops or the time runs out."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        time.sleep(interval)
        state = ready()
        if state is not None:
            return state
        if not is_running(name):
            return False
    return False


def print_gpu_memory(nvidia_query: str, nvidia_process_filter: str | None = None) -> None:
    """Print the GPU memory use with amd-smi and nvidia-smi, if they are installed.

    ``nvidia_query`` is the ``--query-gpu`` fields of nvidia-smi. With ``nvidia_process_filter``, the memory use of
    the compute processes whose name contains it is printed too.
    """
    if shutil.which("amd-smi"):
        out = subprocess.run(["amd-smi", "metric", "-m"], capture_output=True, text=True, check=False).stdout
        for line in out.splitlines():
            if (line.startswith("GPU") or "USED_VRAM:" in line) and "VISIBLE" not in line:
                print(line)
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", f"--query-gpu={nvidia_query}", "--format=csv,noheader"], check=False)
        if nvidia_process_filter:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True, check=False,
            ).stdout
            for line in out.splitlines():
                if nvidia_process_filter in line:
                    print(line)
