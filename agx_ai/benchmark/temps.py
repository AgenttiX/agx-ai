"""Sample the temperatures of the GPUs (including their memory) and the CPU during a benchmark.

Sources, all optional (a missing one is skipped):

- hwmon in sysfs (``/sys/class/hwmon``): ``amdgpu`` (edge, junction and memory of AMD GPUs), ``nvidia_vram`` (the
  VRAM of NVIDIA Ampere and Ada GPUs, from the kernel module ``nvidia_vram_hwmon``; NVIDIA's driver does not report
  it on Linux), and the CPU (``k10temp``, ``zenpower``, ``coretemp``),
- ``nvidia-smi``: the core temperature of NVIDIA GPUs, and their memory temperature where the driver reports it
  (data center GPUs).

GPUs are named by their PCI address and, where nvidia-smi or amd-smi knows it, their model. All sensors of these
kinds are recorded, as it is not always known which device a server uses. Only the Python standard library is needed.

    uv run -m agx_ai.benchmark.temps  # print the current readings
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
from typing import Self

HWMON: Path = Path("/sys/class/hwmon")
GPU_HWMON: frozenset[str] = frozenset({"amdgpu", "nvidia_vram"})
CPU_HWMON: frozenset[str] = frozenset({"k10temp", "zenpower", "coretemp"})


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_names() -> dict[str, str]:
    """GPU models by PCI address (``0000:01:00.0``), from nvidia-smi and amd-smi."""
    names: dict[str, str] = {}
    if shutil.which("nvidia-smi"):
        for line in _run(["nvidia-smi", "--query-gpu=pci.bus_id,name", "--format=csv,noheader"]).splitlines():
            bus, _, name = line.partition(",")
            if name:
                names[bus.strip()[-12:].lower()] = name.strip()
    if shutil.which("amd-smi"):
        try:
            gpus = json.loads(_run(["amd-smi", "static", "--asic", "--bus", "--json"]) or "[]")
            for g in gpus if isinstance(gpus, list) else [gpus]:
                names[g["bus"]["bdf"].lower()] = g["asic"]["market_name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return names


def hwmon_sensors(root: Path = HWMON, names: dict[str, str] | None = None) -> dict[str, Path]:
    """The temperature inputs of the GPU and CPU hwmon devices, by a readable sensor name."""
    names = names or {}
    sensors: dict[str, Path] = {}
    for dev in sorted(root.glob("hwmon*")):
        try:
            kind = (dev / "name").read_text().strip()
        except OSError:
            continue
        if kind not in GPU_HWMON | CPU_HWMON:
            continue
        if kind in GPU_HWMON:
            pci = (dev / "device").resolve().name.lower()
            device = f"{names.get(pci, kind)} ({pci})"
        else:
            device = "CPU"
        for inp in sorted(dev.glob("temp*_input")):
            label_file = inp.with_name(inp.name.replace("_input", "_label"))
            try:
                label = label_file.read_text().strip()
            except OSError:
                label = "VRAM" if kind == "nvidia_vram" else inp.name.removesuffix("_input")
            sensors[f"{device} {label}"] = inp
    return sensors


def read_nvidia_smi(names: dict[str, str]) -> dict[str, float]:
    """The core (and, if reported, memory) temperatures of the NVIDIA GPUs."""
    out = _run(["nvidia-smi", "--query-gpu=pci.bus_id,temperature.gpu,temperature.memory",
                "--format=csv,noheader,nounits"])
    temps: dict[str, float] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        pci = parts[0][-12:].lower()
        device = f"{names.get(pci, 'NVIDIA GPU')} ({pci})"
        for label, value in (("core", parts[1]), ("memory", parts[2])):
            if value.replace(".", "", 1).isdigit():  # not "N/A"
                temps[f"{device} {label}"] = float(value)
    return temps


def read_hwmon(sensors: dict[str, Path]) -> dict[str, float]:
    """The current readings in °C (sensors that fail to read, e.g. a suspended GPU, are left out)."""
    temps: dict[str, float] = {}
    for name, path in sensors.items():
        with contextlib.suppress(OSError, ValueError):
            temps[name] = int(path.read_text()) / 1000
    return temps


class TempSampler:
    """Samples the temperatures in a background thread; ``summary()`` gives the min, mean and max per sensor."""

    def __init__(self, interval: float = 1.0, root: Path = HWMON) -> None:
        self.interval: float = interval
        self.names: dict[str, str] = gpu_names()
        self.sensors: dict[str, Path] = hwmon_sensors(root, self.names)
        self.nvidia: bool = shutil.which("nvidia-smi") is not None
        self.samples: dict[str, list[float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def read(self) -> dict[str, float]:
        """One reading of all sensors."""
        temps = read_nvidia_smi(self.names) if self.nvidia else {}
        temps.update(read_hwmon(self.sensors))
        return temps

    def sample(self) -> None:
        """Add one reading to the samples."""
        for name, value in self.read().items():
            self.samples.setdefault(name, []).append(value)

    def summary(self) -> dict[str, dict[str, float]]:
        """``{sensor: {"min", "mean", "max", "samples"}}`` in °C, the GPUs first."""
        def order(name: str) -> tuple[bool, str]:
            return name.startswith("CPU"), name

        return {
            name: {"min": min(v), "mean": round(statistics.fmean(v), 1), "max": max(v), "samples": len(v)}
            for name, v in sorted(self.samples.items(), key=lambda kv: order(kv[0])) if v
        }

    def __enter__(self) -> Self:
        self.sample()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.sample()


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    """One line: ``sensor min / mean / max``, separated by commas."""
    return ", ".join(f"{name} {s['min']:g} / {s['mean']:g} / {s['max']:g}" for name, s in summary.items())


def main() -> int:
    """Print the current readings."""
    sampler = TempSampler()
    for name, value in sampler.read().items():
        print(f"{name}: {value:g} °C")
    if sampler.nvidia and not any(name.endswith(("VRAM", "memory")) for name in sampler.read()):
        print("(no VRAM temperature of an NVIDIA GPU: load the nvidia_vram_hwmon kernel module for it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
