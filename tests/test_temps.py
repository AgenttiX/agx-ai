"""Tests of agx_ai.benchmark.temps with a fake hwmon tree."""

from pathlib import Path
import sys

import pytest

from agx_ai.benchmark import temps


def _hwmon(root: Path, n: int, name: str, device: Path, sensors: dict[str, tuple[str | None, int]]) -> None:
    dev = root / f"hwmon{n}"
    dev.mkdir(parents=True)
    (dev / "name").write_text(name + "\n")
    device.mkdir(parents=True, exist_ok=True)
    (dev / "device").symlink_to(device)
    for key, (label, millideg) in sensors.items():
        (dev / f"{key}_input").write_text(f"{millideg}\n")
        if label is not None:
            (dev / f"{key}_label").write_text(label + "\n")


@pytest.mark.skipif(sys.platform == "win32", reason="hwmon is Linux-specific")
def test_hwmon_sensors_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "hwmon"
    _hwmon(root, 0, "amdgpu", tmp_path / "pci" / "0000:4f:00.0", {"temp1": ("edge", 40000), "temp3": ("mem", 45500)})
    _hwmon(root, 1, "nvidia_vram", tmp_path / "pci" / "0000:01:00.0", {"temp1": (None, 64000)})
    _hwmon(root, 2, "zenpower", tmp_path / "pci" / "0000:00:18.3", {"temp1": ("Tdie", 50000)})
    _hwmon(root, 3, "nvme", tmp_path / "pci" / "0000:21:00.0", {"temp1": ("Composite", 30000)})
    names = {"0000:4f:00.0": "AMD Radeon VII"}
    sensors = temps.hwmon_sensors(root, names)
    assert set(sensors) == {
        "AMD Radeon VII (0000:4f:00.0) edge", "AMD Radeon VII (0000:4f:00.0) mem",
        "nvidia_vram (0000:01:00.0) VRAM", "CPU Tdie",
    }
    assert temps.read_hwmon(sensors)["AMD Radeon VII (0000:4f:00.0) mem"] == 45.5

    sampler = temps.TempSampler(root=root)
    sampler.nvidia = False
    sampler.sensors = sensors
    sampler.sample()
    (root / "hwmon0" / "temp1_input").write_text("50000\n")
    sampler.sample()
    summary = sampler.summary()
    assert summary["AMD Radeon VII (0000:4f:00.0) edge"] == {"min": 40.0, "mean": 45.0, "max": 50.0, "samples": 2}
    assert list(summary)[-1] == "CPU Tdie"  # the GPUs first
    assert "CPU Tdie 50 / 50 / 50" in temps.format_summary(summary)
