r"""Reserve the CPU expert pool's cores for Qwen3.8-Flash-Next while it serves requests (a prototype, for review).

While vLLM reports running requests (``vllm:num_requests_running`` in ``/metrics``), the programs of the users
(``user.slice``) and the movable interrupts are kept off the reserved CPUs; when the server has been idle for a few
seconds, everything gets all CPUs back. The CPU pool's threads sleep between requests, so the reserved cores are free
for other programs then. This complements ``QWEN38_CPU_EXPERTS_FIFO`` (see the README): the real-time priority makes
other threads give way on the pool's CPUs, and this moves them (and the interrupts) elsewhere altogether.

It has to run as root on the host (``systemctl set-property`` and ``/proc/irq``), e.g. as a systemd service:

    sudo python3 isolate_cpus.py --url http://localhost:9932 --cpus 1-2,5-6,9-10,13-14,17-18,21-22,25-26,29-30,\
33-34,37-38,41-42,45-46,49-50,53-54,57-58,61-62
    python3 isolate_cpus.py --dry-run ...   # without root: print what it would do

The default CPUs are the pool's and the loaders' cores (16 cores with their SMT siblings) of docker-compose.yml.
Kernel threads that are bound to a CPU (per-CPU workers, ksoftirqd) and the timer tick stay; only the kernel
command line can remove those (``nohz_full=``, ``isolcpus=``), permanently. Only the Python standard library is used.
"""

import argparse
import contextlib
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

DEFAULT_CPUS: str = ("1-2,5-6,9-10,13-14,17-18,21-22,25-26,29-30,"
                     "33-34,37-38,41-42,45-46,49-50,53-54,57-58,61-62")
IRQ_DIR: Path = Path("/proc/irq")
RUNNING: re.Pattern[str] = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)", re.MULTILINE)


def parse_cpus(spec: str) -> set[int]:
    """'1-2,5' -> {1, 2, 5}."""
    cpus: set[int] = set()
    for part in spec.split(","):
        if part.strip():
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def format_cpus(cpus: set[int]) -> str:
    """{1, 2, 5} -> '1-2,5'."""
    out: list[str] = []
    for c in sorted(cpus):
        if out and out[-1].split("-")[-1] == str(c - 1):
            out[-1] = f"{out[-1].split('-')[0]}-{c}"
        else:
            out.append(str(c))
    return ",".join(out)


def running_requests(url: str, timeout: float = 2.0) -> int | None:
    """The number of running requests of the vLLM server (None if it does not answer)."""
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=timeout) as r:
            text = r.read().decode()
    except OSError:
        return None
    return int(sum(float(v) for v in RUNNING.findall(text)))


class Isolation:
    """Applies and reverts the reservation of the CPUs."""

    def __init__(self, reserved: set[int], all_cpus: set[int], dry_run: bool) -> None:
        self.others: str = format_cpus(all_cpus - reserved)
        self.dry_run: bool = dry_run
        self.active: bool = False
        self.saved_irqs: dict[Path, str] = {}

    def _run(self, cmd: list[str]) -> None:
        if self.dry_run:
            print("would run:", " ".join(cmd))
        else:
            subprocess.run(cmd, check=False)

    def _write(self, path: Path, value: str) -> None:
        if self.dry_run:
            print(f"would write {value} to {path}")
            return
        with contextlib.suppress(OSError):  # e.g. per-CPU or managed interrupts cannot be moved
            path.write_text(value)

    def apply(self) -> None:
        """Keep user.slice and the interrupts on the other CPUs."""
        self._run(["systemctl", "set-property", "--runtime", "user.slice", f"AllowedCPUs={self.others}"])
        self.saved_irqs = {}
        for f in sorted(IRQ_DIR.glob("*/smp_affinity_list")):
            with contextlib.suppress(OSError):
                self.saved_irqs[f] = f.read_text().strip()
            self._write(f, self.others)
        self.active = True

    def revert(self) -> None:
        """Give user.slice and the interrupts all CPUs again."""
        self._run(["systemctl", "set-property", "--runtime", "user.slice", "AllowedCPUs="])
        for f, value in self.saved_irqs.items():
            self._write(f, value)
        self.active = False


def main() -> int:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9932", help="the vLLM server")
    parser.add_argument("--cpus", default=DEFAULT_CPUS, help="the CPUs to reserve during requests")
    parser.add_argument("--interval", type=float, default=0.25, help="seconds between checks")
    parser.add_argument("--idle", type=float, default=5.0, help="seconds without requests before releasing")
    parser.add_argument("--dry-run", action="store_true", help="print the actions instead of doing them")
    parser.add_argument("--once", action="store_true", help="check once and exit (for testing)")
    args = parser.parse_args()
    reserved = parse_cpus(args.cpus)
    all_cpus = parse_cpus(Path("/sys/devices/system/cpu/online").read_text().strip())
    iso = Isolation(reserved, all_cpus, args.dry_run)
    last_busy = 0.0
    try:
        while True:
            n = running_requests(args.url)
            now = time.monotonic()
            if n:
                last_busy = now
                if not iso.active:
                    print(f"{n} running requests: reserving CPUs {format_cpus(reserved)}", flush=True)
                    iso.apply()
            elif iso.active and now - last_busy > args.idle:
                print("idle: releasing the CPUs", flush=True)
                iso.revert()
            if args.once:
                print(f"running requests: {n}, reserved: {iso.active}")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if iso.active:
            iso.revert()
    return 0


if __name__ == "__main__":
    sys.exit(main())
