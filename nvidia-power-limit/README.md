# NVIDIA power limit

A script and a systemd service that lock the core clock (and optionally set the power limit) of the RTX 3070 on the
Proxmox host `agx-h12` at boot, and create the NVIDIA device nodes for the `agx-ai` LXC container.

The GPU is from a friend's old cryptocurrency mining setup and has therefore degraded over time: at the default
settings, it crashes with Xid 79 ("GPU has fallen off the bus") under long full loads, but it has been stable with the
core clock locked to 1600 MHz or with a 170 W power limit. The measurements are in
[`../llama-cpp-rtx3070`](../llama-cpp-rtx3070/README.md#xid-79-crashes), and the clock lock applies to both
[`../llama-cpp-rtx3070`](../llama-cpp-rtx3070/README.md) and [`../vllm-rtx3070`](../vllm-rtx3070/README.md).

## Setup

The clock lock and the power limit reset at every reboot of the host. Install the script and the systemd service on
`agx-h12` (from this directory):

```sh
sudo install -m 755 nvidia-power-limit.sh /usr/local/sbin/nvidia-power-limit.sh
sudo install -m 644 nvidia-power-limit.service /etc/systemd/system/nvidia-power-limit.service
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power-limit.service
systemctl status nvidia-power-limit.service
nvidia-smi -q -d CLOCK | grep -A2 "Locked\|Max Clocks"
```

The script and the service were called `gpu-power-limit` until 2026-10-31. To replace an installed old version (before
installing the new one as above):

```sh
sudo systemctl disable --now gpu-power-limit.service
[ -e /etc/default/gpu-power-limit ] && sudo mv /etc/default/gpu-power-limit /etc/default/nvidia-power-limit
sudo rm /etc/systemd/system/gpu-power-limit.service /usr/local/sbin/gpu-power-limit.sh
```

The script waits up to 300 s for the driver and the GPU after boot, enables persistence mode (so that the limits are kept
when no program uses the GPU) and locks the core clock to 210-1600 MHz. By default, it leaves the power limit at the
card's default of 240 W, see [Locking the GPU clock](../llama-cpp-rtx3070/README.md#locking-the-gpu-clock). It also creates the NVIDIA device nodes, including
`/dev/nvidia-uvm-tools`, before Proxmox starts the guests. Previously, the autostart of the `agx-ai` container failed
with `TASK ERROR: Device /dev/nvidia-uvm-tools does not exist`, as the node only appears when something first uses
the GPU, so the container can now be set to start at boot. The settings (`POWER_LIMIT_W`, `MAX_CLOCK_MHZ`,
`GPU_BUS_ID`, `WAIT_TIMEOUT_S`) can be overridden in `/etc/default/nvidia-power-limit`, e.g.:

```sh
echo 'MAX_CLOCK_MHZ=1700' | sudo tee -a /etc/default/nvidia-power-limit
sudo systemctl restart nvidia-power-limit.service
```

An empty value leaves that setting unchanged; `POWER_LIMIT_W` is empty by default. To also set the 170 W power
limit, e.g. as a safety net against short power spikes:

```sh
echo 'POWER_LIMIT_W=170' | sudo tee -a /etc/default/nvidia-power-limit
sudo systemctl restart nvidia-power-limit.service
```

Restarting the service does not raise a power limit that was set earlier (also not one set by an earlier version of
the script, which set 170 W by default), so after removing or raising it, also run
`sudo nvidia-smi -i 00000000:81:00.0 -pl 240` or reboot.
