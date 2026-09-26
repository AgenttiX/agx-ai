#!/usr/bin/env bash
# Sets the power limit (and optionally a core clock limit) of the NVIDIA GPU on the Proxmox host agx-h12 at boot.
#
# The RTX 3070 crashes with Xid 79 ("GPU has fallen off the bus") under long full loads at the default settings,
# but has been stable with a 170 W power limit, and with the core clock locked to 1600 MHz (with or without the power
# limit). Both are set, as the clock lock also keeps the GPU cooler. See ../README.md.
# The settings reset at every reboot, hence this script.
#
# The script also creates the NVIDIA device nodes, including /dev/nvidia-uvm and /dev/nvidia-uvm-tools,
# as they only appear when something first uses the GPU. Without them, Proxmox fails to autostart the agx-ai
# LXC container with "TASK ERROR: Device /dev/nvidia-uvm-tools does not exist".
#
# Settings can be overridden in /etc/default/gpu-power-limit.
set -u

# Power limit in W (nvidia-smi -pl). Empty = not changed (the default of the card is 240 W).
POWER_LIMIT_W=170
# Maximum core clock in MHz (nvidia-smi -lgc). Empty = not locked. See ../README.md.
MAX_CLOCK_MHZ=1600
# PCI bus ID of the GPU, as shown by nvidia-smi
GPU_BUS_ID="00000000:81:00.0"
# How long to wait for the driver and the GPU to become available after boot
WAIT_TIMEOUT_S=300
WAIT_INTERVAL_S=5

if [ -r /etc/default/gpu-power-limit ]; then
    # shellcheck source=/dev/null
    . /etc/default/gpu-power-limit
fi

log() {
    echo "gpu-power-limit: $*"
}

create_device_nodes() {
    if command -v nvidia-modprobe >/dev/null 2>&1; then
        # -c 0: /dev/nvidia0 and /dev/nvidiactl, -u: /dev/nvidia-uvm and /dev/nvidia-uvm-tools
        nvidia-modprobe -c 0 -u || log "nvidia-modprobe failed"
    else
        modprobe nvidia_uvm || log "modprobe nvidia_uvm failed"
    fi
}

gpu_available() {
    nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | grep -qi "$GPU_BUS_ID"
}

deadline=$(( $(date +%s) + WAIT_TIMEOUT_S ))
until create_device_nodes && gpu_available; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "GPU $GPU_BUS_ID did not appear within $WAIT_TIMEOUT_S s, power limit not set"
        exit 1
    fi
    log "waiting for GPU $GPU_BUS_ID..."
    sleep "$WAIT_INTERVAL_S"
done

for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [ -e "$node" ] || log "warning: $node does not exist"
done

# Persistence mode keeps the driver state, including the power limit, when no program is using the GPU.
nvidia-smi -i "$GPU_BUS_ID" -pm 1 || log "could not enable persistence mode"

if [ -n "$POWER_LIMIT_W" ]; then
    if ! nvidia-smi -i "$GPU_BUS_ID" -pl "$POWER_LIMIT_W"; then
        log "setting the power limit failed"
        exit 1
    fi
fi
current=$(nvidia-smi -i "$GPU_BUS_ID" --query-gpu=power.limit --format=csv,noheader,nounits)
log "power limit is now $current W"

if [ -n "$MAX_CLOCK_MHZ" ]; then
    if nvidia-smi -i "$GPU_BUS_ID" -lgc "210,$MAX_CLOCK_MHZ"; then
        log "core clock locked to 210-$MAX_CLOCK_MHZ MHz"
    else
        log "locking the core clock failed"
        exit 1
    fi
fi
