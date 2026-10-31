# NVIDIA GPU and VRAM overclocking

Instructions for viewing the VRAM temperature and overclocking the RTX 3090 of agx-z2e and its GDDR6X memory on
Linux (Wayland). Not applied yet: these are the instructions for the first try.

- [olealgoritme/gddr6](https://github.com/olealgoritme/gddr6): the VRAM temperature
- [LACT](https://github.com/ilya-zlobintsev/LACT) (Linux GPU Configuration Tool): clock offsets and the power limit,
  applied by a daemon at boot and after a resume
- [Dreaming-Codes/nvidia_oc](https://github.com/Dreaming-Codes/nvidia_oc): a command-line alternative to LACT

## Why

- **Token generation is limited by the memory bandwidth**, so it should get faster about in proportion to the memory
  clock: +500 MHz on the 9751 MHz of the RTX 3090 is ~5 % more bandwidth, i.e. maybe +3-5 % generation.
  Prompt processing is limited by the compute (and the power limit) and should gain little.
- **CUDA lowers the memory clock**: in the P2 state that GeForce cards use for CUDA, the memory runs at 9501 MHz
  instead of 9751 MHz (`nvidia-smi --query-gpu=pstate,clocks.mem --format=csv` during a vLLM benchmark,
  2026-10-31). An offset of the same size gives back the full clock.
- **The prompt processing is power limited**: `nvidia-smi -q -d PERFORMANCE` shows a large counter for
  `SW Power Capping` already at the default 350 W, which is also the maximum of this card (`nvidia-smi -q -d POWER`:
  100-350 W). A core clock offset would therefore help little; an undervolt (LACT's V/F curve) might help more.
  Not tested.

## Current state of agx-z2e (2026-10-31)

| Item | Value |
| --- | --- |
| GPU | ZOTAC RTX 3090 (GA102, 24 GB GDDR6X), with [gpu-unlock](../nvidia-unlock/README.md) |
| Driver | `nvidia-driver-open` 610.43.02 |
| Memory clock | max 9751 MHz, 9501 MHz in P2 (CUDA) |
| NVML offset range | memory -2000 to +6000, graphics -1000 to +1000 (`nvmlDeviceGet*ClkMinMaxVfOffset`), both currently 0 |
| Power limit | 350 W (default and maximum), minimum 100 W |
| VRAM temperature | not reported by the driver: `nvidia-smi --query-gpu=temperature.memory --format=csv` gives `N/A` |
| Secure Boot | disabled (`mokutil --sb-state`), as gddr6 requires |
| Rust | `cargo` in `~/.cargo/bin` (for nvidia_oc) |

## VRAM temperature: gddr6

NVIDIA reports the memory junction temperature of GeForce cards only on Windows (e.g. HWiNFO, GPU-Z). `gddr6` reads it
from the card's registers through `/dev/mem`, based on reverse engineering of the driver, and supports the RTX 3090.
It needs root and Secure Boot disabled.

```bash
sudo apt install libpci-dev
cd ~/Git
git clone https://github.com/olealgoritme/gddr6
cd gddr6
./build_install.sh   # CMake build; answer y to install to /usr/local with sudo
sudo gddr6
```

If it cannot map the registers, add `iomem=relaxed` to the kernel command line
(the README of gddr6 says that some distributions need it and others do not):
```bash
sudo nano /etc/default/grub
# GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.modeset=0 nvidia-drm.modeset=1 gpu_unlock.targets=all iomem=relaxed"
sudo update-grub
```
and reboot. `iomem=relaxed` lets root processes map the memory regions of devices that have a driver, so it weakens
the kernel's protection a little; leave it out if gddr6 works without it.
On agx-z2e (2026-10-31), gddr6 works without it (the kernel has `CONFIG_STRICT_DEVMEM=y` but not
`CONFIG_IO_STRICT_DEVMEM`, and no lockdown), and shows `VRAM Temps: |  62°C |` at idle.

### Without sudo

gddr6 opens `/dev/mem` (`root:kmem`, mode 0640) and maps the registers of the GPU from it. The kernel allows opening
`/dev/mem` only with `CAP_SYS_RAWIO`, so a user needs both that capability and read access to the file. The narrowest
way is to give them to the binary alone: the group `kmem` with setgid, and the capability as a file capability.
```bash
sudo chgrp kmem /usr/local/bin/gddr6
sudo chmod 2755 /usr/local/bin/gddr6
sudo setcap cap_sys_rawio+ep /usr/local/bin/gddr6
gddr6
```
Reinstalling gddr6 (`./build_install.sh`) replaces the binary and drops these, so run them again after an update.
With them, any local user can run gddr6, which reads only the temperature registers. The kernel ignores
`LD_PRELOAD` and similar for binaries with capabilities, but a bug in gddr6 could expose more, as `/dev/mem` gives
access to the registers of all devices (not to RAM, which `CONFIG_STRICT_DEVMEM` blocks).
Adding the user to `kmem` instead of the setgid bit is not enough without the capability, and would give all of the
user's processes read access to `/dev/mem`.

Alternatives: a sudoers rule without a password
(`mika ALL=(root) NOPASSWD: /usr/local/bin/gddr6` in `sudo visudo -f /etc/sudoers.d/gddr6`, still run with `sudo`), or
a root service that writes the temperature periodically to a file or a metrics exporter that other programs read.

Limits: [Tom's Hardware](https://www.tomshardware.com/uk/news/replacing-geforce-rtx-3090-thermal-pads-improves-temps-by-25c)
gives Micron's rating of GDDR6X as up to 95 °C, and the RTX 3090 measured 104 °C in their test; the card lowers the
memory clock itself at ~110 °C. Half of the memory chips of the RTX 3090 are on the back of the board, cooled only by
the backplate, so they are the ones to watch. Measure the temperature at stock clocks first, during a long prompt and
generation (e.g. the 128k depth test below), to know the margin.

## Overclocking: LACT

On Wayland, `nvidia-settings` cannot set clock offsets (its Coolbits work only under X11). NVML can, without a display
server, and LACT uses it. LACT has a GUI and a daemon (`lactd`) that applies the settings of `/etc/lact/config.yaml` at
boot and again after a resume (the driver may reset them in a suspend, and gpu-unlock resets the card in a resume).
It requires the proprietary or open NVIDIA driver with the CUDA libraries, and driver 555 or newer for clock
offsets (565+ recommended).

### Install

Download the `.deb` for Ubuntu 24.04+ from the [releases](https://github.com/ilya-zlobintsev/LACT/releases/latest)
(v0.10.1 on 2026-10-31) and install it:
```bash
sudo apt install ./lact_*.deb
sudo systemctl enable --now lactd
lact cli list-gpus   # the ID of the GPU in config.yaml
```

### Configure

In the GUI (`lact`): select the RTX 3090, then the OC tab: set the VRAM clock offset, apply, and confirm within the
time limit (`apply_settings_timer`, 5 s by default), otherwise LACT reverts the change. The settings are saved to
`/etc/lact/config.yaml`; the equivalent there (the ID from `lact cli list-gpus`):

```yaml
gpus:
  10de:2204-XXXX:XXXX-0000:01:00.0:
    power_cap: 350.0
    mem_clock_offsets:
      0: 500   # per P-state; see the checks below
```
and `sudo systemctl restart lactd`.

Two things to check after the first change, as LACT's documentation does not say them:
1. **The unit of the memory offset.** NVML's range of -2000 to +6000 for this card suggests that the offset is in the
   units of the transfer rate, i.e. twice the memory clock that `nvidia-smi` shows (so +500 would be +250 MHz in
   `nvidia-smi`), as with the old `GPUMemoryTransferRateOffset` of `nvidia-settings`. Compare `clocks.mem` under load
   before and after and write the result here.
2. **Whether the offset applies in P2.** CUDA runs in P2. Check `clocks.mem` while vLLM generates
   (`nvidia-smi --query-gpu=pstate,clocks.mem --format=csv -l 1`); if it stays at 9501 MHz, set the offset for P-state
   2 as well.

## Alternative: nvidia_oc

A single binary that sets the offsets and the power limit through NVML, e.g. from a systemd unit at boot:
```bash
cargo install nvidia_oc
sudo ~/.cargo/bin/nvidia_oc set --index 0 --mem-offset 500 --power-limit 350000
```
(`--power-limit` appears to be in milliwatts; check `nvidia_oc set --help`, as its README documents only an example.)
Unlike LACT, it does not reapply the settings after a resume, so a resume would need a systemd sleep hook too.
Copy the binary to a root-owned path (e.g. `/usr/local/sbin`) before running it from a root service.

## Finding the offset

GDDR6X detects transmission errors and retransmits the data, so an offset that is too high usually does not crash or
corrupt anything at first: it makes the memory slower again. The speed, not stability, finds the limit:

1. Stock clocks: measure the VRAM temperature and the speed with the vLLM configuration of the GPU
   (here [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md)):
   ```bash
   cd ../benchmark
   ./llama_cpp_bench_http.py --url http://localhost:9931 --env-file ../llama-cpp/llama-cpp.env --no-embedding \
       --tests chat depth edit --repeat 2 --depth 6400 32000 100000 128000 --edit-depth 0 32000 --label "..."
   ```
   and `watch -n 1 sudo gddr6` in another terminal.
2. Raise the memory offset in steps (first the 250 MHz that P2 takes away, then +250 MHz at a time in `nvidia-smi`
   clocks) and run the depth test at 6.4k and 32k after each step. Generation should rise about in proportion;
   when it stops rising or falls, the errors cost more than the clock gives: go back one or two steps.
3. Run the full test set at the chosen offset, and check the outputs with
   [`../benchmark/llama_cpp_quality_http.py`](../benchmark/README.md) against the stock clocks.
4. Keep the VRAM temperature below ~100 °C in the longest test, with a margin for summer.
5. Record the offset, the speeds and the temperatures here and in the README of the vLLM configuration.

## Recovery

- The driver does not store the offsets: they are gone after a reboot unless `lactd` (or a unit of nvidia_oc)
  applies them again. `sudo systemctl disable --now lactd` and a reboot give the stock clocks back.
- If the desktop freezes at boot because of an offset (unlikely for a memory offset), boot into recovery mode or add
  `systemd.mask=lactd.service` to the kernel command line in GRUB (press Shift or Esc during the boot, then `e`).
- LACT's own instructions: [Recovering from a bad overclock](https://github.com/ilya-zlobintsev/LACT/blob/master/docs/Recovering-from-a-bad-overclock.md).
