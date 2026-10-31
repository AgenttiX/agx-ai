# NVIDIA GPU unlock

[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)

- [RTX 3090 (GA102) compatibility report](https://github.com/Zagitta/gpu-unlock/issues/25)
- [Initramfs binding issue](https://github.com/Zagitta/gpu-unlock/issues/11)
- [Wccftech news article](https://wccftech.com/nvidia-geforce-rtx-30-gpus-get-up-to-2x-faster-ai-compute-performance-with-new-gpu-unlock-mod-real-world-gains-of-up-to-60/)

`gpu_unlock` is a kernel module that removes the half-rate cap of the FP16/BF16/TF32 tensor core math
(with FP32 accumulation) of GeForce cards.
It runs at boot before the NVIDIA driver binds the card and resets the card after the unlock.
It is expected to speed up prompt processing of unquantized and AWQ/GPTQ models (e.g. in vLLM),
but not token generation or quantized GGUF models in llama.cpp.

## agx-z2e setup

| Item | Value |
| --- | --- |
| OS | Kubuntu 26.04.1 LTS, kernel 7.0.0-42-generic |
| GPU | ZOTAC RTX 3090 (GA102, `10de:2204`) at `0000:01:00.0`, the only active display output |
| Other GPUs | GTX Titan (GK110, not supported by the unlock, no driver bound), Radeon VII (display output disabled with `amdgpu.modeset=0` to keep the desktop on the RTX 3090) |
| NVIDIA driver | `nvidia-driver-open` 610.43.02 (Ubuntu packages, DKMS) |
| gpu-unlock | 2.0.0, built from `../gpu-unlock` with `make install` (`gpu-unlock-dkms` + `gpu-unlock-firmware`) |
| Root filesystem | LUKS-encrypted (`crypto-vg`), passphrase asked in the initramfs |
| Kernel command line | `amdgpu.modeset=0 nvidia-drm.modeset=1 gpu_unlock.targets=all` |

## Black screen at boot (solved)

The first attempt with gpu-unlock 1.0.0 (2026-10-26) did not boot: the screen turned black during the boot.
The boot left no journal at all, so it never got past the initramfs.

Cause: the root filesystem is encrypted, so the initramfs stops to ask for the LUKS passphrase.
On the display card, `gpu_unlock` removes the firmware framebuffer (simpledrm) and resets the card,
and the screen stays blank until `nvidia-drm` takes over.
`nvidia.ko` was not in the initramfs, so `nvidia-drm` could only load from the root filesystem,
i.e. after the passphrase had been entered.
The passphrase prompt was therefore invisible, and the boot was most likely waiting for it.
With the Radeon VII console disabled, there is no other output to show the prompt on.

Fix: load the NVIDIA driver in the initramfs, so that `nvidia-drm` shows the prompt.

1. Add the modules to `/etc/initramfs-tools/modules`:
   ```
   nvidia
   nvidia_modeset
   nvidia_drm
   ```
2. Use gpu-unlock 2.0.0 or newer.
   It installs `/usr/share/initramfs-tools/scripts/init-top/gpu-unlock`,
   which loads `gpu_unlock` before `init-top/udev` loads `nvidia` for the card.
   Without it, `nvidia` in the initramfs binds the card first and the unlock fails with `-EBUSY` (-16)
   (see [issue #11](https://github.com/Zagitta/gpu-unlock/issues/11)),
   because `modules.d` and `/etc/initramfs-tools/modules` are only loaded after the udev coldplug.
   With 1.0.0, `softdep nvidia pre: gpu_unlock` in `/etc/modprobe.d/gpu-unlock.conf` is reported to work instead.
3. `sudo update-initramfs -u`, add `gpu_unlock.targets=all` to `GRUB_CMDLINE_LINUX_DEFAULT`, `sudo update-grub`
   and power the computer fully off before booting.

Verify the initramfs:
```bash
lsinitramfs /boot/initrd.img-$(uname -r) | grep -E 'nvidia(-modeset|-drm)?\.ko|gpu_unlock|init-top/gpu-unlock'
```

This makes the initramfs ~100 MB larger (286 MB instead of 183 MB),
mostly due to the GSP firmware `gsp_ga10x.bin` (84 MB). `/boot` is 2 GB.
Only the initramfs of the newest kernel has been rebuilt.
Booting an older kernel with the unlock enabled gives the black screen again,
but typing the passphrase blind should still work.

## Verification

After a cold boot, `gpu_unlock: unlocked` should appear before `NVRM: loading`:
```bash
journalctl -k -b | grep -E 'gpu_unlock|NVRM: loading'
lspci -nnk -s 0000:01:00.0
```

## Suspend (works)

Tested on 2026-10-31 with a suspend to RAM (S3, `deep`) and resume:
the unlock is re-applied during the resume, before `PM: suspend exit`,
and the kernel log shows `gpu_unlock: unlocked again after resume` with `ret=0` for both passes.
The `nvidia-resume` systemd service runs only after this, and there were no NVRM errors or Xids after the resume.

This works with the package defaults of the driver: `UseKernelSuspendNotifiers: 0`
(see `/proc/driver/nvidia/params`) together with `NVreg_PreserveVideoMemoryAllocations=1`
and the `nvidia-suspend`/`nvidia-resume` systemd services.
Changing `NVreg_UseKernelSuspendNotifiers` is therefore not needed.

Check after a suspend:
```bash
journalctl -k -b | grep -E 'gpu_unlock|PM: suspend'
```

## Results

- [`../vllm-rtx3090-hyperqwen`](../vllm-rtx3090-hyperqwen/README.md#gpu-unlock) (Qwen3.8-27B W4A16, vLLM, 2026-10-31):
  prompt processing 40-52 % faster (1247 vs 822 t/s at 100k tokens of context), generation unchanged except
  +11-13 % past 100k tokens of context.

## Recovery

- To boot once without the unlock, press Shift or Esc during boot to show the GRUB menu (`GRUB_TIMEOUT_STYLE=hidden`),
  press `e` and remove `gpu_unlock.targets=all`.
- If the `nvidia` module in the initramfs breaks, add `modprobe.blacklist=nvidia` in the same way.
- `kernel.sysrq = 176` allows a safe reboot from a black screen with Alt+SysRq+S, U, B.
- To disable the unlock permanently, remove `gpu_unlock.targets=` from `/etc/default/grub` and run `sudo update-grub`.
