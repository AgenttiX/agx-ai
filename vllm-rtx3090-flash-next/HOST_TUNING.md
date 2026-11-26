# Host settings for Flash-Next on agx-z2e: L3 cache allocation, 1 GiB huge pages and the CPU's power policy

These settings change the host and need root, so they are not applied automatically. The decode speed of
Qwen3.8-Flash-Next is bound by the CPU's expert reads from RAM (~60-70 % of a decode step, see
[README.md](README.md#cpu-kernel-and-prefetch-2026-11-24)). Each setting below protects that path from something that
can slow it down. They are not measured yet; expect small gains (estimates below), and keep a setting only if the
benchmark shows a gain beyond the noise (±1-4 %).

The CPU pool of the runtime (from [docker-compose.yml](docker-compose.yml)), on the Threadripper 3970X's 8 CCXs of 4
cores (each CCX shares 16 MB of L3 with 16 ways; CPU n and n + 32 are the SMT siblings of core n):

| CPUs per CCX k (k = 0..7) | Used by |
|---|---|
| 4k + 1 | a pool thread (`QWEN38_CPU_EXPERTS_CPUS`, computes the experts) |
| 4k + 2 | its loader thread (`QWEN38_CPU_EXPERTS_LOADER_CPUS`, reads the expert ahead into the L3) |
| 4k, 4k + 3 and their SMT siblings | the serving processes, the Radeon VII worker's thread, the desktop (`QWEN38_MAIN_CPUS`) |
| 4k + 33, 4k + 34 (SMT siblings of the pool and loader) | kept idle by the runtime; other programs may run there |

## Measuring

Measure each setting on its own against the same image, in alternating order and twice if the results are close
(stop the production container first, see [test_server.sh](test_server.sh)):

```sh
docker stop vllm-flash-next
./test_server.sh ghcr.io/agenttix/agx-ai/vllm-flash-next:latest
cd .. && uv run -m agx_ai.benchmark.bench_http --url http://localhost:9933 --env-file llama-cpp/llama-cpp.env \
  --no-embedding --tests chat depth edit --repeat 2 --depth 6400 32000 --edit-depth 0 --label "z2e Flash-Next <setting>"
docker rm -f fn-test && docker start vllm-flash-next
```

The edit test is the most stable (~98 % of the drafts accepted); the chat and depth tests also vary with the drafts'
acceptance. With `QWEN38_CPU_EXPERTS_STATS=200` as an extra argument of `test_server.sh`, the container log shows the
CPU's time per decode step directly (`docker logs fn-test 2>&1 | grep "decode stats"`).

## L3 cache allocation (resctrl CAT)

**Why:** the pool and loader threads read the experts of a layer into their CCX's L3 (the prefetch during the GPU's
~0.2 ms between the layers, and the loaders ahead of the pool threads), and the pool threads then read them from
there at 2-3 times the speed of the RAM. The other two cores of each CCX (the serving processes, the desktop, the
Radeon VII worker's polling thread) share the same 16 MB and can evict those lines. Cache allocation technology
(CAT; the CPU flags `cat_l3` and `rdt_a`, and the kernel's `CONFIG_X86_CPU_RESCTRL`, are present) restricts which
L3 ways the other cores can fill, so that most of each CCX's L3 is kept for the pool. Estimate: 0-5 % in decode,
more while other programs run on the computer.

On Zen 2, the L3 is a victim cache: a line enters the L3 when the core's L2 evicts it, into one of the ways that
the core's class of service allows. The classes are assigned per CPU here (the runtime pins its threads), so no
process IDs are needed.

### Apply (until the next reboot)

```sh
sudo mount -t resctrl resctrl /sys/fs/resctrl
cat /sys/fs/resctrl/info/L3/cbm_mask      # ffff: 16 ways per L3 (one L3 per CCX, ids 0-7)
cat /sys/fs/resctrl/schemata              # L3:0=ffff;1=ffff;...;7=ffff (the default group: all CPUs)
# The pool and loader cores: all 16 ways of their CCX's L3.
sudo mkdir /sys/fs/resctrl/qwen38-pool
echo 1-2,5-6,9-10,13-14,17-18,21-22,25-26,29-30 | sudo tee /sys/fs/resctrl/qwen38-pool/cpus_list
echo "L3:0=ffff;1=ffff;2=ffff;3=ffff;4=ffff;5=ffff;6=ffff;7=ffff" | sudo tee /sys/fs/resctrl/qwen38-pool/schemata
# Everything else (the default group): 4 of the 16 ways (4 MB per CCX).
echo "L3:0=000f;1=000f;2=000f;3=000f;4=000f;5=000f;6=000f;7=000f" | sudo tee /sys/fs/resctrl/schemata
cat /sys/fs/resctrl/qwen38-pool/schemata /sys/fs/resctrl/schemata
```

Variants worth measuring: the default group with 8 ways (`00ff`) if the serving processes or the desktop become
slower (the image test, `--tests image`, runs the vision tower on the serving processes' CPUs), and the pool with
only the other 12 ways (`fff0`, exclusive) instead of all 16.

`cpus_list` must not include the SMT siblings (33-34, 37-38, ...): other programs that run there would get the
pool's ways. If the CPU pool's settings in `docker-compose.yml` change, change `cpus_list` with them.

### Revert

```sh
echo "L3:0=ffff;1=ffff;2=ffff;3=ffff;4=ffff;5=ffff;6=ffff;7=ffff" | sudo tee /sys/fs/resctrl/schemata
sudo rmdir /sys/fs/resctrl/qwen38-pool
sudo umount /sys/fs/resctrl
```

### Keep it after reboots

If it helps, a systemd unit can apply it at boot, e.g. `/etc/systemd/system/qwen38-l3.service`:

```ini
[Unit]
Description=L3 cache allocation for the Flash-Next CPU expert pool
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'mountpoint -q /sys/fs/resctrl || mount -t resctrl resctrl /sys/fs/resctrl; \
  mkdir -p /sys/fs/resctrl/qwen38-pool; \
  echo 1-2,5-6,9-10,13-14,17-18,21-22,25-26,29-30 > /sys/fs/resctrl/qwen38-pool/cpus_list; \
  echo "L3:0=ffff;1=ffff;2=ffff;3=ffff;4=ffff;5=ffff;6=ffff;7=ffff" > /sys/fs/resctrl/qwen38-pool/schemata; \
  echo "L3:0=000f;1=000f;2=000f;3=000f;4=000f;5=000f;6=000f;7=000f" > /sys/fs/resctrl/schemata'
ExecStop=/bin/sh -c 'echo "L3:0=ffff;1=ffff;2=ffff;3=ffff;4=ffff;5=ffff;6=ffff;7=ffff" > /sys/fs/resctrl/schemata; \
  rmdir /sys/fs/resctrl/qwen38-pool'

[Install]
WantedBy=multi-user.target
```

```sh
sudo systemctl daemon-reload && sudo systemctl enable --now qwen38-l3.service
```

The restriction of the default group applies to every other program too, also when Flash-Next is not running;
`sudo systemctl stop qwen38-l3.service` gives the whole L3 back.

## 1 GiB huge pages for the expert arena

**Why:** the runtime keeps the 23424 cold experts in a 55.3 GiB arena in RAM and asks for transparent huge pages
(2 MiB). Right after a boot it gets them for the whole arena (57.9 GB of `AnonHugePages` now), but after days of
uptime the RAM is fragmented: on 2026-10-14 only 23.6 of 56 GiB were on huge pages. Then the CPU's random reads of
~2.5 MB per expert walk more page tables and miss the TLB more often. Reserved 1 GiB pages make the arena's pages
independent of the uptime. Gain over a fully covered arena: probably none; after long uptimes, an estimated 0-3 %.

The image (since 2026-11-26) takes the arena from 1 GiB pages with `QWEN38_EXPERT_STORE_HUGETLB=1G`
([`expert-store-hugetlb.patch`](expert-store-hugetlb.patch)). If there are not enough free 1 GiB pages, it logs
a warning and uses transparent huge pages as before. The arena needs 56 pages (55.3 GiB, rounded up); the
startup log shows the size (`Expert store: ... GiB`). With `QWEN38_HOT_ONLY` or the memory limit changed, check it.

**The cost:** reserved pages are taken from everything else, also while Flash-Next is not running: 56 of the 125.6
GiB. The other configurations of this computer that keep experts in RAM (e.g. [`../vllm-rtx3090`](../vllm-rtx3090),
[`../llama-cpp-rtx3090`](../llama-cpp-rtx3090)) then have ~56 GiB less. Reserving them only while Flash-Next runs
(option B) avoids this, but can fail when the RAM is fragmented, which is the case that the pages are for.

### Option A: reserve them at boot (reliable)

```sh
sudoedit /etc/default/grub   # append to GRUB_CMDLINE_LINUX_DEFAULT: hugepagesz=1G hugepages=56
sudo update-grub && sudo reboot
grep -E "HugePages_(Total|Free)|Hugepagesize" /proc/meminfo
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages   # 56
```

### Option B: reserve them before starting the server

```sh
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches && echo 1 | sudo tee /proc/sys/vm/compact_memory
echo 56 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages   # fewer than 56: not enough contiguous RAM
```

Give them back after stopping the server: `echo 0 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages`.

### Use them

In [docker-compose.yml](docker-compose.yml), add to the environment of `vllm-flash-next`:

```yaml
      QWEN38_EXPERT_STORE_HUGETLB: "1G"
```

and restart it (`docker compose up -d vllm-flash-next`). The log should show `Expert store: the arena is on 56
reserved 1 GiB huge pages`, and `HugePages_Free` in `/proc/meminfo` drops to 0. No volume or capability is needed:
the arena is an anonymous mapping, and the container's `memlock` limit is already unlimited. The pages are not
counted in the container's memory limit (`mem_limit`); the runtime sizes its arena from that limit, so leave it as is.

## The CPU's power policy

**Why:** the pool threads alternate between short bursts of work and spinning every ~0.2 ms, and each layer waits for
the slowest of the 8 threads. The host runs the `powersave` governor of `amd-pstate-epp` with the energy-performance
preference `balance_performance` (and the `balanced` platform profile). The preference `performance` keeps the cores
at their highest clock without ramping up. Estimate: 0-3 % in decode, at a higher idle power.

```sh
cat /sys/devices/system/cpu/cpu1/cpufreq/energy_performance_preference   # balance_performance
powerprofilesctl set performance        # sets the preference of all CPUs (power-profiles-daemon); back: balanced
# or only the pool's and loaders' CPUs, without power-profiles-daemon:
for c in 1 2 5 6 9 10 13 14 17 18 21 22 25 26 29 30; do
  echo performance | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/energy_performance_preference
done
```

The second variant is reset at reboot and by power-profiles-daemon when the profile changes.
