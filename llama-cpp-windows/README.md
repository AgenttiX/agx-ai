# llama.cpp natively on Windows

Generic scripts for running the llama.cpp router server (`llama-server.exe`) natively on Windows, without Docker and
WSL 2, with the same `config.ini`/preset files as the Docker configurations. The configuration of each computer is in
its own directory, e.g. [`../llama-cpp-t550/windows`](../llama-cpp-t550/windows).

- [`install-llama-cpp.ps1`](install-llama-cpp.ps1): downloads an official release from
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases), verifies the SHA-256 digests that GitHub
  publishes, and installs it to `%LOCALAPPDATA%\llama.cpp\<build>-<backend>`. The previous versions are kept, and
  `%LOCALAPPDATA%\llama.cpp\current.txt` names the one to run.
- [`start-llama-server.ps1`](start-llama-server.ps1): starts the router with a configuration directory.
- [`register-autostart.ps1`](register-autostart.ps1): a scheduled task that starts a computer's server at logon.

No administrator rights are needed. CUDA builds need only the NVIDIA driver: the release includes the CUDA runtime
and cuBLAS DLLs (the separate `cudart-llama-bin-win-cuda-*.zip`), so the CUDA toolkit is not needed.

## Usage

```powershell
.\install-llama-cpp.ps1                           # newest release with a CUDA 13.4 build
.\install-llama-cpp.ps1 -Backend cuda-12.4        # for drivers older than CUDA 13.4 (see below)
.\install-llama-cpp.ps1 -Build b11262 -Backend vulkan
..\llama-cpp-t550\windows\start.ps1               # a computer's start script, runs in the foreground
.\register-autostart.ps1 -StartScript ..\llama-cpp-t550\windows\start.ps1
```

`install-llama-cpp.ps1` ends by running `llama-server.exe --version` and `--list-devices`, which shows whether
Windows allows running it (application control policies may block downloaded executables) and which GPUs it sees.
To go back to an earlier installed version, write its directory name into `current.txt`.

## Configuration directory

The same files as for Docker, see e.g. [`../llama-cpp-t550/windows`](../llama-cpp-t550/windows):

- `config.ini`: the `[*]` section with the server-wide settings. In the Docker image, llama-server reads
  `/etc/llama.cpp/config.ini` by itself, but it has no option for a config file, so on Windows
  `start-llama-server.ps1` passes the settings as arguments (an option is a flag if `llama-server --help` lists it
  without an argument, e.g. `kv-unified = 1` -> `--kv-unified`). A relative `models-preset` path is relative to the
  configuration directory, so the Windows and Docker configurations of a computer can share one preset file.
- An env file with `KEY=VALUE` lines, e.g. `LLAMA_API_KEY` (not committed). The script sets them as environment
  variables of the server and never prints them.

## Notes

- **Model cache:** the models are downloaded to `%LOCALAPPDATA%\llama.cpp\models` (`HF_HUB_CACHE`/`LLAMA_CACHE`), not
  to `%USERPROFILE%\.cache\huggingface\hub`. When Docker containers have used that one, its snapshot entries are Linux
  symbolic links that Windows programs cannot open. Without administrator rights or Developer Mode, Windows does not
  allow llama.cpp to create symbolic links either (`finalize_file: failed to create symlink: A required privilege is
  not held by the client`); llama.cpp then stores the files directly in the snapshot directory, which works.
- **CUDA version:** a build for a newer CUDA version than the driver supports ("CUDA Version" in `nvidia-smi`) may
  still start, as CUDA allows newer minor versions, but fail when the GPU needs its kernels compiled from PTX:
  `CUDA error: the provided PTX was compiled with an unsupported toolchain`. The Windows release builds do not include
  native kernels for all GPUs (e.g. not for Turing on 2026-09-30), unlike the Docker images. Unlike in Docker
  (`NVIDIA_DISABLE_REQUIRE`), this cannot be overridden: use a build for an older CUDA version (e.g. `cuda-12.4`) or
  update the driver.
- **Network:** with `host = 127.0.0.1`, only local programs can connect. With `0.0.0.0`, Windows asks on the first
  start whether to allow `llama-server.exe` through the firewall.
- **Old versions:** GitHub keeps only the most recent build releases, so a build that is used for benchmarks should
  be kept installed.
