# Shared llama.cpp settings

Files used by the llama.cpp configurations of this repository (and, for the API key, by the vLLM configurations):

- [`config.ini`](config.ini): the server-wide settings of the llama.cpp router (`--config`), mounted as
  `/etc/llama.cpp/config.ini` by all configurations except [`../llama-cpp-rtx3070`](../llama-cpp-rtx3070) and
  [`../llama-cpp-rtx-a4000`](../llama-cpp-rtx-a4000), which has its own. One model is loaded at a time
  (`models-max = 1`).
- [`config-dual.ini`](config-dual.ini): the same with two models loaded at a time (`models-max = 2`), for
  [`../llama-cpp-rtx3070`](../llama-cpp-rtx3070), which keeps its GPU and CPU models loaded. The models and their settings
  are in the `preset.ini` files of the configurations. The native Windows configuration of
  [`../llama-cpp-t550`](../llama-cpp-t550/README.md) has its own copy with the host and paths for Windows.
- `llama-cpp.env` (not in Git): the API key of the servers of this computer, `LLAMA_API_KEY=...`, read by the
  llama.cpp and vLLM configurations and by the benchmarks (`--env-file ../llama-cpp/llama-cpp.env`). Create it once
  on each computer:

  ```sh
  echo "LLAMA_API_KEY=$(openssl rand -base64 32)" > llama-cpp.env
  ```
