# AgenttiX AI configs

[Structure of the setup](structure/README.md) as an SVG figure

## Server software stack
- [GPT Researcher](https://docs.gptr.dev/)
- [LiteLLM](https://www.litellm.ai/)
  - Routes requests to multiple backends, including both cloud services and local models on multiple devices.
- [LiteLLM Claude](https://github.com/cabinlab/litellm-claude-code)
  - Use Claude Pro or Max subscription instead of an API key
- [Nginx](https://nginx.org/)
  - Reverse proxy (=HTTPS encryption etc.)
  - Using custom [nginx-server](https://github.com/AgenttiX/nginx-server) config
- [Open Terminal](https://github.com/open-webui/open-terminal)
  - Docker container for LLM terminal access
- [Open WebUI](https://openwebui.com/)
  - User interface for chats
  - Android client: [Conduit](https://conduit.mobile/)


## Backends
These are the configurations that run the local LLMs.

- [Benchmarks](benchmark/README.md): measure the served configuration over HTTP, or run `llama bench` in the container
- [llama.cpp](https://llama-cpp.com/)
  - [Shared server settings (config.ini)](llama-cpp/README.md)
  - [Native Windows config](llama-cpp-windows/README.md)
  - [AMD Radeon VII (16 GB): custom ROCm container](llama-cpp-radeon-vii/README.md)
  - [NVIDIA RTX A4000 (2x 16 GB): PaperQA2 RAG](llama-cpp-rtx-a4000/README.md)
  - [NVIDIA RTX 3070 (8 GB)](llama-cpp-rtx3070/README.md)
  - [NVIDIA RTX 3090 (24 GB)](llama-cpp-rtx3090/README.md)
  - [NVIDIA T550 Laptop GPU (4 GB) with for Docker Desktop on Windows](llama-cpp-t550/README.md)
  - [ThinkPad L14 Gen 5 (Intel Core Ultra 5 125U) iGPU](llama-cpp-l14/README.md)
  - [ThinkPad T480 (Intel Core i7-8550U, NVIDIA GeForce MX150 2 GB) CPU + dGPU config](llama-cpp-t480/README.md)
- [LM Studio](https://lmstudio.ai/)
  - Much easier to set up than llama.cpp or vLLM, but not as optimized.
    Depending on the hardware, you may get multiple times the performance (tokens/s)
    using llama.cpp or vLLM with proper configuration.
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
  - [ThinkPad L14 Gen 5 NPU (Intel Core Ultra 5 125U, Meteor Lake NPU 3720)](openvino-l14/README.md)
  - [ThinkPad L14 Gen 5 NPU: EmbeddingGemma 2 with OpenVINO](openvino-l14-npu-embeddinggemma2/README.md)
  - [ThinkPad L14 Gen 5 NPU: Gemma 4 E4B with a third-party server](openvino-l14-npu-e4b/README.md)
- [vLLM](https://vllm.ai/)
  - [AMD Radeon VII (16 GB): custom ROCm container](vllm-radeon-vii/README.md)
  - [NVIDIA RTX 3070 (8 GB)](vllm-rtx3070/README.md)  
  - [NVIDIA RTX 3090 (24 GB)](vllm-rtx3090/README.md)
  - [NVIDIA RTX 3090 (24 GB): HyperQwen (Qwen3.8-27B)](vllm-rtx3090-hyperqwen/README.md)
  - [NVIDIA RTX 3090 (24 GB): Qwen3.8-Flash-Next with custom container](vllm-rtx3090-flash-next)
  - [NVIDIA RTX A4000 (2x 16 GB): PaperQA2 RAG](vllm-rtx-a4000/README.md)


## Cloud providers
- [Claude](https://claude.ai/login)


## Other software
- [Claude Desktop for Linux](https://github.com/aaddrick/claude-desktop-debian)


## The attitudes this setup is built with
<a href="https://www.smbc-comics.com/comic/2011-02-17">
<img src="https://www.smbc-comics.com/comics/20110217.gif" width="500">
</a>
<br>
Image source:
<a href="https://www.smbc-comics.com/comic/2011-02-17">SMBC</a>
<br><br>
<a href="https://www.reddit.com/r/BrandNewSentence/comments/1wgya3y/we_made_runes_so_complex_that_sand_started/">
<img src="https://i.redd.it/r2ts99qzaoph1.jpeg" width="500">
</a>
<br>
Image source:
<a href="https://www.reddit.com/r/BrandNewSentence/comments/1wgya3y/we_made_runes_so_complex_that_sand_started/">Reddit</a>

These images reflect quite well the duality of emotions when starting a local LLM for the first time.
Witnessing your computer think in natural language can be both a humbling and a greatly rewarding experience,
and even more so than with the cloud, as you get to physically see the thing doing the thinking.
To be a bit poetic, these are both aliens, intricate clockwork, and children we are building in our own image.


## Notes
- [Don't oversaturate the CPU with too many threads](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md#verifying-that-the-cpu-is-not-oversaturated)


### Connecting LiteLLM to a remote LLM backend server over SSH
This can be done using the SSH sidecar container in [`docker-compose.yml`](./docker-compose.yml).
First, create `./ssh/config/private.conf` based on `./ssh/config/private-template.conf`,
and run `chown root:root ./ssh/config/private.conf`.
Start the SSH container with `cd ./ssh && sudo docker compose run --rm ssh-debug`.
Create an SSH key from within the container with `ssh-keygen -t ed25519 -C "YOUR_COMMENT"`.
Your public key is now in `./ssh/config/id_ed25519.pub`.
On the jump host, set in `~/.ssh/authorized_keys` to allow the SSH key to be used only for SSH JumpHost to the target server:
```
restrict,port-forwarding,command="/bin/false",permitlisten="127.0.0.1:1",permitopen="LLAMA_HOSTNAME:22" YOUR_PUBLIC_KEY
```
On the llama.cpp server, set in `~/.ssh/authorized_keys` to allow the SSH key to be used only for forwarding the LLM backend port:
```
restrict,port-forwarding,command="/bin/false",permitlisten="127.0.0.1:1",permitopen="127.0.0.1:9931" YOUR_PUBLIC_KEY
```
Either populate `./ssh/config/known_hosts` manually,
or disable `StrictHostKeyChecking yes` in `./ssh/config` and run
`ssh -N -F /ssh/config -L 0.0.0.0:9931:127.0.0.1:9931 remote`.
