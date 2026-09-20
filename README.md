# AgenttiX AI server configs

## Software stack
- [GPT Researcher](https://docs.gptr.dev/)
- [LiteLLM](https://www.litellm.ai/)
  - Routes requests to multiple backends, including both cloud services and local models on multiple devices.
- [LiteLLM Claude](https://github.com/cabinlab/litellm-claude-code)
  - Use Claude Pro or Max subscription instead of an API key
- [llama.cpp](https://llama-cpp.com/)
  - For running local models
  - [Custom container for Radeon VII with ROCm](llama-cpp-radeon-vii/docker-compose.yml)
  - [Dual RTX A4000 config for PaperQA2 RAG](llama-cpp-big-machine/docker-compose.yml)
- [LM Studio](https://lmstudio.ai/)
  - For running local models
- [Open Terminal](https://github.com/open-webui/open-terminal)
  - Docker container for LLM terminal access
- [Open WebUI](https://openwebui.com/)
  - User interface for chats
  - Android client: [Conduit](https://conduit.mobile/)


## Cloud providers
- [Claude](https://claude.ai/login)


## Other software
- [Claude Desktop for Linux](https://github.com/aaddrick/claude-desktop-debian)


## Notes
- [Don't oversaturate the CPU with too many threads](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md#verifying-that-the-cpu-is-not-oversaturated)


### Connecting LiteLLM to a remote llama.cpp server over SSH
This can be done using the SSH sidecar container in [`docker-compose.yml`](./docker-compose.yml).
First, create `./ssh/config/private.conf` based on `./ssh/config/private-template.conf`,
and run `chown root:root ./ssh/config/private.conf`.
Start the SSH container with `cd ./ssh && sudo docker compose run --rm ssh-debug`.
Create an SSH key from within the container with `ssh-keygen -t ed25519 -C "YOUR_COMMENT"`.
Your public key is now in `./ssh/config/id_ed25519.pub`.
On the jump host, set in `~/.ssh/authorized_keys`:
```
restrict,port-forwarding,command="/bin/false",permitlisten="127.0.0.1:1",permitopen="LLAMA_HOSTNAME:22" YOUR_PUBLIC_KEY
```
On the llama.cpp server, set in `~/.ssh/authorized_keys`:
```
restrict,port-forwarding,command="/bin/false",permitlisten="127.0.0.1:1",permitopen="127.0.0.1:9931" YOUR_PUBLIC_KEY
```
Either populate `./ssh/config/known_hosts` manually,
or disable `StrictHostKeyChecking yes` in `./ssh/config` and run
`ssh -N -F /ssh/config -L 0.0.0.0:9931:127.0.0.1:9931 remote`.
