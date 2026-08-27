# AgenttiX AI server configs

## Software stack
- [GPT Researcher](https://docs.gptr.dev/)
- [LiteLLM](https://www.litellm.ai/)
  - Routes requests to multiple backends, including both cloud services and local models on multiple devices.
- [LiteLLM Claude](https://github.com/cabinlab/litellm-claude-code)
  - Use Claude Pro or Max subscription instead of an API key
- [llama.cpp](https://llama-cpp.com/)
  - For running local models
  - [Custom container for Radeon VII with ROCm](./llama-cpp/docker-compose.yml)
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
