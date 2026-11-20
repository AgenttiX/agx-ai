# llama.cpp with DiffusionGemma support for NVIDIA GPUs (CUDA), see README.md.
# DiffusionGemma is not in upstream llama.cpp yet. This builds the OpenAI-compatible llama-diffusion-gemma-server of
# the draft pull request https://github.com/ggml-org/llama.cpp/pull/24427 at a fixed commit, with the
# llama-server-compatible /completion, /tokenize and /detokenize endpoints of server-endpoints.patch for the
# benchmarks of ../agx_ai/benchmark.
# Built by ../.github/workflows/deploy-docker-llama-cpp-diffusion-gemma.yml.
# Locally: docker build -t ghcr.io/agenttix/agx-ai/llama-cpp-diffusion-gemma:latest .
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# The head of the pull request on 2026-11-20 (last pushed 2026-06-24).
ARG LLAMA_CPP_REPO=lnigam/llama.cpp
ARG LLAMA_CPP_COMMIT=dd0cf04459b0c4f43aa6667dbc0879ac0cd50323
# Ampere (RTX 30x0, RTX A4000), Ada and Blackwell consumer GPUs.
ARG CUDA_ARCHITECTURES="86;89;120"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN curl -fsSL "https://codeload.github.com/${LLAMA_CPP_REPO}/tar.gz/${LLAMA_CPP_COMMIT}" | tar xz --strip-components=1
COPY server-endpoints.patch /tmp/
RUN patch -p1 < /tmp/server-endpoints.patch

RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" \
        --target llama-diffusion-gemma-server llama-diffusion-gemma-cli \
    && mkdir -p /app \
    && cp build/bin/llama-diffusion-gemma-server build/bin/llama-diffusion-gemma-cli /app/ \
    && find build -name "*.so*" -exec cp -P {} /app/ \;

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
ENV LD_LIBRARY_PATH=/app
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-diffusion-gemma-server" ]
