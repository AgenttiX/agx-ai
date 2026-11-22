# llama.cpp server for NVIDIA GPUs (CUDA) with the DSpark block drafter of Inkling 2 Mini, see README.md.
# Upstream llama.cpp supports Inkling 2 Mini since b12611, but its DSpark drafter (--spec-type draft-dspark, 1.2-1.7
# times as fast) is only in the open pull request https://github.com/ggml-org/llama.cpp/pull/32218. This builds release
# b12627 with the commit of that pull request (dspark.patch, which applies to b12627 unchanged), like upstream's
# .devops/cuda.Dockerfile, but only llama-server, without the embedded web UI (config.ini has no-ui) and for the
# consumer GPUs of these machines. Drop it for the upstream server-cuda13 image once the pull request is merged.
# Built by ../.github/workflows/deploy-docker-llama-cpp-inkling.yml.
# Locally: docker build -t ghcr.io/agenttix/agx-ai/llama-cpp-inkling:latest .
ARG CUDA_VERSION=13.5.0
ARG UBUNTU_VERSION=24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# b12627 (2026-11-22), the first release image after the Inkling 2 Mini fix ggml-org/llama.cpp#32704.
ARG LLAMA_CPP_TAG=b12627
ARG LLAMA_CPP_COMMIT=225a5345bbb0e0c70859af5441ba922beb666fcf
# Ampere (RTX 30x0, RTX A4000), Ada and Blackwell consumer GPUs.
ARG CUDA_ARCHITECTURES="86;89;120"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates libssl-dev patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN curl -fsSL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${LLAMA_CPP_COMMIT}" | tar xz --strip-components=1
COPY dspark.patch /tmp/
RUN patch -p1 < /tmp/dspark.patch

RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_NUMBER="${LLAMA_CPP_TAG#b}" -DLLAMA_BUILD_COMMIT="${LLAMA_CPP_COMMIT}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server \
    && mkdir -p /app \
    && cp build/bin/llama-server /app/ \
    && find build -name "*.so*" -exec cp -P {} /app/ \;

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ffmpeg decodes videos (video_url parts), as in upstream's images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/app
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]
