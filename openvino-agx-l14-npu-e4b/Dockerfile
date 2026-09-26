# syntax=docker/dockerfile:1
# check=error=true
# https://docs.docker.com/build/checks/#fail-build-on-check-violations

# Container for ryugyosoft/gemma-4-E4B-it-npu (Gemma 4 E4B with every model component on the Intel NPU),
# so that the third-party code never runs on the host. See README.md for the usage and the review notes.
# Only x86-64 is supported, as the Intel NPU driver is available only for it.

# Ubuntu 26.04 includes Python 3.14.
ARG UBUNTU_IMAGE="ubuntu:26.04"

# The uv image is used for copying the uv binary to the other stages.
# https://docs.astral.sh/uv/guides/integration/docker/
ARG UV_IMAGE="ghcr.io/astral-sh/uv:0.12"

# The Python libraries are installed in a virtual environment,
# as the system Python of Ubuntu is externally managed (PEP 668).
ARG VENV_PATH="/opt/venv"

# The Intel NPU user-mode driver from Intel's official GitHub releases.
# The kernel driver (intel_vpu) and the NPU firmware are on the host.
ARG NPU_DRIVER_URL="https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2604.tar.gz"
ARG NPU_DRIVER_SHA256="83685e6d92f4db1ca65e442846d487c7feb1335634b33994292ceb38f933bcc7"

# The server code at the reviewed commit. The checksums are those of the files that were reviewed on 2026-09-26.
ARG REPO_URL="https://huggingface.co/ryugyosoft/gemma-4-E4B-it-npu/raw/e120bd9ca8cdee3bd9cbfd42a5b6045aef69657c"


# ---
# uv stage: provides the uv binary
# ---
FROM ${UV_IMAGE} AS uv


# ---
# Download stage: fetches the NPU driver and the server code, and verifies their checksums
# ---
FROM ${UBUNTU_IMAGE} AS download

ARG TARGETARCH
ARG NPU_DRIVER_URL
ARG NPU_DRIVER_SHA256
ARG REPO_URL

# The automatic cleanup is disabled to preserve the apt caches.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl

WORKDIR /download
RUN curl -fsSL -o npu.tgz "${NPU_DRIVER_URL}" \
    && echo "${NPU_DRIVER_SHA256}  npu.tgz" | sha256sum -c - \
    && mkdir npu-driver \
    && tar xzf npu.tgz -C npu-driver

WORKDIR /download/app
RUN mkdir web \
    && for f in server.py gemma4_npu.py static_lm.py npu_worker.py npu_fixes.py web/index.html; do \
        curl -fsSL -o "${f}" "${REPO_URL}/${f}" || exit 1; done \
    && printf '%s\n' \
        "8758ea888b4fcb3ae0ea72541884114e6dc82bbcca0c82cabbab1533a3b72c77  server.py" \
        "d8c4b6fb6613c7e8c135fc6b9bb7e9c05de8cf7b533403874b91d21f2db94182  gemma4_npu.py" \
        "31ae03041c72ea069a9473b2ca1a770a676aa721eb374fd282a12077e36ee166  static_lm.py" \
        "4fe4ef2ca4d594e168604b9f8be5aeea60a2adb6bbf87285ee415f25080c6999  npu_worker.py" \
        "d2b0da15fe59eb3da977c83d212bc3769a45b18e2356404d12b2b62118318bce  npu_fixes.py" \
        "08816248b8aea3447032d48347a3cd33e66f46fa14d161d36af836885e7c6b69  web/index.html" \
        | sha256sum -c -


# ---
# Builder stage: installs the Python dependencies in a virtual environment
# ---
FROM ${UBUNTU_IMAGE} AS builder

# The architecture is used for separating the build caches of the different platforms.
ARG TARGETARCH
ARG VENV_PATH

# Setting these environment variables is equivalent to activating the virtual environment.
ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"
# Make uv use the virtual environment above instead of creating .venv in the project directory.
ENV UV_PROJECT_ENVIRONMENT="${VENV_PATH}"
# Use the system Python instead of downloading a uv-managed Python,
# as the final image contains only the system Python.
ENV UV_PYTHON_DOWNLOADS=never
# Copy the packages from the cache mount instead of hardlinking, as the cache is on a different filesystem.
ENV UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/

RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3

# The dependencies are installed from the lock file.
# The extra package indexes are restricted to their packages in pyproject.toml (see the comments there).
RUN \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked,id=uv-${TARGETARCH} \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project


# ---
# Final stage: contains only Python, the NPU driver, the virtual environment and the server code
# ---
FROM ${UBUNTU_IMAGE}

ARG TARGETARCH
ARG VENV_PATH

ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
# The container is run as the host user, which has no home directory in the image.
ENV HOME=/tmp

# Python has to be installed, as the virtual environment links to the system interpreter.
# libze1 is the Level Zero loader, and libtbb12, libzstd1 and zlib1g are dependencies of the NPU compiler.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    --mount=type=bind,from=download,source=/download/npu-driver,target=/tmp/npu-driver \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends libtbb12 libze1 libzstd1 python3 zlib1g \
    && dpkg -i /tmp/npu-driver/intel-level-zero-npu_*.deb /tmp/npu-driver/intel-driver-compiler-npu_*.deb

COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=download /download/app /app

WORKDIR /app
EXPOSE 8000
ENTRYPOINT ["python", "/app/server.py", "--host", "0.0.0.0", "--port", "8000"]
