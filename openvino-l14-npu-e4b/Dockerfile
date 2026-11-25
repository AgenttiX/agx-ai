# syntax=docker/dockerfile:1
# check=error=true
# https://docs.docker.com/build/checks/#fail-build-on-check-violations

# Converter that prepares Gemma 4 E4B with symmetric INT4 weights for OpenVINO Model Server on the Intel NPU.
# See README.md for the usage and convert.py for the details.

# Ubuntu 26.04 includes Python 3.14.
ARG UBUNTU_IMAGE="ubuntu:26.04"

# The uv image is used for copying the uv binary to the builder stage.
# https://docs.astral.sh/uv/guides/integration/docker/
ARG UV_IMAGE="ghcr.io/astral-sh/uv:0.13"

# The Python libraries are installed in a virtual environment,
# as the system Python of Ubuntu is externally managed (PEP 668).
ARG VENV_PATH="/opt/venv"

# The converted commits of OpenVINO/gemma-4-E4B-it-int4-ov and OpenVINO/gemma-4-E4B-it-fp16-ov.
ARG INT4_REVISION="c157bc8d31bae5041874d4378667bc19ae0ee5c6"
ARG FP16_REVISION="2331737b18db63cb3c7141b8996139834fcde055"


# ---
# uv stage: provides the uv binary
# ---
FROM ${UV_IMAGE} AS uv


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

# The automatic cleanup is disabled to preserve the apt caches.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3

# The dependencies are installed from the lock file.
RUN \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked,id=uv-${TARGETARCH} \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project


# ---
# Final stage: contains only Python, the virtual environment and the converter
# ---
FROM ${UBUNTU_IMAGE}

ARG TARGETARCH
ARG VENV_PATH
ARG INT4_REVISION
ARG FP16_REVISION

ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
# The container is run as the host user, which has no home directory in the image.
ENV HOME=/tmp
ENV INT4_REVISION="${INT4_REVISION}"
ENV FP16_REVISION="${FP16_REVISION}"

# Python has to be installed, as the virtual environment links to the system interpreter.
# ca-certificates is needed for downloading the model files, as huggingface_hub uses the system certificates.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3

COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY convert.py /app/convert.py
# A new named volume gets the permissions of this directory, which is writable by any user (like /tmp),
# so that the user can be set in docker-compose.yml, and OVMS can write its compilation cache to the same volume.
RUN mkdir -m 1777 /models

# The shell form expands the revisions, and exec replaces the shell with Python.
ENTRYPOINT ["sh", "-c", "exec python /app/convert.py /models/gemma-4-E4B-it-int4-sym-ov --int4-revision \"${INT4_REVISION}\" --fp16-revision \"${FP16_REVISION}\""]
