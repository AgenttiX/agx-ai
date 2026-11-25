# syntax=docker/dockerfile:1
# check=error=true
# https://docs.docker.com/build/checks/#fail-build-on-check-violations

# Converters that prepare models for OpenVINO Model Server (OVMS) on the Intel NPU (../agx_ai/openvino).
# The converter and its arguments are given in the docker-compose.yml files of the configurations, see README.md.
# The build context is the root of the repository, and Dockerfile.dockerignore limits it to the needed files:
# docker build -f openvino-npu-convert/Dockerfile -t ghcr.io/agenttix/agx-ai/openvino-npu-convert:latest .

# Ubuntu 26.04 includes Python 3.14.
ARG UBUNTU_IMAGE="ubuntu:26.04"

# The uv image is used for copying the uv binary to the builder stage.
# https://docs.astral.sh/uv/guides/integration/docker/
ARG UV_IMAGE="ghcr.io/astral-sh/uv:0.13"

# The Python libraries are installed in a virtual environment,
# as the system Python of Ubuntu is externally managed (PEP 668).
ARG VENV_PATH="/opt/venv"


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
# The PyTorch CPU index is restricted to torch in pyproject.toml.
# The OpenVINO version of the converter should match the OVMS images (see pyproject.toml).
RUN \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked,id=uv-${TARGETARCH} \
    --mount=type=bind,source=openvino-npu-convert/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=openvino-npu-convert/uv.lock,target=uv.lock \
    uv sync --locked --no-install-project


# ---
# Final stage: contains only Python, the virtual environment and the converters
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
# The converters are run as modules of agx_ai, e.g. "python -m agx_ai.openvino.convert_gemma4".
ENV PYTHONPATH=/app

# Python has to be installed, as the virtual environment links to the system interpreter.
# ca-certificates is needed for downloading the model files, as huggingface_hub uses the system certificates.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3

COPY --from=builder ${VENV_PATH} ${VENV_PATH}
# Only the converters of agx_ai are copied, as they do not depend on the rest of it.
COPY agx_ai/__init__.py /app/agx_ai/__init__.py
COPY agx_ai/openvino/*.py /app/agx_ai/openvino/
# A new named volume gets the permissions of this directory, which is writable by any user (like /tmp),
# so that the user can be set in docker-compose.yml, and OVMS can write its compilation cache to the same volume.
RUN mkdir -m 1777 /models

# The converter and its arguments are given as the command in the docker-compose.yml files.
CMD ["python", "-m", "agx_ai.openvino.convert_gemma4", "--help"]
