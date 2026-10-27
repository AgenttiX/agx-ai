# syntax=docker/dockerfile:1
# check=error=true
# https://docs.docker.com/build/checks/#fail-build-on-check-violations

# Converter that exports the text model of google/embeddinggemma-2 to OpenVINO IR for OpenVINO Model Server.
# See README.md for the usage and convert.py for the details.

# Ubuntu 26.04 includes Python 3.14.
ARG UBUNTU_IMAGE="ubuntu:26.04"

# The uv image is used for copying the uv binary to the builder stage.
# https://docs.astral.sh/uv/guides/integration/docker/
ARG UV_IMAGE="ghcr.io/astral-sh/uv:0.13"

# The Python libraries are installed in a virtual environment,
# as the system Python of Ubuntu is externally managed (PEP 668).
ARG VENV_PATH="/opt/venv"

# The converted commit of google/embeddinggemma-2.
ARG MODEL_REVISION="914f7f89142e33e77833254d9c9b90c3cef7303b"


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
# The OpenVINO version of the converter has to match the OVMS image in docker-compose.yml (see pyproject.toml).
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
ARG MODEL_REVISION

ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
# The container is run as the host user, which has no home directory in the image.
ENV HOME=/tmp
ENV MODEL_REVISION="${MODEL_REVISION}"

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

# The shell form expands MODEL_REVISION, and exec replaces the shell with Python.
ENTRYPOINT ["sh", "-c", "exec python /app/convert.py /models/embeddinggemma-2 --revision \"${MODEL_REVISION}\""]
