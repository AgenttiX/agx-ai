# GPT Researcher MCP server (https://github.com/assafelovic/gptr-mcp)
#
# Debian instead of Alpine, as a dependency of gpt-researcher (unstructured -> numba -> llvmlite) has no musllinux wheels.
# The plain ghcr.io/astral-sh/uv image contains only the uv binary, so the debian-slim variant is used.
# The test dependencies of the upstream requirements.txt are not installed.
# google-genai is an optional dependency of gpt-researcher, needed for image generation with Google models.
FROM ghcr.io/astral-sh/uv:debian-slim

# curl is needed by the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON=3.13 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

ARG GPT_RESEARCHER_VERSION=0.16.1
RUN uv venv "$VIRTUAL_ENV" \
    && uv pip install \
        fastapi \
        fastmcp \
        google-genai \
        "gpt-researcher==${GPT_RESEARCHER_VERSION}" \
        loguru \
        python-dotenv \
    && python -c "import gpt_researcher, fastmcp, google.genai"

# Fix image generation never being enabled: Config stores the settings in lowercase, but the image generator reads them
# in uppercase. Remove once https://github.com/assafelovic/gpt-researcher/pull/2199 is released.
RUN f="$VIRTUAL_ENV/lib/python3.13/site-packages/gpt_researcher/skills/image_generator.py" \
    && grep -q "'IMAGE_GENERATION_ENABLED'" "$f" \
    && sed -i "s/getattr(self.cfg, 'IMAGE_GENERATION_\([A-Z_]*\)'/getattr(self.cfg, 'image_generation_\L\1'/" "$f" \
    && ! grep -q "'IMAGE_GENERATION_" "$f" \
    && python -m compileall -q "$f"

WORKDIR /app
ARG GPTR_MCP_COMMIT=63884773685b1f12c7f0d9e283b3d71a5b9b5fda
ADD https://raw.githubusercontent.com/assafelovic/gptr-mcp/${GPTR_MCP_COMMIT}/server.py \
    https://raw.githubusercontent.com/assafelovic/gptr-mcp/${GPTR_MCP_COMMIT}/utils.py \
    /app/

ENV MCP_TRANSPORT=sse \
    DOCKER_CONTAINER=true \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=7s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "server.py"]
