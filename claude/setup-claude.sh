#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
CLAUDE_SETTINGS="${CLAUDE_DIR}/settings.json"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AGENTS_DIR="$(dirname "${SCRIPT_DIR}")/agents"

if [ ! -d "${CLAUDE_DIR}" ]; then
  echo "Claude directory does not exist. Install Claude first."
  exit 1
fi

if [ -L "${CLAUDE_SETTINGS}" ]; then
  echo "Claude config is already a symlink. Skipping symlink creation."
else
  if [ -f "${CLAUDE_SETTINGS}" ]; then
    mv "${CLAUDE_SETTINGS}" "${CLAUDE_DIR}/settings-old.json"
  fi
  ln -s "${SCRIPT_DIR}/settings.json" "${CLAUDE_SETTINGS}"
fi

# Link the skills and sub-agents to the repository so that they are available globally.
for NAME in skills agents; do
  CLAUDE_SUBDIR="${CLAUDE_DIR}/${NAME}"
  if [ -L "${CLAUDE_SUBDIR}" ]; then
    echo "Claude ${NAME} directory is already a symlink. Skipping symlink creation."
  else
    if [ -d "${CLAUDE_SUBDIR}" ]; then
      # Keep the existing files, but don't overwrite the ones in the repository.
      cp -a -n "${CLAUDE_SUBDIR}/." "${AGENTS_DIR}/${NAME}/"
      mv "${CLAUDE_SUBDIR}" "${CLAUDE_DIR}/${NAME}-old"
    fi
    ln -s "${AGENTS_DIR}/${NAME}" "${CLAUDE_SUBDIR}"
  fi
done

# set +e
# https://dash.plotly.com/mcp
# claude mcp add plotly-dash-documentation --transport http --scope user https://dash.plotly.com/_mcp

echo "Installing language servers (LSP)"
uv tool install --upgrade pyright
if command -v npm &>/dev/null; then
  npm install -g typescript-language-server typescript
fi

echo "Claude configured."
