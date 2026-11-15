#!/usr/bin/env bash
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
CLAUDE_SETTINGS="${CLAUDE_DIR}/settings.json"
CLAUDE_SKILLS="${CLAUDE_DIR}/skills"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SKILLS_DIR="$(dirname "${SCRIPT_DIR}")/agents/skills"

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

if [ -L "${CLAUDE_SKILLS}" ]; then
  echo "Claude skills directory is already a symlink. Skipping symlink creation."
else
  if [ -d "${CLAUDE_SKILLS}" ]; then
    # Keep the existing skills, but don't overwrite the ones in the repository.
    cp -a -n "${CLAUDE_SKILLS}/." "${SKILLS_DIR}/"
    mv "${CLAUDE_SKILLS}" "${CLAUDE_DIR}/skills-old"
  fi
  ln -s "${SKILLS_DIR}" "${CLAUDE_SKILLS}"
fi

# set +e
# https://dash.plotly.com/mcp
# claude mcp add plotly-dash-documentation --transport http --scope user https://dash.plotly.com/_mcp

echo "Installing language servers (LSP)"
uv tool install --upgrade pyright
if command -v npm &>/dev/null; then
  npm install -g typescript-language-server typescript
fi

echo "Claude configured."
