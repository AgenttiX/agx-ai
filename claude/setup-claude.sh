#!/usr/bin/env sh
set -eu

CLAUDE_DIR="${HOME}/.claude"
CLAUDE_SETTINGS="${CLAUDE_DIR}/settings.json"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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
