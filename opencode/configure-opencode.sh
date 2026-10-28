#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -eq 0 ]; then
  echo "This script should not be run as root."
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

CONFIG_DIR="${HOME}/.config/opencode"
CONFIG_PATH="${CONFIG_DIR}/opencode.jsonc"
OPENCODE_CLI_PATH="/opt/OpenCode/resources/opencode-cli"

if [ ! -d "${CONFIG_DIR}" ]; then
  echo "Config directory ${CONFIG_DIR} does not exist. Creating."
  mkdir -p "${CONFIG_DIR}"
fi

if [ -f "${CONFIG_PATH}" ] && [ ! -L "${CONFIG_PATH}" ]; then
  echo "Config file already exists. Backing up."
  mv "${CONFIG_PATH}" "${CONFIG_DIR}/opencode-old.jsonc"
fi

if [ -f "${OPENCODE_CLI_PATH}" ]; then
  echo "Symlinking opencode-cli from OpenCode Desktop as opencode."
  ln -f -s "${OPENCODE_CLI_PATH}" "${HOME}/.local/bin/opencode"
else
  echo "This computer doesn't have OpenCode Desktop installed. Skipping its opencode-cli symlink creation."
fi

echo "Creating symlinks."
ln -f -s "${SCRIPT_DIR}/opencode.jsonc" "${CONFIG_PATH}"
echo "Configuration ready."

echo "Removing old models from OpenCode cache."
rm -f ~/.cache/opencode-litellm/models-*.json

echo "Restarting OpenCode service."
opencode service restart
