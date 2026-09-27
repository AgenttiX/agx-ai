#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "This script should be run as root."
  exit 1
fi

ufw allow 9931 comment "llama.cpp"
ufw enable
