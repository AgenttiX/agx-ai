#!/bin/sh
# Packs the experts into the cache (once per HOT and N) and runs the worker. See README.md.
#   RVII_HOT        experts per layer that the RTX 3090 keeps (the worker takes the next ranks), default 24
#   RVII_PER_LAYER  experts per layer in the Radeon VII's VRAM, default: as many as fit (RVII_VRAM_MIB, if set,
#                   else the free VRAM minus RVII_RESERVE_MIB, default 640)
#   RVII_CACHE      the cache directory, default /root/.cache/vllm-models/flash-next-rvii
set -e
HOT="${RVII_HOT:-24}"
CACHE="${RVII_CACHE:-/root/.cache/vllm-models/flash-next-rvii}"
N="${RVII_PER_LAYER:-}"
EXPERT_BYTES=2534400
if [ -z "$N" ]; then
  if [ -n "$RVII_VRAM_MIB" ]; then
    N=$((RVII_VRAM_MIB * 1048576 / (48 * EXPERT_BYTES)))
  else
    N=$(rvii_worker --max-per-layer)
  fi
fi
RANKINGS="${RVII_RANKINGS:-/opt/rvii/rankings.json}"
# The cache of a ranking file is keyed by its hash, as in prepare.py.
DIR="$CACHE/hot$HOT-n$N-$(sha256sum "$RANKINGS" | cut -c1-8)"
if [ ! -f "$DIR/meta.json" ]; then
  python /opt/rvii/prepare.py --hot "$HOT" --per-layer "$N" --out "$CACHE"
fi
exec rvii_worker "$DIR"
