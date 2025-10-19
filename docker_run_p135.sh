#!/usr/bin/env bash
# Build and run RCKangaroo in Docker with NVIDIA GPU for Puzzle #135
# Prereqs on VM: Docker + NVIDIA Container Toolkit (nvidia-container-toolkit)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_TAG=${IMAGE_TAG:-rckangaroo:cuda12.4}
GPU_MASK=${GPU_MASK:-all}   # "all" or a specific device index list
WIN_BITS=${WIN_BITS:-40}
DP=${DP:-16}
DP_SLOTS=${DP_SLOTS:-8}
MAX=${MAX:-2.5}
CHECKPOINT_SECS=${CHECKPOINT_SECS:-300}
ORDER=${ORDER:-permute}
SEED=${SEED:-12345}
TAMES_FILE=${TAMES_FILE:-tames_p135_w${WIN_BITS}.dat}
LOG_DIR=${LOG_DIR:-logs135}

PUBKEY=02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
START_HEX=4000000000000000000000000000000000
END_HEX=7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

# 1) Build image
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  docker build -t "$IMAGE_TAG" "$SCRIPT_DIR"
fi

# 2) Ensure log dir exists on host
mkdir -p "$SCRIPT_DIR/$LOG_DIR"

# 3) Run scan inside container, mounting the repo so results persist
exec docker run --rm \
  --gpus $GPU_MASK \
  -v "$WS_DIR":/ws \
  -w /ws/RCKangaroo \
  -e PUBKEY_COMPRESSED="$PUBKEY" \
  -e START_HEX="$START_HEX" \
  -e END_HEX="$END_HEX" \
  -e WIN_BITS="$WIN_BITS" \
  -e DP="$DP" \
  -e DP_SLOTS="$DP_SLOTS" \
  -e MAX="$MAX" \
  -e CHECKPOINT_SECS="$CHECKPOINT_SECS" \
  -e ORDER="$ORDER" \
  -e SEED="$SEED" \
  -e TAMES_FILE="$TAMES_FILE" \
  -e LOG_DIR="$LOG_DIR" \
  "$IMAGE_TAG" \
  bash -lc "chmod +x scan_puzzle135.sh && ./scan_puzzle135.sh"
