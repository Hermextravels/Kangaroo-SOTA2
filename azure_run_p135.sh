#!/usr/bin/env bash
# Quick launcher for Puzzle #135 on Azure NVads A10 v5
# - Builds RCKangaroo if needed
# - Runs scan_puzzle135.sh with tuned defaults for A10
# - Logs to logs135 and supports resume
set -euo pipefail
cd "$(dirname "$0")"

# 1) Build
if [ ! -x ./rckangaroo ]; then
  echo "Building rckangaroo..."
  make -j || { echo "Build failed. Ensure CUDA is installed and CUDA_PATH points to /usr/local/cuda"; exit 2; }
fi

# 2) Defaults for Puzzle #135
export PUBKEY_COMPRESSED="02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
export START_HEX="4000000000000000000000000000000000"
export END_HEX="7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
# A10: start fairly low DP and moderate slots, auto-tune on clamping
export WIN_BITS=${WIN_BITS:-40}
export DP=${DP:-16}
export DP_SLOTS=${DP_SLOTS:-8}
export MAX=${MAX:-2.5}
export GPU_MASK=${GPU_MASK:-0} # change to 0,1 if multiple GPUs are present
export CHECKPOINT_SECS=${CHECKPOINT_SECS:-300}
export TAMES_FILE=${TAMES_FILE:-tames_p135_w${WIN_BITS}.dat}
export VERBOSE=${VERBOSE:-1}
export ORDER=${ORDER:-permute}
export SEED=${SEED:-12345}
export SKIP_VISITED=${SKIP_VISITED:-1}

# 3) Run
./scan_puzzle135.sh
