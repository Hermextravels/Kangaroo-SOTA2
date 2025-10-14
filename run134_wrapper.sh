#!/usr/bin/env bash
# Wrapper to accept your original one-liner flags and run a windowed scan safely
# Usage example (same shape as your command):
#   ./run134_wrapper.sh -pubkey <hex> -start 4000...000 -range 134 -dp 25 -gpu 0 -resume -checkpoint-secs 600
# Under the hood, this switches to randomized 44-bit windows with tuned DP/slots and time-based per-window MAX.
set -euo pipefail

# Defaults
PUBKEY=""
START=""
RANGE=134
DP=25
GPU=""
RESUME=0
CHECKPOINT=600
# Windowed defaults
WIN_BITS=${WIN_BITS:-44}
TARGET_SEC=${TARGET_SEC:-900}
THROUGHPUT_GS=${THROUGHPUT_GS:-1.0}
DP_AUTO=${DP_AUTO:-1}
DP_SLOTS=${DP_SLOTS:-12}
N=${N:-1000}
RCK_BIN=${RCK_BIN:-"./rckangaroo"}

# Parse minimal flags
while [[ $# -gt 0 ]]; do
  k="$1"; shift
  case "$k" in
    -pubkey) PUBKEY="$1"; shift;;
    -start)  START="$1"; shift;;
    -range)  RANGE="$1"; shift;;
    -dp)     DP="$1"; shift;;
    -gpu)    GPU="$1"; shift;;
    -resume) RESUME=1;;
    -checkpoint-secs) CHECKPOINT="$1"; shift;;
    *) echo "Unknown arg: $k"; exit 1;;
  esac
done

if [[ -z "$PUBKEY" || -z "$START" ]]; then
  echo "error: -pubkey and -start are required" >&2
  exit 1
fi

# Honor GPU as given; if user wrote 01, pass it through; solver expects a digit list.
EXTRA=()
[[ "$RESUME" -eq 1 ]] && EXTRA+=("-resume")
EXTRA+=("-checkpoint-secs" "$CHECKPOINT")
[[ -n "$GPU" ]] && EXTRA+=("-gpu" "$GPU")

# Delegate to the window scanner with safe, tuned parameters
PUBKEY="$PUBKEY" \
BASE_START_HEX="$START" \
SPAN_BITS="$RANGE" \
WIN_BITS="$WIN_BITS" \
N="$N" \
DP_AUTO="$DP_AUTO" \
DP_SLOTS="$DP_SLOTS" \
TARGET_SEC="$TARGET_SEC" \
THROUGHPUT_GS="$THROUGHPUT_GS" \
CHECKPOINT_SECS="$CHECKPOINT" \
RCK_BIN="$RCK_BIN" \
GPU="${GPU}" \
./scan_puzzle135.sh
