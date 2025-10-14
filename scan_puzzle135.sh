#!/usr/bin/env bash
# Auto window scanner for Bitcoin Puzzle #135 using RCKangaroo
# - Generates many small subranges (windows) inside the 134-bit span
# - Runs rckangaroo on each window with your chosen DP settings
# - Logs output per window and supports safe resume/stop
#
# Requirements: bash, python3, NVIDIA GPU with CUDA (or your existing setup)
# Usage examples:
#   chmod +x scan_puzzle135.sh
#   ./scan_puzzle135.sh                # run with defaults (WIN_BITS=40, N=200)
#   WIN_BITS=44 N=100 ./scan_puzzle135.sh
#   GPU=0 N=500 MAX=1.0 ./scan_puzzle135.sh
#   STOP_FILE=STOP ./scan_puzzle135.sh  # create STOP to stop between windows

set -euo pipefail

# --- USER TUNABLES (can be overridden via env) ---
PUBKEY=${PUBKEY:-"02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"}
BASE_START_HEX=${BASE_START_HEX:-"4000000000000000000000000000000000"}
SPAN_BITS=${SPAN_BITS:-134}
WIN_BITS=${WIN_BITS:-40}              # try 40 or 44
N=${N:-200}                           # number of windows to generate
SEED=${SEED:-"puzzle135_seed_v1"}     # change to vary sequence deterministically
ALIGN=${ALIGN:-1}                     # 1=align window start to 2^WIN_BITS boundary; 0=no align
MAX=${MAX:-1.0}                       # -max factor per window (1.0 ~ expected work per window)
DP=${DP:-22}
DP_SLOTS=${DP_SLOTS:-8}
# If enabled, override DP/DP_SLOTS (and maybe MAX) based on WIN_BITS heuristics
DP_AUTO=${DP_AUTO:-1}                # 1=auto-tune DP/slots from WIN_BITS; 0=use provided DP/slots as-is
RANDOMIZE=${RANDOMIZE:-1}            # 1=shuffle window order; 0=sorted
GPU=${GPU:-""}                        # e.g. "0" to pin GPU 0; empty = auto
CHECKPOINT_SECS=${CHECKPOINT_SECS:-300}
RCK_BIN=${RCK_BIN:-"./rckangaroo"}
LOG_DIR=${LOG_DIR:-"logs135"}
LIST_FILE=${LIST_FILE:-""}           # if provided, read starts from this file instead of generating
STOP_FILE=${STOP_FILE:-"STOP_SCAN"}   # create this file to stop between windows
TIMEOUT_SEC=${TIMEOUT_SEC:-0}         # optional per-window wall time cap; 0=disabled
# Skip windows that already finished successfully (detected via a completion marker in the log)
SKIP_DONE=${SKIP_DONE:-1}
# Optional: target a wall-clock duration per window; we'll auto-compute MAX from it.
# Assumes throughput around THROUGHPUT_GS Gsteps/sec (set this if you know your speed)
TARGET_SEC=${TARGET_SEC:-0}           # e.g., 900 for ~15 minutes; 0 disables auto MAX
THROUGHPUT_GS=${THROUGHPUT_GS:-1.0}   # measured Gsteps/sec (Tesla T4 often ~0.9–1.1)

mkdir -p "$LOG_DIR"

if [[ ! -x "$RCK_BIN" ]]; then
  echo "Error: rckangaroo binary not found: $RCK_BIN" >&2
  echo "Tip: build in this folder (Makefile) or point RCK_BIN to your binary path." >&2
  exit 1
fi

# --- helper: generate window starts ---
# Prints N hex start values to stdout
_gen_starts() {
  python3 - "$BASE_START_HEX" "$SPAN_BITS" "$WIN_BITS" "$N" "$SEED" "$ALIGN" << 'PY'
import sys, secrets
BASE_HEX, SPAN_BITS, WIN_BITS, N, SEED, ALIGN = sys.argv[1:]
SPAN_BITS = int(SPAN_BITS); WIN_BITS = int(WIN_BITS); N = int(N); ALIGN=int(ALIGN)
BASE = int(BASE_HEX, 16)
SPAN = 1 << SPAN_BITS
WIN  = 1 << WIN_BITS
rng = secrets.SystemRandom(SEED)

# Avoid duplicates by using a set; oversample then trim
starts = set()
limit = SPAN - WIN
mask_align = ~(WIN - 1)
for _ in range(max(N*3, 1000)):
    off = rng.randrange(0, limit + 1)
    if ALIGN:
        off &= mask_align
    s = BASE + off
    # ensure within [BASE, BASE+SPAN-WIN]
    if s < BASE or s + WIN > BASE + SPAN:
        continue
    starts.add(s)
    if len(starts) >= N:
        break

for s in sorted(starts):
    # print lower-case hex without 0x
    print(f"{s:x}")
PY
}

# --- prepare starts ---
STARTS_FILE="$LOG_DIR/starts_win${WIN_BITS}_n${N}.txt"
if [[ -n "$LIST_FILE" ]]; then
  echo "Using provided list file: $LIST_FILE"
  cp "$LIST_FILE" "$STARTS_FILE"
else
  echo "Generating $N window starts (WIN_BITS=$WIN_BITS, ALIGN=$ALIGN)…"
  _gen_starts > "$STARTS_FILE"
fi

# Shuffle window order if requested (random sampling helps luck-per-day)
if [[ "$RANDOMIZE" -eq 1 ]]; then
  if command -v shuf >/dev/null 2>&1; then
    tmpf=$(mktemp)
    shuf "$STARTS_FILE" > "$tmpf" && mv "$tmpf" "$STARTS_FILE"
  else
    # Fallback shuffle via Python
    python3 - "$STARTS_FILE" << 'PY'
import sys, random
p = sys.argv[1]
with open(p,'r') as f:
    lines = [l for l in f if l.strip()]
random.shuffle(lines)
with open(p,'w') as f:
    f.writelines(lines)
PY
  fi
fi

# Auto-tune DP/slots for the chosen window size, unless disabled
if [[ "$DP_AUTO" -eq 1 ]]; then
  # Heuristic: for tiny windows, lower DP to emit enough DPs per kangaroo
  # and keep slots modest; for larger windows, raise DP to cut DP traffic.
  case "$WIN_BITS" in
    ''|*[!0-9]*) true ;; # leave defaults if WIN_BITS not numeric
    *)
      wb=$WIN_BITS
      if   (( wb <= 40 )); then DP=16; DP_SLOTS=6; MAX=${MAX:-1.2};
      elif (( wb <= 44 )); then DP=16; DP_SLOTS=10; MAX=${MAX:-1.1};
      elif (( wb <= 50 )); then DP=20; DP_SLOTS=6; MAX=${MAX:-1.05};
      else                       DP=22; DP_SLOTS=8; MAX=${MAX:-1.0};
      fi
      ;;
  esac
  echo "Auto-tuned parameters for WIN_BITS=$WIN_BITS -> DP=$DP, DP_SLOTS=$DP_SLOTS, MAX=$MAX"
fi

# --- iterate windows ---
idx=0
while IFS= read -r START_HEX; do
  idx=$((idx+1))
  [[ -z "$START_HEX" ]] && continue

  if [[ -f "$STOP_FILE" ]]; then
    echo "Stop file present ($STOP_FILE). Exiting before window #$idx." >&2
    exit 0
  fi

  # per-window log file
  LOG="$LOG_DIR/w${WIN_BITS}_i${idx}_start_${START_HEX}.log"

  # Skip previously completed windows if requested
  if [[ "$SKIP_DONE" -eq 1 && -f "$LOG" ]] && grep -q "WINDOW_DONE" "$LOG" 2>/dev/null; then
    echo "[Window $idx] already completed, skipping: start=$START_HEX" | tee -a "$LOG_DIR/scan_summary.log"
    continue
  fi

  # Optional: compute MAX from desired time budget and estimated throughput
  if [[ "$TARGET_SEC" -gt 0 ]]; then
    # MAX is multiplicative of the window's expected op count, but we don't know that here.
    # rckangaroo prints an estimated ops like 2^k; empirically, MAX~ops_multiplier. We'll approximate
    # using time = MAX * baseline_seconds; assume baseline ~ 1 second for 2^WIN_BITS windows at default.
    # Instead, derive MAX from throughput (Gsteps/s) and time: ops_target = THROUGHPUT_GS * 1e9 * TARGET_SEC.
    # Then convert to a MAX scale by dividing by an empirical baseline per window (~2^22 for 44-bit SOTA shown in logs).
    # Use a heuristic baseline depending on WIN_BITS to keep units sane.
    case "$WIN_BITS" in
      ''|*[!0-9]*) base_ops=4194304 ;; # default ~2^22
      *)
        wb=$WIN_BITS
        if   (( wb <= 40 )); then base_ops=2097152;   # 2^21 approx
        elif (( wb <= 44 )); then base_ops=4194304;   # 2^22 approx (seen in logs)
        elif (( wb <= 50 )); then base_ops=8388608;   # 2^23 approx
        else                     base_ops=16777216;  # 2^24 approx
        fi
        ;;
    esac
    ops_target=$(python3 - <<PY
import math
throughput_gs = float("$THROUGHPUT_GS")
target_sec = float("$TARGET_SEC")
print(int(throughput_gs * 1e9 * target_sec))
PY
)
    # Avoid zero or too small
    if [[ "$ops_target" -lt 1 ]]; then ops_target=1; fi
    # Compute MAX ~ ops_target / base_ops, but clamp to a reasonable range
    new_max=$(python3 - <<PY
import math
ops_target = int("$ops_target")
base_ops = int("$base_ops")
m = max(1.0, ops_target / max(1, base_ops))
# clamp
print(min(m, 1e7))
PY
)
    echo "Auto MAX from TARGET_SEC=$TARGET_SEC sec @ ${THROUGHPUT_GS}Gsteps/s (ops_target=$ops_target, base_ops=$base_ops) -> MAX=$new_max" | tee -a "$LOG"
    MAX="$new_max"
  fi

  # build command
  cmd=("$RCK_BIN" "-dp" "$DP" "-dp-slots" "$DP_SLOTS" "-resume" "-checkpoint-secs" "$CHECKPOINT_SECS" \
       "-start" "$START_HEX" "-range" "$WIN_BITS" "-max" "$MAX" "-pubkey" "$PUBKEY")
  if [[ -n "$GPU" ]]; then
    cmd=("${cmd[@]}" "-gpu" "$GPU")
  fi

  echo "[Window $idx] start=$START_HEX range=$WIN_BITS dp=$DP slots=$DP_SLOTS max=$MAX" | tee -a "$LOG"
  echo "Logging to: $LOG"

  if [[ "$TIMEOUT_SEC" -gt 0 ]]; then
    # Use timeout if available; otherwise run normally
    if command -v timeout >/dev/null 2>&1; then
      timeout "$TIMEOUT_SEC" "${cmd[@]}" 2>&1 | tee -a "$LOG" || true
    else
      echo "timeout command not found; running without time limit" | tee -a "$LOG"
      "${cmd[@]}" 2>&1 | tee -a "$LOG"
    fi
  else
    "${cmd[@]}" 2>&1 | tee -a "$LOG"
  fi

  # Quick check for success
  if grep -qi "Private key" "$LOG" || grep -qi "RESULTS.TXT" "$LOG"; then
    echo "[Window $idx] Potential result detected. Review $LOG and RESULTS.TXT." | tee -a "$LOG"
    exit 0
  fi

  # Mark window completion
  echo "WINDOW_DONE" >> "$LOG"

done < "$STARTS_FILE"

echo "All $idx windows completed. No result detected."
