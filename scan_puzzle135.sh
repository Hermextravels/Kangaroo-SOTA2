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
WIN_BITS=${WIN_BITS:-40}              # default single window size (ignored if MULTI_WIN_BITS set)
# Comma-separated list of window bit sizes to cycle or randomize through (e.g. "44,42,46,40").
# Using multiple sizes slightly varies collision structure & DP emission; does not change fundamental odds
MULTI_WIN_BITS=${MULTI_WIN_BITS:-""}
WIN_CYCLE_MODE=${WIN_CYCLE_MODE:-roundrobin} # roundrobin|random
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
LEDGER_FILE=${LEDGER_FILE:-"$LOG_DIR/ledger.csv"}  # CSV summary of windows
LIST_FILE=${LIST_FILE:-""}           # if provided, read starts from this file instead of generating
STOP_FILE=${STOP_FILE:-"STOP_SCAN"}   # create this file to stop between windows
TIMEOUT_SEC=${TIMEOUT_SEC:-0}         # optional per-window wall time cap; 0=disabled
# Skip windows that already finished successfully (detected via a completion marker in the log)
SKIP_DONE=${SKIP_DONE:-1}
# Optional: target a wall-clock duration per window; we'll auto-compute MAX from it.
# Assumes throughput around THROUGHPUT_GS Gsteps/sec (set this if you know your speed)
TARGET_SEC=${TARGET_SEC:-0}           # e.g., 900 for ~15 minutes; 0 disables auto MAX
THROUGHPUT_GS=${THROUGHPUT_GS:-1.0}   # measured Gsteps/sec (Tesla T4 often ~0.9–1.1)
MEM_CAP_MB=${MEM_CAP_MB:-3800}        # soft cap for DP tables (approx check)
KANG_CNT_HINT=${KANG_CNT_HINT:-1310720} # override if solver prints different KangCnt
ADAPTIVE_DP=${ADAPTIVE_DP:-1}         # 1=enable per-window adaptive DP/slots adjustments
CLAMP_THRESHOLD=${CLAMP_THRESHOLD:-120} # events above this may trigger slot/DP change
LOW_DP_COUNT_THRESHOLD=${LOW_DP_COUNT_THRESHOLD:-8000} # iteration DP count below this -> lower DP bits
HIGH_DP_COUNT_THRESHOLD=${HIGH_DP_COUNT_THRESHOLD:-40000} # if above & no clamp may raise DP to reduce overhead
MAX_DP_SLOTS=${MAX_DP_SLOTS:-16}
MIN_DP=${MIN_DP:-12}
MAX_DP=${MAX_DP_VAL:-22}
COLLISION_THRESHOLD=${COLLISION_THRESHOLD:-25}  # if Collision Error lines in a window exceed this, raise DP (rarer DPs)
# Optional: shift the base heuristic DP upward/downward (e.g., DP_BASE_SHIFT=+2 to start 2 bits higher to suppress early collisions)
DP_BASE_SHIFT=${DP_BASE_SHIFT:-0}
# Optional: force a minimum DP after heuristic + shift (e.g., MIN_DP_OVERRIDE=18)
MIN_DP_OVERRIDE=${MIN_DP_OVERRIDE:-0}

mkdir -p "$LOG_DIR"

# Initialize ledger header if not present
if [[ ! -f "$LEDGER_FILE" ]]; then
  echo "timestamp,window_index,win_bits,start_hex,dp,dp_slots,max,dp_count,clamp_events,collision_count,last_speed_mkeys_s,duration_sec" > "$LEDGER_FILE"
fi

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

# Internal function: base heuristic (used at start or when changing window size)
_base_dp_heuristic() {
  local wb="$1"
  local new_dp="$DP" new_slots="$DP_SLOTS" new_max="$MAX"
  if [[ "$DP_AUTO" -eq 1 ]]; then
    if [[ "$wb" =~ ^[0-9]+$ ]]; then
      if   (( wb <= 40 )); then new_dp=16; new_slots=6;   [[ -z "${MAX_SET:-}" ]] && new_max=${MAX:-1.2};
      elif (( wb <= 44 )); then new_dp=16; new_slots=10;  [[ -z "${MAX_SET:-}" ]] && new_max=${MAX:-1.1};
      elif (( wb <= 50 )); then new_dp=20; new_slots=6;   [[ -z "${MAX_SET:-}" ]] && new_max=${MAX:-1.05};
      else                     new_dp=22; new_slots=8;   [[ -z "${MAX_SET:-}" ]] && new_max=${MAX:-1.0};
      fi
    fi
  fi
  # Apply optional global shift & override to mitigate early Collision Errors
  if [[ "$DP_BASE_SHIFT" != "0" ]]; then
    new_dp=$(( new_dp + DP_BASE_SHIFT ))
  fi
  if (( MIN_DP_OVERRIDE > 0 && new_dp < MIN_DP_OVERRIDE )); then
    new_dp=$MIN_DP_OVERRIDE
  fi
  if (( new_dp > MAX_DP )); then
    new_dp=$MAX_DP
  fi
  echo "$new_dp $new_slots $new_max"
}

# Initialize DP/slots for initial WIN_BITS (single mode) only if no multi list
if [[ -z "$MULTI_WIN_BITS" ]]; then
  if [[ "$DP_AUTO" -eq 1 ]]; then
    read DP DP_SLOTS MAX < <(_base_dp_heuristic "$WIN_BITS")
    echo "Auto-tuned parameters for WIN_BITS=$WIN_BITS -> DP=$DP, DP_SLOTS=$DP_SLOTS, MAX=$MAX"
  fi
else
  IFS=',' read -r -a _WIN_LIST <<< "$MULTI_WIN_BITS"
  echo "Multi window sizes enabled: ${_WIN_LIST[*]} (mode=$WIN_CYCLE_MODE)"
fi

# (legacy block kept for backward compatibility if user inspects script)
if [[ -z "$MULTI_WIN_BITS" && "$DP_AUTO" -eq 1 ]]; then
  # Heuristic: for tiny windows, lower DP to emit enough DPs per kangaroo
  # and keep slots modest; for larger windows, raise DP to cut DP traffic.
  : # already handled above, kept for clarity
fi

# --- iterate windows ---
idx=0
_win_index=0
while IFS= read -r START_HEX; do
  idx=$((idx+1))
  [[ -z "$START_HEX" ]] && continue

  if [[ -f "$STOP_FILE" ]]; then
    echo "Stop file present ($STOP_FILE). Exiting before window #$idx." >&2
    exit 0
  fi

  # Determine active window bits (possibly cycling)
  ACTIVE_WIN_BITS="$WIN_BITS"
  if [[ -n "$MULTI_WIN_BITS" ]]; then
    if [[ "$WIN_CYCLE_MODE" == "random" ]]; then
      # shell-safe random pick
      ACTIVE_WIN_BITS=${_WIN_LIST[$RANDOM % ${#_WIN_LIST[@]}]}
    else
      ACTIVE_WIN_BITS=${_WIN_LIST[$((_win_index % ${#_WIN_LIST[@]}))]}
    fi
    _win_index=$((_win_index+1))
    # Recompute base heuristic if size changed
    read DP DP_SLOTS MAX < <(_base_dp_heuristic "$ACTIVE_WIN_BITS")
  fi

  # per-window log file (include active bits for clarity)
  LOG="$LOG_DIR/w${ACTIVE_WIN_BITS}_i${idx}_start_${START_HEX}.log"

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
    "-start" "$START_HEX" "-range" "$ACTIVE_WIN_BITS" "-max" "$MAX" "-pubkey" "$PUBKEY")
  if [[ -n "$GPU" ]]; then
    cmd=("${cmd[@]}" "-gpu" "$GPU")
  fi

  echo "[Window $idx] start=$START_HEX range=$ACTIVE_WIN_BITS dp=$DP slots=$DP_SLOTS max=$MAX" | tee -a "$LOG"
  echo "Logging to: $LOG"

  window_t0=$(date +%s)

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

  # Adaptive DP/slots tuning for next window (analyze tail metrics)
  if [[ "$ADAPTIVE_DP" -eq 1 ]]; then
    # Grab last iteration lines (avoid huge greps)
    last_iter_line=$(grep -E "Iteration DP count=" "$LOG" | tail -n1 || true)
    clamp_line=$(grep -E "DP clamped events" "$LOG" | tail -n1 || true)
    if [[ -n "$last_iter_line" ]]; then
      # Extract DP count and speed
      dp_count=$(echo "$last_iter_line" | sed -E 's/.*Iteration DP count=([0-9]+).*/\1/')
    else
      dp_count=0
    fi
    if [[ -n "$clamp_line" ]]; then
      clamp_events=$(echo "$clamp_line" | sed -E 's/.*events this iteration: ([0-9]+).*/\1/')
    else
      clamp_events=0
    fi
    adjust_msg=""

    # Memory estimator before changes
    current_mem_mb=$(( KANG_CNT_HINT * (DP_SLOTS * 16 + 4) / 1000000 ))

    # If clamp events high -> try increasing slots first
    if (( clamp_events > CLAMP_THRESHOLD )); then
      if (( DP_SLOTS < MAX_DP_SLOTS )); then
        DP_SLOTS=$((DP_SLOTS + 2))
        adjust_msg+="clamp>${CLAMP_THRESHOLD}: +slots -> ${DP_SLOTS}; "
      elif (( DP < MAX_DP )); then
        DP=$((DP + 1))
        adjust_msg+="clamp>${CLAMP_THRESHOLD}: +DP -> ${DP}; "
      fi
    else
      # Low clamp pressure: maybe reduce DP if DP density too sparse
      if (( dp_count < LOW_DP_COUNT_THRESHOLD && DP > MIN_DP )); then
        DP=$((DP - 1))
        adjust_msg+="lowDP<${LOW_DP_COUNT_THRESHOLD}: -DP -> ${DP}; "
      elif (( dp_count > HIGH_DP_COUNT_THRESHOLD && DP < MAX_DP )); then
        # plenty of DPs, can raise DP to trim overhead if not clamping
        DP=$((DP + 1))
        adjust_msg+="highDP>${HIGH_DP_COUNT_THRESHOLD}: +DP -> ${DP}; "
      fi
    fi

    # Re-estimate memory; revert if exceeding cap
    new_mem_mb=$(( KANG_CNT_HINT * (DP_SLOTS * 16 + 4) / 1000000 ))
    if (( new_mem_mb > MEM_CAP_MB )); then
      # revert slots change if that was the difference
      if (( new_mem_mb > MEM_CAP_MB && DP_SLOTS > 2 )); then
        DP_SLOTS=$((DP_SLOTS - 2))
        new_mem_mb=$(( KANG_CNT_HINT * (DP_SLOTS * 16 + 4) / 1000000 ))
        adjust_msg+="memCap: revert slots -> ${DP_SLOTS}; "
      fi
    fi

    if [[ -n "$adjust_msg" ]]; then
      echo "ADAPT_NEXT Window=$((idx+1)) dp=$DP slots=$DP_SLOTS estMem=${new_mem_mb}MB (${adjust_msg})" | tee -a "$LOG_DIR/scan_summary.log"
    fi

    # Secondary adaptation: Collision Error frequency
    collision_count=$(grep -c "Collision Error" "$LOG" || true)
    if (( collision_count > COLLISION_THRESHOLD )); then
      if (( DP < MAX_DP )); then
        old_dp=$DP
        # Escalate by 1 (or 2 if extremely high)
        if (( collision_count > COLLISION_THRESHOLD * 3 && DP + 2 <= MAX_DP )); then
          DP=$((DP+2))
          bump=2
        else
          DP=$((DP+1))
          bump=1
        fi
        # Optionally shrink slots slightly to free memory (keep >=4)
        if (( DP_SLOTS > 6 )); then
          DP_SLOTS=$((DP_SLOTS-2))
        fi
        new_mem_mb=$(( KANG_CNT_HINT * (DP_SLOTS * 16 + 4) / 1000000 ))
        echo "ADAPT_NEXT Window=$((idx+1)) dp=$DP slots=$DP_SLOTS estMem=${new_mem_mb}MB (collisionCount=${collision_count}: +${bump} DP, slots adjust)" | tee -a "$LOG_DIR/scan_summary.log"
      else
        echo "ADAPT_NEXT Window=$((idx+1)) dp=$DP slots=$DP_SLOTS (collisionCount=${collision_count}: at MAX_DP, no change)" | tee -a "$LOG_DIR/scan_summary.log"
      fi
    fi
  fi

  # --- Ledger collection (independent of adaptive logic) ---
  # Gather metrics from this window's log and append one CSV row.
  last_iter_line=$(grep -E "Iteration DP count=" "$LOG" | tail -n1 || true)
  last_speed="" dp_count_val="" clamp_events_val="" collision_count_val=""
  if [[ -n "$last_iter_line" ]]; then
    dp_count_val=$(echo "$last_iter_line" | sed -E 's/.*Iteration DP count=([0-9]+).*/\1/')
    last_speed=$(echo "$last_iter_line" | sed -E 's/.*speed=([0-9]+) MKeys\/s.*/\1/')
  fi
  clamp_events_line=$(grep -E "DP clamped events" "$LOG" | tail -n1 || true)
  if [[ -n "$clamp_events_line" ]]; then
    clamp_events_val=$(echo "$clamp_events_line" | sed -E 's/.*events this iteration: ([0-9]+).*/\1/')
  else
    clamp_events_val=0
  fi
  collision_count_val=$(grep -c "Collision Error" "$LOG" || true)
  window_t1=$(date +%s)
  duration=$((window_t1-window_t0))
  ts_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$ts_iso,$idx,$ACTIVE_WIN_BITS,$START_HEX,$DP,$DP_SLOTS,$MAX,${dp_count_val:-0},${clamp_events_val:-0},${collision_count_val:-0},${last_speed:-0},$duration" >> "$LEDGER_FILE"

done < "$STARTS_FILE"

echo "All $idx windows completed. No result detected."
