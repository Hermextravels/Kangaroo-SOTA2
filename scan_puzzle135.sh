#!/usr/bin/env bash
# Smart RCKangaroo scan for Bitcoin Puzzle #135
# - Range: 0x4000...000 to 0x7FFF...FFF (34 hex nibbles)
# - Big-int safe math/comparisons using bc
# - Correct hex width padding (34 nibbles)
# - Auto-tunes DP/slots when clamping detected
# - Logs each window and stops when key found in RESULTS.TXT

set -u

# ----- USER CONFIG -----
PUBKEY_COMPRESSED="02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"

# Range for Puzzle #135 (inclusive): allow environment overrides for focused runs
START_HEX=${START_HEX:-"4000000000000000000000000000000000"}  # 34 hex nibbles
END_HEX=${END_HEX:-"7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"}      # 34 hex nibbles

# Windowing and solver tuning
WIN_BITS=${WIN_BITS:-40}     # window size in bits
DP=${DP:-15}                 # initial DP bits (min supported is 15)
DP_SLOTS=${DP_SLOTS:-6}      # initial per-kang slots
MAX_OPS=${MAX:-2.0}          # max ops per window

# Files/paths
RESULTS_FILE=${RESULTS_FILE:-RESULTS.TXT}
LOG_DIR=${LOG_DIR:-logs135}
KANGAROO_BIN=${KANGAROO_BIN:-./rckangaroo}
GPU_MASK=${GPU_MASK:-0}           # default to GPU 0; set to "01" to use devices 0 and 1
CHECKPOINT_SECS=${CHECKPOINT_SECS:-300}
LEDGER_FILE=${LEDGER_FILE:-$LOG_DIR/scan_ledger.csv}
SKIP_VISITED=${SKIP_VISITED:-1}  # 1=skip windows whose start_hex already appears in ledger (fast de-dup and resume)
VERBOSE=${VERBOSE:-0}            # 1=print extra diagnostics and executed commands
MIN_DP=${MIN_DP:-15}             # solver requires DP >= 15
MAX_SLOTS=${MAX_SLOTS:-16}       # solver supports dp-slots in 1..16
TAMES_FILE=${TAMES_FILE:-}       # optional: path to -tames file to reuse (auto-generated if missing when -max is set)

# Optional: focus on a high-hex prefix (e.g., FOCUS_PREFIX=6 or 60 or 6000...) to scan that subrange first
# When set, the script constrains START/END to the block defined by the prefix, clamped to the puzzle range.
FOCUS_PREFIX=${FOCUS_PREFIX:-}

# Window order: seq (default), permute (LCG permutation over windows), zigzag (alternate from start/end),
# random (random window index per step), randstart (random start value anywhere in span). Seed affects modes.
ORDER=${ORDER:-seq}
SEED=${SEED:-0}
SPAN_BITS=134  # total keys in span = 2^134, from 0x4000.. to 0x7FFF..
LIMIT=${LIMIT:-0}         # when ORDER=random or ORDER=randstart, limit number of iterations (0 = use TOTAL_WINDOWS or run until interrupted)

# ----- PRECHECKS -----
command -v bc >/dev/null 2>&1 || { echo "Error: 'bc' is required but not installed." >&2; exit 1; }
# Auto-detect binary if default isn't present
if [ ! -x "$KANGAROO_BIN" ]; then
  # Common candidates to try automatically
  CANDIDATES=(
    "./rckangaroo"
    "./kangaroo"
    "../rckangaroo"
    "../kangaroo"
    "../Kangaroo/rckangaroo"
    "../Kangaroo/kangaroo"
    "../Kangaroo-256-bit/kangaroo"
  )
  for c in "${CANDIDATES[@]}"; do
    if [ -x "$c" ]; then KANGAROO_BIN="$c"; break; fi
  done
  if [ ! -x "$KANGAROO_BIN" ]; then
    # Try PATH
    if command -v rckangaroo >/dev/null 2>&1; then KANGAROO_BIN="$(command -v rckangaroo)"; fi
  fi
  if [ ! -x "$KANGAROO_BIN" ]; then
    if command -v kangaroo >/dev/null 2>&1; then KANGAROO_BIN="$(command -v kangaroo)"; fi
  fi
fi
[ -n "$KANGAROO_BIN" ] && [ -x "$KANGAROO_BIN" ] || { echo "Error: RCKangaroo binary not found/executable. Tried default and common names (rckangaroo/kangaroo). Set KANGAROO_BIN to the correct path, or place the binary next to this script." >&2; exit 1; }
mkdir -p "$LOG_DIR"
[ -f "$LEDGER_FILE" ] || echo "ts,idx,start_hex,win_bits,dp,slots,max,clamp_count" > "$LEDGER_FILE"

# Hex width (nibbles) for padding. Use START_HEX length to preserve full precision.
HEX_WIDTH=${#START_HEX}

# ----- Helpers (big-int safe) -----
uppercase() {
  if [ $# -gt 0 ]; then
    printf "%s" "$1" | tr '[:lower:]' '[:upper:]'
  else
    tr '[:lower:]' '[:upper:]'
  fi
}

# Compare two hex numbers: returns 1 if a<=b else 0
hex_le() {
  local A=$(uppercase "$1"); local B=$(uppercase "$2")
  echo "ibase=16; $A <= $B" | bc
}

# Add two hex numbers, return hex (unpadded, uppercase)
hex_add() {
  local A=$(uppercase "$1"); local B=$(uppercase "$2")
  echo "obase=16; ibase=16; $A + $B" | bc | tr -d '\n'
}

# Convert decimal to hex and pad to HEX_WIDTH
dec_to_hex_pad() {
  local D="$1"
  local HX=$(echo "obase=16; $D" | bc)
  local PADDED=$(printf "%0${HEX_WIDTH}s" "$HX" | tr ' ' '0')
  uppercase "$PADDED"
}

# Convert hex to decimal
hex_to_dec() {
  local H=$(uppercase "$1")
  echo "ibase=16; $H" | bc
}

# Compute 2^WIN_BITS in decimal and hex using Python (portable, big-int safe)
pow2_dec() { python3 - <<PY
print(1 << $WIN_BITS)
PY
}

pow2_hex() { python3 - <<PY
print(format(1 << $WIN_BITS, 'x').upper())
PY
}

# Align start to a window boundary: aligned = floor(start / 2^WIN_BITS) * 2^WIN_BITS
align_start_to_window() {
  local H=$(uppercase "$1")
  local START_DEC=$(hex_to_dec "$H")
  local WIN_SIZE_DEC=$(pow2_dec)
  local ALIGNED_DEC=$(python3 - <<PY
start_dec=int("$START_DEC")
win=int("$WIN_SIZE_DEC")
print((start_dec // win) * win)
PY
)
  dec_to_hex_pad "$ALIGNED_DEC"
}

# Precompute window size in hex
WIN_SIZE_HEX=$(pow2_hex)

START_HEX=$(uppercase "$START_HEX")
END_HEX=$(uppercase "$END_HEX")

if [ ${#END_HEX} -ne $HEX_WIDTH ]; then
  echo "Error: START_HEX and END_HEX must have same width (got $HEX_WIDTH and ${#END_HEX})." >&2
  exit 1
fi

# If FOCUS_PREFIX is provided, compute its block subrange and clamp to the puzzle range
if [ -n "$FOCUS_PREFIX" ]; then
  FOCUS_PREFIX=$(uppercase "$FOCUS_PREFIX")
  if ! echo "$FOCUS_PREFIX" | grep -Eq '^[0-9A-F]+$'; then
    echo "Error: FOCUS_PREFIX must be hex (0-9A-F). Got '$FOCUS_PREFIX'" >&2
    exit 1
  fi
  if [ ${#FOCUS_PREFIX} -gt $HEX_WIDTH ]; then
    echo "Error: FOCUS_PREFIX length (${#FOCUS_PREFIX}) exceeds hex width ($HEX_WIDTH)." >&2
    exit 1
  fi
  read -r FOCUS_START_HEX FOCUS_END_HEX <<EOF
$(python3 - <<PY
pref = int("$FOCUS_PREFIX", 16)
hex_width = $HEX_WIDTH
total_bits = hex_width * 4
pref_bits = len("$FOCUS_PREFIX") * 4
shift = total_bits - pref_bits
block_start = pref << shift
block_end = block_start | ((1 << shift) - 1)
puz_start = int("$START_HEX",16)
puz_end = int("$END_HEX",16)
focus_start = max(block_start, puz_start)
focus_end = min(block_end, puz_end)
if focus_start > focus_end:
    # No overlap with the puzzle range
    print("")
else:
    fmt = "{:0%dx}" % hex_width
    print(fmt.format(focus_start).upper(), fmt.format(focus_end).upper())
PY
)
EOF
  if [ -z "$FOCUS_START_HEX" ] || [ -z "$FOCUS_END_HEX" ]; then
    echo "Warning: FOCUS_PREFIX block doesn't intersect the puzzle range; ignoring FOCUS_PREFIX=$FOCUS_PREFIX" >&2
  else
    echo "Focusing scan on prefix 0x$FOCUS_PREFIX -> $FOCUS_START_HEX .. $FOCUS_END_HEX"
    START_HEX="$FOCUS_START_HEX"
    END_HEX="$FOCUS_END_HEX"
  fi
fi

ALIGNED_START_HEX=$(align_start_to_window "$START_HEX")

echo "Puzzle #135 scan starting"
echo "Range: $ALIGNED_START_HEX .. $END_HEX (HEX_WIDTH=${HEX_WIDTH}, WIN_BITS=$WIN_BITS)"
echo "Using binary: $KANGAROO_BIN"

# Enforce minimum DP before starting
if [ "$DP" -lt "$MIN_DP" ]; then
  echo "DP ($DP) below minimum ($MIN_DP), raising to $MIN_DP"
  DP=$MIN_DP
fi
# Clamp DP_SLOTS to supported range before starting
if [ "$DP_SLOTS" -lt 1 ]; then
  echo "DP_SLOTS ($DP_SLOTS) below minimum (1), raising to 1"
  DP_SLOTS=1
fi
if [ "$DP_SLOTS" -gt "$MAX_SLOTS" ]; then
  echo "DP_SLOTS ($DP_SLOTS) above maximum ($MAX_SLOTS), clamping to $MAX_SLOTS"
  DP_SLOTS=$MAX_SLOTS
fi

# Compute total number of windows = floor((end-start)/2^WIN_BITS) + 1 (big-int safe)
TOTAL_WINDOWS=$(python3 - <<PY
start_int=int("$ALIGNED_START_HEX",16)
end_int=int("$END_HEX",16)
win=1<<$WIN_BITS
print(((end_int - start_int)//win) + 1)
PY
)
echo "Windows to scan: $TOTAL_WINDOWS (order=$ORDER seed=$SEED)"

trap 'echo "\nInterrupted."; exit 130' INT TERM

found=0
win_index=0

# Determine how many iterations to run
if { [ "$ORDER" = "random" ] || [ "$ORDER" = "randstart" ]; } && [ "$LIMIT" -gt 0 ]; then
  END_COUNT="$LIMIT"
else
  END_COUNT="$TOTAL_WINDOWS"
fi

get_start_for_index() {
  python3 - "$win_index" <<PY
import sys
idx=int(sys.argv[1])
WIN_BITS=$WIN_BITS
HEX_WIDTH=$HEX_WIDTH
SPAN_BITS=$SPAN_BITS
N_BITS=SPAN_BITS-WIN_BITS
A=6364136223846793005  # odd, co-prime to 2^N
B=$SEED
order="$ORDER"
start=int("$ALIGNED_START_HEX",16)
totwin=int("$TOTAL_WINDOWS")
if order == "permute":
    mask=(1<<N_BITS)-1
    idx=(A*idx + B) & mask
elif order == "random":
    import random
    rnd = random.Random(B + idx)
    idx = rnd.randrange(0, totwin)
elif order == "zigzag":
  # 0, last, 1, last-1, 2, last-2, ...
  if (idx & 1) == 0:
    idx = idx >> 1
  else:
    idx = (totwin - 1) - (idx >> 1)
elif order == "randstart":
  import random
  rnd = random.Random(B + idx)
  start_int = start
  end_int = int("$END_HEX",16)
  win = 1<<WIN_BITS
  upper_start = end_int - (win - 1)
  if upper_start < start_int:
    upper_start = start_int
  span = upper_start - start_int + 1
  offset = rnd.randrange(0, span)
  start_val = start_int + offset
  print(("{:0%dx}"%HEX_WIDTH).format(start_val).upper())
  sys.exit(0)
start_val = start + (idx << WIN_BITS)
print(("{:0%dx}"%HEX_WIDTH).format(start_val).upper())
PY
}

while [ "$(echo "$win_index < $END_COUNT" | bc)" -eq 1 ]; do
  current_hex=$(get_start_for_index)
  log_file="$LOG_DIR/w${WIN_BITS}_i${win_index}_start_${current_hex}.log"
  # Skip if already visited (resume/de-dup)
  if [ "$SKIP_VISITED" = "1" ] && grep -Fq ",$current_hex," "$LEDGER_FILE" 2>/dev/null; then
    echo "Skipping window $win_index (already visited): start=$current_hex"
    win_index=$((win_index+1))
    continue
  fi
  echo "Scanning window $win_index: start=$current_hex, DP=$DP, slots=$DP_SLOTS"
  # Optional: show end-of-window coverage (start + 2^WIN_BITS - 1)
  # end_hex=$(echo "obase=16; ibase=16; $current_hex + $WIN_SIZE_HEX - 1" | bc | tr -d '\n' | awk '{printf "%0* s", '$HEX_WIDTH', toupper($0)}' | tr ' ' '0')
  # echo "Covers: $current_hex .. $end_hex"

  # Build CLI
  # Re-clamp DP and slots each iteration in case tuning changed them
  if [ "$DP" -lt "$MIN_DP" ]; then DP=$MIN_DP; fi
  if [ "$DP_SLOTS" -lt 1 ]; then DP_SLOTS=1; fi
  if [ "$DP_SLOTS" -gt "$MAX_SLOTS" ]; then DP_SLOTS=$MAX_SLOTS; fi
  args=( -pubkey "$PUBKEY_COMPRESSED" -start "$current_hex" -range "$WIN_BITS" -dp "$DP" -dp-slots "$DP_SLOTS" -max "$MAX_OPS" -resume -checkpoint-secs "$CHECKPOINT_SECS" )
  # Reuse or generate tames per WIN_BITS range if provided
  if [ -n "$TAMES_FILE" ]; then
    args+=( -tames "$TAMES_FILE" )
  fi
  if [ -n "$GPU_MASK" ]; then
    args=( -gpu "$GPU_MASK" "${args[@]}" )
  fi
  if [ "$VERBOSE" = "1" ]; then
    echo "Exec: $KANGAROO_BIN ${args[*]}" | tee -a "$log_file"
  fi
  "${KANGAROO_BIN}" "${args[@]}" > "$log_file" 2>&1
  rc=$?
  if [ "$VERBOSE" = "1" ]; then
    echo "Exit code: $rc" | tee -a "$log_file"
  fi
  # One-time retry if DP is below minimum (some builds require DP>=15)
  if [ $rc -ne 0 ] && [ "$DP" -lt "$MIN_DP" ]; then
    echo "Solver failed with DP=$DP on window $win_index; raising DP to $MIN_DP and retrying" | tee -a "$log_file"
    DP=$MIN_DP
    args=( -pubkey "$PUBKEY_COMPRESSED" -start "$current_hex" -range "$WIN_BITS" -dp "$DP" -dp-slots "$DP_SLOTS" -max "$MAX_OPS" -resume -checkpoint-secs "$CHECKPOINT_SECS" )
    if [ -n "$GPU_MASK" ]; then
      args=( -gpu "$GPU_MASK" "${args[@]}" )
    fi
    if [ "$VERBOSE" = "1" ]; then
      echo "Retry Exec: $KANGAROO_BIN ${args[*]}" | tee -a "$log_file"
    fi
    "${KANGAROO_BIN}" "${args[@]}" >> "$log_file" 2>&1
    rc=$?
    if [ "$VERBOSE" = "1" ]; then
      echo "Retry exit code: $rc" | tee -a "$log_file"
    fi
  fi
  if [ $rc -ne 0 ]; then
    echo "Solver exited with non-zero code ($rc) on window $win_index. Showing first 40 lines of log:"
    sed -n '1,40p' "$log_file" >&2
    exit $rc
  fi
  if ! grep -qE "CUDA devices:|GPUs started" "$log_file"; then
    echo "Warning: CUDA init markers not detected in log for window $win_index. Showing first 40 lines:" >&2
    sed -n '1,40p' "$log_file" >&2
  fi

  if grep -q "DP clamped events" "$log_file"; then
    clamp_count=$(grep -m1 "DP clamped events" "$log_file" | sed -E 's/.*clamped events this iteration: ([0-9]+).*/\1/' || echo 0)
    if [ -n "$clamp_count" ] && [ "$clamp_count" -gt 80 ]; then
      DP=$(( DP>MIN_DP ? DP-1 : DP ))
      # Increase slots up to MAX_SLOTS, step by 2 for stability
      if [ "$DP_SLOTS" -lt "$MAX_SLOTS" ]; then
        inc=$(( DP_SLOTS+2 ))
        DP_SLOTS=$(( inc>MAX_SLOTS ? MAX_SLOTS : inc ))
      fi
      echo "Auto-tuning due to clamping=$clamp_count -> DP=$DP, slots=$DP_SLOTS"
    fi
  fi

  # Append ledger line
  ts_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$ts_iso,$win_index,$current_hex,$WIN_BITS,$DP,$DP_SLOTS,$MAX_OPS,${clamp_count:-0}" >> "$LEDGER_FILE"

  if [ -f "$RESULTS_FILE" ] && grep -q "PRIVATE KEY" "$RESULTS_FILE" 2>/dev/null; then
    echo "Key found in window $win_index! See $RESULTS_FILE"
    found=1
    break
  fi

  win_index=$((win_index+1))
done

if [ "$found" -eq 0 ]; then
  echo "Scan complete. No key found in scanned range."
fi
