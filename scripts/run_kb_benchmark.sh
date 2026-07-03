#!/bin/bash
# ============================================================================
# KernelBench L1+L2 Full Benchmark Runner
#
# Usage: bash run_kb_benchmark.sh <gpu_id> <policy> <beam_width> <level> <start> <end>
#
# Example:
#   bash run_kb_benchmark.sh 0 greedy 1 1 1 100    # L1 kernels 001-100, greedy
#   bash run_kb_benchmark.sh 2 beam 4 2 1 100      # L2 kernels 001-100, beam width=4
#
# Output structure:
#   kb_benchmark_results/<policy>_L<level>_gpu<gpu>/
#     kernel_XXX/
#       run.log              — full pipeline output
#       pipeline_checkpoint.json
#       best_kernel.py
#       optimization_log.md
#       synthesis/
#       opt_*/
# ============================================================================

set -euo pipefail

GPU=$1
POLICY=$2
BW=$3
LEVEL=$4
START=$5
END=$6

BASE="$(cd "$(dirname "$0")/.." && pwd)"
OUT=$BASE/kb_benchmark_results/${POLICY}_L${LEVEL}_gpu${GPU}
LOGFILE=$OUT/master.log

export CUDA_VISIBLE_DEVICES=$GPU
# Ensure ncu is on PATH
which ncu >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"

mkdir -p "$OUT"

# Config
T=5
N=4
K_ATTEMPTS=4
E2E_RETRIES=3

echo "============================================" | tee -a "$LOGFILE"
echo "KernelBench L${LEVEL} Benchmark" | tee -a "$LOGFILE"
echo "GPU: $GPU | Policy: $POLICY | Beam: $BW" | tee -a "$LOGFILE"
echo "Config: T=$T N=$N K=$K_ATTEMPTS E2E_RETRIES=$E2E_RETRIES" | tee -a "$LOGFILE"
echo "Kernels: ${START}-${END}" | tee -a "$LOGFILE"
echo "Output: $OUT" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "============================================" | tee -a "$LOGFILE"

TOTAL=0
PASS=0
FAIL=0
SPEEDUPS=""

for i in $(seq $START $END); do
    # Format kernel ID with proper padding (3 digits for L1/L2)
    K=$(printf "%03d" $i)
    KPATH="$BASE/tests_suite/benchmarks/kernelbench/level_${LEVEL}/kernel_${K}.py"

    # Skip if kernel file doesn't exist
    if [ ! -f "$KPATH" ]; then
        continue
    fi

    KOUT="$OUT/kernel_${K}"

    # Resume support: skip if already completed
    # Multi-kernel saves checkpoints in subdirs (e.g. opt_fused_*/pipeline_checkpoint.json)
    # Also check for final_model.py as a completion indicator
    if [ -f "$KOUT/pipeline_checkpoint.json" ] || [ -f "$KOUT/final_model.py" ]; then
        echo "[GPU $GPU] kernel_${K}: SKIP (already done)" | tee -a "$LOGFILE"
        continue
    fi

    TOTAL=$((TOTAL + 1))
    START_TIME=$(date +%s)
    echo "[GPU $GPU ${POLICY}] L${LEVEL}_kernel_${K}: START $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"

    mkdir -p "$KOUT"

    kernelopt run "$KPATH" \
        -T $T -N $N -K $K_ATTEMPTS \
        --no-run-subdir -o "$KOUT" \
        --kernel-args "--profile-only" \
        --launch-skip 0 --launch-count 10 \
        --search-policy "$POLICY" \
        --beam-width "$BW" \
        --e2e-retries "$E2E_RETRIES" \
        --multi-kernel \
        > "$KOUT/run.log" 2>&1 || true

    STATUS=$?
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))

    # Extract key results — end-to-end model speedup (eager, compile, optimized)
    RESULT="no output"
    FINAL_MODEL="$KOUT/final_model.py"
    if [ -f "$FINAL_MODEL" ]; then
        RESULT=$(python3 -c "
from kernelopt.functions.benchmark import benchmark_model_e2e
e2e = benchmark_model_e2e('$KPATH', '$FINAL_MODEL')
if not e2e.get('correct', True):
    print(f'CORRECTNESS FAILED: {e2e.get(\"error\", \"unknown\")[:80]}')
else:
    eager = e2e.get('eager_ms', -1)
    comp = e2e.get('compile_ms', -1)
    opt = e2e.get('optimized_ms', -1)
    vs_c = e2e.get('vs_compile', 0)
    vs_e = e2e.get('vs_eager', 0)
    parts = []
    if eager > 0: parts.append(f'eager={eager:.3f}ms')
    if comp > 0: parts.append(f'compile={comp:.3f}ms')
    if opt > 0: parts.append(f'opt={opt:.3f}ms')
    if vs_c > 0 and vs_c >= 1.0:
        parts.append(f'vs_compile={vs_c}x')
    elif vs_c > 0 and vs_c < 1.0:
        parts.append(f'vs_compile=1.0x (no improvement, actual={vs_c}x)')
    if vs_e > 0: parts.append(f'vs_eager={vs_e}x')
    if parts:
        print(' | '.join(parts))
    elif 'error' in e2e:
        print(f'e2e_error: {e2e.get(\"error\", \"\")[:80]}')
    else:
        print('e2e_benchmark_failed')
" 2>/dev/null || echo "e2e_benchmark_error")
        PASS=$((PASS + 1))
    else
        # Check for synthesis failure
        if grep -q "No valid kernels\|No Triton kernels" "$KOUT/run.log" 2>/dev/null; then
            RESULT="synthesis failed"
        fi
        FAIL=$((FAIL + 1))
    fi

    echo "[GPU $GPU] L${LEVEL}_kernel_${K}: ${RESULT} (${MINS}m ${SECS}s) exit=$STATUS" | tee -a "$LOGFILE"
done

echo "============================================" | tee -a "$LOGFILE"
echo "SUMMARY: Total=$TOTAL Pass=$PASS Fail=$FAIL" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "============================================" | tee -a "$LOGFILE"
