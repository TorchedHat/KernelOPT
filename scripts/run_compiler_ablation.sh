#!/bin/bash
set -uo pipefail

# Compiler Context Ablation v4
# Fixes: TRITON_CACHE_DIR on /mnt, .pyc exclusion, per-kernel timing
# Usage: bash run_compiler_ablation_v4.sh <GPU> <with|without> <L1|L2>

GPU=$1
MODE=$2
LEVEL=$3

BASE="/mnt/podman_storage/ahpoddar/kernelopt_ablation"
KPT="$BASE/kernel-profiling-tool"
KB="$BASE/KernelBench/KernelBench"
RESULTS="$BASE/compiler_ablation_v4/${MODE}_gpu${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH

# CRITICAL: Triton JIT cache on 7TB /mnt partition, NOT /home (250GB)
# Each GPU gets its own cache dir to prevent cross-process corruption
# See: triton-lang/triton#5933, python/cpython#92452
export TRITON_CACHE_DIR="/mnt/podman_storage/ahpoddar/triton_cache_gpu${GPU}"
mkdir -p "$TRITON_CACHE_DIR"

eval "$(/home/ahpoddar/.conda/bin/conda shell.bash hook 2>/dev/null)"
conda activate /mnt/podman_storage/ahpoddar/conda_envs/kopt
export PATH="/mnt/podman_storage/ahpoddar/conda_envs/kopt/bin:$PATH"
cd "$KPT"

T=5
N=4
K=4
BW=4
E2E_RETRIES=2

CONTEXT_FLAG=""
if [ "$MODE" = "without" ]; then
    CONTEXT_FLAG="--no-compiler-context"
fi

# L1: 20 kernels — mix of op types
L1_KERNELS="12 7 10 15 36 39 85 89 3 5 17 22 24 30 42 50 58 65 72 91"

# L2: 20 kernels — mix of compound ops
L2_KERNELS="18 53 9 22 64 29 67 8 3 14 25 33 41 55 60 71 80 88 95 99"

if [ "$LEVEL" = "L1" ]; then
    KERNELS="$L1_KERNELS"
    KB_LEVEL="$KB/level1"
elif [ "$LEVEL" = "L2" ]; then
    KERNELS="$L2_KERNELS"
    KB_LEVEL="$KB/level2"
else
    echo "ERROR: LEVEL must be L1 or L2"
    exit 1
fi

mkdir -p "$RESULTS"
SUMMARY_FILE="$RESULTS/summary.jsonl"
> "$SUMMARY_FILE"

echo "[GPU $GPU] $MODE context, level=$LEVEL, 20 kernels — started $(date)" | tee "$RESULTS/run.log"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR" | tee -a "$RESULTS/run.log"
echo "  CUDA_VISIBLE_DEVICES=$GPU" | tee -a "$RESULTS/run.log"

for K_NUM in $KERNELS; do
    # CRITICAL: -maxdepth 1 skips __pycache__/, -name "*.py" excludes .pyc
    KPATH=$(find "$KB_LEVEL" -maxdepth 1 -name "${K_NUM}_*.py" -type f 2>/dev/null | head -1)
    if [ -z "$KPATH" ]; then
        echo "  SKIP $LEVEL kernel $K_NUM (not found)" | tee -a "$RESULTS/run.log"
        continue
    fi

    KOUT="$RESULTS/${LEVEL}_kernel_${K_NUM}"

    # Skip if already completed (for resumability)
    if [ -f "$KOUT/timing.json" ]; then
        echo "  SKIP $LEVEL kernel $K_NUM (already completed)" | tee -a "$RESULTS/run.log"
        continue
    fi

    START_TIME=$(date +%s)
    START_TS=$(date -Iseconds)
    echo "  [$(date +%H:%M:%S)] $LEVEL kernel $K_NUM: $(basename $KPATH) — START" | tee -a "$RESULTS/run.log"

    kernelopt run "$KPATH" \
        -T $T -N $N -K $K \
        --beam-width $BW --search-policy beam \
        --e2e-retries $E2E_RETRIES \
        --e2e-perf-gate --e2e-perf-margin 1.03 \
        --e2e-correctness-timeout 180 \
        --no-run-subdir -o "$KOUT" \
        --provider vertex \
        $CONTEXT_FLAG \
        >> "$RESULTS/run.log" 2>&1
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    END_TS=$(date -Iseconds)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    # Check triton cache size
    CACHE_SIZE=$(du -sh "$TRITON_CACHE_DIR" 2>/dev/null | awk '{print $1}')
    DISK_AVAIL=$(df -h / 2>/dev/null | tail -1 | awk '{print $4}')

    echo "  [$(date +%H:%M:%S)] $LEVEL kernel $K_NUM: done (${ELAPSED_MIN}m ${ELAPSED_SEC}s, exit=$EXIT_CODE, cache=$CACHE_SIZE, disk_avail=$DISK_AVAIL)" | tee -a "$RESULTS/run.log"

    echo "{\"kernel\": $K_NUM, \"level\": \"$LEVEL\", \"mode\": \"$MODE\", \"gpu\": $GPU, \"elapsed_seconds\": $ELAPSED, \"exit_code\": $EXIT_CODE, \"start\": \"$START_TS\", \"end\": \"$END_TS\", \"kernel_file\": \"$(basename $KPATH)\", \"cache_size\": \"$CACHE_SIZE\", \"disk_avail\": \"$DISK_AVAIL\"}" > "$KOUT/timing.json"
    echo "{\"kernel\": $K_NUM, \"level\": \"$LEVEL\", \"mode\": \"$MODE\", \"gpu\": $GPU, \"elapsed_seconds\": $ELAPSED, \"exit_code\": $EXIT_CODE}" >> "$SUMMARY_FILE"
done

echo "[GPU $GPU] $MODE $LEVEL — completed $(date)" | tee -a "$RESULTS/run.log"
echo "Summary: $(wc -l < $SUMMARY_FILE) kernels completed" | tee -a "$RESULTS/run.log"
