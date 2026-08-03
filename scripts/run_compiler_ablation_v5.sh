#!/bin/bash
set -uo pipefail

# Compiler Context Ablation v5 — Stratified Kernel Selection
# Targets reductions, norms, compound fusions where context should matter
# Usage: bash run_compiler_ablation_v5.sh <GPU> <with|without> <L1|L2>

GPU=$1
MODE=$2
LEVEL=$3

BASE="/mnt/podman_storage/ahpoddar/kernelopt_ablation"
KPT="$BASE/kernel-profiling-tool"
KB="$BASE/KernelBench/KernelBench"
RESULTS="$BASE/compiler_ablation_v5/${MODE}_gpu${GPU}"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export PATH="/mnt/podman_storage/ahpoddar/conda_envs/kopt/bin:/usr/local/cuda-12.8/bin:$PATH"

# Triton cache: isolated per GPU, on /mnt (7TB), not /home (250GB)
# See: triton-lang/triton#5933, python/cpython#92452
export TRITON_CACHE_DIR="/mnt/podman_storage/ahpoddar/triton_cache_v5_gpu${GPU}"
mkdir -p "$TRITON_CACHE_DIR"

eval "$(/home/ahpoddar/.conda/bin/conda shell.bash hook 2>/dev/null)"
conda activate /mnt/podman_storage/ahpoddar/conda_envs/kopt
cd "$KPT"

# Matching original benchmark config (T=5, N=4, K=4, beam_width=4)
T=5
N=4
K=4
BW=4
E2E_RETRIES=2

# WITH: explicitly enable compiler context (default is OFF)
# WITHOUT: no flag needed (default is OFF)
CONTEXT_FLAG=""
if [ "$MODE" = "with" ]; then
    CONTEXT_FLAG="--use-compiler-context"
fi

# L1: 10 reduction/norm kernels where compiler context should matter
# Softmax, LogSoftmax, BatchNorm, InstanceNorm, GroupNorm, RMSNorm,
# LayerNorm, Sum reduction, MinGPTNewGelu, ScaledDotProductAttention
L1_KERNELS="23 24 33 34 35 36 40 47 88 97"

# L2: 20 compound kernels (10 optimized + 10 norm chains)
# Optimized (>1.0x in original benchmark):
#   18(33.11x) 22(7.18x) 55(5.46x) 64(8.30x) 80(5.78x)
#   53(1.60x) 9(1.47x) 40(1.51x) 70(1.43x) 8(1.19x)
# Norm chains:
#   3(LN) 11(BN+GN) 34(LN) 37(GN) 60(GN)
#   88(GN) 92(GN+LSE) 94(GN) 97(BN) 28(IN)
L2_KERNELS="18 22 55 64 80 53 9 40 70 8 3 11 34 37 60 88 92 94 97 28"

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

echo "[GPU $GPU] $MODE context, level=$LEVEL — started $(date)" | tee "$RESULTS/run.log"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR" | tee -a "$RESULTS/run.log"
echo "  CUDA_VISIBLE_DEVICES=$GPU" | tee -a "$RESULTS/run.log"
echo "  CONTEXT_FLAG=$CONTEXT_FLAG" | tee -a "$RESULTS/run.log"
echo "  Kernels: $KERNELS" | tee -a "$RESULTS/run.log"

for K_NUM in $KERNELS; do
    # -maxdepth 1 skips __pycache__/, -name "*.py" excludes .pyc
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

    CACHE_SIZE=$(du -sh "$TRITON_CACHE_DIR" 2>/dev/null | awk '{print $1}')
    DISK_AVAIL=$(df -h / 2>/dev/null | tail -1 | awk '{print $4}')

    echo "  [$(date +%H:%M:%S)] $LEVEL kernel $K_NUM: done (${ELAPSED_MIN}m ${ELAPSED_SEC}s, exit=$EXIT_CODE, cache=$CACHE_SIZE, disk_avail=$DISK_AVAIL)" | tee -a "$RESULTS/run.log"

    echo "{\"kernel\": $K_NUM, \"level\": \"$LEVEL\", \"mode\": \"$MODE\", \"gpu\": $GPU, \"elapsed_seconds\": $ELAPSED, \"exit_code\": $EXIT_CODE, \"start\": \"$START_TS\", \"end\": \"$END_TS\", \"kernel_file\": \"$(basename $KPATH)\", \"cache_size\": \"$CACHE_SIZE\", \"disk_avail\": \"$DISK_AVAIL\"}" > "$KOUT/timing.json"
    echo "{\"kernel\": $K_NUM, \"level\": \"$LEVEL\", \"mode\": \"$MODE\", \"gpu\": $GPU, \"elapsed_seconds\": $ELAPSED, \"exit_code\": $EXIT_CODE}" >> "$SUMMARY_FILE"
done

echo "[GPU $GPU] $MODE $LEVEL — completed $(date)" | tee -a "$RESULTS/run.log"
echo "Summary: $(wc -l < $SUMMARY_FILE) kernels completed" | tee -a "$RESULTS/run.log"
