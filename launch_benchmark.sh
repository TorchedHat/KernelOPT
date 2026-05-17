#!/usr/bin/env bash
#
# Launch Phase 2 benchmark in tmux.
# Usage:
#   bash launch_benchmark.sh [phase] [gpu_list]
#   bash launch_benchmark.sh 1 2,3,4         # Phase 1 on GPUs 2,3,4
#   bash launch_benchmark.sh 1 2,3,4,5,6,7   # Phase 1 on GPUs 2-7
#   bash launch_benchmark.sh 2 2,3,4         # Phase 2 on GPUs 2,3,4

set -euo pipefail

PHASE="${1:-1}"
GPU_LIST="${2:-2,3,4}"
SESSION="kpt_phase${PHASE}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/usr/bin/python3.11"
export PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda-12.8/bin:$PATH"
T=4
N=3
K=2
MODEL="claude-sonnet-4-6"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "============================================"
echo "  kernelopt Phase ${PHASE} Benchmark Launch"
echo "  GPUs: ${GPU_LIST} (${NUM_GPUS} total)"
echo "  Config: T=${T} N=${N} K=${K}"
echo "  Session: ${SESSION}"
echo "============================================"

# GPU assignment map — distributes 531 problems across available GPUs
# With 3 GPUs (2,3,4):
#   GPU 2: kb_l1 + kb_l2                     = 200 problems
#   GPU 3: kb_l3 + kb_l4 + sol_train + sol_flashinfer = 122 problems
#   GPU 4: sol_l1_lo + sol_l1_hi + sol_l2 + sol_quant = 209 problems
#
# With 6 GPUs (2,3,4,5,6,7):
#   GPU 2: kb_l1                              = 100 problems
#   GPU 3: kb_l2                              = 100 problems
#   GPU 4: kb_l3 + kb_l4                      = 70 problems
#   GPU 5: sol_train + sol_flashinfer + sol_quant = 85 problems
#   GPU 6: sol_l1_lo + sol_l1_hi              = 94 problems
#   GPU 7: sol_l2                             = 82 problems
#
# With 8 GPUs (0-7):
#   GPU 0: kb_l1                              = 100 problems
#   GPU 1: kb_l2                              = 100 problems
#   GPU 2: kb_l3 + kb_l4                      = 70 problems
#   GPU 3: sol_train + sol_flashinfer         = 52 problems
#   GPU 4: sol_l1_lo                          = 47 problems
#   GPU 5: sol_l1_hi                          = 47 problems
#   GPU 6: sol_l2                             = 82 problems
#   GPU 7: sol_quant                          = 33 problems

get_assignments() {
    local gpu=$1
    local num_gpus=$2

    if [ "$num_gpus" -le 3 ]; then
        case "$gpu" in
            "${GPUS[0]}") echo "kb_l1,kb_l2" ;;
            "${GPUS[1]}") echo "kb_l3,kb_l4,sol_train,sol_flashinfer" ;;
            "${GPUS[2]}") echo "sol_l1_lo,sol_l1_hi,sol_l2,sol_quant" ;;
        esac
    elif [ "$num_gpus" -le 6 ]; then
        case "$gpu" in
            "${GPUS[0]}") echo "kb_l1" ;;
            "${GPUS[1]}") echo "kb_l2" ;;
            "${GPUS[2]}") echo "kb_l3,kb_l4" ;;
            "${GPUS[3]}") echo "sol_train,sol_flashinfer,sol_quant" ;;
            "${GPUS[4]}") echo "sol_l1_lo,sol_l1_hi" ;;
            "${GPUS[5]}") echo "sol_l2" ;;
        esac
    else
        case "$gpu" in
            "${GPUS[0]}") echo "kb_l1" ;;
            "${GPUS[1]}") echo "kb_l2" ;;
            "${GPUS[2]}") echo "kb_l3,kb_l4" ;;
            "${GPUS[3]}") echo "sol_train,sol_flashinfer" ;;
            "${GPUS[4]}") echo "sol_l1_lo" ;;
            "${GPUS[5]}") echo "sol_l1_hi" ;;
            "${GPUS[6]}") echo "sol_l2" ;;
            "${GPUS[7]}") echo "sol_quant" ;;
        esac
    fi
}

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 0.5

# Create session with first GPU
FIRST_GPU="${GPUS[0]}"
FIRST_ASSIGNMENTS=$(get_assignments "$FIRST_GPU" "$NUM_GPUS")
FIRST_CMD="export PYTHONUNBUFFERED=1 PATH=/usr/local/cuda-12.8/bin:\$PATH && cd ${PROJ_DIR} && ${PYTHON} eval_benchmark.py --gpu ${FIRST_GPU} --assignments '${FIRST_ASSIGNMENTS}' --phase ${PHASE} -T ${T} -N ${N} -K ${K} --model ${MODEL} 2>&1 | tee eval_benchmark_results/phase${PHASE}/gpu${FIRST_GPU}/run.log"

tmux new-session -d -s "$SESSION" -n "gpu${FIRST_GPU}" "$FIRST_CMD"

# Create windows for remaining GPUs
for idx in $(seq 1 $((NUM_GPUS - 1))); do
    GPU="${GPUS[$idx]}"
    ASSIGNMENTS=$(get_assignments "$GPU" "$NUM_GPUS")
    if [ -z "$ASSIGNMENTS" ]; then
        continue
    fi

    # Ensure output dir exists for tee
    mkdir -p "${PROJ_DIR}/eval_benchmark_results/phase${PHASE}/gpu${GPU}"

    CMD="export PYTHONUNBUFFERED=1 PATH=/usr/local/cuda-12.8/bin:\$PATH && cd ${PROJ_DIR} && ${PYTHON} eval_benchmark.py --gpu ${GPU} --assignments '${ASSIGNMENTS}' --phase ${PHASE} -T ${T} -N ${N} -K ${K} --model ${MODEL} 2>&1 | tee eval_benchmark_results/phase${PHASE}/gpu${GPU}/run.log"
    tmux new-window -t "$SESSION" -n "gpu${GPU}" "$CMD"
done

echo ""
echo "Launched tmux session: ${SESSION}"
echo "  Attach: tmux attach -t ${SESSION}"
echo "  List windows: tmux list-windows -t ${SESSION}"
echo ""
echo "GPU assignments:"
for GPU in "${GPUS[@]}"; do
    A=$(get_assignments "$GPU" "$NUM_GPUS")
    if [ -n "$A" ]; then
        echo "  GPU ${GPU}: ${A}"
    fi
done
echo ""
echo "Monitor progress:"
echo "  watch -n 60 'for f in eval_benchmark_results/phase${PHASE}/gpu*/results.json; do echo \$f; python3 -c \"import json; d=json.load(open(\\\"\$f\\\")); print(f\\\"  done={len(d)} pass={sum(1 for r in d if r.get(\\\\\\\"status\\\\\\\")==\\\\\\\"pass\\\\\\\")}\\\")\"; done'"
