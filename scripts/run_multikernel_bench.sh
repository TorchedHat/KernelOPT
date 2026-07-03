#!/bin/bash
# Run multi-kernel optimization on top-3 multi-kernel problems
# T=3 iterations, N=3 plans, K=2 attempts, UCB search
set -e

OUTDIR=pipeline_output/multikernel_bench
mkdir -p $OUTDIR
BENCH_PY=/workspaces/pytorch-devcontainers/TOOL/.bench_venv/bin/python3
DEV_PY=python3

PROBLEMS=(
  "kb_l3_005|tests_suite/benchmarks/kernelbench/level_3/kernel_005.py"
  "sol_l1_003|pipeline_output/benchmark_suite/sol_wrappers/sol_l1_003.py"
  "kb_l2_005|tests_suite/benchmarks/kernelbench/level_2/kernel_005.py"
)

time_pytorch() {
  local file=$1 mode=$2
  local compile_line=""
  [ "$mode" = "compile" ] && compile_line="model = torch.compile(model)"
  $BENCH_PY -c "
import json, torch, importlib.util, sys
torch.cuda.set_device(0)
spec = importlib.util.spec_from_file_location('m', '$file')
mod = importlib.util.module_from_spec(spec); sys.modules['m'] = mod; spec.loader.exec_module(mod)
Model = getattr(mod, 'Model', None)
get_ii = getattr(mod, 'get_init_inputs', lambda: [])
get_i = getattr(mod, 'get_inputs', None)
if not Model or not get_i: print(json.dumps({'error':'missing'})); sys.exit(0)
model = Model(*get_ii()).cuda().eval()
$compile_line
for _ in range(5):
    inp = [t.cuda() if isinstance(t,torch.Tensor) else t for t in get_i()]
    with torch.no_grad(): model(*inp)
    torch.cuda.synchronize()
import time
times = []
for _ in range(20):
    inp = [t.cuda() if isinstance(t,torch.Tensor) else t for t in get_i()]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(): model(*inp)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(json.dumps({'median_ms': round(times[len(times)//2], 3)}))
" 2>/dev/null || echo '{"error":"failed"}'
}

time_synth() {
  local file=$1
  $DEV_PY -c "
import json, torch, importlib.util, sys, time
torch.cuda.set_device(0)
spec = importlib.util.spec_from_file_location('m', '$file')
mod = importlib.util.module_from_spec(spec); sys.modules['m'] = mod; spec.loader.exec_module(mod)
fn = getattr(mod, 'kernel_function', None) or getattr(mod, 'call', None)
gi = getattr(mod, 'get_inputs', None)
if not fn or not gi: print(json.dumps({'error':'missing'})); sys.exit(0)
for _ in range(5):
    inp = [t.clone() if isinstance(t,torch.Tensor) else t for t in gi()]
    with torch.no_grad(): fn(*inp)
    torch.cuda.synchronize()
times = []
for _ in range(20):
    inp = [t.clone() if isinstance(t,torch.Tensor) else t for t in gi()]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(): fn(*inp)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(json.dumps({'median_ms': round(times[len(times)//2], 3)}))
" 2>/dev/null || echo '{"error":"failed"}'
}

for entry in "${PROBLEMS[@]}"; do
  IFS='|' read -r name file <<< "$entry"
  echo ""
  echo "================================================================"
  echo "  $name (multi-kernel, T=3, N=3, K=2, UCB)"
  echo "================================================================"
  echo "  Started: $(date)"

  if [ ! -f "$file" ]; then
    echo "  MISSING: $file"
    continue
  fi

  # 1. Eager wall-clock
  printf "  Eager:    "
  eager_json=$(time_pytorch "$file" "eager")
  eager=$(echo "$eager_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('median_ms',-1))" 2>/dev/null || echo "-1")
  echo "${eager} ms"

  # 2. Compile wall-clock
  printf "  Compile:  "
  compile_json=$(time_pytorch "$file" "compile")
  compile=$(echo "$compile_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('median_ms',-1))" 2>/dev/null || echo "-1")
  echo "${compile} ms"

  # 3. Run multi-kernel pipeline
  pipedir=$OUTDIR/$name
  mkdir -p $pipedir
  echo "  Pipeline: running multi-kernel (T=3, N=3, K=2, UCB)..."
  kernelopt run "$file" \
    -T 3 -N 3 -K 2 \
    --multi-kernel \
    --no-run-subdir -o "$pipedir" \
    --kernel-args '--profile-only' \
    --launch-skip 0 --launch-count 10 \
    --search-policy ucb \
    2>&1 | tee "$pipedir/pipeline.log"

  # 4. Time synth (baseline compiled model)
  synth_file="$pipedir/synthesis/synthesized_triton.py"
  synth_ms="-1"
  if [ -f "$synth_file" ]; then
    printf "  Synth:    "
    synth_json=$(time_synth "$synth_file")
    synth_ms=$(echo "$synth_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('median_ms',-1))" 2>/dev/null || echo "-1")
    echo "${synth_ms} ms"
  fi

  # 5. Time final optimized model (last restitched or final_model)
  best_ms="-1"
  final_model=$(ls "$pipedir"/final_model*.py 2>/dev/null | tail -1)
  [ -z "$final_model" ] && final_model=$(ls "$pipedir"/*restitched*.py 2>/dev/null | tail -1)
  if [ -n "$final_model" ]; then
    printf "  Optimized: "
    best_json=$(time_synth "$final_model")
    best_ms=$(echo "$best_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('median_ms',-1))" 2>/dev/null || echo "-1")
    echo "${best_ms} ms ($(basename $final_model))"
  else
    echo "  Optimized: no final model produced"
  fi

  # 6. Summary
  echo ""
  echo "  ┌─────────────────────────────────────────────┐"
  echo "  │ RESULTS: $name"
  echo "  ├─────────────────────────────────────────────┤"
  printf "  │ Eager:     %8s ms\n" "$eager"
  printf "  │ Compile:   %8s ms\n" "$compile"
  printf "  │ Synth:     %8s ms\n" "$synth_ms"
  printf "  │ Optimized: %8s ms\n" "$best_ms"
  vs_eager=$(python3 -c "e=$eager; b=$best_ms; print(round(e/b,2) if b>0 and e>0 else 0)" 2>/dev/null || echo "0")
  vs_compile=$(python3 -c "c=$compile; b=$best_ms; print(round(c/b,2) if b>0 and c>0 else 0)" 2>/dev/null || echo "0")
  opt_sp=$(python3 -c "s=$synth_ms; b=$best_ms; print(round(s/b,2) if b>0 and s>0 else 0)" 2>/dev/null || echo "0")
  printf "  │ vs_eager:  %8sx\n" "$vs_eager"
  printf "  │ vs_compile:%8sx\n" "$vs_compile"
  printf "  │ opt_sp:    %8sx\n" "$opt_sp"
  echo "  └─────────────────────────────────────────────┘"
  echo "  Finished: $(date)"
  echo ""
done

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "================================================================"
