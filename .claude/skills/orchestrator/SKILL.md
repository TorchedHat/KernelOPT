---
name: kernel-opt-orchestrator
description: >
  Orchestrate the full GPU kernel optimization pipeline across multiple iterations.
  Works for any kernel framework — CUDA, Triton, Helion, or other.
  Use this skill whenever a user wants to run the full automated optimization loop,
  improve a kernel end-to-end, or coordinate the profiler/planner/executor/summarizer
  agents together. Trigger on: "optimize my kernel", "run the full pipeline", "improve
  kernel performance", or "start the optimization loop".
---

# Kernel Optimization Orchestrator

You coordinate the four sub-skills in an iterative loop to improve a kernel.
The pipeline is framework-agnostic — it works for CUDA, Triton, Helion, or any GPU language
that NCU can profile.

---

## Pipeline

```
User kernel  +  output.csv
       ↓
[kernel-profiler]     →  profiling_context JSON
       ↓
[kernel-planner] × N  →  N optimization_plan JSONs
       ↓
[kernel-executor] × K →  optimized_kernel + execution_result (per plan)
       ↓
Correctness gate       →  reject status != "success"
       ↓
Profile valid kernels  →  profiling_context for each
       ↓
[kernel-summarizer]    →  experience_items → update memory queue
       ↓
Select best kernel     →  keep best by latency → next iteration
       ↓
Repeat T iterations (or until user stops)
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `T` | 5 | Iterations (increase for deeper search) |
| `N` | 4 | Plans generated per iteration |
| `K` | 1 | Executor attempts per plan |
| `TopK` | 4 | Max experience items added to memory per iteration |
| `ExpN` | 8 | Memory queue capacity |
| `t_pos` | 1.04 | Min speedup to store a positive experience item |
| `t_neg` | 1.15 | Min slowdown factor to store a negative experience item |

Minimal smoke-test run: `T=1, N=2, K=1`.

---

## Before iteration 1

Ask the user for:
1. The kernel source file (any framework)
2. An NCU CSV (`output.csv`) for the baseline kernel
3. A correctness harness (optional but recommended)
4. The target framework if not obvious from the source

**Setup**:
```bash
# Create output directory (if it doesn't exist)
mkdir -p pipeline_output
```

Initialize:
- `current_kernel = baseline_kernel`
- `memory = []`
- `best_kernel = baseline_kernel`
- `best_latency = baseline_latency` (from `GPU Speed Of Light Throughput → Duration`)

---

## Each iteration

### Phase 1 — Profile
Invoke **kernel-profiler** on the current kernel → get `profiling_context`.

### Phase 2 — Plan
Invoke **kernel-planner** N times (with varied context order for diversity) →
collect N `optimization_plan` blocks.

### Phase 3 — Execute
For each plan, invoke **kernel-executor** K times → collect `optimized_kernel` +
`execution_result`. Reject any with `ready_for_profiling: false`.

### Phase 4 — Profile generated kernels
For each accepted kernel, run NCU profiling to generate CSV and .ncu-rep files.

**NCU Command Format** (framework-agnostic):

```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:{kernel_pattern}' --launch-skip 0 --launch-count {N} -o {output_path} -f {execution_command}
```

**Examples**:

Triton kernel (forward only):
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_forward_kernel' --launch-skip 0 --launch-count 1 -o pipeline_output/baseline -f python softmax.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --profile-only
```

Triton kernel (forward + backward):
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_(forward|backward)_kernel' --launch-skip 0 --launch-count 2 -o pipeline_output/softmax_iter2_planA_pipelining -f python pipeline_output/softmax_iter2_planA_pipelining.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --backward --profile-only
```

CUDA kernel:
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:my_cuda_kernel' --launch-skip 0 --launch-count 1 -o pipeline_output/cuda_kernel_iter1_planB_tiling -f ./cuda_kernel --profile-only
```

Helion kernel:
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:helion_.*' --launch-skip 0 --launch-count 1 -o pipeline_output/helion_kernel_iter3_planC_fusion -f python helion_kernel.py --profile-only
```

**Flags explained**:
- `--set full` - capture all metric sections (SOL, Occupancy, Memory, etc.)
- `--replay-mode kernel` - replay individual kernel invocations for accurate profiling
- `--target-processes all` - profile all spawned processes
- `--kernel-name-base demangled` - match against demangled kernel names
- `--kernel-name 'regex:...'` - filter kernels by regex pattern
- `--launch-skip 0` - start profiling from first kernel launch
- `--launch-count N` - profile N kernel launches (use 1 for forward-only, 2 for fwd+bwd)
- `-o {path}` - output basename (generates `{path}_output.csv` and `{path}_profile.ncu-rep`)
- `-f` - force overwrite existing files

**This generates**:
- `{output_path}_output.csv` - sections-based CSV with metrics and NCU rules
- `{output_path}_profile.ncu-rep` - binary report for Nsight Compute GUI

Then parse each CSV with **kernel-profiler** → one `profiling_context` per kernel.

**File naming pattern** (from pipeline_output/):
- Baseline: `baseline_output.csv`, `baseline_profile.ncu-rep`
- Iterations: `{kernel}_iter{N}_plan{X}_{direction}_output.csv`, `_profile.ncu-rep`
- All files stored in **flat** `pipeline_output/` directory (no subdirectories)

**Framework Detection**:
The orchestrator must infer the kernel name pattern from the framework:
- Triton: `regex:.*triton.*` or specific function names
- CUDA: `regex:.*` or specific kernel names from source
- Helion: `regex:helion_.*` or specific patterns
- PyTorch: `regex:.*at::.*` for ATen kernels

### Phase 5 — Summarize
Sort kernel pairs by |speedup − 1.0| descending.
Take top `TopK` pairs that meet the curation thresholds.
Invoke **kernel-summarizer** for each → apply `memory_update` actions.

### Phase 6 — Select best kernel
```
1. Keep only the fastest correct kernel from all N plans
2. If faster than current_kernel: update current_kernel
3. Update best_kernel and best_latency if this is the fastest ever seen
```

---

## Progress display

After each iteration:

```
Iteration 2/5  |  Framework: triton  |  Best: 1.31×  |  Memory items: 5
──────────────────────────────────────────────────────────────────────────
Direction                   | Speedup | Correctness | Kept?
────────────────────────────┼─────────┼─────────────┼──────
reduce_register_pressure    | 1.31×   | passed      | ✓
increase_tile_k             | 0.94×   | passed      | ✗
fuse_fp_instructions        | 1.08×   | passed      | ✓
adjust_grid_size            | fail    | failed      | ✗
```

---

## Stopping conditions

Stop when any of these are true:
- All `T` iterations are complete
- The user asks to stop
- Two consecutive iterations produce no improvement
- All NCU rule opportunities have been addressed (planner reports low confidence
  on all generated plans)

---

## Output artifacts

At the end, produce:
1. `best_kernel.<ext>` — best kernel source in the original framework's extension
2. `optimization_memory.json` — final memory queue (reusable for future kernels)
3. `optimization_log.md` — per-iteration table of plans tried, speedups, and outcomes

---

## Sub-skills required

This skill requires: `kernel-profiler`, `kernel-planner`, `kernel-executor`,
`kernel-summarizer`. If any are missing, list which ones need installing first.
