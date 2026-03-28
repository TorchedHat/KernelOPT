# Orchestrator Implementation Guide

## File Naming Pattern (from pipeline_output/)

Based on the successful pipeline run in `pipeline_output/`, the file naming follows:

### Baseline
```
baseline_output.csv
baseline_profile.ncu-rep
```

### Iteration kernels
```
softmax_iter{N}_plan{X}_{direction}.py
softmax_iter{N}_plan{X}_{direction}_output.csv
softmax_iter{N}_plan{X}_{direction}_profile.ncu-rep
```

Where:
- `{N}` = iteration number (1, 2, 3, ...)
- `{X}` = plan letter (A, B, C, D, ...)
- `{direction}` = optimization direction from plan (e.g., "pipelining", "fp_fusion", "single_pass")

### Alternative naming (early iterations)
```
softmax_plan{N}_{direction}.py
softmax_plan{N}_{direction}_output.csv
softmax_plan{N}_{direction}_profile.ncu-rep
```

All files stored in **flat** `pipeline_output/` directory - no subdirectories.

---

## NCU Profiling Command (Framework-Agnostic)

For each kernel variant to profile, use the full NCU command with kernel filtering:

```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:{kernel_pattern}' --launch-skip 0 --launch-count {N} -o {output_path} -f {execution_command}
```

This generates:
- `{output_path}_output.csv` - sections-based CSV with metrics and rules
- `{output_path}_profile.ncu-rep` - binary report for GUI analysis

### Parameters

- `--set full` - capture all metric sections (SOL, Occupancy, Memory, Warp State, etc.)
- `--replay-mode kernel` - replay individual kernel invocations for accurate profiling
- `--target-processes all` - profile all spawned processes (handles Python multiprocessing)
- `--kernel-name-base demangled` - match against demangled (human-readable) kernel names
- `--kernel-name 'regex:...'` - filter which kernels to profile (avoid profiling unrelated kernels)
- `--launch-skip 0` - start profiling from the first kernel launch
- `--launch-count N` - number of kernel launches to profile (1 for forward-only, 2 for fwd+bwd, etc.)
- `-o {path}` - output basename (NCU appends `_output.csv` and `_profile.ncu-rep`)
- `-f` - force overwrite existing files

### Examples by Framework

**Triton (forward only)**:
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_forward_kernel' --launch-skip 0 --launch-count 1 -o pipeline_output/baseline -f python softmax.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --profile-only
```

**Triton (forward + backward)**:
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_(forward|backward)_kernel' --launch-skip 0 --launch-count 2 -o pipeline_output/softmax_iter2_planA_pipelining -f python pipeline_output/softmax_iter2_planA_pipelining.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --backward --profile-only
```

**CUDA**:
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:my_cuda_kernel' --launch-skip 0 --launch-count 1 -o pipeline_output/cuda_kernel_iter1_planB -f ./cuda_kernel_opt --profile-only
```

**Helion**:
```bash
ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:helion_.*' --launch-skip 0 --launch-count 1 -o pipeline_output/helion_kernel_iter3_planC -f python helion_kernel.py --profile-only
```

---

## Kernel Requirements

Each kernel file must support `--profile-only` mode:

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true",
                       help="Run single iteration for NCU profiling")
    args = parser.parse_args()

    if args.profile_only:
        # Single forward pass for profiling
        x = torch.randn(131072, 4096, device='cuda')  # fixed size
        output = triton_softmax_forward(x)
    else:
        # Normal benchmark mode
        benchmark_kernels()
```

---

## Pipeline Flow with NCU

### Iteration 0 (Baseline)

1. User provides `softmax.py` (baseline kernel)
2. **Profile baseline**:
   ```bash
   ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_forward_kernel' --launch-skip 0 --launch-count 1 -o pipeline_output/baseline -f python softmax.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --profile-only
   ```
3. Parse `pipeline_output/baseline_output.csv` → `profiling_context_baseline.json`
4. Generate N=4 plans
5. Execute each plan → generate optimized kernels
6. **Profile each optimized kernel**:
   ```bash
   ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_forward_kernel' --launch-skip 0 --launch-count 1 -o pipeline_output/softmax_plan1_grid_align -f python pipeline_output/softmax_plan1_grid_align.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --profile-only
   # Repeat for plan2, plan3, plan4
   ```
7. Parse each CSV → `profiling_context_{plan}.json`
8. Summarize → update `optimization_memory.json`
9. Select best kernel for iteration 1

### Iteration N (N ≥ 1)

1. Start with current best kernel from previous iteration
2. Generate N=4 plans
3. Execute each plan → `softmax_iter{N}_plan{X}_{direction}.py`
4. **Profile each kernel**:
   ```bash
   ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_(forward|backward)_kernel' --launch-skip 0 --launch-count 2 -o pipeline_output/softmax_iter2_planA_pipelining -f python pipeline_output/softmax_iter2_planA_pipelining.py --kernel triton --batch 32 --sequence 4096 --hidden 256 --backward --profile-only
   ```
5. Parse CSVs → profiling contexts
6. Summarize → update memory
7. Select best kernel for next iteration

---

## Orchestrator Execution Steps

When user says "optimize my kernel":

### 1. Initialization
- Create output directory: `mkdir -p pipeline_output`
- Identify baseline kernel: `softmax.py`
- Check if baseline profiling exists:
  - If `pipeline_output/baseline_output.csv` exists → skip profiling
  - Else → run NCU profiling
- Parse baseline CSV → `profiling_context_baseline.json`

### 2. Iteration Loop (T=5)
For iter in range(T):

  **Phase 1: Plan Generation**
  - Invoke `kernel-planner` N=4 times on current kernel
  - Get N optimization plans

  **Phase 2: Kernel Generation**
  - For each plan:
    - Invoke `kernel-executor`
    - Save to `pipeline_output/softmax_iter{N}_plan{X}_{direction}.py`
    - Verify correctness

  **Phase 3: Profiling**
  - For each valid kernel:
    ```python
    kernel_file = f"pipeline_output/softmax_iter{iter}_plan{plan_letter}_{direction}.py"
    output_base = kernel_file.replace(".py", "")

    cmd = f"ncu --set full --replay-mode kernel --target-processes all --kernel-name-base demangled --kernel-name 'regex:softmax_triton_(forward|backward)_kernel' --launch-skip 0 --launch-count 2 -o {output_base} -f python {kernel_file} --kernel triton --batch 32 --sequence 4096 --hidden 256 --backward --profile-only"
    subprocess.run(cmd, shell=True, check=True)

    # This creates:
    # - {output_base}_output.csv
    # - {output_base}_profile.ncu-rep
    ```

  **Phase 4: Parse Profiling**
  - For each generated CSV:
    - Invoke `kernel-profiler`
    - Get `profiling_context_{kernel}.json`

  **Phase 5: Summarization**
  - Compare slow→fast pairs
  - Invoke `kernel-summarizer`
  - Update `optimization_memory.json`

  **Phase 6: Select Best Kernel**
  - Select fastest kernel by latency
  - Update current_kernel and best_kernel if improved

### 3. Final Output
- `pipeline_output/best_kernel.py` - copy of best performer
- `pipeline_output/optimization_memory.json` - final memory
- `pipeline_output/optimization_log.md` - iteration summary

---

## Critical Missing Piece

**The original orchestrator skill does NOT execute NCU commands.**

It says "Profile all accepted kernels → one profiling_context per kernel" but doesn't specify:
- What command to run
- Where to store the output
- How to handle the .ncu-rep files

**Fix**: Add explicit NCU execution to Phase 3 of each iteration.
