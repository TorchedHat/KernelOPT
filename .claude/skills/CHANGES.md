# Skills Update - NCU Profiling Integration

## Problem Identified

The original orchestrator skill said "Profile all accepted kernels → one profiling_context per kernel" but did NOT specify:
- **What command to run** - no actual NCU execution
- **Where to store files** - subdirectories vs flat structure
- **What flags to use** - simple `--export` vs full profiling command

User feedback: "there was no profiler ran i dont see any csv being generated"

## Changes Made

### 1. Orchestrator Skill (`.claude/skills/orchestrator/SKILL.md`)

**Added Phase 4 - Profile generated kernels** with complete NCU command structure:

```bash
ncu --set full --csv \
    --replay-mode kernel \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name 'regex:{kernel_pattern}' \
    --launch-skip 0 \
    --launch-count {N} \
    -o {output_path} \
    -f \
    {execution_command}
```

**Why these flags**:
- `--replay-mode kernel` - accurate per-kernel profiling (not averaged)
- `--target-processes all` - handles Python multiprocessing
- `--kernel-name-base demangled` - human-readable kernel names
- `--kernel-name 'regex:...'` - filter specific kernels (avoid profiling PyTorch internals)
- `--launch-skip 0 --launch-count N` - control which kernel launches to profile
- `-o {path}` - output basename (generates `{path}_output.csv` and `{path}_profile.ncu-rep`)
- `-f` - force overwrite

**Framework examples**:
- Triton: `--kernel-name 'regex:softmax_triton_forward_kernel'`
- CUDA: `--kernel-name 'regex:my_cuda_kernel'`
- Helion: `--kernel-name 'regex:helion_.*'`
- PyTorch: `--kernel-name 'regex:.*at::.*'`

### 2. File Structure Correction

**WRONG** (my earlier implementation):
```
pipeline_output/
  iteration_0/
    kernel_0_0.py
    plan_0_0.json
  iteration_1/
    ...
```

**CORRECT** (from restored pipeline_output/):
```
pipeline_output/   # FLAT directory
  baseline_output.csv
  baseline_profile.ncu-rep
  softmax_iter2_planA_pipelining.py
  softmax_iter2_planA_pipelining_output.csv
  softmax_iter2_planA_pipelining_profile.ncu-rep
  optimization_memory.json
```

**File naming pattern**:
- Baseline: `baseline_output.csv`, `baseline_profile.ncu-rep`
- Iterations: `{kernel}_iter{N}_plan{X}_{direction}.py`, `_output.csv`, `_profile.ncu-rep`
- N = iteration number (1, 2, 3, ...)
- X = plan letter (A, B, C, D, ...)
- direction = optimization name from plan

### 3. Memory Updates (`memory/MEMORY.md`)

Updated with:
- Correct flat file structure
- NCU profiling command examples
- Framework-agnostic approach
- File naming pattern documentation

### 4. Reference Documentation

Created `orchestrator_impl.md` with:
- Complete NCU command breakdown
- Per-framework examples
- Kernel requirements (--profile-only flag)
- Pipeline execution flow

## Framework Agnostic Support

The pipeline now explicitly supports:
- **Triton** - Python-based GPU kernels
- **CUDA** - C/C++ CUDA kernels
- **Helion** - Helion framework kernels
- **PyTorch** - ATen kernels (e.g., `at::native::...`)
- **Any other** - generic regex pattern matching

Each framework gets appropriate:
- Kernel name regex pattern
- Execution command format
- Launch count (1 for forward-only, 2 for fwd+bwd, etc.)

## What This Fixes

Before:
- ❌ No actual profiling happening
- ❌ No CSV files generated
- ❌ Wrong directory structure (subdirectories)
- ❌ Profiler couldn't re-run on generated kernels

After:
- ✅ Explicit NCU profiling commands in orchestrator
- ✅ CSV and .ncu-rep files generated for each kernel
- ✅ Flat pipeline_output/ structure matching original
- ✅ Profiler can parse generated CSVs for next iteration
- ✅ Framework-agnostic (CUDA/Triton/Helion/etc.)

## Ready for Production

When user says **"optimize my kernel"**, the orchestrator will:
1. Profile baseline with NCU → `baseline_output.csv`
2. Parse CSV → `profiling_context_baseline.json`
3. Generate N=4 plans → `plan_*.json`
4. Execute plans → `softmax_iter{N}_plan{X}_{direction}.py`
5. **Profile each kernel with NCU** → `*_output.csv`, `*_profile.ncu-rep`
6. Parse CSVs → profiling contexts
7. Summarize → update `optimization_memory.json`
8. Select best kernel
9. Repeat for T=5 iterations

All files stored in **flat** `pipeline_output/` with correct naming pattern.
