# kernelopt Phase 2 -- Complete Status and Next Steps

## Chat Context Summary (for continuation in new chat)

This document captures the full state of Phase 2 development as of May 16, 2026.
Use this to resume work in a new chat session.

---

## What Was Built

### Feature 4: PyTorch-to-Triton Synthesis (COMPLETE)

**Files created/modified:**

| File | Status | What it does |
|------|--------|-------------|
| `kernelopt/synthesis/__init__.py` | NEW | Package exports |
| `kernelopt/synthesis/compile_extract.py` | NEW | Subprocess torch.compile + Triton extraction |
| `kernelopt/synthesis/wrap_kernel.py` | NEW | Keep Inductor intact + standalone kernel for LLM |
| `kernelopt/agents/profiler/ncu_runner.py` | MODIFIED | `_is_inductor_output()`, `_is_pytorch_source()` detection |
| `kernelopt/agents/profiler/__init__.py` | MODIFIED | `launch_count=None` for Triton, `launch_skip=0` |
| `kernelopt/agents/profiler/prompts.py` | MODIFIED | Multi-kernel awareness in profiler prompt |
| `kernelopt/agents/planner/prompts.py` | MODIFIED | Strategy-first, fusion strategies, optimization_strategy aspect |
| `kernelopt/agents/planner/tools.py` | MODIFIED | `optimization_strategy` + `multi_kernel_summary` queryable |
| `kernelopt/agents/planner/__init__.py` | MODIFIED | Strategy-aware diversity hints for all plans |
| `kernelopt/agents/executor/prompts.py` | MODIFIED | Complete file requirements, fusion awareness |
| `kernelopt/agents/correctness.py` | MODIFIED | GEMM-aware tolerance prompt + `atol=1e-3` floor |
| `kernelopt/functions/csv_parser.py` | MODIFIED | `find_bottleneck_kernel()` multi-kernel discovery |
| `kernelopt/langgraph_pipeline.py` | MODIFIED | `maybe_synthesize`, `analyze_strategy`, multi-kernel profiling |
| `kernelopt/state.py` | MODIFIED | `optimization_strategy`, `_original_pytorch_path/source` |

**Pipeline flow:**
```
Input -> detect framework -> [maybe_synthesize] -> profile_baseline (all kernels)
  -> analyze_strategy (Phase 1) -> generate_plans (Phase 2, N plans)
  -> execute_plans (K attempts) -> correctness -> [autotune_candidates (disabled)]
  -> profile_candidates -> summarize_results (writes memory bank) -> select_best
  -> [iterate or stop]
```

**Synthesis pass rates (final, Python 3.11 + torch 2.11):**

| Dataset | Pass | Total | Rate |
|---------|------|-------|------|
| KB L1 | 100 | 100 | 100% |
| KB L2 | 100 | 100 | 100% |
| KB L3 | 41 | 50 | 82% |
| KB L4 | 0 | 20 | 0% |
| SOL train | 17 | 26 | 65% |
| SOL FlashInfer | 19 | 26 | 73% |
| SOL L1 | 51 | 94 | 54% |
| SOL L2 | 32 | 82 | 39% |
| SOL Quant | 17 | 33 | 52% |
| **TOTAL** | **377** | **531** | **71.0%** |

### Feature 2: Optimization Memory Bank (COMPLETE)

**Files:**

| File | Status | What it does |
|------|--------|-------------|
| `kernelopt/memory_bank.py` | NEW | Persistent cross-run memory bank (JSON, ~/.kernelopt/) |
| `kernelopt/reference_kernels.py` | NEW | Triton tutorial reference kernel database |

**Memory bank features:**
- `MemoryEntry`: kernel_structure, hardware, strategy, outcome, speedup, failure_reason, constraint, optimization_insight, code_pattern
- `MemoryBank`: JSON file at `~/.kernelopt/memory_bank.json`, max 500 entries
- `classify_kernel_structure()`: gemm_epilogue, pointwise, reduction, softmax, layernorm, rmsnorm, attention, conv_epilogue, cuda
- Structural retrieval by kernel_type + hardware + outcome
- `format_for_llm()`: renders past experiences + code patterns as LLM context
- Reference kernels from official Triton tutorials (matmul, softmax, layernorm, attention, persistent_matmul)

**Integration:**
- `analyze_strategy`: queries failures + successes + reference kernels BEFORE deciding strategy
- `summarize_results`: writes EVERY candidate outcome (success/failure/degradation) with code patterns

### Feature 1: Decoupled Autotuner (DISABLED)

- Code exists in `autotuner.py` and pipeline node `autotune_candidates`
- Currently a pass-through (returns candidates unchanged)
- Will be validated separately after Feature 4 benchmarking

### Multi-Kernel Architecture (COMPLETE)

- `find_bottleneck_kernel()`: discovers all kernels, classifies as user/library/runtime
- Baseline profiling: profiles ALL kernels, selects user kernel for comparison
- Candidate profiling: bottleneck discovery on candidate CSV
- Strategy analysis: sees full kernel list, proposes fusion when appropriate
- Two-phase planning: Phase 1 (strategy) -> Phase 2 (N plans aligned with strategy)

### Two-Phase Planning (COMPLETE)

- `analyze_strategy` node: ONE LLM call decides macro strategy (fuse/optimize/rewrite)
- Queries memory bank for past failures/successes
- Includes reference kernel examples
- Planner diversity hints: Plan A forced to follow strategy, B/C get strategy prefix + NCU rules
- All plans see strategy context AND NCU metrics

---

## What Was Tested

### kernel_081 (KB L2: gemm + swish + div + clamp + tanh + clamp)

**Best result achieved: 1.57x speedup on Triton activation kernel (24.58µs -> 15.62µs)**

```
Iter 1: 1.48x — eliminate_tanh_branch_divergence_with_polynomial_approximation
Iter 2: 1.57x (cumulative) — store_output_as_fp16_to_enable_l2_compression
Iter 3: no improvement
```

**Fusion attempts:** Strategy correctly identified "fuse" but LLM-written GEMM failed:
- 4.7% numerical error (wrong matmul math)
- Shared memory overflow (block sizes too large)
- 580x slower (naive unoptimized GEMM)

**Root cause:** 8192x8192 GEMM exceeds Triton shared memory limits on H200. Even Inductor's own templates fail. cuBLAS IS optimal for this size.

---

## TODO 2: Benchmark Plan

### Objective

Full benchmark on ALL 531 problems (KB L1-L4 + SOL all splits):

```
Path A: PyTorch -> torch.compile -> Triton (Inductor) -> kernelopt optimize
        (for problems where synthesis succeeds -- 377/531 = 71%)
Path A': PyTorch -> kernelopt optimize directly (fallback for 154 problems)
Path B: PyTorch eager baseline (wall-clock do_bench)
Path C: torch.compile baseline (Inductor, no LLM optimization)
```

Both paths go through the FULL kpt pipeline. The only difference is whether
synthesis succeeded (Path A) or fell back to raw PyTorch (Path A').

### Execution Strategy

**Phase 1: First pass (populates memory bank)**

Run `kernelopt run` on ALL 531 problems:
- Config: T=3, N=3, K=1, model=claude-sonnet-4-6
- 8 GPUs in parallel (~66 problems per GPU)
- Memory bank starts EMPTY, grows organically
- Estimated time: ~66 problems × 15min/problem / 1 GPU = ~16 hours per GPU
- With 8 GPUs: ~16 hours total

Script: write a batch runner that runs `kernelopt run` sequentially per GPU,
collecting results in a JSON file per problem.

**Phase 2: Second pass (benefits from memory bank)**

Rerun ALL 531 problems WITH the populated memory bank:
- Strategy analysis now has past failures/successes from Phase 1
- Planner sees relevant code patterns and reference kernels
- Should produce better results than Phase 1

**Phase 3: Collect and analyze results**

For each problem, record:
- problem_id, dataset (KB L1/L2/L3/L4/SOL), kernel_type
- synthesis_status (pass/fail -- which path was taken)
- n_triton_kernels, has_extern_kernels
- baseline_latency_ns, best_latency_ns, speedup
- strategy_type, plans_tried, correctness_pass_rate
- best_direction (what optimization worked)
- memory_bank_entries_added

Aggregate metrics:
- Median speedup (synthesized vs fallback, overall)
- % of problems where kpt achieved >1.0x, >1.2x, >1.5x
- Synthesis path vs fallback path comparison
- Per-dataset breakdown
- Memory bank growth curve
- LLM call count and cost

**Phase 4: Compare against KernelAgent**

Run KernelAgent on the same problems for direct comparison.

### GPU Allocation

```
GPU 0: KB L1 (100 problems)
GPU 1: KB L2 (100 problems)
GPU 2: KB L3 + L4 (70 problems)
GPU 3: SOL train + FlashInfer (52 problems)
GPU 4: SOL L1 first half (47 problems)
GPU 5: SOL L1 second half (47 problems)
GPU 6: SOL L2 (82 problems)
GPU 7: SOL Quant (33 problems)
```

### Output

- `eval_benchmark_results/phase1/` -- first pass results + populated memory bank
- `eval_benchmark_results/phase2/` -- second pass results (with memory)
- `~/.kernelopt/memory_bank.json` -- starter bank to ship with tool
- `docs/benchmark_results.md` -- full analysis and comparison tables

---

## Known Limitations

1. **Fusion quality**: LLM-written Triton GEMM is unreliable (4.7% error, shared memory overflow). For large GEMMs (>4096x4096), cuBLAS is optimal. Memory bank learns this.

2. **RNN/LSTM/GRU**: torch.compile can't trace (PyTorch issue #158007). Graceful degradation.

3. **KB L4**: Complex multi-model pipelines exceed Dynamo tracing limits. WIP.

4. **SOL-ExecBench**: 39-73% pass rate depending on split. Complex backward passes and FP8 ops.

5. **Autotuner**: Disabled pending validation. Feature 1 needs separate benchmarking.

---

## GitLab

- **Branch**: `feature4/pytorch-synthesis` (clean, on `origin/main`)
- **Working branch**: `phase2/feature4-pytorch-synthesis` (includes autotuner from phase2/research)
- **MR**: https://gitlab.com/Sanskaaar1/kernel-profiling-tool/-/merge_requests/28
- **Remote**: git@gitlab.com:Sanskaaar1/kernel-profiling-tool.git

---

## Key Design Decisions (for reference)

1. **Synthesis**: Keep Inductor output INTACT as execution engine. Don't disassemble. Append kernelopt interface.

2. **Measurement**: Profile user Triton kernel for baseline comparison. Show ALL kernels to planner for strategy. NCU for hardware analysis, not for speedup measurement.

3. **Strategy**: Two-phase planning. Phase 1 decides macro strategy (fuse/optimize). Phase 2 generates N plans aligned with strategy. Memory bank prevents repeating failures.

4. **Correctness**: `atol=1e-3` floor for large tensors (GEMM chains). Matches KernelAgent default.

5. **Memory**: Persistent JSON bank at `~/.kernelopt/memory_bank.json`. Ships with starter from benchmarks. Grows organically from every run.

6. **No library filtering**: The planner sees ALL kernels (cuBLAS, cuDNN, user) and decides strategy. We don't silently filter.

7. **Holistic changes**: All modifications work for ANY input (PyTorch, Triton, CUDA, Helion). No special-casing for synthesized files.

---

## How to Resume

1. Read this document
2. Check git status: `cd kernelopt && git log --oneline -10`
3. Check memory bank: `cat ~/.kernelopt/memory_bank.json 2>/dev/null | python3 -m json.tool | head -20`
4. The next step is TODO 2 benchmark -- write the batch runner script and launch on GPUs 2-7
