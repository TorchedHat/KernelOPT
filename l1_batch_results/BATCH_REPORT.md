# KernelBench Level 1 Batch Optimization Report

**Date:** 2026-06-01
**GPU:** NVIDIA H200 (143 GB VRAM)
**Backend:** Claude on Vertex AI (claude-sonnet-4-6)
**Pipeline config:** T=3 iterations, N=3 plans/iter, K=3 attempts/plan, TopK=4

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Total kernels attempted | 33 |
| Successful (status=ok) | 15 (45.5%) |
| Failed (status=fail) | 18 (54.5%) |
| Kernels with speedup > 1.0x | 8 of 15 (53.3%) |
| Kernels with speedup = 1.0x | 7 of 15 (46.7%) |
| Average speedup (passing, > 1.0x only) | 2.24x |
| Geometric mean speedup (all passing) | 1.30x |
| Best single-kernel speedup | 4.92x (kernel 012) |
| Total wall-clock time (all 33 kernels) | ~8.7 hours |

**Key findings:**
- The pipeline achieves meaningful speedups (>1.5x) on 5 out of 33 kernels, with the strongest gains on elementwise/pointwise kernels with suboptimal tile sizes.
- 18 kernels failed entirely, dominated by two systematic failure modes: **standalone build failures** for template (triton_tem) kernels (TypeError: wrong argument count), and **torch.compile synthesis failures** for transposed convolutions (NoValidChoicesError).
- The last 4 kernels (090-099) failed due to a **corrupted torch environment** (circular import), not a pipeline bug.

---

## 2. Results Table

| ID | Problem Description | Status | Baseline | Best | Speedup | Failure Mode |
|---|---|---|---|---|---|---|
| 003 | Batched matrix multiply (BMM) | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 006 | MatMul with large K dimension | OK | 3040 ns | 2940 ns | 1.03x | -- |
| 009 | Tall-skinny matrix multiply | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 012 | Diagonal matrix multiply | OK | 82750 ns | 16830 ns | 4.92x | -- |
| 015 | Lower triangular matmul | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 018 | Transposed matmul (A^T * B^T) | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 021 | Sigmoid activation | OK | 7.73 ms | 7.73 ms | 1.0x | -- |
| 024 | LogSoftmax activation | OK | 5.01 ms | 5.01 ms | 1.0x | -- |
| 027 | SELU activation (ELU variant) | OK | 7.73 ms | 3.01 ms | 2.57x | -- |
| 030 | Softsign activation | FAIL | -- | -- | -- | Standalone build (compile crash) |
| 033 | Batch Normalization (inference) | OK | 5.16 ms | 2.04 ms | 2.53x | -- |
| 036 | RMS Normalization | OK | 9.03 ms | 3.62 ms | 2.49x | -- |
| 039 | L2 Normalization | OK | 12.35 ms | 6.74 ms | 1.83x | -- |
| 042 | Max Pooling 2D | OK | 3.19 ms | 3.14 ms | 1.02x | -- |
| 045 | Average Pooling 2D | OK | 20.16 ms | 20.16 ms | 1.0x | -- |
| 048 | Mean reduction | OK | 67.44 ms | 67.44 ms | 1.0x | -- |
| 051 | Argmax reduction | OK | 84.65 ms | 84.65 ms | 1.0x | -- |
| 054 | 3D Convolution | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 057 | Transposed Conv2D (square) | FAIL | -- | -- | -- | Synthesis failed (NoValidChoicesError) |
| 060 | 3D Conv (asymmetric kernel) | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 063 | 2D Convolution (square) | OK | 679 us | 516 us | 1.23x | -- |
| 066 | 3D Conv (asymmetric input+kernel) | FAIL | -- | -- | -- | Standalone build (TypeError: wrong arg count) |
| 069 | Transposed Conv2D (asymmetric) | FAIL | -- | -- | -- | Synthesis failed (NoValidChoicesError) |
| 072 | Transposed Conv3D (asymmetric) | FAIL | -- | -- | -- | Synthesis failed (NoValidChoicesError) |
| 075 | Transposed Conv2D (grouped+dilated) | FAIL | -- | -- | -- | Synthesis failed (NoValidChoicesError) |
| 078 | Transposed Conv2D (asymmetric+padded) | FAIL | -- | -- | -- | Synthesis failed (NoValidChoicesError) |
| 081 | Transposed Conv2D (dilated+strided) | FAIL | -- | -- | -- | Synthesis failed (NoValidChoicesError) |
| 084 | Depthwise Conv2D | OK | 685 us | 513 us | 1.34x | -- |
| 087 | Pointwise Conv2D (1x1) | OK | 6.09 ms | 6.09 ms | 1.0x | -- |
| 090 | Cumulative product | FAIL | -- | -- | -- | Torch env corrupted (circular import) |
| 093 | Masked cumulative sum | FAIL | -- | -- | -- | Torch env corrupted (circular import) |
| 096 | Smooth L1 (Huber) loss | FAIL | -- | -- | -- | Torch env corrupted (circular import) |
| 099 | Triplet margin loss | FAIL | -- | -- | -- | Torch env corrupted (circular import) |

---

## 3. Detailed Per-Kernel Reports

### Kernel 003 -- Batched Matrix Multiply (BMM)
- **Triton kernels found:** 1 (`triton_tem_fused_bmm_0`)
- **Fusion:** No fusible groups detected
- **Failure:** Standalone build succeeded (4191 chars), but NCU profiling crashed with `TypeError: dynamic_func() takes 3 positional arguments but 5 were given`. The standalone wrapper passes extra grid args that template kernels don't accept.
- **Elapsed:** 53s

### Kernel 006 -- MatMul with Large K
- **Triton kernels found:** 1 (`triton_poi_fused_0`)
- **Fusion:** No fusible groups
- **Baseline:** 3040 ns
- **Iteration 1:** 3 plans generated (autotune/tile-size variants). All 3 passed correctness. Best plan C (increase_occupancy_and_coalescing) achieved 1.03x (2940 ns). 9 total search nodes.
- **Iteration 2:** 3 plans. Plan C (1D grid) signature mismatch, degraded, correctness failed. Plans A and B passed but regressed (0.98x, 0.24x).
- **Iteration 3:** 3 plans. All passed correctness but all regressed (0.92x-0.96x).
- **Final result:** 1.03x speedup. The kernel was already near-optimal at 3040 ns.
- **Elapsed:** 22m 03s

### Kernel 009 -- Tall-Skinny MatMul
- **Triton kernels found:** 1 (`triton_tem_fused_mm_0`)
- **Failure:** Same template kernel standalone TypeError as kernel 003.
- **Elapsed:** 40s

### Kernel 012 -- Diagonal Matrix Multiply
- **Triton kernels found:** 2 (`triton_poi_fused_diag_embed_0`, `triton_tem_fused_diag_embed_mm_1`)
- **Fusion:** 2 fusion groups detected, both failed validation (unicode chars, invalid syntax in generated code).
- **Kernel 1 (diag_embed pointwise):**
  - Baseline: 82750 ns
  - Iter 1: Plan C (larger XBLOCK + vectorized loads) achieved **4.90x** (16900 ns). Plan A: 3.68x.
  - Iter 2: Plans explored eviction policies and algorithmic changes; plateaued at 1.0x relative to best.
  - Iter 3: Software pipelining matched 1.0x relative; autotune regressed to 0.20x.
  - **Final: 4.92x** (16830 ns) -- best result in entire batch.
- **Kernel 2 (template GEMM):** Standalone build TypeError, failed.
- **Elapsed:** 24m 48s

### Kernel 015 -- Lower Triangular MatMul
- **Triton kernels found:** 1 (`triton_tem_fused_mm_tril_0`)
- **Failure:** Template kernel standalone TypeError.
- **Elapsed:** 38s

### Kernel 018 -- Transposed MatMul (A^T * B^T)
- **Triton kernels found:** 1 (`triton_tem_fused_mm_permute_0`)
- **Failure:** Template kernel standalone TypeError.
- **Elapsed:** 47s

### Kernel 021 -- Sigmoid Activation
- **Triton kernels found:** 1 (`triton_poi_fused_sigmoid_0`)
- **Baseline:** 7.73 ms
- **Iteration 1:** All 4 plans (A-D) failed with `INVALID TRITON API: tl.sigmoid does not exist -- use libdevice.sigmoid`. Zero valid candidates.
- **Iteration 2:** Same API error on all 3 plans. Early stopped after 2 consecutive no-improvement iters.
- **Final: 1.0x** -- all plans broke the kernel by using non-existent Triton API.
- **Root cause:** Executor LLM consistently rewrites `libdevice.sigmoid` as `tl.sigmoid`.
- **Elapsed:** 8m 11s

### Kernel 024 -- LogSoftmax Activation
- **Triton kernels found:** 1 (`triton_red_fused__log_softmax_prepare_softmax_online_0`)
- **Baseline:** 5.01 ms
- **Iteration 1:** All 3 plans failed with `INVALID TRITON API: tl.log does not exist -- use tl.math.log or libdevice.log`. Degraded plans also failed.
- **Iteration 2:** Same error pattern. Early stopped.
- **Final: 1.0x** -- executor LLM cannot preserve `tl.math.log` API.
- **Elapsed:** 9m 08s

### Kernel 027 -- SELU (ELU) Activation
- **Triton kernels found:** 1 (`triton_poi_fused_elu_0`)
- **Baseline:** 7.73 ms
- **Iteration 1:** All 3 plans passed correctness. Plan A (increase XBLOCK + autotune) achieved **2.57x** (3.01 ms). Plans B and C: 2.36x.
- **Iteration 2:** Plans tried FMA fusion and cache hints -- all plateaued at 1.0x relative. Autotune backup regressed (0.39x).
- **Iteration 3:** Same plateau pattern; autotune still 0.39x.
- **Final: 2.57x** -- tile size increase was the key optimization.
- **Elapsed:** 41m 01s

### Kernel 030 -- Softsign Activation
- **Triton kernels found:** 1 (`triton_poi_fused_abs_add_div_0`)
- **Failure:** Standalone build compiled, but NCU profiling crashed at kernel launch time (compile error during tile-size specialization).
- **Elapsed:** 43s

### Kernel 033 -- Batch Normalization (Inference)
- **Triton kernels found:** 1 (`triton_poi_fused__native_batch_norm_legit_no_training_0`)
- **Baseline:** 5.16 ms
- **Iteration 1:** All 3 plans passed. Plan A (increase XBLOCK + autotune hide L1TEX latency) achieved **2.53x** (2.04 ms). Plans B and C: 2.37x, 2.39x.
- **Iterations 2-3:** All subsequent plans plateaued at 1.0x relative. Autotune consistently regressed (0.40x).
- **Final: 2.53x** -- larger tile sizes reduced wave overhead on this memory-bound kernel.
- **Elapsed:** 32m 22s

### Kernel 036 -- RMS Normalization
- **Triton kernels found:** 2 (`triton_per_fused_mean_pow_0`, `triton_poi_fused_add_div_mean_pow_sqrt_1`)
- **Fusion:** 2 groups detected, both failed validation (string literal errors).
- **Kernel 1 (reduction):**
  - Baseline: 1.77 ms. Iter 1: Plan A (contiguous hints) 1.01x. Iters 2-3: All correctness failures on reduction tile changes.
  - Final: 1.01x
- **Kernel 2 (pointwise):**
  - Baseline: 9.03 ms. Iter 1: Plan B (autotune xblock/warps) achieved **2.10x** (4.30 ms).
  - Iter 2: Plan A (2D grid restructuring) improved to **2.49x** (3.62 ms).
  - Iter 3: Cache eviction policies regressed.
  - Final: 2.49x
- **Elapsed:** 62m 54s

### Kernel 039 -- L2 Normalization
- **Triton kernels found:** 1 (`triton_red_fused_div_linalg_vector_norm_0`)
- **Baseline:** 12.35 ms
- **Iteration 1:** Plans A and C exhausted all 3 dry-run attempts. Only Plan B passed (memory coalescing autotune) but regressed (0.85x).
- **Iteration 2:** Plans A and B again failed dry-run. Plan C (software pipelining via `tl.range num_stages`) achieved **1.80x** (6.86 ms).
- **Iteration 3:** All 3 plans passed. Plan C (asymmetric num_stages) improved slightly to **1.83x** (6.74 ms).
- **Final: 1.83x** -- software pipelining was the key optimization for this reduction kernel.
- **Elapsed:** 32m 03s

### Kernel 042 -- Max Pooling 2D
- **Triton kernels found:** 1 (`triton_poi_fused_max_pool2d_with_indices_0`)
- **Baseline:** 3.19 ms
- **Iteration 1:** Plans passed correctness but achieved only ~1.0x due to address computation overhead.
- **Iteration 2:** Software pipelining (num_stages) plans all around 1.0x.
- **Iteration 3:** 2D tiling plans hit signature mismatch. Autotune backup: 0.98x.
- **Final: 1.02x** -- the kernel's bottleneck is address computation for strided pooling, not memory bandwidth.
- **Elapsed:** 36m 57s

### Kernel 045 -- Average Pooling 2D
- **Triton kernels found:** 1 (`triton_red_fused_avg_pool2d_0`)
- **Baseline:** 20.16 ms
- **Iteration 1:** All 3 plans failed correctness (large diffs ~5.8).
- **Iteration 2:** Plan A passed correctness but no speedup (20.21 ms, 1.0x). Others failed. Early stopped.
- **Final: 1.0x** -- reduction kernel with correctness-sensitive pooling; optimizer couldn't improve.
- **Elapsed:** 18m 09s

### Kernel 048 -- Mean Reduction
- **Triton kernels found:** 1 (`triton_red_fused_mean_0`)
- **Baseline:** 67.44 ms
- **Iterations 1-2:** All plans either failed dry-run, failed correctness, or hit max_diff errors. Plans that tried to restructure the reduction axis broke correctness.
- **Final: 1.0x** -- this reduction kernel has register spill issues (local memory thrashing) that the optimizer cannot fix with tile-size/pipelining changes alone.
- **Elapsed:** 19m 47s

### Kernel 051 -- Argmax Reduction
- **Triton kernels found:** 1 (`triton_red_fused_argmax_0`)
- **Baseline:** 84.65 ms
- **Iterations 1-2:** Plans either timed out (>60s) or failed correctness (max_diff 4095). The kernel is extremely slow due to uncoalesced access patterns, but attempts to restructure break correctness on the argmax index tracking.
- **Final: 1.0x** -- argmax's index-tracking logic is fragile; coalescing changes break the max-index correspondence.
- **Elapsed:** 28m 03s

### Kernel 054 -- 3D Convolution
- **Triton kernels found:** 1 (`triton_tem_fused_convolution_0`)
- **Failure:** Template kernel standalone TypeError.
- **Elapsed:** 43s

### Kernel 057 -- Transposed Conv2D (Square)
- **Failure:** `torch.compile` synthesis failed: `NoValidChoicesError: No choices exist for backend` for `aten.convolution.default` with transposed=True. The inductor cannot generate Triton code for this conv_transpose operation.
- **Elapsed:** 18s

### Kernel 060 -- 3D Conv (Asymmetric Kernel)
- **Triton kernels found:** 1 (`triton_tem_fused_convolution_0`)
- **Failure:** Template kernel standalone TypeError.
- **Elapsed:** 42s

### Kernel 063 -- 2D Convolution (Square)
- **Triton kernels found:** 4 (2 pointwise transposes, 1 template GEMM, 1 pointwise epilogue)
- **Fusion:** 4 groups attempted, all failed (invalid syntax, unicode chars in generated code).
- **Kernel 1 (transpose prologue):** Baseline 679 us. Iter 3: autotune achieved 1.32x (516 us).
- **Kernel 2 (bias add):** Baseline 3200 ns. Iter 1: vectorized loads achieved 1.07x (2980 ns).
- **Kernel 3 (template GEMM):** Standalone TypeError, failed.
- **Kernel 4 (transpose epilogue):** Baseline 326 us. Iter 1: autotune tile sizes achieved 1.22x (268 us). Iter 2: further to 1.23x (266 us).
- **Final: 1.23x** (best kernel speedup reported).
- **Elapsed:** 82m 18s

### Kernel 066 -- 3D Conv (Asymmetric Input+Kernel)
- **Triton kernels found:** 1 (`triton_tem_fused_convolution_0`)
- **Failure:** Template kernel standalone TypeError.
- **Elapsed:** 39s

### Kernel 069 -- Transposed Conv2D (Asymmetric)
- **Failure:** Synthesis failed (NoValidChoicesError for conv_transpose).
- **Elapsed:** 16s

### Kernel 072 -- Transposed Conv3D (Asymmetric)
- **Failure:** Synthesis failed (NoValidChoicesError for conv_transpose3d with grouped ops).
- **Elapsed:** 15s

### Kernel 075 -- Transposed Conv2D (Grouped+Dilated)
- **Failure:** Synthesis failed (NoValidChoicesError for conv_transpose with dilation+groups).
- **Elapsed:** 15s

### Kernel 078 -- Transposed Conv2D (Asymmetric+Padded)
- **Failure:** Synthesis failed (NoValidChoicesError for conv_transpose with asymmetric kernel).
- **Elapsed:** 16s

### Kernel 081 -- Transposed Conv2D (Dilated+Strided)
- **Failure:** Synthesis failed (NoValidChoicesError for conv_transpose with dilation+stride).
- **Elapsed:** 17s

### Kernel 084 -- Depthwise Conv2D
- **Triton kernels found:** 2 (`triton_poi_fused_convolution_0`, `triton_tem_fused_convolution_1`)
- **Fusion:** 2 groups attempted, both failed (invalid syntax).
- **Kernel 1 (pointwise transpose):**
  - Baseline: 685 us. Iter 1: Plan C (autotune tile sizes) achieved **1.33x** (513 us).
  - Iter 2: Shared-memory transpose attempts all failed (correctness or dry-run).
  - Iter 3: Shared-memory plans passed correctness. Plans A (0.99x) and B (1.00x) couldn't beat autotune.
  - Final: 1.34x (from cumulative best across iterations).
- **Kernel 2 (template GEMM):** Standalone TypeError, failed.
- **Elapsed:** 46m 22s

### Kernel 087 -- Pointwise Conv2D (1x1)
- **Triton kernels found:** 3 (2 pointwise transposes, 1 template GEMM)
- **Fusion:** 2 groups attempted, both failed (unicode '×' in generated code).
- **Kernel 1 (transpose):** Baseline: 6.09 ms. Iter 1: Plans passed correctness but NCU profiling failed (torch env corrupt -- `narrow_scatter` AttributeError from local pytorch source tree). Iter 2: all dry-run failures. Early stopped at 1.0x.
- **Kernels 2-3:** Both failed NCU profiling due to same torch env issue.
- **Final: 1.0x** -- partially corrupted torch env blocked profiling of optimized candidates.
- **Elapsed:** 20m 01s

### Kernel 090 -- Cumulative Product
- **Failure:** `kernelopt` CLI import crash: `AttributeError: partially initialized module 'torch' has no attribute 'narrow_scatter' (most likely due to a circular import)`. The local pytorch source tree at `/workspaces/pytorch-devcontainers/pytorch/` shadows the installed torch.
- **Elapsed:** 1s

### Kernel 093 -- Masked Cumulative Sum
- **Failure:** Same torch environment corruption as kernel 090.
- **Elapsed:** 1s

### Kernel 096 -- Smooth L1 (Huber) Loss
- **Failure:** Same torch environment corruption as kernel 090.
- **Elapsed:** 1s

### Kernel 099 -- Triplet Margin Loss
- **Failure:** Same torch environment corruption as kernel 090.
- **Elapsed:** 1s

---

## 4. Failure Analysis

### Failure Mode Breakdown

| Failure Mode | Count | Kernels | Suggested Fix |
|---|---|---|---|
| **Standalone build: TypeError (template kernels)** | 8 | 003, 009, 015, 018, 054, 060, 066 (+ sub-kernels in 012, 063, 084, 087) | Fix `wrap_kernel.py` to not pass extra grid-size args to `triton_tem_*` template kernels. Template kernels use `@triton.jit` with fixed grid, not dynamic args. |
| **Synthesis failed: NoValidChoicesError (conv_transpose)** | 6 | 057, 069, 072, 075, 078, 081 | Add `ATEN` to `max_autotune_conv_backends` config in torch.compile, or fall back to ATEN backend for transposed convolutions. |
| **Torch env corrupted (circular import)** | 4 | 090, 093, 096, 099 | Ensure `/workspaces/pytorch-devcontainers/pytorch/` is not on `PYTHONPATH` or `sys.path`. The local pytorch source tree shadows the installed torch package. |
| **Standalone build crash (non-template)** | 1 | 030 | Investigate the specific compile error for the softsign kernel's tile-size specialization. |
| **Executor LLM uses wrong Triton API** | 2 (as 1.0x) | 021, 024 | Add `tl.sigmoid -> libdevice.sigmoid` and `tl.log -> tl.math.log` to the executor's API validation or prompt. These are already caught by the validator but the LLM keeps regenerating the same error. |

### Summary by Category

| Category | Count | % |
|---|---|---|
| Standalone build failures (template TypeError) | 8 | 44% of failures |
| Synthesis failures (torch.compile) | 6 | 33% of failures |
| Environment corruption | 4 | 22% of failures |
| Standalone build failures (other) | 1 | 6% of failures |

**Note:** Kernels 021, 024, 045, 048, 051 are marked as "OK" in the CSV (status=ok, speedup=1.0x) despite achieving no speedup, because they completed the pipeline without errors -- the optimizer simply couldn't find improvements.

---

## 5. Top Performers

### Rank by Speedup

| Rank | Kernel | Problem | Speedup | Key Optimization |
|---|---|---|---|---|
| 1 | **012** | Diagonal matrix multiply | **4.92x** | Larger XBLOCK with vectorized loads/stores; reduced grid overhead for the `diag_embed` pointwise kernel from 82750 ns to 16830 ns. |
| 2 | **027** | SELU activation | **2.57x** | Increased XBLOCK with autotune to reduce scheduling overhead and hide memory latency. Went from 7.73 ms to 3.01 ms. |
| 3 | **033** | Batch Normalization | **2.53x** | Increased XBLOCK with autotune to hide L1TEX latency on the batch norm pointwise kernel. 5.16 ms to 2.04 ms. |
| 4 | **036** | RMS Normalization | **2.49x** | Autotune XBLOCK/num_warps on the pointwise div/sqrt kernel, plus 2D grid restructuring. 9.03 ms to 3.62 ms. |
| 5 | **039** | L2 Normalization | **1.83x** | Software pipelining via `tl.range(num_stages=2)` to hide L1TEX latency on the reduction kernel. 12.35 ms to 6.74 ms. |

### Common Optimization Patterns

1. **Tile size increase (XBLOCK) with autotune** -- The single most effective optimization. Every top-5 kernel benefited from increasing XBLOCK from the default (usually 128) to larger values (512-2048) with `@triton.autotune` configs. This reduces grid launch overhead and improves memory coalescing.

2. **Software pipelining (num_stages)** -- Effective for reduction kernels (039). Adding `num_stages=2` to `tl.range` hides memory latency by overlapping loads with computation.

3. **Vectorized loads via contiguous hints** -- Marginal improvements (1-5%) on several kernels by adding `tl.max_contiguous` / `tl.multiple_of` hints.

4. **Cache eviction policies** -- Rarely effective in practice. Plans using `evict_first`/`evict_last` policies showed minimal (<1%) impact on H200.

### Anti-Patterns (Optimizations That Consistently Regress)

1. **`@triton.autotune` as backup plan** -- When added to already-optimal kernels with small tile sizes, autotune's warmup overhead dominates, causing 2-5x regression (seen in kernels 027, 033 as 0.39-0.40x).

2. **Signature-changing plans (2D tiling, algorithm changes)** -- Plans that rename parameters or add YBLOCK/CBLOCK consistently hit signature mismatch, degrade, and fail correctness. This is the most common plan failure mode.

3. **Reduction axis restructuring** -- Attempts to swap or restructure reduction dimensions consistently break correctness on reduction kernels (036, 045, 048, 051).

---

## 6. Recommendations

### High-Impact Fixes (address 78% of failures)

1. **Fix template kernel standalone wrappers** -- The `wrap_kernel.py` module passes grid-size arguments (e.g., `128, 128`) to template kernels that don't accept them. This causes `TypeError: dynamic_func() takes 3 positional arguments but 5 were given` on all `triton_tem_*` kernels. Fixing this would unlock 8 additional kernels (003, 009, 015, 018, 054, 060, 066 + sub-kernels).

2. **Enable ATEN backend for conv_transpose** -- Add `ATEN` to `max_autotune_conv_backends` in the torch.compile config. This would allow synthesis to succeed for 6 transposed convolution kernels (057, 069, 072, 075, 078, 081).

3. **Fix PYTHONPATH** -- Remove the local pytorch source tree from the Python path to prevent circular imports on kernels 090-099.

### Medium-Impact Improvements

4. **Teach executor about Triton API differences** -- Add explicit examples of `libdevice.sigmoid` (not `tl.sigmoid`) and `tl.math.log` (not `tl.log`) to the executor prompt or enforce via post-processing validation that rejects these common errors before attempting a dry-run.

5. **Prevent signature-changing plans** -- The planner repeatedly generates plans that rename kernel parameters (e.g., `xnumel` -> `N`, adding `YBLOCK`). These always fail. Add a hard constraint in the planner that kernel parameter names/count are immutable.

6. **Fix fusion code generation** -- All fusion attempts across the batch failed with syntax errors (unicode chars like `--`, `->`, `x`), unterminated strings, or `@jit functions should be defined in a Python file`. The fusion code generator needs to sanitize its output.
