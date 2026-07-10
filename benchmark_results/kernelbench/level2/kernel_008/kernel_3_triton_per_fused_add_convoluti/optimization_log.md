# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2430 ns — **Best latency**: 2430 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_008/kernel_3_triton_per_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 28
- **Duration**: 2430 ns
- **SM Throughput**: 0.02 % | **Memory**: 0.44 % | **DRAM**: 0.11 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.30 inst/cycle | **No Eligible**: 90.06 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | reduce_xblock_to_increase_grid_parallelism | 1.031× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2430 ns | 2430 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp |
| Memory Throughput (%) | 0.44 % | 0.41 % | -0.03 pp |
| DRAM Throughput (%) | 0.11 % | 0.03 % | -0.08 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.06 % | 95.11 % | +5.05 pp |
| IPC Active | 0.30 inst/cycle | 0.04 inst/cycle | -0.26 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune_parallelism | 0.984× | 1.115× | ✓ |  |
| B | fix_grid_launch_to_use_autotuned_xblock | 1.014× | 1.071× | ✓ |  |
| C | vectorized_loads_with_evict_first_streaming_policy | 1.029× | 1.099× | ✓ |  |
| D | fix_grid_lambda_to_enable_xblock_autotune_parallelism | 1.020× | 1.115× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_grid_launch_t... | Δ | vectorized_loads_... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2430 ns | 2180 ns | -250 ns (-10.3%) | 2270 ns | -160 ns (-6.6%) | 2210 ns | -220 ns (-9.1%) | 2180 ns | -250 ns (-10.3%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp |
| Memory Throughput (%) | 0.44 % | 0.55 % | +0.11 pp | 0.54 % | +0.10 pp | 0.41 % | -0.03 pp | 0.49 % | +0.05 pp |
| DRAM Throughput (%) | 0.11 % | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp | 0.04 % | -0.07 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.06 % | 95.03 % | +4.97 pp | 95.15 % | +5.09 pp | 94.27 % | +4.21 pp | 95.81 % | +5.75 pp |
| IPC Active | 0.30 inst/cycle | 0.03 inst/cycle | -0.27 | 0.04 inst/cycle | -0.26 | 0.04 inst/cycle | -0.26 | 0.03 inst/cycle | -0.27 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_enable_autotune_parallelism` (1.115×)

When a Triton kernel uses @triton.autotune with multiple XBLOCK configs but the caller computes the grid with a hardcoded constant (e.g., cdiv(xnumel, 128)) instead of meta['XBLOCK'], the grid size is always 1 regardless of which XBLOCK autotune selects, causing partial output and correctness failures. Changing the grid to a lambda `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` allows autotune to correctly explore smaller XBLOCK values that spread work across more SMs. This reduced elapsed cycles from 3605 to 3249 (~10%) and warp cycles per issued instruction from 28.1 to 22.6, yielding a 1.11x speedup.

> **Applicable when**: Triton kernel has @triton.autotune with multiple XBLOCK configs, but the caller grid formula uses a hardcoded constant instead of meta['XBLOCK']. LaunchConfiguration NCU rule fires with high speedup_pct (>90%). Grid size is 1 while SM count is large (e.g., 132). Achieved occupancy is near 0% (1-2%) vs theoretical 50%+.

> **Avoid when**: Grid is already a lambda using meta['XBLOCK']. Kernel has no @triton.autotune or only one XBLOCK config. The workload is intentionally single-block (e.g., a global reduction where all threads must cooperate). Correctness requires all x-elements to be processed by a single program_id.

#### ✅ Plan B: `fix_grid_launch_to_use_autotuned_xblock` (1.071×)

When a Triton kernel uses @triton.autotune with multiple XBLOCK configs but the launch site hardcodes the grid as `(cdiv(xnumel, FIXED_XBLOCK),)`, the grid size never adapts to the winning config, leaving most SMs idle. Replacing the hardcoded grid with `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` allows the grid to scale with the autotuner's chosen XBLOCK. In this case the kernel was tiny (grid=1, 2.43 µs) so the improvement was modest (~7%), but the fix is structurally correct and prevents correctness failures that occurred in prior iterations when XBLOCK was reduced without updating the grid expression.

> **Applicable when**: A Triton kernel has @triton.autotune with multiple XBLOCK configs but the kernel_function launch site uses a hardcoded constant (e.g. cdiv(xnumel, 128)) instead of reading XBLOCK from meta. NCU LaunchConfiguration rule fires with high speedup_pct (>90%). Grid size is 1 or very small relative to SM count.

> **Avoid when**: The kernel does not use @triton.autotune, or the grid is already a lambda reading from meta['XBLOCK']. Also not applicable when the kernel body lacks an xmask guard (xindex < xnumel) — without the mask, non-power-of-two xnumel with small XBLOCK will cause out-of-bounds writes.

#### ✅ Plan D: `fix_grid_lambda_to_enable_xblock_autotune_parallelism` (1.115×)

When a Triton kernel's grid is hardcoded as a static tuple (e.g., `(cdiv(N, XBLOCK_CONSTANT),)`), the autotuner can vary XBLOCK in the kernel configs but the grid size never changes, so all parallelism benefit is lost. Replacing the static grid with a lambda `lambda meta: (cdiv(N, meta['XBLOCK']),)` lets the grid adapt dynamically to whichever XBLOCK the autotuner selects, correctly distributing work across SMs. In this case the fix reduced elapsed cycles from 3605 to 3246 (10.5%) and warp cycles per issued instruction from 28.1 to 21.7, even though the workload is tiny (128 elements) and the kernel remains latency-bound.

> **Applicable when**: Triton kernel has @triton.autotune with multiple XBLOCK configs but the call-site grid is a static tuple using a hardcoded XBLOCK constant. LaunchConfiguration NCU rule fires with high speedup_pct (>90%). Grid size is 1 or very small relative to SM count. Achieved occupancy is far below theoretical occupancy.

> **Avoid when**: The kernel body does not have an xmask guard (partial tiles would be unsafe). The grid is already a lambda or the XBLOCK is not an autotuned parameter. The workload is large enough that a single block already saturates the SM.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | num_stages_pipelining_with_evict_first_streaming_load | 0.985× | 1.000× | ✓ |  |
| B | add_num_stages_pipelining_and_evict_first_streaming_loads | 1.027× | 1.099× | ✓ |  |
| C | expand_autotune_configs_with_higher_num_warps_and_num_stages | 1.026× | 1.099× | ✓ |  |
| D | add_num_stages_and_larger_num_warps_to_autotune_configs | 1.028× | 1.056× | ✓ |  |


### Metric Comparison

| Metric | Baseline | num_stages_pipeli... | Δ | add_num_stages_pi... | Δ | expand_autotune_c... | Δ | add_num_stages_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2430 ns | 2430 ns | +0 ns (+0.0%) | 2210 ns | -220 ns (-9.1%) | 2210 ns | -220 ns (-9.1%) | 2300 ns | -130 ns (-5.3%) |
| SM Throughput (%) | 0.02 % | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp | 0.00 % | -0.02 pp |
| Memory Throughput (%) | 0.44 % | 0.46 % | +0.02 pp | 0.42 % | -0.02 pp | 0.45 % | +0.01 pp | 0.41 % | -0.03 pp |
| DRAM Throughput (%) | 0.11 % | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.06 % | 94.20 % | +4.14 pp | 94.10 % | +4.04 pp | 94.14 % | +4.08 pp | 95.99 % | +5.93 pp |
| IPC Active | 0.30 inst/cycle | 0.03 inst/cycle | -0.27 | 0.05 inst/cycle | -0.25 | 0.04 inst/cycle | -0.26 | 0.03 inst/cycle | -0.27 |


### Key Learnings

#### ✅ Plan D: `add_num_stages_and_larger_num_warps_to_autotune_configs` (1.056×)

Adding num_stages=2 variants to autotune configs for very small kernels (41 instructions, grid=1, ~2 µs) does not reliably improve performance and can cause regression. In this case, elapsed cycles increased from 3249 to 3435 (+5.7%) and Warp Cycles Per Issued Instruction worsened from 22.6 to 28.9, despite the CPIStall rule suggesting L1TEX stalls as the bottleneck. The kernel is too small and fully unrolled (R0_BLOCK=16 compile-time constant) for software pipelining to provide meaningful latency hiding; the autotuner overhead and register pressure from num_stages=2 outweigh any benefit.

> **Applicable when**: Kernel has a large loop body with many iterations, significant compute per load, and is NOT fully unrolled at compile time. CPIStall reports 40%+ of cycles stalled on L1TEX. Kernel duration is >10 µs with multiple waves across SMs.

> **Avoid when**: Kernel is extremely small (<50 instructions, <3 µs, grid=1). The reduction dimension (R0_BLOCK) is a compile-time constant causing full loop unrolling — software pipelining cannot reorder already-unrolled instructions. Achieved occupancy is already near theoretical maximum. The kernel has no meaningful loop body for the pipeline to overlap.

---

## Cumulative Progress

```
Baseline:  2430 ns  (1.000×)

Final best: 2430 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},  num_warps=1),
+        triton.Config({'XBLOCK': 2},  num_warps=1),
+        triton.Config({'XBLOCK': 4},  num_warps=1),
+        triton.Config({'XBLOCK': 8},  num_warps=2),
+        triton.Config({'XBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 128
@@ -23,7 +37,7 @@

     r0_1 = r0_index
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
-    tmp3 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
     tmp1 = tl.full([1, 1], 6727.0, tl.float32)
     tmp2 = (tmp0 / tmp1)
     tmp4 = tmp2 + tmp3
@@ -32,8 +46,6 @@

     tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
     tl.store(out_ptr0 + (x0), tmp8, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -41,7 +53,7 @@

     xnumel = 128
     r0_numel = 16
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3[grid](arg_0, arg_1, arg_2, 128, 16, XBLOCK=128)
+    triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3[grid](arg_0, arg_1, arg_2, 128, 16)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
