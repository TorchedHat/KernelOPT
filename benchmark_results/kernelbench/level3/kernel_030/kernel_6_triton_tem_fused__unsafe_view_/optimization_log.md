# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4900 ns — **Best latency**: 4450 ns (**1.1011×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_grid_with_correct_3d_batch_decomposition_and_split_k`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_030/kernel_6_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 96
- **Duration**: 4900 ns
- **SM Throughput**: 0.18 % | **Memory**: 0.53 % | **DRAM**: 0.14 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 1.43 inst/cycle | **No Eligible**: 65.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_and_enable_tf32_tensor_cores | 0.977× | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_and_enab... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4900 ns | 4770 ns | -130 ns (-2.7%) |
| SM Throughput (%) | 0.18 % | 0.20 % | +0.02 pp |
| Memory Throughput (%) | 0.53 % | 0.46 % | -0.07 pp |
| DRAM Throughput (%) | 0.14 % | 0.14 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 65.11 % | 64.12 % | -0.99 pp |
| IPC Active | 1.43 inst/cycle | 1.46 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_with_correct_3d_batch_decomposition_and_split_k | 1.004× | 1.101× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_with_cor... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4900 ns | 4450 ns | -450 ns (-9.2%) |
| SM Throughput (%) | 0.18 % | 0.19 % | +0.01 pp |
| Memory Throughput (%) | 0.53 % | 0.55 % | +0.02 pp |
| DRAM Throughput (%) | 0.14 % | 0.15 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 65.11 % | 59.47 % | -5.64 pp |
| IPC Active | 1.43 inst/cycle | 1.61 inst/cycle | +0.18 |

---

## Iteration 3

**Best so far**: 1.1011× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_launch_3d_decomposition_with_smaller_tiles | 0.980× | 1.094× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_3... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4900 ns | 4480 ns | -420 ns (-8.6%) |
| SM Throughput (%) | 0.18 % | 0.19 % | +0.01 pp |
| Memory Throughput (%) | 0.53 % | 0.49 % | -0.04 pp |
| DRAM Throughput (%) | 0.14 % | 0.15 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 65.11 % | 59.15 % | -5.96 pp |
| IPC Active | 1.43 inst/cycle | 1.62 inst/cycle | +0.19 |

---

## Iteration 4

**Best so far**: 1.1011× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_grid_launch_site_only_minimal_3d_batch | 0.981× | 1.094× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_s... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 4900 ns | 4480 ns | -420 ns (-8.6%) |
| SM Throughput (%) | 0.18 % | 0.19 % | +0.01 pp |
| Memory Throughput (%) | 0.53 % | 0.53 % | +0.00 pp |
| DRAM Throughput (%) | 0.14 % | 0.15 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 96.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 65.11 % | 62.53 % | -2.58 pp |
| IPC Active | 1.43 inst/cycle | 1.61 inst/cycle | +0.18 |


### Key Learnings

#### ✅ Plan D: `fix_grid_launch_site_only_minimal_3d_batch` (1.094×)

This iteration applied a minimal grid-only change (grid=(1,1,1) → grid=(1,3,640)) to a batched matmul kernel whose body already contained correct 3D batch decomposition logic. However, NCU profiling still shows grid_size=1 and waves_per_sm=0.0 after the change, meaning the grid expansion was not captured in the profiled kernel — the 1.09x wall-clock speedup is not attributable to parallelism expansion. The profiling metrics are nearly identical before and after (same duration, same occupancy, same grid_size=1), with minor secondary changes (L2 Hit Rate improved from 57.22% to 61.98%, Warp Cycles Per Issued Instruction slightly increased from 2.47 to 2.79). The LaunchConfiguration bottleneck (99.24% potential speedup) remains unresolved in the profiled kernel, suggesting the grid change either did not take effect in the profiled invocation or the profiler captured a different kernel instance.

> **Applicable when**: Batched matmul kernel with multi-dimensional batch (outer_batch × num_heads) where the kernel body already uses tl.program_id(1) and tl.program_id(2) for batch decomposition but the launch grid is hardcoded to (1,1,1). NCU LaunchConfiguration rule fires with speedup_pct >90%, grid_size=1, waves_per_sm=0.0. The kernel body pointer arithmetic uses separate strides per batch sub-dimension.

> **Avoid when**: The grid change does not register in NCU profiling (grid_size remains 1 after change) — this indicates the profiler is capturing a different kernel invocation or the change is not being applied to the hot path. Do not rely on this minimal grid-only fix if prior iterations with the same grid shape (1,3,640) also failed to show grid expansion in profiling. Multiple prior attempts with similar grid shapes all showed correctness failures (max_diff ~20), suggesting the kernel body indexing may not be as correct as assumed.

---

## Cumulative Progress

```
Baseline:  4900 ns  (1.000×)
Iter 2     4450 ns  (1.101×)  — fix_grid_with_correct_3d_batch_decomposition_and_split_k

Final best: 4450 ns  (1.1011×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6(arg_A, arg_B, out_ptr0):
@@ -58,9 +59,17 @@

 
     rk = tl.arange(0, BLOCK_K)
 
-    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
-    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
-    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)
+    # 3D batch decomposition: pid[1]=head (0..2), pid[2]=outer_batch (0..639)
+    pid_head = tl.program_id(1).to(INDEX_DTYPE)   # head index: 0..2
+    pid_outer = tl.program_id(2).to(INDEX_DTYPE)  # outer batch index: 0..639
+    idx_q = 3 * pid_outer + pid_head              # flat batch index for output
+
+    # A has shape (640, 3, 49, 32) with strides (4704, 1568, 32, 1)
+    # outer batch stride = 4704, head stride = 1568
+    A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + pid_head * stride_aq + pid_outer * 4704)
+    # B has shape (640, 3, 32, 49) with strides (4704, 1568, 49, 1)
+    # outer batch stride = 4704, head stride = 1568
+    B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + pid_head * stride_bq + pid_outer * 4704)
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
     for k in range(K, 0, -BLOCK_K):
@@ -77,7 +86,10 @@

     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
+    # recompute batch indices
+    pid_head = tl.program_id(1).to(INDEX_DTYPE)
+    pid_outer = tl.program_id(2).to(INDEX_DTYPE)
+    idx_q = 3 * pid_outer + pid_head
     idx_m = rm[:, None]
     idx_n = rn[None, :]
     mask = (idx_m < M) & (idx_n < N)
@@ -85,8 +97,6 @@

     # inductor generates a suffix
     xindex = idx_n + 49*idx_m + 2401*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
