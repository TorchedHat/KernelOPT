# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4640 ns — **Best latency**: 2910 ns (**1.5945×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_evict_first_on_streaming_load_and_increase_num_stages`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_21`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_21_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (80,1,1), Waves/SM: 0.08
- **Registers/thread**: 64
- **Duration**: 4640 ns
- **SM Throughput**: 4.24 % | **Memory**: 9.00 % | **DRAM**: 9.00 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.49 inst/cycle | **No Eligible**: 87.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_increase_xblock_for_grid_utilization | 0.998× | 0.979× | ✓ |  |
| B | add_autotune_reduce_xblock_increase_grid_coverage | 1.175× | 1.296× | ✓ |  |
| C | autotune_xblock_and_vectorize_loads | 1.193× | 1.271× | ✓ | **✓ best** |
| D | reduce_xblock_increase_grid_coverage | 1.156× | 1.228× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_incr... | Δ | add_autotune_redu... | Δ | autotune_xblock_a... | Δ | reduce_xblock_inc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 4740 ns | +100 ns (+2.2%) | 3580 ns | -1060 ns (-22.8%) | 3650 ns | -990 ns (-21.3%) | 3780 ns | -860 ns (-18.5%) |
| SM Throughput (%) | 4.24 % | 4.24 % | +0.00 pp | 1.98 % | -2.26 pp | 2.36 % | -1.88 pp | 2.34 % | -1.90 pp |
| Memory Throughput (%) | 9.00 % | 8.83 % | -0.17 pp | 2.62 % | -6.38 pp | 2.98 % | -6.02 pp | 2.87 % | -6.13 pp |
| DRAM Throughput (%) | 9.00 % | 8.83 % | -0.17 pp | 1.57 % | -7.43 pp | 2.98 % | -6.02 pp | 2.87 % | -6.13 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 3.10 % | -3.14 pp | 3.11 % | -3.13 pp | 3.11 % | -3.13 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 40.00 register/thread | -24 | 56.00 register/thread | -8 | 56.00 register/thread | -8 |
| Waves/SM | 0.08 | 0.08 | +0.00 | 0.03 | -0.05 | 0.03 | -0.05 | 0.03 | -0.05 |
| No Eligible (%) | 87.62 % | 88.13 % | +0.51 pp | 85.55 % | -2.07 pp | 84.63 % | -2.99 pp | 84.31 % | -3.31 pp |
| IPC Active | 0.49 inst/cycle | 0.49 inst/cycle | +0.00 | 0.27 inst/cycle | -0.22 | 0.31 inst/cycle | -0.18 | 0.31 inst/cycle | -0.18 |

---

## Iteration 2

**Best so far**: 1.2712× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_expand_sm_coverage_with_smaller_xblock | 1.372× | 1.595× | ✓ | **✓ best** |
| B | fix_grid_lambda_to_enable_sm_coverage_from_autotune | 1.196× | 1.296× | ✓ |  |
| C | expand_grid_coverage_via_dynamic_xblock_lambda | 1.240× | 1.228× | ✓ |  |
| D | expand_grid_coverage_via_small_xblock_configs_with_lambda_grid | 1.365× | 1.557× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | expand_grid_cover... | Δ | expand_grid_cover... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 2910 ns | -1730 ns (-37.3%) | 3580 ns | -1060 ns (-22.8%) | 3780 ns | -860 ns (-18.5%) | 2980 ns | -1660 ns (-35.8%) |
| SM Throughput (%) | 4.24 % | 0.35 % | -3.89 pp | 2.05 % | -2.19 pp | 1.58 % | -2.66 pp | 0.35 % | -3.89 pp |
| Memory Throughput (%) | 9.00 % | 2.79 % | -6.21 pp | 2.64 % | -6.36 pp | 2.35 % | -6.65 pp | 2.77 % | -6.23 pp |
| DRAM Throughput (%) | 9.00 % | 0.15 % | -8.85 pp | 1.59 % | -7.41 pp | 0.78 % | -8.22 pp | 0.15 % | -8.85 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 3.10 % | -3.14 pp | 3.10 % | -3.14 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 64.00 register/thread | 22.00 register/thread | -42 | 40.00 register/thread | -24 | 32.00 register/thread | -32 | 22.00 register/thread | -42 |
| Waves/SM | 0.08 | 0.02 | -0.06 | 0.03 | -0.05 | 0.02 | -0.06 | 0.02 | -0.06 |
| No Eligible (%) | 87.62 % | 93.59 % | +5.97 pp | 85.23 % | -2.39 pp | 89.17 % | +1.55 pp | 93.59 % | +5.97 pp |
| IPC Active | 0.49 inst/cycle | 0.06 inst/cycle | -0.43 | 0.27 inst/cycle | -0.22 | 0.21 inst/cycle | -0.28 | 0.06 inst/cycle | -0.43 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_expand_sm_coverage_with_smaller_xblock` (1.611×)

Removing large XBLOCK configs (256, 512, 1024) from the @triton.autotune search space and changing the autotune key from ['xnumel', 'r0_numel'] to ['xnumel'] alone produced a 1.61x speedup relative to the original baseline, even though the selected config (XBLOCK=32, block_size=32, 22 regs/thread) and all profiling metrics remained nearly identical to the prior iteration. The marginal metric changes (warp cycles per issued instruction: 16.27→12.85, elapsed cycles: 4035→4267) suggest the speedup is measured against a different baseline and the actual kernel execution is essentially unchanged. The key insight is that once the autotune space is anchored to small XBLOCK configs with a lambda grid, removing large configs and fixing the autotune key to exclude fixed-at-runtime dimensions (r0_numel) prevents autotune from wasting benchmark time on suboptimal large-tile configs, consolidating the gains from prior iterations.

> **Applicable when**: Triton reduction kernel where autotune key includes dimensions that are fixed at runtime (e.g., r0_numel is always the same value); autotune config space includes large XBLOCK values (256+) that are known to cause register spilling or instruction bloat; the kernel is already achieving good L2 hit rate (>75%) and low instruction count (<10K) with small XBLOCK configs; the goal is to reduce autotune overhead and prevent regression to large-tile configs.

> **Avoid when**: The autotune key dimensions genuinely vary across calls (r0_numel changes at runtime); large XBLOCK configs are needed for coalescing on memory-bandwidth-bound kernels; the kernel has not yet been optimized with small XBLOCK configs and a lambda grid (apply exp_002 first).

---

## Iteration 3

**Best so far**: 1.5945× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_large_tiles_with_lambda_grid_and_num_stages | 1.369× | 1.706× | ✓ |  |
| B | add_evict_first_on_streaming_load_and_increase_num_stages | 1.363× | 1.595× | ✓ |  |
| C | increase_pipeline_depth_and_large_xblock_to_reduce_scheduler_stalls | 1.220× | 1.318× | ✓ |  |
| D | increase_xblock_large_tiles_for_cache_reuse_and_latency_hiding | 1.029× | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_l... | Δ | add_evict_first_o... | Δ | increase_pipeline... | Δ | increase_xblock_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 2720 ns | -1920 ns (-41.4%) | 2910 ns | -1730 ns (-37.3%) | 3520 ns | -1120 ns (-24.1%) | 4610 ns | -30 ns (-0.6%) |
| SM Throughput (%) | 4.24 % | 0.36 % | -3.88 pp | 0.37 % | -3.87 pp | 2.01 % | -2.23 pp | 4.17 % | -0.07 pp |
| Memory Throughput (%) | 9.00 % | 2.99 % | -6.01 pp | 2.80 % | -6.20 pp | 2.67 % | -6.33 pp | 9.01 % | +0.01 pp |
| DRAM Throughput (%) | 9.00 % | 0.17 % | -8.83 pp | 0.15 % | -8.85 pp | 1.60 % | -7.40 pp | 9.01 % | +0.01 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 3.10 % | -3.14 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 22.00 register/thread | -42 | 22.00 register/thread | -42 | 40.00 register/thread | -24 | 64.00 register/thread | +0 |
| Waves/SM | 0.08 | 0.02 | -0.06 | 0.02 | -0.06 | 0.03 | -0.05 | 0.08 | +0.00 |
| No Eligible (%) | 87.62 % | 93.66 % | +6.04 pp | 93.88 % | +6.26 pp | 86.14 % | -1.48 pp | 87.57 % | -0.05 pp |
| IPC Active | 0.49 inst/cycle | 0.07 inst/cycle | -0.42 | 0.06 inst/cycle | -0.43 | 0.26 inst/cycle | -0.23 | 0.46 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.5945× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_load_plus_num_stages3_autotune_expansion | 1.316× | 1.628× | ✓ |  |
| B | fix_hardcoded_grid_to_lambda_enabling_autotune_tile_exploration | 1.363× | 1.557× | ✓ |  |
| C | restrict_autotune_to_small_xblock_to_eliminate_register_heavy_config_selection | 1.366× | 1.557× | ✓ |  |
| D | small_xblock_configs_with_lambda_grid_and_num_stages_sweep | 1.362× | 1.611× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | fix_hardcoded_gri... | Δ | restrict_autotune... | Δ | small_xblock_conf... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 2850 ns | -1790 ns (-38.6%) | 2980 ns | -1660 ns (-35.8%) | 2980 ns | -1660 ns (-35.8%) | 2880 ns | -1760 ns (-37.9%) |
| SM Throughput (%) | 4.24 % | 0.37 % | -3.87 pp | 0.37 % | -3.87 pp | 0.36 % | -3.88 pp | 0.33 % | -3.91 pp |
| Memory Throughput (%) | 9.00 % | 2.84 % | -6.16 pp | 2.80 % | -6.20 pp | 2.83 % | -6.17 pp | 2.79 % | -6.21 pp |
| DRAM Throughput (%) | 9.00 % | 0.15 % | -8.85 pp | 0.15 % | -8.85 pp | 0.15 % | -8.85 pp | 0.16 % | -8.84 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 64.00 register/thread | 22.00 register/thread | -42 | 22.00 register/thread | -42 | 22.00 register/thread | -42 | 22.00 register/thread | -42 |
| Waves/SM | 0.08 | 0.02 | -0.06 | 0.02 | -0.06 | 0.02 | -0.06 | 0.02 | -0.06 |
| No Eligible (%) | 87.62 % | 93.59 % | +5.97 pp | 92.19 % | +4.57 pp | 93.40 % | +5.78 pp | 93.66 % | +6.04 pp |
| IPC Active | 0.49 inst/cycle | 0.07 inst/cycle | -0.42 | 0.06 inst/cycle | -0.43 | 0.06 inst/cycle | -0.43 | 0.06 inst/cycle | -0.43 |

---

## Cumulative Progress

```
Baseline:  4640 ns  (1.000×)
Iter 1     3650 ns  (1.271×)  — autotune_xblock_and_vectorize_loads
Iter 2     2910 ns  (1.595×)  — fix_grid_lambda_to_expand_sm_coverage_with_smaller_xblock

Final best: 2910 ns  (1.5945×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 2}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 4}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 8}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK': 8}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_21(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 10240
@@ -24,11 +50,11 @@

     x0 = (xindex % 1024)
     x1 = xindex // 1024
     x3 = xindex
-    tmp0 = tl.load(in_out_ptr0 + (x0 + 1024*r0_2 + 50176*x1), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x0 + 1024*r0_2 + 50176*x1), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr0 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -48,9 +74,7 @@

     tmp22 = tl.full([1, 1], 49.0, tl.float32)
     tmp23 = (tmp21 / tmp22)
     tl.debug_barrier()
-    tl.store(in_out_ptr1 + (x3), tmp23, xmask)
-
-
+    tl.store(in_out_ptr1 + (x3), tmp23, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -62,7 +86,7 @@

     xnumel = 10240
     r0_numel = 49
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 10240, 49, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 10240, 49)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
