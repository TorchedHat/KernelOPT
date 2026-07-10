# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 2340 ns (**2.5171×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_static_grid_to_xnumel_blocks_with_xblock1_for_full_sm_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_096/kernel_2_triton_per_fused_clamp_convolu/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.03
- **Registers/thread**: 128
- **Duration**: 5890 ns
- **SM Throughput**: 2.85 % | **Memory**: 3.32 % | **DRAM**: 1.97 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.00 inst/cycle | **No Eligible**: 74.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_parallelism_via_smaller_xblock_and_autotune | 1.572× | 1.841× | ✓ |  |
| B | increase_grid_coverage_via_autotune_xblock | 1.673× | 2.561× | ✓ | **✓ best** |
| C | autotune_xblock_and_num_warps_to_fill_sms | 1.620× | 2.485× | ✓ |  |
| D | reduce_xblock_to_increase_grid_size_and_lower_register_pressure | 1.644× | 2.301× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_grid_par... | Δ | increase_grid_cov... | Δ | autotune_xblock_a... | Δ | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 3200 ns | -2690 ns (-45.7%) | 2300 ns | -3590 ns (-61.0%) | 2370 ns | -3520 ns (-59.8%) | 2560 ns | -3330 ns (-56.5%) |
| SM Throughput (%) | 2.85 % | 0.62 % | -2.23 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp |
| Memory Throughput (%) | 3.32 % | 0.74 % | -2.58 pp | 0.50 % | -2.82 pp | 0.43 % | -2.89 pp | 0.41 % | -2.91 pp |
| DRAM Throughput (%) | 1.97 % | 0.49 % | -1.48 pp | 0.07 % | -1.90 pp | 0.06 % | -1.91 pp | 0.06 % | -1.91 pp |
| Achieved Occupancy (%) | 6.25 % | 3.10 % | -3.15 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 128.00 register/thread | 40.00 register/thread | -88 | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 |
| Waves/SM | 0.03 | 0.01 | -0.02 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 |
| No Eligible (%) | 74.85 % | 86.01 % | +11.16 pp | 96.09 % | +21.24 pp | 95.78 % | +20.93 pp | 95.17 % | +20.32 pp |
| IPC Active | 1.00 inst/cycle | 0.28 inst/cycle | -0.72 | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 |

---

## Iteration 2

**Best so far**: 2.5609× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_cache_reuse_and_warp_utilization | 1.672× | 2.328× | ✓ |  |
| B | fix_hardcoded_grid_and_expand_xblock_autotune_configs | 1.682× | 2.356× | ✓ | **✓ best** |
| C | fix_callsite_static_grid_and_expand_autotune_tile_sizes | 1.629× | 2.454× | ✓ |  |
| D | fix_hardcoded_grid_and_expand_autotune_to_smaller_xblock | 1.625× | 2.561× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | fix_hardcoded_gri... | Δ | fix_callsite_stat... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 2530 ns | -3360 ns (-57.0%) | 2500 ns | -3390 ns (-57.6%) | 2400 ns | -3490 ns (-59.3%) | 2300 ns | -3590 ns (-61.0%) |
| SM Throughput (%) | 2.85 % | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp |
| Memory Throughput (%) | 3.32 % | 0.38 % | -2.94 pp | 0.38 % | -2.94 pp | 0.40 % | -2.92 pp | 0.42 % | -2.90 pp |
| DRAM Throughput (%) | 1.97 % | 0.06 % | -1.91 pp | 0.06 % | -1.91 pp | 0.06 % | -1.91 pp | 0.07 % | -1.90 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 |
| Waves/SM | 0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 |
| No Eligible (%) | 74.85 % | 95.16 % | +20.31 pp | 95.35 % | +20.50 pp | 96.45 % | +21.60 pp | 95.14 % | +20.29 pp |
| IPC Active | 1.00 inst/cycle | 0.05 inst/cycle | -0.95 | 0.04 inst/cycle | -0.96 | 0.05 inst/cycle | -0.95 | 0.04 inst/cycle | -0.96 |

---

## Iteration 3

**Best so far**: 2.5609× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_callsite_autotune_bypass_via_lambda_grid_and_remove_hardcoded_xblock | 1.693× | 2.517× | ✓ | **✓ best** |
| B | fix_callsite_grid_lambda_and_autotune_num_stages_for_latency_hiding | 1.638× | 2.561× | ✓ |  |
| C | fix_static_grid_to_dynamic_lambda_so_autotune_xblock_takes_effect | 1.690× | 2.561× | ✓ |  |
| D | fix_callsite_to_enable_autotune_and_add_num_stages_pipelining | 1.632× | 2.485× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_callsite_auto... | Δ | fix_callsite_grid... | Δ | fix_static_grid_t... | Δ | fix_callsite_to_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 2340 ns | -3550 ns (-60.3%) | 2300 ns | -3590 ns (-61.0%) | 2300 ns | -3590 ns (-61.0%) | 2370 ns | -3520 ns (-59.8%) |
| SM Throughput (%) | 2.85 % | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp |
| Memory Throughput (%) | 3.32 % | 0.45 % | -2.87 pp | 0.42 % | -2.90 pp | 0.48 % | -2.84 pp | 0.43 % | -2.89 pp |
| DRAM Throughput (%) | 1.97 % | 0.07 % | -1.90 pp | 0.07 % | -1.90 pp | 0.07 % | -1.90 pp | 0.07 % | -1.90 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 |
| Waves/SM | 0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 |
| No Eligible (%) | 74.85 % | 95.16 % | +20.31 pp | 95.16 % | +20.31 pp | 96.24 % | +21.39 pp | 95.28 % | +20.43 pp |
| IPC Active | 1.00 inst/cycle | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 | 0.03 inst/cycle | -0.97 |

---

## Iteration 4

**Best so far**: 2.5609× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_single_block_for_full_cache_reuse | 1.702× | 2.517× | ✓ | **✓ best** |
| B | increase_tile_coverage_via_persistent_x_loop_within_block | 0.127× | 0.082× | ✓ |  |
| C | fix_callsite_bypass_and_add_large_xblock_configs_for_cache_reuse | 1.639× | 2.165× | ✓ |  |
| D | increase_tile_sizes_via_lambda_grid_and_larger_xblock_autotune_configs | 1.621× | 2.485× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_tile_cov... | Δ | fix_callsite_bypa... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 2340 ns | -3550 ns (-60.3%) | 71740 ns | +65850 ns (+1118.0%) | 2720 ns | -3170 ns (-53.8%) | 2370 ns | -3520 ns (-59.8%) |
| SM Throughput (%) | 2.85 % | 0.04 % | -2.81 pp | 0.15 % | -2.70 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp |
| Memory Throughput (%) | 3.32 % | 0.43 % | -2.89 pp | 0.16 % | -3.16 pp | 0.36 % | -2.96 pp | 0.42 % | -2.90 pp |
| DRAM Throughput (%) | 1.97 % | 0.07 % | -1.90 pp | 0.16 % | -1.81 pp | 0.06 % | -1.91 pp | 0.06 % | -1.91 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 20.00 register/thread | -108 | 16.00 register/thread | -112 | 16.00 register/thread | -112 |
| Waves/SM | 0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 |
| No Eligible (%) | 74.85 % | 95.24 % | +20.39 pp | 94.68 % | +19.83 pp | 95.16 % | +20.31 pp | 95.17 % | +20.32 pp |
| IPC Active | 1.00 inst/cycle | 0.04 inst/cycle | -0.96 | 0.05 inst/cycle | -0.95 | 0.04 inst/cycle | -0.96 | 0.03 inst/cycle | -0.97 |

---

## Iteration 5

**Best so far**: 2.5609× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_xnumel_blocks_with_xblock1_for_full_sm_coverage | 1.603× | 2.517× | ✓ |  |
| B | remove_debug_barrier_and_fix_callsite_lambda_grid_for_autotune | 1.670× | 2.485× | ✓ |  |
| C | add_num_stages_pipelining_and_fix_grid_lambda_for_autotune | 1.622× | 2.485× | ✓ |  |
| D | eliminate_persistent_loop_and_restore_single_pass_vectorized_reduction | 1.613× | 2.119× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | remove_debug_barr... | Δ | add_num_stages_pi... | Δ | eliminate_persist... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 2340 ns | -3550 ns (-60.3%) | 2370 ns | -3520 ns (-59.8%) | 2370 ns | -3520 ns (-59.8%) | 2780 ns | -3110 ns (-52.8%) |
| SM Throughput (%) | 2.85 % | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp | 0.04 % | -2.81 pp |
| Memory Throughput (%) | 3.32 % | 0.44 % | -2.88 pp | 0.41 % | -2.91 pp | 0.41 % | -2.91 pp | 0.40 % | -2.92 pp |
| DRAM Throughput (%) | 1.97 % | 0.07 % | -1.90 pp | 0.07 % | -1.90 pp | 0.07 % | -1.90 pp | 0.06 % | -1.91 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 128.00 register/thread | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 | 16.00 register/thread | -112 |
| Waves/SM | 0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 | 0.00 | -0.03 |
| No Eligible (%) | 74.85 % | 95.24 % | +20.39 pp | 96.34 % | +21.49 pp | 96.60 % | +21.75 pp | 95.24 % | +20.39 pp |
| IPC Active | 1.00 inst/cycle | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 | 0.04 inst/cycle | -0.96 |

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     2300 ns  (2.561×)  — increase_grid_coverage_via_autotune_xblock

Final best: 2340 ns  (2.5171×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},    num_warps=1,  num_stages=1),
+        triton.Config({'XBLOCK': 1},    num_warps=1,  num_stages=2),
+        triton.Config({'XBLOCK': 2},    num_warps=1,  num_stages=1),
+        triton.Config({'XBLOCK': 2},    num_warps=1,  num_stages=2),
+        triton.Config({'XBLOCK': 4},    num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 4},    num_warps=2,  num_stages=2),
+        triton.Config({'XBLOCK': 8},    num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 8},    num_warps=2,  num_stages=3),
+        triton.Config({'XBLOCK': 16},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 32},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 64},   num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 2048
@@ -22,7 +52,7 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 66*x0), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_1 + 66*x0), mask=r0_mask & xmask, other=0.0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
@@ -33,9 +63,7 @@

     tmp9 = tl.full([1, 1], 1.0, tl.float32)
     tmp10 = tl.minimum(tmp8, tmp9)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
-
-
+    tl.store(in_out_ptr0 + (x0), tmp10, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +71,7 @@

     xnumel = 2048
     r0_numel = 66
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2[grid](arg_0, arg_1, 2048, 66, XBLOCK=128)
+    triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2[grid](arg_0, arg_1, 2048, 66)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
