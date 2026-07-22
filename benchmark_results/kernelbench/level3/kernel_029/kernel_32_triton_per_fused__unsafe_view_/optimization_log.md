# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__unsafe_view_native_layer_norm_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 44670 ns — **Best latency**: 2690 ns (**16.6059×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`expand_autotune_with_large_xblock_tiles_and_fix_dynamic_grid_for_cache_reuse`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__unsafe_view_native_layer_norm_32`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_32_triton_per_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.06
- **Registers/thread**: 255
- **Duration**: 44670 ns
- **SM Throughput**: 2.34 % | **Memory**: 3.19 % | **DRAM**: 1.43 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.47 inst/cycle | **No Eligible**: 88.26 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_autotune_xblock_and_maxnreg | 7.594× | 16.423× | ✓ | **✓ best** |
| D | autotune_xblock_and_reduce_register_pressure | 6.071× | 9.690× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 44670 ns | 2720 ns | -41950 ns (-93.9%) | 4610 ns | -40060 ns (-89.7%) |
| SM Throughput (%) | 2.34 % | 0.17 % | -2.17 pp | 1.07 % | -1.27 pp |
| Memory Throughput (%) | 3.19 % | 0.49 % | -2.70 pp | 1.84 % | -1.35 pp |
| DRAM Throughput (%) | 1.43 % | 0.22 % | -1.21 pp | 1.84 % | +0.41 pp |
| Achieved Occupancy (%) | 6.25 % | 3.07 % | -3.18 pp | 3.11 % | -3.14 pp |
| Registers/Thread | 255.00 register/thread | 23.00 register/thread | -232 | 164.00 register/thread | -91 |
| Waves/SM | 0.06 | 0.00 | -0.06 | 0.02 | -0.04 |
| No Eligible (%) | 88.26 % | 93.64 % | +5.38 pp | 70.90 % | -17.36 pp |
| IPC Active | 0.47 inst/cycle | 0.15 inst/cycle | -0.32 | 0.60 inst/cycle | +0.13 |

---

## Iteration 2

**Best so far**: 16.4228× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_site_to_enable_dynamic_grid_from_autotune | 7.480× | 16.423× | ✓ |  |
| B | fix_launch_site_dynamic_grid_and_trim_r0_block_padding | 7.486× | 17.247× | ✓ |  |
| D | fix_r0_block_exact_size_and_expand_autotune_to_small_xblock | 7.600× | 16.244× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_launch_site_t... | Δ | fix_launch_site_d... | Δ | fix_r0_block_exac... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44670 ns | 2720 ns | -41950 ns (-93.9%) | 2590 ns | -42080 ns (-94.2%) | 2750 ns | -41920 ns (-93.8%) |
| SM Throughput (%) | 2.34 % | 0.17 % | -2.17 pp | 0.08 % | -2.26 pp | 0.16 % | -2.18 pp |
| Memory Throughput (%) | 3.19 % | 0.45 % | -2.74 pp | 0.49 % | -2.70 pp | 0.45 % | -2.74 pp |
| DRAM Throughput (%) | 1.43 % | 0.22 % | -1.21 pp | 0.23 % | -1.20 pp | 0.22 % | -1.21 pp |
| Achieved Occupancy (%) | 6.25 % | 3.08 % | -3.17 pp | 1.56 % | -4.69 pp | 3.07 % | -3.18 pp |
| Registers/Thread | 255.00 register/thread | 23.00 register/thread | -232 | 24.00 register/thread | -231 | 23.00 register/thread | -232 |
| Waves/SM | 0.06 | 0.00 | -0.06 | 0.00 | -0.06 | 0.00 | -0.06 |
| No Eligible (%) | 88.26 % | 91.96 % | +3.70 pp | 93.75 % | +5.49 pp | 92.74 % | +4.48 pp |
| IPC Active | 0.47 inst/cycle | 0.12 inst/cycle | -0.35 | 0.07 inst/cycle | -0.40 | 0.14 inst/cycle | -0.33 |

---

## Iteration 3

**Best so far**: 16.4228× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_2_pipelining_to_hide_l1tex_latency_and_fix_static_grid | 7.773× | 16.606× | ✓ | **✓ best** |
| B | fix_static_grid_to_enable_autotune_dynamic_xblock_dispatch | 7.494× | 14.990× | ✓ |  |
| C | expand_autotune_num_warps_variants_with_dynamic_grid_lambda | 7.450× | 15.840× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_2_... | Δ | fix_static_grid_t... | Δ | expand_autotune_n... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44670 ns | 2690 ns | -41980 ns (-94.0%) | 2980 ns | -41690 ns (-93.3%) | 2820 ns | -41850 ns (-93.7%) |
| SM Throughput (%) | 2.34 % | 0.06 % | -2.28 pp | 0.17 % | -2.17 pp | 0.07 % | -2.27 pp |
| Memory Throughput (%) | 3.19 % | 0.50 % | -2.69 pp | 0.42 % | -2.77 pp | 0.43 % | -2.76 pp |
| DRAM Throughput (%) | 1.43 % | 0.22 % | -1.21 pp | 0.20 % | -1.23 pp | 0.21 % | -1.22 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 3.07 % | -3.18 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 255.00 register/thread | 24.00 register/thread | -231 | 23.00 register/thread | -232 | 24.00 register/thread | -231 |
| Waves/SM | 0.06 | 0.00 | -0.06 | 0.00 | -0.06 | 0.00 | -0.06 |
| No Eligible (%) | 88.26 % | 92.52 % | +4.26 pp | 92.11 % | +3.85 pp | 93.67 % | +5.41 pp |
| IPC Active | 0.47 inst/cycle | 0.07 inst/cycle | -0.40 | 0.15 inst/cycle | -0.32 | 0.06 inst/cycle | -0.41 |

---

## Iteration 4

**Best so far**: 16.6059× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_autotune_dynamic_grid_and_exact_r0_block_for_sm_saturation | 7.614× | 17.449× | ✓ |  |
| B | fix_static_grid_lambda_and_trim_r0block_to_exact_size | 7.282× | 15.194× | ✓ |  |
| C | increase_tile_sizes_with_dynamic_grid_and_num_stages_pipelining | 7.228× | 16.068× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_autotune_d... | Δ | fix_static_grid_l... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44670 ns | 2560 ns | -42110 ns (-94.3%) | 2940 ns | -41730 ns (-93.4%) | 2780 ns | -41890 ns (-93.8%) |
| SM Throughput (%) | 2.34 % | 0.07 % | -2.27 pp | 0.17 % | -2.17 pp | 0.07 % | -2.27 pp |
| Memory Throughput (%) | 3.19 % | 0.49 % | -2.70 pp | 0.43 % | -2.76 pp | 0.45 % | -2.74 pp |
| DRAM Throughput (%) | 1.43 % | 0.23 % | -1.20 pp | 0.20 % | -1.23 pp | 0.21 % | -1.22 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 3.07 % | -3.18 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 255.00 register/thread | 24.00 register/thread | -231 | 23.00 register/thread | -232 | 24.00 register/thread | -231 |
| Waves/SM | 0.06 | 0.00 | -0.06 | 0.00 | -0.06 | 0.00 | -0.06 |
| No Eligible (%) | 88.26 % | 92.64 % | +4.38 pp | 93.81 % | +5.55 pp | 93.90 % | +5.64 pp |
| IPC Active | 0.47 inst/cycle | 0.06 inst/cycle | -0.41 | 0.13 inst/cycle | -0.34 | 0.07 inst/cycle | -0.40 |

---

## Iteration 5

**Best so far**: 16.6059× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_with_large_xblock_tiles_and_fix_dynamic_grid_for_cache_reuse | 7.462× | 16.606× | ✓ |  |
| B | enable_dynamic_grid_lambda_to_unblock_autotune_xblock_dispatch | 7.301× | 13.959× | ✓ |  |
| C | fix_static_grid_to_enable_autotune_dynamic_xblock_dispatch_v2 | 7.446× | 17.449× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_w... | Δ | enable_dynamic_gr... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 44670 ns | 2690 ns | -41980 ns (-94.0%) | 3200 ns | -41470 ns (-92.8%) | 2560 ns | -42110 ns (-94.3%) |
| SM Throughput (%) | 2.34 % | 0.07 % | -2.27 pp | 0.16 % | -2.18 pp | 0.07 % | -2.27 pp |
| Memory Throughput (%) | 3.19 % | 0.49 % | -2.70 pp | 0.40 % | -2.79 pp | 0.50 % | -2.69 pp |
| DRAM Throughput (%) | 1.43 % | 0.22 % | -1.21 pp | 0.18 % | -1.25 pp | 0.23 % | -1.20 pp |
| Achieved Occupancy (%) | 6.25 % | 1.56 % | -4.69 pp | 3.07 % | -3.18 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 255.00 register/thread | 24.00 register/thread | -231 | 23.00 register/thread | -232 | 24.00 register/thread | -231 |
| Waves/SM | 0.06 | 0.00 | -0.06 | 0.00 | -0.06 | 0.00 | -0.06 |
| No Eligible (%) | 88.26 % | 92.78 % | +4.52 pp | 92.45 % | +4.19 pp | 93.63 % | +5.37 pp |
| IPC Active | 0.47 inst/cycle | 0.06 inst/cycle | -0.41 | 0.15 inst/cycle | -0.32 | 0.07 inst/cycle | -0.40 |

---

## Cumulative Progress

```
Baseline:  44670 ns  (1.000×)
Iter 1     2720 ns  (16.423×)  — reduce_register_pressure_via_autotune_xblock_and_maxnreg
Iter 3     2690 ns  (16.606×)  — add_num_stages_2_pipelining_to_hide_l1tex_latency_and_fix_static_grid

Final best: 2690 ns  (16.6059×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=1, num_stages=1),  # NEW
+        triton.Config({'XBLOCK': 1},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2},   num_warps=4, num_stages=1),  # NEW
+        triton.Config({'XBLOCK': 4},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=8, num_stages=1),  # NEW
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=8, num_stages=1),  # NEW
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__unsafe_view_native_layer_norm_32(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 1960
@@ -22,7 +42,7 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), mask=r0_mask & xmask, other=0.0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
     tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
@@ -36,10 +56,8 @@

     tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
     tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
     tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x0), tmp10, xmask)
-    tl.store(out_ptr1 + (x0), tmp16, xmask)
-
-
+    tl.store(out_ptr0 + (x0), tmp10, mask=xmask)
+    tl.store(out_ptr1 + (x0), tmp16, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -48,7 +66,7 @@

     xnumel = 1960
     r0_numel = 384
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__unsafe_view_native_layer_norm_32[grid](arg_0, arg_1, arg_2, 1960, 384, XBLOCK=128)
+    triton_per_fused__unsafe_view_native_layer_norm_32[grid](arg_0, arg_1, arg_2, 1960, 384)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
