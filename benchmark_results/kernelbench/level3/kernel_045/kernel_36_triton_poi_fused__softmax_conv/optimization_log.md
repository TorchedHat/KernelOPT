# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_sub_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3040 ns — **Best latency**: 2880 ns (**1.0556×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`streaming_eviction_policy_for_transpose_l1tex_stall`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_sub_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_36_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 27
- **Duration**: 3040 ns
- **SM Throughput**: 13.23 % | **Memory**: 13.17 % | **DRAM**: 0.92 %
- **Achieved Occupancy**: 40.18 %
- **IPC Active**: 1.41 inst/cycle | **No Eligible**: 59.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | — | 0.016× | ✓ |  |
| B | autotune_tile_sizes_fix_grid_underutilization | — | 1.020× | ✓ |  |
| C | fix_tile_dimensions_and_autotune_for_underutilized_transpose_kernel | — | 1.056× | ✓ | **✓ best** |
| D | autotune_tile_sizes_fix_xblock_yblock_for_underutilized_grid | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 185020 ns | +181980 ns (+5986.2%) | 2980 ns | -60 ns (-2.0%) | 2880 ns | -160 ns (-5.3%) |
| SM Throughput (%) | 13.23 % | 21.53 % | +8.30 pp | 9.59 % | -3.64 pp | 2.27 % | -10.96 pp |
| Memory Throughput (%) | 13.17 % | 14.47 % | +1.30 pp | 5.53 % | -7.64 pp | 5.53 % | -7.64 pp |
| DRAM Throughput (%) | 0.92 % | 0.01 % | -0.91 pp | 3.67 % | +2.75 pp | 3.81 % | +2.89 pp |
| Achieved Occupancy (%) | 40.18 % | 24.16 % | -16.02 pp | 39.48 % | -0.70 pp | 9.37 % | -30.81 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 | 16.00 register/thread | -11 | 16.00 register/thread | -11 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.48 | +0.00 | 0.24 | -0.24 |
| No Eligible (%) | 59.53 % | 81.75 % | +22.22 pp | 72.28 % | +12.75 pp | 91.36 % | +31.83 pp |
| IPC Active | 1.41 inst/cycle | 0.73 inst/cycle | -0.68 | 1.02 inst/cycle | -0.39 | 0.27 inst/cycle | -1.14 |

---

## Iteration 2

**Best so far**: 1.0556× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staged_transpose_for_coalesced_stores | — | 1.056× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3040 ns | 2880 ns | -160 ns (-5.3%) |
| SM Throughput (%) | 13.23 % | 2.28 % | -10.95 pp |
| Memory Throughput (%) | 13.17 % | 5.58 % | -7.59 pp |
| DRAM Throughput (%) | 0.92 % | 3.79 % | +2.87 pp |
| Achieved Occupancy (%) | 40.18 % | 9.37 % | -30.81 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 |
| Waves/SM | 0.48 | 0.24 | -0.24 |
| No Eligible (%) | 59.53 % | 91.36 % | +31.83 pp |
| IPC Active | 1.41 inst/cycle | 0.27 inst/cycle | -1.14 |

---

## Iteration 3

**Best so far**: 1.0556× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_transpose_l1tex_stall | — | 1.056× | ✓ |  |
| C | evict_first_load_policy_for_streaming_transpose | — | 1.067× | ✓ |  |
| D | fuse_exp_sub_elementwise_into_transpose_kernel | — | 1.067× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | evict_first_load_... | Δ | fuse_exp_sub_elem... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3040 ns | 2880 ns | -160 ns (-5.3%) | 2850 ns | -190 ns (-6.2%) | 2850 ns | -190 ns (-6.2%) |
| SM Throughput (%) | 13.23 % | 2.24 % | -10.99 pp | 2.27 % | -10.96 pp | 2.27 % | -10.96 pp |
| Memory Throughput (%) | 13.17 % | 5.65 % | -7.52 pp | 5.61 % | -7.56 pp | 5.62 % | -7.55 pp |
| DRAM Throughput (%) | 0.92 % | 3.81 % | +2.89 pp | 3.81 % | +2.89 pp | 3.82 % | +2.90 pp |
| Achieved Occupancy (%) | 40.18 % | 9.60 % | -30.58 pp | 9.50 % | -30.68 pp | 9.82 % | -30.36 pp |
| Registers/Thread | 27.00 register/thread | 16.00 register/thread | -11 | 16.00 register/thread | -11 | 16.00 register/thread | -11 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.24 | -0.24 | 0.24 | -0.24 |
| No Eligible (%) | 59.53 % | 91.45 % | +31.92 pp | 92.19 % | +32.66 pp | 91.09 % | +31.56 pp |
| IPC Active | 1.41 inst/cycle | 0.27 inst/cycle | -1.14 | 0.26 inst/cycle | -1.15 | 0.24 inst/cycle | -1.17 |

---

## Cumulative Progress

```
Baseline:  3040 ns  (1.000×)
Iter 1     2880 ns  (1.056×)  — fix_tile_dimensions_and_autotune_for_underutilized_transpose_kernel

Final best: 2880 ns  (1.0556×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 32},  num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 32},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'YBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 2, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
@@ -19,10 +36,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +45,7 @@

     ynumel = 8192
     xnumel = 4
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
