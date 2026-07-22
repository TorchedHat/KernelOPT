# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 3010 ns (**1.1694×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_sizes_and_add_autotune_for_underutilized_pointwise_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3520 ns
- **SM Throughput**: 21.68 % | **Memory**: 20.00 % | **DRAM**: 2.60 %
- **Achieved Occupancy**: 46.69 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 50.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_and_add_autotune_for_underutilized_pointwise_kernel | 1.139× | 1.169× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.063× | 1.169× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 3010 ns | -510 ns (-14.5%) | 3010 ns | -510 ns (-14.5%) |
| SM Throughput (%) | 21.68 % | 8.17 % | -13.51 pp | 12.54 % | -9.14 pp |
| Memory Throughput (%) | 20.00 % | 3.06 % | -16.94 pp | 5.60 % | -14.40 pp |
| DRAM Throughput (%) | 2.60 % | 3.06 % | +0.46 pp | 3.03 % | +0.43 pp |
| Achieved Occupancy (%) | 46.69 % | 38.33 % | -8.36 pp | 45.91 % | -0.78 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 20.00 register/thread | -8 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 50.70 % | 66.45 % | +15.75 pp | 50.02 % | -0.68 pp |
| IPC Active | 1.67 inst/cycle | 1.13 inst/cycle | -0.54 | 1.75 inst/cycle | +0.08 |

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 1     3010 ns  (1.169×)  — fix_tile_sizes_and_add_autotune_for_underutilized_pointwise_kernel

Final best: 3010 ns  (1.1694×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +38,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
