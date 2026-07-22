# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2660 ns — **Best latency**: 2560 ns (**1.0391×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_36_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.05
- **Registers/thread**: 28
- **Duration**: 2660 ns
- **SM Throughput**: 1.95 % | **Memory**: 1.86 % | **DRAM**: 0.63 %
- **Achieved Occupancy**: 6.11 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_small_kernel | 1.034× | 1.039× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2660 ns | 2560 ns | -100 ns (-3.8%) |
| SM Throughput (%) | 1.95 % | 0.61 % | -1.34 pp |
| Memory Throughput (%) | 1.86 % | 0.56 % | -1.30 pp |
| DRAM Throughput (%) | 0.63 % | 0.42 % | -0.21 pp |
| Achieved Occupancy (%) | 6.11 % | 6.04 % | -0.07 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.05 | 0.05 | +0.00 |
| No Eligible (%) | 91.03 % | 91.68 % | +0.65 pp |
| IPC Active | 0.35 inst/cycle | 0.24 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  2660 ns  (1.000×)
Iter 1     2560 ns  (1.039×)  — autotune_tile_sizes_for_small_kernel

Final best: 2560 ns  (1.0391×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},   num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 768
@@ -22,15 +40,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 768
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
