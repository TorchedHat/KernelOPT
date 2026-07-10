# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6240 ns — **Best latency**: 5890 ns (**1.0594×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_and_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_005/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6240 ns
- **SM Throughput**: 15.98 % | **Memory**: 39.14 % | **DRAM**: 19.77 %
- **Achieved Occupancy**: 55.03 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.263× | 1.059× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.246× | 1.454× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6240 ns | 5890 ns | -350 ns (-5.6%) | 4290 ns | -1950 ns (-31.2%) |
| SM Throughput (%) | 15.98 % | 17.17 % | +1.19 pp | 28.05 % | +12.07 pp |
| Memory Throughput (%) | 39.14 % | 41.75 % | +2.61 pp | 22.26 % | -16.88 pp |
| DRAM Throughput (%) | 19.77 % | 21.03 % | +1.26 pp | 3.86 % | -15.91 pp |
| Achieved Occupancy (%) | 55.03 % | 56.12 % | +1.09 pp | 66.93 % | +11.90 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.29 % | 70.60 % | +0.31 pp | 44.58 % | -25.71 pp |
| IPC Active | 1.13 inst/cycle | 1.06 inst/cycle | -0.07 | 2.32 inst/cycle | +1.19 |

---

## Cumulative Progress

```
Baseline:  6240 ns  (1.000×)
Iter 1     5890 ns  (1.059×)  — vectorized_loads_and_autotune_tile_sizes

Final best: 5890 ns  (1.0594×)
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
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=2, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting to 2D
+    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +38,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
