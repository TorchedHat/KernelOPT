# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 3170 ns (**1.8991×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_layout_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_065/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.19 % | **Memory**: 41.06 % | **DRAM**: 20.52 %
- **Achieved Occupancy**: 55.62 %
- **IPC Active**: 1.12 inst/cycle | **No Eligible**: 71.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.250× | 1.306× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_layout_transpose | 1.417× | 1.899× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4610 ns | -1410 ns (-23.4%) | 3170 ns | -2850 ns (-47.3%) |
| SM Throughput (%) | 16.19 % | 21.15 % | +4.96 pp | 4.74 % | -11.45 pp |
| Memory Throughput (%) | 41.06 % | 30.46 % | -10.60 pp | 11.17 % | -29.89 pp |
| DRAM Throughput (%) | 20.52 % | 14.37 % | -6.15 pp | 2.62 % | -17.90 pp |
| Achieved Occupancy (%) | 55.62 % | 47.15 % | -8.47 pp | 26.26 % | -29.36 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.42 % | 55.75 % | -15.67 pp | 81.34 % | +9.92 pp |
| IPC Active | 1.12 inst/cycle | 1.69 inst/cycle | +0.57 | 0.68 inst/cycle | -0.44 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3170 ns  (1.899×)  — autotune_tile_sizes_and_warps_for_layout_transpose

Final best: 3170 ns  (1.8991×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1},  num_warps=2,  num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1},  num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 8},  num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4,  num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 32}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=8,  num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +43,13 @@

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
