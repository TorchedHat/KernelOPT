# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 3230 ns (**1.8638×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_improve_occupancy_and_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_036/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 16.08 % | **Memory**: 40.98 % | **DRAM**: 20.40 %
- **Achieved Occupancy**: 54.79 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.53 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.383× | 1.447× | ✓ |  |
| B | autotune_tile_sizes_to_improve_occupancy_and_coalescing | 1.393× | 1.864× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4160 ns | -1860 ns (-30.9%) | 3230 ns | -2790 ns (-46.3%) |
| SM Throughput (%) | 16.08 % | 26.67 % | +10.59 pp | 4.57 % | -11.51 pp |
| Memory Throughput (%) | 40.98 % | 20.93 % | -20.05 pp | 11.26 % | -29.72 pp |
| DRAM Throughput (%) | 20.40 % | 3.98 % | -16.42 pp | 2.59 % | -17.81 pp |
| Achieved Occupancy (%) | 54.79 % | 68.20 % | +13.41 pp | 27.49 % | -27.30 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.37 | -0.37 |
| No Eligible (%) | 70.53 % | 39.94 % | -30.59 pp | 79.98 % | +9.45 pp |
| IPC Active | 1.14 inst/cycle | 2.44 inst/cycle | +1.30 | 0.52 inst/cycle | -0.62 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     3230 ns  (1.864×)  — autotune_tile_sizes_to_improve_occupancy_and_coalescing

Final best: 3230 ns  (1.8638×)
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
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +42,13 @@

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
