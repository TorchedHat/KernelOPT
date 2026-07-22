# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6050 ns — **Best latency**: 3230 ns (**1.8731×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_044/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6050 ns
- **SM Throughput**: 16.05 % | **Memory**: 41.15 % | **DRAM**: 20.44 %
- **Achieved Occupancy**: 54.70 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.145× | 1.112× | ✓ |  |
| B | autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel | 1.357× | 1.873× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6050 ns | 5440 ns | -610 ns (-10.1%) | 3230 ns | -2820 ns (-46.6%) |
| SM Throughput (%) | 16.05 % | 34.97 % | +18.92 pp | 10.21 % | -5.84 pp |
| Memory Throughput (%) | 41.15 % | 35.84 % | -5.31 pp | 11.02 % | -30.13 pp |
| DRAM Throughput (%) | 20.44 % | 12.12 % | -8.32 pp | 2.58 % | -17.86 pp |
| Achieved Occupancy (%) | 54.70 % | 74.39 % | +19.69 pp | 53.40 % | -1.30 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.62 % | 37.79 % | -33.83 pp | 68.46 % | -3.16 pp |
| IPC Active | 1.14 inst/cycle | 2.42 inst/cycle | +1.28 | 1.18 inst/cycle | +0.04 |

---

## Cumulative Progress

```
Baseline:  6050 ns  (1.000×)
Iter 1     3230 ns  (1.873×)  — autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel

Final best: 3230 ns  (1.8731×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8,  num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +41,13 @@

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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
