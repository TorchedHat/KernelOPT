# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6140 ns — **Best latency**: 3200 ns (**1.9187×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_078/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6140 ns
- **SM Throughput**: 16.25 % | **Memory**: 40.04 % | **DRAM**: 20.04 %
- **Achieved Occupancy**: 55.38 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.273× | 1.048× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.450× | 1.919× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 5860 ns | -280 ns (-4.6%) | 3200 ns | -2940 ns (-47.9%) |
| SM Throughput (%) | 16.25 % | 23.84 % | +7.59 pp | 5.11 % | -11.14 pp |
| Memory Throughput (%) | 40.04 % | 35.25 % | -4.79 pp | 11.34 % | -28.70 pp |
| DRAM Throughput (%) | 20.04 % | 11.24 % | -8.80 pp | 2.62 % | -17.42 pp |
| Achieved Occupancy (%) | 55.38 % | 51.30 % | -4.08 pp | 27.24 % | -28.14 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 0.37 | -0.37 |
| No Eligible (%) | 70.65 % | 53.74 % | -16.91 pp | 83.97 % | +13.32 pp |
| IPC Active | 1.13 inst/cycle | 1.81 inst/cycle | +0.68 | 0.70 inst/cycle | -0.43 |

---

## Cumulative Progress

```
Baseline:  6140 ns  (1.000×)
Iter 1     3200 ns  (1.919×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 3200 ns  (1.9187×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +44,13 @@

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
