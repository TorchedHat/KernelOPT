# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5700 ns — **Best latency**: 3170 ns (**1.7981×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_uncoalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_001/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5700 ns
- **SM Throughput**: 15.59 % | **Memory**: 44.04 % | **DRAM**: 21.76 %
- **Achieved Occupancy**: 54.56 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints_and_tile_resize | 1.261× | 1.236× | ✓ |  |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 1.383× | 1.798× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5700 ns | 4610 ns | -1090 ns (-19.1%) | 3170 ns | -2530 ns (-44.4%) |
| SM Throughput (%) | 15.59 % | 21.05 % | +5.46 pp | 5.18 % | -10.41 pp |
| Memory Throughput (%) | 44.04 % | 30.45 % | -13.59 pp | 11.30 % | -32.74 pp |
| DRAM Throughput (%) | 21.76 % | 14.35 % | -7.41 pp | 2.63 % | -19.13 pp |
| Achieved Occupancy (%) | 54.56 % | 45.62 % | -8.94 pp | 26.41 % | -28.15 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.12 % | 55.75 % | -15.37 pp | 84.18 % | +13.06 pp |
| IPC Active | 1.16 inst/cycle | 1.68 inst/cycle | +0.52 | 0.67 inst/cycle | -0.49 |

---

## Cumulative Progress

```
Baseline:  5700 ns  (1.000×)
Iter 1     3170 ns  (1.798×)  — autotune_tile_sizes_and_fix_uncoalesced_stores

Final best: 3170 ns  (1.7981×)
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
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 1},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 2},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +39,13 @@

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
