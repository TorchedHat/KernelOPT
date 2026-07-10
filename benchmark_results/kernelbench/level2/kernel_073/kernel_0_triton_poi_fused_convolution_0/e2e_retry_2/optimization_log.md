# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 3390 ns (**1.7640×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_reduce_uncoalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_073/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.11 % | **Memory**: 41.79 % | **DRAM**: 20.61 %
- **Achieved Occupancy**: 55.29 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_tile_size_increase | 0.990× | 0.974× | ✓ |  |
| B | autotune_tile_sizes_reduce_uncoalesced_stores | 1.396× | 1.764× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 6140 ns | +160 ns (+2.7%) | 3390 ns | -2590 ns (-43.3%) |
| SM Throughput (%) | 16.11 % | 15.72 % | -0.39 pp | 17.71 % | +1.60 pp |
| Memory Throughput (%) | 41.79 % | 41.05 % | -0.74 pp | 20.53 % | -21.26 pp |
| DRAM Throughput (%) | 20.61 % | 20.04 % | -0.57 pp | 4.93 % | -15.68 pp |
| Achieved Occupancy (%) | 55.29 % | 55.64 % | +0.35 pp | 44.08 % | -11.21 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.70 % | 71.20 % | +0.50 pp | 55.00 % | -15.70 pp |
| IPC Active | 1.15 inst/cycle | 1.10 inst/cycle | -0.05 | 1.70 inst/cycle | +0.55 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     3390 ns  (1.764×)  — autotune_tile_sizes_reduce_uncoalesced_stores

Final best: 3390 ns  (1.7640×)
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
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2},  num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
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
