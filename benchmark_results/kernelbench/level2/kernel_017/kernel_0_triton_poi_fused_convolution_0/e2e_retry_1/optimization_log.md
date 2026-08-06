# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 4450 ns (**1.3011×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_017/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.38 % | **Memory**: 43.47 % | **DRAM**: 21.24 %
- **Achieved Occupancy**: 55.09 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 71.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.284× | 1.016× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.400× | 1.301× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 5700 ns | -90 ns (-1.6%) | 4450 ns | -1340 ns (-23.1%) |
| SM Throughput (%) | 16.38 % | 35.11 % | +18.73 pp | 24.71 % | +8.33 pp |
| Memory Throughput (%) | 43.47 % | 42.74 % | -0.73 pp | 19.62 % | -23.85 pp |
| DRAM Throughput (%) | 21.24 % | 11.53 % | -9.71 pp | 3.70 % | -17.54 pp |
| Achieved Occupancy (%) | 55.09 % | 52.88 % | -2.21 pp | 68.31 % | +13.22 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.15 % | 39.77 % | -31.38 pp | 37.52 % | -33.63 pp |
| IPC Active | 1.15 inst/cycle | 2.38 inst/cycle | +1.23 | 2.37 inst/cycle | +1.22 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     4450 ns  (1.301×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 4450 ns  (1.3011×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -22,15 +36,13 @@

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
