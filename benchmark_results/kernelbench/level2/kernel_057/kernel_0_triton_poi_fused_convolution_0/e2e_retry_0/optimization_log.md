# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6340 ns — **Best latency**: 3520 ns (**1.8011×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_to_increase_wave_count_and_reduce_uncoalesced_stores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6340 ns
- **SM Throughput**: 16.11 % | **Memory**: 39.68 % | **DRAM**: 19.55 %
- **Achieved Occupancy**: 55.14 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.02 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.258× | 1.060× | ✓ |  |
| B | autotune_tile_sizes_to_increase_wave_count_and_reduce_uncoalesced_stores | 1.322× | 1.801× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6340 ns | 5980 ns | -360 ns (-5.7%) | 3520 ns | -2820 ns (-44.5%) |
| SM Throughput (%) | 16.11 % | 16.75 % | +0.64 pp | 9.75 % | -6.36 pp |
| Memory Throughput (%) | 39.68 % | 42.16 % | +2.48 pp | 16.90 % | -22.78 pp |
| DRAM Throughput (%) | 19.55 % | 20.53 % | +0.98 pp | 8.80 % | -10.75 pp |
| Achieved Occupancy (%) | 55.14 % | 55.36 % | +0.22 pp | 29.96 % | -25.18 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 20.00 register/thread | -6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.02 % | 71.53 % | +0.51 pp | 73.38 % | +2.36 pp |
| IPC Active | 1.16 inst/cycle | 1.14 inst/cycle | -0.02 | 1.01 inst/cycle | -0.15 |

---

## Cumulative Progress

```
Baseline:  6340 ns  (1.000×)
Iter 1     3520 ns  (1.801×)  — autotune_tile_sizes_to_increase_wave_count_and_reduce_uncoalesced_stores

Final best: 3520 ns  (1.8011×)
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
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 8},  num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16}, num_warps=4, num_stages=4),
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

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
