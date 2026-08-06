# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3620 ns — **Best latency**: 3040 ns (**1.1908×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_tile_dimensions_and_autotune_for_underutilized_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3620 ns
- **SM Throughput**: 21.53 % | **Memory**: 19.89 % | **DRAM**: 2.55 %
- **Achieved Occupancy**: 52.83 %
- **IPC Active**: 1.79 inst/cycle | **No Eligible**: 54.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_dimensions_and_autotune_for_underutilized_grid | 1.153× | 1.191× | ✓ | **✓ best** |
| B | fix_tile_sizes_and_add_autotune_for_underutilized_transpose_kernel | 1.105× | 1.257× | ✓ |  |
| C | add_triton_autotune | 1.043× | 0.992× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_dimensio... | Δ | fix_tile_sizes_an... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 3040 ns | -580 ns (-16.0%) | 2880 ns | -740 ns (-20.4%) | 3650 ns | +30 ns (+0.8%) |
| SM Throughput (%) | 21.53 % | 4.88 % | -16.65 pp | 4.82 % | -16.71 pp | 23.00 % | +1.47 pp |
| Memory Throughput (%) | 19.89 % | 3.02 % | -16.87 pp | 3.19 % | -16.70 pp | 21.22 % | +1.33 pp |
| DRAM Throughput (%) | 2.55 % | 3.02 % | +0.47 pp | 3.19 % | +0.64 pp | 2.52 % | -0.03 pp |
| Achieved Occupancy (%) | 52.83 % | 16.12 % | -36.71 pp | 16.47 % | -36.36 pp | 51.55 % | -1.28 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.73 | 0.36 | -0.37 | 0.36 | -0.37 | 0.73 | +0.00 |
| No Eligible (%) | 54.58 % | 80.08 % | +25.50 pp | 81.09 % | +26.51 pp | 45.62 % | -8.96 pp |
| IPC Active | 1.79 inst/cycle | 0.59 inst/cycle | -1.20 | 0.60 inst/cycle | -1.19 | 2.01 inst/cycle | +0.22 |

---

## Cumulative Progress

```
Baseline:  3620 ns  (1.000×)
Iter 1     3040 ns  (1.191×)  — fix_tile_dimensions_and_autotune_for_underutilized_grid

Final best: 3040 ns  (1.1908×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +44,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
