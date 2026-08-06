# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6140 ns — **Best latency**: 5950 ns (**1.0319×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_and_fix_coalescing`** beats baseline latency.

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
- **Duration**: 6140 ns
- **SM Throughput**: 15.97 % | **Memory**: 40.20 % | **DRAM**: 20.04 %
- **Achieved Occupancy**: 54.99 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.194× | 0.770× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.292× | 1.032× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_underutilized_layout_permutation_kernel | 1.288× | 1.547× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 7970 ns | +1830 ns (+29.8%) | 5950 ns | -190 ns (-3.1%) | 3970 ns | -2170 ns (-35.3%) |
| SM Throughput (%) | 15.97 % | 41.14 % | +25.17 pp | 13.56 % | -2.41 pp | 27.74 % | +11.77 pp |
| Memory Throughput (%) | 40.20 % | 51.54 % | +11.34 pp | 24.74 % | -15.46 pp | 22.03 % | -18.17 pp |
| DRAM Throughput (%) | 20.04 % | 15.41 % | -4.63 pp | 5.17 % | -14.87 pp | 4.16 % | -15.88 pp |
| Achieved Occupancy (%) | 54.99 % | 78.59 % | +23.60 pp | 43.56 % | -11.43 pp | 68.98 % | +13.99 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 24.00 register/thread | -2 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 71.33 % | 44.88 % | -26.45 pp | 68.87 % | -2.46 pp | 37.47 % | -33.86 pp |
| IPC Active | 1.14 inst/cycle | 2.27 inst/cycle | +1.13 | 1.22 inst/cycle | +0.08 | 2.40 inst/cycle | +1.26 |

---

## Iteration 2

**Best so far**: 1.0319× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_on_pointer_offset_with_evict_first_store | 1.268× | 1.389× | ✓ |  |
| B | vectorized_loads_with_fixed_autotune_grid_and_expanded_configs | 1.291× | 1.512× | ✓ |  |
| C | fix_autotune_grid_lambda_to_match_selected_tile_sizes | 1.187× | 0.767× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 4420 ns | -1720 ns (-28.0%) | 4060 ns | -2080 ns (-33.9%) | 8000 ns | +1860 ns (+30.3%) |
| SM Throughput (%) | 15.97 % | 27.80 % | +11.83 pp | 26.10 % | +10.13 pp | 40.44 % | +24.47 pp |
| Memory Throughput (%) | 40.20 % | 20.88 % | -19.32 pp | 20.72 % | -19.48 pp | 50.68 % | +10.48 pp |
| DRAM Throughput (%) | 20.04 % | 3.73 % | -16.31 pp | 4.05 % | -15.99 pp | 15.40 % | -4.64 pp |
| Achieved Occupancy (%) | 54.99 % | 68.74 % | +13.75 pp | 67.64 % | +12.65 pp | 78.38 % | +23.39 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 71.33 % | 37.33 % | -34.00 pp | 38.02 % | -33.31 pp | 43.02 % | -28.31 pp |
| IPC Active | 1.14 inst/cycle | 2.43 inst/cycle | +1.29 | 2.37 inst/cycle | +1.23 | 2.26 inst/cycle | +1.12 |

---

## Iteration 3

**Best so far**: 1.0319× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_eliminate_partial_wave_tail | 1.268× | 1.476× | ✓ |  |
| B | fix_autotune_grid_lambda_for_wave_efficiency | 1.125× | 1.431× | ✓ |  |
| D | add_triton_autotune | 1.291× | 1.402× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 4160 ns | -1980 ns (-32.2%) | 4290 ns | -1850 ns (-30.1%) | 4380 ns | -1760 ns (-28.7%) |
| SM Throughput (%) | 15.97 % | 26.13 % | +10.16 pp | 28.78 % | +12.81 pp | 25.26 % | +9.29 pp |
| Memory Throughput (%) | 40.20 % | 20.69 % | -19.51 pp | 21.64 % | -18.56 pp | 19.95 % | -20.25 pp |
| DRAM Throughput (%) | 20.04 % | 3.99 % | -16.05 pp | 3.89 % | -16.15 pp | 3.77 % | -16.27 pp |
| Achieved Occupancy (%) | 54.99 % | 69.31 % | +14.32 pp | 71.01 % | +16.02 pp | 67.52 % | +12.53 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 71.33 % | 37.63 % | -33.70 pp | 37.27 % | -34.06 pp | 38.78 % | -32.55 pp |
| IPC Active | 1.14 inst/cycle | 2.42 inst/cycle | +1.28 | 2.22 inst/cycle | +1.08 | 2.38 inst/cycle | +1.24 |

---

## Cumulative Progress

```
Baseline:  6140 ns  (1.000×)
Iter 1     5950 ns  (1.032×)  — autotune_tile_sizes_and_fix_coalescing

Final best: 5950 ns  (1.0319×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=2),
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

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
