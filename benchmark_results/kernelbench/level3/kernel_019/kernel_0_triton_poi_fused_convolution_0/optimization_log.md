# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5860 ns — **Best latency**: 3140 ns (**1.8662×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5860 ns
- **SM Throughput**: 16.70 % | **Memory**: 41.92 % | **DRAM**: 21.03 %
- **Achieved Occupancy**: 54.98 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.222× | 1.429× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.407× | 1.866× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.212× | 1.419× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 4100 ns | -1760 ns (-30.0%) | 3140 ns | -2720 ns (-46.4%) | 4130 ns | -1730 ns (-29.5%) |
| SM Throughput (%) | 16.70 % | 17.77 % | +1.07 pp | 14.78 % | -1.92 pp | 27.24 % | +10.54 pp |
| Memory Throughput (%) | 41.92 % | 33.62 % | -8.30 pp | 21.96 % | -19.96 pp | 21.68 % | -20.24 pp |
| DRAM Throughput (%) | 21.03 % | 16.03 % | -5.00 pp | 5.30 % | -15.73 pp | 4.00 % | -17.03 pp |
| Achieved Occupancy (%) | 54.98 % | 54.58 % | -0.40 pp | 48.47 % | -6.51 pp | 69.13 % | +14.15 pp |
| Registers/Thread | 26.00 register/thread | 20.00 register/thread | -6 | 16.00 register/thread | -10 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.66 % | 63.34 % | -7.32 pp | 59.98 % | -10.68 pp | 38.12 % | -32.54 pp |
| IPC Active | 1.15 inst/cycle | 1.44 inst/cycle | +0.29 | 1.37 inst/cycle | +0.22 | 2.41 inst/cycle | +1.26 |

---

## Iteration 2

**Best so far**: 1.8662× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads | 1.341× | 1.776× | ✓ |  |
| B | software_pipelining_via_num_stages_autotune | 1.356× | 1.454× | ✓ |  |
| C | vectorized_loads_with_fixed_autotune_callsite | 1.313× | 1.465× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads | Δ | software_pipelini... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 3300 ns | -2560 ns (-43.7%) | 4030 ns | -1830 ns (-31.2%) | 4000 ns | -1860 ns (-31.7%) |
| SM Throughput (%) | 16.70 % | 15.93 % | -0.77 pp | 18.12 % | +1.42 pp | 27.62 % | +10.92 pp |
| Memory Throughput (%) | 41.92 % | 20.95 % | -20.97 pp | 34.11 % | -7.81 pp | 21.93 % | -19.99 pp |
| DRAM Throughput (%) | 21.03 % | 5.03 % | -16.00 pp | 16.27 % | -4.76 pp | 4.16 % | -16.87 pp |
| Achieved Occupancy (%) | 54.98 % | 46.26 % | -8.72 pp | 54.63 % | -0.35 pp | 67.84 % | +12.86 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 20.00 register/thread | -6 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.66 % | 58.88 % | -11.78 pp | 62.95 % | -7.71 pp | 39.31 % | -31.35 pp |
| IPC Active | 1.15 inst/cycle | 1.50 inst/cycle | +0.35 | 1.43 inst/cycle | +0.28 | 2.38 inst/cycle | +1.23 |

---

## Iteration 3

**Best so far**: 1.8662× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_bypassed_autotune_callsite_and_expand_wave_saturating_configs | 1.358× | 1.236× | ✓ |  |
| B | fix_autotuner_bypass_to_eliminate_partial_wave | 1.340× | 1.389× | ✓ |  |
| C | fix_autotune_bypass_at_callsite_to_enable_tile_selection | 1.359× | 1.419× | ✓ |  |
| D | add_triton_autotune | 1.229× | 0.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_bypassed_auto... | Δ | fix_autotuner_byp... | Δ | fix_autotune_bypa... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5860 ns | 4740 ns | -1120 ns (-19.1%) | 4220 ns | -1640 ns (-28.0%) | 4130 ns | -1730 ns (-29.5%) | 5980 ns | +120 ns (+2.0%) |
| SM Throughput (%) | 16.70 % | 41.39 % | +24.69 pp | 27.36 % | +10.66 pp | 17.57 % | +0.87 pp | 19.49 % | +2.79 pp |
| Memory Throughput (%) | 41.92 % | 18.09 % | -23.83 pp | 21.70 % | -20.22 pp | 33.63 % | -8.29 pp | 40.54 % | -1.38 pp |
| DRAM Throughput (%) | 21.03 % | 3.48 % | -17.55 pp | 3.91 % | -17.12 pp | 16.04 % | -4.99 pp | 20.65 % | -0.38 pp |
| Achieved Occupancy (%) | 54.98 % | 74.37 % | +19.39 pp | 69.46 % | +14.48 pp | 55.10 % | +0.12 pp | 56.23 % | +1.25 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 18.00 register/thread | -8 | 20.00 register/thread | -6 | 28.00 register/thread | +2 |
| Waves/SM | 0.74 | 2.97 | +2.23 | 1.48 | +0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.66 % | 27.83 % | -42.83 pp | 39.41 % | -31.25 pp | 62.52 % | -8.14 pp | 67.54 % | -3.12 pp |
| IPC Active | 1.15 inst/cycle | 2.96 inst/cycle | +1.81 | 2.34 inst/cycle | +1.19 | 1.40 inst/cycle | +0.25 | 1.26 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  5860 ns  (1.000×)
Iter 1     3140 ns  (1.866×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3140 ns  (1.8662×)
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
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +29,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    _xrange = tl.multiple_of(xoffset + tl.arange(0, XBLOCK), 16)
+    _xrange = tl.max_contiguous(_xrange, 16)
+    xindex = _xrange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +40,13 @@

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
