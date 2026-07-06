# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 4030 ns (**1.4839×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_081/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.65 % | **Memory**: 41.08 % | **DRAM**: 20.56 %
- **Achieved Occupancy**: 55.06 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune_tile_sizes | 1.442× | 1.812× | ✓ |  |
| B | autotune_tile_sizes_and_fix_coalescing | 1.259× | 1.049× | ✓ |  |
| C | memory_coalescing_fix_strided_store | 1.419× | 1.105× | ✓ |  |
| D | autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel | 1.480× | 1.484× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3300 ns | -2680 ns (-44.8%) | 5700 ns | -280 ns (-4.7%) | 5410 ns | -570 ns (-9.5%) | 4030 ns | -1950 ns (-32.6%) |
| SM Throughput (%) | 16.65 % | 9.80 % | -6.85 pp | 24.33 % | +7.68 pp | 18.53 % | +1.88 pp | 19.37 % | +2.72 pp |
| Memory Throughput (%) | 41.08 % | 20.99 % | -20.09 pp | 35.99 % | -5.09 pp | 39.73 % | -1.35 pp | 22.30 % | -18.78 pp |
| DRAM Throughput (%) | 20.56 % | 5.01 % | -15.55 pp | 11.52 % | -9.04 pp | 22.87 % | +2.31 pp | 4.09 % | -16.47 pp |
| Achieved Occupancy (%) | 55.06 % | 56.33 % | +1.27 pp | 50.88 % | -4.18 pp | 51.60 % | -3.46 pp | 48.72 % | -6.34 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 38.00 register/thread | +12 | 32.00 register/thread | +6 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.99 | +0.25 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.34 % | 72.81 % | +1.47 pp | 54.66 % | -16.68 pp | 68.07 % | -3.27 pp | 51.57 % | -19.77 pp |
| IPC Active | 1.14 inst/cycle | 0.89 inst/cycle | -0.25 | 1.77 inst/cycle | +0.63 | 1.29 inst/cycle | +0.15 | 1.82 inst/cycle | +0.68 |

---

## Iteration 2

**Best so far**: 1.4839× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_arange_axes_for_coalesced_nhwc_store | 1.410× | 1.812× | ✓ |  |
| B | vectorized_load_hints_on_contiguous_xarange | 1.442× | 1.458× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.435× | 1.748× | ✓ |  |
| D | vectorized_loads_with_max_contiguous_multiple_of_hints | 1.409× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_arange_axes_... | Δ | vectorized_load_h... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3300 ns | -2680 ns (-44.8%) | 4100 ns | -1880 ns (-31.4%) | 3420 ns | -2560 ns (-42.8%) | 5890 ns | -90 ns (-1.5%) |
| SM Throughput (%) | 16.65 % | 9.78 % | -6.87 pp | 19.07 % | +2.42 pp | 9.81 % | -6.84 pp | 24.53 % | +7.88 pp |
| Memory Throughput (%) | 41.08 % | 21.12 % | -19.96 pp | 22.05 % | -19.03 pp | 20.25 % | -20.83 pp | 36.13 % | -4.95 pp |
| DRAM Throughput (%) | 20.56 % | 5.01 % | -15.55 pp | 4.05 % | -16.51 pp | 4.83 % | -15.73 pp | 11.18 % | -9.38 pp |
| Achieved Occupancy (%) | 55.06 % | 56.07 % | +1.01 pp | 50.20 % | -4.86 pp | 51.79 % | -3.27 pp | 53.08 % | -1.98 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 16.00 register/thread | -10 | 38.00 register/thread | +12 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.99 | +0.25 |
| No Eligible (%) | 71.34 % | 74.41 % | +3.07 pp | 51.83 % | -19.51 pp | 70.06 % | -1.28 pp | 56.12 % | -15.22 pp |
| IPC Active | 1.14 inst/cycle | 0.90 inst/cycle | -0.24 | 1.75 inst/cycle | +0.61 | 1.06 inst/cycle | -0.08 | 1.69 inst/cycle | +0.55 |

---

## Iteration 3

**Best so far**: 1.4839× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_load_with_evict_first_store_and_expanded_autotune | 1.416× | 1.458× | ✓ |  |
| B | evict_first_store_policy_with_expanded_yblock_autotune | 1.431× | 1.812× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_tile_exploration | 1.449× | 1.448× | ✓ |  |
| D | reduce_register_pressure_via_index_simplification | 1.429× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_load_wi... | Δ | evict_first_store... | Δ | fix_autotune_grid... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4100 ns | -1880 ns (-31.4%) | 3300 ns | -2680 ns (-44.8%) | 4130 ns | -1850 ns (-30.9%) | 5980 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 16.65 % | 17.34 % | +0.69 pp | 10.62 % | -6.03 pp | 18.40 % | +1.75 pp | 22.53 % | +5.88 pp |
| Memory Throughput (%) | 41.08 % | 33.83 % | -7.25 pp | 21.18 % | -19.90 pp | 21.25 % | -19.83 pp | 33.25 % | -7.83 pp |
| DRAM Throughput (%) | 20.56 % | 16.19 % | -4.37 pp | 5.02 % | -15.54 pp | 3.99 % | -16.57 pp | 11.05 % | -9.51 pp |
| Achieved Occupancy (%) | 55.06 % | 54.38 % | -0.68 pp | 52.57 % | -2.49 pp | 48.70 % | -6.36 pp | 50.90 % | -4.16 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 16.00 register/thread | -10 | 25.00 register/thread | -1 | 38.00 register/thread | +12 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.99 | +0.25 |
| No Eligible (%) | 71.34 % | 63.52 % | -7.82 pp | 72.89 % | +1.55 pp | 52.42 % | -18.92 pp | 54.09 % | -17.25 pp |
| IPC Active | 1.14 inst/cycle | 1.39 inst/cycle | +0.25 | 1.21 inst/cycle | +0.07 | 1.83 inst/cycle | +0.69 | 1.78 inst/cycle | +0.64 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     4030 ns  (1.484×)  — autotune_tile_sizes_and_warps_for_underutilized_transpose_kernel

Final best: 4030 ns  (1.4839×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,40 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=8, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
     xnumel = 50176
+    xoffset = tl.program_id(0) * XBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # NOW slow outer axis
+    xmask = xindex < xnumel
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    # Apply vectorization hints to 1D arange before broadcasting (now on fast y axis)
+    yarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
+    yindex = yoffset + yarange[None, :]                 # NOW fast inner axis (stride-1 output dim)
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 3)
+    y0 = (yindex % 3)   # stride-1 in output -- now on fast axis -> coalesced stores
     y1 = yindex // 3
     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +46,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
