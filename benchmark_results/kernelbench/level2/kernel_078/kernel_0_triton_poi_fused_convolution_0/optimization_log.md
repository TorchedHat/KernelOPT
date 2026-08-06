# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 4130 ns (**1.4722×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_num_stages_for_software_pipelining`** beats baseline latency.

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
- **Duration**: 6080 ns
- **SM Throughput**: 15.80 % | **Memory**: 40.65 % | **DRAM**: 20.32 %
- **Achieved Occupancy**: 55.05 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.270× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_to_improve_coalescing_and_occupancy | 1.271× | 0.590× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.424× | 1.472× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 6080 ns | +0 ns (+0.0%) | 10300 ns | +4220 ns (+69.4%) | 4130 ns | -1950 ns (-32.1%) |
| SM Throughput (%) | 15.80 % | 16.90 % | +1.10 pp | 14.42 % | -1.38 pp | 19.17 % | +3.37 pp |
| Memory Throughput (%) | 40.65 % | 40.84 % | +0.19 pp | 33.46 % | -7.19 pp | 22.16 % | -18.49 pp |
| DRAM Throughput (%) | 20.32 % | 20.21 % | -0.11 pp | 5.99 % | -14.33 pp | 4.01 % | -16.31 pp |
| Achieved Occupancy (%) | 55.05 % | 54.77 % | -0.28 pp | 52.89 % | -2.16 pp | 49.96 % | -5.09 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.48 % | 70.71 % | +0.23 pp | 75.12 % | +4.64 pp | 52.11 % | -18.37 pp |
| IPC Active | 1.14 inst/cycle | 1.15 inst/cycle | +0.01 | 1.02 inst/cycle | -0.12 | 1.73 inst/cycle | +0.59 |

---

## Iteration 2

**Best so far**: 1.4722× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_yblock_configs_to_cover_full_y_dimension_for_coalesced_stores | 1.418× | 1.498× | ✓ |  |
| B | expand_autotune_configs_large_xblock_small_yblock_with_lambda_grid | 1.391× | 1.543× | ✓ |  |
| C | fix_autotune_grid_lambda_and_expand_configs_for_layout_permute | 1.420× | 1.520× | ✓ |  |
| D | add_triton_autotune | 1.388× | 1.543× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_yblock_con... | Δ | expand_autotune_c... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4060 ns | -2020 ns (-33.2%) | 3940 ns | -2140 ns (-35.2%) | 4000 ns | -2080 ns (-34.2%) | 3940 ns | -2140 ns (-35.2%) |
| SM Throughput (%) | 15.80 % | 19.14 % | +3.34 pp | 19.61 % | +3.81 pp | 18.84 % | +3.04 pp | 18.98 % | +3.18 pp |
| Memory Throughput (%) | 40.65 % | 22.16 % | -18.49 pp | 22.68 % | -17.97 pp | 21.76 % | -18.89 pp | 21.97 % | -18.68 pp |
| DRAM Throughput (%) | 20.32 % | 4.08 % | -16.24 pp | 4.19 % | -16.13 pp | 4.16 % | -16.16 pp | 4.18 % | -16.14 pp |
| Achieved Occupancy (%) | 55.05 % | 48.38 % | -6.67 pp | 48.95 % | -6.10 pp | 51.01 % | -4.04 pp | 48.92 % | -6.13 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.48 % | 56.39 % | -14.09 pp | 51.86 % | -18.62 pp | 52.35 % | -18.13 pp | 52.29 % | -18.19 pp |
| IPC Active | 1.14 inst/cycle | 1.83 inst/cycle | +0.69 | 1.83 inst/cycle | +0.69 | 1.64 inst/cycle | +0.50 | 1.86 inst/cycle | +0.72 |

---

## Iteration 3

**Best so far**: 1.4722× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_launch_grid_and_add_small_xblock_large_yblock_configs | 1.388× | 1.543× | ✓ |  |
| B | fix_static_launch_grid_to_lambda_and_add_num_stages_to_autotune | 1.417× | 1.441× | ✓ |  |
| C | fix_static_grid_and_add_contiguous_hints_for_vectorized_stores | 1.355× | 1.543× | ✓ |  |
| D | autotune_num_stages_for_software_pipelining | 1.395× | 1.472× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_launch... | Δ | fix_static_launch... | Δ | fix_static_grid_a... | Δ | autotune_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 3940 ns | -2140 ns (-35.2%) | 4220 ns | -1860 ns (-30.6%) | 3940 ns | -2140 ns (-35.2%) | 4130 ns | -1950 ns (-32.1%) |
| SM Throughput (%) | 15.80 % | 18.74 % | +2.94 pp | 19.34 % | +3.54 pp | 21.06 % | +5.26 pp | 19.51 % | +3.71 pp |
| Memory Throughput (%) | 40.65 % | 21.68 % | -18.97 pp | 22.38 % | -18.27 pp | 21.69 % | -18.96 pp | 22.60 % | -18.05 pp |
| DRAM Throughput (%) | 20.32 % | 4.21 % | -16.11 pp | 3.93 % | -16.39 pp | 4.20 % | -16.12 pp | 4.02 % | -16.30 pp |
| Achieved Occupancy (%) | 55.05 % | 48.48 % | -6.57 pp | 48.97 % | -6.08 pp | 48.00 % | -7.05 pp | 48.26 % | -6.79 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 26.00 register/thread | +0 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.48 % | 56.05 % | -14.43 pp | 52.01 % | -18.47 pp | 46.08 % | -24.40 pp | 52.49 % | -17.99 pp |
| IPC Active | 1.14 inst/cycle | 1.82 inst/cycle | +0.68 | 1.85 inst/cycle | +0.71 | 2.02 inst/cycle | +0.88 | 1.82 inst/cycle | +0.68 |

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     4130 ns  (1.472×)  — autotune_tile_sizes_for_occupancy_and_coalescing

Final best: 4130 ns  (1.4722×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +31,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply multiple_of/max_contiguous to 1D arange first, then reshape
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +42,13 @@

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
