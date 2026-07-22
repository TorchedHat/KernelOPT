# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 131360 ns — **Best latency**: 42820 ns (**3.0677×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`software_pipeline_num_stages_to_hide_load_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_3_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (58800,1,1), Waves/SM: 27.84
- **Registers/thread**: 27
- **Duration**: 131360 ns
- **SM Throughput**: 25.69 % | **Memory**: 84.18 % | **DRAM**: 71.45 %
- **Achieved Occupancy**: 93.11 %
- **IPC Active**: 0.85 inst/cycle | **No Eligible**: 78.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.683× | 1.000× | ✓ |  |
| B | shared_memory_tiled_transpose_for_coalesced_stores | 2.175× | 0.954× | ✓ |  |
| D | increase_yblock_for_coalesced_stores_with_autotune | 2.914× | 2.127× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_til... | Δ | increase_yblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 131360 ns | 131360 ns | +0 ns (+0.0%) | 137660 ns | +6300 ns (+4.8%) | 61760 ns | -69600 ns (-53.0%) |
| SM Throughput (%) | 25.69 % | 25.60 % | -0.09 pp | 23.99 % | -1.70 pp | 58.10 % | +32.41 pp |
| Memory Throughput (%) | 84.18 % | 84.21 % | +0.03 pp | 80.54 % | -3.64 pp | 85.97 % | +1.79 pp |
| DRAM Throughput (%) | 71.45 % | 71.47 % | +0.02 pp | 68.20 % | -3.25 pp | 13.10 % | -58.35 pp |
| Achieved Occupancy (%) | 93.11 % | 93.18 % | +0.07 pp | 93.04 % | -0.07 pp | 65.76 % | -27.35 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 27.00 register/thread | +0 | 38.00 register/thread | +11 |
| Waves/SM | 27.84 | 27.84 | +0.00 | 27.84 | +0.00 | 37.12 | +9.28 |
| No Eligible (%) | 78.79 % | 78.52 % | -0.27 pp | 81.44 % | +2.65 pp | 40.14 % | -38.65 pp |
| IPC Active | 0.85 inst/cycle | 0.85 inst/cycle | +0.00 | 0.73 inst/cycle | -0.12 | 2.38 inst/cycle | +1.53 |

---

## Iteration 2

**Best so far**: 2.1269× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_num_warps_and_num_stages_autotune | 2.906× | 2.128× | ✓ |  |
| B | autotune_large_yblock_for_coalesced_scatter_stores | 3.140× | 3.068× | ✓ | **✓ best** |
| C | expand_autotune_configs_for_coalesced_stores_via_large_yblock | 2.841× | 3.050× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | autotune_large_yb... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 131360 ns | 61730 ns | -69630 ns (-53.0%) | 42820 ns | -88540 ns (-67.4%) | 43070 ns | -88290 ns (-67.2%) |
| SM Throughput (%) | 25.69 % | 58.12 % | +32.43 pp | 54.58 % | +28.89 pp | 51.39 % | +25.70 pp |
| Memory Throughput (%) | 84.18 % | 86.00 % | +1.82 pp | 65.67 % | -18.51 pp | 64.87 % | -19.31 pp |
| DRAM Throughput (%) | 71.45 % | 13.08 % | -58.37 pp | 7.57 % | -63.88 pp | 7.49 % | -63.96 pp |
| Achieved Occupancy (%) | 93.11 % | 65.72 % | -27.39 pp | 36.29 % | -56.82 pp | 38.19 % | -54.92 pp |
| Registers/Thread | 27.00 register/thread | 38.00 register/thread | +11 | 23.00 register/thread | -4 | 23.00 register/thread | -4 |
| Waves/SM | 27.84 | 37.12 | +9.28 | 27.84 | +0.00 | 27.84 | +0.00 |
| No Eligible (%) | 78.79 % | 40.37 % | -38.42 pp | 41.81 % | -36.98 pp | 44.49 % | -34.30 pp |
| IPC Active | 0.85 inst/cycle | 2.39 inst/cycle | +1.54 | 2.35 inst/cycle | +1.50 | 2.21 inst/cycle | +1.36 |

---

## Iteration 3

**Best so far**: 3.0677× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_num_stages_to_hide_load_latency | 3.139× | 3.068× | ✓ |  |
| B | vectorized_loads_xindex_contiguity_hints | 2.842× | 3.061× | ✓ |  |
| C | transpose_arange_layout_with_vectorized_load_hints_for_coalesced_stores | 2.845× | 3.043× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | vectorized_loads_... | Δ | transpose_arange_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 131360 ns | 42820 ns | -88540 ns (-67.4%) | 42910 ns | -88450 ns (-67.3%) | 43170 ns | -88190 ns (-67.1%) |
| SM Throughput (%) | 25.69 % | 54.73 % | +29.04 pp | 51.76 % | +26.07 pp | 51.31 % | +25.62 pp |
| Memory Throughput (%) | 84.18 % | 65.81 % | -18.37 pp | 65.38 % | -18.80 pp | 64.72 % | -19.46 pp |
| DRAM Throughput (%) | 71.45 % | 7.52 % | -63.93 pp | 7.50 % | -63.95 pp | 7.47 % | -63.98 pp |
| Achieved Occupancy (%) | 93.11 % | 36.44 % | -56.67 pp | 37.91 % | -55.20 pp | 37.89 % | -55.22 pp |
| Registers/Thread | 27.00 register/thread | 23.00 register/thread | -4 | 23.00 register/thread | -4 | 23.00 register/thread | -4 |
| Waves/SM | 27.84 | 27.84 | +0.00 | 27.84 | +0.00 | 27.84 | +0.00 |
| No Eligible (%) | 78.79 % | 41.96 % | -36.83 pp | 45.06 % | -33.73 pp | 44.67 % | -34.12 pp |
| IPC Active | 0.85 inst/cycle | 2.32 inst/cycle | +1.47 | 2.20 inst/cycle | +1.35 | 2.22 inst/cycle | +1.37 |

---

## Iteration 4

**Best so far**: 3.0677× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_meta_for_autotune_adaptation | 3.140× | 3.063× | ✓ |  |
| B | fix_static_grid_to_enable_autotune_coalesced_stores | 2.860× | 3.059× | ✓ |  |
| C | streaming_load_eviction_policy_to_reduce_l1_cache_pollution | 2.815× | 3.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | streaming_load_ev... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 131360 ns | 42880 ns | -88480 ns (-67.4%) | 42940 ns | -88420 ns (-67.3%) | 43460 ns | -87900 ns (-66.9%) |
| SM Throughput (%) | 25.69 % | 54.70 % | +29.01 pp | 51.76 % | +26.07 pp | 54.93 % | +29.24 pp |
| Memory Throughput (%) | 84.18 % | 65.81 % | -18.37 pp | 65.35 % | -18.83 pp | 64.37 % | -19.81 pp |
| DRAM Throughput (%) | 71.45 % | 7.53 % | -63.92 pp | 7.53 % | -63.92 pp | 7.37 % | -64.08 pp |
| Achieved Occupancy (%) | 93.11 % | 36.24 % | -56.87 pp | 37.84 % | -55.27 pp | 36.37 % | -56.74 pp |
| Registers/Thread | 27.00 register/thread | 23.00 register/thread | -4 | 23.00 register/thread | -4 | 23.00 register/thread | -4 |
| Waves/SM | 27.84 | 27.84 | +0.00 | 27.84 | +0.00 | 27.84 | +0.00 |
| No Eligible (%) | 78.79 % | 41.84 % | -36.95 pp | 44.98 % | -33.81 pp | 40.71 % | -38.08 pp |
| IPC Active | 0.85 inst/cycle | 2.33 inst/cycle | +1.48 | 2.20 inst/cycle | +1.35 | 2.35 inst/cycle | +1.50 |

---

## Cumulative Progress

```
Baseline:  131360 ns  (1.000×)
Iter 1     61760 ns  (2.127×)  — increase_yblock_for_coalesced_stores_with_autotune
Iter 2     42820 ns  (3.068×)  — autotune_large_yblock_for_coalesced_scatter_stores

Final best: 42820 ns  (3.0677×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1200
@@ -13,7 +38,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +47,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 120*x2 + 6021120*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 1200
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_3[grid](arg_0, arg_1, 1200, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_clone_convolution_transpose_view_3[grid](arg_0, arg_1, 1200, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
