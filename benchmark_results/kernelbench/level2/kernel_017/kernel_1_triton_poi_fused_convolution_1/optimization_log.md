# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286460 ns — **Best latency**: 144160 ns (**1.9871×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_static_grid_to_lambda_enable_autotune_tile_exploration`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_017/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286460 ns
- **SM Throughput**: 19.80 % | **Memory**: 78.63 % | **DRAM**: 72.02 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.78 inst/cycle | **No Eligible**: 80.42 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | shared_memory_transpose_with_autotune | 2.379× | 1.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_tra... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286460 ns | 143810 ns | -142650 ns (-49.8%) |
| SM Throughput (%) | 19.80 % | 46.59 % | +26.79 pp |
| Memory Throughput (%) | 78.63 % | 40.56 % | -38.07 pp |
| DRAM Throughput (%) | 72.02 % | 34.22 % | -37.80 pp |
| Achieved Occupancy (%) | 94.05 % | 89.95 % | -4.10 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.42 % | 56.00 % | -24.42 pp |
| IPC Active | 0.78 inst/cycle | 1.71 inst/cycle | +0.93 |

---

## Iteration 2

**Best so far**: 1.9919× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_autotune_grid_to_use_meta_block_sizes | 2.375× | 2.000× | ✓ |  |
| D | fix_autotune_grid_lambda_and_expand_tile_configs | 2.383× | 2.016× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 143260 ns | -143200 ns (-50.0%) | 142110 ns | -144350 ns (-50.4%) |
| SM Throughput (%) | 19.80 % | 47.51 % | +27.71 pp | 47.80 % | +28.00 pp |
| Memory Throughput (%) | 78.63 % | 41.41 % | -37.22 pp | 41.73 % | -36.90 pp |
| DRAM Throughput (%) | 72.02 % | 34.41 % | -37.61 pp | 34.61 % | -37.41 pp |
| Achieved Occupancy (%) | 94.05 % | 88.94 % | -5.11 pp | 89.98 % | -4.07 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.42 % | 57.41 % | -23.01 pp | 57.06 % | -23.36 pp |
| IPC Active | 0.78 inst/cycle | 1.72 inst/cycle | +0.94 | 1.74 inst/cycle | +0.96 |

---

## Iteration 3

**Best so far**: 2.0158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_for_autotune_activation | 2.386× | 2.014× | ✓ |  |
| C | vectorized_loads_fixed_autotune_grid | 3.151× | 2.006× | ✓ | **✓ best** |
| D | fix_static_grid_to_lambda_to_enable_autotune | 2.380× | 2.012× | ✓ |  |
| E | add_triton_autotune | 2.384× | 1.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | vectorized_loads_... | Δ | fix_static_grid_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 142210 ns | -144250 ns (-50.4%) | 142820 ns | -143640 ns (-50.1%) | 142370 ns | -144090 ns (-50.3%) | 143550 ns | -142910 ns (-49.9%) |
| SM Throughput (%) | 19.80 % | 47.93 % | +28.13 pp | 47.97 % | +28.17 pp | 47.35 % | +27.55 pp | 47.31 % | +27.51 pp |
| Memory Throughput (%) | 78.63 % | 41.80 % | -36.83 pp | 41.83 % | -36.80 pp | 41.26 % | -37.37 pp | 41.25 % | -37.38 pp |
| DRAM Throughput (%) | 72.02 % | 34.60 % | -37.42 pp | 34.52 % | -37.50 pp | 34.59 % | -37.43 pp | 34.29 % | -37.73 pp |
| Achieved Occupancy (%) | 94.05 % | 89.67 % | -4.38 pp | 89.30 % | -4.75 pp | 87.39 % | -6.66 pp | 86.86 % | -7.19 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.42 % | 56.52 % | -23.90 pp | 56.25 % | -24.17 pp | 56.68 % | -23.74 pp | 56.99 % | -23.43 pp |
| IPC Active | 0.78 inst/cycle | 1.72 inst/cycle | +0.94 | 1.71 inst/cycle | +0.93 | 1.71 inst/cycle | +0.93 | 1.71 inst/cycle | +0.93 |

---

## Iteration 4

**Best so far**: 2.0158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_static_grid_to_lambda_enable_autotune_tile_exploration | 3.155× | 2.016× | ✓ |  |
| C | fix_static_grid_to_lambda_enable_autotune_tile_exploration | 3.156× | 1.987× | ✓ | **✓ best** |
| D | fix_static_grid_to_lambda_enable_autotune_tile_exploration | 2.384× | 2.001× | ✓ |  |
| E | add_triton_autotune | 2.380× | 2.014× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 142110 ns | -144350 ns (-50.4%) | 144160 ns | -142300 ns (-49.7%) | 143170 ns | -143290 ns (-50.0%) | 142270 ns | -144190 ns (-50.3%) |
| SM Throughput (%) | 19.80 % | 46.77 % | +26.97 pp | 47.09 % | +27.29 pp | 47.47 % | +27.67 pp | 46.83 % | +27.03 pp |
| Memory Throughput (%) | 78.63 % | 40.81 % | -37.82 pp | 41.05 % | -37.58 pp | 41.49 % | -37.14 pp | 40.80 % | -37.83 pp |
| DRAM Throughput (%) | 72.02 % | 34.63 % | -37.39 pp | 34.10 % | -37.92 pp | 34.42 % | -37.60 pp | 34.53 % | -37.49 pp |
| Achieved Occupancy (%) | 94.05 % | 89.48 % | -4.57 pp | 89.12 % | -4.93 pp | 89.56 % | -4.49 pp | 86.25 % | -7.80 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.42 % | 56.51 % | -23.91 pp | 57.03 % | -23.39 pp | 56.49 % | -23.93 pp | 57.06 % | -23.36 pp |
| IPC Active | 0.78 inst/cycle | 1.77 inst/cycle | +0.99 | 1.72 inst/cycle | +0.94 | 1.72 inst/cycle | +0.94 | 1.72 inst/cycle | +0.94 |

---

## Iteration 5

**Best so far**: 2.0158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_static_grid_to_lambda_enable_autotune_coalescing | 2.387× | 1.976× | ✓ |  |
| D | coalesced_load_via_lambda_grid_enabling_autotune_coalescing_configs | 3.154× | 1.991× | ✓ |  |
| E | add_triton_autotune | 0.984× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | coalesced_load_vi... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286460 ns | 144990 ns | -141470 ns (-49.4%) | 143900 ns | -142560 ns (-49.8%) | 289730 ns | +3270 ns (+1.1%) |
| SM Throughput (%) | 19.80 % | 47.58 % | +27.78 pp | 47.33 % | +27.53 pp | 18.30 % | -1.50 pp |
| Memory Throughput (%) | 78.63 % | 41.57 % | -37.06 pp | 41.34 % | -37.29 pp | 77.94 % | -0.69 pp |
| DRAM Throughput (%) | 72.02 % | 34.00 % | -38.02 pp | 34.26 % | -37.76 pp | 71.22 % | -0.80 pp |
| Achieved Occupancy (%) | 94.05 % | 89.00 % | -5.05 pp | 89.15 % | -4.90 pp | 93.91 % | -0.14 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.42 % | 56.51 % | -23.91 pp | 57.33 % | -23.09 pp | 81.57 % | +1.15 pp |
| IPC Active | 0.78 inst/cycle | 1.70 inst/cycle | +0.92 | 1.71 inst/cycle | +0.93 | 0.74 inst/cycle | -0.04 |

---

## Cumulative Progress

```
Baseline:  286460 ns  (1.000×)
Iter 1     143810 ns  (1.992×)  — shared_memory_transpose_with_autotune
Iter 2     142110 ns  (2.016×)  — fix_autotune_grid_lambda_and_expand_tile_configs

Final best: 144160 ns  (1.9871×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
+def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
     ynumel = 4096
     xnumel = 31500
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
+    y0 = yindex % 128
     y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (y0 + 128*xindex + 4032000*y1), xmask & ymask, other=0.0)
+    tl.store(out_ptr0 + (xindex + 31500*yindex), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
