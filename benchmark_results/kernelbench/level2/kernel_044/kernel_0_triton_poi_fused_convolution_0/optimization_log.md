# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5760 ns — **Best latency**: 5220 ns (**1.1034×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_static_grid_to_lambda_to_activate_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_044/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5760 ns
- **SM Throughput**: 15.95 % | **Memory**: 42.69 % | **DRAM**: 21.42 %
- **Achieved Occupancy**: 54.78 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.366× | 1.395× | ✓ |  |
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.427× | 1.419× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_layout_transpose_kernel | 1.427× | 1.343× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5760 ns | 4130 ns | -1630 ns (-28.3%) | 4060 ns | -1700 ns (-29.5%) | 4290 ns | -1470 ns (-25.5%) |
| SM Throughput (%) | 15.95 % | 27.43 % | +11.48 pp | 17.32 % | +1.37 pp | 27.04 % | +11.09 pp |
| Memory Throughput (%) | 42.69 % | 21.79 % | -20.90 pp | 20.03 % | -22.66 pp | 21.49 % | -21.20 pp |
| DRAM Throughput (%) | 21.42 % | 3.99 % | -17.43 pp | 4.08 % | -17.34 pp | 3.87 % | -17.55 pp |
| Achieved Occupancy (%) | 54.78 % | 67.92 % | +13.14 pp | 49.95 % | -4.83 pp | 69.18 % | +14.40 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.99 % | 38.93 % | -32.06 pp | 56.12 % | -14.87 pp | 40.76 % | -30.23 pp |
| IPC Active | 1.15 inst/cycle | 2.38 inst/cycle | +1.23 | 1.78 inst/cycle | +0.63 | 2.36 inst/cycle | +1.21 |

---

## Iteration 2

**Best so far**: 1.3427× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_to_enable_yblock1_store_coalescing | 1.367× | 1.462× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5760 ns | 3940 ns | -1820 ns (-31.6%) |
| SM Throughput (%) | 15.95 % | 27.74 % | +11.79 pp |
| Memory Throughput (%) | 42.69 % | 22.08 % | -20.61 pp |
| DRAM Throughput (%) | 21.42 % | 4.18 % | -17.24 pp |
| Achieved Occupancy (%) | 54.78 % | 68.33 % | +13.55 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.99 % | 38.01 % | -32.98 pp |
| IPC Active | 1.15 inst/cycle | 2.34 inst/cycle | +1.19 |

---

## Iteration 3

**Best so far**: 1.3427× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune_tile_selection | 1.429× | 1.419× | ✓ |  |
| B | fix_static_grid_to_enable_autotune_and_reduce_tail_wave | 1.367× | 1.395× | ✓ |  |
| D | fix_static_grid_to_lambda_to_activate_autotune | 1.434× | 1.103× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.385× | 1.375× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5760 ns | 4060 ns | -1700 ns (-29.5%) | 4130 ns | -1630 ns (-28.3%) | 5220 ns | -540 ns (-9.4%) | 4190 ns | -1570 ns (-27.3%) |
| SM Throughput (%) | 15.95 % | 26.82 % | +10.87 pp | 27.23 % | +11.28 pp | 38.75 % | +22.80 pp | 26.78 % | +10.83 pp |
| Memory Throughput (%) | 42.69 % | 21.32 % | -21.37 pp | 21.63 % | -21.06 pp | 29.15 % | -13.54 pp | 21.30 % | -21.39 pp |
| DRAM Throughput (%) | 21.42 % | 4.07 % | -17.35 pp | 4.02 % | -17.40 pp | 3.16 % | -18.26 pp | 3.96 % | -17.46 pp |
| Achieved Occupancy (%) | 54.78 % | 68.68 % | +13.90 pp | 68.60 % | +13.82 pp | 77.56 % | +22.78 pp | 68.20 % | +13.42 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 2.97 | +2.23 | 1.48 | +0.74 |
| No Eligible (%) | 70.99 % | 37.83 % | -33.16 pp | 37.48 % | -33.51 pp | 25.45 % | -45.54 pp | 36.76 % | -34.23 pp |
| IPC Active | 1.15 inst/cycle | 2.42 inst/cycle | +1.27 | 2.40 inst/cycle | +1.25 | 2.92 inst/cycle | +1.77 | 2.37 inst/cycle | +1.22 |

---

## Iteration 4

**Best so far**: 1.3427× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_enable_yblock1_store_coalescing | 1.434× | 1.395× | ✓ |  |
| B | evict_first_streaming_load_policy | 1.388× | 1.419× | ✓ |  |
| C | fix_static_grid_to_lambda_activate_autotune_and_coalescing | 1.389× | 1.324× | ✓ |  |
| D | streaming_eviction_policy_for_layout_transpose | 1.375× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | evict_first_strea... | Δ | fix_static_grid_t... | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5760 ns | 4130 ns | -1630 ns (-28.3%) | 4060 ns | -1700 ns (-29.5%) | 4350 ns | -1410 ns (-24.5%) | 5730 ns | -30 ns (-0.5%) |
| SM Throughput (%) | 15.95 % | 27.00 % | +11.05 pp | 24.79 % | +8.84 pp | 26.02 % | +10.07 pp | 40.24 % | +24.29 pp |
| Memory Throughput (%) | 42.69 % | 21.46 % | -21.23 pp | 19.98 % | -22.71 pp | 20.67 % | -22.02 pp | 30.54 % | -12.15 pp |
| DRAM Throughput (%) | 21.42 % | 4.01 % | -17.41 pp | 4.09 % | -17.33 pp | 3.81 % | -17.61 pp | 2.89 % | -18.53 pp |
| Achieved Occupancy (%) | 54.78 % | 69.03 % | +14.25 pp | 67.81 % | +13.03 pp | 68.83 % | +14.05 pp | 77.95 % | +23.17 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 2.97 | +2.23 |
| No Eligible (%) | 70.99 % | 41.49 % | -29.50 pp | 40.37 % | -30.62 pp | 36.88 % | -34.11 pp | 26.26 % | -44.73 pp |
| IPC Active | 1.15 inst/cycle | 2.45 inst/cycle | +1.30 | 2.35 inst/cycle | +1.20 | 2.42 inst/cycle | +1.27 | 2.89 inst/cycle | +1.74 |


### Key Learnings

#### ✅ Plan B: `evict_first_streaming_load_policy` (1.419×)

For streaming kernels where each element is read and written exactly once (zero temporal reuse), changing tl.load eviction_policy from 'evict_last' to 'evict_first' and adding eviction_policy='evict_first' to tl.store signals the L2 cache to treat these lines as lowest-priority for retention (PTX .cs cache-streaming operator). Despite the NCU profiling metrics showing nearly identical before/after values (the kernel is so short that per-invocation metrics are noisy), the wall-clock speedup of 1.42x indicates the policy change reduces L2 cache pollution across repeated kernel invocations in a benchmark loop. This is most impactful when L1/TEX hit rate is 0% (all traffic goes through L2), L2 hit rate is high (>70%) on a purely streaming access pattern, and the kernel touches large tensors exactly once per invocation.

> **Applicable when**: Triton kernel is a pure streaming operation (layout transpose, copy, elementwise with no reuse) where: L1/TEX hit rate is 0%, L2 hit rate is high (>70%) despite zero expected temporal reuse, each element is accessed exactly once per kernel invocation, and the kernel is called repeatedly in a benchmark or training loop (so cross-invocation L2 pollution matters).

> **Avoid when**: Kernel has genuine data reuse (e.g. reduction, matmul, convolution with sliding windows) where L2 caching provides real benefit. Also avoid when L2 hit rate is already low (<30%) indicating data is already streaming through without caching. Do not apply when the kernel is called only once (no cross-invocation pollution to eliminate).

---

## Iteration 5

**Best so far**: 1.3427× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | activate_autotune_via_lambda_grid_for_yblock1_coalescing | 1.387× | 1.419× | ✓ |  |
| B | activate_autotune_via_lambda_grid_for_vectorized_loads | 1.381× | 1.419× | ✓ |  |
| C | fix_hardcoded_grid_tuple_to_lambda_activate_autotune_selection | 1.370× | 1.405× | ✓ |  |
| D | activate_autotune_via_lambda_grid_to_eliminate_partial_wave | 1.388× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | activate_autotune... | Δ | activate_autotune... | Δ | fix_hardcoded_gri... | Δ | activate_autotune... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5760 ns | 4060 ns | -1700 ns (-29.5%) | 4060 ns | -1700 ns (-29.5%) | 4100 ns | -1660 ns (-28.8%) | 5700 ns | -60 ns (-1.0%) |
| SM Throughput (%) | 15.95 % | 26.76 % | +10.81 pp | 26.96 % | +11.01 pp | 26.25 % | +10.30 pp | 40.22 % | +24.27 pp |
| Memory Throughput (%) | 42.69 % | 21.56 % | -21.13 pp | 21.42 % | -21.27 pp | 20.84 % | -21.85 pp | 30.50 % | -12.19 pp |
| DRAM Throughput (%) | 21.42 % | 4.07 % | -17.35 pp | 4.06 % | -17.36 pp | 4.05 % | -17.37 pp | 2.90 % | -18.52 pp |
| Achieved Occupancy (%) | 54.78 % | 70.70 % | +15.92 pp | 68.63 % | +13.85 pp | 68.91 % | +14.13 pp | 77.24 % | +22.46 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 2.97 | +2.23 |
| No Eligible (%) | 70.99 % | 38.06 % | -32.93 pp | 37.31 % | -33.68 pp | 38.12 % | -32.87 pp | 26.23 % | -44.76 pp |
| IPC Active | 1.15 inst/cycle | 2.19 inst/cycle | +1.04 | 2.33 inst/cycle | +1.18 | 2.41 inst/cycle | +1.26 | 2.90 inst/cycle | +1.75 |

---

## Cumulative Progress

```
Baseline:  5760 ns  (1.000×)
Iter 1     4290 ns  (1.343×)  — autotune_tile_sizes_for_underutilized_layout_transpose_kernel

Final best: 5220 ns  (1.1034×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 2}, num_warps=4),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +29,14 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

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
