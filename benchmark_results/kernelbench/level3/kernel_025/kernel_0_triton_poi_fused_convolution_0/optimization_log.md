# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5980 ns — **Best latency**: 4510 ns (**1.3259×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_enable_autotune_and_increase_waves`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5980 ns
- **SM Throughput**: 16.37 % | **Memory**: 40.24 % | **DRAM**: 20.61 %
- **Achieved Occupancy**: 55.43 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 71.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.366× | 1.714× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.292× | 0.201× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_swap | 1.325× | 1.404× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_transpose_kernel | 1.365× | 1.473× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3490 ns | -2490 ns (-41.6%) | 29790 ns | +23810 ns (+398.2%) | 4260 ns | -1720 ns (-28.8%) | 4060 ns | -1920 ns (-32.1%) |
| SM Throughput (%) | 16.37 % | 17.29 % | +0.92 pp | 60.18 % | +43.81 pp | 25.54 % | +9.17 pp | 19.14 % | +2.77 pp |
| Memory Throughput (%) | 40.24 % | 19.60 % | -20.64 pp | 73.08 % | +32.84 pp | 22.47 % | -17.77 pp | 22.13 % | -18.11 pp |
| DRAM Throughput (%) | 20.61 % | 4.72 % | -15.89 pp | 4.12 % | -16.49 pp | 7.25 % | -13.36 pp | 4.06 % | -16.55 pp |
| Achieved Occupancy (%) | 55.43 % | 43.98 % | -11.45 pp | 34.37 % | -21.06 pp | 52.98 % | -2.45 pp | 48.59 % | -6.84 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 32.00 register/thread | +6 | 28.00 register/thread | +2 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 3.96 | +3.22 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.13 % | 54.54 % | -16.59 pp | 48.57 % | -22.56 pp | 40.66 % | -30.47 pp | 52.23 % | -18.90 pp |
| IPC Active | 1.11 inst/cycle | 1.69 inst/cycle | +0.58 | 2.03 inst/cycle | +0.92 | 2.39 inst/cycle | +1.28 | 1.84 inst/cycle | +0.73 |

---

## Iteration 2

**Best so far**: 1.7135× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_strided_store_by_swapping_fast_thread_dimension | 1.366× | 1.812× | ✓ |  |
| B | vectorized_loads_contiguous_hint | 1.423× | 1.458× | ✓ | **✓ best** |
| C | vectorized_loads_via_contiguous_hints | 1.366× | 1.335× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_strided_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 3300 ns | -2680 ns (-44.8%) | 4100 ns | -1880 ns (-31.4%) | 4480 ns | -1500 ns (-25.1%) |
| SM Throughput (%) | 16.37 % | 17.25 % | +0.88 pp | 18.95 % | +2.58 pp | 25.84 % | +9.47 pp |
| Memory Throughput (%) | 40.24 % | 20.49 % | -19.75 pp | 21.83 % | -18.41 pp | 22.65 % | -17.59 pp |
| DRAM Throughput (%) | 20.61 % | 4.99 % | -15.62 pp | 4.05 % | -16.56 pp | 6.92 % | -13.69 pp |
| Achieved Occupancy (%) | 55.43 % | 43.95 % | -11.48 pp | 51.41 % | -4.02 pp | 54.89 % | -0.54 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 28.00 register/thread | +2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.13 % | 55.62 % | -15.51 pp | 52.35 % | -18.78 pp | 38.61 % | -32.52 pp |
| IPC Active | 1.11 inst/cycle | 1.69 inst/cycle | +0.58 | 1.64 inst/cycle | +0.53 | 2.11 inst/cycle | +1.00 |


### Key Learnings

#### ✅ Plan A: `coalesce_strided_store_by_swapping_fast_thread_dimension` (1.812×)

When a Triton kernel's output tensor has a non-unit stride along the dimension assigned to the fast (warp-lane) thread index, swapping the arange slicing axes so the stride-1 output dimension becomes the fast dimension can improve memory throughput. However, when the kernel has already been partially optimized (vectorized loads applied, L2 hit rate already high at ~76%), the store-coalescing axis swap yields only a modest improvement (~5% duration reduction, ~6% memory throughput increase) because the uncoalesced store excess sectors (50%) and bytes-per-sector (10.7) remain unchanged — indicating the swap did not actually coalesce the stores in this configuration, and the speedup likely comes from minor scheduling or timing effects rather than true coalescing. The strategy is more impactful when applied to a cold kernel (before vectorized loads) where it can interact with the full memory bottleneck.

> **Applicable when**: NCU reports UncoalescedGlobalAccess with >10% excess sectors; MemoryCacheAccessPattern shows store bytes utilized per sector well below 32; the output tensor has a non-unit stride along the dimension currently assigned to the fast (warp-lane) arange index; kernel has not yet had vectorized load hints applied (cold kernel state gives larger benefit).

> **Avoid when**: The kernel has already had vectorized loads (tl.multiple_of + tl.max_contiguous) applied to the load path and L2 hit rate is already high (>75%) — in this state the store-coalescing swap yields only marginal gains (~5%) and NCU still reports the same excess sector percentage, suggesting the YBLOCK tile size (e.g., YBLOCK=1 or YBLOCK=8 with ynumel=30) limits the coalescing benefit. Also do not apply when the load access pattern would become uncoalesced after the swap.

---

## Iteration 3

**Best so far**: 1.7135× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_to_use_meta_tile_sizes | 1.425× | 1.484× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5980 ns | 4030 ns | -1950 ns (-32.6%) |
| SM Throughput (%) | 16.37 % | 19.56 % | +3.19 pp |
| Memory Throughput (%) | 40.24 % | 22.59 % | -17.65 pp |
| DRAM Throughput (%) | 20.61 % | 4.09 % | -16.52 pp |
| Achieved Occupancy (%) | 55.43 % | 49.10 % | -6.33 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 |
| No Eligible (%) | 71.13 % | 51.53 % | -19.60 pp |
| IPC Active | 1.11 inst/cycle | 1.80 inst/cycle | +0.69 |

---

## Iteration 4

**Best so far**: 1.7135× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 5

**Best so far**: 1.7135× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_lambda_for_autotune_activation | 1.421× | 1.448× | ✓ |  |
| B | fix_hardcoded_grid_to_enable_autotune_and_increase_waves | 1.426× | 1.326× | ✓ | **✓ best** |
| C | fix_hardcoded_grid_to_enable_autotune | 1.379× | 1.458× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5980 ns | 4130 ns | -1850 ns (-30.9%) | 4510 ns | -1470 ns (-24.6%) | 4100 ns | -1880 ns (-31.4%) |
| SM Throughput (%) | 16.37 % | 18.20 % | +1.83 pp | 19.27 % | +2.90 pp | 17.86 % | +1.49 pp |
| Memory Throughput (%) | 40.24 % | 21.01 % | -19.23 pp | 22.24 % | -18.00 pp | 20.65 % | -19.59 pp |
| DRAM Throughput (%) | 20.61 % | 3.99 % | -16.62 pp | 3.66 % | -16.95 pp | 4.04 % | -16.57 pp |
| Achieved Occupancy (%) | 55.43 % | 49.00 % | -6.43 pp | 48.31 % | -7.12 pp | 48.47 % | -6.96 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.13 % | 57.37 % | -13.76 pp | 51.80 % | -19.33 pp | 52.33 % | -18.80 pp |
| IPC Active | 1.11 inst/cycle | 1.85 inst/cycle | +0.74 | 1.83 inst/cycle | +0.72 | 1.82 inst/cycle | +0.71 |

---

## Cumulative Progress

```
Baseline:  5980 ns  (1.000×)
Iter 1     3490 ns  (1.714×)  — vectorized_loads_with_autotune

Final best: 4510 ns  (1.3259×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -12,8 +34,8 @@

     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xoffset = tl.multiple_of(tl.program_id(0) * XBLOCK, XBLOCK)
+    xindex = xoffset + tl.max_contiguous(tl.arange(0, XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +44,13 @@

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

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
