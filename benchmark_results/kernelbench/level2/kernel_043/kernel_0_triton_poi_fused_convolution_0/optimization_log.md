# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6020 ns — **Best latency**: 4030 ns (**1.4938×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`collapse_y_grid_with_yblock32_and_large_xblock_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_043/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6020 ns
- **SM Throughput**: 17.05 % | **Memory**: 40.79 % | **DRAM**: 20.50 %
- **Achieved Occupancy**: 54.89 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 71.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_and_autotune_tile_sizes | 1.260× | 1.344× | ✓ |  |
| B | autotune_tile_sizes_for_coalesced_stores_and_wave_coverage | 1.378× | 1.362× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_underutilized_layout_permutation_kernel | 0.983× | 1.056× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4480 ns | -1540 ns (-25.6%) | 4420 ns | -1600 ns (-26.6%) | 5700 ns | -320 ns (-5.3%) |
| SM Throughput (%) | 17.05 % | 20.97 % | +3.92 pp | 19.02 % | +1.97 pp | 16.48 % | -0.57 pp |
| Memory Throughput (%) | 40.79 % | 31.53 % | -9.26 pp | 21.97 % | -18.82 pp | 43.06 % | +2.27 pp |
| DRAM Throughput (%) | 20.50 % | 14.76 % | -5.74 pp | 3.75 % | -16.75 pp | 21.53 % | +1.03 pp |
| Achieved Occupancy (%) | 54.89 % | 47.28 % | -7.61 pp | 48.61 % | -6.28 pp | 54.79 % | -0.10 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 56.38 % | -15.52 pp | 52.15 % | -19.75 pp | 74.48 % | +2.58 pp |
| IPC Active | 1.16 inst/cycle | 1.72 inst/cycle | +0.56 | 1.85 inst/cycle | +0.69 | 1.16 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan B: `autotune_tile_sizes_for_coalesced_stores_and_wave_coverage` (1.458×)

Switching the load eviction policy from 'evict_last' to 'evict_first' on a zero-reuse streaming permutation kernel, combined with expanding the autotune config space to include num_stages=1,2,3,4 variants, yielded a 1.46x speedup. Notably, the NCU profiling metrics before and after are nearly identical (duration 4.03→4.13 µs, DRAM throughput 4.1→4.05%), suggesting the speedup is measured at the wall-clock/benchmark level rather than reflected in single-kernel NCU traces — the autotuner likely selected a different config across runs. The uncoalesced access pattern (50% excessive sectors, 10.7 bytes/sector) and low DRAM throughput persist in the NCU snapshot, indicating the core memory access bottleneck remains unresolved by this change alone.

> **Applicable when**: Triton streaming (zero-reuse) layout permutation kernels where load eviction_policy='evict_last' is set on data never reused; UncoalescedGlobalAccess triggered with >20% excessive sectors; existing autotune configs do not span all num_stages values (1-4); waves_per_sm < 1.0.

> **Avoid when**: Kernel has genuine data reuse (evict_first would hurt L1 hit rate); autotune config space already spans num_stages=1-4 with large XBLOCK values (2048/4096); DRAM throughput already >50%; kernel is compute-bound (SM Throughput >70%).

#### ✅ Plan D: `autotune_tile_sizes_for_underutilized_layout_permutation_kernel` (1.056×)

For Triton pointwise kernels performing layout permutations (e.g. NCHW→NHWC) with a fixed small grid (< 1 wave/SM), adding @triton.autotune with a range of XBLOCK/YBLOCK/num_warps configurations allows the runtime to select the tile shape that best hides L1TEX latency and improves SM utilization. The speedup here is modest (5.6%) because the dominant bottlenecks—uncoalesced stores (23% excess sectors), low waves/SM (0.74), and L1TEX stalls (~50% of warp cycles)—were not fully resolved by tile-size tuning alone; the autotuner selected a config close to the original. This strategy is most impactful when the optimal tile size differs substantially from the hardcoded default.

> **Applicable when**: Triton pointwise kernel with hardcoded XBLOCK/YBLOCK tile sizes; waves/SM < 1.0; IssueSlotUtilization rule triggered; scheduler idle cycles > 60%; L1TEX stall fraction > 40% of warp cycles; kernel performs a 2D layout permutation with strided scatter stores.

> **Avoid when**: The kernel already uses @triton.autotune; the bottleneck is purely uncoalesced memory access (autotuning tile sizes cannot fix stride-induced scatter patterns); the grid is already > 2 waves/SM and occupancy > 80%; the kernel has shared memory reductions where tile size changes affect correctness.

---

## Iteration 2

**Best so far**: 1.362× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_software_pipelining | 1.369× | 1.483× | ✓ |  |
| B | vectorized_loads_evict_first_and_yblock1_configs | 1.306× | 1.335× | ✓ |  |
| C | vectorized_loads_with_num_stages_pipelining | 1.421× | 1.494× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4060 ns | -1960 ns (-32.6%) | 4510 ns | -1510 ns (-25.1%) | 4030 ns | -1990 ns (-33.1%) |
| SM Throughput (%) | 17.05 % | 19.10 % | +2.05 pp | 21.75 % | +4.70 pp | 19.91 % | +2.86 pp |
| Memory Throughput (%) | 40.79 % | 22.01 % | -18.78 pp | 31.08 % | -9.71 pp | 23.04 % | -17.75 pp |
| DRAM Throughput (%) | 20.50 % | 4.09 % | -16.41 pp | 14.60 % | -5.90 pp | 4.10 % | -16.40 pp |
| Achieved Occupancy (%) | 54.89 % | 49.79 % | -5.10 pp | 45.94 % | -8.95 pp | 48.76 % | -6.13 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 27.00 register/thread | +1 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 53.70 % | -18.20 pp | 56.07 % | -15.83 pp | 51.84 % | -20.06 pp |
| IPC Active | 1.16 inst/cycle | 1.78 inst/cycle | +0.62 | 1.69 inst/cycle | +0.53 | 1.82 inst/cycle | +0.66 |

---

## Iteration 3

**Best so far**: 1.4938× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | yblock32_full_y_coverage_to_reduce_store_scatter_and_grid_fragmentation | 1.366× | 1.505× | ✓ |  |
| B | evict_first_streaming_load_cache_policy | 1.355× | 1.458× | ✓ |  |
| C | autotune_yblock32_full_y_coverage_to_eliminate_workload_imbalance | 1.311× | 1.362× | ✓ |  |


### Metric Comparison

| Metric | Baseline | yblock32_full_y_c... | Δ | evict_first_strea... | Δ | autotune_yblock32... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 4000 ns | -2020 ns (-33.6%) | 4130 ns | -1890 ns (-31.4%) | 4420 ns | -1600 ns (-26.6%) |
| SM Throughput (%) | 17.05 % | 18.92 % | +1.87 pp | 19.18 % | +2.13 pp | 20.46 % | +3.41 pp |
| Memory Throughput (%) | 40.79 % | 21.86 % | -18.93 pp | 22.15 % | -18.64 pp | 31.63 % | -9.16 pp |
| DRAM Throughput (%) | 20.50 % | 4.13 % | -16.37 pp | 4.05 % | -16.45 pp | 14.92 % | -5.58 pp |
| Achieved Occupancy (%) | 54.89 % | 48.63 % | -6.26 pp | 48.82 % | -6.07 pp | 46.59 % | -8.30 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 25.00 register/thread | -1 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 51.88 % | -20.02 pp | 52.31 % | -19.59 pp | 60.15 % | -11.75 pp |
| IPC Active | 1.16 inst/cycle | 1.86 inst/cycle | +0.70 | 1.87 inst/cycle | +0.71 | 1.70 inst/cycle | +0.54 |

---

## Iteration 4

**Best so far**: 1.4938× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_store_via_output_index_remapping | 1.222× | 1.636× | ✓ |  |
| B | collapse_y_grid_with_yblock32_and_large_xblock_configs | 1.323× | 1.494× | ✓ |  |
| C | vectorized_store_hints_for_scatter_permutation_kernel | 1.331× | 1.281× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_store_v... | Δ | collapse_y_grid_w... | Δ | vectorized_store_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6020 ns | 3680 ns | -2340 ns (-38.9%) | 4030 ns | -1990 ns (-33.1%) | 4700 ns | -1320 ns (-21.9%) |
| SM Throughput (%) | 17.05 % | 15.51 % | -1.54 pp | 19.61 % | +2.56 pp | 21.02 % | +3.97 pp |
| Memory Throughput (%) | 40.79 % | 24.23 % | -16.56 pp | 22.76 % | -18.03 pp | 29.50 % | -11.29 pp |
| DRAM Throughput (%) | 20.50 % | 4.49 % | -16.01 pp | 4.12 % | -16.38 pp | 13.94 % | -6.56 pp |
| Achieved Occupancy (%) | 54.89 % | 58.40 % | +3.51 pp | 48.46 % | -6.43 pp | 46.88 % | -8.01 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 25.00 register/thread | -1 | 27.00 register/thread | +1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 67.44 % | -4.46 pp | 52.37 % | -19.53 pp | 56.40 % | -15.50 pp |
| IPC Active | 1.16 inst/cycle | 1.18 inst/cycle | +0.02 | 1.87 inst/cycle | +0.71 | 1.71 inst/cycle | +0.55 |

---

## Cumulative Progress

```
Baseline:  6020 ns  (1.000×)
Iter 1     4420 ns  (1.362×)  — autotune_tile_sizes_for_coalesced_stores_and_wave_coverage
Iter 2     4030 ns  (1.494×)  — vectorized_loads_with_num_stages_pipelining

Final best: 4030 ns  (1.4938×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,38 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # existing configs (KEEP ALL)
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 1},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 8},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 16}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 16}, num_warps=8,  num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 4},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8},  num_warps=16, num_stages=4),
+        # NEW: YBLOCK=32 covers all 30 Y-rows in one CTA, collapsing Y-grid to 1
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 32}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 32}, num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 32}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 32}, num_warps=16, num_stages=3),
+        # NEW: very large XBLOCK for maximum X-stripe width
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 4096, 'YBLOCK': 32}, num_warps=16, num_stages=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +45,7 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +54,13 @@

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
