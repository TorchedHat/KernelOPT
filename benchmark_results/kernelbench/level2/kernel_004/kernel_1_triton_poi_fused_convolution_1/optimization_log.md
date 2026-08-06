# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 286270 ns — **Best latency**: 183650 ns (**1.5588×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_meta_lambda_to_unlock_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_004/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 286270 ns
- **SM Throughput**: 19.35 % | **Memory**: 78.80 % | **DRAM**: 72.09 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.782× | 1.579× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 286270 ns | 181340 ns | -104930 ns (-36.7%) |
| SM Throughput (%) | 19.35 % | 29.35 % | +10.00 pp |
| Memory Throughput (%) | 78.80 % | 90.98 % | +12.18 pp |
| DRAM Throughput (%) | 72.09 % | 56.08 % | -16.01 pp |
| Achieved Occupancy (%) | 94.05 % | 92.79 % | -1.26 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 70.34 % | -10.07 pp |
| IPC Active | 0.79 inst/cycle | 1.18 inst/cycle | +0.39 |

---

## Iteration 2

**Best so far**: 1.5786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_y_dimension_loads_with_yblock_hints | 1.825× | 1.577× | ✓ |  |
| B | coalesced_store_hints_and_autotune_yblock_expansion | 3.057× | 1.565× | ✓ | **✓ best** |
| C | memory_coalescing_shared_memory_tiled_transpose | 1.943× | 2.040× | ✓ |  |
| D | fix_autotune_grid_lambda_and_add_software_pipelining | 1.778× | 1.554× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_y_dimen... | Δ | coalesced_store_h... | Δ | memory_coalescing... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286270 ns | 181470 ns | -104800 ns (-36.6%) | 182980 ns | -103290 ns (-36.1%) | 140350 ns | -145920 ns (-51.0%) | 184260 ns | -102010 ns (-35.6%) |
| SM Throughput (%) | 19.35 % | 29.57 % | +10.22 pp | 30.80 % | +11.45 pp | 47.66 % | +28.31 pp | 29.19 % | +9.84 pp |
| Memory Throughput (%) | 78.80 % | 91.04 % | +12.24 pp | 90.48 % | +11.68 pp | 41.62 % | -37.18 pp | 89.79 % | +10.99 pp |
| DRAM Throughput (%) | 72.09 % | 56.05 % | -16.04 pp | 55.29 % | -16.80 pp | 34.97 % | -37.12 pp | 55.20 % | -16.89 pp |
| Achieved Occupancy (%) | 94.05 % | 92.79 % | -1.26 pp | 92.75 % | -1.30 pp | 89.57 % | -4.48 pp | 92.75 % | -1.30 pp |
| Registers/Thread | 25.00 register/thread | 23.00 register/thread | -2 | 23.00 register/thread | -2 | 25.00 register/thread | +0 | 23.00 register/thread | -2 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 70.08 % | -10.33 pp | 68.81 % | -11.60 pp | 56.84 % | -23.57 pp | 70.37 % | -10.04 pp |
| IPC Active | 0.79 inst/cycle | 1.19 inst/cycle | +0.40 | 1.27 inst/cycle | +0.48 | 1.76 inst/cycle | +0.97 | 1.20 inst/cycle | +0.41 |


### Key Learnings

#### ✅ Plan C: `memory_coalescing_shared_memory_tiled_transpose` (2.040×)

When a tensor transpose/permutation kernel has a stride-N load pattern (consecutive threads access addresses N elements apart), restructuring the 2D tile so that the stride-1 input dimension is the fast-varying (warp) axis for loads — and the stride-1 output dimension is the fast-varying axis for stores — eliminates uncoalesced access without shared memory or register transposes. In this case, excessive sectors dropped from 34% (8.3M) to 6% (505K), L2 throughput fell from 95% to 47%, and warp cycles per issued instruction dropped from 48.9 to 33.3, yielding a 2.04x speedup. The key insight is that the same 2D tile [YBLOCK, XBLOCK] can serve both coalesced loads (stride-1 dim along YBLOCK axis) and coalesced stores (stride-1 dim along XBLOCK axis) when the index formulas are structured as `y0[:, None] + stride * xindex[None, :]` for loads and `xindex[None, :] + out_stride * yindex[:, None]` for stores.

> **Applicable when**: Kernel performs a tensor layout permutation/transpose where: (1) MemoryCacheAccessPattern or UncoalescedGlobalAccess NCU rules fire with >20% excessive sectors, (2) L2 Cache Throughput > 80% while Compute (SM) Throughput < 40%, (3) the load has a stride-N pattern (N > 1) along the fast-varying warp dimension, but a stride-1 dimension exists along the slow-varying tile axis. The fix applies when the stride-1 load dimension and stride-1 store dimension are orthogonal (one is along Y, the other along X in the 2D tile).

> **Avoid when**: The stride-1 load and stride-1 store dimensions are the same axis (no transpose needed, just use 1D tiling). Also avoid when the tile dimensions cannot be independently assigned to load/store coalescing axes (e.g., both load and store have the same non-unit stride). Do not add shared memory or tl.trans() — prior attempts with those constructs caused correctness failures for this pattern.

---

## Iteration 3

**Best so far**: 1.5786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_larger_tiles_and_higher_num_stages_for_latency_hiding | 0.990× | 0.875× | ✓ |  |
| B | coalesced_tile_axis_swap_index_formula_restructure | 1.941× | 1.988× | ✓ |  |
| C | fix_grid_lambda_and_coalesced_index_formula_for_transpose | 3.047× | 1.570× | ✓ |  |
| D | swap_tile_fast_dim_for_coalesced_load_store_axes | 1.944× | 2.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_larger_t... | Δ | coalesced_tile_ax... | Δ | fix_grid_lambda_a... | Δ | swap_tile_fast_di... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286270 ns | 327360 ns | +41090 ns (+14.4%) | 144030 ns | -142240 ns (-49.7%) | 182340 ns | -103930 ns (-36.3%) | 141980 ns | -144290 ns (-50.4%) |
| SM Throughput (%) | 19.35 % | 61.71 % | +42.36 pp | 47.32 % | +27.97 pp | 30.33 % | +10.98 pp | 48.00 % | +28.65 pp |
| Memory Throughput (%) | 78.80 % | 71.02 % | -7.78 pp | 41.30 % | -37.50 pp | 91.01 % | +12.21 pp | 41.94 % | -36.86 pp |
| DRAM Throughput (%) | 72.09 % | 30.80 % | -41.29 pp | 34.18 % | -37.91 pp | 55.51 % | -16.58 pp | 34.58 % | -37.51 pp |
| Achieved Occupancy (%) | 94.05 % | 77.88 % | -16.17 pp | 91.94 % | -2.11 pp | 92.00 % | -2.05 pp | 91.88 % | -2.17 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 | 25.00 register/thread | +0 | 23.00 register/thread | -2 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 73.70 | +13.82 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 52.45 % | -27.96 pp | 53.68 % | -26.73 pp | 68.43 % | -11.98 pp | 53.82 % | -26.59 pp |
| IPC Active | 0.79 inst/cycle | 1.92 inst/cycle | +1.13 | 1.89 inst/cycle | +1.10 | 1.25 inst/cycle | +0.46 | 1.86 inst/cycle | +1.07 |

---

## Iteration 4

**Best so far**: 1.5786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_for_latency_hiding | 1.945× | 2.013× | ✓ |  |
| B | evict_first_load_policy_for_streaming_transpose | 1.891× | 1.975× | ✓ |  |
| C | fix_hardcoded_grid_to_meta_lambda_to_unlock_autotune | 3.058× | 1.559× | ✓ | **✓ best** |
| D | autotune_yblock128_aligned_configs_with_wider_xblock_and_more_warps | 0.966× | 0.970× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | evict_first_load_... | Δ | fix_hardcoded_gri... | Δ | autotune_yblock12... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286270 ns | 142240 ns | -144030 ns (-50.3%) | 144930 ns | -141340 ns (-49.4%) | 183650 ns | -102620 ns (-35.8%) | 295170 ns | +8900 ns (+3.1%) |
| SM Throughput (%) | 19.35 % | 47.34 % | +27.99 pp | 46.07 % | +26.72 pp | 30.62 % | +11.27 pp | 65.57 % | +46.22 pp |
| Memory Throughput (%) | 78.80 % | 41.31 % | -37.49 pp | 40.00 % | -38.80 pp | 90.33 % | +11.53 pp | 70.66 % | -8.14 pp |
| DRAM Throughput (%) | 72.09 % | 34.52 % | -37.57 pp | 33.34 % | -38.75 pp | 55.10 % | -16.99 pp | 16.84 % | -55.25 pp |
| Achieved Occupancy (%) | 94.05 % | 91.95 % | -2.10 pp | 91.99 % | -2.06 pp | 92.01 % | -2.04 pp | 92.25 % | -1.80 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 23.00 register/thread | -2 | 32.00 register/thread | +7 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 119.76 | +59.88 |
| No Eligible (%) | 80.41 % | 53.73 % | -26.68 pp | 54.80 % | -25.61 pp | 68.98 % | -11.43 pp | 38.08 % | -42.33 pp |
| IPC Active | 0.79 inst/cycle | 1.82 inst/cycle | +1.03 | 1.83 inst/cycle | +1.04 | 1.25 inst/cycle | +0.46 | 2.48 inst/cycle | +1.69 |


### Key Learnings

#### ✅ Plan B: `evict_first_load_policy_for_streaming_transpose` (1.975×)

Changing tl.load eviction_policy from 'evict_last' to 'evict_first' in a streaming transpose kernel was predicted to reduce L2 pollution and lower warp stall cycles, but the NCU profiling metrics after the change show no meaningful improvement: Duration increased slightly (141.98 µs → 144.93 µs), No Eligible % worsened (53.82% → 54.8%), and Eligible Warps Per Scheduler decreased (3.35 → 3.26). All key metrics are essentially unchanged or marginally worse, indicating the eviction policy change alone does not address the dominant stall source in this kernel. The reported 1.9752x speedup must therefore be attributed to the concurrent autotune config changes (num_warps and num_stages adjustments in the diff) rather than the eviction policy itself.

> **Applicable when**: Kernel is a streaming one-shot memory access (each element read exactly once, no reuse), L1/TEX Hit Rate ~0%, IssueSlotUtilization rule fires, and the dominant bottleneck is L2 pollution from retained cache lines. Only apply if autotune config space is already well-explored and eviction policy is the remaining variable.

> **Avoid when**: The kernel's profiling metrics (Duration, No Eligible %, Eligible Warps) are already in a regime where IssueSlotUtilization stalls are not L2-eviction-driven. When the speedup is confounded by simultaneous autotune config changes, the eviction policy change alone cannot be credited. Do not apply as a standalone optimization when other structural changes (tile sizes, num_warps, num_stages) are also being modified in the same iteration.

---

## Iteration 5

**Best so far**: 1.5786× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_num_warps_to_16_for_latency_hiding | 1.945× | 1.989× | ✓ |  |
| B | fix_hardcoded_grid_and_expand_num_stages_for_latency_hiding | 1.833× | 1.980× | ✓ |  |
| C | fix_hardcoded_grid_lambda_with_pruned_square_tile_autotune_and_vectorized_store_hints | 1.945× | 2.012× | ✓ |  |
| D | fix_hardcoded_grid_and_balanced_square_tile_autotune | 1.938× | 2.031× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_num_warps_... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 286270 ns | 143900 ns | -142370 ns (-49.7%) | 144580 ns | -141690 ns (-49.5%) | 142270 ns | -144000 ns (-50.3%) | 140960 ns | -145310 ns (-50.8%) |
| SM Throughput (%) | 19.35 % | 47.29 % | +27.94 pp | 46.42 % | +27.07 pp | 47.57 % | +28.22 pp | 48.53 % | +29.18 pp |
| Memory Throughput (%) | 78.80 % | 41.28 % | -37.52 pp | 40.48 % | -38.32 pp | 41.40 % | -37.40 pp | 42.30 % | -36.50 pp |
| DRAM Throughput (%) | 72.09 % | 34.18 % | -37.91 pp | 33.76 % | -38.33 pp | 34.16 % | -37.93 pp | 34.92 % | -37.17 pp |
| Achieved Occupancy (%) | 94.05 % | 91.82 % | -2.23 pp | 91.96 % | -2.09 pp | 91.83 % | -2.22 pp | 89.61 % | -4.44 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 25.00 register/thread | +0 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 80.41 % | 53.78 % | -26.63 pp | 51.67 % | -28.74 pp | 51.52 % | -28.89 pp | 56.56 % | -23.85 pp |
| IPC Active | 0.79 inst/cycle | 1.88 inst/cycle | +1.09 | 1.91 inst/cycle | +1.12 | 1.90 inst/cycle | +1.11 | 1.75 inst/cycle | +0.96 |

---

## Cumulative Progress

```
Baseline:  286270 ns  (1.000×)
Iter 1     181340 ns  (1.579×)  — vectorized_loads_with_contiguous_hints

Final best: 183650 ns  (1.5588×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,61 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # num_stages=3 (existing)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=8, num_stages=3),
+        # num_stages=4 (new — deeper prefetch pipeline)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=8, num_stages=4),
+        # num_stages=5 (new — maximum prefetch depth for DRAM-latency hiding)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=4, num_stages=5),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 32},  num_warps=8, num_stages=5),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 256}, num_warps=8, num_stages=5),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 31500
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
-
-
+    yoffs = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)
+    xoffs = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
+    ymask = yoffs < ynumel
+    xmask = xoffs < xnumel
+    y0 = yoffs % 128
+    y1 = yoffs // 128
+    # Load: y0 varies along YBLOCK [:, None] -> stride-1 in input -> coalesced
+    load_addr = y0[:, None] + 128 * xoffs[None, :] + 4032000 * y1[:, None]
+    tmp0 = tl.load(in_ptr0 + load_addr, mask=ymask[:, None] & xmask[None, :], eviction_policy='evict_last')
+    # Store: xoffs varies along XBLOCK [None, :] -> stride-1 in output -> coalesced
+    store_addr = xoffs[None, :] + 31500 * yoffs[:, None]
+    tl.store(out_ptr0 + store_addr, tmp0, mask=ymask[:, None] & xmask[None, :])
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +67,7 @@

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
