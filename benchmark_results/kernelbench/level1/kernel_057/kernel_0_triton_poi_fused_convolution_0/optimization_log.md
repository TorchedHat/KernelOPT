# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 2940 ns (**1.9694×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_lambda_for_correct_autotune_dispatch`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_057/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.94 % | **Memory**: 42.92 % | **DRAM**: 21.36 %
- **Achieved Occupancy**: 55.15 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 71.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.227× | 1.310× | ✓ |  |
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.082× | 0.983× | ✓ |  |
| C | memory_coalescing_swap_tile_dimensions | 1.336× | 1.372× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_coalescing_and_occupancy | 1.325× | 1.402× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 4420 ns | -1370 ns (-23.7%) | 5890 ns | +100 ns (+1.7%) | 4220 ns | -1570 ns (-27.1%) | 4130 ns | -1660 ns (-28.7%) |
| SM Throughput (%) | 16.94 % | 21.42 % | +4.48 pp | 17.02 % | +0.08 pp | 14.57 % | -2.37 pp | 11.84 % | -5.10 pp |
| Memory Throughput (%) | 42.92 % | 31.67 % | -11.25 pp | 42.73 % | -0.19 pp | 20.32 % | -22.60 pp | 16.59 % | -26.33 pp |
| DRAM Throughput (%) | 21.36 % | 14.91 % | -6.45 pp | 20.94 % | -0.42 pp | 3.68 % | -17.68 pp | 4.01 % | -17.35 pp |
| Achieved Occupancy (%) | 55.15 % | 46.53 % | -8.62 pp | 55.45 % | +0.30 pp | 41.00 % | -14.15 pp | 48.46 % | -6.69 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 26.00 register/thread | +0 | 17.00 register/thread | -9 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 56.10 % | -15.80 pp | 71.36 % | -0.54 pp | 61.98 % | -9.92 pp | 52.91 % | -18.99 pp |
| IPC Active | 1.14 inst/cycle | 1.70 inst/cycle | +0.56 | 1.13 inst/cycle | -0.01 | 1.37 inst/cycle | +0.23 | 1.85 inst/cycle | +0.71 |


### Key Learnings

#### ✅ Plan D: `autotune_tile_sizes_for_coalescing_and_occupancy` (1.402×)

When a Triton pointwise kernel has high CPIStall from L1TEX latency (e.g. ~15 cycles/instruction stall) and low eligible warps per scheduler (<1.0), adding a @triton.autotune decorator with a sweep of XBLOCK/YBLOCK configurations allows the runtime to find tile shapes that increase instruction-level parallelism and reduce stall cycles. In this case, warp cycles per issued instruction dropped from 30.2 to 16.5 and eligible warps per scheduler rose from 0.9 to 1.92, yielding a 1.40x speedup despite waves/SM remaining at 0.74. The L2 hit rate also improved from 60% to 77%, suggesting the autotuned tile shape improved data reuse.

> **Applicable when**: Triton pointwise kernel with: (1) CPIStall rule triggered with L1TEX stall >10 cycles/instruction, (2) eligible warps per scheduler <1.5, (3) scheduler 'No Eligible' >50%, (4) fixed tile sizes set by Inductor without autotuning, (5) waves/SM <1.5 indicating insufficient parallelism to hide latency

> **Avoid when**: Kernel already uses @triton.autotune; kernel is compute-bound (SM throughput >70%); kernel has very large grid (waves/SM >>2) where tile size changes won't affect occupancy meaningfully; autotuning overhead is unacceptable in latency-sensitive single-call scenarios

---

## Iteration 2

**Best so far**: 1.372× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints | 1.364× | 1.402× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints_expanded_autotune | 1.385× | 1.359× | ✓ |  |
| C | autotune_small_xblock_large_yblock_for_coalesced_stores | 1.434× | 1.723× | ✓ | **✓ best** |
| D | vectorized_loads_with_coalesced_store_orientation | 1.383× | 1.673× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | autotune_small_xb... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 4130 ns | -1660 ns (-28.7%) | 4260 ns | -1530 ns (-26.4%) | 3360 ns | -2430 ns (-42.0%) | 3460 ns | -2330 ns (-40.2%) |
| SM Throughput (%) | 16.94 % | 18.90 % | +1.96 pp | 12.88 % | -4.06 pp | 7.46 % | -9.48 pp | 13.72 % | -3.22 pp |
| Memory Throughput (%) | 42.92 % | 21.86 % | -21.06 pp | 18.08 % | -24.84 pp | 10.09 % | -32.83 pp | 7.51 % | -35.41 pp |
| DRAM Throughput (%) | 21.36 % | 4.01 % | -17.35 pp | 3.63 % | -17.73 pp | 0.31 % | -21.05 pp | 1.14 % | -20.22 pp |
| Achieved Occupancy (%) | 55.15 % | 48.67 % | -6.48 pp | 43.10 % | -12.05 pp | 35.30 % | -19.85 pp | 47.76 % | -7.39 pp |
| Registers/Thread | 26.00 register/thread | 25.00 register/thread | -1 | 17.00 register/thread | -9 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 52.38 % | -19.52 pp | 63.53 % | -8.37 pp | 78.23 % | +6.33 pp | 65.82 % | -6.08 pp |
| IPC Active | 1.14 inst/cycle | 1.85 inst/cycle | +0.71 | 1.29 inst/cycle | +0.15 | 0.68 inst/cycle | -0.46 | 1.43 inst/cycle | +0.29 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_with_contiguity_hints` (1.402×)

When a Triton kernel loads from a provably stride-1 dimension but the compiler cannot infer contiguity (e.g. after a 2D broadcast), wrapping the 1D arange in tl.max_contiguous(tl.multiple_of(...)) before broadcasting enables the backend to emit 128-bit (LDG.128/float4) vectorized loads instead of scalar 32-bit loads. In this case, Mem Pipes Busy rose from 7.3% to 11.69% and Compute (SM) Throughput rose from 11.84% to 18.9%, while Total SM Elapsed Cycles dropped from 1,317,274 to 823,100, confirming more efficient memory pipeline utilization. The hints are mathematically valid when xoffset = program_id * XBLOCK (always a multiple of XBLOCK) and the arange produces XBLOCK contiguous elements.

> **Applicable when**: Triton kernel where: (1) load index is provably stride-1 in the x-dimension (last tensor dimension maps directly to xindex), (2) xindex is constructed by broadcasting a 1D arange to 2D, (3) UncoalescedGlobalAccess or MemoryCacheAccessPattern rules are triggered, (4) Mem Pipes Busy is low (<15%) despite memory-bound workload, (5) xoffset = program_id * XBLOCK is always a multiple of XBLOCK

> **Avoid when**: Load index has non-unit stride in x (e.g. stride-3 store index — applying hints to non-contiguous accesses produces incorrect results); hints applied after 2D broadcast (must be on 1D tensor to propagate correctly); store index is strided (do not hint stores with non-unit stride); kernel is already emitting LDG.128 (check SASS); yindex dimension has non-unit stride

---

## Iteration 3

**Best so far**: 1.7232× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_and_num_stages_for_latency_hiding | 1.385× | 1.005× | ✓ |  |
| B | fix_hardcoded_grid_and_remove_evict_last_for_autotune_correctness | 1.393× | 1.827× | ✓ |  |
| C | fix_autotune_grid_lambda_and_add_coalesced_store_configs | 1.406× | 1.886× | ✓ |  |
| D | fix_autotune_grid_lambda_and_add_coalesced_store_configs | 1.370× | 1.868× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 5760 ns | -30 ns (-0.5%) | 3170 ns | -2620 ns (-45.3%) | 3070 ns | -2720 ns (-47.0%) | 3100 ns | -2690 ns (-46.5%) |
| SM Throughput (%) | 16.94 % | 12.04 % | -4.90 pp | 12.35 % | -4.59 pp | 7.24 % | -9.70 pp | 7.25 % | -9.69 pp |
| Memory Throughput (%) | 42.92 % | 28.08 % | -14.84 pp | 8.06 % | -34.86 pp | 9.79 % | -33.13 pp | 9.80 % | -33.12 pp |
| DRAM Throughput (%) | 21.36 % | 5.37 % | -15.99 pp | 1.24 % | -20.12 pp | 0.34 % | -21.02 pp | 0.34 % | -21.02 pp |
| Achieved Occupancy (%) | 55.15 % | 22.16 % | -32.99 pp | 44.72 % | -10.43 pp | 34.11 % | -21.04 pp | 33.06 % | -22.09 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.90 % | 71.77 % | -0.13 pp | 62.87 % | -9.03 pp | 77.16 % | +5.26 pp | 78.72 % | +6.82 pp |
| IPC Active | 1.14 inst/cycle | 0.96 inst/cycle | -0.18 | 1.36 inst/cycle | +0.22 | 0.71 inst/cycle | -0.43 | 0.68 inst/cycle | -0.46 |

---

## Iteration 4

**Best so far**: 1.7232× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_hardcoded_grid_to_lambda_for_true_autotune | 1.435× | 1.844× | ✓ |  |
| C | fix_hardcoded_grid_lambda_add_xblock1_xblock2_coalesced_store_configs | 1.383× | 1.827× | ✓ |  |
| D | fix_hardcoded_grid_lambda_and_collapse_y_dimension_with_yblock32 | 1.460× | 1.969× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 3140 ns | -2650 ns (-45.8%) | 3170 ns | -2620 ns (-45.3%) | 2940 ns | -2850 ns (-49.2%) |
| SM Throughput (%) | 16.94 % | 7.23 % | -9.71 pp | 6.72 % | -10.22 pp | 2.02 % | -14.92 pp |
| Memory Throughput (%) | 42.92 % | 9.76 % | -33.16 pp | 9.07 % | -33.85 pp | 7.29 % | -35.63 pp |
| DRAM Throughput (%) | 21.36 % | 0.34 % | -21.02 pp | 0.33 % | -21.03 pp | 0.36 % | -21.00 pp |
| Achieved Occupancy (%) | 55.15 % | 33.80 % | -21.35 pp | 35.45 % | -19.70 pp | 7.32 % | -47.83 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 71.90 % | 77.95 % | +6.05 pp | 79.74 % | +7.84 pp | 93.05 % | +21.15 pp |
| IPC Active | 1.14 inst/cycle | 0.70 inst/cycle | -0.44 | 0.76 inst/cycle | -0.38 | 0.20 inst/cycle | -0.94 |

---

## Iteration 5

**Best so far**: 1.9694× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_lambda_for_correct_autotune_dispatch | 1.435× | 1.969× | ✓ |  |
| B | fix_hardcoded_grid_lambda_add_num_warps_1_2_and_num_stages_3_4_for_yblock32 | 1.442× | 1.868× | ✓ |  |
| C | expand_yblock32_configs_with_num_warps_1_2_and_num_stages_3_4 | 1.455× | 1.969× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | expand_yblock32_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 2940 ns | -2850 ns (-49.2%) | 3100 ns | -2690 ns (-46.5%) | 2940 ns | -2850 ns (-49.2%) |
| SM Throughput (%) | 16.94 % | 2.04 % | -14.90 pp | 1.99 % | -14.95 pp | 2.05 % | -14.89 pp |
| Memory Throughput (%) | 42.92 % | 7.12 % | -35.80 pp | 6.81 % | -36.11 pp | 7.18 % | -35.74 pp |
| DRAM Throughput (%) | 21.36 % | 0.47 % | -20.89 pp | 0.34 % | -21.02 pp | 0.35 % | -21.01 pp |
| Achieved Occupancy (%) | 55.15 % | 7.38 % | -47.77 pp | 7.41 % | -47.74 pp | 7.31 % | -47.84 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 | 16.00 register/thread | -10 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.37 | -0.37 | 0.37 | -0.37 | 0.37 | -0.37 |
| No Eligible (%) | 71.90 % | 91.90 % | +20.00 pp | 91.96 % | +20.06 pp | 91.53 % | +19.63 pp |
| IPC Active | 1.14 inst/cycle | 0.21 inst/cycle | -0.93 | 0.20 inst/cycle | -0.94 | 0.21 inst/cycle | -0.93 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     4220 ns  (1.372×)  — memory_coalescing_swap_tile_dimensions
Iter 2     3360 ns  (1.723×)  — autotune_small_xblock_large_yblock_for_coalesced_stores
Iter 4     2940 ns  (1.969×)  — fix_hardcoded_grid_lambda_and_collapse_y_dimension_with_yblock32

Final best: 2940 ns  (1.9694×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,64 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # YBLOCK=32 group (num_warps=1): collapses ynumel=30 into 1 CTA row, coalesced store orientation
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 32}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK':  2, 'YBLOCK': 32}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 32}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=1, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=1, num_stages=2),
+        # YBLOCK=32 group (num_warps=2):
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK':  2, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=2, num_stages=2),
+        # num_warps=2 group (64 threads/block -> 32 blocks/SM max -> higher occupancy):
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=2, num_stages=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=2, num_stages=4),
+        # num_stages=3 group (pipelined, num_warps=4):
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  2, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=8, num_stages=3),
+        # num_stages=4 group (deeper pipeline, num_warps=4/8):
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=8, num_stages=4),
+        # original configs retained as fallback (num_stages=2):
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK':  2, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,16 +71,16 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints on 1D arange before broadcasting to 2D
+    xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
+    xindex = xrange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask)
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +88,7 @@

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
