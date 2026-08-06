# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950 ns — **Best latency**: 4450 ns (**1.3371×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5950 ns
- **SM Throughput**: 16.30 % | **Memory**: 41.56 % | **DRAM**: 20.77 %
- **Achieved Occupancy**: 55.75 %
- **IPC Active**: 1.08 inst/cycle | **No Eligible**: 70.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune_tile_sizes | 1.398× | 1.337× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_underutilized_transpose_kernel | 1.351× | 1.441× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_sizes | 1.059× | 1.033× | ✓ |  |
| D | autotune_tile_sizes_for_coalescing_and_occupancy | 1.107× | 0.657× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 4450 ns | -1500 ns (-25.2%) | 4130 ns | -1820 ns (-30.6%) | 5760 ns | -190 ns (-3.2%) | 9060 ns | +3110 ns (+52.3%) |
| SM Throughput (%) | 16.30 % | 19.85 % | +3.55 pp | 16.32 % | +0.02 pp | 16.81 % | +0.51 pp | 20.30 % | +4.00 pp |
| Memory Throughput (%) | 41.56 % | 31.44 % | -10.12 pp | 33.42 % | -8.14 pp | 43.98 % | +2.42 pp | 37.93 % | -3.63 pp |
| DRAM Throughput (%) | 20.77 % | 14.86 % | -5.91 pp | 16.07 % | -4.70 pp | 21.47 % | +0.70 pp | 13.57 % | -7.20 pp |
| Achieved Occupancy (%) | 55.75 % | 46.96 % | -8.79 pp | 53.78 % | -1.97 pp | 54.90 % | -0.85 pp | 51.56 % | -4.19 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 21.00 register/thread | -5 | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.56 % | 55.79 % | -14.77 pp | 65.13 % | -5.43 pp | 70.30 % | -0.26 pp | 63.91 % | -6.65 pp |
| IPC Active | 1.08 inst/cycle | 1.70 inst/cycle | +0.62 | 1.35 inst/cycle | +0.27 | 1.16 inst/cycle | +0.08 | 1.35 inst/cycle | +0.27 |

---

## Iteration 2

**Best so far**: 1.3371× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguity_hints | 1.347× | 1.476× | ✓ |  |
| B | store_eviction_policy_and_load_eviction_streaming | 1.290× | 1.358× | ✓ |  |
| C | vectorized_loads_with_corrected_autotune_grid | 1.057× | 0.995× | ✓ |  |
| D | vectorized_loads_with_fixed_autotune_grid | 1.078× | 0.657× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | store_eviction_po... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 4030 ns | -1920 ns (-32.3%) | 4380 ns | -1570 ns (-26.4%) | 5980 ns | +30 ns (+0.5%) | 9060 ns | +3110 ns (+52.3%) |
| SM Throughput (%) | 16.30 % | 16.94 % | +0.64 pp | 21.07 % | +4.77 pp | 16.28 % | -0.02 pp | 21.57 % | +5.27 pp |
| Memory Throughput (%) | 41.56 % | 34.18 % | -7.38 pp | 32.15 % | -9.41 pp | 42.26 % | +0.70 pp | 40.16 % | -1.40 pp |
| DRAM Throughput (%) | 20.77 % | 16.34 % | -4.43 pp | 15.10 % | -5.67 pp | 20.62 % | -0.15 pp | 13.58 % | -7.19 pp |
| Achieved Occupancy (%) | 55.75 % | 55.16 % | -0.59 pp | 49.63 % | -6.12 pp | 55.18 % | -0.57 pp | 51.92 % | -3.83 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 27.00 register/thread | +1 | 26.00 register/thread | +0 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.56 % | 65.41 % | -5.15 pp | 54.44 % | -16.12 pp | 70.42 % | -0.14 pp | 64.51 % | -6.05 pp |
| IPC Active | 1.08 inst/cycle | 1.27 inst/cycle | +0.19 | 1.58 inst/cycle | +0.50 | 1.12 inst/cycle | +0.04 | 1.40 inst/cycle | +0.32 |


### Key Learnings

#### ✅ Plan A: `vectorized_loads_via_contiguity_hints` (1.476×)

Adding tl.max_contiguous + tl.multiple_of hints to the innermost index of a layout-permutation kernel, without also autotuning tile sizes, produced only a marginal 1.48x speedup (duration 4.13→4.03 µs, elapsed cycles 6075→5979). Critically, the UncoalescedGlobalAccess excess sector percentage remained unchanged at 20%, and CPIStall L1TEX scoreboard stall fraction stayed at ~37%, indicating the hints alone did not successfully eliminate uncoalesced accesses in this configuration. The small gain likely comes from minor instruction-count reduction (issued instructions 556K→552K) rather than true vectorization. The strategy is most effective when combined with autotuning tile sizes so the compiler can prove alignment across all selected XBLOCK values.

> **Applicable when**: Triton pointwise/permutation kernel with stride-1 contiguous loads in the innermost dimension; UncoalescedGlobalAccess fires with >15% excess sectors; CPIStall L1TEX scoreboard stalls >35% of warp cycles; hints are applied alongside autotuning (see exp_001_triton_autotune_vectorized_loads_layout_permute for the combined strategy).

> **Avoid when**: Hints are applied in isolation without autotuning tile sizes — this iteration shows the standalone gain is marginal (~1.48x) and does not resolve the underlying uncoalesced access pattern. Do not apply when the load dimension is non-contiguous or strided. Do not apply when the kernel is already autotuned with a lambda grid.

---

## Iteration 3

**Best so far**: 1.3371× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_with_num_stages_pipelining | 1.298× | 1.466× | ✓ |  |
| B | expand_num_warps_and_yblock_to_hide_latency_in_small_grid_kernel | 1.251× | 0.444× | ✓ |  |
| C | streaming_eviction_policy_with_num_stages_pipelining | 1.033× | 1.015× | ✓ |  |
| D | fix_autotune_grid_mismatch_with_lambda_grid | 1.357× | 1.617× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | expand_num_warps_... | Δ | streaming_evictio... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950 ns | 4060 ns | -1890 ns (-31.8%) | 13410 ns | +7460 ns (+125.4%) | 5860 ns | -90 ns (-1.5%) | 3680 ns | -2270 ns (-38.2%) |
| SM Throughput (%) | 16.30 % | 17.13 % | +0.83 pp | 49.72 % | +33.42 pp | 16.75 % | +0.45 pp | 15.31 % | -0.99 pp |
| Memory Throughput (%) | 41.56 % | 33.92 % | -7.64 pp | 60.58 % | +19.02 pp | 41.01 % | -0.55 pp | 19.27 % | -22.29 pp |
| DRAM Throughput (%) | 20.77 % | 16.27 % | -4.50 pp | 9.17 % | -11.60 pp | 21.02 % | +0.25 pp | 9.03 % | -11.74 pp |
| Achieved Occupancy (%) | 55.75 % | 55.44 % | -0.31 pp | 87.79 % | +32.04 pp | 55.19 % | -0.56 pp | 56.06 % | +0.31 pp |
| Registers/Thread | 26.00 register/thread | 21.00 register/thread | -5 | 32.00 register/thread | +6 | 26.00 register/thread | +0 | 16.00 register/thread | -10 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 2.97 | +2.23 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 70.56 % | 65.48 % | -5.08 pp | 41.06 % | -29.50 pp | 71.36 % | +0.80 pp | 61.76 % | -8.80 pp |
| IPC Active | 1.08 inst/cycle | 1.15 inst/cycle | +0.07 | 2.34 inst/cycle | +1.26 | 1.22 inst/cycle | +0.14 | 1.40 inst/cycle | +0.32 |


### Key Learnings

#### ✅ Plan D: `fix_autotune_grid_mismatch_with_lambda_grid` (1.617×)

When a Triton kernel uses @triton.autotune but the launch grid is a static tuple computed with hardcoded tile constants, the autotuner may select a tile configuration that mismatches the grid, causing redundant or incorrect block coverage and inflated instruction counts. Replacing the static grid with a lambda that reads meta['XBLOCK'] and meta['YBLOCK'] restores correct coverage for every autotuner config, reducing executed instructions from 533K to 420K and elapsed cycles from 5979 to 5446. Additionally, capping XBLOCK to values that keep the grid above one full wave (e.g., XBLOCK ≤ 128 for a problem with ~1568 blocks at 132 SMs) prevents the SOLBottleneck from worsening, and also reduces register pressure (21→16 registers/thread) as the compiler finds a more efficient allocation for the smaller tile space.

> **Applicable when**: Triton 2D tiled kernels where: (1) @triton.autotune is present but the launch grid is a static tuple using hardcoded tile constants; (2) autotune configs include XBLOCK values large enough to reduce the grid below one full SM wave; (3) IssueSlotUtilization and UncoalescedGlobalAccess rules fire; (4) executed instruction count is inflated relative to the theoretical minimum for the problem size.

> **Avoid when**: Grid size is independent of tile sizes (e.g., one block per output element); kernel is not autotuned; XBLOCK is already capped at values that keep the grid above one wave; the kernel body is compute-bound (Compute SM Throughput >70%).

---

## Cumulative Progress

```
Baseline:  5950 ns  (1.000×)
Iter 1     4450 ns  (1.337×)  — vectorized_loads_with_autotune_tile_sizes

Final best: 4450 ns  (1.3371×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK':  64, 'YBLOCK':  4}, num_warps=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK':  8}, num_warps=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK':  64, 'YBLOCK': 32}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +27,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    _xrange = tl.arange(0, XBLOCK)
+    _xrange = tl.max_contiguous(tl.multiple_of(_xrange, XBLOCK), XBLOCK)
+    xindex = xoffset + _xrange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +38,13 @@

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
