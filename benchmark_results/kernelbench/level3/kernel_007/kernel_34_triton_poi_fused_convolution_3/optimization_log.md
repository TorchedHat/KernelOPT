# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_34` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4290 ns — **Best latency**: 6050 ns (**0.7091×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_34`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_34_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2496,1,1), Waves/SM: 1.18
- **Registers/thread**: 28
- **Duration**: 4290 ns
- **SM Throughput**: 32.29 % | **Memory**: 30.23 % | **DRAM**: 3.44 %
- **Achieved Occupancy**: 68.45 %
- **IPC Active**: 2.49 inst/cycle | **No Eligible**: 39.90 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_mismatch_and_autotune_for_partial_wave_elimination | 0.985× | 1.147× | ✓ |  |
| B | fix_tile_sizes_eliminate_partial_wave | 1.009× | 0.993× | ✓ |  |
| C | fix_tile_sizes_eliminate_partial_wave_and_improve_coalescing | 1.079× | 1.240× | ✓ | **✓ best** |
| D | fix_tile_dimensions_and_vectorize_loads | 1.040× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_mismatch... | Δ | fix_tile_sizes_el... | Δ | fix_tile_sizes_el... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4290 ns | 3740 ns | -550 ns (-12.8%) | 4320 ns | +30 ns (+0.7%) | 3460 ns | -830 ns (-19.3%) | 4260 ns | -30 ns (-0.7%) |
| SM Throughput (%) | 32.29 % | 10.18 % | -22.11 pp | 31.21 % | -1.08 pp | 12.70 % | -19.59 pp | 40.38 % | +8.09 pp |
| Memory Throughput (%) | 30.23 % | 10.78 % | -19.45 pp | 29.22 % | -1.01 pp | 4.81 % | -25.42 pp | 27.39 % | -2.84 pp |
| DRAM Throughput (%) | 3.44 % | 3.99 % | +0.55 pp | 3.43 % | -0.01 pp | 4.30 % | +0.86 pp | 3.49 % | +0.05 pp |
| Achieved Occupancy (%) | 68.45 % | 17.04 % | -51.41 pp | 66.84 % | -1.61 pp | 48.17 % | -20.28 pp | 65.97 % | -2.48 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 16.00 register/thread | -12 | 27.00 register/thread | -1 |
| Waves/SM | 1.18 | 0.59 | -0.59 | 1.18 | +0.00 | 1.18 | +0.00 | 1.18 | +0.00 |
| No Eligible (%) | 39.90 % | 58.82 % | +18.92 pp | 35.70 % | -4.20 pp | 46.21 % | +6.31 pp | 28.39 % | -11.51 pp |
| IPC Active | 2.49 inst/cycle | 1.39 inst/cycle | -1.10 | 2.35 inst/cycle | -0.14 | 1.90 inst/cycle | -0.59 | 2.85 inst/cycle | +0.36 |

---

## Iteration 2

**Best so far**: 1.2399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_tile_scheduling_for_partial_wave_elimination | 0.010× | 0.006× | ✓ |  |
| C | persistent_tile_scheduling_for_partial_wave_elimination | 0.002× | 0.001× | ✓ |  |
| D | persistent_tile_scheduling_for_sm_utilization | 0.002× | 0.001× | ✓ |  |
| E | add_triton_autotune | 0.044× | 0.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4290 ns | 694720 ns | +690430 ns (+16093.9%) | 4730000 ns | +4725710 ns (+110156.4%) | 4720000 ns | +4715710 ns (+109923.3%) | 677220 ns | +672930 ns (+15686.0%) |
| SM Throughput (%) | 32.29 % | 41.95 % | +9.66 pp | 78.53 % | +46.24 pp | 78.62 % | +46.33 pp | 13.79 % | -18.50 pp |
| Memory Throughput (%) | 30.23 % | 68.12 % | +37.89 pp | 77.74 % | +47.51 pp | 77.84 % | +47.61 pp | 66.17 % | +35.94 pp |
| DRAM Throughput (%) | 3.44 % | 0.03 % | -3.41 pp | 0.01 % | -3.43 pp | 0.01 % | -3.43 pp | 66.17 % | +62.73 pp |
| Achieved Occupancy (%) | 68.45 % | 52.72 % | -15.73 pp | 72.24 % | +3.79 pp | 72.23 % | +3.78 pp | 36.12 % | -32.33 pp |
| Registers/Thread | 28.00 register/thread | 37.00 register/thread | +9 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 1.18 | 0.79 | -0.39 | 1.18 | +0.00 | 1.18 | +0.00 | 3.15 | +1.97 |
| No Eligible (%) | 39.90 % | 54.94 % | +15.04 pp | 16.99 % | -22.91 pp | 16.99 % | -22.91 pp | 92.19 % | +52.29 pp |
| IPC Active | 2.49 inst/cycle | 1.80 inst/cycle | -0.69 | 3.32 inst/cycle | +0.83 | 3.33 inst/cycle | +0.84 | 0.31 inst/cycle | -2.18 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling_for_partial_wave_elimination` (0.001×)

Rewriting a Triton pointwise kernel as a persistent kernel with an intra-CTA tile loop (using tl.range over tile_id) caused a ~1000x regression because the launch site was NOT updated: the grid remained (cdiv(xnumel,128), cdiv(ynumel,8)) = (1, 2496) CTAs with XBLOCK=128 and YBLOCK=8 passed as kwargs, while the kernel body now contained a loop over all tiles. Each of the 2,496 CTAs independently iterated over all 156 tiles, causing every element to be processed ~2,496 times instead of once. Executed instructions exploded from 330K to 2.94 billion (8,900x increase), duration jumped from 3.46µs to 4.73ms (~1,368x slower). The structural anti-pattern is implementing a persistent kernel loop in the kernel body without simultaneously reducing the grid to exactly num_sms (or total_tiles) CTAs at the launch site.

> **Applicable when**: Never apply this pattern (persistent kernel with intra-CTA tile loop) without simultaneously updating the launch grid to (num_sms,) or (total_tiles,). This negative experience is a warning for any persistent kernel rewrite.

> **Avoid when**: Any time the launch site grid is not updated to match the persistent kernel's CTA count. If the grid remains the original large value (e.g., 2496 CTAs) while the kernel body loops over all tiles, every CTA will redundantly process all tiles, causing catastrophic instruction explosion proportional to grid_size * total_tiles.

#### ❌ Plan C: `persistent_tile_scheduling_for_partial_wave_elimination` (0.001×)

Rewriting a Triton pointwise kernel as a persistent kernel with an intra-CTA tile loop (using tl.range over tile_id) caused a ~1000x regression because the launch site was NOT updated: the grid remained (cdiv(xnumel,128), cdiv(ynumel,8)) = (1, 2496) CTAs with XBLOCK=128 and YBLOCK=8 passed as kwargs, while the kernel body now contained a loop over all tiles. Each of the 2,496 CTAs independently iterated over all 156 tiles, causing every element to be processed ~2,496 times instead of once. Executed instructions exploded from 330K to 2.94 billion (8,900x increase), duration jumped from 3.46µs to 4.73ms (~1,368x slower). The structural anti-pattern is implementing a persistent kernel loop in the kernel body without simultaneously reducing the grid to exactly num_sms (or total_tiles) CTAs at the launch site.

> **Applicable when**: Never apply this pattern (persistent kernel with intra-CTA tile loop) without simultaneously updating the launch grid to (num_sms,) or (total_tiles,). This negative experience is a warning for any persistent kernel rewrite.

> **Avoid when**: Any time the launch site grid is not updated to match the persistent kernel's CTA count. If the grid remains the original large value (e.g., 2496 CTAs) while the kernel body loops over all tiles, every CTA will redundantly process all tiles, causing catastrophic instruction explosion proportional to grid_size * total_tiles.

#### ❌ Plan D: `persistent_tile_scheduling_for_sm_utilization` (0.001×)

Replacing a standard 2D grid launch with a persistent 1D grid (one CTA per SM, while-loop over tiles) caused a ~1380x increase in executed instructions (330K → 2.94B) and a ~1365x increase in kernel duration (3.46 µs → 4.72 ms). The root cause is that the persistent kernel's while-loop tile-index arithmetic (cdiv, integer division, modulo) is computed in scalar/non-constexpr fashion inside the kernel body, generating enormous instruction overhead per iteration. Additionally, registers per thread doubled (16 → 32) and dynamic shared memory was introduced (0 → 4 KB/block), both of which constrain occupancy and add overhead. The partial-wave penalty (NCU LaunchConfiguration) was not the dominant cost for this ultra-short kernel (3.46 µs); the persistent loop overhead vastly outweighed any launch-configuration savings.

> **Applicable when**: Never applicable for ultra-short kernels (<10 µs duration); avoid when total tile count is only marginally above SM count (e.g., 157 tiles / 132 SMs = 1.19 tiles/SM), as the loop overhead exceeds any partial-wave savings.

> **Avoid when**: Kernel duration is short (<100 µs); total tile count is close to SM count (tiles/SM < 4); tile-index arithmetic (cdiv, div, mod) cannot be hoisted out of the loop as constexpr; the kernel body is a simple pointwise or transpose op where per-iteration overhead dominates; registers per thread would increase due to loop variables.

---

## Iteration 3

**Best so far**: 1.2399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_staged_coalesced_transpose_with_fixed_autotune_grid | 0.788× | 0.749× | ✓ |  |
| B | revert_persistent_loop_to_simple_2d_grid_kernel | 0.993× | 1.046× | ✓ |  |
| D | fix_broken_persistent_kernel_and_optimize_eviction_policy_for_streaming_transpose | 1.093× | 1.240× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.044× | 0.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_sta... | Δ | revert_persistent... | Δ | fix_broken_persis... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4290 ns | 5730 ns | +1440 ns (+33.6%) | 4100 ns | -190 ns (-4.4%) | 3460 ns | -830 ns (-19.3%) | 677630 ns | +673340 ns (+15695.6%) |
| SM Throughput (%) | 32.29 % | 53.15 % | +20.86 pp | 29.20 % | -3.09 pp | 11.04 % | -21.25 pp | 13.79 % | -18.50 pp |
| Memory Throughput (%) | 30.23 % | 44.34 % | +14.11 pp | 32.48 % | +2.25 pp | 4.29 % | -25.94 pp | 66.20 % | +35.97 pp |
| DRAM Throughput (%) | 3.44 % | 2.59 % | -0.85 pp | 3.62 % | +0.18 pp | 4.29 % | +0.85 pp | 66.20 % | +62.76 pp |
| Achieved Occupancy (%) | 68.45 % | 67.28 % | -1.17 pp | 40.81 % | -27.64 pp | 49.73 % | -18.72 pp | 36.11 % | -32.34 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 29.00 register/thread | +1 | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 1.18 | 1.18 | +0.00 | 0.59 | -0.59 | 1.18 | +0.00 | 3.15 | +1.97 |
| No Eligible (%) | 39.90 % | 20.22 % | -19.68 pp | 37.54 % | -2.36 pp | 54.90 % | +15.00 pp | 92.19 % | +52.29 pp |
| IPC Active | 2.49 inst/cycle | 3.15 inst/cycle | +0.66 | 2.38 inst/cycle | -0.11 | 1.68 inst/cycle | -0.81 | 0.31 inst/cycle | -2.18 |


### Key Learnings

#### ✅ Plan B: `revert_persistent_loop_to_simple_2d_grid_kernel` (1.046×)

When a Triton kernel contains a persistent while-loop (tile_id += num_sms) that was introduced to eliminate partial waves, but the workload is tiny (duration < 10 µs, elements < 1M), reverting to a simple 2D grid where each CTA handles exactly one tile eliminates catastrophic instruction overhead. The persistent loop caused warp cycles per issued instruction to balloon to 73.95 (vs 10.54 after), executed instructions to inflate 42x (41M → 973K), and dynamic shared memory to consume 32 KB/block — all of which collapsed theoretical occupancy to 37.5% and stalled schedulers 92% of cycles. Restoring the 2D grid with autotuned XBLOCK/YBLOCK configs recovers theoretical occupancy to 100% and reduces duration from 677 µs to 4.1 µs (~165x improvement).

> **Applicable when**: A Triton kernel contains a persistent while-loop (tile_id += num_sms) and profiling shows: warp cycles per issued instruction > 30, scheduler 'No Eligible' > 70%, executed instructions are orders of magnitude higher than expected for the workload size, dynamic shared memory is unexpectedly large (> 4 KB/block), and theoretical occupancy is severely limited by shared memory. The workload is small (duration < 100 µs, total elements < 1M).

> **Avoid when**: The kernel is genuinely large (duration > 1 ms, elements > 10M) and waves_per_sm < 0.5 — in that case persistent scheduling may legitimately help. Also do not apply if the kernel has complex reduction logic that requires the persistent loop structure for correctness.

---

## Iteration 4

**Best so far**: 1.2399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_dead_autotune_with_lambda_grid_and_store_eviction_hint | 1.057× | 1.229× | ✓ |  |
| B | fix_hardcoded_launch_grid_and_eviction_policy_for_streaming_transpose | 1.050× | 1.057× | ✓ |  |
| C | fix_broken_autotune_launch_site_and_streaming_eviction_policy | 1.091× | 1.240× | ✓ |  |
| D | fix_shared_memory_occupancy_limiter_and_autotune_tile_configs | 1.059× | 1.240× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_dead_autotune... | Δ | fix_hardcoded_lau... | Δ | fix_broken_autotu... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4290 ns | 3490 ns | -800 ns (-18.6%) | 4060 ns | -230 ns (-5.4%) | 3460 ns | -830 ns (-19.3%) | 3460 ns | -830 ns (-19.3%) |
| SM Throughput (%) | 32.29 % | 11.09 % | -21.20 pp | 30.18 % | -2.11 pp | 10.95 % | -21.34 pp | 10.76 % | -21.53 pp |
| Memory Throughput (%) | 30.23 % | 4.29 % | -25.94 pp | 33.57 % | +3.34 pp | 4.33 % | -25.90 pp | 4.30 % | -25.93 pp |
| DRAM Throughput (%) | 3.44 % | 4.29 % | +0.85 pp | 3.65 % | +0.21 pp | 4.33 % | +0.89 pp | 4.30 % | +0.86 pp |
| Achieved Occupancy (%) | 68.45 % | 57.11 % | -11.34 pp | 40.62 % | -27.83 pp | 47.07 % | -21.38 pp | 50.64 % | -17.81 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 29.00 register/thread | +1 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 1.18 | 1.18 | +0.00 | 0.59 | -0.59 | 1.18 | +0.00 | 1.18 | +0.00 |
| No Eligible (%) | 39.90 % | 54.06 % | +14.16 pp | 37.83 % | -2.07 pp | 53.78 % | +13.88 pp | 53.77 % | +13.87 pp |
| IPC Active | 2.49 inst/cycle | 1.36 inst/cycle | -1.13 | 2.38 inst/cycle | -0.11 | 1.61 inst/cycle | -0.88 | 1.45 inst/cycle | -1.04 |

---

## Iteration 5

**Best so far**: 1.2399× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_launch_grid_to_enable_autotune_and_add_store_eviction_hint | 1.053× | 1.254× | ✓ |  |
| B | fix_hardcoded_grid_and_coalesce_stores_via_xblock9_configs | 1.094× | 0.709× | ✓ | **✓ best** |
| C | fix_dead_autotune_launch_grid_and_l2_swizzle_for_occupancy_and_workload_balance | 1.030× | 0.986× | ✓ |  |
| D | activate_dead_autotune_via_lambda_grid_and_fix_thread_waste | 1.048× | 1.240× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_lau... | Δ | fix_hardcoded_gri... | Δ | fix_dead_autotune... | Δ | activate_dead_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4290 ns | 3420 ns | -870 ns (-20.3%) | 6050 ns | +1760 ns (+41.0%) | 4350 ns | +60 ns (+1.4%) | 3460 ns | -830 ns (-19.3%) |
| SM Throughput (%) | 32.29 % | 10.87 % | -21.42 pp | 53.05 % | +20.76 pp | 35.22 % | +2.93 pp | 10.98 % | -21.31 pp |
| Memory Throughput (%) | 30.23 % | 4.32 % | -25.91 pp | 41.58 % | +11.35 pp | 31.83 % | +1.60 pp | 4.31 % | -25.92 pp |
| DRAM Throughput (%) | 3.44 % | 4.32 % | +0.88 pp | 2.46 % | -0.98 pp | 3.42 % | -0.02 pp | 4.31 % | +0.87 pp |
| Achieved Occupancy (%) | 68.45 % | 46.80 % | -21.65 pp | 65.44 % | -3.01 pp | 41.25 % | -27.20 pp | 45.80 % | -22.65 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 1.18 | 1.18 | +0.00 | 1.18 | +0.00 | 0.59 | -0.59 | 1.18 | +0.00 |
| No Eligible (%) | 39.90 % | 53.88 % | +13.98 pp | 21.61 % | -18.29 pp | 31.35 % | -8.55 pp | 54.05 % | +14.15 pp |
| IPC Active | 2.49 inst/cycle | 1.60 inst/cycle | -0.89 | 3.09 inst/cycle | +0.60 | 2.67 inst/cycle | +0.18 | 1.56 inst/cycle | -0.93 |

---

## Cumulative Progress

```
Baseline:  4290 ns  (1.000×)
Iter 1     3460 ns  (1.240×)  — fix_tile_sizes_eliminate_partial_wave_and_improve_coalescing

Final best: 6050 ns  (0.7091×)
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
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 2048}, num_warps=16),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 2048}, num_warps=16),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 19968
-    xnumel = 9
+def triton_poi_fused_convolution_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
+    # Clean 2D grid: program_id(0)=X-tile, program_id(1)=Y-tile, NO persistent loop
+    xoffset = tl.program_id(0) * XBLOCK
     yoffset = tl.program_id(1) * YBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    xmask = xindex < xnumel
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 96)
+    y0 = yindex % 96
     y1 = yindex // 96
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    # evict_first: streaming load, data not reused after this store
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_first', other=0.0)
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 19968
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_34[grid](arg_0, arg_1, 19968, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_34[grid](arg_0, arg_1, 19968, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
