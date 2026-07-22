# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 2940 ns (**1.1973×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_to_use_autotuned_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3520 ns
- **SM Throughput**: 22.92 % | **Memory**: 21.19 % | **DRAM**: 2.61 %
- **Achieved Occupancy**: 52.77 %
- **IPC Active**: 1.99 inst/cycle | **No Eligible**: 46.94 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_underutilized_2d_pointwise_kernel | 1.061× | 1.222× | ✓ | **✓ best** |
| D | fix_tile_sizes_and_vectorize_loads_for_transpose_kernel | 1.051× | 1.135× | ✓ |  |
| E | add_triton_autotune | 1.030× | 0.528× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_sizes_an... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2880 ns | -640 ns (-18.2%) | 3100 ns | -420 ns (-11.9%) | 6660 ns | +3140 ns (+89.2%) |
| SM Throughput (%) | 22.92 % | 7.73 % | -15.19 pp | 14.49 % | -8.43 pp | 43.44 % | +20.52 pp |
| Memory Throughput (%) | 21.19 % | 3.18 % | -18.01 pp | 13.20 % | -7.99 pp | 51.61 % | +30.42 pp |
| DRAM Throughput (%) | 2.61 % | 3.18 % | +0.57 pp | 2.94 % | +0.33 pp | 1.38 % | -1.23 pp |
| Achieved Occupancy (%) | 52.77 % | 36.08 % | -16.69 pp | 25.85 % | -26.92 pp | 45.70 % | -7.07 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 30.00 register/thread | +2 | 48.00 register/thread | +20 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.36 | -0.37 | 1.16 | +0.43 |
| No Eligible (%) | 46.94 % | 65.36 % | +18.42 pp | 50.56 % | +3.62 pp | 35.94 % | -11.00 pp |
| IPC Active | 1.99 inst/cycle | 1.07 inst/cycle | -0.92 | 1.75 inst/cycle | -0.24 | 2.54 inst/cycle | +0.55 |

---

## Iteration 2

**Best so far**: 1.2222× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_with_persistent_tile_scheduling | 1.074× | 1.222× | ✓ | **✓ best** |
| B | persistent_tile_scheduling_with_correct_grid_lambda | 0.004× | 0.002× | ✓ |  |
| C | persistent_tile_scheduling_to_eliminate_partial_wave | 0.003× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2880 ns | -640 ns (-18.2%) | 1520000 ns | +1516480 ns (+43081.8%) | 1880000 ns | +1876480 ns (+53309.1%) |
| SM Throughput (%) | 22.92 % | 7.11 % | -15.81 pp | 62.66 % | +39.74 pp | 76.48 % | +53.56 pp |
| Memory Throughput (%) | 21.19 % | 3.18 % | -18.01 pp | 75.65 % | +54.46 pp | 75.63 % | +54.44 pp |
| DRAM Throughput (%) | 2.61 % | 3.18 % | +0.57 pp | 0.01 % | -2.60 pp | 0.01 % | -2.60 pp |
| Achieved Occupancy (%) | 52.77 % | 45.63 % | -7.14 pp | 66.07 % | +13.30 pp | 57.47 % | +4.70 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 46.94 % | 66.28 % | +19.34 pp | 33.23 % | -13.71 pp | 21.08 % | -25.86 pp |
| IPC Active | 1.99 inst/cycle | 0.88 inst/cycle | -1.11 | 2.67 inst/cycle | +0.68 | 3.15 inst/cycle | +1.16 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling_with_correct_grid_lambda` (0.002×)

For the fourth consecutive iteration (iterations 2, 3, 4, and 5), converting a 2D Triton pointwise kernel to a persistent tile-loop kernel using `tl.range(sm_id, total_tiles, num_sms)` caused a ~1,860x increase in executed instructions (406K → 755M) and a ~528x increase in kernel duration (2.88 µs → 1.52 ms). The root cause is identical each time: the grid at the call site was NOT updated from the original `(cdiv(xnumel, 128), cdiv(ynumel, 8)) = (1, 1536)` to `(NUM_SMS,) = (132,)`. With 1536 CTAs launched and `num_sms = tl.num_programs(0) = 1536`, each CTA iterates over ceil(total_tiles/1536) tiles but the sheer overhead of 1536 CTAs each running the persistent loop body, doubled register pressure (16→32 registers/thread), and 4 KB dynamic shared memory allocation collectively cause catastrophic regression. The persistent loop pattern is ONLY valid when the grid is simultaneously reduced to ≤ num_SMs CTAs at the launch site.

> **Applicable when**: Never apply this broken pattern. This is the fifth confirmation of the same anti-pattern: persistent loop body added but grid launch site left unchanged at original large CTA count.

> **Avoid when**: Always: when converting to a persistent kernel with an intra-CTA tile loop using tl.range(sm_id, total_tiles, num_sms), the grid MUST be simultaneously reduced to (NUM_SMS,). Leaving the original large grid (e.g., 1536 CTAs) while adding a tile loop causes catastrophic O(grid_size) work multiplication. This anti-pattern has now caused regression in iterations 2, 3, 4, and 5.

#### ❌ Plan C: `persistent_tile_scheduling_to_eliminate_partial_wave` (0.007×)

Converting a standard 2D-grid Triton kernel into a persistent kernel using `tl.range(sm_id, total_tiles, num_sms)` while keeping the original grid size (1536 CTAs instead of 132) caused a ~628x increase in executed instructions (406K → 254M) and a ~175x increase in kernel duration (2.88 µs → 504 µs). The grid was not changed to (NUM_SMS,) = 132 blocks; instead 1536 CTAs were launched, each iterating over all tiles via the persistent loop, producing 1536 × (total_tiles / num_sms) redundant tile executions. Despite the L1 hit rate rising to 93.8% and L2 hit rate to 100% (data fits in cache due to repeated access), the sheer instruction explosion dominates. This is the same anti-pattern as exp-002 (persistent while-loop), now manifesting with `tl.range` instead of a while-loop: the grid MUST be changed to (NUM_SMS,) atomically with the persistent loop body.

> **Applicable when**: Never apply this pattern. The persistent tl.range loop is only valid when the grid is simultaneously changed to (NUM_SMS,) at the call site.

> **Avoid when**: Do NOT add a persistent tl.range loop to a Triton kernel body without simultaneously changing the launch grid to (NUM_SMS,) = (132,). If the grid remains at the original tile-count size (e.g., 1536), each CTA will redundantly execute the full tile loop, causing O(grid_size × total_tiles / num_sms) work explosion. This is especially dangerous when the kernel is already short (<10 µs) and the grid is large relative to SM count. Also avoid when autotune is active and the grid lambda does not account for the persistent scheduling — the autotuner may select a config that makes the mismatch worse.

---

## Iteration 3

**Best so far**: 1.2222× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_autotuned_tile_sizes | 1.115× | 1.197× | ✓ | **✓ best** |
| B | shared_memory_tile_transpose_for_coalesced_access | 0.004× | 0.002× | ✓ |  |
| C | revert_broken_persistent_loop_and_fix_tile_sizes | 0.997× | 0.992× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | shared_memory_til... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2940 ns | -580 ns (-16.5%) | 1580000 ns | +1576480 ns (+44786.4%) | 3550 ns | +30 ns (+0.9%) |
| SM Throughput (%) | 22.92 % | 8.14 % | -14.78 pp | 62.85 % | +39.93 pp | 22.28 % | -0.64 pp |
| Memory Throughput (%) | 21.19 % | 3.09 % | -18.10 pp | 75.88 % | +54.69 pp | 19.99 % | -1.20 pp |
| DRAM Throughput (%) | 2.61 % | 3.09 % | +0.48 pp | 0.01 % | -2.60 pp | 2.60 % | -0.01 pp |
| Achieved Occupancy (%) | 52.77 % | 36.50 % | -16.27 pp | 65.64 % | +12.87 pp | 52.35 % | -0.42 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 30.00 register/thread | +2 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 46.94 % | 65.35 % | +18.41 pp | 33.01 % | -13.93 pp | 46.38 % | -0.56 pp |
| IPC Active | 1.99 inst/cycle | 1.07 inst/cycle | -0.92 | 2.67 inst/cycle | +0.68 | 2.06 inst/cycle | +0.07 |

---

## Iteration 4

**Best so far**: 1.2222× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_match_autotuned_tile_sizes_with_coalescing_configs | 1.066× | 1.222× | ✓ |  |
| C | fix_persistent_kernel_grid_to_num_sms_with_correct_tile_sizes | 0.004× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2880 ns | -640 ns (-18.2%) | 1490000 ns | +1486480 ns (+42229.5%) |
| SM Throughput (%) | 22.92 % | 17.85 % | -5.07 pp | 62.94 % | +40.02 pp |
| Memory Throughput (%) | 21.19 % | 6.57 % | -14.62 pp | 75.97 % | +54.78 pp |
| DRAM Throughput (%) | 2.61 % | 3.15 % | +0.54 pp | 0.01 % | -2.60 pp |
| Achieved Occupancy (%) | 52.77 % | 59.07 % | +6.30 pp | 65.30 % | +12.53 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.73 | 1.45 | +0.72 | 0.73 | +0.00 |
| No Eligible (%) | 46.94 % | 43.96 % | -2.98 pp | 32.96 % | -13.98 pp |
| IPC Active | 1.99 inst/cycle | 2.13 inst/cycle | +0.14 | 2.62 inst/cycle | +0.63 |

---

## Iteration 5

**Best so far**: 1.2222× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_call_site_grid_lambda_to_use_autotuned_tile_sizes_correctly | 1.066× | 1.210× | ✓ |  |
| B | fix_grid_lambda_and_improve_store_coalescing_via_loop_reorder | 1.070× | 0.964× | ✓ |  |
| C | fix_persistent_kernel_grid_and_xblock_for_correct_sm_utilization | 0.004× | 0.002× | ✓ |  |
| D | increase_yblock_and_fix_xblock_waste_for_large_tile_cache_reuse | 0.026× | 0.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_call_site_gri... | Δ | fix_grid_lambda_a... | Δ | fix_persistent_ke... | Δ | increase_yblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2910 ns | -610 ns (-17.3%) | 3650 ns | +130 ns (+3.7%) | 1520000 ns | +1516480 ns (+43081.8%) | 504060 ns | +500540 ns (+14219.9%) |
| SM Throughput (%) | 22.92 % | 17.69 % | -5.23 pp | 21.88 % | -1.04 pp | 61.71 % | +38.79 pp | 64.88 % | +41.96 pp |
| Memory Throughput (%) | 21.19 % | 6.51 % | -14.68 pp | 9.61 % | -11.58 pp | 74.51 % | +53.32 pp | 28.98 % | +7.79 pp |
| DRAM Throughput (%) | 2.61 % | 3.15 % | +0.54 pp | 2.53 % | -0.08 pp | 0.01 % | -2.60 pp | 0.03 % | -2.58 pp |
| Achieved Occupancy (%) | 52.77 % | 59.19 % | +6.42 pp | 66.13 % | +13.36 pp | 65.18 % | +12.41 pp | 55.47 % | +2.70 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 20.00 register/thread | -8 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 1.45 | +0.72 | 1.45 | +0.72 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 46.94 % | 41.13 % | -5.81 pp | 34.19 % | -12.75 pp | 34.51 % | -12.43 pp | 31.36 % | -15.58 pp |
| IPC Active | 1.99 inst/cycle | 2.17 inst/cycle | +0.18 | 2.50 inst/cycle | +0.51 | 2.61 inst/cycle | +0.62 | 2.72 inst/cycle | +0.73 |

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 1     2880 ns  (1.222×)  — autotune_tile_sizes_for_underutilized_2d_pointwise_kernel

Final best: 2940 ns  (1.1973×)
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
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK':  64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK':  32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +38,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
