# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_59` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2560 ns — **Best latency**: 2530 ns (**1.0119×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_with_safe_lambda_and_xnumel25_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_59`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_59_triton_poi_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.12
- **Registers/thread**: 28
- **Duration**: 2560 ns
- **SM Throughput**: 4.87 % | **Memory**: 5.30 % | **DRAM**: 1.70 %
- **Achieved Occupancy**: 10.94 %
- **IPC Active**: 0.54 inst/cycle | **No Eligible**: 85.20 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune | 0.986× | 0.921× | ✓ |  |
| B | autotune_tile_sizes_for_small_kernel | 1.052× | 1.080× | ✓ | **✓ best** |
| C | increase_tile_sizes_and_autotune_for_underutilized_kernel | 0.925× | 0.871× | ✓ |  |
| D | autotune_tile_sizes_for_underutilized_small_grid | 0.977× | 0.921× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | increase_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 2780 ns | +220 ns (+8.6%) | 2370 ns | -190 ns (-7.4%) | 2940 ns | +380 ns (+14.8%) | 2780 ns | +220 ns (+8.6%) |
| SM Throughput (%) | 4.87 % | 0.80 % | -4.07 pp | 1.65 % | -3.22 pp | 4.46 % | -0.41 pp | 5.31 % | +0.44 pp |
| Memory Throughput (%) | 5.30 % | 7.86 % | +2.56 pp | 1.18 % | -4.12 pp | 8.79 % | +3.49 pp | 9.20 % | +3.90 pp |
| DRAM Throughput (%) | 1.70 % | 7.86 % | +6.16 pp | 1.18 % | -0.52 pp | 5.75 % | +4.05 pp | 6.09 % | +4.39 pp |
| Achieved Occupancy (%) | 10.94 % | 5.51 % | -5.43 pp | 10.86 % | -0.08 pp | 5.69 % | -5.25 pp | 11.21 % | +0.27 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.06 | -0.06 | 0.12 | +0.00 | 0.06 | -0.06 | 0.12 | +0.00 |
| No Eligible (%) | 85.20 % | 97.57 % | +12.37 pp | 92.07 % | +6.87 pp | 86.20 % | +1.00 pp | 85.82 % | +0.62 pp |
| IPC Active | 0.54 inst/cycle | 0.07 inst/cycle | -0.47 | 0.32 inst/cycle | -0.22 | 0.50 inst/cycle | -0.04 | 0.58 inst/cycle | +0.04 |

---

## Iteration 2

**Best so far**: 1.0802× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_lambda_meta_for_autotune | 1.011× | 1.041× | ✓ |  |
| C | persistent_tile_scheduling_loop_over_work | 0.051× | 0.016× | ✓ |  |
| D | persistent_tile_scheduling_flat_grid | 0.044× | 0.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 2460 ns | -100 ns (-3.9%) | 156740 ns | +154180 ns (+6022.7%) | 170850 ns | +168290 ns (+6573.8%) |
| SM Throughput (%) | 4.87 % | 1.73 % | -3.14 pp | 16.98 % | +12.11 pp | 23.29 % | +18.42 pp |
| Memory Throughput (%) | 5.30 % | 1.11 % | -4.19 pp | 23.13 % | +17.83 pp | 25.49 % | +20.19 pp |
| DRAM Throughput (%) | 1.70 % | 1.11 % | -0.59 pp | 0.04 % | -1.66 pp | 0.03 % | -1.67 pp |
| Achieved Occupancy (%) | 10.94 % | 10.66 % | -0.28 pp | 12.12 % | +1.18 pp | 12.12 % | +1.18 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 85.20 % | 88.00 % | +2.80 pp | 82.59 % | -2.61 pp | 76.49 % | -8.71 pp |
| IPC Active | 0.54 inst/cycle | 0.38 inst/cycle | -0.16 | 0.72 inst/cycle | +0.18 | 0.94 inst/cycle | +0.40 |


### Key Learnings

#### ❌ Plan C: `persistent_tile_scheduling_loop_over_work` (0.016×)

Adding a while-loop inside a Triton kernel to implement persistent tile scheduling caused executed instructions to explode from 28,864 to 21,264,384 (~737x increase), and kernel duration to increase from 2.37 µs to 156.74 µs (~66x regression). The root cause is that the persistent loop body was executed by all 256 CTAs (the grid was not changed from the original), so each CTA iterated over all 800 tiles instead of a subset, processing the entire workload 256 times. Additionally, the while-loop with tl.cdiv, modulo, and division operations inside the kernel body added significant per-iteration overhead, and registers-per-thread doubled (16→32) while dynamic shared memory reappeared (0→4 KB), further reducing occupancy headroom. The structural anti-pattern is implementing a persistent loop without actually reducing the grid size to match the intended number of persistent workers — the grid must be changed from the original tile-covering size to the smaller persistent-worker count (e.g., num_SMs * 4), otherwise every original CTA runs the full loop.

> **Applicable when**: Never apply this negative pattern. The lesson is: when implementing a persistent kernel loop, the launch grid MUST be set to the number of persistent workers (e.g., num_SMs * oversubscription_factor), not the original tile-covering grid size.

> **Avoid when**: Do not use a persistent while-loop inside a Triton kernel unless the grid size is simultaneously reduced to the persistent worker count. If the grid size is left at the original tile-covering value, every CTA will redundantly process all tiles, causing an instruction count explosion proportional to (original_grid_size / persistent_worker_count) * total_tiles.

#### ❌ Plan D: `persistent_tile_scheduling_flat_grid` (0.015×)

Converting a Triton kernel body to a persistent while-loop over tiles while leaving the launch grid unchanged caused executed instructions to explode from 28,864 to 31,247,360 (~1083x), ballooning duration from 2.46 µs to 170.59 µs (~69x regression). The root cause is that the grid remained (cdiv(xnumel,128), cdiv(ynumel,8)) = (1, 256) instead of being changed to a 1-D persistent grid of ~132 CTAs, so each of the 256 CTAs iterated over all tiles redundantly, processing the entire workload 256 times. Additionally, the while-loop overhead doubled registers-per-thread (16→32) and reintroduced 4 KB dynamic shared memory, further reducing occupancy headroom. The persistent tile pattern is only valid when the grid launch site is atomically updated to the persistent worker count (e.g., min(SM_COUNT, num_tiles)).

> **Applicable when**: Never apply this broken pattern. The persistent tile loop strategy is valid in principle only when the grid launch site is simultaneously updated to SM_COUNT CTAs.

> **Avoid when**: Any time the kernel body is converted to a persistent while-loop but the grid launch site is left unchanged at the original tile-covering size. This causes each CTA to redundantly process the entire workload, multiplying total instructions by the original grid size. Also avoid when the kernel is already very short (<5 µs) and the persistent loop overhead (register pressure increase, barrier stalls from CPIStall at 44% of cycles) would dominate over any occupancy gain.

---

## Iteration 3

**Best so far**: 1.0802× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_lambda_grid_and_add_evict_first_store_for_streaming_transpose | 1.059× | 1.067× | ✓ | **✓ best** |
| C | fix_persistent_kernel_grid_and_right_size_xblock | 0.044× | 0.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_lambda_grid_a... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 2400 ns | -160 ns (-6.2%) | 170590 ns | +168030 ns (+6563.7%) |
| SM Throughput (%) | 4.87 % | 1.72 % | -3.15 pp | 23.04 % | +18.17 pp |
| Memory Throughput (%) | 5.30 % | 1.15 % | -4.15 pp | 25.21 % | +19.91 pp |
| DRAM Throughput (%) | 1.70 % | 1.14 % | -0.56 pp | 0.03 % | -1.67 pp |
| Achieved Occupancy (%) | 10.94 % | 11.14 % | +0.20 pp | 12.12 % | +1.18 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 85.20 % | 88.25 % | +3.05 pp | 76.04 % | -9.16 pp |
| IPC Active | 0.54 inst/cycle | 0.25 inst/cycle | -0.29 | 0.97 inst/cycle | +0.43 |

---

## Iteration 4

**Best so far**: 1.0802× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_add_num_stages_pipelining | 1.020× | 1.080× | ✓ |  |
| B | shared_memory_coalesced_transpose_to_eliminate_uncoalesced_store | 1.016× | 1.041× | ✓ |  |
| C | revert_broken_persistent_loop_restore_autotuned_simple_kernel_with_evict_first_store | 1.019× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | shared_memory_coa... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 2370 ns | -190 ns (-7.4%) | 2460 ns | -100 ns (-3.9%) | 2620 ns | +60 ns (+2.3%) |
| SM Throughput (%) | 4.87 % | 1.56 % | -3.31 pp | 0.84 % | -4.03 pp | 1.70 % | -3.17 pp |
| Memory Throughput (%) | 5.30 % | 1.28 % | -4.02 pp | 1.25 % | -4.05 pp | 1.12 % | -4.18 pp |
| DRAM Throughput (%) | 1.70 % | 1.17 % | -0.53 pp | 1.13 % | -0.57 pp | 1.05 % | -0.65 pp |
| Achieved Occupancy (%) | 10.94 % | 10.33 % | -0.61 pp | 4.78 % | -6.16 pp | 10.86 % | -0.08 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.06 | -0.06 | 0.12 | +0.00 |
| No Eligible (%) | 85.20 % | 88.18 % | +2.98 pp | 95.72 % | +10.52 pp | 92.40 % | +7.20 pp |
| IPC Active | 0.54 inst/cycle | 0.38 inst/cycle | -0.16 | 0.15 inst/cycle | -0.39 | 0.31 inst/cycle | -0.23 |

---

## Iteration 5

**Best so far**: 1.0802× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_large_block_configs_for_wave_coverage | 1.049× | 0.977× | ✓ |  |
| B | fix_hardcoded_grid_with_safe_lambda_and_xnumel25_configs | 1.066× | 1.012× | ✓ | **✓ best** |
| C | fix_hardcoded_grid_lambda_correct_formula | 1.066× | 1.012× | ✓ |  |
| D | add_triton_autotune | 1.020× | 1.024× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2560 ns | 2620 ns | +60 ns (+2.3%) | 2530 ns | -30 ns (-1.2%) | 2530 ns | -30 ns (-1.2%) | 2500 ns | -60 ns (-2.3%) |
| SM Throughput (%) | 4.87 % | 5.98 % | +1.11 pp | 0.88 % | -3.99 pp | 0.87 % | -4.00 pp | 0.77 % | -4.10 pp |
| Memory Throughput (%) | 5.30 % | 5.51 % | +0.21 pp | 1.15 % | -4.15 pp | 1.21 % | -4.09 pp | 1.15 % | -4.15 pp |
| DRAM Throughput (%) | 1.70 % | 1.64 % | -0.06 pp | 1.10 % | -0.60 pp | 1.10 % | -0.60 pp | 1.10 % | -0.60 pp |
| Achieved Occupancy (%) | 10.94 % | 10.36 % | -0.58 pp | 4.86 % | -6.08 pp | 4.66 % | -6.28 pp | 4.64 % | -6.30 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.06 | -0.06 | 0.06 | -0.06 | 0.06 | -0.06 |
| No Eligible (%) | 85.20 % | 79.96 % | -5.24 pp | 93.87 % | +8.67 pp | 94.09 % | +8.89 pp | 95.92 % | +10.72 pp |
| IPC Active | 0.54 inst/cycle | 0.85 inst/cycle | +0.31 | 0.12 inst/cycle | -0.42 | 0.14 inst/cycle | -0.40 | 0.14 inst/cycle | -0.40 |

---

## Cumulative Progress

```
Baseline:  2560 ns  (1.000×)
Iter 1     2370 ns  (1.080×)  — autotune_tile_sizes_for_small_kernel

Final best: 2530 ns  (1.0119×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,44 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK':  2, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK':  2, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  32}, num_warps=2),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK':  8, 'YBLOCK':  32}, num_warps=4),
+        triton.Config({'XBLOCK':  8, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  32}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  64}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 2048
-    xnumel = 25
-    yoffset = tl.program_id(1) * YBLOCK
+def triton_poi_fused_convolution_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
+    pid_x = tl.program_id(0)
+    pid_y = tl.program_id(1)
+    xoffset = pid_x * XBLOCK
+    yoffset = pid_y * YBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
+    ymask = yindex < ynumel
+    mask = xmask & ymask
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 32)
+    y0 = yindex % 32
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 25 * y3), mask=mask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 32 * x2 + 800 * y1), tmp0, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +50,7 @@

     ynumel = 2048
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
