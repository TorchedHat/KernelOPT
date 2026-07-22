# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 3140 ns (**1.1210×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`remove_in_body_constexpr_promotion_and_restore_clean_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_6_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3520 ns
- **SM Throughput**: 21.92 % | **Memory**: 20.77 % | **DRAM**: 2.59 %
- **Achieved Occupancy**: 49.51 %
- **IPC Active**: 1.86 inst/cycle | **No Eligible**: 47.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_overprovisioned_xblock_with_autotune | 1.125× | 1.197× | ✓ | **✓ best** |
| C | fix_tile_sizes_for_small_xdim_and_add_autotune | 1.067× | 1.100× | ✓ |  |
| D | fix_tile_sizes_for_coalesced_vectorized_access | 1.106× | 1.029× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_overprovision... | Δ | fix_tile_sizes_fo... | Δ | fix_tile_sizes_fo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2940 ns | -580 ns (-16.5%) | 3200 ns | -320 ns (-9.1%) | 3420 ns | -100 ns (-2.8%) |
| SM Throughput (%) | 21.92 % | 12.87 % | -9.05 pp | 14.79 % | -7.13 pp | 31.13 % | +9.21 pp |
| Memory Throughput (%) | 20.77 % | 5.74 % | -15.03 pp | 13.82 % | -6.95 pp | 21.19 % | +0.42 pp |
| DRAM Throughput (%) | 2.59 % | 3.11 % | +0.52 pp | 2.88 % | +0.29 pp | 2.67 % | +0.08 pp |
| Achieved Occupancy (%) | 49.51 % | 46.58 % | -2.93 pp | 26.65 % | -22.86 pp | 49.92 % | +0.41 pp |
| Registers/Thread | 28.00 register/thread | 18.00 register/thread | -10 | 30.00 register/thread | +2 | 27.00 register/thread | -1 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.36 | -0.37 | 0.73 | +0.00 |
| No Eligible (%) | 47.12 % | 51.31 % | +4.19 pp | 49.63 % | +2.51 pp | 32.46 % | -14.66 pp |
| IPC Active | 1.86 inst/cycle | 1.78 inst/cycle | -0.08 | 1.83 inst/cycle | -0.03 | 2.61 inst/cycle | +0.75 |

---

## Iteration 2

**Best so far**: 1.1973× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_constexpr_scalar_promotion | 0.004× | 0.002× | ✓ |  |
| B | persistent_tile_scheduling | 0.004× | 0.002× | ✓ |  |
| C | persistent_tile_scheduling_with_correct_autotune_launch | 1.111× | 0.867× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 1470000 ns | +1466480 ns (+41661.4%) | 1840000 ns | +1836480 ns (+52172.7%) | 4060 ns | +540 ns (+15.3%) |
| SM Throughput (%) | 21.92 % | 63.65 % | +41.73 pp | 76.52 % | +54.60 pp | 31.42 % | +9.50 pp |
| Memory Throughput (%) | 20.77 % | 75.93 % | +55.16 pp | 75.05 % | +54.28 pp | 15.53 % | -5.24 pp |
| DRAM Throughput (%) | 2.59 % | 0.01 % | -2.58 pp | 0.01 % | -2.58 pp | 2.25 % | -0.34 pp |
| Achieved Occupancy (%) | 49.51 % | 66.26 % | +16.75 pp | 57.82 % | +8.31 pp | 70.52 % | +21.01 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 1.45 | +0.72 |
| No Eligible (%) | 47.12 % | 30.40 % | -16.72 pp | 19.31 % | -27.81 pp | 25.61 % | -21.51 pp |
| IPC Active | 1.86 inst/cycle | 2.69 inst/cycle | +0.83 | 3.22 inst/cycle | +1.36 | 2.87 inst/cycle | +1.01 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling` (0.009×)

Replacing a 2D-tiled Triton kernel with a 1D-flattened persistent while-loop (XBLOCK=1 configs, YBLOCK as effective block size) caused a ~132x increase in executed instructions (1.0M → 132.9M) and a ~96,000x increase in kernel duration (4.06 µs → 391.6 ms). The root cause is that the grid remained at (1536,) — the original Y-dimension tile count — while the persistent loop used tl.num_programs(0)=1536 as the stride, but the total tile count (cdiv(110592, BLOCK_SIZE)) was much larger than 1536 for small BLOCK_SIZE configs, causing each of the 1536 CTAs to iterate over many tiles sequentially. Additionally, L1 hit rate jumped to 97% (data fits in L1 due to repeated access of the same small working set per CTA), confirming that each CTA was redundantly re-processing tiles. The persistent while-loop pattern requires the grid to be set to exactly (num_sms,) = (132,), not the original tile-decomposed grid size. Using the original grid size as both the CTA count and the loop stride causes O(total_tiles / grid_size) iterations per CTA instead of O(total_tiles / num_sms).

> **Applicable when**: Never apply this specific anti-pattern. The persistent kernel pattern itself is valid only when the grid is set to exactly (num_sms,) and the loop stride equals num_sms.

> **Avoid when**: Do NOT add a persistent while-loop to a kernel without simultaneously changing the launch grid to a 1D grid of exactly num_sms CTAs. Do NOT keep the original tile-decomposed grid size (e.g., 1536) as the grid when adding a persistent loop — this causes each CTA to process O(total_tiles/grid_size) tiles sequentially instead of distributing work across SMs. Do NOT use tl.num_programs(0) as the tile stride when the grid is not set to num_sms. Do NOT apply persistent scheduling to kernels with duration < 10 µs.

---

## Iteration 3

**Best so far**: 1.1973× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_autotune_meta_lambda | 1.104× | 0.859× | ✓ |  |
| B | fix_broken_persistent_loop_revert_to_simple_2d_tiled_with_right_sized_xblock | 1.159× | 1.121× | ✓ | **✓ best** |
| C | flatten_to_1d_tiled_kernel_with_autotune | 0.020× | 0.009× | ✓ |  |
| D | add_triton_autotune | 1.148× | 0.887× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_broken_persis... | Δ | flatten_to_1d_til... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 4100 ns | +580 ns (+16.5%) | 3140 ns | -380 ns (-10.8%) | 391620 ns | +388100 ns (+11025.6%) | 3970 ns | +450 ns (+12.8%) |
| SM Throughput (%) | 21.92 % | 30.40 % | +8.48 pp | 15.65 % | -6.27 pp | 42.86 % | +20.94 pp | 32.29 % | +10.37 pp |
| Memory Throughput (%) | 20.77 % | 15.02 % | -5.75 pp | 13.73 % | -7.04 pp | 69.57 % | +48.80 pp | 15.94 % | -4.83 pp |
| DRAM Throughput (%) | 2.59 % | 2.25 % | -0.34 pp | 2.95 % | +0.36 pp | 0.03 % | -2.56 pp | 2.32 % | -0.27 pp |
| Achieved Occupancy (%) | 49.51 % | 70.47 % | +20.96 pp | 27.97 % | -21.54 pp | 70.97 % | +21.46 pp | 70.38 % | +20.87 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 30.00 register/thread | +2 | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.73 | 1.45 | +0.72 | 0.36 | -0.37 | 0.73 | +0.00 | 1.45 | +0.72 |
| No Eligible (%) | 47.12 % | 25.27 % | -21.85 pp | 48.24 % | +1.12 pp | 57.33 % | +10.21 pp | 25.95 % | -21.17 pp |
| IPC Active | 1.86 inst/cycle | 2.87 inst/cycle | +1.01 | 1.81 inst/cycle | -0.05 | 1.69 inst/cycle | -0.17 | 2.86 inst/cycle | +1.00 |


### Key Learnings

#### ❌ Plan A: `fix_hardcoded_grid_to_use_autotune_meta_lambda` (0.859×)

Replacing a hardcoded grid tuple with a lambda (grid = lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK']))) was expected to allow the autotuner to select large-YBLOCK configs and reduce the CTA count from 1536 to ~6, eliminating the partial wave. However, the profiling shows zero change: grid_size remained 1536, waves_per_sm stayed at 1.45, duration increased slightly from 4.06 to 4.10 µs (0.86x regression). The diff was empty, indicating the code change either was not applied or the autotuner had already cached a config that reproduced the same (1, 1536) grid. When the autotune cache already contains a winning config that matches the original hardcoded grid dimensions, switching to a lambda grid produces no observable benefit and may introduce minor overhead.

> **Applicable when**: Never: this specific change (hardcoded grid → lambda) produced a regression and zero metric improvement. The autotune cache or config set caused the same grid to be selected.

> **Avoid when**: Do NOT apply this change alone when the autotune config set already includes the original tile sizes as a candidate — the autotuner may select the same config, leaving grid_size, waves_per_sm, and duration unchanged while adding lambda overhead. Do NOT apply when the diff is empty (no actual code change landed). Do NOT apply when waves_per_sm > 1.0 and the partial wave is the primary bottleneck but the autotune configs do not include sufficiently large YBLOCK values to actually reduce the grid.

---

## Iteration 4

**Best so far**: 1.1973× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_config_space_for_better_wave_coverage | 1.109× | 1.248× | ✓ |  |
| C | shared_memory_staged_coalesced_store_for_strided_transpose | 1.104× | 1.222× | ✓ |  |
| D | fix_hardcoded_grid_lambda_and_add_eviction_policy_for_streaming_transpose | 1.153× | 0.335× | ✓ |  |
| E | add_triton_autotune | 1.111× | 1.110× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | shared_memory_sta... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2820 ns | -700 ns (-19.9%) | 2880 ns | -640 ns (-18.2%) | 10500 ns | +6980 ns (+198.3%) | 3170 ns | -350 ns (-9.9%) |
| SM Throughput (%) | 21.92 % | 4.96 % | -16.96 pp | 7.63 % | -14.29 pp | 63.55 % | +41.63 pp | 15.33 % | -6.59 pp |
| Memory Throughput (%) | 20.77 % | 3.26 % | -17.51 pp | 3.21 % | -17.56 pp | 61.06 % | +40.29 pp | 13.43 % | -7.34 pp |
| DRAM Throughput (%) | 2.59 % | 3.26 % | +0.67 pp | 3.21 % | +0.62 pp | 0.88 % | -1.71 pp | 2.91 % | +0.32 pp |
| Achieved Occupancy (%) | 49.51 % | 19.67 % | -29.84 pp | 34.55 % | -14.96 pp | 60.98 % | +11.47 pp | 26.78 % | -22.73 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 40.00 register/thread | +12 | 30.00 register/thread | +2 |
| Waves/SM | 0.73 | 0.36 | -0.37 | 0.73 | +0.00 | 1.94 | +1.21 | 0.36 | -0.37 |
| No Eligible (%) | 47.12 % | 80.43 % | +33.31 pp | 71.99 % | +24.87 pp | 19.75 % | -27.37 pp | 51.71 % | +4.59 pp |
| IPC Active | 1.86 inst/cycle | 0.67 inst/cycle | -1.19 | 0.95 inst/cycle | -0.91 | 3.19 inst/cycle | +1.33 | 1.89 inst/cycle | +0.03 |

---

## Iteration 5

**Best so far**: 1.1973× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_launch_grid_meta_lambda_and_add_small_warp_configs | 1.101× | 1.248× | ✓ |  |
| C | remove_in_body_constexpr_promotion_and_restore_clean_autotune | 1.156× | 1.121× | ✓ |  |
| D | streaming_load_eviction_policy_with_num_stages_pipelining | 1.129× | 1.210× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_m... | Δ | remove_in_body_co... | Δ | streaming_load_ev... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2820 ns | -700 ns (-19.9%) | 3140 ns | -380 ns (-10.8%) | 2910 ns | -610 ns (-17.3%) |
| SM Throughput (%) | 21.92 % | 4.29 % | -17.63 pp | 15.83 % | -6.09 pp | 8.44 % | -13.48 pp |
| Memory Throughput (%) | 20.77 % | 3.26 % | -17.51 pp | 13.46 % | -7.31 pp | 3.14 % | -17.63 pp |
| DRAM Throughput (%) | 2.59 % | 3.26 % | +0.67 pp | 2.93 % | +0.34 pp | 3.14 % | +0.55 pp |
| Achieved Occupancy (%) | 49.51 % | 18.53 % | -30.98 pp | 26.47 % | -23.04 pp | 36.90 % | -12.61 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 30.00 register/thread | +2 | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.36 | -0.37 | 0.36 | -0.37 | 0.73 | +0.00 |
| No Eligible (%) | 47.12 % | 82.24 % | +35.12 pp | 45.12 % | -2.00 pp | 63.30 % | +16.18 pp |
| IPC Active | 1.86 inst/cycle | 0.36 inst/cycle | -1.50 | 1.95 inst/cycle | +0.09 | 1.17 inst/cycle | -0.69 |

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 1     2940 ns  (1.197×)  — fix_overprovisioned_xblock_with_autotune

Final best: 3140 ns  (1.1210×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,23 +5,54 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Small YBLOCK + XBLOCK=1: maximize CTA count, fill all SMs
+        # XBLOCK=1,YBLOCK=16 -> grid=(9,768)=6912 CTAs=52.4 waves
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 16},  num_warps=2),
+        # XBLOCK=1,YBLOCK=32 -> grid=(9,384)=3456 CTAs=26.2 waves
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        # XBLOCK=16,YBLOCK=16 -> grid=(1,768)=768 CTAs=5.8 waves
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=2),
+        # XBLOCK=16,YBLOCK=32 -> grid=(1,384)=384 CTAs=2.9 waves (current best)
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        # Medium YBLOCK configs (unchanged from before)
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
+        # Large YBLOCK configs (unchanged from before)
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+        # NEW: very large YBLOCK with num_warps=16 for high per-CTA occupancy
+        # XBLOCK=1,YBLOCK=1024 -> grid=(9,12)=108 CTAs, 16 warps=512 threads/CTA
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 1024}, num_warps=16),
+        # XBLOCK=1,YBLOCK=2048 -> grid=(9,6)=54 CTAs, 16 warps
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 2048}, num_warps=16),
+        # XBLOCK=1,YBLOCK=4096 -> grid=(9,3)=27 CTAs, 16 warps
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 4096}, num_warps=16),
+        # Pipelined: num_stages=2 to overlap memory latency on medium tile
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 12288
-    xnumel = 9
+def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
     yoffset = tl.program_id(1) * YBLOCK
+    xoffset = tl.program_id(0) * XBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
+    y0 = yindex % 64
+    y1 = yindex // 64
     y3 = yindex
-    y0 = (yindex % 64)
-    y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64 * x2 + 576 * y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -29,7 +60,7 @@

     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
