# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5820 ns — **Best latency**: 4000 ns (**1.4550×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_069/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5820 ns
- **SM Throughput**: 16.04 % | **Memory**: 42.50 % | **DRAM**: 21.11 %
- **Achieved Occupancy**: 55.20 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.127× | 0.905× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.163× | 1.282× | ✓ |  |
| D | autotune_tile_sizes_for_coalescing_and_parallelism | 1.369× | 1.455× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 6430 ns | +610 ns (+10.5%) | 4540 ns | -1280 ns (-22.0%) | 4000 ns | -1820 ns (-31.3%) |
| SM Throughput (%) | 16.04 % | 24.95 % | +8.91 pp | 14.43 % | -1.61 pp | 18.81 % | +2.77 pp |
| Memory Throughput (%) | 42.50 % | 39.20 % | -3.30 pp | 30.11 % | -12.39 pp | 21.75 % | -20.75 pp |
| DRAM Throughput (%) | 21.11 % | 19.17 % | -1.94 pp | 14.53 % | -6.58 pp | 4.18 % | -16.93 pp |
| Achieved Occupancy (%) | 55.20 % | 48.58 % | -6.62 pp | 23.52 % | -31.68 pp | 48.58 % | -6.62 pp |
| Registers/Thread | 26.00 register/thread | 32.00 register/thread | +6 | 34.00 register/thread | +8 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.49 | -0.25 | 0.74 | +0.00 |
| No Eligible (%) | 70.54 % | 55.45 % | -15.09 pp | 68.13 % | -2.41 pp | 52.04 % | -18.50 pp |
| IPC Active | 1.13 inst/cycle | 1.76 inst/cycle | +0.63 | 1.10 inst/cycle | -0.03 | 1.87 inst/cycle | +0.74 |

---

## Iteration 2

**Best so far**: 1.455× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_alignment_hints | 1.324× | 1.477× | ✓ |  |
| B | vectorized_loads | 1.161× | 1.181× | ✓ |  |
| C | streaming_eviction_policy_and_fix_autotune_grid | 1.287× | 1.409× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads | Δ | streaming_evictio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 3940 ns | -1880 ns (-32.3%) | 4930 ns | -890 ns (-15.3%) | 4130 ns | -1690 ns (-29.0%) |
| SM Throughput (%) | 16.04 % | 21.51 % | +5.47 pp | 17.21 % | +1.17 pp | 20.14 % | +4.10 pp |
| Memory Throughput (%) | 42.50 % | 22.27 % | -20.23 pp | 27.88 % | -14.62 pp | 23.17 % | -19.33 pp |
| DRAM Throughput (%) | 21.11 % | 4.21 % | -16.90 pp | 13.37 % | -7.74 pp | 3.99 % | -17.12 pp |
| Achieved Occupancy (%) | 55.20 % | 46.96 % | -8.24 pp | 21.46 % | -33.74 pp | 49.17 % | -6.03 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 32.00 register/thread | +6 | 25.00 register/thread | -1 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 |
| No Eligible (%) | 70.54 % | 46.48 % | -24.06 pp | 62.02 % | -8.52 pp | 53.69 % | -16.85 pp |
| IPC Active | 1.13 inst/cycle | 2.09 inst/cycle | +0.96 | 1.43 inst/cycle | +0.30 | 1.86 inst/cycle | +0.73 |

---

## Iteration 3

**Best so far**: 1.455× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_to_use_meta_lambda | 1.354× | 1.444× | ✓ |  |
| B | fix_autotune_grid_lambda_and_expand_yblock_configs | 1.296× | 1.357× | ✓ |  |
| C | fix_autotune_grid_lambda_to_restore_occupancy | 1.241× | 1.317× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5820 ns | 4030 ns | -1790 ns (-30.8%) | 4290 ns | -1530 ns (-26.3%) | 4420 ns | -1400 ns (-24.1%) |
| SM Throughput (%) | 16.04 % | 21.86 % | +5.82 pp | 19.69 % | +3.65 pp | 17.04 % | +1.00 pp |
| Memory Throughput (%) | 42.50 % | 22.61 % | -19.89 pp | 22.75 % | -19.75 pp | 30.81 % | -11.69 pp |
| DRAM Throughput (%) | 21.11 % | 4.11 % | -17.00 pp | 3.85 % | -17.26 pp | 14.92 % | -6.19 pp |
| Achieved Occupancy (%) | 55.20 % | 46.64 % | -8.56 pp | 50.92 % | -4.28 pp | 21.29 % | -33.91 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 25.00 register/thread | -1 | 32.00 register/thread | +6 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.37 | -0.37 |
| No Eligible (%) | 70.54 % | 46.20 % | -24.34 pp | 51.61 % | -18.93 pp | 61.16 % | -9.38 pp |
| IPC Active | 1.13 inst/cycle | 2.07 inst/cycle | +0.94 | 1.67 inst/cycle | +0.54 | 1.45 inst/cycle | +0.32 |


### Key Learnings

#### ✅ Plan A: `fix_autotune_grid_to_use_meta_lambda` (1.444×)

When a Triton kernel already uses @triton.autotune with a META-driven grid lambda, adding additional configs with num_warps=8 (e.g., XBLOCK=1024/YBLOCK=4, XBLOCK=512/YBLOCK=8, XBLOCK=256/YBLOCK=16) can yield modest improvements by allowing the autotuner to find better warp-level parallelism for the given problem shape. In this case, the NCU per-kernel metrics were essentially unchanged (Duration 3.94→4.03µs, grid size fixed at 1568 CTAs, waves_per_sm=0.74), suggesting the autotuner selected a config similar to the previous best, but the wall-clock speedup of 1.44x indicates the benefit came from the autotuner finding a marginally better config across the expanded search space. The strategy is incremental when the grid structure is already correct and the bottleneck (uncoalesced stores, low wave count) is structural rather than config-driven.

> **Applicable when**: Triton kernel already uses @triton.autotune with a META-driven grid lambda; autotune config space only covers num_warps=4 for large tile sizes (XBLOCK>=512, YBLOCK>=4); IssueSlotUtilization fires with no-eligible-warp cycles >40%; kernel has a 2D tile structure with one small dimension (ynumel small) and one large dimension (xnumel large); waves_per_sm < 1.0.

> **Avoid when**: The autotune grid is still hardcoded (not using META lambda) — fix that first; the kernel is already compute- or bandwidth-bound rather than warp-latency-bound; num_warps=8 configs would exceed register file limits and reduce occupancy below the benefit threshold; the problem is so small that autotuning overhead dominates.

#### ✅ Plan C: `fix_autotune_grid_lambda_to_restore_occupancy` (1.317×)

When a Triton kernel's launch site uses a hardcoded grid (e.g., `cdiv(xnumel, 128), cdiv(ynumel, 8)`) instead of a META-driven lambda, the autotuner selects a winning config but the grid is always computed with the hardcoded tile sizes, causing a mismatch. In this case the autotuner selected XBLOCK=64 (smaller than the hardcoded 128), halving the thread count and cutting waves_per_sm from 0.74 to 0.37 and achieved occupancy from 47% to 21%, which more than offset per-warp latency improvements. The fix is to replace the hardcoded grid with `lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))` so the launch configuration always matches the autotuner-selected tile sizes.

> **Applicable when**: Triton kernel has @triton.autotune but the launch site uses a hardcoded grid tuple instead of a META lambda; profiler shows waves_per_sm and achieved occupancy much lower than expected given the grid size; the autotuner is effectively disabled because the grid does not adapt to the selected config.

> **Avoid when**: The grid is already a META lambda and correctly reflects the autotuner config; the kernel does not use @triton.autotune; the hardcoded tile sizes in the grid intentionally differ from XBLOCK/YBLOCK for correctness reasons (e.g., partial tiles handled separately).

---

## Cumulative Progress

```
Baseline:  5820 ns  (1.000×)
Iter 1     4000 ns  (1.455×)  — autotune_tile_sizes_for_coalescing_and_parallelism

Final best: 4000 ns  (1.4550×)
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
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 2}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -14,6 +30,7 @@

     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    xindex = tl.max_contiguous(tl.multiple_of(xindex, [1, 16]), [1, XBLOCK])  # alignment/contiguity hints for LDG.128
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +39,13 @@

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
