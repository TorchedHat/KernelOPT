# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_gelu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2500 ns — **Best latency**: 2430 ns (**1.0288×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`true_single_block_full_tile_xblock4096_num_warps16`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_gelu_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_14_triton_poi_fused_addmm_gelu_14/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.02
- **Registers/thread**: 16
- **Duration**: 2500 ns
- **SM Throughput**: 0.41 % | **Memory**: 0.59 % | **DRAM**: 0.23 %
- **Achieved Occupancy**: 6.12 %
- **IPC Active**: 0.24 inst/cycle | **No Eligible**: 93.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 1.018× | 0.471× | ✓ | **✓ best** |
| B | autotune_xblock_increase_grid_coverage | 1.008× | 0.988× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_tiny_pointwise | 0.995× | 0.940× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_i... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2500 ns | 5310 ns | +2810 ns (+112.4%) | 2530 ns | +30 ns (+1.2%) | 2660 ns | +160 ns (+6.4%) |
| SM Throughput (%) | 0.41 % | 0.13 % | -0.28 pp | 0.12 % | -0.29 pp | 0.24 % | -0.17 pp |
| Memory Throughput (%) | 0.59 % | 0.47 % | -0.12 pp | 0.50 % | -0.09 pp | 0.63 % | +0.04 pp |
| DRAM Throughput (%) | 0.23 % | 0.18 % | -0.05 pp | 0.09 % | -0.14 pp | 0.23 % | +0.00 pp |
| Achieved Occupancy (%) | 6.12 % | 6.21 % | +0.09 pp | 1.56 % | -4.56 pp | 1.56 % | -4.56 pp |
| Registers/Thread | 16.00 register/thread | 80.00 register/thread | +64 | 16.00 register/thread | +0 | 27.00 register/thread | +11 |
| Waves/SM | 0.02 | 0.04 | +0.02 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 93.27 % | 94.68 % | +1.41 pp | 92.53 % | -0.74 pp | 88.24 % | -5.03 pp |
| IPC Active | 0.24 inst/cycle | 0.18 inst/cycle | -0.06 | 0.06 inst/cycle | -0.18 | 0.12 inst/cycle | -0.12 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune` (0.471×)

Adding a while-loop based persistent tile scheduling pattern inside a Triton kernel caused registers per thread to explode from 16 to 80, reducing theoretical occupancy from 100% to 37.5% and doubling kernel duration (2.5us → 5.31us). The register pressure increase was caused by the compiler needing to keep the loop induction variable (tile_id) and all intermediate values live across loop iterations, preventing the compiler from reusing registers. For tiny workloads where the entire dataset fits in a single large tile (e.g., XBLOCK=4096 covers all 4096 elements), a persistent loop is unnecessary overhead — a single-tile kernel with a large XBLOCK is strictly better.

> **Applicable when**: Never beneficial for tiny workloads where XBLOCK can cover all elements in one tile. Persistent loops are only useful when the grid is intentionally smaller than the number of tiles (i.e., num_tiles >> num_SMs) and register pressure is not a concern.

> **Avoid when**: Workload is small enough that a single large XBLOCK (e.g., XBLOCK=4096 for 4096 elements) covers all data in one tile per block. Adding a while-loop in this case adds register pressure without reducing grid launches, causing theoretical occupancy to drop and duration to increase. Also avoid when registers per thread are already near the SM limit (e.g., >32 registers with 128-thread blocks on H200).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | single_large_tile_max_warps_no_loop | 1.051× | 0.988× | ✓ | **✓ best** |
| B | fix_broken_autotune_grid_formula | 0.997× | 1.029× | ✓ |  |
| C | remove_persistent_loop_restore_single_tile_kernel | 0.997× | 1.029× | ✓ |  |


### Metric Comparison

| Metric | Baseline | single_large_tile... | Δ | fix_broken_autotu... | Δ | remove_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2500 ns | 2530 ns | +30 ns (+1.2%) | 2430 ns | -70 ns (-2.8%) | 2430 ns | -70 ns (-2.8%) |
| SM Throughput (%) | 0.41 % | 0.11 % | -0.30 pp | 0.34 % | -0.07 pp | 0.46 % | +0.05 pp |
| Memory Throughput (%) | 0.59 % | 0.44 % | -0.15 pp | 0.59 % | +0.00 pp | 0.58 % | -0.01 pp |
| DRAM Throughput (%) | 0.23 % | 0.10 % | -0.13 pp | 0.24 % | +0.01 pp | 0.24 % | +0.01 pp |
| Achieved Occupancy (%) | 6.12 % | 1.56 % | -4.56 pp | 3.10 % | -3.02 pp | 6.16 % | +0.04 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 19.00 register/thread | +3 | 16.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.02 | +0.00 |
| No Eligible (%) | 93.27 % | 92.52 % | -0.75 pp | 91.33 % | -1.94 pp | 93.10 % | -0.17 pp |
| IPC Active | 0.24 inst/cycle | 0.06 inst/cycle | -0.18 | 0.20 inst/cycle | -0.04 | 0.19 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_respect_autotuned_xblock | 1.034× | 1.042× | ✓ |  |
| B | single_block_full_tile_max_warps_gelu_pointwise | 1.005× | 1.087× | ✓ |  |
| C | fix_grid_formula_to_respect_autotune_xblock_single_block | 0.998× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | single_block_full... | Δ | fix_grid_formula_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2500 ns | 2400 ns | -100 ns (-4.0%) | 2300 ns | -200 ns (-8.0%) | 2560 ns | +60 ns (+2.4%) |
| SM Throughput (%) | 0.41 % | 0.32 % | -0.09 pp | 0.40 % | -0.01 pp | 1.18 % | +0.77 pp |
| Memory Throughput (%) | 0.59 % | 0.61 % | +0.02 pp | 0.60 % | +0.01 pp | 0.58 % | -0.01 pp |
| DRAM Throughput (%) | 0.23 % | 0.25 % | +0.02 pp | 0.25 % | +0.02 pp | 0.23 % | +0.00 pp |
| Achieved Occupancy (%) | 6.12 % | 3.05 % | -3.07 pp | 5.86 % | -0.26 pp | 11.49 % | +5.37 pp |
| Registers/Thread | 16.00 register/thread | 19.00 register/thread | +3 | 16.00 register/thread | +0 | 19.00 register/thread | +3 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.02 | +0.00 | 0.03 | +0.01 |
| No Eligible (%) | 93.27 % | 89.79 % | -3.48 pp | 94.60 % | +1.33 pp | 71.16 % | -22.11 pp |
| IPC Active | 0.24 inst/cycle | 0.16 inst/cycle | -0.08 | 0.22 inst/cycle | -0.02 | 1.03 inst/cycle | +0.79 |


### Key Learnings

#### ✅ Plan A: `fix_hardcoded_grid_to_respect_autotuned_xblock` (1.042×)

When a Triton kernel wrapper hard-codes the grid divisor (e.g., `cdiv(xnumel, 128)`) instead of using a lambda that reads `meta['XBLOCK']`, the autotuner cannot correctly evaluate larger XBLOCK configs because the grid size never shrinks to match. Replacing the static grid tuple with `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` allows the autotuner to explore all tile sizes correctly. In this case the fix yielded a modest 1.04x speedup (duration 2.43→2.40 µs, elapsed cycles 3626→3504) with active warps per scheduler improving from 0.83 to 1.25, suggesting the autotuner found a slightly better config, though the workload remains severely launch-bound (grid=32 vs 132 SMs) so the fundamental bottleneck persists.

> **Applicable when**: A Triton kernel wrapper uses a static grid tuple with a hardcoded tile size constant instead of a lambda that reads meta['XBLOCK']. The @triton.autotune decorator is present with multiple XBLOCK configs. NCU reports LaunchConfiguration and AchievedOccupancy warnings with grid_size << sm_count.

> **Avoid when**: The grid is already a lambda reading meta['XBLOCK']. The kernel has no @triton.autotune decorator. The workload is large enough that even the smallest XBLOCK config fills all SMs (waves_per_sm >= 1.0). Note: this fix alone cannot overcome a fundamentally undersized workload — if xnumel is too small to fill all SMs at any XBLOCK, the speedup will be marginal.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | true_single_block_full_tile_xblock4096_num_warps16 | 1.086× | 1.029× | ✓ | **✓ best** |
| B | true_single_block_collapse_xblock4096_no_mask_no_autotune | 1.044× | 1.055× | ✓ |  |
| C | true_single_block_collapse_xblock4096_no_mask_no_autotune | 1.002× | 1.087× | ✓ |  |


### Metric Comparison

| Metric | Baseline | true_single_block... | Δ | true_single_block... | Δ | true_single_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2500 ns | 2430 ns | -70 ns (-2.8%) | 2370 ns | -130 ns (-5.2%) | 2300 ns | -200 ns (-8.0%) |
| SM Throughput (%) | 0.41 % | 0.38 % | -0.03 pp | 0.43 % | +0.02 pp | 0.44 % | +0.03 pp |
| Memory Throughput (%) | 0.59 % | 0.52 % | -0.07 pp | 0.63 % | +0.04 pp | 0.69 % | +0.10 pp |
| DRAM Throughput (%) | 0.23 % | 0.04 % | -0.19 pp | 0.24 % | +0.01 pp | 0.25 % | +0.02 pp |
| Achieved Occupancy (%) | 6.12 % | 6.19 % | +0.07 pp | 6.16 % | +0.04 pp | 5.89 % | -0.23 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 93.27 % | 95.56 % | +2.29 pp | 93.33 % | +0.06 pp | 93.21 % | -0.06 pp |
| IPC Active | 0.24 inst/cycle | 0.22 inst/cycle | -0.02 | 0.18 inst/cycle | -0.06 | 0.22 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0288× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hardcoded_single_block_constexpr_no_xnumel_param | 1.044× | 1.087× | ✓ |  |
| B | true_single_block_xblock4096_grid1_num_warps32 | 1.001× | 0.988× | ✓ |  |
| C | increase_xblock_and_num_warps_to_hide_l1tex_latency | 0.998× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hardcoded_single_... | Δ | true_single_block... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2500 ns | 2300 ns | -200 ns (-8.0%) | 2530 ns | +30 ns (+1.2%) | 2460 ns | -40 ns (-1.6%) |
| SM Throughput (%) | 0.41 % | 0.39 % | -0.02 pp | 0.41 % | +0.00 pp | 0.42 % | +0.01 pp |
| Memory Throughput (%) | 0.59 % | 0.57 % | -0.02 pp | 0.65 % | +0.06 pp | 0.59 % | +0.00 pp |
| DRAM Throughput (%) | 0.23 % | 0.04 % | -0.19 pp | 0.23 % | +0.00 pp | 0.23 % | +0.00 pp |
| Achieved Occupancy (%) | 6.12 % | 6.15 % | +0.03 pp | 6.08 % | -0.04 pp | 6.08 % | -0.04 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 93.27 % | 95.57 % | +2.30 pp | 93.34 % | +0.07 pp | 93.19 % | -0.08 pp |
| IPC Active | 0.24 inst/cycle | 0.21 inst/cycle | -0.03 | 0.24 inst/cycle | +0.00 | 0.25 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  2500 ns  (1.000×)
Iter 4     2430 ns  (1.029×)  — true_single_block_full_tile_xblock4096_num_warps16

Final best: 2430 ns  (1.0288×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,16 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_addmm_gelu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
-    xnumel = 4096
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x0 = (xindex % 2048)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_ptr0 + (x0), eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2))
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0.5, tl.float32)
     tmp4 = tmp2 * tmp3
@@ -24,9 +23,7 @@

     tmp8 = tl.full([1], 1.0, tl.float32)
     tmp9 = tmp7 + tmp8
     tmp10 = tmp4 * tmp9
-    tl.store(in_out_ptr0 + (x2), tmp10, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp10)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
