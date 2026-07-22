# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 38940 ns — **Best latency**: 38940 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_36_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (96,1,1), Waves/SM: 0.09
- **Registers/thread**: 64
- **Duration**: 38940 ns
- **SM Throughput**: 9.46 % | **Memory**: 60.89 % | **DRAM**: 0.96 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.545× | 0.522× | ✓ |  |
| B | tma_loads_with_larger_tiles_to_fix_bank_conflicts_and_occupancy | 0.511× | 0.490× | ✓ |  |
| C | autotune_tile_sizes_and_pipeline_stages_to_fix_occupancy_and_bank_conflicts | 0.546× | 0.519× | ✓ |  |
| D | increase_tile_sizes_with_autotune | 0.453× | 0.412× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_la... | Δ | autotune_tile_siz... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38940 ns | 74530 ns | +35590 ns (+91.4%) | 79390 ns | +40450 ns (+103.9%) | 75010 ns | +36070 ns (+92.6%) | 94590 ns | +55650 ns (+142.9%) |
| SM Throughput (%) | 9.46 % | 16.43 % | +6.97 pp | 15.46 % | +6.00 pp | 16.40 % | +6.94 pp | 13.29 % | +3.83 pp |
| Memory Throughput (%) | 60.89 % | 63.66 % | +2.77 pp | 60.41 % | -0.48 pp | 63.56 % | +2.67 pp | 52.41 % | -8.48 pp |
| DRAM Throughput (%) | 0.96 % | 0.51 % | -0.45 pp | 0.48 % | -0.48 pp | 0.51 % | -0.45 pp | 0.40 % | -0.56 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 128.00 register/thread | +64 | 128.00 register/thread | +64 | 128.00 register/thread | +64 | 128.00 register/thread | +64 |
| Waves/SM | 0.09 | 0.18 | +0.09 | 0.18 | +0.09 | 0.18 | +0.09 | 0.18 | +0.09 |
| No Eligible (%) | 86.31 % | 76.62 % | -9.69 pp | 78.08 % | -8.23 pp | 76.63 % | -9.68 pp | 81.05 % | -5.26 pp |
| IPC Active | 0.55 inst/cycle | 0.93 inst/cycle | +0.38 | 0.88 inst/cycle | +0.33 | 0.94 inst/cycle | +0.39 | 0.76 inst/cycle | +0.21 |


### Key Learnings

#### ❌ Plan D: `increase_tile_sizes_with_autotune` (0.412×)

Increasing BLOCK_M/BLOCK_N/BLOCK_K from 32 to 64 caused registers per thread to double from 64 to 128, which halved theoretical occupancy from 50% to 25% and doubled shared memory per block from 16 to 33 KB. Despite the grid and launch config remaining the same (96 blocks, 132 SMs), achieved occupancy stayed at 6.25% while elapsed cycles nearly tripled (58K to 142K) and executed instructions increased 3.4x (2.9M to 9.9M), indicating the larger tiles did far more redundant work without improving SM utilization. The autotune decorator was applied only over num_warps/num_stages without varying tile sizes as constexprs, so the tile size increase was unconditional and the register blowup was unavoidable.

> **Applicable when**: Never — this specific pattern (increasing tile sizes as hard-coded constexprs without making them autotune keys, while autotune only varies num_warps/num_stages) is always harmful when register pressure is already near the limit.

> **Avoid when**: When registers per thread are already at or near 64 on a device where block limit is register-bound (Block Limit Registers = 8 blocks/SM at 64 regs/thread). Doubling tile sizes in this regime doubles register usage, halves theoretical occupancy, and increases total instruction count without improving SM utilization. Also avoid when the grid is not updated to reflect larger tiles — the same 96-block grid with 64x64 tiles covers less of the problem space per block, causing redundant boundary masking overhead.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_tile_sizes_to_restore_register_occupancy | 0.993× | 0.996× | ✓ |  |
| B | reduce_tile_sizes_to_restore_register_occupancy_and_fix_bank_conflicts | 0.999× | 0.807× | ✓ |  |
| C | reduce_register_pressure_by_reverting_to_small_tiles_with_proper_autotune | 1.000× | 1.005× | ✓ |  |
| D | reduce_tile_sizes_to_restore_occupancy_and_grid_coverage | 0.992× | 0.801× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_tile_sizes... | Δ | reduce_tile_sizes... | Δ | reduce_register_p... | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38940 ns | 39100 ns | +160 ns (+0.4%) | 48220 ns | +9280 ns (+23.8%) | 38750 ns | -190 ns (-0.5%) | 48640 ns | +9700 ns (+24.9%) |
| SM Throughput (%) | 9.46 % | 9.45 % | -0.01 pp | 7.28 % | -2.18 pp | 9.33 % | -0.13 pp | 7.25 % | -2.21 pp |
| Memory Throughput (%) | 60.89 % | 60.83 % | -0.06 pp | 48.99 % | -11.90 pp | 61.12 % | +0.23 pp | 49.08 % | -11.81 pp |
| DRAM Throughput (%) | 0.96 % | 0.96 % | +0.00 pp | 0.78 % | -0.18 pp | 0.97 % | +0.01 pp | 0.77 % | -0.19 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 62.00 register/thread | -2 | 64.00 register/thread | +0 | 62.00 register/thread | -2 |
| Waves/SM | 0.09 | 0.09 | +0.00 | 0.09 | +0.00 | 0.09 | +0.00 | 0.09 | +0.00 |
| No Eligible (%) | 86.31 % | 86.33 % | +0.02 pp | 89.44 % | +3.13 pp | 86.42 % | +0.11 pp | 89.64 % | +3.33 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.42 inst/cycle | -0.13 | 0.54 inst/cycle | -0.01 | 0.42 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan B: `reduce_tile_sizes_to_restore_register_occupancy_and_fix_bank_conflicts` (0.807×)

Reducing BLOCK_M/BLOCK_N from 64 to 32 and replacing the transposed B load with a standard [BLOCK_K, BLOCK_N] load successfully halved registers per thread (128→62), doubled theoretical occupancy (25%→50%), and reduced instruction count 3.5x (9.7M→2.8M). However, the kernel still regressed vs. the original baseline (0.81x speedup) because warp cycles per issued instruction nearly doubled (4.28→9.51 cycles), waves per SM halved (0.18→0.09), and achieved occupancy remained stuck at 6.25% — the grid of 96 blocks on 132 SMs is too small to fill the GPU regardless of tile size. The shared memory bank conflict rate remained at ~78% (12-way conflicts), indicating the B-load rewrite did not eliminate the root cause of MIO stalls. The net effect was fewer total instructions but each instruction took longer to issue, resulting in only a modest wall-time improvement that still fell short of the original kernel.

> **Applicable when**: Never — this specific combination (reducing tile sizes from 64→32 while adding autotune over num_warps/num_stages only, with key=[]) on a kernel with grid=96 and 132 SMs does not recover the original baseline performance. The grid is too small to benefit from occupancy improvements.

> **Avoid when**: Do not apply tile size reduction as the primary optimization when the kernel's grid size is already smaller than the SM count (waves_per_sm < 0.2) and achieved occupancy is stuck at 6.25% regardless of theoretical occupancy. In this regime, reducing tile sizes reduces total work per block without increasing parallelism, and the autotune overhead (key=[]) adds latency without exploring meaningful configurations. Also do not expect that switching from transposed B load to standard B load eliminates bank conflicts when the conflict rate remains at ~78% after the change.

#### ❌ Plan D: `reduce_tile_sizes_to_restore_occupancy_and_grid_coverage` (0.801×)

Reducing BLOCK_M/BLOCK_N from 64 to 32 successfully cut registers per thread from 128 to 62 and doubled theoretical occupancy from 25% to 50%, and reduced instruction count from 9.7M to 2.8M. However, achieved occupancy remained stuck at 6.25% (same as before), and warp cycles per issued instruction worsened from 4.28 to 9.64 — the scheduler idle rate increased from 76.6% to 89.6%. The kernel ran faster in absolute terms (74.53µs → 48.64µs) but still slower than the original baseline (38.94µs), yielding a 0.8x speedup relative to the immediately prior iteration. The persistent 12-way shared memory bank conflicts (77-78% excessive wavefronts) and MIO stalls (5.6 cycles/warp) remain the dominant bottleneck regardless of tile size, and the small grid (96 blocks on 132 SMs, 0.09 waves/SM) means the GPU cannot hide latency through parallelism.

> **Applicable when**: When registers_per_thread is very high (>=128) due to large tile sizes and Block Limit Registers is the binding occupancy constraint — reducing tile sizes will lower register pressure and improve theoretical occupancy. However, this alone is insufficient when shared memory bank conflicts (12-way, >75% excessive wavefronts) and MIO stalls dominate, and when the grid is too small to fill the GPU (waves_per_sm < 0.2).

> **Avoid when**: Do not expect tile size reduction alone to fully recover performance when the kernel has persistent 12-way shared memory bank conflicts (UncoalescedSharedAccess >75% excessive wavefronts) and the grid is too small to fill the GPU (96 blocks on 132 SMs). The bank conflict stalls will dominate regardless of tile size, and achieved occupancy will remain low. The autotune decorator with empty configs {} (no tile size keys) provides no benefit and adds overhead.

---

## Cumulative Progress

```
Baseline:  38940 ns  (1.000×)

Final best: 38940 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,14 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 64
+    BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -90,7 +91,6 @@

     xindex = idx_n + 192*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
