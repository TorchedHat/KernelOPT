# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 39520 ns — **Best latency**: 40670 ns (**0.9717×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_25_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.19
- **Registers/thread**: 94
- **Duration**: 39520 ns
- **SM Throughput**: 12.36 % | **Memory**: 83.65 % | **DRAM**: 1.64 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.56 inst/cycle | **No Eligible**: 86.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 0.996× | ✓ |  |
| B | add_autotune_reduce_register_pressure_fix_bank_conflicts | 1.282× | 0.972× | ✓ | **✓ best** |
| C | autotune_tile_sizes_to_eliminate_bank_conflicts_and_improve_occupancy | 1.282× | 1.310× | ✓ |  |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.272× | 0.903× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | add_autotune_redu... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39520 ns | 39680 ns | +160 ns (+0.4%) | 40670 ns | +1150 ns (+2.9%) | 30180 ns | -9340 ns (-23.6%) | 43740 ns | +4220 ns (+10.7%) |
| SM Throughput (%) | 12.36 % | 12.31 % | -0.05 pp | 12.48 % | +0.12 pp | 15.41 % | +3.05 pp | 11.14 % | -1.22 pp |
| Memory Throughput (%) | 83.65 % | 83.31 % | -0.34 pp | 81.43 % | -2.22 pp | 58.36 % | -25.29 pp | 80.29 % | -3.36 pp |
| DRAM Throughput (%) | 1.64 % | 1.63 % | -0.01 pp | 1.59 % | -0.05 pp | 2.14 % | +0.50 pp | 1.48 % | -0.16 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 94.00 register/thread | +0 | 96.00 register/thread | +2 | 128.00 register/thread | +34 | 92.00 register/thread | -2 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.19 | +0.00 | 0.12 | -0.07 | 0.19 | +0.00 |
| No Eligible (%) | 86.09 % | 86.11 % | +0.02 pp | 85.92 % | -0.17 pp | 64.77 % | -21.32 pp | 87.45 % | +1.36 pp |
| IPC Active | 0.56 inst/cycle | 0.56 inst/cycle | +0.00 | 0.56 inst/cycle | +0.00 | 0.71 inst/cycle | +0.15 | 0.50 inst/cycle | -0.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | promote_tile_sizes_to_kernel_params_and_autotune_full_gemm_config | 1.249× | 0.921× | ✓ |  |
| D | promote_tile_sizes_to_kernel_params_for_autotune | 1.249× | 1.282× | ✓ |  |


### Metric Comparison

| Metric | Baseline | promote_tile_size... | Δ | promote_tile_size... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 39520 ns | 42910 ns | +3390 ns (+8.6%) | 30820 ns | -8700 ns (-22.0%) |
| SM Throughput (%) | 12.36 % | 10.58 % | -1.78 pp | 15.08 % | +2.72 pp |
| Memory Throughput (%) | 83.65 % | 45.53 % | -38.12 pp | 63.44 % | -20.21 pp |
| DRAM Throughput (%) | 1.64 % | 1.51 % | -0.13 pp | 2.10 % | +0.46 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 128.00 register/thread | +34 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.12 | -0.07 |
| No Eligible (%) | 86.09 % | 75.81 % | -10.28 pp | 65.31 % | -20.78 pp |
| IPC Active | 0.56 inst/cycle | 0.49 inst/cycle | -0.07 | 0.69 inst/cycle | +0.13 |


### Key Learnings

#### ⚠️ Plan D: `promote_tile_sizes_to_kernel_params_for_autotune` (0.000×)

Plan 'promote_tile_sizes_to_kernel_params_for_autotune' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=1.007e+02 > atol=0.001
seed=42 output[0]: max_diff=8.390e+01 > atol=0.001
seed=1337 output[0]: max_diff=9.614e+01 > atol=0.001

> **Applicable when**: The kernel has BLOCK_M=32, BLOCK_N=64, BLOCK_K=16 hardcoded as constexpr body assignments. The 32x64 accumulator tile alone requires ~2048 float registers, driving register count to 128/thread and capping theoretical occupancy at 25% (8 blocks/SM). Achieved occupancy is only 3.12% because only 124 blocks are launched across 132 SMs (0.12 waves). Reducing tile sizes (e.g., BLOCK_M=16, BLOCK_N=32 or BLOCK_M=32, BLOCK_N=32) would cut register pressure, raise occupancy, and allow more warps to hide the 11.9-way shared memory bank conflict stalls. Prior iteration (exp_001) already autotuned num_warps/num_stages but kept tile sizes fixed — the memory item explicitly identifies promoting tile sizes to kernel signature parameters as the next required step for full benefit.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=1.007e+02 > atol=0.001
seed=42 output[0]: max_diff=8.390e+01 > atol=0.001
seed=1337 output[0]: max_diff=9.614e+01 > atol=0.001

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts | 1.261× | 1.274× | ✓ |  |
| B | tma_loads_via_block_ptr_for_async_bulk_copy | 1.198× | 1.222× | ✓ |  |
| D | tma_loads_via_block_ptr | 1.196× | 0.888× | ✓ |  |
| E | add_triton_autotune | 0.935× | 0.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_via_blo... | Δ | tma_loads_via_blo... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39520 ns | 31010 ns | -8510 ns (-21.5%) | 32350 ns | -7170 ns (-18.1%) | 44480 ns | +4960 ns (+12.6%) | 42560 ns | +3040 ns (+7.7%) |
| SM Throughput (%) | 12.36 % | 15.12 % | +2.76 pp | 14.69 % | +2.33 pp | 10.50 % | -1.86 pp | 11.90 % | -0.46 pp |
| Memory Throughput (%) | 83.65 % | 63.60 % | -20.05 pp | 60.88 % | -22.77 pp | 44.38 % | -39.27 pp | 82.67 % | -0.98 pp |
| DRAM Throughput (%) | 1.64 % | 2.09 % | +0.45 pp | 2.00 % | +0.36 pp | 1.45 % | -0.19 pp | 1.52 % | -0.12 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 94.00 register/thread | 128.00 register/thread | +34 | 128.00 register/thread | +34 | 128.00 register/thread | +34 | 96.00 register/thread | +2 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.12 | -0.07 | 0.12 | -0.07 | 0.19 | +0.00 |
| No Eligible (%) | 86.09 % | 65.38 % | -20.71 pp | 66.40 % | -19.69 pp | 75.14 % | -10.95 pp | 86.65 % | +0.56 pp |
| IPC Active | 0.56 inst/cycle | 0.69 inst/cycle | +0.13 | 0.67 inst/cycle | +0.11 | 0.50 inst/cycle | -0.06 | 0.53 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan B: `tma_loads_via_block_ptr_for_async_bulk_copy` (1.222×)

Replacing manual scalar pointer arithmetic in a Triton GEMM k-loop with tl.make_block_ptr() + tl.load() + tl.advance() was expected to emit TMA (cp.async.bulk.tensor) instructions on Hopper and bypass shared memory bank conflicts. However, the profiling data shows the bank conflict metrics (13.3-way, 79.84% conflicted wavefronts) and MIO stall fraction (48.5% → 50.3%) were completely unchanged after the rewrite, and kernel duration actually increased slightly (30.82 → 32.35 µs). The 1.22x speedup reported is inconsistent with the profiling showing a regression in duration, suggesting the speedup may reflect measurement noise or a different kernel path; the structural bottleneck (shared memory bank conflicts) was not addressed by this change.

> **Applicable when**: Triton GEMM kernel with manual scalar pointer arithmetic in the k-loop, SharedMemoryConflicts rule triggered with >10-way bank conflicts and >75% conflicted wavefronts, CPIStall MIO fraction >40%, running on Hopper (compute capability 9.0) where TMA hardware is available.

> **Avoid when**: The shared memory bank conflict metrics (bank_conflict_way_average, conflicted_wavefronts_pct) are driven by the shared memory layout of the accumulator or epilogue, not the load path — in that case block_ptr loads will not reduce conflicts. Also avoid when the speedup target is >20% from bank conflict elimination alone, as this approach did not move the conflict metrics at all in this case. Do not use as the sole optimization when bank conflicts remain at 13+ way after application.

---

## Cumulative Progress

```
Baseline:  39520 ns  (1.000×)

Final best: 40670 ns  (0.9717×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=5),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
-    BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 64
+    BLOCK_N : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -72,9 +88,7 @@

         xindex = idx_n + 96*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
 
-
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +100,6 @@

     # inductor generates a suffix
     xindex = idx_n + 96*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
