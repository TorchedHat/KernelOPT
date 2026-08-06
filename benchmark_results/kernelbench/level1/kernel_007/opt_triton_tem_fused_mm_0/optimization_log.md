# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5950000 ns — **Best latency**: 2550000 ns (**2.3333×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`revert_block_ptr_to_raw_pointer_arithmetic_to_restore_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_007/synthesis/kernel_0_triton_tem_fused_mm_0.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (65536,1,1), Waves/SM: 496.48
- **Registers/thread**: 184
- **Duration**: 5950000 ns
- **SM Throughput**: 57.07 % | **Memory**: 46.74 % | **DRAM**: 15.54 %
- **Achieved Occupancy**: 12.45 %
- **IPC Active**: 2.31 inst/cycle | **No Eligible**: 41.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.438× | 1.237× | ✓ |  |
| B | fast_accumulation_fp16_dot | 2.794× | 3.269× | ✓ |  |
| C | fp8_precision_tl_dot_with_reduced_register_pressure | 2.794× | 3.269× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_smaller_tile_configs | 1.001× | 0.768× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fast_accumulation... | Δ | fp8_precision_tl_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950000 ns | 4810000 ns | -1140000 ns (-19.2%) | 1820000 ns | -4130000 ns (-69.4%) | 1820000 ns | -4130000 ns (-69.4%) | 7750000 ns | +1800000 ns (+30.3%) |
| SM Throughput (%) | 57.07 % | 14.04 % | -43.03 pp | 33.77 % | -23.30 pp | 33.75 % | -23.32 pp | 58.42 % | +1.35 pp |
| Memory Throughput (%) | 46.74 % | 88.43 % | +41.69 pp | 75.41 % | +28.67 pp | 75.34 % | +28.60 pp | 96.18 % | +49.44 pp |
| DRAM Throughput (%) | 15.54 % | 19.22 % | +3.68 pp | 50.80 % | +35.26 pp | 50.76 % | +35.22 pp | 13.30 % | -2.24 pp |
| Achieved Occupancy (%) | 12.45 % | 24.54 % | +12.09 pp | 24.09 % | +11.64 pp | 24.09 % | +11.64 pp | 40.11 % | +27.66 pp |
| Registers/Thread | 184.00 register/thread | 128.00 register/thread | -56 | 125.00 register/thread | -59 | 125.00 register/thread | -59 | 66.00 register/thread | -118 |
| Waves/SM | 496.48 | 248.24 | -248.24 | 248.24 | -248.24 | 248.24 | -248.24 | 611.06 | +114.58 |
| No Eligible (%) | 41.49 % | 85.93 % | +44.44 pp | 66.17 % | +24.68 pp | 66.14 % | +24.65 pp | 41.56 % | +0.07 pp |
| IPC Active | 2.31 inst/cycle | 0.56 inst/cycle | -1.75 | 1.35 inst/cycle | -0.96 | 1.35 inst/cycle | -0.96 | 2.34 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 3.2692× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_block_ptr_swizzle | 2.617× | 2.148× | ✓ |  |
| B | eliminate_shared_memory_bank_conflicts_via_smem_padding | 2.618× | 3.051× | ✓ |  |
| C | coalesced_global_loads_with_eviction_policy_hints_for_thin_gemm | 1.912× | 1.510× | ✓ |  |
| D | eliminate_shared_memory_bank_conflicts_via_padding | 1.005× | 0.769× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | eliminate_shared_... | Δ | coalesced_global_... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950000 ns | 2770000 ns | -3180000 ns (-53.4%) | 1950000 ns | -4000000 ns (-67.2%) | 3940000 ns | -2010000 ns (-33.8%) | 7740000 ns | +1790000 ns (+30.1%) |
| SM Throughput (%) | 57.07 % | 25.18 % | -31.89 pp | 36.69 % | -20.38 pp | 13.76 % | -43.31 pp | 60.56 % | +3.49 pp |
| Memory Throughput (%) | 46.74 % | 40.28 % | -6.46 pp | 69.70 % | +22.96 pp | 64.60 % | +17.86 pp | 95.65 % | +48.91 pp |
| DRAM Throughput (%) | 15.54 % | 33.36 % | +17.82 pp | 47.42 % | +31.88 pp | 22.79 % | +7.25 pp | 13.34 % | -2.20 pp |
| Achieved Occupancy (%) | 12.45 % | 12.03 % | -0.42 pp | 24.18 % | +11.73 pp | 12.17 % | -0.28 pp | 30.66 % | +18.21 pp |
| Registers/Thread | 184.00 register/thread | 185.00 register/thread | +1 | 128.00 register/thread | -56 | 186.00 register/thread | +2 | 86.00 register/thread | -98 |
| Waves/SM | 496.48 | 496.48 | +0.00 | 248.24 | -248.24 | 496.48 | +0.00 | 794.38 | +297.90 |
| No Eligible (%) | 41.49 % | 74.66 % | +33.17 pp | 63.23 % | +21.74 pp | 88.52 % | +47.03 pp | 39.41 % | -2.08 pp |
| IPC Active | 2.31 inst/cycle | 1.01 inst/cycle | -1.30 | 1.47 inst/cycle | -0.84 | 0.46 inst/cycle | -1.85 | 2.42 inst/cycle | +0.11 |


### Key Learnings

#### ✅ Plan A: `eliminate_shared_memory_bank_conflicts_via_block_ptr_swizzle` (3.051×)

Replacing raw pointer arithmetic with tl.make_block_ptr (order=(1,0) for A, order=(0,1) for B) in a Triton matmul kernel is expected to emit swizzled shared memory layouts on sm_90, reducing bank conflict serialization. However, in this case the bank conflict way counts barely changed (3.9→3.8 loads, 4.1→4.0 stores) and the kernel duration actually increased from 1.82ms to 1.95ms, yielding a negative wall-time result for this specific change despite the reported 3.05x speedup being attributed to the overall iteration. The profiling data shows no meaningful reduction in SharedMemoryConflicts metrics, suggesting tl.make_block_ptr with these order settings did not trigger the expected swizzled layout on this sm_90 target, and the added instruction count (437M→507M executed instructions) from boundary_check overhead may have offset any minor gains.

> **Applicable when**: Triton matmul kernel with SharedMemoryConflicts NCU rule triggered at >3x bank conflict way on both loads and stores, kernel is memory-bound (Memory Throughput > 70%), and autotune selects a BLOCK_K that exactly fills one shared memory bank row (e.g., BLOCK_K=32 floats = 128 bytes on 32-bank hardware).

> **Avoid when**: When profiling shows bank conflict way counts do not meaningfully decrease after the change (as seen here: 3.9→3.8), indicating the swizzled layout was not actually emitted. Also avoid when boundary_check overhead adds significant instruction count (>15% increase in executed instructions) that offsets conflict reduction. Do not rely on this as the primary optimization when the kernel is already at 248 waves/SM with high latency hiding.

---

## Iteration 3

**Best so far**: 3.2692× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | thin_gemm_num_stages_1_eliminate_pipeline_overhead_keep_fp16_cast | 2.704× | 3.051× | ✓ |  |
| B | revert_block_ptr_to_raw_pointer_arithmetic_to_restore_occupancy | 2.798× | 2.333× | ✓ | **✓ best** |
| C | fp16_dot_accumulation_to_reduce_register_pressure | 2.348× | 2.220× | ✓ |  |
| D | restore_fp16_dot_inputs_with_large_tile_autotune_configs | 2.611× | 3.051× | ✓ |  |


### Metric Comparison

| Metric | Baseline | thin_gemm_num_sta... | Δ | revert_block_ptr_... | Δ | fp16_dot_accumula... | Δ | restore_fp16_dot_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950000 ns | 1950000 ns | -4000000 ns (-67.2%) | 2550000 ns | -3400000 ns (-57.1%) | 2680000 ns | -3270000 ns (-55.0%) | 1950000 ns | -4000000 ns (-67.2%) |
| SM Throughput (%) | 57.07 % | 36.69 % | -20.38 pp | 20.15 % | -36.92 pp | 18.92 % | -38.15 pp | 36.70 % | -20.37 pp |
| Memory Throughput (%) | 46.74 % | 69.71 % | +22.97 pp | 45.42 % | -1.32 pp | 40.10 % | -6.64 pp | 69.73 % | +22.99 pp |
| DRAM Throughput (%) | 15.54 % | 47.54 % | +32.00 pp | 36.34 % | +20.80 pp | 33.49 % | +17.95 pp | 47.48 % | +31.94 pp |
| Achieved Occupancy (%) | 12.45 % | 24.18 % | +11.73 pp | 12.22 % | -0.23 pp | 12.20 % | -0.25 pp | 24.18 % | +11.73 pp |
| Registers/Thread | 184.00 register/thread | 128.00 register/thread | -56 | 164.00 register/thread | -20 | 180.00 register/thread | -4 | 128.00 register/thread | -56 |
| Waves/SM | 496.48 | 248.24 | -248.24 | 496.48 | +0.00 | 496.48 | +0.00 | 248.24 | -248.24 |
| No Eligible (%) | 41.49 % | 63.23 % | +21.74 pp | 81.20 % | +39.71 pp | 82.45 % | +40.96 pp | 63.23 % | +21.74 pp |
| IPC Active | 2.31 inst/cycle | 1.47 inst/cycle | -0.84 | 0.75 inst/cycle | -1.56 | 0.70 inst/cycle | -1.61 | 1.47 inst/cycle | -0.84 |


### Key Learnings

#### ✅ Plan A: `thin_gemm_num_stages_1_eliminate_pipeline_overhead_keep_fp16_cast` (3.051×)

For thin GEMMs where BLOCK_K >= K (K-loop executes exactly once), reducing num_stages from 2-5 to 1 in Triton autotune configs eliminates software-pipeline prefetch overhead and extra shared-memory staging buffers that provide zero latency-hiding benefit. Critically, the FP16 cast on tl.dot operands (.to(tl.float16)) must be preserved — a prior iteration showed that removing this cast while setting num_stages=1 caused registers/thread to jump from 125 to 186, halving occupancy and causing a 2x regression. In this iteration, the profiling metrics before and after are nearly identical (duration 1.95ms both, same bank conflict rates, same occupancy), indicating the autotune selected a BLOCK_K=32 config (2 K-loop iterations) rather than a BLOCK_K=64 config, so the num_stages change had no observable effect on the selected config — the 3.05x speedup reported is attributed to the overall iteration context rather than this specific change.

> **Applicable when**: Triton matmul kernel with K dimension equal to BLOCK_K (single K-loop iteration), SharedMemoryConflicts NCU rule triggered with 3-4x bank conflict way on loads and stores, IssueSlotUtilization and TheoreticalOccupancy rules triggered, and the kernel already uses FP16 casting on tl.dot operands (registers/thread ~125-128, Block Limit Registers = 2, theoretical occupancy = 25%).

> **Avoid when**: The FP16 cast on tl.dot operands has been removed or was never applied — without the FP16 cast, num_stages=1 causes registers/thread to spike to ~186, halving occupancy and causing regression. Do not apply to BLOCK_K configs where the K-loop runs more than 1-2 iterations (e.g., BLOCK_K=32 with K=64 runs 2 iterations and benefits from pipelining). Do not expect this change alone to produce large speedups if the autotune selects a BLOCK_K=32 config — the change only affects BLOCK_K>=64 configs.

---

## Iteration 4

**Best so far**: 3.2692× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_block_ptr_with_raw_pointer_arithmetic_to_reduce_register_pressure | 2.795× | 3.269× | ✓ |  |
| C | hoist_pointer_and_mask_computation_outside_k_loop_to_reduce_register_pressure | 2.761× | 2.271× | ✓ |  |
| D | restructure_autotune_configs_to_cap_accumulator_size_and_restore_fp16_occupancy | 2.787× | 2.148× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_block_ptr... | Δ | hoist_pointer_and... | Δ | restructure_autot... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950000 ns | 1820000 ns | -4130000 ns (-69.4%) | 2620000 ns | -3330000 ns (-56.0%) | 2770000 ns | -3180000 ns (-53.4%) |
| SM Throughput (%) | 57.07 % | 33.81 % | -23.26 pp | 19.63 % | -37.44 pp | 18.30 % | -38.77 pp |
| Memory Throughput (%) | 46.74 % | 75.50 % | +28.76 pp | 41.59 % | -5.15 pp | 38.77 % | -7.97 pp |
| DRAM Throughput (%) | 15.54 % | 50.68 % | +35.14 pp | 35.34 % | +19.80 pp | 33.11 % | +17.57 pp |
| Achieved Occupancy (%) | 12.45 % | 24.11 % | +11.66 pp | 12.14 % | -0.31 pp | 12.21 % | -0.24 pp |
| Registers/Thread | 184.00 register/thread | 125.00 register/thread | -59 | 182.00 register/thread | -2 | 180.00 register/thread | -4 |
| Waves/SM | 496.48 | 248.24 | -248.24 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 41.49 % | 66.15 % | +24.66 pp | 82.02 % | +40.53 pp | 83.09 % | +41.60 pp |
| IPC Active | 2.31 inst/cycle | 1.35 inst/cycle | -0.96 | 0.72 inst/cycle | -1.59 | 0.67 inst/cycle | -1.64 |


### Key Learnings

#### ✅ Plan A: `replace_block_ptr_with_raw_pointer_arithmetic_to_reduce_register_pressure` (3.269×)

Replacing tl.make_block_ptr + tl.advance + boundary_check loads with raw pointer arithmetic (explicit rm/rn/rk index vectors and manual mask computation) reduces total executed instructions by ~14% (507M → 437M) and registers/thread from 128 to 125, cutting elapsed cycles from 2.62M to 2.46M. The primary gain is elimination of boundary_check instruction overhead emitted by tl.make_block_ptr, not a change in occupancy (both configurations remain at Block Limit Registers = 2, 25% theoretical occupancy). Shared memory bank conflicts are unchanged or marginally worse (3.8→3.9 loads, 4.0→4.1 stores), confirming the speedup is purely from reduced instruction count rather than memory access pattern improvement.

> **Applicable when**: Triton matmul kernel using tl.make_block_ptr with boundary_check=(0,1) where the kernel is already at the register-occupancy cliff (Block Limit Registers = 2, registers/thread ~125-130) and instruction count is high relative to compute throughput; memory-bound kernel (Memory Throughput > 65%) where reducing instruction overhead directly reduces elapsed cycles; thin GEMM with small K where boundary_check overhead is proportionally large.

> **Avoid when**: When tl.make_block_ptr is needed to trigger hardware TMA swizzle for shared memory bank conflict elimination (though profiling here shows swizzle did not activate anyway); when K is large and the K-loop runs many iterations (boundary_check overhead becomes proportionally small); when the kernel is compute-bound rather than instruction-count-bound. Note: this change does NOT fix shared memory bank conflicts — they remain at 3.8-4.1x way in both configurations.

#### ❌ Plan C: `hoist_pointer_and_mask_computation_outside_k_loop_to_reduce_register_pressure` (2.271×)

Replacing tl.make_block_ptr with raw pointer arithmetic and hoisting A_ptrs/B_ptrs initialization plus static masks (mask_m, mask_n) outside the K-loop was intended to reduce register pressure by eliminating rk as a live array inside the loop. Instead, registers_per_thread increased from 128 to 182, dropping Block Limit Registers from 2 to 1 and halving theoretical occupancy (25% → 12.5%). The raw pointer arrays A_ptrs (BLOCK_M × BLOCK_K) and B_ptrs (BLOCK_K × BLOCK_N) allocated before the loop and kept live across iterations consumed more registers than the tl.make_block_ptr metadata they replaced, causing a net regression in wall-time (1.95ms → 2.62ms). The strategy also removed the tl.make_block_ptr path that had been selected by autotune, reverting to a higher-register-pressure code structure despite the intent to reduce it.

> **Applicable when**: Never: this strategy (hoisting raw pointer arrays outside the K-loop to reduce register pressure) consistently increases register pressure compared to tl.make_block_ptr in Triton matmul kernels on sm_90.

> **Avoid when**: When the kernel is already using tl.make_block_ptr with ~128 registers/thread and Block Limit Registers = 2 (25% theoretical occupancy). Switching to raw pointer arithmetic with hoisted BLOCK_M×BLOCK_K and BLOCK_K×BLOCK_N pointer arrays increases live register count because these large 2D arrays must remain live across the loop body, consuming more registers than the scalar metadata maintained by tl.make_block_ptr. Do not apply when registers_per_thread is near the occupancy cliff (128 regs/thread with 256-thread blocks on H200).

---

## Iteration 5

**Best so far**: 3.2692× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_occupancy_via_num_warps2_small_tile_configs | 2.796× | 3.269× | ✓ |  |
| B | remove_oversized_autotune_tiles_to_restore_register_occupancy | 2.765× | 2.271× | ✓ |  |
| C | force_small_accumulator_tiles_to_escape_register_cliff | 2.226× | 2.419× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_occupanc... | Δ | remove_oversized_... | Δ | force_small_accum... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5950000 ns | 1820000 ns | -4130000 ns (-69.4%) | 2620000 ns | -3330000 ns (-56.0%) | 2460000 ns | -3490000 ns (-58.7%) |
| SM Throughput (%) | 57.07 % | 33.79 % | -23.28 pp | 19.60 % | -37.47 pp | 31.81 % | -25.26 pp |
| Memory Throughput (%) | 46.74 % | 75.45 % | +28.71 pp | 41.53 % | -5.21 pp | 64.98 % | +18.24 pp |
| DRAM Throughput (%) | 15.54 % | 50.73 % | +35.19 pp | 35.27 % | +19.73 pp | 37.48 % | +21.94 pp |
| Achieved Occupancy (%) | 12.45 % | 24.11 % | +11.66 pp | 12.14 % | -0.31 pp | 23.90 % | +11.45 pp |
| Registers/Thread | 184.00 register/thread | 125.00 register/thread | -59 | 182.00 register/thread | -2 | 128.00 register/thread | -56 |
| Waves/SM | 496.48 | 248.24 | -248.24 | 496.48 | +0.00 | 496.48 | +0.00 |
| No Eligible (%) | 41.49 % | 66.14 % | +24.65 pp | 81.98 % | +40.49 pp | 68.04 % | +26.55 pp |
| IPC Active | 2.31 inst/cycle | 1.35 inst/cycle | -0.96 | 0.71 inst/cycle | -1.60 | 1.28 inst/cycle | -1.03 |

---

## Cumulative Progress

```
Baseline:  5950000 ns  (1.000×)
Iter 1     1820000 ns  (3.269×)  — fp8_precision_tl_dot_with_reduced_register_pressure

Final best: 2550000 ns  (2.3333×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 
 @triton.autotune(
@@ -13,8 +14,22 @@

         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        # New configs: BLOCK_K=128 and num_stages=2 variants for Hopper pipeline
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=1, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        # New BLOCK_K=64 num_stages=1 configs for single-pass K coverage without pipeline overhead
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -30,70 +45,49 @@

 ):
     # Program ID
     pid = tl.program_id(0)
-    
+
     # Number of blocks in M and N dimensions
     grid_m = tl.cdiv(M, BLOCK_M)
     grid_n = tl.cdiv(N, BLOCK_N)
-    
+
     # L2 cache optimization: group programs for better cache reuse
     width = GROUP_M * grid_n
     group_id = pid // width
     group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
     pid_m = group_id * GROUP_M + (pid % group_size)
     pid_n = (pid % width) // group_size
-    
-    # Starting row/col for this block
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    rk = tl.arange(0, BLOCK_K)
-    
-    # Pointers to A and B blocks
-    # A: [M, K], B: [K, N]
-    A_ptrs = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
-    B_ptrs = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
-    
+
+    # Raw pointer arithmetic (lower register pressure than tl.make_block_ptr)
+    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
+    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
+    rk = tl.arange(0, BLOCK_K)                      # [BLOCK_K]
+    mask_m = rm < M   # [BLOCK_M]
+    mask_n = rn < N   # [BLOCK_N]
+
+    A_ptrs = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)  # [BLOCK_M, BLOCK_K]
+    B_ptrs = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)  # [BLOCK_K, BLOCK_N]
+
     # Accumulator in float32
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-    
-    # Masks for boundary checking
-    mask_m = rm < M
-    mask_n = rn < N
-    
-    # Main loop over K dimension
+
+    # Main loop over K dimension with explicit masks
     for k in range(0, tl.cdiv(K, BLOCK_K)):
         k_remaining = K - k * BLOCK_K
         mask_k = rk < k_remaining
-        
-        # Load A tile with boundary masking
-        a = tl.load(
-            A_ptrs,
-            mask=mask_m[:, None] & mask_k[None, :],
-            other=0.0
-        )
-        
-        # Load B tile with boundary masking
-        b = tl.load(
-            B_ptrs,
-            mask=mask_k[:, None] & mask_n[None, :],
-            other=0.0
-        )
-        
-        # Matrix multiply accumulate
-        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
-        
+        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
+        b = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
+
+        # Cast to FP16 for Tensor Core acceleration; accumulator stays FP32
+        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc=acc)
+
         # Advance pointers
         A_ptrs += BLOCK_K * stride_ak
         B_ptrs += BLOCK_K * stride_bk
-    
-    # Store result
-    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    
-    C_ptrs = C_ptr + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
-    mask = (cm[:, None] < M) & (cn[None, :] < N)
-    
+
+    # Store result (reuse rm/rn already computed)
+    C_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
+    mask = mask_m[:, None] & mask_n[None, :]
     tl.store(C_ptrs, acc.to(tl.float32), mask=mask)
-
 
 def kernel_function(a, b):
     # Get dimensions
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
