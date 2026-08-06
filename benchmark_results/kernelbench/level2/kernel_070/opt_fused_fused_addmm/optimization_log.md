# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_sigmoid_scale_residual_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8000000 ns — **Best latency**: 2780000 ns (**2.8777×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_accumulator_tile_size_to_break_255_register_ceiling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_sigmoid_scale_residual_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_070/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 8000000 ns
- **SM Throughput**: 43.30 % | **Memory**: 59.66 % | **DRAM**: 0.99 %
- **Achieved Occupancy**: 12.49 %
- **IPC Active**: 1.84 inst/cycle | **No Eligible**: 54.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_smaller_tiles_and_maxnreg | 0.881× | 0.605× | ✓ |  |
| C | eviction_policies_streaming_gemm_tiles | 1.002× | 1.008× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_maxnreg_and_smaller_tiles | 1.002× | 0.609× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8000000 ns | 13230000 ns | +5230000 ns (+65.4%) | 7940000 ns | -60000 ns (-0.7%) | 13130000 ns | +5130000 ns (+64.1%) |
| SM Throughput (%) | 43.30 % | 25.15 % | -18.15 pp | 43.31 % | +0.01 pp | 24.74 % | -18.56 pp |
| Memory Throughput (%) | 59.66 % | 94.47 % | +34.81 pp | 59.68 % | +0.02 pp | 93.41 % | +33.75 pp |
| DRAM Throughput (%) | 0.99 % | 1.19 % | +0.20 pp | 1.02 % | +0.03 pp | 1.10 % | +0.11 pp |
| Achieved Occupancy (%) | 12.49 % | 23.93 % | +11.44 pp | 12.42 % | -0.07 pp | 18.18 % | +5.69 pp |
| Registers/Thread | 255.00 register/thread | 128.00 register/thread | -127 | 255.00 register/thread | +0 | 164.00 register/thread | -91 |
| Waves/SM | 1.94 | 3.88 | +1.94 | 1.94 | +0.00 | 5.17 | +3.23 |
| No Eligible (%) | 54.33 % | 74.04 % | +19.71 pp | 54.03 % | -0.30 pp | 74.36 % | +20.03 pp |
| IPC Active | 1.84 inst/cycle | 1.04 inst/cycle | -0.80 | 1.84 inst/cycle | +0.00 | 1.03 inst/cycle | -0.81 |

---

## Iteration 2

**Best so far**: 1.0076× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 3.731× | 1.975× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8000000 ns | 4050000 ns | -3950000 ns (-49.4%) |
| SM Throughput (%) | 43.30 % | 27.91 % | -15.39 pp |
| Memory Throughput (%) | 59.66 % | 53.47 % | -6.19 pp |
| DRAM Throughput (%) | 0.99 % | 3.41 % | +2.42 pp |
| Achieved Occupancy (%) | 12.49 % | 12.41 % | -0.08 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 |
| No Eligible (%) | 54.33 % | 76.48 % | +22.15 pp |
| IPC Active | 1.84 inst/cycle | 0.94 inst/cycle | -0.90 |

---

## Iteration 3

**Best so far**: 1.9753× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_tail_via_grid_aligned_tiling | 3.826× | 2.036× | ✓ | **✓ best** |
| B | tma_loads_via_block_ptr | 3.682× | 1.214× | ✓ |  |
| C | reduce_local_memory_spills_via_smaller_block_k | 3.188× | 1.975× | ✓ |  |
| D | increase_block_k_and_align_grid_to_eliminate_partial_wave | 3.724× | 1.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | tma_loads_via_blo... | Δ | reduce_local_memo... | Δ | increase_block_k_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8000000 ns | 3930000 ns | -4070000 ns (-50.9%) | 6590000 ns | -1410000 ns (-17.6%) | 4050000 ns | -3950000 ns (-49.4%) | 4040000 ns | -3960000 ns (-49.5%) |
| SM Throughput (%) | 43.30 % | 28.33 % | -14.97 pp | 17.22 % | -26.08 pp | 27.91 % | -15.39 pp | 27.76 % | -15.54 pp |
| Memory Throughput (%) | 59.66 % | 54.58 % | -5.08 pp | 33.11 % | -26.55 pp | 53.30 % | -6.36 pp | 53.57 % | -6.09 pp |
| DRAM Throughput (%) | 0.99 % | 3.45 % | +2.46 pp | 1.39 % | +0.40 pp | 3.43 % | +2.44 pp | 3.43 % | +2.44 pp |
| Achieved Occupancy (%) | 12.49 % | 12.18 % | -0.31 pp | 6.25 % | -6.24 pp | 12.45 % | -0.04 pp | 12.45 % | -0.04 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 3.88 | +1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.33 % | 76.57 % | +22.24 pp | 85.57 % | +31.24 pp | 76.61 % | +22.28 pp | 76.57 % | +22.24 pp |
| IPC Active | 1.84 inst/cycle | 0.93 inst/cycle | -0.91 | 0.58 inst/cycle | -1.26 | 0.94 inst/cycle | -0.90 | 0.94 inst/cycle | -0.90 |

---

## Iteration 4

**Best so far**: 2.0356× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | prune_large_tile_configs_to_reduce_local_memory_spill_traffic | 3.080× | 1.980× | ✓ |  |
| B | increase_num_stages_to_hide_l1tex_latency | 3.755× | 2.005× | ✓ |  |
| C | reduce_local_memory_spills_via_tf32_single_pass | 3.137× | 2.051× | ✓ |  |
| D | reduce_smem_per_block_via_lower_num_stages_to_restore_occupancy | 3.198× | 1.951× | ✓ |  |


### Metric Comparison

| Metric | Baseline | prune_large_tile_... | Δ | increase_num_stag... | Δ | reduce_local_memo... | Δ | reduce_smem_per_b... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8000000 ns | 4040000 ns | -3960000 ns (-49.5%) | 3990000 ns | -4010000 ns (-50.1%) | 3900000 ns | -4100000 ns (-51.2%) | 4100000 ns | -3900000 ns (-48.8%) |
| SM Throughput (%) | 43.30 % | 27.84 % | -15.46 pp | 27.98 % | -15.32 pp | 28.62 % | -14.68 pp | 27.35 % | -15.95 pp |
| Memory Throughput (%) | 59.66 % | 53.98 % | -5.68 pp | 54.48 % | -5.18 pp | 55.46 % | -4.20 pp | 54.15 % | -5.51 pp |
| DRAM Throughput (%) | 0.99 % | 3.64 % | +2.65 pp | 3.94 % | +2.95 pp | 3.81 % | +2.82 pp | 4.14 % | +3.15 pp |
| Achieved Occupancy (%) | 12.49 % | 12.25 % | -0.24 pp | 12.39 % | -0.10 pp | 12.28 % | -0.21 pp | 12.27 % | -0.22 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 54.33 % | 76.98 % | +22.65 pp | 76.46 % | +22.13 pp | 76.11 % | +21.78 pp | 77.00 % | +22.67 pp |
| IPC Active | 1.84 inst/cycle | 0.92 inst/cycle | -0.92 | 0.94 inst/cycle | -0.90 | 0.95 inst/cycle | -0.89 | 0.92 inst/cycle | -0.92 |


### Key Learnings

#### ✅ Plan A: `prune_large_tile_configs_to_reduce_local_memory_spill_traffic` (1.980×)

Removing large-tile autotune configs (BLOCK_M/N=256, BLOCK_K=64 with num_stages>=3) and replacing them with smaller tiles capped at BLOCK_M/N<=128 with num_stages<=3 and num_stages=2 variants reduced dynamic shared memory per block from 98 KB to 65 KB (block_limit_shared_mem improved from 2 to 3), but registers/thread remained at 255 and all key stall metrics were essentially unchanged: MemoryCacheAccessPattern on local loads/stores stayed at ~52%, CPIStall at ~41%, and no-eligible-warp at ~77%. The profiling shows the autotuner selected a config that still drives 255 registers/thread even from the pruned set, meaning the accumulator tile size (BLOCK_M=128, BLOCK_N=128) alone is sufficient to saturate the register file regardless of num_stages. The measured speedup of ~1.98x reflects the overall optimization chain (TF32 + grid padding) rather than this specific pruning step, which contributed negligibly (duration actually increased slightly from 3.93ms to 4.04ms in the profiling snapshot).

> **Applicable when**: Never: pruning autotune configs to smaller tiles and lower num_stages does NOT reduce register spill when the accumulator tile (BLOCK_M x BLOCK_N float32 elements) alone saturates the register file. For BLOCK_M=128, BLOCK_N=128, the accumulator requires 128*128*4 bytes = 65,536 bytes = 16,384 float registers, which exceeds the per-thread register budget at any num_stages setting.

> **Avoid when**: When registers/thread is at 255 and the dominant register consumer is the accumulator tile (BLOCK_M * BLOCK_N * dtype_bytes / threads_per_block > register_file_per_thread). In this regime, reducing num_stages or removing large-tile configs from the autotune search space will not reduce register count because the winning config still selects BLOCK_M=128, BLOCK_N=128 which alone drives 255 registers. The shared memory reduction (98KB -> 65KB) is real but does not translate to latency improvement when registers remain the binding occupancy constraint.

---

## Iteration 5

**Best so far**: 2.0356× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_warps_per_block_to_hide_l1tex_spill_latency | 3.245× | 2.041× | ✓ |  |
| B | reduce_accumulator_tile_size_to_break_255_register_ceiling | 3.196× | 2.878× | ✓ | **✓ best** |
| C | force_small_accumulator_tile_to_break_register_ceiling | — | fail | ✓ |  |
| D | smaller_output_tiles_to_break_accumulator_register_saturation | 3.382× | 2.867× | ✓ |  |
| E | add_triton_autotune | 3.120× | 2.051× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_warps_pe... | Δ | reduce_accumulato... | Δ | smaller_output_ti... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8000000 ns | 3920000 ns | -4080000 ns (-51.0%) | 2780000 ns | -5220000 ns (-65.3%) | 2790000 ns | -5210000 ns (-65.1%) | 3900000 ns | -4100000 ns (-51.2%) |
| SM Throughput (%) | 43.30 % | 28.62 % | -14.68 pp | 43.38 % | +0.08 pp | 42.87 % | -0.43 pp | 28.64 % | -14.66 pp |
| Memory Throughput (%) | 59.66 % | 55.14 % | -4.52 pp | 56.29 % | -3.37 pp | 56.33 % | -3.33 pp | 55.40 % | -4.26 pp |
| DRAM Throughput (%) | 0.99 % | 3.74 % | +2.75 pp | 5.40 % | +4.41 pp | 5.30 % | +4.31 pp | 3.80 % | +2.81 pp |
| Achieved Occupancy (%) | 12.49 % | 12.35 % | -0.14 pp | 17.88 % | +5.39 pp | 17.91 % | +5.42 pp | 12.27 % | -0.22 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 140.00 register/thread | -115 | 146.00 register/thread | -109 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 5.17 | +3.23 | 5.17 | +3.23 | 1.94 | +0.00 |
| No Eligible (%) | 54.33 % | 76.02 % | +21.69 pp | 54.85 % | +0.52 pp | 55.30 % | +0.97 pp | 76.04 % | +21.71 pp |
| IPC Active | 1.84 inst/cycle | 0.96 inst/cycle | -0.88 | 1.81 inst/cycle | -0.03 | 1.79 inst/cycle | -0.05 | 0.96 inst/cycle | -0.88 |


### Key Learnings

#### ✅ Plan B: `reduce_accumulator_tile_size_to_break_255_register_ceiling` (2.878×)

When a Triton GEMM kernel hits the 255-register/thread hardware ceiling due to a large accumulator tile (BLOCK_M × BLOCK_N × dtype_bytes), the MemoryCacheAccessPattern NCU rule fires on both local loads and local stores at ~50%+ speedup potential with only 1 byte utilized per 32-byte sector — the definitive signature of register-file overflow spilling the accumulator to L2-backed local memory. The root fix is to reduce BLOCK_M and/or BLOCK_N to 64 in the autotune configs, which cuts the accumulator from 128×128×4=65KB to 64×64×4=16KB, allowing the compiler to fit the kernel under ~140 registers/thread. This raised block_limit_registers from 2 to 3, increased theoretical occupancy from 12.5% to 18.75%, doubled eligible warps per scheduler (0.28→0.70), reduced no-eligible-warp stalls from 76% to 55%, and eliminated the MemoryCacheAccessPattern local load/store rule entirely, yielding a 2.88× speedup (3.9ms→2.78ms). No other strategy (tf32 precision mode, num_stages tuning, grid alignment, block_ptr TMA, config pruning) was able to break the 255-register ceiling because none of them reduced the accumulator tile size.

> **Applicable when**: Triton GEMM kernel where: (1) registers_per_thread is at or near 255 (hardware maximum); (2) MemoryCacheAccessPattern NCU rule fires on BOTH local loads AND local stores with >40% speedup potential and ~1 byte utilized per 32-byte sector; (3) block_limit_registers <= 2 and theoretical_occupancy <= 12.5%; (4) no_eligible_warp_pct > 70%; (5) all autotune configs have BLOCK_M >= 128 AND BLOCK_N >= 128. The accumulator tile size (BLOCK_M × BLOCK_N × 4 bytes) must be the dominant register consumer.

> **Avoid when**: Kernel is already using BLOCK_M <= 64 or BLOCK_N <= 64; registers_per_thread is below 200 (register pressure is not the binding constraint); MemoryCacheAccessPattern fires on global or shared loads rather than local loads (different root cause — not register spill); the problem size is very small and smaller tiles would cause too many waves with insufficient arithmetic intensity per tile; compute throughput is already near peak (>80% SM utilization).

#### ✅ Plan D: `smaller_output_tiles_to_break_accumulator_register_saturation` (2.867×)

When a Triton GEMM kernel is stuck at 255 registers/thread (hardware maximum) with MemoryCacheAccessPattern firing on local loads AND local stores at >50% speedup potential each, the root cause is accumulator register saturation: the BLOCK_M × BLOCK_N float32 accumulator tile consumes the entire register file, forcing the compiler to spill pointer-tracking and pipeline-buffer registers to L2-backed local memory. Reducing BLOCK_M × BLOCK_N from 128×128 to 64×64 (or 64×128 / 128×64) cuts per-thread accumulator register demand by 4× (from ~128 to ~32 float registers), dropping registers/thread from 255 to 146, eliminating local memory spill entirely, and raising theoretical occupancy from 12.5% to 18.75%. The grid grows proportionally (512 → 2048 tiles), increasing waves/SM from 1.94 to 5.17, which more than compensates for the reduced arithmetic intensity per tile — compute throughput rose from 28.6% to 42.9% and duration fell from 3.9ms to 2.79ms (1.4× on this sub-step, 2.87× overall).

> **Applicable when**: Triton GEMM kernel where: (1) registers/thread is at or near 255 (hardware maximum); (2) MemoryCacheAccessPattern fires on local loads AND local stores simultaneously with >40% speedup potential each (indicating accumulator spill to L2-backed local memory, not global memory access issues); (3) block_limit_registers = block_limit_shared_mem = 2 (both resources are binding occupancy constraints); (4) all autotune configs use BLOCK_M >= 128 AND BLOCK_N >= 128; (5) prior attempts to reduce registers via maxnreg, num_stages tuning, or config pruning have failed to break below 255 registers/thread.

> **Avoid when**: Registers/thread is already below ~200 (accumulator is not saturating the register file); MemoryCacheAccessPattern fires only on global loads/stores (different root cause — coalescing issue, not spill); the kernel is already using BLOCK_M=64 or BLOCK_N=64; the problem size is very small such that 4× more tiles creates excessive grid overhead; arithmetic intensity is the primary bottleneck (compute-bound regime where larger tiles are needed for efficiency).

---

## Cumulative Progress

```
Baseline:  8000000 ns  (1.000×)
Iter 1     7940000 ns  (1.008×)  — eviction_policies_streaming_gemm_tiles
Iter 2     4050000 ns  (1.975×)  — tf32_tensor_cores
Iter 3     3930000 ns  (2.036×)  — eliminate_partial_wave_tail_via_grid_aligned_tiling
Iter 5     2780000 ns  (2.878×)  — reduce_accumulator_tile_size_to_break_255_register_ceiling

Final best: 2780000 ns  (2.8777×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,35 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        # BLOCK_M=64, BLOCK_N=64 configs
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # BLOCK_M=64, BLOCK_N=128 configs
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
+        # BLOCK_M=128, BLOCK_N=64 configs
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -58,20 +77,22 @@

         x_tile = tl.load(
             X_ptrs,
             mask=(offs_m[:, None] < M) & (mask_k[None, :]),
-            other=0.0
+            other=0.0,
+            eviction_policy='evict_first'  # single-use per K-iter
         )
         w_tile = tl.load(
             W_ptrs,
             mask=(mask_k[:, None]) & (offs_n[None, :] < N),
-            other=0.0
+            other=0.0,
+            eviction_policy='evict_first'  # single-use per K-iter
         )
 
-        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
+        acc = tl.dot(x_tile, w_tile, acc, input_precision='tf32x3')
 
         X_ptrs += BLOCK_K * stride_xk
         W_ptrs += BLOCK_K * stride_wk
 
-    # Load bias
+    # Load bias (keep default eviction so bias stays in L2 across M-tiles)
     bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
 
     # Add bias -> this is original_x
@@ -89,8 +110,7 @@

     # Store
     out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
     mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
-    tl.store(out_ptrs, result.to(tl.float32), mask=mask_out)
-
+    tl.store(out_ptrs, result.to(tl.float32), mask=mask_out, eviction_policy='evict_first')
 
 def kernel_function(p_gemm_weight, p_gemm_bias, x):
     # p_gemm_weight: [N, K] = [8192, 8192]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
