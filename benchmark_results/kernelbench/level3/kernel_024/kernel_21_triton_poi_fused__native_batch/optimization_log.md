# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2460 ns — **Best latency**: 2340 ns (**1.0513×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`streaming_eviction_policy_for_output_to_improve_broadcast_param_l1_hit_rate`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3,1,1), Waves/SM: 0.00
- **Registers/thread**: 18
- **Duration**: 2460 ns
- **SM Throughput**: 0.05 % | **Memory**: 0.41 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 5.79 %
- **IPC Active**: 0.20 inst/cycle | **No Eligible**: 93.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune_xblock | 0.985× | 0.569× | ✓ |  |
| B | autotune_xblock_for_tiny_pointwise_kernel | 1.041× | 0.817× | ✓ |  |
| C | autotune_xblock_to_increase_sm_utilization | 1.043× | 1.012× | ✓ | **✓ best** |
| D | reduce_xblock_increase_grid_coverage | 1.040× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_f... | Δ | autotune_xblock_t... | Δ | reduce_xblock_inc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 4320 ns | +1860 ns (+75.6%) | 3010 ns | +550 ns (+22.4%) | 2430 ns | -30 ns (-1.2%) | 2400 ns | -60 ns (-2.4%) |
| SM Throughput (%) | 0.05 % | 0.02 % | -0.03 pp | 0.08 % | +0.03 pp | 0.01 % | -0.04 pp | 0.01 % | -0.04 pp |
| Memory Throughput (%) | 0.41 % | 0.38 % | -0.03 pp | 0.36 % | -0.05 pp | 0.40 % | -0.01 pp | 0.41 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.04 % | -0.03 pp | 0.07 % | +0.00 pp | 0.05 % | -0.02 pp | 0.04 % | -0.03 pp |
| Achieved Occupancy (%) | 5.79 % | 1.56 % | -4.23 pp | 5.69 % | -0.10 pp | 1.56 % | -4.23 pp | 1.56 % | -4.23 pp |
| Registers/Thread | 18.00 register/thread | 28.00 register/thread | +10 | 32.00 register/thread | +14 | 19.00 register/thread | +1 | 19.00 register/thread | +1 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.33 % | 92.38 % | -0.95 pp | 85.31 % | -8.02 pp | 93.72 % | +0.39 pp | 94.28 % | +0.95 pp |
| IPC Active | 0.20 inst/cycle | 0.07 inst/cycle | -0.13 | 0.54 inst/cycle | +0.34 | 0.07 inst/cycle | -0.13 | 0.07 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune_xblock` (0.569×)

Adding a persistent while-loop stride pattern to a tiny pointwise kernel (384 elements, 3 CTAs) caused a 1.76x slowdown: duration increased from 2.46us to 4.32us and elapsed cycles from 3687 to 6435. The persistent loop body increased register usage from 18 to 28 registers/thread, which cut theoretical occupancy from 100% to 50% and achieved occupancy from 5.79% to 1.56%. Critically, the grid remained at 3 blocks (autotune did not change the launch grid as intended), so the persistent loop added overhead without distributing work across more SMs — the anti-pattern is applying persistent scheduling to a kernel whose grid is already fixed and too small to benefit.

> **Applicable when**: Never beneficial in this form. The persistent loop pattern is intended for kernels where the launch grid is explicitly set to num_SMs and the loop strides over all tiles. It only helps when grid_size >= sm_count.

> **Avoid when**: The kernel has a tiny total element count (e.g., <10K elements) and the grid remains fixed at a small number of blocks (e.g., 3). Adding a persistent while-loop without also fixing the grid launch to use num_SMs blocks increases register pressure, reduces theoretical occupancy, and adds loop overhead with no SM utilization benefit. Also avoid when autotune configs do not include a grid lambda that scales with num_SMs.

---

## Iteration 2

**Best so far**: 1.0123× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_respond_to_autotuned_xblock | 1.044× | 0.950× | ✓ |  |
| B | sm_count_fixed_grid_persistent_stride_loop | 1.029× | 0.939× | ✓ |  |
| C | minimize_launch_overhead_single_warp_single_block | 1.033× | 1.000× | ✓ |  |
| D | remove_persistent_loop_restore_flat_pointwise_kernel | 1.059× | 1.012× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | sm_count_fixed_gr... | Δ | minimize_launch_o... | Δ | remove_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 2590 ns | +130 ns (+5.3%) | 2620 ns | +160 ns (+6.5%) | 2460 ns | +0 ns (+0.0%) | 2430 ns | -30 ns (-1.2%) |
| SM Throughput (%) | 0.05 % | 0.02 % | -0.03 pp | 0.06 % | +0.01 pp | 0.05 % | +0.00 pp | 0.05 % | +0.00 pp |
| Memory Throughput (%) | 0.41 % | 0.38 % | -0.03 pp | 0.41 % | +0.00 pp | 0.43 % | +0.02 pp | 0.42 % | +0.01 pp |
| DRAM Throughput (%) | 0.07 % | 0.03 % | -0.04 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 5.79 % | 1.56 % | -4.23 pp | 6.20 % | +0.41 pp | 6.16 % | +0.37 pp | 6.19 % | +0.40 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 28.00 register/thread | +10 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.33 % | 95.64 % | +2.31 pp | 93.44 % | +0.11 pp | 93.41 % | +0.08 pp | 93.42 % | +0.09 pp |
| IPC Active | 0.20 inst/cycle | 0.04 inst/cycle | -0.16 | 0.28 inst/cycle | +0.08 | 0.21 inst/cycle | +0.01 | 0.20 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0123× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | single_block_full_width_xblock_384 | 1.039× | 0.961× | ✓ |  |
| B | reduce_instruction_count_via_vectorized_broadcast_loads | 1.036× | 1.000× | ✓ |  |
| C | preload_broadcast_params_into_shared_memory_single_cta | 1.002× | 1.012× | ✓ |  |
| D | arithmetic_simplification_rsqrt_to_reduce_register_pressure_and_instruction_count | 1.032× | 1.051× | ✓ |  |


### Metric Comparison

| Metric | Baseline | single_block_full... | Δ | reduce_instructio... | Δ | preload_broadcast... | Δ | arithmetic_simpli... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 2560 ns | +100 ns (+4.1%) | 2460 ns | +0 ns (+0.0%) | 2430 ns | -30 ns (-1.2%) | 2340 ns | -120 ns (-4.9%) |
| SM Throughput (%) | 0.05 % | 0.11 % | +0.06 pp | 0.05 % | +0.00 pp | 0.05 % | +0.00 pp | 0.04 % | -0.01 pp |
| Memory Throughput (%) | 0.41 % | 0.41 % | +0.00 pp | 0.42 % | +0.01 pp | 0.42 % | +0.01 pp | 0.49 % | +0.08 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 5.79 % | 11.27 % | +5.48 pp | 6.06 % | +0.27 pp | 5.88 % | +0.09 pp | 6.19 % | +0.40 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.33 % | 79.01 % | -14.32 pp | 93.23 % | -0.10 pp | 93.39 % | +0.06 pp | 95.11 % | +1.78 pp |
| IPC Active | 0.20 inst/cycle | 0.80 inst/cycle | +0.60 | 0.25 inst/cycle | +0.05 | 0.22 inst/cycle | +0.02 | 0.18 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.0123× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_output_to_improve_broadcast_param_l1_hit_rate | 1.064× | 1.051× | ✓ | **✓ best** |
| B | arithmetic_simplification_rsqrt_replace_verbose_reciprocal_sqrt_chain | 1.062× | 1.084× | ✓ |  |
| C | arithmetic_simplification_rsqrt_replace_verbose_reciprocal_sqrt_chain | 1.026× | 1.051× | ✓ |  |
| D | fused_arithmetic_simplification_rsqrt_and_remove_broken_autotune | 1.033× | 1.084× | ✓ |  |
| E | add_triton_autotune | 1.062× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | arithmetic_simpli... | Δ | arithmetic_simpli... | Δ | fused_arithmetic_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 2340 ns | -120 ns (-4.9%) | 2270 ns | -190 ns (-7.7%) | 2340 ns | -120 ns (-4.9%) | 2270 ns | -190 ns (-7.7%) | 2430 ns | -30 ns (-1.2%) |
| SM Throughput (%) | 0.05 % | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp |
| Memory Throughput (%) | 0.41 % | 0.46 % | +0.05 pp | 0.43 % | +0.02 pp | 0.43 % | +0.02 pp | 0.45 % | +0.04 pp | 0.41 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.06 % | -0.01 pp |
| Achieved Occupancy (%) | 5.79 % | 6.18 % | +0.39 pp | 6.22 % | +0.43 pp | 5.88 % | +0.09 pp | 6.18 % | +0.39 pp | 6.17 % | +0.38 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.33 % | 95.53 % | +2.20 pp | 95.09 % | +1.76 pp | 95.10 % | +1.77 pp | 95.79 % | +2.46 pp | 96.49 % | +3.16 pp |
| IPC Active | 0.20 inst/cycle | 0.15 inst/cycle | -0.05 | 0.15 inst/cycle | -0.05 | 0.15 inst/cycle | -0.05 | 0.17 inst/cycle | -0.03 | 0.17 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan B: `arithmetic_simplification_rsqrt_replace_verbose_reciprocal_sqrt_chain` (1.084×)

Replacing an explicit `tl.full([1], 1e-05, tl.float32)` temporary with an inline float literal (`tmp3 + 1e-05`) and removing the intermediate `pid` variable reduced elapsed cycles from 3455 to 3352 (2.9% wall-clock improvement) in a tiny 384-element pointwise kernel. The key observable metric change was Average L2 Active Cycles dropping from 422.96 to 273.46 (35% reduction), suggesting the compiler was able to fold the constant more aggressively and reduce L2 pressure. Instruction count remained identical at 600, so the gain came from reduced memory subsystem activity rather than fewer instructions.

> **Applicable when**: Tiny pointwise kernels (< 1K elements) where tl.full([1], constant, dtype) is used to create scalar constant tensors that are immediately consumed in arithmetic; kernel is already instruction-count-optimized (rsqrt applied) and further gains must come from compiler constant folding or reduced memory subsystem pressure; L2 Active Cycles are disproportionately high relative to DRAM cycles.

> **Avoid when**: The tl.full constant is reused multiple times (inlining would duplicate computation); the kernel is large enough that instruction count dominates over L2 pressure; the constant dtype matters for precision (e.g., explicit fp16 vs fp32 casting is required).

#### ✅ Plan C: `arithmetic_simplification_rsqrt_replace_verbose_reciprocal_sqrt_chain` (1.084×)

Replacing an explicit `tl.full([1], 1e-05, tl.float32)` temporary with an inline float literal (`tmp3 + 1e-05`) and removing the intermediate `pid` variable reduced elapsed cycles from 3455 to 3352 (2.9% wall-clock improvement) in a tiny 384-element pointwise kernel. The key observable metric change was Average L2 Active Cycles dropping from 422.96 to 273.46 (35% reduction), suggesting the compiler was able to fold the constant more aggressively and reduce L2 pressure. Instruction count remained identical at 600, so the gain came from reduced memory subsystem activity rather than fewer instructions.

> **Applicable when**: Tiny pointwise kernels (< 1K elements) where tl.full([1], constant, dtype) is used to create scalar constant tensors that are immediately consumed in arithmetic; kernel is already instruction-count-optimized (rsqrt applied) and further gains must come from compiler constant folding or reduced memory subsystem pressure; L2 Active Cycles are disproportionately high relative to DRAM cycles.

> **Avoid when**: The tl.full constant is reused multiple times (inlining would duplicate computation); the kernel is large enough that instruction count dominates over L2 pressure; the constant dtype matters for precision (e.g., explicit fp16 vs fp32 casting is required).

#### ✅ Plan E: `add_triton_autotune` (1.070×)

Adding a @triton.autotune decorator exploring XBLOCK in {128, 256, 512, 1024} with num_warps in {4, 8} on a tiny 384-element pointwise kernel (grid=3, 132 SMs) produced a 1.07x wall-clock speedup despite the NCU profiling metrics showing no meaningful improvement — elapsed cycles actually increased slightly (3352→3387) and achieved occupancy dropped (6.22%→5.67%). The speedup is likely attributable to the autotuner selecting a configuration that reduces kernel launch overhead or host-side scheduling latency rather than any GPU-side metric improvement. This pattern is safe to apply (no kernel body changes, purely mechanical parameter search) but yields only modest gains on severely launch-bound tiny kernels where the fundamental bottleneck (grid << sm_count) cannot be addressed by XBLOCK tuning alone.

> **Applicable when**: Tiny pointwise kernels (< 1K elements, grid < 10 blocks) where XBLOCK is currently hardcoded and no other optimization avenue remains. The kernel body is correct and the only question is whether a different tile size or warp count reduces scheduling overhead. Safe to apply as a last-resort mechanical search with no correctness risk.

> **Avoid when**: The kernel is already severely launch-bound (grid << sm_count) and the grid lambda is not updated to scale with the autotuned XBLOCK — in that case the grid remains fixed and XBLOCK tuning cannot improve SM utilization. Do not expect >1.1x speedup from XBLOCK autotune alone on tiny kernels; the fundamental bottleneck (too few blocks) requires grid-level changes. Also avoid if the autotune overhead (compilation of N configs) is not amortized over enough kernel invocations.

---

## Iteration 5

**Best so far**: 1.0513× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | inline_epsilon_constant_and_fma_fusion | 1.028× | 1.084× | ✓ |  |
| B | fma_fusion_and_evict_first_on_streaming_output_load | 1.025× | 1.070× | ✓ |  |
| C | replace_modulo_with_conditional_subtraction_to_reduce_nonfused_fp_instructions | 1.064× | 0.972× | ✓ |  |
| D | fma_fusion_and_inline_epsilon_constant_folding | 1.063× | 1.084× | ✓ |  |
| E | add_triton_autotune | 1.026× | 1.070× | ✓ |  |


### Metric Comparison

| Metric | Baseline | inline_epsilon_co... | Δ | fma_fusion_and_ev... | Δ | replace_modulo_wi... | Δ | fma_fusion_and_in... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 2270 ns | -190 ns (-7.7%) | 2300 ns | -160 ns (-6.5%) | 2530 ns | +70 ns (+2.8%) | 2270 ns | -190 ns (-7.7%) | 2300 ns | -160 ns (-6.5%) |
| SM Throughput (%) | 0.05 % | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp | 0.04 % | -0.01 pp |
| Memory Throughput (%) | 0.41 % | 0.44 % | +0.03 pp | 0.46 % | +0.05 pp | 0.53 % | +0.12 pp | 0.43 % | +0.02 pp | 0.44 % | +0.03 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 5.79 % | 6.18 % | +0.39 pp | 6.23 % | +0.44 pp | 6.17 % | +0.38 pp | 6.18 % | +0.39 pp | 5.67 % | -0.12 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.33 % | 95.77 % | +2.44 pp | 95.15 % | +1.82 pp | 96.13 % | +2.80 pp | 96.20 % | +2.87 pp | 95.87 % | +2.54 pp |
| IPC Active | 0.20 inst/cycle | 0.17 inst/cycle | -0.03 | 0.15 inst/cycle | -0.05 | 0.20 inst/cycle | +0.00 | 0.18 inst/cycle | -0.02 | 0.15 inst/cycle | -0.05 |


### Key Learnings

#### ✅ Plan A: `inline_epsilon_constant_and_fma_fusion` (1.084×)

Replacing a separate `tl.full([1], 1e-05, tl.float32)` constant tensor with an inline Python float literal and fusing the trailing FMUL+FADD pair into `libdevice.fma(a, b, c)` yielded a 1.08x speedup on a tiny 384-element batch-norm pointwise kernel. The most notable metric change was L2 Compression Input Sectors dropping from 93 to 63 (32% reduction), indicating fewer L2 transactions from eliminating the constant tensor materialization. Elapsed cycles decreased marginally from 3352 to 3336 (~0.5%), while instruction count remained identical at 600, confirming the gain came from reduced memory subsystem pressure rather than fewer instructions. The FMA fusion did not measurably reduce Warp Cycles Per Issued Instruction (20.54→20.67), suggesting the FFMA latency benefit was masked by the kernel's severe launch-bound condition (3 CTAs vs 132 SMs).

> **Applicable when**: Tiny pointwise kernels where tl.full([1], constant, dtype) is used for scalar constants immediately consumed in arithmetic, AND a trailing multiply-add pair (FMUL+FADD) exists that can be fused into FFMA. NCU FPInstructions rule is triggered with high non-fused FP32 instruction count. L2 Compression Input Sectors are disproportionately high relative to actual data volume.

> **Avoid when**: The tl.full constant is reused multiple times (inlining would not reduce materialization cost). The kernel is large enough that instruction count dominates over L2 sector pressure. Strict IEEE-754 double-rounding semantics are required (libdevice.fma uses single rounding vs separate FMUL+FADD). The kernel is so severely launch-bound that even micro-optimizations yield <1% improvement and measurement noise dominates.

#### ✅ Plan D: `fma_fusion_and_inline_epsilon_constant_folding` (1.084×)

Combining libdevice.fma() fusion, inline epsilon constant folding (replacing tl.full([1], 1e-05, tl.float32) with 1e-05), and tl.max_contiguous/tl.multiple_of hints on broadcast index loads yielded a 1.08x speedup on a tiny 384-element batch-norm pointwise kernel. Notably, the profiling metrics did not improve in the expected directions: Warp Cycles Per Issued Instruction increased from 20.54 to 24.33, L1/TEX Cache Throughput increased from 26.71% to 32.35%, and Total SM Elapsed Cycles actually decreased (499042→456812), suggesting the wall-clock gain came from reduced total SM active cycles rather than reduced per-instruction latency. The instruction count remained identical at 600, and L2 metrics were essentially unchanged, so the speedup is likely attributable to the tl.max_contiguous/tl.multiple_of hints enabling better memory coalescing for the broadcast parameter loads, reducing total SM active cycles by ~8.5%.

> **Applicable when**: Tiny pointwise kernels with broadcast parameter loads indexed by a modulo expression (e.g., xindex % C) where the modulus is a power-of-2 or known multiple; kernel also has separate FMUL+FADD pairs that can be fused; NCU FPInstructions rule flags high non-fused FP32 instruction ratio. Most effective when Total SM Elapsed Cycles is the binding constraint rather than per-instruction latency.

> **Avoid when**: The broadcast index is not a clean multiple (tl.multiple_of hint would be incorrect and could cause misaligned loads); the kernel is already using libdevice.fma(); the FP32 precision difference between fma() and separate mul+add matters for correctness. Also avoid when the kernel is large enough that the contiguous hint overhead outweighs the coalescing benefit.

#### ✅ Plan E: `add_triton_autotune` (1.070×)

Adding a @triton.autotune decorator exploring XBLOCK in {128, 256, 512, 1024} with num_warps in {4, 8} on a tiny 384-element pointwise kernel (grid=3, 132 SMs) produced a 1.07x wall-clock speedup despite the NCU profiling metrics showing no meaningful improvement — elapsed cycles actually increased slightly (3352→3387) and achieved occupancy dropped (6.22%→5.67%). The speedup is likely attributable to the autotuner selecting a configuration that reduces kernel launch overhead or host-side scheduling latency rather than any GPU-side metric improvement. This pattern is safe to apply (no kernel body changes, purely mechanical parameter search) but yields only modest gains on severely launch-bound tiny kernels where the fundamental bottleneck (grid << sm_count) cannot be addressed by XBLOCK tuning alone.

> **Applicable when**: Tiny pointwise kernels (< 1K elements, grid < 10 blocks) where XBLOCK is currently hardcoded and no other optimization avenue remains. The kernel body is correct and the only question is whether a different tile size or warp count reduces scheduling overhead. Safe to apply as a last-resort mechanical search with no correctness risk.

> **Avoid when**: The kernel is already severely launch-bound (grid << sm_count) and the grid lambda is not updated to scale with the autotuned XBLOCK — in that case the grid remains fixed and XBLOCK tuning cannot improve SM utilization. Do not expect >1.1x speedup from XBLOCK autotune alone on tiny kernels; the fundamental bottleneck (too few blocks) requires grid-level changes. Also avoid if the autotune overhead (compilation of N configs) is not amortized over enough kernel invocations.

---

## Cumulative Progress

```
Baseline:  2460 ns  (1.000×)
Iter 1     2430 ns  (1.012×)  — autotune_xblock_to_increase_sm_utilization
Iter 4     2340 ns  (1.051×)  — streaming_eviction_policy_for_output_to_improve_broadcast_param_l1_hit_rate

Final best: 2340 ns  (1.0513×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -21,17 +22,11 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    inv_std = tl.math.rsqrt(tmp5)
+    tmp11 = tmp2 * inv_std
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
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
