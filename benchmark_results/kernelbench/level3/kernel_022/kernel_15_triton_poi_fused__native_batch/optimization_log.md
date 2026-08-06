# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8100 ns — **Best latency**: 8100 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_15_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8820,1,1), Waves/SM: 4.18
- **Registers/thread**: 18
- **Duration**: 8100 ns
- **SM Throughput**: 46.88 % | **Memory**: 18.59 % | **DRAM**: 11.42 %
- **Achieved Occupancy**: 59.72 %
- **IPC Active**: 2.60 inst/cycle | **No Eligible**: 33.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_improve_occupancy | 0.803× | 0.767× | ✓ |  |
| B | autotune_xblock_for_occupancy_and_l1_reuse | 0.815× | 0.765× | ✓ |  |
| C | increase_xblock_and_add_autotune_for_vectorized_loads | 0.977× | 0.258× | ✓ |  |
| D | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_improve_occupancy | 0.978× | 0.258× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_f... | Δ | increase_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8100 ns | 10560 ns | +2460 ns (+30.4%) | 10590 ns | +2490 ns (+30.7%) | 31390 ns | +23290 ns (+287.5%) | 31390 ns | +23290 ns (+287.5%) |
| SM Throughput (%) | 46.88 % | 66.54 % | +19.66 pp | 66.23 % | +19.35 pp | 80.58 % | +33.70 pp | 80.61 % | +33.73 pp |
| Memory Throughput (%) | 18.59 % | 20.08 % | +1.49 pp | 19.99 % | +1.40 pp | 16.83 % | -1.76 pp | 16.81 % | -1.78 pp |
| DRAM Throughput (%) | 11.42 % | 8.76 % | -2.66 pp | 8.72 % | -2.70 pp | 2.94 % | -8.48 pp | 2.94 % | -8.48 pp |
| Achieved Occupancy (%) | 59.72 % | 82.50 % | +22.78 pp | 82.34 % | +22.62 pp | 55.58 % | -4.14 pp | 55.64 % | -4.08 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 48.00 register/thread | +30 | 48.00 register/thread | +30 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 13.36 | +9.18 | 13.36 | +9.18 |
| No Eligible (%) | 33.72 % | 18.28 % | -15.44 pp | 18.44 % | -15.28 pp | 13.77 % | -19.95 pp | 13.59 % | -20.13 pp |
| IPC Active | 2.60 inst/cycle | 3.21 inst/cycle | +0.61 | 3.23 inst/cycle | +0.63 | 3.45 inst/cycle | +0.85 | 3.45 inst/cycle | +0.85 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_reduce_partial_wave_and_improve_occupancy` (0.767×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (18→32), which did not reduce occupancy (it actually improved from 59.7%→82.5%) but nearly doubled the total executed instruction count (3.03M→5.55M), increasing kernel duration from 8.1µs to 10.56µs. The instruction explosion suggests the autotuner selected a config that caused the compiler to unroll loops or generate redundant computation, not that the tile size itself was wrong. When the grid is fixed (hardcoded) rather than updated to use the meta lambda, the autotune configs are ignored at launch but the compiler still generates code for all configs, inflating register usage and instruction count.

> **Applicable when**: Never apply this specific pattern (autotune decorator with fixed hardcoded grid). This is an implementation flaw, not a valid strategy.

> **Avoid when**: Do not add @triton.autotune without also updating the grid to use a meta lambda (lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)). A mismatched autotune+fixed-grid causes the compiler to generate bloated multi-config code while the runtime still uses the original tile size, increasing instruction count by ~83% and duration by ~30% with no benefit.

#### ⚠️ Plan B: `autotune_xblock_for_occupancy_and_l1_reuse` (0.765×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused the compiler to allocate more registers per thread (18→32), increasing instruction count by ~83% (3.03M→5.55M executed instructions) and elapsed cycles by ~31% (12050→15794), despite occupancy improving from 59.7% to 82.3%. The larger tile sizes forced the compiler to keep more live values in registers simultaneously, inflating register pressure and total work beyond what the occupancy gain could offset. When a pointwise kernel already has low register usage (≤20/thread) and the bottleneck is occupancy/scheduling, increasing XBLOCK via autotune can backfire if the selected config causes the compiler to emit significantly more instructions due to register spilling or loop unrolling overhead.

> **Applicable when**: Kernel has low achieved occupancy (<65%) due to scheduling stalls, low eligible warps per scheduler, and registers per thread are reported as low (≤20). Partial-wave overhead is significant (>15%). Kernel is purely pointwise with no shared memory.

> **Avoid when**: The compiler's register allocation is sensitive to XBLOCK size — even if static analysis suggests headroom (e.g., 18 regs/thread at XBLOCK=128), larger XBLOCK values can cause the compiler to allocate significantly more registers (e.g., 32/thread at XBLOCK=1024+), inflating instruction count and negating occupancy gains. Do not apply when instruction count increase outpaces the occupancy improvement. Also avoid when the grid is already fixed and autotune cannot override it (the diff shows the grid was not updated to use meta['XBLOCK'], remaining hardcoded to cdiv(xnumel, 128)).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_arithmetic_to_expose_fma_instructions | 0.941× | 0.944× | ✓ |  |
| B | enable_fma_fusion_by_replacing_int_division_with_fp32_rsqrt | 0.961× | 0.941× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_and_small_xblock_autotune_configs | 1.010× | 0.989× | ✓ | **✓ best** |
| D | fix_grid_lambda_and_reduce_register_pressure_via_small_xblock_configs | 0.997× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rewrite_arithmeti... | Δ | enable_fma_fusion... | Δ | reduce_register_p... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8100 ns | 8580 ns | +480 ns (+5.9%) | 8610 ns | +510 ns (+6.3%) | 8190 ns | +90 ns (+1.1%) | 8100 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 46.88 % | 37.49 % | -9.39 pp | 37.57 % | -9.31 pp | 47.16 % | +0.28 pp | 45.89 % | -0.99 pp |
| Memory Throughput (%) | 18.59 % | 26.59 % | +8.00 pp | 26.55 % | +7.96 pp | 18.53 % | -0.06 pp | 18.56 % | -0.03 pp |
| DRAM Throughput (%) | 11.42 % | 10.81 % | -0.61 pp | 10.76 % | -0.66 pp | 11.26 % | -0.16 pp | 11.41 % | -0.01 pp |
| Achieved Occupancy (%) | 59.72 % | 42.63 % | -17.09 pp | 43.51 % | -16.21 pp | 59.82 % | +0.10 pp | 59.21 % | -0.51 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.72 % | 43.58 % | +9.86 pp | 43.17 % | +9.45 pp | 33.79 % | +0.07 pp | 36.73 % | +3.01 pp |
| IPC Active | 2.60 inst/cycle | 2.25 inst/cycle | -0.35 | 2.26 inst/cycle | -0.34 | 2.61 inst/cycle | +0.01 | 2.54 inst/cycle | -0.06 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_division_with_rsqrt_multiply_to_expose_fma_instructions | 1.030× | 1.025× | ✓ | **✓ best** |
| B | replace_integer_division_with_rsqrt_and_fuse_multiply_chain_for_fma | 1.029× | 1.025× | ✓ |  |
| C | streaming_eviction_policy_for_large_tensor_plus_fix_grid_lambda | 1.016× | 1.009× | ✓ |  |
| D | replace_libdevice_rsqrt_with_native_tl_rsqrt_and_hoist_channel_invariant_computation | 0.961× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_divi... | Δ | replace_integer_d... | Δ | streaming_evictio... | Δ | replace_libdevice... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8100 ns | 7900 ns | -200 ns (-2.5%) | 7900 ns | -200 ns (-2.5%) | 8030 ns | -70 ns (-0.9%) | 8510 ns | +410 ns (+5.1%) |
| SM Throughput (%) | 46.88 % | 29.04 % | -17.84 pp | 28.93 % | -17.95 pp | 18.06 % | -28.82 pp | 37.47 % | -9.41 pp |
| Memory Throughput (%) | 18.59 % | 19.00 % | +0.41 pp | 18.89 % | +0.30 pp | 17.87 % | -0.72 pp | 26.70 % | +8.11 pp |
| DRAM Throughput (%) | 11.42 % | 11.66 % | +0.24 pp | 11.70 % | +0.28 pp | 11.50 % | +0.08 pp | 10.84 % | -0.58 pp |
| Achieved Occupancy (%) | 59.72 % | 47.37 % | -12.35 pp | 45.57 % | -14.15 pp | 36.81 % | -22.91 pp | 43.25 % | -16.47 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 24.00 register/thread | +6 | 32.00 register/thread | +14 |
| Waves/SM | 4.18 | 4.18 | +0.00 | 4.18 | +0.00 | 2.09 | -2.09 | 4.18 | +0.00 |
| No Eligible (%) | 33.72 % | 57.63 % | +23.91 pp | 57.82 % | +24.10 pp | 66.76 % | +33.04 pp | 43.31 % | +9.59 pp |
| IPC Active | 2.60 inst/cycle | 1.65 inst/cycle | -0.95 | 1.65 inst/cycle | -0.95 | 1.26 inst/cycle | -1.34 | 2.23 inst/cycle | -0.37 |

---

## Iteration 4

**Best so far**: 1.0253× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_without_changing_xblock | 1.028× | 0.477× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_latency_without_xblock_increase | 0.993× | 0.885× | ✓ |  |
| C | fix_grid_lambda_and_increase_num_warps_for_l1tex_latency_hiding | 1.027× | 1.012× | ✓ |  |
| D | fix_autotune_grid_lambda_and_add_small_xblock_configs_to_eliminate_partial_wave | 0.982× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_grid_lambda_a... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8100 ns | 16990 ns | +8890 ns (+109.8%) | 9150 ns | +1050 ns (+13.0%) | 8000 ns | -100 ns (-1.2%) | 8060 ns | -40 ns (-0.5%) |
| SM Throughput (%) | 46.88 % | 53.37 % | +6.49 pp | 48.81 % | +1.93 pp | 30.61 % | -16.27 pp | 28.32 % | -18.56 pp |
| Memory Throughput (%) | 18.59 % | 30.82 % | +12.23 pp | 29.82 % | +11.23 pp | 16.68 % | -1.91 pp | 18.97 % | +0.38 pp |
| DRAM Throughput (%) | 11.42 % | 5.42 % | -6.00 pp | 10.09 % | -1.33 pp | 5.78 % | -5.64 pp | 11.46 % | +0.04 pp |
| Achieved Occupancy (%) | 59.72 % | 75.69 % | +15.97 pp | 73.61 % | +13.89 pp | 45.99 % | -13.73 pp | 45.93 % | -13.79 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 19.00 register/thread | +1 | 19.00 register/thread | +1 | 16.00 register/thread | -2 |
| Waves/SM | 4.18 | 16.70 | +12.52 | 8.35 | +4.17 | 4.18 | +0.00 | 4.18 | +0.00 |
| No Eligible (%) | 33.72 % | 37.12 % | +3.40 pp | 35.64 % | +1.92 pp | 54.16 % | +20.44 pp | 59.06 % | +25.34 pp |
| IPC Active | 2.60 inst/cycle | 2.51 inst/cycle | -0.09 | 2.54 inst/cycle | -0.06 | 1.79 inst/cycle | -0.81 | 1.64 inst/cycle | -0.96 |


### Key Learnings

#### ❌ Plan A: `increase_num_warps_to_hide_l1tex_latency_without_changing_xblock` (0.477×)

Increasing num_warps from 4 to 16 (block_size 128→512) while keeping XBLOCK=128 fixed caused a 3.9x explosion in executed instructions (1.83M→7.16M) and doubled kernel duration (7.9→17.0 µs), despite improving achieved occupancy (47.4%→75.7%) and eligible warps per scheduler (1.01→2.84). The autotuner selected a config with block_size=512 (num_warps=16), which caused Block Limit Registers to drop from 21 to 5 blocks/SM even though registers per thread only increased marginally (18→19). The waves_per_sm ballooned from 4.18 to 16.7, indicating the grid lambda correctly adapted to the winning XBLOCK but the winning config (XBLOCK=64, num_warps=8 or similar) produced far more blocks than expected, multiplying total work. When a kernel's bottleneck is L1TEX stall latency and occupancy is already moderate, simply adding more warps per block does not reduce per-element work and can dramatically increase total instruction count if the autotuner selects a smaller XBLOCK config with more warps.

> **Applicable when**: Never: increasing num_warps alone (without reducing per-thread work) does not reduce total instruction count and can cause the autotuner to select a config that multiplies waves_per_sm and total instructions. Document as anti-pattern for L1TEX-stall kernels with moderate occupancy.

> **Avoid when**: Do not increase num_warps via autotune configs when: (1) the kernel's total instruction count is already proportional to grid_size × block_size — adding more warps per block multiplies total instructions if the autotuner selects a config with smaller XBLOCK; (2) Block Limit Registers is already constrained (≤21 blocks/SM at 18 regs/thread) — larger block sizes (512 threads) can drop Block Limit Registers from 21 to 5, collapsing SM-level parallelism; (3) waves_per_sm is already low (4.18) — selecting a config that increases waves_per_sm to 16.7 means 4x more kernel launches with no reduction in per-element work. The correct fix for L1TEX stall with low eligible warps is to reduce memory access latency (improve coalescing, increase L1 hit rate) rather than adding more warps.

---

## Iteration 5

**Best so far**: 1.0253× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_channel_invariant_bn_params_into_shared_memory | 0.855× | 0.846× | ✓ |  |
| B | fix_broken_autotune_grid_lambda_to_enable_correct_config_selection | 0.983× | 1.020× | ✓ |  |
| C | align_xblock_to_channel_size_for_perfect_channel_param_reuse | 1.030× | 1.000× | ✓ | **✓ best** |
| D | replace_libdevice_rsqrt_with_tl_math_rsqrt_and_align_xblock_to_channel_width | 0.997× | 0.480× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_channel_i... | Δ | fix_broken_autotu... | Δ | align_xblock_to_c... | Δ | replace_libdevice... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8100 ns | 9570 ns | +1470 ns (+18.1%) | 7940 ns | -160 ns (-2.0%) | 8100 ns | +0 ns (+0.0%) | 16860 ns | +8760 ns (+108.1%) |
| SM Throughput (%) | 46.88 % | 97.00 % | +50.12 pp | 28.21 % | -18.67 pp | 29.01 % | -17.87 pp | 54.35 % | +7.47 pp |
| Memory Throughput (%) | 18.59 % | 55.31 % | +36.72 pp | 18.92 % | +0.33 pp | 18.91 % | +0.32 pp | 31.44 % | +12.85 pp |
| DRAM Throughput (%) | 11.42 % | 9.67 % | -1.75 pp | 11.62 % | +0.20 pp | 11.38 % | -0.04 pp | 5.46 % | -5.96 pp |
| Achieved Occupancy (%) | 59.72 % | 78.52 % | +18.80 pp | 46.78 % | -12.94 pp | 46.68 % | -13.04 pp | 75.32 % | +15.60 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 19.00 register/thread | +1 |
| Waves/SM | 4.18 | 8.35 | +4.17 | 4.18 | +0.00 | 4.18 | +0.00 | 16.70 | +12.52 |
| No Eligible (%) | 33.72 % | 28.76 % | -4.96 pp | 58.66 % | +24.94 pp | 57.72 % | +24.00 pp | 37.17 % | +3.45 pp |
| IPC Active | 2.60 inst/cycle | 2.92 inst/cycle | +0.32 | 1.63 inst/cycle | -0.97 | 1.69 inst/cycle | -0.91 | 2.47 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan A: `preload_channel_invariant_bn_params_into_shared_memory` (0.846×)

Preloading per-channel batch normalization parameters into register arrays via tl.gather (replacing per-element L1TEX loads) caused a 2.25x explosion in executed instructions (1.96M → 4.41M) and increased kernel duration from 8.0 µs to 9.57 µs, despite improving achieved occupancy (46% → 78.5%), eligible warps per scheduler (1.14 → 3.48), and compute SM throughput (30.6% → 97%). The prologue preload (4 masked loads + rsqrt over XBLOCK=256 lanes) added substantial instruction overhead that outweighed the per-element stall reduction. Additionally, the approach introduced shared memory bank conflicts (SharedMemoryConflicts rule triggered, 12.71% of wavefronts conflicted) and uncoalesced shared accesses, indicating tl.gather internally routes through shared memory with non-ideal access patterns. The net effect is that the memory pipe became the new bottleneck (Mem Pipes Busy 26.9% → 97%) while total work more than doubled.

> **Applicable when**: Never apply tl.gather-based register preloading for small channel-invariant parameter arrays when the baseline already achieves high L1TEX hit rate (>85%) via evict_last. The prologue overhead and gather-induced shared memory bank conflicts outweigh the per-element stall reduction.

> **Avoid when**: Do not use tl.gather to preload small (≤256 element) channel-invariant arrays from L1TEX into register arrays when: (1) the baseline L1/TEX hit rate is already high (>85%) — the parameters are effectively cached and per-element loads are cheap; (2) XBLOCK is large (≥256) — the prologue masked loads over XBLOCK lanes add significant instruction overhead proportional to XBLOCK; (3) the gather operation routes through shared memory — this introduces bank conflicts (SharedMemoryConflicts rule) that can consume 8-13% of memory bandwidth; (4) total instruction count is sensitive — the prologue adds O(XBLOCK) instructions per block that multiply across the grid. The strategy is also inapplicable when the resulting Mem Pipes Busy approaches 100%, shifting the bottleneck from compute stalls to memory pipe saturation.

#### ❌ Plan D: `replace_libdevice_rsqrt_with_tl_math_rsqrt_and_align_xblock_to_channel_width` (0.480×)

Replacing libdevice.rsqrt with tl.math.rsqrt and simultaneously changing autotune configs to include higher num_warps options caused the autotuner to select a block_size=512 config, which dropped Block Limit Registers from 21 to 5 blocks/SM (despite registers/thread staying at 19) and caused a 3.7x explosion in executed instructions (1.96M → 7.16M). The rsqrt substitution itself did not reduce CPIStall stall cycles per instruction (15.97 → 19.36, actually worse), and the dominant regression driver was the same autotune-selects-high-num_warps anti-pattern seen in prior iterations: waves_per_sm ballooned from 4.18 to 16.7, multiplying total work with no per-element benefit. When the autotune config space includes high num_warps options (e.g., num_warps=16 with XBLOCK=128), the autotuner consistently selects these on this kernel, causing instruction count to scale with block_size rather than element count.

> **Applicable when**: Never: this combination of rsqrt swap + autotune config expansion consistently triggers the high-num_warps selection anti-pattern on this kernel class, causing instruction explosion regardless of the rsqrt change.

> **Avoid when**: Do not add autotune configs with num_warps > 4 for this kernel when XBLOCK <= 128. The autotuner consistently selects high num_warps configs (e.g., num_warps=16, block_size=512) which drops Block Limit Registers from 21 to 5 and multiplies executed instructions by ~3.7x. The libdevice.rsqrt → tl.math.rsqrt substitution alone did not reduce CPIStall stall cycles (15.97 → 19.36 cycles/instruction, worsened). Do not conflate the rsqrt change with the autotune config change — the regression is driven by the autotune config space, not the rsqrt instruction.

---

## Cumulative Progress

```
Baseline:  8100 ns  (1.000×)
Iter 3     7900 ns  (1.025×)  — replace_sqrt_division_with_rsqrt_multiply_to_expose_fma_instructions

Final best: 8100 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1128960
@@ -21,21 +33,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp10 = libdevice.rsqrt(tmp5)
+    tmp15 = tl.fma(tmp2 * tmp10, tmp12, tmp14)
     tmp16 = tl.full([1], 0.0, tl.float32)
     tmp17 = tl.maximum(tmp15, tmp16)
     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +49,7 @@

     arg_4 = inputs[4]
     xnumel = 1128960
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
