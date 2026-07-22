# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13760 ns — **Best latency**: 6400 ns (**2.1500×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`force_fma_and_rcp_to_eliminate_non_fused_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_047/kernel_1_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1600,1,1), Waves/SM: 1.52
- **Registers/thread**: 64
- **Duration**: 13760 ns
- **SM Throughput**: 51.63 % | **Memory**: 44.65 % | **DRAM**: 40.78 %
- **Achieved Occupancy**: 37.65 %
- **IPC Active**: 2.53 inst/cycle | **No Eligible**: 35.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.999× | 1.002× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.994× | 1.014× | ✓ |  |
| C | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_reduce_register_pressure | 1.653× | 1.225× | ✓ | **✓ best** |
| D | increase_xblock_eliminate_partial_wave_and_add_vectorized_loads | 0.299× | 0.166× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_xblock_to_... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13760 ns | 13730 ns | -30 ns (-0.2%) | 13570 ns | -190 ns (-1.4%) | 11230 ns | -2530 ns (-18.4%) | 82780 ns | +69020 ns (+501.6%) |
| SM Throughput (%) | 51.63 % | 51.57 % | -0.06 pp | 50.82 % | -0.81 pp | 36.82 % | -14.81 pp | 36.04 % | -15.59 pp |
| Memory Throughput (%) | 44.65 % | 44.71 % | +0.06 pp | 45.10 % | +0.45 pp | 18.14 % | -26.51 pp | 22.14 % | -22.51 pp |
| DRAM Throughput (%) | 40.78 % | 41.70 % | +0.92 pp | 41.34 % | +0.56 pp | 11.96 % | -28.82 pp | 10.01 % | -30.77 pp |
| Achieved Occupancy (%) | 37.65 % | 37.42 % | -0.23 pp | 37.92 % | +0.27 pp | 22.67 % | -14.98 pp | 12.01 % | -25.64 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 91.00 register/thread | +27 | 255.00 register/thread | +191 |
| Waves/SM | 1.52 | 1.52 | +0.00 | 1.52 | +0.00 | 1.21 | -0.31 | 6.06 | +4.54 |
| No Eligible (%) | 35.56 % | 34.60 % | -0.96 pp | 34.92 % | -0.64 pp | 48.11 % | +12.55 pp | 61.33 % | +25.77 pp |
| IPC Active | 2.53 inst/cycle | 2.53 inst/cycle | +0.00 | 2.55 inst/cycle | +0.02 | 1.97 inst/cycle | -0.56 | 1.54 inst/cycle | -0.99 |

---

## Iteration 2

**Best so far**: 1.2253× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | online_softmax_attention | 1.699× | 1.165× | ✓ | **✓ best** |
| B | online_softmax_attention | 1.653× | 1.205× | ✓ |  |
| C | online_softmax_attention | 1.002× | 0.986× | ✓ |  |
| D | fix_register_spill_by_replacing_large_xblock_autotune_configs_with_small_xblock_configs | 1.653× | 1.201× | ✓ |  |
| E | add_triton_autotune | 1.657× | 1.201× | ✓ |  |


### Metric Comparison

| Metric | Baseline | online_softmax_at... | Δ | online_softmax_at... | Δ | online_softmax_at... | Δ | fix_register_spil... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13760 ns | 11810 ns | -1950 ns (-14.2%) | 11420 ns | -2340 ns (-17.0%) | 13950 ns | +190 ns (+1.4%) | 11460 ns | -2300 ns (-16.7%) | 11460 ns | -2300 ns (-16.7%) |
| SM Throughput (%) | 51.63 % | 32.18 % | -19.45 pp | 36.80 % | -14.83 pp | 51.51 % | -0.12 pp | 36.19 % | -15.44 pp | 36.72 % | -14.91 pp |
| Memory Throughput (%) | 44.65 % | 16.62 % | -28.03 pp | 18.18 % | -26.47 pp | 43.89 % | -0.76 pp | 17.87 % | -26.78 pp | 18.15 % | -26.50 pp |
| DRAM Throughput (%) | 40.78 % | 11.36 % | -29.42 pp | 11.74 % | -29.04 pp | 40.93 % | +0.15 pp | 11.73 % | -29.05 pp | 11.70 % | -29.08 pp |
| Achieved Occupancy (%) | 37.65 % | 32.67 % | -4.98 pp | 22.79 % | -14.86 pp | 37.51 % | -0.14 pp | 22.66 % | -14.99 pp | 23.40 % | -14.25 pp |
| Registers/Thread | 64.00 register/thread | 78.00 register/thread | +14 | 91.00 register/thread | +27 | 64.00 register/thread | +0 | 91.00 register/thread | +27 | 91.00 register/thread | +27 |
| Waves/SM | 1.52 | 1.01 | -0.51 | 1.21 | -0.31 | 1.52 | +0.00 | 1.21 | -0.31 | 1.21 | -0.31 |
| No Eligible (%) | 35.56 % | 45.06 % | +9.50 pp | 48.47 % | +12.91 pp | 35.42 % | -0.14 pp | 48.49 % | +12.93 pp | 49.53 % | +13.97 pp |
| IPC Active | 2.53 inst/cycle | 2.07 inst/cycle | -0.46 | 1.97 inst/cycle | -0.56 | 2.52 inst/cycle | -0.01 | 1.91 inst/cycle | -0.62 | 2.03 inst/cycle | -0.50 |

---

## Iteration 3

**Best so far**: 1.2253× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_softmax_intermediates_to_reduce_register_pressure | 1.684× | 1.194× | ✓ |  |
| B | eliminate_redundant_softmax_intermediates_to_reduce_register_pressure | 1.702× | 1.188× | ✓ | **✓ best** |
| C | eliminate_redundant_softmax_intermediates_to_reduce_register_pressure | 1.700× | 1.178× | ✓ |  |
| D | replace_int_div_sqrt_with_rsqrt_to_enable_fma_fusion | 0.986× | 0.905× | ✓ |  |
| E | add_triton_autotune | 0.512× | 0.344× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | eliminate_redunda... | Δ | eliminate_redunda... | Δ | replace_int_div_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13760 ns | 11520 ns | -2240 ns (-16.3%) | 11580 ns | -2180 ns (-15.8%) | 11680 ns | -2080 ns (-15.1%) | 15200 ns | +1440 ns (+10.5%) | 40030 ns | +26270 ns (+190.9%) |
| SM Throughput (%) | 51.63 % | 32.62 % | -19.01 pp | 32.73 % | -18.90 pp | 32.35 % | -19.28 pp | 38.45 % | -13.18 pp | 41.36 % | -10.27 pp |
| Memory Throughput (%) | 44.65 % | 16.82 % | -27.83 pp | 17.30 % | -27.35 pp | 16.72 % | -27.93 pp | 39.72 % | -4.93 pp | 20.66 % | -23.99 pp |
| DRAM Throughput (%) | 40.78 % | 11.61 % | -29.17 pp | 11.58 % | -29.20 pp | 11.46 % | -29.32 pp | 37.20 % | -3.58 pp | 14.11 % | -26.67 pp |
| Achieved Occupancy (%) | 37.65 % | 32.54 % | -5.11 pp | 32.63 % | -5.02 pp | 32.76 % | -4.89 pp | 38.99 % | +1.34 pp | 17.41 % | -20.24 pp |
| Registers/Thread | 64.00 register/thread | 78.00 register/thread | +14 | 78.00 register/thread | +14 | 78.00 register/thread | +14 | 64.00 register/thread | +0 | 167.00 register/thread | +103 |
| Waves/SM | 1.52 | 1.01 | -0.51 | 1.01 | -0.51 | 1.01 | -0.51 | 1.52 | +0.00 | 4.04 | +2.52 |
| No Eligible (%) | 35.56 % | 45.78 % | +10.22 pp | 47.06 % | +11.50 pp | 46.69 % | +11.13 pp | 49.59 % | +14.03 pp | 52.30 % | +16.74 pp |
| IPC Active | 2.53 inst/cycle | 2.03 inst/cycle | -0.50 | 2.07 inst/cycle | -0.46 | 1.98 inst/cycle | -0.55 | 1.96 inst/cycle | -0.57 | 1.91 inst/cycle | -0.62 |


### Key Learnings

#### ✅ Plan A: `eliminate_redundant_softmax_intermediates_to_reduce_register_pressure` (1.188×)

When a Triton softmax kernel materializes redundant intermediate tensors—multiple tl.broadcast_to calls on the same value and a second tl.exp on an already-exponentiated tensor—the excess live values inflate register pressure. Replacing the multi-variable chain (double broadcast + double exp) with a minimal 4-variable form (row_max, shifted, exps, row_sum) reduced registers/thread from 91 to 78, raised Block Limit Registers from 10 to 12 blocks/SM, and improved theoretical occupancy from 31.25% to 37.5% and achieved occupancy from 22.79% to 32.63%. The instruction count also dropped from 3.28M to 2.96M, confirming the compiler could not CSE the redundant broadcast intermediates on its own.

> **Applicable when**: Triton softmax kernel has multiple tl.broadcast_to calls on the same tensor and/or two separate tl.exp calls on provably identical arguments; TheoreticalOccupancy NCU rule fires with register-limited Block Limit Registers; registers_per_thread >= 85; executed instruction count decreases after the rewrite (confirming compiler did not already CSE the redundancy).

> **Avoid when**: The kernel already uses a minimal single-pass softmax form; the two exp arguments differ in any way (different masks, dtypes, or intermediate stores); registers_per_thread is already low (<64) and occupancy is not register-limited; the broadcast intermediates are needed for correctness (e.g., different shapes or dtypes).

#### ✅ Plan B: `eliminate_redundant_softmax_intermediates_to_reduce_register_pressure` (1.188×)

When a Triton softmax kernel materializes redundant intermediate tensors—multiple tl.broadcast_to calls on the same value and a second tl.exp on an already-exponentiated tensor—the excess live values inflate register pressure. Replacing the multi-variable chain (double broadcast + double exp) with a minimal 4-variable form (row_max, shifted, exps, row_sum) reduced registers/thread from 91 to 78, raised Block Limit Registers from 10 to 12 blocks/SM, and improved theoretical occupancy from 31.25% to 37.5% and achieved occupancy from 22.79% to 32.63%. The instruction count also dropped from 3.28M to 2.96M, confirming the compiler could not CSE the redundant broadcast intermediates on its own.

> **Applicable when**: Triton softmax kernel has multiple tl.broadcast_to calls on the same tensor and/or two separate tl.exp calls on provably identical arguments; TheoreticalOccupancy NCU rule fires with register-limited Block Limit Registers; registers_per_thread >= 85; executed instruction count decreases after the rewrite (confirming compiler did not already CSE the redundancy).

> **Avoid when**: The kernel already uses a minimal single-pass softmax form; the two exp arguments differ in any way (different masks, dtypes, or intermediate stores); registers_per_thread is already low (<64) and occupancy is not register-limited; the broadcast intermediates are needed for correctness (e.g., different shapes or dtypes).

#### ✅ Plan C: `eliminate_redundant_softmax_intermediates_to_reduce_register_pressure` (1.188×)

When a Triton softmax kernel materializes redundant intermediate tensors—multiple tl.broadcast_to calls on the same value and a second tl.exp on an already-exponentiated tensor—the excess live values inflate register pressure. Replacing the multi-variable chain (double broadcast + double exp) with a minimal 4-variable form (row_max, shifted, exps, row_sum) reduced registers/thread from 91 to 78, raised Block Limit Registers from 10 to 12 blocks/SM, and improved theoretical occupancy from 31.25% to 37.5% and achieved occupancy from 22.79% to 32.63%. The instruction count also dropped from 3.28M to 2.96M, confirming the compiler could not CSE the redundant broadcast intermediates on its own.

> **Applicable when**: Triton softmax kernel has multiple tl.broadcast_to calls on the same tensor and/or two separate tl.exp calls on provably identical arguments; TheoreticalOccupancy NCU rule fires with register-limited Block Limit Registers; registers_per_thread >= 85; executed instruction count decreases after the rewrite (confirming compiler did not already CSE the redundancy).

> **Avoid when**: The kernel already uses a minimal single-pass softmax form; the two exp arguments differ in any way (different masks, dtypes, or intermediate stores); registers_per_thread is already low (<64) and occupancy is not register-limited; the broadcast intermediates are needed for correctness (e.g., different shapes or dtypes).

---

## Iteration 4

**Best so far**: 1.2253× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_arithmetic_via_rsqrt_and_eliminate_redundant_multiply | 2.128× | 2.170× | ✓ |  |
| B | enable_fma_fusion_by_replacing_int_div_with_float_rsqrt_in_batchnorm_normalization | 2.154× | 2.150× | ✓ | **✓ best** |
| C | replace_int_div_sqrt_chain_with_rsqrt_to_enable_fma_fusion_and_reduce_register_pressure | 2.130× | 2.117× | ✓ |  |
| D | fuse_bn_arithmetic_into_fma_and_eliminate_redundant_broadcasts | 1.008× | 0.979× | ✓ |  |
| E | add_triton_autotune | 0.527× | 0.350× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_ar... | Δ | enable_fma_fusion... | Δ | replace_int_div_s... | Δ | fuse_bn_arithmeti... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13760 ns | 6340 ns | -7420 ns (-53.9%) | 6400 ns | -7360 ns (-53.5%) | 6500 ns | -7260 ns (-52.8%) | 14050 ns | +290 ns (+2.1%) | 39330 ns | +25570 ns (+185.8%) |
| SM Throughput (%) | 51.63 % | 28.42 % | -23.21 pp | 28.70 % | -22.93 pp | 28.65 % | -22.98 pp | 34.69 % | -16.94 pp | 40.03 % | -11.60 pp |
| Memory Throughput (%) | 44.65 % | 30.58 % | -14.07 pp | 30.91 % | -13.74 pp | 30.79 % | -13.86 pp | 45.66 % | +1.01 pp | 21.05 % | -23.60 pp |
| DRAM Throughput (%) | 40.78 % | 21.20 % | -19.58 pp | 20.95 % | -19.83 pp | 20.62 % | -20.16 pp | 41.11 % | +0.33 pp | 14.24 % | -26.54 pp |
| Achieved Occupancy (%) | 37.65 % | 32.63 % | -5.02 pp | 32.64 % | -5.01 pp | 32.74 % | -4.91 pp | 64.80 % | +27.15 pp | 17.57 % | -20.08 pp |
| Registers/Thread | 64.00 register/thread | 44.00 register/thread | -20 | 44.00 register/thread | -20 | 44.00 register/thread | -20 | 32.00 register/thread | -32 | 164.00 register/thread | +100 |
| Waves/SM | 1.52 | 0.61 | -0.91 | 0.61 | -0.91 | 0.61 | -0.91 | 0.76 | -0.76 | 4.04 | +2.52 |
| No Eligible (%) | 35.56 % | 56.60 % | +21.04 pp | 55.10 % | +19.54 pp | 57.26 % | +21.70 pp | 55.42 % | +19.86 pp | 54.14 % | +18.58 pp |
| IPC Active | 2.53 inst/cycle | 1.75 inst/cycle | -0.78 | 1.74 inst/cycle | -0.79 | 1.72 inst/cycle | -0.81 | 1.73 inst/cycle | -0.80 | 1.82 inst/cycle | -0.71 |


### Key Learnings

#### ✅ Plan C: `replace_int_div_sqrt_chain_with_rsqrt_to_enable_fma_fusion_and_reduce_register_pressure` (2.117×)

A batch normalization normalization chain that uses an int32 literal (tl.full([1,1], 1, tl.int32)) divided by tl.sqrt_rn, followed by a redundant multiply by 1.0, blocks FMA fusion and inflates register pressure due to type coercion and extra SSA temporaries. Replacing the 7-temporary chain (eps literal + add + sqrt + int32 literal + int-div + float literal + multiply) with a single tl.math.rsqrt call reduced registers/thread from 78 to 44, raised Block Limit Registers from 12 to 20, lifted theoretical occupancy from 37.5% to 62.5%, eliminated all branch divergence (Avg. Active Threads Per Warp: 26.85 → 32.0, Branch Efficiency: 83.51% → 100%), and cut executed instructions from 3.02M to 1.44M — a 52% reduction — yielding a 2.12x end-to-end speedup. The int32 literal forces an I2F conversion that breaks the FP dependency chain and prevents the compiler from fusing subsequent multiply-add operations into FMA instructions.

> **Applicable when**: Triton BN normalization kernel uses an int32 literal (tl.full([...], 1, tl.int32)) in a division-by-sqrt pattern; FPInstructions NCU rule fires with high non-fused FP32 instruction count (e.g., 6:1 ratio of non-fused to fused); registers_per_thread is elevated (>64) due to excess SSA temporaries from the type-coercion chain; ThreadDivergence rule fires (Avg. Active Threads Per Warp < 32) indicating predicated instructions from the int32 branch.

> **Avoid when**: The variance can be negative (rsqrt is only mathematically equivalent to 1/sqrt for non-negative inputs); epsilon must be exactly representable as float32 (1e-05 is fine); tl.math.rsqrt is unavailable in the Triton version (fallback: libdevice.rsqrt); the normalization chain is already using a pure FP32 path without int32 literals.

---

## Iteration 5

**Best so far**: 2.15× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_softmax_fdiv_with_rcp_multiply_to_enable_fma_fusion_and_reduce_non_fused_fp32_instructions | 2.121× | 2.140× | ✓ |  |
| B | replace_softmax_element_division_with_reciprocal_multiply_to_enable_fma_fusion | 2.152× | 2.160× | ✓ |  |
| C | force_fma_and_rcp_to_eliminate_non_fused_fp32_instructions | 2.151× | 2.150× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_memory_latency | 2.125× | 1.963× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_softmax_f... | Δ | replace_softmax_e... | Δ | force_fma_and_rcp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13760 ns | 6430 ns | -7330 ns (-53.3%) | 6370 ns | -7390 ns (-53.7%) | 6400 ns | -7360 ns (-53.5%) | 7010 ns | -6750 ns (-49.1%) |
| SM Throughput (%) | 51.63 % | 27.03 % | -24.60 pp | 27.69 % | -23.94 pp | 27.64 % | -23.99 pp | 27.16 % | -24.47 pp |
| Memory Throughput (%) | 44.65 % | 30.32 % | -14.33 pp | 31.03 % | -13.62 pp | 30.99 % | -13.66 pp | 28.57 % | -16.08 pp |
| DRAM Throughput (%) | 40.78 % | 20.85 % | -19.93 pp | 21.10 % | -19.68 pp | 20.97 % | -19.81 pp | 19.12 % | -21.66 pp |
| Achieved Occupancy (%) | 37.65 % | 32.61 % | -5.04 pp | 32.61 % | -5.04 pp | 33.09 % | -4.56 pp | 61.10 % | +23.45 pp |
| Registers/Thread | 64.00 register/thread | 46.00 register/thread | -18 | 46.00 register/thread | -18 | 46.00 register/thread | -18 | 38.00 register/thread | -26 |
| Waves/SM | 1.52 | 0.61 | -0.91 | 0.61 | -0.91 | 0.61 | -0.91 | 1.01 | -0.51 |
| No Eligible (%) | 35.56 % | 56.94 % | +21.38 pp | 56.07 % | +20.51 pp | 57.79 % | +22.23 pp | 49.22 % | +13.66 pp |
| IPC Active | 2.53 inst/cycle | 1.68 inst/cycle | -0.85 | 1.66 inst/cycle | -0.87 | 1.49 inst/cycle | -1.04 | 1.99 inst/cycle | -0.54 |


### Key Learnings

#### ✅ Plan D: `increase_num_warps_to_hide_l1tex_memory_latency` (1.963×)

When a kernel is bottlenecked by L1TEX scoreboard stalls (CPIStall rule, ~48% of cycles with no eligible warp, only ~1.07 eligible warps/scheduler), expanding the autotune config space to include higher num_warps values (up to 16) gives the warp scheduler more independent warps to issue while others wait on global memory loads. In this case, the autotuner selected a config with larger block size (64→128 threads) and fewer registers/thread (44→38), which nearly doubled achieved occupancy (32.6%→61.1%), reduced No Eligible cycles from 56.6% to 49.2%, and increased eligible warps/scheduler from 1.07 to 1.57, yielding a 1.96x end-to-end speedup despite a slightly longer per-kernel NCU duration (6.34→7.01 µs) due to better SM wave coverage (0.61→1.01 waves/SM).

> **Applicable when**: CPIStall NCU rule fires with L1TEX scoreboard stalls accounting for >40% of warp cycles; No Eligible % is high (>50%); Eligible Warps Per Scheduler is low (<1.5); registers_per_thread is moderate (<64) so increasing num_warps does not cause register spill; waves_per_sm is below 1.0 (partial wave, SMs underutilized); the existing autotune config set caps num_warps at 4 or 8.

> **Avoid when**: Registers per thread is already high (>64) — increasing num_warps raises threads/block and total register demand, risking spill; the kernel is compute-bound rather than memory-latency-bound (Compute SM Throughput > 60%); XBLOCK * num_warps * 32 would exceed 1024 threads/block hardware limit; the kernel has no existing autotune decorator and adding one risks mismatched grid dimensions.

---

## Cumulative Progress

```
Baseline:  13760 ns  (1.000×)
Iter 1     11230 ns  (1.225×)  — autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_reduce_register_pressure
Iter 4     6400 ns  (2.150×)  — enable_fma_fusion_by_replacing_int_div_with_float_rsqrt_in_batchnorm_normalization

Final best: 6400 ns  (2.1500×)
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
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 204800
@@ -14,7 +26,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
     r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
@@ -22,35 +34,22 @@

     rindex = r0_index
     r0_1 = r0_index
     x0 = xindex
-    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), None)
-    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
+    tmp1 = tl.load(in_ptr0 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
+    tmp3 = tl.load(in_ptr1 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
+    tmp12 = tl.load(in_ptr2 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
+    tmp14 = tl.load(in_ptr3 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1, 1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1, 1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1, 1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
-    tmp18 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
-    tmp20 = triton_helpers.max2(tmp18, 1)[:, None].to(tl.float32)
-    tmp21 = tmp16 - tmp20
-    tmp22 = tl.exp(tmp21)
-    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
-    tmp25 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
-    tmp26 = tmp15 - tmp20
-    tmp27 = tl.exp(tmp26)
-    tmp28 = (tmp27 / tmp25)
-    tl.store(in_out_ptr0 + (r0_1 + 32*x0), tmp28, None)
-
-
+    tmp5 = tmp3 + tl.full([1, 1], 1e-05, tl.float32)
+    inv_std = libdevice.rsqrt(tmp5)
+    tmp11 = tmp2 * inv_std
+    tmp15 = tmp11 * tmp12 + tmp14
+    row_max = tl.max(tmp15, axis=1)[:, None]
+    shifted = tmp15 - row_max
+    exps = tl.exp(shifted)
+    row_sum = tl.sum(exps, axis=1)[:, None]
+    result = exps / row_sum
+    tl.store(in_out_ptr0 + (r0_1 + 32*x0), result, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -61,7 +60,7 @@

     xnumel = 204800
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
