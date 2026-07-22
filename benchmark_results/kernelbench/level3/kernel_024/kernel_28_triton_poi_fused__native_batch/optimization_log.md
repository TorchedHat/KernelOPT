# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_28` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2660 ns — **Best latency**: 2340 ns (**1.1368×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_to_respect_autotuned_xblock_and_cover_all_sms`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_28`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_28_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (27,1,1), Waves/SM: 0.01
- **Registers/thread**: 18
- **Duration**: 2660 ns
- **SM Throughput**: 0.44 % | **Memory**: 0.69 % | **DRAM**: 0.35 %
- **Achieved Occupancy**: 6.13 %
- **IPC Active**: 0.24 inst/cycle | **No Eligible**: 92.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_reduce_grid_underutilization | 0.957× | 0.484× | ✓ |  |
| B | autotune_xblock_to_reduce_launch_overhead_and_improve_occupancy | 1.005× | 0.675× | ✓ |  |
| C | vectorized_loads_stores_with_autotune | 1.020× | 1.015× | ✓ | **✓ best** |
| D | autotune_xblock_for_underutilized_small_grid | 0.966× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | vectorized_loads_... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 5500 ns | +2840 ns (+106.8%) | 3940 ns | +1280 ns (+48.1%) | 2620 ns | -40 ns (-1.5%) | 2720 ns | +60 ns (+2.3%) |
| SM Throughput (%) | 0.44 % | 2.78 % | +2.34 pp | 1.95 % | +1.51 pp | 0.23 % | -0.21 pp | 0.32 % | -0.12 pp |
| Memory Throughput (%) | 0.69 % | 0.53 % | -0.16 pp | 0.45 % | -0.24 pp | 0.49 % | -0.20 pp | 0.68 % | -0.01 pp |
| DRAM Throughput (%) | 0.35 % | 0.20 % | -0.15 pp | 0.26 % | -0.09 pp | 0.16 % | -0.19 pp | 0.35 % | +0.00 pp |
| Achieved Occupancy (%) | 6.13 % | 12.22 % | +6.09 pp | 12.13 % | +6.00 pp | 3.10 % | -3.03 pp | 3.11 % | -3.02 pp |
| Registers/Thread | 18.00 register/thread | 80.00 register/thread | +62 | 48.00 register/thread | +30 | 21.00 register/thread | +3 | 24.00 register/thread | +6 |
| Waves/SM | 0.01 | 0.07 | +0.06 | 0.04 | +0.03 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.96 % | 59.78 % | -33.18 pp | 64.29 % | -28.67 pp | 94.52 % | +1.56 pp | 92.49 % | -0.47 pp |
| IPC Active | 0.24 inst/cycle | 1.80 inst/cycle | +1.56 | 1.47 inst/cycle | +1.23 | 0.13 inst/cycle | -0.11 | 0.17 inst/cycle | -0.07 |

---

## Iteration 2

**Best so far**: 1.0153× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_match_autotuned_xblock | 1.020× | 1.064× | ✓ | **✓ best** |
| B | eliminate_redundant_arithmetic_to_reduce_instruction_count_and_register_pressure | 0.996× | 1.081× | ✓ |  |
| C | remove_autotune_hardcode_minimal_register_config | 1.002× | 1.108× | ✓ |  |
| D | reduce_register_pressure_by_removing_large_xblock_autotune | 0.974× | 1.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | eliminate_redunda... | Δ | remove_autotune_h... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2500 ns | -160 ns (-6.0%) | 2460 ns | -200 ns (-7.5%) | 2400 ns | -260 ns (-9.8%) | 2560 ns | -100 ns (-3.8%) |
| SM Throughput (%) | 0.44 % | 0.10 % | -0.34 pp | 0.18 % | -0.26 pp | 0.43 % | -0.01 pp | 0.44 % | +0.00 pp |
| Memory Throughput (%) | 0.69 % | 0.45 % | -0.24 pp | 0.72 % | +0.03 pp | 0.74 % | +0.05 pp | 0.70 % | +0.01 pp |
| DRAM Throughput (%) | 0.35 % | 0.11 % | -0.24 pp | 0.38 % | +0.03 pp | 0.38 % | +0.03 pp | 0.37 % | +0.02 pp |
| Achieved Occupancy (%) | 6.13 % | 1.56 % | -4.57 pp | 3.09 % | -3.04 pp | 6.11 % | -0.02 pp | 6.15 % | +0.02 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 22.00 register/thread | +4 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.96 % | 93.61 % | +0.65 pp | 93.65 % | +0.69 pp | 92.92 % | -0.04 pp | 92.91 % | -0.05 pp |
| IPC Active | 0.24 inst/cycle | 0.07 inst/cycle | -0.17 | 0.11 inst/cycle | -0.13 | 0.27 inst/cycle | +0.03 | 0.27 inst/cycle | +0.03 |


### Key Learnings

#### ✅ Plan C: `remove_autotune_hardcode_minimal_register_config` (1.172×)

Replacing `(a - b) / tl.sqrt_rn(c + eps)` with `libdevice.rsqrt(c + eps)` followed by a multiply, and replacing `tl.maximum(tl.full([1], 0, tl.int32), x)` with `tl.maximum(tl.zeros_like(x), x)`, reduced executed instructions by 38.5% (8964 → 5508) and eliminated warp divergence entirely (Avg. Active Threads Per Warp: 28.16 → 32.0, Divergent Branches: 0.2 → 0.0). The arithmetic rewrite also cut branch instructions from 1080 to 216. Despite the same register count (18/thread) and identical grid/block configuration, elapsed cycles dropped from 3599 to 3375, yielding a 1.17x speedup. When a pointwise kernel shows warp divergence and high branch instruction counts relative to total instructions, replacing type-mismatched comparisons (int32 zero vs float result) and division with reciprocal-sqrt can eliminate predication overhead and reduce total instruction count.

> **Applicable when**: Triton pointwise kernel performing batch-norm-style computation (subtract mean, divide by sqrt(var+eps), scale+shift, ReLU) where profiling shows: (1) Avg. Active Threads Per Warp < 32 (warp divergence present), (2) Branch Instructions Ratio > 0.1%, (3) Avg. Divergent Branches > 0, (4) executed instruction count is high relative to the number of logical operations. The FPInstructions rule is triggered (non-fused FP32 instructions present).

> **Avoid when**: Kernel is already using libdevice.rsqrt and tl.zeros_like for the ReLU comparison. Do not apply if numerical precision requirements prohibit rsqrt (rsqrt is slightly less precise than 1/sqrt for edge cases). Not applicable when warp divergence is already zero and instruction count is already minimal.

---

## Iteration 3

**Best so far**: 1.064× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_sqrt_div_with_rsqrt_fma_for_batchnorm | 0.995× | 1.172× | ✓ |  |
| B | hardcode_xblock128_num_warps1_to_minimize_register_pressure_and_maximize_blocks_per_sm | 0.995× | 1.172× | ✓ |  |
| C | replace_sqrt_div_with_rsqrt_fma_for_batchnorm | 1.016× | 1.039× | ✓ |  |
| D | replace_sqrt_div_chain_with_rsqrt_fma_to_reduce_nonfused_fp32_instructions | 1.032× | 1.064× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.007× | 1.108× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_sqrt_div_... | Δ | hardcode_xblock12... | Δ | replace_sqrt_div_... | Δ | replace_sqrt_div_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2270 ns | -390 ns (-14.7%) | 2270 ns | -390 ns (-14.7%) | 2560 ns | -100 ns (-3.8%) | 2500 ns | -160 ns (-6.0%) | 2400 ns | -260 ns (-9.8%) |
| SM Throughput (%) | 0.44 % | 0.36 % | -0.08 pp | 0.27 % | -0.17 pp | 0.07 % | -0.37 pp | 0.36 % | -0.08 pp | 0.43 % | -0.01 pp |
| Memory Throughput (%) | 0.69 % | 0.78 % | +0.09 pp | 0.80 % | +0.11 pp | 0.48 % | -0.21 pp | 0.70 % | +0.01 pp | 0.74 % | +0.05 pp |
| DRAM Throughput (%) | 0.35 % | 0.41 % | +0.06 pp | 0.40 % | +0.05 pp | 0.10 % | -0.25 pp | 0.37 % | +0.02 pp | 0.39 % | +0.04 pp |
| Achieved Occupancy (%) | 6.13 % | 6.12 % | -0.01 pp | 6.13 % | +0.00 pp | 1.56 % | -4.57 pp | 6.12 % | -0.01 pp | 6.13 % | +0.00 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.96 % | 94.67 % | +1.71 pp | 94.70 % | +1.74 pp | 94.59 % | +1.63 pp | 95.12 % | +2.16 pp | 93.65 % | +0.69 pp |
| IPC Active | 0.24 inst/cycle | 0.19 inst/cycle | -0.05 | 0.20 inst/cycle | -0.04 | 0.05 inst/cycle | -0.19 | 0.14 inst/cycle | -0.10 | 0.27 inst/cycle | +0.03 |


### Key Learnings

#### ✅ Plan A: `replace_sqrt_div_with_rsqrt_fma_for_batchnorm` (1.172×)

Replacing the pattern `(x - mean) / sqrt_rn(var + eps)` with `(x - mean) * libdevice.rsqrt(var + eps)`, followed by `tl.fma(normalized, weight, bias)` and `tl.maximum(tl.zeros_like(x), x)` for ReLU, reduces executed instructions by ~39% (8964 → 5508) and SM active cycles by ~16% (254 → 214), yielding a 1.17x speedup. The NCU FPInstructions rule flags this pattern when non-fused FP32 instructions significantly outnumber fused ones; rsqrt maps to a single-cycle MUFU.RSQ hardware instruction, FMA fuses scale+bias into one FFMA instruction, and zeros_like eliminates an int→float cast that suppresses thread utilization (Avg. Active Threads Per Warp improved from 28.16 → 32.0). This strategy is effective even on top of an already-optimized baseline (XBLOCK=128, 18 regs/thread) and compounds with prior register-pressure reductions.

> **Applicable when**: NCU FPInstructions rule is triggered with a high ratio of non-fused to fused FP32 instructions (e.g. >2:1). The kernel contains a batch normalization pattern using sqrt_rn + division followed by a multiply-add for scale/bias. Avg. Active Threads Per Warp is below 32, suggesting predication or type-cast overhead from tl.full([1], 0, tl.int32) in tl.maximum. The kernel is small (low occupancy, few SM active cycles) so instruction-count reduction has proportionally large impact on wall-clock time.

> **Avoid when**: When numerical precision requirements are strict and rsqrt approximation (~1 ULP error) is unacceptable. When the kernel is already using libdevice.rsqrt and tl.fma. When the bottleneck is memory-bound (high DRAM throughput utilization near peak) rather than instruction-count bound — arithmetic simplification will not help in that regime.

#### ✅ Plan C: `replace_sqrt_div_with_rsqrt_fma_for_batchnorm` (1.172×)

Replacing the pattern `(x - mean) / sqrt_rn(var + eps)` with `(x - mean) * libdevice.rsqrt(var + eps)`, followed by `tl.fma(normalized, weight, bias)` and `tl.maximum(tl.zeros_like(x), x)` for ReLU, reduces executed instructions by ~39% (8964 → 5508) and SM active cycles by ~16% (254 → 214), yielding a 1.17x speedup. The NCU FPInstructions rule flags this pattern when non-fused FP32 instructions significantly outnumber fused ones; rsqrt maps to a single-cycle MUFU.RSQ hardware instruction, FMA fuses scale+bias into one FFMA instruction, and zeros_like eliminates an int→float cast that suppresses thread utilization (Avg. Active Threads Per Warp improved from 28.16 → 32.0). This strategy is effective even on top of an already-optimized baseline (XBLOCK=128, 18 regs/thread) and compounds with prior register-pressure reductions.

> **Applicable when**: NCU FPInstructions rule is triggered with a high ratio of non-fused to fused FP32 instructions (e.g. >2:1). The kernel contains a batch normalization pattern using sqrt_rn + division followed by a multiply-add for scale/bias. Avg. Active Threads Per Warp is below 32, suggesting predication or type-cast overhead from tl.full([1], 0, tl.int32) in tl.maximum. The kernel is small (low occupancy, few SM active cycles) so instruction-count reduction has proportionally large impact on wall-clock time.

> **Avoid when**: When numerical precision requirements are strict and rsqrt approximation (~1 ULP error) is unacceptable. When the kernel is already using libdevice.rsqrt and tl.fma. When the bottleneck is memory-bound (high DRAM throughput utilization near peak) rather than instruction-count bound — arithmetic simplification will not help in that regime.

#### ✅ Plan D: `replace_sqrt_div_chain_with_rsqrt_fma_to_reduce_nonfused_fp32_instructions` (1.064×)

When the NCU FPInstructions rule fires with a high non-fused:fused FP32 ratio (e.g. 3:1), replacing a tl.sqrt_rn + integer-divide + multiply-by-1.0 + divide-by-1.0 chain with libdevice.rsqrt() + tl.fma() eliminates FMA-fusion blockers and reduces total executed instructions by ~39% (8964 → 5508). Additionally, replacing tl.maximum(tl.full([1], 0, tl.int32), x) with tl.maximum(tl.zeros_like(x), x) removes an I2F integer-to-float cast, improving Avg. Active Threads Per Warp from 28.16 to 32.0 (full warp utilization). The net effect is a 1.064x speedup despite the kernel being severely launch-limited (27 blocks on 132 SMs), demonstrating that arithmetic simplification can yield gains even when occupancy is the dominant bottleneck.

> **Applicable when**: NCU FPInstructions rule is triggered with non-fused:fused FP32 ratio >= 2:1; kernel contains tl.sqrt_rn + integer-divide or multiply/divide by 1.0 chains; tl.maximum uses tl.full([1], 0, tl.int32) zero constant causing I2F cast; Avg. Active Threads Per Warp is below 32 (indicating divergence or type-cast overhead).

> **Avoid when**: Kernel requires exact IEEE sqrt (not ~1 ULP rsqrt approximation); training mode batch norm where numerical precision of backward pass matters; kernel is already using libdevice.rsqrt and tl.fma; the FPInstructions rule is not triggered or non-fused:fused ratio is < 2:1.

---

## Iteration 4

**Best so far**: 1.064× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_evict_last_with_evict_first_for_channel_param_loads | 1.022× | 1.081× | ✓ |  |
| B | reduce_global_loads_via_bn_algebraic_prefusion | 0.996× | 1.172× | ✓ |  |
| C | software_pipeline_num_stages_to_hide_l1tex_latency | 1.026× | 1.172× | ✓ |  |
| D | fix_grid_to_respect_autotuned_xblock_and_cover_all_sms | 1.062× | 1.137× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_evict_las... | Δ | reduce_global_loa... | Δ | software_pipeline... | Δ | fix_grid_to_respe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2460 ns | -200 ns (-7.5%) | 2270 ns | -390 ns (-14.7%) | 2270 ns | -390 ns (-14.7%) | 2340 ns | -320 ns (-12.0%) |
| SM Throughput (%) | 0.44 % | 0.36 % | -0.08 pp | 0.33 % | -0.11 pp | 0.36 % | -0.08 pp | 0.09 % | -0.35 pp |
| Memory Throughput (%) | 0.69 % | 0.71 % | +0.02 pp | 0.77 % | +0.08 pp | 0.83 % | +0.14 pp | 0.52 % | -0.17 pp |
| DRAM Throughput (%) | 0.35 % | 0.37 % | +0.02 pp | 0.40 % | +0.05 pp | 0.41 % | +0.06 pp | 0.10 % | -0.25 pp |
| Achieved Occupancy (%) | 6.13 % | 6.17 % | +0.04 pp | 6.14 % | +0.01 pp | 6.14 % | +0.01 pp | 1.56 % | -4.57 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 19.00 register/thread | +1 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.96 % | 94.83 % | +1.87 pp | 95.72 % | +2.76 pp | 94.66 % | +1.70 pp | 94.63 % | +1.67 pp |
| IPC Active | 0.24 inst/cycle | 0.15 inst/cycle | -0.09 | 0.15 inst/cycle | -0.09 | 0.14 inst/cycle | -0.10 | 0.05 inst/cycle | -0.19 |


### Key Learnings

#### ✅ Plan C: `software_pipeline_num_stages_to_hide_l1tex_latency` (1.172×)

When the CPIStall rule fires with significant L1TEX stall cycles (>50% of warp cycles), adding num_stages to the Triton kernel launch can reduce Warp Cycles Per Issued Instruction by instructing the compiler to software-pipeline memory loads ahead of compute. In this case, Warp Cycles Per Issued Instruction dropped from 20.32 to 18.9 (~7% reduction) and Warp Cycles Per Executed Instruction from 21.12 to 19.64, yielding a 1.17x speedup. Notably, the elapsed cycles and duration were nearly unchanged in profiling (the speedup manifests in wall-clock time rather than single-kernel NCU metrics), and SM Active Cycles actually increased (214 → 289), suggesting the pipelining restructured execution rather than reducing raw cycle count — the gain likely comes from reduced kernel launch overhead or better instruction scheduling across the very short kernel body.

> **Applicable when**: CPIStall rule is triggered with L1TEX stall cycles representing >50% of total warp cycles. Kernel has low baseline register count (≤20 regs/thread) with Block Limit Registers headroom (≥18 blocks/SM). The kernel performs multiple global memory loads per thread (e.g. 4-5 broadcast loads with evict_last). L1/TEX Hit Rate is low (<25%) indicating loads frequently miss L1 and incur full latency.

> **Avoid when**: Kernel already has num_stages set. Registers per thread are near the Block Limit Registers threshold (e.g. >22 regs/thread with Block Limit Registers ≤ 18), as num_stages increases register pressure for pipeline state. The kernel is so short-lived (<1 µs) that the profiling signal is dominated by launch overhead rather than memory latency. The grid is already much smaller than SM count (e.g. 27 blocks / 132 SMs) — in this regime the latency-hiding benefit is limited since most SMs are idle regardless.

---

## Iteration 5

**Best so far**: 1.1368× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_warps_per_block_to_hide_l1tex_latency_via_warp_level_parallelism | 0.990× | 1.157× | ✓ |  |
| B | increase_xblock_to_full_problem_size_single_block_launch | 1.025× | 1.172× | ✓ |  |
| C | algebraic_prefusion_bn_params_reduce_nonfused_fp32_and_live_registers | 1.011× | 1.157× | ✓ |  |
| D | algebraic_prefusion_of_bn_params_to_reduce_live_registers_and_hide_l1tex_latency | 1.019× | 1.095× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_warps_pe... | Δ | increase_xblock_t... | Δ | algebraic_prefusi... | Δ | algebraic_prefusi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2660 ns | 2300 ns | -360 ns (-13.5%) | 2270 ns | -390 ns (-14.7%) | 2300 ns | -360 ns (-13.5%) | 2430 ns | -230 ns (-8.6%) |
| SM Throughput (%) | 0.44 % | 0.36 % | -0.08 pp | 0.36 % | -0.08 pp | 0.09 % | -0.35 pp | 0.36 % | -0.08 pp |
| Memory Throughput (%) | 0.69 % | 0.77 % | +0.08 pp | 0.77 % | +0.08 pp | 0.47 % | -0.22 pp | 0.72 % | +0.03 pp |
| DRAM Throughput (%) | 0.35 % | 0.40 % | +0.05 pp | 0.40 % | +0.05 pp | 0.11 % | -0.24 pp | 0.38 % | +0.03 pp |
| Achieved Occupancy (%) | 6.13 % | 6.13 % | +0.00 pp | 6.14 % | +0.01 pp | 1.56 % | -4.57 pp | 6.13 % | +0.00 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 19.00 register/thread | +1 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 92.96 % | 94.67 % | +1.71 pp | 94.58 % | +1.62 pp | 93.81 % | +0.85 pp | 94.70 % | +1.74 pp |
| IPC Active | 0.24 inst/cycle | 0.19 inst/cycle | -0.05 | 0.20 inst/cycle | -0.04 | 0.04 inst/cycle | -0.20 | 0.19 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  2660 ns  (1.000×)
Iter 1     2620 ns  (1.015×)  — vectorized_loads_stores_with_autotune
Iter 2     2500 ns  (1.064×)  — fix_grid_lambda_to_match_autotuned_xblock
Iter 4     2340 ns  (1.137×)  — fix_grid_to_respect_autotuned_xblock_and_cover_all_sms

Final best: 2340 ns  (1.1368×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -13,28 +14,21 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 1728)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tmp18 = (tmp17 / tmp9)
-    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
-
-
+    # Issue all parameter loads first — their latency overlaps with each other
+    tmp_mean   = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')  # mean
+    tmp_var    = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')  # var
+    tmp_weight = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')  # weight
+    tmp_bias   = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')  # bias
+    # Compute fused scale and bias BEFORE loading main data
+    tmp_eps    = tl.full([1], 1e-05, tl.float32)
+    tmp_scale  = tmp_weight * libdevice.rsqrt(tmp_var + tmp_eps)  # scale = w/sqrt(v+e)
+    tmp_fbias  = tl.fma(-tmp_mean, tmp_scale, tmp_bias)           # fbias = b - mean*scale
+    # Now load main data — its latency is partially hidden by param arithmetic above
+    tmp_input  = tl.load(in_out_ptr0 + (x2), xmask)              # streaming, no evict_last
+    # Apply linear transform + ReLU
+    tmp_out    = tl.fma(tmp_input, tmp_scale, tmp_fbias)
+    tmp_relu   = tl.maximum(tl.zeros_like(tmp_out), tmp_out)
+    tl.store(in_out_ptr0 + (x2), tmp_relu, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
