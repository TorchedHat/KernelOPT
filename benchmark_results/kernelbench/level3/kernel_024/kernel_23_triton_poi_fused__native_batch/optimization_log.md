# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2530 ns — **Best latency**: 2430 ns (**1.0412×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_warps_to_hide_l1tex_latency_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_23_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (18,1,1), Waves/SM: 0.01
- **Registers/thread**: 18
- **Duration**: 2530 ns
- **SM Throughput**: 0.28 % | **Memory**: 0.68 % | **DRAM**: 0.26 %
- **Achieved Occupancy**: 6.21 %
- **IPC Active**: 0.22 inst/cycle | **No Eligible**: 93.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.936× | 0.861× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_small_grid | 1.014× | 1.012× | ✓ |  |
| C | add_vectorized_loads_with_autotune | 0.975× | 0.682× | ✓ |  |
| D | autotune_xblock_for_small_grid_occupancy | 1.047× | 1.028× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | add_vectorized_lo... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2940 ns | +410 ns (+16.2%) | 2500 ns | -30 ns (-1.2%) | 3710 ns | +1180 ns (+46.6%) | 2460 ns | -70 ns (-2.8%) |
| SM Throughput (%) | 0.28 % | 0.26 % | -0.02 pp | 0.08 % | -0.20 pp | 1.33 % | +1.05 pp | 0.08 % | -0.20 pp |
| Memory Throughput (%) | 0.68 % | 0.50 % | -0.18 pp | 0.44 % | -0.24 pp | 0.52 % | -0.16 pp | 0.42 % | -0.26 pp |
| DRAM Throughput (%) | 0.26 % | 0.23 % | -0.03 pp | 0.08 % | -0.18 pp | 0.20 % | -0.06 pp | 0.08 % | -0.18 pp |
| Achieved Occupancy (%) | 6.21 % | 3.09 % | -3.12 pp | 1.56 % | -4.65 pp | 11.84 % | +5.63 pp | 1.56 % | -4.65 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 19.00 register/thread | +1 | 48.00 register/thread | +30 | 19.00 register/thread | +1 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.03 | +0.02 | 0.00 | -0.01 |
| No Eligible (%) | 93.22 % | 88.51 % | -4.71 pp | 92.81 % | -0.41 pp | 62.27 % | -30.95 pp | 94.12 % | +0.90 pp |
| IPC Active | 0.22 inst/cycle | 0.25 inst/cycle | +0.03 | 0.06 inst/cycle | -0.16 | 1.43 inst/cycle | +1.21 | 0.06 inst/cycle | -0.16 |

---

## Iteration 2

**Best so far**: 1.0285× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | simplify_arithmetic_chain_to_reduce_register_pressure_and_instruction_count | 1.046× | 1.012× | ✓ |  |
| B | simplify_redundant_int_cast_arithmetic_to_reduce_register_pressure | 1.077× | 1.100× | ✓ | **✓ best** |
| C | simplify_arithmetic_to_reduce_register_pressure | 0.986× | 0.966× | ✓ |  |
| D | remove_autotune_restore_low_register_pressure | 1.028× | 0.966× | ✓ |  |
| E | add_triton_autotune | 0.992× | 0.930× | ✓ |  |


### Metric Comparison

| Metric | Baseline | simplify_arithmet... | Δ | simplify_redundan... | Δ | simplify_arithmet... | Δ | remove_autotune_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2500 ns | -30 ns (-1.2%) | 2300 ns | -230 ns (-9.1%) | 2620 ns | +90 ns (+3.6%) | 2620 ns | +90 ns (+3.6%) | 2720 ns | +190 ns (+7.5%) |
| SM Throughput (%) | 0.28 % | 0.07 % | -0.21 pp | 0.06 % | -0.22 pp | 0.28 % | +0.00 pp | 0.30 % | +0.02 pp | 0.30 % | +0.02 pp |
| Memory Throughput (%) | 0.68 % | 0.46 % | -0.22 pp | 0.49 % | -0.19 pp | 0.59 % | -0.09 pp | 0.60 % | -0.08 pp | 0.58 % | -0.10 pp |
| DRAM Throughput (%) | 0.26 % | 0.09 % | -0.17 pp | 0.08 % | -0.18 pp | 0.25 % | -0.01 pp | 0.25 % | -0.01 pp | 0.24 % | -0.02 pp |
| Achieved Occupancy (%) | 6.21 % | 1.56 % | -4.65 pp | 1.56 % | -4.65 pp | 5.98 % | -0.23 pp | 6.01 % | -0.20 pp | 6.07 % | -0.14 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 93.22 % | 92.74 % | -0.48 pp | 96.10 % | +2.88 pp | 93.75 % | +0.53 pp | 93.33 % | +0.11 pp | 93.28 % | +0.06 pp |
| IPC Active | 0.22 inst/cycle | 0.06 inst/cycle | -0.16 | 0.05 inst/cycle | -0.17 | 0.25 inst/cycle | +0.03 | 0.25 inst/cycle | +0.03 | 0.24 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.1× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_use_single_large_block_to_reduce_launch_overhead | 1.077× | 1.041× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_to_spread_blocks_across_sms | 1.048× | 1.028× | ✓ |  |
| C | consolidate_to_larger_blocks_increase_warps_per_sm | 0.992× | 0.966× | ✓ |  |
| D | collapse_grid_to_single_block_and_replace_sqrt_with_rsqrt | 1.022× | 1.081× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_autotune_grid... | Δ | consolidate_to_la... | Δ | collapse_grid_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2430 ns | -100 ns (-4.0%) | 2460 ns | -70 ns (-2.8%) | 2620 ns | +90 ns (+3.6%) | 2340 ns | -190 ns (-7.5%) |
| SM Throughput (%) | 0.28 % | 0.06 % | -0.22 pp | 0.08 % | -0.20 pp | 0.31 % | +0.03 pp | 0.24 % | -0.04 pp |
| Memory Throughput (%) | 0.68 % | 0.52 % | -0.16 pp | 0.43 % | -0.25 pp | 0.59 % | -0.09 pp | 0.75 % | +0.07 pp |
| DRAM Throughput (%) | 0.26 % | 0.08 % | -0.18 pp | 0.08 % | -0.18 pp | 0.25 % | -0.01 pp | 0.27 % | +0.01 pp |
| Achieved Occupancy (%) | 6.21 % | 1.56 % | -4.65 pp | 1.56 % | -4.65 pp | 6.13 % | -0.08 pp | 6.14 % | -0.07 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 93.22 % | 94.79 % | +1.57 pp | 93.41 % | +0.19 pp | 93.21 % | -0.01 pp | 95.04 % | +1.82 pp |
| IPC Active | 0.22 inst/cycle | 0.05 inst/cycle | -0.17 | 0.07 inst/cycle | -0.15 | 0.26 inst/cycle | +0.04 | 0.19 inst/cycle | -0.03 |

---

## Iteration 4

**Best so far**: 1.1× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_stalls | 1.054× | 1.041× | ✓ |  |
| B | persistent_kernel_spread_across_all_sms | 1.037× | 0.966× | ✓ |  |
| C | single_block_full_tile_with_rsqrt_fusion | 1.057× | 1.081× | ✓ |  |
| D | replace_sqrt_reciprocal_with_rsqrt_and_eliminate_scoreboard_stalls | 1.027× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | persistent_kernel... | Δ | single_block_full... | Δ | replace_sqrt_reci... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2430 ns | -100 ns (-4.0%) | 2620 ns | +90 ns (+3.6%) | 2340 ns | -190 ns (-7.5%) | 2590 ns | +60 ns (+2.4%) |
| SM Throughput (%) | 0.28 % | 0.24 % | -0.04 pp | 0.31 % | +0.03 pp | 0.22 % | -0.06 pp | 0.22 % | -0.06 pp |
| Memory Throughput (%) | 0.68 % | 0.67 % | -0.01 pp | 0.58 % | -0.10 pp | 0.65 % | -0.03 pp | 0.59 % | -0.09 pp |
| DRAM Throughput (%) | 0.26 % | 0.26 % | +0.00 pp | 0.25 % | -0.01 pp | 0.28 % | +0.02 pp | 0.25 % | -0.01 pp |
| Achieved Occupancy (%) | 6.21 % | 6.09 % | -0.12 pp | 6.08 % | -0.13 pp | 6.12 % | -0.09 pp | 6.08 % | -0.13 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 93.22 % | 95.11 % | +1.89 pp | 95.79 % | +2.57 pp | 95.67 % | +2.45 pp | 95.39 % | +2.17 pp |
| IPC Active | 0.22 inst/cycle | 0.16 inst/cycle | -0.06 | 0.17 inst/cycle | -0.05 | 0.18 inst/cycle | -0.04 | 0.14 inst/cycle | -0.08 |


### Key Learnings

#### ✅ Plan A: `increase_num_warps_to_hide_l1tex_latency_stalls` (1.041×)

Increasing num_warps from 4 to 8 (doubling threads per block from 128 to 256) on a tiny kernel (18 blocks, 132-SM GPU, 2304 total elements) did not reduce L1TEX scoreboard stalls as expected. Despite the plan predicting active warps per scheduler would rise from 1.1 to ~2.2, the profiling shows it only moved from 1.1 to 1.2, No Eligible% stayed at ~95%, and Warp Cycles Per Issued Instruction actually increased from 22.21 to 24.46 cycles. The kernel duration increased from 2.34 µs to 2.43 µs in the profiled kernel, yet the overall wall-clock speedup was 1.04x — suggesting the profiled kernel is not the primary bottleneck and the speedup came from elsewhere. For launch-overhead-dominated kernels near the GPU latency floor (~2 µs), increasing num_warps does not meaningfully improve latency hiding because the grid is too small to amortize the overhead.

> **Applicable when**: Never: increasing num_warps alone does not help when the kernel is launch-overhead dominated (grid_size << SM count, waves_per_sm < 0.05, duration < 3 µs on H100/H200 class GPUs). The CPIStall L1TEX signal is present but the bottleneck is the tiny grid, not warp-level latency hiding capacity.

> **Avoid when**: When grid_size is much smaller than SM count (waves_per_sm < 0.05) and kernel duration is near the GPU launch latency floor (~2 µs). In this regime, increasing num_warps does not increase active warps per scheduler meaningfully because there are too few blocks to distribute across SMs. The CPIStall rule may be triggered but the root cause is grid underutilization, not insufficient warp parallelism within each block.

---

## Iteration 5

**Best so far**: 1.1× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_arithmetic_for_fma_fusion_to_reduce_instruction_fetch_stalls | 1.024× | 0.977× | ✓ |  |
| B | collapse_to_single_block_xblock_4096_maximize_l1_reuse | 1.056× | 1.081× | ✓ |  |
| C | single_block_full_tile_l1_broadcast_cache_reuse | 1.027× | 1.100× | ✓ |  |
| D | fix_persistent_kernel_grid_to_use_num_sms_blocks | 1.001× | 0.951× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_arith... | Δ | collapse_to_singl... | Δ | single_block_full... | Δ | fix_persistent_ke... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2530 ns | 2590 ns | +60 ns (+2.4%) | 2340 ns | -190 ns (-7.5%) | 2300 ns | -230 ns (-9.1%) | 2660 ns | +130 ns (+5.1%) |
| SM Throughput (%) | 0.28 % | 0.21 % | -0.07 pp | 0.24 % | -0.04 pp | 0.24 % | -0.04 pp | 0.31 % | +0.03 pp |
| Memory Throughput (%) | 0.68 % | 0.58 % | -0.10 pp | 0.69 % | +0.01 pp | 0.67 % | -0.01 pp | 0.58 % | -0.10 pp |
| DRAM Throughput (%) | 0.26 % | 0.25 % | -0.01 pp | 0.28 % | +0.02 pp | 0.27 % | +0.01 pp | 0.24 % | -0.02 pp |
| Achieved Occupancy (%) | 6.21 % | 6.10 % | -0.11 pp | 6.15 % | -0.06 pp | 6.14 % | -0.07 pp | 6.16 % | -0.05 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 93.22 % | 95.43 % | +2.21 pp | 96.31 % | +3.09 pp | 95.00 % | +1.78 pp | 95.45 % | +2.23 pp |
| IPC Active | 0.22 inst/cycle | 0.18 inst/cycle | -0.04 | 0.18 inst/cycle | -0.04 | 0.19 inst/cycle | -0.03 | 0.15 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  2530 ns  (1.000×)
Iter 1     2460 ns  (1.028×)  — autotune_xblock_for_small_grid_occupancy
Iter 2     2300 ns  (1.100×)  — simplify_redundant_int_cast_arithmetic_to_reduce_register_pressure

Final best: 2430 ns  (1.0412×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -21,20 +22,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    inv_std = tl.rsqrt(tmp5)
+    tmp11 = tmp2 * inv_std
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tmp18 = (tmp17 / tmp9)
-    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
-
-
+    tmp17 = tl.maximum(0.0, tmp15)
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
