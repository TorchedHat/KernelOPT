# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8990 ns — **Best latency**: 6430 ns (**1.3981×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_partial_wave_tail_via_smaller_xblock_and_num_warps1`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_48_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 32
- **Duration**: 8990 ns
- **SM Throughput**: 54.41 % | **Memory**: 20.55 % | **DRAM**: 20.55 %
- **Achieved Occupancy**: 77.23 %
- **IPC Active**: 2.85 inst/cycle | **No Eligible**: 25.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_eliminate_partial_wave | 1.004× | 0.955× | ✓ | **✓ best** |
| B | autotune_xblock_eliminate_partial_wave | 0.795× | 0.710× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.984× | 0.419× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave_and_improve_cache_reuse | 1.002× | 0.958× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_e... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 9410 ns | +420 ns (+4.7%) | 12670 ns | +3680 ns (+40.9%) | 21470 ns | +12480 ns (+138.8%) | 9380 ns | +390 ns (+4.3%) |
| SM Throughput (%) | 54.41 % | 63.28 % | +8.87 pp | 63.03 % | +8.62 pp | 71.77 % | +17.36 pp | 64.25 % | +9.84 pp |
| Memory Throughput (%) | 20.55 % | 23.61 % | +3.06 pp | 18.27 % | -2.28 pp | 17.92 % | -2.63 pp | 23.98 % | +3.43 pp |
| DRAM Throughput (%) | 20.55 % | 19.65 % | -0.90 pp | 14.57 % | -5.98 pp | 8.58 % | -11.97 pp | 19.75 % | -0.80 pp |
| Achieved Occupancy (%) | 77.23 % | 80.25 % | +3.02 pp | 62.64 % | -14.59 pp | 65.64 % | -11.59 pp | 80.13 % | +2.90 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 34.00 register/thread | +2 | 34.00 register/thread | +2 | 32.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 3.71 | +0.93 | 7.42 | +4.64 | 2.78 | +0.00 |
| No Eligible (%) | 25.19 % | 17.68 % | -7.51 pp | 24.14 % | -1.05 pp | 20.39 % | -4.80 pp | 17.53 % | -7.66 pp |
| IPC Active | 2.85 inst/cycle | 3.27 inst/cycle | +0.42 | 3.01 inst/cycle | +0.16 | 3.19 inst/cycle | +0.34 | 3.27 inst/cycle | +0.42 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_arithmetic_into_fma_instructions | 0.994× | 0.952× | ✓ |  |
| B | replace_sqrt_rcp_with_rsqrt_to_reduce_nonfused_fp_instructions | 1.064× | 1.180× | ✓ | **✓ best** |
| C | restore_register_budget_with_small_xblock_autotune_configs | 0.985× | 1.000× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.978× | 0.670× | ✓ |  |
| E | add_triton_autotune | 1.003× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_ar... | Δ | replace_sqrt_rcp_... | Δ | restore_register_... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 9440 ns | +450 ns (+5.0%) | 7620 ns | -1370 ns (-15.2%) | 8990 ns | +0 ns (+0.0%) | 13410 ns | +4420 ns (+49.2%) | 8990 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 54.41 % | 63.38 % | +8.97 pp | 57.28 % | +2.87 pp | 54.19 % | -0.22 pp | 69.27 % | +14.86 pp | 53.66 % | -0.75 pp |
| Memory Throughput (%) | 20.55 % | 23.58 % | +3.03 pp | 30.07 % | +9.52 pp | 20.53 % | -0.02 pp | 25.15 % | +4.60 pp | 20.53 % | -0.02 pp |
| DRAM Throughput (%) | 20.55 % | 19.61 % | -0.94 pp | 24.30 % | +3.75 pp | 20.53 % | -0.02 pp | 13.73 % | -6.82 pp | 20.53 % | -0.02 pp |
| Achieved Occupancy (%) | 77.23 % | 80.48 % | +3.25 pp | 76.39 % | -0.84 pp | 76.54 % | -0.69 pp | 81.79 % | +4.56 pp | 76.70 % | -0.53 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 5.57 | +2.79 | 2.78 | +0.00 |
| No Eligible (%) | 25.19 % | 20.93 % | -4.26 pp | 21.55 % | -3.64 pp | 25.36 % | +0.17 pp | 17.50 % | -7.69 pp | 25.45 % | +0.26 pp |
| IPC Active | 2.85 inst/cycle | 3.22 inst/cycle | +0.37 | 3.08 inst/cycle | +0.23 | 2.90 inst/cycle | +0.05 | 3.29 inst/cycle | +0.44 | 2.93 inst/cycle | +0.08 |


### Key Learnings

#### ✅ Plan B: `replace_sqrt_rcp_with_rsqrt_to_reduce_nonfused_fp_instructions` (1.180×)

Replacing a two-step `sqrt(x)` followed by `1/result` pattern with a single `rsqrt(x)` call (e.g., `libdevice.rsqrt()` in Triton) collapses multiple non-fused FP32 instructions into a single MUFU.RSQ hardware instruction. In this case, executed instructions dropped from 4.65M to 3.40M (~27% reduction) and kernel duration fell from 9.38 µs to 7.62 µs (1.18x speedup). The NCU FPInstructions rule flagging a high ratio of non-fused to fused FP32 instructions (4.3:1) is the key signal. Additionally, eliminating intermediate integer-division steps removed divergent branches (Avg. Divergent Branches: 44.55 → 0.0) and improved average active threads per warp (30.37 → 32.0), indicating the original pattern also caused warp divergence.

> **Applicable when**: NCU FPInstructions rule is triggered with a high ratio of non-fused to fused FP32 instructions (>3:1); kernel contains a sqrt(x) followed by 1/result or 1.0/sqrt(x) pattern; the operand is guaranteed positive (e.g., variance + epsilon in batch norm); Avg. Active Threads Per Warp is below 32 suggesting predicated/divergent paths from integer ops mixed with float ops.

> **Avoid when**: The operand may be zero or negative (rsqrt is undefined for non-positive inputs); the Triton compiler version already auto-fuses sqrt+rcp into MUFU.RSQ internally (verify by checking if non-fused instruction count is already low); the kernel is memory-bound rather than compute-bound (Memory SOL >> Compute SOL), where reducing FP instruction count yields minimal benefit.

---

## Iteration 3

**Best so far**: 1.1798× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_evict_first_for_streaming_loads | 1.099× | 1.180× | ✓ | **✓ best** |
| B | change_large_tensor_loads_to_evict_first_to_reduce_l1tex_stall | 0.966× | 1.000× | ✓ |  |
| C | fuse_sqrt_rcp_into_rsqrt_to_eliminate_nonfused_fp_instructions | 1.087× | 1.186× | ✓ |  |
| D | fuse_sqrt_rcp_into_rsqrt_to_eliminate_nonfused_fp32_instructions | 1.089× | 0.763× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | change_large_tens... | Δ | fuse_sqrt_rcp_int... | Δ | fuse_sqrt_rcp_int... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 7620 ns | -1370 ns (-15.2%) | 8990 ns | +0 ns (+0.0%) | 7580 ns | -1410 ns (-15.7%) | 11780 ns | +2790 ns (+31.0%) |
| SM Throughput (%) | 54.41 % | 51.82 % | -2.59 pp | 55.40 % | +0.99 pp | 49.75 % | -4.66 pp | 66.39 % | +11.98 pp |
| Memory Throughput (%) | 20.55 % | 24.31 % | +3.76 pp | 20.56 % | +0.01 pp | 24.37 % | +3.82 pp | 29.26 % | +8.71 pp |
| DRAM Throughput (%) | 20.55 % | 24.31 % | +3.76 pp | 20.56 % | +0.01 pp | 24.37 % | +3.82 pp | 15.69 % | -4.86 pp |
| Achieved Occupancy (%) | 77.23 % | 75.61 % | -1.62 pp | 76.71 % | -0.52 pp | 76.44 % | -0.79 pp | 82.07 % | +4.84 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 | 5.57 | +2.79 |
| No Eligible (%) | 25.19 % | 27.45 % | +2.26 pp | 25.44 % | +0.25 pp | 27.02 % | +1.83 pp | 19.47 % | -5.72 pp |
| IPC Active | 2.85 inst/cycle | 2.87 inst/cycle | +0.02 | 2.93 inst/cycle | +0.08 | 2.83 inst/cycle | -0.02 | 3.20 inst/cycle | +0.35 |

---

## Iteration 4

**Best so far**: 1.1798× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_alleviate_cpi_stall_scheduling_contention | 1.130× | 1.405× | ✓ | **✓ best** |
| B | reduce_num_warps_to_alleviate_cpi_not_selected_stall | 1.121× | 1.405× | ✓ |  |
| C | use_streaming_store_cache_modifier_to_enable_l2_delta_compression | 1.058× | 1.200× | ✓ |  |
| D | rewrite_batchnorm_epilogue_as_explicit_fma_to_reduce_nonfused_fp32_instructions | 1.115× | 0.778× | ✓ |  |
| E | add_triton_autotune | 1.126× | 1.405× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | reduce_num_warps_... | Δ | use_streaming_sto... | Δ | rewrite_batchnorm... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 6400 ns | -2590 ns (-28.8%) | 6400 ns | -2590 ns (-28.8%) | 7490 ns | -1500 ns (-16.7%) | 11550 ns | +2560 ns (+28.5%) | 6400 ns | -2590 ns (-28.8%) |
| SM Throughput (%) | 54.41 % | 34.05 % | -20.36 pp | 34.07 % | -20.34 pp | 51.29 % | -3.12 pp | 66.34 % | +11.93 pp | 34.21 % | -20.20 pp |
| Memory Throughput (%) | 20.55 % | 28.86 % | +8.31 pp | 28.88 % | +8.33 pp | 24.69 % | +4.14 pp | 29.26 % | +8.71 pp | 28.85 % | +8.30 pp |
| DRAM Throughput (%) | 20.55 % | 28.86 % | +8.31 pp | 28.88 % | +8.33 pp | 24.69 % | +4.14 pp | 15.97 % | -4.58 pp | 28.85 % | +8.30 pp |
| Achieved Occupancy (%) | 77.23 % | 62.46 % | -14.77 pp | 62.96 % | -14.27 pp | 75.50 % | -1.73 pp | 82.37 % | +5.14 pp | 62.33 % | -14.90 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 2.78 | 1.39 | -1.39 | 1.39 | -1.39 | 2.78 | +0.00 | 5.57 | +2.79 | 1.39 | -1.39 |
| No Eligible (%) | 25.19 % | 34.69 % | +9.50 pp | 35.37 % | +10.18 pp | 27.19 % | +2.00 pp | 19.29 % | -5.90 pp | 34.45 % | +9.26 pp |
| IPC Active | 2.85 inst/cycle | 2.27 inst/cycle | -0.58 | 2.48 inst/cycle | -0.37 | 2.88 inst/cycle | +0.03 | 3.17 inst/cycle | +0.32 | 2.47 inst/cycle | -0.38 |


### Key Learnings

#### ✅ Plan A: `reduce_num_warps_to_alleviate_cpi_stall_scheduling_contention` (1.405×)

When CPIStall 'not-selected' accounts for >40% of stall cycles and eligible warps per scheduler exceeds ~5, reducing num_warps (halving warps per block from 4 to 2) cuts per-scheduler warp pressure and reduces total executed instructions. In this case, halving num_warps from 4 to 2 at fixed XBLOCK=128 dropped executed instructions by 43% (3.03M to 1.72M), reduced kernel duration from 7.58 µs to 6.40 µs (1.40x speedup), and improved memory throughput from 1.19 TB/s to 1.41 TB/s. The Block Limit Registers constraint relaxed from 16 to 32 blocks/SM (fewer warps per block means more blocks can fit), and Warp Cycles Per Issued Instruction improved slightly (16.63 → 15.97 cycles), confirming reduced scheduling contention.

> **Applicable when**: CPIStall 'not-selected' accounts for >35% of total stall cycles; eligible warps per scheduler exceeds 4-5; the kernel is not memory-bandwidth-bound (DRAM SOL <40%); XBLOCK is already at its minimum safe value (cannot be reduced further without correctness issues); registers per thread are at 32 and Block Limit Registers is the binding occupancy constraint.

> **Avoid when**: The kernel is already memory-bandwidth-bound (DRAM SOL >60%) where reducing warps would starve the memory pipeline; the kernel has very few waves per SM (<1.0) where reducing warps per block would leave SMs idle; num_warps is already at minimum (1 or 2) and further reduction would cause insufficient parallelism to hide memory latency.

#### ✅ Plan B: `reduce_num_warps_to_alleviate_cpi_not_selected_stall` (1.405×)

When the NCU CPIStall rule flags 'not-selected' stalls as the dominant bottleneck (>40% of warp cycles), reducing num_warps per block (e.g., from 4 to 2, halving block_size from 128 to 64 threads) directly reduces scheduler contention and yields measurable wall-clock improvement. In this case, halving num_warps reduced kernel duration from 7.58 µs to 6.40 µs (1.40x speedup) by cutting total executed instructions nearly in half (3.03M → 1.72M) and improving memory throughput from 1.19 TB/s to 1.41 TB/s. Although achieved occupancy dropped from 76.4% to 63.0% and Block Limit Registers relaxed from 16 to 32 (fewer warps per block means less register pressure per SM), the reduced scheduler contention and improved cache locality per warp more than compensated for the lower occupancy.

> **Applicable when**: NCU CPIStall rule is triggered with 'not-selected' stalls comprising >35% of warp cycles; Active Warps Per Scheduler is high (>10) relative to Eligible Warps Per Scheduler; achieved occupancy is well above 50% (meaning there is headroom to trade occupancy for reduced contention); the kernel is a short-running pointwise/elementwise kernel where scheduler contention dominates over memory latency hiding.

> **Avoid when**: The kernel is memory-latency-bound (DRAM throughput >60% of peak) where high warp count is needed to hide memory latency; achieved occupancy is already near 50% or below, leaving no headroom to reduce warps; the kernel has a large working set per warp that benefits from L2 reuse across many concurrent warps; waves_per_sm is already ≤1 (reducing warps further would leave SMs idle).

#### ✅ Plan D: `rewrite_batchnorm_epilogue_as_explicit_fma_to_reduce_nonfused_fp32_instructions` (1.392×)

This iteration applied three simultaneous changes: (1) replacing the two-instruction BN epilogue MUL+ADD (tmp49*tmp50; tmp51+tmp52) with tl.math.fma(tmp49, tmp50, tmp52), (2) switching input streaming loads from evict_last to evict_first, and (3) adding num_warps=1 to the autotune config set while removing num_stages=2 configs. The profiling data shows the kernel duration barely changed (6.40us → 6.46us per NCU), yet the reported end-to-end speedup is 1.39x — suggesting the improvement comes from reduced kernel launch overhead or autotuner selecting num_warps=1 (block_size=32) rather than from the FMA arithmetic change itself. Notably, executed instructions actually increased slightly (1.72M → 1.78M) and Executed IPC Active improved (2.27 → 2.48), indicating the FMA did not reduce instruction count but may have improved instruction throughput. The dominant causal factor is likely the autotune config pruning (removing num_stages=2 configs and adding num_warps=1) rather than the FMA rewrite.

> **Applicable when**: FPInstructions NCU rule is triggered with high non-fused to fused FP32 ratio (>5:1); kernel has a BN epilogue with a MUL followed by ADD on the same intermediate (a*b + c pattern); autotune config set includes num_stages=2 configs that add overhead without benefit for short-running (<10us) pointwise kernels; CPIStall not-selected is still present (>30% of stall cycles) suggesting num_warps=1 may further reduce contention.

> **Avoid when**: The Triton/LLVM backend already auto-fuses MUL+ADD into FFMA (verify by checking if non-fused instruction count is already low after prior optimizations); the kernel is memory-bandwidth-bound (DRAM SOL >60%) where FMA fusion yields no benefit; the FMA change is bundled with autotune config changes that include XBLOCK>128 for kernels with cat()/where() boundary logic (XBLOCK inflation will dominate and mask any FMA benefit).

---

## Iteration 5

**Best so far**: 1.4047× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_partial_wave_tail_via_smaller_xblock_and_num_warps1 | 1.154× | 1.398× | ✓ | **✓ best** |
| B | rewrite_bn_epilogue_as_fma_to_reduce_nonfused_fp32_instructions | 1.117× | 1.392× | ✓ |  |
| C | reduce_num_warps_to_2_to_alleviate_cpi_not_selected_scheduler_contention | 1.061× | 1.332× | ✓ |  |
| D | restructure_bn_epilogue_to_explicit_fma_chain_to_reduce_nonfused_fp32_instructions | 1.116× | 0.763× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_partial_wa... | Δ | rewrite_bn_epilog... | Δ | reduce_num_warps_... | Δ | restructure_bn_ep... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8990 ns | 6430 ns | -2560 ns (-28.5%) | 6460 ns | -2530 ns (-28.1%) | 6750 ns | -2240 ns (-24.9%) | 11780 ns | +2790 ns (+31.0%) |
| SM Throughput (%) | 54.41 % | 16.85 % | -37.56 pp | 35.28 % | -19.13 pp | 19.87 % | -34.54 pp | 66.40 % | +11.99 pp |
| Memory Throughput (%) | 20.55 % | 14.39 % | -6.16 pp | 28.72 % | +8.17 pp | 27.34 % | +6.79 pp | 29.27 % | +8.72 pp |
| DRAM Throughput (%) | 20.55 % | 14.39 % | -6.16 pp | 28.72 % | +8.17 pp | 27.34 % | +6.79 pp | 15.67 % | -4.88 pp |
| Achieved Occupancy (%) | 77.23 % | 21.34 % | -55.89 pp | 64.80 % | -12.43 pp | 27.56 % | -49.67 pp | 82.18 % | +4.95 pp |
| Registers/Thread | 32.00 register/thread | 42.00 register/thread | +10 | 32.00 register/thread | +0 | 60.00 register/thread | +28 | 32.00 register/thread | +0 |
| Waves/SM | 2.78 | 1.39 | -1.39 | 1.39 | -1.39 | 1.39 | -1.39 | 5.57 | +2.79 |
| No Eligible (%) | 25.19 % | 62.05 % | +36.86 pp | 33.90 % | +8.71 pp | 63.01 % | +37.82 pp | 21.46 % | -3.73 pp |
| IPC Active | 2.85 inst/cycle | 1.36 inst/cycle | -1.49 | 2.48 inst/cycle | -0.37 | 1.36 inst/cycle | -1.49 | 3.21 inst/cycle | +0.36 |

---

## Cumulative Progress

```
Baseline:  8990 ns  (1.000×)
Iter 2     7620 ns  (1.180×)  — replace_sqrt_rcp_with_rsqrt_to_reduce_nonfused_fp_instructions
Iter 4     6400 ns  (1.405×)  — reduce_num_warps_to_alleviate_cpi_stall_scheduling_contention

Final best: 6430 ns  (1.3981×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 64},  num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
     xnumel = 752640
@@ -16,21 +26,21 @@

     x2 = xindex // 5376
     x3 = xindex // 384
     x4 = xindex
-    tmp39 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
-    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
-    tmp50 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
-    tmp52 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
+    tmp39 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp41 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp50 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp52 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 256, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -40,21 +50,21 @@

     tmp17 = tl.full([1], 288, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 320, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 352, tl.int64)
     tmp28 = tmp0 < tmp27
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), mask=tmp29 & xmask, eviction_policy='evict_last', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1], 384, tl.int64)
     tmp33 = tmp0 < tmp32
-    tmp34 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp34 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), mask=tmp31 & xmask, eviction_policy='evict_last', other=0.0)
     tmp35 = tl.where(tmp29, tmp30, tmp34)
     tmp36 = tl.where(tmp24, tmp25, tmp35)
     tmp37 = tl.where(tmp19, tmp20, tmp36)
@@ -62,19 +72,13 @@

     tmp40 = tmp38 - tmp39
     tmp42 = tl.full([1], 1e-05, tl.float32)
     tmp43 = tmp41 + tmp42
-    tmp44 = tl.sqrt_rn(tmp43)
-    tmp45 = tl.full([1], 1, tl.int32)
-    tmp46 = (tmp45 / tmp44)
-    tmp47 = tl.full([1], 1.0, tl.float32)
-    tmp48 = tmp46 * tmp47
+    tmp48 = libdevice.rsqrt(tmp43)
     tmp49 = tmp40 * tmp48
     tmp51 = tmp49 * tmp50
     tmp53 = tmp51 + tmp52
     tmp54 = tl.full([1], 0, tl.int32)
     tmp55 = tl.maximum(tmp54, tmp53)
-    tl.store(in_out_ptr0 + (x4), tmp55, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp55, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -89,7 +93,7 @@

     arg_9 = inputs[9]
     xnumel = 752640
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 752640, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 752640)
     return arg_9
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
