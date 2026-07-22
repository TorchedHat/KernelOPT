# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 21440 ns — **Best latency**: 13540 ns (**1.5835×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`align_xblock_to_concat_boundaries_to_eliminate_cross_branch_predication_and_fix_workload_imbalance`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_23_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (17640,1,1), Waves/SM: 8.35
- **Registers/thread**: 32
- **Duration**: 21440 ns
- **SM Throughput**: 72.59 % | **Memory**: 26.04 % | **DRAM**: 20.43 %
- **Achieved Occupancy**: 85.20 %
- **IPC Active**: 3.24 inst/cycle | **No Eligible**: 18.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 0.996× | 0.983× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy_and_fma | 1.112× | 0.462× | ✓ | **✓ best** |
| C | autotune_xblock_for_vectorized_loads_and_fma_fusion | 0.881× | 0.861× | ✓ |  |
| D | autotune_xblock_for_fma_and_occupancy | 0.987× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | autotune_xblock_f... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21440 ns | 21820 ns | +380 ns (+1.8%) | 46400 ns | +24960 ns (+116.4%) | 24900 ns | +3460 ns (+16.1%) | 21920 ns | +480 ns (+2.2%) |
| SM Throughput (%) | 72.59 % | 73.21 % | +0.62 pp | 73.73 % | +1.14 pp | 77.33 % | +4.74 pp | 72.95 % | +0.36 pp |
| Memory Throughput (%) | 26.04 % | 26.26 % | +0.22 pp | 26.02 % | -0.02 pp | 27.23 % | +1.19 pp | 26.17 % | +0.13 pp |
| DRAM Throughput (%) | 20.43 % | 19.96 % | -0.47 pp | 9.38 % | -11.05 pp | 17.46 % | -2.97 pp | 19.84 % | -0.59 pp |
| Achieved Occupancy (%) | 85.20 % | 85.46 % | +0.26 pp | 51.89 % | -33.31 pp | 85.90 % | +0.70 pp | 85.63 % | +0.43 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 42.00 register/thread | +10 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 8.35 | 8.35 | +0.00 | 26.73 | +18.38 | 8.35 | +0.00 | 8.35 | +0.00 |
| No Eligible (%) | 18.56 % | 18.46 % | -0.10 pp | 22.25 % | +3.69 pp | 15.84 % | -2.72 pp | 18.51 % | -0.05 pp |
| IPC Active | 3.24 inst/cycle | 3.24 inst/cycle | +0.00 | 3.10 inst/cycle | -0.14 | 3.36 inst/cycle | +0.12 | 3.21 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_reciprocal_sqrt_with_libdevice_rsqrt | 1.117× | 1.147× | ✓ |  |
| B | replace_sqrt_div_sequence_with_rsqrt_for_fma_fusion | 1.113× | 1.157× | ✓ |  |
| C | replace_sqrt_reciprocal_with_rsqrt_and_fix_xblock_to_128 | 1.130× | 1.159× | ✓ |  |
| D | restore_small_xblock_to_recover_register_limited_occupancy | 1.130× | 1.151× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.987× | 0.965× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_re... | Δ | replace_sqrt_div_... | Δ | replace_sqrt_reci... | Δ | restore_small_xbl... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21440 ns | 18690 ns | -2750 ns (-12.8%) | 18530 ns | -2910 ns (-13.6%) | 18500 ns | -2940 ns (-13.7%) | 18620 ns | -2820 ns (-13.2%) | 22210 ns | +770 ns (+3.6%) |
| SM Throughput (%) | 72.59 % | 67.43 % | -5.16 pp | 68.60 % | -3.99 pp | 68.14 % | -4.45 pp | 68.41 % | -4.18 pp | 72.96 % | +0.37 pp |
| Memory Throughput (%) | 26.04 % | 30.34 % | +4.30 pp | 30.87 % | +4.83 pp | 30.66 % | +4.62 pp | 30.79 % | +4.75 pp | 26.18 % | +0.14 pp |
| DRAM Throughput (%) | 20.43 % | 23.29 % | +2.86 pp | 23.48 % | +3.05 pp | 23.52 % | +3.09 pp | 23.34 % | +2.91 pp | 19.61 % | -0.82 pp |
| Achieved Occupancy (%) | 85.20 % | 83.22 % | -1.98 pp | 83.48 % | -1.72 pp | 82.89 % | -2.31 pp | 83.16 % | -2.04 pp | 85.32 % | +0.12 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 8.35 | 8.35 | +0.00 | 8.35 | +0.00 | 8.35 | +0.00 | 8.35 | +0.00 | 8.35 | +0.00 |
| No Eligible (%) | 18.56 % | 22.16 % | +3.60 pp | 21.99 % | +3.43 pp | 21.86 % | +3.30 pp | 22.19 % | +3.63 pp | 18.14 % | -0.42 pp |
| IPC Active | 3.24 inst/cycle | 3.09 inst/cycle | -0.15 | 3.08 inst/cycle | -0.16 | 3.10 inst/cycle | -0.14 | 3.10 inst/cycle | -0.14 | 3.24 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.1515× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_num_warps_to_cut_scheduler_contention_and_improve_cache_locality | 1.116× | 1.163× | ✓ |  |
| B | reduce_num_warps_to_cut_scheduler_contention_and_improve_l1_coherence | 1.218× | 1.580× | ✓ |  |
| C | replace_constant_division_with_multiply_for_fma_fusion | 1.122× | 1.151× | ✓ |  |
| D | fuse_batchnorm_epilogue_with_explicit_fma_and_sweep_num_warps | 1.219× | 1.155× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_num_warps_... | Δ | reduce_num_warps_... | Δ | replace_constant_... | Δ | fuse_batchnorm_ep... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21440 ns | 18430 ns | -3010 ns (-14.0%) | 13570 ns | -7870 ns (-36.7%) | 18620 ns | -2820 ns (-13.2%) | 18560 ns | -2880 ns (-13.4%) |
| SM Throughput (%) | 72.59 % | 68.59 % | -4.00 pp | 52.74 % | -19.85 pp | 68.55 % | -4.04 pp | 68.71 % | -3.88 pp |
| Memory Throughput (%) | 26.04 % | 30.89 % | +4.85 pp | 32.03 % | +5.99 pp | 30.88 % | +4.84 pp | 30.92 % | +4.88 pp |
| DRAM Throughput (%) | 20.43 % | 23.64 % | +3.21 pp | 32.03 % | +11.60 pp | 23.36 % | +2.93 pp | 23.44 % | +3.01 pp |
| Achieved Occupancy (%) | 85.20 % | 82.86 % | -2.34 pp | 39.04 % | -46.16 pp | 83.02 % | -2.18 pp | 83.07 % | -2.13 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 34.00 register/thread | +2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 8.35 | 8.35 | +0.00 | 5.57 | -2.78 | 8.35 | +0.00 | 8.35 | +0.00 |
| No Eligible (%) | 18.56 % | 21.74 % | +3.18 pp | 34.73 % | +16.17 pp | 22.45 % | +3.89 pp | 22.95 % | +4.39 pp |
| IPC Active | 3.24 inst/cycle | 3.09 inst/cycle | -0.15 | 2.72 inst/cycle | -0.52 | 3.10 inst/cycle | -0.14 | 3.10 inst/cycle | -0.14 |


### Key Learnings

#### ✅ Plan B: `reduce_num_warps_to_cut_scheduler_contention_and_improve_l1_coherence` (1.547×)

Reducing num_warps from 2 to 1 (block_size 64→32) further halved active warps per scheduler from 6.04 to 3.21 and cut Warp Cycles Per Issued Instruction from 9.25 to 7.47, yet kernel duration barely changed (13.57 µs → 13.86 µs, effectively flat or a slight regression at the single-kernel level). The speedup of 1.55x reported is likely an end-to-end measurement artifact rather than a per-kernel improvement, since the NCU profiling shows duration increased slightly. Critically, registers_per_thread nearly doubled (34 → 64), which collapsed theoretical occupancy from 75% to 50% and achieved occupancy from 39% to 19%, and the IssueSlotUtilization rule was newly triggered (57.11% idle slots). The instruction count dropped from 5.64M to 3.44M (39% reduction), but the scheduler became so starved (only 0.78 eligible warps/scheduler) that the latency-hiding benefit of fewer warps was lost. This reveals a register-pressure cliff: at num_warps=1 with XBLOCK=128, each thread processes 4 elements and the compiler allocates 64 registers/thread, which is the binding occupancy constraint.

> **Applicable when**: num_warps=2 is already selected and registers_per_thread remains stable (does not increase) when moving to num_warps=1; CPIStall not-selected stall is still dominant (>40% of stall cycles); achieved occupancy is above 30% at num_warps=2 so there is headroom to reduce further without starving the scheduler.

> **Avoid when**: registers_per_thread doubles when moving from num_warps=2 to num_warps=1 (as happened here: 34→64 regs/thread), which collapses theoretical occupancy and triggers IssueSlotUtilization; when achieved occupancy is already below 40% at the current num_warps setting; when the IssueSlotUtilization rule is triggered after the change (indicating scheduler starvation); when eligible warps per scheduler drops below 1.0, meaning the scheduler has no ready warps to hide latency.

#### ✅ Plan C: `replace_constant_division_with_multiply_for_fma_fusion` (1.151×)

Replacing a float division by a compile-time constant (e.g., `x / 4.0` via `tl.full([1], 4.0, tl.float32)`) with multiplication by its reciprocal (`x * 0.25`) did not reduce the executed instruction count (9.878M before and after) and did not improve kernel duration (18.5 µs → 18.62 µs, a slight regression). The FPInstructions NCU rule remained triggered with the same non-fused instruction count, indicating that the LLVM/Triton compiler already handles constant-divisor float division efficiently and the substitution did not enable additional FMA fusion. When the dominant source of non-fused FP32 instructions has already been addressed (e.g., via rsqrt), remaining non-fused instructions may be structurally unfusable regardless of arithmetic reformulation.

> **Applicable when**: Never: this substitution did not improve performance even when the FPInstructions NCU rule was triggered. The compiler already optimizes constant-divisor float division to FMUL internally.

> **Avoid when**: When the dominant non-fused FP32 instruction source has already been eliminated (e.g., rsqrt already applied). When executed instruction count does not decrease after the substitution, the FMA fusion opportunity does not exist. Do not apply when the kernel is already near its arithmetic throughput ceiling and the remaining non-fused instructions are structurally unfusable (e.g., independent FADD chains without a multiply partner).

---

## Iteration 4

**Best so far**: 1.1552× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_per_thread_to_close_occupancy_gap | 1.234× | 1.555× | ✓ | **✓ best** |
| B | differentiate_eviction_policies_streaming_vs_broadcast_tensors | 1.143× | 1.143× | ✓ |  |
| C | minimize_warps_per_block_with_pipelining_to_drain_scheduler_queue | 1.207× | 1.547× | ✓ |  |
| D | reduce_xblock_to_64_to_cut_predicated_branch_overhead_in_concat_kernel | 1.231× | 1.569× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_per... | Δ | differentiate_evi... | Δ | minimize_warps_pe... | Δ | reduce_xblock_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21440 ns | 13790 ns | -7650 ns (-35.7%) | 18750 ns | -2690 ns (-12.5%) | 13860 ns | -7580 ns (-35.4%) | 13660 ns | -7780 ns (-36.3%) |
| SM Throughput (%) | 72.59 % | 53.36 % | -19.23 pp | 68.85 % | -3.74 pp | 31.76 % | -40.83 pp | 46.51 % | -26.08 pp |
| Memory Throughput (%) | 26.04 % | 31.64 % | +5.60 pp | 31.00 % | +4.96 pp | 31.48 % | +5.44 pp | 21.04 % | -5.00 pp |
| DRAM Throughput (%) | 20.43 % | 31.64 % | +11.21 pp | 23.17 % | +2.74 pp | 31.48 % | +11.05 pp | 15.75 % | -4.68 pp |
| Achieved Occupancy (%) | 85.20 % | 38.47 % | -46.73 pp | 83.07 % | -2.13 pp | 19.12 % | -66.08 pp | 40.91 % | -44.29 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 | 32.00 register/thread | +0 | 64.00 register/thread | +32 | 32.00 register/thread | +0 |
| Waves/SM | 8.35 | 5.57 | -2.78 | 8.35 | +0.00 | 4.18 | -4.17 | 4.18 | -4.17 |
| No Eligible (%) | 18.56 % | 32.12 % | +13.56 pp | 22.02 % | +3.46 pp | 57.11 % | +38.55 pp | 36.86 % | +18.30 pp |
| IPC Active | 3.24 inst/cycle | 2.70 inst/cycle | -0.54 | 3.10 inst/cycle | -0.14 | 1.63 inst/cycle | -1.61 | 2.49 inst/cycle | -0.75 |


### Key Learnings

#### ✅ Plan D: `reduce_xblock_to_64_to_cut_predicated_branch_overhead_in_concat_kernel` (1.592×)

Adding XBLOCK=256 and XBLOCK=512 autotune configs (with num_warps=4 and num_warps=8) alongside the existing XBLOCK=64/128 configs produced a marginal 1.3% duration reduction (13.66 µs → 13.47 µs) with the autotuner still selecting XBLOCK=64/num_warps=2 as the winner (block_size=64, registers_per_thread=32 unchanged). The explicit tl.math.fma() rewrite and tl.full([1], 1e-05) → scalar literal simplification also had no measurable effect on instruction count (4,939,200 before and after), consistent with prior findings that Triton's JIT already auto-fuses sequential multiply-add into FFMA. The small speedup (1.59x vs prior 1.57x) is attributable to the expanded autotune search space allowing the autotuner to confirm XBLOCK=64 more confidently, with a slight improvement in eligible warps per scheduler (2.55 → 2.61) and warp cycles per issued instruction (10.23 → 10.05).

> **Applicable when**: Autotune config space is small (≤3 configs) and the winning config is at the boundary of the search space (e.g., smallest XBLOCK); adding larger XBLOCK configs with proportionally larger num_warps allows the autotuner to confirm the optimal config and may expose a better config. Kernel is already near its performance floor with CPIStall as the dominant bottleneck.

> **Avoid when**: The autotuner has already explored a wide range of XBLOCK values and the winning config is stable; adding more configs only increases autotuning overhead without changing the selected config. Do not expect tl.math.fma() or scalar literal substitutions (tl.full([1], c, dtype) → c) to reduce instruction count when Triton's JIT already auto-fuses these patterns.

---

## Iteration 5

**Best so far**: 1.5547× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_non_fma_fp32_instructions_via_explicit_multiply_add_rewrite_and_grid_alignment | 1.231× | 1.592× | ✓ |  |
| B | align_xblock_to_concat_boundaries_to_eliminate_cross_branch_predication_and_fix_workload_imbalance | 1.267× | 1.583× | ✓ | **✓ best** |
| C | align_xblock_to_channel_period_to_eliminate_predicated_branch_divergence | 1.205× | 1.405× | ✓ |  |
| D | add_triton_autotune | 1.126× | 1.159× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_non_fma_fp32... | Δ | align_xblock_to_c... | Δ | align_xblock_to_c... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 21440 ns | 13470 ns | -7970 ns (-37.2%) | 13540 ns | -7900 ns (-36.8%) | 15260 ns | -6180 ns (-28.8%) | 18500 ns | -2940 ns (-13.7%) |
| SM Throughput (%) | 72.59 % | 46.41 % | -26.18 pp | 47.38 % | -25.21 pp | 57.25 % | -15.34 pp | 68.35 % | -4.24 pp |
| Memory Throughput (%) | 26.04 % | 21.01 % | -5.03 pp | 20.05 % | -5.99 pp | 30.84 % | +4.80 pp | 30.79 % | +4.75 pp |
| DRAM Throughput (%) | 20.43 % | 15.99 % | -4.44 pp | 7.96 % | -12.47 pp | 28.46 % | +8.03 pp | 23.53 % | +3.10 pp |
| Achieved Occupancy (%) | 85.20 % | 42.37 % | -42.83 pp | 43.79 % | -41.41 pp | 42.56 % | -42.64 pp | 82.55 % | -2.65 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 42.00 register/thread | +10 | 32.00 register/thread | +0 |
| Waves/SM | 8.35 | 4.18 | -4.17 | 4.18 | -4.17 | 6.68 | -1.67 | 8.35 | +0.00 |
| No Eligible (%) | 18.56 % | 35.75 % | +17.19 pp | 34.63 % | +16.07 pp | 30.74 % | +12.18 pp | 21.95 % | +3.39 pp |
| IPC Active | 3.24 inst/cycle | 2.53 inst/cycle | -0.71 | 2.57 inst/cycle | -0.67 | 2.74 inst/cycle | -0.50 | 3.10 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  21440 ns  (1.000×)
Iter 2     18620 ns  (1.151×)  — restore_small_xblock_to_recover_register_limited_occupancy
Iter 3     18560 ns  (1.155×)  — fuse_batchnorm_epilogue_with_explicit_fma_and_sweep_num_warps
Iter 4     13790 ns  (1.555×)  — cap_registers_per_thread_to_close_occupancy_gap
Iter 5     13540 ns  (1.583×)  — align_xblock_to_concat_boundaries_to_eliminate_cross_branch_predication_and_fix_workload_imbalance

Final best: 13540 ns  (1.5835×)
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
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2),  # PRIMARY: 1 segment/block, no cross-boundary predication
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=3),  # PRIMARY+pipeline
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),  # 2 segments fallback
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=3),  # 2 segments + pipeline fallback
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),  # baseline fallback
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=3),  # baseline + pipeline fallback
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2257920
@@ -16,21 +28,21 @@

     x2 = xindex // 8064
     x3 = xindex // 288
     x4 = xindex
-    tmp45 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
-    tmp47 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
-    tmp56 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
-    tmp58 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
+    tmp45 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp47 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp56 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp58 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 128, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -40,26 +52,26 @@

     tmp17 = tl.full([1], 160, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 192, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 224, tl.int64)
     tmp28 = tmp0 < tmp27
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), mask=tmp29 & xmask, eviction_policy='evict_last', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1], 256, tl.int64)
     tmp33 = tmp0 < tmp32
     tmp34 = tmp31 & tmp33
-    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-224) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-224) + x0)), mask=tmp34 & xmask, eviction_policy='evict_last', other=0.0)
     tmp36 = tmp0 >= tmp32
     tmp37 = tl.full([1], 288, tl.int64)
     tmp38 = tmp0 < tmp37
-    tmp39 = tl.load(in_ptr5 + (32*x3 + ((-256) + x0)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp39 = tl.load(in_ptr5 + (32*x3 + ((-256) + x0)), mask=tmp36 & xmask, eviction_policy='evict_last', other=0.0)
     tmp40 = tl.where(tmp34, tmp35, tmp39)
     tmp41 = tl.where(tmp29, tmp30, tmp40)
     tmp42 = tl.where(tmp24, tmp25, tmp41)
@@ -68,19 +80,13 @@

     tmp46 = tmp44 - tmp45
     tmp48 = tl.full([1], 1e-05, tl.float32)
     tmp49 = tmp47 + tmp48
-    tmp50 = tl.sqrt_rn(tmp49)
-    tmp51 = tl.full([1], 1, tl.int32)
-    tmp52 = (tmp51 / tmp50)
-    tmp53 = tl.full([1], 1.0, tl.float32)
-    tmp54 = tmp52 * tmp53
-    tmp55 = tmp46 * tmp54
+    tmp50 = libdevice.rsqrt(tmp49)
+    tmp55 = tmp46 * tmp50
     tmp57 = tmp55 * tmp56
     tmp59 = tmp57 + tmp58
     tmp60 = tl.full([1], 0, tl.int32)
     tmp61 = tl.maximum(tmp60, tmp59)
-    tl.store(in_out_ptr0 + (x4), tmp61, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp61, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -96,7 +102,7 @@

     arg_10 = inputs[10]
     xnumel = 2257920
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 2257920, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 2257920)
     return arg_10
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
