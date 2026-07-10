# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 39230 ns — **Best latency**: 11620 ns (**3.3761×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_redundant_broadcast_and_int32_cast_to_reduce_nonfused_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_094/kernel_1_triton_per_fused_add_addmm_har/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 3.88
- **Registers/thread**: 106
- **Duration**: 39230 ns
- **SM Throughput**: 58.09 % | **Memory**: 18.25 % | **DRAM**: 18.02 %
- **Achieved Occupancy**: 23.55 %
- **IPC Active**: 2.51 inst/cycle | **No Eligible**: 37.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_num_warps_eliminate_partial_wave | 1.089× | 0.657× | ✓ |  |
| B | reduce_register_pressure_via_autotune_xblock | 3.153× | 3.397× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_n... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 39230 ns | 59680 ns | +20450 ns (+52.1%) | 11550 ns | -27680 ns (-70.6%) |
| SM Throughput (%) | 58.09 % | 38.89 % | -19.20 pp | 51.92 % | -6.17 pp |
| Memory Throughput (%) | 18.25 % | 11.77 % | -6.48 pp | 16.04 % | -2.21 pp |
| DRAM Throughput (%) | 18.02 % | 11.77 % | -6.25 pp | 14.95 % | -3.07 pp |
| Achieved Occupancy (%) | 23.55 % | 12.26 % | -11.29 pp | 37.45 % | +13.90 pp |
| Registers/Thread | 106.00 register/thread | 211.00 register/thread | +105 | 54.00 register/thread | -52 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 0.86 | -3.02 |
| No Eligible (%) | 37.65 % | 58.08 % | +20.43 pp | 33.65 % | -4.00 pp |
| IPC Active | 2.51 inst/cycle | 1.67 inst/cycle | -0.84 | 2.56 inst/cycle | +0.05 |


### Key Learnings

#### ✅ Plan B: `reduce_register_pressure_via_autotune_xblock` (3.388×)

When a Triton autotune config list excludes small XBLOCK values (e.g., minimum XBLOCK=128 or 256), the autotuner is forced to select a large-tile configuration that drives up registers-per-thread (96 at XBLOCK=128 vs. 54 at XBLOCK=64), collapsing Block Limit Registers from 18 to 5 and theoretical occupancy from 56.25% to 31.25%. Restoring small XBLOCK configs (32, 64) to the autotune search space allows the autotuner to rediscover the register-efficient configuration, recovering the full 3.4x speedup. The key signal is TheoreticalOccupancy NCU rule firing with Block Limit Registers as the binding constraint and registers-per-thread significantly higher than a known-good smaller-block baseline.

> **Applicable when**: TheoreticalOccupancy NCU rule fires; Block Limit Registers is the binding occupancy constraint (much lower than Block Limit Warps or Block Limit SM); registers-per-thread is elevated (>64) due to autotuner selecting a large XBLOCK; a prior known-good configuration with smaller XBLOCK and lower register count exists (e.g., from profiling history or earlier iterations); the autotune config list has been narrowed or modified to exclude small XBLOCK values.

> **Avoid when**: The autotune config list already includes small XBLOCK values (32, 64); the kernel is memory-bandwidth bound (DRAM throughput >70%); XBLOCK>=256 is required for correctness or problem-size alignment; the register pressure source is the reduction dimension (R0_BLOCK) rather than XBLOCK; xnumel is not divisible by smaller XBLOCK candidates without adding boundary masking that changes correctness.

---

## Iteration 2

**Best so far**: 3.3965× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_tile_size_fix_grid_lambda_expand_autotune | 3.220× | 3.388× | ✓ | **✓ best** |
| B | fix_hardcoded_grid_and_streaming_eviction_policy | 3.131× | 3.330× | ✓ |  |
| C | increase_xblock_tile_size_to_improve_parallelism_and_eliminate_partial_wave | 0.643× | 0.386× | ✓ |  |
| D | welford_online_variance_to_reduce_register_pressure | 1.092× | 1.054× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ | fix_hardcoded_gri... | Δ | increase_xblock_t... | Δ | welford_online_va... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39230 ns | 11580 ns | -27650 ns (-70.5%) | 11780 ns | -27450 ns (-70.0%) | 101700 ns | +62470 ns (+159.2%) | 37220 ns | -2010 ns (-5.1%) |
| SM Throughput (%) | 58.09 % | 52.02 % | -6.07 pp | 52.24 % | -5.85 pp | 42.83 % | -15.26 pp | 63.58 % | +5.49 pp |
| Memory Throughput (%) | 18.25 % | 16.07 % | -2.18 pp | 16.00 % | -2.25 pp | 10.61 % | -7.64 pp | 19.37 % | +1.12 pp |
| DRAM Throughput (%) | 18.02 % | 14.92 % | -3.10 pp | 14.71 % | -3.31 pp | 6.93 % | -11.09 pp | 18.94 % | +0.92 pp |
| Achieved Occupancy (%) | 23.55 % | 37.41 % | +13.86 pp | 37.47 % | +13.92 pp | 12.34 % | -11.21 pp | 28.32 % | +4.77 pp |
| Registers/Thread | 106.00 register/thread | 54.00 register/thread | -52 | 54.00 register/thread | -52 | 212.00 register/thread | +106 | 96.00 register/thread | -10 |
| Waves/SM | 3.88 | 0.86 | -3.02 | 0.86 | -3.02 | 7.76 | +3.88 | 3.10 | -0.78 |
| No Eligible (%) | 37.65 % | 33.93 % | -3.72 pp | 33.70 % | -3.95 pp | 54.95 % | +17.30 pp | 29.88 % | -7.77 pp |
| IPC Active | 2.51 inst/cycle | 2.57 inst/cycle | +0.06 | 2.58 inst/cycle | +0.07 | 1.81 inst/cycle | -0.70 | 2.82 inst/cycle | +0.31 |


### Key Learnings

#### ✅ Plan D: `welford_online_variance_to_reduce_register_pressure` (3.376×)

Replacing a 2-pass variance (sum then sum-of-squared-deviations) with a single-pass sum+sum-of-squares approach, removing a redundant broadcast, and collapsing a 3-way addition to enable FMA fusion produced essentially no change in the profiled kernel's NCU metrics: duration remained ~11.6 µs, registers/thread stayed at 54, occupancy held at ~37-38%, and executed instruction count was nearly identical (4.66M vs 4.68M). The 3.38x reported speedup is attributable to the concurrent removal of large XBLOCK autotune configs (XBLOCK=256, 512 removed from search space), not to the arithmetic restructuring itself. When the autotune config space is already constrained to small XBLOCK (≤128) and register pressure is stable, algebraic simplification of the variance computation and FMA-enabling rewrites do not produce measurable per-kernel improvement.

> **Applicable when**: Kernel performs 2-pass group norm variance with a redundant broadcast and int32 constant cast; NCU FPInstructions rule fires with >1M non-fused FP32 instructions; registers/thread is already low (≤54) and occupancy is register-limited but stable; autotune config space is already constrained to small XBLOCK (≤128).

> **Avoid when**: Do not expect this arithmetic restructuring alone to produce measurable speedup when the kernel is already at a stable operating point (54 regs/thread, 56% theoretical occupancy, ~11.6 µs). The NCU FPInstructions rule may overestimate the benefit if Triton's LLVM backend already performs some FMA fusion. The 3.38x speedup in this iteration was driven by autotune config pruning (removing XBLOCK=256/512), not by the variance or FMA changes. Do not conflate the two effects.

---

## Iteration 3

**Best so far**: 3.3965× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_lambda_and_streaming_eviction_for_large_tensor | 3.210× | 3.359× | ✓ |  |
| B | restore_small_xblock_configs_to_recover_register_limited_occupancy | 3.151× | 3.388× | ✓ |  |
| C | restructure_arithmetic_for_fma_fusion_to_reduce_non_fused_fp32_instructions | 3.140× | 3.376× | ✓ |  |
| D | revert_to_small_xblock_configs_to_restore_register_limited_occupancy | 3.215× | 3.388× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | restore_small_xbl... | Δ | restructure_arith... | Δ | revert_to_small_x... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39230 ns | 11680 ns | -27550 ns (-70.2%) | 11580 ns | -27650 ns (-70.5%) | 11620 ns | -27610 ns (-70.4%) | 11580 ns | -27650 ns (-70.5%) |
| SM Throughput (%) | 58.09 % | 52.21 % | -5.88 pp | 50.75 % | -7.34 pp | 51.70 % | -6.39 pp | 51.92 % | -6.17 pp |
| Memory Throughput (%) | 18.25 % | 15.99 % | -2.26 pp | 15.66 % | -2.59 pp | 15.99 % | -2.26 pp | 16.03 % | -2.22 pp |
| DRAM Throughput (%) | 18.02 % | 14.79 % | -3.23 pp | 14.92 % | -3.10 pp | 14.89 % | -3.13 pp | 14.92 % | -3.10 pp |
| Achieved Occupancy (%) | 23.55 % | 37.52 % | +13.97 pp | 37.44 % | +13.89 pp | 38.04 % | +14.49 pp | 37.53 % | +13.98 pp |
| Registers/Thread | 106.00 register/thread | 54.00 register/thread | -52 | 54.00 register/thread | -52 | 54.00 register/thread | -52 | 54.00 register/thread | -52 |
| Waves/SM | 3.88 | 0.86 | -3.02 | 0.86 | -3.02 | 0.86 | -3.02 | 0.86 | -3.02 |
| No Eligible (%) | 37.65 % | 33.89 % | -3.76 pp | 33.94 % | -3.71 pp | 33.85 % | -3.80 pp | 34.23 % | -3.42 pp |
| IPC Active | 2.51 inst/cycle | 2.58 inst/cycle | +0.07 | 2.56 inst/cycle | +0.05 | 2.54 inst/cycle | +0.03 | 2.55 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 3.3965× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_slice_balance_via_program_id_swizzle | 3.198× | 3.322× | ✓ |  |
| B | l2_pid_swizzle_to_reduce_workload_imbalance_and_improve_achieved_occupancy | 3.167× | 3.313× | ✓ |  |
| C | eliminate_redundant_broadcast_and_int32_cast_to_reduce_nonfused_fp32_instructions | 3.221× | 3.376× | ✓ | **✓ best** |
| D | fix_grid_lambda_to_align_with_autotune_xblock_selection | 3.205× | 3.397× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_slice_balance_... | Δ | l2_pid_swizzle_to... | Δ | eliminate_redunda... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39230 ns | 11810 ns | -27420 ns (-69.9%) | 11840 ns | -27390 ns (-69.8%) | 11620 ns | -27610 ns (-70.4%) | 11550 ns | -27680 ns (-70.6%) |
| SM Throughput (%) | 58.09 % | 52.82 % | -5.27 pp | 48.05 % | -10.04 pp | 51.54 % | -6.55 pp | 51.96 % | -6.13 pp |
| Memory Throughput (%) | 18.25 % | 15.95 % | -2.30 pp | 15.56 % | -2.69 pp | 15.92 % | -2.33 pp | 15.91 % | -2.34 pp |
| DRAM Throughput (%) | 18.02 % | 14.68 % | -3.34 pp | 14.60 % | -3.42 pp | 14.91 % | -3.11 pp | 14.96 % | -3.06 pp |
| Achieved Occupancy (%) | 23.55 % | 37.84 % | +14.29 pp | 38.68 % | +15.13 pp | 37.41 % | +13.86 pp | 37.74 % | +14.19 pp |
| Registers/Thread | 106.00 register/thread | 54.00 register/thread | -52 | 48.00 register/thread | -58 | 54.00 register/thread | -52 | 54.00 register/thread | -52 |
| Waves/SM | 3.88 | 0.86 | -3.02 | 0.78 | -3.10 | 0.86 | -3.02 | 0.86 | -3.02 |
| No Eligible (%) | 37.65 % | 32.87 % | -4.78 pp | 38.96 % | +1.31 pp | 33.81 % | -3.84 pp | 32.98 % | -4.67 pp |
| IPC Active | 2.51 inst/cycle | 2.57 inst/cycle | +0.06 | 2.37 inst/cycle | -0.14 | 2.58 inst/cycle | +0.07 | 2.59 inst/cycle | +0.08 |


### Key Learnings

#### ❌ Plan A: `l2_slice_balance_via_program_id_swizzle` (3.388×)

Removing a GROUP_SIZE=8 program_id swizzle from a short-duration Triton kernel (~11.5 µs) produced essentially no change in any profiling metric: duration moved from 11.55 µs to 11.58 µs, achieved occupancy from 37.74% to 37.60%, and executed instruction count was identical (4.70M). The profiling data confirms that neither the presence nor absence of the swizzle materially affects this kernel's performance — the kernel is compute-bound at ~52% SM throughput with low memory pressure (<15% DRAM), and the swizzle overhead and its removal are both negligible at this scale. The 3.39x speedup reported is attributable entirely to prior iterations' autotune config restoration (small XBLOCK), not to this swizzle removal.

> **Applicable when**: Never applicable as a standalone optimization for this kernel class. Removing a PID swizzle from a short-duration, compute-bound kernel with low DRAM utilization (<15%) produces no measurable speedup.

> **Avoid when**: Do not expect swizzle removal to produce speedup when: (1) the kernel is already compute-bound (SM throughput ~50%+); (2) kernel duration is short (<15 µs); (3) DRAM throughput is low (<15%); (4) executed instruction count is unchanged between swizzle and no-swizzle variants. The WorkloadImbalance NCU rule firing does not imply swizzle removal will help — the imbalance is not the binding bottleneck in this regime.

#### ❌ Plan B: `l2_pid_swizzle_to_reduce_workload_imbalance_and_improve_achieved_occupancy` (3.305×)

A GROUP_SIZE=8 PID swizzle using group_id/local_id remapping was applied to a ~11.6 µs kernel targeting the WorkloadImbalance NCU rule (L2 slice imbalance ~15-22% above/below average). The result was a slight regression: duration increased from 11.55 µs to 11.87 µs, No-Eligible scheduler cycles worsened from 32.98% to 34.01%, and executed instructions increased from 4.70M to 4.81M, while L2 hit rate was essentially unchanged (18.7% → 18.57%). The swizzle integer arithmetic (cdiv, //, %, minimum) adds per-CTA overhead that exceeds any L2 locality benefit for kernels with short duration and modest absolute L2 traffic (~535 KB).

> **Applicable when**: Never applicable for short-duration kernels (<20 µs) with modest absolute L2 traffic (<1 MB). PID swizzling may help when kernel duration exceeds 100 µs, L2 traffic is large (>10 MB), and L2 slice imbalance is severe (>30% above/below average).

> **Avoid when**: Do NOT apply any variant of PID swizzle (linear group swizzle, Triton-style swizzle, or group_id/local_id remapping) when kernel duration is short (<20 µs) and absolute L2 traffic is modest (<1 MB). The integer arithmetic overhead of the swizzle adds per-CTA compute cost and increases instruction count, worsening No-Eligible scheduler cycles rather than hiding memory latency. This kernel has now been confirmed to regress with two different swizzle implementations (iterations 4 and 5); the WorkloadImbalance NCU rule speedup estimate is unreliable for short kernels where the absolute imbalance duration is negligible.

---

## Iteration 5

**Best so far**: 3.3965× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_num_stages_to_hide_load_latency | 3.166× | 3.313× | ✓ |  |
| B | fix_grid_lambda_and_evict_first_on_streaming_tensor_to_improve_l2_reuse | 3.126× | 3.305× | ✓ |  |
| C | add_num_stages_pipelining_to_hide_memory_latency | 3.195× | 3.313× | ✓ |  |
| D | remove_baked_in_swizzle_and_fix_grid_lambda_to_close_achieved_occupancy_gap | 3.138× | 3.388× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | fix_grid_lambda_a... | Δ | add_num_stages_pi... | Δ | remove_baked_in_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 39230 ns | 11840 ns | -27390 ns (-69.8%) | 11870 ns | -27360 ns (-69.7%) | 11840 ns | -27390 ns (-69.8%) | 11580 ns | -27650 ns (-70.5%) |
| SM Throughput (%) | 58.09 % | 47.41 % | -10.68 pp | 52.44 % | -5.65 pp | 51.15 % | -6.94 pp | 52.12 % | -5.97 pp |
| Memory Throughput (%) | 18.25 % | 15.35 % | -2.90 pp | 15.69 % | -2.56 pp | 15.46 % | -2.79 pp | 15.96 % | -2.29 pp |
| DRAM Throughput (%) | 18.02 % | 14.61 % | -3.41 pp | 14.57 % | -3.45 pp | 14.62 % | -3.40 pp | 14.94 % | -3.08 pp |
| Achieved Occupancy (%) | 23.55 % | 38.74 % | +15.19 pp | 37.44 % | +13.89 pp | 37.96 % | +14.41 pp | 37.60 % | +14.05 pp |
| Registers/Thread | 106.00 register/thread | 48.00 register/thread | -58 | 54.00 register/thread | -52 | 54.00 register/thread | -52 | 54.00 register/thread | -52 |
| Waves/SM | 3.88 | 0.78 | -3.10 | 0.86 | -3.02 | 0.86 | -3.02 | 0.86 | -3.02 |
| No Eligible (%) | 37.65 % | 38.87 % | +1.22 pp | 34.01 % | -3.64 pp | 33.96 % | -3.69 pp | 34.83 % | -2.82 pp |
| IPC Active | 2.51 inst/cycle | 2.36 inst/cycle | -0.15 | 2.61 inst/cycle | +0.10 | 2.56 inst/cycle | +0.05 | 2.58 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan A: `software_pipeline_num_stages_to_hide_load_latency` (3.313×)

Adding num_stages=2 and num_stages=3 variants to a Triton autotune config list for a short-duration (~11.6 µs) kernel that is already ~52% compute-bound produced a slight regression (3.31x vs 3.40x baseline). Despite registers-per-thread dropping from 54 to 48 (raising theoretical occupancy from 56.25% to 62.5%), the achieved occupancy barely moved (37.74% → 38.74%) and No-Eligible scheduler cycles worsened (32.98% → 38.87%), indicating that the winning config selected by the autotuner included the PID swizzle (bundled in the diff) rather than benefiting from software pipelining. The profiling signature — modest DRAM throughput (~15%), ~52% compute SOL, and short duration — indicates the kernel is compute-bound rather than memory-latency-bound, so num_stages>1 provides no benefit because there is no dominant streaming load stall to hide.

> **Applicable when**: num_stages>1 is beneficial when: DRAM throughput is high (>40%), No-Eligible scheduler cycles are high (>40%) due to confirmed memory latency stalls, kernel duration is long (>50 µs), and the kernel has a dominant streaming load that is not already cached in L2. The AchievedOccupancy NCU rule alone is insufficient justification if DRAM throughput is low.

> **Avoid when**: Do NOT add num_stages>1 autotune configs when the kernel is already compute-bound (SM throughput >45%, DRAM throughput <20%). In this regime, the autotuner may select a config that bundles other changes (e.g., PID swizzle) causing regression, and the software pipeline provides no benefit because there is no memory latency to hide. Also avoid when No-Eligible scheduler cycles are <35% — the stalls are likely compute dependencies, not memory latency. Do not bundle num_stages changes with other structural changes (PID swizzle) in the same iteration, as it makes causal attribution impossible.

#### ❌ Plan C: `add_num_stages_pipelining_to_hide_memory_latency` (3.313×)

Adding num_stages=2 and num_stages=3 autotune configs to a Triton kernel that performs a single-pass load (no inner reduction loop over tiles) produced a slight regression: duration increased from 11.55 µs to 11.84 µs, No-Eligible scheduler stalls worsened slightly (32.98% → 33.96%), and executed instructions increased by ~74K. Triton's software pipelining via num_stages emits cp.async prefetch instructions that overlap global memory loads with compute across loop iterations; when the kernel body has no tiled inner loop (all loads happen once per CTA), there is no loop boundary across which to pipeline, so num_stages>1 adds overhead (extra instructions, slightly more register pressure) without hiding any latency. The AchievedOccupancy gap vs. theoretical occupancy remained unchanged, confirming that the No-Eligible stalls are not caused by global memory latency but by a structural compute bottleneck that pipelining cannot address.

> **Applicable when**: Never applicable as described for this kernel class. num_stages>1 is only beneficial when the Triton kernel has an explicit inner loop over tiles (e.g., matmul K-dimension loop, reduction over many blocks) where the compiler can emit cp.async prefetches across loop iterations.

> **Avoid when**: Do NOT add num_stages>1 configs when the Triton kernel performs a single-pass load with no inner tiled loop — i.e., all global memory loads happen once per CTA and there is no loop boundary for the compiler to pipeline across. In this regime, num_stages>1 adds instruction overhead and may slightly increase register pressure without hiding any memory latency. The AchievedOccupancy gap and No-Eligible stall rate will not improve. Also do not apply when the No-Eligible stall rate (~33%) is caused by compute dependencies (ALU chains) rather than global memory latency — pipelining cannot hide compute-bound stalls.

---

## Cumulative Progress

```
Baseline:  39230 ns  (1.000×)
Iter 1     11550 ns  (3.397×)  — reduce_register_pressure_via_autotune_xblock

Final best: 11620 ns  (3.3761×)
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
+        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 262144
@@ -14,7 +26,7 @@

     RBLOCK: tl.constexpr = R0_BLOCK
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
+    xmask = xindex < xnumel
     r0_index = tl.arange(0, R0_BLOCK)[None, :]
     r0_offset = 0
     r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
@@ -23,9 +35,9 @@

     r0_2 = r0_index
     x0 = (xindex % 256)
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (r0_2 + 32*x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (r0_2 + 32*x3), None)
-    tmp3 = tl.load(in_ptr2 + (r0_2 + 32*x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (r0_2 + 32*x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (r0_2 + 32*x3), xmask, eviction_policy='evict_first')
+    tmp3 = tl.load(in_ptr2 + (r0_2 + 32*x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tmp4 = tmp2 + tmp3
     tmp5 = tl.full([1, 1], -1.0, tl.float32)
@@ -49,10 +61,8 @@

     tmp25 = tmp24 * tmp24
     tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
     tmp28 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
-    tl.store(out_ptr0 + (x3), tmp23, None)
-    tl.store(out_ptr1 + (x3), tmp28, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp23, xmask)
+    tl.store(out_ptr1 + (x3), tmp28, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -63,7 +73,7 @@

     xnumel = 262144
     r0_numel = 32
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 262144, 32, XBLOCK=128)
+    triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 262144, 32)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
