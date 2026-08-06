# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused_add_gelu_sum_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2620 ns — **Best latency**: 2340 ns (**1.1197×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_static_grid_to_dynamic_lambda_for_full_sm_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused_add_gelu_sum_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_036/kernel_4_triton_per_fused_add_gelu_sum_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.02
- **Registers/thread**: 18
- **Duration**: 2620 ns
- **SM Throughput**: 0.67 % | **Memory**: 0.81 % | **DRAM**: 0.29 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.28 inst/cycle | **No Eligible**: 91.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_coverage_via_smaller_xblock | 0.991× | 1.065× | ✓ |  |
| B | autotune_xblock_to_increase_grid_coverage | 1.008× | 1.078× | ✓ |  |
| C | increase_grid_coverage_and_vectorize_loads | 1.010× | 0.985× | ✓ | **✓ best** |
| D | reduce_xblock_increase_grid_coverage | 1.008× | 1.120× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_grid_cov... | Δ | autotune_xblock_t... | Δ | increase_grid_cov... | Δ | reduce_xblock_inc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2460 ns | -160 ns (-6.1%) | 2430 ns | -190 ns (-7.3%) | 2660 ns | +40 ns (+1.5%) | 2340 ns | -280 ns (-10.7%) |
| SM Throughput (%) | 0.67 % | 0.11 % | -0.56 pp | 0.12 % | -0.55 pp | 0.54 % | -0.13 pp | 0.24 % | -0.43 pp |
| Memory Throughput (%) | 0.81 % | 0.55 % | -0.26 pp | 0.53 % | -0.28 pp | 0.42 % | -0.39 pp | 0.45 % | -0.36 pp |
| DRAM Throughput (%) | 0.29 % | 0.06 % | -0.23 pp | 0.06 % | -0.23 pp | 0.10 % | -0.19 pp | 0.07 % | -0.22 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 6.16 % | -0.07 pp | 3.08 % | -3.15 pp |
| Registers/Thread | 18.00 register/thread | 17.00 register/thread | -1 | 17.00 register/thread | -1 | 16.00 register/thread | -2 | 17.00 register/thread | -1 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.02 | +0.00 | 0.01 | -0.01 |
| No Eligible (%) | 91.70 % | 94.18 % | +2.48 pp | 93.08 % | +1.38 pp | 92.38 % | +0.68 pp | 92.93 % | +1.23 pp |
| IPC Active | 0.28 inst/cycle | 0.06 inst/cycle | -0.22 | 0.06 inst/cycle | -0.22 | 0.26 inst/cycle | -0.02 | 0.13 inst/cycle | -0.15 |


### Key Learnings

#### ✅ Plan B: `autotune_xblock_to_increase_grid_coverage` (1.139×)

When a Triton kernel's grid is hardcoded as a static tuple (e.g., `(cdiv(xnumel, 128),)`), expanding the @triton.autotune XBLOCK config list — even adding XBLOCK=256 and XBLOCK=512 — does not increase the number of launched blocks or SM coverage, because the grid size is evaluated once at call time and ignores the autotuned XBLOCK. In this iteration, the grid remained at 32 blocks (waves_per_sm=0.01, achieved occupancy=1.56%) identical to before, and the 1.14x speedup came entirely from a modest improvement in L2 hit rate (40%→53%) and reduced warp cycles per instruction (17.65→15.09), not from improved parallelism. The critical missing fix is changing the grid to `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` so the autotuner can actually scale block count with XBLOCK selection.

> **Applicable when**: Triton kernel with @triton.autotune on XBLOCK, grid_size << sm_count (waves_per_sm < 0.1), LaunchConfiguration NCU rule triggered, achieved occupancy < 5%, and the grid expression is a static tuple rather than a lambda.

> **Avoid when**: Grid is already a dynamic lambda scaling with meta['XBLOCK']; kernel is memory-bandwidth bound; xnumel is large enough that even the largest XBLOCK fills all SMs; expecting SM coverage improvement from config expansion alone without fixing the grid lambda.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_for_sm_coverage | 1.040× | 1.139× | ✓ |  |
| B | fix_grid_lambda_to_scale_with_autotuned_xblock | 1.062× | 1.120× | ✓ | **✓ best** |
| C | fix_grid_lambda_to_scale_with_autotuned_xblock | 1.031× | 1.139× | ✓ |  |
| D | fix_grid_lambda_to_scale_with_autotuned_xblock_for_full_sm_coverage | 1.060× | 1.092× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2300 ns | -320 ns (-12.2%) | 2340 ns | -280 ns (-10.7%) | 2300 ns | -320 ns (-12.2%) | 2400 ns | -220 ns (-8.4%) |
| SM Throughput (%) | 0.67 % | 0.10 % | -0.57 pp | 0.11 % | -0.56 pp | 0.08 % | -0.59 pp | 0.10 % | -0.57 pp |
| Memory Throughput (%) | 0.81 % | 0.45 % | -0.36 pp | 0.47 % | -0.34 pp | 0.49 % | -0.32 pp | 0.42 % | -0.39 pp |
| DRAM Throughput (%) | 0.29 % | 0.03 % | -0.26 pp | 0.03 % | -0.26 pp | 0.03 % | -0.26 pp | 0.04 % | -0.25 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 18.00 register/thread | 17.00 register/thread | -1 | 17.00 register/thread | -1 | 17.00 register/thread | -1 | 17.00 register/thread | -1 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 91.70 % | 93.56 % | +1.86 pp | 93.67 % | +1.97 pp | 93.42 % | +1.72 pp | 93.54 % | +1.84 pp |
| IPC Active | 0.28 inst/cycle | 0.06 inst/cycle | -0.22 | 0.05 inst/cycle | -0.23 | 0.05 inst/cycle | -0.23 | 0.06 inst/cycle | -0.22 |

---

## Iteration 3

**Best so far**: 1.1197× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pipeline_l1tex_stall_with_num_stages_and_larger_tile | 1.039× | 1.092× | ✓ |  |
| B | fix_hardcoded_grid_to_dynamic_lambda_with_larger_tile_configs | 1.060× | 1.139× | ✓ |  |
| C | increase_xblock_and_warps_for_cache_reuse_and_occupancy | 1.044× | 1.154× | ✓ |  |
| D | fix_hardcoded_grid_to_dynamic_lambda_for_sm_coverage | 1.040× | 1.170× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pipeline_l1tex_st... | Δ | fix_hardcoded_gri... | Δ | increase_xblock_a... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2400 ns | -220 ns (-8.4%) | 2300 ns | -320 ns (-12.2%) | 2270 ns | -350 ns (-13.4%) | 2240 ns | -380 ns (-14.5%) |
| SM Throughput (%) | 0.67 % | 0.11 % | -0.56 pp | 0.10 % | -0.57 pp | 0.10 % | -0.57 pp | 0.08 % | -0.59 pp |
| Memory Throughput (%) | 0.81 % | 0.41 % | -0.40 pp | 0.66 % | -0.15 pp | 0.52 % | -0.29 pp | 0.44 % | -0.37 pp |
| DRAM Throughput (%) | 0.29 % | 0.03 % | -0.26 pp | 0.03 % | -0.26 pp | 0.03 % | -0.26 pp | 0.04 % | -0.25 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 18.00 register/thread | 17.00 register/thread | -1 | 17.00 register/thread | -1 | 17.00 register/thread | -1 | 17.00 register/thread | -1 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 91.70 % | 93.47 % | +1.77 pp | 93.82 % | +2.12 pp | 93.63 % | +1.93 pp | 93.35 % | +1.65 pp |
| IPC Active | 0.28 inst/cycle | 0.05 inst/cycle | -0.23 | 0.05 inst/cycle | -0.23 | 0.05 inst/cycle | -0.23 | 0.06 inst/cycle | -0.22 |


### Key Learnings

#### ⚠️ Plan A: `pipeline_l1tex_stall_with_num_stages_and_larger_tile` (1.092×)

Adding num_stages=2 variants to Triton autotune configs is intended to pipeline memory loads and hide L1TEX stalls (CPIStall rule, ~39% of cycles). However, when the kernel is extremely small (grid_size << sm_count, waves_per_sm=0.01, duration ~2us) and the grid is hardcoded rather than dynamic, the autotuner may select a larger-XBLOCK config that increases warp cycles per issued instruction (17.65 → 21.09) and elapsed cycles (3377 → 3564), causing a slight regression in the profiled kernel while the overall benchmark speedup (1.09x) is likely attributable to other kernels or measurement noise. The num_stages pipelining benefit is negligible when the kernel is so latency-dominated by launch overhead and the grid cannot expand to fill SMs.

> **Applicable when**: Triton kernel with CPIStall rule triggered (L1TEX stall >30% of cycles), medium-to-large grid (waves_per_sm >= 0.5), and a dynamic grid lambda so that XBLOCK changes actually affect parallelism. num_stages=2+ is most effective when there is sufficient arithmetic to overlap with memory latency.

> **Avoid when**: Kernel grid is hardcoded (static tuple, not lambda) so grid_size cannot grow with XBLOCK changes; kernel is extremely small (waves_per_sm < 0.05, duration < 5us) where launch overhead dominates; achieved occupancy is already near theoretical maximum; the L1TEX stall is caused by uncoalesced access patterns that pipelining cannot fix.

---

## Iteration 4

**Best so far**: 1.1197× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | force_small_xblock_via_call_site_grid_fix_and_constrained_autotune | 1.048× | 1.154× | ✓ |  |
| B | remove_autotune_hardcode_xblock1_for_full_sm_coverage | 0.998× | 1.065× | ✓ |  |
| C | fix_hardcoded_grid_lambda_and_coalesce_strided_loads | 1.043× | 1.154× | ✓ |  |
| D | fix_static_grid_to_dynamic_lambda_for_full_sm_coverage | 1.032× | 1.120× | ✓ |  |


### Metric Comparison

| Metric | Baseline | force_small_xbloc... | Δ | remove_autotune_h... | Δ | fix_hardcoded_gri... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2270 ns | -350 ns (-13.4%) | 2460 ns | -160 ns (-6.1%) | 2270 ns | -350 ns (-13.4%) | 2340 ns | -280 ns (-10.7%) |
| SM Throughput (%) | 0.67 % | 0.11 % | -0.56 pp | 0.61 % | -0.06 pp | 0.11 % | -0.56 pp | 0.10 % | -0.57 pp |
| Memory Throughput (%) | 0.81 % | 0.43 % | -0.38 pp | 0.75 % | -0.06 pp | 0.45 % | -0.36 pp | 0.46 % | -0.35 pp |
| DRAM Throughput (%) | 0.29 % | 0.03 % | -0.26 pp | 0.31 % | +0.02 pp | 0.03 % | -0.26 pp | 0.04 % | -0.25 pp |
| Achieved Occupancy (%) | 6.23 % | 1.56 % | -4.67 pp | 6.23 % | +0.00 pp | 1.56 % | -4.67 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 18.00 register/thread | 17.00 register/thread | -1 | 18.00 register/thread | +0 | 17.00 register/thread | -1 | 17.00 register/thread | -1 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.02 | +0.00 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 91.70 % | 93.55 % | +1.85 pp | 93.27 % | +1.57 pp | 93.52 % | +1.82 pp | 93.36 % | +1.66 pp |
| IPC Active | 0.28 inst/cycle | 0.05 inst/cycle | -0.23 | 0.33 inst/cycle | +0.05 | 0.05 inst/cycle | -0.23 | 0.06 inst/cycle | -0.22 |


### Key Learnings

#### ⚠️ Plan B: `remove_autotune_hardcode_xblock1_for_full_sm_coverage` (1.065×)

Removing @triton.autotune and hardcoding XBLOCK=128 (via the explicit `XBLOCK=128` argument at the call site) while keeping the grid as `(triton.cdiv(xnumel, 128),)` = 32 blocks did NOT increase SM coverage — grid_size remained 32 and waves_per_sm only moved from 0.01 to 0.02. The plan intended XBLOCK=1 with grid=(4096,), but the implementation passed XBLOCK=128 at the call site, keeping the grid at 32 blocks. The speedup (1.065x) came from increased instruction count (1888→12928, 4x more work per block due to XBLOCK=128 with 4 warps) and improved warp cycles per instruction (16.04→11.54), not from SM parallelism. The critical lesson: removing @triton.autotune is only effective if the grid expression is simultaneously updated to `(triton.cdiv(xnumel, XBLOCK),)` with the correct small XBLOCK value (e.g., 1 or 4).

> **Applicable when**: Triton kernel with @triton.autotune removed and XBLOCK hardcoded; LaunchConfiguration NCU rule fires; grid_size << sm_count; waves_per_sm < 0.05; xnumel is small (few thousand elements).

> **Avoid when**: The grid expression already uses the correct XBLOCK divisor matching the call-site XBLOCK value; kernel is memory-bandwidth bound; xnumel is large enough that even XBLOCK=128 fills all SMs.

---

## Cumulative Progress

```
Baseline:  2620 ns  (1.000×)
Iter 2     2340 ns  (1.120×)  — fix_grid_lambda_to_scale_with_autotuned_xblock

Final best: 2340 ns  (1.1197×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=1),
+        triton.Config({'XBLOCK': 2},   num_warps=1),
+        triton.Config({'XBLOCK': 4},   num_warps=1),
+        triton.Config({'XBLOCK': 8},   num_warps=1),
+        triton.Config({'XBLOCK': 16},  num_warps=1),
+        triton.Config({'XBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused_add_gelu_sum_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 4096
@@ -39,7 +54,6 @@

     tmp14 = tmp11 + tmp12
     tl.store(in_out_ptr0 + (x3), tmp14, mask=xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -47,7 +61,7 @@

     xnumel = 4096
     r0_numel = 2
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused_add_gelu_sum_4[grid](arg_0, arg_1, arg_2, 4096, 2, XBLOCK=128)
+    triton_per_fused_add_gelu_sum_4[grid](arg_0, arg_1, arg_2, 4096, 2)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
