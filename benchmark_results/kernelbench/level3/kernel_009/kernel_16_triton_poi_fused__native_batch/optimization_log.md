# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3710 ns — **Best latency**: 3710 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_16_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (784,1,1), Waves/SM: 0.37
- **Registers/thread**: 30
- **Duration**: 3710 ns
- **SM Throughput**: 17.36 % | **Memory**: 5.39 % | **DRAM**: 4.53 %
- **Achieved Occupancy**: 32.24 %
- **IPC Active**: 1.58 inst/cycle | **No Eligible**: 60.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 0.926× | 0.768× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.999× | 1.072× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.963× | 1.008× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy | 0.970× | 1.045× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | add_autotune_with... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 4830 ns | +1120 ns (+30.2%) | 3460 ns | -250 ns (-6.7%) | 3680 ns | -30 ns (-0.8%) | 3550 ns | -160 ns (-4.3%) |
| SM Throughput (%) | 17.36 % | 27.81 % | +10.45 pp | 17.35 % | -0.01 pp | 21.40 % | +4.04 pp | 16.08 % | -1.28 pp |
| Memory Throughput (%) | 5.39 % | 7.21 % | +1.82 pp | 5.44 % | +0.05 pp | 7.20 % | +1.81 pp | 5.03 % | -0.36 pp |
| DRAM Throughput (%) | 4.53 % | 3.47 % | -1.06 pp | 4.83 % | +0.30 pp | 4.54 % | +0.01 pp | 4.70 % | +0.17 pp |
| Achieved Occupancy (%) | 32.24 % | 24.05 % | -8.19 pp | 32.60 % | +0.36 pp | 27.61 % | -4.63 pp | 33.03 % | +0.79 pp |
| Registers/Thread | 30.00 register/thread | 54.00 register/thread | +24 | 30.00 register/thread | +0 | 32.00 register/thread | +2 | 30.00 register/thread | +0 |
| Waves/SM | 0.37 | 0.66 | +0.29 | 0.37 | +0.00 | 0.37 | +0.00 | 0.37 | +0.00 |
| No Eligible (%) | 60.89 % | 53.24 % | -7.65 pp | 62.46 % | +1.57 pp | 53.10 % | -7.79 pp | 61.70 % | +0.81 pp |
| IPC Active | 1.58 inst/cycle | 2.05 inst/cycle | +0.47 | 1.56 inst/cycle | -0.02 | 1.87 inst/cycle | +0.29 | 1.56 inst/cycle | -0.02 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_underutilized_pointwise_kernel` (0.768×)

Adding @triton.autotune with large XBLOCK configs (512–2048) to a pointwise kernel that already had low achieved occupancy caused register pressure to increase from 30 to 54 registers/thread. This reduced theoretical occupancy from 100% to 56.25% (Block Limit Registers dropped from 16 to 9 blocks/SM), achieved occupancy fell further from 32.2% to 24.1%, and total executed instructions nearly doubled (479K→942K), increasing kernel duration from 3.71µs to 4.83µs. When a kernel's register count is near the boundary where larger tile sizes push it over a register-file occupancy cliff, autotune with large XBLOCK configs will cause a net regression despite improving ILP per block.

> **Applicable when**: Kernel has very low achieved occupancy due to small grid (waves_per_sm < 0.5) AND registers_per_thread is well below the occupancy-limiting threshold (e.g., <24 regs for 100% theoretical occupancy on H100/H200). The grid lambda must correctly use META['XBLOCK'] so block count scales with tile size.

> **Avoid when**: Kernel already uses 28+ registers/thread — increasing XBLOCK will push register usage over the occupancy cliff (Block Limit Registers will drop), reducing theoretical occupancy and causing net regression. Also avoid when the grid is hardcoded and not updated to use META['XBLOCK'], as the block count won't decrease proportionally, wasting launch overhead. Do not apply when the kernel body has many live variables that scale with XBLOCK (e.g., complex multi-input pointwise fusions with 9+ tensor loads).

#### ✅ Plan B: `autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel` (1.072×)

When a pointwise kernel has fewer than 0.5 waves/SM and achieved occupancy well below theoretical (e.g., 32% vs 100%), adding @triton.autotune with larger XBLOCK configs (512, 1024, 2048) and varying num_warps allows the runtime to select tile sizes that reduce launch overhead and improve per-CTA work density. The key observable improvement is reduced Warp Cycles Per Issued Instruction (14.28 → 12.97) and lower elapsed cycles (5495 → 5143), even when waves/SM and achieved occupancy remain similarly constrained. This strategy requires no algorithmic changes and is safe for any pure elementwise kernel with no shared memory or inter-thread dependencies.

> **Applicable when**: Pointwise (elementwise) Triton kernel with waves_per_sm < 0.5, achieved occupancy < 50% of theoretical, IssueSlotUtilization No Eligible > 50%, no shared memory usage, no inter-thread reductions, and register count per thread leaves headroom (e.g., < 40 registers/thread with 64 max warps/SM).

> **Avoid when**: Kernels with reductions, shared memory tiling, or inter-warp synchronization where XBLOCK is semantically constrained. Also avoid when the kernel is already at >= 1 full wave and occupancy is near theoretical, or when register pressure is near the limit (>= 48 registers/thread) making larger blocks cause register spilling.

#### ✅ Plan D: `autotune_xblock_and_num_warps_for_occupancy` (1.045×)

For pointwise kernels with very small grids (waves/SM < 0.5) and high no-eligible-warp percentages (>50%), adding @triton.autotune over a range of XBLOCK values (128–1024) and num_warps (4–16) allows the autotuner to find a configuration that better hides latency. The key change is replacing a hardcoded XBLOCK and static grid with an autotune decorator and a lambda grid, requiring no algorithmic changes. In this case the improvement was modest (~4.5%) because the grid remained under 1 wave even with larger XBLOCK; warp cycles per issued instruction dropped from 14.28 to 13.12 cycles, indicating slightly better latency hiding.

> **Applicable when**: Triton pointwise kernel with waves/SM < 0.5, achieved occupancy < 40%, no-eligible-warp % > 50%, and no inter-element dependencies (pure elementwise ops). Kernel uses a hardcoded XBLOCK and static grid.

> **Avoid when**: Kernels with reductions or inter-element dependencies where XBLOCK affects correctness. Kernels already at or above 1 wave/SM where occupancy is not the bottleneck. Cases where the grid is already large enough to saturate the device. Avoid when autotune overhead (first-run latency) is unacceptable in production.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_effective_tile_size_exploration | 0.969× | 1.054× | ✓ |  |
| B | fix_hardcoded_grid_to_use_autotune_meta_xblock | 0.967× | 1.054× | ✓ |  |
| C | fix_autotune_grid_lambda_and_cap_xblock_to_avoid_register_spill | 0.849× | 1.063× | ✓ |  |
| D | reduce_register_pressure_via_small_xblock_and_fix_grid_lambda | 0.965× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3520 ns | -190 ns (-5.1%) | 3520 ns | -190 ns (-5.1%) | 3490 ns | -220 ns (-5.9%) | 3710 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 17.36 % | 17.34 % | -0.02 pp | 17.50 % | +0.14 pp | 16.28 % | -1.08 pp | 16.29 % | -1.07 pp |
| Memory Throughput (%) | 5.39 % | 5.43 % | +0.04 pp | 5.48 % | +0.09 pp | 5.11 % | -0.28 pp | 4.90 % | -0.49 pp |
| DRAM Throughput (%) | 4.53 % | 4.77 % | +0.24 pp | 4.77 % | +0.24 pp | 4.78 % | +0.25 pp | 2.30 % | -2.23 pp |
| Achieved Occupancy (%) | 32.24 % | 33.15 % | +0.91 pp | 33.67 % | +1.43 pp | 31.78 % | -0.46 pp | 33.45 % | +1.21 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 0.37 | 0.37 | +0.00 | 0.37 | +0.00 | 0.37 | +0.00 | 0.37 | +0.00 |
| No Eligible (%) | 60.89 % | 66.36 % | +5.47 pp | 59.52 % | -1.37 pp | 59.55 % | -1.34 pp | 59.32 % | -1.57 pp |
| IPC Active | 1.58 inst/cycle | 1.59 inst/cycle | +0.01 | 1.54 inst/cycle | -0.04 | 1.51 inst/cycle | -0.07 | 1.32 inst/cycle | -0.26 |

---

## Cumulative Progress

```
Baseline:  3710 ns  (1.000×)

Final best: 3710 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
     xnumel = 100352
@@ -47,8 +60,6 @@

     tmp31 = tl.maximum(tmp30, tmp29)
     tl.store(in_out_ptr0 + (x2), tmp31, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -62,7 +73,7 @@

     arg_9 = inputs[9]
     xnumel = 100352
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 100352, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 100352)
     return arg_9
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
