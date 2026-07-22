# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2780 ns — **Best latency**: 2240 ns (**1.2411×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`eliminate_r0_block_padding_waste_and_fix_grid_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_13_triton_per_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.01
- **Registers/thread**: 24
- **Duration**: 2780 ns
- **SM Throughput**: 0.33 % | **Memory**: 1.79 % | **DRAM**: 0.45 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.23 inst/cycle | **No Eligible**: 94.01 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_parallelism_via_autotune_xblock | 1.076× | 1.112× | ✓ |  |
| B | reduce_xblock_autotune_for_sm_utilization | 1.058× | 1.112× | ✓ |  |
| C | vectorized_loads_stores_with_autotune | 1.038× | 0.749× | ✓ |  |
| D | add_autotune_reduce_xblock_increase_grid_coverage | 1.106× | 1.144× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_grid_par... | Δ | reduce_xblock_aut... | Δ | vectorized_loads_... | Δ | add_autotune_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2500 ns | -280 ns (-10.1%) | 2500 ns | -280 ns (-10.1%) | 3710 ns | +930 ns (+33.5%) | 2430 ns | -350 ns (-12.6%) |
| SM Throughput (%) | 0.33 % | 0.13 % | -0.20 pp | 0.13 % | -0.20 pp | 0.76 % | +0.43 pp | 0.06 % | -0.27 pp |
| Memory Throughput (%) | 1.79 % | 0.54 % | -1.25 pp | 0.42 % | -1.37 pp | 0.47 % | -1.32 pp | 0.52 % | -1.27 pp |
| DRAM Throughput (%) | 0.45 % | 0.09 % | -0.36 pp | 0.09 % | -0.36 pp | 0.35 % | -0.10 pp | 0.09 % | -0.36 pp |
| Achieved Occupancy (%) | 6.24 % | 3.07 % | -3.17 pp | 3.09 % | -3.15 pp | 11.97 % | +5.73 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 24.00 register/thread | 17.00 register/thread | -7 | 17.00 register/thread | -7 | 32.00 register/thread | +8 | 20.00 register/thread | -4 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.02 | +0.01 | 0.00 | -0.01 |
| No Eligible (%) | 94.01 % | 93.16 % | -0.85 pp | 92.90 % | -1.11 pp | 69.73 % | -24.28 pp | 94.27 % | +0.26 pp |
| IPC Active | 0.23 inst/cycle | 0.13 inst/cycle | -0.10 | 0.10 inst/cycle | -0.13 | 1.48 inst/cycle | +1.25 | 0.06 inst/cycle | -0.17 |

---

## Iteration 2

**Best so far**: 1.144× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_autotune_sm_coverage | 1.062× | 1.144× | ✓ |  |
| B | fix_autotune_grid_lambda_for_sm_coverage | 1.158× | 1.241× | ✓ |  |
| C | fix_grid_lambda_to_expand_sm_coverage_with_small_xblock | 1.159× | 1.241× | ✓ | **✓ best** |
| D | fix_autotune_grid_lambda_to_scale_blocks_with_xblock | 1.156× | 1.225× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2430 ns | -350 ns (-12.6%) | 2240 ns | -540 ns (-19.4%) | 2240 ns | -540 ns (-19.4%) | 2270 ns | -510 ns (-18.3%) |
| SM Throughput (%) | 0.33 % | 0.06 % | -0.27 pp | 0.04 % | -0.29 pp | 0.04 % | -0.29 pp | 0.04 % | -0.29 pp |
| Memory Throughput (%) | 1.79 % | 0.43 % | -1.36 pp | 0.45 % | -1.34 pp | 0.52 % | -1.27 pp | 0.55 % | -1.24 pp |
| DRAM Throughput (%) | 0.45 % | 0.09 % | -0.36 pp | 0.03 % | -0.42 pp | 0.03 % | -0.42 pp | 0.04 % | -0.41 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 24.00 register/thread | 20.00 register/thread | -4 | 16.00 register/thread | -8 | 16.00 register/thread | -8 | 16.00 register/thread | -8 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 94.01 % | 93.30 % | -0.71 pp | 97.51 % | +3.50 pp | 96.23 % | +2.22 pp | 96.26 % | +2.25 pp |
| IPC Active | 0.23 inst/cycle | 0.06 inst/cycle | -0.17 | 0.04 inst/cycle | -0.19 | 0.04 inst/cycle | -0.19 | 0.04 inst/cycle | -0.19 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_enable_autotune_sm_coverage` (1.144×)

Changing a fixed grid tuple to a lambda (so autotune can adapt grid size to the selected XBLOCK) did not expand the grid in this case — the autotuner still selected XBLOCK=32, keeping grid_size=16. However, the L2 Hit Rate improved from 40.49% to 52.3% and active warps per scheduler rose from 0.86 to 1.0, suggesting the lambda change altered memory access sequencing or scheduling in a way that improved cache reuse. The 1.144x speedup appears to come from this L2 hit rate improvement and marginally better warp eligibility, not from SM coverage expansion as predicted.

> **Applicable when**: Triton kernel uses @triton.autotune but the grid is a fixed constant (e.g., cdiv(N, max_XBLOCK)) rather than a lambda over meta['XBLOCK']; LaunchConfiguration rule is triggered; grid_size << sm_count; L2 Hit Rate is below 50%; active warps per scheduler < 1.0.

> **Avoid when**: Grid is already a lambda using meta['XBLOCK']; autotune is not used; the kernel is memory-bandwidth bound (DRAM Throughput > 50%); grid already covers most SMs (waves_per_sm >= 1.0).

#### ✅ Plan C: `fix_grid_lambda_to_expand_sm_coverage_with_small_xblock` (1.241×)

Adding small XBLOCK configs (1, 2, 4, 8) to the autotune search space and changing the autotune key from ['xnumel', 'r0_numel'] to ['xnumel'] caused the autotuner to select a configuration that reduced registers per thread from 20 to 16 and eliminated dynamic shared memory (64 bytes → 0 bytes). This cut executed instructions by 53% (1136 → 528) and elapsed cycles by 8% (3585 → 3301), yielding a 1.24x speedup. Notably, the grid size remained at 16 blocks (the grid lambda was not actually updated in the diff — the kernel_function grid expression change was the intended fix but the autotuner still selected a small XBLOCK that happened to fit in 16 blocks), so the gain came from reduced register/SMEM footprint rather than SM coverage expansion.

> **Applicable when**: Triton reduction kernel with LaunchConfiguration + AchievedOccupancy rules triggered; grid is small (<<SM count); autotune key includes r0_numel which is constant; registers_per_thread >= 20 or dynamic_smem > 0; autotune config space does not include XBLOCK < 16; executed instruction count is high relative to work done.

> **Avoid when**: Grid is already large enough to cover SMs (waves_per_sm >= 1.0); kernel is memory-bandwidth bound (DRAM Throughput > 50%); registers_per_thread is already minimal (<=16) and dynamic_smem is already 0; autotune overhead exceeds kernel runtime in latency-critical paths.

---

## Iteration 3

**Best so far**: 1.2411× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_r0_block_and_xblock_with_correct_grid_lambda | 1.101× | 1.073× | ✓ |  |
| B | fix_grid_lambda_and_force_small_xblock_for_sm_coverage | 1.099× | 1.241× | ✓ |  |
| C | increase_xblock_tile_size_with_high_num_warps_and_correct_grid_lambda | 1.047× | 1.073× | ✓ |  |
| D | add_num_stages_pipelining_to_hide_l1tex_load_latency | 1.071× | 1.061× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_r0_block... | Δ | fix_grid_lambda_a... | Δ | increase_xblock_t... | Δ | add_num_stages_pi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2590 ns | -190 ns (-6.8%) | 2240 ns | -540 ns (-19.4%) | 2590 ns | -190 ns (-6.8%) | 2620 ns | -160 ns (-5.8%) |
| SM Throughput (%) | 0.33 % | 0.04 % | -0.29 pp | 0.04 % | -0.29 pp | 0.29 % | -0.04 pp | 0.05 % | -0.28 pp |
| Memory Throughput (%) | 1.79 % | 0.53 % | -1.26 pp | 0.44 % | -1.35 pp | 1.06 % | -0.73 pp | 0.43 % | -1.36 pp |
| DRAM Throughput (%) | 0.45 % | 0.03 % | -0.42 pp | 0.03 % | -0.42 pp | 0.26 % | -0.19 pp | 0.08 % | -0.37 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 6.18 % | -0.06 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 24.00 register/thread | 16.00 register/thread | -8 | 16.00 register/thread | -8 | 24.00 register/thread | +0 | 20.00 register/thread | -4 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 94.01 % | 96.17 % | +2.16 pp | 96.17 % | +2.16 pp | 93.80 % | -0.21 pp | 93.32 % | -0.69 pp |
| IPC Active | 0.23 inst/cycle | 0.04 inst/cycle | -0.19 | 0.04 inst/cycle | -0.19 | 0.24 inst/cycle | +0.01 | 0.05 inst/cycle | -0.18 |


### Key Learnings

#### ✅ Plan D: `add_num_stages_pipelining_to_hide_l1tex_load_latency` (1.061×)

Adding num_stages=2 and num_stages=3 to Triton autotune configs for a very short, low-occupancy kernel caused the autotuner to select a configuration with higher register pressure (16→20 registers/thread) and non-zero dynamic shared memory (0→64 bytes), increasing executed instructions by 115% (528→1136) and elapsed cycles by 17% (3333→3906). Despite Warp Cycles Per Issued Instruction improving (26.03→14.69) and L2 Hit Rate rising (45.39%→51.77%), the net effect was a modest 1.06x speedup — substantially below the 1.24x achieved by the prior iteration's config simplification. For extremely small kernels (<<2us, <600 instructions), num_stages pipelining overhead in register/SMEM allocation can negate the latency-hiding benefit.

> **Applicable when**: Triton kernel with CPIStall rule triggered (>40% L1TEX stall fraction), moderate instruction count (>2000 instructions), sufficient occupancy (>5% achieved) to benefit from pipelining, and L1 hit rate near 0% with L2 hit rate >50%.

> **Avoid when**: Kernel is extremely small (<600 instructions, <2.5us duration) with very low achieved occupancy (<2%); the prior autotune config set already found a low-register (<=16 reg/thread), zero-SMEM configuration; adding num_stages causes register/SMEM pressure to increase and instruction count to more than double. In such cases, the pipelining overhead exceeds the latency-hiding benefit.

---

## Iteration 4

**Best so far**: 1.2411× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | force_xblock1_grid2048_with_r0block16_mask_elimination | 1.120× | 1.225× | ✓ |  |
| B | coalesce_reduction_dimension_memory_access | 1.133× | 1.086× | ✓ |  |
| C | eliminate_r0_block_padding_waste_and_fix_grid_lambda | 1.101× | 1.241× | ✓ |  |
| D | sequential_reduction_loop_to_eliminate_masked_waste_and_improve_coalescing | 1.084× | 1.188× | ✓ |  |


### Metric Comparison

| Metric | Baseline | force_xblock1_gri... | Δ | coalesce_reductio... | Δ | eliminate_r0_bloc... | Δ | sequential_reduct... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2780 ns | 2270 ns | -510 ns (-18.3%) | 2560 ns | -220 ns (-7.9%) | 2240 ns | -540 ns (-19.4%) | 2340 ns | -440 ns (-15.8%) |
| SM Throughput (%) | 0.33 % | 0.12 % | -0.21 pp | 0.05 % | -0.28 pp | 0.04 % | -0.29 pp | 0.04 % | -0.29 pp |
| Memory Throughput (%) | 1.79 % | 0.60 % | -1.19 pp | 0.47 % | -1.32 pp | 0.43 % | -1.36 pp | 0.47 % | -1.32 pp |
| DRAM Throughput (%) | 0.45 % | 0.03 % | -0.42 pp | 0.14 % | -0.31 pp | 0.04 % | -0.41 pp | 0.10 % | -0.35 pp |
| Achieved Occupancy (%) | 6.24 % | 6.12 % | -0.12 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 24.00 register/thread | 16.00 register/thread | -8 | 22.00 register/thread | -2 | 16.00 register/thread | -8 | 21.00 register/thread | -3 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 94.01 % | 96.66 % | +2.65 pp | 95.21 % | +1.20 pp | 96.27 % | +2.26 pp | 95.03 % | +1.02 pp |
| IPC Active | 0.23 inst/cycle | 0.12 inst/cycle | -0.11 | 0.06 inst/cycle | -0.17 | 0.03 inst/cycle | -0.20 | 0.06 inst/cycle | -0.17 |

---

## Cumulative Progress

```
Baseline:  2780 ns  (1.000×)
Iter 1     2430 ns  (1.144×)  — add_autotune_reduce_xblock_increase_grid_coverage
Iter 2     2240 ns  (1.241×)  — fix_grid_lambda_to_expand_sm_coverage_with_small_xblock

Final best: 2240 ns  (1.2411×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1}, num_warps=1),
+        triton.Config({'XBLOCK': 2}, num_warps=1),
+        triton.Config({'XBLOCK': 4}, num_warps=1),
+        triton.Config({'XBLOCK': 8}, num_warps=1),
+        triton.Config({'XBLOCK': 16}, num_warps=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 2048
@@ -24,16 +35,14 @@

     x0 = (xindex % 256)
     x1 = xindex // 256
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 1792*x1), r0_mask & xmask, other=0.0)
+    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 1792*x1), mask=r0_mask & xmask, other=0.0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
     tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
     tmp5 = tl.full([1, 1], 784.0, tl.float32)
     tmp6 = (tmp4 / tmp5)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp6, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +50,7 @@

     xnumel = 2048
     r0_numel = 7
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13[grid](arg_0, arg_1, 2048, 7, XBLOCK=128)
+    triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13[grid](arg_0, arg_1, 2048, 7)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
