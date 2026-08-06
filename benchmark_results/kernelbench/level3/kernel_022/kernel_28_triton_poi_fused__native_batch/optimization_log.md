# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3650 ns — **Best latency**: 3650 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_28_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1225,1,1), Waves/SM: 0.58
- **Registers/thread**: 22
- **Duration**: 3650 ns
- **SM Throughput**: 15.17 % | **Memory**: 7.04 % | **DRAM**: 7.04 %
- **Achieved Occupancy**: 44.77 %
- **IPC Active**: 1.60 inst/cycle | **No Eligible**: 59.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_sm_utilization | 0.966× | 0.815× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.928× | 0.943× | ✓ |  |
| D | autotune_xblock_increase_tile_size | 0.971× | 0.559× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_a... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3650 ns | 4480 ns | +830 ns (+22.7%) | 3870 ns | +220 ns (+6.0%) | 6530 ns | +2880 ns (+78.9%) |
| SM Throughput (%) | 15.17 % | 42.67 % | +27.50 pp | 27.47 % | +12.30 pp | 56.84 % | +41.67 pp |
| Memory Throughput (%) | 7.04 % | 11.05 % | +4.01 pp | 8.85 % | +1.81 pp | 12.74 % | +5.70 pp |
| DRAM Throughput (%) | 7.04 % | 5.75 % | -1.29 pp | 6.74 % | -0.30 pp | 3.94 % | -3.10 pp |
| Achieved Occupancy (%) | 44.77 % | 70.72 % | +25.95 pp | 41.00 % | -3.77 pp | 79.32 % | +34.55 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 0.58 | 1.16 | +0.58 | 0.58 | +0.00 | 2.32 | +1.74 |
| No Eligible (%) | 59.45 % | 24.79 % | -34.66 pp | 45.83 % | -13.62 pp | 18.74 % | -40.71 pp |
| IPC Active | 1.60 inst/cycle | 3.00 inst/cycle | +1.40 | 2.17 inst/cycle | +0.57 | 3.20 inst/cycle | +1.60 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_tile_size_for_sm_utilization` (0.815×)

Adding @triton.autotune to a pointwise kernel with a small grid (0.58 waves/SM) and low register pressure (22 regs/thread) selected a larger block size (256) that increased registers per thread from 22 to 32, reduced Block Limit Registers from 21 to 8 blocks/SM, and caused a 3.66x explosion in executed instructions (421K → 1.54M). Although occupancy improved (44.8% → 70.7%) and compute throughput rose (15% → 43%), the net effect was a 23% slower kernel (3.65 µs → 4.48 µs) because the instruction count increase dominated. For very short-duration kernels (<5 µs) where the bottleneck is grid size rather than per-SM occupancy, autotuning larger XBLOCK tiles can backfire if the selected config inflates register usage and instruction count beyond what the latency savings from fewer waves can compensate.

> **Applicable when**: Grid is severely under-saturated (waves/SM < 0.5), compute and memory throughput both below 20%, eligible warps/scheduler < 2, and kernel duration is long enough (>20 µs) that per-wave latency savings outweigh instruction overhead from larger tiles.

> **Avoid when**: Kernel duration is very short (<5 µs); register count is near the block-limit boundary (e.g., 22 regs where doubling block size pushes to 32 regs and halves Block Limit Registers); or the total instruction count is already low (< 500K) where any tile-size increase multiplies instructions faster than it reduces wave count. Also avoid when the autotune key does not vary across calls, making autotuning overhead amortization poor.

#### ❌ Plan D: `autotune_xblock_increase_tile_size` (0.559×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) on a small pointwise kernel caused a 7x increase in executed instructions (421K → 3.01M) despite the same grid size (1225 blocks), resulting in 1.79x longer duration. The autotune selected XBLOCK=512 but the grid was not updated to use meta['XBLOCK'] — it remained hardcoded to cdiv(xnumel, 128), so each block processed 4x more elements than intended, causing redundant computation. When the grid lambda is not correctly wired to meta['XBLOCK'], larger tile configs multiply total work rather than redistributing it, turning a latency-bound kernel into a compute-bound one with worse wall time.

> **Applicable when**: Never apply this pattern as implemented. The correct version (lambda grid) is applicable when waves_per_sm < 1.0 and the kernel is a simple pointwise loop with no inter-element dependencies.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs unless the launch grid is simultaneously updated to a lambda that reads meta['XBLOCK']. If the grid remains hardcoded to cdiv(xnumel, FIXED_BLOCK), each block will process XBLOCK/FIXED_BLOCK times more elements than the data covers, causing massive instruction count inflation and regression.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_persistent_tile_scheduling | 0.965× | 1.077× | ✓ |  |
| B | persistent_tile_scheduling | 0.963× | 1.000× | ✓ |  |
| C | persistent_tile_scheduling | 0.963× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3650 ns | 3390 ns | -260 ns (-7.1%) | 3650 ns | +0 ns (+0.0%) | 3710 ns | +60 ns (+1.6%) |
| SM Throughput (%) | 15.17 % | 15.67 % | +0.50 pp | 15.41 % | +0.24 pp | 16.19 % | +1.02 pp |
| Memory Throughput (%) | 7.04 % | 7.63 % | +0.59 pp | 7.12 % | +0.08 pp | 6.94 % | -0.10 pp |
| DRAM Throughput (%) | 7.04 % | 7.63 % | +0.59 pp | 7.12 % | +0.08 pp | 6.94 % | -0.10 pp |
| Achieved Occupancy (%) | 44.77 % | 45.54 % | +0.77 pp | 47.33 % | +2.56 pp | 46.79 % | +2.02 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 |
| No Eligible (%) | 59.45 % | 60.50 % | +1.05 pp | 59.44 % | -0.01 pp | 59.57 % | +0.12 pp |
| IPC Active | 1.60 inst/cycle | 1.57 inst/cycle | -0.03 | 1.55 inst/cycle | -0.05 | 1.49 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  3650 ns  (1.000×)

Final best: 3650 ns  (1.0000×)
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
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xnumel = 156800
@@ -33,8 +45,6 @@

     tmp17 = tmp15 + tmp16
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +54,7 @@

     arg_5 = inputs[5]
     xnumel = 156800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 156800, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_hardtanh_28[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 156800)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
