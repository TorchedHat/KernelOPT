# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5060 ns — **Best latency**: 5060 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_22_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3675,1,1), Waves/SM: 1.74
- **Registers/thread**: 18
- **Duration**: 5060 ns
- **SM Throughput**: 32.02 % | **Memory**: 11.97 % | **DRAM**: 7.63 %
- **Achieved Occupancy**: 59.75 %
- **IPC Active**: 2.31 inst/cycle | **No Eligible**: 43.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.900× | 0.350× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.905× | 0.904× | ✓ |  |
| C | vectorized_loads_stores_with_larger_xblock | 0.978× | 0.948× | ✓ |  |
| D | autotune_xblock_and_num_warps_to_eliminate_partial_wave | 0.978× | 0.958× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 14460 ns | +9400 ns (+185.8%) | 5600 ns | +540 ns (+10.7%) | 5340 ns | +280 ns (+5.5%) | 5280 ns | +220 ns (+4.3%) |
| SM Throughput (%) | 32.02 % | 73.79 % | +41.77 pp | 50.68 % | +18.66 pp | 39.35 % | +7.33 pp | 39.22 % | +7.20 pp |
| Memory Throughput (%) | 11.97 % | 15.31 % | +3.34 pp | 15.04 % | +3.07 pp | 14.22 % | +2.25 pp | 14.11 % | +2.14 pp |
| DRAM Throughput (%) | 7.63 % | 2.67 % | -4.96 pp | 6.92 % | -0.71 pp | 7.25 % | -0.38 pp | 7.34 % | -0.29 pp |
| Achieved Occupancy (%) | 59.75 % | 54.29 % | -5.46 pp | 73.72 % | +13.97 pp | 69.70 % | +9.95 pp | 68.52 % | +8.77 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 24.00 register/thread | +6 |
| Waves/SM | 1.74 | 5.57 | +3.83 | 1.74 | +0.00 | 1.74 | +0.00 | 1.74 | +0.00 |
| No Eligible (%) | 43.97 % | 14.87 % | -29.10 pp | 20.21 % | -23.76 pp | 25.14 % | -18.83 pp | 25.09 % | -18.88 pp |
| IPC Active | 2.31 inst/cycle | 3.39 inst/cycle | +1.08 | 3.16 inst/cycle | +0.85 | 2.82 inst/cycle | +0.51 | 2.93 inst/cycle | +0.62 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.350×)

Adding @triton.autotune with large XBLOCK configs (512, 1024, 2048) to a pointwise kernel caused registers per thread to jump from 18 to 48, reducing Block Limit Registers from 21 to 5 and theoretical occupancy from 100% to 62.5%. More critically, executed instructions exploded from ~1.3M to ~8.4M (6.5x increase), indicating the autotuner selected a config that dramatically increased per-thread work or caused register spilling, resulting in 2.86x longer kernel duration despite improved scheduler utilization. When a kernel already has low register pressure (≤20 registers/thread) and the autotune decorator itself causes register count to balloon, the occupancy loss and instruction overhead far outweigh any partial-wave tail benefit.

> **Applicable when**: Never: this specific pattern (autotune with large XBLOCK on a kernel with originally low register count) caused severe regression. Do not apply when the kernel has ≤20 registers/thread and adding autotune causes register count to increase by >2x.

> **Avoid when**: Kernel has low register pressure (≤20 regs/thread) AND adding autotune decorator causes register count to balloon (>2x increase). The LaunchConfiguration NCU rule may flag partial-wave overhead, but if the fix causes register spilling or instruction count explosion, the cure is worse than the disease. Also avoid when the autotune overhead itself (compilation of multiple configs) is not amortized by repeated kernel calls.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_and_fuse_fp32_instructions_via_fma | 1.000× | 0.969× | ✓ |  |
| B | fix_autotune_grid_lambda_and_tune_num_warps_for_occupancy | 0.973× | 1.008× | ✓ |  |
| C | fix_grid_lambda_to_use_meta_xblock_with_divisor_aligned_tile_sizes | 0.975× | 0.969× | ✓ |  |
| D | remove_large_xblock_autotune_restore_low_register_pressure | 0.972× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ | remove_large_xblo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5060 ns | 5220 ns | +160 ns (+3.2%) | 5020 ns | -40 ns (-0.8%) | 5220 ns | +160 ns (+3.2%) | 5060 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 32.02 % | 30.63 % | -1.39 pp | 32.05 % | +0.03 pp | 22.90 % | -9.12 pp | 31.17 % | -0.85 pp |
| Memory Throughput (%) | 11.97 % | 12.11 % | +0.14 pp | 12.01 % | +0.04 pp | 10.27 % | -1.70 pp | 11.70 % | -0.27 pp |
| DRAM Throughput (%) | 7.63 % | 7.39 % | -0.24 pp | 7.71 % | +0.08 pp | 7.42 % | -0.21 pp | 7.69 % | +0.06 pp |
| Achieved Occupancy (%) | 59.75 % | 62.04 % | +2.29 pp | 59.51 % | -0.24 pp | 61.03 % | +1.28 pp | 58.62 % | -1.13 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 24.00 register/thread | +6 | 18.00 register/thread | +0 |
| Waves/SM | 1.74 | 1.74 | +0.00 | 1.74 | +0.00 | 0.87 | -0.87 | 1.74 | +0.00 |
| No Eligible (%) | 43.97 % | 44.06 % | +0.09 pp | 41.30 % | -2.67 pp | 41.15 % | -2.82 pp | 46.87 % | +2.90 pp |
| IPC Active | 2.31 inst/cycle | 2.04 inst/cycle | -0.27 | 2.23 inst/cycle | -0.08 | 2.43 inst/cycle | +0.12 | 2.25 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  5060 ns  (1.000×)

Final best: 5060 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 470400
@@ -35,8 +49,6 @@

     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -45,7 +57,7 @@

     arg_4 = inputs[4]
     xnumel = 470400
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 470400, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 470400)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
