# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4320 ns — **Best latency**: 4320 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_17_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2450,1,1), Waves/SM: 1.16
- **Registers/thread**: 18
- **Duration**: 4320 ns
- **SM Throughput**: 24.54 % | **Memory**: 11.47 % | **DRAM**: 6.00 %
- **Achieved Occupancy**: 59.62 %
- **IPC Active**: 1.97 inst/cycle | **No Eligible**: 47.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_autotune_xblock | 0.967× | 1.046× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave | 0.968× | 1.031× | ✓ |  |
| C | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.966× | 0.993× | ✓ |  |
| D | add_autotune_with_larger_xblock_for_vectorized_loads_and_wave_elimination | 0.968× | 1.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_t... | Δ | increase_xblock_w... | Δ | add_autotune_with... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4320 ns | 4130 ns | -190 ns (-4.4%) | 4190 ns | -130 ns (-3.0%) | 4350 ns | +30 ns (+0.7%) | 4160 ns | -160 ns (-3.7%) |
| SM Throughput (%) | 24.54 % | 31.24 % | +6.70 pp | 31.47 % | +6.93 pp | 42.99 % | +18.45 pp | 31.52 % | +6.98 pp |
| Memory Throughput (%) | 11.47 % | 14.08 % | +2.61 pp | 14.04 % | +2.57 pp | 14.63 % | +3.16 pp | 14.22 % | +2.75 pp |
| DRAM Throughput (%) | 6.00 % | 6.26 % | +0.26 pp | 6.19 % | +0.19 pp | 5.93 % | -0.07 pp | 6.20 % | +0.20 pp |
| Achieved Occupancy (%) | 59.62 % | 72.13 % | +12.51 pp | 72.39 % | +12.77 pp | 69.60 % | +9.98 pp | 72.03 % | +12.41 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 47.59 % | 28.32 % | -19.27 pp | 28.24 % | -19.35 pp | 23.84 % | -23.75 pp | 29.71 % | -17.88 pp |
| IPC Active | 1.97 inst/cycle | 2.82 inst/cycle | +0.85 | 2.79 inst/cycle | +0.82 | 2.98 inst/cycle | +1.01 | 2.80 inst/cycle | +0.83 |


### Key Learnings

#### ✅ Plan A: `eliminate_partial_wave_via_autotune_xblock` (1.046×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) to a pointwise kernel that was fixed at XBLOCK=128 improved achieved occupancy from 59.6% to 72.1% and reduced idle scheduler cycles (No Eligible) from 47.6% to 28.3%, yielding a 1.046x speedup. The larger tile sizes allow more warps to be active per SM, increasing eligible warps per scheduler from 2.26 to 3.89 and Issue Slots Busy from 49.6% to 70.9%. Note: the grid size and waves_per_sm did not change in the profiled config (autotune selected a config that kept grid=2450), so the primary benefit came from better warp-level scheduling rather than partial-wave elimination — the occupancy and IPC gains drove the improvement.

> **Applicable when**: Triton pointwise kernels with fixed small XBLOCK (e.g. 128) where achieved occupancy is below 65%, No Eligible warp % is above 40%, Issue Slots Busy is below 55%, and the NCU LaunchConfiguration or IssueSlotUtilization rules are triggered. Especially effective when the kernel is short-duration (<10 us) and latency-bound.

> **Avoid when**: Kernel is already memory-bandwidth bound (DRAM Throughput >60%) — larger tiles won't help and may hurt cache reuse. Also avoid when the tensor has in-place writes and restore_value is not set in autotune (correctness risk). Do not apply when register pressure is already near the block limit, as larger num_warps configs may reduce occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.982× | 0.986× | ✓ |  |
| B | persistent_tile_scheduling | 0.983× | 1.024× | ✓ |  |
| C | persistent_tile_scheduling | 0.983× | 1.000× | ✓ |  |
| D | persistent_tile_scheduling | 0.989× | 0.971× | ✓ |  |
| E | add_triton_autotune | 0.965× | 1.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4320 ns | 4380 ns | +60 ns (+1.4%) | 4220 ns | -100 ns (-2.3%) | 4320 ns | +0 ns (+0.0%) | 4450 ns | +130 ns (+3.0%) | 4160 ns | -160 ns (-3.7%) |
| SM Throughput (%) | 24.54 % | 27.20 % | +2.66 pp | 27.77 % | +3.23 pp | 27.81 % | +3.27 pp | 26.11 % | +1.57 pp | 24.37 % | -0.17 pp |
| Memory Throughput (%) | 11.47 % | 11.45 % | -0.02 pp | 11.70 % | +0.23 pp | 11.74 % | +0.27 pp | 11.40 % | -0.07 pp | 11.73 % | +0.26 pp |
| DRAM Throughput (%) | 6.00 % | 5.89 % | -0.11 pp | 6.12 % | +0.12 pp | 5.96 % | -0.04 pp | 5.81 % | -0.19 pp | 6.18 % | +0.18 pp |
| Achieved Occupancy (%) | 59.62 % | 64.42 % | +4.80 pp | 64.18 % | +4.56 pp | 64.52 % | +4.90 pp | 64.19 % | +4.57 pp | 58.86 % | -0.76 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 30.00 register/thread | +12 | 18.00 register/thread | +0 |
| Waves/SM | 1.16 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 | 1.16 | +0.00 |
| No Eligible (%) | 47.59 % | 43.57 % | -4.02 pp | 44.50 % | -3.09 pp | 44.62 % | -2.97 pp | 52.01 % | +4.42 pp | 47.85 % | +0.26 pp |
| IPC Active | 1.97 inst/cycle | 2.16 inst/cycle | +0.19 | 2.17 inst/cycle | +0.20 | 2.16 inst/cycle | +0.19 | 2.14 inst/cycle | +0.17 | 1.97 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  4320 ns  (1.000×)

Final best: 4320 ns  (1.0000×)
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
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 313600
@@ -31,8 +42,6 @@

     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -41,7 +50,7 @@

     arg_4 = inputs[4]
     xnumel = 313600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 313600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 313600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
