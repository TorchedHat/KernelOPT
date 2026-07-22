# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4640 ns — **Best latency**: 4420 ns (**1.0498×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reduce_register_pressure_via_maxnreg_to_restore_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_16_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2813,1,1), Waves/SM: 1.33
- **Registers/thread**: 18
- **Duration**: 4640 ns
- **SM Throughput**: 26.91 % | **Memory**: 12.93 % | **DRAM**: 6.39 %
- **Achieved Occupancy**: 60.44 %
- **IPC Active**: 2.11 inst/cycle | **No Eligible**: 45.15 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_xblock | 0.952× | 0.408× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.979× | 0.955× | ✓ |  |
| C | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.998× | 0.662× | ✓ |  |
| D | increase_xblock_and_add_autotune_for_vectorized_loads | 0.978× | 0.941× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_xblock_e... | Δ | increase_xblock_w... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 11360 ns | +6720 ns (+144.8%) | 4860 ns | +220 ns (+4.7%) | 7010 ns | +2370 ns (+51.1%) | 4930 ns | +290 ns (+6.2%) |
| SM Throughput (%) | 26.91 % | 68.17 % | +41.26 pp | 45.99 % | +19.08 pp | 59.68 % | +32.77 pp | 45.82 % | +18.91 pp |
| Memory Throughput (%) | 12.93 % | 15.66 % | +2.73 pp | 15.67 % | +2.74 pp | 15.56 % | +2.63 pp | 15.62 % | +2.69 pp |
| DRAM Throughput (%) | 6.39 % | 2.61 % | -3.78 pp | 6.13 % | -0.26 pp | 4.21 % | -2.18 pp | 6.03 % | -0.36 pp |
| Achieved Occupancy (%) | 60.44 % | 52.78 % | -7.66 pp | 71.38 % | +10.94 pp | 78.83 % | +18.39 pp | 72.89 % | +12.45 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.33 | 4.26 | +2.93 | 1.33 | +0.00 | 2.66 | +1.33 | 1.33 | +0.00 |
| No Eligible (%) | 45.15 % | 16.76 % | -28.39 pp | 22.65 % | -22.50 pp | 17.12 % | -28.03 pp | 22.56 % | -22.59 pp |
| IPC Active | 2.11 inst/cycle | 3.30 inst/cycle | +1.19 | 3.03 inst/cycle | +0.92 | 3.28 inst/cycle | +1.17 | 2.87 inst/cycle | +0.76 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_sm_loop_scheduling | 0.985× | 1.013× | ✓ |  |
| B | persistent_tile_scheduling_fixed_grid_sm_loop | 0.316× | 0.207× | ✓ |  |
| C | fuse_rsqrt_to_reduce_nonfused_fp32_instructions | — | fail | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.999× | 1.050× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_sm_loo... | Δ | persistent_tile_s... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4640 ns | 4580 ns | -60 ns (-1.3%) | 22370 ns | +17730 ns (+382.1%) | 4420 ns | -220 ns (-4.7%) |
| SM Throughput (%) | 26.91 % | 27.14 % | +0.23 pp | 63.29 % | +36.38 pp | 26.85 % | -0.06 pp |
| Memory Throughput (%) | 12.93 % | 12.16 % | -0.77 pp | 29.08 % | +16.15 pp | 12.88 % | -0.05 pp |
| DRAM Throughput (%) | 6.39 % | 6.46 % | +0.07 pp | 1.32 % | -5.07 pp | 6.73 % | +0.34 pp |
| Achieved Occupancy (%) | 60.44 % | 64.15 % | +3.71 pp | 86.59 % | +26.15 pp | 60.90 % | +0.46 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 29.00 register/thread | +11 | 18.00 register/thread | +0 |
| Waves/SM | 1.33 | 1.33 | +0.00 | 1.33 | +0.00 | 1.33 | +0.00 |
| No Eligible (%) | 45.15 % | 48.80 % | +3.65 pp | 22.02 % | -23.13 pp | 48.03 % | +2.88 pp |
| IPC Active | 2.11 inst/cycle | 1.89 inst/cycle | -0.22 | 3.10 inst/cycle | +0.99 | 2.12 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0498× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0498× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  4640 ns  (1.000×)
Iter 2     4420 ns  (1.050×)  — reduce_register_pressure_via_maxnreg_to_restore_occupancy

Final best: 4420 ns  (1.0498×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1, maxnreg=32),
+        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1, maxnreg=32),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 360000
@@ -31,7 +40,6 @@

     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -40,7 +48,7 @@

     arg_4 = inputs[4]
     xnumel = 360000
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 360000, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_16[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 360000)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
