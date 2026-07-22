# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3300 ns — **Best latency**: 3580 ns (**0.9218×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_43_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1225,1,1), Waves/SM: 0.58
- **Registers/thread**: 18
- **Duration**: 3300 ns
- **SM Throughput**: 15.09 % | **Memory**: 6.11 % | **DRAM**: 3.96 %
- **Achieved Occupancy**: 46.88 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 62.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_for_persistent_tile_scheduling | 0.999× | 0.516× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 1.003× | 0.922× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 6400 ns | +3100 ns (+93.9%) | 3580 ns | +280 ns (+8.5%) |
| SM Throughput (%) | 15.09 % | 51.03 % | +35.94 pp | 27.15 % | +12.06 pp |
| Memory Throughput (%) | 6.11 % | 10.79 % | +4.68 pp | 7.46 % | +1.35 pp |
| DRAM Throughput (%) | 3.96 % | 2.05 % | -1.91 pp | 3.68 % | -0.28 pp |
| Achieved Occupancy (%) | 46.88 % | 49.09 % | +2.21 pp | 37.56 % | -9.32 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 32.00 register/thread | +14 |
| Waves/SM | 0.58 | 1.86 | +1.28 | 0.58 | +0.00 |
| No Eligible (%) | 62.46 % | 21.84 % | -40.62 pp | 43.75 % | -18.71 pp |
| IPC Active | 1.64 inst/cycle | 3.10 inst/cycle | +1.46 | 2.26 inst/cycle | +0.62 |

---

## Cumulative Progress

```
Baseline:  3300 ns  (1.000×)

Final best: 3580 ns  (0.9218×)
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
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 156800
@@ -31,8 +44,6 @@

     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -41,7 +52,7 @@

     arg_4 = inputs[4]
     xnumel = 156800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 156800, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 156800)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
