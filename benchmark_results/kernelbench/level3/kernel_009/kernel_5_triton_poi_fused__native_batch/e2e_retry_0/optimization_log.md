# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 11780 ns — **Best latency**: 11870 ns (**0.9924×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (14700,1,1), Waves/SM: 6.96
- **Registers/thread**: 18
- **Duration**: 11780 ns
- **SM Throughput**: 55.46 % | **Memory**: 25.58 % | **DRAM**: 13.05 %
- **Achieved Occupancy**: 60.74 %
- **IPC Active**: 2.76 inst/cycle | **No Eligible**: 31.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 1.014× | 1.003× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy_and_coalescing | 1.015× | 0.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 11780 ns | 11740 ns | -40 ns (-0.3%) | 11870 ns | +90 ns (+0.8%) |
| SM Throughput (%) | 55.46 % | 54.74 % | -0.72 pp | 55.36 % | -0.10 pp |
| Memory Throughput (%) | 25.58 % | 25.23 % | -0.35 pp | 25.57 % | -0.01 pp |
| DRAM Throughput (%) | 13.05 % | 13.08 % | +0.03 pp | 12.94 % | -0.11 pp |
| Achieved Occupancy (%) | 60.74 % | 62.51 % | +1.77 pp | 60.99 % | +0.25 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 6.96 | 6.96 | +0.00 | 6.96 | +0.00 |
| No Eligible (%) | 31.04 % | 31.15 % | +0.11 pp | 30.87 % | -0.17 pp |
| IPC Active | 2.76 inst/cycle | 2.70 inst/cycle | -0.06 | 2.75 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  11780 ns  (1.000×)

Final best: 11870 ns  (0.9924×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=16),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1881600
@@ -33,7 +47,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -42,7 +55,7 @@

     arg_4 = inputs[4]
     xnumel = 1881600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1881600)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
