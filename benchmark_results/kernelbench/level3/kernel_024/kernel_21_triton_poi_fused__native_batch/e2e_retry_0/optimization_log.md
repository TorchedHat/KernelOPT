# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2620 ns — **Best latency**: 2460 ns (**1.0650×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`persistent_tile_scheduling_via_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_21_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3,1,1), Waves/SM: 0.00
- **Registers/thread**: 18
- **Duration**: 2620 ns
- **SM Throughput**: 0.05 % | **Memory**: 0.40 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.17 %
- **IPC Active**: 0.21 inst/cycle | **No Eligible**: 93.14 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune | 1.041× | 1.065× | ✓ | **✓ best** |
| B | autotune_xblock_and_num_warps_for_tiny_pointwise_kernel | 1.010× | 1.092× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2460 ns | -160 ns (-6.1%) | 2400 ns | -220 ns (-8.4%) |
| SM Throughput (%) | 0.05 % | 0.01 % | -0.04 pp | 0.01 % | -0.04 pp |
| Memory Throughput (%) | 0.40 % | 0.51 % | +0.11 pp | 0.49 % | +0.09 pp |
| DRAM Throughput (%) | 0.07 % | 0.04 % | -0.03 pp | 0.05 % | -0.02 pp |
| Achieved Occupancy (%) | 6.17 % | 1.56 % | -4.61 pp | 1.56 % | -4.61 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 18.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 93.14 % | 92.79 % | -0.35 pp | 93.09 % | -0.05 pp |
| IPC Active | 0.21 inst/cycle | 0.06 inst/cycle | -0.15 | 0.06 inst/cycle | -0.15 |

---

## Cumulative Progress

```
Baseline:  2620 ns  (1.000×)
Iter 1     2460 ns  (1.065×)  — persistent_tile_scheduling_via_autotune

Final best: 2460 ns  (1.0650×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16},  num_warps=1),
+        triton.Config({'XBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 32},  num_warps=1),
+        triton.Config({'XBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 512}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 384
@@ -31,8 +50,6 @@

     tmp15 = tmp13 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -41,7 +58,7 @@

     arg_4 = inputs[4]
     xnumel = 384
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 384, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_21[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 384)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
