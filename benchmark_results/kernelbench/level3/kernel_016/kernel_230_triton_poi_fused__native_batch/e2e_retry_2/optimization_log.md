# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7200 ns — **Best latency**: 7040 ns (**1.0227×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_increase_occupancy_and_l1_reuse`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_230`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_230_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7105,1,1), Waves/SM: 3.36
- **Registers/thread**: 22
- **Duration**: 7200 ns
- **SM Throughput**: 44.02 % | **Memory**: 16.66 % | **DRAM**: 10.46 %
- **Achieved Occupancy**: 60.89 %
- **IPC Active**: 2.55 inst/cycle | **No Eligible**: 37.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_reduce_partial_wave_and_improve_l1_reuse | 0.922× | 0.787× | ✓ |  |
| B | autotune_xblock_increase_occupancy_and_l1_reuse | 1.018× | 1.023× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_i... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7200 ns | 9150 ns | +1950 ns (+27.1%) | 7040 ns | -160 ns (-2.2%) |
| SM Throughput (%) | 44.02 % | 64.21 % | +20.19 pp | 43.52 % | -0.50 pp |
| Memory Throughput (%) | 16.66 % | 17.33 % | +0.67 pp | 16.56 % | -0.10 pp |
| DRAM Throughput (%) | 10.46 % | 8.18 % | -2.28 pp | 10.63 % | +0.17 pp |
| Achieved Occupancy (%) | 60.89 % | 81.26 % | +20.37 pp | 61.70 % | +0.81 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 3.36 | 3.36 | +0.00 | 3.36 | +0.00 |
| No Eligible (%) | 37.78 % | 17.72 % | -20.06 pp | 38.35 % | +0.57 pp |
| IPC Active | 2.55 inst/cycle | 3.26 inst/cycle | +0.71 | 2.50 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  7200 ns  (1.000×)
Iter 1     7040 ns  (1.023×)  — autotune_xblock_increase_occupancy_and_l1_reuse

Final best: 7040 ns  (1.0227×)
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
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_230(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 909440
@@ -33,8 +46,6 @@

     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +55,7 @@

     arg_5 = inputs[5]
     xnumel = 909440
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_230[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 909440, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_230[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 909440)
     return arg_5
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
