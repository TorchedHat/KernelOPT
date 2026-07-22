# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5250 ns — **Best latency**: 5090 ns (**1.0314×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_xblock_eliminate_partial_wave`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_24_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 1.86
- **Registers/thread**: 18
- **Duration**: 5250 ns
- **SM Throughput**: 33.50 % | **Memory**: 12.08 % | **DRAM**: 7.85 %
- **Achieved Occupancy**: 58.96 %
- **IPC Active**: 2.31 inst/cycle | **No Eligible**: 38.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.966× | 0.891× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 1.006× | 1.031× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5250 ns | 5890 ns | +640 ns (+12.2%) | 5090 ns | -160 ns (-3.0%) |
| SM Throughput (%) | 33.50 % | 53.84 % | +20.34 pp | 42.13 % | +8.63 pp |
| Memory Throughput (%) | 12.08 % | 14.56 % | +2.48 pp | 18.24 % | +6.16 pp |
| DRAM Throughput (%) | 7.85 % | 7.00 % | -0.85 pp | 8.08 % | +0.23 pp |
| Achieved Occupancy (%) | 58.96 % | 73.95 % | +14.99 pp | 66.03 % | +7.07 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 31.00 register/thread | +13 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 38.72 % | 18.23 % | -20.49 pp | 31.67 % | -7.05 pp |
| IPC Active | 2.31 inst/cycle | 3.24 inst/cycle | +0.93 | 2.88 inst/cycle | +0.57 |

---

## Cumulative Progress

```
Baseline:  5250 ns  (1.000×)
Iter 1     5090 ns  (1.031×)  — autotune_xblock_eliminate_partial_wave

Final best: 5090 ns  (1.0314×)
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
+        triton.Config({'XBLOCK': 512}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 501760
@@ -21,19 +33,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = 1.0 / tl.sqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    tmp17 = tl.maximum(0.0, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +48,7 @@

     arg_4 = inputs[4]
     xnumel = 501760
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760)
     return arg_4
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
