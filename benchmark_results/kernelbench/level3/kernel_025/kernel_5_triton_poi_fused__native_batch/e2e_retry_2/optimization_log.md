# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 124990 ns — **Best latency**: 99710 ns (**1.2535×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`vectorized_loads_via_contiguous_hints`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_025/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 29.70
- **Registers/thread**: 62
- **Duration**: 124990 ns
- **SM Throughput**: 59.34 % | **Memory**: 75.54 % | **DRAM**: 61.04 %
- **Achieved Occupancy**: 47.58 %
- **IPC Active**: 2.44 inst/cycle | **No Eligible**: 38.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_contiguous_hints | 1.871× | 1.254× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 124990 ns | 99710 ns | -25280 ns (-20.2%) |
| SM Throughput (%) | 59.34 % | 62.07 % | +2.73 pp |
| Memory Throughput (%) | 75.54 % | 81.27 % | +5.73 pp |
| DRAM Throughput (%) | 61.04 % | 37.68 % | -23.36 pp |
| Achieved Occupancy (%) | 47.58 % | 53.35 % | +5.77 pp |
| Registers/Thread | 62.00 register/thread | 56.00 register/thread | -6 |
| Waves/SM | 29.70 | 26.40 | -3.30 |
| No Eligible (%) | 38.74 % | 36.07 % | -2.67 pp |
| IPC Active | 2.44 inst/cycle | 2.55 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  124990 ns  (1.000×)
Iter 1     99710 ns  (1.254×)  — vectorized_loads_via_contiguous_hints

Final best: 99710 ns  (1.2535×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,30 +5,46 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 8}, num_warps=2, num_stages=2),
+    ],
+    key=['xnumel', 'ynumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 640
     xnumel = 50176
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    _yrange = tl.arange(0, YBLOCK)
+    _yrange = tl.max_contiguous(tl.multiple_of(_yrange, YBLOCK), YBLOCK)
+    yindex = yoffset + _yrange[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    _xrange = tl.arange(0, XBLOCK)
+    _xrange = tl.max_contiguous(tl.multiple_of(_xrange, XBLOCK), XBLOCK)
+    xindex = xoffset + _xrange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y0 = (yindex % 64)
     y1 = yindex // 64
     y3 = yindex
-    tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), xmask & ymask, eviction_policy='evict_last')
-    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (y0 + 64*x2 + 3211264*y1), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr0 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (y0 + 64*x2 + 3211264*y1), mask=xmask & ymask, eviction_policy='evict_last')
+    tmp17 = tl.load(in_ptr5 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr6 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr7 + (y0), mask=ymask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr8 + (y0), mask=ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1, 1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
@@ -51,9 +67,7 @@

     tmp29 = tmp15 + tmp28
     tmp30 = tl.full([1, 1], 0, tl.int32)
     tmp31 = tl.maximum(tmp30, tmp29)
-    tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x2 + 50176*y3), tmp31, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -70,7 +84,7 @@

     ynumel = 640
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 640, 50176)
     return arg_10
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
